"""A model on any server that speaks OpenAI Chat Completions.

Works with vLLM, SGLang, llama.cpp, Ollama, OpenAI itself and gateways in
front of them. It talks HTTP directly with ``httpx``; no vendor SDK is needed.

Retry policy (the same conservative one the QCloud SDK uses): a request is
sent again only when it provably never ran, that is, on a 429 or when the
connection could not be established. A timeout, a reset or a 5xx after the
request was sent is raised as ``ambiguous`` and left to the caller.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx

from ..errors import (
    ModelAuthenticationError,
    ModelBadRequestError,
    ModelBillingError,
    ModelConnectionError,
    ModelError,
    ModelRateLimitError,
    ModelResponseError,
    ModelServerError,
    ModelTimeoutError,
)
from ._openai_wire import ChatAccumulator, build_chat_body
from .base import Model, ModelCapabilities, ModelRequest, ModelStreamEvent, ResponseDone

_NEVER_SENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)


class OpenAIChatModel(Model):
    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 600.0,
        max_retries: int = 2,
        http_client: httpx.AsyncClient | None = None,
        provider: str = "openai-compatible",
        max_tokens_param: str = "max_tokens",
        include_usage: bool = True,
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        self.name = model
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._headers = {"Content-Type": "application/json", **(headers or {})}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._timeout = timeout
        self._client = http_client
        self._owns_client = http_client is None
        self._max_tokens_param = max_tokens_param
        self._include_usage = include_usage
        self._capabilities = capabilities or ModelCapabilities()

    async def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        body = build_chat_body(
            self.name,
            request,
            stream=True,
            max_tokens_param=self._max_tokens_param,
            include_usage=self._include_usage,
        )
        response = await self._send(body)
        accumulator = ChatAccumulator(provider=self.provider, model=self.name)
        finished = False
        try:
            async for payload in _sse_payloads(response):
                if payload.strip() == "[DONE]":
                    finished = True
                    break
                try:
                    chunk = json.loads(payload)
                except ValueError as error:
                    raise self._error(
                        ModelResponseError,
                        "The server sent invalid JSON in its stream.",
                        response=response,
                        ambiguous=True,
                    ) from error
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("error"):
                    raise self._error(
                        ModelServerError,
                        _error_message(chunk),
                        response=response,
                        code=_error_code(chunk),
                        ambiguous=True,
                    )
                for event in accumulator.feed(chunk):
                    yield event
        except httpx.TimeoutException as error:
            raise self._error(
                ModelTimeoutError,
                "The server stopped sending before the response was complete.",
                response=response,
                ambiguous=True,
            ) from error
        except httpx.TransportError as error:
            raise self._error(
                ModelConnectionError,
                f"The connection broke mid-response: {error}",
                response=response,
                ambiguous=True,
            ) from error
        finally:
            await response.aclose()
        if not finished and accumulator.finish_reason is None:
            raise self._error(
                ModelResponseError,
                "The response stream ended before the model finished.",
                response=response,
                ambiguous=True,
            )
        result = accumulator.result()
        result.usage.cost_usd = self._capabilities.cost(result.usage)
        yield ResponseDone(response=result)

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout, follow_redirects=False)
        return self._client

    async def _send(self, body: dict[str, Any]) -> httpx.Response:
        client = self._http()
        url = f"{self.base_url}/chat/completions"
        for attempt in range(self.max_retries + 1):
            last = attempt == self.max_retries
            try:
                request = client.build_request("POST", url, json=body, headers=self._headers)
                response = await client.send(request, stream=True)
            except _NEVER_SENT as error:
                if not last:
                    await asyncio.sleep(_backoff(attempt))
                    continue
                raise self._error(
                    ModelConnectionError,
                    f"Could not connect to {self.base_url}: {error}",
                    retryable=True,
                ) from error
            except httpx.TimeoutException as error:
                raise self._error(
                    ModelTimeoutError, "The request timed out after it was sent.", ambiguous=True
                ) from error
            except httpx.TransportError as error:
                raise self._error(
                    ModelConnectionError,
                    f"The connection failed after the request was sent: {error}",
                    ambiguous=True,
                ) from error
            if response.status_code < 400:
                return response
            raw = await response.aread()
            await response.aclose()
            error = self._status_error(response, raw)
            if isinstance(error, ModelRateLimitError) and not last:
                delay = error.retry_after
                await asyncio.sleep(
                    delay if delay is not None and delay <= 60 else _backoff(attempt)
                )
                continue
            raise error
        raise AssertionError("unreachable")

    def _status_error(self, response: httpx.Response, raw: bytes) -> ModelError:
        try:
            data = json.loads(raw)
        except ValueError:
            data = None
        message = _error_message(data) if isinstance(data, dict) else ""
        if not message:
            message = raw.decode("utf-8", errors="replace")[:500] or response.reason_phrase
        message = f"[{response.status_code}] {message}"
        code = _error_code(data) if isinstance(data, dict) else None
        status = response.status_code
        if status in (400, 404, 422):
            return self._error(ModelBadRequestError, message, response=response, code=code)
        if status in (401, 403):
            return self._error(ModelAuthenticationError, message, response=response, code=code)
        if status == 402:
            return self._error(ModelBillingError, message, response=response, code=code)
        if status == 429:
            return self._error(
                ModelRateLimitError, message, response=response, code=code, retryable=True
            )
        return self._error(
            ModelServerError, message, response=response, code=code, retryable=True, ambiguous=True
        )

    def _error(
        self,
        kind: type[ModelError],
        message: str,
        *,
        response: httpx.Response | None = None,
        code: str | None = None,
        retryable: bool = False,
        ambiguous: bool = False,
    ) -> ModelError:
        headers = response.headers if response is not None else {}
        return kind(
            message,
            provider=self.provider,
            model=self.name,
            status_code=response.status_code if response is not None else None,
            code=code,
            request_id=headers.get("x-request-id") or headers.get("request-id"),
            retry_after=_retry_after(headers.get("retry-after")),
            retryable=retryable,
            ambiguous=ambiguous,
        )


async def _sse_payloads(response: httpx.Response) -> AsyncIterator[str]:
    """Yield the ``data`` of each server-sent event."""
    data: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if data:
                yield "\n".join(data)
                data = []
            continue
        if line == "data" or line.startswith("data:"):
            value = line[5:]
            data.append(value[1:] if value.startswith(" ") else value)
    if data:
        yield "\n".join(data)


def _error_message(data: dict[str, Any]) -> str:
    error = data.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    if isinstance(error, str):
        return error
    return ""


def _error_code(data: dict[str, Any]) -> str | None:
    error = data.get("error")
    if isinstance(error, dict) and isinstance(error.get("code"), str):
        return error["code"]
    return None


def _retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        delay = float(value)
    except ValueError:
        return None
    return delay if delay >= 0 else None


def _backoff(attempt: int) -> float:
    return min(0.5 * 2**attempt, 8.0) * (0.75 + random.random() * 0.25)
