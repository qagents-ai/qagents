"""A model on any server that speaks OpenAI Chat Completions.

Works with vLLM, SGLang, llama.cpp, Ollama, OpenAI itself and gateways in
front of them. It talks HTTP directly with ``httpx``; no vendor SDK is needed.

Retry policy (the same conservative one the QCloud SDK uses): a request is
sent again only when it provably never ran, that is, on a 429 or when the
connection could not be established. A timeout, a reset, a 408, a 409 or a 5xx
after the request was sent is raised as ``ambiguous`` and left to the caller.
Other 4xx answers and redirects are bad requests: neither retried nor ambiguous.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import AsyncIterator, Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from ..errors import (
    ConfigurationError,
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
from .base import (
    Model,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ResponseDone,
)

logger = logging.getLogger("qagents.openai_chat")

_NEVER_SENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)

DEFAULT_TIMEOUT = 600.0
"""Seconds allowed for connecting, and between two chunks of a response."""

_HTTPX_DEFAULT_TIMEOUT = httpx.Timeout(5.0)
_EXCERPT_CHARS = 300


class _Unset:
    def __repr__(self) -> str:
        return "DEFAULT_TIMEOUT"


_UNSET = _Unset()


class OpenAIChatModel(Model):
    """A model behind an OpenAI-compatible ``/chat/completions`` endpoint.

    Args:
        model: The id the server knows the model by.
        base_url: The API root, for example ``http://localhost:8000/v1``.
        api_key: Sent as ``Authorization: Bearer ...``. Surrounding whitespace
            (a trailing newline from a file) is removed.
        headers: Extra HTTP headers.
        timeout: Applies to every request, also through ``http_client``.
            Without it, the adapter's own client uses `DEFAULT_TIMEOUT`, and an
            ``http_client`` keeps its own timeout, unless that is httpx's 5 s
            default, which is too short for generation and is replaced by
            `DEFAULT_TIMEOUT`.
        max_retries: How often a request that provably never ran is sent again.
        http_client: An ``httpx.AsyncClient`` to send requests with (proxies,
            certificates). The adapter does not close it.
        provider: The provider label used in errors, traces and responses.
        max_tokens_param: ``"max_completion_tokens"`` for servers that require it.
        include_usage: Ask for token usage in the stream. Set ``False`` for
            servers that reject ``stream_options``; usage is then unknown, so
            ``cost_usd`` is ``None`` and token limits do not see the requests.
        capabilities: What is known about the model, prices included.

    Raises:
        ConfigurationError: ``base_url`` is not an http(s) URL, or ``api_key``
            or a header contains a line break, another control character or a
            character that is not ASCII. The value is never quoted.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | httpx.Timeout | None | _Unset = _UNSET,
        max_retries: int = 2,
        http_client: httpx.AsyncClient | None = None,
        provider: str = "openai-compatible",
        max_tokens_param: str = "max_tokens",
        include_usage: bool = True,
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        if not base_url.lower().startswith(("http://", "https://")):
            raise ConfigurationError(
                f"base_url must start with http:// or https://, not {_shown(base_url)!r}."
            )
        self.name = model
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._headers = {"Content-Type": "application/json", **(headers or {})}
        # Checked now, and never quoted: httpx would fail later, quoting the value.
        for name, value in self._headers.items():
            problem = _header_problem(name) or _header_problem(value)
            if problem:
                raise ConfigurationError(
                    f"The header {str(name).encode('ascii', 'replace').decode()!r} contains "
                    f"{problem}, which an HTTP header cannot carry."
                )
        api_key = api_key.strip() if api_key else None
        if api_key:
            problem = _header_problem(api_key)
            if problem:
                raise ConfigurationError(
                    f"api_key contains {problem}, which an HTTP header cannot carry. "
                    "Check how the key is read or pasted."
                )
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._timeout = timeout
        self._client = http_client
        self._owns_client = http_client is None
        self._max_tokens_param = max_tokens_param
        self._include_usage = include_usage
        self._capabilities = capabilities or ModelCapabilities()
        self._warned_usage = False

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
        content_type = response.headers.get("content-type", "")
        if "json" in content_type and "event-stream" not in content_type:
            # Not a stream: an error, or a whole completion from a server that ignores
            # "stream". Or a stream with the wrong label, read below from the body.
            completion = await self._json_body(response)
            if completion is not None:
                events = accumulator.feed_completion(completion)
                if accumulator.finish_reason is None:
                    # No first choice, or one without a finish_reason: it may be cut off.
                    raise self._error(
                        ModelResponseError,
                        "The server's completion has no choice that says how it finished.",
                        response=response,
                        ambiguous=True,
                    )
                for event in events:
                    yield event
                yield ResponseDone(response=self._result(accumulator, response))
                return
        finished = False
        payloads = 0
        ignored: list[str] = []
        try:
            async for payload in _sse_payloads(response, ignored):
                payloads += 1
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
            message = "The response stream ended before the model finished."
            excerpt = _excerpt(" ".join(ignored))
            if payloads == 0 and "event-stream" in content_type:
                message = "The event stream ended before the server sent any data."
                if excerpt:
                    message += f" It sent: {excerpt}"
            elif payloads == 0:
                message = (
                    f"The server answered with content type {content_type or '(none)'!r} "
                    f"instead of an event stream: {excerpt or 'an empty body'}"
                )
            raise self._error(ModelResponseError, message, response=response, ambiguous=True)
        yield ResponseDone(response=self._result(accumulator, response))

    def _result(self, accumulator: ChatAccumulator, response: httpx.Response) -> ModelResponse:
        if accumulator.failed:
            raise self._error(
                ModelResponseError,
                f"The server ended the response with finish_reason "
                f"{accumulator.finish_reason!r}: the generation failed.",
                response=response,
                ambiguous=True,
            )
        result = accumulator.result()
        if accumulator.usage_reported:
            result.usage.cost_usd = self._capabilities.cost(result.usage)
        else:
            # Unknown cost: Limits.max_cost_usd fails closed on it (Usage.unpriced_requests).
            result.usage.unpriced_requests = 1
            if not self._warned_usage:
                self._warned_usage = True
                logger.warning(
                    "%s:%s did not report token usage. Its responses count as 0 tokens with an "
                    "unknown cost: token limits do not see them, and a run with a cost limit "
                    "stops at the first one.",
                    self.provider,
                    self.name,
                )
        return result

    async def _json_body(self, response: httpx.Response) -> dict[str, Any] | None:
        """The completion a server sent instead of a stream, or the error it reports.

        ``None`` when the body is an event stream after all; it has been read,
        and ``response`` still yields it.
        """
        try:
            raw = await response.aread()
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
        try:
            data = json.loads(raw)
        except ValueError:
            if any(line.startswith(b"data:") for line in raw.splitlines()):
                return None  # an event stream labelled as JSON by a server or proxy
            data = None
        if isinstance(data, dict) and data.get("error"):
            raise self._error(
                ModelServerError,
                _error_message(data) or _excerpt(raw.decode("utf-8", errors="replace")),
                response=response,
                code=_error_code(data),
                ambiguous=True,
            )
        if not isinstance(data, dict) or not isinstance(data.get("choices"), list):
            raise self._error(
                ModelResponseError,
                "The server answered with JSON that is neither a stream nor a completion: "
                f"{_excerpt(raw.decode('utf-8', errors='replace'))}",
                response=response,
                ambiguous=True,
            )
        return data

    def _request_timeout(self, client: httpx.AsyncClient) -> float | httpx.Timeout | None:
        if not isinstance(self._timeout, _Unset):
            return self._timeout
        if self._owns_client or client.timeout == _HTTPX_DEFAULT_TIMEOUT:
            return DEFAULT_TIMEOUT
        return client.timeout

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=DEFAULT_TIMEOUT if isinstance(self._timeout, _Unset) else self._timeout,
                follow_redirects=False,
            )
        return self._client

    async def _send(self, body: dict[str, Any]) -> httpx.Response:
        client = self._http()
        url = f"{self.base_url}/chat/completions"
        for attempt in range(self.max_retries + 1):
            last = attempt == self.max_retries
            try:
                request = client.build_request(
                    "POST",
                    url,
                    json=body,
                    headers=self._headers,
                    timeout=self._request_timeout(client),
                )
                response = await client.send(request, stream=True)
            # Raised before anything was sent. The messages are fixed: the
            # original ones can quote a header, and so the API key.
            except httpx.UnsupportedProtocol:
                raise ConfigurationError(
                    f"base_url must start with http:// or https://, not {_shown(self.base_url)!r}."
                ) from None
            except httpx.InvalidURL:
                raise ConfigurationError(
                    f"base_url {_shown(self.base_url)!r} is not a valid URL."
                ) from None
            except httpx.LocalProtocolError:
                raise ConfigurationError(
                    "The request could not be sent: a header name or value is not valid HTTP. "
                    "Check api_key and headers for line breaks or other control characters."
                ) from None
            except _NEVER_SENT as error:
                if not last:
                    await asyncio.sleep(_backoff(attempt))
                    continue
                raise self._error(
                    ModelConnectionError,
                    f"Could not connect to {_shown(self.base_url)}: {error}",
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
            if response.status_code < 300:
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
        status = response.status_code
        if 300 <= status < 400:
            # Not followed: a POST must not be re-sent, with the key, to another address.
            return self._error(
                ModelBadRequestError,
                f"[{status}] {_shown(self.base_url)} redirected the request to "
                f"{response.headers.get('location') or 'another address'}. Set base_url to "
                "the final address; redirects are not followed.",
                response=response,
            )
        try:
            data = json.loads(raw)
        except ValueError:
            data = None
        message = _error_message(data) if isinstance(data, dict) else ""
        if not message:
            message = raw.decode("utf-8", errors="replace")[:500] or response.reason_phrase
        message = f"[{status}] {message}"
        code = _error_code(data) if isinstance(data, dict) else None
        if status in (401, 403):
            return self._error(ModelAuthenticationError, message, response=response, code=code)
        if status == 402:
            return self._error(ModelBillingError, message, response=response, code=code)
        if status == 429:
            return self._error(
                ModelRateLimitError, message, response=response, code=code, retryable=True
            )
        if status == 408:
            return self._error(
                ModelTimeoutError, message, response=response, code=code, retryable=True,
                ambiguous=True,
            )  # fmt: skip
        if 400 <= status < 500 and status != 409:
            return self._error(ModelBadRequestError, message, response=response, code=code)
        # A 5xx, or a 409: the server may have run the request.
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


async def _sse_payloads(
    response: httpx.Response, ignored: list[str] | None = None
) -> AsyncIterator[str]:
    """Yield the ``data`` of each server-sent event.

    The first few lines that are not part of an event go to ``ignored``, so
    an error page sent instead of a stream can be quoted.
    """
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
        elif ignored is not None and len(ignored) < 5 and not line.startswith(":"):
            ignored.append(line)
    if data:
        yield "\n".join(data)


def _shown(url: str) -> str:
    """``url`` without user name, password, query or fragment, for error messages.

    Error texts reach logs, traces, the journal and the model, so credentials
    written into ``base_url`` must not appear in them.
    """
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        port = f":{parts.port}" if parts.port is not None else ""
    except ValueError:
        return "base_url"
    return urlunsplit((parts.scheme, host + port, parts.path, "", ""))


def _header_problem(text: str | bytes) -> str | None:
    """Why ``text`` cannot go into an HTTP header, or ``None`` when it can."""
    if isinstance(text, bytes):  # httpx sends bytes as they are
        text = text.decode("latin-1")
    if any((ord(char) < 32 and char != "\t") or ord(char) == 127 for char in text):
        return "a line break or another control character"
    if not text.isascii():
        return "a character that is not ASCII (a typographic quote or a non-breaking space?)"
    return None


def _excerpt(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= _EXCERPT_CHARS else f"{flat[:_EXCERPT_CHARS]}..."


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
