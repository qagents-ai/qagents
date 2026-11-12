"""QCloud, through the official ``qcloud_ai`` SDK.

The SDK keeps ownership of transport: authentication, connection pooling,
its retry policy and stream validation. This adapter adds what an agent loop
needs on top:

* translation between qagents messages and Chat Completions;
* a local check of request fields against QCloud's published allow-list, and
  of their values against its published limits, so a typo fails here with a
  clear message instead of as an HTTP 400;
* the model's catalog entry (capabilities, limits, prices), fetched once, used
  to fail fast on unsupported features and to put a cost on every response;
* QCloud's typed errors mapped to qagents errors, keeping the request id.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import weakref
from collections.abc import AsyncIterator
from typing import Any

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
from ..messages import ImagePart
from ._openai_wire import ChatAccumulator, build_chat_body
from .base import (
    Model,
    ModelCapabilities,
    ModelRequest,
    ModelSettings,
    ModelStreamEvent,
    ResponseDone,
)

logger = logging.getLogger("qagents.qcloud")

ALLOWED_PARAMS = frozenset(
    {
        "model", "messages", "stream", "stream_options", "max_tokens",
        "max_completion_tokens", "n", "temperature", "top_p", "frequency_penalty",
        "presence_penalty", "stop", "seed", "logit_bias", "logprobs", "top_logprobs",
        "user", "tools", "tool_choice", "parallel_tool_calls", "response_format",
    }
)  # fmt: skip
"""Request fields QCloud's Chat Completions endpoint accepts. Anything else is
rejected by the API with HTTP 400 before dispatch (sdk-compatibility.md)."""

CATALOG_RETRY_S = 300.0
"""After a failed catalog lookup, how long requests go on without the entry
before the lookup is tried again."""

_INSTALL_HINT = (
    "The QCloud provider needs the qcloud-ai SDK. Until it is published on PyPI, install it "
    'from source: pip install "git+https://github.com/qcloud-host/qcloud-python.git"'
)

# Failures while a connection was being set up: the request was never sent.
_NEVER_SENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)


class QCloudModel(Model):
    """A model from the QCloud catalog.

    ``client`` is an ``qcloud_ai.AsyncQCloud`` you already have; share one per
    process. Without it, a client is created from ``api_key`` / ``base_url`` /
    ``organization`` or the ``QCLOUD_*`` environment variables.

    With ``use_catalog=True`` (the default) the first request also reads the
    model's catalog entry, once. A lookup that fails or takes longer than
    ``catalog_timeout`` seconds does not hold up inference: requests go on
    without cost or capability checks, and the lookup is tried again after
    `CATALOG_RETRY_S`. Set ``use_catalog=False`` to skip it; cost is then not
    computed and capability checks are left to the API.
    """

    provider = "qcloud"

    def __init__(
        self,
        model: str,
        *,
        client: Any = None,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        use_catalog: bool = True,
        catalog_timeout: float = 10.0,
    ) -> None:
        self.name = model
        self._client = client
        self._owns_client = client is None
        self._options = {
            "api_key": api_key,
            "base_url": base_url,
            "organization": organization,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        self._use_catalog = use_catalog
        self._catalog_timeout = catalog_timeout
        self._capabilities: ModelCapabilities | None = None
        self._catalog_retry_at = 0.0
        self._catalog_lock: tuple[weakref.ref[Any], asyncio.Lock] | None = None
        self._warned_usage = False

    @property
    def client(self) -> Any:
        """The ``AsyncQCloud`` client, created on first use.

        Raises:
            ConfigurationError: The SDK is not installed, or the client cannot
                be created (no API key, for example).
        """
        if self._client is None:
            try:
                from qcloud_ai import AsyncQCloud
            except ImportError as error:
                raise ConfigurationError(_INSTALL_HINT) from error
            options = {key: value for key, value in self._options.items() if value is not None}
            try:
                self._client = AsyncQCloud(**options)
            except (ValueError, TypeError) as error:
                raise ConfigurationError(
                    f"The QCloud client could not be created: {error}"
                ) from error
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.close()
            self._client = None

    def validate_settings(self, settings: ModelSettings) -> None:
        """Check the field names and values of ``settings`` against what QCloud accepts.

        Raises:
            ModelBadRequestError: A field is not on `ALLOWED_PARAMS`, or a value
                is outside QCloud's published limits.
        """
        unknown = sorted(set(settings.extra) - ALLOWED_PARAMS)
        if unknown:
            raise self._unknown_fields(unknown)
        self._check_values(settings)

    async def capabilities(self) -> ModelCapabilities:
        if self._capabilities is not None:
            return self._capabilities
        if not self._use_catalog:
            return ModelCapabilities()
        async with self._lock():  # concurrent first requests share one lookup
            if self._capabilities is not None:
                return self._capabilities
            if time.monotonic() < self._catalog_retry_at:
                return ModelCapabilities()
            client = self.client
            try:
                entry = await asyncio.wait_for(
                    client.models.retrieve(self.name), self._catalog_timeout
                )
            except TimeoutError:
                failure = f"no answer within catalog_timeout={self._catalog_timeout:g} s"
            except Exception as error:
                mapped = self._map_error(error)
                if isinstance(mapped, ModelBadRequestError | ModelAuthenticationError):
                    raise mapped from error
                failure = str(error)
            else:
                self._capabilities = _capabilities(entry)
                return self._capabilities
            # The catalog being unreachable must not block inference.
            self._catalog_retry_at = time.monotonic() + CATALOG_RETRY_S
            logger.warning(
                "Could not read the QCloud catalog entry for %s (%s). Requests go on without "
                "cost or capability checks; the lookup is tried again in %g s.",
                self.name,
                failure,
                CATALOG_RETRY_S,
            )
            return ModelCapabilities()

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        body = build_chat_body(self.name, request, stream=True)
        unknown = sorted(set(body) - ALLOWED_PARAMS)
        if unknown:
            raise self._unknown_fields(unknown)
        self._check_values(request.settings)
        capabilities = await self.capabilities()
        self._check_supported(request, capabilities)

        client = self.client
        params = {
            key: value for key, value in body.items() if key not in ("model", "messages", "stream")
        }
        accumulator = ChatAccumulator(provider=self.provider, model=self.name)
        started = False
        try:
            stream = await client.chat.completions.create(
                model=self.name, messages=body["messages"], stream=True, **params
            )
            started = True
            async with stream:
                async for chunk in stream:
                    for event in accumulator.feed(chunk.model_dump()):
                        yield event
        except (ModelError, ConfigurationError):
            raise
        except Exception as error:
            raise self._map_error(error, started=started) from error
        if accumulator.failed:
            raise ModelResponseError(
                f"QCloud ended the response with finish_reason {accumulator.finish_reason!r}: "
                "the generation failed.",
                provider=self.provider,
                model=self.name,
                ambiguous=True,
            )
        response = accumulator.result()
        if accumulator.usage_reported:
            response.usage.cost_usd = capabilities.cost(response.usage)
        else:
            # Unknown cost: Limits.max_cost_usd fails closed on it (Usage.unpriced_requests).
            response.usage.unpriced_requests = 1
            if not self._warned_usage:
                self._warned_usage = True
                logger.warning(
                    "QCloud did not report token usage for %s. Such responses count as 0 "
                    "tokens with an unknown cost: token limits do not see them, and a run with "
                    "a cost limit stops at the first one.",
                    self.name,
                )
        yield ResponseDone(response=response)

    def _lock(self) -> asyncio.Lock:
        """The catalog lock of the running event loop."""
        loop = asyncio.get_running_loop()
        held = self._catalog_lock
        if held is None or held[0]() is not loop:
            held = self._catalog_lock = (weakref.ref(loop), asyncio.Lock())
        return held[1]

    def _unknown_fields(self, unknown: list[str]) -> ModelBadRequestError:
        return ModelBadRequestError(
            f"QCloud does not accept the request field(s) {', '.join(unknown)}. "
            f"Accepted fields: {', '.join(sorted(ALLOWED_PARAMS))}.",
            provider=self.provider,
            model=self.name,
        )

    def _check_values(self, settings: ModelSettings) -> None:
        """Reject values QCloud answers with HTTP 400, before anything is sent."""
        sent: dict[str, Any] = {
            key: value
            for key, value in (
                ("max_tokens", settings.max_tokens),
                ("temperature", settings.temperature),
                ("top_p", settings.top_p),
                ("stop", settings.stop or None),
                ("seed", settings.seed),
                ("parallel_tool_calls", settings.parallel_tool_calls),
            )
            if value is not None
        }
        sent.update(settings.extra)  # extra fields are sent last and win
        problems: list[str] = []
        for key, low, high in (
            ("temperature", 0, 2),
            ("top_p", 0, 1),
            ("frequency_penalty", -2, 2),
            ("presence_penalty", -2, 2),
        ):
            if key in sent and not _number_in(sent[key], low, high):
                problems.append(f"{key} must be a number from {low} to {high}")
        stop = sent.get("stop")
        if stop is not None and not (
            isinstance(stop, str)
            or (isinstance(stop, list) and len(stop) <= 4 and all(isinstance(s, str) for s in stop))
        ):
            problems.append("stop must be a string or a list of at most 4 strings")
        if "max_tokens" in sent and "max_completion_tokens" in sent:
            problems.append(
                "give only one of max_tokens and max_completion_tokens "
                "(ModelSettings.max_tokens is sent as max_tokens)"
            )
        for key, low, high, limits in (
            ("max_tokens", 1, None, " of at least 1"),
            ("max_completion_tokens", 1, None, " of at least 1"),
            ("n", 1, 8, " from 1 to 8"),
            ("top_logprobs", 0, 20, " from 0 to 20"),
            ("seed", None, None, ""),
        ):
            if key in sent and not _integer_in(sent[key], low, high):
                problems.append(f"{key} must be an integer{limits}")
        for key in ("logprobs", "parallel_tool_calls"):
            if key in sent and not isinstance(sent[key], bool):
                problems.append(f"{key} must be true or false")
        user = sent.get("user")
        if user is not None and not (isinstance(user, str) and len(user) <= 256):
            problems.append("user must be a string of at most 256 characters")
        response_format = sent.get("response_format")
        if response_format is not None and not (
            isinstance(response_format, dict)
            and response_format.get("type") in ("text", "json_object", "json_schema")
        ):
            problems.append("response_format.type must be 'text', 'json_object' or 'json_schema'")
        if problems:
            raise ModelBadRequestError(
                f"QCloud would reject these settings: {'; '.join(problems)}.",
                provider=self.provider,
                model=self.name,
            )

    def _check_supported(self, request: ModelRequest, capabilities: ModelCapabilities) -> None:
        if request.tools and capabilities.tools is False:
            raise ModelBadRequestError(
                f"The QCloud catalog does not list the 'tools' capability for {self.name}. "
                "Pick a model that supports tool calling, or remove the agent's tools.",
                provider=self.provider,
                model=self.name,
            )
        if capabilities.vision is False and any(
            isinstance(part, ImagePart) for message in request.messages for part in message.parts
        ):
            raise ModelBadRequestError(
                f"The QCloud catalog does not list the 'vision' capability for {self.name}, "
                "but the conversation contains an image.",
                provider=self.provider,
                model=self.name,
            )

    def _map_error(self, error: Exception, *, started: bool = False) -> Exception:
        """Turn a ``qcloud_ai`` exception into the matching qagents error.

        ``started`` is true once QCloud has answered and the stream is being
        read: from then on every failure may follow billable output.
        """
        try:
            import qcloud_ai
        except ImportError:
            return error
        details: dict[str, Any] = {"provider": self.provider, "model": self.name}
        if isinstance(error, qcloud_ai.APIError):
            status = error.status_code
            details.update(
                status_code=status,
                code=error.code,
                request_id=error.request_id,
                retry_after=error.retry_after,
            )
            message = str(error)
            if isinstance(error, qcloud_ai.RateLimitError):
                return ModelRateLimitError(message, retryable=True, **details)
            if isinstance(error, qcloud_ai.InsufficientCreditsError):
                return ModelBillingError(message, **details)
            if isinstance(error, qcloud_ai.AuthenticationError | qcloud_ai.PermissionDeniedError):
                return ModelAuthenticationError(message, **details)
            if status < 300 or status >= 500 or status == 409:
                # An error event inside a started stream (reported as 200), a 5xx
                # or a 409: the provider may have produced billable output already.
                return ModelServerError(message, retryable=True, ambiguous=True, **details)
            if status == 408:
                return ModelTimeoutError(message, retryable=True, ambiguous=True, **details)
            if status < 400:
                return ModelBadRequestError(
                    f"{message} (QCloud answered with a redirect, which is not followed; "
                    "check base_url)",
                    **details,
                )
            return ModelBadRequestError(message, **details)  # 400, 404, 405, 413, 422, ...
        if isinstance(error, qcloud_ai.APIConnectionError):  # timeouts included
            kind = (
                ModelTimeoutError
                if isinstance(error, qcloud_ai.APITimeoutError)
                else ModelConnectionError
            )
            # The SDK sets __cause__ for every transport failure; .cause not for timeouts.
            cause = error.__cause__ or getattr(error, "cause", None)
            if not started and isinstance(cause, _NEVER_SENT):
                # The SDK has used up its retries on a connection that was never made.
                return kind(f"Could not connect to QCloud: {cause}", retryable=True, **details)
            return kind(str(error), ambiguous=True, **details)
        if isinstance(error, qcloud_ai.APIResponseValidationError):
            return ModelResponseError(str(error), ambiguous=True, **details)
        if started and isinstance(error, ValueError | TypeError):
            # A chunk the SDK could not read (pydantic's ValidationError is a ValueError).
            return ModelResponseError(
                f"QCloud sent a chunk that could not be read: {error}", ambiguous=True, **details
            )
        return error


def _capabilities(entry: Any) -> ModelCapabilities:
    listed = set(entry.capabilities or [])
    pricing = entry.pricing or {}
    return ModelCapabilities(
        # An empty list means the catalog did not say, not that nothing is supported.
        tools=("tools" in listed) if listed else None,
        vision=("vision" in listed) if listed else None,
        context_length=entry.context_length,
        max_output_tokens=entry.max_output_tokens,
        input_price_per_million=_number(pricing.get("input_per_million")),
        output_price_per_million=_number(pricing.get("output_per_million")),
    )


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if value >= 0 else None


def _number_in(value: Any, low: float, high: float) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return math.isfinite(value) and low <= value <= high


def _integer_in(value: Any, low: int | None, high: int | None) -> bool:
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    return (low is None or value >= low) and (high is None or value <= high)
