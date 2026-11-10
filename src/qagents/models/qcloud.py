"""QCloud, through the official ``qcloud_ai`` SDK.

The SDK keeps ownership of transport: authentication, connection pooling,
its retry policy and stream validation. This adapter adds what an agent loop
needs on top:

* translation between qagents messages and Chat Completions;
* a local check of request fields against QCloud's published allow-list, so a
  typo fails here with a clear message instead of as an HTTP 400;
* the model's catalog entry (capabilities, limits, prices), fetched once, used
  to fail fast on unsupported features and to put a cost on every response;
* QCloud's typed errors mapped to qagents errors, keeping the request id.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

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

_INSTALL_HINT = (
    "The QCloud provider needs the qcloud-ai SDK. Until it is published on PyPI, install it "
    'from source: pip install "git+https://github.com/qcloud-host/qcloud-python.git"'
)


class QCloudModel(Model):
    """A model from the QCloud catalog.

    ``client`` is an ``qcloud_ai.AsyncQCloud`` you already have; share one per
    process. Without it, a client is created from ``api_key`` / ``base_url`` /
    ``organization`` or the ``QCLOUD_*`` environment variables.

    With ``use_catalog=True`` (the default) the first request also reads the
    model's catalog entry once. Set it to ``False`` to skip that request; cost
    is then not computed and capability checks are left to the API.
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
        self._capabilities: ModelCapabilities | None = None

    @property
    def client(self) -> Any:
        """The ``AsyncQCloud`` client, created on first use."""
        if self._client is None:
            try:
                from qcloud_ai import AsyncQCloud
            except ImportError as error:
                raise ConfigurationError(_INSTALL_HINT) from error
            options = {key: value for key, value in self._options.items() if value is not None}
            self._client = AsyncQCloud(**options)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.close()
            self._client = None

    def validate_settings(self, settings: ModelSettings) -> None:
        unknown = sorted(set(settings.extra) - ALLOWED_PARAMS)
        if unknown:
            raise ModelBadRequestError(
                f"QCloud does not accept the request field(s) {', '.join(unknown)}. "
                f"Accepted fields: {', '.join(sorted(ALLOWED_PARAMS))}.",
                provider=self.provider,
                model=self.name,
            )

    async def capabilities(self) -> ModelCapabilities:
        if self._capabilities is not None:
            return self._capabilities
        if not self._use_catalog:
            return ModelCapabilities()
        try:
            entry = await self.client.models.retrieve(self.name)
        except ConfigurationError:
            raise
        except Exception as error:
            mapped = self._map_error(error)
            if isinstance(mapped, ModelBadRequestError | ModelAuthenticationError):
                raise mapped from error
            # The catalog being unreachable must not block inference.
            logger.warning("Could not read the QCloud catalog entry for %s: %s", self.name, error)
            return ModelCapabilities()
        listed = set(entry.capabilities or [])
        pricing = entry.pricing or {}
        self._capabilities = ModelCapabilities(
            # An empty list means the catalog did not say, not that nothing is supported.
            tools=("tools" in listed) if listed else None,
            vision=("vision" in listed) if listed else None,
            context_length=entry.context_length,
            max_output_tokens=entry.max_output_tokens,
            input_price_per_million=_number(pricing.get("input_per_million")),
            output_price_per_million=_number(pricing.get("output_per_million")),
        )
        return self._capabilities

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        body = build_chat_body(self.name, request, stream=True)
        unknown = sorted(set(body) - ALLOWED_PARAMS)
        if unknown:
            raise ModelBadRequestError(
                f"QCloud does not accept the request field(s) {', '.join(unknown)}. "
                f"Accepted fields: {', '.join(sorted(ALLOWED_PARAMS))}.",
                provider=self.provider,
                model=self.name,
            )
        capabilities = await self.capabilities()
        self._check_supported(request, capabilities)

        params = {
            key: value for key, value in body.items() if key not in ("model", "messages", "stream")
        }
        accumulator = ChatAccumulator(provider=self.provider, model=self.name)
        try:
            stream = await self.client.chat.completions.create(
                model=self.name, messages=body["messages"], stream=True, **params
            )
            async with stream:
                async for chunk in stream:
                    for event in accumulator.feed(chunk.model_dump()):
                        yield event
        except (ModelError, ConfigurationError):
            raise
        except Exception as error:
            raise self._map_error(error) from error
        response = accumulator.result()
        response.usage.cost_usd = capabilities.cost(response.usage)
        yield ResponseDone(response=response)

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

    def _map_error(self, error: Exception) -> Exception:
        """Turn a ``qcloud_ai`` exception into the matching qagents error."""
        try:
            import qcloud_ai
        except ImportError:
            return error
        details: dict[str, Any] = {"provider": self.provider, "model": self.name}
        if isinstance(error, qcloud_ai.APIError):
            details.update(
                status_code=error.status_code,
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
            if error.status_code in (400, 404, 422):
                return ModelBadRequestError(message, **details)
            # A 5xx, a 409, or an error event inside a started stream: the
            # provider may have produced billable output already.
            return ModelServerError(message, retryable=True, ambiguous=True, **details)
        # The SDK already retried the failures that provably never reached the
        # API. What surfaces here happened after the request was sent.
        if isinstance(error, qcloud_ai.APITimeoutError):
            return ModelTimeoutError(str(error), ambiguous=True, **details)
        if isinstance(error, qcloud_ai.APIConnectionError):
            return ModelConnectionError(str(error), ambiguous=True, **details)
        if isinstance(error, qcloud_ai.APIResponseValidationError):
            return ModelResponseError(str(error), ambiguous=True, **details)
        return error


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if value >= 0 else None
