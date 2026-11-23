"""Claude, through the Messages API and the official ``anthropic`` SDK.

The SDK keeps ownership of transport and authentication. This adapter adds
what the agent loop needs on top:

* translation between qagents messages and Messages API content blocks, with
  thinking blocks kept byte for byte; on the models that run the API's
  history-editing check (`PREFIX_CHECK_MODELS`) a block is replayed only while
  the conversation before it is unchanged;
* prompt caching with a stable prefix: tools, then system, then the
  conversation;
* the request checks the API would otherwise fail with a 400, made locally
  before anything is sent;
* stream events mapped to qagents events, usage with cache reads and writes,
  and a cost from a price table;
* SDK errors mapped to qagents errors with the request id and the
  ``retryable`` / ``ambiguous`` flags the loop acts on.

The ``anthropic`` package is an optional dependency. Importing this module
does not import it; the first request does.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from .. import prompts
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
from ..messages import (
    ImagePart,
    Message,
    Part,
    ReasoningPart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    Usage,
)
from .base import (
    FinishReason,
    Model,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelSettings,
    ModelStreamEvent,
    ReasoningDelta,
    ResponseDone,
    TextDelta,
    ToolCallDelta,
)

if TYPE_CHECKING:
    import anthropic

logger = logging.getLogger("qagents.anthropic")

Effort = Literal["low", "medium", "high", "xhigh", "max"]
ThinkingDisplay = Literal["summarized", "omitted", "updates"]

DEFAULT_MODEL = "claude-opus-5-5"
"""The model used when none is given: the current default Claude model."""

DEFAULT_MAX_TOKENS = 64_000
"""``max_tokens`` sent when ``ModelSettings.max_tokens`` is not set.

Thinking counts toward ``max_tokens``, and current models think on every
request, so the limit is sized for thinking plus the reply. It is capped at the
model's maximum output when the model is known."""

FALLBACK_BETA = "server-side-fallback-2026-07-01"
"""Beta header for server-side ``fallbacks`` (both the ``"default"`` and the list form)."""

THINKING_UPDATES_BETA = "thinking-display-updates-2026-08-18"
"""Beta header for ``thinking.display = "updates"`` (progress updates between tool calls)."""

THINKING_UPDATES_MODELS = frozenset(
    {
        "claude-fable-5", "claude-fable-5-1", "claude-fable-5-2", "claude-mythos-5-1",
        "claude-mythos-5-2", "claude-opus-5-5", "claude-sonnet-5-5",
    }
)  # fmt: skip
"""Known models that accept ``thinking.display = "updates"``. On another model in
`MODEL_LIMITS`, ``thinking_display="updates"`` fails when the model is built."""

FALLBACK_MODELS = frozenset(
    {
        "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5",
        "claude-fable-5-1", "claude-fable-5-2", "claude-mythos-5-1", "claude-mythos-5-2",
    }
)  # fmt: skip
"""Models with safety classifiers that support server-side fallback. ``fallbacks="default"``
is sent only to these."""

NO_SERVER_FALLBACK = frozenset({"claude-haiku-5-5"})
"""Models without server-side fallback: ``fallbacks: "default"`` does nothing on them and a
list of fallback models is a 400, so a list fails when the model or agent is built."""

PREFIX_CHECK_MODELS = frozenset(
    {
        "claude-fable-5-1", "claude-fable-5-2", "claude-mythos-5-1", "claude-mythos-5-2",
        "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5",
    }
)  # fmt: skip
"""Models that run the API's history-editing check on replayed thinking blocks: a block is
valid only while the system prompt, the tools and every earlier message are unchanged. Other
models, and every request with manual thinking (``thinking.type = "enabled"``, which none of
these models accepts), get every block back unchanged."""

NO_EFFORT = frozenset({"claude-haiku-4-5"})
"""Models that reject ``output_config.effort``."""

NO_XHIGH_EFFORT = frozenset({"claude-opus-4-6", "claude-sonnet-4-6"})
"""Models released before the ``xhigh`` effort level."""

MANUAL_THINKING_ONLY = frozenset({"claude-haiku-4-5"})
"""Models with manual extended thinking only (``budget_tokens``): adaptive thinking, which
``thinking_display`` sends, is rejected."""

NO_FORCED_TOOL_CHOICE = frozenset(
    {
        "claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-fable-5-2",
        "claude-mythos-5-1", "claude-mythos-5-2",
    }
)  # fmt: skip
"""Models that return a 400 for ``tool_choice`` ``any`` or ``tool``. A required or named
tool choice is sent to them as ``auto``, with a warning."""

NO_STRUCTURED_OUTPUT: frozenset[str] = frozenset()
"""Kept for backward compatibility; empty. Every current model supports structured outputs
(``output_config.format``). Where an organization cannot use them, the API's 400 is raised
as `ModelBadRequestError`."""

ALLOWED_EXTRA = frozenset(
    {
        "betas", "cache_control", "compaction", "container", "context_management",
        "diagnostics", "fallback_credit_token", "fallbacks", "inference_geo", "metadata",
        "output_config", "service_tier", "speed", "stop_sequences", "thinking",
        "user_profile_id", "workspace_id",
    }
)  # fmt: skip
"""``ModelSettings.extra`` keys passed to the Messages API as they are. ``model``,
``messages``, ``system``, ``tools``, ``tool_choice`` and ``max_tokens`` are built from the
request and its settings, so they cannot be set here."""

_UNSUPPORTED_EXTRA = {
    "thread": "this adapter sends the whole conversation, the system prompt and the tools on "
    "every request, so a server-side thread cannot be continued",
    "mcp_servers": "the MCP connector needs an mcp_toolset entry in tools, which this adapter "
    "does not build; use qagents.toolkits.mcp.MCPToolkit",
}

REQUEST_OPTIONS = frozenset({"extra_headers", "extra_query", "extra_body", "timeout"})
"""SDK per-request options accepted in ``request_options`` besides `ALLOWED_EXTRA`."""

_REMOVED_SAMPLING = ("temperature", "top_p", "top_k")
_IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})
_DATA_URL = re.compile(r"^data:(?P<media_type>[^;,]+);base64,(?P<data>.*)$", re.DOTALL)
_DATE_SUFFIX = re.compile(r"-\d{8}$")
_DEFAULT_HOST = "https://api.anthropic.com"
_THINKING_TYPES = ("thinking", "redacted_thinking")
_TOOL_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_CUT_OFF = ("max_tokens", "model_context_window_exceeded")

_INSTALL_HINT = (
    "The Anthropic provider needs the anthropic SDK (2.x). Install it with "
    'pip install "anthropic>=2.8,<3", or the qagents extra that includes it.'
)
_MISSING_CREDENTIALS = (
    "No Anthropic credentials were found. Set the ANTHROPIC_API_KEY environment variable, "
    "run `ant auth login`, or pass api_key=... or client=anthropic.AsyncAnthropic(...) to "
    "AnthropicModel."
)


# --- model table and prices ---------------------------------------------------------------


class AnthropicPricing(BaseModel):
    """US dollars per million tokens for one model.

    ``input`` is the price of input tokens that were neither read from nor
    written to the prompt cache.
    """

    input: float
    output: float
    cache_read: float
    cache_write_5m: float
    """Price of tokens written to the cache with the default 5-minute lifetime."""
    cache_write_1h: float
    """Price of tokens written to the cache with the 1-hour lifetime."""
    long_context_threshold: int | None = None
    """Prompts with more input tokens than this are billed at ``long_context`` prices."""
    long_context: AnthropicPricing | None = None

    def cost(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_5m_tokens: int = 0,
        cache_write_1h_tokens: int = 0,
    ) -> float:
        """The cost in US dollars. ``input_tokens`` excludes cache reads and writes."""
        prompt = input_tokens + cache_read_tokens + cache_write_5m_tokens + cache_write_1h_tokens
        prices = self
        if (
            self.long_context is not None
            and self.long_context_threshold is not None
            and prompt > self.long_context_threshold
        ):
            prices = self.long_context
        return (
            input_tokens * prices.input
            + output_tokens * prices.output
            + cache_read_tokens * prices.cache_read
            + cache_write_5m_tokens * prices.cache_write_5m
            + cache_write_1h_tokens * prices.cache_write_1h
        ) / 1_000_000


def _prices(
    input: float, output: float, read: float, *, write_5m: float | None = None
) -> AnthropicPricing:
    return AnthropicPricing(
        input=input,
        output=output,
        cache_read=read,
        cache_write_5m=write_5m if write_5m is not None else input * 1.25,
        cache_write_1h=input * 2,
    )


PRICING: dict[str, AnthropicPricing] = {
    "claude-opus-5-5": _prices(4.0, 20.0, 0.20),
    "claude-opus-5": _prices(5.0, 25.0, 0.50),
    "claude-opus-4-8": _prices(5.0, 25.0, 0.50),
    "claude-fable-5-2": _prices(8.0, 40.0, 0.20),
    "claude-mythos-5-2": _prices(8.0, 40.0, 0.20),
    "claude-fable-5-1": _prices(10.0, 50.0, 0.25),
    "claude-mythos-5-1": _prices(10.0, 50.0, 0.25),
    "claude-fable-5": _prices(10.0, 50.0, 1.0),
    "claude-mythos-5": _prices(10.0, 50.0, 1.0),
    "claude-sonnet-5-5": _prices(2.0, 10.0, 0.10),
    "claude-sonnet-5": _prices(2.0, 10.0, 0.20),
    "claude-haiku-5-5": _prices(0.10, 0.50, 0.01).model_copy(
        update={"long_context_threshold": 100_000, "long_context": _prices(0.50, 2.50, 0.05)}
    ),
}
"""Standard API prices per million tokens, from Anthropic's published rates (November 2026).
Pass ``pricing=`` to `AnthropicModel` for other rates or models. Batch, fast mode and
regional prices are not covered."""

MODEL_LIMITS: dict[str, tuple[int, int]] = {
    "claude-fable-5-2": (1_000_000, 128_000),
    "claude-mythos-5-2": (1_000_000, 128_000),
    "claude-fable-5-1": (1_000_000, 128_000),
    "claude-mythos-5-1": (1_000_000, 128_000),
    "claude-fable-5": (1_000_000, 128_000),
    "claude-mythos-5": (1_000_000, 128_000),
    "claude-opus-5-5": (1_000_000, 128_000),
    "claude-opus-5": (1_000_000, 128_000),
    "claude-opus-4-8": (1_000_000, 128_000),
    "claude-opus-4-7": (1_000_000, 128_000),
    "claude-opus-4-6": (1_000_000, 128_000),
    "claude-sonnet-5-5": (1_000_000, 128_000),
    "claude-sonnet-5": (1_000_000, 128_000),
    "claude-sonnet-4-6": (1_000_000, 128_000),
    "claude-haiku-5-5": (1_000_000, 128_000),
    "claude-haiku-4-5": (200_000, 64_000),
}
"""Context window and maximum output tokens of current models. All of them accept images
and client tools."""


def _base_id(model: str) -> str:
    """``claude-haiku-4-5-20251001`` -> ``claude-haiku-4-5``, for table lookups."""
    return _DATE_SUFFIX.sub("", model)


def _open_maps(schema: Any, path: str) -> list[str]:
    """JSON paths of objects with free-form keys (``additionalProperties`` other than
    false, ``patternProperties``, or an object with no ``properties`` that is not
    closed). Objects with ``properties`` that only omit ``additionalProperties`` are
    not open maps: closing them keeps their declared properties."""
    if not isinstance(schema, dict):
        return []
    extra = schema.get("additionalProperties")
    kind = schema.get("type")
    bare = (
        (kind == "object" or (isinstance(kind, list) and "object" in kind))
        and "properties" not in schema
        and extra is not False
    )
    found = (
        [path]
        if (extra is not None and extra is not False) or "patternProperties" in schema or bare
        else []
    )
    for key in ("properties", "$defs", "definitions", "patternProperties"):
        if isinstance(schema.get(key), dict):
            for name, sub in schema[key].items():
                found += _open_maps(sub, f"{path}.{key}.{name}")
    for key in ("items", "additionalProperties", "not"):
        found += _open_maps(schema.get(key), f"{path}.{key}")
    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        if isinstance(schema.get(key), list):
            for position, sub in enumerate(schema[key]):
                found += _open_maps(sub, f"{path}.{key}[{position}]")
    return found


# --- the model ------------------------------------------------------------------------------


class AnthropicModel(Model):
    """A Claude model on the Claude API.

    Every request is streamed. ``client`` is an ``anthropic.AsyncAnthropic`` you
    already have; share one per process. Without it, a client is created on the
    first request from ``api_key`` / ``base_url`` / ``timeout`` or from the
    environment (``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN``, an
    ``ant auth login`` profile). Missing credentials fail on that first request,
    not when the model is built.

    Retries: the SDK's own retries are turned off (``max_retries=0``), also on a
    client you pass in. The SDK would repeat 5xx responses and timeouts, which may
    already have been processed and billed. The qagents loop repeats only errors
    that provably did not run (429, 529 overloaded, a connection that was never
    made), and never a request whose stream had started. Once one segment of a
    paused turn (``pause_turn``, ``compaction``) has run, every later error of
    that model request is ambiguous, since repeating it would run the billed
    segment again.

    Fallbacks: with ``fallbacks="default"`` (the default), requests to models
    with safety classifiers (`FALLBACK_MODELS`, which includes the default model)
    carry ``fallbacks: "default"`` and the ``server-side-fallback-2026-07-01``
    beta header. When the classifier declines, the API re-runs the request on
    the model Anthropic recommends for that category, inside the same call, and
    ``ModelResponse.model`` names the model that answered. Pass a list such as
    ``[{"model": "claude-opus-5"}]`` to choose the targets, or ``fallbacks=None``
    to turn fallback off; a declined request then ends with
    ``finish_reason="content_filter"``. Models in `NO_SERVER_FALLBACK` reject a
    list. Usage and cost come from ``usage.iterations`` when the API reports it,
    so declined attempts and server-side compaction are counted at their own
    model's prices.

    Thinking: thinking blocks are kept in ``ReasoningPart.opaque`` and sent back
    unchanged. The models in `PREFIX_CHECK_MODELS` run a history-editing check:
    a block is valid only while the system prompt, the tools and every earlier
    message are unchanged, and after an edit there (compaction that keeps recent
    turns, instructions that change between turns, a different tool set) the API
    may reject it (by default on accounts created on or after 2026-08-31, and on
    every account for Claude Fable 5.2 and Claude Mythos 5.1 / 5.2). For those
    models, and for model ids this adapter does not know, it records a
    fingerprint of that prefix with each block and leaves out the blocks whose
    prefix changed, so those requests still succeed without the old reasoning.
    Every other model, and every request with manual thinking
    (``thinking.type = "enabled"``, which an open tool round needs to keep its
    thinking block for), gets every block back unchanged; the API ignores the
    ones a model cannot read.

    Exhausted continuations: when a turn is still paused after
    ``max_continuations`` continuations, the partial turn is returned with
    ``finish_reason="length"`` and ``"incomplete": "max_continuations"`` in the
    message's ``anthropic`` metadata. Server tool calls left without their result
    are not sent back in later requests.

    Args:
        model: The model id, such as ``"claude-opus-5-5"`` or ``"claude-sonnet-5-5"``.
        client: An ``anthropic.AsyncAnthropic`` client to use.
        api_key: API key for a client created by the model.
        base_url: Base URL for a client created by the model.
        timeout: Request timeout in seconds for a client created by the model.
        max_retries: SDK-level retries. Keep 0 unless you accept repeated, billed requests.
        max_tokens: ``max_tokens`` when ``ModelSettings.max_tokens`` is unset. Defaults to
            `DEFAULT_MAX_TOKENS`, capped at the model's maximum output. A value above the
            maximum output of a known model is a `ConfigurationError`.
        effort: ``output_config.effort``: ``"low"``, ``"medium"``, ``"high"``, ``"xhigh"`` or
            ``"max"``. ``None`` leaves the model's default (``medium`` on Claude Opus 5.5,
            ``high`` on most others).
        thinking_display: ``"summarized"`` returns readable summaries of the reasoning (as
            ``ReasoningDelta`` events and ``ReasoningPart.text``), ``"updates"`` returns only
            progress updates between tool calls (beta; the models in
            `THINKING_UPDATES_MODELS` and unknown ids), ``"omitted"`` returns empty thinking
            text. ``None`` sends no ``thinking`` field, so the model's default applies.
        cache: Prompt caching: a breakpoint after the stable prefix (the system prompt, or
            the last tool when there is none) plus automatic caching of the conversation.
        cache_ttl: Cache lifetime, ``"5m"`` or ``"1h"``.
        fallbacks: ``"default"``, a list of fallback entries, or ``None``. See above.
        eager_input_streaming: Stream tool arguments as they are generated instead of after
            the server validated each parameter. ``None`` turns it on for the Claude API
            and off for a custom ``base_url`` (a proxy may reject the field). Arguments are
            always parsed strictly and validated by the tool before it runs.
        pricing: Prices for this model, replacing the `PRICING` entry.
        request_options: Extra fields for every request (keys of `ALLOWED_EXTRA`) and SDK
            options (``extra_headers``, ``extra_query``, ``extra_body``, ``timeout``).
            ``ModelSettings.extra`` takes precedence. An ``anthropic-beta`` value in
            ``extra_headers`` is merged with the betas the adapter sends.
        max_continuations: How many times a ``pause_turn`` response is continued within one
            model request.
    """

    provider = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        client: anthropic.AsyncAnthropic | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int = 0,
        max_tokens: int | None = None,
        effort: Effort | None = None,
        thinking_display: ThinkingDisplay | None = None,
        cache: bool = True,
        cache_ttl: Literal["5m", "1h"] = "5m",
        fallbacks: Literal["default"] | list[dict[str, Any]] | None = "default",
        eager_input_streaming: bool | None = None,
        pricing: AnthropicPricing | None = None,
        request_options: Mapping[str, Any] | None = None,
        max_continuations: int = 5,
    ) -> None:
        if not model:
            raise ConfigurationError("AnthropicModel needs a model id, such as 'claude-opus-5-5'.")
        if effort is not None and effort not in ("low", "medium", "high", "xhigh", "max"):
            raise ConfigurationError(
                f"effort must be one of low, medium, high, xhigh, max; got {effort!r}."
            )
        if thinking_display is not None and thinking_display not in (
            "summarized",
            "omitted",
            "updates",
        ):
            raise ConfigurationError(
                "thinking_display must be 'summarized', 'omitted', 'updates' or None; "
                f"got {thinking_display!r}."
            )
        if cache_ttl not in ("5m", "1h"):
            raise ConfigurationError(f"cache_ttl must be '5m' or '1h'; got {cache_ttl!r}.")
        if not (fallbacks is None or fallbacks == "default" or isinstance(fallbacks, list)):
            raise ConfigurationError(
                "fallbacks must be 'default', a list such as [{'model': 'claude-opus-5'}], "
                f"or None; got {fallbacks!r}."
            )
        if max_retries < 0 or max_continuations < 0:
            raise ConfigurationError("max_retries and max_continuations cannot be negative.")
        base = _base_id(model)
        limits = MODEL_LIMITS.get(base)
        if max_tokens is not None and (max_tokens < 1 or (limits and max_tokens > limits[1])):
            raise ConfigurationError(
                f"max_tokens must be between 1 and {limits[1] if limits else 'the model maximum'}"
                f" for {model}; got {max_tokens}."
            )
        if effort is not None and base in NO_EFFORT:
            raise ConfigurationError(f"{model} has no effort parameter; leave effort=None.")
        if effort == "xhigh" and base in NO_XHIGH_EFFORT:
            raise ConfigurationError(f"{model} has no 'xhigh' effort; use 'high' or 'max'.")
        if thinking_display is not None and base in MANUAL_THINKING_ONLY:
            raise ConfigurationError(
                f"{model} has no adaptive thinking, which thinking_display needs. Leave "
                "thinking_display=None and pass ModelSettings(extra={'thinking': {'type': "
                "'enabled', 'budget_tokens': N}}) for manual extended thinking."
            )
        if thinking_display == "updates" and limits and base not in THINKING_UPDATES_MODELS:
            raise ConfigurationError(
                f"{model} has no thinking_display='updates' (progress updates between tool "
                "calls); use 'summarized' or 'omitted'."
            )
        if isinstance(fallbacks, list) and base in NO_SERVER_FALLBACK:
            raise ConfigurationError(
                f"{model} has no server-side fallback and rejects a fallbacks list. Use "
                "fallbacks=None and handle finish_reason='content_filter'."
            )
        self.name = model
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.effort: Effort | None = effort
        self.thinking_display: ThinkingDisplay | None = thinking_display
        self.cache = cache
        self.cache_ttl: Literal["5m", "1h"] = cache_ttl
        self.fallbacks = fallbacks
        self.eager_input_streaming = eager_input_streaming
        self.pricing = pricing
        self.max_continuations = max_continuations
        self.request_options = dict(request_options or {})
        unknown = sorted(set(self.request_options) - ALLOWED_EXTRA - REQUEST_OPTIONS)
        if unknown:
            reasons = "".join(
                f" {name}: {_UNSUPPORTED_EXTRA[name]}."
                for name in unknown
                if name in _UNSUPPORTED_EXTRA
            )
            raise ConfigurationError(
                f"request_options has key(s) the Messages API does not accept: "
                f"{', '.join(unknown)}.{reasons} Accepted: "
                f"{', '.join(sorted(ALLOWED_EXTRA | REQUEST_OPTIONS))}."
            )
        self._check_extra(self.request_options)
        self._header_betas = self._take_header_betas()
        self._client = client
        self._owns_client = client is None
        self._request_client: Any = None
        self._client_options = {"api_key": api_key, "base_url": base_url, "timeout": timeout}
        self._warned_tool_choice = False
        self._rejected_signatures: set[str] = set()

    def _take_header_betas(self) -> list[str]:
        """Move ``anthropic-beta`` out of ``extra_headers``: the SDK would let it replace
        the header built from ``betas``, dropping the betas the adapter adds."""
        headers = self.request_options.get("extra_headers")
        if headers is None:
            return []
        if not isinstance(headers, Mapping):
            raise ConfigurationError("request_options['extra_headers'] must be a mapping.")
        betas: list[str] = []
        kept: dict[str, Any] = {}
        for key, value in headers.items():
            if str(key).lower() != "anthropic-beta":
                kept[key] = value
            elif isinstance(value, str):
                betas.extend(item.strip() for item in value.split(",") if item.strip())
            else:
                raise ConfigurationError(
                    "An anthropic-beta header in extra_headers must be a string; pass betas "
                    "in request_options['betas'] instead."
                )
        self.request_options["extra_headers"] = kept
        return betas

    # -- client ------------------------------------------------------------------------------

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        """The ``AsyncAnthropic`` client, created on first use."""
        if self._client is None:
            try:
                import anthropic
            except ImportError as error:
                raise ConfigurationError(_INSTALL_HINT) from error
            options = {k: v for k, v in self._client_options.items() if v is not None}
            try:
                self._client = anthropic.AsyncAnthropic(max_retries=self.max_retries, **options)
            except anthropic.CredentialsError as error:
                # A configured profile that cannot be read fails here, not at the request.
                raise ModelAuthenticationError(
                    f"{_MISSING_CREDENTIALS} ({error})", provider=self.provider, model=self.name
                ) from error
        return self._client

    def _sdk(self) -> Any:
        """The client requests go through, with this model's ``max_retries``."""
        if self._request_client is None:
            client = self.client
            if self._owns_client:
                self._request_client = client
            else:
                self._request_client = client.with_options(max_retries=self.max_retries)
        return self._request_client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.close()
            self._client = None
            self._request_client = None

    # -- what is known about the model ---------------------------------------------------------

    async def capabilities(self) -> ModelCapabilities:
        """Known limits and base prices from `MODEL_LIMITS` and `PRICING`; no request is made."""
        base = _base_id(self.name)
        limits = MODEL_LIMITS.get(base)
        prices = self.pricing or PRICING.get(base)
        return ModelCapabilities(
            tools=True if limits else None,
            vision=True if limits else None,
            context_length=limits[0] if limits else None,
            max_output_tokens=limits[1] if limits else None,
            input_price_per_million=prices.input if prices else None,
            output_price_per_million=prices.output if prices else None,
        )

    def validate_settings(self, settings: ModelSettings) -> None:
        """Reject settings the Messages API would reject, before any request is made."""
        removed = [name for name in ("temperature", "top_p") if getattr(settings, name) is not None]
        if removed:
            raise self._bad_request(
                f"The Claude API rejects {' and '.join(removed)} on every model (API version "
                "2026-10-21, sent by the anthropic SDK 2.x). Remove it from ModelSettings; "
                "use effort= on AnthropicModel to trade quality against latency and cost."
            )
        if settings.seed is not None:
            raise self._bad_request(
                "The Claude API has no seed parameter. Remove seed from ModelSettings."
            )
        unknown = sorted(set(settings.extra) - ALLOWED_EXTRA)
        if unknown:
            sampling = [name for name in unknown if name in _REMOVED_SAMPLING]
            if sampling:
                raise self._bad_request(
                    f"The Claude API rejects {', '.join(sampling)} on every model (API version "
                    "2026-10-21). Remove it from ModelSettings.extra."
                )
            unsupported = [name for name in unknown if name in _UNSUPPORTED_EXTRA]
            if unsupported:
                name = unsupported[0]
                raise self._bad_request(
                    f"ModelSettings.extra[{name!r}] is not supported: {_UNSUPPORTED_EXTRA[name]}."
                )
            raise self._bad_request(
                f"The Messages API does not accept the request field(s) {', '.join(unknown)} "
                f"in ModelSettings.extra. Accepted: {', '.join(sorted(ALLOWED_EXTRA))}. "
                "max_tokens, stop and tool_choice have their own ModelSettings fields."
            )
        self._check_extra(settings.extra)
        limits = MODEL_LIMITS.get(_base_id(self.name))
        if settings.max_tokens is not None:
            if settings.max_tokens < 1:
                raise self._bad_request("ModelSettings.max_tokens must be at least 1.")
            if limits and settings.max_tokens > limits[1]:
                raise self._bad_request(
                    f"{self.name} produces at most {limits[1]} output tokens; "
                    f"ModelSettings.max_tokens is {settings.max_tokens}."
                )

    def _check_extra(self, extra: Mapping[str, Any]) -> None:
        betas = extra.get("betas")
        if betas is not None and not (
            isinstance(betas, list) and all(isinstance(item, str) for item in betas)
        ):
            raise self._bad_request("betas must be a list of beta header values (strings).")
        for name in ("output_config", "thinking", "metadata"):
            if name in extra and extra[name] is not None and not isinstance(extra[name], dict):
                raise self._bad_request(f"{name} must be a JSON object (a dict).")
        base = _base_id(self.name)
        effort = (extra.get("output_config") or {}).get("effort")
        if effort is not None and base in NO_EFFORT:
            raise self._bad_request(f"{self.name} has no effort parameter (output_config.effort).")
        if effort == "xhigh" and base in NO_XHIGH_EFFORT:
            raise self._bad_request(f"{self.name} has no 'xhigh' effort; use 'high' or 'max'.")
        thinking = extra.get("thinking")
        if (
            base == "claude-opus-5-5"
            and thinking is not None
            and thinking.get("type") != "adaptive"
        ):
            raise self._bad_request(
                f"{self.name} requires adaptive thinking. Omit thinking or set its type "
                "to 'adaptive' and use output_config.effort to control reasoning depth."
            )
        if thinking and thinking.get("type") == "adaptive" and base in MANUAL_THINKING_ONLY:
            raise self._bad_request(
                f"{self.name} has no adaptive thinking; use {{'type': 'enabled', "
                "'budget_tokens': N}."
            )
        if isinstance(extra.get("fallbacks"), list) and base in NO_SERVER_FALLBACK:
            raise self._bad_request(
                f"{self.name} has no server-side fallback and rejects a fallbacks list. Use "
                "fallbacks=None and handle finish_reason='content_filter'."
            )

    # -- request building ------------------------------------------------------------------------

    def build_params(self, request: ModelRequest) -> dict[str, Any]:
        """The keyword arguments this model passes to ``client.beta.messages.create``.

        The result is what is sent, apart from ``stream=True``. Useful to log or
        inspect a request; it raises the same early errors a request would.
        """
        params, _ = self._build(request)
        return params

    def _build(self, request: ModelRequest) -> tuple[dict[str, Any], str]:
        """Request parameters, and the fingerprint recorded with thinking blocks it produces."""
        settings = request.settings
        self.validate_settings(settings)
        base = _base_id(self.name)
        system_blocks, prefinal = _translate(request.messages, self._bad_request)
        tools = self._tools(request)
        named_in_history = _tool_names_in(prefinal)
        stub_tools = not tools and bool(named_in_history)
        if stub_tools:
            # A request whose history holds tool_use blocks is rejected when it defines no
            # tools at all. An agent with no tools of its own (a handoff target, say) still
            # carries earlier calls, so they are declared as stubs and tool_choice is none.
            tools = [
                {"name": name, "input_schema": {"type": "object"}}
                for name in sorted(named_in_history)
            ]
        fingerprint = _Fingerprint(system_blocks, tools)
        options = {**self.request_options, **settings.extra}
        # Manual thinking needs the open tool round to start with its thinking block, and
        # no model with the history-editing check accepts it, so its blocks are all kept.
        if not _manual_thinking(options) and (
            base in PREFIX_CHECK_MODELS or base not in MODEL_LIMITS
        ):
            prefixes: list[str | None] = list(fingerprint.prefixes(prefinal))
        else:
            prefixes = [None] * len(prefinal)  # no history-editing check: replay every block
        messages = _finalize(prefinal, prefixes, self._rejected_signatures)
        if not messages:
            raise self._bad_request("The request has no messages to send.")
        if messages[0]["role"] != "user":
            raise self._bad_request(
                "The Messages API needs the conversation to start with a user message; this "
                f"one starts with a {messages[0]['role']} message."
            )

        params: dict[str, Any] = {
            "model": self.name,
            "max_tokens": self._max_tokens(settings),
            "messages": messages,
        }
        if system_blocks:
            system = [dict(block) for block in system_blocks]
            if self.cache:
                system[-1]["cache_control"] = self._cache_control()
            params["system"] = system
        if tools:
            sent_tools = [dict(tool) for tool in tools]
            if self.cache and not system_blocks:
                sent_tools[-1]["cache_control"] = self._cache_control()
            params["tools"] = sent_tools
            choice = {"type": "none"} if stub_tools else self._tool_choice(settings, base)
            if choice is not None:
                params["tool_choice"] = choice
        if self.cache:
            params["cache_control"] = self._cache_control()
        if settings.stop:
            params["stop_sequences"] = list(settings.stop)

        output_config: dict[str, Any] = {}
        if self.effort is not None:
            output_config["effort"] = self.effort
        if request.output is not None:
            output_config["format"] = self._output_format(request)
        if isinstance(options.get("output_config"), dict):
            output_config.update(options.pop("output_config"))
        if output_config:
            params["output_config"] = output_config
        if self.thinking_display is not None:
            params["thinking"] = {"type": "adaptive", "display": self.thinking_display}

        betas: list[str] = [*self._header_betas]
        if self.thinking_display == "updates":
            betas.append(THINKING_UPDATES_BETA)
        if "fallbacks" not in options:
            if isinstance(self.fallbacks, list):
                params["fallbacks"] = [dict(entry) for entry in self.fallbacks]
            elif self.fallbacks == "default" and base in FALLBACK_MODELS:
                params["fallbacks"] = "default"
        elif options["fallbacks"] is None:
            options.pop("fallbacks")
        betas.extend(options.pop("betas", None) or [])
        params.update(options)
        if params.get("fallbacks") is not None:
            betas.append(FALLBACK_BETA)
        if betas:
            params["betas"] = list(dict.fromkeys(betas))
        return params, fingerprint.of(prefinal)

    def _max_tokens(self, settings: ModelSettings) -> int:
        if settings.max_tokens is not None:
            return settings.max_tokens
        if self.max_tokens is not None:
            return self.max_tokens
        limits = MODEL_LIMITS.get(_base_id(self.name))
        return min(DEFAULT_MAX_TOKENS, limits[1]) if limits else DEFAULT_MAX_TOKENS

    def _cache_control(self) -> dict[str, Any]:
        control: dict[str, Any] = {"type": "ephemeral"}
        if self.cache_ttl == "1h":
            control["ttl"] = "1h"
        return control

    def _eager(self) -> bool:
        if self.eager_input_streaming is not None:
            return self.eager_input_streaming
        # The base URL the SDK resolves (argument, ANTHROPIC_BASE_URL, profile), so the
        # first request and the later ones send the same tool definitions.
        return str(self.client.base_url).rstrip("/") == _DEFAULT_HOST

    def _tools(self, request: ModelRequest) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        eager = self._eager() if request.tools else False
        for spec in request.tools:
            schema = spec.parameters
            if spec.strict:
                schema = self._strict_schema(schema, f"tool {spec.name!r}")
            tool: dict[str, Any] = {"name": spec.name}
            if spec.description:
                tool["description"] = spec.description
            tool["input_schema"] = schema
            if spec.strict:
                tool["strict"] = True
            if eager:
                tool["eager_input_streaming"] = True
            tools.append(tool)
        return tools

    def _tool_choice(self, settings: ModelSettings, base: str) -> dict[str, Any] | None:
        choice = settings.tool_choice
        wire: dict[str, Any] | None
        if choice is None:
            wire = None
        elif choice == "auto":
            wire = {"type": "auto"}
        elif choice == "none":
            return {"type": "none"}
        else:
            if base in NO_FORCED_TOOL_CHOICE:
                if not self._warned_tool_choice:
                    self._warned_tool_choice = True
                    logger.warning(
                        "%s rejects a forced tool choice (tool_choice=%r); sending 'auto' "
                        "instead. Name the tool in the instructions, and use strict=True on it "
                        "for schema-valid arguments.",
                        self.name,
                        choice,
                    )
                wire = {"type": "auto"}
            elif choice == "required":
                wire = {"type": "any"}
            else:
                wire = {"type": "tool", "name": choice}
        if settings.parallel_tool_calls is False:
            wire = {**(wire or {"type": "auto"}), "disable_parallel_tool_use": True}
        return wire

    def _output_format(self, request: ModelRequest) -> dict[str, Any]:
        assert request.output is not None
        schema = self._strict_schema(request.output.json_schema, "the output schema")
        return {"type": "json_schema", "schema": schema}

    def _strict_schema(self, schema: dict[str, Any], what: str) -> dict[str, Any]:
        """The schema in the form structured outputs require, via the SDK's ``transform_schema``.

        Objects get ``additionalProperties: false``; constraints the API does not
        enforce (``minimum``, ``maxLength``, ...) move into the description. The
        loop still validates the result against the original schema. Output schemas
        are always sent in this form: ``output_config.format`` has no unenforced mode.

        An open map (``dict[str, X]``, a bare ``dict``, ``patternProperties``) cannot be
        expressed: closing it would let the model return only ``{}`` for it. Such a
        schema is refused here, before anything is sent.
        """
        open_maps = _open_maps(schema, "$")
        if open_maps:
            raise self._bad_request(
                f"{what} has an open map at {', '.join(open_maps)} (a dict with free-form "
                "keys), which Claude structured outputs cannot express. Model it as a list of "
                'key/value objects, or use output_mode="prompted" (or output_mode="tool" '
                "without output_strict), or drop strict=True on the tool."
            )
        try:
            from anthropic import transform_schema
        except ImportError as error:
            raise ConfigurationError(_INSTALL_HINT) from error
        try:
            return transform_schema(schema)
        except (ValueError, TypeError) as error:
            raise self._bad_request(
                f"The Messages API cannot enforce {what}: {error} Remove strict=True, or use "
                'output_mode="prompted".'
            ) from error

    # -- streaming -------------------------------------------------------------------------------

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        params, fingerprint = self._build(request)
        client = self._sdk()
        responses: list[_Accumulator] = []
        tool_index = 0
        while True:
            accumulator = _Accumulator(
                model=self.name, fingerprint=fingerprint, first_tool_index=tool_index
            )
            try:
                stream, params = await self._open(client, params)
                try:
                    async for raw in stream:
                        for event in accumulator.feed(raw):
                            yield event
                except ModelError:
                    raise
                except Exception as error:
                    raise self._map_error(error, started=accumulator.started) from error
                finally:
                    await stream.close()
            except ModelError as error:
                if responses:
                    # An earlier segment of this turn ran and was billed, and it may have
                    # emitted nothing: repeating the model request would run it again.
                    error.ambiguous = True
                raise
            if not accumulator.started or not accumulator.complete:
                raise ModelResponseError(
                    "The response stream ended before the model finished.",
                    provider=self.provider,
                    model=self.name,
                    ambiguous=True,
                )
            responses.append(accumulator)
            tool_index = accumulator.next_tool_index
            if (
                accumulator.stop_reason not in ("pause_turn", "compaction")
                or len(responses) > self.max_continuations
            ):
                break
            # The server paused a long turn: send the partial turn back and it continues.
            # Every segment ends up in one assistant message, so its thinking blocks keep
            # the fingerprint of the prefix before that message.
            params = _continue(params, accumulator)
        yield ResponseDone(response=self._response(responses))

    async def _open(self, client: Any, params: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        """The open stream, and the parameters actually sent (thinking may be stripped)."""
        try:
            return await self._create(client, params), params
        except ModelError:
            raise
        except Exception as error:
            if not _is_thinking_signature_error(error) or (
                _manual_thinking(params) and _in_tool_round(params["messages"])
            ):
                # Under manual thinking an open tool round must keep its thinking block,
                # so a request without the blocks would only be rejected again.
                raise self._map_error(error, started=False) from error
            # A thinking block failed the API's history check although its recorded
            # prefix matched. The request was rejected before it ran, so it is safe to
            # send once more without thinking blocks.
            stripped, signatures = _strip_thinking(params["messages"])
            if not signatures:
                raise self._map_error(error, started=False) from error
            logger.warning(
                "The Claude API rejected %d replayed thinking block(s) for %s; retrying once "
                "without them. Keep the system prompt, the tools and earlier messages unchanged "
                "within a conversation to keep the model's earlier reasoning.",
                len(signatures),
                self.name,
            )
            self._rejected_signatures.update(signatures)
            sent = {**params, "messages": stripped}
            try:
                return await self._create(client, sent), sent
            except ModelError:
                raise
            except Exception as retry_error:
                raise self._map_error(retry_error, started=False) from retry_error

    async def _create(self, client: Any, params: dict[str, Any]) -> Any:
        import anthropic

        try:
            return await client.beta.messages.create(**params, stream=True)
        except anthropic.CredentialsError as error:
            raise ModelAuthenticationError(
                f"{_MISSING_CREDENTIALS} ({error})", provider=self.provider, model=self.name
            ) from error
        except TypeError as error:
            # The SDK raises TypeError when it finds no credentials at all.
            if "authentication" in str(error).lower():
                raise ModelAuthenticationError(
                    _MISSING_CREDENTIALS, provider=self.provider, model=self.name
                ) from error
            raise

    def _response(self, responses: list[_Accumulator]) -> ModelResponse:
        last = responses[-1]
        if last.stop_reason is None or last.stop_reason not in _FINISH_REASONS:
            # A message_stop alone does not say whether a tool or answer was cut off.
            raise ModelResponseError(
                "The response stream ended without a recognized stop reason.",
                provider=self.provider,
                model=self.name,
                ambiguous=True,
            )
        parts: list[Part] = []
        usage = Usage()
        metadata = last.metadata()
        cache_creation_tokens = 0
        for accumulator in responses:
            parts.extend(accumulator.parts())
            segment = accumulator.usage()
            segment.cost_usd = self._cost(accumulator)
            # Usage.add classifies each segment by whether its price is known.
            usage = usage.add(segment)
            cache_creation_tokens += sum(
                entry.cache_creation_tokens
                for entry in accumulator.iterations or [accumulator.totals()]
            )
        if cache_creation_tokens:
            metadata["cache_creation_input_tokens"] = cache_creation_tokens
        # The size of the last request sent, unlike usage, which sums every segment.
        context_tokens = metadata["context_tokens"] = last.context_tokens()
        finish = _FINISH_REASONS[last.stop_reason]
        if last.stop_reason in ("pause_turn", "compaction"):
            logger.warning(
                "%s paused its turn %d times; returning the partial turn. Raise "
                "max_continuations to let it finish.",
                self.name,
                len(responses),
            )
            metadata["incomplete"] = "max_continuations"
        if finish == "content_filter":
            # A refusal can cut a tool call off mid-input: never run that turn's tools.
            parts = [part for part in parts if not isinstance(part, ToolCallPart)]
        elif finish == "stop" and any(isinstance(part, ToolCallPart) for part in parts):
            finish = "tool_calls"
        return ModelResponse(
            message=Message(
                role="assistant",
                parts=parts,
                metadata={"anthropic": metadata} if metadata else {},
            ),
            finish_reason=finish,
            usage=usage,
            model=last.served_by or self.name,
            provider=self.provider,
            response_id=last.response_id,
            context_tokens=context_tokens,
        )

    def _prices(self, model: str) -> AnthropicPricing | None:
        base = _base_id(model)
        if self.pricing is not None and base == _base_id(self.name):
            return self.pricing
        return PRICING.get(base)

    def _cost(self, accumulator: _Accumulator) -> float | None:
        if accumulator.iterations:
            # Per attempt, each at its own model's prices: a declined attempt (counted even
            # when it may be credited, so spend limits fail closed) and a compaction, which
            # top-level usage leaves out.
            total = 0.0
            for entry in accumulator.iterations:
                model = entry.model or (
                    self.name if entry.kind == "compaction" else accumulator.served_by or self.name
                )
                piece = self._entry_cost(model, entry)
                if piece is None:
                    return None
                total += piece
            return total
        return self._entry_cost(accumulator.served_by or self.name, accumulator.totals())

    def _entry_cost(self, model: str, entry: _UsageEntry) -> float | None:
        prices = self._prices(model)
        if prices is None:
            return None
        write_5m, write_1h = entry.cache_writes(default_ttl=self.cache_ttl)
        return prices.cost(
            input_tokens=entry.input_tokens,
            output_tokens=entry.output_tokens,
            cache_read_tokens=entry.cache_read_tokens,
            cache_write_5m_tokens=write_5m,
            cache_write_1h_tokens=write_1h,
        )

    # -- errors ------------------------------------------------------------------------------------

    def _bad_request(self, message: str) -> ModelBadRequestError:
        return ModelBadRequestError(message, provider=self.provider, model=self.name)

    def _map_error(self, error: Exception, *, started: bool) -> Exception:
        """Turn an ``anthropic`` exception into the matching qagents error.

        ``started`` is true once the response stream has begun: from then on the
        request may have produced billable output, so every error is ambiguous.
        """
        try:
            import anthropic
            import httpx2
        except ImportError:
            return error
        details: dict[str, Any] = {"provider": self.provider, "model": self.name}
        if isinstance(error, anthropic.APIStatusError):
            status = error.status_code
            kind = error.type
            details.update(
                status_code=status,
                code=error.error_code or kind,
                request_id=error.request_id,
                retry_after=_retry_after(error.response.headers),
            )
            message = f"[{status}] {error.api_message or error.message}"
            if started or status < 400:
                # An error event inside a stream that had already started.
                if kind == "invalid_request_error":
                    return ModelBadRequestError(message, ambiguous=True, **details)
                return ModelServerError(message, retryable=True, ambiguous=True, **details)
            if status == 429:
                return ModelRateLimitError(message, retryable=True, **details)
            if status == 529 or kind == "overloaded_error":
                # Refused at admission: the request did not run.
                return ModelServerError(message, retryable=True, **details)
            if status in (401, 403):
                return ModelAuthenticationError(message, **details)
            if status == 402 or kind == "billing_error":
                return ModelBillingError(message, **details)
            if status == 408:
                return ModelTimeoutError(message, retryable=True, ambiguous=True, **details)
            if 400 <= status < 500 and status != 409:
                return ModelBadRequestError(message, **details)
            return ModelServerError(message, retryable=True, ambiguous=True, **details)
        if isinstance(error, anthropic.APIConnectionError):
            cause = error.__cause__
            never_sent = isinstance(
                cause, httpx2.ConnectError | httpx2.ConnectTimeout | httpx2.PoolTimeout
            )
            ambiguous = started or not never_sent
            kind = (
                ModelTimeoutError
                if isinstance(error, anthropic.APITimeoutError)
                else ModelConnectionError
            )
            return kind(str(error), retryable=True, ambiguous=ambiguous, **details)
        if isinstance(error, anthropic.APIResponseValidationError | ValueError):
            return ModelResponseError(str(error), ambiguous=True, **details)
        if isinstance(error, httpx2.TransportError):
            timeout = isinstance(error, httpx2.TimeoutException)
            return (ModelTimeoutError if timeout else ModelConnectionError)(
                f"The connection broke mid-response: {error}",
                retryable=True,
                ambiguous=True,
                **details,
            )
        return error


# --- translation: qagents messages -> content blocks ----------------------------------------


class _Bound:
    """A replayable thinking block with the prefix fingerprint it was produced under."""

    __slots__ = ("block", "prefix")

    def __init__(self, block: dict[str, Any], prefix: str | None) -> None:
        self.block = block
        self.prefix = prefix


def _translate(
    messages: list[Message], bad_request: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """System blocks, and wire messages before thinking blocks are checked.

    System messages become top-level system blocks, in order (the Messages API
    has no system role inside ``messages`` on every model). Tool results become
    user messages of ``tool_result`` blocks, and consecutive messages of the
    same role are merged.
    """
    system: list[dict[str, Any]] = []
    wire: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            if message.text:
                system.append({"type": "text", "text": message.text})
            continue
        if message.role == "assistant":
            role, content = "assistant", _assistant_content(message)
        elif message.role == "tool":
            role, content = "user", [_tool_result(part) for part in message.tool_results]
        else:
            role, content = "user", _user_content(message, bad_request)
        if not content:
            continue
        if wire and wire[-1]["role"] == role:
            wire[-1]["content"].extend(content)
        else:
            wire.append({"role": role, "content": content})
    _unique_tool_ids(wire)
    return system, wire


def _unique_tool_ids(wire: list[dict[str, Any]]) -> None:
    """Give a tool call id that repeats an earlier call's id a suffix (``_2``, ``_3``, ...).

    The Messages API needs unique ``tool_use`` ids, and some providers reuse ids such
    as ``functions.search:0`` in every turn. A result goes with the latest call of its
    id. The suffix depends only on the position in the history, so it is stable.
    """
    used: set[str] = set()
    latest: dict[str, str] = {}
    for message in wire:
        for block in message["content"]:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                original = new = block["id"]
                count = 1
                while new in used:
                    count += 1
                    new = f"{original}_{count}"
                used.add(new)
                latest[original] = block["id"] = new
            elif block.get("type") == "tool_result" and block["tool_use_id"] in latest:
                block["tool_use_id"] = latest[block["tool_use_id"]]


def _user_content(message: Message, bad_request: Any) -> list[Any]:
    content: list[Any] = []
    for part in message.parts:
        if isinstance(part, TextPart):
            if part.text:
                content.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            content.append(_image(part, bad_request))
        elif isinstance(part, ToolResultPart):
            content.append(_tool_result(part))
    return content


def _image(part: ImagePart, bad_request: Any) -> dict[str, Any]:
    if part.data:
        media_type, data = part.media_type or "image/png", part.data
    elif part.url and part.url.startswith("data:"):
        match = _DATA_URL.match(part.url)
        if match is None:
            raise bad_request("An image data: URL must be base64-encoded (data:<type>;base64,...).")
        media_type, data = match["media_type"], match["data"]
    elif part.url:
        return {"type": "image", "source": {"type": "url", "url": part.url}}
    else:
        raise bad_request("An ImagePart needs either url or data.")
    if media_type not in _IMAGE_TYPES:
        raise bad_request(
            f"The Messages API accepts images of type {', '.join(sorted(_IMAGE_TYPES))}; "
            f"got {media_type}."
        )
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}


def _wire_id(value: str) -> str:
    """A tool call id the Messages API accepts (``^[A-Za-z0-9_-]+$``).

    Ids from other providers (``functions.search:0``, say) are mapped to a stable
    valid id, the same for a call and its result in every request; valid ids are
    kept as they are.
    """
    if _TOOL_ID.match(value):
        return value
    digest = hashlib.sha256(value.encode()).hexdigest()[:12]
    return f"{re.sub(r'[^A-Za-z0-9_-]', '_', value)[:48]}_{digest}"


def _tool_result(part: ToolResultPart) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "tool_result", "tool_use_id": _wire_id(part.tool_call_id)}
    if part.content:
        block["content"] = part.content
    if part.is_error:
        block["is_error"] = True
    return block


def _assistant_content(message: Message) -> list[Any]:
    content: list[Any] = []
    for part in message.parts:
        if isinstance(part, TextPart):
            if part.text:
                content.append({"type": "text", "text": part.text})
        elif isinstance(part, ToolCallPart):
            content.append(
                {
                    "type": "tool_use",
                    "id": _wire_id(part.id),
                    "name": part.name,
                    "input": part.arguments,
                }
            )
        elif isinstance(part, ReasoningPart):
            opaque = part.opaque or {}
            block = opaque.get("block")
            if opaque.get("provider") != "anthropic" or not isinstance(block, dict):
                continue  # reasoning from another provider cannot be replayed here
            if block.get("type") in _THINKING_TYPES:
                prefix = opaque.get("prefix")
                content.append(_Bound(block, prefix if isinstance(prefix, str) else None))
            else:
                content.append(block)
    return content


def _tool_names_in(messages: list[dict[str, Any]]) -> set[str]:
    return {
        block["name"]
        for message in messages
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "tool_use"
    }


class _Fingerprint:
    """Hashes of what the API binds a thinking block to: system, tools, earlier messages.

    Thinking blocks and ``cache_control`` markers are left out, as the API's own
    check leaves them out. Tools are compared as a name-sorted set.
    """

    def __init__(self, system: list[dict[str, Any]], tools: list[dict[str, Any]]) -> None:
        self._head = hashlib.sha256()
        self._head.update(_canonical({"system": system}))
        self._head.update(_canonical({"tools": sorted(tools, key=lambda tool: tool["name"])}))

    def prefixes(self, messages: list[dict[str, Any]]) -> list[str]:
        """The fingerprint of the prefix before each message."""
        running = self._head.copy()
        result = []
        for message in messages:
            result.append(running.hexdigest())
            running.update(_canonical(_without_thinking(message)))
        return result

    def of(self, messages: list[dict[str, Any]]) -> str:
        """The fingerprint of the whole conversation."""
        running = self._head.copy()
        for message in messages:
            running.update(_canonical(_without_thinking(message)))
        return running.hexdigest()


def _without_thinking(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": message["role"],
        "content": [
            block
            for block in message["content"]
            if not isinstance(block, _Bound)
            and not (isinstance(block, dict) and block.get("type") in _THINKING_TYPES)
        ],
    }


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    ).encode()


def _finalize(
    messages: list[dict[str, Any]], prefixes: list[str | None], rejected: set[str]
) -> list[dict[str, Any]]:
    """Resolve thinking blocks: keep a block only while its recorded prefix still matches.

    A ``None`` prefix (a model without the history-editing check) keeps every block.
    Server tool calls without their result are left out of every assistant message
    but the last, where the API resumes them.
    """
    final: list[dict[str, Any]] = []
    last = len(messages) - 1
    for position, (message, prefix) in enumerate(zip(messages, prefixes, strict=True)):
        blocks = message["content"]
        if message["role"] == "assistant" and position < last:
            blocks = _without_unpaired_server_calls(blocks)
        content: list[dict[str, Any]] = []
        for block in blocks:
            if isinstance(block, _Bound):
                matches = block.prefix is None or prefix is None or block.prefix == prefix
                if matches and _signature(block.block) not in rejected:
                    content.append(block.block)
            else:
                content.append(block)
        if not content:
            continue
        if final and final[-1]["role"] == message["role"]:
            final[-1]["content"].extend(content)
        else:
            final.append({"role": message["role"], "content": content})
    return final


def _is_server_call(block: Any) -> bool:
    """A ``server_tool_use``, ``mcp_tool_use`` or other server-side tool call block."""
    kind = block.get("type") if isinstance(block, dict) else None
    return isinstance(kind, str) and kind.endswith("tool_use") and kind != "tool_use"


def _without_unpaired_server_calls(content: list[Any]) -> list[Any]:
    """Content without server tool calls that have no result block (a refused turn, or
    one still paused after ``max_continuations``); the API rejects them mid-history."""
    answered = {block.get("tool_use_id") for block in content if isinstance(block, dict)}
    return [
        block
        for block in content
        if not (_is_server_call(block) and block.get("id") not in answered)
    ]


def _signature(block: dict[str, Any]) -> str:
    return str(block.get("signature") or block.get("data") or "")


def _strip_thinking(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[str]]:
    signatures: set[str] = set()
    stripped: list[dict[str, Any]] = []
    for message in messages:
        content = []
        for block in message["content"]:
            if isinstance(block, dict) and block.get("type") in _THINKING_TYPES:
                signatures.add(_signature(block))
            else:
                content.append(block)
        if not content:
            continue
        if stripped and stripped[-1]["role"] == message["role"]:
            stripped[-1]["content"].extend(content)
        else:
            stripped.append({"role": message["role"], "content": content})
    return stripped, signatures


def _in_tool_round(messages: list[dict[str, Any]]) -> bool:
    """Whether the last assistant message called tools: the request continues its round."""
    for message in reversed(messages):
        if message["role"] == "assistant":
            return any(
                isinstance(block, dict) and block.get("type") == "tool_use"
                for block in message["content"]
            )
    return False


def _manual_thinking(options: Mapping[str, Any]) -> bool:
    """Whether the request uses manual extended thinking (``thinking.type = "enabled"``)."""
    thinking = options.get("thinking")
    return isinstance(thinking, Mapping) and thinking.get("type") == "enabled"


def _is_thinking_signature_error(error: Exception) -> bool:
    try:
        import anthropic
    except ImportError:
        return False
    if not isinstance(error, anthropic.BadRequestError):
        return False
    text = str(error.api_message or error.message)
    return "signature" in text and "thinking" in text


def _continue(params: dict[str, Any], accumulator: _Accumulator) -> dict[str, Any]:
    """The request that continues a paused turn: the partial turn appended as it was received."""
    messages = [dict(message) for message in params["messages"]]
    blocks = accumulator.replay_blocks()
    if messages and messages[-1]["role"] == "assistant":
        messages[-1] = {"role": "assistant", "content": [*messages[-1]["content"], *blocks]}
    else:
        messages.append({"role": "assistant", "content": blocks})
    return {**params, "messages": messages}


# --- translation: stream events -> qagents ------------------------------------------------


_FINISH_REASONS: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "refusal": "content_filter",
    # Only a turn still paused after max_continuations ends on these: it is incomplete.
    "pause_turn": "length",
    "compaction": "length",
}


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def _breakdown(value: Any) -> tuple[int, int] | None:
    """The (5-minute, 1-hour) split of a ``cache_creation`` usage object."""
    if value is None:
        return None
    return (
        _field(value, "ephemeral_5m_input_tokens") or 0,
        _field(value, "ephemeral_1h_input_tokens") or 0,
    )


class _UsageEntry:
    """Token counts of one attempt (an entry of ``usage.iterations``) or of a whole response."""

    def __init__(
        self,
        *,
        kind: str = "message",
        model: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
        breakdown: tuple[int, int] | None = None,
    ) -> None:
        self.kind = kind
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_tokens = cache_read_tokens
        self.cache_creation_tokens = cache_creation_tokens
        self.breakdown = breakdown

    @classmethod
    def of(cls, item: Any) -> _UsageEntry:
        model = _field(item, "model")
        return cls(
            kind=str(_field(item, "type") or "message"),
            model=model if isinstance(model, str) else None,
            input_tokens=_field(item, "input_tokens") or 0,
            output_tokens=_field(item, "output_tokens") or 0,
            cache_read_tokens=_field(item, "cache_read_input_tokens") or 0,
            cache_creation_tokens=_field(item, "cache_creation_input_tokens") or 0,
            breakdown=_breakdown(_field(item, "cache_creation")),
        )

    @property
    def prompt_tokens(self) -> int:
        """The whole prompt: qagents counts cache reads and writes as input too."""
        return self.input_tokens + self.cache_read_tokens + self.cache_creation_tokens

    def cache_writes(self, *, default_ttl: str) -> tuple[int, int]:
        """Cache-write tokens with the 5-minute and the 1-hour lifetime.

        The reported total is the authority. The TTL breakdown splits it while it
        covers no more than the total; writes it does not cover (added later in a
        server-side loop) get the request's lifetime.
        """
        total = self.cache_creation_tokens
        if self.breakdown is not None and 0 < sum(self.breakdown) <= total:
            write_5m, write_1h = self.breakdown
            rest = total - write_5m - write_1h
            if default_ttl == "1h":
                return write_5m, write_1h + rest
            return write_5m + rest, write_1h
        return (0, total) if default_ttl == "1h" else (total, 0)


class _Accumulator:
    """Builds one response from the raw stream events of the Messages API.

    Tool arguments are collected as raw JSON text and parsed strictly at the end;
    the SDK's own helper parses them tolerantly and raises mid-stream on JSON it
    cannot repair, which would lose the whole turn.
    """

    def __init__(self, *, model: str, fingerprint: str, first_tool_index: int = 0) -> None:
        self.model = model
        self.fingerprint = fingerprint
        self.served_by: str | None = None
        self.response_id: str | None = None
        self.stop_reason: str | None = None
        self.stop_details: dict[str, Any] | None = None
        self.started = False
        self.complete = False
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0
        self.cache_creation_tokens = 0
        self.thinking_tokens = 0
        self.iterations: list[_UsageEntry] | None = None
        """``usage.iterations``: one entry per attempt, when the API reports them."""
        self._cache_breakdown: tuple[int, int] | None = None
        self._blocks: dict[int, dict[str, Any]] = {}
        self._json: dict[int, str] = {}
        """Raw input text of every block that streams one: client and server tool calls."""
        self._tool_index: dict[int, int] = {}
        self.next_tool_index = first_tool_index

    def feed(self, event: Any) -> list[ModelStreamEvent]:
        kind = getattr(event, "type", None)
        if kind == "message_start":
            message = event.message
            self.started = True
            self.response_id = message.id
            self.served_by = message.model
            self._usage(message.usage)
        elif kind == "content_block_start":
            return self._start(event.index, event.content_block)
        elif kind == "content_block_delta":
            return self._delta(event.index, event.delta)
        elif kind == "message_delta":
            self.stop_reason = event.delta.stop_reason or self.stop_reason
            details = event.delta.stop_details
            if details is not None:
                self.stop_details = details.to_dict(mode="json")
            self._usage(event.usage)
        elif kind == "message_stop":
            self.complete = True
        return []

    def _usage(self, usage: Any) -> None:
        if usage is None:
            return
        # Counts on message_delta are running totals: they replace, they do not add.
        if usage.input_tokens is not None:
            self.input_tokens = usage.input_tokens
        if usage.output_tokens is not None:
            self.output_tokens = usage.output_tokens
        if usage.cache_read_input_tokens is not None:
            self.cache_read_tokens = usage.cache_read_input_tokens
        if usage.cache_creation_input_tokens is not None:
            self.cache_creation_tokens = usage.cache_creation_input_tokens
        details = getattr(usage, "output_tokens_details", None)
        if details is not None and details.thinking_tokens is not None:
            self.thinking_tokens = details.thinking_tokens
        breakdown = getattr(usage, "cache_creation", None)
        if breakdown is not None:
            self._cache_breakdown = _breakdown(breakdown)
        iterations = getattr(usage, "iterations", None)
        if iterations:
            # A snapshot of every attempt so far: the newest replaces the older one.
            self.iterations = [_UsageEntry.of(item) for item in iterations]

    def _start(self, index: int, block: Any) -> list[ModelStreamEvent]:
        data = block.to_dict(mode="json")
        self._blocks[index] = data
        kind = data.get("type")
        if kind == "fallback":
            target = (data.get("to") or {}).get("model")
            if isinstance(target, str):
                self.served_by = target
            self._cache_breakdown = None  # it described the declined attempt
        if isinstance(kind, str) and kind.endswith("tool_use"):
            self._json[index] = ""  # server_tool_use and mcp_tool_use stream input too
        if kind != "tool_use":
            return []
        position = self.next_tool_index
        self._tool_index[index] = position
        self.next_tool_index += 1
        return [ToolCallDelta(index=position, id=data.get("id"), name=data.get("name") or "")]

    def _delta(self, index: int, delta: Any) -> list[ModelStreamEvent]:
        block = self._blocks.get(index)
        if block is None:
            return []
        kind = delta.type
        if kind == "text_delta":
            block["text"] = block.get("text", "") + delta.text
            return [TextDelta(text=delta.text)] if delta.text else []
        if kind == "thinking_delta":
            block["thinking"] = block.get("thinking", "") + delta.thinking
            return [ReasoningDelta(text=delta.thinking)] if delta.thinking else []
        if kind == "signature_delta":
            block["signature"] = delta.signature
        elif kind == "input_json_delta" and index in self._json:
            self._json[index] += delta.partial_json
            if index not in self._tool_index:
                return []  # a server-side call: not a tool for the loop to run
            return [
                ToolCallDelta(
                    index=self._tool_index[index],
                    id=block.get("id"),
                    name=block.get("name") or "",
                    arguments_delta=delta.partial_json,
                )
            ]
        elif kind == "compaction_delta":
            block["content"] = delta.content
            block["encrypted_content"] = delta.encrypted_content
        return []

    def _ordered(self) -> list[tuple[int, dict[str, Any]]]:
        """Blocks in order, without the declined part of a turn that fell back mid-output.

        Before the last ``fallback`` block only text survives (and the fallback
        blocks themselves, which must stay where they appeared): the declined
        model's thinking and tool calls are not part of the answer.
        """
        ordered = sorted(self._blocks.items())
        boundary = max(
            (
                position
                for position, (_, block) in enumerate(ordered)
                if block["type"] == "fallback"
            ),
            default=-1,
        )
        return [
            (index, block)
            for position, (index, block) in enumerate(ordered)
            if position >= boundary or block["type"] in ("text", "fallback")
        ]

    def parts(self) -> list[Part]:
        parts: list[Part] = []
        ordered = self._ordered()
        # The output limit can cut off only the block that was open when it hit.
        cut_index = ordered[-1][0] if ordered and self.stop_reason in _CUT_OFF else None
        for index, block in ordered:
            kind = block.get("type")
            if kind == "text":
                if block.get("text"):
                    parts.append(TextPart(text=block["text"]))
            elif kind == "tool_use":
                parts.append(self._tool_call(index, block, cut=index == cut_index))
            elif kind == "thinking":
                text = block.get("thinking") or ""
                if block.get("signature"):
                    parts.append(
                        ReasoningPart(text=text, provider="anthropic", opaque=self._opaque(block))
                    )
                elif text:
                    parts.append(ReasoningPart(text=text, provider="anthropic"))
            else:
                # redacted_thinking, fallback, compaction and server-side blocks: kept as
                # received (server tool calls with their streamed input) and sent back.
                opaque = self._opaque(self._with_input(index, block))
                parts.append(ReasoningPart(text="", provider="anthropic", opaque=opaque))
        return parts

    def _with_input(self, index: int, block: dict[str, Any]) -> dict[str, Any]:
        """A server-side tool call block with the input streamed for it."""
        raw = self._json.get(index, "")
        if block.get("type") == "tool_use" or not raw.strip():
            return block
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = None
        if not isinstance(parsed, dict):
            if self.stop_reason in (*_CUT_OFF, "refusal"):
                return block  # cut off before it ran; never replayed, being unpaired
            # Never journal or replay a server call without the input it ran with.
            raise ModelResponseError(
                f"The input of a {block.get('type')} block in the response was not a JSON object.",
                provider="anthropic",
                model=self.model,
                ambiguous=True,
            )
        return {**block, "input": parsed}

    def _opaque(self, block: dict[str, Any]) -> dict[str, Any]:
        opaque: dict[str, Any] = {
            "provider": "anthropic",
            "model": self.served_by or self.model,
            "block": block,
        }
        if block.get("type") in _THINKING_TYPES:
            opaque["prefix"] = self.fingerprint
        return opaque

    def _tool_call(self, index: int, block: dict[str, Any], *, cut: bool = False) -> ToolCallPart:
        call_id, name = str(block.get("id") or ""), str(block.get("name") or "")
        raw = self._json.get(index, "")
        if cut:
            # The output limit ended the response inside this call: its input may be
            # incomplete even when what arrived parses (or nothing arrived). Never run it.
            return ToolCallPart(
                id=call_id,
                name=name,
                raw_arguments=raw,
                parse_error=prompts.TOOL_CALL_TRUNCATED,
            )
        if not raw.strip():
            arguments = block.get("input")
            if isinstance(arguments, dict):
                return ToolCallPart(
                    id=call_id,
                    name=name,
                    arguments=arguments,
                    raw_arguments=json.dumps(arguments, ensure_ascii=False),
                )
            return ToolCallPart(id=call_id, name=name, arguments={}, raw_arguments="{}")
        try:
            parsed = json.loads(raw)
        except ValueError as error:
            return ToolCallPart(id=call_id, name=name, raw_arguments=raw, parse_error=str(error))
        if not isinstance(parsed, dict):
            return ToolCallPart(
                id=call_id,
                name=name,
                raw_arguments=raw,
                parse_error=f"expected a JSON object, got {type(parsed).__name__}",
            )
        return ToolCallPart(id=call_id, name=name, arguments=parsed, raw_arguments=raw)

    def replay_blocks(self) -> list[dict[str, Any]]:
        """The received blocks as request content, for continuing a paused turn."""
        blocks = []
        for index, block in self._ordered():
            if block.get("type") == "tool_use":
                call = self._tool_call(index, block)
                block = {**block, "input": call.arguments}
            elif block.get("type") == "text" and not block.get("text"):
                continue
            else:
                block = self._with_input(index, block)
            blocks.append(block)
        return blocks

    def totals(self) -> _UsageEntry:
        """The top-level usage: the attempt that produced the returned message."""
        return _UsageEntry(
            model=self.served_by,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_read_tokens=self.cache_read_tokens,
            cache_creation_tokens=self.cache_creation_tokens,
            breakdown=self._cache_breakdown,
        )

    def usage(self) -> Usage:
        # Every attempt when the API itemizes them (declined fallback attempts and
        # compaction are absent from the top-level counts), else the top-level counts.
        entries = self.iterations or [self.totals()]
        return Usage(
            requests=1,
            # qagents counts the whole prompt as input; the API reports cached parts apart.
            input_tokens=sum(entry.prompt_tokens for entry in entries),
            output_tokens=sum(entry.output_tokens for entry in entries),
            cached_input_tokens=sum(entry.cache_read_tokens for entry in entries),
            reasoning_tokens=self.thinking_tokens,
        )

    def context_tokens(self) -> int:
        """Prompt plus output of the last sampling step: the size of the conversation now.

        From the last ``message`` entry of ``usage.iterations`` when there is one,
        never from a ``compaction`` entry or from summed counts.
        """
        sampled = [
            entry
            for entry in self.iterations or []
            if entry.kind in ("message", "fallback_message")
        ]
        last = sampled[-1] if sampled else self.totals()
        return last.prompt_tokens + last.output_tokens

    def metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if self.stop_reason and self.stop_reason not in ("end_turn", "tool_use"):
            metadata["stop_reason"] = self.stop_reason
        if self.stop_details:
            metadata["stop_details"] = self.stop_details
        if self.served_by and _base_id(self.served_by) != _base_id(self.model):
            metadata["served_by"] = self.served_by
        return metadata


def _retry_after(headers: Mapping[str, str]) -> float | None:
    milliseconds = headers.get("retry-after-ms")
    if milliseconds is not None:
        try:
            value = float(milliseconds) / 1000
        except ValueError:
            value = -1.0
        if value >= 0:
            return value
    seconds = headers.get("retry-after")
    if seconds is None:
        return None
    try:
        value = float(seconds)
    except ValueError:
        return None
    return value if value >= 0 else None
