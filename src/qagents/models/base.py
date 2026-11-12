"""The contract between the agent loop and a model provider.

A provider implements one method, `Model.stream`. It receives a
`ModelRequest` and yields stream events, ending with exactly one
`ResponseDone` that carries the complete `ModelResponse`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..errors import ConfigurationError, ModelResponseError
from ..messages import Message, Usage

FinishReason = Literal["stop", "tool_calls", "length", "content_filter"]


class ModelSettings(BaseModel):
    """Sampling and tool-choice settings. Unset fields are not sent."""

    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    seed: int | None = None
    tool_choice: str | None = None
    """``"auto"``, ``"none"``, ``"required"``, or the name of one tool."""
    parallel_tool_calls: bool | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    """Extra request fields passed to the provider as they are. A provider that
    publishes an allow-list (QCloud does) rejects unknown keys before sending."""

    def merged(self, override: ModelSettings | None) -> ModelSettings:
        if override is None:
            return self
        changes = override.model_dump(exclude_unset=True)
        extra = {**self.extra, **changes.pop("extra", {})}
        return self.model_copy(update={**changes, "extra": extra})


class ToolSpec(BaseModel):
    """What the model is told about one tool."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    strict: bool = False


class OutputSpec(BaseModel):
    """A JSON Schema the final answer must conform to."""

    name: str = "final_answer"
    json_schema: dict[str, Any]
    strict: bool = False


class ModelRequest(BaseModel):
    messages: list[Message]
    tools: list[ToolSpec] = Field(default_factory=list)
    settings: ModelSettings = Field(default_factory=ModelSettings)
    output: OutputSpec | None = None


class ModelResponse(BaseModel):
    message: Message
    finish_reason: FinishReason = "stop"
    usage: Usage = Field(default_factory=Usage)
    model: str | None = None
    provider: str | None = None
    response_id: str | None = None
    """The provider's id for this response, for support requests and logs."""
    context_tokens: int | None = None
    """Prompt plus output tokens of the last request the provider ran for this response:
    the size of the conversation now. Set when ``usage`` sums several requests (continued
    or server-compacted turns, fallback attempts), so it is not their total; ``None`` means
    the input plus output of ``usage`` is that size."""


class ModelCapabilities(BaseModel):
    """What a provider's catalog says about a model. ``None`` means unknown."""

    tools: bool | None = None
    vision: bool | None = None
    context_length: int | None = None
    max_output_tokens: int | None = None
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None

    def cost(self, usage: Usage) -> float | None:
        if self.input_price_per_million is None or self.output_price_per_million is None:
            return None
        return (
            usage.input_tokens * self.input_price_per_million
            + usage.output_tokens * self.output_price_per_million
        ) / 1_000_000


# --- stream events -----------------------------------------------------------


class TextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class ReasoningDelta(BaseModel):
    type: Literal["reasoning_delta"] = "reasoning_delta"
    text: str


class ToolCallDelta(BaseModel):
    """A tool call is being written. ``name`` is the name received so far."""

    type: Literal["tool_call_delta"] = "tool_call_delta"
    index: int
    id: str | None = None
    name: str = ""
    arguments_delta: str = ""


class ResponseDone(BaseModel):
    type: Literal["response_done"] = "response_done"
    response: ModelResponse


ModelStreamEvent = TextDelta | ReasoningDelta | ToolCallDelta | ResponseDone


class Model(ABC):
    """A chat model behind some provider."""

    name: str
    """The provider's id for the model."""
    provider: str = "unknown"
    """Short provider label used in errors, traces and ``ModelResponse.provider``."""

    @abstractmethod
    def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        """Send ``request`` and yield events, ending with one `ResponseDone`."""

    async def request(self, request: ModelRequest) -> ModelResponse:
        """Send ``request`` and return the complete response."""
        response: ModelResponse | None = None
        async for event in self.stream(request):
            if isinstance(event, ResponseDone):
                response = event.response
        if response is None:
            raise ModelResponseError(
                "The model stream ended without a response.",
                provider=self.provider,
                model=self.name,
                ambiguous=True,
            )
        return response

    async def capabilities(self) -> ModelCapabilities:
        """What is known about this model. The default knows nothing."""
        return ModelCapabilities()

    def validate_settings(self, settings: ModelSettings) -> None:  # noqa: B027 - optional
        """Raise if the provider is known to reject ``settings``. Called when an agent is built."""

    async def aclose(self) -> None:  # noqa: B027 - optional to override
        """Release connections the model owns."""


# --- "provider:model-id" strings ---------------------------------------------

_PROVIDERS: dict[str, Callable[[str], Model]] = {}


def register_provider(prefix: str, factory: Callable[[str], Model]) -> None:
    """Make ``Agent("<prefix>:<model-id>")`` build a model with ``factory(model_id)``."""
    _PROVIDERS[prefix] = factory


def resolve_model(model: Model | str) -> Model:
    if isinstance(model, Model):
        return model
    prefix, separator, name = model.partition(":")
    if not separator or not name:
        raise ConfigurationError(
            f"Cannot tell which provider {model!r} belongs to. Pass a Model instance, "
            "or a string of the form 'provider:model-id' such as 'qcloud:<model-id>'."
        )
    factory = _PROVIDERS.get(prefix)
    if factory is None:
        known = ", ".join(sorted(_PROVIDERS)) or "none"
        raise ConfigurationError(
            f"Unknown model provider {prefix!r} (registered: {known}). "
            "Register one with qagents.register_provider()."
        )
    return factory(name)
