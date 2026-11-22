"""The provider-neutral conversation model.

A conversation is a list of `Message`. Each message has a role and a
list of typed parts. Provider adapters translate this model to and from their
wire format; nothing else in qagents knows what a provider's JSON looks like.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, SerializerFunctionWrapHandler, model_serializer

Role = Literal["system", "user", "assistant", "tool"]


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImagePart(BaseModel):
    """An image for the model to look at.

    Give either ``url`` (http(s) or ``data:``) or base64 ``data`` with its
    ``media_type``.
    """

    type: Literal["image"] = "image"
    url: str | None = None
    data: str | None = None
    media_type: str | None = None

    def as_url(self) -> str:
        if self.url:
            return self.url
        if self.data:
            return f"data:{self.media_type or 'image/png'};base64,{self.data}"
        raise ValueError("ImagePart needs either url or data")


class ToolCallPart(BaseModel):
    """A request from the model to run a tool."""

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw_arguments: str | None = None
    """The argument text exactly as the model produced it."""
    parse_error: str | None = None
    """Set when ``raw_arguments`` was not a JSON object; ``arguments`` is then empty."""


class ToolResultPart(BaseModel):
    """What a tool call produced. ``content`` is the text the model reads."""

    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReasoningPart(BaseModel):
    """Reasoning the model exposed before answering.

    ``opaque`` holds provider data (signatures, encrypted items) that must be
    sent back unchanged to the same provider; other providers ignore it.
    """

    type: Literal["reasoning"] = "reasoning"
    text: str = ""
    provider: str | None = None
    opaque: dict[str, Any] | None = None


Part = Annotated[
    TextPart | ImagePart | ToolCallPart | ToolResultPart | ReasoningPart,
    Field(discriminator="type"),
]


class Message(BaseModel):
    role: Role
    parts: list[Part] = Field(default_factory=list)
    name: str | None = None
    """Name of the agent that wrote an assistant message."""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def system(cls, text: str) -> Message:
        return cls(role="system", parts=[TextPart(text=text)])

    @classmethod
    def user(cls, content: str | list[Part]) -> Message:
        parts: list[Part] = [TextPart(text=content)] if isinstance(content, str) else list(content)
        return cls(role="user", parts=parts)

    @classmethod
    def assistant(cls, text: str = "", *, tool_calls: list[ToolCallPart] | None = None) -> Message:
        parts: list[Part] = []
        if text:
            parts.append(TextPart(text=text))
        parts.extend(tool_calls or [])
        return cls(role="assistant", parts=parts)

    @property
    def text(self) -> str:
        return "".join(part.text for part in self.parts if isinstance(part, TextPart))

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        return [part for part in self.parts if isinstance(part, ToolCallPart)]

    @property
    def tool_results(self) -> list[ToolResultPart]:
        return [part for part in self.parts if isinstance(part, ToolResultPart)]


class Usage(BaseModel):
    """Token and cost accounting, summed over one or more model requests."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float | None = None
    """The sum of the costs that are known; ``None`` when no request reported or
    allowed computing one. Requests without a cost are counted in
    ``unpriced_requests``, so a total that leaves some out says so."""
    unpriced_requests: int = 0
    """Requests that used tokens but whose cost is unknown, so ``cost_usd`` does
    not include them. A sum (`add`) counts a single response that used tokens
    and has no ``cost_usd`` as unpriced. A response with no tokens and no cost
    counts as free, so a model whose server reported no usage at all must say so
    by setting ``unpriced_requests=1`` (``Limits.max_cost_usd`` then stops the
    run). Left out of the serialised form while 0."""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_complete(self) -> bool:
        """Every request that used tokens has a known cost."""
        return _unpriced(self) == 0

    def add(self, other: Usage) -> Usage:
        known = [usage.cost_usd for usage in (self, other) if usage.cost_usd is not None]
        return Usage(
            requests=self.requests + other.requests,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cost_usd=sum(known) if known else None,
            unpriced_requests=_unpriced(self) + _unpriced(other),
        )

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        if not self.unpriced_requests:  # journals and reports keep their old shape
            data.pop("unpriced_requests", None)
        return data


def _unpriced(usage: Usage) -> int:
    """Requests of ``usage`` whose cost is unknown.

    A provider reports one response without a cost as ``cost_usd=None``; it is
    counted here, so a sum that mixes priced and unpriced responses keeps the count.
    """
    if usage.unpriced_requests or usage.cost_usd is not None:
        return usage.unpriced_requests
    return max(usage.requests, 1) if usage.total_tokens > 0 else 0
