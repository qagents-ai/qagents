"""The provider-neutral conversation model.

A conversation is a list of `Message`. Each message has a role and a
list of typed parts. Provider adapters translate this model to and from their
wire format; nothing else in qagents knows what a provider's JSON looks like.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

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
    """``None`` when no request reported or allowed computing a cost."""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, other: Usage) -> Usage:
        if self.cost_usd is None and other.cost_usd is None:
            cost = None
        else:
            cost = (self.cost_usd or 0.0) + (other.cost_usd or 0.0)
        return Usage(
            requests=self.requests + other.requests,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cost_usd=cost,
        )
