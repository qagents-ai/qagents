"""Keeping a long conversation inside the model's context window.

A `Compactor` is asked before every model request whether the history
should be replaced by a shorter one. The replacement is written to the session
journal as a ``compaction`` entry; the original messages stay in the journal
for audit and are simply no longer sent.
"""

from __future__ import annotations

from typing import Any

from . import prompts
from .messages import Message, ReasoningPart, TextPart, ToolCallPart, ToolResultPart
from .models.base import Model, ModelRequest, ModelResponse, ModelSettings, resolve_model
from .tools import RunContext


class Compactor:
    async def compact(
        self,
        ctx: RunContext[Any],
        messages: list[Message],
        last_response: ModelResponse,
        model: Model,
    ) -> list[Message] | None:
        """Return a shorter history to use from now on, or ``None`` to keep it."""
        raise NotImplementedError


class SummarizingCompactor(Compactor):
    """Replaces the older part of the conversation with a model-written summary.

    It triggers when the last request used more than ``trigger_tokens`` tokens
    (input plus output), keeps the most recent ``keep_last`` messages verbatim,
    and never separates a tool call from its result. The summary is written by
    ``model`` when given, otherwise by the agent's own model; its usage is
    counted in the run.
    """

    def __init__(
        self,
        *,
        trigger_tokens: int,
        keep_last: int = 6,
        model: Model | str | None = None,
        max_summary_tokens: int = 2000,
    ) -> None:
        if keep_last < 1:
            raise ValueError("keep_last must be at least 1")
        self.trigger_tokens = trigger_tokens
        self.keep_last = keep_last
        self.model = resolve_model(model) if model is not None else None
        self.max_summary_tokens = max_summary_tokens

    async def compact(
        self,
        ctx: RunContext[Any],
        messages: list[Message],
        last_response: ModelResponse,
        model: Model,
    ) -> list[Message] | None:
        if last_response.usage.total_tokens < self.trigger_tokens:
            return None
        cut = _safe_cut(messages, self.keep_last)
        if cut <= 0:
            return None
        older, recent = messages[:cut], messages[cut:]
        request = ModelRequest(
            messages=[
                Message.user(f"{render_transcript(older)}\n\n---\n\n{prompts.SUMMARY_REQUEST}")
            ],
            settings=ModelSettings(max_tokens=self.max_summary_tokens),
        )
        response = await (self.model or model).request(request)
        ctx.add_usage(response.usage)
        summary = response.message.text.strip()
        if not summary:
            return None
        return [Message.user(prompts.SUMMARY_PREFIX + summary), *recent]


def _safe_cut(messages: list[Message], keep_last: int) -> int:
    """Index where the kept tail starts, moved back so it never starts with a tool result."""
    cut = len(messages) - keep_last
    while cut > 0 and messages[cut].role == "tool":
        cut -= 1
    return cut


def render_transcript(messages: list[Message]) -> str:
    """A plain-text rendering of messages, safe to send to any model without tool definitions."""
    lines: list[str] = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, TextPart) and part.text:
                lines.append(f"{message.role.capitalize()}: {part.text}")
            elif isinstance(part, ToolCallPart):
                arguments = part.raw_arguments if part.raw_arguments is not None else part.arguments
                lines.append(f"Assistant called tool {part.name}: {arguments}")
            elif isinstance(part, ToolResultPart):
                label = "Tool error" if part.is_error else "Tool result"
                lines.append(f"{label} ({part.name}): {part.content}")
            elif isinstance(part, ReasoningPart):
                continue
    return "\n".join(lines)
