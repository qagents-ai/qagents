"""Translation between qagents messages and the OpenAI Chat Completions wire format.

Shared by every adapter that speaks Chat Completions (QCloud, and any other
OpenAI-compatible server). It tolerates the ways compatible servers differ:
tool-call deltas without an index, arguments sent as an object, a repeated
function name, a missing call id, and reasoning under either field name.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from ..messages import (
    ImagePart,
    Message,
    Part,
    ReasoningPart,
    TextPart,
    ToolCallPart,
    Usage,
)
from .base import (
    FinishReason,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ReasoningDelta,
    TextDelta,
    ToolCallDelta,
)

_FINISH_REASONS: dict[str, FinishReason] = {
    "stop": "stop",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "length": "length",
    "content_filter": "content_filter",
}


def to_wire_messages(messages: list[Message]) -> list[dict[str, Any]]:
    wire: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            wire.append({"role": "system", "content": message.text})
        elif message.role == "user":
            wire.append({"role": "user", "content": _user_content(message)})
        elif message.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": message.text or None}
            calls = message.tool_calls
            if calls:
                entry["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": _arguments_text(call)},
                    }
                    for call in calls
                ]
            elif entry["content"] is None:
                entry["content"] = ""
            wire.append(entry)
        else:  # tool
            for result in message.tool_results:
                wire.append(
                    {"role": "tool", "tool_call_id": result.tool_call_id, "content": result.content}
                )
    return wire


def _user_content(message: Message) -> str | list[dict[str, Any]]:
    if all(isinstance(part, TextPart) for part in message.parts):
        return message.text
    content: list[dict[str, Any]] = []
    for part in message.parts:
        if isinstance(part, TextPart):
            content.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            content.append({"type": "image_url", "image_url": {"url": part.as_url()}})
    return content


def _arguments_text(call: ToolCallPart) -> str:
    # Replay the model's own text when it was valid, so a server-side prompt
    # cache sees the same bytes. Invalid text is replaced: some chat templates
    # parse the arguments and fail the whole request on bad JSON.
    if call.raw_arguments is not None and call.parse_error is None:
        return call.raw_arguments
    return json.dumps(call.arguments, ensure_ascii=False)


def build_chat_body(
    model: str,
    request: ModelRequest,
    *,
    stream: bool,
    max_tokens_param: str = "max_tokens",
    include_usage: bool = True,
) -> dict[str, Any]:
    settings = request.settings
    body: dict[str, Any] = {"model": model, "messages": to_wire_messages(request.messages)}
    if settings.max_tokens is not None:
        body[max_tokens_param] = settings.max_tokens
    if settings.temperature is not None:
        body["temperature"] = settings.temperature
    if settings.top_p is not None:
        body["top_p"] = settings.top_p
    if settings.stop:
        body["stop"] = settings.stop
    if settings.seed is not None:
        body["seed"] = settings.seed
    if request.tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                    **({"strict": True} if spec.strict else {}),
                },
            }
            for spec in request.tools
        ]
        choice = settings.tool_choice
        if choice in ("auto", "none", "required"):
            body["tool_choice"] = choice
        elif choice:
            body["tool_choice"] = {"type": "function", "function": {"name": choice}}
        if settings.parallel_tool_calls is not None:
            body["parallel_tool_calls"] = settings.parallel_tool_calls
    if request.output is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": request.output.name,
                "schema": request.output.json_schema,
                "strict": request.output.strict,
            },
        }
    body.update(settings.extra)
    if stream:
        body["stream"] = True
        if include_usage:
            body["stream_options"] = {"include_usage": True}
    return body


def parse_usage(raw: Any) -> Usage:
    if not isinstance(raw, dict):
        return Usage(requests=1)
    prompt_details = raw.get("prompt_tokens_details") or {}
    completion_details = raw.get("completion_tokens_details") or {}
    return Usage(
        requests=1,
        input_tokens=_int(raw.get("prompt_tokens")),
        output_tokens=_int(raw.get("completion_tokens")),
        cached_input_tokens=_int(prompt_details.get("cached_tokens"))
        if isinstance(prompt_details, dict)
        else 0,
        reasoning_tokens=_int(completion_details.get("reasoning_tokens"))
        if isinstance(completion_details, dict)
        else 0,
    )


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


class _PartialCall:
    __slots__ = ("id", "name", "arguments")

    def __init__(self) -> None:
        self.id: str | None = None
        self.name = ""
        self.arguments = ""


class ChatAccumulator:
    """Builds a `ModelResponse` from Chat Completions chunks."""

    def __init__(self, *, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        self.response_id: str | None = None
        self.finish_reason: str | None = None
        self._text: list[str] = []
        self._reasoning: list[str] = []
        self._calls: dict[int, _PartialCall] = {}
        self._usage: Any = None

    def feed(self, chunk: dict[str, Any]) -> list[ModelStreamEvent]:
        """Take one streamed chunk; return the events it produced."""
        self._note_envelope(chunk)
        events: list[ModelStreamEvent] = []
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict):
                events.extend(self._take(delta))
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
        return events

    def feed_completion(self, completion: dict[str, Any]) -> None:
        """Take a complete, non-streamed response."""
        self._note_envelope(completion)
        for choice in completion.get("choices") or []:
            if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                self._take(message)
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]

    def _note_envelope(self, data: dict[str, Any]) -> None:
        if isinstance(data.get("id"), str) and data["id"]:
            self.response_id = data["id"]
        if isinstance(data.get("model"), str) and data["model"]:
            self.model = data["model"]
        if data.get("usage"):
            self._usage = data["usage"]

    def _take(self, delta: dict[str, Any]) -> list[ModelStreamEvent]:
        events: list[ModelStreamEvent] = []
        # vLLM renamed reasoning_content to reasoning; servers in the wild send either.
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            self._reasoning.append(reasoning)
            events.append(ReasoningDelta(text=reasoning))
        text = delta.get("content")
        if isinstance(text, str) and text:
            self._text.append(text)
            events.append(TextDelta(text=text))
        for position, raw in enumerate(delta.get("tool_calls") or []):
            if not isinstance(raw, dict):
                continue
            index = raw.get("index")
            if not isinstance(index, int):
                index = position
            call = self._calls.setdefault(index, _PartialCall())
            if isinstance(raw.get("id"), str) and raw["id"]:
                call.id = raw["id"]
            function = raw.get("function")
            fragment = ""
            if isinstance(function, dict):
                name = function.get("name")
                if isinstance(name, str) and name and name != call.name:
                    call.name += name
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    fragment = arguments
                    call.arguments += arguments
                elif isinstance(arguments, dict):
                    fragment = json.dumps(arguments, ensure_ascii=False)
                    call.arguments = fragment
            events.append(
                ToolCallDelta(index=index, id=call.id, name=call.name, arguments_delta=fragment)
            )
        return events

    @property
    def partial_text(self) -> str:
        return "".join(self._text)

    def result(self) -> ModelResponse:
        parts: list[Part] = []
        reasoning = "".join(self._reasoning)
        if reasoning:
            parts.append(ReasoningPart(text=reasoning, provider=self.provider))
        text = "".join(self._text)
        if text:
            parts.append(TextPart(text=text))
        seen: set[str] = set()
        for index in sorted(self._calls):
            partial = self._calls[index]
            if not partial.name:
                continue
            call_id = partial.id
            if not call_id or call_id in seen:
                call_id = f"call_{uuid.uuid4().hex[:24]}"
            seen.add(call_id)
            parts.append(_finish_call(call_id, partial))
        has_calls = any(isinstance(part, ToolCallPart) for part in parts)
        finish: FinishReason = _FINISH_REASONS.get(self.finish_reason or "", "stop")
        if has_calls and finish == "stop":
            finish = "tool_calls"
        return ModelResponse(
            message=Message(role="assistant", parts=parts),
            finish_reason=finish,
            usage=parse_usage(self._usage),
            model=self.model,
            provider=self.provider,
            response_id=self.response_id,
        )


def _finish_call(call_id: str, partial: _PartialCall) -> ToolCallPart:
    raw = partial.arguments
    if not raw.strip():
        return ToolCallPart(id=call_id, name=partial.name, arguments={}, raw_arguments="{}")
    try:
        parsed = json.loads(raw)
    except ValueError as error:
        return ToolCallPart(
            id=call_id, name=partial.name, raw_arguments=raw, parse_error=str(error)
        )
    if not isinstance(parsed, dict):
        return ToolCallPart(
            id=call_id,
            name=partial.name,
            raw_arguments=raw,
            parse_error=f"expected a JSON object, got {type(parsed).__name__}",
        )
    return ToolCallPart(id=call_id, name=partial.name, arguments=parsed, raw_arguments=raw)
