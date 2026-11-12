"""Translation between qagents messages and the OpenAI Chat Completions wire format.

Shared by every adapter that speaks Chat Completions (QCloud, and any other
OpenAI-compatible server). It tolerates the ways compatible servers differ:
tool-call deltas without an index (told apart by call id, name, position in
the delta and complete arguments), arguments sent as an object, a repeated function name, a missing
call id, reasoning under either field name, and finish reasons spelled
differently (``model_length``, ``MAX_TOKENS``, ...).
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from .. import prompts
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
    # How other servers and gateways say that the output token limit was reached.
    "max_tokens": "length",
    "max_output_tokens": "length",
    "model_length": "length",
    "model_context_window_exceeded": "length",
    "content_filter": "content_filter",
}

FAILED_FINISH_REASONS = frozenset({"error", "abort"})
"""Finish reasons with which a server reports that the generation itself failed."""

TRUNCATED_ARGUMENTS = prompts.TOOL_CALL_TRUNCATED
"""``parse_error`` of the last tool call of a response cut off by ``max_tokens``: the
default of `prompts.TOOL_CALL_TRUNCATED`, which is read when the call is built (kept as an
alias for compatibility).

The loop then refuses the call (``prompts.INVALID_ARGUMENTS_JSON``) instead of
running the tool with arguments the model never finished writing."""


def to_wire_messages(messages: list[Message]) -> list[dict[str, Any]]:
    wire: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            wire.append({"role": "system", "content": _joined_text(message)})
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


def _joined_text(message: Message) -> str:
    # Separate text parts stay separate: one per line, never run together.
    return "\n".join(part.text for part in message.parts if isinstance(part, TextPart))


def _user_content(message: Message) -> str | list[dict[str, Any]]:
    if all(isinstance(part, TextPart) for part in message.parts):
        return _joined_text(message)
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
    """The Chat Completions request body for ``request``.

    A strict tool or output schema is sent with ``additionalProperties: false``
    on every object, which strict mode requires. OpenAI's strict mode also
    requires every property to be listed in ``required``, so a strict tool
    there cannot have parameters with defaults.
    """
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
                    "parameters": _strict_schema(spec.parameters)
                    if spec.strict
                    else spec.parameters,
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
        schema = request.output.json_schema
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": request.output.name,
                "schema": _strict_schema(schema) if request.output.strict else schema,
                "strict": request.output.strict,
            },
        }
    body.update(settings.extra)
    if stream:
        body["stream"] = True
        if include_usage:
            body["stream_options"] = {"include_usage": True}
    return body


_SUBSCHEMA_MAPS = ("properties", "$defs", "definitions", "patternProperties")
_SUBSCHEMA_LISTS = ("anyOf", "allOf", "oneOf", "prefixItems")
_SUBSCHEMAS = ("items", "additionalProperties", "not")


def _strict_schema(schema: Any) -> Any:
    """A copy of ``schema`` with ``additionalProperties: false`` on every object."""
    if not isinstance(schema, dict):
        return schema
    strict = dict(schema)
    for key in _SUBSCHEMA_MAPS:
        if isinstance(strict.get(key), dict):
            strict[key] = {name: _strict_schema(sub) for name, sub in strict[key].items()}
    for key in _SUBSCHEMA_LISTS:
        if isinstance(strict.get(key), list):
            strict[key] = [_strict_schema(sub) for sub in strict[key]]
    for key in _SUBSCHEMAS:
        if isinstance(strict.get(key), dict):
            strict[key] = _strict_schema(strict[key])
    if strict.get("type") == "object" or "properties" in strict:
        strict.setdefault("additionalProperties", False)
    return strict


def parse_usage(raw: Any) -> Usage:
    """The usage a response reported. Without one, every count is zero: check
    `ChatAccumulator.usage_reported` before treating that as a free response."""
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


def _finish_text(value: Any) -> str | None:
    if isinstance(value, dict):  # some gateways send {"type": "stop", ...}
        value = value.get("type")
    return value.lower() if isinstance(value, str) and value else None


def _is_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except ValueError:
        return False


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
        """The server's finish reason, lower-cased."""
        self._text: list[str] = []
        self._reasoning: list[str] = []
        self._refusal: list[str] = []
        self._calls: dict[int, _PartialCall] = {}
        self._slot_of_id: dict[str, int] = {}
        self._slot_at: dict[int, int] = {}  # position in a delta -> the call last seen there
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
                self.finish_reason = _finish_text(choice["finish_reason"])
        return events

    def feed_completion(self, completion: dict[str, Any]) -> list[ModelStreamEvent]:
        """Take a complete, non-streamed response; return the events it amounts to."""
        self._note_envelope(completion)
        events: list[ModelStreamEvent] = []
        for choice in completion.get("choices") or []:
            if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                events.extend(self._take(message))
            if choice.get("finish_reason"):
                self.finish_reason = _finish_text(choice["finish_reason"])
        return events

    @property
    def usage_reported(self) -> bool:
        """Whether the server reported token usage for this response."""
        return isinstance(self._usage, dict)

    @property
    def failed(self) -> bool:
        """Whether the server ended the response with a failure (`FAILED_FINISH_REASONS`)."""
        return self.finish_reason in FAILED_FINISH_REASONS

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
        refusal = delta.get("refusal")  # OpenAI sends a refusal here, with no content
        if isinstance(refusal, str) and refusal:
            self._refusal.append(refusal)
            events.append(TextDelta(text=refusal))
        taken: set[int] = set()
        for position, raw in enumerate(delta.get("tool_calls") or []):
            if not isinstance(raw, dict):
                continue
            index = self._slot(raw, position, taken)
            taken.add(index)
            call = self._calls.setdefault(index, _PartialCall())
            if isinstance(raw.get("id"), str) and raw["id"]:
                call.id = raw["id"]
                self._slot_of_id.setdefault(raw["id"], index)
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

    def _slot(self, raw: dict[str, Any], position: int, taken: set[int]) -> int:
        """Which call a tool-call entry belongs to.

        An ``index`` decides. Without one, a known id decides. Otherwise the
        entry goes on with the call last seen at the same position of a delta
        (with one entry per delta, the call opened last), unless it plainly
        starts a new one: a new id, a different name, or the same name starting
        a new object after complete arguments. Two entries of one delta are
        never joined into one call.
        """
        index = raw.get("index")
        if isinstance(index, int) and not isinstance(index, bool):
            slot = index
        else:
            call_id = raw.get("id")
            before = self._slot_at.get(position)
            if isinstance(call_id, str) and call_id in self._slot_of_id:
                slot = self._slot_of_id[call_id]
            elif (
                before is not None and before not in taken and _continues(self._calls[before], raw)
            ):
                slot = before
            else:
                slot = max(self._calls, default=-1) + 1
        self._slot_at[position] = slot
        return slot

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
        refusal = "".join(self._refusal)
        if refusal:
            parts.append(TextPart(text=refusal))
        finish: FinishReason = _FINISH_REASONS.get(self.finish_reason or "", "stop")
        named = [index for index in sorted(self._calls) if self._calls[index].name]
        seen: set[str] = set()
        for index in [] if refusal else named:  # never run the tools of a refused turn
            partial = self._calls[index]
            call_id = partial.id
            if not call_id or call_id in seen:
                call_id = f"call_{uuid.uuid4().hex[:24]}"
            seen.add(call_id)
            # Only the last call can have been cut off: the model went on from the others.
            cut_off = finish == "length" and index == named[-1]
            parts.append(_finish_call(call_id, partial, cut_off=cut_off))
        has_calls = any(isinstance(part, ToolCallPart) for part in parts)
        if refusal:
            finish = "content_filter"
        elif has_calls and finish == "stop":
            finish = "tool_calls"
        details: dict[str, Any] = {}
        if self.finish_reason is not None and self.finish_reason not in (
            "stop",
            "tool_calls",
            "length",
            "content_filter",
        ):
            details["finish_reason"] = self.finish_reason  # keep what the server said
        if refusal:
            details["refusal"] = True
        return ModelResponse(
            message=Message(
                role="assistant",
                parts=parts,
                metadata={self.provider: details} if details else {},
            ),
            finish_reason=finish,
            usage=parse_usage(self._usage),
            model=self.model,
            provider=self.provider,
            response_id=self.response_id,
        )


def _continues(call: _PartialCall, raw: dict[str, Any]) -> bool:
    """Whether a tool-call entry without an index goes on with ``call``."""
    call_id = raw.get("id")
    if isinstance(call_id, str) and call_id and call.id and call_id != call.id:
        return False
    function = raw.get("function")
    if not isinstance(function, dict):
        return True
    name = function.get("name")
    if not (isinstance(name, str) and name):
        return True
    if call.name and name != call.name:
        return False
    # The same name again, starting an object after complete arguments: a
    # second call. Anything else (whitespace, nothing) goes on with this one.
    return not (_starts_object(function.get("arguments")) and _is_object(call.arguments))


def _starts_object(arguments: Any) -> bool:
    return isinstance(arguments, dict) or (
        isinstance(arguments, str) and arguments.lstrip().startswith("{")
    )


def _finish_call(call_id: str, partial: _PartialCall, *, cut_off: bool = False) -> ToolCallPart:
    raw = partial.arguments
    if cut_off and not _is_object(raw):
        return ToolCallPart(
            id=call_id,
            name=partial.name,
            raw_arguments=raw,
            parse_error=prompts.TOOL_CALL_TRUNCATED,
        )
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
