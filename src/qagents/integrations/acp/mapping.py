"""ACP v1 wire formats on the qagents side: updates to events, stop reasons to statuses.

Pure translation, no I/O. `TurnUpdates` maps one turn's ``session/update``
notifications onto the `ExternalTurnContext` (which emits the qagents events);
`outcome_for_stop_reason` and `outcome_for_prompt_error` map how
``session/prompt`` ended; `prompt_blocks` turns a task into ACP content blocks.

| ACP ``sessionUpdate`` | qagents |
|---|---|
| ``agent_message_chunk``, text | `TextDelta` (a block for Devvi, a token chunk for others) |
| ``agent_message_chunk``, other | ``external_event`` ``image``, ``audio``, ``resource_link``... |
| ``agent_thought_chunk`` | `ReasoningDelta` |
| ``tool_call`` | `ToolCallStarted` ``"<prefix>.<tool>"``; counts toward ``max_external_turns`` |
| ``tool_call_update`` | new arguments or name; ``completed``/``failed``: `ToolCallFinished` |
| ``usage_update {used, size}`` | ``external_event`` ``context`` (context pressure, not tokens) |
| ``config_option_update`` | ``external_event`` ``config`` |
| ``plan`` | ``external_event`` ``plan`` |
| anything else | ``external_event`` ``unknown`` (``raw_events=True``), else counted |

``external_event`` is `ExternalAgentEvent` with that ``kind``; bytes of images are never
forwarded. Kinds that are dropped are counted in ``ExternalResult.raw["ignored_events"]``
(``{kind: count}``).

| ``stopReason`` | status | ``stop_reason`` |
|---|---|---|
| ``end_turn`` | ``completed`` | ``end_turn`` |
| ``max_tokens`` | ``limit_reached`` | ``max_tokens`` |
| ``max_turn_requests`` | ``limit_reached`` | ``max_turn_requests`` |
| ``refusal`` | ``completed`` | ``content_filter`` |
| ``cancelled`` | ``cancelled`` | ``cancelled`` |
| JSON-RPC -32603 ``"Internal error: turn failed: X"`` | ``error`` | ``turn failed: X`` |
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ... import prompts as _prompts
from ...external import (
    ExternalAgentConfigurationError,
    ExternalAgentProtocolError,
    ExternalOutcome,
    ExternalStatus,
    ExternalTurnContext,
)
from ...messages import ImagePart, Part, TextPart
from .._jsonrpc import INTERNAL_ERROR, REQUEST_CANCELLED, JsonRpcError

logger = logging.getLogger("qagents.integrations.acp")

PROTOCOL_VERSION = 1
"""The ACP version this client speaks. An agent that answers another one is refused."""

AUTH_REQUIRED = -32000
"""ACP's error code for "authenticate first"."""

RESOURCE_NOT_FOUND = -32002
"""ACP's error code for a missing resource, such as an unknown session."""

STOP_REASONS: dict[str, tuple[ExternalStatus, str]] = {
    "end_turn": ("completed", "end_turn"),
    "max_tokens": ("limit_reached", "max_tokens"),
    "max_turn_requests": ("limit_reached", "max_turn_requests"),
    "refusal": ("completed", "content_filter"),
    "cancelled": ("cancelled", "cancelled"),
}
"""ACP ``stopReason`` -> (status, ``stop_reason``)."""

FINISHED_STATUSES = frozenset({"completed", "failed"})

ToolNamer = Callable[[Mapping[str, Any]], str]
"""``(tool call fields) -> tool name``: the name permission rules match after the prefix."""

TITLES_ARE_TOOL_NAMES = frozenset({"deepseek-harness-acp"})
"""``agentInfo.name`` of agents whose tool call ``title`` is the name of the tool the model
called, set by the agent's code: Devvi's ACP bridge. `ACPAgent` uses `title_tool_name` for
them and `default_tool_name` for every other agent."""

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]{0,63}$")


def default_tool_name(call: Mapping[str, Any]) -> str:
    """The tool name of an ACP tool call, for permission rules and the result footer.

    ACP gives a tool call a ``title`` for display, an optional ``name`` and a
    ``kind`` (``read``, ``edit``, ``execute``, ...). The name is, in order:

    1. ``name``, when the agent sends one;
    2. ``kind`` (``execute``, ``edit``, ...);
    3. ``"unknown"``.

    The title is never used. Most agents build it from the tool's input (``"make"``,
    ``"setup.py"``, ``"Run ls -la"``), so the model would choose the name that rules
    match, and a title such as ``"rm"`` would get past a deny rule on ``execute``.
    Use `title_tool_name` for an agent that puts real tool names there.
    """
    name = call.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    kind = call.get("kind")
    if isinstance(kind, str) and kind.strip():
        return kind.strip()
    return "unknown"


def title_tool_name(call: Mapping[str, Any]) -> str:
    """`default_tool_name`, but a ``title`` that looks like a tool name comes before ``kind``.

    For agents whose code sets the title to the tool's name, such as Devvi (``bash``,
    ``mcp__db__query``, with ``kind`` always ``other``). `ACPAgent` uses it by itself for
    the agents in `TITLES_ARE_TOOL_NAMES`; pass ``tool_name=title_tool_name`` for another
    agent that does the same. A title that is display text (``"Run ls -la"``) falls back
    to ``kind``.
    """
    name = call.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    title = call.get("title")
    if isinstance(title, str) and _IDENTIFIER.match(title.strip()):
        return title.strip()
    return default_tool_name(call)


def outcome_for_stop_reason(result: Any) -> ExternalOutcome:
    """The `ExternalOutcome` of a ``session/prompt`` result ``{"stopReason": ...}``.

    Raises:
        ExternalAgentProtocolError: The result has no ``stopReason`` string.
    """
    reason = result.get("stopReason") if isinstance(result, Mapping) else None
    if not isinstance(reason, str) or not reason:
        raise ExternalAgentProtocolError(
            f"The agent answered session/prompt with {result!r}, which has no stopReason.",
            reason=_prompts.ACP_BAD_ANSWER,
        )
    known = STOP_REASONS.get(reason)
    if known is None:
        logger.warning("Unknown ACP stopReason %r; treating the turn as completed", reason)
        status: ExternalStatus = "completed"
        stop_reason = reason
    else:
        status, stop_reason = known
    return ExternalOutcome(status=status, stop_reason=stop_reason, raw={"acp_stop_reason": reason})


def error_text(error: JsonRpcError) -> str:
    """The agent's error message without the generic JSON-RPC prefix.

    Devvi answers a failed turn with ``"Internal error: turn failed: <why>"``;
    the useful part is ``"turn failed: <why>"``. A message that is only the
    prefix falls back to ``data`` (``details`` or ``message`` in it, or the
    whole value).
    """
    message = (error.message or "").strip()
    if message == "Internal error":
        message = ""
    elif message.startswith("Internal error: "):
        message = message[len("Internal error: ") :].strip()
    if message:
        return message
    data = error.data
    if isinstance(data, Mapping):
        for key in ("details", "message", "error"):
            if isinstance(data.get(key), str) and data[key].strip():
                return data[key].strip()
    if data is not None:
        return str(data)
    return error.message or f"JSON-RPC error {error.code}"


def outcome_for_prompt_error(error: JsonRpcError, *, cancelling: bool) -> ExternalOutcome:
    """The `ExternalOutcome` of a ``session/prompt`` that was answered with an error.

    -32603 (a failed turn) and any other code become ``error`` with the agent's
    message as ``stop_reason``; -32800 (request cancelled) becomes ``cancelled``,
    and so does any error while the turn is being cancelled (agents should answer
    ``stopReason: "cancelled"`` then, but some answer an error).
    """
    raw = {"error_code": error.code, "error": error.message}
    if error.code == REQUEST_CANCELLED or cancelling:
        return ExternalOutcome(status="cancelled", stop_reason="cancelled", raw=raw)
    if error.code != INTERNAL_ERROR:
        logger.info("session/prompt failed with JSON-RPC error %s: %s", error.code, error.message)
    return ExternalOutcome(status="error", stop_reason=error_text(error), raw=raw)


def prompt_blocks(parts: Sequence[Part], prompt_capabilities: Mapping[str, Any]) -> list[dict]:
    """ACP content blocks for a task: text, and images when the agent accepts them.

    Raises:
        ExternalAgentConfigurationError: An image for an agent whose
            ``promptCapabilities.image`` is not true, or an image given only by
            an http(s) URL (ACP carries images as base64 data).
    """
    blocks: list[dict] = []
    for part in parts:
        if isinstance(part, TextPart):
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            if prompt_capabilities.get("image") is not True:
                raise ExternalAgentConfigurationError(
                    "The agent does not accept images in prompts (its promptCapabilities.image "
                    "is not true). Send text only."
                )
            data, mime_type = _image_data(part)
            blocks.append({"type": "image", "mimeType": mime_type, "data": data})
        else:  # _as_parts only lets text and images through; be explicit anyway
            raise ExternalAgentConfigurationError(
                f"An ACP prompt can hold text and images, not {type(part).__name__}."
            )
    return blocks


def _image_data(part: ImagePart) -> tuple[str, str]:
    if part.data:
        return part.data, part.media_type or "image/png"
    url = part.url or ""
    found = re.match(r"^data:([^;,]+);base64,(.*)$", url, re.DOTALL)
    if found:
        return found.group(2), found.group(1)
    raise ExternalAgentConfigurationError(
        "ACP sends images as base64 data; this image is a URL. Download it and pass "
        "ImagePart(data=..., media_type=...)."
    )


def content_text(content: Any) -> str:
    """The text of a tool call's ``content`` list (ACP ``ToolCallContent``), for its result.

    Text blocks are joined with newlines; a diff, a terminal or a non-text block
    becomes a short bracketed note (``[diff: path]``), without its bytes.
    """
    if not isinstance(content, list):
        return ""
    pieces: list[str] = []
    for item in content:
        if not isinstance(item, Mapping):
            continue
        kind = item.get("type")
        if kind == "content":
            block = item.get("content")
            if isinstance(block, Mapping):
                text = _block_text(block)
                if text:
                    pieces.append(text)
        elif kind == "diff":
            pieces.append(f"[diff: {item.get('path', '')}]")
        elif kind == "terminal":
            pieces.append(f"[terminal: {item.get('terminalId', '')}]")
    return "\n".join(pieces)


def _block_text(block: Mapping[str, Any]) -> str:
    kind = block.get("type")
    if kind == "text":
        text = block.get("text")
        return text if isinstance(text, str) else ""
    if kind in ("image", "audio"):
        return f"[{kind}: {block.get('mimeType', '')}]"
    if kind == "resource_link":
        return f"[resource_link: {block.get('uri', '')}]"
    if kind == "resource":
        resource = block.get("resource")
        if isinstance(resource, Mapping) and isinstance(resource.get("text"), str):
            return resource["text"]
        uri = resource.get("uri", "") if isinstance(resource, Mapping) else ""
        return f"[resource: {uri}]"
    return ""


def _arguments(raw_input: Any) -> tuple[dict[str, Any], str | None, str | None]:
    """``rawInput`` as tool-call arguments: (arguments, raw text, parse error)."""
    if raw_input is None:
        return {}, None, None
    if isinstance(raw_input, Mapping):
        return dict(raw_input), None, None
    if isinstance(raw_input, str):  # Devvi keeps model output that is not JSON as a string
        return {}, raw_input, "the agent's tool input is not a JSON object"
    return {"input": raw_input}, None, None


@dataclass
class _Call:
    """What is known about one tool call; ACP updates carry only what changed."""

    fields: dict[str, Any] = field(default_factory=dict)
    name: str = "unknown"
    started: bool = False
    finished: bool = False

    def merge(self, update: Mapping[str, Any]) -> bool:
        """Take the fields an update sets; whether it set new arguments."""
        new_input = False
        for key in ("title", "name", "kind", "status", "content", "locations"):
            if update.get(key) is not None:
                self.fields[key] = update[key]
        if update.get("rawInput") is not None:
            new_input = self.fields.get("rawInput") != update["rawInput"]
            self.fields["rawInput"] = update["rawInput"]
        return new_input


class TurnUpdates:
    """Maps one turn's ``session/update`` notifications onto its `ExternalTurnContext`.

    Args:
        turn: The turn the updates belong to.
        raw_events: Forward update kinds without a mapping as `ExternalAgentEvent`
            kind ``unknown`` (with the whole update) instead of only counting them.
        tool_name: How a tool call's name is chosen (default: `default_tool_name`). It
            runs again when a later update brings a ``name``, ``kind`` or ``title``.
        on_config: Called with the ``configOptions`` of a ``config_option_update``.
        chunks_are_blocks: Each text ``agent_message_chunk`` is a whole committed block
            (Devvi), not a token chunk: consecutive blocks are then separated by a blank
            line even when they share a ``messageId``. Otherwise chunks with the same
            ``messageId`` (or with none) are joined as they come, and a new
            ``messageId`` starts a new block.
    """

    def __init__(
        self,
        turn: ExternalTurnContext,
        *,
        raw_events: bool = False,
        tool_name: ToolNamer | None = None,
        on_config: Callable[[Any], None] | None = None,
        chunks_are_blocks: bool = False,
    ) -> None:
        self.turn = turn
        self.raw_events = raw_events
        self.tool_name = tool_name or default_tool_name
        self.on_config = on_config
        self.chunks_are_blocks = chunks_are_blocks
        self.calls: dict[str, _Call] = {}
        self._blocks = 0

    # -- entry points --------------------------------------------------------------------

    def apply(self, update: Any) -> None:
        """Map one ``params.update`` of a ``session/update`` notification."""
        if not isinstance(update, Mapping) or not isinstance(update.get("sessionUpdate"), str):
            self._ignore("malformed")
            return
        kind = update["sessionUpdate"]
        handler = self._handlers.get(kind)
        if handler is None:
            self._unknown(kind, update)
            return
        handler(self, update)

    def note_tool_call(self, tool_call: Any) -> str | None:
        """Take the fields a permission request's ``toolCall`` carries; returns its id.

        Devvi sends the ``tool_call`` update first and only the id here; other
        agents may describe the call in the request itself, so it is started (or
        updated) from these fields too.
        """
        if not isinstance(tool_call, Mapping):
            return None
        call_id = tool_call.get("toolCallId")
        if not isinstance(call_id, str) or not call_id:
            return None
        if call_id not in self.calls or any(
            tool_call.get(key) is not None for key in ("title", "name", "kind", "rawInput")
        ):
            self._tool(tool_call)
        return call_id

    def name_of(self, call_id: str) -> str:
        """The tool name chosen for ``call_id`` (``"unknown"`` when it was never seen)."""
        call = self.calls.get(call_id)
        return call.name if call is not None else "unknown"

    # -- the kinds -------------------------------------------------------------------------

    def _message(self, update: Mapping[str, Any]) -> None:
        content = update.get("content")
        if not isinstance(content, Mapping):
            self._ignore("agent_message_chunk")
            return
        kind = content.get("type")
        if kind == "text":
            text = content.get("text")
            if isinstance(text, str):
                self.turn.text_delta(text, message_id=self._block_id(update))
            return
        if kind in ("image", "audio"):
            self.turn.event(kind, {"mimeType": content.get("mimeType")})
        elif kind == "resource_link":
            data = {key: content.get(key) for key in ("name", "uri", "mimeType")}
            self.turn.event("resource_link", {k: v for k, v in data.items() if v is not None})
        elif kind == "resource":
            resource = content.get("resource")
            data = {}
            if isinstance(resource, Mapping):
                data = {k: resource[k] for k in ("uri", "mimeType") if resource.get(k) is not None}
            self.turn.event("resource", data)
        else:
            self._ignore(f"agent_message_chunk:{kind}")

    def _block_id(self, update: Mapping[str, Any]) -> str | None:
        """The id that tells text blocks apart (see ``chunks_are_blocks``)."""
        message_id = update.get("messageId")
        message_id = message_id if isinstance(message_id, str) and message_id else None
        if not self.chunks_are_blocks:
            return message_id
        self._blocks += 1
        return f"{message_id or 'block'}:{self._blocks}"

    def _thought(self, update: Mapping[str, Any]) -> None:
        content = update.get("content")
        if isinstance(content, Mapping) and isinstance(content.get("text"), str):
            self.turn.reasoning_delta(content["text"])
        else:
            self._ignore("agent_thought_chunk")

    def _tool(self, update: Mapping[str, Any]) -> None:
        call_id = update.get("toolCallId")
        if not isinstance(call_id, str) or not call_id:
            self._ignore("malformed")
            return
        call = self.calls.get(call_id)
        if call is None:
            call = self.calls[call_id] = _Call()
        new_input = call.merge(update)
        if call.started and any(update.get(key) is not None for key in ("name", "kind", "title")):
            self._rename(call_id, call)
        if not call.started:
            call.name = self.tool_name(call.fields)
            arguments, raw, problem = _arguments(call.fields.get("rawInput"))
            self.turn.tool_started(
                call_id,
                call.name,
                arguments,
                raw_arguments=raw,
                parse_error=problem,
                counts_as_step=True,
            )
            call.started = True
        elif new_input:  # new input replaces the arguments (and says if it is not JSON)
            arguments, raw, problem = _arguments(call.fields.get("rawInput"))
            self.turn.tool_started(
                call_id, call.name, arguments, raw_arguments=raw, parse_error=problem
            )
        status = call.fields.get("status")
        if status in FINISHED_STATUSES and not call.finished:
            call.finished = True
            metadata = {
                key: call.fields[key]
                for key in ("title", "kind")
                if isinstance(call.fields.get(key), str)
            }
            self.turn.tool_finished(
                call_id,
                content_text(call.fields.get("content")),
                is_error=status == "failed",
                metadata=metadata,
            )

    def _rename(self, call_id: str, call: _Call) -> None:
        """A later update described the call (a name, a kind): permission requests use it.

        A ``tool_call`` may come without a ``kind`` and a ``tool_call_update`` add it; the
        request that follows is decided on the name from everything known by then. The
        ``tool_call_started`` event already sent keeps the name the call started with; the
        result's tool call takes the new one.
        """
        name = self.tool_name(call.fields)
        if name == call.name:
            return
        call.name = name
        record = self.turn.tool_calls.get(call_id)
        if record is not None:
            record.name = name

    def _usage(self, update: Mapping[str, Any]) -> None:
        data = {key: update.get(key) for key in ("used", "size")}
        if update.get("cost") is not None:
            data["cost"] = update["cost"]
        self.turn.event("context", data)

    def _config(self, update: Mapping[str, Any]) -> None:
        options = update.get("configOptions")
        if self.on_config is not None and isinstance(options, list):
            self.on_config(options)
        self.turn.event("config", {"configOptions": options})

    def _plan(self, update: Mapping[str, Any]) -> None:
        self.turn.event("plan", {"entries": update.get("entries")})

    def _replayed(self, update: Mapping[str, Any]) -> None:
        self._ignore("user_message_chunk")  # an echo of the prompt, or history being replayed

    _handlers: dict[str, Callable[[TurnUpdates, Mapping[str, Any]], None]] = {
        "agent_message_chunk": _message,
        "agent_thought_chunk": _thought,
        "tool_call": _tool,
        "tool_call_update": _tool,
        "usage_update": _usage,
        "config_option_update": _config,
        "plan": _plan,
        "user_message_chunk": _replayed,
    }

    # -- the rest ----------------------------------------------------------------------------

    def _unknown(self, kind: str, update: Mapping[str, Any]) -> None:
        if self.raw_events:
            self.turn.event("unknown", {"sessionUpdate": kind, "update": dict(update)})
        else:
            self._ignore(kind)

    def _ignore(self, kind: str) -> None:
        counts = self.turn.raw.setdefault("ignored_events", {})
        counts[kind] = counts.get(kind, 0) + 1
