"""Devvi ``SessionEvent`` (session format v4) to qagents events, and token usage.

`SessionEventMapper` reports the events of one turn to an
`qagents.external.ExternalTurnContext`; the SDK transport feeds it the ``session.event``
notifications of the turn, and crash re-attach feeds it the rows of Devvi's session log.
The mapping (design 2.5):

* ``step/start``: ``turn_started`` (a Devvi step is one model request).
* ``assistant/message``: ``reasoning_delta`` per ``reasoning`` block, ``text_delta`` per
  ``text`` block (two blocks with no tool call between them are separated by a blank
  line), then ``model_responded`` with the usage (added to the calling run at once).
* ``tool/call``: ``tool_call_started`` (``<agent>.<name>``, parsed ``arguments``,
  ``raw_arguments``).
* ``tool/result``: ``tool_call_finished`` (``is_error`` when ``isError`` or ``error``).
* ``assistant/attempt``: ``external_event`` ``attempt``.
* ``approval/asked``, ``approval/decided``: ``external_event`` ``approval_asked`` /
  ``approval_decided``, Devvi's audit records. The decision itself is the bridge's
  ``permission_requested`` / ``permission_decided`` event.
* ``llm/retry``, ``llm/retry-started``: ``external_event`` ``retry``.
* ``compaction/*``: ``external_event`` ``compaction``. A ``compaction/summary`` is a model
  request of its own (Devvi summarizes the conversation): its ``usage`` is added to the turn
  at once (`auxiliary_usage`), so token and cost budgets see it; one without ``usage`` that
  says it called the model (``llmStreamCall``) counts as one unpriced request.
* ``web/deepseek-search-llm-request``: the DeepSeek web search's own model request, which
  Devvi records without usage: one unpriced request (`WEB_SEARCH_USAGE`), and an
  ``external_event`` ``web_search``. Under ``Limits.max_cost_usd`` an unpriced request
  stops the turn (the spend can no longer be known), so disable Devvi's web tools there
  (the patch row ``{"id": "tool-web", "disabled": True}``).
* ``job/registered``, ``job/settled``: ``external_event`` ``job``.
* ``turn/end``: the turn's outcome (`turn_end_outcome`), with ``stop_reason`` in the
  vocabulary every external agent uses (``end_turn``, ``max_tokens``, ``cancelled``).
* Other known types are not forwarded (``raw_events=True``: ``external_event`` ``raw``).
* Unknown types are ignored and counted in ``raw["ignored_events"]``.

Events of a child session (after ``subagent.started``) become ``external_event``
``subagent_event`` with the child's ``session_id`` and a short summary; a child's token
usage (its messages, compactions and web searches) is added to the turn too, so budgets
see what subagents spend.

The mapper also keeps the arguments of every tool call of the tree, by session and call id,
for Devvi's approval requests (`SessionEventMapper.arguments_for`): call ids come from each
session's model, so they are unique only within one session.

Text arrives per committed block, never per token: Devvi sends no token stream.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from ...external import ExternalOutcome, ExternalStatus
from ...messages import Message, Part, ReasoningPart, TextPart, ToolCallPart, Usage

if TYPE_CHECKING:
    from ...external import ExternalTurnContext

logger = logging.getLogger("qagents.integrations.devvi")

KNOWN_EVENT_TYPES = frozenset(
    {
        "agent-preset/selected", "agent/inbox/spliced", "approval/asked", "approval/decided",
        "approval/policy", "assistant/attempt", "assistant/message", "command/done",
        "command/run", "compaction/end", "compaction/prune", "compaction/start",
        "compaction/summary", "deliverables/presented", "developer/message",
        "feedback/message-delete", "feedback/message-put", "feedback/record",
        "goal-workflow/bind", "goal-workflow/round", "goal/change", "hook/invoked",
        "hook/result", "image/offload", "job/registered", "job/settled", "llm/retry",
        "llm/retry-started", "model/selection", "permission/preset", "plan/mode",
        "request/context", "request/header", "sandbox/mode", "schedule/change",
        "session-log-deepseek/delivery-accepted", "session/end-seed", "session/title",
        "session/title-llm-request", "step/end", "step/start", "subagent/catalog",
        "subagent/descriptor", "subagent/model-selection-policy", "system/message",
        "team/member", "team/message/delivered", "team/message/queued", "team/task",
        "todo/write", "tool-workflow/agent-end", "tool-workflow/agent-start",
        "tool-workflow/run-end", "tool-workflow/run-start", "tool/call", "tool/ptc-dispatch",
        "tool/ptc-dispatch-start", "tool/result", "turn/end", "turn/start", "user/message",
        "web/deepseek-search-llm-request", "working-directory/change", "workspace/changes",
    }
)  # fmt: skip
"""Devvi's ``packages/core/session/src/known-event-types.ts`` (0.2.1-rc.3). Types outside
this set are counted as ignored; Devvi's log is merge-extensible."""

_TEXT_LIMIT = 2000


# --- usage -------------------------------------------------------------------------------------


class TokenPricing(BaseModel):
    """Token prices in US dollars per million tokens, for the SDK transport's cost.

    qagents has no price table for Devvi's models: supply the prices of the route you use
    (and the devvi.sh platform's margin, if it bills you through the platform).

        DevviAgent(..., transport="sdk",
                   pricing=TokenPricing(input=0.27, output=1.10, cache_read=0.07))

    Devvi reports disjoint counts: ``inputTokens`` is uncached input only, cache reads and
    cache writes are counted apart (``packages/llm/llm`` ``TokenUsage``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    input: float = Field(ge=0)
    """Uncached input tokens."""
    output: float = Field(ge=0)
    """Output tokens (reasoning included)."""
    cache_read: float | None = Field(default=None, ge=0)
    """Input tokens read from the cache; ``None``: the ``input`` price."""
    cache_write: float | None = Field(default=None, ge=0)
    """Input tokens written to the cache; ``None``: the ``input`` price."""

    def cost(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        """The cost in US dollars; ``input_tokens`` excludes cache reads and writes."""
        read = self.input if self.cache_read is None else self.cache_read
        write = self.input if self.cache_write is None else self.cache_write
        return (
            input_tokens * self.input
            + output_tokens * self.output
            + cache_read_tokens * read
            + cache_write_tokens * write
        ) / 1_000_000


def usage_from_devvi(raw: Mapping[str, Any] | None, pricing: TokenPricing | None = None) -> Usage:
    """One ``assistant/message`` usage as a qagents `Usage` (one request).

    qagents counts the whole prompt as ``input_tokens`` (cached parts included) and the
    cached part again as ``cached_input_tokens``, so ``input_tokens`` is
    ``inputTokens + cacheReadTokens + cacheWriteTokens``. ``cost_usd`` is set only with
    ``pricing``; a message without usage is counted as one unpriced request, so a cost
    budget fails closed.
    """
    if not isinstance(raw, Mapping):
        return Usage(requests=1, unpriced_requests=1)
    uncached = _count(raw.get("inputTokens"))
    output = _count(raw.get("outputTokens"))
    read = _count(raw.get("cacheReadTokens"))
    write = _count(raw.get("cacheWriteTokens"))
    cost = None
    if pricing is not None:
        cost = pricing.cost(
            input_tokens=uncached,
            output_tokens=output,
            cache_read_tokens=read,
            cache_write_tokens=write,
        )
    return Usage(
        requests=1,
        input_tokens=uncached + read + write,
        output_tokens=output,
        cached_input_tokens=read,
        reasoning_tokens=_count(raw.get("reasoningTokens")),
        cost_usd=cost,
    )


WEB_SEARCH_EVENT = "web/deepseek-search-llm-request"
"""The event Devvi's DeepSeek web search records for its own model request
(``packages/web/web-search-deepseek``): the request only, never its usage."""

WEB_SEARCH_USAGE = Usage(requests=1, unpriced_requests=1)
"""What one web search request counts: a request whose tokens and cost are unknown."""


def auxiliary_usage(data: Mapping[str, Any], pricing: TokenPricing | None = None) -> Usage | None:
    """The usage of a ``compaction/summary`` event: Devvi's summarization request.

    Returns ``usage`` as `usage_from_devvi` reads it; one unpriced request when the event
    has no usage but says it called the model (``llmStreamCall: true``); ``None`` for a
    summary that made no model call (a template summarizer).
    """
    raw = data.get("usage")
    if isinstance(raw, Mapping):
        return usage_from_devvi(raw, pricing)
    if data.get("llmStreamCall") is True:
        return Usage(requests=1, unpriced_requests=1)
    return None


def _count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


# --- outcomes ----------------------------------------------------------------------------------


TURN_END_KINDS: dict[str, tuple[ExternalStatus, str]] = {
    "completed": ("completed", "end_turn"),
    "forked": ("completed", "forked"),
    "max-tokens": ("limit_reached", "max_tokens"),
    "aborted": ("cancelled", "cancelled"),
    "blocked": ("error", "blocked"),
    "interrupted": ("interrupted", "interrupted"),
}
"""Devvi's ``turn/end.reason.kind`` -> (status, ``stop_reason``).

The ``stop_reason`` values are the ones every external agent of qagents reports for the
same outcome (ACP's ``end_turn``, ``max_tokens`` and ``cancelled``), so a check such as
``result.stop_reason == "max_tokens"`` works on both Devvi transports. Devvi's own word is
kept in ``raw["agent_stop_reason"]``. ``forked`` (a turn closed at a fork boundary, which
only fork seeds carry) and ``blocked`` (a hook refused the turn) have no common word and
pass through."""


def turn_end_outcome(reason: Mapping[str, Any] | None) -> ExternalOutcome:
    """``turn/end.reason`` as an `ExternalOutcome` (design 2.5).

    | ``kind`` | status | ``stop_reason`` |
    |---|---|---|
    | ``completed`` | ``completed`` | ``end_turn`` |
    | ``forked`` | ``completed`` | ``forked`` |
    | ``max-tokens`` | ``limit_reached`` | ``max_tokens`` |
    | ``aborted`` | ``cancelled`` | ``cancelled`` |
    | ``blocked`` | ``error`` | ``blocked`` |
    | ``error`` | ``error`` | the failure's message (``raw["error_code"]``: its code) |
    | ``interrupted`` | ``interrupted`` | ``interrupted`` (``raw["ambiguous"]``) |
    | anything else | ``error`` | the kind |

    ``raw["agent_stop_reason"]`` is Devvi's ``kind`` (see `TURN_END_KINDS`).
    """
    kind = reason.get("kind") if isinstance(reason, Mapping) else None
    raw: dict[str, Any] = {"agent_stop_reason": kind} if isinstance(kind, str) else {}
    known = TURN_END_KINDS.get(kind) if isinstance(kind, str) else None
    if known is not None:
        status, stop_reason = known
        if status == "interrupted":
            raw["ambiguous"] = True
        return ExternalOutcome(status=status, stop_reason=stop_reason, raw=raw)
    if kind == "error":
        failure = reason.get("error") if isinstance(reason, Mapping) else None
        message = failure.get("message") if isinstance(failure, Mapping) else None
        code = failure.get("code") if isinstance(failure, Mapping) else None
        if isinstance(code, str):
            raw["error_code"] = code
        text = message if isinstance(message, str) and message.strip() else "error"
        return ExternalOutcome(status="error", stop_reason=text, raw=raw)
    return ExternalOutcome(status="error", stop_reason=str(kind or "unknown turn end"), raw=raw)


# --- small helpers -----------------------------------------------------------------------------


def parse_arguments(raw: Any) -> tuple[dict[str, Any], str | None]:
    """A ``tool/call`` arguments string as ``(arguments, parse_error)``."""
    if isinstance(raw, Mapping):
        return dict(raw), None
    if not isinstance(raw, str) or not raw.strip():
        return {}, None
    try:
        value = json.loads(raw)
    except ValueError as error:
        return {}, f"the arguments are not JSON: {error}"
    if not isinstance(value, dict):
        return {}, "the arguments are not a JSON object"
    return value, None


def content_text(content: Any) -> str:
    """The text of a list of content blocks (``text`` blocks joined)."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        str(block.get("text") or "")
        for block in content
        if isinstance(block, Mapping) and block.get("type") == "text"
    )


def user_receipt(event: Mapping[str, Any]) -> list[tuple[str, str]]:
    """``(message id, text)`` of the user messages an ``agent/inbox/spliced`` event inserted."""
    if event.get("type") != "agent/inbox/spliced":
        return []
    data = event.get("data")
    inserted = data.get("inserted") if isinstance(data, Mapping) else None
    found: list[tuple[str, str]] = []
    for message in inserted if isinstance(inserted, list) else []:
        if not isinstance(message, Mapping) or not isinstance(message.get("id"), str):
            continue
        source = message.get("source")
        if isinstance(source, Mapping) and source.get("kind") not in (None, "user"):
            continue
        found.append((message["id"], content_text(message.get("content"))))
    return found


def _short(text: str, limit: int = _TEXT_LIMIT) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _scalars(data: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    """The listed keys of ``data`` whose values are scalars (strings shortened)."""
    if not isinstance(data, Mapping):
        return {}
    picked: dict[str, Any] = {}
    for key in keys:
        value = data.get(key)
        if isinstance(value, str):
            picked[key] = _short(value, 500)
        elif isinstance(value, (int, float, bool)) or value is None and key in data:
            picked[key] = value
    return picked


# --- the mapper ---------------------------------------------------------------------------------


class SessionEventMapper:
    """Report Devvi session events of one turn to an `ExternalTurnContext`.

    Args:
        turn: The turn to report to.
        root: The session id of the turn (the root of the session tree).
        pricing: Token prices; ``None`` leaves ``cost_usd`` unknown.
        raw_events: Forward events that have no qagents counterpart as ``raw``.
    """

    def __init__(
        self,
        turn: ExternalTurnContext,
        *,
        root: str,
        pricing: TokenPricing | None = None,
        raw_events: bool = False,
    ) -> None:
        self.turn = turn
        self.root = root
        self.pricing = pricing
        self.raw_events = raw_events
        self.tree: dict[str, str | None] = {root: None}
        """Session ids of the tree: child id -> parent id (the root's parent is ``None``)."""
        self.turn_end: Mapping[str, Any] | None = None
        """``reason`` of the root's last ``turn/end``."""
        self.turns_ended = 0
        self.turns_started = 0
        self.ignored = 0
        self._calls: dict[tuple[str, str], dict[str, Any]] = {}
        """Arguments of the tool calls of every session in the tree, by ``(session id, call
        id)``: each session's model makes its own call ids, so two sessions can share one."""
        self._messages = 0
        """Root ``assistant/message`` events mapped so far: each text block gets its own id
        (``text_delta(message_id=...)``), so blocks are separated by a blank line."""

    # -- routing ---------------------------------------------------------------------------

    def owns(self, session_id: Any) -> bool:
        return isinstance(session_id, str) and session_id in self.tree

    def depth_of(self, session_id: str) -> int:
        depth, current = 0, self.tree.get(session_id)
        while current is not None:
            depth += 1
            current = self.tree.get(current)
        return depth

    def handle_notification(self, method: str, params: Any) -> None:
        """One SDK notification (``session.event``, ``subagent.started``, ...)."""
        if not isinstance(params, Mapping):
            return
        if method == "session.event":
            event = params.get("event")
            if isinstance(event, Mapping):
                self.handle_event(params.get("sessionId"), event)
        elif method == "subagent.started":
            self.subagent_started(params)
        elif method == "subagent.finished":
            self.subagent_finished(params)

    def handle_event(self, session_id: Any, event: Mapping[str, Any]) -> None:
        """One SessionEvent of a session in the tree (others are dropped)."""
        if not self.owns(session_id):
            return
        try:
            if session_id == self.root:
                self._root_event(event)
            else:
                self._child_event(session_id, event)
        except Exception:  # a malformed event must not end the turn
            logger.exception("Could not map Devvi event %r", event.get("type"))
            self._count_ignored()

    def arguments_for(self, session_id: Any, call_id: str | None) -> dict[str, Any] | None:
        """The arguments of tool call ``call_id`` of session ``session_id``, for the approval
        request that session sent about it.

        Only that session's own calls are looked at: call ids are unique within one session,
        not across the tree, so another session's call with the same id says nothing about
        this request.

        Returns:
            The call's arguments; ``{}`` when the request names no call (the decision then
            rests on the tool name and the reason); ``None`` when it names a call that this
            session has not started, or a session outside the tree. Devvi records a call
            before it asks about it, so the caller refuses such a request.
        """
        if call_id is None:
            return {}
        if not self.owns(session_id):
            return None
        found = self._calls.get((session_id, call_id))
        return dict(found) if found is not None else None

    # -- subagents -------------------------------------------------------------------------

    def subagent_started(self, params: Mapping[str, Any]) -> None:
        parent, child = params.get("parentSessionId"), params.get("childSessionId")
        if not self.owns(parent) or not isinstance(child, str) or child in self.tree:
            return
        self.tree[child] = parent
        self.turn.raw.setdefault("subagents", []).append(child)
        self.turn.event(
            "subagent_started",
            {"parent_session_id": parent, "depth": self.turn.depth + self.depth_of(child)},
            session_id=child,
        )

    def subagent_finished(self, params: Mapping[str, Any]) -> None:
        child = params.get("childSessionId")
        if not self.owns(child) or child == self.root:
            return
        data = _scalars(params, ("provider", "status", "stopReason"))
        data["text"] = _short(content_text(params.get("lastAssistantMessage")))
        self.turn.event("subagent_finished", data, session_id=child)

    # -- root events -----------------------------------------------------------------------

    def _root_event(self, event: Mapping[str, Any]) -> None:
        kind = event.get("type")
        data = event.get("data")
        data = data if isinstance(data, Mapping) else {}
        turn = self.turn
        if kind == "assistant/message":
            self._assistant_message(data)
        elif kind == "tool/call":
            call_id = data.get("callId")
            if isinstance(call_id, str):
                arguments, error = parse_arguments(data.get("arguments"))
                raw = data.get("arguments") if isinstance(data.get("arguments"), str) else None
                self._calls[(self.root, call_id)] = arguments
                turn.tool_started(
                    call_id,
                    str(data.get("name") or "unknown"),
                    arguments,
                    raw_arguments=raw,
                    parse_error=error,
                )
        elif kind == "tool/result":
            self._tool_result(data)
        elif kind == "step/start":
            turn.step_started()
        elif kind == "turn/start":
            self.turns_started += 1
        elif kind == "turn/end":
            self.turns_ended += 1
            reason = data.get("reason")
            self.turn_end = reason if isinstance(reason, Mapping) else {"kind": None}
        elif kind == "assistant/attempt":
            turn.event("attempt", _scalars(data, ("turn", "step")))
        elif kind in ("approval/asked", "approval/decided"):
            name = "approval_asked" if kind == "approval/asked" else "approval_decided"
            turn.event(name, _scalars(data, ("id", "toolName", "callId", "reason", "outcome")))
        elif kind in ("llm/retry", "llm/retry-started"):
            turn.event("retry", {"type": kind, **_scalars(data, ("attempt", "delayMs", "reason"))})
        elif isinstance(kind, str) and kind.startswith("compaction/"):
            summary: dict[str, Any] = {"type": kind, **_scalars(data, ("reason", "turn"))}
            if kind == "compaction/summary":
                summary.update(_scalars(data, ("provider", "model")))
                usage = self._add_auxiliary(data)
                if usage is not None:
                    summary["usage"] = usage.model_dump(mode="json")
            turn.event("compaction", summary)
        elif kind == WEB_SEARCH_EVENT:
            usage = self._add_web_search()
            data_out = {"type": kind, **_scalars(data, ("endpoint", "apiVersion"))}
            turn.event("web_search", {**data_out, "usage": usage.model_dump(mode="json")})
        elif kind in ("job/registered", "job/settled"):
            turn.event(
                "job",
                {"type": kind, **_scalars(data, ("jobId", "kind", "label", "status", "detail"))},
            )
        elif kind in KNOWN_EVENT_TYPES:
            if self.raw_events:
                turn.event("raw", {"type": kind, "seq": event.get("seq"), "data": dict(data)})
        else:
            self._count_ignored()
            if self.raw_events:
                turn.event("unknown", {"type": kind, "seq": event.get("seq")})

    def _assistant_message(self, data: Mapping[str, Any]) -> None:
        turn = self.turn
        message = data.get("message")
        message = message if isinstance(message, Mapping) else {}
        parts: list[Part] = []
        calls = 0
        self._messages += 1
        for index, block in enumerate(message.get("content") or []):
            if not isinstance(block, Mapping):
                continue
            kind = block.get("type")
            if kind == "reasoning":
                text = str(block.get("text") or "")
                turn.reasoning_delta(text)
                parts.append(ReasoningPart(text=text, provider="devvi"))
            elif kind == "text":
                text = str(block.get("text") or "")
                # each committed block is its own: "Fixed it." and "Tests pass." read as two
                turn.text_delta(text, message_id=f"m{self._messages}:{index}")
                parts.append(TextPart(text=text))
            elif kind == "tool-call" and isinstance(block.get("id"), str):
                arguments, error = parse_arguments(block.get("arguments"))
                raw = block.get("arguments") if isinstance(block.get("arguments"), str) else None
                parts.append(
                    ToolCallPart(
                        id=block["id"],
                        name=f"{turn.name}.{block.get('name') or 'unknown'}",
                        arguments=arguments,
                        raw_arguments=raw,
                        parse_error=error,
                    )
                )
                calls += 1
            elif kind == "image":
                media = block.get("mediaType") or block.get("mimeType")
                turn.event("image", {"mimeType": media if isinstance(media, str) else None})
        source = message.get("source")
        source = source if isinstance(source, Mapping) else {}
        usage = usage_from_devvi(data.get("usage"), self.pricing)
        self._cache_writes(data.get("usage"))
        turn.model_responded(
            Message(role="assistant", parts=parts, name=turn.name),
            usage=usage,
            model=source.get("model") if isinstance(source.get("model"), str) else None,
            provider=source.get("provider") if isinstance(source.get("provider"), str) else None,
            finish_reason="tool_calls" if calls else "stop",
        )

    def _tool_result(self, data: Mapping[str, Any]) -> None:
        message = data.get("message")
        message = message if isinstance(message, Mapping) else {}
        call_id = message.get("toolCallId")
        if not isinstance(call_id, str):
            return
        error = data.get("error")
        is_error = bool(message.get("isError")) or isinstance(error, Mapping)
        metadata = {"error": dict(error)} if isinstance(error, Mapping) else {}
        self.turn.tool_finished(
            call_id, content_text(message.get("content")), is_error=is_error, metadata=metadata
        )

    def _add_auxiliary(self, data: Mapping[str, Any]) -> Usage | None:
        """Count a compaction's model request against the turn (and the calling run)."""
        usage = auxiliary_usage(data, self.pricing)
        if usage is not None:
            self._cache_writes(data.get("usage"))
            self.turn.add_usage(usage)
        return usage

    def _add_web_search(self) -> Usage:
        """Count a web search's model request: one request with an unknown cost."""
        usage = WEB_SEARCH_USAGE.model_copy()
        self.turn.add_usage(usage)
        return usage

    def _cache_writes(self, usage: Any) -> None:
        if isinstance(usage, Mapping) and _count(usage.get("cacheWriteTokens")):
            raw = self.turn.raw
            raw["cache_write_tokens"] = raw.get("cache_write_tokens", 0) + _count(
                usage.get("cacheWriteTokens")
            )

    # -- child events ----------------------------------------------------------------------

    def _child_event(self, session_id: str, event: Mapping[str, Any]) -> None:
        kind = event.get("type")
        data = event.get("data")
        data = data if isinstance(data, Mapping) else {}
        summary: dict[str, Any] = {
            "type": kind,
            "depth": self.turn.depth + self.depth_of(session_id),
        }
        if kind == "assistant/message":
            message = data.get("message")
            content = message.get("content") if isinstance(message, Mapping) else None
            usage = usage_from_devvi(data.get("usage"), self.pricing)
            self._cache_writes(data.get("usage"))
            self.turn.add_usage(usage)  # what a subagent spends counts against the budget
            summary["text"] = _short(content_text(content))
            summary["usage"] = usage.model_dump(mode="json")
        elif kind == "tool/call":
            arguments, _ = parse_arguments(data.get("arguments"))
            call_id = data.get("callId")
            if isinstance(call_id, str):
                self._calls[(session_id, call_id)] = arguments
            summary.update(_scalars(data, ("callId", "name")))
        elif kind == "tool/result":
            message = data.get("message")
            if isinstance(message, Mapping):
                summary["callId"] = message.get("toolCallId")
                summary["isError"] = bool(message.get("isError")) or "error" in data
        elif kind == "turn/end":
            reason = data.get("reason")
            summary["reason"] = reason.get("kind") if isinstance(reason, Mapping) else None
        elif kind == "compaction/summary":
            usage = self._add_auxiliary(data)  # what a subagent's compaction spends counts too
            if usage is not None:
                summary["usage"] = usage.model_dump(mode="json")
        elif kind == WEB_SEARCH_EVENT:
            summary["usage"] = self._add_web_search().model_dump(mode="json")
        elif kind not in ("turn/start", "step/start", "job/registered", "job/settled") and (
            not self.raw_events
        ):
            if kind not in KNOWN_EVENT_TYPES:
                self._count_ignored()
            return
        self.turn.event("subagent_event", summary, session_id=session_id)

    def _count_ignored(self) -> None:
        self.ignored += 1
        self.turn.raw["ignored_events"] = self.ignored

    # -- the end ---------------------------------------------------------------------------

    def outcome(self) -> ExternalOutcome | None:
        """How the root's last turn ended; ``None`` when no ``turn/end`` arrived."""
        if self.turn_end is None:
            return None
        return turn_end_outcome(self.turn_end)
