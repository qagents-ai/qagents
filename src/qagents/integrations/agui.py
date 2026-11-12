"""AG-UI: a run's events as Agent-User Interaction protocol events.

[AG-UI](https://docs.ag-ui.com) is the event protocol that CopilotKit and other
agent front ends read. `to_agui_events` projects a qagents event stream onto
it, and `agui_sse` encodes the result as server-sent events for an HTTP
response:

    from fastapi.responses import StreamingResponse
    from qagents.integrations.agui import AGUI_MEDIA_TYPE, agui_sse

    @app.post("/agent")
    async def run_agent(body: dict) -> StreamingResponse:
        session = Session(store, id=body["threadId"])
        stream = agent.stream(body["messages"][-1]["content"], session=session)
        events = agui_sse(stream, thread_id=body["threadId"], run_id=body["runId"])
        return StreamingResponse(events, media_type=AGUI_MEDIA_TYPE)

The mapping:

| qagents event | AG-UI events |
|---|---|
| `run_started` | `RUN_STARTED` |
| `turn_started` | `STEP_STARTED`, after `STEP_FINISHED` for the previous turn |
| `text_delta` | `TEXT_MESSAGE_START`, `TEXT_MESSAGE_CONTENT` ..., `TEXT_MESSAGE_END` |
| `reasoning_delta` | `REASONING_START` and `REASONING_MESSAGE_*` events, `REASONING_END` |
| `tool_call_delta`, `model_responded` | `TOOL_CALL_START`, `TOOL_CALL_ARGS`, `TOOL_CALL_END` |
| `tool_call_finished` | `TOOL_CALL_RESULT` |
| `tool_call_started` of an external call (see below) | `TOOL_CALL_START`, `_ARGS`, `_END` |
| `state_updated` | `STATE_SNAPSHOT` and a `CUSTOM` event |
| `approval_required`, `input_required` | `CUSTOM`, named after the event |
| `agent_switched`, `guardrail_tripped` | `CUSTOM`, named after the event |
| `subagent_event` | `CUSTOM` named `subagent_event`, or the `SUBAGENT_*` events |
| `run_finished` | `RUN_FINISHED`, with an interrupt outcome when the run paused |
| `external_event` `turn_finished` of a direct external turn | `RUN_FINISHED` or `RUN_ERROR` |
| an exception | `RUN_ERROR` |

An external agent (`qagents.external.ExternalAgent`) reports its tool calls, and a
tool server bound to a run its served calls, with ``tool_call_started`` and
``tool_call_finished`` only: no model response carries them. Such a call is
announced with ``TOOL_CALL_START``, ``TOOL_CALL_ARGS`` and ``TOOL_CALL_END`` on
the first of its events, so every result has a start. A resumed qagents run is
the exception: the calls it runs or closes before its first model request are
those of the response the paused run showed, which the client already has. The
calls those serve (`RunContext.call_tool`) are new, and are announced.

A direct external turn (``ExternalAgent.stream(...)``) has no ``run_finished``: its
last event, the ``external_event`` of kind ``turn_finished``, ends the run, with
``RUN_FINISHED`` when the turn's status is ``completed``, ``cancelled`` or
``limit_reached``, and with ``RUN_ERROR`` (code ``external_error`` or
``external_interrupted``) when it is ``error`` or ``interrupted``.

Events of other types are skipped, so a qagents release that adds event types
never breaks a client. The ``value`` of a `CUSTOM` event is the qagents event
as JSON, without its ``type``, and without what the client should not get: a
guardrail's ``reason`` (unless ``guardrail_reasons=True``), the text of
exceptions that tools raised (unless ``expose_errors=True``), and in the events
of sub-agents their usage, provider replay data and, with ``reasoning=False``,
their reasoning.

Every start event gets its end event, and messages and tool calls are never
interleaved: a text message ends before a tool call starts, which is what
strict AG-UI clients check. Tool calls are announced when the model has
finished writing them if the provider does not stream them. Message ids are
made from the AG-UI run id and a counter.

Output guardrails check the final answer after the model has written it, so
its text has already been streamed to the client when a guardrail blocks it.
A run blocked that way therefore ends with a ``MESSAGES_SNAPSHOT`` without the
blocked answer, whatever ``messages_snapshot`` says: clients replace their
messages with a snapshot, which takes the streamed text back. The text of
sub-agents in ``"custom"`` and ``"native"`` mode is not taken back. With
``hold_final_answer=True`` the answer is not streamed at all: each model
response is sent once the run goes on past it, so a blocked answer never
reaches the client. A blocked answer stays in the conversation
(``RunResult.messages``); a later run of the session reads it back marked
``metadata["blocked"]``, and its snapshots leave it out. A sub-agent in
``"native"`` mode holds its answer too, and sends it when its tool call ends
without an error.

A run that pauses for approvals or for deferred tools ends with a
``RUN_FINISHED`` event whose ``outcome`` lists the interrupts. When the client
answers them, `agui_resume` turns its ``resume`` entries back into the
``approvals`` and ``results`` of ``agent.resume``.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import AsyncGenerator, AsyncIterable, Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ValidationError
from pydantic_core import to_jsonable_python

from .. import prompts
from ..errors import (
    ApprovalTokenError,
    ConfigurationError,
    ModelConnectionError,
    ModelError,
    ModelRateLimitError,
    SessionError,
    ToolError,
)
from ..messages import ImagePart, Message, ReasoningPart, TextPart
from ..permissions import Approval, InputResult

# The text sent to the model (INPUT_CANCELLED) is in qagents.prompts.

# --- text and names sent to the client ----------------------------------------------

AGUI_MEDIA_TYPE = "text/event-stream"
"""Media type of the response that carries `agui_sse` output."""

STEP_NAME = "{agent}: turn {turn}"
"""``stepName`` of the step events that frame one model request and its tool calls."""

RUN_ERROR_MESSAGE = "The run failed. The server log has the details."
"""``RUN_ERROR`` message for an exception that has none in `RUN_ERROR_MESSAGES`,
unless ``expose_errors=True``."""

RUN_ERROR_MESSAGES: tuple[tuple[type[BaseException], str], ...] = (
    (ModelRateLimitError, "The model is busy. Try again shortly."),
    (ModelConnectionError, "The model provider could not be reached."),
    (ModelError, "The model request failed."),
    (ApprovalTokenError, "The answer to the paused run was refused."),
    (SessionError, "The conversation could not be loaded, resumed or saved."),
)
"""``RUN_ERROR`` messages by exception class, first match wins, unless
``expose_errors=True``. The text of the exception that ended the run is not
sent by default: qagents errors included, it can name hosts, paths and
provider responses. The exception's class name is sent as ``code``; the
server log has the text. For a tool that raised, see `TOOL_ERROR_MESSAGE`."""

TOOL_ERROR_MESSAGE = "{name!r} failed. The server log has the details."
"""What the client gets, unless ``expose_errors=True``, as the result of a
tool call that raised an exception (an agent used as a tool included): in
``TOOL_CALL_RESULT``, ``SUBAGENT_ERROR``, ``MESSAGES_SNAPSHOT`` and ``CUSTOM``
values. The model reads the exception's text, `prompts.TOOL_FAILED`; a
`ToolError`'s text is written for the model and is sent as it is."""

STREAM_ENDED_MESSAGE = "The event stream ended before the run finished."
"""``RUN_ERROR`` message when the qagents stream stops without ``run_finished``."""

CUSTOM_EVENTS: frozenset[str] = frozenset(
    {"approval_required", "input_required", "state_updated", "agent_switched", "guardrail_tripped"}
)
"""qagents event types forwarded as AG-UI ``CUSTOM`` events by default."""

APPROVAL_INTERRUPT_PREFIX = "approval:"
"""Interrupt ids of pending approvals are this prefix plus the tool call id."""

INPUT_INTERRUPT_PREFIX = "input:"
"""Interrupt ids of pending inputs (deferred tool calls) are this prefix plus the tool call id."""

APPROVAL_RESPONSE_SCHEMA: dict[str, Any] = Approval.model_json_schema()
"""``responseSchema`` of an approval interrupt: the client answers with an
`Approval`, without ``arguments`` (see `agui_resume`)."""
APPROVAL_RESPONSE_SCHEMA["properties"].pop("arguments")

SubagentMode = Literal["custom", "native", "ignore"]
"""How events of agents running as tools are projected; see `AGUIProjector`."""

logger = logging.getLogger("qagents.agui")

_LIFECYCLE = frozenset({"run_started", "run_finished", "subagent_event"})

_HOLDERS = {"model_responded": "response", "run_finished": "result"}
"""The field of an event that holds its usage and messages."""

_HELD = frozenset(
    {"text_delta", "reasoning_delta", "tool_call_delta", "model_responded", "tool_call_finished"}
)
"""With ``hold_final_answer``: events whose AG-UI events wait, a model response and the
results recorded with it."""

_GOES_ON = frozenset(
    {
        "turn_started",
        "tool_call_started",
        "approval_required",
        "input_required",
        "agent_switched",
        "subagent_event",
        "run_finished",
    }
)
"""With ``hold_final_answer``: events after which what waits is sent, unless the run
was blocked: the response was not the final answer, or the answer passed."""

_EXTERNAL_TURN_END = "turn_finished"
"""``kind`` of the ``external_event`` that ends a direct external turn
(``ExternalAgent.stream``), which has no ``run_finished``."""

_EXTERNAL_RUN_PREFIX = "ext:"
"""Start of the ``run_id`` of an external turn (``ext:<session>:<token>``). Its
``resumed`` says that a session was continued, not that a paused run goes on."""

_SERVED_ID = re.compile(r"^[A-Za-z0-9_-]{1,32}-[0-9a-f]{32}$")
"""The id `RunContext.call_tool` gives a call: its ``source``, ``-`` and 32 hex digits."""


@dataclass
class _StreamedCall:
    """A tool call the provider is streaming, keyed by its index in the response."""

    id: str | None = None
    name: str = ""
    buffer: str = ""
    """Arguments received before the call was started."""
    sent: str = ""
    """Arguments sent to the client."""
    started: bool = False
    ended: bool = False


class AGUIProjector:
    """Turns qagents events into AG-UI events, one at a time.

    `to_agui_events` drives one for you. Use it directly when the events reach
    you another way, for example through ``Hooks.on_event``:

        projector = AGUIProjector(thread_id="thread-1")
        for item in projector.feed(event):
            await queue.put(item)
        ...
        for item in projector.fail(error):  # or projector.close() if the events stop
            await queue.put(item)

    Events are dicts with the protocol's camelCase field names; fields without a
    value are left out.

    Args:
        thread_id: AG-UI ``threadId``. Defaults to the session id from ``run_started``.
        run_id: AG-UI ``runId``; pass the ``runId`` of the client's
            ``RunAgentInput``. Defaults to the qagents run id. qagents keeps the
            run id when a run is resumed, while AG-UI wants a new id for every
            run, so a resumed run gets the qagents id plus a random suffix.
        steps: Frame each model request and its tool calls with
            ``STEP_STARTED`` and ``STEP_FINISHED``.
        reasoning: Forward the reasoning text that models expose, sub-agents'
            included. Set it to ``False`` to keep reasoning away from the client.
        state_snapshots: Send ``STATE_SNAPSHOT`` with the full session state
            after every ``state_updated``.
        messages_snapshot: Send ``MESSAGES_SNAPSHOT`` with the whole
            conversation before ``RUN_FINISHED``. Off by default: the
            conversation can be large, and the client already has it. A run
            blocked by an output guardrail always gets one, without the
            blocked answer.
        interrupts: Put the pending approvals and inputs of a paused run in the
            ``outcome`` of ``RUN_FINISHED``, as AG-UI interrupts.
        subagents: Events of an agent running as a tool become one ``CUSTOM``
            event each (``"custom"``, safe for every client), the protocol's
            ``SUBAGENT_STARTED`` / ``SUBAGENT_FINISHED`` / ``SUBAGENT_ERROR``
            with the inner events tagged by ``subagentRunId`` (``"native"``, for
            clients that support subagents), or nothing (``"ignore"``).
        custom_events: qagents event types also sent as ``CUSTOM`` events,
            after any other mapping. Add the ``type`` of events your tools emit
            with ``ctx.emit``.
        expose_errors: Send the text of exceptions: in ``RUN_ERROR``, and as
            the result of a tool call that raised. Off by default, because
            error text, qagents errors' included, can name hosts, paths and
            provider responses: the client gets a fixed message from
            `RUN_ERROR_MESSAGES` and the exception's class name, or
            `TOOL_ERROR_MESSAGE` for a tool.
        guardrail_reasons: Keep the ``reason`` in ``guardrail_tripped``
            events. Off by default: the reason is meant for your code and
            logs, and often quotes what was blocked.
        hold_final_answer: Send the text, reasoning and tool calls of a model
            response only once the run goes on past it (a tool starts, an
            approval is asked, the next turn starts) or finishes without
            being blocked. A final answer that an output guardrail blocks
            then never reaches the client. Off by default, because the
            client then sees the final answer only when the run ends. Turn it
            on when the agent, or an agent it hands off to, has output
            guardrails. A run that raises drops what waits. Sub-agents hold
            their answers too in ``"native"`` mode, not in ``"custom"`` mode.

    Raises:
        ConfigurationError: ``subagents`` is not one of the modes above.
    """

    def __init__(
        self,
        *,
        thread_id: str | None = None,
        run_id: str | None = None,
        steps: bool = True,
        reasoning: bool = True,
        state_snapshots: bool = True,
        messages_snapshot: bool = False,
        interrupts: bool = True,
        subagents: SubagentMode = "custom",
        custom_events: Collection[str] = CUSTOM_EVENTS,
        expose_errors: bool = False,
        guardrail_reasons: bool = False,
        hold_final_answer: bool = False,
    ) -> None:
        if subagents not in ("custom", "native", "ignore"):
            raise ConfigurationError(
                f"subagents={subagents!r} is not a mode. Use 'custom', 'native' or 'ignore'."
            )
        self.thread_id = thread_id
        """The AG-UI ``threadId``; known once ``run_started`` was fed."""
        self.run_id = run_id
        """The AG-UI ``runId``; known once ``run_started`` was fed."""
        self.steps = steps
        self.reasoning = reasoning
        self.state_snapshots = state_snapshots
        self.messages_snapshot = messages_snapshot
        self.interrupts = interrupts
        self.subagents: SubagentMode = subagents
        self.custom_events = frozenset(custom_events)
        self.expose_errors = expose_errors
        self.guardrail_reasons = guardrail_reasons
        self.hold_final_answer = hold_final_answer

        self._held: list[dict[str, Any]] | None = None
        """AG-UI events that wait (``hold_final_answer``); ``None`` when nothing waits."""
        self._subagent_run_id: str | None = None
        """Set on the projector of an agent running as a tool (``subagents="native"``)."""
        self._started = False
        self._finished = False
        self._counter = 0
        self._step: str | None = None
        self._text_id: str | None = None
        self._reasoning_id: str | None = None
        self._open_call: _StreamedCall | None = None
        self._calls: dict[int, _StreamedCall] = {}
        """Tool calls of the current model response that the provider streams."""
        self._parent_id: str | None = None
        """The assistant message that tool calls of the current response belong to."""
        self._streamed_text = False
        self._streamed_reasoning = False
        self._unresulted: dict[str, str] = {}
        """Tool calls of the last response without a result: id -> name."""
        self._resulted: set[str] = set()
        """Tool calls this projector sent a result for."""
        self._announced: set[str] = set()
        """Tool calls this projector sent ``TOOL_CALL_START`` for."""
        self._replaying = False
        """A resumed qagents run before its first model request: the calls it runs or
        closes then belong to the response the paused run showed the client, except the
        calls they serve (`_served`)."""
        self._running: dict[str, bool] = {}
        """Calls with a ``tool_call_started`` and no ``tool_call_finished`` yet: id ->
        whether it is served (`_served`, decided when it started)."""
        self._unseen_handoffs: list[str] = []
        """Agents handed off to by a call this projector never saw (a resumed run)."""
        self._children: dict[tuple[str, str], AGUIProjector] = {}
        """Open sub-agent projectors by (tool call id, agent): the parallel children of one
        ``fan_out`` call share the call id and differ by agent label."""
        self._handlers = {
            "run_started": self._on_run_started,
            "turn_started": self._on_turn_started,
            "text_delta": self._on_text_delta,
            "reasoning_delta": self._on_reasoning_delta,
            "tool_call_delta": self._on_tool_call_delta,
            "model_responded": self._on_model_responded,
            "tool_call_started": self._on_tool_call_started,
            "tool_call_finished": self._on_tool_call_finished,
            "state_updated": self._on_state_updated,
            "agent_switched": self._on_agent_switched,
            "subagent_event": self._on_subagent_event,
            "external_event": self._on_external_event,
            "run_finished": self._on_run_finished,
        }

    @property
    def finished(self) -> bool:
        """True once ``RUN_FINISHED`` or ``RUN_ERROR`` was produced. Later events are ignored."""
        return self._finished

    def feed(self, event: Any) -> list[dict[str, Any]]:
        """Project one qagents event.

        Args:
            event: A qagents event: a pydantic model, or the same as a dict
                with a ``type``, such as ``event.model_dump(mode="json")``.
                Events of unknown types produce nothing, unless their type is
                in ``custom_events``.

        Returns:
            The AG-UI events it produces, in order; often none or one.
        """
        if self._finished:
            return []
        kind = _event_type(event)
        out: list[dict[str, Any]] = []
        target = self._hold(kind, event, out) if self.hold_final_answer else out
        handler = self._handlers.get(kind or "")
        if handler is not None:
            handler(event, target)
        if kind in self.custom_events and kind not in _LIFECYCLE and not self._finished:
            self._close_segments(target)
            value = self._public(kind, _dump(event))
            target.append(self._event("CUSTOM", name=kind, value=value))
        return out

    def fail(self, error: BaseException) -> list[dict[str, Any]]:
        """End the projection because the run raised ``error``.

        Returns:
            End events for whatever is still open, then ``RUN_ERROR``. Nothing
            if the run already finished.
        """
        if self._finished:
            return []
        out: list[dict[str, Any]] = []
        if self.expose_errors:
            message = str(error) or type(error).__name__
        else:
            message = next(
                (text for kind, text in RUN_ERROR_MESSAGES if isinstance(error, kind)),
                RUN_ERROR_MESSAGE,
            )
        self._close_all(out, error=message)
        out.append(self._event("RUN_ERROR", message=message, code=type(error).__name__))
        self._finished = True
        return out

    def close(self) -> list[dict[str, Any]]:
        """End the projection because the qagents events stopped.

        Returns:
            Nothing after ``run_finished``. Otherwise end events for whatever is
            open, then a ``RUN_ERROR`` saying the stream ended early.
        """
        if self._finished:
            return []
        out: list[dict[str, Any]] = []
        self._close_all(out, error=STREAM_ENDED_MESSAGE)
        out.append(self._event("RUN_ERROR", message=STREAM_ENDED_MESSAGE, code="stream_ended"))
        self._finished = True
        return out

    # -- lifecycle -------------------------------------------------------------------

    def _on_run_started(self, event: Any, out: list[dict[str, Any]]) -> None:
        if self._started:
            return
        self._started = True
        self._replaying = _replays(event)
        if self._subagent_run_id is not None:
            return  # the parent announced this run with SUBAGENT_STARTED
        if self.thread_id is None:
            self.thread_id = str(_field(event, "session_id") or uuid.uuid4().hex)
        if self.run_id is None:
            run_id = str(_field(event, "run_id") or uuid.uuid4().hex)
            if _field(event, "resumed"):
                run_id = f"{run_id}-{uuid.uuid4().hex[:8]}"
            self.run_id = run_id
        out.append(self._event("RUN_STARTED", threadId=self.thread_id, runId=self.run_id))

    def _on_run_finished(self, event: Any, out: list[dict[str, Any]]) -> None:
        result = _field(event, "result")
        self._close_all(out)
        messages = list(_field(result, "messages") or [])
        self._send_unseen_handoffs(messages, out)
        summary = {
            "status": _field(result, "status"),
            "stop_reason": _field(result, "stop_reason"),
            "output": to_jsonable_python(_field(result, "output"), fallback=str),
        }
        self._finished = True
        if self._subagent_run_id is not None:
            out.append(self._event("SUBAGENT_FINISHED", result=summary))
            return
        blocked = _blocks_the_answer("run_finished", event)
        if blocked:  # the snapshot replaces the client's messages: the streamed answer goes
            messages = _before_last_answer(messages)
        if messages and (self.messages_snapshot or blocked):
            snapshot = to_agui_messages(
                messages, id_prefix=f"{self.thread_id}-msg", expose_errors=self.expose_errors
            )
            out.append(self._event("MESSAGES_SNAPSHOT", messages=snapshot))
        outcome = None
        if self.interrupts:
            interrupts = _interrupts(result)
            if interrupts:
                outcome = {"type": "interrupt", "interrupts": interrupts}
        out.append(
            self._event(
                "RUN_FINISHED",
                threadId=self.thread_id,
                runId=self.run_id,
                result=summary,
                outcome=outcome,
            )
        )

    def _on_external_event(self, event: Any, out: list[dict[str, Any]]) -> None:
        """A direct external turn has no ``run_finished``: its ``turn_finished`` event ends
        the run. ``completed``, ``cancelled`` and ``limit_reached`` end with ``RUN_FINISHED``
        (as a qagents run with that status does); ``error`` and ``interrupted`` with
        ``RUN_ERROR``. Inside a sub-agent the parent's tool result ends it, as before."""
        if _field(event, "kind") != _EXTERNAL_TURN_END or self._subagent_run_id is not None:
            return
        if not self._started:
            return
        if self._held is not None:  # hold_final_answer: a finished turn was not blocked
            out.extend(self._held)
            self._held = None
        data = _field(event, "data") or {}
        status = data.get("status")
        self._close_all(out)
        self._finished = True
        if status in ("error", "interrupted"):
            message = RUN_ERROR_MESSAGE
            if self.expose_errors:
                message = str(data.get("stop_reason") or status)
            out.append(self._event("RUN_ERROR", message=message, code=f"external_{status}"))
            return
        summary = {"status": status, "stop_reason": data.get("stop_reason"), "output": None}
        out.append(
            self._event(
                "RUN_FINISHED",
                threadId=self.thread_id,
                runId=self.run_id,
                result=summary,
                outcome=None,
            )
        )

    def _on_turn_started(self, event: Any, out: list[dict[str, Any]]) -> None:
        self._replaying = False
        self._close_segments(out)
        self._end_response()
        if not self.steps:
            return
        self._close_step(out)
        self._step = STEP_NAME.format(
            agent=_field(event, "agent") or "agent", turn=_field(event, "turn") or 0
        )
        out.append(self._event("STEP_STARTED", stepName=self._step))

    # -- model output -------------------------------------------------------------------

    def _on_text_delta(self, event: Any, out: list[dict[str, Any]]) -> None:
        text = _field(event, "text")
        if not text:
            return  # AG-UI content deltas must not be empty
        self._close_reasoning(out)
        self._close_call(out)
        if self._text_id is None:
            self._text_id = self._parent_id = self._new_id("msg")
            out.append(self._event("TEXT_MESSAGE_START", messageId=self._text_id, role="assistant"))
        out.append(self._event("TEXT_MESSAGE_CONTENT", messageId=self._text_id, delta=text))
        self._streamed_text = True

    def _on_reasoning_delta(self, event: Any, out: list[dict[str, Any]]) -> None:
        text = _field(event, "text")
        if not self.reasoning or not text:
            return
        self._close_text(out)
        self._close_call(out)
        if self._reasoning_id is None:
            self._reasoning_id = self._new_id("reasoning")
            out.append(self._event("REASONING_START", messageId=self._reasoning_id))
            out.append(
                self._event(
                    "REASONING_MESSAGE_START", messageId=self._reasoning_id, role="reasoning"
                )
            )
        out.append(
            self._event("REASONING_MESSAGE_CONTENT", messageId=self._reasoning_id, delta=text)
        )
        self._streamed_reasoning = True

    def _on_tool_call_delta(self, event: Any, out: list[dict[str, Any]]) -> None:
        self._close_text(out)
        self._close_reasoning(out)
        call = self._calls.setdefault(_field(event, "index") or 0, _StreamedCall())
        call.id = call.id or _field(event, "id")
        call.name = _field(event, "name") or call.name
        fragment = _field(event, "arguments_delta") or ""
        if call.ended:
            return  # ended early by text or reasoning, or left for model_responded
        if call.started:
            if fragment:
                out.append(self._event("TOOL_CALL_ARGS", toolCallId=call.id, delta=fragment))
                call.sent += fragment
            return
        call.buffer += fragment
        # Start once the id is known and the arguments begin, by which point
        # providers have sent the whole name. Without an id, wait for the response.
        if not (call.id and call.name and call.buffer):
            return
        if any(other is not call and other.id == call.id for other in self._calls.values()):
            call.ended = True  # a repeated id: the response gives this call its own
            return
        if self._open_call is not None and not _is_json(self._open_call.sent):
            return  # the open call's arguments are not complete: this call waits
        self._close_call(out)
        self._start_call(out, call.id, call.name)
        out.append(self._event("TOOL_CALL_ARGS", toolCallId=call.id, delta=call.buffer))
        call.sent, call.buffer = call.buffer, ""
        call.started = True
        self._open_call = call

    def _on_model_responded(self, event: Any, out: list[dict[str, Any]]) -> None:
        self._replaying = False
        self._close_segments(out)
        message = _field(_field(event, "response"), "message")
        if isinstance(message, Mapping):
            try:
                message = Message.model_validate(dict(message))
            except ValidationError:
                logger.warning("model_responded carries a message that is not one; skipped")
        if isinstance(message, Message):
            # What the provider did not stream is sent whole, in the message's order.
            streamed_text, streamed_reasoning = self._streamed_text, self._streamed_reasoning
            for part in message.parts:
                if isinstance(part, ReasoningPart) and part.text and not streamed_reasoning:
                    self._on_reasoning_delta(part, out)
                    self._close_reasoning(out)
                elif isinstance(part, TextPart) and part.text and not streamed_text:
                    self._on_text_delta(part, out)
                    self._close_text(out)
            streamed = {call.id for call in self._calls.values() if call.started}
            self._unresulted = {}
            for call in message.tool_calls:
                self._unresulted[call.id] = call.name
                if call.id in streamed:
                    continue
                self._start_call(out, call.id, call.name)
                arguments = call.raw_arguments
                if arguments is None:
                    arguments = json.dumps(call.arguments, ensure_ascii=False)
                if arguments:
                    out.append(self._event("TOOL_CALL_ARGS", toolCallId=call.id, delta=arguments))
                out.append(self._event("TOOL_CALL_END", toolCallId=call.id))
        self._end_response()

    # -- tools, state, agents -----------------------------------------------------------

    def _on_tool_call_started(self, event: Any, out: list[dict[str, Any]]) -> None:
        call = _field(event, "call")
        call_id = _field(call, "id")
        if call_id is not None:
            call_id = str(call_id)
            served = self._running.get(call_id)
            if served is None:
                served = self._running[call_id] = self._served(event, call_id)
            self._announce(out, call_id, _field(call, "name"), call, served)

    def _served(self, event: Any, call_id: str) -> bool:
        """Whether a call is one that a running call serves (`RunContext.call_tool`), not a
        call of a model response. Such a call is new even in a resumed run that goes on
        with the paused response: the paused run never showed it.

        The event may name the calling call (``caller_tool_call_id``). Without that, a
        served call is one with an id of the form ``call_tool`` gives (``"<source>-"`` and
        32 hex digits) that came while another call of this run runs: the serving call
        must be running, and a provider's id of that form is then not taken for one unless
        calls of the paused response also run in parallel.
        """
        if _field(event, "caller_tool_call_id"):
            return True
        others = any(running != call_id for running in self._running)
        return others and _SERVED_ID.match(call_id) is not None

    def _announce(
        self, out: list[dict[str, Any]], call_id: str, name: Any, call: Any, served: bool
    ) -> None:
        """``TOOL_CALL_START``, ``TOOL_CALL_ARGS`` and ``TOOL_CALL_END`` for a call that no
        model response announced: an external agent's, or one a bound tool server ran.
        Nothing for a call already announced or resulted, or one a resumed run goes on
        with (the paused run announced it) unless it is ``served`` (`_served`)."""
        if self._replaying and not served:
            return
        if call_id in self._announced or call_id in self._resulted:
            return
        self._close_segments(out)
        name = str(name or "tool")
        self._start_call(out, call_id, name)
        arguments = _field(call, "raw_arguments")
        if arguments is None:
            arguments = json.dumps(
                to_jsonable_python(_field(call, "arguments") or {}, fallback=str),
                ensure_ascii=False,
            )
        if arguments:
            out.append(self._event("TOOL_CALL_ARGS", toolCallId=call_id, delta=arguments))
        out.append(self._event("TOOL_CALL_END", toolCallId=call_id))
        self._unresulted[call_id] = name

    def _on_tool_call_finished(self, event: Any, out: list[dict[str, Any]]) -> None:
        self._close_segments(out)
        result = _field(event, "result")
        call_id = _field(_field(event, "call"), "id") or _field(result, "tool_call_id")
        if call_id is not None:  # a refused served call has no tool_call_started
            call_id = str(call_id)
            served = self._running.pop(call_id, None)
            if served is None:  # finished only: decided now
                served = self._served(event, call_id)
            call = _field(event, "call")
            name = _field(call, "name") or _field(result, "name")
            self._announce(out, call_id, name, call, served)
        content = _field(result, "content") or ""
        if not self.expose_errors and _raised(result):
            content = TOOL_ERROR_MESSAGE.format(name=_field(result, "name") or "tool")
        keys = [key for key in self._children if key[0] == str(call_id)]
        for key in keys:
            child = self._children.pop(key)
            if _field(result, "is_error"):
                child._close_all(out, error=content)
                out.extend(child._end_subagent(error=content))
            else:
                # One child answers with the call's result; parallel children share one.
                # Its run ended unblocked (a blocked answer was dropped when its
                # guardrail tripped), so what it holds is sent before it closes.
                child._release_held(out)
                child._close_all(out)
                output = content if len(keys) == 1 else None
                out.extend(child._end_subagent(output=output))
        self._unresulted.pop(call_id, None)
        self._send_result(out, call_id, content)

    def _on_state_updated(self, event: Any, out: list[dict[str, Any]]) -> None:
        values = _field(event, "values")
        if self.state_snapshots and isinstance(values, Mapping):
            self._close_segments(out)
            snapshot = to_jsonable_python(dict(values), fallback=str)
            out.append(self._event("STATE_SNAPSHOT", snapshot=snapshot))

    def _on_agent_switched(self, event: Any, out: list[dict[str, Any]]) -> None:
        # The loop records the result of the accepted handoff call without an
        # event; every other call of that turn has had its tool_call_finished.
        # The event names the call; without the name, it is the handoff call of
        # the last response that has no result. A resumed run may not have seen
        # that response: the result is then taken from the messages at the end.
        target = _field(event, "to_agent") or ""
        call_id = _field(event, "tool_call_id") or next(
            (id_ for id_, name in self._unresulted.items() if name.startswith("transfer_to_")),
            None,
        )
        if call_id is None:
            self._unseen_handoffs.append(target)
            return
        self._unresulted.pop(call_id, None)
        self._close_segments(out)
        self._send_result(out, call_id, prompts.HANDOFF_RESULT.format(name=target))

    def _send_unseen_handoffs(self, messages: list[Any], out: list[dict[str, Any]]) -> None:
        """``TOOL_CALL_RESULT`` for the handoff calls this projector never saw, from the
        conversation: the latest result of a handoff to each agent that has none sent."""
        for target in self._unseen_handoffs:
            content = prompts.HANDOFF_RESULT.format(name=target)
            found = (
                str(_field(part, "tool_call_id"))
                for message in reversed(messages)
                for part in reversed(_field(message, "parts") or [])
                if _field(part, "type") == "tool_result"
                and _field(part, "content") == content
                and not _field(part, "is_error")
                and str(_field(part, "tool_call_id")) not in self._resulted
            )
            call_id = next(found, None)
            if call_id is not None:
                self._send_result(out, call_id, content)
        self._unseen_handoffs = []

    def _send_result(self, out: list[dict[str, Any]], call_id: Any, content: str) -> None:
        self._resulted.add(str(call_id))
        out.append(
            self._event(
                "TOOL_CALL_RESULT",
                messageId=self._new_id("result"),
                toolCallId=call_id,
                content=content,
                role="tool",
            )
        )

    def _on_subagent_event(self, event: Any, out: list[dict[str, Any]]) -> None:
        if self.subagents == "ignore":
            return
        if self.subagents == "custom":
            value = self._public("subagent_event", _dump(event))
            if value is not None:
                self._close_segments(out)
                out.append(self._event("CUSTOM", name="subagent_event", value=value))
            return
        inner = _field(event, "event")
        agent = str(_field(event, "agent") or "agent")
        call_id = _field(event, "tool_call_id")
        key = (str(call_id or ""), agent)
        child = self._children.get(key)
        if child is None:
            inner_run = _field(inner, "run_id") if _event_type(inner) == "run_started" else None
            scope = f"{call_id}-{agent}" if call_id else f"agent:{agent}"
            child_id = str(inner_run or f"{self.run_id or 'run'}-{scope}")
            child = AGUIProjector(
                thread_id=self.thread_id,
                run_id=child_id,
                steps=self.steps,
                reasoning=self.reasoning,
                state_snapshots=self.state_snapshots,
                interrupts=self.interrupts,
                subagents=self.subagents,
                custom_events=self.custom_events,
                expose_errors=self.expose_errors,
                guardrail_reasons=self.guardrail_reasons,
                hold_final_answer=self.hold_final_answer,
            )
            child._subagent_run_id = child_id
            child._started = True
            child._replaying = _event_type(inner) == "run_started" and _replays(inner)
            self._children[key] = child
            out.append(
                self._event(
                    "SUBAGENT_STARTED",
                    subagentRunId=child_id,
                    name=agent,
                    parentToolCallId=call_id,
                    parentSubagentRunId=self._subagent_run_id,
                )
            )
        out.extend(child.feed(inner))
        if child.finished:
            del self._children[key]

    def _end_subagent(
        self, *, output: Any = None, error: str | None = None
    ) -> list[dict[str, Any]]:
        """``SUBAGENT_FINISHED`` or ``SUBAGENT_ERROR`` for this (child) projector."""
        if self._finished:
            return []
        self._finished = True
        if error is not None:
            return [self._event("SUBAGENT_ERROR", message=error)]
        summary = {"status": "completed", "output": output} if output is not None else None
        return [self._event("SUBAGENT_FINISHED", result=summary)]

    # -- plumbing -------------------------------------------------------------------------

    def _hold(
        self, kind: str | None, event: Any, out: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Where the AG-UI events of a ``kind`` event go with ``hold_final_answer``.

        A model response, and the results recorded with it, wait. What waits is
        dropped when an output guardrail blocks the run, and sent before an
        event that shows the run went on, or finished unblocked. Other events
        are sent at once, after the waiting messages and calls are ended.
        """
        if kind in _HELD and (self._held is not None or kind != "tool_call_finished"):
            if self._held is None:
                self._held = []
            return self._held
        if self._held is None:
            return out
        if _blocks_the_answer(kind, event):
            self._drop_held()
        elif kind in _GOES_ON:
            out.extend(self._held)
            self._held = None
        else:
            self._close_segments(self._held)
        return out

    def _release_held(self, out: list[dict[str, Any]]) -> None:
        """Send what waits (``hold_final_answer``), its children's first."""
        for child in self._children.values():
            child._release_held(out)
        if self._held is not None:
            out.extend(self._held)
            self._held = None

    def _drop_held(self) -> None:
        """Forget what waits; its open messages and calls were never started for the client."""
        if self._held is not None:
            self._close_segments([])
            self._held = None

    def _public(self, kind: str | None, value: Any) -> Any:
        """The JSON ``value`` of a ``kind`` event as the client may see it; ``None``: nothing.

        A guardrail's reason is left out unless ``guardrail_reasons``, and the
        text of tool calls that raised unless ``expose_errors``. From the
        events of sub-agents, at any depth, usage and provider replay data
        (``opaque``) are left out, as the parent's own events never carry them,
        and so is reasoning unless ``reasoning``.
        """
        if not isinstance(value, dict):
            return value
        if not self.expose_errors:
            value = _without_exception_text(value)
        if kind == "guardrail_tripped" and not self.guardrail_reasons:
            return {key: item for key, item in value.items() if key != "reason"}
        inner = value.get("event")
        if kind != "subagent_event" or not isinstance(inner, dict):
            return value
        inner_kind = inner.get("type")
        if inner_kind == "reasoning_delta" and not self.reasoning:
            return None
        holder = _HOLDERS.get(inner_kind or "")
        if holder is not None and isinstance(inner.get(holder), dict):
            data = {key: item for key, item in inner[holder].items() if key != "usage"}
            inner = {**inner, holder: _without_reasoning(data, keep_text=self.reasoning)}
        inner = self._public(inner_kind, inner)
        return None if inner is None else {**value, "event": inner}

    def _event(self, type_: str, **fields: Any) -> dict[str, Any]:
        event: dict[str, Any] = {"type": type_}
        event.update((key, value) for key, value in fields.items() if value is not None)
        if self._subagent_run_id is not None and "subagentRunId" not in event:
            event["subagentRunId"] = self._subagent_run_id
        return event

    def _new_id(self, kind: str) -> str:
        self._counter += 1
        return f"{self.run_id or 'run'}-{kind}-{self._counter}"

    def _start_call(self, out: list[dict[str, Any]], call_id: str, name: str) -> None:
        self._announced.add(str(call_id))
        if self._parent_id is None:
            self._parent_id = self._new_id("msg")
        out.append(
            self._event(
                "TOOL_CALL_START",
                toolCallId=call_id,
                toolCallName=name,
                parentMessageId=self._parent_id,
            )
        )

    def _close_text(self, out: list[dict[str, Any]]) -> None:
        if self._text_id is not None:
            out.append(self._event("TEXT_MESSAGE_END", messageId=self._text_id))
            self._text_id = None

    def _close_reasoning(self, out: list[dict[str, Any]]) -> None:
        if self._reasoning_id is not None:
            out.append(self._event("REASONING_MESSAGE_END", messageId=self._reasoning_id))
            out.append(self._event("REASONING_END", messageId=self._reasoning_id))
            self._reasoning_id = None

    def _close_call(self, out: list[dict[str, Any]]) -> None:
        if self._open_call is not None:
            out.append(self._event("TOOL_CALL_END", toolCallId=self._open_call.id))
            self._open_call.ended = True
            self._open_call = None

    def _close_segments(self, out: list[dict[str, Any]]) -> None:
        self._close_text(out)
        self._close_reasoning(out)
        self._close_call(out)

    def _close_step(self, out: list[dict[str, Any]]) -> None:
        if self._step is not None:
            out.append(self._event("STEP_FINISHED", stepName=self._step))
            self._step = None

    def _close_all(self, out: list[dict[str, Any]], *, error: str | None = None) -> None:
        """End every open message, tool call, step and subagent, children first.

        What waits (``hold_final_answer``) is dropped: the run ended without
        showing that it went on past it.
        """
        self._drop_held()
        self._close_segments(out)
        for child in self._children.values():
            child._close_all(out, error=error)
            out.extend(child._end_subagent(error=error))
        self._children = {}
        self._close_step(out)

    def _end_response(self) -> None:
        self._calls = {}
        self._parent_id = None
        self._streamed_text = False
        self._streamed_reasoning = False


def to_agui_events(
    stream: AsyncIterable[Any],
    *,
    thread_id: str | None = None,
    run_id: str | None = None,
    **options: Any,
) -> AsyncGenerator[dict[str, Any], None]:
    """Project a qagents event stream onto AG-UI events.

        stream = agent.stream("Plan my trip", session=session)
        async for event in to_agui_events(stream, thread_id=thread, run_id=run):
            await websocket.send_json(event)

    An exception from the stream does not propagate: it is logged on the
    ``qagents.agui`` logger and reported to the client as ``RUN_ERROR``. When
    you stop iterating (the client went away), the qagents stream is closed,
    which stops the run.

    Args:
        stream: ``agent.stream(...)`` or ``agent.resume_stream(...)``, a direct
            external turn (``ExternalAgent.stream(...)``), or any async iterable of
            qagents events.
        thread_id: AG-UI ``threadId``; defaults to the session id.
        run_id: AG-UI ``runId``; pass the one from the client's ``RunAgentInput``.
        **options: The other options of `AGUIProjector`.

    Returns:
        An async generator of AG-UI events as dicts, ending with
        ``RUN_FINISHED`` or ``RUN_ERROR``.

    Raises:
        TypeError: An option is not an `AGUIProjector` option (raised here, not
            when iterating).
        ConfigurationError: An option has a value it does not accept.
    """
    projector = AGUIProjector(thread_id=thread_id, run_id=run_id, **options)
    return _project(stream, projector)


async def _project(
    stream: AsyncIterable[Any], projector: AGUIProjector
) -> AsyncGenerator[dict[str, Any], None]:
    try:
        try:
            async for event in stream:
                for item in projector.feed(event):
                    yield item
        except Exception as error:
            logger.exception("The run failed; reporting RUN_ERROR to the AG-UI client")
            for item in projector.fail(error):
                yield item
            return
        for item in projector.close():
            yield item
    finally:
        close = getattr(stream, "aclose", None)
        if close is not None:
            await close()


def encode_sse(event: Mapping[str, Any]) -> str:
    """Encode one AG-UI event as a server-sent event: ``data: <json>`` and a blank line."""
    data = json.dumps(event, ensure_ascii=False, separators=(",", ":"), default=str)
    return f"data: {data}\n\n"


async def agui_sse(stream: AsyncIterable[Any], **options: Any) -> AsyncGenerator[str, None]:
    """`to_agui_events`, encoded as server-sent events.

    Send it as the body of a response with media type `AGUI_MEDIA_TYPE`.

    Args:
        stream: ``agent.stream(...)``, ``agent.resume_stream(...)`` or any async
            iterable of qagents events.
        **options: ``thread_id``, ``run_id`` and the options of `AGUIProjector`.

    Yields:
        One ``data: ...`` message per AG-UI event.
    """
    events = to_agui_events(stream, **options)
    try:
        async for event in events:
            yield encode_sse(event)
    finally:
        await events.aclose()


def to_agui_messages(
    messages: Iterable[Message | Mapping[str, Any]],
    *,
    id_prefix: str = "msg",
    expose_errors: bool = False,
) -> list[dict[str, Any]]:
    """Convert a conversation into AG-UI messages, as sent in ``MESSAGES_SNAPSHOT``.

    qagents messages have no ids, so ids are ``<id_prefix>-<position>``; tool
    results become one ``tool`` message each, ``<id_prefix>-<position>-<n>``.
    Reasoning is left out, and an assistant message with neither text nor tool
    calls is skipped, and so is an answer an output guardrail blocked
    (``metadata["blocked"]``, set when the session is read back) with the
    results of its tool calls.

    Args:
        messages: The conversation, for example ``result.messages``, as
            `Message` objects or their JSON dicts.
        id_prefix: Start of every message id. Use something unique to the thread.
        expose_errors: Keep the text of the exception in the result of a tool
            call that raised. Off by default: it becomes `TOOL_ERROR_MESSAGE`.

    Returns:
        AG-UI message dicts with camelCase field names.

    Raises:
        pydantic.ValidationError: A dict is not a message.
    """
    converted: list[dict[str, Any]] = []
    hidden: set[str] = set()  # calls of answers an output guardrail blocked
    for index, item in enumerate(messages):
        message = item if isinstance(item, Message) else Message.model_validate(dict(item))
        base = f"{id_prefix}-{index}"
        if message.role == "assistant" and message.metadata.get("blocked"):
            hidden.update(call.id for call in message.tool_calls)
            continue
        if message.role == "tool":
            for position, result in enumerate(message.tool_results):
                if result.tool_call_id in hidden:
                    continue
                content = result.content
                if not expose_errors and _raised(result):
                    content = TOOL_ERROR_MESSAGE.format(name=result.name)
                converted.append(
                    {
                        "id": f"{base}-{position}",
                        "role": "tool",
                        "content": content,
                        "toolCallId": result.tool_call_id,
                    }
                )
        elif message.role == "assistant":
            item: dict[str, Any] = {"id": base, "role": "assistant"}
            if message.text:
                item["content"] = message.text
            if message.name:
                item["name"] = message.name
            calls = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.raw_arguments
                        if call.raw_arguments is not None
                        else json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ]
            if calls:
                item["toolCalls"] = calls
            if "content" in item or calls:
                converted.append(item)
        elif message.role == "user" and any(
            not isinstance(part, TextPart) for part in message.parts
        ):
            content = [_input_content(part) for part in message.parts]
            parts = [part for part in content if part is not None]
            converted.append({"id": base, "role": "user", "content": parts})
        else:
            converted.append({"id": base, "role": message.role, "content": message.text})
    return converted


def agui_resume(
    resume: Iterable[Mapping[str, Any]], *, allow_argument_edits: bool = False
) -> dict[str, dict[str, Any]]:
    """Turn the ``resume`` entries of an AG-UI ``RunAgentInput`` into resume arguments.

        await agent.resume(session, **agui_resume(body["resume"]))

    Interrupt ids are the ones `to_agui_events` issued:

    * ``approval:<tool call id>``: the payload is an `Approval` as a dict
      (``{"approved": false, "reason": "..."}``) or a bare boolean. A
      ``cancelled`` entry declines the call, whatever its payload says.
    * ``input:<tool call id>``: the payload is the tool's result. A
      ``cancelled`` entry gives the call an error result, `prompts.INPUT_CANCELLED`.

    With signed pauses (``Permissions(approval_secret=...)``) the client sends
    back the ``token`` it received in the interrupt's metadata:
    ``{"approved": true, "token": "..."}`` for an approval and
    ``{"token": "...", "value": ...}`` for an input. Both pass through
    unchanged, and ``agent.resume`` checks them. A ``cancelled`` entry needs
    the token too, as its payload ``{"token": "..."}``; without it
    ``agent.resume`` raises `ApprovalTokenError` and nothing runs.

    Args:
        resume: The entries, each with ``interruptId``, ``status``
            (``"resolved"`` or ``"cancelled"``) and an optional ``payload``.
        allow_argument_edits: Accept ``arguments`` in an approval, which runs
            the tool with the client's arguments instead of the model's. Off
            by default, because the client would choose what runs, and a
            pause's token signs the model's arguments, not the client's. Turn
            it on only for clients you trust to do that, and advertise your own
            ``responseSchema``: `APPROVAL_RESPONSE_SCHEMA` has no ``arguments``.

    Returns:
        ``{"approvals": {...}, "results": {...}}``, keyed by tool call id.

    Raises:
        ValueError: An entry has an unknown interrupt id or status, an
            approval payload that is not an `Approval`, or ``arguments``
            without ``allow_argument_edits``.
    """
    approvals: dict[str, Approval] = {}
    results: dict[str, Any] = {}
    for entry in resume:
        interrupt_id = str(entry.get("interruptId") or "")
        status = entry.get("status", "resolved")
        if status not in ("resolved", "cancelled"):
            raise ValueError(
                f"Resume entry {interrupt_id!r} has status {status!r}; "
                "AG-UI allows 'resolved' or 'cancelled'."
            )
        payload = entry.get("payload")
        if interrupt_id.startswith(APPROVAL_INTERRUPT_PREFIX):
            call_id = interrupt_id.removeprefix(APPROVAL_INTERRUPT_PREFIX)
            approval = _as_approval(interrupt_id, status, payload)
            if approval.arguments is not None and not allow_argument_edits:
                raise ValueError(
                    f"The payload for {interrupt_id!r} changes the call's arguments. That "
                    "lets the client choose what runs, so it is refused unless "
                    "agui_resume(..., allow_argument_edits=True)."
                )
            approvals[call_id] = approval
        elif interrupt_id.startswith(INPUT_INTERRUPT_PREFIX):
            call_id = interrupt_id.removeprefix(INPUT_INTERRUPT_PREFIX)
            if status == "resolved":
                results[call_id] = payload
            else:
                cancelled = ToolError(prompts.INPUT_CANCELLED)
                token = _token_of(payload)
                results[call_id] = (
                    cancelled if token is None else InputResult(value=cancelled, token=token)
                )
        else:
            raise ValueError(
                f"Interrupt id {interrupt_id!r} was not issued by qagents: expected "
                f"'{APPROVAL_INTERRUPT_PREFIX}<tool call id>' or "
                f"'{INPUT_INTERRUPT_PREFIX}<tool call id>'."
            )
    return {"approvals": approvals, "results": results}


# --- helpers -----------------------------------------------------------------------------


def _event_type(event: Any) -> str | None:
    kind = _field(event, "type")
    return kind if isinstance(kind, str) else None


def _field(source: Any, name: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def _dump(event: Any) -> Any:
    """A qagents event as JSON-ready data, without its ``type``."""
    if isinstance(event, BaseModel):
        return event.model_dump(mode="json", exclude={"type"})
    if isinstance(event, Mapping):
        return to_jsonable_python(
            {key: value for key, value in event.items() if key != "type"}, fallback=str
        )
    return to_jsonable_python(event, fallback=str)


def _interrupts(result: Any) -> list[dict[str, Any]]:
    interrupts: list[dict[str, Any]] = []
    for approval in _field(result, "pending_approvals") or []:
        call_id = _field(approval, "tool_call_id")
        interrupts.append(
            _without_none(
                {
                    "id": f"{APPROVAL_INTERRUPT_PREFIX}{call_id}",
                    "reason": "confirmation",
                    "message": _field(approval, "reason"),
                    "toolCallId": call_id,
                    "responseSchema": APPROVAL_RESPONSE_SCHEMA,
                    "metadata": _without_none(
                        {
                            "name": _field(approval, "name"),
                            "arguments": to_jsonable_python(
                                _field(approval, "arguments"), fallback=str
                            ),
                            "token": _field(approval, "token"),
                        }
                    ),
                }
            )
        )
    for pending in _field(result, "pending_inputs") or []:
        call_id = _field(pending, "tool_call_id")
        interrupts.append(
            _without_none(
                {
                    "id": f"{INPUT_INTERRUPT_PREFIX}{call_id}",
                    "reason": "input_required",
                    "message": _field(pending, "reason"),
                    "toolCallId": call_id,
                    "metadata": _without_none(
                        {
                            "name": _field(pending, "name"),
                            "arguments": to_jsonable_python(
                                _field(pending, "arguments"), fallback=str
                            ),
                            "metadata": to_jsonable_python(
                                _field(pending, "metadata") or None, fallback=str
                            ),
                            "token": _field(pending, "token"),
                        }
                    ),
                }
            )
        )
    return interrupts


def _as_approval(interrupt_id: str, status: str, payload: Any) -> Approval:
    if status == "cancelled":
        return Approval(approved=False, token=_token_of(payload))
    if isinstance(payload, bool):
        return Approval(approved=payload)
    if isinstance(payload, Mapping):
        try:
            return Approval.model_validate(dict(payload))
        except ValidationError as error:
            raise ValueError(
                f"The payload for {interrupt_id!r} is not an approval: {error}"
            ) from error
    raise ValueError(
        f"The payload for {interrupt_id!r} must be true, false or an object like "
        '{"approved": true}; got ' + repr(payload) + "."
    )


def _token_of(payload: Any) -> str | None:
    """The ``token`` of a resume payload, if it carries one."""
    token = payload.get("token") if isinstance(payload, Mapping) else None
    return token if isinstance(token, str) else None


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def _replays(run_started: Any) -> bool:
    """Whether a ``run_started`` event begins a resumed qagents run, whose first tool calls
    are those of the response the paused run showed. An external turn's ``resumed`` says
    that its session was continued: its tool calls are all new."""
    run_id = str(_field(run_started, "run_id") or "")
    return bool(_field(run_started, "resumed")) and not run_id.startswith(_EXTERNAL_RUN_PREFIX)


def _blocks_the_answer(kind: str | None, event: Any) -> bool:
    """Whether ``event`` says an output guardrail blocked the final answer."""
    if kind == "guardrail_tripped":
        return _field(event, "stage") == "output"
    if kind == "run_finished":
        result = _field(event, "result")
        return (
            _field(result, "status") == "blocked"
            and _field(_field(result, "guardrail"), "stage") == "output"
        )
    return False


def _raised(result: Any) -> bool:
    """Whether ``result`` is a tool result that records an exception the tool raised."""
    metadata = _field(result, "metadata")
    return (
        _field(result, "tool_call_id") is not None
        and bool(_field(result, "is_error"))
        and isinstance(metadata, Mapping)
        and bool(metadata.get("exception"))
    )


def _without_exception_text(value: Any) -> Any:
    """``value`` (JSON data) with the content of every tool result that records an
    exception replaced by `TOOL_ERROR_MESSAGE`."""
    if isinstance(value, list):
        return [_without_exception_text(item) for item in value]
    if not isinstance(value, dict):
        return value
    cleaned = {key: _without_exception_text(item) for key, item in value.items()}
    if _raised(cleaned):
        cleaned["content"] = TOOL_ERROR_MESSAGE.format(name=cleaned.get("name") or "tool")
    return cleaned


def _before_last_answer(messages: list[Any]) -> list[Any]:
    """``messages`` up to, not including, the last assistant message."""
    for index in range(len(messages) - 1, -1, -1):
        if _field(messages[index], "role") == "assistant":
            return messages[:index]
    return messages


def _without_reasoning(value: Any, *, keep_text: bool) -> Any:
    """``value`` with the reasoning parts of its messages left out, or with only
    their provider replay data (``opaque``) left out when ``keep_text``."""
    if isinstance(value, list):
        return [_without_reasoning(item, keep_text=keep_text) for item in value]
    if not isinstance(value, dict):
        return value
    cleaned = {key: _without_reasoning(item, keep_text=keep_text) for key, item in value.items()}
    if isinstance(cleaned.get("parts"), list):
        parts = []
        for part in cleaned["parts"]:
            if isinstance(part, dict) and part.get("type") == "reasoning":
                if not keep_text:
                    continue
                part = {key: item for key, item in part.items() if key != "opaque"}
            parts.append(part)
        cleaned["parts"] = parts
    return cleaned


def _input_content(part: Any) -> dict[str, Any] | None:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, ImagePart):
        content: dict[str, Any] = {"type": "binary", "mimeType": part.media_type or "image/png"}
        if part.url:
            content["url"] = part.url
        else:
            content["data"] = part.data
        return content
    return None


def _without_none(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


__all__ = [
    "AGUIProjector",
    "AGUI_MEDIA_TYPE",
    "APPROVAL_INTERRUPT_PREFIX",
    "APPROVAL_RESPONSE_SCHEMA",
    "CUSTOM_EVENTS",
    "INPUT_INTERRUPT_PREFIX",
    "RUN_ERROR_MESSAGE",
    "RUN_ERROR_MESSAGES",
    "STEP_NAME",
    "STREAM_ENDED_MESSAGE",
    "TOOL_ERROR_MESSAGE",
    "agui_resume",
    "agui_sse",
    "encode_sse",
    "to_agui_events",
    "to_agui_messages",
]
