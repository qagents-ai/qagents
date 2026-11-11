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
| `state_updated` | `STATE_SNAPSHOT` and a `CUSTOM` event |
| `approval_required`, `input_required` | `CUSTOM`, named after the event |
| `agent_switched`, `guardrail_tripped` | `CUSTOM`, named after the event |
| `subagent_event` | `CUSTOM` named `subagent_event`, or the `SUBAGENT_*` events |
| `run_finished` | `RUN_FINISHED`, with an interrupt outcome when the run paused |
| an exception | `RUN_ERROR` |

Events of other types are skipped, so a qagents release that adds event types
never breaks a client. The ``value`` of a `CUSTOM` event is the qagents event
as JSON, without its ``type``.

Every start event gets its end event, and messages and tool calls are never
interleaved: a text message ends before a tool call starts, which is what
strict AG-UI clients check. Tool calls are announced when the model has
finished writing them if the provider does not stream them. Message ids are
made from the AG-UI run id and a counter.

A run that pauses for approvals or for deferred tools ends with a
``RUN_FINISHED`` event whose ``outcome`` lists the interrupts. When the client
answers them, `agui_resume` turns its ``resume`` entries back into the
``approvals`` and ``results`` of ``agent.resume``.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncGenerator, AsyncIterable, Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ValidationError
from pydantic_core import to_jsonable_python

from .. import prompts
from ..errors import ConfigurationError, QAgentsError, ToolError
from ..messages import ImagePart, Message, ReasoningPart, TextPart
from ..permissions import Approval

# The text sent to the model (INPUT_CANCELLED) is in qagents.prompts.

# --- text and names sent to the client ----------------------------------------------

AGUI_MEDIA_TYPE = "text/event-stream"
"""Media type of the response that carries `agui_sse` output."""

STEP_NAME = "{agent}: turn {turn}"
"""``stepName`` of the step events that frame one model request and its tool calls."""

RUN_ERROR_MESSAGE = "The run failed. The server log has the details."
"""``RUN_ERROR`` message for exceptions that are not qagents errors, unless
``expose_errors=True``. Their text can contain internal details."""

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
"""``responseSchema`` of an approval interrupt: the client answers with an `Approval`."""

SubagentMode = Literal["custom", "native", "ignore"]
"""How events of agents running as tools are projected; see `AGUIProjector`."""

logger = logging.getLogger("qagents.agui")

_LIFECYCLE = frozenset({"run_started", "run_finished", "subagent_event"})


@dataclass
class _StreamedCall:
    """A tool call the provider is streaming, keyed by its index in the response."""

    id: str | None = None
    name: str = ""
    buffer: str = ""
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
        reasoning: Forward the reasoning text that models expose. Set it to
            ``False`` to keep reasoning away from the client.
        state_snapshots: Send ``STATE_SNAPSHOT`` with the full session state
            after every ``state_updated``.
        messages_snapshot: Send ``MESSAGES_SNAPSHOT`` with the whole
            conversation before ``RUN_FINISHED``. Off by default: the
            conversation can be large, and the client already has it.
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
        expose_errors: Put the text of any exception in ``RUN_ERROR``. By
            default only qagents errors, which are written for developers and
            carry no internals, are shown; others get `RUN_ERROR_MESSAGE`.

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
            "tool_call_finished": self._on_tool_call_finished,
            "state_updated": self._on_state_updated,
            "agent_switched": self._on_agent_switched,
            "subagent_event": self._on_subagent_event,
            "run_finished": self._on_run_finished,
        }

    @property
    def finished(self) -> bool:
        """True once ``RUN_FINISHED`` or ``RUN_ERROR`` was produced. Later events are ignored."""
        return self._finished

    def feed(self, event: Any) -> list[dict[str, Any]]:
        """Project one qagents event.

        Args:
            event: A qagents event (a pydantic model or a dict with a ``type``).
                Events of unknown types produce nothing, unless their type is
                in ``custom_events``.

        Returns:
            The AG-UI events it produces, in order; often none or one.
        """
        if self._finished:
            return []
        kind = _event_type(event)
        out: list[dict[str, Any]] = []
        handler = self._handlers.get(kind or "")
        if handler is not None:
            handler(event, out)
        if kind in self.custom_events and kind not in _LIFECYCLE and not self._finished:
            self._close_segments(out)
            out.append(self._event("CUSTOM", name=kind, value=_dump(event)))
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
        if isinstance(error, QAgentsError) or self.expose_errors:
            message = str(error) or type(error).__name__
        else:
            message = RUN_ERROR_MESSAGE
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
        summary = {
            "status": _field(result, "status"),
            "stop_reason": _field(result, "stop_reason"),
            "output": to_jsonable_python(_field(result, "output"), fallback=str),
        }
        self._finished = True
        if self._subagent_run_id is not None:
            out.append(self._event("SUBAGENT_FINISHED", result=summary))
            return
        messages = _field(result, "messages")
        if self.messages_snapshot and messages:
            snapshot = to_agui_messages(messages, id_prefix=f"{self.thread_id}-msg")
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

    def _on_turn_started(self, event: Any, out: list[dict[str, Any]]) -> None:
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
            return  # another call started meanwhile; the final arguments are in the history
        if call.started:
            if fragment:
                out.append(self._event("TOOL_CALL_ARGS", toolCallId=call.id, delta=fragment))
            return
        call.buffer += fragment
        # Start once the id is known and the arguments begin, by which point
        # providers have sent the whole name. Without an id, wait for the response.
        if call.id and call.name and call.buffer:
            self._close_call(out)
            self._start_call(out, call.id, call.name)
            out.append(self._event("TOOL_CALL_ARGS", toolCallId=call.id, delta=call.buffer))
            call.buffer = ""
            call.started = True
            self._open_call = call

    def _on_model_responded(self, event: Any, out: list[dict[str, Any]]) -> None:
        self._close_segments(out)
        message = _field(_field(event, "response"), "message")
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

    def _on_tool_call_finished(self, event: Any, out: list[dict[str, Any]]) -> None:
        self._close_segments(out)
        result = _field(event, "result")
        call_id = _field(_field(event, "call"), "id") or _field(result, "tool_call_id")
        keys = [key for key in self._children if key[0] == str(call_id)]
        for key in keys:
            child = self._children.pop(key)
            if _field(result, "is_error"):
                child._close_all(out, error=_field(result, "content"))
                out.extend(child._end_subagent(error=_field(result, "content") or ""))
            else:
                # One child answers with the call's result; parallel children share one.
                child._close_all(out)
                output = _field(result, "content") if len(keys) == 1 else None
                out.extend(child._end_subagent(output=output))
        self._unresulted.pop(call_id, None)
        out.append(
            self._event(
                "TOOL_CALL_RESULT",
                messageId=self._new_id("result"),
                toolCallId=call_id,
                content=_field(result, "content") or "",
                role="tool",
            )
        )

    def _on_state_updated(self, event: Any, out: list[dict[str, Any]]) -> None:
        values = _field(event, "values")
        if self.state_snapshots and isinstance(values, Mapping):
            self._close_segments(out)
            snapshot = to_jsonable_python(dict(values), fallback=str)
            out.append(self._event("STATE_SNAPSHOT", snapshot=snapshot))

    def _on_agent_switched(self, event: Any, out: list[dict[str, Any]]) -> None:
        # The loop records the result of the accepted handoff call without an
        # event; every other call of that turn has had its tool_call_finished.
        target = _field(event, "to_agent") or ""
        for call_id, name in list(self._unresulted.items()):
            if name.startswith("transfer_to_"):
                self._close_segments(out)
                out.append(
                    self._event(
                        "TOOL_CALL_RESULT",
                        messageId=self._new_id("result"),
                        toolCallId=call_id,
                        content=prompts.HANDOFF_RESULT.format(name=target),
                        role="tool",
                    )
                )
                del self._unresulted[call_id]
                break

    def _on_subagent_event(self, event: Any, out: list[dict[str, Any]]) -> None:
        if self.subagents == "ignore":
            return
        if self.subagents == "custom":
            self._close_segments(out)
            out.append(self._event("CUSTOM", name="subagent_event", value=_dump(event)))
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
            )
            child._subagent_run_id = child_id
            child._started = True
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
        """End every open message, tool call, step and subagent, children first."""
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
        stream: ``agent.stream(...)`` or ``agent.resume_stream(...)``, or any
            async iterable of qagents events.
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
    messages: Iterable[Message], *, id_prefix: str = "msg"
) -> list[dict[str, Any]]:
    """Convert a conversation into AG-UI messages, as sent in ``MESSAGES_SNAPSHOT``.

    qagents messages have no ids, so ids are ``<id_prefix>-<position>``; tool
    results become one ``tool`` message each, ``<id_prefix>-<position>-<n>``.
    Reasoning is left out, and an assistant message with neither text nor tool
    calls is skipped.

    Args:
        messages: The conversation, for example ``result.messages``.
        id_prefix: Start of every message id. Use something unique to the thread.

    Returns:
        AG-UI message dicts with camelCase field names.
    """
    converted: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        base = f"{id_prefix}-{index}"
        if message.role == "tool":
            for position, result in enumerate(message.tool_results):
                converted.append(
                    {
                        "id": f"{base}-{position}",
                        "role": "tool",
                        "content": result.content,
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


def agui_resume(resume: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Turn the ``resume`` entries of an AG-UI ``RunAgentInput`` into resume arguments.

        await agent.resume(session, **agui_resume(body["resume"]))

    Interrupt ids are the ones `to_agui_events` issued:

    * ``approval:<tool call id>``: the payload is an `Approval` as a dict
      (``{"approved": false, "reason": "..."}``) or a bare boolean. A
      ``cancelled`` entry declines the call.
    * ``input:<tool call id>``: the payload is the tool's result. A
      ``cancelled`` entry gives the call an error result, `prompts.INPUT_CANCELLED`.

    With signed pauses (``Permissions(approval_secret=...)``) the client sends
    back the ``token`` it received in the interrupt's metadata:
    ``{"approved": true, "token": "..."}`` for an approval and
    ``{"token": "...", "value": ...}`` for an input. Both pass through
    unchanged, and ``agent.resume`` checks them.

    Args:
        resume: The entries, each with ``interruptId``, ``status``
            (``"resolved"`` or ``"cancelled"``) and an optional ``payload``.

    Returns:
        ``{"approvals": {...}, "results": {...}}``, keyed by tool call id.

    Raises:
        ValueError: An entry has an unknown interrupt id or status, or an
            approval payload that is not an `Approval`.
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
            approvals[call_id] = _as_approval(interrupt_id, status, payload)
        elif interrupt_id.startswith(INPUT_INTERRUPT_PREFIX):
            call_id = interrupt_id.removeprefix(INPUT_INTERRUPT_PREFIX)
            results[call_id] = (
                ToolError(prompts.INPUT_CANCELLED) if status == "cancelled" else payload
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
        return Approval(approved=False)
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
    "STEP_NAME",
    "STREAM_ENDED_MESSAGE",
    "agui_resume",
    "agui_sse",
    "encode_sse",
    "to_agui_events",
    "to_agui_messages",
]
