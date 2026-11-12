"""Events a run emits, in the order things happen.

``agent.stream(...)`` yields them; ``Hooks.on_event`` receives them. Every
event is a pydantic model with a ``type`` string, so it serialises straight to
JSON for a websocket or an SSE response.

A typical run:

    run_started
    turn_started
      text_delta ...            (and reasoning_delta / tool_call_delta)
      model_responded
      tool_call_started
      tool_call_finished
    turn_started
      text_delta ...
      model_responded
    run_finished

A run that pauses ends with ``approval_required`` and ``input_required``
events before ``run_finished``. ``state_updated`` follows every change of
``ctx.state``. Turns of an external agent (`qagents.external`) also emit
``external_event`` (`ExternalAgentEvent`). A run that a guardrail blocks emits
``guardrail_tripped`` before ``run_finished``: right after ``run_started`` for
an input guardrail, after the last ``model_responded`` for an output guardrail.

A final answer the run does not accept (it does not match ``output_type``, an
output validator rejects it, or in ``output_mode="tool"`` it was not given
with the final-answer tool) is followed by ``output_rejected``, and the model
is asked again.

Text is streamed as the model writes it, before the answer is checked: an
output guardrail that blocks the answer cannot recall ``text_delta`` events
already sent. A front end that must never show a blocked answer holds the
text back until ``run_finished`` and checks ``result.status``.

A run that raised emits no ``run_finished``: the error is raised from the
stream. An ``on_event`` hook that raises fails the run the same way (the
journal records ``run_failed``), except on ``run_finished``, when the run has
already ended.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .messages import ToolCallPart, ToolResultPart
from .models.base import ModelResponse, ReasoningDelta, TextDelta, ToolCallDelta
from .results import GuardrailStage, RunResult
from .session import PendingApproval, PendingInput


class RunStarted(BaseModel):
    type: Literal["run_started"] = "run_started"
    run_id: str
    session_id: str
    agent: str
    resumed: bool = False
    parent_run_id: str | None = None
    """The calling run, for a child run started with ``parent=ctx``."""
    depth: int = 0
    """0 for a top-level run, 1 for a run it started, and so on."""


class TurnStarted(BaseModel):
    """A model request is about to be sent."""

    type: Literal["turn_started"] = "turn_started"
    turn: int
    agent: str


class ModelResponded(BaseModel):
    """The model finished one response. ``response.message`` is the full assistant message."""

    type: Literal["model_responded"] = "model_responded"
    response: ModelResponse
    agent: str


class ToolCallStarted(BaseModel):
    type: Literal["tool_call_started"] = "tool_call_started"
    call: ToolCallPart
    agent: str


class ToolCallFinished(BaseModel):
    """A tool call has a result: it ran, failed, was refused or was cancelled.

    Every call of the run's turns gets one, including the calls closed when
    the run is cancelled, times out or is refused by the provider. Calls that
    a previous run left open, and that a new run closes when it starts, get
    none: they belong to the previous run.
    """

    type: Literal["tool_call_finished"] = "tool_call_finished"
    call: ToolCallPart
    result: ToolResultPart
    duration_ms: float = 0.0
    agent: str


class ApprovalRequired(BaseModel):
    """A tool call waits for a human. The run ends as ``paused`` after these."""

    type: Literal["approval_required"] = "approval_required"
    approval: PendingApproval
    agent: str


class InputRequired(BaseModel):
    """A tool call waits for a result from outside the run.

    Emitted for calls of tools declared ``deferred=True`` and for calls whose
    tool raised `DeferToolCall`. The run ends as ``paused`` after these;
    supply the result with ``agent.resume(session, results=...)``.
    """

    type: Literal["input_required"] = "input_required"
    input: PendingInput
    agent: str


class StateUpdated(BaseModel):
    """``ctx.state`` changed and the new values were journaled."""

    type: Literal["state_updated"] = "state_updated"
    values: dict[str, Any]
    """The full state after the change (a copy)."""
    agent: str


class OutputRejected(BaseModel):
    """A final answer was not accepted; the model is asked to correct it.

    In every output mode: the answer did not match ``output_type``, an output
    validator raised `ModelRetry`, or (``output_mode="tool"``) the answer was
    plain text or the final-answer call was invalid (that call also gets an
    error result, with its ``tool_call_finished``). When
    ``Limits.output_retries`` are used up the run raises
    `OutputValidationError` instead.
    """

    type: Literal["output_rejected"] = "output_rejected"
    reason: str
    """What is wrong, as the model is told."""
    attempt: int
    """1 for the first rejected answer of the run, 2 for the second, and so on."""
    agent: str


class GuardrailTriggered(BaseModel):
    """A guardrail blocked the prompt or the final answer. The run ends as ``blocked``."""

    type: Literal["guardrail_tripped"] = "guardrail_tripped"
    name: str
    """Name of the guardrail that tripped."""
    reason: str
    """The reason given to `GuardrailTripped`."""
    stage: GuardrailStage
    """``"input"`` for the prompt, ``"output"`` for the final answer."""
    agent: str


class AgentSwitched(BaseModel):
    """A handoff made another agent the active one."""

    type: Literal["agent_switched"] = "agent_switched"
    from_agent: str
    to_agent: str
    tool_call_id: str | None = None
    """The handoff tool call that made the switch."""


class SubagentEvent(BaseModel):
    """An event from an agent running as a tool.

    It is wrapped so that it cannot be mistaken for one of the parent's events.
    """

    type: Literal["subagent_event"] = "subagent_event"
    agent: str
    tool_call_id: str | None = None
    event: Any


class ExternalAgentEvent(BaseModel):
    """An event of an external agent with no qagents counterpart.

    ``kind`` says what it is: ``permission_requested`` and ``permission_decided``
    (every request and its answer), ``tool_arguments`` (a tool call already
    started got new input: ``{tool_call_id, name, arguments}``), ``context``
    (ACP context pressure: ``{used, size}``, not token usage), ``config``,
    ``plan`` (ACP), ``image`` and ``audio`` (``{mimeType}``; the bytes are not
    forwarded), ``resource_link`` and ``resource`` (ACP; never the content),
    and, from Devvi's SDK transport, ``approval_asked``, ``approval_decided``
    (Devvi's own audit records), ``approval_refused`` (a request about a call
    its session never started), ``retry``, ``compaction``, ``web_search``,
    ``job``, ``attempt``, ``subagent_started``, ``subagent_finished`` and
    ``subagent_event``; ``unknown`` or ``raw`` (with raw events switched on),
    and ``turn_finished``, the last event of every turn, with the result's
    status, session, backend and footer data (``changed_files``: the first 100
    paths, ``changed_files_count``: all of them; `ExternalResult.changed_files`
    has every path).
    """

    type: Literal["external_event"] = "external_event"
    agent: str
    kind: str
    session_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class RunFinished(BaseModel):
    """Always the last event of a run that did not raise."""

    type: Literal["run_finished"] = "run_finished"
    result: RunResult


Event = (
    RunStarted
    | TurnStarted
    | TextDelta
    | ReasoningDelta
    | ToolCallDelta
    | ModelResponded
    | ToolCallStarted
    | ToolCallFinished
    | ApprovalRequired
    | InputRequired
    | StateUpdated
    | OutputRejected
    | GuardrailTriggered
    | AgentSwitched
    | SubagentEvent
    | ExternalAgentEvent
    | RunFinished
)

__all__ = [
    "AgentSwitched",
    "ApprovalRequired",
    "Event",
    "ExternalAgentEvent",
    "GuardrailTriggered",
    "InputRequired",
    "ModelResponded",
    "OutputRejected",
    "ReasoningDelta",
    "RunFinished",
    "RunStarted",
    "StateUpdated",
    "SubagentEvent",
    "TextDelta",
    "ToolCallDelta",
    "ToolCallFinished",
    "ToolCallStarted",
    "TurnStarted",
]
