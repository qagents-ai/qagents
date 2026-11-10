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
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from .messages import ToolCallPart, ToolResultPart
from .models.base import ModelResponse, ReasoningDelta, TextDelta, ToolCallDelta
from .results import RunResult
from .session import PendingApproval


class RunStarted(BaseModel):
    type: Literal["run_started"] = "run_started"
    run_id: str
    session_id: str
    agent: str
    resumed: bool = False


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
    """A tool call has a result: it ran, failed, was refused or was cancelled."""

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


class AgentSwitched(BaseModel):
    """A handoff made another agent the active one."""

    type: Literal["agent_switched"] = "agent_switched"
    from_agent: str
    to_agent: str


class SubagentEvent(BaseModel):
    """An event from an agent running as a tool.

    It is wrapped so that it cannot be mistaken for one of the parent's events.
    """

    type: Literal["subagent_event"] = "subagent_event"
    agent: str
    tool_call_id: str | None = None
    event: Any


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
    | AgentSwitched
    | SubagentEvent
    | RunFinished
)

__all__ = [
    "AgentSwitched",
    "ApprovalRequired",
    "Event",
    "ModelResponded",
    "ReasoningDelta",
    "RunFinished",
    "RunStarted",
    "SubagentEvent",
    "TextDelta",
    "ToolCallDelta",
    "ToolCallFinished",
    "ToolCallStarted",
    "TurnStarted",
]
