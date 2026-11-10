"""What a run returns."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, PrivateAttr

from .messages import Message, Usage
from .session import PendingApproval

if TYPE_CHECKING:
    from .session import Session

RunStatus = Literal["completed", "paused", "limit_reached", "cancelled"]


class RunResult(BaseModel):
    """The outcome of ``agent.run``.

    Check ``status`` before using ``output``:

    * ``completed``: the agent answered. ``output`` is the text, or the
      validated ``output_type`` value.
    * ``paused``: tool calls in ``pending_approvals`` wait for a decision;
      continue with ``agent.resume``.
    * ``limit_reached``: a limit stopped the run; ``stop_reason`` names it
      (``max_turns``, ``max_total_tokens``, ``max_cost_usd``, ``timeout``).
    * ``cancelled``: ``RunStream.cancel`` was called.

    Provider failures and broken hooks are raised as exceptions instead.
    """

    run_id: str
    session_id: str
    agent: str
    """The agent that was active when the run stopped (differs from the one
    you called after a handoff)."""
    status: RunStatus
    stop_reason: str | None = None
    output: Any = None
    text: str = ""
    """Text of the last assistant message."""
    messages: list[Message] = Field(default_factory=list)
    """The whole conversation, including earlier runs of the session."""
    new_messages: list[Message] = Field(default_factory=list)
    """Only what this run added."""
    usage: Usage = Field(default_factory=Usage)
    turns: int = 0
    pending_approvals: list[PendingApproval] = Field(default_factory=list)

    _session: Any = PrivateAttr(default=None)

    @property
    def session(self) -> Session:
        """The session the run used. Pass it to ``agent.resume`` or to the next ``agent.run``."""
        return self._session

    @property
    def completed(self) -> bool:
        return self.status == "completed"
