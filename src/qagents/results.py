"""What a run returns."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, PrivateAttr

from .messages import Message, Usage
from .session import PendingApproval, PendingInput

if TYPE_CHECKING:
    from .session import Session

RunStatus = Literal["completed", "paused", "limit_reached", "cancelled", "blocked"]
GuardrailStage = Literal["input", "output"]


class TrippedGuardrail(BaseModel):
    """The guardrail that blocked a run (``RunResult.guardrail``)."""

    name: str
    """The guardrail's name: the ``name`` given to `GuardrailTripped`, or the
    guardrail function's ``__name__``."""
    reason: str
    """The reason given to `GuardrailTripped`."""
    stage: GuardrailStage
    """``"input"``: the prompt was blocked and never joined the conversation.
    ``"output"``: the final answer was blocked; it stays in the journal, and a
    later read of the session (`Session.state`, a later run's ``messages``)
    marks that assistant message with ``metadata["blocked"] = True``."""


class RunResult(BaseModel):
    """The outcome of ``agent.run``.

    Check ``status`` before using ``output``:

    * ``completed``: the agent answered. ``output`` is the text, or the
      validated ``output_type`` value.
    * ``paused``: tool calls in ``pending_approvals`` wait for a decision, and
      calls in ``pending_inputs`` wait for a result from outside the run;
      continue with ``agent.resume(session, approvals=..., results=...)``.
    * ``limit_reached``: a limit stopped the run; ``stop_reason`` names it
      (``max_turns``, ``max_total_tokens``, ``max_cost_usd``, ``timeout``,
      ``max_tool_calls``).
    * ``cancelled``: ``RunStream.cancel`` was called.
    * ``blocked``: a guardrail tripped. ``guardrail`` says which one and why,
      ``stop_reason`` is its name, and ``output`` and ``text`` are empty so a
      blocked answer is not shown by mistake (it stays in ``messages``).

    A ``completed`` run with ``stop_reason == "content_filter"`` was ended by
    the provider's refusal; ``output`` is the refusal text for a plain-text
    agent and ``None`` when an ``output_type`` is set.

    Provider failures and broken hooks are raised as exceptions instead, and
    so is a final answer that is still rejected when ``Limits.output_retries``
    are used up (`OutputValidationError`): there is no ``limit_reached``
    result for it. The run stays open in the journal as failed.

    Each assistant message a model wrote carries, in its ``metadata``, the
    ``model`` and ``provider`` that answered (with `FallbackModel`, the model
    that actually answered, as far as the provider reports it) and the
    ``finish_reason``.
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
    """Text of the last assistant message; empty when the run was blocked.

    It is what the model wrote, not the accepted answer: use ``output``. They
    differ when an output validator replaced the answer, and in
    ``output_mode="tool"``, where the answer is the final-answer call's
    arguments and ``text`` is usually empty."""
    messages: list[Message] = Field(default_factory=list)
    """The whole conversation, including earlier runs of the session."""
    new_messages: list[Message] = Field(default_factory=list)
    """Only what this run added."""
    usage: Usage = Field(default_factory=Usage)
    """Usage of the whole run: a resumed run includes what it used before it
    stopped, as the journal records it (model responses and the usage tools
    added with ``ctx.add_usage``)."""
    turns: int = 0
    """Model requests of the whole run that produced a response, across resumes."""
    pending_approvals: list[PendingApproval] = Field(default_factory=list)
    """Tool calls that wait for a human decision (``status == "paused"``)."""
    pending_inputs: list[PendingInput] = Field(default_factory=list)
    """Tool calls that wait for a result: calls of deferred tools and calls
    whose tool raised `DeferToolCall` (``status == "paused"``)."""

    guardrail: TrippedGuardrail | None = None
    """The guardrail that blocked the run (``status == "blocked"``)."""

    _session: Any = PrivateAttr(default=None)

    @property
    def session(self) -> Session:
        """The session the run used. Pass it to ``agent.resume`` or to the next ``agent.run``."""
        return self._session

    @property
    def completed(self) -> bool:
        return self.status == "completed"
