"""`ScriptedExternalAgent` and its steps: an in-memory external agent for offline tests.

It plays scripted turns the way `qagents.models.testing.ScriptedModel` plays model replies:
each turn is a list of steps (text, `ToolStep`, `PermissionStep`, `WaitStep`, ...), reported
through the same `ExternalTurnContext` a real adapter uses, so the whole base class (the
permission bridge, locks, budgets, cancellation, ``as_tool``) runs as it does for Devvi or
an ACP agent. It is a small adapter, and an example of one.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..messages import Usage
from .agent import ExternalAgent
from .contract import ExternalCapabilities, ExternalOutcome, ExternalStatus
from .permissions import PermissionAnswer
from .turn import ExternalTurnContext


@dataclass
class ToolStep:
    """A scripted tool call: started, optionally asked about, then finished.

    With ``ask=True`` the agent asks permission first (after the start, as
    Devvi does) and runs the tool only when allowed; a refusal finishes the
    call with an error result.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str = "OK"
    is_error: bool = False
    ask: bool = False
    id: str | None = None


@dataclass
class PermissionStep:
    """A scripted permission request without a tool call (the arguments are sent with it)."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None


@dataclass
class ReasoningStep:
    """Scripted reasoning text."""

    text: str


@dataclass
class EventStep:
    """A scripted `ExternalAgentEvent`."""

    kind: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class WaitStep:
    """Wait ``seconds``, or until the turn is cancelled (``seconds=None``).

    ``ignore_cancel=True`` keeps waiting through a cancel, like an agent that
    does not answer: qagents abandons the turn after ``cancel_grace_s``.
    """

    seconds: float | None = None
    ignore_cancel: bool = False


@dataclass
class EndStep:
    """End the turn here, with this status."""

    status: ExternalStatus = "completed"
    stop_reason: str | None = "end_turn"
    text: str | None = None


@dataclass
class ScriptedCall:
    """What a `ScriptedExternalAgent` received for one turn."""

    task: str
    session: str | None
    session_id: str
    cwd: str | None
    resumed: bool
    checkpoints: dict[str, Any]
    answers: list[PermissionAnswer] = field(default_factory=list)
    cancelled: str | None = None


class ScriptedExternalAgent(ExternalAgent):
    """An in-memory external agent that plays scripted turns, for offline tests.

        coder = ScriptedExternalAgent([
            ["Reading the code.", ToolStep("read", {"path": "a.py"}, result="..."),
             ToolStep("bash", {"command": "pytest"}, ask=True, result="1 passed"),
             Usage(requests=1, input_tokens=120, output_tokens=40),
             "All tests pass."],
        ], name="coder")
        lead = Agent(ScriptedModel([call("coder", task="Fix it"), "Done."]),
                     tools=[coder.as_tool()], permissions=Permissions(allow=["coder.bash"]))

    Each turn is a list of steps: a string (text), `ReasoningStep`,
    `ToolStep`, `PermissionStep`, `EventStep`, `WaitStep`, `EndStep`, a `Usage`
    (reported usage), an exception instance (raised: an
    `ExternalAgentConnectionError` simulates a crash), or a function of the
    `ExternalTurnContext` (sync or async). A turn may also be a single step.

    Each string is a committed message (Devvi sends whole blocks, not
    tokens): two strings in a row are separated by a blank line in the turn's
    text, as two blocks of a real agent are. ``grant_ignored_arguments`` sets
    `ExternalAgent.grant_ignored_arguments` (arguments a grant across a pause
    need not match).

    ``calls`` records what each turn received, ``answers`` every permission
    answer. A new session is named ``"<name>-<n>"`` (or
    ``turn.suggested_session_id`` with ``client_session_ids``); a given
    ``session`` is continued (``resumed=True``).
    """

    backend = "scripted"

    def __init__(
        self,
        turns: Iterable[Any],
        *,
        name: str = "scripted",
        description: str = "",
        capabilities: ExternalCapabilities | None = None,
        resumable_turns: bool = False,
        grant_ignored_arguments: Iterable[str] = (),
        **options: Any,
    ) -> None:
        super().__init__(name=name, description=description, **options)
        self.grant_ignored_arguments = frozenset(grant_ignored_arguments)
        self._capabilities = capabilities or ExternalCapabilities(
            token_usage=True, cancel_turn=True, resume_sessions=True, permission_requests=True
        )
        self.resumable_turns = resumable_turns
        self._script: list[list[Any]] = [
            list(turn) if isinstance(turn, list | tuple) else [turn] for turn in turns
        ]
        self.calls: list[ScriptedCall] = []
        self.answers: list[PermissionAnswer] = []
        self.connects = 0
        self.disconnects = 0
        self.cancel_requests: list[str] = []
        self._sessions = 0

    @property
    def prompts(self) -> list[str]:
        """The prompt text of every turn, in order."""
        return [call.task for call in self.calls]

    async def capabilities(self) -> ExternalCapabilities:
        return self._capabilities

    async def connect(self) -> None:
        self.connects += 1

    async def disconnect(self) -> None:
        self.disconnects += 1

    async def cancel_turn(self, turn: ExternalTurnContext) -> None:
        self.cancel_requests.append(turn.cancel_reason or "cancelled")

    async def run_turn(self, turn: ExternalTurnContext) -> ExternalOutcome:
        if not self._script:
            raise AssertionError(
                f"ScriptedExternalAgent {self.name!r} was asked for turn #{len(self.calls) + 1} "
                "but the script has no more turns."
            )
        steps = self._script.pop(0)
        if turn.session is not None:
            session_id = turn.session
        elif self._capabilities.client_session_ids:
            session_id = turn.suggested_session_id
        else:
            self._sessions += 1
            session_id = f"{self.name}-{self._sessions}"
        record = ScriptedCall(
            task=turn.prompt_text,
            session=turn.session,
            session_id=session_id,
            cwd=turn.cwd,
            resumed=turn.session is not None,
            checkpoints=dict(turn.checkpoints),
        )
        self.calls.append(record)
        await turn.begin(session_id, resumed=turn.session is not None)
        calls = 0
        for number, step in enumerate(steps):
            if turn.cancel_requested:
                record.cancelled = turn.cancel_reason
                return ExternalOutcome(status="cancelled", stop_reason="cancelled")
            if isinstance(step, str):
                turn.text_delta(step, message_id=f"{session_id}-message-{number}")
            elif isinstance(step, ReasoningStep):
                turn.reasoning_delta(step.text)
            elif isinstance(step, Usage):
                turn.add_usage(step)
            elif isinstance(step, EventStep):
                turn.event(step.kind, step.data)
            elif isinstance(step, ToolStep):
                calls += 1
                await self._tool(turn, record, step, step.id or f"{session_id}-call-{calls}")
            elif isinstance(step, PermissionStep):
                answer = await turn.ask_permission(step.name, step.arguments, reason=step.reason)
                record.answers.append(answer)
                self.answers.append(answer)
            elif isinstance(step, WaitStep):
                await self._wait(turn, step)
            elif isinstance(step, EndStep):
                return ExternalOutcome(
                    status=step.status, stop_reason=step.stop_reason, text=step.text
                )
            elif isinstance(step, BaseException):
                raise step
            elif callable(step):
                value = step(turn)
                if inspect.isawaitable(value):
                    await value
            else:
                raise TypeError(f"Unknown scripted step: {step!r}")
            await asyncio.sleep(0)  # let the consumer see each step
        if turn.cancel_requested:
            record.cancelled = turn.cancel_reason
            return ExternalOutcome(status="cancelled", stop_reason="cancelled")
        return ExternalOutcome(status="completed", stop_reason="end_turn")

    async def _tool(
        self, turn: ExternalTurnContext, record: ScriptedCall, step: ToolStep, call_id: str
    ) -> None:
        turn.tool_started(call_id, step.name, step.arguments, counts_as_step=True)
        if step.ask:
            answer = await turn.ask_permission(step.name, tool_call_id=call_id)
            record.answers.append(answer)
            self.answers.append(answer)
            if not answer.approved:
                outcome = "cancelled" if answer.outcome == "cancel" else "rejected"
                turn.tool_finished(call_id, f"The user {outcome} this tool call.", is_error=True)
                return
        turn.tool_finished(call_id, step.result, is_error=step.is_error)

    @staticmethod
    async def _wait(turn: ExternalTurnContext, step: WaitStep) -> None:
        if step.ignore_cancel:
            await asyncio.sleep(step.seconds if step.seconds is not None else 3600)
            return
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(turn.wait_cancelled(), step.seconds)
