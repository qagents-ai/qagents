"""One turn: what an adapter reports through, and what drives it and streams its events.

`ExternalTurnContext` is what an adapter's ``run_turn`` receives: the task, the session and
the working directory, and the methods it reports the turn through (text, tool calls,
usage, permission requests, checkpoints). It turns each report into a qagents event and
keeps what the `ExternalResult` is built from.

`ExternalTurn` is what `ExternalAgent.stream` returns: it runs the adapter's ``run_turn``
in a task of its own, holds the workspace and session locks, snapshots git around the turn,
and hands the events to whoever iterates it. It also owns cancellation: ``cancel_turn``
first, then abandoning the turn after ``cancel_grace_s``.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

from .. import prompts
from ..errors import (
    ExternalAgentConfigurationError,
    ExternalAgentConnectionError,
    ExternalAgentUnavailableError,
    ToolError,
)
from ..events import (
    ExternalAgentEvent,
    ModelResponded,
    RunStarted,
    ToolCallFinished,
    ToolCallStarted,
    TurnStarted,
)
from ..messages import Message, Part, TextPart, ToolCallPart, ToolResultPart, Usage, _unpriced
from ..models.base import FinishReason, ModelResponse, ReasoningDelta, TextDelta
from ..tools import RunContext
from .changes import _changed_since, _git_snapshot, _relative_to
from .contract import (
    LIMIT_REASONS,
    WRITE_TOOLS,
    ExternalOutcome,
    ExternalResult,
    ExternalStatus,
    ExternalToolCall,
    _result_summary,
)
from .permissions import ExternalPermission, PermissionAnswer, PermissionBridge
from .tool import _SESSION_CHECKPOINT, _depth_refusal, _remember

if TYPE_CHECKING:
    from .agent import ExternalAgent

logger = logging.getLogger("qagents.external")


class ExternalTurnContext:
    """What an adapter's `ExternalAgent.run_turn` receives: the task, and how to report.

    Report the turn as it happens, in this order:

    1. ``await turn.begin(session_id, resumed=...)`` once the session is known
       (it emits ``run_started`` and, for a resumable call, journals the
       session id so a repeated call finds it).
    2. While the agent works: `text_delta` / `reasoning_delta` for committed
       text, `tool_started` / `tool_finished` for its tool calls,
       `step_started` per model request (when the protocol says so),
       `add_usage` or `model_responded` for token usage, `event` for
       anything else, and ``await ask_permission(...)`` when the agent asks.
    3. Watch `cancel_requested` (or ``await wait_cancelled()``); qagents also
       calls `ExternalAgent.cancel_turn`. Then return.
    4. Return an `ExternalOutcome`. Raise `ExternalAgentConnectionError` when
       the connection broke (the turn becomes ``interrupted``),
       `ExternalAgentUnavailableError` for a failure that will repeat.

    A repeated call (`resumable`) that finds its turn finished in the agent and
    reports it again from the agent's own record does so inside ``with
    turn.replaying():``, so the calling run does not count its usage twice.
    """

    def __init__(
        self,
        *,
        agent: ExternalAgent,
        name: str,
        task: list[Part],
        session: str | None,
        cwd: str | None,
        parent: RunContext[Any] | None,
        bridge: PermissionBridge,
        checkpoints: Mapping[str, Any] | None,
        resumable: bool,
        emit: Callable[[Any], None],
        on_cancel: Callable[[str], None],
        remember: bool = False,
    ) -> None:
        self.agent = agent
        self.name = name
        """Prefix of tool names and the ``agent`` of every event: the tool name the
        calling model sees, or the agent's name for a direct `ExternalAgent.stream`."""
        self.task = task
        """The prompt as parts (`TextPart`, `ImagePart`)."""
        self.session = session
        """The session to continue, as the caller gave it; ``None`` for a new one."""
        self.cwd = cwd
        """The working directory (absolute), or ``None`` when neither the call nor
        the agent named one."""
        self.parent = parent
        """The calling run's context (a tool call), or ``None`` for a direct run."""
        self.operation_id = parent.operation_id if parent is not None else None
        """Stable across a crash and a resume of the calling run: the base of client-chosen
        session ids (`suggested_session_id`)."""
        self.checkpoints: dict[str, Any] = dict(checkpoints or {})
        """Values journaled by an earlier attempt of this same call (`checkpoint`), for
        re-attaching after a crash. Empty on a first attempt."""
        self.resumable = resumable
        """The call is declared idempotent (the agent's ``resumable_turns``): after a
        crash the calling run repeats it, and the adapter must re-attach instead of
        starting over (see `checkpoints`)."""
        self.bridge = bridge
        self.session_id: str | None = None
        """The session id, once `begin` was called."""
        self.resumed = False
        self.backend: str | None = None
        self.usage = Usage()
        self._replayed: Usage | None = None
        """Inside `replaying`: the usage reported so far, which the calling run gets at
        the end of the block (only what its journal does not hold yet)."""
        self.tool_calls: dict[str, ExternalToolCall] = {}
        self.steps = 0
        self.raw: dict[str, Any] = {}
        """Backend extras for ``ExternalResult.raw``; adapters may add to it."""
        self.deferred_permission: ExternalPermission | None = None
        self.cancel_reason: str | None = None
        self._cancelled = asyncio.Event()
        self._emit_event = emit
        self._on_cancel = on_cancel
        self._remember = remember
        self._began = False
        self.run_id: str | None = None
        """``ext:<session>:<12 hex>``, set when ``run_started`` is emitted. Unique: random
        for each turn, except that a call of ``as_tool()`` derives it from its
        `operation_id`, so a call repeated after a crash keeps its id."""
        self._text: list[str] = []
        self._text_closed = False
        self._text_message: str | None = None
        """The ``message_id`` of the last `text_delta`."""
        self._text_override: str | None = None
        self._started_at: dict[str, float] = {}
        self._parts: dict[str, ToolCallPart] = {}
        self._parse_errors: dict[str, str] = {}
        """Tool calls whose latest input could not be read as a JSON object."""
        seed = self.operation_id or uuid.uuid4().hex
        self._suggested_session_id = "qa-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]
        if remember and self.operation_id:  # one turn per as_tool() call: stable on a repeat
            run_seed = f"run:{self.operation_id}"
            self._run_token = hashlib.sha256(run_seed.encode("utf-8")).hexdigest()[:12]
        else:
            self._run_token = uuid.uuid4().hex[:12]

    # -- what the adapter reads --------------------------------------------------------------

    @property
    def prompt_text(self) -> str:
        """The text parts of the task, joined with `prompts.EXTERNAL_AGENT_TASK_SEPARATOR` (a
        blank line)."""
        texts = (part.text for part in self.task if isinstance(part, TextPart))
        return prompts.EXTERNAL_AGENT_TASK_SEPARATOR.join(texts)

    @property
    def suggested_session_id(self) -> str:
        """A session id for agents that let the client choose one: ``"qa-"`` + 32 hex
        characters, derived from `operation_id`, so a repeated call gets the same id;
        random for a direct run (chosen once per turn, so every read gives the same id)."""
        return self._suggested_session_id

    @property
    def depth(self) -> int:
        """Nesting depth of this turn: the calling run's depth + 1, or 0 for a direct run."""
        return self.parent.depth + 1 if self.parent is not None else 0

    @property
    def cancel_requested(self) -> bool:
        return self._cancelled.is_set()

    async def wait_cancelled(self) -> str:
        """Wait until the turn is asked to stop; returns the reason."""
        await self._cancelled.wait()
        return self.cancel_reason or "cancelled"

    def request_cancel(self, reason: str = "cancelled") -> None:
        """Ask the turn to stop (the first reason wins). qagents then calls
        `ExternalAgent.cancel_turn`, and abandons the turn after ``cancel_grace_s``."""
        if self._cancelled.is_set():
            return
        self.cancel_reason = reason
        self._cancelled.set()
        self._on_cancel(reason)

    # -- reporting ---------------------------------------------------------------------------

    async def begin(
        self, session_id: str, *, resumed: bool = False, backend: str | None = None
    ) -> None:
        """The turn has a session: emit ``run_started`` (once) and journal the session id.

        Args:
            session_id: The external agent's session id.
            resumed: An existing session is continued (``session/resume``).
            backend: What serves this turn, for ``ExternalResult.backend``.
        """
        self.session_id = session_id
        self.resumed = resumed
        if backend is not None:
            self.backend = backend
        self._start()
        if self._remember and self.parent is not None:
            # Known now, not only when the turn ends: a call that is cancelled or times out
            # still leaves the session in ctx.state["qagents.external.sessions"].
            backend = self.backend or self.agent.backend
            _remember(self.parent, self.name, session_id, self.cwd, backend)
        if self.resumable:
            await self.checkpoint(_SESSION_CHECKPOINT, {"session_id": session_id, "cwd": self.cwd})

    def emit(self, event: Any) -> None:
        """Put a qagents event on the turn's stream (``run_started`` first, if not yet)."""
        self._start()
        self._emit_event(event)

    def text_delta(self, text: str, *, message_id: str | None = None) -> None:
        """Assistant text. A tool call in between starts a new message.

        Pass ``message_id`` when the protocol says which message (or which
        content block) the text belongs to. Text whose id differs from the
        text just before it is another block: it is separated by
        `prompts.EXTERNAL_AGENT_TEXT_BLOCK_SEPARATOR` (a blank line), in the
        turn's text and at the start of its ``text_delta`` event. Without ids,
        deltas are joined as they come, which is right for a token stream: for
        whole committed blocks, give each block its own id.
        """
        if not text:
            return
        if self._text_closed:
            self._text = []
            self._text_closed = False
            self._text_message = None
        if (
            message_id is not None
            and self._text
            and self._text_message is not None
            and message_id != self._text_message
        ):
            text = prompts.EXTERNAL_AGENT_TEXT_BLOCK_SEPARATOR + text
        if message_id is not None:
            self._text_message = message_id
        self._text.append(text)
        self.emit(TextDelta(text=text))

    def reasoning_delta(self, text: str) -> None:
        """Committed reasoning text."""
        if text:
            self.emit(ReasoningDelta(text=text))

    def set_text(self, text: str) -> None:
        """Set the turn's final text outright (instead of the last message's deltas)."""
        self._text_override = text

    @property
    def text(self) -> str:
        """The final text so far: the override, or the last message's text."""
        return self._text_override if self._text_override is not None else "".join(self._text)

    def step_started(self) -> int:
        """One model request of the agent begins: emit ``turn_started`` and count it.

        Beyond ``max_external_turns`` the turn is cancelled as ``limit_reached``.
        Returns the step number.
        """
        self.steps += 1
        self.emit(TurnStarted(turn=self.steps, agent=self.name))
        self._check_steps()
        return self.steps

    def tool_started(
        self,
        call_id: str,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        raw_arguments: str | None = None,
        parse_error: str | None = None,
        counts_as_step: bool = False,
    ) -> ToolCallPart:
        """The agent started a tool: emit ``tool_call_started`` with the prefixed name.

        Calling it again for the same id with ``arguments`` (an update that
        brings new input, also ``{}``) replaces the call's arguments, together
        with ``raw_arguments`` and ``parse_error``, and emits an
        `ExternalAgentEvent` of kind ``tool_arguments`` (``{tool_call_id,
        name, arguments}``). The part that ``tool_call_started`` carried is
        not changed; ``tool_call_finished`` carries the updated one. Without
        ``arguments`` an update changes nothing. ``counts_as_step`` counts the
        call toward ``max_external_turns`` (for protocols without step events).
        A permission request about a call whose latest input has a
        ``parse_error`` is refused (fail closed).
        """
        self._text_closed = True
        known = self._parts.get(call_id)
        if known is not None:
            if arguments is None:
                return known  # an update without new input
            new = (dict(arguments), raw_arguments, parse_error)
            if new == (known.arguments, known.raw_arguments, known.parse_error):
                return known
            updated = known.model_copy(
                update={
                    "arguments": dict(arguments),
                    "raw_arguments": raw_arguments,
                    "parse_error": parse_error,
                }
            )
            self._parts[call_id] = updated  # the part tool_call_started carried stays as it was
            self.tool_calls[call_id].arguments = dict(arguments)
            self._note_parse_error(call_id, parse_error)
            self.event(
                "tool_arguments",
                {"tool_call_id": call_id, "name": updated.name, "arguments": dict(arguments)},
            )
            return updated
        part = ToolCallPart(
            id=call_id,
            name=f"{self.name}.{name}",
            arguments=dict(arguments or {}),
            raw_arguments=raw_arguments,
            parse_error=parse_error,
        )
        self._parts[call_id] = part
        self._note_parse_error(call_id, parse_error)
        self.tool_calls[call_id] = ExternalToolCall(
            id=call_id, name=name, arguments=dict(arguments or {})
        )
        self._started_at[call_id] = time.perf_counter()
        self.emit(ToolCallStarted(call=part, agent=self.name))
        if counts_as_step:
            self.steps += 1
            self._check_steps()
        return part

    def tool_finished(
        self,
        call_id: str,
        content: str,
        *,
        is_error: bool = False,
        metadata: Mapping[str, Any] | None = None,
        name: str | None = None,
    ) -> ToolResultPart:
        """The agent's tool call ended: emit ``tool_call_finished`` with the measured duration.

        A call that was never started is started first (``name`` names it).
        """
        if call_id not in self._parts:
            self.tool_started(call_id, name or "unknown")
        part = self._parts[call_id]
        record = self.tool_calls[call_id]
        began = self._started_at.get(call_id, time.perf_counter())
        duration = (time.perf_counter() - began) * 1000
        record.is_error = is_error
        record.finished = True
        record.duration_ms = duration
        result = ToolResultPart(
            tool_call_id=call_id,
            name=part.name,
            content=content,
            is_error=is_error,
            metadata=dict(metadata or {}),
        )
        self.emit(ToolCallFinished(call=part, result=result, duration_ms=duration, agent=self.name))
        return result

    def add_usage(self, usage: Usage) -> None:
        """Count token usage now: in the turn, in the calling run, and against its budgets.

        When the calling run's token or cost budget is used up, the turn is
        cancelled as ``limit_reached`` (``max_total_tokens`` / ``max_cost_usd``).
        Inside `replaying`, the calling run gets its share when the block ends.
        """
        self.usage = self.usage.add(usage)
        if self._replayed is not None:
            self._replayed = self._replayed.add(usage)
            return
        self._give_parent(usage)

    @contextlib.contextmanager
    def replaying(self, *, since: Usage | None = None) -> Iterator[None]:
        """Report again the events of a turn that an earlier attempt of this call already
        reported: a call repeated after a crash, whose turn had finished in the agent and
        is read back from the agent's own log instead of being run again.

        Inside the block `add_usage` counts usage in the turn only. When the block ends,
        the calling run gets what the replayed usage adds to what its journal already
        holds of this call's usage (``RunContext.journaled_usage``): usage that rode on
        any journal entry before the crash is not counted twice, and usage the crash lost
        is counted once. ``since``: the part of that journaled usage that belongs to
        older turns of the call (journaled before the replayed turn was prompted), which
        the replayed usage does not cover.
        """
        self._replayed = Usage()
        try:
            yield
        finally:
            replayed, self._replayed = self._replayed, None
        if self.parent is None:
            return
        held = _beyond(self.parent.journaled_usage, since or Usage())
        owed = _beyond(replayed, held)
        if owed != Usage():
            self._give_parent(owed)

    def _give_parent(self, usage: Usage) -> None:
        if self.parent is None:
            return
        self.parent.add_usage(usage)
        remaining = self.parent.remaining_budget()
        for key in ("max_total_tokens", "max_cost_usd"):
            value = remaining.get(key)
            if value is not None and value <= 0:
                self.request_cancel(key)
                return

    def model_responded(
        self,
        message: Message,
        *,
        usage: Usage | None = None,
        model: str | None = None,
        provider: str | None = None,
        finish_reason: FinishReason = "stop",
    ) -> None:
        """One model response of the agent: emit ``model_responded`` and `add_usage` its usage."""
        response = ModelResponse(
            message=message,
            usage=usage or Usage(),
            model=model,
            provider=provider,
            finish_reason=finish_reason,
        )
        self.emit(ModelResponded(response=response, agent=self.name))
        if usage is not None:
            self.add_usage(usage)

    def event(
        self, kind: str, data: Mapping[str, Any] | None = None, *, session_id: str | None = None
    ) -> None:
        """Emit an `ExternalAgentEvent` of ``kind``."""
        self.emit(
            ExternalAgentEvent(
                agent=self.name,
                kind=kind,
                session_id=session_id or self.session_id,
                data=dict(data or {}),
            )
        )

    async def ask_permission(
        self,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        tool_call_id: str | None = None,
        reason: str | None = None,
        options: Sequence[Mapping[str, Any]] | None = None,
    ) -> PermissionAnswer:
        """Decide a permission request of the agent (through the `PermissionBridge`).

        Without ``arguments``, those of the tool call started with
        ``tool_call_id`` are used (ACP sends the ``tool_call`` update with its
        ``rawInput`` before the request, which carries only the id).
        ``tool_calls`` holds the calls of the turn's own session only: an adapter
        that relays requests from other sessions (subagents) passes their
        ``arguments`` itself (``{}`` when unknown), never ``None``. Pass the
        request's ``options`` (ACP) and answer with ``answer.option_id``
        (``None``: answer "cancelled"). A request about a call whose latest
        input could not be read (``parse_error``) is answered "cancelled":
        the rules cannot see what would run. It never raises.
        """
        parse_error: str | None = None
        if arguments is None and tool_call_id is not None and tool_call_id in self.tool_calls:
            arguments = self.tool_calls[tool_call_id].arguments
            parse_error = self._parse_errors.get(tool_call_id)
        return await self.bridge.decide(
            tool,
            arguments,
            tool_call_id=tool_call_id,
            reason=reason,
            options=options,
            parse_error=parse_error,
        )

    async def checkpoint(self, key: str, value: Any) -> bool:
        """Journal a fact about this call now, before the call ends (a JSON value).

        It is written with ``RunContext.checkpoint`` of the calling tool call
        before this returns, so record it before acting on it. When the calling
        run repeats this call after a crash, the value is in `checkpoints`.
        Returns ``False`` when there is no running tool call to journal into (a
        direct run, or a parent context that is not a tool call's).

        Raises:
            SessionError: ``value`` is not JSON.
        """
        parent = self.parent
        if parent is None or parent.tool_call_id is None or parent.limits is None:
            self.checkpoints[key] = value
            return False
        await parent.checkpoint(key, value)
        self.checkpoints[key] = parent.checkpoints.get(key, value)  # as a repeat will read it
        return True

    # -- internals ------------------------------------------------------------------------------

    def _note_parse_error(self, call_id: str, parse_error: str | None) -> None:
        if parse_error:
            self._parse_errors[call_id] = parse_error
        else:
            self._parse_errors.pop(call_id, None)

    def _start(self) -> None:
        if self._began:
            return
        self._began = True
        session = self.session_id or self.session or ""
        self.run_id = f"ext:{session}:{self._run_token}"
        self._emit_event(
            RunStarted(
                run_id=self.run_id,
                session_id=session,
                agent=self.name,
                resumed=self.resumed,
                parent_run_id=self.parent.run_id if self.parent is not None else None,
                depth=self.depth,
            )
        )

    def _check_steps(self) -> None:
        limit = self.agent.max_external_turns
        if limit is not None and self.steps > limit:
            self.request_cancel("max_external_turns")

    def _record_permission(self, record: ExternalPermission, kind: str) -> None:
        data: dict[str, Any] = {
            "tool": record.tool,
            "arguments": record.arguments,
            "tool_call_id": record.tool_call_id,
        }
        if kind == "permission_decided":
            data.update(
                approved=record.approved,
                outcome=record.outcome,
                rule=record.decided_by,
                decision=record.decision,
                reason=record.reason,
            )
        self.event(kind, data)

    def _defer(self, record: ExternalPermission) -> bool:
        if self.deferred_permission is not None or self.parent is None:
            return False  # one pause per turn; a direct run has no run to pause
        self.deferred_permission = record
        self.request_cancel("permission")
        return True

    def _result(
        self, outcome: ExternalOutcome, changed_files: list[str] | None, git_root: str | None
    ) -> ExternalResult:
        status: ExternalStatus = outcome.status
        stop_reason = outcome.stop_reason
        reason = self.cancel_reason
        not_started = bool(outcome.raw.get("not_started"))  # its stop reason is final
        if reason is not None and status in ("cancelled", "limit_reached") and not not_started:
            if reason in LIMIT_REASONS:
                status, stop_reason = "limit_reached", reason
            elif reason in ("permission", "closed"):
                status, stop_reason = "cancelled", reason
            elif stop_reason is None:
                stop_reason = reason
        raw = {**self.raw, **outcome.raw}
        if self.deferred_permission is not None:
            record = self.deferred_permission
            raw["permission_deferred"] = {
                "agent": self.name,
                "external_session": self.session_id or self.session or "",
                "tool": record.tool,
                "external_tool": record.external_tool,
                "arguments": record.arguments,
                "fingerprint": record.fingerprint,
                "tool_call_id": record.tool_call_id,
            }
        if git_root is not None:
            raw["git_root"] = git_root
        files = outcome.changed_files if outcome.changed_files is not None else changed_files
        if files is not None:
            files = sorted({*files, *self._written_paths(git_root)})
        return ExternalResult(
            status=status,
            stop_reason=stop_reason,
            text=outcome.text if outcome.text is not None else self.text,
            # Only a session the turn began in (`begin`): a turn refused or broken before
            # that names none, so the footer never offers a session that was not confirmed.
            session_id=self.session_id or "",
            backend=self.backend or self.agent.backend,
            usage=self.usage,
            tool_calls=list(self.tool_calls.values()),
            permissions=list(self.bridge.records),
            changed_files=files,
            workspace=self.cwd,
            raw=raw,
        )

    def _written_paths(self, git_root: str | None) -> list[str]:
        """Paths from the arguments of successful write tools (a cross-check of git)."""
        found: list[str] = []
        for call in self.tool_calls.values():
            if call.name not in WRITE_TOOLS or call.is_error or not call.finished:
                continue
            path = call.arguments.get("path") or call.arguments.get("file_path")
            if not isinstance(path, str) or not path:
                continue
            relative = _relative_to(path, git_root or self.cwd)
            if relative is not None:
                found.append(relative)
        return found


_COUNTS = ("requests", "input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens")


def _beyond(total: Usage, already: Usage) -> Usage:
    """What ``total`` holds beyond ``already``, field by field and never below zero: the
    usage of a replayed turn that the journal does not hold yet."""
    counts = {name: max(0, getattr(total, name) - getattr(already, name)) for name in _COUNTS}
    unpriced = max(0, _unpriced(total) - _unpriced(already))
    cost: float | None = None
    if total.cost_usd is not None:
        cost = max(0.0, total.cost_usd - (already.cost_usd or 0.0))
    if not any(counts.values()) and not unpriced and not cost:
        return Usage()  # nothing beyond (a cost of 0.0 left over is nothing too)
    return Usage(**counts, cost_usd=cost, unpriced_requests=unpriced)


class _End:
    pass


_END = _End()


@dataclass
class _Failure:
    error: BaseException


class ExternalTurn:
    """One turn of an external agent: an async iterator of qagents events, plus control.

        async with coder.stream("Add type hints to utils.py") as turn:
            async for event in turn:
                print(event.type)
        print(turn.result.status, turn.result.changed_files)

    Events are the ordinary qagents ones (``run_started``, ``text_delta``,
    ``reasoning_delta``, ``tool_call_started``, ``tool_call_finished``,
    ``turn_started``, ``model_responded``) plus `ExternalAgentEvent`; the last
    one is ``external_event`` of kind ``turn_finished`` (there is no
    ``run_finished``: the result is `result`, an `ExternalResult`). Text
    arrives per committed block, not per token, for agents such as Devvi that
    send no token stream.

    Like a `RunStream` it has `run_id`, `text`, `cancel`, `wait` and
    `aclose`. Unlike a run, the agent keeps working while nobody reads the
    events: it runs in its own process. So use the turn in ``async with``, or
    read it to the end: leaving ``async with`` early, or breaking out of
    ``async for``, cancels the turn (``stop_reason`` ``"closed"``).

    Iterate it once. ``cancel()`` asks the agent to stop (ACP:
    ``session/cancel``), and after ``cancel_grace_s`` the turn is abandoned
    (the adapter stops the process). A turn cancelled before it starts, or
    while it waits for its workspace (one turn at a time per workspace), ends
    at once without sending anything (``raw["not_started"]``). Leaving
    ``async with`` early cancels the turn and waits for it to end. A setup
    failure (`ExternalAgentConfigurationError`) or a bug in the adapter is
    raised from the iteration; a broken connection is not: the result says
    ``interrupted``. A child turn (``parent=``) that would nest beyond
    ``Limits.max_depth``, or that would run an agent reporting no token usage
    under the calling run's token or cost budget (without
    ``allow_unmetered=True``), is refused before the agent gets the task: a
    `ToolError` with the text the calling model reads is raised from the
    iteration. A child turn's usage is added to the calling run as it is
    reported, so ``ctx.add_usage(turn.result.usage)`` adds nothing again (as
    for a child `Agent` run).
    """

    def __init__(self, agent: ExternalAgent, context: ExternalTurnContext) -> None:
        self.agent = agent
        self.context = context
        self.result: ExternalResult | None = None
        """Set once the turn has ended."""
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._driver: asyncio.Task[None] | None = None
        self._cancel_task: asyncio.Task[None] | None = None
        self._forced = False
        self._done = False
        self._waiting = False
        """The driver waits for the workspace or session lock: nothing was sent yet."""
        self._stopped_waiting = False
        """The driver was cancelled while it waited for the locks."""

    @property
    def session_id(self) -> str | None:
        """The external session id, once known."""
        return self.context.session_id or self.context.session

    @property
    def run_id(self) -> str | None:
        """The turn's run id (``ext:<session>:<12 hex>``), once ``run_started`` was emitted;
        unique, unlike the session id, which every turn of a session shares."""
        return self.context.run_id

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._events()

    async def _events(self) -> AsyncIterator[Any]:
        """The events, for ``async for``. A loop left before the end (``break``, an error)
        cancels the turn: the agent would otherwise go on working unobserved."""
        reason = "closed"
        try:
            while True:
                try:
                    event = await self.__anext__()
                except StopAsyncIteration:
                    return
                except asyncio.CancelledError:
                    reason = "cancelled"
                    raise
                yield event
        finally:
            if not self._done:
                self.cancel(reason)

    async def __anext__(self) -> Any:
        if self._done:
            raise StopAsyncIteration
        if self._driver is None:
            self._driver = asyncio.get_running_loop().create_task(
                self._drive(), name=f"external:{self.context.name}"
            )
        item = await self._queue.get()
        if isinstance(item, _End):
            self._done = True
            raise StopAsyncIteration
        if isinstance(item, _Failure):
            self._done = True
            raise item.error
        return item

    def cancel(self, reason: str = "cancelled") -> None:
        """Ask the turn to stop. Keep iterating (or ``await turn.wait()``) for the result."""
        self.context.request_cancel(reason)

    async def wait(self) -> ExternalResult:
        """Consume the remaining events and return the result."""
        async with self:
            async for _ in self:
                pass
        assert self.result is not None
        return self.result

    async def text(self) -> AsyncIterator[str]:
        """Only the text the agent writes, as it arrives (as `RunStream.text`).

        Check ``turn.result.status`` afterwards: the text of a turn that did
        not complete has been yielded too.
        """
        async with self:
            async for event in self:
                if isinstance(event, TextDelta):
                    yield event.text

    async def aclose(self, reason: str = "closed") -> None:
        """Stop the turn if it still runs, and wait (bounded by the grace period) for it to end."""
        driver = self._driver
        if driver is None:
            self._done = True
            return
        if not driver.done():
            self.cancel(reason)
            try:
                await asyncio.shield(driver)
            except asyncio.CancelledError:
                if not driver.done():  # cancelled again while waiting: abandon the turn now
                    self._forced = True
                    driver.cancel()
                raise

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: Any, *_: Any) -> None:
        cancelled = exc_type is not None and issubclass(exc_type, asyncio.CancelledError)
        await self.aclose("cancelled" if cancelled else "closed")

    # -- driving -----------------------------------------------------------------------------

    def _on_cancel(self, reason: str) -> None:
        driver = self._driver
        if driver is None or driver.done() or self._cancel_task is not None:
            return  # not started yet: the driver checks the cancel before anything else
        if self._waiting:  # nothing was sent: stop waiting for the workspace now
            self._stopped_waiting = True
            driver.cancel()
            return
        self._cancel_task = asyncio.get_running_loop().create_task(self._cancel_flow(driver))

    async def _cancel_flow(self, driver: asyncio.Task[None]) -> None:
        """Ask the adapter to stop the turn; abandon it after ``cancel_grace_s``."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, self.agent.cancel_grace_s)
        try:
            async with asyncio.timeout_at(deadline):
                await self.agent.cancel_turn(self.context)
        except TimeoutError:
            pass
        except Exception:
            logger.exception("cancel_turn of %s raised", self.context.name)
        if driver.done():
            return
        await asyncio.wait({driver}, timeout=max(0.0, deadline - loop.time()))
        if not driver.done():
            logger.warning(
                "%s did not stop within %g seconds; abandoning the turn",
                self.context.name,
                self.agent.cancel_grace_s,
            )
            self._forced = True
            driver.cancel()

    def _not_started(self) -> ExternalOutcome:
        """The outcome of a turn cancelled before anything was sent to the agent. A
        ``timeout`` then means the time ran out while it waited: ``busy``."""
        reason = self.context.cancel_reason or "cancelled"
        stop_reason = "busy" if reason == "timeout" else reason
        return ExternalOutcome(
            status="cancelled", stop_reason=stop_reason, raw={"not_started": True}
        )

    async def _drive(self) -> None:
        ctx = self.context
        try:
            outcome: ExternalOutcome | None = None
            changed: list[str] | None = None
            root: str | None = None
            try:
                if (refusal := _depth_refusal(ctx.parent, ctx.name)) is not None:
                    raise ToolError(refusal)
                if ctx.cancel_requested:  # cancelled before it was first iterated
                    outcome = self._not_started()
                else:
                    outcome, changed, root = await self._locked_turn()
            except asyncio.CancelledError:
                if outcome is None and self._stopped_waiting:
                    outcome = self._not_started()
                elif outcome is None:  # abandoned before the turn ended (not while git ran)
                    outcome = ExternalOutcome(
                        status="cancelled", stop_reason=ctx.cancel_reason, raw={"forced": True}
                    )
                ctx.cancel_reason = ctx.cancel_reason or "cancelled"
            result = ctx._result(outcome, changed, root)
            self.result = result
            if ctx.parent is not None and result.usage != Usage():
                # Added to the calling run as it was reported: adding the total again is a no-op.
                ctx.parent.counted(result.usage)
            ctx.emit(
                ExternalAgentEvent(
                    agent=ctx.name,
                    kind="turn_finished",
                    session_id=result.session_id or None,
                    data=_result_summary(result),
                )
            )
        except BaseException as error:  # setup failures and adapter bugs reach the consumer
            self._queue.put_nowait(_Failure(error))
            if not isinstance(error, Exception):
                raise
        finally:
            self._queue.put_nowait(_END)

    async def _locked_turn(self) -> tuple[ExternalOutcome, list[str] | None, str | None]:
        """Connect, snapshot and run the turn while holding its workspace and session locks."""
        ctx, agent = self.context, self.agent
        changed: list[str] | None = None
        root: str | None = None
        self._waiting = True
        try:
            async with agent._turn_locks(ctx.cwd, ctx.session):
                self._waiting = False
                if ctx.cancel_requested:  # cancelled the moment the locks were free
                    return self._not_started(), None, None
                # Connected inside the locks: a turn that waited behind one whose process
                # died finds the agent disconnected and starts it again.
                await agent.ensure_connected()
                refusal = await agent._unmetered_refusal(ctx.parent, ctx.name)
                if refusal is not None:
                    raise ToolError(refusal)
                snapshot = await _git_snapshot(ctx.cwd) if agent._tracks(ctx.cwd) else None
                if ctx.cancel_requested:
                    outcome = ExternalOutcome(status="cancelled", stop_reason=ctx.cancel_reason)
                else:
                    outcome = await self._run_turn()
                if snapshot is not None:
                    root = snapshot[0]
                    try:
                        changed = await _changed_since(snapshot)
                    except asyncio.CancelledError:
                        # Cancelled while git ran after the turn: the turn's outcome stands.
                        ctx.cancel_reason = ctx.cancel_reason or "cancelled"
                        return outcome, None, root
        finally:
            self._waiting = False
        return outcome, changed, root

    async def _run_turn(self) -> ExternalOutcome:
        ctx = self.context
        try:
            return await self.agent.run_turn(ctx)
        except asyncio.CancelledError:
            if not self._forced:
                raise
            self.agent.mark_disconnected()  # it did not stop: the adapter stops the process
            return ExternalOutcome(
                status="cancelled", stop_reason=ctx.cancel_reason, raw={"forced": True}
            )
        except ExternalAgentUnavailableError as error:
            return ExternalOutcome(
                status="error", stop_reason=str(error), raw={"unavailable": True}
            )
        except ExternalAgentConfigurationError:
            raise
        except ExternalAgentConnectionError as error:
            self.agent.mark_disconnected()
            if not error.ambiguous:  # nothing of the turn was sent: its outcome is known
                logger.warning("%s: the turn was not sent: %s", ctx.name, error)
                return ExternalOutcome(
                    status="error", stop_reason=error.reason, raw={"error": str(error)}
                )
            logger.warning("%s: the connection broke during the turn: %s", ctx.name, error)
            raw: dict[str, Any] = {"error": str(error), "ambiguous": True}
            if error.stderr_tail:
                raw["stderr_tail"] = error.stderr_tail
            if error.returncode is not None:
                raw["returncode"] = error.returncode
            return ExternalOutcome(status="interrupted", stop_reason=error.reason, raw=raw)
