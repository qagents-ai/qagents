"""`ExternalAgent`: the base class of every adapter.

What an adapter implements (`ExternalAgent.capabilities`, ``run_turn``, and optionally
``connect``, ``disconnect`` and ``cancel_turn``), the connection's lifecycle (connect on
first use, connect again after a broken connection or on another event loop, close), and
how a turn starts: `ExternalAgent.stream` and ``run`` directly, ``as_tool`` as a tool of
a calling run (its body is in `qagents.external.tool`).

Turns take two locks first: one per workspace, shared by every external agent of the
process (one writer per directory), and one per session of the agent.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy as copying
import logging
import os
import weakref
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Self

from .. import prompts
from ..errors import ConfigurationError
from ..messages import ImagePart, Part, TextPart
from ..permissions import Permissions
from ..tools import RunContext, Tool
from .contract import ExternalCapabilities, ExternalOutcome, ExternalResult
from .permissions import _DEFAULT_TIMEOUTS, PermissionBridge, PermissionPosture
from .tool import _build_tool
from .turn import ExternalTurn, ExternalTurnContext

logger = logging.getLogger("qagents.external")


class _HeldLock:
    __slots__ = ("lock", "users")

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.users = 0
        """Tasks that hold the lock or wait for it."""


class _LoopLocks:
    """``asyncio.Lock`` objects by key, separately for each event loop.

    A lock that nobody holds or waits for is dropped, so a long-lived agent
    that sees many sessions and workspaces keeps only the locks in use.
    """

    def __init__(self) -> None:
        self._by_loop: weakref.WeakKeyDictionary[Any, dict[Any, _HeldLock]] = (
            weakref.WeakKeyDictionary()
        )

    @contextlib.asynccontextmanager
    async def hold(self, key: Any) -> AsyncIterator[None]:
        """Hold the lock of ``key`` for the block."""
        locks = self._by_loop.setdefault(asyncio.get_running_loop(), {})
        entry = locks.get(key)
        if entry is None:
            entry = locks[key] = _HeldLock()
        entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            entry.users -= 1
            if entry.users == 0 and locks.get(key) is entry:
                del locks[key]

    def __len__(self) -> int:
        """Locks in use on the running event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return 0
        return len(self._by_loop.get(loop, {}))


_WORKSPACE_LOCKS = _LoopLocks()
"""One writer per workspace, across every external agent of the process."""


def _workspace_key(cwd: str) -> str:
    return os.path.normcase(os.path.abspath(cwd))


class ExternalAgent(ABC):
    """Another agent runtime, driven over a control protocol. Subclass it to add one.

    An adapter implements:

    * `capabilities` (abstract): what the agent supports.
    * `run_turn` (abstract): run one turn and report it through the
      `ExternalTurnContext`; return an `ExternalOutcome`.
    * `connect` / `disconnect` (optional): start and stop the process and
      run the handshake. `connect` runs once, on first use (or on
      ``async with``); raise `ExternalAgentConfigurationError` when it fails.
    * `cancel_turn` (optional): ask the agent to stop the current turn
      (ACP ``session/cancel``). If `run_turn` has not returned
      ``cancel_grace_s`` seconds after a cancel, it is cancelled with
      ``CancelledError``: then kill the process (``ManagedProcess.kill()``)
      before re-raising.

    After a broken connection (`run_turn` raised `ExternalAgentConnectionError`)
    or an abandoned turn the agent is marked disconnected (`mark_disconnected`):
    the next turn calls `disconnect` and then `connect` again, so `connect` can
    assume a clean slate.

    The base class provides `stream`, `run` and `as_tool`, the permission
    bridge, the per-workspace and per-session locks (one turn at a time per
    workspace and per session), git change tracking, budgets, cancellation and
    journaling.

    Args:
        name: Name of the agent; also the prefix of permission names for a
            direct `stream` (``as_tool`` uses the tool name).
        description: What it does, for the calling model (``as_tool``).
        workspace: Default working directory of turns (made absolute).
        permissions: Rules for its permission requests. ``None``: the calling
            run's rules (``as_tool``), else ``Permissions()``. When these
            have no ``on_ask``, the calling run's ``on_ask`` is used.
        on_permission: The posture for requests that need a human when no
            ``on_ask`` answers: ``"rules"`` (refuse), ``"pause"`` (pause the
            calling run until a human answers) or ``"live"``.
        approval_timeout_s: Seconds for one permission decision; ``None``: the
            posture's default (60 s, or 15 min for ``"live"``).
        cancel_grace_s: Seconds a cancelled turn gets to stop before it is abandoned.
        max_external_turns: Steps (model requests, or tool calls for protocols
            without step events) per turn before it is stopped as
            ``limit_reached``; ``None``: no limit.
        allow_unmetered: Let a calling run with a token or cost budget start
            this agent even though it reports no token usage. Its spend is
            then not counted.
        track_changes: Record ``changed_files`` with ``git status``: ``None``
            when the workspace is a git repository, ``True`` or ``False``.
            Git runs with a small environment (no secrets), no network
            transport, and with every program the repository's config could
            name switched off (fsmonitor, filters, external diff, textconv,
            pager), because the agent can write that config. Submodules'
            work trees are not looked into.

    Lifecycle: use the agent inside ``async with`` (or call `aclose`) on the
    event loop that runs it. ``Agent.run_sync`` starts a new event loop per
    call; an agent used across such calls stops its process and starts a new
    one on the next loop, and a process left running is stopped when Python
    exits. `disconnect` always runs to its end: an `aclose` (or a reconnect)
    that is cancelled while it waits leaves it running, and the next one waits
    for it.
    """

    backend: str = "external"
    """What serves the turns, for ``ExternalResult.backend``; adapters refine it per turn."""

    grant_ignored_arguments: frozenset[str] = frozenset()
    """Top-level arguments of a permission request that an approval given across a pause
    (``on_permission="pause"``) need not match: free text the agent's model writes anew on
    every attempt and that does not change what runs (Devvi: ``description`` and
    ``justification``). Every other argument must be the same. Empty: all must match."""

    resumable_turns: bool = False
    """A repeated call (the calling run resumed after a crash) re-attaches to the earlier
    attempt instead of starting over, using `ExternalTurnContext.checkpoints` or a
    client-chosen session id. ``as_tool()`` then declares the tool idempotent, so the
    calling run repeats it; otherwise the calling model is told its outcome is unknown."""

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        workspace: str | Path | None = None,
        permissions: Permissions | None = None,
        on_permission: PermissionPosture = "rules",
        approval_timeout_s: float | None = None,
        cancel_grace_s: float = 5.0,
        max_external_turns: int | None = 50,
        allow_unmetered: bool = False,
        track_changes: bool | None = None,
    ) -> None:
        if on_permission not in _DEFAULT_TIMEOUTS:
            raise ConfigurationError(
                f"on_permission must be 'rules', 'pause' or 'live', not {on_permission!r}."
            )
        if not name:
            raise ConfigurationError("An external agent needs a name.")
        self.name = name
        self.description = description
        self.workspace = os.path.abspath(workspace) if workspace is not None else None
        self.permissions = permissions
        self.on_permission: PermissionPosture = on_permission
        self.approval_timeout_s = approval_timeout_s
        self.cancel_grace_s = cancel_grace_s
        self.max_external_turns = max_external_turns
        self.allow_unmetered = allow_unmetered
        self.track_changes = track_changes
        self._connected = False
        """The connection is usable."""
        self._opened = False
        """`connect` succeeded and `disconnect` has not run since: something to release."""
        self._loop: asyncio.AbstractEventLoop | None = None
        """The event loop the connection was made on."""
        self._connect_lock: asyncio.Lock | None = None
        self._connect_lock_loop: asyncio.AbstractEventLoop | None = None
        self._turns: weakref.WeakSet[ExternalTurn] = weakref.WeakSet()
        self._session_locks = _LoopLocks()
        self._releasing: asyncio.Task[None] | None = None
        """The `disconnect` in progress, which a cancelled caller leaves running."""

    # -- what adapters implement ------------------------------------------------------------

    @abstractmethod
    async def capabilities(self) -> ExternalCapabilities:
        """What this agent supports (may connect first)."""

    @abstractmethod
    async def run_turn(self, turn: ExternalTurnContext) -> ExternalOutcome:
        """Run one turn: send ``turn.task`` to ``turn.session`` (or a new session) in
        ``turn.cwd``, report what happens through ``turn``, return how it ended."""

    async def connect(self) -> None:  # noqa: B027 - optional hook
        """Start the agent (process, handshake). Called once, on first use."""

    async def disconnect(self) -> None:  # noqa: B027 - optional hook
        """Stop the agent and release what it holds. Called by `aclose`."""

    async def cancel_turn(self, turn: ExternalTurnContext) -> None:  # noqa: B027 - optional hook
        """Ask the agent to stop ``turn`` (``turn.cancel_reason`` says why). Return quickly."""

    def grant_arguments(self, tool: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """The arguments of a permission request that an approval across a pause must match.

        An approval answers the request with the same fingerprint (tool name and
        these arguments). By default every argument except `grant_ignored_arguments`.
        Override it for a finer rule; keep everything that changes what would run.

        Args:
            tool: The agent's own tool name (``bash``, not ``coder.bash``).
            arguments: The request's arguments.
        """
        ignored = self.grant_ignored_arguments
        return {key: value for key, value in arguments.items() if key not in ignored}

    # -- lifecycle ------------------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._connected

    def mark_disconnected(self) -> None:
        """The connection is gone (the process exited, or was killed after a turn that
        would not stop): the next turn calls `disconnect`, then `connect` again.

        The base class calls it itself when `run_turn` raises
        `ExternalAgentConnectionError` and when a turn is abandoned after
        ``cancel_grace_s``. Adapters call it when they notice a dead connection.
        """
        self._connected = False

    async def ensure_connected(self) -> None:
        """Run `connect` if the agent is not connected; return at once if it is.

        After `mark_disconnected`, `disconnect` runs first, to release what the
        broken connection held (the old process). The same happens when the
        connection was made on another event loop: ``Agent.run_sync`` (and
        each ``asyncio.run``) uses a new loop per call, and a connection does
        not survive its loop. Its process is then stopped and a new one is
        started. Close the agent (``async with`` or `aclose`) inside the loop
        that uses it to stop the process as soon as you are done.
        """
        loop = asyncio.get_running_loop()
        if self._connected and self._loop is loop:
            return
        async with self._lock_for_connect(loop):
            if self._connected and self._loop is loop:
                return
            if self._opened:
                if self._loop is not loop:
                    logger.info(
                        "%s was connected on an event loop that has ended; connecting again",
                        self.name,
                    )
                self._connected = False
                try:
                    await self._release()
                except Exception:
                    logger.exception("disconnect of %s raised; connecting again anyway", self.name)
            await self.connect()
            self._loop = loop
            self._opened = self._connected = True

    async def aclose(self) -> None:
        """Cancel the turns still running, then `disconnect`. Safe to call more than once.

        A `connect` still in progress (a turn that is starting) is waited for, so
        the agent never stays connected after `aclose` returns. Turns created
        but not iterated yet are cancelled too: they end without connecting.
        When `aclose` itself is cancelled, `disconnect` still runs to its end,
        and a later `aclose` (or the next turn) waits for it.
        """
        loop = asyncio.get_running_loop()
        turns: list[ExternalTurn] = []
        for turn in list(self._turns):
            if turn._driver is None and not turn._done:
                turn.cancel("closed")  # not started: it ends at once, without connecting
            elif turn._driver is not None and turn._driver.get_loop() is loop:
                turns.append(turn)
        await asyncio.gather(*(turn.aclose("closed") for turn in turns), return_exceptions=True)
        async with self._lock_for_connect(loop):
            self._connected = False
            if self._opened:
                await self._release()

    async def _release(self) -> None:
        """Run `disconnect` to its end, whatever happens to the caller.

        It runs in a task of its own, awaited through ``asyncio.shield``: a
        caller that is cancelled (a timeout, a cancelled run) leaves it running,
        so the process is still stopped, and the next caller awaits the same
        task. ``_opened`` is cleared only when it has ended, so it is never
        forgotten half done. A `disconnect` that raises is not retried.
        """
        loop = asyncio.get_running_loop()
        task = self._releasing
        if task is None or task.done() or task.get_loop() is not loop:
            task = loop.create_task(self._disconnect_once(), name=f"external:{self.name}:close")
            self._releasing = task
        await asyncio.shield(task)

    async def _disconnect_once(self) -> None:
        try:
            await self.disconnect()
        except asyncio.CancelledError:
            raise  # only when the loop shuts down: _opened stays set, the next call retries
        except BaseException:
            self._opened = False
            raise
        else:
            self._opened = False
        finally:
            if self._releasing is asyncio.current_task():
                self._releasing = None

    def _lock_for_connect(self, loop: asyncio.AbstractEventLoop) -> asyncio.Lock:
        """The lock that orders `connect` and `disconnect`, one per event loop."""
        if self._connect_lock is None or self._connect_lock_loop is not loop:
            self._connect_lock = asyncio.Lock()
            self._connect_lock_loop = loop
        return self._connect_lock

    async def __aenter__(self) -> Self:
        await self.ensure_connected()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    # -- running ----------------------------------------------------------------------------------

    def stream(
        self,
        task: str | Sequence[Part],
        *,
        session: str | None = None,
        cwd: str | Path | None = None,
        parent: RunContext[Any] | None = None,
        permissions: Permissions | None = None,
    ) -> ExternalTurn:
        """Start a turn and return its event stream.

        Args:
            task: The prompt: text, or parts (`TextPart`, `ImagePart`).
            session: An external session id to continue; ``None`` starts a new one.
            cwd: Working directory; defaults to the agent's ``workspace``.
            parent: The calling tool call's context: the turn becomes its child
                (depth, usage, budgets, the caller's permissions and ``on_ask``).
            permissions: Rules for this turn's permission requests, instead of
                the agent's (or the caller's).
        """
        return self._open_turn(
            task, session=session, cwd=cwd, parent=parent, permissions=permissions, label=None
        )

    async def run(
        self,
        task: str | Sequence[Part],
        *,
        session: str | None = None,
        cwd: str | Path | None = None,
        parent: RunContext[Any] | None = None,
        permissions: Permissions | None = None,
    ) -> ExternalResult:
        """Run a turn to its end and return the result. Takes the arguments of `stream`."""
        turn = self.stream(task, session=session, cwd=cwd, parent=parent, permissions=permissions)
        return await turn.wait()

    def as_tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        timeout: float | None = None,
        continuable: bool = True,
        requires_approval: bool = False,
        idempotent: bool | None = None,
    ) -> Tool:
        """Expose the agent as a tool: ``name(task, session=None)``.

        Each call is one turn, run as a child of the calling run: at depth + 1
        (refused beyond ``Limits.max_depth``), with its usage added to the
        caller as it is reported and checked against the caller's budgets,
        its events forwarded as `SubagentEvent` (``run_started`` included),
        cancelled when the caller is cancelled or times out, with its
        permission requests decided by the caller's `Permissions` on names
        ``"<name>.<tool>"``. The result is the final text plus the footer
        (`ExternalResult.for_model`); a turn that did not complete is an
        error result that still names the session. The session is
        remembered in ``ctx.state["qagents.external.sessions"]`` as soon as
        the agent opens it, so it is there also when the call is cancelled.

        Guarantees across a crash of the calling run: a call whose result was
        journaled never runs again (the agent loop's guarantee). A call that
        was running is repeated only when the agent's turns are resumable
        (``resumable_turns``): the tool is then idempotent, the repeat
        re-attaches, and a turn that had already finished returns its
        journaled result without running again. Otherwise the calling model
        is told the outcome is unknown.

        Args:
            name: Tool name; defaults to the agent's name (made tool-safe). It
                is also the prefix of the permission names.
            description: Tool description; defaults to the agent's description,
                or `EXTERNAL_AGENT_DESCRIPTION`.
            timeout: Seconds for one turn. When it expires the turn is
                cancelled and the model is told so, with the session to
                continue. ``None``: the caller's ``Limits.tool_timeout_s``
                (120 s by default), which is short for a coding agent. The
                turn is then cancelled ``cancel_grace_s + 5`` seconds before
                ``tool_timeout_s``, so the cancel is graceful and the model
                is told the session, but only when ``tool_timeout_s`` is more
                than twice that (more than 20 s with the default
                ``cancel_grace_s``). A shorter ``tool_timeout_s`` is ended by
                the calling loop itself: the turn is abandoned and the model
                reads the loop's timeout text, which names no session (the
                session is still in ``ctx.state``). Give coding agents a
                ``timeout``. The time also runs while the call waits for its
                workspace (one turn at a time per workspace, across agents,
                and per session): a call whose time runs out there was never
                started, and the model reads `EXTERNAL_AGENT_BUSY` (stop
                reason ``busy``).
            continuable: Offer the ``session`` parameter. The model may continue
                only a session this tool opened in this qagents session (they are
                kept in ``ctx.state["qagents.external.sessions"]``, the latest 50);
                any other id is refused with an error result, before the agent is
                reached. Code that calls `stream` or `run` itself may continue any
                session. Without the parameter every call starts a new session,
                except the call after a pause (``on_permission="pause"``): it
                continues the paused session by itself, so the human's answer applies.
            requires_approval: Ask the caller's human once per delegation
                (and run the agent with a permissive policy in a disposable
                workspace). The caller's permission rules apply as usual.
            idempotent: Override the default (``resumable_turns``).
        """
        return _build_tool(
            self,
            name=name,
            description=description,
            timeout=timeout,
            continuable=continuable,
            requires_approval=requires_approval,
            idempotent=idempotent,
        )

    # -- internals --------------------------------------------------------------------------------

    def _open_turn(
        self,
        task: str | Sequence[Part],
        *,
        session: str | None,
        cwd: str | Path | None,
        parent: RunContext[Any] | None,
        permissions: Permissions | None,
        label: str | None,
        grants: Mapping[str, bool] | None = None,
        checkpoints: Mapping[str, Any] | None = None,
        resumable: bool = False,
    ) -> ExternalTurn:
        prefix = label or self.name
        directory = os.path.abspath(cwd) if cwd is not None else self.workspace
        holder: list[ExternalTurn] = []
        context_ref: list[ExternalTurnContext] = []
        bridge = PermissionBridge(
            prefix=prefix,
            permissions=self._permissions_for(parent, permissions),
            ctx=parent,
            posture=self.on_permission,
            approval_timeout_s=self.approval_timeout_s,
            grants=grants,
            on_record=lambda record, kind: context_ref[0]._record_permission(record, kind),
            on_defer=lambda record: context_ref[0]._defer(record),
            is_cancelling=lambda: context_ref[0].cancel_requested,
            grant_arguments=self.grant_arguments,
        )

        def emit(event: Any) -> None:
            holder[0]._queue.put_nowait(event)

        def on_cancel(reason: str) -> None:
            holder[0]._on_cancel(reason)

        context = ExternalTurnContext(
            agent=self,
            name=prefix,
            task=_as_parts(task),
            session=session,
            cwd=directory,
            parent=parent,
            bridge=bridge,
            checkpoints=checkpoints,
            resumable=resumable,
            emit=emit,
            on_cancel=on_cancel,
            remember=label is not None,  # as_tool(): the session goes into ctx.state
        )
        context_ref.append(context)
        turn = ExternalTurn(self, context)
        holder.append(turn)
        self._turns.add(turn)
        return turn

    def _permissions_for(
        self, parent: RunContext[Any] | None, explicit: Permissions | None
    ) -> Permissions:
        """The turn's rules: explicit, the agent's own, else the calling run's; with the
        calling run's ``on_ask`` when they have none (as a child run does)."""
        inherited = parent.permissions if parent is not None else None
        own = explicit or self.permissions
        if own is None:
            return inherited or Permissions()
        if own.on_ask is None and inherited is not None and inherited.on_ask is not None:
            own = copying.copy(own)
            own.on_ask = inherited.on_ask
        return own

    @contextlib.asynccontextmanager
    async def _turn_locks(self, cwd: str | None, session: str | None) -> AsyncIterator[None]:
        async with contextlib.AsyncExitStack() as stack:
            if cwd is not None:
                await stack.enter_async_context(_WORKSPACE_LOCKS.hold(_workspace_key(cwd)))
            if session is not None:
                await stack.enter_async_context(self._session_locks.hold(session))
            yield

    def _tracks(self, cwd: str | None) -> bool:
        return cwd is not None and self.track_changes is not False

    async def _unmetered_refusal(self, parent: RunContext[Any] | None, label: str) -> str | None:
        """Why a child turn may not start: the calling run has a token or cost budget and this
        agent reports no token usage (and ``allow_unmetered`` is off); ``None`` when it may."""
        if parent is None or self.allow_unmetered:
            return None
        budget = parent.remaining_budget()
        if all(budget.get(key) is None for key in ("max_total_tokens", "max_cost_usd")):
            return None
        capabilities = await self.capabilities()
        if capabilities.token_usage:
            return None
        return prompts.EXTERNAL_AGENT_UNMETERED.format(name=label)


def _as_parts(task: str | Sequence[Part]) -> list[Part]:
    if isinstance(task, str):
        return [TextPart(text=task)]
    parts = list(task)
    for part in parts:
        if not isinstance(part, TextPart | ImagePart):
            raise ConfigurationError(
                f"An external agent takes text and image parts, not {type(part).__name__}."
            )
    return parts
