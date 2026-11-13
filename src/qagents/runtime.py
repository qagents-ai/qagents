"""Long-running agents: background runs, leases and crash recovery.

`AgentRuntime` runs agents as background tasks on top of a session store, so
a run outlives the request that started it and the process that ran it:

    store = PostgresStore()
    runtime = AgentRuntime(store, [support], deps_factory=make_deps)

    handle = await runtime.start("support", "Refund order A-1001", session_id="ticket-77")
    async for event in handle.events():
        ...  # forward to a websocket, or ignore
    result = await handle.wait()

    # In every worker process: resume the runs whose worker died.
    await runtime.run_forever()

How it fits together:

- **Leases.** Before a run starts, the runtime takes the session's lease in
  the store. A heartbeat renews it every ``heartbeat_s`` seconds from the
  moment it is taken (also while the run is prepared or queued), and every
  journal write is checked against it in the same transaction
  (`SessionStore.append_leased`), so at most one worker writes to a run at a
  time, even after a network partition. A worker that finds its lease taken
  over stops its run.
- **Start.** `AgentRuntime.start` journals the new run (its ``run_started``
  entry and the prompt) before it returns, also when the run then waits for
  a free slot. A run that start() accepted is therefore never lost: after a
  shutdown or a crash it is open in the journal and is recovered.
- **Recovery.** `AgentRuntime.recover` lists the sessions whose run is
  ``"running"`` or ``"failed"``, takes each free or expired lease and resumes
  the run with the agent named in the journal. Paused runs wait for a
  person and are never resumed automatically. `run_forever` calls it every
  ``poll_interval_s`` seconds.
- **Shutdown.** `AgentRuntime.shutdown` asks each run to stop at the next
  safe point (before a model request, after a model response, or after a
  tool call when no tool of the run is executing and no journal write is in
  flight), cancels runs that do not reach one in time, and releases the
  leases. The runs stay open in the journal, so another worker resumes them
  at once instead of waiting for a lease to expire.

Delivery guarantees, stated plainly. They are the journal's guarantees; the
runtime adds liveness, not exactly-once execution:

- A tool call with a recorded result is never run again, on any worker.
- A tool call that was executing when its worker died or was cancelled is
  run again on recovery only if the tool is ``read_only`` or ``idempotent``;
  otherwise the model is told its outcome is unknown. Delivery is
  at-least-once: give tools with side effects ``ctx.operation_id`` as an
  idempotency key.
- A model request that was in flight when the worker *died* is sent again on
  recovery. Its first response was lost, so this costs tokens; no tool had
  acted on it. This is the one case where recovery repeats a request that
  may have been billed: the journal cannot tell a request cut off by a
  crash from one that was never sent.
- A run that failed (it raised, for example on a provider error) is resumed
  automatically until it has failed ``max_failures`` times, except when its
  last model request failed *ambiguously* (the provider may have processed
  and billed it; ``ModelError.ambiguous``) and the run's limits do not set
  ``retry_ambiguous=True``. Such a run waits for a person, who resumes it
  with `AgentRuntime.resume`. Set ``max_failures=0`` to leave every failed
  run to a person.
- A run whose worker dies on every attempt (a tool or a response that
  crashes the process) is resumed at most ``max_recoveries`` times in a row
  without making progress (a new message or tool result); then it is marked
  failed for a person.
- Limits given to `AgentRuntime.start` or `AgentRuntime.resume` are journaled,
  so a run recovered on another worker keeps them.
- Liveness depends on a worker calling `recover` (or `run_forever`). A run
  whose worker died is resumed once its lease expires, after at most
  ``lease_ttl_s`` plus ``poll_interval_s`` seconds.

Journal entries the runtime adds to a run (readers ignore kinds they do not
know): ``runtime_limits`` (``limits``: the `Limits` the run was started or
resumed with) and ``run_recovered`` (``worker``, ``status``: an automatic
recovery resumed the run; a recovery that a shutdown stops before then is not
journaled or counted). A ``run_failed`` entry the runtime writes for a crash loop
has ``needs_person: true``.

Rules for the store: a plain ``agent.run`` or ``agent.resume`` on a store with
leases holds a lease of its own while it executes, so `recover` leaves it
alone; continuing a session with ``result.session`` is safe. A store subclass
that overrides ``append`` but not ``append_leased`` gets no lease from a
plain run, and under a runtime its ``append`` is bypassed (every write goes
through ``append_leased``): override both. Sessions that belong to a
`qagents.workflow.Workflow` (legacy ``"workflow:"`` ids and encoded
``"workflow-journal:"`` / ``"workflow-step:"`` ids) are never recovered:
resume the workflow instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import os
import socket
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from .errors import ConfigurationError, RunInterrupted, SessionError
from .events import (
    AgentSwitched,
    Event,
    ModelResponded,
    RunFinished,
    RunStarted,
    StateUpdated,
    ToolCallFinished,
    TurnStarted,
)
from .session import Entry, LeaseError, Session, SessionStatus, SessionStore, _owner_token

if TYPE_CHECKING:
    from .agent import Agent, Limits, RunStream
    from .permissions import Approval
    from .results import RunResult

logger = logging.getLogger("qagents.runtime")

HandleStatus = Literal["queued", "running", "finished", "failed", "interrupted"]
"""Where a `RunHandle` stands.

* ``queued``: waiting for a free slot (``max_concurrent``). A started run is
  already journaled while it waits.
* ``running``: the run is being driven.
* ``finished``: the run returned a `RunResult` (completed, paused, cancelled, ...).
* ``failed``: the run raised; the error is in ``RunHandle.error``.
* ``interrupted``: shutdown or a lost lease stopped it; the run stays open
  (`RunInterrupted` says how to continue it).
"""

DepsFactory = Callable[[str], Any]
"""Builds ``deps`` for a session id. May be async (return an awaitable)."""

WORKFLOW_PREFIX = ("workflow:", "workflow-journal:", "workflow-step:")
"""Sessions whose id starts with this belong to a workflow and are not recovered."""

LIMITS_KIND = "runtime_limits"
"""Journal entry with the `Limits` a run was started or resumed with (``limits``)."""

RECOVERED_KIND = "run_recovered"
"""Journal entry written when a run that `AgentRuntime.recover` took resumes (``worker``,
``status``); `AgentRuntime` counts them for ``max_recoveries``."""

_RECOVERABLE = ("running", "failed")
_PROGRESS = frozenset({"message", "tool_result", "run_failed"})
"""Entry kinds after which a run's recoveries in a row are counted from zero again."""
_SAFE_AFTER = (TurnStarted, ModelResponded, AgentSwitched, ToolCallFinished)


class RunHandle:
    """A run that an `AgentRuntime` drives in the background.

    Created by `AgentRuntime.start`, `AgentRuntime.resume` and
    `AgentRuntime.recover`; not meant to be created directly.
    """

    def __init__(
        self,
        runtime: AgentRuntime,
        session_id: str,
        agent: str,
        *,
        resumed: bool,
        buffer: int,
        answers: bool = False,
        unsaved: Iterable[str] = (),
    ) -> None:
        self.session_id = session_id
        self.agent = agent
        """Name of the agent the run was started or resumed with."""
        self.resumed = resumed
        """True when the handle continues an unfinished run."""
        self.run_id: str | None = None
        """Set once the run has started (its ``run_started`` event)."""
        self.status: HandleStatus = "queued"
        """``queued``, ``running``, ``finished``, ``failed`` or ``interrupted`` (`HandleStatus`)."""
        self.result: RunResult | None = None
        """Set when ``status == "finished"``."""
        self.error: BaseException | None = None
        """Set when ``status`` is ``"failed"`` or ``"interrupted"``."""
        self._runtime = runtime
        self._events: deque[Event] = deque(maxlen=buffer)
        self._published = 0
        self._changed = asyncio.Event()
        self._done = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._preparing_task = asyncio.current_task()
        self._heartbeat: asyncio.Task[None] | None = None
        """Renews the lease from the moment it is taken, through preparation and the run."""
        self._stream: RunStream | None = None
        self._prepared: RunStream | None = None
        """A started run's stream, already journaled, until the handle drives it."""
        self._store: _LeasedStore | None = None
        self._cancel_reason: str | None = None
        self._stop_requested = False
        self._wake_up = asyncio.Event()
        """Set by `cancel` and by a stop request: a queued run stops waiting for its slot."""
        self._lease_lost = False
        self._lease_owner = ""  # set by _new_handle immediately after acquisition
        self._lease_deadline = asyncio.get_running_loop().time() + runtime.lease_ttl_s
        self._answers = answers
        """The run was resumed with approvals or results it has not finished using."""
        self._unsaved = set(unsaved)
        """Arguments given to resume() (``state``, ``limits``) not journaled yet."""

    @property
    def session(self) -> Session:
        """The run's session, for reading its journal or state."""
        return Session(self._runtime.store, id=self.session_id)

    def done(self) -> bool:
        """True once the run has finished, failed or been interrupted."""
        return self._done.is_set()

    async def wait(self) -> RunResult:
        """Wait for the run to end and return its result.

        Raises:
            RunInterrupted: Shutdown or a lost lease stopped the run; it stays open.
            Exception: Whatever the run raised (it stays open for a resume too).
        """
        await self._done.wait()
        if self.result is not None:
            return self.result
        assert self.error is not None
        raise self.error

    def cancel(self, reason: str = "cancelled") -> None:
        """Cancel the run, as `RunStream.cancel` does.

        The run ends with ``status == "cancelled"`` and is closed in the
        journal; it is not resumed later. A run that is still queued is
        cancelled at once, without waiting for a slot (a cancel runs no model
        request and no tool), and a shutdown that comes later still lets it
        close. To stop a worker but keep its runs open, use
        `AgentRuntime.shutdown` instead.
        """
        if self._done.is_set():
            return
        self._cancel_reason = reason
        self._wake_up.set()
        if self._stream is not None:
            self._stream.cancel(reason)

    async def events(self) -> AsyncIterator[Event]:
        """Yield the run's events as they happen, starting from the oldest one kept.

        Several subscribers may iterate at once, and a subscriber that starts
        late first receives the events already emitted. The handle keeps the
        last ``event_buffer`` events (see `AgentRuntime`); a subscriber that
        falls further behind skips the oldest ones. The run never waits for a
        subscriber.

        Raises:
            RunInterrupted: After the last event, when the run was interrupted.
            Exception: After the last event, whatever the run raised.
        """
        seen = self._published - len(self._events)
        while True:
            changed = self._changed
            finished = self._done.is_set()
            while seen < self._published:
                oldest = self._published - len(self._events)
                seen = max(seen, oldest)
                event = self._events[seen - oldest]
                seen += 1
                yield event
            if finished:
                break
            await changed.wait()
        if self.error is not None:
            raise self.error

    def __repr__(self) -> str:
        return (
            f"RunHandle(session_id={self.session_id!r}, run_id={self.run_id!r}, "
            f"status={self.status!r})"
        )

    # -- used by the runtime -----------------------------------------------------------

    def _publish(self, event: Event) -> None:
        if isinstance(event, RunStarted):
            self.run_id = event.run_id
        elif isinstance(event, TurnStarted | ModelResponded | AgentSwitched):
            self._answers = False  # the answered turn is over: every answer was used
            self._unsaved.discard("state")  # journaled right after the run started
        elif isinstance(event, StateUpdated):
            self._unsaved.discard("state")
        self._events.append(event)
        self._published += 1
        self._wake()

    def _wake(self) -> None:
        changed, self._changed = self._changed, asyncio.Event()
        changed.set()

    def _at_safe_point(self, event: Event) -> bool:
        """True when stopping right after ``event`` interrupts no model request or tool.

        Asked of the run itself, not of the events seen so far: tool calls of a
        group run ahead of the consumer, so a call may already execute while its
        ``ToolCallStarted`` still waits in the queue. Answers given to a resume
        are only in memory until the answered turn is over, so no point before
        that is safe.
        """
        if self._answers or not isinstance(event, _SAFE_AFTER):
            return False
        run = getattr(self._stream, "_run", None)
        if run is not None and getattr(run, "_running", None):
            return False  # a tool call journaled its start and has no result yet
        return self._store is None or self._store.writing == 0

    def _request_stop(self) -> None:
        self._stop_requested = True
        self._wake_up.set()

    def _finish(
        self,
        status: HandleStatus,
        *,
        result: RunResult | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.status, self.result, self.error = status, result, error
        self._done.set()
        self._wake()


OpenStream = Callable[[Session, Any, "Limits | None"], "RunStream"]


class AgentRuntime:
    """Runs agents in the background with leases, heartbeats and crash recovery.

    Args:
        store: Where sessions live. It must support run status and leases:
            `InMemoryStore` (one process), `SQLiteStore` (one host) or
            `qagents.stores.postgres.PostgresStore` (several hosts).
        agents: The agents this runtime can run, as a list or as a mapping
            from ``agent.name`` to agent. Runs are recorded by agent name, so
            names must be unique, and a worker can only recover runs of
            agents it knows. Agents reached through handoffs need not be
            listed: the run is resumed on the agent that started it.
        worker_id: This worker's diagnostic name. Defaults to host name,
            process id and a random suffix. Each acquisition appends a fresh
            token, so runtimes that reuse a name still have distinct leases.
        lease_ttl_s: Seconds a lease lasts without a heartbeat. A dead
            worker's runs are recovered after about this long.
        heartbeat_s: Seconds between lease renewals. Defaults to a third of
            ``lease_ttl_s``.
        max_concurrent: Runs this runtime drives at once. More are queued,
            holding their lease while they wait.
        deps_factory: Builds ``deps`` for a session id when a run is started
            or resumed without ``deps``, and always for recovered runs. It may
            be async.
        max_failures: `recover` stops resuming a failed run once it has failed
            this many times (`SessionStatus.failures`, counted in the journal
            and shared by all workers). ``0`` leaves every failed run to a
            person; ``None`` retries without a limit. A run whose last
            failure was an ambiguous model error is never resumed
            automatically (see the module documentation).
        max_recoveries: `recover` stops resuming a run whose worker died once
            it was recovered this many times in a row without progress (no
            new message or tool result in between), and marks it failed for
            a person. ``None`` recovers without a limit.
        session_prefix: When set, `recover` considers only sessions whose id
            starts with it, so several applications can share one store.
        event_buffer: Events each `RunHandle` keeps for late subscribers.

    Raises:
        ConfigurationError: An argument is invalid, or two agents share a name.
    """

    def __init__(
        self,
        store: SessionStore,
        agents: Mapping[str, Agent] | Sequence[Agent],
        *,
        worker_id: str | None = None,
        lease_ttl_s: float = 30.0,
        heartbeat_s: float | None = None,
        max_concurrent: int = 8,
        deps_factory: DepsFactory | None = None,
        max_failures: int | None = 3,
        max_recoveries: int | None = 5,
        session_prefix: str | None = None,
        event_buffer: int = 4096,
    ) -> None:
        heartbeat = lease_ttl_s / 3 if heartbeat_s is None else heartbeat_s
        if not 0 < lease_ttl_s < float("inf"):
            raise ConfigurationError(f"lease_ttl_s must be a positive number, not {lease_ttl_s}.")
        if not 0 < heartbeat < lease_ttl_s:
            raise ConfigurationError(
                f"heartbeat_s={heartbeat} must be above 0 and below lease_ttl_s={lease_ttl_s}, "
                "or the lease expires between renewals. A third of the TTL is a good value."
            )
        if max_concurrent < 1:
            raise ConfigurationError(f"max_concurrent must be at least 1, not {max_concurrent}.")
        if max_failures is not None and max_failures < 0:
            raise ConfigurationError(
                f"max_failures must be None or at least 0, not {max_failures}."
            )
        if max_recoveries is not None and max_recoveries < 1:
            raise ConfigurationError(
                f"max_recoveries must be None or at least 1, not {max_recoveries}."
            )
        if event_buffer < 1:
            raise ConfigurationError(f"event_buffer must be at least 1, not {event_buffer}.")
        self.store = store
        self.agents: dict[str, Agent] = _registry(agents)
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self.lease_ttl_s = float(lease_ttl_s)
        self.heartbeat_s = float(heartbeat)
        self.max_concurrent = max_concurrent
        self.deps_factory = deps_factory
        self.max_failures = max_failures
        self.max_recoveries = max_recoveries
        self.session_prefix = session_prefix
        self.event_buffer = event_buffer
        self._slots = asyncio.Semaphore(max_concurrent)
        self._handles: dict[str, RunHandle] = {}
        self._claiming: set[str] = set()
        self._claims_idle = asyncio.Event()
        self._claims_idle.set()
        self._closing = asyncio.Event()
        self._background: set[asyncio.Future[Any]] = set()
        self._skipped: set[tuple[str, str]] = set()
        """Sessions recover() already logged as skipped, with the reason."""
        self._held_back: dict[str, tuple[str | None, int, str]] = {}
        """Sessions recover() leaves to a person, with the (run id, failures, status) seen,
        so later polls skip them without a lease write until their journal changes."""

    def __repr__(self) -> str:
        return (
            f"AgentRuntime(worker_id={self.worker_id!r}, agents={sorted(self.agents)}, "
            f"store={type(self.store).__name__})"
        )

    async def __aenter__(self) -> AgentRuntime:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.shutdown()

    # -- runs ------------------------------------------------------------------------

    @property
    def handles(self) -> dict[str, RunHandle]:
        """The runs this runtime is driving or has queued, by session id."""
        return dict(self._handles)

    def get(self, session_id: str) -> RunHandle | None:
        """The handle of this runtime's run on ``session_id``, if it has one."""
        return self._handles.get(session_id)

    async def start(
        self,
        agent_name: str,
        prompt: Any = None,
        *,
        session_id: str | None = None,
        deps: Any = None,
        state: Mapping[str, Any] | None = None,
        limits: Limits | None = None,
    ) -> RunHandle:
        """Start a run in the background, as ``agent.stream`` would.

        Takes the session's lease and journals the run (``run_started``, the
        ``state`` values and the prompt) before returning, so a run that waits
        for a free slot is not lost when the worker stops or dies. Like
        ``agent.run``, a new run on a session whose run is paused or failed
        closes that run first. Input guardrails and the toolkits' loading run
        here, outside ``max_concurrent``.

        Args:
            agent_name: Name of a registered agent.
            prompt: The new user message. May be left out when the session
                already ends with one.
            session_id: The session to run in. A new id is made when left out.
            deps: Passed to tools and hooks as ``ctx.deps``. When left out,
                ``deps_factory`` builds it.
            state: Values merged into the session state when the run starts.
            limits: Replaces the agent's limits for this run. They are
                journaled, so the run keeps them when another worker recovers it.

        Raises:
            LeaseError: Another worker is driving this session, or this
                runtime already runs it.
            ConfigurationError: No agent has that name.
            RuntimeError: The runtime is shut down.
            Exception: What the run raised before it was journaled (nothing
                to run, an input guardrail that failed, a session in use).
        """
        agent = self._agent(agent_name)
        session_id = session_id or uuid.uuid4().hex

        def open_stream(session: Session, deps: Any, limits: Limits | None) -> RunStream:
            return agent.stream(prompt, deps=deps, session=session, state=state, limits=limits)

        return await self._begin(session_id, agent, open_stream, deps, resumed=False, limits=limits)

    async def resume(
        self,
        session_id: str,
        *,
        approvals: Mapping[str, Approval | bool] | None = None,
        results: Mapping[str, Any] | None = None,
        deps: Any = None,
        state: Mapping[str, Any] | None = None,
        limits: Limits | None = None,
    ) -> RunHandle:
        """Continue the session's unfinished run in the background, as ``agent.resume`` would.

        The run is resumed on the agent that started it, as recorded in the
        journal. Use it to answer a paused run, or to resume an interrupted
        or failed one right away: a person may resume a run that `recover`
        leaves alone (an ambiguous failure, a crash loop).

        Args:
            session_id: The session whose run is unfinished.
            approvals: Answers for ``result.pending_approvals``.
            results: Results for ``result.pending_inputs``.
            deps: Passed to tools and hooks; ``deps_factory`` builds it when left out.
            state: Values merged into the session state when the run resumes.
            limits: Replaces the run's limits from now on (journaled). When
                left out, the limits the run was started with through this
                runtime apply, else the agent's.

        Raises:
            SessionError: The session has no unfinished run.
            LeaseError: Another worker is driving this session.
            ConfigurationError: The run's agent is not registered here.
            RuntimeError: The runtime is shut down.
        """
        status = await self.store.status(session_id)
        if status.open_run_id is None:
            raise SessionError(
                f"Session {session_id!r} has no unfinished run to resume. "
                "Start a new one with runtime.start()."
            )
        agent = self._agent(status.agent)

        def open_stream(session: Session, deps: Any, limits: Limits | None) -> RunStream:
            return agent.resume_stream(
                session, approvals=approvals, results=results, deps=deps, state=state,
                limits=limits,
            )  # fmt: skip

        return await self._begin(
            session_id, agent, open_stream, deps, resumed=True, limits=limits,
            answers=bool(approvals or results),
            unsaved=[name for name, given in (("state", state), ("limits", limits)) if given],
        )  # fmt: skip

    # -- recovery --------------------------------------------------------------------

    async def recover(self, *, limit: int | None = None) -> list[RunHandle]:
        """Resume the runs whose worker died or that failed, and return their handles.

        Looks for sessions whose run is ``"running"`` or ``"failed"`` and whose
        lease is free or expired, takes the lease and resumes the run with
        the agent named in the journal (``deps`` from ``deps_factory``, the
        limits journaled for the run). Sessions are skipped, with a log
        message, when the agent is not registered here, when a failed run has
        reached ``max_failures`` or failed ambiguously, or when a run is in a
        crash loop (``max_recoveries``). Paused runs are never resumed. When
        two workers recover at once, the lease decides: each run is taken by
        exactly one of them.

        A session whose listed lease is held by another worker and has not
        expired is passed over without a write. The check uses this process's
        clock; with `PostgresStore` (the database's clock) a skew between the
        two only delays the takeover by that much.

        Args:
            limit: Resume at most this many runs. Defaults to the free slots
                of ``max_concurrent``, so a worker does not take more runs
                than it can drive while others are idle.
        """
        if self._closing.is_set():
            return []
        room = self.max_concurrent - len(self._handles) if limit is None else limit
        if room <= 0:
            return []
        candidates = await self.store.list_sessions(status=_RECOVERABLE, prefix=self.session_prefix)
        listed = {candidate.session_id for candidate in candidates}
        self._skipped = {key for key in self._skipped if key[0] in listed}
        self._held_back = {key: seen for key, seen in self._held_back.items() if key in listed}
        handles: list[RunHandle] = []
        for candidate in candidates:
            if len(handles) >= room or self._closing.is_set():
                break
            session_id = candidate.session_id
            if session_id in self._handles or session_id in self._claiming:
                continue
            if candidate.lease_active():
                continue  # another worker drives it; acquire_lease would only refuse
            if self._held_back.get(session_id) == _seen(candidate):
                continue  # left to a person, and nothing changed since
            if self._skip_reason(candidate) is not None:
                continue
            handle = await self._recover_one(session_id)
            if handle is not None:
                handles.append(handle)
        return handles

    async def run_forever(self, poll_interval_s: float = 5.0) -> None:
        """Call `recover` every ``poll_interval_s`` seconds until `shutdown` is called.

        An error while recovering is logged and retried at the next poll.
        """
        if not poll_interval_s > 0:
            raise ValueError(f"poll_interval_s must be above 0, not {poll_interval_s}.")
        while not self._closing.is_set():
            try:
                await self.recover()
            except Exception:
                logger.exception("Worker %s could not recover runs; retrying", self.worker_id)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._closing.wait(), poll_interval_s)

    async def shutdown(self, timeout_s: float = 10.0) -> None:
        """Stop this worker and keep its runs open and resumable.

        No new runs start: a `start`, `resume` or `recover` still taking its
        lease when shutdown begins either backs out (releasing the lease) or
        hands its run to the shutdown like the others. Each run is asked to
        stop at its next safe point: before a model request, after a model
        response, or after a tool call when no tool of the run is executing.
        A run resumed with approvals or results first finishes the turn they
        answer. Runs still going after ``timeout_s`` seconds are cancelled,
        which interrupts their model request or tool call (see the delivery
        guarantees in the module documentation). A run that `RunHandle.cancel`
        was asked to cancel is closed as cancelled, not left open. Their
        handles end with status ``"interrupted"``.

        Leases are released before this returns, so another worker can resume
        the runs at once, with one exception: a `start`, `resume` or `recover`
        still preparing when ``timeout_s`` runs out (a slow ``deps_factory``,
        say) is not waited for any longer. It backs out or stops at once when
        its preparation ends, and releases its lease then; until then its
        heartbeat keeps the lease.
        """
        self._closing.set()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(timeout_s, 0.0)
        with contextlib.suppress(TimeoutError):  # claims in flight launch or back out
            await asyncio.wait_for(self._claims_idle.wait(), max(deadline - loop.time(), 0.0))
        handles = list(self._handles.values())
        for handle in handles:
            handle._request_stop()
        tasks = [handle._task for handle in handles if handle._task is not None]
        if tasks:
            _, late = await asyncio.wait(tasks, timeout=max(deadline - loop.time(), 0.0))
            for task in late:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        # Lease releases (and stream closes) that a cancel left to run on their own.
        while pending := [future for future in self._background if not future.done()]:
            await asyncio.gather(*pending, return_exceptions=True)

    # -- internals -------------------------------------------------------------------

    def _agent(self, name: str | None) -> Agent:
        agent = self.agents.get(name) if name is not None else None
        if agent is None:
            raise ConfigurationError(
                f"No agent named {name!r} is registered in this runtime. "
                f"Registered: {sorted(self.agents)}. Pass it in AgentRuntime(agents=...)."
            )
        return agent

    def _shut_down(self) -> RuntimeError:
        return RuntimeError(
            f"AgentRuntime {self.worker_id!r} is shut down. Create a new runtime to run agents."
        )

    def _skip_reason(self, candidate: SessionStatus) -> str | None:
        """Why ``recover`` leaves this session alone, or None. Logs each reason once."""
        if candidate.session_id.startswith(WORKFLOW_PREFIX):
            return "workflow"  # resumed by its workflow; not worth a log line
        if candidate.open_run_id is None:
            return "closed"
        reason = self._failures_reason(candidate)
        if reason is None and candidate.agent not in self.agents:
            reason = (
                f"its run belongs to agent {candidate.agent!r}, which this worker does not have "
                f"(registered: {sorted(self.agents)})"
            )
        if reason is not None:
            self._log_skip(candidate.session_id, reason)
        return reason

    def _failures_reason(self, status: SessionStatus) -> str | None:
        """Why a failed run has failed too often to be resumed automatically, or None."""
        if status.status != "failed" or self.max_failures is None:
            return None
        if status.failures < self.max_failures:
            return None
        return (
            f"its run failed {status.failures} times (max_failures={self.max_failures}); "
            "resume it with runtime.resume() once the cause is fixed"
        )

    def _log_skip(self, session_id: str, reason: str) -> None:
        key = (session_id, reason)
        if key not in self._skipped:
            self._skipped.add(key)
            logger.warning(
                "Worker %s does not recover session %r: %s.", self.worker_id, session_id, reason
            )

    def _held_reason(
        self, current: SessionStatus, entries: list[Entry], limits: Limits
    ) -> tuple[str, bool] | None:
        """Why the open run must wait for a person, and whether it is a crash loop to record."""
        run_id = current.open_run_id
        own = [entry for entry in entries if entry.run_id == run_id]
        if current.status == "failed":
            failure = next((entry for entry in reversed(own) if entry.kind == "run_failed"), None)
            data = failure.data if failure is not None else {}
            if data.get("needs_person"):
                return f"{data.get('message') or 'it needs a person'}", False
            if "ambiguous" not in data:
                return (
                    "its failure was journaled without saying whether its last model request "
                    "may have been billed (an older qagents wrote it); resume it with "
                    "runtime.resume() if sending it again is safe"
                ), False
            if data["ambiguous"] and not limits.retry_ambiguous:
                return (
                    f"its last model request failed ambiguously ({data.get('error_type')}): the "
                    "provider may have processed and billed it, so it is not sent again "
                    "automatically; resume it with runtime.resume() once that is safe, or "
                    "start it with Limits(retry_ambiguous=True)"
                ), False
        if self.max_recoveries is not None:
            attempts = 0
            for entry in own:
                if entry.kind == RECOVERED_KIND:
                    attempts += 1
                elif entry.kind in _PROGRESS:
                    attempts = 0
            if attempts >= self.max_recoveries:
                return (
                    f"it was recovered {attempts} times in a row without progress "
                    f"(max_recoveries={self.max_recoveries}): its worker probably dies on it "
                    "every time; find the cause, then resume it with runtime.resume()"
                ), True
        return None

    async def _recover_one(self, session_id: str) -> RunHandle | None:
        with self._claim(session_id):
            owner = _owner_token(self.worker_id)
            deadline = asyncio.get_running_loop().time() + self.lease_ttl_s
            try:
                if not await self.store.acquire_lease(session_id, owner, self.lease_ttl_s):
                    return None  # another worker is driving it, or won the race
            except Exception:
                logger.warning(
                    "Worker %s could not take the lease of session %r; it tries again later",
                    self.worker_id,
                    session_id,
                    exc_info=True,
                )
                return None
            handle: RunHandle | None = None
            try:
                # The run may have ended between the listing and the lease.
                current = await self.store.status(session_id)
                if current.status not in _RECOVERABLE or self._skip_reason(current) is not None:
                    await self._release(session_id, owner)
                    return None
                agent = self.agents[current.agent or ""]
                handle = self._new_handle(
                    session_id, agent, owner=owner, deadline=deadline, resumed=True
                )
                entries = await self.store.load(session_id)
                limits = _journaled_limits(entries, current.open_run_id)
                held = self._held_reason(current, entries, limits or agent.limits)
                if held is not None:
                    reason, crash_loop = held
                    if crash_loop:
                        await self.store.append_leased(
                            session_id,
                            [_crash_loop_entry(current.open_run_id, reason)],
                            owner,
                        )
                        current = await self.store.status(session_id)
                    self._held_back[session_id] = _seen(current)
                    self._log_skip(session_id, reason)
                    await self._back_out(handle)
                    return None
                deps = await self._deps(session_id, None)
            except Exception:
                logger.exception(
                    "Worker %s could not prepare session %r for recovery",
                    self.worker_id,
                    session_id,
                )
                if handle is None:
                    await self._release(session_id, owner)
                else:
                    await self._back_out(handle)
                return None
            except BaseException:
                if handle is not None and handle._heartbeat is not None:
                    handle._heartbeat.cancel()  # the lease expires; never renew it forever
                raise
            # No await from here to the launch, so a shutdown cannot slip in between.
            if self._closing.is_set() or handle._lease_lost:
                await self._back_out(handle)
                return None
            logger.info(
                "Worker %s recovers session %r (run %s, status %s, agent %r)",
                self.worker_id,
                session_id,
                current.open_run_id,
                current.status,
                agent.name,
            )

            def open_stream(session: Session, deps: Any, limits: Limits | None) -> RunStream:
                return agent.resume_stream(session, deps=deps, limits=limits)

            recovered = {"worker": self.worker_id, "status": current.status}
            return self._launch(
                handle, open_stream, deps, limits=limits, journal=None, recovered=recovered
            )

    async def _begin(
        self,
        session_id: str,
        agent: Agent,
        open_stream: OpenStream,
        deps: Any,
        *,
        resumed: bool,
        limits: Limits | None,
        answers: bool = False,
        unsaved: Iterable[str] = (),
    ) -> RunHandle:
        if self._closing.is_set():
            raise self._shut_down()
        if session_id in self._handles or session_id in self._claiming:
            raise LeaseError(
                f"Session {session_id!r} already has a run in this runtime. Wait for "
                f"runtime.get({session_id!r}) to finish before starting or resuming another."
            )
        with self._claim(session_id):
            owner = _owner_token(self.worker_id)
            deadline = asyncio.get_running_loop().time() + self.lease_ttl_s
            if not await self.store.acquire_lease(session_id, owner, self.lease_ttl_s):
                holder = await self.store.status(session_id)
                raise LeaseError(
                    f"Session {session_id!r} is being driven by worker {holder.lease_owner!r}. "
                    "Wait for its run to stop; if that worker died, recover() takes the run "
                    "over once the lease expires."
                )
            # The heartbeat runs from now on: preparing the run (deps_factory, input
            # guardrails, toolkits) may take longer than the lease lasts.
            handle = self._new_handle(
                session_id,
                agent,
                owner=owner,
                deadline=deadline,
                resumed=resumed,
                answers=answers,
                unsaved=unsaved,
            )
            prepared: RunStream | None = None
            journal = limits  # limits given here are journaled for a recovery
            try:
                deps = await self._deps(session_id, deps)
                if resumed and limits is None:
                    status = await self.store.status(session_id)
                    entries = await self.store.load(session_id)
                    limits = _journaled_limits(entries, status.open_run_id)
                if self._closing.is_set():
                    raise self._shut_down()
                if not resumed:
                    # Journal the run now, under the lease: a run that waits for a slot is
                    # then open in the journal, so a shutdown or a crash does not lose it.
                    store = _LeasedStore(self.store, handle._lease_owner)
                    prepared = open_stream(Session(store, session_id), deps, limits)
                    if journal is not None:  # before the run starts: it governs the whole run
                        await store.append(session_id, [_limits_entry(prepared.run_id, journal)])
                        journal = None
                    first = await prepared.__anext__()
                if handle._lease_lost:
                    raise LeaseError(
                        f"Worker {self.worker_id!r} lost the lease of session {session_id!r} "
                        "while it prepared the run: another worker took the session over."
                    )
            except BaseException:
                if prepared is not None:
                    with contextlib.suppress(Exception):
                        await prepared.aclose()
                await self._back_out(handle)
                raise
            self._launch(handle, open_stream, deps, limits=limits, journal=journal)
            if prepared is not None:
                handle._prepared, handle._store = prepared, store
                handle._publish(first)
            if self._closing.is_set():  # shut down while the run was journaled: it stays open
                handle._request_stop()
            return handle

    @contextlib.contextmanager
    def _claim(self, session_id: str) -> Iterator[None]:
        """Mark ``session_id`` as being taken; `shutdown` waits for claims in flight."""
        self._claiming.add(session_id)
        self._claims_idle.clear()
        try:
            yield
        finally:
            self._claiming.discard(session_id)
            if not self._claiming:
                self._claims_idle.set()

    def _new_handle(
        self,
        session_id: str,
        agent: Agent,
        *,
        owner: str,
        deadline: float,
        resumed: bool,
        answers: bool = False,
        unsaved: Iterable[str] = (),
    ) -> RunHandle:
        """A handle for a session whose lease was just taken; its heartbeat starts now."""
        handle = RunHandle(
            self, session_id, agent.name, resumed=resumed, buffer=self.event_buffer,
            answers=answers, unsaved=unsaved,
        )  # fmt: skip
        handle._lease_owner = owner
        handle._lease_deadline = deadline
        handle._heartbeat = asyncio.create_task(
            self._heartbeat(handle), name=f"qagents-lease:{session_id}"
        )
        return handle

    async def _back_out(self, handle: RunHandle) -> None:
        """Give up a handle that will not run: stop its heartbeat and release the lease."""
        heartbeat = handle._heartbeat
        if heartbeat is not None:
            heartbeat.cancel()
            await asyncio.wait([heartbeat])  # a renewal in flight ends before the release
        if not handle._lease_lost:
            await self._release(handle.session_id, handle._lease_owner)

    def _launch(
        self,
        handle: RunHandle,
        open_stream: OpenStream,
        deps: Any,
        *,
        limits: Limits | None,
        journal: Limits | None,
        recovered: Mapping[str, Any] | None = None,
    ) -> RunHandle:
        self._handles[handle.session_id] = handle
        handle._preparing_task = None
        task = asyncio.create_task(
            self._drive(handle, open_stream, deps, limits, journal, recovered),
            name=f"qagents-run:{handle.session_id}",
        )
        task.add_done_callback(lambda _: self._settle(handle))
        handle._task = task
        return handle

    def _settle(self, handle: RunHandle) -> None:
        """Finish a handle whose task was cancelled before it ever ran (so no finally ran)."""
        if handle.done():
            return
        if handle._heartbeat is not None:
            handle._heartbeat.cancel()
        if self._handles.get(handle.session_id) is handle:
            del self._handles[handle.session_id]
        self._keep(asyncio.ensure_future(self._abandon(handle)))

    async def _abandon(self, handle: RunHandle) -> None:
        note: RunInterrupted | None = None
        try:
            if handle._prepared is not None:
                with contextlib.suppress(Exception):
                    await handle._prepared.aclose()
            note = await self._interrupted(handle, "shutdown")
            await self._back_out(handle)
        finally:
            handle._finish("interrupted", error=note or _interrupted_text(handle, "shutdown"))

    def _keep(self, future: asyncio.Future[Any]) -> None:
        """Keep a background task alive until it is done; `shutdown` waits for it."""
        self._background.add(future)
        future.add_done_callback(self._background.discard)

    async def _slot(self, handle: RunHandle) -> bool:
        """Wait for a free slot; a cancel or a stop request ends the wait. True: one is held."""
        if handle._cancel_reason is not None or handle._stop_requested:
            return False
        acquire = asyncio.ensure_future(self._slots.acquire())
        woken = asyncio.ensure_future(handle._wake_up.wait())
        try:
            await asyncio.wait({acquire, woken}, return_when=asyncio.FIRST_COMPLETED)
        except BaseException:
            acquire.cancel()
            if acquire.done() and not acquire.cancelled():
                self._slots.release()
            raise
        finally:
            woken.cancel()
        if acquire.done():
            return True
        acquire.cancel()
        return False

    async def _drive(
        self,
        handle: RunHandle,
        open_stream: OpenStream,
        deps: Any,
        limits: Limits | None,
        journal: Limits | None,
        recovered: Mapping[str, Any] | None,
    ) -> None:
        session_id = handle.session_id
        heartbeat = handle._heartbeat or asyncio.create_task(self._heartbeat(handle))
        status: HandleStatus = "interrupted"
        result: RunResult | None = None
        error: BaseException | None = None
        why: str | None = "shutdown"
        slot = False
        stream: RunStream | None = handle._prepared
        try:
            slot = await self._slot(handle)
            # A cancel is honoured even during shutdown: it closes the run, it runs nothing.
            if handle._stop_requested and handle._cancel_reason is None:
                return  # shut down while queued: nothing more has run
            handle.status = "running"
            if stream is None:
                handle._store = _LeasedStore(self.store, handle._lease_owner)
                stream = open_stream(Session(handle._store, session_id), deps, limits)
            handle._stream = stream
            if handle._cancel_reason is not None:
                stream.cancel(handle._cancel_reason)
            async with stream:
                async for event in stream:
                    if isinstance(event, RunStarted):
                        await self._journal_start(handle, event, journal, recovered)
                    elif isinstance(event, RunFinished):
                        # Callers continue with result.session: give them the plain store (a
                        # plain run on it takes a lease of its own, so recover() leaves it alone).
                        event.result._session = Session(self.store, session_id)
                    handle._publish(event)
                    if (
                        handle._stop_requested
                        and handle._cancel_reason is None
                        and handle._at_safe_point(event)
                    ):
                        break
            if stream.result is not None:
                status, result, why = "finished", stream.result, None
        except asyncio.CancelledError:
            why = "the worker lost its lease" if handle._lease_lost else "shutdown"
            raise
        except Exception as raised:
            if handle._lease_lost or isinstance(raised, LeaseError):
                handle._lease_lost = True
                why, error = "the worker lost its lease", raised
            else:
                logger.warning(
                    "Run of session %r raised %s; it stays open for a resume",
                    session_id,
                    type(raised).__name__,
                )
                status, error, why = "failed", raised, None
        finally:
            heartbeat.cancel()
            if slot:
                self._slots.release()
            # The outcome is known: record it on the handle even if a cancel (shutdown's, after
            # its timeout) lands while the lease is released. The release goes on regardless.
            wind_down = asyncio.ensure_future(self._wind_down(handle, heartbeat, stream, why))
            self._keep(wind_down)
            cancelled = False
            try:
                note = await asyncio.shield(wind_down)
            except asyncio.CancelledError:
                cancelled = True
                note = _interrupted_text(handle, why) if why is not None else None
            if note is not None:
                note.__cause__ = error if error is not None else None
                error = note
            if self._handles.get(session_id) is handle:
                del self._handles[session_id]
            # Last, so a caller woken by wait() finds the lease released and the slot free.
            handle._finish(status, result=result, error=error)
            if cancelled:
                raise asyncio.CancelledError

    async def _journal_start(
        self,
        handle: RunHandle,
        event: RunStarted,
        journal: Limits | None,
        recovered: Mapping[str, Any] | None,
    ) -> None:
        """Journal what the runtime adds to a run that has just started or resumed."""
        entries: list[Entry] = []
        if journal is not None:
            entries.append(_limits_entry(event.run_id, journal))
        # A recovery counts towards max_recoveries only once the run has resumed, and not
        # when a shutdown already stops it: it then stops before its first model request.
        if recovered is not None and not handle._stop_requested:
            entries.append(Entry(kind=RECOVERED_KIND, run_id=event.run_id, data=dict(recovered)))
        if entries:
            assert handle._store is not None
            await handle._store.append(handle.session_id, entries)
        handle._unsaved.discard("limits")

    async def _wind_down(
        self,
        handle: RunHandle,
        heartbeat: asyncio.Task[None],
        stream: RunStream | None,
        why: str | None,
    ) -> RunInterrupted | None:
        """Close the run, say how it was interrupted (``why``), and release the lease."""
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat
        if stream is not None:  # a no-op when `async with stream` closed it already
            with contextlib.suppress(Exception):
                await stream.aclose()
        note = await self._interrupted(handle, why) if why is not None else None
        if not handle._lease_lost:
            await self._release(handle.session_id, handle._lease_owner)
        return note

    async def _heartbeat(self, handle: RunHandle) -> None:
        session_id = handle.session_id
        loop = asyncio.get_running_loop()
        deadline = handle._lease_deadline
        while True:
            await asyncio.sleep(min(self.heartbeat_s, max(0, deadline - loop.time())))
            if loop.time() >= deadline:
                self._lose_lease(handle)
                return
            began = loop.time()
            try:
                held = await asyncio.wait_for(
                    self.store.renew_lease(session_id, handle._lease_owner, self.lease_ttl_s),
                    timeout=min(self.heartbeat_s, deadline - began),
                )
            except Exception as failure:
                logger.warning(
                    "Worker %s could not renew its lease on session %r (%s); retrying",
                    self.worker_id,
                    session_id,
                    type(failure).__name__,
                )
                logger.debug("The renewal failed with", exc_info=True)
                continue  # failures never extend the last confirmed local deadline
            if not held or loop.time() >= deadline:
                self._lose_lease(handle)
                return
            deadline = began + self.lease_ttl_s

    def _lose_lease(self, handle: RunHandle) -> None:
        logger.error(
            "Worker %s lost or could not confirm its lease on session %r; stopping its run",
            self.worker_id,
            handle.session_id,
        )
        handle._lease_lost = True
        task = handle._task or handle._preparing_task
        if task is not None:
            task.cancel()

    async def _release(self, session_id: str, owner: str) -> None:
        """Release the lease; on failure it simply expires."""
        try:
            await asyncio.wait_for(
                self.store.release_lease(session_id, owner),
                timeout=min(self.lease_ttl_s, 10.0),
            )
        except Exception as failure:
            logger.warning(
                "Worker %s could not release its lease on session %r (%s); it expires in %g s",
                self.worker_id,
                session_id,
                type(failure).__name__,
                self.lease_ttl_s,
            )
            logger.debug("The release failed with", exc_info=True)

    async def _deps(self, session_id: str, deps: Any) -> Any:
        if deps is not None or self.deps_factory is None:
            return deps
        value = self.deps_factory(session_id)
        if inspect.isawaitable(value):
            value = await value
        return value

    async def _interrupted(self, handle: RunHandle, why: str) -> RunInterrupted:
        """The error of an interrupted handle, worded after what the journal holds now."""
        timeout = min(self.lease_ttl_s, 10.0)
        try:
            stored = await asyncio.wait_for(self.store.status(handle.session_id), timeout)
        except (Exception, asyncio.CancelledError):
            return _interrupted_text(handle, why)
        held: str | None = None
        if stored.status == "failed":
            try:
                entries = await asyncio.wait_for(self.store.load(handle.session_id), timeout)
            except (Exception, asyncio.CancelledError):
                entries = None
            held = self._failures_reason(stored)
            agent = self.agents.get(stored.agent or "")
            if held is None and entries is not None and agent is not None:
                limits = _journaled_limits(entries, stored.open_run_id) or agent.limits
                reason = self._held_reason(stored, entries, limits)
                held = reason[0] if reason is not None else None
        return _interrupted_text(handle, why, stored, held)


def _interrupted_text(
    handle: RunHandle, why: str, stored: SessionStatus | None = None, held: str | None = None
) -> RunInterrupted:
    """``stored``: the session's status after the stop, when it could be read; ``held``:
    why `AgentRuntime.recover` leaves the run to a person, if it does."""
    session_id = handle.session_id
    lead = f"The run of session {session_id!r} was interrupted by {why} before it finished."
    if stored is not None and stored.open_run_id is None:
        return RunInterrupted(f"{lead} The journal has no open run left, so nothing continues it.")
    if handle._unsaved:
        given = " and ".join(f"{name}=" for name in sorted(handle._unsaved))
        verb, them = ("were", "them") if len(handle._unsaved) > 1 else ("was", "it")
        lead += f" The {given} given to resume() {verb} not journaled: pass {them} again."
    if stored is not None and stored.status == "paused":
        unused = " It had not used the approvals or results given to resume()." * handle._answers
        return RunInterrupted(
            f"{lead}{unused} It is still paused: call runtime.resume({session_id!r}, "
            "approvals=..., results=...) with its answers. recover() never resumes a paused run."
        )
    if held is not None:
        text = f"{lead} It stays open in the journal, but recover() does not resume it: {held}."
    else:
        text = (
            f"{lead} It stays open in the journal: runtime.resume({session_id!r}) continues it, "
            "and so does recover() on any worker."
        )
    if handle._answers:
        text += (
            " Approvals or results given to resume() that it had not used yet are not kept: "
            "the run asks for them again."
        )
    return RunInterrupted(text)


def _seen(status: SessionStatus) -> tuple[str | None, int, str]:
    return status.open_run_id, status.failures, status.status


def _limits_entry(run_id: str, limits: Limits) -> Entry:
    return Entry(kind=LIMITS_KIND, run_id=run_id, data={"limits": limits.model_dump(mode="json")})


def _journaled_limits(entries: list[Entry], run_id: str | None) -> Limits | None:
    """The limits last journaled for ``run_id`` (by a runtime, or in its ``run_started``)."""
    from .agent import Limits

    if run_id is None:
        return None
    for entry in reversed(entries):
        if entry.run_id != run_id:
            continue
        if entry.kind == LIMITS_KIND or (entry.kind == "run_started" and "limits" in entry.data):
            return Limits.model_validate(entry.data["limits"])
    return None


def _crash_loop_entry(run_id: str | None, reason: str) -> Entry:
    message = f"recover() stopped resuming the run: {reason}"
    return Entry(
        kind="run_failed",
        run_id=run_id,
        data={
            "error_type": "RecoveryLimitReached",
            "message": message,
            "retryable": False,
            "ambiguous": False,
            "error": f"RecoveryLimitReached: {message}",
            "needs_person": True,
        },
    )


class _LeasedStore(SessionStore):
    """One run's view of the store: every write is checked against this worker's lease.

    ``writing`` counts the writes in flight, so a stop never lands in the middle of one.
    """

    def __init__(self, store: SessionStore, owner: str) -> None:
        self._store = store
        self._owner = owner
        self.writing = 0

    async def append(self, session_id: str, entries: list[Entry]) -> None:
        await self.append_leased(session_id, entries, self._owner)

    async def append_leased(self, session_id: str, entries: list[Entry], owner: str) -> None:
        self.writing += 1
        try:
            await self._store.append_leased(session_id, entries, owner)
        finally:
            self.writing -= 1

    async def load(self, session_id: str) -> list[Entry]:
        return await self._store.load(session_id)

    async def delete(self, session_id: str) -> None:
        await self._store.delete(session_id)

    async def status(self, session_id: str) -> SessionStatus:
        return await self._store.status(session_id)

    async def list_sessions(
        self,
        status: str | Iterable[str] | None = None,
        prefix: str | None = None,
        limit: int | None = None,
    ) -> list[SessionStatus]:
        return await self._store.list_sessions(status, prefix, limit)

    async def acquire_lease(self, session_id: str, owner: str, ttl_s: float) -> bool:
        return await self._store.acquire_lease(session_id, owner, ttl_s)

    async def renew_lease(self, session_id: str, owner: str, ttl_s: float) -> bool:
        return await self._store.renew_lease(session_id, owner, ttl_s)

    async def release_lease(self, session_id: str, owner: str) -> None:
        await self._store.release_lease(session_id, owner)


def _registry(agents: Mapping[str, Agent] | Sequence[Agent]) -> dict[str, Agent]:
    registry: dict[str, Agent] = {}
    if isinstance(agents, Mapping):
        for key, agent in agents.items():
            if key != agent.name:
                raise ConfigurationError(
                    f"agents[{key!r}] is an agent named {agent.name!r}. Runs are recorded by "
                    "agent.name, so key each agent by its name (or pass a list)."
                )
            registry[key] = agent
    else:
        for agent in agents:
            if registry.get(agent.name, agent) is not agent:
                raise ConfigurationError(
                    f"Two different agents are named {agent.name!r}. Runs are recorded by "
                    "agent.name, so give each agent a unique name=."
                )
            registry[agent.name] = agent
    if not registry:
        raise ConfigurationError("AgentRuntime needs at least one agent.")
    return registry
