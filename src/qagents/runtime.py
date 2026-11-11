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
  the store. A heartbeat renews it every ``heartbeat_s`` seconds, and every
  journal write is checked against it in the same transaction
  (`SessionStore.append_leased`), so at most one worker writes to a run at a
  time, even after a network partition. A worker that finds its lease taken
  over stops its run.
- **Recovery.** `AgentRuntime.recover` lists the sessions whose run is
  ``"running"`` or ``"failed"``, takes each free or expired lease and resumes
  the run with the agent named in the journal. Paused runs wait for a
  person and are never resumed automatically. `run_forever` calls it every
  ``poll_interval_s`` seconds.
- **Shutdown.** `AgentRuntime.shutdown` asks each run to stop at the next
  safe point (before a model request, after a model response, or after a
  tool call when no other tool is running), cancels runs that do not reach
  one in time, and releases the leases. The runs stay open in the journal,
  so another worker resumes them at once instead of waiting for a lease to
  expire.

Delivery guarantees, stated plainly. They are the journal's guarantees; the
runtime adds liveness, not exactly-once execution:

- A tool call with a recorded result is never run again, on any worker.
- A tool call that was executing when its worker died or was cancelled is
  run again on recovery only if the tool is ``read_only`` or ``idempotent``;
  otherwise the model is told its outcome is unknown. Delivery is
  at-least-once: give tools with side effects ``ctx.operation_id`` as an
  idempotency key.
- A model request that was in flight when the worker died is sent again on
  recovery. Its first response was lost, so this costs tokens; no tool had
  acted on it.
- A run that failed (it raised, for example on a provider error) is resumed
  automatically until it has failed ``max_failures`` times. Set
  ``max_failures=0`` to leave failed runs to a person.
- Liveness depends on a worker calling `recover` (or `run_forever`). A run
  whose worker died is resumed once its lease expires, after at most
  ``lease_ttl_s`` plus ``poll_interval_s`` seconds.

Rules for the store: every run on sessions that a runtime recovers must go
through a runtime. A run started with plain ``agent.run`` holds no lease, so
a worker would take it for a dead run and resume it while it is still
running. Sessions that belong to a `qagents.workflow.Workflow` (ids starting
with ``"workflow:"``) are never recovered: resume the workflow instead.
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
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from .errors import ConfigurationError, RunInterrupted, SessionError
from .events import (
    AgentSwitched,
    Event,
    ModelResponded,
    RunFinished,
    RunStarted,
    ToolCallFinished,
    ToolCallStarted,
    TurnStarted,
)
from .session import Entry, LeaseError, Session, SessionStatus, SessionStore

if TYPE_CHECKING:
    from .agent import Agent, Limits, RunStream
    from .permissions import Approval
    from .results import RunResult

logger = logging.getLogger("qagents.runtime")

HandleStatus = Literal["queued", "running", "finished", "failed", "interrupted"]
"""Where a `RunHandle` stands.

* ``queued``: waiting for a free slot (``max_concurrent``).
* ``running``: the run is being driven.
* ``finished``: the run returned a `RunResult` (completed, paused, cancelled, ...).
* ``failed``: the run raised; the error is in ``RunHandle.error``.
* ``interrupted``: shutdown or a lost lease stopped it; the run stays open.
"""

DepsFactory = Callable[[str], Any]
"""Builds ``deps`` for a session id. May be async (return an awaitable)."""

WORKFLOW_PREFIX = "workflow:"
"""Sessions whose id starts with this belong to a workflow and are not recovered."""

_RECOVERABLE = ("running", "failed")


class RunHandle:
    """A run that an `AgentRuntime` drives in the background.

    Created by `AgentRuntime.start`, `AgentRuntime.resume` and
    `AgentRuntime.recover`; not meant to be created directly.
    """

    def __init__(
        self, runtime: AgentRuntime, session_id: str, agent: str, *, resumed: bool, buffer: int
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
        self._stream: RunStream | None = None
        self._cancel_reason: str | None = None
        self._stop_requested = False
        self._lease_lost = False
        self._tools_running: set[str] = set()

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
        cancelled as soon as it starts. To stop a worker but keep its runs
        open, use `AgentRuntime.shutdown` instead.
        """
        if self._done.is_set():
            return
        self._cancel_reason = reason
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
        if isinstance(event, ToolCallStarted):
            self._tools_running.add(event.call.id)
        elif isinstance(event, ToolCallFinished):
            self._tools_running.discard(event.call.id)
        self._events.append(event)
        self._published += 1
        self._wake()

    def _wake(self) -> None:
        changed, self._changed = self._changed, asyncio.Event()
        changed.set()

    def _at_safe_point(self, event: Event) -> bool:
        """True when stopping right after ``event`` interrupts no model request or tool."""
        if isinstance(event, (RunStarted, TurnStarted, ModelResponded, AgentSwitched)):
            return True
        return isinstance(event, ToolCallFinished) and not self._tools_running

    def _request_stop(self) -> None:
        self._stop_requested = True  # a queued run checks it when it gets a slot

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
        worker_id: This worker's name in leases. Defaults to host name,
            process id and a random suffix. Give each runtime its own id: two
            runtimes with one id would share leases.
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
            person; ``None`` retries without a limit.
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
        self.session_prefix = session_prefix
        self.event_buffer = event_buffer
        self._slots = asyncio.Semaphore(max_concurrent)
        self._handles: dict[str, RunHandle] = {}
        self._claiming: set[str] = set()
        self._closing = asyncio.Event()
        self._background: set[asyncio.Future[None]] = set()
        self._skipped: set[tuple[str, str]] = set()
        """Sessions recover() already logged as skipped, with the reason."""

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

        Takes the session's lease before returning. Like ``agent.run``, a new
        run on a session whose run is paused or failed closes that run first.

        Args:
            agent_name: Name of a registered agent.
            prompt: The new user message. May be left out when the session
                already ends with one.
            session_id: The session to run in. A new id is made when left out.
            deps: Passed to tools and hooks as ``ctx.deps``. When left out,
                ``deps_factory`` builds it.
            state: Values merged into the session state when the run starts.
            limits: Replaces the agent's limits for this run.

        Raises:
            LeaseError: Another worker is driving this session, or this
                runtime already runs it.
            ConfigurationError: No agent has that name.
        """
        agent = self._agent(agent_name)
        session_id = session_id or uuid.uuid4().hex

        def open_stream(session: Session, deps: Any) -> RunStream:
            return agent.stream(prompt, deps=deps, session=session, state=state, limits=limits)

        return await self._begin(session_id, agent, open_stream, deps, resumed=False)

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
        or failed one right away.

        Args:
            session_id: The session whose run is unfinished.
            approvals: Answers for ``result.pending_approvals``.
            results: Results for ``result.pending_inputs``.
            deps: Passed to tools and hooks; ``deps_factory`` builds it when left out.
            state: Values merged into the session state when the run resumes.
            limits: Replaces the agent's limits for the resumed run.

        Raises:
            SessionError: The session has no unfinished run.
            LeaseError: Another worker is driving this session.
            ConfigurationError: The run's agent is not registered here.
        """
        status = await self.store.status(session_id)
        if status.open_run_id is None:
            raise SessionError(
                f"Session {session_id!r} has no unfinished run to resume. "
                "Start a new one with runtime.start()."
            )
        agent = self._agent(status.agent)

        def open_stream(session: Session, deps: Any) -> RunStream:
            return agent.resume_stream(
                session, approvals=approvals, results=results, deps=deps, state=state,
                limits=limits,
            )  # fmt: skip

        return await self._begin(session_id, agent, open_stream, deps, resumed=True)

    # -- recovery --------------------------------------------------------------------

    async def recover(self, *, limit: int | None = None) -> list[RunHandle]:
        """Resume the runs whose worker died or that failed, and return their handles.

        Looks for sessions whose run is ``"running"`` or ``"failed"`` and whose
        lease is free or expired, takes the lease and resumes the run with
        the agent named in the journal (``deps`` from ``deps_factory``).
        Sessions are skipped, with a log message, when the agent is not
        registered here or when a failed run has reached ``max_failures``.
        Paused runs are never resumed. When two workers recover at once,
        the lease decides: each run is taken by exactly one of them.

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
        handles: list[RunHandle] = []
        for candidate in candidates:
            if len(handles) >= room or self._closing.is_set():
                break
            session_id = candidate.session_id
            if session_id in self._handles or session_id in self._claiming:
                continue
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

        No new runs start. Each run is asked to stop at its next safe point:
        before a model request, after a model response, or after a tool call
        when no other tool is running. Runs still going after ``timeout_s``
        seconds are cancelled, which interrupts their model request or tool
        call (see the delivery guarantees in the module documentation).
        Leases are released, so another worker can resume the runs at once.
        Their handles end with status ``"interrupted"``.
        """
        self._closing.set()
        handles = list(self._handles.values())
        for handle in handles:
            handle._request_stop()
        tasks = [handle._task for handle in handles if handle._task is not None]
        if not tasks:
            return
        _, late = await asyncio.wait(tasks, timeout=max(timeout_s, 0.0))
        for task in late:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # -- internals -------------------------------------------------------------------

    def _agent(self, name: str | None) -> Agent:
        agent = self.agents.get(name) if name is not None else None
        if agent is None:
            raise ConfigurationError(
                f"No agent named {name!r} is registered in this runtime. "
                f"Registered: {sorted(self.agents)}. Pass it in AgentRuntime(agents=...)."
            )
        return agent

    def _skip_reason(self, candidate: SessionStatus) -> str | None:
        """Why ``recover`` leaves this session alone, or None. Logs each reason once."""
        reason: str | None = None
        if candidate.session_id.startswith(WORKFLOW_PREFIX):
            return "workflow"  # resumed by its workflow; not worth a log line
        if candidate.open_run_id is None:
            return "closed"
        if (
            candidate.status == "failed"
            and self.max_failures is not None
            and candidate.failures >= self.max_failures
        ):
            reason = (
                f"its run failed {candidate.failures} times (max_failures={self.max_failures}); "
                "resume it with runtime.resume() once the cause is fixed"
            )
        elif candidate.agent not in self.agents:
            reason = (
                f"its run belongs to agent {candidate.agent!r}, which this worker does not have "
                f"(registered: {sorted(self.agents)})"
            )
        if reason is not None:
            key = (candidate.session_id, reason)
            if key not in self._skipped:
                self._skipped.add(key)
                logger.warning(
                    "Worker %s does not recover session %r: %s.",
                    self.worker_id,
                    candidate.session_id,
                    reason,
                )
        return reason

    async def _recover_one(self, session_id: str) -> RunHandle | None:
        self._claiming.add(session_id)
        try:
            if not await self.store.acquire_lease(session_id, self.worker_id, self.lease_ttl_s):
                return None  # another worker is driving it, or won the race
            try:
                # The run may have ended between the listing and the lease.
                current = await self.store.status(session_id)
                if current.status not in _RECOVERABLE or self._skip_reason(current) is not None:
                    await self._release(session_id)
                    return None
                agent = self.agents[current.agent or ""]
                deps = await self._deps(session_id, None)
            except Exception:
                logger.exception(
                    "Worker %s could not prepare session %r for recovery",
                    self.worker_id,
                    session_id,
                )
                await self._release(session_id)
                return None
            logger.info(
                "Worker %s recovers session %r (run %s, status %s, agent %r)",
                self.worker_id,
                session_id,
                current.open_run_id,
                current.status,
                agent.name,
            )

            def open_stream(session: Session, deps: Any) -> RunStream:
                return agent.resume_stream(session, deps=deps)

            return self._launch(session_id, agent, open_stream, deps, resumed=True)
        finally:
            self._claiming.discard(session_id)

    async def _begin(
        self,
        session_id: str,
        agent: Agent,
        open_stream: Callable[[Session, Any], RunStream],
        deps: Any,
        *,
        resumed: bool,
    ) -> RunHandle:
        if self._closing.is_set():
            raise RuntimeError(
                f"AgentRuntime {self.worker_id!r} is shut down. Create a new runtime to run agents."
            )
        if session_id in self._handles or session_id in self._claiming:
            raise LeaseError(
                f"Session {session_id!r} already has a run in this runtime. Wait for "
                f"runtime.get({session_id!r}) to finish before starting or resuming another."
            )
        self._claiming.add(session_id)
        try:
            if not await self.store.acquire_lease(session_id, self.worker_id, self.lease_ttl_s):
                holder = await self.store.status(session_id)
                raise LeaseError(
                    f"Session {session_id!r} is being driven by worker {holder.lease_owner!r}. "
                    "Wait for its run to stop; if that worker died, recover() takes the run "
                    "over once the lease expires."
                )
            try:
                deps = await self._deps(session_id, deps)
            except BaseException:
                await self._release(session_id)
                raise
            return self._launch(session_id, agent, open_stream, deps, resumed=resumed)
        finally:
            self._claiming.discard(session_id)

    def _launch(
        self,
        session_id: str,
        agent: Agent,
        open_stream: Callable[[Session, Any], RunStream],
        deps: Any,
        *,
        resumed: bool,
    ) -> RunHandle:
        handle = RunHandle(self, session_id, agent.name, resumed=resumed, buffer=self.event_buffer)
        self._handles[session_id] = handle
        task = asyncio.create_task(
            self._drive(handle, open_stream, deps), name=f"qagents-run:{session_id}"
        )
        task.add_done_callback(lambda _: self._settle(handle))
        handle._task = task
        return handle

    def _settle(self, handle: RunHandle) -> None:
        """Finish a handle whose task was cancelled before it ever ran (so no finally ran)."""
        if handle.done():
            return
        if self._handles.get(handle.session_id) is handle:
            del self._handles[handle.session_id]
        handle._finish("interrupted", error=self._interrupted(handle, "shutdown"))
        release = asyncio.ensure_future(self._release(handle.session_id))
        self._background.add(release)
        release.add_done_callback(self._background.discard)

    async def _drive(
        self, handle: RunHandle, open_stream: Callable[[Session, Any], RunStream], deps: Any
    ) -> None:
        session_id = handle.session_id
        heartbeat = asyncio.create_task(self._heartbeat(handle))
        status: HandleStatus = "interrupted"
        result: RunResult | None = None
        error: BaseException | None = None
        try:
            async with self._slots:
                if handle._stop_requested:  # shut down while queued: nothing has run
                    error = self._interrupted(handle, "shutdown")
                    return
                handle.status = "running"
                stream = open_stream(
                    Session(_LeasedStore(self.store, self.worker_id), session_id), deps
                )
                handle._stream = stream
                if handle._cancel_reason is not None:
                    stream.cancel(handle._cancel_reason)
                async with stream:
                    async for event in stream:
                        if isinstance(event, RunStarted):
                            handle.run_id = event.run_id
                        elif isinstance(event, RunFinished):
                            # Callers continue with result.session: give them the plain store.
                            event.result._session = Session(self.store, session_id)
                        handle._publish(event)
                        if handle._stop_requested and handle._at_safe_point(event):
                            break
            if stream.result is not None:
                status, result = "finished", stream.result
            else:
                error = self._interrupted(handle, "shutdown")
        except asyncio.CancelledError:
            why = "the worker lost its lease" if handle._lease_lost else "shutdown"
            error = self._interrupted(handle, why)
            raise
        except Exception as raised:
            if handle._lease_lost or isinstance(raised, LeaseError):
                handle._lease_lost = True
                error = self._interrupted(handle, "the worker lost its lease")
                error.__cause__ = raised
            else:
                logger.warning(
                    "Run of session %r raised %s; it stays open for a resume",
                    session_id,
                    type(raised).__name__,
                )
                status, error = "failed", raised
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
            if not handle._lease_lost:
                await self._release(session_id)
            if self._handles.get(session_id) is handle:
                del self._handles[session_id]
            # Last, so a caller woken by wait() finds the lease released and the slot free.
            handle._finish(status, result=result, error=error)

    async def _heartbeat(self, handle: RunHandle) -> None:
        session_id = handle.session_id
        while True:
            await asyncio.sleep(self.heartbeat_s)
            try:
                held = await asyncio.wait_for(
                    self.store.renew_lease(session_id, self.worker_id, self.lease_ttl_s),
                    timeout=self.heartbeat_s,
                )
            except Exception:
                logger.warning(
                    "Worker %s could not renew its lease on session %r; retrying",
                    self.worker_id,
                    session_id,
                    exc_info=True,
                )
                continue
            if not held:
                logger.error(
                    "Worker %s lost its lease on session %r to another worker; stopping its run",
                    self.worker_id,
                    session_id,
                )
                handle._lease_lost = True
                if handle._task is not None:
                    handle._task.cancel()
                return

    async def _release(self, session_id: str) -> None:
        """Release the lease; on failure it simply expires."""
        try:
            await asyncio.wait_for(
                self.store.release_lease(session_id, self.worker_id),
                timeout=min(self.lease_ttl_s, 10.0),
            )
        except Exception:
            logger.warning(
                "Worker %s could not release its lease on session %r; it expires in %g s",
                self.worker_id,
                session_id,
                self.lease_ttl_s,
                exc_info=True,
            )

    async def _deps(self, session_id: str, deps: Any) -> Any:
        if deps is not None or self.deps_factory is None:
            return deps
        value = self.deps_factory(session_id)
        if inspect.isawaitable(value):
            value = await value
        return value

    def _interrupted(self, handle: RunHandle, why: str) -> RunInterrupted:
        return RunInterrupted(
            f"The run of session {handle.session_id!r} was interrupted by {why} before it "
            f"finished. It stays open in the journal: runtime.resume({handle.session_id!r}) "
            "continues it, and so does recover() on any worker."
        )


class _LeasedStore(SessionStore):
    """One run's view of the store: every write is checked against this worker's lease."""

    def __init__(self, store: SessionStore, owner: str) -> None:
        self._store = store
        self._owner = owner

    async def append(self, session_id: str, entries: list[Entry]) -> None:
        await self._store.append_leased(session_id, entries, self._owner)

    async def append_leased(self, session_id: str, entries: list[Entry], owner: str) -> None:
        await self._store.append_leased(session_id, entries, owner)

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
