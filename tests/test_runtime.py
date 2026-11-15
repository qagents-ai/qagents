"""AgentRuntime: background runs, leases, heartbeats, crash recovery and shutdown.

The crash-recovery and racing tests also run against SQLiteStore and, when
QAGENTS_TEST_POSTGRES_DSN is set, PostgresStore (see test_postgres_store.py).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from qagents import Agent, Limits, Message, RunContext, Session, SQLiteStore, tool
from qagents.errors import ConfigurationError, ModelTimeoutError, SessionError
from qagents.models.testing import FunctionModel, ScriptedModel, call
from qagents.runtime import LIMITS_KIND, RECOVERED_KIND, AgentRuntime, RunHandle, RunInterrupted
from qagents.session import Entry, InMemoryStore, LeaseError, SessionStatus, SessionStore

DSN = os.environ.get("QAGENTS_TEST_POSTGRES_DSN")
BACKENDS = [
    "memory",
    "sqlite",
    pytest.param(
        "postgres",
        marks=pytest.mark.skipif(not DSN, reason="set QAGENTS_TEST_POSTGRES_DSN to test Postgres"),
    ),
]
FAST = {"lease_ttl_s": 0.3, "heartbeat_s": 0.05}
STEADY = {"lease_ttl_s": 30.0}
"""A lease no loaded machine outlives by accident: a test ends it with `Backend.expire_lease`."""
CUE = {"lease_ttl_s": 30.0, "heartbeat_s": 0.01}
"""A lease that is lost only when its store refuses a renewal, noticed at the next heartbeat."""


class CrashableStore(SessionStore):
    """One worker's connection to a shared store. ``crash()`` makes it fail like a dead process."""

    def __init__(self, inner: SessionStore) -> None:
        self.inner = inner
        self.crashed = False

    def crash(self) -> None:
        self.crashed = True

    def _alive(self) -> SessionStore:
        if self.crashed:
            raise ConnectionError("this worker is dead")
        return self.inner

    async def append(self, session_id, entries):
        await self._alive().append(session_id, entries)

    async def append_leased(self, session_id, entries, owner):
        await self._alive().append_leased(session_id, entries, owner)

    async def load(self, session_id):
        return await self._alive().load(session_id)

    async def delete(self, session_id):
        await self._alive().delete(session_id)

    async def status(self, session_id):
        return await self._alive().status(session_id)

    async def list_sessions(self, status=None, prefix=None, limit=None):
        return await self._alive().list_sessions(status, prefix, limit)

    async def acquire_lease(self, session_id, owner, ttl_s):
        return await self._alive().acquire_lease(session_id, owner, ttl_s)

    async def renew_lease(self, session_id, owner, ttl_s):
        return await self._alive().renew_lease(session_id, owner, ttl_s)

    async def release_lease(self, session_id, owner):
        await self._alive().release_lease(session_id, owner)


def receipt_tools(log: list[str], gate: asyncio.Event, started: asyncio.Event):
    """Charging finishes at once; sending the receipt blocks until ``gate`` is set."""

    @tool
    def charge_card(amount: int) -> str:
        """Charge the customer's card."""
        log.append(f"charge {amount}")
        return "charged"

    @tool(idempotent=True)
    async def send_receipt(ctx: RunContext[Any], to: str) -> str:
        """Email the receipt."""
        log.append(f"receipt {to} via {ctx.deps}")
        started.set()
        await gate.wait()
        return "sent"

    return [charge_card, send_receipt]


def receipt_agent(log, gate, started, *, replies=None) -> Agent:
    script = [[call("charge_card", amount=40), call("send_receipt", to="m@x.gr")], "All done."]
    return Agent(
        ScriptedModel(script if replies is None else replies),
        name="billing",
        tools=receipt_tools(log, gate, started),
    )


async def wait_until(condition: Callable[[], Awaitable[bool]], timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not await condition():  # noqa: ASYNC110 - polls the store, which has no event
            await asyncio.sleep(0.02)

    await asyncio.wait_for(poll(), timeout)


# -- background runs and handles -------------------------------------------------------


async def test_start_runs_in_the_background_and_releases_the_lease():
    store = InMemoryStore()
    runtime = AgentRuntime(store, [Agent(ScriptedModel(["Hello!"]), name="greeter")])
    handle = await runtime.start("greeter", "Hi", session_id="s1")
    assert isinstance(handle, RunHandle) and handle.session_id == "s1"
    assert runtime.get("s1") is handle
    lease = await store.status("s1")
    assert lease.lease_owner.startswith(runtime.worker_id + "#") and lease.lease_active()

    result = await handle.wait()
    assert result.status == "completed" and result.text == "Hello!"
    assert handle.status == "finished" and handle.run_id == result.run_id and handle.done()
    status = await store.status("s1")
    assert (status.status, status.lease_owner) == ("idle", None)
    assert runtime.get("s1") is None and runtime.handles == {}

    # result.session uses the plain store, so the conversation can go on without the runtime.
    follow = await Agent(ScriptedModel(["Again!"])).run("Hi again", session=result.session)
    assert follow.status == "completed"
    await runtime.shutdown()


async def test_several_subscribers_and_late_ones_see_every_event():
    model = ScriptedModel(["one two three four"], delay=0.01)
    runtime = AgentRuntime(InMemoryStore(), [Agent(model, name="a")])
    handle = await runtime.start("a", "count")

    async def collect() -> list[str]:
        return [event.type async for event in handle.events()]

    early = asyncio.create_task(collect())
    await asyncio.sleep(0.02)
    middle = asyncio.create_task(collect())
    await handle.wait()
    late = await collect()  # the run is over: replayed from the buffer
    assert (await early) == (await middle) == late
    assert late[0] == "run_started" and late[-1] == "run_finished"
    assert late.count("text_delta") == 4
    await runtime.shutdown()


async def test_a_small_event_buffer_keeps_only_the_newest_events():
    runtime = AgentRuntime(
        InMemoryStore(), [Agent(ScriptedModel(["a b c d e f"]), name="a")], event_buffer=3
    )
    handle = await runtime.start("a", "go")
    await handle.wait()
    assert [event.type async for event in handle.events()][-1] == "run_finished"
    assert len([event async for event in handle.events()]) == 3
    await runtime.shutdown()


async def test_deps_come_from_the_caller_or_the_factory():
    seen: list[Any] = []

    @tool(read_only=True)
    def whoami(ctx: RunContext[Any]) -> str:
        """Say who the deps belong to."""
        seen.append(ctx.deps)
        return "ok"

    def agent() -> Agent:
        return Agent(ScriptedModel([call("whoami"), "done"]), name="a", tools=[whoami])

    async def factory(session_id: str) -> str:
        return f"deps for {session_id}"

    runtime = AgentRuntime(InMemoryStore(), [agent()], deps_factory=factory)
    await (await runtime.start("a", "go", session_id="x")).wait()
    runtime.agents["a"] = agent()
    await (await runtime.start("a", "go", session_id="y", deps="explicit")).wait()
    assert seen == ["deps for x", "explicit"]
    await runtime.shutdown()


async def test_a_session_driven_elsewhere_is_refused():
    store = InMemoryStore()
    gate, started = asyncio.Event(), asyncio.Event()
    runtime = AgentRuntime(store, [receipt_agent([], gate, started)], worker_id="w1")
    await store.acquire_lease("taken", "w2", 30)
    with pytest.raises(LeaseError, match="driven by worker 'w2'"):
        await runtime.start("billing", "go", session_id="taken")

    handle = await runtime.start("billing", "go", session_id="mine")
    with pytest.raises(LeaseError, match="already has a run in this runtime"):
        await runtime.start("billing", "again", session_id="mine")
    gate.set()
    await handle.wait()
    await runtime.shutdown()


async def test_runs_beyond_max_concurrent_wait_in_the_queue():
    gate, started = asyncio.Event(), asyncio.Event()
    log: list[str] = []
    agents = {
        "billing": receipt_agent(log, gate, started),
        "greeter": Agent(ScriptedModel(["Hi"]), name="greeter"),
    }
    runtime = AgentRuntime(InMemoryStore(), agents, max_concurrent=1)
    first = await runtime.start("billing", "charge")
    await asyncio.wait_for(started.wait(), 5)
    second = await runtime.start("greeter", "hello")
    await asyncio.sleep(0.05)
    assert (first.status, second.status) == ("running", "queued")
    gate.set()
    assert (await second.wait()).text == "Hi"
    assert (await first.wait()).status == "completed"
    await runtime.shutdown()


async def test_cancel_ends_the_run_and_closes_it():
    store = InMemoryStore()
    gate, started = asyncio.Event(), asyncio.Event()
    runtime = AgentRuntime(store, [receipt_agent([], gate, started)])
    handle = await runtime.start("billing", "go", session_id="c")
    await asyncio.wait_for(started.wait(), 5)
    handle.cancel()
    result = await handle.wait()
    assert result.status == "cancelled"
    assert (await store.status("c")).status == "idle"
    assert await runtime.recover() == []
    await runtime.shutdown()


async def test_a_paused_run_is_resumed_through_the_runtime_but_never_recovered():
    store = InMemoryStore()
    refunds: list[str] = []

    @tool(requires_approval=True)
    def refund(order: str) -> str:
        """Refund an order."""
        refunds.append(order)
        return "refunded"

    agent = Agent(
        ScriptedModel([call("refund", order="A-1"), "Done."]), name="support", tools=[refund]
    )
    runtime = AgentRuntime(store, [agent])
    paused = await (await runtime.start("support", "Refund A-1", session_id="p")).wait()
    assert paused.status == "paused"
    assert (await store.status("p")).status == "paused"

    other = AgentRuntime(store, [agent], worker_id="other")
    assert await other.recover() == []  # waits for a person

    handle = await runtime.resume("p", approvals={paused.pending_approvals[0].tool_call_id: True})
    assert handle.resumed
    assert (await handle.wait()).status == "completed" and refunds == ["A-1"]
    with pytest.raises(SessionError, match="no unfinished run"):
        await runtime.resume("p")
    await runtime.shutdown()
    await other.shutdown()


# -- crash recovery ----------------------------------------------------------------------


def run(coroutine: Awaitable[None]) -> None:
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
        runner.run(coroutine)  # type: ignore[arg-type]


class Backend:
    def __init__(self, kind: str, path: str) -> None:
        self.kind = kind
        self.path = path
        self.schema = f"qagents_test_{uuid.uuid4().hex[:12]}"
        self._memory: InMemoryStore | None = None
        self._opened: list[SessionStore] = []
        self.ahead = 0.0
        """Seconds the memory and SQLite store clocks run ahead of wall time."""

    def clock(self) -> float:
        return time.time() + self.ahead

    def store(self) -> SessionStore:
        if self.kind == "memory":
            # SPEC 7.7.6: legacy recovery tests deliberately use sub-second takeover.
            self._memory = self._memory or InMemoryStore(clock=self.clock, takeover_grace_s=0.0)
            return self._memory
        if self.kind == "sqlite":
            store: SessionStore = SQLiteStore(self.path, clock=self.clock, takeover_grace_s=0.0)
        else:
            from qagents.stores.postgres import PostgresStore

            store = PostgresStore(DSN, schema=self.schema, pool_size=4)
        self._opened.append(store)
        return store

    def run(self, test: Callable[[], Awaitable[None]]) -> None:
        async def main() -> None:
            try:
                await test()
            finally:
                for store in self._opened:
                    await store.aclose()

        run(main())

    async def expire_lease(self, store: SessionStore, session_id: str) -> None:
        """End the lease a dead worker left on ``session_id`` now, instead of waiting for it.

        The memory and SQLite store clocks move past it. Postgres leases run on the
        database's clock, so there the lease's expiry moves into the past instead.
        """
        if self.kind != "postgres":
            expires = (await store.status(session_id)).lease_expires_at
            assert expires is not None
            self.ahead += max(expires - self.clock(), 0.0) + 1.0
            return
        import psycopg

        async with await psycopg.AsyncConnection.connect(DSN, autocommit=True) as connection:
            await connection.execute(
                f'UPDATE "{self.schema}".sessions'
                " SET lease_expires_at = clock_timestamp() - interval '1 hour'"
                " WHERE session_id = %s",
                (session_id,),
            )

    def drop(self) -> None:
        if self.kind != "postgres":
            return

        async def drop() -> None:
            import psycopg

            async with await psycopg.AsyncConnection.connect(DSN, autocommit=True) as connection:
                await connection.execute(f'DROP SCHEMA IF EXISTS "{self.schema}" CASCADE')

        run(drop())


@pytest.fixture(params=BACKENDS)
def backend(request: pytest.FixtureRequest, tmp_path) -> Backend:
    made = Backend(request.param, str(tmp_path / "agents.db"))
    yield made
    made.drop()


async def crash_worker(
    store: SessionStore, log: list[str], timing: dict[str, float] = FAST
) -> AgentRuntime:
    """Worker A starts a run and dies while the receipt is being sent."""
    gate, started = asyncio.Event(), asyncio.Event()
    connection = CrashableStore(store)
    worker = AgentRuntime(
        connection,
        [receipt_agent(log, gate, started)],
        worker_id="worker-a",
        deps_factory=lambda session_id: "A",
        **timing,
    )
    handle = await worker.start("billing", "charge and send the receipt", session_id="order-7")
    await asyncio.wait_for(started.wait(), 5)
    connection.crash()  # the process dies: no more writes, no release, no heartbeat
    assert handle._task is not None
    handle._task.cancel()
    with pytest.raises(RunInterrupted):
        await handle.wait()
    return worker


def recovering_worker(
    store: SessionStore, log: list[str], name: str, timing: dict[str, float] = FAST
) -> AgentRuntime:
    gate, started = asyncio.Event(), asyncio.Event()
    gate.set()
    return AgentRuntime(
        store,
        [receipt_agent(log, gate, started, replies=["The card was charged and the receipt sent."])],
        worker_id=name,
        deps_factory=lambda session_id: name,
        **timing,
    )


def test_a_dead_workers_run_is_recovered_without_repeating_finished_tools(backend):
    async def check() -> None:
        log: list[str] = []
        await crash_worker(backend.store(), log, STEADY)
        store = backend.store()
        status = await store.status("order-7")
        assert status.status == "running" and status.lease_owner.startswith("worker-a#")

        worker = recovering_worker(store, log, "worker-b", STEADY)
        assert await worker.recover() == []  # the lease has not expired yet
        await backend.expire_lease(store, "order-7")  # no wall-clock wait for it to run out
        (handle,) = await worker.recover()
        assert handle.resumed and handle.agent == "billing"
        result = await handle.wait()

        assert result.status == "completed"
        assert result.text == "The card was charged and the receipt sent."
        # charge_card finished before the crash and never ran again; send_receipt is
        # idempotent, so it ran again, with the deps that worker-b's factory built.
        assert log == ["charge 40", "receipt m@x.gr via A", "receipt m@x.gr via worker-b"]
        status = await store.status("order-7")
        assert (status.status, status.lease_owner) == ("idle", None)
        assert await worker.recover() == []
        await worker.shutdown()

    backend.run(check)


def test_two_workers_racing_to_recover_a_run_have_one_winner(backend):
    async def check() -> None:
        log: list[str] = []
        # STEADY leases: under load a sub-second lease ran out while worker A started its run
        # (T4). The dead worker's lease is ended on purpose, so nothing races the wall clock.
        await crash_worker(backend.store(), log, STEADY)
        await backend.expire_lease(backend.store(), "order-7")
        workers = [recovering_worker(backend.store(), log, f"worker-{n}", STEADY) for n in "bcd"]
        found = await asyncio.gather(*[worker.recover() for worker in workers])
        handles = [handle for handles in found for handle in handles]
        assert len(handles) == 1
        assert (await handles[0].wait()).status == "completed"
        assert log.count("charge 40") == 1 and len(log) == 3
        for worker in workers:
            await worker.shutdown()

    backend.run(check)


async def test_run_forever_recovers_until_shutdown():
    store = InMemoryStore()
    log: list[str] = []
    await crash_worker(store, log)
    worker = recovering_worker(store, log, "worker-b")
    loop = asyncio.create_task(worker.run_forever(poll_interval_s=0.05))

    async def finished() -> bool:
        return (await store.status("order-7")).status == "idle"

    await wait_until(finished)
    await worker.shutdown()
    await asyncio.wait_for(loop, 2)
    assert log.count("charge 40") == 1
    with pytest.raises(RuntimeError, match="shut down"):
        await worker.start("billing", "again")


async def test_failed_runs_are_retried_up_to_max_failures(caplog):
    store = InMemoryStore()

    def failing_agent() -> Agent:
        return Agent(ScriptedModel([RuntimeError("provider down")] * 5), name="flaky")

    runtime = AgentRuntime(store, [failing_agent()], max_failures=2)
    handle = await runtime.start("flaky", "go", session_id="f")
    with pytest.raises(RuntimeError, match="provider down"):
        await handle.wait()
    assert handle.status == "failed" and isinstance(handle.error, RuntimeError)
    assert ((await store.status("f")).status, (await store.status("f")).failures) == ("failed", 1)

    (retry,) = await runtime.recover()
    with pytest.raises(RuntimeError):
        await retry.wait()
    assert (await store.status("f")).failures == 2
    with caplog.at_level(logging.WARNING, logger="qagents.runtime"):
        assert await runtime.recover() == []
        assert await runtime.recover() == []
    skipped = [r.message for r in caplog.records if "does not recover" in r.message]
    assert len(skipped) == 1 and "failed 2 times (max_failures=2)" in skipped[0]

    # A person fixes the cause and resumes it explicitly.
    runtime.agents["flaky"] = Agent(ScriptedModel(["Recovered."]), name="flaky")
    assert (await (await runtime.resume("f")).wait()).text == "Recovered."
    await runtime.shutdown()


async def test_runs_of_unknown_agents_and_workflow_sessions_are_left_alone(caplog):
    store = InMemoryStore()
    started = Entry(kind="run_started", run_id="r", data={"agent": "elsewhere"})
    await store.append("foreign", [started])
    for session_id in (
        "workflow:report:research",
        "workflow-journal:report",
        "workflow-step:report:research",
    ):
        await store.append(session_id, [started.model_copy(update={"data": {"agent": "a"}})])
    runtime = AgentRuntime(store, [Agent(ScriptedModel([]), name="a")])
    with caplog.at_level(logging.WARNING, logger="qagents.runtime"):
        assert await runtime.recover() == []
    messages = [r.message for r in caplog.records]
    assert any("'foreign'" in m and "agent 'elsewhere'" in m for m in messages)
    assert not any("workflow" in m for m in messages)
    await runtime.shutdown()


async def test_session_prefix_limits_recovery():
    # SPEC 7.7.6: this legacy test expects takeover below the default one-second grace.
    store = InMemoryStore(takeover_grace_s=0.0)
    log: list[str] = []
    await crash_worker(store, log)
    await asyncio.sleep(FAST["lease_ttl_s"] + 0.1)
    other_app = recovering_worker(store, log, "other")
    other_app.session_prefix = "tenant-b:"
    assert await other_app.recover() == []
    worker = recovering_worker(store, log, "b")
    (handle,) = await worker.recover(limit=1)
    assert (await handle.wait()).status == "completed"
    await worker.shutdown()


# -- shutdown and lost leases ------------------------------------------------------------


async def test_shutdown_stops_at_a_safe_point_and_the_run_resumes_elsewhere():
    store = InMemoryStore()
    log: list[str] = []
    gate, started = asyncio.Event(), asyncio.Event()
    model = ScriptedModel([[call("charge_card", amount=40), call("send_receipt", to="m@x.gr")]])
    agent = Agent(model, name="billing", tools=receipt_tools(log, gate, started))
    worker = AgentRuntime(store, [agent], worker_id="worker-a", deps_factory=lambda s: "A")
    handle = await worker.start("billing", "go", session_id="s")
    await asyncio.wait_for(started.wait(), 5)

    stopping = asyncio.create_task(worker.shutdown(timeout_s=5))
    await asyncio.sleep(0.05)
    gate.set()  # the tool call finishes; the run stops before asking the model again
    await stopping
    assert handle.status == "interrupted"
    with pytest.raises(RunInterrupted, match="interrupted by shutdown.*stays open"):
        await handle.wait()
    with pytest.raises(RunInterrupted):
        async for _ in handle.events():
            pass
    assert len(model.requests) == 1  # no request was cut off
    status = await store.status("s")
    assert (status.status, status.lease_owner) == ("running", None)  # open, lease released

    other = recovering_worker(store, log, "worker-b")
    (resumed,) = await other.recover()  # no need to wait for the lease to expire
    assert (await resumed.wait()).status == "completed"
    assert log == ["charge 40", "receipt m@x.gr via A"]  # nothing ran twice
    await other.shutdown()


async def test_shutdown_cancels_a_run_that_does_not_reach_a_safe_point():
    store = InMemoryStore()
    started, never = asyncio.Event(), asyncio.Event()

    @tool
    async def send_email(to: str) -> str:
        """Send an email."""
        started.set()
        await never.wait()
        return "sent"

    agents = [
        Agent(ScriptedModel([call("send_email", to="a@b.c")]), name="mailer", tools=[send_email]),
        Agent(ScriptedModel(["Hi"]), name="greeter"),
    ]
    worker = AgentRuntime(store, agents, max_concurrent=1)
    handle = await worker.start("mailer", "mail", session_id="m")
    await asyncio.wait_for(started.wait(), 5)
    queued = await worker.start("greeter", "hello", session_id="q")
    await worker.shutdown(timeout_s=0.1)
    assert handle.status == queued.status == "interrupted"
    assert (await store.status("m")).lease_owner is None
    # The queued run was journaled by start(): it is open, with its lease released.
    assert [e.kind for e in await store.load("q")] == ["run_started", "message"]
    assert ((await store.status("q")).status, (await store.status("q")).lease_owner) == (
        "running",
        None,
    )

    model = ScriptedModel(["I could not confirm the email."])
    resumed = await Agent(model, name="mailer", tools=[send_email]).resume(Session(store, id="m"))
    assert "outcome is unknown" in resumed.messages[2].tool_results[0].content


async def test_a_queued_run_is_not_started_after_shutdown_begins():
    store = InMemoryStore()
    log: list[str] = []
    gate, started = asyncio.Event(), asyncio.Event()
    agents = [receipt_agent(log, gate, started), Agent(ScriptedModel(["Hi"]), name="greeter")]
    worker = AgentRuntime(store, agents, max_concurrent=1)
    running = await worker.start("billing", "charge", session_id="b")
    await asyncio.wait_for(started.wait(), 5)
    queued = await worker.start("greeter", "hello", session_id="g")
    assert set(worker.handles) == {"b", "g"}

    stopping = asyncio.create_task(worker.shutdown(timeout_s=5))
    await asyncio.sleep(0.05)
    gate.set()
    await stopping
    assert running.status == queued.status == "interrupted"
    assert (await store.status("g")).lease_owner is None
    assert "message" not in [e.kind for e in await store.load("g")][2:]  # it never ran
    assert worker.handles == {}


async def test_a_worker_that_loses_its_lease_stops_and_cannot_write():
    store = InMemoryStore()
    log: list[str] = []
    gate, started = asyncio.Event(), asyncio.Event()
    worker = AgentRuntime(store, [receipt_agent(log, gate, started)], worker_id="w1", **FAST)
    handle = await worker.start("billing", "go", session_id="s")
    await asyncio.wait_for(started.wait(), 5)

    # w1 stalls past its lease; w2 takes the session over.
    await store.release_lease("s", handle._lease_owner)
    assert await store.acquire_lease("s", "w2", 30)
    entries = len(await store.load("s"))
    with pytest.raises(RunInterrupted, match="lost its lease"):
        await asyncio.wait_for(handle.wait(), 5)
    # Once launched, a lost lease cancels the runtime's own task, never the caller's.
    caller = asyncio.current_task()
    assert caller is not None and caller.cancelling() == 0
    assert handle._task is not None and handle._task.cancelled()
    gate.set()
    await asyncio.sleep(0.05)
    assert len(await store.load("s")) == entries  # w1 wrote nothing after losing the lease
    assert (await store.status("s")).lease_owner == "w2"  # and did not release w2's lease
    await worker.shutdown()


async def test_a_fenced_write_failure_marks_the_handle_interrupted():
    store = InMemoryStore()
    log: list[str] = []
    gate, started = asyncio.Event(), asyncio.Event()
    # Heartbeats too slow to notice: the fenced write is what stops the run.
    worker = AgentRuntime(
        store,
        [receipt_agent(log, gate, started)],
        worker_id="w1",
        lease_ttl_s=60,
        heartbeat_s=30,
    )
    handle = await worker.start("billing", "go", session_id="s")
    await asyncio.wait_for(started.wait(), 5)
    await store.release_lease("s", handle._lease_owner)
    assert await store.acquire_lease("s", "w2", 30)
    gate.set()  # the tool returns and the run tries to record its result
    with pytest.raises(RunInterrupted, match="lost its lease") as raised:
        await asyncio.wait_for(handle.wait(), 5)
    assert isinstance(raised.value.__cause__, LeaseError)
    assert [e.kind for e in await store.load("s")][-1] == "tool_started"
    await worker.shutdown()


# -- configuration -----------------------------------------------------------------------


def test_configuration_errors_say_what_to_do():
    agent = Agent(ScriptedModel([]), name="a")
    store = InMemoryStore()
    with pytest.raises(ConfigurationError, match="below lease_ttl_s"):
        AgentRuntime(store, [agent], lease_ttl_s=5, heartbeat_s=5)
    with pytest.raises(ConfigurationError, match="lease_ttl_s"):
        AgentRuntime(store, [agent], lease_ttl_s=0)
    with pytest.raises(ConfigurationError, match="max_concurrent"):
        AgentRuntime(store, [agent], max_concurrent=0)
    with pytest.raises(ConfigurationError, match="Two different agents are named 'a'"):
        AgentRuntime(store, [agent, Agent(ScriptedModel([]), name="a")])
    with pytest.raises(ConfigurationError, match="key each agent by its name"):
        AgentRuntime(store, {"b": agent})
    with pytest.raises(ConfigurationError, match="at least one agent"):
        AgentRuntime(store, [])
    assert AgentRuntime(store, [agent, agent]).agents == {"a": agent}
    runtime = AgentRuntime(store, {"a": agent})
    assert runtime.heartbeat_s == 10.0 and runtime.worker_id


async def test_unknown_agents_and_stores_without_leases_fail_early():
    runtime = AgentRuntime(InMemoryStore(), [Agent(ScriptedModel([]), name="a")])
    with pytest.raises(ConfigurationError, match=r"No agent named 'b'.*Registered: \['a'\]"):
        await runtime.start("b", "go")

    class JournalOnly(SessionStore):
        async def append(self, session_id, entries):
            pass

        async def load(self, session_id):
            return []

        async def delete(self, session_id):
            pass

    plain = AgentRuntime(JournalOnly(), [Agent(ScriptedModel([]), name="a")])
    with pytest.raises(NotImplementedError, match="JournalOnly does not support leases"):
        await plain.start("a", "go")
    assert isinstance(await InMemoryStore().status("x"), SessionStatus)


# -- fixes: what recovery must not repeat ------------------------------------------------


async def test_an_ambiguous_failure_is_left_to_a_person_and_not_sent_again(caplog):
    store = InMemoryStore()
    model = ScriptedModel(
        [ModelTimeoutError("no answer", retryable=True, ambiguous=True), "answer"]
    )
    runtime = AgentRuntime(store, [Agent(model, name="a")])
    with pytest.raises(ModelTimeoutError):
        await (await runtime.start("a", "go", session_id="s")).wait()
    assert (await store.status("s")).status == "failed"

    other = AgentRuntime(store, [Agent(model, name="a")], worker_id="other")
    with caplog.at_level(logging.WARNING, logger="qagents.runtime"):
        assert await other.recover() == []
        assert await other.recover() == []
    assert len(model.requests) == 1  # the possibly billed request was not sent again
    skipped = [r.message for r in caplog.records if "does not recover" in r.message]
    assert len(skipped) == 1 and "may have processed and billed it" in skipped[0]

    # A person decides that sending it again is fine.
    assert (await (await other.resume("s")).wait()).text == "answer"
    assert len(model.requests) == 2
    await runtime.shutdown()
    await other.shutdown()


async def test_a_failure_journaled_without_its_flags_is_left_to_a_person():
    store = InMemoryStore()
    prompt = Message.user("go").model_dump(mode="json")
    await store.append(
        "old",
        [
            Entry(kind="run_started", run_id="r", data={"agent": "a"}),
            Entry(kind="message", run_id="r", data={"message": prompt}),
            Entry(kind="run_failed", run_id="r", data={"error": "ModelServerError: 502"}),
        ],
    )
    model = ScriptedModel(["answer"])
    runtime = AgentRuntime(store, [Agent(model, name="a")])
    assert await runtime.recover() == []
    assert model.requests == [] and (await store.status("old")).lease_owner is None
    await runtime.shutdown()


async def test_limits_given_to_start_are_journaled_and_kept_by_recovery():
    store = InMemoryStore()
    flaky = ScriptedModel([ModelTimeoutError("no answer", retryable=True, ambiguous=True)])
    runtime = AgentRuntime(store, [Agent(flaky, name="a")])
    # retry_ambiguous=True was given to start(), not set on the agent: recovery honours it.
    limits = Limits(retry_ambiguous=True, model_retries=0, max_turns=4)
    with pytest.raises(ModelTimeoutError):
        await (await runtime.start("a", "go", session_id="s", limits=limits)).wait()

    other = AgentRuntime(store, [Agent(ScriptedModel(["answer"]), name="a")], worker_id="other")
    (handle,) = await other.recover()
    assert (await handle.wait()).text == "answer"
    recorded = [e for e in await store.load("s") if e.kind == LIMITS_KIND]
    assert len(recorded) == 1 and recorded[0].data["limits"]["max_turns"] == 4
    await runtime.shutdown()
    await other.shutdown()


async def test_limits_given_to_a_plain_run_are_journaled_and_kept_by_recovery():
    store = InMemoryStore()
    flaky = ScriptedModel([ModelTimeoutError("no answer", retryable=True, ambiguous=True)])
    limits = Limits(retry_ambiguous=True, model_retries=0, max_turns=4)
    session = Session(store, id="s")
    with pytest.raises(ModelTimeoutError):
        await Agent(flaky, name="a").run("go", session=session, limits=limits)
    (started,) = [e for e in await store.load("s") if e.kind == "run_started"]
    assert started.data["limits"]["max_turns"] == 4
    # retry_ambiguous=True came with the run, not with the agent: recovery honours it.
    other = AgentRuntime(store, [Agent(ScriptedModel(["answer"]), name="a")], worker_id="other")
    (handle,) = await other.recover()
    assert (await handle.wait()).text == "answer"
    await other.shutdown()


async def test_a_recovered_run_keeps_the_max_turns_it_was_started_with():
    # SPEC 7.7.6: this legacy test expects takeover below the default one-second grace.
    store = InMemoryStore(takeover_grace_s=0.0)
    gate, started = asyncio.Event(), asyncio.Event()

    @tool(read_only=True)
    async def search(query: str) -> str:
        """Search."""
        started.set()
        await gate.wait()
        return "more"

    def researcher() -> Agent:
        return Agent(ScriptedModel([call("search", query="q")] * 10), name="r", tools=[search])

    connection = CrashableStore(store)
    worker = AgentRuntime(connection, [researcher()], worker_id="a", **FAST)
    handle = await worker.start("r", "go", session_id="t1", limits=Limits(max_turns=2))
    await asyncio.wait_for(started.wait(), 5)
    connection.crash()
    assert handle._task is not None
    handle._task.cancel()
    with pytest.raises(RunInterrupted):
        await handle.wait()

    gate.set()
    await asyncio.sleep(FAST["lease_ttl_s"] + 0.1)
    other = AgentRuntime(store, [researcher()], worker_id="b", **FAST)
    (resumed,) = await other.recover()
    result = await resumed.wait()
    assert (result.status, result.stop_reason) == ("limit_reached", "max_turns")
    assert result.turns == 2
    await other.shutdown()


async def test_a_run_that_crashes_its_worker_every_time_is_left_to_a_person(caplog):
    store = InMemoryStore()
    prompt = Message.user("go").model_dump(mode="json")
    await store.append(
        "loop",
        [
            Entry(kind="run_started", run_id="r", data={"agent": "a"}),
            Entry(kind="message", run_id="r", data={"message": prompt}),
            Entry(kind=RECOVERED_KIND, run_id="r", data={"worker": "w1", "status": "running"}),
            Entry(kind=RECOVERED_KIND, run_id="r", data={"worker": "w2", "status": "running"}),
        ],
    )
    model = ScriptedModel(["Done."])
    runtime = AgentRuntime(store, [Agent(model, name="a")], max_recoveries=2)
    with caplog.at_level(logging.WARNING, logger="qagents.runtime"):
        assert await runtime.recover() == []
        assert await runtime.recover() == []
    status = await store.status("loop")
    assert (status.status, status.lease_owner) == ("failed", None)
    failure = (await store.load("loop"))[-1]
    assert failure.kind == "run_failed" and failure.data["needs_person"] is True
    assert failure.data["error_type"] == "RecoveryLimitReached"
    skipped = [r.message for r in caplog.records if "does not recover" in r.message]
    assert len(skipped) == 1 and "2 times in a row without progress" in skipped[0]
    assert model.requests == []

    # A person resumes it.
    assert (await (await runtime.resume("loop")).wait()).text == "Done."
    await runtime.shutdown()


async def test_each_recovery_is_journaled_and_progress_resets_the_count():
    # SPEC 7.7.6: this legacy test expects takeover below the default one-second grace.
    store = InMemoryStore(takeover_grace_s=0.0)
    log: list[str] = []
    await crash_worker(store, log)
    await store.append(
        "order-7",
        [Entry(kind=RECOVERED_KIND, run_id=(await store.status("order-7")).open_run_id)],
    )
    await asyncio.sleep(FAST["lease_ttl_s"] + 0.1)
    worker = recovering_worker(store, log, "worker-b")
    worker.max_recoveries = 2
    (handle,) = await worker.recover()  # one earlier recovery: still below the bound
    assert (await handle.wait()).status == "completed"
    recovered = [e.data for e in await store.load("order-7") if e.kind == RECOVERED_KIND]
    assert recovered[-1] == {"worker": "worker-b", "status": "running"}
    await worker.shutdown()


# -- fixes: start() journals the run -----------------------------------------------------


def echo_agent(name: str = "worker", input_guardrails: list[Any] | None = None) -> Agent:
    def reply(request: Any) -> str:
        last = next(m for m in reversed(request.messages) if m.role == "user")
        return f"answering: {last.text}"

    return Agent(FunctionModel(reply), name=name, input_guardrails=input_guardrails or [])


async def test_a_queued_start_is_journaled_and_a_later_worker_answers_the_new_prompt():
    store = InMemoryStore()
    old = AgentRuntime(store, [Agent(ScriptedModel([RuntimeError("down")]), name="worker")])
    with pytest.raises(RuntimeError):
        await (await old.start("worker", "old question", session_id="b")).wait()
    await old.shutdown()

    gate, started = asyncio.Event(), asyncio.Event()
    busy = receipt_agent([], gate, started)
    runtime = AgentRuntime(store, [busy, echo_agent()], max_concurrent=1)
    await runtime.start("billing", "charge", session_id="a")
    await asyncio.wait_for(started.wait(), 5)
    queued = await runtime.start("worker", "NEW task B", session_id="b")
    assert queued.status == "queued" and queued.run_id is not None
    messages = await Session(store, id="b").messages()
    assert messages[-1].text == "NEW task B"  # durable before it got a slot
    await runtime.shutdown(timeout_s=0.1)
    with pytest.raises(RunInterrupted, match="stays open in the journal"):
        await queued.wait()

    other = AgentRuntime(store, [echo_agent()], worker_id="other")
    (recovered,) = [h for h in await other.recover() if h.session_id == "b"]
    assert (await recovered.wait()).text == "answering: NEW task B"
    await other.shutdown()


async def test_start_raises_what_the_run_raises_before_it_is_journaled():
    store = InMemoryStore()
    runtime = AgentRuntime(store, [Agent(ScriptedModel([]), name="a")])
    with pytest.raises(ConfigurationError, match="Nothing to run"):
        await runtime.start("a", session_id="empty")
    assert await store.load("empty") == []
    assert (await store.status("empty")).lease_owner is None and runtime.handles == {}
    await runtime.shutdown()


# -- fixes: shutdown races and safe points ---------------------------------------------


async def test_shutdown_waits_for_a_start_in_flight_which_then_backs_out():
    store = InMemoryStore()
    release = asyncio.Event()
    ran: list[str] = []

    async def slow_deps(session_id: str) -> str:
        await release.wait()
        return "deps"

    @tool
    def charge() -> str:
        """Charge."""
        ran.append("charge")
        return "ok"

    agent = Agent(ScriptedModel([call("charge"), "done"]), name="a", tools=[charge])
    runtime = AgentRuntime(store, [agent], deps_factory=slow_deps)
    starting = asyncio.create_task(runtime.start("a", "go", session_id="s1"))
    await asyncio.sleep(0.05)
    stopping = asyncio.create_task(runtime.shutdown(timeout_s=5))
    await asyncio.sleep(0.05)
    assert not stopping.done()  # it waits for the claim in flight
    release.set()
    with pytest.raises(RuntimeError, match="shut down"):
        await starting
    await stopping
    assert runtime.handles == {} and ran == []
    assert await store.load("s1") == [] and (await store.status("s1")).lease_owner is None


async def test_shutdown_during_recover_launches_nothing_and_frees_the_lease():
    # SPEC 7.7.6: this legacy test expects takeover below the default one-second grace.
    store = InMemoryStore(takeover_grace_s=0.0)
    log: list[str] = []
    await crash_worker(store, log)
    await asyncio.sleep(FAST["lease_ttl_s"] + 0.1)
    release, preparing = asyncio.Event(), asyncio.Event()

    async def slow_deps(session_id: str) -> str:
        preparing.set()
        await release.wait()
        return "B"

    gate, started = asyncio.Event(), asyncio.Event()
    worker = AgentRuntime(
        store, [receipt_agent(log, gate, started)], worker_id="b", deps_factory=slow_deps, **FAST
    )
    recovering = asyncio.create_task(worker.recover())
    try:
        async with asyncio.timeout(10):
            await preparing.wait()
            stopping = asyncio.create_task(worker.shutdown(timeout_s=5))
            await worker._closing.wait()
            release.set()
            assert await recovering == []
            await stopping
    finally:
        release.set()
    status = await store.status("order-7")
    assert (status.status, status.lease_owner) == ("running", None)
    assert RECOVERED_KIND not in [e.kind for e in await store.load("order-7")]
    assert log == ["charge 40", "receipt m@x.gr via A"]


async def test_shutdown_never_cuts_a_gated_tool_call_that_already_started():
    store = InMemoryStore()
    transfers: list[int] = []
    notified = asyncio.Event()
    finish_notify = asyncio.Event()

    @tool
    async def notify(to: str) -> str:
        """Notify someone."""
        notified.set()
        await finish_notify.wait()
        return "notified"

    @tool
    def transfer(amount: int) -> str:
        """Move money."""
        time.sleep(0.2)
        transfers.append(amount)
        return "moved"

    def agent(replies: list[Any]) -> Agent:
        return Agent(
            ScriptedModel(replies),
            name="bank",
            tools=[notify, transfer],
            limits=Limits(parallel_tools="all", max_parallel_tools=1),
        )

    runtime = AgentRuntime(store, [agent([[call("notify", to="x"), call("transfer", amount=5)]])])
    handle = await runtime.start("bank", "go", session_id="t")
    await asyncio.wait_for(notified.wait(), 5)
    stopping = asyncio.create_task(runtime.shutdown(timeout_s=5))
    await asyncio.sleep(0.05)
    finish_notify.set()
    await stopping
    assert handle.status == "interrupted"
    assert (await Session(store, id="t").state()).started_tools == []  # nothing was cut off

    other = AgentRuntime(store, [agent(["Moved."])], worker_id="other")
    (resumed,) = await other.recover()
    result = await resumed.wait()
    assert result.status == "completed" and transfers == [5]
    contents = [p.content for m in result.messages if m.role == "tool" for p in m.tool_results]
    assert not any("outcome is unknown" in content for content in contents)
    await other.shutdown()


class SlowLoadStore(InMemoryStore):
    """``load`` waits for ``gate`` while it is armed."""

    def __init__(self) -> None:
        super().__init__()
        self.armed = False
        self.gate = asyncio.Event()
        self.waiting = asyncio.Event()

    async def load(self, session_id):
        if self.armed:
            self.waiting.set()
            await self.gate.wait()
        return await super().load(session_id)


def refund_agent(refunds: list[str], replies: list[Any]) -> Agent:
    @tool(requires_approval=True)
    def refund(order: str) -> str:
        """Refund an order."""
        refunds.append(order)
        return "refunded"

    return Agent(ScriptedModel(replies), name="support", tools=[refund])


async def test_a_shutdown_during_a_resume_keeps_its_approvals_and_state():
    store = SlowLoadStore()
    refunds: list[str] = []
    runtime = AgentRuntime(store, [refund_agent(refunds, [call("refund", order="A-1")])])
    paused = await (await runtime.start("support", "Refund A-1", session_id="p")).wait()
    approval = {paused.pending_approvals[0].tool_call_id: True}

    handle = await runtime.resume("p", approvals=approval, state={"ticket": "T-9"})
    store.armed = True  # before its task runs: the resumed run waits while it reads the journal
    await asyncio.wait_for(store.waiting.wait(), 5)
    stopping = asyncio.create_task(runtime.shutdown(timeout_s=5))
    await asyncio.sleep(0.05)
    store.armed = False
    store.gate.set()
    await stopping
    assert handle.status == "interrupted"
    assert refunds == ["A-1"]  # the approved call ran before the run stopped
    assert (await Session(store, id="p").state()).values == {"ticket": "T-9"}

    other = AgentRuntime(store, [refund_agent(refunds, ["Refunded."])], worker_id="other")
    (resumed,) = await other.recover()
    assert (await resumed.wait()).text == "Refunded." and refunds == ["A-1"]
    await other.shutdown()


async def test_a_paused_run_whose_resume_was_queued_says_it_is_still_paused():
    store = InMemoryStore()
    refunds: list[str] = []
    gate, started = asyncio.Event(), asyncio.Event()
    agents = [
        refund_agent(refunds, [call("refund", order="A-1")]),
        receipt_agent([], gate, started),
    ]
    runtime = AgentRuntime(store, agents, max_concurrent=1)
    paused = await (await runtime.start("support", "Refund A-1", session_id="p")).wait()
    await runtime.start("billing", "charge", session_id="busy")
    await asyncio.wait_for(started.wait(), 5)

    queued = await runtime.resume("p", approvals={paused.pending_approvals[0].tool_call_id: True})
    await runtime.shutdown(timeout_s=0.1)
    with pytest.raises(RunInterrupted, match="still paused") as raised:
        await queued.wait()
    assert "recover() never resumes a paused run" in str(raised.value)
    assert (await store.status("p")).status == "paused" and refunds == []


async def test_a_queued_run_that_was_cancelled_is_closed_even_by_a_shutdown():
    store = InMemoryStore()
    sent: list[str] = []

    @tool
    def send_newsletter(to: str) -> str:
        """Send the newsletter."""
        sent.append(to)
        return "sent"

    def mailer(replies: list[Any]) -> Agent:
        return Agent(ScriptedModel(replies), name="mailer", tools=[send_newsletter])

    first = AgentRuntime(store, [mailer([RuntimeError("down")])])
    with pytest.raises(RuntimeError):
        await (await first.start("mailer", "send it", session_id="news-1")).wait()
    await first.shutdown()

    gate, started = asyncio.Event(), asyncio.Event()
    newsletter = mailer([call("send_newsletter", to="all@list"), "Sent."])
    worker = AgentRuntime(store, [newsletter, receipt_agent([], gate, started)], max_concurrent=1)
    await worker.start("billing", "charge", session_id="busy")
    await asyncio.wait_for(started.wait(), 5)
    queued = await worker.resume("news-1")
    queued.cancel("operator stopped the newsletter")
    await worker.shutdown(timeout_s=0.1)
    result = await queued.wait()
    assert (result.status, result.stop_reason) == ("cancelled", "operator stopped the newsletter")
    assert (await store.status("news-1")).status == "idle"

    other = AgentRuntime(store, [mailer([call("send_newsletter", to="all@list"), "Sent."])])
    assert "news-1" not in [h.session_id for h in await other.recover()]
    assert sent == [] and newsletter.model.requests == []
    await other.shutdown(timeout_s=0.1)


class SlowReleaseStore(InMemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.releasing = asyncio.Event()

    async def release_lease(self, session_id, owner):
        self.releasing.set()
        await asyncio.sleep(0.3)
        await super().release_lease(session_id, owner)


async def test_a_finished_run_keeps_its_result_when_shutdown_cancels_its_lease_release():
    store = SlowReleaseStore()
    runtime = AgentRuntime(store, [Agent(ScriptedModel(["The answer is 42."]), name="a")])
    handle = await runtime.start("a", "go", session_id="s")
    await asyncio.wait_for(store.releasing.wait(), 5)
    assert (await store.status("s")).status == "idle" and not handle.done()
    await runtime.shutdown(timeout_s=0.05)
    assert handle.status == "finished"
    assert (await handle.wait()).text == "The answer is 42."
    assert (await store.status("s")).lease_owner is None  # released before shutdown returned


async def test_a_plain_resume_of_a_runtime_session_is_not_taken_over_by_recover():
    store = InMemoryStore()
    refunds: list[str] = []
    gate, started = asyncio.Event(), asyncio.Event()

    @tool(requires_approval=True)
    async def refund(order: str) -> str:
        """Refund an order."""
        started.set()
        await gate.wait()
        refunds.append(order)
        return f"refunded {order}"

    def support(replies: list[Any]) -> Agent:
        return Agent(ScriptedModel(replies), name="support", tools=[refund])

    runtime = AgentRuntime(store, [support([call("refund", order="A-1")])])
    paused = await (await runtime.start("support", "Refund A-1", session_id="t1")).wait()
    decisions = {paused.pending_approvals[0].tool_call_id: True}
    # What RunResult documents: continue with agent.resume(result.session, ...).
    resuming = asyncio.create_task(support(["Done."]).resume(paused.session, approvals=decisions))
    await asyncio.wait_for(started.wait(), 5)
    other = AgentRuntime(store, [support(["Done."])], worker_id="other")
    assert await other.recover() == []  # the plain run holds a lease of its own
    gate.set()
    assert (await resuming).text == "Done."
    tool_messages = [m for m in await Session(store, id="t1").messages() if m.role == "tool"]
    assert len(tool_messages) == 1 and refunds == ["A-1"]
    await runtime.shutdown()
    await other.shutdown()


# -- fixes: polling cost and logs ----------------------------------------------------------


class CountingStore(InMemoryStore):
    def __init__(self, broken: str | None = None) -> None:
        super().__init__()
        self.acquired: list[str] = []
        self.broken = broken

    async def acquire_lease_grant(self, session_id, owner, ttl_s):
        self.acquired.append(session_id)
        if session_id == self.broken:
            raise ConnectionError("database is locked")
        return await super().acquire_lease_grant(session_id, owner, ttl_s)


async def test_recover_passes_over_live_leases_and_survives_a_failing_candidate():
    store = CountingStore(broken="bad")
    started = Entry(kind="run_started", run_id="r", data={"agent": "a"})
    prompt = Entry(kind="message", run_id="r", data={"message": Message.user("hi").model_dump()})
    for session_id in ("bad", "busy", "dead"):
        await store.append(session_id, [started, prompt])
    assert await store.acquire_lease("busy", "elsewhere", 30)
    store.acquired.clear()

    runtime = AgentRuntime(store, [Agent(ScriptedModel(["Hello."]), name="a")])
    handles = await runtime.recover()
    assert [handle.session_id for handle in handles] == ["dead"]
    assert "busy" not in store.acquired  # no lease write for a session another worker drives
    assert (await handles[0].wait()).text == "Hello."
    await runtime.shutdown()


async def test_an_expected_renewal_failure_is_logged_without_a_traceback(caplog):
    class FlakyRenewals(InMemoryStore):
        async def renew_lease(self, session_id, owner, ttl_s):
            raise ConnectionError("the database restarted")

    gate, started = asyncio.Event(), asyncio.Event()
    runtime = AgentRuntime(FlakyRenewals(), [receipt_agent([], gate, started)], **FAST)
    with caplog.at_level(logging.WARNING, logger="qagents.runtime"):
        handle = await runtime.start("billing", "go")
        await asyncio.wait_for(started.wait(), 5)
        await asyncio.sleep(0.12)
        gate.set()
        await handle.wait()
    renewals = [r for r in caplog.records if "could not renew" in r.message]
    assert renewals and all(r.exc_info is None for r in renewals)
    assert "ConnectionError" in renewals[0].message
    await runtime.shutdown()


# -- second pass: preparation under the lease, and what an interruption says -------------


async def test_a_slow_start_keeps_its_lease_so_no_worker_resumes_the_old_run():
    store = InMemoryStore()
    old = AgentRuntime(store, [Agent(ScriptedModel([RuntimeError("down")]), name="worker")])
    with pytest.raises(RuntimeError):
        await (await old.start("worker", "old question", session_id="b")).wait()
    await old.shutdown()

    async def moderation(ctx: Any, message: Any) -> None:
        await asyncio.sleep(4 * FAST["lease_ttl_s"])  # longer than the lease lasts

    starter = AgentRuntime(
        store, [echo_agent(input_guardrails=[moderation])], worker_id="starter", **FAST
    )
    poller = AgentRuntime(store, [echo_agent()], worker_id="poller", **FAST)
    starting = asyncio.create_task(starter.start("worker", "NEW question", session_id="b"))
    await asyncio.sleep(2 * FAST["lease_ttl_s"])
    assert await poller.recover() == []  # the starter's heartbeat keeps its lease
    handle = await starting
    assert (await handle.wait()).text == "answering: NEW question"
    texts = [m.text for m in await Session(store, id="b").messages()]
    assert "answering: old question" not in texts
    await starter.shutdown()
    await poller.shutdown()


async def test_a_slow_recovery_keeps_its_lease_so_only_one_worker_resumes_the_run():
    # SPEC 7.7.6: this legacy test expects takeover below the default one-second grace.
    store = InMemoryStore(takeover_grace_s=0.0)
    log: list[str] = []
    await crash_worker(store, log)
    await asyncio.sleep(FAST["lease_ttl_s"] + 0.1)

    async def slow_deps(session_id: str) -> str:
        await asyncio.sleep(4 * FAST["lease_ttl_s"])
        return "B"

    slow = recovering_worker(store, log, "worker-b")
    slow.deps_factory = slow_deps
    other = recovering_worker(store, log, "worker-c")
    recovering = asyncio.create_task(slow.recover())
    await asyncio.sleep(2 * FAST["lease_ttl_s"])
    assert await other.recover() == []
    (handle,) = await recovering
    assert (await handle.wait()).status == "completed"
    assert log == ["charge 40", "receipt m@x.gr via A", "receipt m@x.gr via B"]
    await slow.shutdown()
    await other.shutdown()


async def test_a_queued_resume_stopped_by_shutdown_says_its_state_was_not_kept():
    store = InMemoryStore()
    first = AgentRuntime(store, [Agent(ScriptedModel([RuntimeError("down")]), name="a")])
    with pytest.raises(RuntimeError):
        await (await first.start("a", "who am I?", session_id="s")).wait()
    await first.shutdown()

    gate, started = asyncio.Event(), asyncio.Event()
    agents = [Agent(ScriptedModel(["done"]), name="a"), receipt_agent([], gate, started)]
    runtime = AgentRuntime(store, agents, max_concurrent=1)
    await runtime.start("billing", "charge", session_id="busy")
    await asyncio.wait_for(started.wait(), 5)
    queued = await runtime.resume("s", state={"tenant": "globex"})
    await runtime.shutdown(timeout_s=0.1)
    with pytest.raises(RunInterrupted, match=r"state= given to resume\(\) was not journaled"):
        await queued.wait()
    assert (await Session(store, id="s").state()).values == {}
    gate.set()


async def test_an_interrupted_run_that_recover_holds_back_says_so():
    store = InMemoryStore()
    timeout = ModelTimeoutError("no answer", retryable=True, ambiguous=True)
    first = AgentRuntime(store, [Agent(ScriptedModel([timeout]), name="a")])
    with pytest.raises(ModelTimeoutError):
        await (await first.start("a", "go", session_id="s")).wait()
    await first.shutdown()

    gate, started = asyncio.Event(), asyncio.Event()
    agents = [Agent(ScriptedModel(["answer"]), name="a"), receipt_agent([], gate, started)]
    runtime = AgentRuntime(store, agents, max_concurrent=1)
    await runtime.start("billing", "charge", session_id="busy")
    await asyncio.wait_for(started.wait(), 5)
    queued = await runtime.resume("s")
    await runtime.shutdown(timeout_s=0.1)
    with pytest.raises(RunInterrupted, match=r"recover\(\) does not resume it") as raised:
        await queued.wait()
    assert "failed ambiguously" in str(raised.value)
    assert "so does recover()" not in str(raised.value)
    gate.set()


async def test_a_recovery_stopped_by_shutdown_before_it_resumed_is_not_counted():
    # SPEC 7.7.6: this legacy test expects takeover below the default one-second grace.
    store = InMemoryStore(takeover_grace_s=0.0)
    log: list[str] = []
    await crash_worker(store, log)
    await asyncio.sleep(FAST["lease_ttl_s"] + 0.1)

    gate, started = asyncio.Event(), asyncio.Event()
    worker = AgentRuntime(
        store, [receipt_agent([], gate, started)], worker_id="b", max_concurrent=1, **FAST
    )
    await worker.start("billing", "charge", session_id="busy")
    await asyncio.wait_for(started.wait(), 5)
    (queued,) = await worker.recover(limit=1)  # it waits for the busy run's slot
    await worker.shutdown(timeout_s=0.1)
    with pytest.raises(RunInterrupted):
        await queued.wait()
    assert RECOVERED_KIND not in [e.kind for e in await store.load("order-7")]
    assert (await store.status("order-7")).lease_owner is None
    gate.set()


@pytest.mark.parametrize("failure", ["raise", "hang"])
async def test_renewal_errors_stop_the_run_at_its_last_confirmed_deadline(failure):
    class UnreachableRenewal(InMemoryStore):
        renewals = 0

        async def renew_lease(self, session_id, owner, ttl):
            self.renewals += 1
            if failure == "hang":
                await asyncio.Event().wait()
            raise ConnectionError("renewal unavailable")

    stopped = asyncio.Event()

    async def reply(request):
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    store = UnreachableRenewal()
    agent = Agent(FunctionModel(reply), name="a")
    runtime = AgentRuntime(store, [agent], lease_ttl_s=0.15, heartbeat_s=0.03)
    try:
        handle = await runtime.start("a", "go", session_id="s")
        with pytest.raises(RunInterrupted):
            await asyncio.wait_for(handle.wait(), timeout=1)
        assert handle._lease_lost and stopped.is_set()
        renewals = store.renewals
        await asyncio.sleep(0.08)
        assert store.renewals == renewals
    finally:
        await runtime.shutdown(timeout_s=0)


async def test_unconfirmed_lease_also_stops_preparation_with_a_lease_error():
    class UnreachableRenewal(InMemoryStore):
        async def renew_lease(self, session_id, owner, ttl):
            raise ConnectionError("renewal unavailable")

    stopped = asyncio.Event()

    async def deps(session_id):
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    runtime = AgentRuntime(UnreachableRenewal(), [echo_agent()], deps_factory=deps, **FAST)
    starting = asyncio.create_task(runtime.start("worker", "go", session_id="s"))
    try:
        done, _ = await asyncio.wait([starting], timeout=5)
        assert starting in done and not starting.cancelled()
        assert isinstance(starting.exception(), LeaseError)  # not the runtime's own cancel
        assert starting.cancelling() == 0
        assert stopped.is_set()
    finally:
        starting.cancel()
        await asyncio.gather(starting, return_exceptions=True)
        await runtime.shutdown(timeout_s=0)


class LosesLeaseOnCue(InMemoryStore):
    """Renews leases until ``refused`` is set, then refuses, as after a takeover.

    ``stall`` names an entry kind: the first leased write with one sets ``refused`` and
    waits, writing nothing, until it is cancelled. The identical retry goes through.
    """

    def __init__(self) -> None:
        super().__init__()
        self.refused = False
        self.stall: str | None = None

    async def renew_lease(self, session_id, owner, ttl_s):
        if self.refused:
            return False
        return await super().renew_lease(session_id, owner, ttl_s)

    async def append_leased(self, session_id, entries, owner):
        if self.stall is not None and any(entry.kind == self.stall for entry in entries):
            self.stall, self.refused = None, True
            await asyncio.Event().wait()
        return await super().append_leased(session_id, entries, owner)


def open_run(agent: str = "worker") -> list[Entry]:
    """The journal of a run that started and was never finished."""
    return [
        Entry(kind="run_started", run_id="r", data={"agent": agent}),
        Entry(kind="message", run_id="r", data={"message": Message.user("go").model_dump()}),
    ]


@pytest.mark.parametrize("phase", ["start", "resume", "start_journaled"])
async def test_a_lease_lost_while_the_run_is_prepared_raises_lease_error(phase):
    store = LosesLeaseOnCue()
    stopped = asyncio.Event()

    async def deps(session_id: str) -> None:
        store.refused = True  # another worker takes the session over meanwhile
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    if phase == "resume":
        await store.append("s", open_run())
    journaled = phase == "start_journaled"
    if journaled:  # lost after the run is journaled, while its limits are written
        store.stall = LIMITS_KIND
    runtime = AgentRuntime(store, [echo_agent()], deps_factory=None if journaled else deps, **CUE)
    before = [entry.kind for entry in await store.load("s")]
    caller = asyncio.current_task()
    assert caller is not None
    outcome: BaseException | None = None
    try:
        async with asyncio.timeout(5):
            if phase == "resume":
                await runtime.resume("s")
            else:
                limits = Limits(max_turns=3) if journaled else None
                await runtime.start("worker", "go", session_id="s", limits=limits)
    except BaseException as raised:  # a CancelledError too: that was the bug
        outcome = raised
    assert isinstance(outcome, LeaseError), repr(outcome)
    assert "while it prepared the run" in str(outcome)
    assert caller.cancelling() == 0  # the runtime took its own cancel back
    await asyncio.sleep(0)  # and the caller goes on as usual
    assert runtime.handles == {}
    status = await store.status("s")
    assert status.lease_owner is None  # released: recover() need not wait for it to expire
    kinds = [entry.kind for entry in await store.load("s")]
    if phase == "start":
        assert stopped.is_set() and kinds == [] and status.status == "idle"
    elif phase == "resume":
        assert stopped.is_set() and kinds == before and status.status == "running"
    else:  # the run was journaled: it stays open, and recover() resumes it
        assert {"run_started", "message", LIMITS_KIND} <= set(kinds)
        assert status.status == "running"
        store.refused = False
        other = AgentRuntime(store, [echo_agent()], worker_id="other")
        (handle,) = await other.recover()
        assert (await handle.wait()).text == "answering: go"
        await other.shutdown()
    await runtime.shutdown()


@pytest.mark.parametrize("lease_lost_too", [False, True])
async def test_an_outside_cancel_during_start_still_propagates(lease_lost_too):
    store = LosesLeaseOnCue()
    preparing = asyncio.Event()

    async def deps(session_id: str) -> None:
        preparing.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if lease_lost_too:  # the lease is lost as well before start() ends
                store.refused = True
                await asyncio.Event().wait()  # the runtime's own cancel ends this wait
            raise

    runtime = AgentRuntime(store, [echo_agent()], deps_factory=deps, **CUE)
    starting = asyncio.create_task(runtime.start("worker", "go", session_id="s"))
    async with asyncio.timeout(5):
        await preparing.wait()
        starting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await starting
    assert starting.cancelled()
    assert starting.cancelling() == 1  # the outside cancel stays; only the runtime's is gone
    status = await store.status("s")
    assert status.lease_owner is None and await store.load("s") == []
    assert runtime.handles == {}
    await runtime.shutdown()


@pytest.mark.parametrize("entry", ["start", "recover"])
async def test_an_older_cancel_request_on_the_caller_does_not_hide_a_lost_lease(entry):
    """A cancel the caller caught earlier without ``uncancel()`` still counts in its
    ``cancelling()``. It is not a cancel of this preparation from outside (AF-4 review)."""
    store = LosesLeaseOnCue()
    if entry == "recover":
        await store.append("s", open_run())

    async def deps(session_id: str) -> None:
        store.refused = True  # another worker takes the session over meanwhile
        await asyncio.Event().wait()

    runtime = AgentRuntime(store, [echo_agent()], deps_factory=deps, **CUE)

    async def caller() -> tuple[object, int]:
        me = asyncio.current_task()
        assert me is not None
        me.cancel()
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass  # caught without uncancel(): the request stays counted
        assert me.cancelling() == 1
        try:
            async with asyncio.timeout(5):
                if entry == "start":
                    return await runtime.start("worker", "go", session_id="s"), me.cancelling()
                return await runtime.recover(), me.cancelling()
        except BaseException as raised:  # a CancelledError too: that was the bug
            return raised, me.cancelling()

    outcome, cancelling = await asyncio.create_task(caller())
    if entry == "start":
        assert isinstance(outcome, LeaseError), repr(outcome)
        assert "while it prepared the run" in str(outcome)
    else:
        assert outcome == [], repr(outcome)  # recover() passes the session over
    assert cancelling == 1  # the runtime took back its own cancel, not the older one
    status = await store.status("s")
    assert status.lease_owner is None
    assert runtime.handles == {}
    await runtime.shutdown()


@pytest.mark.parametrize("phase", ["start", "resume"])
async def test_an_error_raised_for_the_runtimes_cancel_still_reports_the_lost_lease(phase):
    """Preparation code that turns the runtime's cancel into its own error does not hide
    why the run did not start: the caller gets LeaseError, caused by that error (AF-4)."""
    store = LosesLeaseOnCue()

    class DepsUnavailable(Exception):
        pass

    async def deps(session_id: str) -> None:
        store.refused = True  # another worker takes the session over meanwhile
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as cancel:
            raise DepsUnavailable("pool closed") from cancel

    if phase == "resume":
        await store.append("s", open_run())
    before = [entry.kind for entry in await store.load("s")]
    runtime = AgentRuntime(store, [echo_agent()], deps_factory=deps, **CUE)
    caller = asyncio.current_task()
    assert caller is not None
    outcome: BaseException | None = None
    try:
        async with asyncio.timeout(5):
            if phase == "resume":
                await runtime.resume("s")
            else:
                await runtime.start("worker", "go", session_id="s")
    except BaseException as raised:
        outcome = raised
    assert isinstance(outcome, LeaseError), repr(outcome)
    assert "while it prepared the run" in str(outcome)
    assert isinstance(outcome.__cause__, DepsUnavailable)
    assert caller.cancelling() == 0
    assert runtime.handles == {}
    status = await store.status("s")
    assert status.lease_owner is None
    assert [entry.kind for entry in await store.load("s")] == before
    assert status.status == ("running" if phase == "resume" else "idle")
    await runtime.shutdown()


async def test_a_lease_lost_while_a_recovery_is_prepared_leaves_run_forever_polling():
    store = LosesLeaseOnCue()
    await store.append("s", open_run())
    attempts = 0
    third = asyncio.Event()

    async def deps(session_id: str) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 3:
            third.set()
        store.refused = True
        await asyncio.Event().wait()

    runtime = AgentRuntime(store, [echo_agent()], deps_factory=deps, **CUE)
    caller = asyncio.current_task()
    assert caller is not None
    async with asyncio.timeout(5):
        assert await runtime.recover() == []  # not a CancelledError
    assert caller.cancelling() == 0
    status = await store.status("s")
    assert (status.status, status.lease_owner) == ("running", None)
    assert [entry.kind for entry in await store.load("s")] == ["run_started", "message"]

    polling = asyncio.create_task(runtime.run_forever(poll_interval_s=0.01))
    async with asyncio.timeout(5):
        await third.wait()  # run_forever polled again after its first poll lost the lease
    assert not polling.done()
    await runtime.shutdown(timeout_s=0)
    await asyncio.wait_for(polling, 5)


async def test_reused_worker_name_cannot_share_or_renew_another_acquisition():
    store = InMemoryStore()
    gate = asyncio.Event()

    async def reply(request):
        await gate.wait()
        return "done"

    agent = Agent(FunctionModel(reply), name="a")
    first = AgentRuntime(store, [agent], worker_id="reused", **FAST)
    second = AgentRuntime(store, [agent], worker_id="reused", **FAST)
    try:
        handle = await first.start("a", "go", session_id="s")
        owner = (await store.status("s")).lease_owner
        with pytest.raises(LeaseError):
            await second.resume("s")
        assert owner.startswith("reused#")
        assert await second.recover() == []
        assert (await store.status("s")).lease_owner == owner
        gate.set()
        await handle.wait()
        again = await second.start("a", "again", session_id="s")
        next_owner = (await store.status("s")).lease_owner
        assert next_owner.startswith("reused#") and next_owner != owner
        assert not await store.renew_lease("s", owner, 10)
        await store.release_lease("s", owner)
        assert (await store.status("s")).lease_owner == next_owner
        await again.wait()
    finally:
        gate.set()
        await first.shutdown(timeout_s=0)
        await second.shutdown(timeout_s=0)
