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

from qagents import Agent, RunContext, Session, SQLiteStore, tool
from qagents.errors import ConfigurationError, SessionError
from qagents.models.testing import ScriptedModel, call
from qagents.runtime import AgentRuntime, RunHandle, RunInterrupted
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
    assert lease.lease_owner == runtime.worker_id and lease.lease_active()

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

    def store(self) -> SessionStore:
        if self.kind == "memory":
            self._memory = self._memory or InMemoryStore()
            return self._memory
        if self.kind == "sqlite":
            store: SessionStore = SQLiteStore(self.path)
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


async def crash_worker(store: SessionStore, log: list[str]) -> AgentRuntime:
    """Worker A starts a run and dies while the receipt is being sent."""
    gate, started = asyncio.Event(), asyncio.Event()
    connection = CrashableStore(store)
    worker = AgentRuntime(
        connection,
        [receipt_agent(log, gate, started)],
        worker_id="worker-a",
        deps_factory=lambda session_id: "A",
        **FAST,
    )
    handle = await worker.start("billing", "charge and send the receipt", session_id="order-7")
    await asyncio.wait_for(started.wait(), 5)
    connection.crash()  # the process dies: no more writes, no release, no heartbeat
    assert handle._task is not None
    handle._task.cancel()
    with pytest.raises(RunInterrupted):
        await handle.wait()
    return worker


def recovering_worker(store: SessionStore, log: list[str], name: str) -> AgentRuntime:
    gate, started = asyncio.Event(), asyncio.Event()
    gate.set()
    return AgentRuntime(
        store,
        [receipt_agent(log, gate, started, replies=["The card was charged and the receipt sent."])],
        worker_id=name,
        deps_factory=lambda session_id: name,
        **FAST,
    )


def test_a_dead_workers_run_is_recovered_without_repeating_finished_tools(backend):
    async def check() -> None:
        log: list[str] = []
        await crash_worker(backend.store(), log)
        store = backend.store()
        status = await store.status("order-7")
        assert (status.status, status.lease_owner) == ("running", "worker-a")

        worker = recovering_worker(store, log, "worker-b")
        assert status.lease_expires_at is not None
        if status.lease_expires_at - time.time() > 0.15:  # store clock; Postgres may differ
            assert await worker.recover() == []  # the lease has not expired yet
        await asyncio.sleep(FAST["lease_ttl_s"] + 0.1)
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
        await crash_worker(backend.store(), log)
        await asyncio.sleep(FAST["lease_ttl_s"] + 0.1)
        workers = [recovering_worker(backend.store(), log, f"worker-{n}") for n in "bcd"]
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
    await store.append(
        "workflow:report:research", [started.model_copy(update={"data": {"agent": "a"}})]
    )
    runtime = AgentRuntime(store, [Agent(ScriptedModel([]), name="a")])
    with caplog.at_level(logging.WARNING, logger="qagents.runtime"):
        assert await runtime.recover() == []
    messages = [r.message for r in caplog.records]
    assert any("'foreign'" in m and "agent 'elsewhere'" in m for m in messages)
    assert not any("workflow:" in m for m in messages)
    await runtime.shutdown()


async def test_session_prefix_limits_recovery():
    store = InMemoryStore()
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
    assert await store.load("q") == []  # the queued run never started
    assert (await store.status("q")).lease_owner is None

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
    assert await store.load("g") == [] and (await store.status("g")).lease_owner is None
    assert worker.handles == {}


async def test_a_worker_that_loses_its_lease_stops_and_cannot_write():
    store = InMemoryStore()
    log: list[str] = []
    gate, started = asyncio.Event(), asyncio.Event()
    worker = AgentRuntime(store, [receipt_agent(log, gate, started)], worker_id="w1", **FAST)
    handle = await worker.start("billing", "go", session_id="s")
    await asyncio.wait_for(started.wait(), 5)

    # w1 stalls past its lease; w2 takes the session over.
    await store.release_lease("s", "w1")
    assert await store.acquire_lease("s", "w2", 30)
    entries = len(await store.load("s"))
    with pytest.raises(RunInterrupted, match="lost its lease"):
        await asyncio.wait_for(handle.wait(), 5)
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
    await store.release_lease("s", "w1")
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
