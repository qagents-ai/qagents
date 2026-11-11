"""Run status and leases: the optional store capabilities AgentRuntime builds on.

Every test runs against InMemoryStore, SQLiteStore and, when
QAGENTS_TEST_POSTGRES_DSN is set, PostgresStore. The tests are plain functions
that drive their own selector event loop, because psycopg's async mode cannot
use the proactor loop that asyncio uses on Windows.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import uuid
from collections.abc import Awaitable, Callable

import pytest

from qagents import Agent, Session, SQLiteStore, tool
from qagents.errors import SessionError
from qagents.models.testing import ScriptedModel, call
from qagents.session import (
    Entry,
    InMemoryStore,
    LeaseError,
    SessionStatus,
    SessionStore,
    advance_status,
)

DSN = os.environ.get("QAGENTS_TEST_POSTGRES_DSN")
BACKENDS = [
    "memory",
    "sqlite",
    pytest.param(
        "postgres",
        marks=pytest.mark.skipif(not DSN, reason="set QAGENTS_TEST_POSTGRES_DSN to test Postgres"),
    ),
]


def run(coroutine: Awaitable[None]) -> None:
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
        runner.run(coroutine)  # type: ignore[arg-type]


class Backend:
    """Makes store instances over one backing database, like separate processes would."""

    def __init__(self, kind: str, path: str) -> None:
        self.kind = kind
        self.path = path
        self.schema = f"qagents_test_{uuid.uuid4().hex[:12]}"
        self._memory: InMemoryStore | None = None
        self._opened: list[SessionStore] = []

    def store(self) -> SessionStore:
        if self.kind == "memory":  # one process: every "instance" is the same store
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
                self._opened.clear()

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


def approval_tools(log: list[str]):
    @tool(requires_approval=True)
    def refund(order: str) -> str:
        """Refund an order."""
        log.append(f"refund {order}")
        return "refunded"

    return [refund]


# -- run status ------------------------------------------------------------------------


def test_status_follows_runs_pauses_and_failures(backend):
    async def check() -> None:
        store = backend.store()
        assert await store.status("s") == SessionStatus(session_id="s")

        log: list[str] = []
        model = ScriptedModel([call("refund", order="A-1"), "Refunded."])
        agent = Agent(model, name="support", tools=approval_tools(log))
        paused = await agent.run("Refund A-1", session=Session(store, id="s"))
        status = await store.status("s")
        assert paused.status == "paused"
        assert (status.status, status.open_run_id, status.agent) == (
            "paused",
            paused.run_id,
            "support",
        )
        assert status.updated_at is not None and status.failures == 0

        done = await agent.resume(
            Session(backend.store(), id="s"),
            approvals={paused.pending_approvals[0].tool_call_id: True},
        )
        assert done.status == "completed" and log == ["refund A-1"]
        status = await store.status("s")
        assert (status.status, status.open_run_id, status.agent) == ("idle", None, "support")

        broken = Agent(ScriptedModel([RuntimeError("provider down")]), name="support")
        with pytest.raises(RuntimeError):
            await broken.run("Hello again", session=Session(store, id="s"))
        status = await store.status("s")
        assert (status.status, status.failures) == ("failed", 1)
        assert status.open_run_id is not None

        with pytest.raises(RuntimeError):
            await Agent(ScriptedModel([RuntimeError("still down")]), name="support").resume(
                Session(store, id="s")
            )
        assert (await store.status("s")).failures == 2

        finished = await Agent(ScriptedModel(["Back."]), name="support").resume(
            Session(store, id="s")
        )
        assert finished.status == "completed"
        status = await store.status("s")
        assert (status.status, status.failures, status.open_run_id) == ("idle", 0, None)

    backend.run(check)


def test_a_crashed_run_stays_running(backend):
    async def check() -> None:
        store = backend.store()
        started, gate = asyncio.Event(), asyncio.Event()

        @tool
        async def slow() -> str:
            """Take a while."""
            started.set()
            await gate.wait()
            return "done"

        agent = Agent(ScriptedModel([call("slow"), "ok"]), name="worker", tools=[slow])
        task = asyncio.create_task(agent.run("go", session=Session(store, id="c")))
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        status = await backend.store().status("c")
        assert status.status == "running" and status.open_run_id is not None

    backend.run(check)


def test_list_sessions_filters_and_orders(backend):
    async def check() -> None:
        store = backend.store()
        for session_id, kind in [("b-1", "run_paused"), ("a-1", "run_finished"), ("a-2", None)]:
            entries = [Entry(kind="run_started", run_id="r", data={"agent": "x"})]
            if kind is not None:
                entries.append(Entry(kind=kind, run_id="r"))
            await store.append(session_id, entries)
            await asyncio.sleep(0.01)  # distinct updated_at
        await store.acquire_lease("lease-only", "w", 30)  # no entries: not listed

        listed = await store.list_sessions()
        assert [item.session_id for item in listed] == ["b-1", "a-1", "a-2"]  # oldest update first
        assert [item.status for item in listed] == ["paused", "idle", "running"]
        assert [s.session_id for s in await store.list_sessions(status="running")] == ["a-2"]
        both = await store.list_sessions(status={"running", "paused"})
        assert [s.session_id for s in both] == ["b-1", "a-2"]
        assert [s.session_id for s in await store.list_sessions(prefix="a-")] == ["a-1", "a-2"]
        assert [s.session_id for s in await store.list_sessions(limit=1)] == ["b-1"]
        assert await store.list_sessions(prefix="a-", limit=0) == []

        with pytest.raises(ValueError, match="Unknown session status"):
            await store.list_sessions(status="crashed")
        with pytest.raises(ValueError, match="limit"):
            await store.list_sessions(limit=-1)

    backend.run(check)


def test_delete_removes_status_and_lease(backend):
    async def check() -> None:
        store = backend.store()
        await store.append("d", [Entry(kind="run_started", run_id="r", data={"agent": "x"})])
        assert await store.acquire_lease("d", "w1", 30)
        await store.delete("d")
        assert await store.load("d") == []
        assert await store.status("d") == SessionStatus(session_id="d")
        assert await store.list_sessions() == []

    backend.run(check)


def test_seq_stays_gap_free_and_status_matches_after_many_appends(backend):
    async def check() -> None:
        store = backend.store()
        await store.append("g", [Entry(kind="run_started", run_id="r1", data={"agent": "x"})])
        await asyncio.gather(
            *[store.append("g", [Entry(kind="note", data={"i": i})] * 3) for i in range(10)]
        )
        entries = await backend.store().load("g")
        assert [entry.seq for entry in entries] == list(range(1, 32))
        notes = [entry.data["i"] for entry in entries[1:]]
        assert all(notes[i] == notes[i + 1] == notes[i + 2] for i in range(0, 30, 3))  # batches
        assert (await store.status("g")).status == "running"

    backend.run(check)


# -- leases ----------------------------------------------------------------------------


def test_lease_acquire_renew_expire_and_release(backend):
    async def check() -> None:
        store = backend.store()
        assert await store.acquire_lease("l", "w1", 30)
        assert not await store.acquire_lease("l", "w2", 30)
        assert await store.acquire_lease("l", "w1", 30)  # the owner renews
        assert await store.renew_lease("l", "w1", 30)
        assert not await store.renew_lease("l", "w2", 30)
        status = await backend.store().status("l")
        assert status.lease_owner == "w1" and status.lease_active()
        assert status.updated_at is None  # a lease alone does not make entries

        await store.release_lease("l", "w2")  # not the owner: nothing happens
        assert (await store.status("l")).lease_owner == "w1"
        await store.release_lease("l", "w1")
        status = await store.status("l")
        assert (status.lease_owner, status.lease_expires_at) == (None, None)
        assert not status.lease_active()
        assert not await store.renew_lease("l", "w1", 30)

        assert await store.acquire_lease("l", "w1", 0.2)
        await asyncio.sleep(0.4)
        assert not (await store.status("l")).lease_active()
        assert await store.renew_lease("l", "w1", 0.2)  # expired but not taken: still w1's
        await asyncio.sleep(0.4)
        assert await store.acquire_lease("l", "w2", 30)  # expired: anyone may take it
        assert not await store.renew_lease("l", "w1", 30)
        assert (await store.status("l")).lease_owner == "w2"

    backend.run(check)


def test_concurrent_acquires_have_exactly_one_winner(backend):
    async def check() -> None:
        stores = [backend.store() for _ in range(3)]
        wins = await asyncio.gather(
            *[stores[i % 3].acquire_lease("race", f"w{i}", 30) for i in range(12)]
        )
        assert wins.count(True) == 1

    backend.run(check)


def test_append_leased_writes_only_for_the_lease_holder(backend):
    async def check() -> None:
        store = backend.store()
        started = Entry(kind="run_started", run_id="r", data={"agent": "x"})
        with pytest.raises(LeaseError, match="nobody holds it now"):
            await store.append_leased("f", [started], "w1")
        assert await store.load("f") == []

        assert await store.acquire_lease("f", "w1", 0.2)
        await store.append_leased("f", [started], "w1")
        with pytest.raises(LeaseError, match="worker 'w1' holds it now"):
            await store.append_leased("f", [Entry(kind="message", run_id="r")], "w2")

        await asyncio.sleep(0.4)
        assert await backend.store().acquire_lease("f", "w2", 30)  # w1 stalled; w2 took over
        with pytest.raises(LeaseError, match="no longer holds the lease"):
            await store.append_leased("f", [Entry(kind="run_finished", run_id="r")], "w1")
        await store.append_leased("f", [Entry(kind="run_paused", run_id="r")], "w2")
        assert [entry.kind for entry in await store.load("f")] == ["run_started", "run_paused"]
        assert (await store.status("f")).status == "paused"

    backend.run(check)


def test_lease_ttl_must_be_positive(backend):
    async def check() -> None:
        store = backend.store()
        for ttl in (0, -1, float("inf"), float("nan")):
            with pytest.raises(ValueError, match="ttl_s"):
                await store.acquire_lease("t", "w", ttl)
        with pytest.raises(ValueError, match="ttl_s"):
            await store.renew_lease("t", "w", 0)

    backend.run(check)


# -- pure functions and the base class ---------------------------------------------------


def test_advance_status_ignores_entries_of_other_runs():
    status = SessionStatus(session_id="s")
    for entry in [
        Entry(kind="workflow_step", data={"name": "x"}),  # no run id
        Entry(kind="run_started", run_id="new", data={"agent": "a"}),
        Entry(kind="run_finished", run_id="old", data={"status": "abandoned"}),  # late close
        Entry(kind="run_failed", run_id="new"),
        Entry(kind="tool_started", run_id="other"),
    ]:
        advance_status(status, entry)
    assert (status.status, status.open_run_id, status.failures) == ("failed", "new", 1)
    advance_status(status, Entry(kind="message", run_id="new"))  # resumed
    assert status.status == "running" and status.failures == 1
    advance_status(status, Entry(kind="run_finished", run_id="new"))
    assert (status.status, status.open_run_id, status.agent) == ("idle", None, "a")


async def test_a_store_without_the_optional_methods_says_what_to_do():
    class Minimal(SessionStore):
        async def append(self, session_id, entries):
            pass

        async def load(self, session_id):
            return []

        async def delete(self, session_id):
            pass

    store = Minimal()
    for attempt in (
        store.status("s"),
        store.list_sessions(),
        store.acquire_lease("s", "w", 1),
        store.renew_lease("s", "w", 1),
        store.release_lease("s", "w"),
        store.append_leased("s", [], "w"),
    ):
        with pytest.raises(NotImplementedError, match="Minimal does not support .*PostgresStore"):
            await attempt


# -- SQLite migration --------------------------------------------------------------------


def make_v01_database(path) -> None:
    """A database exactly as qagents 0.1 wrote it: the entries table only, user_version 0."""
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS entries ("
        " session_id TEXT NOT NULL, seq INTEGER NOT NULL, ts REAL NOT NULL,"
        " run_id TEXT, kind TEXT NOT NULL, data TEXT NOT NULL,"
        " PRIMARY KEY (session_id, seq))"
    )
    user = {"role": "user", "parts": [{"type": "text", "text": "Hi"}]}
    answer = {"role": "assistant", "parts": [{"type": "text", "text": "Hello"}]}
    rows = [
        ("done", 1, 10.0, "r1", "run_started", {"agent": "support"}),
        ("done", 2, 11.0, "r1", "message", {"message": user}),
        ("done", 3, 12.0, "r1", "message", {"message": answer}),
        ("done", 4, 13.0, "r1", "run_finished", {"status": "completed", "stop_reason": None}),
        ("crashed", 1, 20.0, "r2", "run_started", {"agent": "support"}),
        ("crashed", 2, 21.0, "r2", "message", {"message": user}),
        ("waiting", 1, 30.0, "r3", "run_started", {"agent": "billing"}),
        ("waiting", 2, 31.0, "r3", "run_paused", {"pending": []}),
        ("broken", 1, 40.0, "r4", "run_started", {"agent": "support"}),
        ("broken", 2, 41.0, "r4", "run_failed", {"error": "RuntimeError: x"}),
    ]
    connection.executemany(
        "INSERT INTO entries VALUES (?, ?, ?, ?, ?, ?)",
        [(s, seq, ts, run, kind, json.dumps(data)) for s, seq, ts, run, kind, data in rows],
    )
    connection.commit()
    connection.close()


def dump(path) -> list[tuple]:
    connection = sqlite3.connect(path)
    try:
        return connection.execute("SELECT * FROM entries ORDER BY session_id, seq").fetchall()
    finally:
        connection.close()


async def test_sqlite_migrates_a_v01_database_without_losing_entries(tmp_path):
    path = tmp_path / "old.db"
    make_v01_database(path)
    before = dump(path)

    store = SQLiteStore(path)
    statuses = {item.session_id: item for item in await store.list_sessions()}
    assert {key: item.status for key, item in statuses.items()} == {
        "done": "idle",
        "crashed": "running",
        "waiting": "paused",
        "broken": "failed",
    }
    assert statuses["crashed"].open_run_id == "r2" and statuses["crashed"].agent == "support"
    assert statuses["waiting"].agent == "billing" and statuses["broken"].failures == 1
    assert statuses["crashed"].updated_at == 21.0  # the time of its last entry
    await store.aclose()
    assert dump(path) == before  # no entry was changed

    connection = sqlite3.connect(path)
    assert connection.execute("PRAGMA user_version").fetchone() == (1,)
    connection.close()

    # The migrated database works as usual: resume the crashed run in "another process".
    store = SQLiteStore(path)
    result = await Agent(ScriptedModel(["Hello again"]), name="support").resume(
        Session(store, id="crashed")
    )
    assert result.status == "completed"
    assert (await store.status("crashed")).status == "idle"
    assert len(await store.load("done")) == 4
    await store.aclose()

    store = SQLiteStore(path)  # opening a migrated database again changes nothing
    assert len(await store.list_sessions()) == 4
    await store.aclose()


async def test_sqlite_refuses_a_database_from_a_newer_version(tmp_path):
    path = tmp_path / "new.db"
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA user_version = 99")
    connection.close()
    store = SQLiteStore(path)
    with pytest.raises(SessionError, match="newer qagents .*schema version 99.*Upgrade"):
        await store.load("x")
    await store.aclose()
