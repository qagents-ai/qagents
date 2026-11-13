"""Store identity, migration and clock boundaries independent of the effect engine."""

from __future__ import annotations

import asyncio
import sqlite3
import threading

import pytest

from qagents.errors import EntryConflict, LeaseError, SessionError
from qagents.session import Entry, InMemoryStore, SQLiteStore


@pytest.fixture(params=["memory", "sqlite"])
async def store(request, tmp_path):
    clock = [100.0]
    if request.param == "memory":
        made = InMemoryStore(clock=lambda: clock[0])
    else:
        made = SQLiteStore(tmp_path / "store.db", clock=lambda: clock[0])
    yield made, clock
    await made.aclose()


async def test_ids_precede_lease_and_partial_overlap_is_atomic(store):
    db, _ = store
    batch = [Entry(kind="custom", data={"value": 1}), Entry(kind="custom")]
    assert await db.acquire_lease("s", "old", 10)
    assert await db.append_leased("s", batch, "old") == [1, 2]
    await db.release_lease("s", "old")
    assert await db.acquire_lease("s", "new", 10)
    assert await db.append_leased("s", batch, "old") == [1, 2]
    with pytest.raises(EntryConflict):
        await db.append_leased("s", [batch[0], Entry(kind="custom")], "old")
    with pytest.raises(LeaseError):
        await db.append_leased("s", [Entry(kind="custom")], "old")
    assert [e.id for e in await db.load("s")] == [e.id for e in batch]


async def test_duplicate_ids_inside_batch_are_refused_without_writing(store):
    db, _ = store
    entry = Entry(kind="custom")
    with pytest.raises(EntryConflict):
        await db.append("s", [entry, entry])
    assert await db.load("s") == []


async def test_invalid_later_entry_does_not_publish_earlier_entry(store):
    db, _ = store
    with pytest.raises(SessionError):
        await db.append("s", [Entry(kind="custom"), Entry(kind="bad", data={"bad": object()})])
    assert await db.load("s") == []
    assert (await db.status("s")).updated_at is None


async def test_cancelled_sqlite_call_is_decided_before_return(tmp_path):
    entered, release = threading.Event(), threading.Event()

    class Blocked(SQLiteStore):
        def _append(self, *args):
            entered.set()
            assert release.wait(5)
            return super()._append(*args)

    async with Blocked(tmp_path / "cancel.db") as db:
        entry = Entry(kind="custom")
        writing = asyncio.create_task(db.append("s", [entry]))
        try:
            await asyncio.to_thread(entered.wait, 2)
            assert entered.is_set()
            writing.cancel()
            await asyncio.sleep(0)
            writing.cancel()  # repeated cancellation must not detach a transaction
            await asyncio.sleep(0)
            assert not writing.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await writing
        assert [row.id for row in await db.load("s")] == [entry.id]


async def test_sqlite_reads_clock_after_waiting_for_write_lock(tmp_path):
    entered, clock = threading.Event(), [100.0]

    class Waiting(SQLiteStore):
        def _acquire(self, *args):
            entered.set()
            return super()._acquire(*args)

    async with Waiting(tmp_path / "clock.db", clock=lambda: clock[0]) as db:
        db._lock.acquire()
        try:
            task = asyncio.create_task(db.acquire_lease_grant("s", "owner", 30))
            assert await asyncio.to_thread(entered.wait, 2)
            clock[0] = 150
        finally:
            db._lock.release()
        grant = await task
        assert grant.acquired_at == 150 and grant.expires_at == 180


async def test_two_sqlite_connections_replay_one_atomic_batch(tmp_path):
    path = tmp_path / "two-writers.db"
    async with SQLiteStore(path) as one, SQLiteStore(path) as two:
        await one.acquire_lease("s", "owner", 30)
        batch = [Entry(kind="custom"), Entry(kind="custom")]
        assert await asyncio.gather(
            one.append_leased("s", batch, "owner"), two.append_leased("s", batch, "owner")
        ) == [[1, 2], [1, 2]]
        assert len(await two.load("s")) == 2


async def test_memory_and_sqlite_do_not_retain_caller_mutations(store):
    db, _ = store
    entry = Entry(kind="custom", data={"nested": [1]})
    await db.append("s", [entry])
    entry.data["nested"].append(2)
    loaded = await db.load("s")
    assert loaded[0].data == {"nested": [1]}
    loaded[0].data["nested"].append(3)
    assert (await db.load("s"))[0].data == {"nested": [1]}


async def test_clock_grace_and_epoch_survive_delete(store):
    db, clock = store
    first = await db.acquire_lease_grant("s", "owner-1", 10)
    assert first.epoch == 1 and first.acquired_at == 100 and first.expires_at == 110
    clock[0] = 110.5
    assert await db.acquire_lease_grant("s", "owner-2", 10) is None
    renewed = await db.acquire_lease_grant("s", "owner-1", 10)
    assert renewed.epoch == 1 and renewed.acquired_at == 100
    clock[0] = 121.5
    second = await db.acquire_lease_grant("s", "owner-2", 10)
    assert second.epoch == 2 and second.acquired_at == clock[0]
    await db.delete("s")
    third = await db.acquire_lease_grant("s", "owner-3", 10)
    assert third.epoch == 3
    await db.append_leased("s", [Entry(kind="custom")], "owner-3")
    assert (await db.status("s")).updated_at == await db.now() == clock[0]


async def test_sqlite_v1_migration_keeps_entries_and_stable_ids(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE entries (session_id TEXT, seq INTEGER, ts REAL, run_id TEXT,
                kind TEXT, data TEXT, PRIMARY KEY(session_id, seq));
            CREATE TABLE sessions (session_id TEXT PRIMARY KEY, status TEXT DEFAULT 'idle',
                open_run_id TEXT, agent TEXT, failures INTEGER DEFAULT 0, updated_at REAL,
                lease_owner TEXT, lease_expires_at REAL);
            INSERT INTO entries VALUES ('s', 1, 12, 'r', 'run_started', '{"agent":"agent"}');
            INSERT INTO sessions VALUES ('s', 'running', 'r', 'agent', 0, 12, 'legacy-owner', 500);
            PRAGMA user_version = 1;
        """)
    async with SQLiteStore(path) as db:
        renewed = await db.acquire_lease_grant("s", "legacy-owner", 30)
        assert renewed.epoch == 1 and renewed.acquired_at is not None
        old = await db.load("s")
        assert old[0].id == "seq:1" and old[0].data == {"agent": "agent"}
        assert await db.append("s", old) == [1]
        assert await db.append("s", [Entry(kind="custom")]) == [2]
        assert (await db.status("s")).open_run_id == "r"
    async with SQLiteStore(path) as db:
        assert (await db.load("s"))[0].id == "seq:1"
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (2,)
