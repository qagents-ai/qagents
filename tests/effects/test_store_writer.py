"""Lease lifetime includes sends running on threads, even after repeated cancellation."""

import asyncio
import concurrent.futures

import pytest

from qagents.effects._ports import Timing
from qagents.errors import JournalOutcomeUnknown, SessionError
from qagents.session import Entry, InMemoryStore, Lease, LeasedWriter, SessionStore, SQLiteStore

from ._samples import failed, intent


def writer(db, **kwargs):
    return LeasedWriter(
        db, "s", worker="worker", ttl_s=kwargs.pop("ttl_s", 30), purpose="run", **kwargs
    )


async def test_writer_acquires_before_read_and_first_write_has_one_acquisition(tmp_path):
    async with SQLiteStore(tmp_path / "writer.db") as db:
        one = writer(db, journal_acquisition=True)
        with pytest.raises(SessionError):
            await one.load()
        assert await one.open()
        assert not await writer(db).open()
        assert await db.load("s") == []  # acquisition is journaled with the first write
        assert await one.append([Entry(kind="run_started", run_id="r")]) == [2]
        assert await one.append([Entry(kind="custom")]) == [3]
        entries = await db.load("s")
        assert [e.kind for e in entries] == ["lease_acquired", "run_started", "custom"]
        assert one.view().acquisitions[0][0] == 1
        await one.close()
        two = writer(db)
        assert await two.open()
        await two.load()
        await two.append([Entry(kind="custom")])
        assert [e.data["epoch"] for e in await db.load("s") if e.kind == "lease_acquired"] == [1, 2]
        await two.close()


async def test_cancel_close_twice_never_releases_before_thread_send_drains():
    class Counted(InMemoryStore):
        renewals = 0

        async def renew_lease(self, *args):
            self.renewals += 1
            return await super().renew_lease(*args)

    db = Counted(takeover_grace_s=0)
    one = writer(db, ttl_s=0.12)
    assert await one.open()
    pending = concurrent.futures.Future()
    one.track(pending)
    task = asyncio.create_task(one.close())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    again = asyncio.create_task(one.close())
    await asyncio.sleep(0)
    again.cancel()
    with pytest.raises(asyncio.CancelledError):
        await again
    await asyncio.sleep(0.16)
    assert db.renewals >= 2
    assert (await db.status("s")).lease_owner == one.lease.owner
    assert not await db.acquire_lease("s", "competitor", 30)
    await asyncio.to_thread(pending.set_result, "sent")
    await one.close()
    assert one.lease.running_sends() == 0
    assert await db.acquire_lease("s", "competitor", 30)


async def test_hanging_renewal_self_fences_on_local_deadline():
    released, lost = asyncio.Event(), asyncio.Event()

    class Hung(InMemoryStore):
        async def renew_lease(self, *args):
            await released.wait()
            return await super().renew_lease(*args)

    db = Hung()
    lease = Lease(
        db,
        "s",
        worker="w",
        ttl_s=0.06,
        purpose="run",
        timing=Timing.for_lease(0.06, 1, effects=False),
    )
    assert await lease.acquire()
    keeping = asyncio.create_task(lease.keep(on_lost=lambda _: lost.set()))
    try:
        await asyncio.wait_for(lost.wait(), 1)
        assert lease.lost()
    finally:
        released.set()
        await keeping
        await lease.release()


async def test_plain_custom_store_and_02_leases_remain_usable():
    class Plain(SessionStore):
        def __init__(self):
            self.entries = []

        async def append(self, session_id, entries):
            self.entries.extend(entries)

        async def load(self, session_id):
            return self.entries[:]

        async def delete(self, session_id):
            self.entries.clear()

    db = Plain()
    one = writer(db)
    assert await one.open() and not one.leased
    assert await one.append([Entry(kind="custom")]) is None
    with pytest.raises(SessionError):
        await one.append_effect([Entry(kind="custom")])
    await one.close()

    class Legacy(InMemoryStore):
        features = frozenset()

        async def acquire_lease(self, session_id, owner, ttl_s):
            return await InMemoryStore.acquire_lease(self, session_id, owner, ttl_s)

    db = Legacy()
    one = writer(db)
    assert await one.open() and one.leased and one.lease.epoch is None
    await one.append([Entry(kind="custom")])
    assert [e.kind for e in await db.load("s")] == ["custom"]
    await one.close()


async def test_unreadable_uncertain_write_poisons_writer():
    class Unreadable(InMemoryStore):
        async def append_leased(self, *args):
            raise OSError("ack unknown")

        async def load(self, *args):
            raise OSError("unreadable")

    one = writer(Unreadable())
    assert await one.open()
    with pytest.raises(JournalOutcomeUnknown):
        await one.append_effect([Entry(kind="custom")])
    assert one.poisoned()
    with pytest.raises(JournalOutcomeUnknown):
        await one.append([Entry(kind="custom")])
    await one.close()


async def test_cancelled_ack_updates_first_write_before_a_later_write():
    class CancelOnce(InMemoryStore):
        calls = 0

        async def append_leased(self, *args):
            result = await super().append_leased(*args)
            self.calls += 1
            if self.calls == 1:
                raise asyncio.CancelledError()
            return result

    db = CancelOnce()
    one = writer(db, journal_acquisition=True)
    assert await one.open()
    with pytest.raises(asyncio.CancelledError):
        await one.append([Entry(kind="custom")])
    assert len(one.view().acquisitions) == 1
    assert await one.append([Entry(kind="custom")]) == [3]
    await one.close()


async def test_close_keeps_renewing_a_lost_lease_until_existing_send_finishes():
    lost = asyncio.Event()

    class FailOnce(InMemoryStore):
        renewals = 0

        async def renew_lease(self, *args):
            self.renewals += 1
            if self.renewals == 1:
                raise OSError("temporary renewal failure")
            return await super().renew_lease(*args)

    db = FailOnce(takeover_grace_s=0)
    one = writer(db, ttl_s=0.12)
    await one.open()
    sending = concurrent.futures.Future()
    one.track(sending)
    keeping = asyncio.create_task(one.keep(on_lost=lambda _: lost.set()))
    await asyncio.wait_for(lost.wait(), 1)
    keeping.cancel()
    await asyncio.gather(keeping, return_exceptions=True)
    closing = asyncio.create_task(one.close())
    try:
        await asyncio.sleep(0.08)
        assert db.renewals >= 2
        assert one.lease.lost()  # renewal cannot authorize fresh sends again
        assert (await db.status("s")).lease_owner == one.lease.owner
    finally:
        sending.set_result(None)
        await closing


async def test_cancelled_renewal_notifies_the_owner():
    lost = asyncio.Event()

    class CancelRenew(InMemoryStore):
        async def renew_lease(self, *args):
            raise asyncio.CancelledError()

    one = writer(CancelRenew(), ttl_s=0.06)
    await one.open()
    keeping = asyncio.create_task(one.keep(on_lost=lambda _: lost.set()))
    try:
        await asyncio.wait_for(lost.wait(), 0.5)
        assert one.lease.lost()
    finally:
        keeping.cancel()
        await asyncio.gather(keeping, return_exceptions=True)
        await one.close()


async def test_late_completion_watcher_journals_before_close_releases_lease():
    db = InMemoryStore()
    one = writer(db, journal_acquisition=True)
    await one.open()
    seqs = await one.append_effect([Entry(kind="effect_intent", data=intent())])
    send = asyncio.get_running_loop().create_future()

    async def watch():
        await send
        assert (await db.status("s")).lease_owner == one.lease.owner
        await one.append_effect([Entry(kind="effect_failed", data=failed(intent_seq=seqs[0]))])
        assert (await db.status("s")).lease_owner == one.lease.owner

    watching = asyncio.create_task(watch())
    one.track(send)
    one.track(watching)  # O2 must track the complete watcher, including its journal append.
    closing = asyncio.create_task(one.close())
    await asyncio.sleep(0)
    with pytest.raises(SessionError):
        await one.append_effect([Entry(kind="effect_intent", data=intent(key="other"))])
    with pytest.raises(SessionError):
        await one.append_effect([Entry(kind="effect_failed", data=failed(intent_seq=999))])
    with pytest.raises(SessionError):
        await one.append([Entry(kind="custom")])
    send.set_result("finished")
    await watching
    await closing
    assert [e.kind for e in await db.load("s")] == [
        "lease_acquired",
        "effect_intent",
        "effect_failed",
    ]
    assert (await db.status("s")).lease_owner is None
