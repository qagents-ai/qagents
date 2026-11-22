"""Read-back resolves uncertain writes before callers act on their outcome."""

import asyncio

import pytest

from qagents.errors import JournalOutcomeUnknown, LeaseError
from qagents.session import Entry, InMemoryStore, append_fenced


async def test_committed_write_replays_same_ids_after_lost_ack_and_lease():
    class LostAck(InMemoryStore):
        calls = 0

        async def append_leased(self, session_id, entries, owner):
            self.calls += 1
            result = await super().append_leased(session_id, entries, owner)
            if self.calls == 1:
                await self.release_lease(session_id, owner)
                await self.acquire_lease(session_id, "new-owner", 30)
                raise OSError("lost acknowledgement")
            return result

    db = LostAck()
    await db.acquire_lease("s", "old-owner", 30)
    entries = [Entry(kind="custom"), Entry(kind="custom")]
    assert await append_fenced(db, "s", entries, "old-owner", fence=lambda: True) == [1, 2]
    assert db.calls == 2
    assert [e.id for e in await db.load("s")] == [e.id for e in entries]


async def test_definite_first_refusal_is_not_retried():
    class Refused(InMemoryStore):
        calls = 0

        async def append_leased(self, *args):
            self.calls += 1
            raise LeaseError("not held")

    db = Refused()
    with pytest.raises(LeaseError):
        await append_fenced(db, "s", [Entry(kind="custom")], "owner", fence=lambda: True)
    assert db.calls == 1


async def test_retry_refusal_still_reads_back_the_original_write():
    class RefusedRetry(InMemoryStore):
        calls = 0

        async def append(self, session_id, entries):
            self.calls += 1
            if self.calls == 1:
                await super().append(session_id, entries)
                raise OSError("ack lost")
            raise LeaseError("later refusal")

    db = RefusedRetry()
    assert await append_fenced(db, "s", [Entry(kind="custom")], None, fence=lambda: True) == [1]
    assert db.calls == 4


async def test_no_retry_after_fence_and_read_failure_is_unknown():
    class Unreadable(InMemoryStore):
        calls = 0

        async def append(self, *args):
            self.calls += 1
            raise OSError("uncertain")

        async def load(self, *args):
            raise OSError("unreadable")

    db = Unreadable()
    with pytest.raises(JournalOutcomeUnknown):
        await append_fenced(db, "s", [Entry(kind="custom")], None, fence=lambda: False)
    assert db.calls == 1


async def test_absent_readback_preserves_failure_and_writes_nothing():
    class NoWrite(InMemoryStore):
        async def append(self, *args):
            raise OSError("definitely absent after readback")

    db = NoWrite()
    with pytest.raises(OSError):
        await append_fenced(db, "s", [Entry(kind="custom")], None, fence=lambda: False)
    assert await db.load("s") == []


async def test_cancelled_write_resolves_readback_despite_second_cancel():
    reading, finish = asyncio.Event(), asyncio.Event()

    class CancelledAck(InMemoryStore):
        async def append(self, session_id, entries):
            await super().append(session_id, entries)
            raise asyncio.CancelledError()

        async def load(self, session_id):
            reading.set()
            await finish.wait()
            return await super().load(session_id)

    db = CancelledAck()
    task = asyncio.create_task(
        append_fenced(db, "s", [Entry(kind="custom")], None, fence=lambda: False)
    )
    await reading.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(await db.load("s")) == 1
