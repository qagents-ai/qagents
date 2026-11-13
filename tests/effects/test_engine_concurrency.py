"""Concurrent callers share work while cancellation preserves its bounded outcome."""

import asyncio
import concurrent.futures
import threading
from types import SimpleNamespace

import pytest

from qagents.effects import _engine as engine
from qagents.effects._canonical import Descriptor
from qagents.errors import EffectConflictError
from qagents.session import LeasedWriter

from ._support import MemoryJournal


def scope(journal=None):
    return SimpleNamespace(journal=journal or MemoryJournal(), session_id="session")


def prepared(*, digest="request", namespace="shop", key="R1"):
    return SimpleNamespace(
        namespace=namespace,
        key=key,
        descriptor=Descriptor("refund", "1", "destination", digest),
    )


async def test_single_flight_shares_result_and_does_not_cache_finished_work():
    host = scope()
    started, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def work():
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return "receipt"

    first = asyncio.create_task(engine._single_flight(prepared(), host, work))
    await started.wait()
    second = asyncio.create_task(engine._single_flight(prepared(), host, work))
    await asyncio.sleep(0)
    release.set()
    assert await asyncio.gather(first, second) == ["receipt", "receipt"]
    assert calls == 1
    assert await engine._single_flight(prepared(), host, work) == "receipt"
    assert calls == 2
    assert id(host.journal.store) not in engine._WORK


async def test_single_flight_rejects_a_different_descriptor_without_starting_it():
    host = scope()
    started, release = asyncio.Event(), asyncio.Event()

    async def work():
        started.set()
        await release.wait()
        return 1

    first = asyncio.create_task(engine._single_flight(prepared(), host, work))
    await started.wait()
    try:
        with pytest.raises(EffectConflictError) as error:
            await engine._single_flight(prepared(digest="changed"), host, work)
        assert error.value.effect == "refund" and error.value.existing_status == "held"
    finally:
        release.set()
        assert await first == 1


@pytest.mark.parametrize("boundary", ["store", "session", "namespace", "key"])
async def test_single_flight_isolated_by_full_identity(boundary):
    one, two = scope(), scope()
    if boundary != "store":
        two.journal.store = one.journal.store
    if boundary == "session":
        two.session_id = "other"
    value = prepared(**({boundary: "other"} if boundary in ("namespace", "key") else {}))
    started, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def work():
        nonlocal calls
        calls += 1
        if calls == 2:
            started.set()
        await release.wait()
        return calls

    first = asyncio.create_task(engine._single_flight(prepared(), one, work))
    second = asyncio.create_task(engine._single_flight(value, two, work))
    try:
        await asyncio.wait_for(started.wait(), 1)
    finally:
        release.set()
        await asyncio.gather(first, second)
    assert calls == 2


async def test_cancelling_a_joiner_waits_for_outcome_without_cancelling_shared_work():
    host = scope()
    started, release = asyncio.Event(), asyncio.Event()
    outcome = []

    async def work():
        started.set()
        await release.wait()
        outcome.append("journaled")
        return 7

    first = asyncio.create_task(engine._single_flight(prepared(), host, work))
    await started.wait()
    second = asyncio.create_task(engine._single_flight(prepared(), host, work))
    await asyncio.sleep(0)
    second.cancel()
    await asyncio.sleep(0)
    second.cancel()
    await asyncio.sleep(0)
    assert not second.done() and not first.done()
    release.set()
    assert await first == 7
    with pytest.raises(asyncio.CancelledError):
        await second
    assert outcome == ["journaled"]
    await host.journal.lease.sends_drained()


async def test_shared_failure_reaches_both_callers_and_releases_registry():
    host = scope()
    started, release = asyncio.Event(), asyncio.Event()

    async def work():
        started.set()
        await release.wait()
        raise ValueError("lost response")

    first = asyncio.create_task(engine._single_flight(prepared(), host, work))
    await started.wait()
    second = asyncio.create_task(engine._single_flight(prepared(), host, work))
    await asyncio.sleep(0)
    release.set()
    outcomes = await asyncio.gather(first, second, return_exceptions=True)
    assert all(isinstance(item, ValueError) for item in outcomes)
    assert id(host.journal.store) not in engine._WORK


async def test_single_flight_shares_across_event_loops_on_different_threads():
    first_host, other_host = scope(), scope()
    other_host.journal.store = first_host.journal.store
    started, release = asyncio.Event(), asyncio.Event()
    joined = threading.Event()
    calls = []

    async def work():
        calls.append(threading.get_ident())
        started.set()
        await release.wait()
        return "shared"

    async def other_loop():
        task = asyncio.create_task(engine._single_flight(prepared(), other_host, work))
        await asyncio.sleep(0)
        joined.set()
        return await task

    first = asyncio.create_task(engine._single_flight(prepared(), first_host, work))
    await started.wait()
    second = asyncio.create_task(asyncio.to_thread(lambda: asyncio.run(other_loop())))
    try:
        assert await asyncio.to_thread(joined.wait, 1)
    finally:
        release.set()
        results = await asyncio.gather(first, second)
    assert results == ["shared", "shared"]
    assert calls == [threading.get_ident()]


async def test_inner_cancellation_does_not_spin_forever_in_shield():
    async def work():
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await engine._shield(work(), scope=scope())


async def test_cancelled_caller_still_waits_for_failure_to_be_accounted():
    host = scope()
    started, release = asyncio.Event(), asyncio.Event()
    accounted = []

    async def work():
        started.set()
        await release.wait()
        accounted.append("held")
        raise ValueError("unknown")

    task = asyncio.create_task(engine._shield(work(), scope=host))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert accounted == ["held"]


def test_abandoned_send_tracking_uses_store_session_record_and_actual_completion():
    first, second = MemoryJournal(), MemoryJournal()
    pending = concurrent.futures.Future()
    engine._track_abandoned(first.store, "s", "record", pending)
    assert engine.send_running(first.store, "s", "record")
    assert not engine.send_running(second.store, "s", "record")
    assert not engine.send_running(first.store, "other", "record")
    assert not engine.send_running(first.store, "s", "other")
    pending.set_result(None)
    assert not engine.send_running(first.store, "s", "record")
    assert id(first.store) not in engine._WORK


def test_forget_store_does_not_cancel_a_live_sender_or_remove_a_new_registry():
    host = MemoryJournal()
    old, new = concurrent.futures.Future(), concurrent.futures.Future()
    engine._track_abandoned(host.store, "s", "record", old)
    engine._forget_store(host.store)
    assert not old.cancelled() and not engine.send_running(host.store, "s", "record")
    engine._track_abandoned(host.store, "s", "record", new)
    old.set_result(None)
    assert engine.send_running(host.store, "s", "record")
    new.set_result(None)
    assert not engine.send_running(host.store, "s", "record")


def test_library_writer_wrappers_share_the_root_send_registry():
    host = MemoryJournal()
    writer = LeasedWriter(host.store, "s", worker="one", ttl_s=30, purpose="run")
    pending = concurrent.futures.Future()
    engine._track_abandoned(writer, "s", "record", pending)
    assert engine.send_running(host.store, "s", "record")
    assert engine.send_running(writer, "s", "record")
    engine._forget_store(writer)
    assert not engine.send_running(host.store, "s", "record")
    pending.set_result(None)
