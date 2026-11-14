"""Crash-harness clock, scheduling, boundary and recovery contracts."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import json
import math
import socket
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import replace

import pytest
from pydantic import BaseModel

from qagents import tools as qtools
from qagents.agent import Agent, _append_overridden_alone
from qagents.effects import _ports, _types
from qagents.effects._canonical import digest, record_id
from qagents.effects._declare import Dedupe, Key, Unique, effect
from qagents.effects._types import Attempt, AttemptDebug, EffectContext, Found, Inquiry
from qagents.errors import ConfigurationError, EffectRejected, LeaseError, ToolError
from qagents.messages import Message
from qagents.models.base import ModelRequest
from qagents.models.recording import request_fingerprint
from qagents.models.testing import ScriptedModel
from qagents.session import Entry, SessionStore
from qagents.testing.crash._clock import ManualClock
from qagents.testing.crash._env import BuildIsolation, CrashEnv, GatedModel
from qagents.testing.crash._explore import (
    Point,
    boundary_points,
    concurrency_variants,
    recovery_points,
    reference_points,
    select_points,
)
from qagents.testing.crash._fakes import ContractFake, ContractRefusal, Fake, FakeStripeRefunds
from qagents.testing.crash._faults import FaultController, FaultPlan, Faults
from qagents.testing.crash._loop import NoTerminalState, VirtualLoop
from qagents.testing.crash._transport import HarnessTransport
from qagents.testing.crash._world import LEG, Boundary, LegCrashed, World, WorldSealed, WorldStore
from qagents.testing.stores import _intent as _store_probe_intent
from qagents.testing.stores import check_store


class _RefundRequest(BaseModel):
    payment_id: str
    amount: int


class _RefundReceipt(BaseModel):
    id: str
    payment_id: str
    amount: int


async def _unexercised_send(ctx: EffectContext, attempt: Attempt[_RefundRequest]) -> _RefundReceipt:
    raise AssertionError("contract-fake unit test must not execute the declared send")


def _refund_effect(contract=None, *, applies_within_s=None, rejected_on=()):
    return effect(
        "test.refund",
        key=Key("refund", "payment_id"),
        dedupe=contract,
        timeout_s=20,
        applies_within_s=applies_within_s,
        rejected_on=rejected_on,
    )(_unexercised_send)


def _attempt(*, amount=10, number=1, idempotency_key="delivery-1", payment_id="pay-1"):
    return Attempt(
        _RefundRequest(payment_id=payment_id, amount=amount),
        "refund:" + payment_id,
        idempotency_key,
        "operation-1",
        20,
        None,
        AttemptDebug(number, 0, 1, 1),
    )


def _intent_for(declared, attempt):
    entry = _store_probe_intent("unit")
    entry.data.update(
        name=declared.name,
        key=attempt.key,
        record_id=record_id("unit", declared.name, attempt.key),
        key_spec_digest=declared.key_spec_digest,
        request=attempt.request.model_dump(mode="json"),
        request_digest=digest(attempt.request.model_dump(mode="json")),
        idempotency_key=attempt.idempotency_key,
        operation_tag=attempt.operation_tag,
        attempt=attempt.debug.number,
        store_ts=0.0,
        first_use_at=0.0,
    )
    return entry


def test_manual_clock_advances_both_domains_and_steps_only_store_time():
    clock = ManualClock(store_time=100, monotonic=20)
    clock.advance(5)
    assert (clock.store_time(), clock.monotonic()) == (105, 25)
    clock.step_store(-3)
    assert (clock.store_time(), clock.monotonic()) == (102, 25)
    clock.step_store(7)
    assert (clock.store_time(), clock.monotonic()) == (109, 25)
    clock.advance(0)
    assert (clock.store_time(), clock.monotonic()) == (109, 25)


@pytest.mark.parametrize("value", [True, "3", None, math.nan, math.inf, -math.inf, 10**400])
def test_manual_clock_rejects_nonfinite_and_non_numeric_values(value):
    with pytest.raises(ValueError):
        ManualClock(store_time=value)
    with pytest.raises(ValueError):
        ManualClock(monotonic=value)
    clock = ManualClock()
    with pytest.raises(ValueError):
        clock.advance(value)
    with pytest.raises(ValueError):
        clock.step_store(value)
    assert clock.store_time() == clock.monotonic() == 0


def test_manual_clock_rejects_backwards_monotonic_advance_and_atomic_overflow():
    clock = ManualClock(store_time=1e308, monotonic=1)
    with pytest.raises(ValueError):
        clock.advance(-1)
    with pytest.raises(ValueError):
        clock.advance(1e308)
    assert clock.monotonic() == 1 and clock.store_time() == 1e308
    with pytest.raises(ValueError):
        clock.step_store(1e308)
    assert clock.monotonic() == 1 and clock.store_time() == 1e308


@pytest.fixture
def virtual_loop():
    loop = VirtualLoop()
    yield loop
    loop.begin_teardown()
    tasks = asyncio.all_tasks(loop)
    for task in tasks:
        task.cancel()
    if tasks:
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.run_until_complete(loop.shutdown_default_executor())
    loop.close()


def test_virtual_sleep_and_deadlines_do_not_wait_wall_time(virtual_loop):
    async def workload():
        started = virtual_loop.time()
        await asyncio.sleep(3600)
        assert virtual_loop.time() == started + 3600
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(20):
                await asyncio.Event().wait()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.Event().wait(), 7)
        return virtual_loop.time()

    started = time.monotonic()
    assert virtual_loop.run_until_complete(workload()) == 3627
    assert time.monotonic() - started < 1


def test_ready_callbacks_run_before_clock_jumps(virtual_loop):
    observed = []

    async def workload():
        virtual_loop.call_soon(lambda: observed.append(virtual_loop.time()))
        await asyncio.sleep(50)
        observed.append(virtual_loop.time())

    virtual_loop.run_until_complete(workload())
    assert observed == [0, 50]


def test_never_set_event_reports_no_terminal_state(virtual_loop):
    with pytest.raises(NoTerminalState, match="no runnable work or timer"):
        virtual_loop.run_until_complete(asyncio.Event().wait())


def test_periodic_renewals_do_not_reset_leg_idle_horizon(virtual_loop):
    virtual_loop.register_leg(1, idle_horizon_s=600)

    async def renew_forever():
        while True:
            await asyncio.sleep(30)
            virtual_loop.note_world_call(1, renewal=True)

    with pytest.raises(NoTerminalState, match="idle horizon"):
        virtual_loop.run_until_complete(renew_forever())
    assert virtual_loop.time() == 600


def test_world_progress_resets_idle_horizon(virtual_loop):
    virtual_loop.register_leg(1, idle_horizon_s=600)

    async def workload():
        await asyncio.sleep(500)
        virtual_loop.note_world_call(1)
        await asyncio.sleep(500)
        virtual_loop.deactivate_leg(1)

    virtual_loop.run_until_complete(workload())
    assert virtual_loop.time() == 1000


def test_unverified_asyncio_shape_fails_closed(monkeypatch):
    monkeypatch.setattr(asyncio.BaseEventLoop, "_run_once", lambda self: None)
    with pytest.raises(ConfigurationError, match="asyncio internals differ"):
        VirtualLoop()


def test_real_time_watchdog_bounds_a_busy_ready_queue():
    loop = VirtualLoop(real_timeout_s=0.02)

    async def spin():
        while True:  # noqa: ASYNC110 -- exercise a pathological, continuously ready task.
            await asyncio.sleep(0)

    task = loop.create_task(spin())
    try:
        with pytest.raises(NoTerminalState, match="real-time bound"):
            loop.run_until_complete(task)
    finally:
        loop.begin_teardown()
        task.cancel()
        loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
        loop.close()


def test_open_connection_keeps_timers_in_real_time_until_close(virtual_loop):
    async def workload():
        server = await virtual_loop.create_server(asyncio.Protocol, "127.0.0.1", 0)
        writer = None
        try:
            address = server.sockets[0].getsockname()
            _, writer = await asyncio.open_connection(*address[:2])
            assert virtual_loop.real_io_active
            started = time.monotonic()
            await asyncio.sleep(0.03)
            assert time.monotonic() - started >= 0.025
            assert virtual_loop.time() < 1
            writer.close()
            await writer.wait_closed()
            assert not virtual_loop.real_io_active
            before = virtual_loop.time()
            await asyncio.sleep(3600)
            assert virtual_loop.time() == before + 3600
        finally:
            if writer is not None:
                writer.close()
                await writer.wait_closed()
            server.close()
            await server.wait_closed()

    virtual_loop.run_until_complete(workload())
    assert virtual_loop.real_io_used


def test_socket_connect_remains_real_time_until_socket_closed(virtual_loop):
    async def workload():
        server = await virtual_loop.create_server(asyncio.Protocol, "127.0.0.1", 0)
        connection = socket.socket()
        connection.setblocking(False)
        try:
            await virtual_loop.sock_connect(connection, server.sockets[0].getsockname())
            assert virtual_loop.real_io_active
            await asyncio.sleep(0.02)
            connection.close()
            assert not virtual_loop.real_io_active
        finally:
            connection.close()
            server.close()
            await server.wait_closed()

    virtual_loop.run_until_complete(workload())
    assert virtual_loop.real_io_used


def test_address_lookup_is_recorded_and_releases_pending_io(virtual_loop):
    assert virtual_loop.run_until_complete(
        virtual_loop.getaddrinfo("127.0.0.1", 80, family=socket.AF_INET)
    )
    assert virtual_loop.real_io_used
    assert not virtual_loop.real_io_active


def test_real_subprocess_uses_actual_interpreter_pid_and_is_reaped(virtual_loop):
    async def workload():
        # Windows venv redirectors may report a launcher PID, so use the base binary.
        interpreter = getattr(sys, "_base_executable", sys.executable)
        child = await asyncio.create_subprocess_exec(
            interpreter,
            "-I",
            "-S",
            "-B",
            "-u",
            "-c",
            "import os,time; print(os.getpid()); time.sleep(0.03)",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            assert virtual_loop.real_io_active
            stdout, stderr = await child.communicate()
            assert child.returncode == 0 and not stderr
            assert int(stdout.strip()) == child.pid
            assert not virtual_loop.real_io_active
        finally:
            if child.returncode is None:
                child.kill()
                await child.wait()

    virtual_loop.run_until_complete(workload())
    assert virtual_loop.real_io_used


def test_executor_without_timer_waits_for_real_thread_and_copies_context(virtual_loop):
    marker = contextvars.ContextVar("crash-test-marker")
    marker.set("copied")

    def worker():
        time.sleep(0.03)
        return marker.get(), virtual_loop.time()

    async def workload():
        return await virtual_loop.run_in_executor(None, worker)

    assert virtual_loop.run_until_complete(workload()) == ("copied", 0)
    assert virtual_loop.pending_threads == ()
    assert virtual_loop.threads_ran_in_real_time


def test_cancelled_executor_awaiter_keeps_loop_busy(virtual_loop):
    entered, release, ended = threading.Event(), threading.Event(), threading.Event()
    observed = []

    def worker():
        entered.set()
        assert release.wait(1)
        observed.append(virtual_loop.time())
        ended.set()

    async def workload():
        future = virtual_loop.run_in_executor(None, worker)
        assert entered.wait(1)
        future.cancel()
        await asyncio.sleep(0)
        assert virtual_loop.busy_threads == 1
        timer = threading.Timer(0.03, release.set)
        timer.start()
        try:
            await asyncio.sleep(20)
            assert ended.is_set()
        finally:
            release.set()
            timer.join(1)

    try:
        virtual_loop.run_until_complete(workload())
    finally:
        release.set()
    assert observed == [0]
    assert virtual_loop.time() == 20
    assert virtual_loop.pending_threads == ()


def test_thread_end_after_wakeup(virtual_loop):
    observed = []

    def worker():
        time.sleep(0.02)
        return "done"

    async def waiter():
        assert await virtual_loop.run_in_executor(None, worker) == "done"
        observed.append(("waiter", virtual_loop.time()))

    async def timer():
        await asyncio.sleep(100)
        observed.append(("timer", virtual_loop.time()))

    async def workload():
        await asyncio.gather(waiter(), timer())

    virtual_loop.run_until_complete(workload())
    assert observed == [("waiter", 0), ("timer", 100)]


def test_observer_counts_until_concurrent_completion_delivery(virtual_loop):
    def worker():
        time.sleep(0.02)
        return 7

    async def workload(executor):
        submitted = executor.submit(worker)
        wrapped = asyncio.wrap_future(submitted)
        virtual_loop.observe_thread(submitted)
        assert await wrapped == 7
        assert virtual_loop.time() == 0
        await virtual_loop.threads_drained()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        virtual_loop.run_until_complete(workload(executor))
    assert not virtual_loop.pending_threads


def test_thread_waiting_on_world_allows_virtual_time_then_returns_to_running(virtual_loop):
    observed = []

    async def world_call():
        await asyncio.sleep(3600)
        return "receipt"

    def send():
        assert virtual_loop.call_from_thread(world_call()) == "receipt"
        observed.append(virtual_loop.busy_threads)
        return virtual_loop.time()

    async def workload():
        return await asyncio.to_thread(send)

    started = time.monotonic()
    assert virtual_loop.run_until_complete(workload()) == 3600
    assert time.monotonic() - started < 1
    assert observed == [1]
    assert virtual_loop._resting_threads == {}


def test_thread_world_completion_cannot_race_waiting_registration(virtual_loop):
    async def immediate():
        return 3

    def sender():
        for _ in range(100):
            assert virtual_loop.call_from_thread(immediate()) == 3
        return "done"

    async def workload():
        return await asyncio.to_thread(sender)

    assert virtual_loop.run_until_complete(workload()) == "done"
    assert virtual_loop._resting_threads == {}
    assert virtual_loop.pending_threads == ()


def test_stuck_thread_bound_marks_inconclusive_and_releases_virtual_timeout():
    loop = VirtualLoop(thread_timeout_s=0.02)
    release = threading.Event()

    async def workload():
        try:
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(20):
                    await loop.run_in_executor(None, release.wait, 1)
            assert loop.pending_threads
            assert loop.time() == 20
            assert loop.inconclusive_reasons == ["thread exceeded real-time bound"]
        finally:
            release.set()

    try:
        loop.run_until_complete(workload())
    finally:
        release.set()
        loop.begin_teardown()
        loop.run_until_complete(loop.shutdown_default_executor())
        loop.close()


def test_frozen_leg_holds_no_medium_lock_and_finally_waits_for_seal(virtual_loop):
    medium, finalized = [], []
    world = World(virtual_loop)
    world.add_leg(0)
    world.add_leg(1)

    async def boundary(point):
        if point.leg == 0 and point.phase == "after":
            world.freeze(0)

    world.boundary = boundary

    async def append(value):
        medium.append(value)

    async def old_process():
        try:
            await world.gate(0, lambda: append("old"), kind="append", store=True, write=True)
        finally:
            finalized.append("old")

    async def workload():
        old = world.create_task(0, old_process())
        await world.crashed.wait()
        assert not world.medium_lock.locked()
        old.cancel()
        await asyncio.sleep(0)
        assert not finalized and not old.done()
        await world.gate(1, lambda: append("new"), kind="append", store=True, write=True)
        assert medium == ["old", "new"]
        assert await world.teardown() == ()
        assert finalized == ["old"]
        assert isinstance(old.exception(), WorldSealed)
        assert world.findings and all(f["kind"] == "post_mortem_call" for f in world.findings)

    virtual_loop.run_until_complete(workload())


def test_zombie_thaw_delivers_one_cancel_and_preserves_asyncio_timeout(virtual_loop):
    world = World(virtual_loop)
    world.add_leg(0)
    observed = []

    async def boundary(point):
        if point.phase == "before":
            world.freeze(0, zombie=True)

    world.boundary = boundary

    async def operation():
        observed.append("must not apply")

    async def old_process():
        try:
            async with asyncio.timeout(20):
                await world.gate(0, operation, kind="send")
        except TimeoutError:
            observed.append("timed out")

    async def workload():
        old = world.create_task(0, old_process())
        await world.crashed.wait()
        old.cancel()
        await asyncio.sleep(0)
        await asyncio.sleep(30)  # the timeout's own cancellation is also swallowed
        assert old.cancelling() == 2
        world.thaw(0)
        await old
        assert old.cancelling() == 0
        assert observed == ["timed out"]
        assert await world.teardown() == ()

    virtual_loop.run_until_complete(workload())


def test_zombie_without_cancellation_continues_after_thaw(virtual_loop):
    world = World(virtual_loop)
    world.add_leg(0)
    applied = []

    async def boundary(point):
        if point.phase == "before":
            world.freeze(0, zombie=True)

    world.boundary = boundary

    async def operation():
        applied.append(LEG.get())
        return "receipt"

    async def workload():
        task = world.create_task(0, world.gate(0, operation, kind="send"))
        await world.crashed.wait()
        assert applied == []
        world.thaw(0)
        assert await task == "receipt"
        assert applied == [0]
        assert await world.teardown() == ()

    virtual_loop.run_until_complete(workload())


def test_exception_mode_runs_finally_at_crash_boundary(virtual_loop):
    world = World(virtual_loop, mode="exception")
    world.add_leg(0)
    observed = []

    async def boundary(point):
        if point.phase == "before":
            world.freeze(0)

    world.boundary = boundary

    async def operation():
        observed.append("applied")

    async def workload():
        try:
            await world.gate(0, operation, kind="append", store=True, write=True)
        finally:
            observed.append("finally")

    with pytest.raises(LegCrashed):
        virtual_loop.run_until_complete(workload())
    assert observed == ["finally"]
    virtual_loop.run_until_complete(world.teardown())


def test_sealed_world_never_delegates_even_for_fresh_legs(virtual_loop):
    world = World(virtual_loop)
    world.add_leg(0)
    applied = []

    async def operation():
        applied.append(True)

    world.seal()
    with pytest.raises(WorldSealed):
        virtual_loop.run_until_complete(world.gate(0, operation, kind="send"))
    with pytest.raises(WorldSealed):
        world.add_leg(1)
    assert applied == []


def test_sync_service_parks_without_busy_count_and_seal_releases_thread(virtual_loop):
    world = World(virtual_loop)
    world.add_leg(0)
    observed = []

    async def boundary(point):
        if point.phase == "after":
            world.freeze(0)

    world.boundary = boundary

    async def service():
        observed.append("applied")

    def send():
        try:
            world.call_from_thread(0, service, kind="send")
        except WorldSealed:
            observed.append("thread sealed")
            return "sealed"

    async def workload():
        task = world.create_task(0, asyncio.to_thread(send))
        await world.crashed.wait()
        assert virtual_loop.busy_threads == 0
        assert "parked" in virtual_loop._resting_threads.values()
        await asyncio.sleep(100)
        world.seal()
        assert await task == "sealed"
        assert not virtual_loop.pending_threads
        assert not virtual_loop._resting_threads
        assert observed == ["applied", "thread sealed"]
        assert await world.teardown() == ()

    virtual_loop.run_until_complete(workload())


def test_medium_order_and_leg_context_survive_executor_handoff(virtual_loop):
    world = World(virtual_loop)
    for ident in (0, 1):
        world.add_leg(ident)
    observed = []

    async def operation():
        ident = LEG.get()
        observed.append(("begin", ident))
        assert await asyncio.to_thread(LEG.get) == ident
        await asyncio.sleep(5)
        observed.append(("end", ident))

    async def workload():
        tasks = [
            world.create_task(ident, world.gate(ident, operation, kind="write", store=True))
            for ident in (0, 1)
        ]
        await asyncio.gather(*tasks)
        assert await world.teardown() == ()

    virtual_loop.run_until_complete(workload())
    assert observed == [("begin", 0), ("end", 0), ("begin", 1), ("end", 1)]
    assert len(world.keep_alive) == 4


def test_teardown_seals_parked_service_even_with_concurrent_cancellation(virtual_loop):
    world = World(virtual_loop)
    world.add_leg(0)
    ended = threading.Event()

    async def boundary(point):
        if point.phase == "before":
            world.freeze(0)

    world.boundary = boundary

    async def service():
        raise AssertionError("a frozen-before call must never delegate")

    def send():
        try:
            with pytest.raises(WorldSealed):
                world.call_from_thread(0, service, kind="send")
        finally:
            ended.set()

    async def workload():
        world.create_task(0, asyncio.to_thread(send))
        await world.crashed.wait()
        assert await world.teardown() == ()
        assert ended.is_set()
        assert not virtual_loop.pending_threads
        assert not virtual_loop._resting_threads
        assert len(asyncio.all_tasks(virtual_loop)) == 1

    virtual_loop.run_until_complete(workload())


def test_teardown_reports_a_worker_that_outlives_its_join_bound(virtual_loop):
    world = World(virtual_loop)
    world.add_leg(0)
    entered, release = threading.Event(), threading.Event()

    def worker():
        entered.set()
        release.wait(1)

    async def workload():
        world.create_task(0, asyncio.to_thread(worker))
        await asyncio.sleep(0)
        assert entered.wait(1)
        try:
            reasons = await world.teardown(timeout_s=0.02)
            assert "threads remained after teardown bound" in reasons
            assert virtual_loop.pending_threads
        finally:
            release.set()
        await virtual_loop.shutdown_default_executor()
        assert not virtual_loop.pending_threads

    try:
        virtual_loop.run_until_complete(workload())
    finally:
        release.set()


def test_store_delegate_failure_releases_medium_lock(virtual_loop):
    world = World(virtual_loop)
    world.add_leg(0)

    async def failure():
        raise OSError("lost acknowledgement")

    async def workload():
        with pytest.raises(OSError, match="acknowledgement"):
            await world.gate(0, failure, kind="append", store=True, write=True)
        assert not world.medium_lock.locked()
        assert await world.teardown() == ()

    virtual_loop.run_until_complete(workload())


def test_refreeze_before_parked_task_resumes_allocates_a_fresh_wait(virtual_loop):
    world = World(virtual_loop)
    world.add_leg(0)
    applied = []

    async def boundary(point):
        if point.phase == "before":
            world.freeze(0, zombie=True)

    world.boundary = boundary

    async def operation():
        applied.append(True)

    async def workload():
        task = world.create_task(0, world.gate(0, operation, kind="send"))
        await world.crashed.wait()
        world.thaw(0)
        world.freeze(0, zombie=True)
        await asyncio.sleep(25)
        assert not task.done() and not applied
        world.thaw(0)
        await task
        assert applied == [True]
        assert await world.teardown() == ()

    virtual_loop.run_until_complete(workload())


def test_unobserved_thread_cannot_hide_an_owned_running_worker(virtual_loop):
    entered, release = threading.Event(), threading.Event()
    rejected = []

    def owned_worker():
        entered.set()
        release.wait(1)

    async def operation():
        raise AssertionError("unobserved worker must not reach the world")

    def external_worker():
        try:
            virtual_loop.call_from_thread(operation())
        except RuntimeError as error:
            rejected.append(str(error))

    async def workload():
        task = virtual_loop.create_task(asyncio.to_thread(owned_worker))
        await asyncio.sleep(0)
        assert entered.wait(1)
        thread = threading.Thread(target=external_worker)
        thread.start()
        thread.join(1)
        assert not thread.is_alive()
        assert rejected == ["world call from an unobserved thread"]
        assert virtual_loop.busy_threads == 1
        assert not virtual_loop._resting_threads
        assert virtual_loop.inconclusive_reasons == rejected
        release.set()
        await task

    try:
        virtual_loop.run_until_complete(workload())
    finally:
        release.set()


def test_inflight_mutation_finishing_after_seal_retains_uncertain_completion(virtual_loop):
    world = World(virtual_loop)
    world.add_leg(0)
    entered, release = asyncio.Event(), asyncio.Event()
    applied = []

    async def mutation():
        entered.set()
        await release.wait()
        applied.append("late")

    async def workload():
        task = world.create_task(0, world.gate(0, mutation, kind="service", mutating=True))
        await entered.wait()
        world.seal()
        release.set()
        with pytest.raises(WorldSealed):
            await task
        assert applied == ["late"]
        assert [finding["kind"] for finding in world.findings] == ["post_seal_call_uncertain"]
        assert await world.teardown() == ()

    virtual_loop.run_until_complete(workload())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_world_store_matches_actual_backend_contract(virtual_loop, tmp_path, backend):
    world = World(virtual_loop)

    def factory():
        leg = len(world.legs)
        world.add_leg(leg)
        return world.store_for_leg(leg, backend=backend, directory=tmp_path)

    async def workload():
        report = await check_store(factory)
        assert report.ok, report.checks
        assert report.checks
        assert await world.teardown() == ()

    virtual_loop.run_until_complete(workload())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_world_store_fences_fresh_leg_and_returns_independent_reads(
    virtual_loop, tmp_path, backend
):
    world = World(virtual_loop)
    world.add_leg(0)
    world.add_leg(1)
    first = world.store_for_leg(0, backend=backend, directory=tmp_path)
    second = world.store_for_leg(1, backend=backend, directory=tmp_path)

    async def workload():
        assert not hasattr(first, "wrapped") and first is not second
        assert not _append_overridden_alone(first)
        assert first.features == first._backend.features
        assert first.takeover_grace_s == first._backend.takeover_grace_s
        old = await first.acquire_lease_grant("s", "old", 10)
        assert old.epoch == 1 and old.acquired_at == 0
        entry = Entry(kind="custom", data={"nested": [1]})
        assert await first.append_leased("s", [entry], "old") == [1]
        loaded = await second.load("s")
        loaded[0].data["nested"].append(2)
        assert (await first.load("s"))[0].data == {"nested": [1]}
        virtual_loop.clock.advance(10 + first.takeover_grace_s + 0.001)
        new = await second.acquire_lease_grant("s", "new", 10)
        assert new.epoch == 2
        with pytest.raises(LeaseError):
            await first.append_leased("s", [Entry(kind="custom")], "old")
        assert await second.append_leased("s", [Entry(kind="custom")], "new") == [2]
        assert await first.salt() == await second.salt()
        assert await first.now() == virtual_loop.clock.store_time()
        if backend == "sqlite":
            assert first._backend is not second._backend
            assert first._backend._executor is not second._backend._executor
            assert first._backend._connection is not second._backend._connection
        assert await world.teardown() == ()
        if backend == "sqlite":
            assert first._backend._executor is second._backend._executor is None

    virtual_loop.run_until_complete(workload())


def test_world_store_exposes_every_store_method_and_masks_only_requested_features(virtual_loop):
    world = World(virtual_loop)
    world.add_leg(0)
    store = world.store_for_leg(0, backend="memory", mask_features=frozenset({"claims"}))
    assert "claims" not in store.features
    assert "epochs" in store.features
    for name, method in SessionStore.__dict__.items():
        if asyncio.iscoroutinefunction(method) and not name.startswith("__"):
            assert name in WorldStore.__dict__, name
    virtual_loop.run_until_complete(world.teardown())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_frozen_store_call_resumes_against_new_lease_not_old_snapshot(
    virtual_loop, tmp_path, backend
):
    world = World(virtual_loop)
    world.add_leg(0)
    world.add_leg(1)
    first = world.store_for_leg(0, backend=backend, directory=tmp_path)
    second = world.store_for_leg(1, backend=backend, directory=tmp_path)

    async def boundary(point):
        if point.leg == 0 and point.kind == "append_leased" and point.phase == "before":
            world.freeze(0, zombie=True)

    world.boundary = boundary

    async def workload():
        await first.acquire_lease_grant("s", "old", 10)
        task = world.create_task(0, first.append_leased("s", [Entry(kind="custom")], "old"))
        await world.crashed.wait()
        virtual_loop.clock.advance(11.001)
        assert (await second.acquire_lease_grant("s", "new", 10)).epoch == 2
        world.thaw(0)
        with pytest.raises(LeaseError):
            await task
        assert await second.load("s") == []
        assert await world.teardown() == ()

    virtual_loop.run_until_complete(workload())


def test_fault_catalog_parses_named_sets_aliases_and_rejects_unknown_kinds():
    assert Faults.default().kinds == Faults.KINDS
    assert Faults.quick().kinds == {
        "kill_before",
        "kill_after",
        "commit_then_lose_response",
        "retry_effect",
    }
    assert Faults.parse("zombie, kill_before").kinds == Faults.ALIASES["zombie"] | {"kill_before"}
    assert Faults.parse("quick").kinds == Faults.quick().kinds
    for invalid in ("", "unknown", "kill_before,", "kill_before,,kill_after"):
        with pytest.raises(ValueError):
            Faults.parse(invalid)
    with pytest.raises(TypeError):
        Faults(kinds="kill_before")
    with pytest.raises(ValueError):
        Faults(kinds=(), stall_s=math.inf)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("committed", [False, True])
def test_acknowledgement_faults_preserve_the_actual_commit_shape(
    virtual_loop, tmp_path, backend, committed
):
    world = World(virtual_loop)
    world.add_leg(0)
    store = world.store_for_leg(0, backend=backend, directory=tmp_path)
    kind = "ack_lost_committed" if committed else "ack_lost_uncommitted"
    faults = FaultController(world, [FaultPlan("selected", kind, 1, method="append")])
    faults.install()

    async def workload():
        assert await store.now() == 0  # reads do not consume write indices
        with pytest.raises(OSError, match="acknowledgement lost"):
            await store.append("s", [Entry(kind="custom", data={"n": 1})])
        assert [entry.data["n"] for entry in await store.load("s")] == ([1] if committed else [])
        await store.append("s", [Entry(kind="custom", data={"n": 2})])
        assert [entry.data["n"] for entry in await store.load("s")] == (
            [1, 2] if committed else [2]
        )
        assert faults.triggered == ["selected"] and not faults.not_triggered
        assert await world.teardown() == ()

    virtual_loop.run_until_complete(workload())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_cancel_during_actual_store_write_does_not_detach_commit(virtual_loop, tmp_path, backend):
    world = World(virtual_loop)
    world.add_leg(0)
    store = world.store_for_leg(0, backend=backend, directory=tmp_path)
    faults = FaultController(world, [FaultPlan("cancel", "cancel_during", 1, method="append")])
    faults.install()

    async def workload():
        task = world.create_task(0, store.append("s", [Entry(kind="custom")]))
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(await store.load("s")) == 1
        assert faults.triggered == ["cancel"]
        assert not world.medium_lock.locked()
        assert await world.teardown() == ()

    virtual_loop.run_until_complete(workload())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_store_stall_fires_virtual_deadline_before_delegation(virtual_loop, tmp_path, backend):
    world = World(virtual_loop)
    world.add_leg(0)
    store = world.store_for_leg(0, backend=backend, directory=tmp_path)
    faults = FaultController(world, [FaultPlan("stall", "stall", 1, method="append")])
    faults.install()

    async def workload():
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(20):
                await store.append("s", [Entry(kind="custom")])
        assert virtual_loop.time() == 20
        assert await store.load("s") == []
        assert faults.triggered == ["stall"]
        assert await world.teardown() == ()

    virtual_loop.run_until_complete(workload())


@pytest.mark.parametrize("shape", ["committed", "uncommitted"])
def test_lease_acknowledgement_loss_reads_back_real_owner(virtual_loop, shape):
    world = World(virtual_loop)
    world.add_leg(0)
    store = world.store_for_leg(0, backend="memory")
    faults = FaultController(
        world,
        [FaultPlan("lease", "lease_ack_lost", 1, method="acquire_lease_grant", value=shape)],
    )
    faults.install()

    async def workload():
        with pytest.raises(OSError, match="lease acknowledgement lost"):
            await store.acquire_lease_grant("s", "owner", 10)
        status = await store.status("s")
        assert status.lease_owner == ("owner" if shape == "committed" else None)
        assert faults.triggered == ["lease"]
        assert await world.teardown() == ()

    virtual_loop.run_until_complete(workload())


@pytest.mark.parametrize("kind", ["lease_renew_refused", "lease_error"])
def test_lease_renewal_fault_does_not_extend_the_lease(virtual_loop, kind):
    world = World(virtual_loop)
    world.add_leg(0)
    store = world.store_for_leg(0, backend="memory")
    faults = FaultController(world, [FaultPlan("renew", kind, 2, method="renew_lease")])
    faults.install()

    async def workload():
        await store.acquire_lease_grant("s", "owner", 10)
        virtual_loop.clock.advance(5)
        if kind == "lease_error":
            with pytest.raises(LeaseError, match="injected"):
                await store.renew_lease("s", "owner", 10)
        else:
            assert await store.renew_lease("s", "owner", 10) is False
        assert (await store.status("s")).lease_expires_at == 10
        assert faults.triggered == ["renew"]
        assert await world.teardown() == ()

    virtual_loop.run_until_complete(workload())


@pytest.mark.parametrize("delta", [-2, -0.5, 0.5, 2])
def test_store_clock_step_records_its_assumption_boundary(virtual_loop, delta):
    world = World(virtual_loop)
    world.add_leg(0)
    store = world.store_for_leg(0, backend="memory")
    beyond = abs(delta) > 1
    kind = "store_step_beyond_sigma" if beyond else "store_step"
    faults = FaultController(world, [FaultPlan("step", kind, 1, phase="before", value=delta)])
    faults.install()

    async def workload():
        await store.append("s", [Entry(kind="custom")])
        assert await store.now() == delta
        assert virtual_loop.time() == 0
        assert faults.outside_a3 == (["step"] if beyond else [])
        assert faults.triggered == ["step"]
        assert await world.teardown() == ()

    virtual_loop.run_until_complete(workload())


def test_fault_with_missing_target_remains_not_triggered(virtual_loop):
    world = World(virtual_loop)
    world.add_leg(0)
    store = world.store_for_leg(0, backend="memory")
    faults = FaultController(world, [FaultPlan("absent", "lease_error", 3, method="renew_lease")])
    faults.install()

    async def workload():
        await store.append("s", [Entry(kind="custom")])
        assert faults.not_triggered == ("absent",) and not faults.triggered
        assert await world.teardown() == ()

    virtual_loop.run_until_complete(workload())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("phase", ["before", "after"])
def test_store_kill_fault_freezes_at_the_selected_commit_boundary(
    virtual_loop, tmp_path, backend, phase
):
    world = World(virtual_loop)
    world.add_leg(0)
    world.add_leg(1)
    first = world.store_for_leg(0, backend=backend, directory=tmp_path)
    second = world.store_for_leg(1, backend=backend, directory=tmp_path)
    faults = FaultController(
        world, [FaultPlan("kill", "kill_" + phase, 1, phase=phase, method="append")]
    )
    faults.install()

    async def workload():
        old = world.create_task(0, first.append("s", [Entry(kind="custom")]))
        await world.crashed.wait()
        assert not old.done() and not world.medium_lock.locked()
        assert len(await second.load("s")) == int(phase == "after")
        assert faults.triggered == ["kill"]
        assert await world.teardown() == ()

    virtual_loop.run_until_complete(workload())


@pytest.mark.parametrize("method", ["renew_lease", "release_lease"])
@pytest.mark.parametrize("shape", ["committed", "uncommitted"])
def test_lease_ack_loss_preserves_renewal_and_release_shapes(virtual_loop, method, shape):
    world = World(virtual_loop)
    world.add_leg(0)
    store = world.store_for_leg(0, backend="memory")
    faults = FaultController(
        world, [FaultPlan("lease", "lease_ack_lost", 2, method=method, value=shape)]
    )
    faults.install()

    async def workload():
        await store.acquire_lease_grant("s", "owner", 10)
        virtual_loop.clock.advance(5)
        with pytest.raises(OSError, match="acknowledgement"):
            if method == "renew_lease":
                await store.renew_lease("s", "owner", 10)
            else:
                await store.release_lease("s", "owner")
        status = await store.status("s")
        if method == "renew_lease":
            assert status.lease_expires_at == (15 if shape == "committed" else 10)
        else:
            assert status.lease_owner == (None if shape == "committed" else "owner")
        assert faults.triggered == ["lease"]
        assert await world.teardown() == ()

    virtual_loop.run_until_complete(workload())


def test_build_isolation_allows_only_exact_shared_infrastructure_roots():
    shared_world = {"durable": []}

    class Client:
        def __init__(self):
            self.world = shared_world
            self.local_cache = []

    check = BuildIsolation(infrastructure=[shared_world])
    for leg in range(2):
        assert check.record(leg, Agent(ScriptedModel(["ok"])), {"client": Client()}) == []
    client = Client()
    assert check.record(2, Agent(ScriptedModel(["ok"])), {"client": client}) == []
    findings = check.record(3, Agent(ScriptedModel(["ok"])), {"client": client})
    assert len(findings) == 1
    assert findings[0]["kind"] == "shared_between_legs"
    assert findings[0]["first_leg"] == 2 and findings[0]["leg"] == 3
    assert findings[0]["type"].endswith("Client")


def test_build_isolation_detects_reused_agent_and_model_clients():
    check = BuildIsolation()
    agent = Agent(ScriptedModel(["ok"]))
    assert check.record(0, agent, None) == []
    assert check.record(1, agent, None)[0]["path"] == "agent"
    client = ScriptedModel(["ok"])
    assert check.record(2, Agent(client), None) == []
    findings = check.record(3, Agent(client), None)
    assert any(item["path"] == "agent.model" for item in findings)


def test_build_isolation_handles_cycles_slots_and_generated_output_schema():
    class Output(BaseModel):
        answer: str

    class Client:
        __slots__ = ("link",)

    shared = Client()
    shared.link = shared
    check = BuildIsolation()
    first = Agent(ScriptedModel(["ok"]), output_type=Output)
    second = Agent(ScriptedModel(["ok"]), output_type=Output)
    assert check.record(0, first, {"nested": (shared,)}) == []
    findings = check.record(1, second, {"nested": (shared,)})
    assert len(findings) == 1 and findings[0]["type"].endswith("Client")


def test_build_isolation_reports_bounded_inspection_instead_of_silent_success():
    check = BuildIsolation(max_nodes=2)
    findings = check.record(0, Agent(ScriptedModel(["ok"])), {"nested": [[[]]]})
    assert any(item["kind"] == "isolation_check_incomplete" for item in findings)


def test_build_isolation_invokes_async_factory_once_per_leg(virtual_loop):
    calls = []
    check = BuildIsolation()

    async def build(env):
        calls.append(env)
        await asyncio.sleep(0)
        return Agent(ScriptedModel(["ok"])), {"client": object()}

    async def workload():
        first = await check.build(0, build, "first input")
        second = await check.build(1, build, "second input")
        assert first[0] is not second[0]
        assert first[1] is not second[1]
        assert not check.findings
        with pytest.raises(ConfigurationError, match="build"):
            await check.build(2, lambda _: "invalid", None)

    virtual_loop.run_until_complete(workload())
    assert calls == ["first input", "second input"]


def test_contract_dedupe_window_starts_at_first_arrival_and_expires_exactly(virtual_loop):
    world = World(virtual_loop)
    declared = _refund_effect(Dedupe(100, "unit destination", concurrent="rejects"))
    fake = ContractFake(world, declared)

    async def workload():
        first = await fake.send(_attempt(), leg=0, key="delivery-1")
        assert first.payment_id == "pay-1" and first.amount == 10
        virtual_loop.clock.advance(40)
        assert await fake.send(_attempt(number=2), leg=1, key="delivery-1") == first
        virtual_loop.clock.advance(59.999)
        assert await fake.send(_attempt(number=3), leg=2, key="delivery-1") == first
        virtual_loop.clock.advance(0.001)
        assert await fake.send(_attempt(number=4), leg=3, key="delivery-1") != first
        assert world.count_applied(declared) == 2
        assert [item.replayed for item in world.applications(declared)] == [
            False,
            True,
            True,
            False,
        ]
        assert [item.leg for item in world.applications(declared)] == [0, 1, 2, 3]

    virtual_loop.run_until_complete(workload())


def test_contract_binding_refuses_changed_payload_without_new_application(virtual_loop):
    world = World(virtual_loop)
    declared = _refund_effect(Dedupe(100, "unit destination", binds_params=True))
    fake = ContractFake(world, declared)

    async def workload():
        await fake.send(_attempt(), leg=0, key="delivery-1")
        with pytest.raises(ContractRefusal, match="changed_params"):
            await fake.send(_attempt(amount=20), leg=1, key="delivery-1")
        assert world.count_applied(declared, amount=10) == 1
        assert world.count_applied(declared, amount=20) == 0

    virtual_loop.run_until_complete(workload())


@pytest.mark.parametrize("behavior", ["rejects", "waits"])
def test_unknown_contract_concurrency_has_two_explicit_scenarios(virtual_loop, behavior):
    world = World(virtual_loop)
    declared = _refund_effect(Dedupe(100, "unit destination", concurrent="unknown"))
    fake = ContractFake(world, declared, unknown_concurrent=behavior)
    entered, release = asyncio.Event(), asyncio.Event()

    async def apply():
        entered.set()
        await release.wait()
        return _RefundReceipt(id="receipt", payment_id="pay-1", amount=10)

    async def workload():
        first = asyncio.create_task(fake.send(_attempt(), leg=0, key="delivery-1", apply=apply))
        await entered.wait()
        if behavior == "rejects":
            with pytest.raises(ContractRefusal, match="concurrent"):
                await fake.send(_attempt(number=2), leg=1, key="delivery-1")
            release.set()
        else:
            second = asyncio.create_task(fake.send(_attempt(number=2), leg=1, key="delivery-1"))
            await asyncio.sleep(0)
            assert not second.done()
            release.set()
            assert await second == await first
        await first
        assert world.count_applied(declared) == 1

    virtual_loop.run_until_complete(workload())


@pytest.mark.parametrize("replayed", [False, True])
@pytest.mark.parametrize("rejection", [False, True])
def test_contract_replays_errors_and_rejections_only_as_declared(virtual_loop, replayed, rejection):
    world = World(virtual_loop)
    contract = Dedupe(100, "unit destination", replays_errors=replayed, replays_rejections=replayed)
    fake = ContractFake(world, _refund_effect(contract))
    exception = ContractRefusal if rejection else ConnectionError

    async def refusal():
        raise exception("first response")

    async def workload():
        with pytest.raises(exception, match="first response"):
            await fake.send(_attempt(), leg=0, key="delivery-1", apply=refusal)
        if replayed:
            with pytest.raises(exception, match="first response"):
                await fake.send(_attempt(number=2), leg=1, key="delivery-1")
            assert world.count_applied(fake.effect) == 0
        else:
            assert isinstance(
                await fake.send(_attempt(number=2), leg=1, key="delivery-1"), _RefundReceipt
            )
            assert world.count_applied(fake.effect) == 1

    virtual_loop.run_until_complete(workload())


def test_unique_uses_declared_duplicate_error_and_missing_key_is_counted(virtual_loop):
    class Duplicate(Exception):
        pass

    world = World(virtual_loop)
    declared = _refund_effect(Unique("permanent unique constraint", on_duplicate=(Duplicate,)))
    fake = ContractFake(world, Fake(declared, identity=lambda request: request.payment_id))

    async def workload():
        await fake.send(_attempt(), leg=0, key="delivery-1")
        with pytest.raises(Duplicate):
            await fake.send(_attempt(number=2), leg=1, key="delivery-1")
        await fake.send(_attempt(number=3), leg=2, key=None, key_passed=False)
        assert world.count_applied(declared) == 2
        applications = world.applications(declared)
        assert applications[0].identity == applications[1].identity == "pay-1"
        assert applications[1].idempotency_key is None and not applications[1].key_passed

    virtual_loop.run_until_complete(workload())


def test_plain_lookup_can_miss_but_copies_visible_receipts(virtual_loop):
    world = World(virtual_loop)
    fake = ContractFake(world, _refund_effect())
    attempt = _attempt()
    inquiry = Inquiry(attempt.request, attempt.key, attempt.operation_tag, ("delivery-1",), (), 0)

    async def workload():
        await fake.send(attempt, leg=0, key="delivery-1")
        found = await fake.lookup(inquiry)
        assert isinstance(found, Found) and found.tag == "operation-1"
        found.receipt.amount = 999
        assert (await fake.lookup(inquiry)).receipt.amount == 10
        assert type(await fake.lookup(inquiry, miss=True)).__name__ == "Absent"

    virtual_loop.run_until_complete(workload())


def test_uncached_error_does_not_restart_the_original_key_window(virtual_loop):
    world = World(virtual_loop)
    declared = _refund_effect(Dedupe(100, "unit destination", replays_errors=False))
    fake = ContractFake(world, declared)

    async def failure():
        raise ConnectionError("before application")

    async def workload():
        with pytest.raises(ConnectionError):
            await fake.send(_attempt(), leg=0, key="delivery-1", apply=failure)
        virtual_loop.clock.advance(40)
        receipt = await fake.send(_attempt(number=2), leg=1, key="delivery-1")
        virtual_loop.clock.advance(60)
        assert await fake.send(_attempt(number=3), leg=2, key="delivery-1") != receipt
        assert world.count_applied(declared) == 2

    virtual_loop.run_until_complete(workload())


@pytest.mark.parametrize("replayed", [True, False])
def test_error_after_application_keeps_application_and_lookup_evidence(virtual_loop, replayed):
    world = World(virtual_loop)
    declared = _refund_effect(Dedupe(100, "unit destination", replays_errors=replayed))
    fake = ContractFake(world, declared)
    attempt = _attempt()
    inquiry = Inquiry(attempt.request, attempt.key, attempt.operation_tag, ("delivery-1",), (), 0)

    async def lose_response(receipt):
        raise ConnectionError("response lost after commit")

    async def workload():
        with pytest.raises(ConnectionError):
            await fake.send(attempt, leg=0, key="delivery-1", after_apply=lose_response)
        assert world.count_applied(declared) == 1
        assert isinstance(await fake.lookup(inquiry), Found)
        if replayed:
            with pytest.raises(ConnectionError):
                await fake.send(_attempt(number=2), leg=1, key="delivery-1")
            assert world.count_applied(declared) == 1
        else:
            await fake.send(_attempt(number=2), leg=1, key="delivery-1")
            assert world.count_applied(declared) == 2

    virtual_loop.run_until_complete(workload())


def test_invalid_identity_fails_before_the_client_mutates(virtual_loop):
    world = World(virtual_loop)
    fake = ContractFake(world, Fake(_refund_effect(), identity=lambda _: []))
    applied = []

    async def client():
        applied.append(True)

    async def workload():
        with pytest.raises(ConfigurationError, match="hashable"):
            await fake.send(_attempt(), leg=0, key="delivery-1", apply=client)
        assert applied == [] and world.applications() == []

    virtual_loop.run_until_complete(workload())


def test_cancelled_call_after_seal_is_uncertain_without_application_evidence(virtual_loop):
    world = World(virtual_loop)
    world.add_leg(0)
    entered = asyncio.Event()

    async def mutation():
        entered.set()
        await asyncio.Event().wait()

    async def workload():
        world.create_task(0, world.gate(0, mutation, kind="service", mutating=True))
        await entered.wait()
        assert await world.teardown() == ()
        assert world.applications() == []
        assert [finding["kind"] for finding in world.findings] == ["post_seal_call_uncertain"]

    virtual_loop.run_until_complete(workload())


def test_new_contract_application_after_seal_is_a_definite_violation(virtual_loop):
    world = World(virtual_loop)
    fake = ContractFake(world, _refund_effect())
    entered, release = asyncio.Event(), asyncio.Event()

    async def client():
        entered.set()
        await release.wait()
        return _RefundReceipt(id="late", payment_id="pay-1", amount=10)

    async def workload():
        task = asyncio.create_task(fake.send(_attempt(), leg=0, key="delivery-1", apply=client))
        await entered.wait()
        world.seal()
        release.set()
        assert (await task).id == "late"
        assert world.count_applied(fake.effect) == 1
        assert [finding["kind"] for finding in world.findings] == ["post_seal_mutation"]

    virtual_loop.run_until_complete(workload())


@pytest.mark.parametrize(
    ("kind", "exception", "applied"),
    [
        ("reject", EffectRejected, 0),
        ("fail_before_commit", ConnectionError, 0),
        ("commit_then_error_response", ConnectionError, 1),
        ("commit_then_reject_shaped_error", ToolError, 1),
    ],
)
def test_effect_fault_records_application_before_failed_response(
    virtual_loop, kind, exception, applied
):
    world = World(virtual_loop)
    fake = ContractFake(world, _refund_effect())
    plan = FaultPlan("fault", kind, 1)
    controller = FaultController(world, [plan])

    async def workload():
        with pytest.raises(exception):
            await fake.send_faulted(
                _attempt(), leg=0, key="delivery-1", fault=plan, controller=controller
            )
        assert world.count_applied(fake.effect) == applied
        assert controller.triggered == ["fault"]

    virtual_loop.run_until_complete(workload())


@pytest.mark.parametrize("cancel_at", [None, 10])
def test_lost_response_preserves_destination_receipt_even_when_caller_is_cancelled(
    virtual_loop, cancel_at
):
    world = World(virtual_loop)
    fake = ContractFake(world, _refund_effect(Dedupe(100, "unit destination")))
    plan = FaultPlan("lost", "commit_then_lose_response", 1)
    controller = FaultController(world, [plan])

    async def workload():
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(cancel_at):
                await fake.send_faulted(
                    _attempt(), leg=0, key="delivery-1", fault=plan, controller=controller
                )
        assert virtual_loop.time() == (20 if cancel_at is None else cancel_at)
        assert world.count_applied(fake.effect) == 1
        receipt = await fake.send(_attempt(number=2), leg=1, key="delivery-1")
        assert receipt.amount == 10
        assert world.count_applied(fake.effect) == 1
        assert controller.triggered == ["lost"]

    virtual_loop.run_until_complete(workload())


def test_crash_after_application_leaves_completed_destination_key_replayable(virtual_loop):
    world = World(virtual_loop)
    world.add_leg(0)
    fake = ContractFake(world, _refund_effect(Dedupe(100, "unit destination")))
    plan = FaultPlan("crash", "commit_then_crash", 1)
    controller = FaultController(world, [plan])

    async def workload():
        world.create_task(
            0,
            fake.send_faulted(
                _attempt(), leg=0, key="delivery-1", fault=plan, controller=controller
            ),
        )
        await world.crashed.wait()
        receipt = await fake.send(_attempt(number=2), leg=1, key="delivery-1")
        assert receipt.amount == 10 and world.count_applied(fake.effect) == 1
        assert controller.triggered == ["crash"]
        assert await world.teardown() == ()

    virtual_loop.run_until_complete(workload())


@pytest.mark.parametrize("kind", ["commit_late", "not_yet_visible_then_commit"])
def test_late_destination_survives_lost_response_and_respects_application_bound(virtual_loop, kind):
    world = World(virtual_loop)
    fake = ContractFake(world, _refund_effect(applies_within_s=5))
    plan = FaultPlan("late", kind, 1, value=5)
    controller = FaultController(world, [plan])
    attempt = _attempt()
    inquiry = Inquiry(attempt.request, attempt.key, attempt.operation_tag, ("delivery-1",), (), 0)

    async def workload():
        with pytest.raises(ConnectionError, match="remains in flight"):
            await fake.send_faulted(
                attempt, leg=0, key="delivery-1", fault=plan, controller=controller
            )
        assert world.count_applied(fake.effect) == 0
        assert type(await fake.lookup(inquiry)).__name__ == (
            "Absent" if kind == "commit_late" else "NotYetVisible"
        )
        await asyncio.sleep(5.001)
        assert world.count_applied(fake.effect) == 1
        assert isinstance(await fake.lookup(inquiry), Found)
        assert world.applications()[0].arrival == 5
        assert not fake.pending_destinations
        assert controller.triggered == ["late"]

    virtual_loop.run_until_complete(workload())


def test_impossible_late_application_is_rejected_before_trigger(virtual_loop):
    world = World(virtual_loop)
    fake = ContractFake(world, _refund_effect(applies_within_s=5))
    plan = FaultPlan("late", "commit_late", 1, value=6)
    controller = FaultController(world, [plan])

    async def workload():
        with pytest.raises(ValueError, match="declared"):
            await fake.send_faulted(
                _attempt(), leg=0, key="delivery-1", fault=plan, controller=controller
            )
        assert controller.not_triggered == ("late",)
        assert not world.applications() and not fake.pending_destinations

    virtual_loop.run_until_complete(workload())


@pytest.mark.parametrize("form", ["large", "non_json"])
def test_unstorable_receipt_fault_preserves_actual_application(virtual_loop, form):
    world = World(virtual_loop)
    fake = ContractFake(world, _refund_effect())
    plan = FaultPlan("receipt", "unstorable_receipt", 1, value=form)
    controller = FaultController(world, [plan])

    async def workload():
        receipt = await fake.send_faulted(
            _attempt(), leg=0, key="delivery-1", fault=plan, controller=controller
        )
        if form == "large":
            assert len(receipt.model_dump_json().encode()) > 16 * 1024
        else:
            with pytest.raises(TypeError):
                json.dumps(receipt)
        assert world.count_applied(fake.effect) == 1
        assert controller.triggered == ["receipt"]

    virtual_loop.run_until_complete(workload())


def test_slow_arrival_reaches_destination_just_before_the_attempt_deadline(virtual_loop):
    world = World(virtual_loop)
    fake = ContractFake(world, _refund_effect())
    plan = FaultPlan("slow", "slow_arrival", 1)
    controller = FaultController(world, [plan])

    async def workload():
        await fake.send_faulted(
            _attempt(), leg=0, key="delivery-1", fault=plan, controller=controller
        )
        assert world.applications()[0].arrival == pytest.approx(19.999)
        assert controller.triggered == ["slow"]

    virtual_loop.run_until_complete(workload())


def test_declared_rejection_fault_uses_the_actual_exception_class(virtual_loop):
    class InvalidRequest(Exception):
        pass

    world = World(virtual_loop)
    fake = ContractFake(world, _refund_effect(rejected_on=(InvalidRequest,)))
    plan = FaultPlan("refused", "rejected_on", 1, value=InvalidRequest)
    controller = FaultController(world, [plan])

    async def workload():
        with pytest.raises(InvalidRequest):
            await fake.send_faulted(
                _attempt(), leg=0, key="delivery-1", fault=plan, controller=controller
            )
        assert world.count_applied(fake.effect) == 0
        assert controller.triggered == ["refused"]

    virtual_loop.run_until_complete(workload())


def test_unique_duplicate_response_keeps_the_one_application(virtual_loop):
    class Duplicate(Exception):
        pass

    world = World(virtual_loop)
    fake = ContractFake(world, _refund_effect(Unique("unit constraint", on_duplicate=(Duplicate,))))
    plan = FaultPlan("duplicate", "duplicate_key", 1)
    controller = FaultController(world, [plan])

    async def workload():
        with pytest.raises(Duplicate):
            await fake.send_faulted(
                _attempt(), leg=0, key="delivery-1", fault=plan, controller=controller
            )
        assert world.count_applied(fake.effect) == 1
        with pytest.raises(Duplicate):
            await fake.send(_attempt(number=2), leg=1, key="delivery-1")
        assert world.count_applied(fake.effect) == 1
        assert controller.triggered == ["duplicate"]

    virtual_loop.run_until_complete(workload())


def test_cancellation_at_effect_gate_retains_actual_application(virtual_loop):
    world = World(virtual_loop)
    world.add_leg(0)
    fake = ContractFake(world, _refund_effect())
    plan = FaultPlan("cancel", "cancel_during", 1)
    controller = FaultController(world, [plan])

    async def workload():
        task = world.create_task(
            0,
            fake.send_faulted(
                _attempt(), leg=0, key="delivery-1", fault=plan, controller=controller
            ),
        )
        with pytest.raises(asyncio.CancelledError):
            await task
        assert world.count_applied(fake.effect) == 1
        assert controller.triggered == ["cancel"]
        assert await world.teardown() == ()

    virtual_loop.run_until_complete(workload())


def test_unfinished_destination_application_is_inconclusive_at_teardown(virtual_loop):
    world = World(virtual_loop)
    fake = ContractFake(world, _refund_effect(applies_within_s=5))
    plan = FaultPlan("late", "commit_late", 1, value=5)
    controller = FaultController(world, [plan])

    async def workload():
        with pytest.raises(ConnectionError):
            await fake.send_faulted(
                _attempt(), leg=0, key="delivery-1", fault=plan, controller=controller
            )
        reasons = await world.teardown()
        assert "destination application pending at teardown" in reasons
        assert world.count_applied(fake.effect) == 0
        assert not fake.pending_destinations

    virtual_loop.run_until_complete(workload())


def test_late_fault_on_a_cached_response_stays_untriggered(virtual_loop):
    world = World(virtual_loop)
    fake = ContractFake(world, _refund_effect(Dedupe(100, "unit destination"), applies_within_s=5))
    plan = FaultPlan("late", "commit_late", 1, value=5)
    controller = FaultController(world, [plan])

    async def workload():
        receipt = await fake.send(_attempt(), leg=0, key="delivery-1")
        assert (
            await fake.send_faulted(
                _attempt(number=2), leg=1, key="delivery-1", fault=plan, controller=controller
            )
            == receipt
        )
        assert controller.not_triggered == ("late",)
        assert not fake.pending_destinations

    virtual_loop.run_until_complete(workload())


def test_client_exceeding_declared_application_bound_records_a2_violation(virtual_loop):
    world = World(virtual_loop)
    fake = ContractFake(world, _refund_effect(applies_within_s=2))

    async def slow_client():
        await asyncio.sleep(3)
        return _RefundReceipt(id="late", payment_id="pay-1", amount=10)

    async def workload():
        await fake.send(_attempt(), leg=0, key="delivery-1", apply=slow_client)
        assert world.count_applied(fake.effect) == 1
        assert [finding["kind"] for finding in world.findings] == ["a2_violation"]

    virtual_loop.run_until_complete(workload())


def test_actual_qagents_thread_observer_owns_sync_world_calls(virtual_loop):
    marker = contextvars.ContextVar("owned-tool-context", default=None)
    marker.set("copied")

    async def world_call():
        assert marker.get() == "copied"
        await asyncio.sleep(20)
        return "receipt"

    def send():
        return threading.get_ident(), virtual_loop.call_from_thread(world_call())

    async def workload():
        wrapped, concurrent = qtools.start_thread("send", send)
        ident, receipt = await wrapped
        assert concurrent._qagents_thread_id == ident
        assert receipt == "receipt"
        assert virtual_loop.time() == 20
        await virtual_loop.threads_drained()
        assert not virtual_loop.pending_threads
        assert not virtual_loop._resting_threads
        assert not virtual_loop.inconclusive_reasons

    previous = qtools._THREAD_OBSERVER
    with virtual_loop.observe_tool_threads():
        virtual_loop.run_until_complete(workload())
    assert qtools._THREAD_OBSERVER is previous


def test_cancelled_actual_tool_awaiter_keeps_owned_worker_counted(virtual_loop):
    entered, release = threading.Event(), threading.Event()

    def worker():
        entered.set()
        release.wait(1)
        return virtual_loop.time()

    async def workload():
        wrapped, concurrent = qtools.start_thread("worker", worker)
        assert entered.wait(1)
        wrapped.cancel()
        await asyncio.sleep(0)
        assert virtual_loop.busy_threads == 1 and not concurrent.done()
        timer = threading.Timer(0.03, release.set)
        timer.start()
        try:
            await asyncio.sleep(100)
            assert concurrent.result() == 0
            assert not virtual_loop.pending_threads
        finally:
            release.set()
            timer.join(1)

    try:
        with virtual_loop.observe_tool_threads():
            virtual_loop.run_until_complete(workload())
    finally:
        release.set()


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_application_evidence_binds_actual_operation_key_and_immutable_journal(
    virtual_loop, tmp_path, backend
):
    world = World(virtual_loop)
    world.add_leg(0)
    store = world.store_for_leg(0, backend=backend, directory=tmp_path)
    declared = _refund_effect()
    fake = ContractFake(world, declared)
    attempt = _attempt()
    intent = _intent_for(declared, attempt)

    async def workload():
        await store.acquire_lease_grant(world.session_id, "owner", 50)
        await store.append_leased(world.session_id, [intent], "owner")
        await fake.send(attempt, leg=0, key=attempt.idempotency_key)
        evidence = world.application_evidence[0]
        assert evidence.application_index == 0
        assert evidence.operation_key == attempt.key
        assert evidence.operation_key != attempt.idempotency_key
        assert evidence.record_id == intent.data["record_id"]
        assert evidence.snapshot.entry_ids == (intent.id,)
        assert evidence.snapshot.independent_read is (backend == "sqlite")
        assert evidence.snapshot.session_id == world.session_id
        snapshot_text = evidence.snapshot.entries_json
        loaded = await store.load(world.session_id)
        loaded[0].data["request"]["amount"] = 999
        await store.append_leased(world.session_id, [Entry(kind="custom")], "owner")
        assert evidence.snapshot.entries_json == snapshot_text
        assert len(evidence.snapshot.entries_json) == 1
        assert len((await world.snapshot()).entries_json) == 2
        assert await world.teardown() == ()

    virtual_loop.run_until_complete(workload())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_late_application_keeps_issue_prefix_even_if_intent_is_written_afterward(
    virtual_loop, tmp_path, backend
):
    world = World(virtual_loop)
    world.add_leg(0)
    store = world.store_for_leg(0, backend=backend, directory=tmp_path)
    declared = _refund_effect(applies_within_s=5)
    fake = ContractFake(world, declared)
    attempt = _attempt()
    plan = FaultPlan("late", "commit_late", 1, value=5)
    controller = FaultController(world, [plan])

    async def workload():
        await store.acquire_lease_grant(world.session_id, "owner", 50)
        with pytest.raises(ConnectionError):
            await fake.send_faulted(
                attempt, leg=0, key=attempt.idempotency_key, fault=plan, controller=controller
            )
        await store.append_leased(world.session_id, [_intent_for(declared, attempt)], "owner")
        await asyncio.sleep(5.001)
        assert world.count_applied(declared) == 1
        evidence = world.application_evidence[0]
        assert evidence.operation_key == attempt.key
        assert evidence.record_id is None
        assert evidence.snapshot.entries_json == ()
        assert len((await world.snapshot()).entries_json) == 1
        assert await world.teardown() == ()

    virtual_loop.run_until_complete(workload())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_model_consultations_capture_independent_immutable_prefixes(
    virtual_loop, tmp_path, backend
):
    world = World(virtual_loop)
    world.add_leg(0)
    store = world.store_for_leg(0, backend=backend, directory=tmp_path)
    model = GatedModel(world, 0, ScriptedModel(["first", "second"]))
    request = ModelRequest(messages=[Message.user("hello")])

    async def workload():
        first = Entry(kind="custom")
        await store.append(world.session_id, [first])
        assert (await model.request(request)).message.text == "first"
        await store.append(world.session_id, [Entry(kind="custom")])
        assert (await model.request(request)).message.text == "second"
        assert [item.consultation for item in world.model_evidence] == [0, 1]
        assert [len(item.snapshot.entry_ids) for item in world.model_evidence] == [1, 2]
        assert all(item.leg == 0 for item in world.model_evidence)
        assert all(
            item.fingerprint == request_fingerprint(request) for item in world.model_evidence
        )
        assert all(
            item.snapshot.independent_read is (backend == "sqlite") for item in world.model_evidence
        )
        assert world.model_evidence[0].snapshot.entry_ids == (first.id,)
        assert await world.teardown() == ()

    virtual_loop.run_until_complete(workload())


def test_dead_leg_model_consultation_parks_before_consuming_reply(virtual_loop):
    world = World(virtual_loop)
    world.add_leg(0)
    world.store_for_leg(0, backend="memory")
    native = ScriptedModel(["answer"])
    model = GatedModel(world, 0, native)
    world.freeze(0, zombie=True)

    async def workload():
        task = world.create_task(0, model.request(ModelRequest(messages=[Message.user("hello")])))
        await asyncio.sleep(0)
        assert not native.requests and not world.model_evidence
        world.thaw(0)
        assert (await task).message.text == "answer"
        assert len(native.requests) == len(world.model_evidence) == 1
        assert await world.teardown() == ()

    virtual_loop.run_until_complete(workload())


def test_model_consultation_ordinals_do_not_collide_between_clients_in_one_leg(virtual_loop):
    world = World(virtual_loop)
    world.add_leg(0)
    world.store_for_leg(0, backend="memory")
    models = [GatedModel(world, 0, ScriptedModel(["ok"])) for _ in range(2)]

    async def workload():
        for model in models:
            await model.request(ModelRequest(messages=[Message.user("hello")]))
        assert [item.consultation for item in world.model_evidence] == [0, 1]
        assert await world.teardown() == ()

    virtual_loop.run_until_complete(workload())


class _ServiceRefusal(Exception):
    pass


class _RefundService:
    mutating = frozenset({"refunds.create"})
    contract_errors = {
        "changed_params": lambda: _ServiceRefusal("changed payload"),
        "concurrent": lambda: _ServiceRefusal("concurrent"),
        "duplicate": lambda: _ServiceRefusal("duplicate"),
    }

    def __init__(self, *, clock, keys):
        assert keys is False
        self.clock, self.keys = clock, keys
        self.refunds = self
        self.rows = []
        self.reads = 0

    async def create(self, payment_id, amount, idempotency_key=None):
        row = _RefundReceipt(
            id=f"refund-{len(self.rows) + 1}", payment_id=payment_id, amount=amount
        )
        self.rows.append(row)
        return row.model_copy(deep=True)

    async def pages(self):
        for row in self.rows:
            self.reads += 1
            yield [row.model_copy(deep=True)]


class _SyncRefundService(_RefundService):
    def create(self, payment_id, amount, idempotency_key=None):
        row = _RefundReceipt(
            id=f"refund-{len(self.rows) + 1}", payment_id=payment_id, amount=amount
        )
        self.rows.append(row)
        return row.model_copy(deep=True)


async def _client_refund_send(
    ctx: EffectContext, attempt: Attempt[_RefundRequest]
) -> _RefundReceipt:
    return await ctx.deps.refunds.create(
        attempt.request.payment_id, attempt.request.amount, attempt.idempotency_key
    )


def _sync_refund_send(ctx: EffectContext, attempt: Attempt[_RefundRequest]) -> _RefundReceipt:
    return ctx.deps.refunds.create(
        attempt.request.payment_id, attempt.request.amount, attempt.idempotency_key
    )


async def _client_refund_lookup(ctx: EffectContext, inquiry: Inquiry[_RefundRequest]):
    async for page in ctx.deps.refunds.pages():
        for row in page:
            if row.payment_id == inquiry.request.payment_id:
                return Found(row, seen={"payment_id": row.payment_id, "amount": row.amount})
    return None


def _client_effect(*, sync=False):
    declared = effect(
        "client.refund",
        key=Key("refund", "payment_id"),
        dedupe=Dedupe(100, "unit", binds_params=True),
    )(_sync_refund_send if sync else _client_refund_send)
    declared.lookup(_client_refund_lookup)
    return declared


@contextmanager
def _live_send(attempt):
    """Bind the real host context for transport-unit tests; no engine is simulated."""
    token = _types._ATTEMPT.set(attempt)
    _ports._set_attempt_live(attempt, True)
    try:
        yield
    finally:
        _ports._set_attempt_live(attempt, False)
        _types._ATTEMPT.reset(token)


async def _dispatch_client(transport, declared, client, attempt, *, leg=0):
    ctx = EffectContext(client)
    with _live_send(attempt):
        return await transport.for_leg(leg).send(
            declared, ctx, attempt, lambda: declared.send(ctx, attempt)
        )


def test_client_transport_executes_real_send_and_shares_only_destination(virtual_loop):
    world = World(virtual_loop)
    world.add_leg(0)
    transport = HarnessTransport(world)
    declared = _client_effect()
    first = transport.service(0, declared, _RefundService)
    world.add_leg(1)
    second = transport.service(1, declared, _RefundService)
    assert first is transport.service(0, declared, _RefundService)
    assert first is not second

    async def workload():
        one = await _dispatch_client(transport, declared, first, _attempt())
        two = await _dispatch_client(transport, declared, second, _attempt(number=2), leg=1)
        assert one == two and one is not two
        assert len(world.service(declared).rows) == world.count_applied(declared) == 1
        assert [row.replayed for row in world.applications(declared)] == [False, True]
        assert all(row.key_passed for row in world.applications(declared))
        assert not world.findings and not transport.not_exercised

    virtual_loop.run_until_complete(workload())


@pytest.mark.parametrize("keys", [(None, None), ("wrong-1", "wrong-2")])
def test_client_missing_or_changed_key_records_actual_duplicate(virtual_loop, keys):
    world = World(virtual_loop)
    world.add_leg(0)
    transport = HarnessTransport(world)
    declared = _client_effect()
    client = transport.service(0, declared, _RefundService)

    async def workload():
        for number, key in enumerate(keys, 1):
            with _live_send(_attempt(number=number)):
                await client.refunds.create("pay-1", 10, idempotency_key=key)
        assert world.count_applied(declared, payment_id="pay-1") == 2
        assert len(world.service(declared).rows) == 2
        assert [row.key_passed for row in world.applications()] == [False, False]
        assert [row["kind"] for row in world.findings] == ["key_not_passed", "key_not_passed"]

    virtual_loop.run_until_complete(workload())


def test_client_contract_refusal_uses_service_exception(virtual_loop):
    world = World(virtual_loop)
    world.add_leg(0)
    transport = HarnessTransport(world)
    declared = _client_effect()
    client = transport.service(0, declared, _RefundService)

    async def workload():
        await _dispatch_client(transport, declared, client, _attempt())
        with pytest.raises(_ServiceRefusal, match="changed payload"):
            await _dispatch_client(transport, declared, client, _attempt(amount=11, number=2))
        assert world.count_applied(declared) == len(world.service(declared).rows) == 1

    virtual_loop.run_until_complete(workload())


def test_client_commit_response_loss_then_recovery_uses_one_application(virtual_loop):
    world = World(virtual_loop)
    world.add_leg(0)
    world.add_leg(1)
    declared = _client_effect()
    plan = FaultPlan("loss", "commit_then_lose_response", 1, method=declared.name)
    controller = FaultController(world, [plan])
    transport = HarnessTransport(world, controller=controller)

    async def workload():
        first = transport.service(0, declared, _RefundService)
        with pytest.raises(TimeoutError):
            await _dispatch_client(transport, declared, first, _attempt())
        second = transport.service(1, declared, _RefundService)
        resumed = replace(_attempt(number=2), deadline=virtual_loop.time() + 20)
        receipt = await _dispatch_client(transport, declared, second, resumed, leg=1)
        assert receipt.id == "refund-1"
        assert world.count_applied(declared) == len(world.service(declared).rows) == 1
        assert controller.not_triggered == ()

    virtual_loop.run_until_complete(workload())


def test_async_generator_lookup_is_gated_for_each_page(virtual_loop):
    world = World(virtual_loop)
    world.add_leg(0)
    transport = HarnessTransport(world)
    declared = _client_effect()
    client = transport.service(0, declared, _RefundService)

    async def workload():
        await _dispatch_client(transport, declared, client, _attempt())
        inquiry = Inquiry(_attempt().request, "refund:pay-1", "operation-1", ("delivery-1",), (), 0)
        ctx = EffectContext(client)
        token = _ports._INQUIRY.set(inquiry)
        try:
            found = await transport.for_leg(0).lookup(
                declared, ctx, inquiry, lambda: declared.lookup_fn(ctx, inquiry)
            )
        finally:
            _ports._INQUIRY.reset(token)
        assert found.receipt.id == "refund-1"
        assert world.service(declared).reads == 1
        assert any(
            item.kind == "service_call" and item.detail.endswith("pages")
            for item in world.boundaries
        )
        assert not world.findings

    virtual_loop.run_until_complete(workload())


def test_frozen_lookup_cannot_consume_next_generator_page(virtual_loop):
    world = World(virtual_loop)
    world.add_leg(0)
    transport = HarnessTransport(world)
    declared = _client_effect()
    client = transport.service(0, declared, _RefundService)

    async def workload():
        await _dispatch_client(transport, declared, client, _attempt())
        inquiry = Inquiry(_attempt().request, "refund:pay-1", "operation-1", (), (), 0)
        token = _ports._INQUIRY.set(inquiry)
        try:
            iterator = client.refunds.pages()
            world.freeze(0, zombie=True)
            task = world.create_task(0, anext(iterator))
            await asyncio.sleep(0)
            assert world.service(declared).reads == 0
            world.thaw(0)
            assert (await task)[0].id == "refund-1"
            await iterator.aclose()
        finally:
            _ports._INQUIRY.reset(token)
        assert await world.teardown() == ()

    virtual_loop.run_until_complete(workload())


def test_client_sync_send_uses_actual_owned_thread(virtual_loop):
    world = World(virtual_loop)
    world.add_leg(0)
    transport = HarnessTransport(world)
    declared = _client_effect(sync=True)
    client = transport.service(0, declared, _SyncRefundService)
    ctx, attempt = EffectContext(client), _attempt()

    async def invoke():
        wrapped, _ = qtools.start_thread("sync refund", declared.send, ctx, attempt)
        return await wrapped

    async def workload():
        with _live_send(attempt):
            result = await transport.for_leg(0).send(declared, ctx, attempt, invoke)
        assert result.id == "refund-1"
        await virtual_loop.threads_drained()
        assert not virtual_loop.pending_threads and not world.findings

    with virtual_loop.observe_tool_threads():
        virtual_loop.run_until_complete(workload())


def test_sync_client_called_on_loop_is_refused_before_mutation(virtual_loop):
    world = World(virtual_loop)
    world.add_leg(0)
    transport = HarnessTransport(world)
    declared = _client_effect(sync=True)
    client = transport.service(0, declared, _SyncRefundService)

    async def workload():
        with _live_send(_attempt()), pytest.raises(ConfigurationError, match="sync send worker"):
            client.refunds.create("pay-1", 10, "delivery-1")
        assert not world.service(declared).rows

    virtual_loop.run_until_complete(workload())


def test_service_call_after_attempt_lifetime_is_reported_and_counted(virtual_loop):
    world = World(virtual_loop)
    world.add_leg(0)
    transport = HarnessTransport(world)
    declared = _client_effect()
    client = transport.service(0, declared, _RefundService)
    attempt = _attempt()

    async def workload():
        token = _types._ATTEMPT.set(attempt)
        try:
            await client.refunds.create("pay-1", 10, "delivery-1")
        finally:
            _types._ATTEMPT.reset(token)
        assert world.count_applied(declared) == 1
        assert [row["kind"] for row in world.findings] == ["a2_violation"]

    virtual_loop.run_until_complete(workload())


def test_service_calls_without_context_are_a2_findings(virtual_loop):
    world = World(virtual_loop)
    world.add_leg(0)
    transport = HarnessTransport(world)
    declared = _client_effect()
    client = transport.service(0, declared, _RefundService)

    async def workload():
        with pytest.raises(ConfigurationError, match="requires an Attempt"):
            await client.refunds.create("pay-1", 10)
        assert [row async for row in client.refunds.pages()] == []
        assert [row["kind"] for row in world.findings] == ["a2_violation", "a2_violation"]
        assert not world.service(declared).rows

    virtual_loop.run_until_complete(workload())


def test_fake_transport_marks_real_send_not_exercised(virtual_loop):
    world = World(virtual_loop)
    world.add_leg(0)
    transport = HarnessTransport(world, mode="fake")
    declared = _refund_effect()
    transport.service(0, declared, _RefundService)
    attempt = _attempt()

    async def workload():
        ctx = EffectContext()
        receipt = await transport.for_leg(0).send(
            declared, ctx, attempt, lambda: declared.send(ctx, attempt)
        )
        assert receipt.payment_id == "pay-1" and world.count_applied(declared) == 1
        assert not world.service(declared).rows
        assert transport.not_exercised == {declared.name}

    virtual_loop.run_until_complete(workload())


def test_forced_client_transport_requires_registered_service(virtual_loop):
    world = World(virtual_loop)
    world.add_leg(0)
    transport = HarnessTransport(world, mode="client")
    declared, attempt = _refund_effect(), _attempt()

    async def workload():
        with pytest.raises(ConfigurationError, match="requires env.service"):
            await transport.for_leg(0).send(declared, EffectContext(), attempt, lambda: None)
        assert world.count_applied(declared) == 0

    virtual_loop.run_until_complete(workload())


def test_crash_env_exposes_fresh_wrappers_and_shared_clock(virtual_loop, tmp_path):
    world = World(virtual_loop)
    world.add_leg(0)
    transport = HarnessTransport(world)
    declared = _client_effect()
    environments = []
    for leg in range(2):
        if leg not in world.legs:
            world.add_leg(leg)
        environments.append(
            CrashEnv(
                world=world,
                transport=transport,
                leg=leg,
                worker_id=f"worker-{leg}",
                recovering=leg > 0,
                store=world.store_for_leg(leg, backend="memory"),
                model=GatedModel(world, leg, ScriptedModel(["ok"])),
                session_id=world.session_id,
                seed=7,
                directory=tmp_path,
            )
        )
    one, two = environments
    assert one.clock is two.clock is world.clock
    assert one.store is not two.store and one.model is not two.model
    assert not one.recovering and two.recovering
    assert (one.leg, two.leg, two.worker_id, one.seed, one.directory) == (
        0,
        1,
        "worker-1",
        7,
        tmp_path,
    )
    assert one.service(declared, _RefundService) is not two.service(declared, _RefundService)


def test_service_registration_rejects_contract_and_factory_drift(virtual_loop):
    world = World(virtual_loop)
    transport = HarnessTransport(world)
    declared = _client_effect()
    transport.service(0, declared, _RefundService)
    with pytest.raises(ConfigurationError, match="changed between legs"):
        transport.service(1, declared, _SyncRefundService)
    with pytest.raises(ConfigurationError, match="changed between legs"):
        transport.service(1, declared, _RefundService, key_arg="another_key")
    changed = _client_effect()
    changed.receipt_identity = ("payment_id",)
    with pytest.raises(ConfigurationError, match="changed between process legs"):
        transport.service(1, changed, _RefundService)
    declared.version = "modified-in-place"
    with pytest.raises(ConfigurationError, match="changed between process legs"):
        transport.service(1, declared, _RefundService)


@pytest.mark.parametrize(
    "attribute,value", [("mutating", {"create"}), ("contract_errors", {"concurrent": 42})]
)
def test_malformed_service_contract_is_rejected(virtual_loop, attribute, value):
    class Broken(_RefundService):
        pass

    setattr(Broken, attribute, value)
    transport = HarnessTransport(World(virtual_loop))
    with pytest.raises(ConfigurationError):
        transport.service(0, _client_effect(), Broken)


def test_mutating_async_generator_is_rejected(virtual_loop):
    class Broken(_RefundService):
        mutating = frozenset({"refunds.pages"})

    world = World(virtual_loop)
    world.add_leg(0)
    declared = _client_effect()
    client = HarnessTransport(world).service(0, declared, Broken)

    async def workload():
        with _live_send(_attempt()), pytest.raises(ConfigurationError, match="async generators"):
            await anext(client.refunds.pages())
        assert not world.service(declared).rows

    virtual_loop.run_until_complete(workload())


def test_other_effect_service_is_not_mislabelled_as_the_current_attempt(virtual_loop):
    world = World(virtual_loop)
    world.add_leg(0)
    transport = HarnessTransport(world)
    declared = _client_effect()
    other = _refund_effect()
    transport.service(0, declared, _RefundService)
    foreign = transport.service(0, other, _RefundService)

    async def workload():
        with _live_send(_attempt()), pytest.raises(ConfigurationError, match="different effect"):
            await transport.for_leg(0).send(
                declared,
                EffectContext(),
                _attempt(),
                lambda: foreign.refunds.create("pay-1", 10, "delivery-1"),
            )
        assert [row["kind"] for row in world.findings] == ["a2_violation"]
        assert not world.applications() and not world.service(other).rows

    virtual_loop.run_until_complete(workload())


def test_keyword_forwarding_service_retains_actual_idempotency_key(virtual_loop):
    class Forwarding(_RefundService):
        async def create(self, **kwargs):
            return await super().create(**kwargs)

    world = World(virtual_loop)
    world.add_leg(0)
    declared = _client_effect()
    client = HarnessTransport(world).service(0, declared, Forwarding)

    async def workload():
        for number in (1, 2):
            with _live_send(_attempt(number=number)):
                await client.refunds.create(
                    payment_id="pay-1", amount=10, idempotency_key="delivery-1"
                )
        assert world.count_applied(declared) == 1
        assert not world.findings
        assert all(row.key_passed for row in world.applications())

    virtual_loop.run_until_complete(workload())


@pytest.mark.parametrize("mismatch", ["leg", "attempt", "inquiry"])
def test_client_dispatch_rejects_foreign_leg_or_context_identity(virtual_loop, mismatch):
    world = World(virtual_loop)
    world.add_leg(0)
    world.add_leg(1)
    transport = HarnessTransport(world)
    declared = _client_effect()
    client = transport.service(0, declared, _RefundService)
    transport.service(1, declared, _RefundService)
    attempt = _attempt()
    ctx = EffectContext(client)

    async def workload():
        if mismatch == "inquiry":
            inquiry = Inquiry(attempt.request, attempt.key, attempt.operation_tag, (), (), 0)
            token = _ports._INQUIRY.set(replace(inquiry))
            try:
                with pytest.raises(ConfigurationError, match="different leg/boundary"):
                    await transport.for_leg(0).lookup(
                        declared, ctx, inquiry, lambda: declared.lookup_fn(ctx, inquiry)
                    )
            finally:
                _ports._INQUIRY.reset(token)
        else:
            with (
                _live_send(attempt),
                pytest.raises(ConfigurationError, match="different leg/boundary"),
            ):
                await transport.for_leg(1 if mismatch == "leg" else 0).send(
                    declared,
                    ctx,
                    attempt if mismatch == "leg" else replace(attempt),
                    lambda: declared.send(ctx, attempt),
                )
        assert [row["kind"] for row in world.findings] == ["a2_violation"]
        assert not world.applications() and not world.service(declared).rows

    virtual_loop.run_until_complete(workload())


def test_stripe_fake_pagination_copies_and_remaining_amount(virtual_loop):
    fake = FakeStripeRefunds(clock=virtual_loop.clock, keys=False)

    async def workload():
        for index in range(25):
            row = await fake.create_refund(charge="ch_A", amount=10, metadata={"index": str(index)})
            assert row.id == f"re_{index + 1}"
            row.metadata["index"] = "outside mutation"
        first = await fake.list_refunds(charge="ch_A")
        assert len(first.data) == 10 and first.has_more
        assert first.data[0].id == "re_25" and first.data[0].metadata == {"index": "24"}
        before = fake.pages_read
        all_rows = [row async for row in fake.iter_refunds(charge="ch_A")]
        assert len(all_rows) == 25 and fake.pages_read - before == 3
        assert all_rows[-1].id == "re_1"
        rest = await fake.create_refund(charge="ch_A")
        assert rest.amount == 9750 and rest.currency == "eur"
        with pytest.raises(fake.InvalidRequest, match="remaining"):
            await fake.create_refund(charge="ch_A", amount=1)

    virtual_loop.run_until_complete(workload())


def test_stripe_fake_keys_false_has_no_hidden_idempotency(virtual_loop):
    fake = FakeStripeRefunds(clock=virtual_loop.clock, keys=False)

    async def workload():
        first = await fake.create_refund(charge="ch_A", amount=10, idempotency_key="same")
        second = await fake.refunds.create(charge="ch_A", amount=10, idempotency_key="same")
        assert (first.id, second.id) == ("re_1", "re_2")
        assert not fake._responses

    virtual_loop.run_until_complete(workload())


def test_stripe_fake_cancelled_pending_request_does_not_poison_key(virtual_loop):
    fake = FakeStripeRefunds(clock=virtual_loop.clock)

    async def workload():
        task = asyncio.create_task(
            fake.create_refund(charge="ch_A", amount=10, idempotency_key="same")
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        result = await fake.create_refund(charge="ch_A", amount=10, idempotency_key="same")
        assert result.id == "re_1"
        assert len([row async for row in fake.iter_refunds(charge="ch_A")]) == 1

    virtual_loop.run_until_complete(workload())


@pytest.mark.parametrize("amount", [True, False, 1.5, -1, 0])
def test_stripe_fake_invalid_amount_does_not_mutate(virtual_loop, amount):
    fake = FakeStripeRefunds(clock=virtual_loop.clock)

    async def workload():
        with pytest.raises(fake.InvalidRequest):
            await fake.create_refund(charge="ch_A", amount=amount, idempotency_key="validation")
        assert not fake._refunds and not fake._responses

    virtual_loop.run_until_complete(workload())


def _boundary(index, kind="custom", *, phase="before", leg=0, method="append", **data):
    entry = Entry(kind=kind, data=data)
    detail = kind + ("/" + data["name"] if "name" in data else "")
    return Boundary(leg, index, phase, method, detail, True, True, 0, (entry.model_dump_json(),))


def test_reference_kill_pruning_retains_aliases_and_never_prunes_zombies():
    trace = [
        _boundary(1, "effect_intent", name="test.refund"),
        _boundary(1, "effect_intent", phase="after", name="test.refund"),
        _boundary(2, "effect_committed", name="test.refund"),
        _boundary(2, "effect_committed", phase="after", name="test.refund"),
    ]
    faults = Faults.parse("kill_before,kill_after,freeze_after_intent")
    raw = reference_points(trace, [_refund_effect()], faults, points="all-raw")
    pruned = reference_points(trace, [_refund_effect()], faults)
    assert len(raw) == 6 and len(pruned) == 5
    merged = [point for point in pruned if point.aliases]
    assert len(merged) == 1
    assert merged[0].aliases == (
        "after(1)/effect_intent/test.refund",
        "before(2)/effect_committed/test.refund",
    )
    assert merged[0].in_effect_window
    assert sum("freeze_after_intent" in point.kinds for point in pruned) == 1
    assert {p.id for p in raw} == {alias for p in pruned for alias in (p.aliases or (p.id,))}


@pytest.mark.parametrize(
    "kind", ["application", "thread_start", "model_request", "service_call", "effect_lookup"]
)
def test_observable_action_prevents_kill_pruning(kind):
    trace = [
        _boundary(1),
        _boundary(1, phase="after"),
        Boundary(0, 0, "observed", kind, "", False, False, 0),
        _boundary(2),
    ]
    points = reference_points(trace, [], Faults.parse("kill_before,kill_after"))
    assert len(points) == 5 and not any(point.aliases for point in points)


def test_effect_window_selection_excludes_unrelated_writes():
    trace = [
        _boundary(1),
        _boundary(2, "effect_intent", name="test.refund"),
        _boundary(3, "custom"),
        _boundary(4, "effect_committed", name="test.refund"),
        _boundary(5, "custom"),
    ]
    points = reference_points(
        trace, [_refund_effect()], Faults.parse("kill_before"), points="effects"
    )
    assert [p.plans[0].index for p in points if p.plans] == [2, 3, 4]


def test_effect_points_are_per_occurrence_and_respect_contract():
    trace = [
        Boundary(0, n, "before", "effect_send", "test.refund", False, False, 0) for n in (1, 2)
    ]
    faults = Faults.parse("commit_late,duplicate_key,rejected_on,unstorable_receipt,slow_arrival")
    opaque = reference_points(trace, [_refund_effect()], faults)
    assert {kind for point in opaque for kind in point.kinds} == {
        "unstorable_receipt",
        "slow_arrival",
    }
    bounded = reference_points(
        trace, [_refund_effect(applies_within_s=2, rejected_on=(ValueError,))], faults
    )
    assert {kind for point in bounded for kind in point.kinds} == {
        "commit_late",
        "rejected_on",
        "unstorable_receipt",
    }
    assert sum("rejected_on" in point.kinds for point in bounded) == 2
    assert any("send(2)" in point.id for point in bounded)
    assert all(p.plans[0].value is ValueError for p in bounded if "rejected_on" in p.kinds)


def test_reference_store_and_lease_shapes_are_distinct():
    trace = [
        _boundary(1, method="acquire_lease_grant"),
        _boundary(2, method="renew_lease"),
        _boundary(3, method="release_lease"),
        _boundary(4, "effect_intent", name="test.refund"),
    ]
    faults = Faults.parse(
        "lease_ack_lost,lease_renew_refused,lease_error,stall,cancel_during,store_step,store_step_beyond_sigma"
    )
    points = reference_points(trace, [_refund_effect()], faults)
    assert sum("lease_ack_lost" in p.kinds for p in points) == 6
    assert sum("lease_renew_refused" in p.kinds for p in points) == 1
    assert sum("lease_error" in p.kinds for p in points) == 1
    assert sum("stall" in p.kinds for p in points) == 1
    assert {p.plans[0].value for p in points if "store_step" in p.kinds} == {-1, 1}
    assert {p.plans[0].value for p in points if "store_step_beyond_sigma" in p.kinds} == {-2, 2}


def test_recovery_depth_two_uses_new_signatures_not_entry_ids():
    reference = [_boundary(1, "effect_committed", name="test.refund", rule="E3", via="send")]
    recovered = [
        _boundary(1, "lease_acquired", leg=1),
        _boundary(2, "effect_committed", leg=1, name="test.refund", rule="E3", via="send"),
        _boundary(3, "effect_committed", leg=1, name="test.refund", rule="P3", via="lookup"),
        _boundary(4, "effect_committed", leg=1, name="test.refund", rule="P3", via="lookup"),
    ]
    prefix = Point("first", (FaultPlan("first", "kill_after", 1, phase="after"),))
    points = recovery_points(prefix, reference, recovered)
    assert len(points) == 4
    assert {p.plans[-1].index for p in points} == {1, 3}
    assert all(p.recovery and p.plans[0] == prefix.plans[0] for p in points)
    assert {p.plans[-1].phase for p in points} == {"before", "after"}


def test_sampling_preserves_reference_kinds_and_first_only_reorders():
    points = [Point("reference")]
    for kind, phase in (("kill_before", "before"), ("kill_after", "after"), ("reject", "call")):
        for index in range(1, 5):
            identity = f"{kind}/{index}"
            points.append(Point(identity, (FaultPlan(identity, kind, index, phase=phase),)))
    chosen = select_points(points, budget=5, seed=7, first=("reject",))
    repeated = select_points(points, budget=5, seed=7, first=("reject",))
    assert chosen == repeated and chosen.sampled and chosen.total == 13
    assert len(chosen.points) == 5 and chosen.points[0].id == "reference"
    assert chosen.covered_kinds == {"kill_before", "kill_after", "reject"}
    assert "reject" in chosen.points[1].kinds
    reordered = select_points(points, first=("reject",))
    assert not reordered.sampled and {p.id for p in reordered.points} == {p.id for p in points}
    with pytest.raises(ValueError, match="cannot preserve"):
        select_points(points, budget=3)


def test_point_caps_and_immutable_variants_are_enforced():
    with pytest.raises(ValueError, match="two crashes"):
        Point(
            "too-many",
            tuple(FaultPlan(str(i), "kill_after", i + 1, phase="after") for i in range(3)),
        )
    with pytest.raises(ValueError, match="effect/store"):
        Point("too-many", (FaultPlan("a", "reject", 1), FaultPlan("b", "stall", 2)))
    with pytest.raises(ValueError, match="one zombie"):
        Point(
            "too-many",
            (FaultPlan("a", "freeze_in_send", 1), FaultPlan("b", "freeze_before_fence2", 2)),
        )
    with pytest.raises(TypeError, match="immutable"):
        Point("mutable", variant=(("list", []),))
    with pytest.raises(TypeError, match="immutable"):
        Point("mutable", (FaultPlan("a", "reject", 1, value={}),))


def test_point_serialization_and_boundary_targets_are_stable():
    prefix = Point("loss", (FaultPlan("loss", "rejected_on", 1, value=ValueError),))
    assert prefix.to_data()["plans"][0]["value"] == {"exception_type": "builtins:ValueError"}
    points = boundary_points(prefix, {"window": 100, "approval": 200})
    assert len(points) == 4
    assert [dict(p.variant)["advance_store_to"] for p in points] == [
        199.999,
        200.001,
        99.999,
        100.001,
    ]
    assert json.dumps([p.to_data() for p in points], sort_keys=True) == json.dumps(
        [p.to_data() for p in boundary_points(prefix, {"approval": 200, "window": 100})],
        sort_keys=True,
    )


def test_actual_world_records_immutable_write_and_thread_application_events(virtual_loop):
    world = World(virtual_loop)
    world.add_leg(0)
    store = world.store_for_leg(0, backend="memory")
    entry = Entry(kind="custom", data={"nested": {"value": 1}})
    fake = ContractFake(world, _refund_effect())

    async def workload():
        await store.append(world.session_id, [entry])
        entry.data["nested"]["value"] = 2
        await fake.send(_attempt(), leg=0, key="delivery-1")
        wrapped, _ = qtools.start_thread("observed", lambda: 7)
        assert await wrapped == 7
        await virtual_loop.threads_drained()
        writes = [b for b in world.boundaries if b.write]
        assert all(json.loads(b.entries_json[0])["data"]["nested"]["value"] == 1 for b in writes)
        assert [b.kind for b in world.boundaries if b.phase == "observed"] == [
            "application",
            "thread_start",
        ]
        assert await world.teardown() == ()

    with virtual_loop.observe_tool_threads():
        virtual_loop.run_until_complete(world.create_task(0, workload()))


def test_effect_window_tracks_distinct_operations_of_same_effect():
    trace = [
        _boundary(1, "effect_intent", name="test.refund", record_id="one"),
        _boundary(2, "effect_intent", name="test.refund", record_id="two"),
        _boundary(3, "effect_committed", name="test.refund", record_id="one"),
        _boundary(4, "custom"),
        _boundary(5, "effect_committed", name="test.refund", record_id="two"),
        _boundary(6, "custom"),
    ]
    points = reference_points(
        trace, [_refund_effect()], Faults.parse("kill_before"), points="effects"
    )
    assert [p.plans[0].index for p in points if p.plans] == [1, 2, 3, 4, 5]


def test_zero_sigma_and_unknown_concurrency_keep_distinct_stable_points():
    points = reference_points([_boundary(1)], [], Faults.parse("store_step"), sigma_s=0)
    assert len(points) == 2 and points[1].plans[0].value == 0
    expanded = concurrency_variants(points, [_refund_effect(Dedupe(100, "unknown"))])
    assert len(expanded) == 3 and expanded[0].id == "reference"
    assert [dict(p.variant)["unknown_concurrent"] for p in expanded[1:]] == ["rejects", "waits"]
    assert (
        concurrency_variants(points, [_refund_effect(Dedupe(100, "known", concurrent="waits"))])
        == points
    )
