from __future__ import annotations

import asyncio
import contextvars
import importlib
import threading
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from pydantic import BaseModel

from qagents import RunContext, tool
from qagents.effects import Attempt, EffectContext, Key, _engine, effect
from qagents.effects._ports import Authorization, CallScope
from qagents.effects._types import _ATTEMPT, current_attempt
from qagents.errors import ConfigurationError, ToolError
from qagents.hooks import HookChain, Hooks
from qagents.messages import ToolCallPart, ToolResultPart

tools = importlib.import_module("qagents.tools")


class Request(BaseModel):
    reference: str


class Receipt(BaseModel):
    reference: str


@effect("sample.send", key=Key("sample", "reference"), timeout_s=1)
async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
    return Receipt(reference=attempt.request.reference)


@pytest.fixture
def pool(monkeypatch):
    monkeypatch.setattr(tools, "_EXECUTOR", None)
    monkeypatch.setattr(tools, "_EXECUTOR_SIZE", 0)
    monkeypatch.setattr(tools, "TOOL_THREADS", 1)
    monkeypatch.setattr(tools, "_ABANDONED", set())
    previous = tools.set_thread_observer(None)
    try:
        yield
    finally:
        tools.set_thread_observer(previous)
        if tools._EXECUTOR is not None:
            tools._EXECUTOR.shutdown(wait=True, cancel_futures=True)


def scope(*, memos=None, record=None, attempt=1):
    async def default_record(name, value):
        pass

    return CallScope(
        session_id="s",
        run_id="r",
        tool_name="refund",
        tool_call_id="c",
        operation_id="op",
        tool_attempt=attempt,
        effects=frozenset({send.name}),
        authorization=Authorization("allow"),
        original_authorization=None,
        arguments_digest=None,
        original_arguments_digest=None,
        composition_refusal=None,
        deps=None,
        emit=lambda _: None,
        journal=None,
        permissions=None,
        memos={} if memos is None else memos,
        record_memo=record or default_record,
        committed_values=lambda: {},
        apply_patch=lambda _: None,
    )


@pytest.mark.parametrize("flags", [{"read_only": True}, {"deferred": True}])
def test_effects_tools_refuse_incompatible_flags(flags):
    async def body(ctx):
        pass

    with pytest.raises(ConfigurationError):
        tool(body, effects=[send], **flags)


def test_effects_tools_refuse_sync_body_and_invalid_declarations():
    with pytest.raises(ConfigurationError, match="async"):
        tool(lambda: None, name="sync", effects=[send])

    async def body(ctx):
        pass

    with pytest.raises(ConfigurationError, match="Effect"):
        tool(body, effects=[object()])


def test_tool_timeout_alias_is_unambiguous_and_effects_are_frozen():
    async def body(ctx):
        pass

    with pytest.raises(ConfigurationError, match="timeout"):
        tool(body, timeout=3, timeout_s=4)
    declarations = [send]
    built = tool(body, effects=declarations, timeout_s=9)
    declarations.clear()
    assert built.effects == (send,) and built.timeout == 9 and built.origin is None


async def test_perform_requires_body_scope_and_delegates_exactly(monkeypatch):
    ctx = RunContext()
    with pytest.raises(ConfigurationError, match="body.*effects"):
        await ctx.perform(send, Request(reference="one"))
    seen = []
    receipt = Receipt(reference="one")

    async def perform(declared, request, *, scope, occurrence):
        seen.append((declared, request, scope, occurrence))
        return receipt

    monkeypatch.setattr(_engine, "perform", perform)
    ctx._call = scope(attempt=3)
    request = Request(reference="one")
    assert await ctx.perform(send, request, occurrence=2) is receipt
    assert seen == [(send, request, ctx._call, 2)]
    assert ctx.tool_attempt == 3 and ctx.is_rerun
    assert RunContext().tool_attempt == 1 and not RunContext().is_rerun


async def test_memo_requires_body_scope():
    with pytest.raises(ConfigurationError, match="body"):
        await RunContext().memo("request", lambda: 1)


async def test_memo_journals_once_and_revives_without_aliases():
    entries = []

    async def record(name, value):
        entries.append((name, value))

    ctx = RunContext(_call=scope(record=record))
    calls = []

    async def compute():
        calls.append(True)
        return Request(reference="original")

    first = await ctx.memo("request", compute, type=Request)
    first.reference = "mutated"
    again = await ctx.memo("request", compute, type=Request)
    assert again.reference == "original"
    replay = RunContext(_call=scope(memos=ctx._call.memos, record=record, attempt=2))
    assert (await replay.memo("request", compute, type=Request)).reference == "original"
    assert calls == [True] and entries == [("request", {"reference": "original"})]
    plain = await replay.memo("request", compute)
    plain["reference"] = "also changed"
    assert replay._call.memos["request"] == {"reference": "original"}


async def test_failed_memo_record_is_not_cached_and_same_name_overlap_is_refused():
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def compute():
        calls.append(True)
        entered.set()
        await release.wait()
        return {"value": 1}

    async def failed_record(name, value):
        raise OSError("journal unavailable")

    ctx = RunContext(_call=scope(record=failed_record))
    first = asyncio.create_task(ctx.memo("one", compute))
    await asyncio.wait_for(entered.wait(), 1)
    try:
        with pytest.raises(ConfigurationError, match="running"):
            await ctx.memo("one", compute)
    finally:
        release.set()
    with pytest.raises(OSError):
        await first
    assert calls == [True] and ctx._call.memos == {}
    ctx._call = scope()
    assert await ctx.memo("one", lambda: None) is None
    assert await ctx.memo("one", lambda: pytest.fail("cached None was lost")) is None


@pytest.mark.parametrize("value", [float("nan"), {"unsafe": 2**53}])
async def test_memo_refuses_noncanonical_values_before_journaling(value):
    recorded = []

    async def record(name, value):
        recorded.append(value)

    ctx = RunContext(_call=scope(record=record))
    with pytest.raises((ValueError, ConfigurationError)):
        await ctx.memo("invalid", lambda: value)
    assert recorded == [] and ctx._call.memos == {}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 2**53])
async def test_memo_refuses_hidden_noncanonical_model_numbers(value):
    class Nested(BaseModel):
        data: dict[str, Any]

    recorded = []

    async def record(name, value):
        recorded.append(value)

    ctx = RunContext(_call=scope(record=record))
    with pytest.raises((ValueError, ConfigurationError)):
        await ctx.memo("invalid", lambda: {"model": Nested(data={"items": [value]})})
    assert recorded == [] and ctx._call.memos == {}


async def test_memo_keeps_lossless_model_datetime_uuid_conversion():
    class Stamp(BaseModel):
        when: datetime
        id: UUID

    expected = Stamp(when=datetime(2026, 1, 1, tzinfo=UTC), id=UUID(int=42))
    ctx = RunContext(_call=scope())
    assert await ctx.memo("stamp", lambda: expected, type=Stamp) == expected
    assert await ctx.memo("stamp", lambda: pytest.fail("recomputed"), type=Stamp) == expected


async def test_call_tool_refuses_effects_before_outside_dispatch():
    called = []

    async def outside(*args, **kwargs):
        called.append(True)

    @tool(effects=[send])
    async def refund(ctx):
        pass

    ctx = RunContext(tool_call_id="parent", _run=SimpleNamespace(_outside_call=outside))
    with pytest.raises(ConfigurationError, match="effects"):
        await ctx.call_tool(refund, {}, source="test")
    assert called == []


async def test_thread_observer_precedes_body_and_result_delivery_precedes_accounting(pool):
    loop = asyncio.get_running_loop()
    marker = contextvars.ContextVar("marker", default="absent")
    marker.set("copied")
    holder, sequence = {}, []
    owner = threading.get_ident()
    counted = asyncio.Event()

    def observe(future):
        assert threading.get_ident() == owner
        holder["future"] = future
        sequence.append("observed")

        def accounted():
            sequence.append(("accounted", holder["wrapped"].done()))
            counted.set()

        future.add_done_callback(lambda _: loop.call_soon_threadsafe(accounted))

    previous = tools.set_thread_observer(observe)
    assert previous is None

    def body():
        future = holder["future"]
        assert future.running()
        assert future._qagents_thread_id == threading.get_ident() != owner
        assert sequence[0] == "observed"
        return marker.get()

    wrapped, raw = tools.start_thread("sample", body)
    holder["wrapped"] = wrapped
    assert await wrapped == "copied"
    await asyncio.wait_for(counted.wait(), 1)
    # wrap_future schedules delivery before the observer schedules accounting.
    assert wrapped.done() and raw.done()
    assert ("accounted", True) in sequence


async def test_thread_context_preserves_current_attempt(pool):
    attempt = object()
    token = _ATTEMPT.set(attempt)
    try:
        wrapped, _ = tools.start_thread("attempt", current_attempt)
        assert await wrapped is attempt
    finally:
        _ATTEMPT.reset(token)
    assert current_attempt() is None


async def test_thread_reservations_are_single_use_and_released_idempotently(pool):
    slot = tools.reserve_thread()
    assert slot is not None and tools.reserve_thread() is None
    with pytest.raises(ToolError):
        tools.start_thread("unreserved", lambda: None)
    slot.release()
    slot.release()
    next_slot = tools.reserve_thread()
    assert next_slot is not None
    with pytest.raises(ConfigurationError):
        tools.start_thread("released", lambda: None, slot=slot)
    wrapped, _ = tools.start_thread("reserved", lambda: 7, slot=next_slot)
    assert await wrapped == 7
    with pytest.raises(ConfigurationError):
        tools.start_thread("reused", lambda: None, slot=next_slot)
    next_slot.release()
    last = tools.reserve_thread()
    assert last is not None
    last.release()


@pytest.mark.parametrize("error", [RuntimeError("observer failed"), asyncio.CancelledError()])
async def test_observer_failure_never_runs_body_or_leaks_slot(pool, error):
    called, seen = [], []

    def broken(future):
        seen.append(future)
        raise error

    slot = tools.reserve_thread()
    tools.set_thread_observer(broken)
    with pytest.raises(type(error)):
        tools.start_thread("blocked", lambda: called.append(True), slot=slot)
    tools.set_thread_observer(None)
    wrapped, _ = tools.start_thread("cleanup", lambda: 5)
    assert await wrapped == 5
    assert called == [] and seen[0].done()
    fresh = tools.reserve_thread()
    assert fresh is not None
    fresh.release()


async def test_submit_failure_releases_reservation(pool, monkeypatch):
    slot = tools.reserve_thread()

    def failed(*args, **kwargs):
        raise RuntimeError("submit failed")

    monkeypatch.setattr(tools._EXECUTOR, "submit", failed)
    with pytest.raises(RuntimeError, match="submit failed"):
        tools.start_thread("failed", lambda: None, slot=slot)
    fresh = tools.reserve_thread()
    assert fresh is not None
    fresh.release()


async def test_queued_thread_cancellation_never_enters_user_body(pool):
    release = threading.Event()
    called = []
    first, _ = tools.start_thread("occupy", lambda: release.wait(5))
    try:
        queued, raw = tools.start_thread("queued", lambda: called.append(True))
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        await asyncio.sleep(0)  # propagate cancellation to the queued concurrent Future
        assert raw.cancelled()
    finally:
        release.set()
        await first
    assert called == []
    slot = tools.reserve_thread()
    assert slot is not None
    slot.release()


async def test_reservation_survives_newly_abandoned_work(pool):
    release = threading.Event()
    first, raw = tools.start_thread("active", lambda: release.wait(5))
    reserved = tools.reserve_thread()
    assert reserved is not None
    try:
        tools._abandon("active", raw, "test")
        # The engine may have journaled its intent since reserving. Admission is
        # still guaranteed, even if actual execution must queue behind old work.
        admitted, _ = tools.start_thread("reserved", lambda: "sent", slot=reserved)
        assert not admitted.done()
    finally:
        release.set()
        await first
    assert await admitted == "sent"


async def test_cancelled_memo_releases_name_without_publishing_a_value():
    entered = asyncio.Event()

    async def compute():
        entered.set()
        await asyncio.Event().wait()

    ctx = RunContext(_call=scope())
    task = asyncio.create_task(ctx.memo("one", compute))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ctx._call.memos == {}
    assert await ctx.memo("one", lambda: {"finished": True}) == {"finished": True}


async def test_distinct_memos_may_run_concurrently_and_publish_after_recording():
    entered, finish = asyncio.Event(), asyncio.Event()

    async def record(name, value):
        if name == "pending":
            entered.set()
            await finish.wait()

    ctx = RunContext(_call=scope(record=record))
    pending = asyncio.create_task(ctx.memo("pending", lambda: {"x": 1}))
    await asyncio.wait_for(entered.wait(), 1)
    try:
        assert "pending" not in ctx._call.memos
        assert await ctx.memo("other", lambda: 2) == 2
    finally:
        finish.set()
    assert await pending == {"x": 1}
    assert ctx._call.memos == {"pending": {"x": 1}, "other": 2}


async def test_abandoned_pool_refuses_before_reservation(pool):
    ready, release = threading.Event(), threading.Event()

    def body():
        ready.set()
        release.wait(5)

    wrapped, raw = tools.start_thread("stuck", body)
    try:
        assert await asyncio.to_thread(ready.wait, 1)
        tools._abandon("stuck", raw, "test")
        assert tools.reserve_thread() is None
        with pytest.raises(ToolError):
            tools.start_thread("later", lambda: None)
    finally:
        release.set()
        await wrapped
    fresh = tools.reserve_thread()
    assert fresh is not None
    fresh.release()


async def test_tool_run_uses_the_thread_observer(pool):
    seen = []
    tools.set_thread_observer(seen.append)

    @tool
    def sync_tool():
        return "done"

    assert await sync_tool.run(RunContext(), {}) == "done"
    assert len(seen) == 1 and seen[0].result() == "done"


async def test_sync_tool_arguments_do_not_collide_with_dispatch_keywords(pool):
    @tool
    def sync_tool(name: str, function: str, slot: str):
        return name, function, slot

    assert await sync_tool.run(
        RunContext(), {"name": "named", "function": "callable", "slot": "reserved"}
    ) == ("named", "callable", "reserved")


class Edit(Hooks):
    async def before_tool(self, ctx, call):
        call.arguments["nested"]["value"] = 2


class LastEdit(Hooks):
    async def before_tool(self, ctx, call):
        return call.model_copy(update={"arguments": {"final": True}})


async def test_hook_trace_reports_last_edit_including_in_place_changes():
    call = ToolCallPart(id="c", name="tool", arguments={"nested": {"value": 1}})
    changed, by = await HookChain([Edit(), Hooks()]).before_tool_traced(RunContext(), call)
    assert changed.arguments == {"nested": {"value": 2}}
    assert by == f"{Edit.__module__}.{Edit.__qualname__}"
    call.arguments = {"nested": {"value": 1}}
    changed, by = await HookChain([Edit(), LastEdit(), Hooks()]).before_tool_traced(
        RunContext(), call
    )
    assert changed.arguments == {"final": True}
    assert by == f"{LastEdit.__module__}.{LastEdit.__qualname__}"


async def test_hook_trace_detects_boolean_integer_change_and_preserves_short_circuit():
    class BoolEdit(Hooks):
        async def before_tool(self, ctx, call):
            call.arguments["value"] = True

    class Stop(Hooks):
        async def before_tool(self, ctx, call):
            return ToolResultPart(tool_call_id=call.id, name=call.name, content="stop")

    class Never(Hooks):
        async def before_tool(self, ctx, call):
            pytest.fail("ran after short circuit")

    call = ToolCallPart(id="c", name="tool", arguments={"value": 1})
    result, by = await HookChain([BoolEdit(), Stop(), Never()]).before_tool_traced(
        RunContext(), call
    )
    assert isinstance(result, ToolResultPart) and result.content == "stop"
    assert by == f"{BoolEdit.__module__}.{BoolEdit.__qualname__}"
    same, by = await HookChain([Hooks()]).before_tool_traced(RunContext(), call)
    assert same is call and by is None


async def test_hook_exception_is_not_swallowed():
    class Broken(Hooks):
        async def before_tool(self, ctx, call):
            raise RuntimeError("hook failed")

    with pytest.raises(RuntimeError, match="hook failed"):
        await HookChain([Broken()]).before_tool_traced(
            RunContext(), ToolCallPart(id="c", name="tool")
        )
