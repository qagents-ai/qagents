"""The real perform engine against a leased journal and a controlled destination."""

from contextlib import asynccontextmanager
from dataclasses import replace

import pytest

from qagents.effects import Attempt, Dedupe, EffectContext, Found, Key, Settle, effect
from qagents.effects import _engine as engine
from qagents.effects import _operator as operator
from qagents.effects._types import Absent, HoldReason, Unknown
from qagents.errors import EffectConflictError, EffectHeld, EffectReceiptError, EffectRejected
from qagents.permissions import Permissions
from qagents.session import InMemoryStore, LeasedWriter, Session, SQLiteStore, derive_state

from .test_engine_dispatch import Receipt, Request, host


@pytest.fixture(params=("memory", "sqlite"))
async def store(request, tmp_path):
    now = [1000.0]

    def clock():
        return now[0]

    value = (
        InMemoryStore(clock=clock)
        if request.param == "memory"
        else SQLiteStore(tmp_path / "effects.sqlite", clock=clock)
    )
    async with value:
        yield value, now


@asynccontextmanager
async def opened(store, *, session="session", operation="run:1:call"):
    writer = LeasedWriter(
        store,
        session,
        worker="test",
        ttl_s=30,
        purpose="run",
        journal_acquisition=True,
    )
    assert await writer.open()
    try:
        await writer.load()
        scope, state, events = host(writer)
        yield replace(scope, session_id=session, operation_id=operation), state, events
    finally:
        await writer.close()


async def test_d1_p3_p1_p2_and_p1k(store):
    db, _ = store
    calls = []

    @effect("refund", key=Key("request", "id"))
    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        entries = await db.load(ctx.session_id)
        assert entries[-1].kind == "effect_intent"
        calls.append(attempt.request.id)
        return Receipt(id="paid")

    async with opened(db) as (scope, _, events):
        assert (await engine.perform(send, Request(id="R1"), scope=scope)).id == "paid"
        assert (await engine.perform(send, Request(id="R1"), scope=scope)).id == "paid"
        assert events[-1].replayed
        with pytest.raises(EffectHeld) as diverged:
            await engine.perform(send, Request(id="R1", amount=2), scope=scope)
        assert diverged.value.held.reason == HoldReason.DIVERGED
        with pytest.raises(EffectHeld) as changed:
            await engine.perform(send, Request(id="R2"), scope=scope)
        assert changed.value.held.reason == HoldReason.KEY_CHANGED
        with pytest.raises(EffectConflictError):
            await engine.perform(
                send,
                Request(id="R1", amount=2),
                scope=replace(scope, operation_id="run:2:call"),
            )
    assert calls == ["R1"]


@pytest.mark.parametrize("cache_rejection", [False, True])
async def test_d2_checks_lookup_and_preserves_or_advances_generation(store, cache_rejection):
    db, _ = store
    calls = []
    looked = []
    contract = Dedupe(3600, contract="fixture", replays_rejections=cache_rejection)

    @effect("refund", key=Key("request", "id"), dedupe=contract)
    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        calls.append(attempt.idempotency_key)
        if len(calls) == 1:
            raise EffectRejected("declined")
        return Receipt(id="paid")

    @send.lookup
    async def find(ctx, inquiry):
        looked.append(inquiry.key)
        return Absent()

    async with opened(db) as (scope, _, _):
        with pytest.raises(EffectRejected, match="declined"):
            await engine.perform(send, Request(id="R1"), scope=scope)
        with pytest.raises(EffectRejected):
            await engine.perform(send, Request(id="R1"), scope=scope)
        assert len(calls) == 1
        later = replace(scope, operation_id="run:2:call")
        assert (await engine.perform(send, Request(id="R1"), scope=later)).id == "paid"
    assert len(looked) == 1
    assert (calls[0] != calls[1]) is cache_rejection


@pytest.mark.parametrize("cached_error", [False, True])
async def test_e4_resends_only_once_and_respects_cached_error(store, cached_error):
    db, _ = store
    calls, looked = [], []
    contract = Dedupe(3600, contract="fixture", concurrent="waits", replays_errors=cached_error)

    @effect("refund", key=Key("request", "id"), dedupe=contract)
    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        calls.append(attempt.idempotency_key)
        raise ConnectionError("ack lost")

    @send.lookup
    async def find(ctx, inquiry):
        looked.append(inquiry)
        return Unknown()

    async with opened(db) as (scope, _, _):
        with pytest.raises(EffectHeld):
            await engine.perform(send, Request(id="R1"), scope=scope)
    assert len(calls) == (1 if cached_error else 2)
    assert len(looked) == len(calls)
    assert len(set(calls)) == 1


async def test_l1_recovered_receipt_commits_and_applies_patch_once(store):
    db, _ = store

    @effect("refund", key=Key("request", "id"))
    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        raise ConnectionError("ack lost")

    @send.lookup
    async def find(ctx, inquiry):
        return Found(Receipt(id="paid"), tag=inquiry.operation_tag)

    @send.on_commit
    def patch(state, request, receipt):
        state["patch_count"] = state.get("patch_count", 0) + 1

    async with opened(db) as (scope, state, _):
        for _ in range(2):
            assert (await engine.perform(send, Request(id="R1"), scope=scope)).id == "paid"
        assert state == {"patch_count": 1}
    entries = await db.load("session")
    assert [e.data["via"] for e in entries if e.kind == "effect_committed"] == ["lookup"]
    assert derive_state(entries).values == {"patch_count": 1}


async def test_d4_waits_for_a_lookup_that_started_after_quiescence(store):
    db, now = store
    calls = []

    @effect("refund", key=Key("request", "id"), applies_within_s=1)
    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        calls.append(attempt.idempotency_key)
        if len(calls) == 1:
            raise ConnectionError("ack lost")
        return Receipt(id="paid")

    @send.lookup(authoritative=True)
    async def find(ctx, inquiry):
        return Absent()

    async with opened(db) as (scope, _, _):
        with pytest.raises(EffectHeld) as first:
            await engine.perform(send, Request(id="R1"), scope=scope)
        assert first.value.held.reason == HoldReason.NOT_YET_QUIESCENT
        assert first.value.held.retry_after is not None
    now[0] = first.value.held.retry_after.timestamp() + 1
    async with opened(db) as (scope, _, _):
        assert (await engine.perform(send, Request(id="R1"), scope=scope)).id == "paid"
    entries = await db.load("session")
    assert [e.data["rule"] for e in entries if e.kind == "effect_intent"] == ["D1", "D4"]
    assert len(set(calls)) == 1
    assert any(e.kind == "effect_observed" and e.data["final"] for e in entries)


async def test_d5_operator_resend_keeps_key_and_is_consumed(store):
    db, now = store
    calls = []
    permissions = Permissions()

    @effect("refund", key=Key("request", "id"))
    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        calls.append(attempt.idempotency_key)
        if len(calls) == 1:
            raise ConnectionError("ack lost")
        return Receipt(id="paid")

    async with opened(db) as (scope, _, _):
        with pytest.raises(EffectHeld) as first:
            await engine.perform(send, Request(id="R1"), scope=scope)
    now[0] += permissions.resend_min_age_s + 1
    await operator.settle(
        Session(db, id="session"),
        Settle.resend(first.value.held.id, by="operator", evidence="destination audit"),
        effects={"refund": send},
        permissions=permissions,
        lease_ttl_s=30,
        worker="operator",
    )
    async with opened(db) as (scope, _, _):
        assert (await engine.perform(send, Request(id="R1"), scope=scope)).id == "paid"
    entries = await db.load("session")
    intents = [e for e in entries if e.kind == "effect_intent"]
    settlement = next(e for e in entries if e.kind == "effect_settled")
    assert [e.data["rule"] for e in intents] == ["D1", "D5"]
    assert intents[-1].data["authorization"]["settlement_seq"] == settlement.seq
    assert len(set(calls)) == 1


async def test_c1_claim_replay_cannot_bypass_descriptor_on_later_reads(store):
    db, _ = store
    calls = []

    @effect("refund", key=Key("request", "id"), namespace="shop")
    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        calls.append(attempt.request.id)
        return Receipt(id="paid")

    async with opened(db, session="one") as (scope, _, _):
        await engine.perform(send, Request(id="R1"), scope=scope)
    async with opened(db, session="two") as (scope, _, _):
        assert (await engine.perform(send, Request(id="R1"), scope=scope)).id == "paid"
        with pytest.raises(EffectConflictError):
            await engine.perform(send, Request(id="R1", amount=9), scope=scope)
    assert calls == ["R1"]
    assert [e.data["via"] for e in await db.load("two") if e.kind == "effect_committed"] == [
        "claim"
    ]


async def test_e2_unstorable_receipt_returns_error_after_commit(store):
    db, _ = store

    @effect("refund", key=Key("request", "id"))
    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        return {"wrong": "receipt"}

    async with opened(db) as (scope, _, _):
        for _ in range(2):
            with pytest.raises(EffectReceiptError):
                await engine.perform(send, Request(id="R1"), scope=scope)
    entries = await db.load("session")
    assert sum(e.kind == "effect_intent" for e in entries) == 1
    assert next(e for e in entries if e.kind == "effect_committed").data["receipt_error"]
