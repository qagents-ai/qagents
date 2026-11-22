"""Actual lookup callbacks, durable journals, and operator integration without sends."""

from __future__ import annotations

import asyncio
import threading

import pytest
from pydantic import BaseModel

from qagents.effects import (
    Attempt,
    EffectContext,
    Found,
    Key,
    Settle,
    _canonical,
    _engine,
    _kinds,
    _operator,
    _ports,
    effect,
)
from qagents.effects._types import MISSING, Absent, NotYetVisible, Unknown
from qagents.errors import ConfigurationError, SettlementConflict
from qagents.permissions import Permissions
from qagents.session import Entry, InMemoryStore, LeasedWriter, Session, SQLiteStore, derive_state

from ._samples import committed, failed, intent


class Request(BaseModel):
    request_id: str
    amount: int


class Receipt(BaseModel):
    id: str
    status: str = "complete"


def declaration(callback, *, timeout=1, patch=None, authoritative=False):
    @effect(
        "refund",
        key=Key("refund", "request_id"),
        namespace="shop",
        timeout_s=timeout,
        applies_within_s=2,
        receipt_identity=("id",),
    )
    async def refund(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        pytest.fail("lookup/operator must never call send")

    if callback is not None:
        refund.lookup(callback, authoritative=authoritative)
    if patch is not None:
        refund.on_commit(patch)
    return refund


@pytest.fixture(params=["memory", "sqlite"])
async def db(request, tmp_path):
    clock = [1000.0]
    store = (
        InMemoryStore(clock=lambda: clock[0])
        if request.param == "memory"
        else SQLiteStore(tmp_path / "lookup.db", clock=lambda: clock[0])
    )
    yield store, clock
    await store.aclose()


async def opened(store):
    writer = LeasedWriter(
        store, "session", worker="lookup", ttl_s=30, purpose="recheck", journal_acquisition=True
    )
    assert await writer.open()
    await writer.load()
    return writer


async def seed(writer, *, attempts=1):
    for number in range(1, attempts + 1):
        data = intent(
            attempt=number,
            generation=number - 1,
            epoch=writer.lease.epoch,
            store_ts=1000.0,
            first_use_at=1000.0,
        )
        data["operation_tag"] = _canonical.operation_tag(
            "shop", "refund", "R1", data["request_digest"]
        )
        seq = (await writer.append_effect([Entry(kind="effect_intent", data=data)]))[0]
        await writer.append_effect(
            [
                Entry(
                    kind="effect_failed",
                    data=failed(
                        attempt=number,
                        intent_seq=seq,
                        ended_at=1000.0,
                    ),
                )
            ]
        )
    return writer.view().get("shop", "refund", "R1")


@pytest.fixture
async def journal(db):
    writer = await opened(db[0])
    await seed(writer)
    yield writer
    await writer.close()


async def lookup(writer, declared, *, deps=MISSING, state=None):
    return await _engine.lookup_and_record(
        declared,
        writer.view().get("shop", "refund", "R1"),
        journal=writer,
        deps=deps,
        purpose="recheck",
        state_values=lambda: {} if state is None else state,
    )


async def entries(writer, kind):
    return [e for e in await writer.store.load(writer.session_id) if e.kind == kind]


@pytest.mark.parametrize("binding", ["tag", "seen", "both"])
async def test_bound_found_commits_one_patch_and_repeated_read_does_not_patch_again(
    journal, binding
):
    calls = []

    async def find(ctx, inquiry):
        calls.append((ctx.session_id, _ports.current_inquiry() is inquiry))
        return Found(
            Receipt(id="receipt"),
            **{
                "tag": inquiry.operation_tag if binding in {"tag", "both"} else None,
                "seen": {"amount": 500} if binding in {"seen", "both"} else None,
            },
        )

    patched = []

    def patch(state, request, receipt):
        patched.append(receipt.id)
        state["total"] = state.get("total", 0) + request.amount

    declared = declaration(find, patch=patch)
    await lookup(journal, declared, state={"total": 7})
    await lookup(journal, declared, state={"total": 507})
    assert calls == [("session", True), ("session", True)]
    assert patched == ["receipt"]
    commits = await entries(journal, "effect_committed")
    assert len(commits) == 1
    assert commits[0].data["via"] == "lookup"
    assert commits[0].data["send_ended"] is None
    assert derive_state(await journal.store.load("session")).values["total"] == 507
    assert journal.view().get("shop", "refund", "R1").state == "C"
    assert _ports.current_inquiry() is None


@pytest.mark.parametrize("binding", ["tag", "seen", "both", "unknown_field", "bool_as_int", "nan"])
async def test_mismatched_binding_is_durable_conflict_without_patch(journal, binding):
    async def find(ctx, inquiry):
        return Found(
            Receipt(id="receipt"),
            **{
                "tag": "wrong" if binding == "tag" else inquiry.operation_tag,
                "seen": {"amount": 1}
                if binding in {"seen", "both"}
                else {"extra": 500}
                if binding == "unknown_field"
                else {"amount": True}
                if binding == "bool_as_int"
                else {"amount": float("nan")}
                if binding == "nan"
                else None,
            },
        )

    def patch(*args):
        pytest.fail("mismatched Found cannot patch state")

    await lookup(journal, declaration(find, patch=patch))
    assert journal.view().get("shop", "refund", "R1").state == "X"
    assert not await entries(journal, "effect_committed")
    (conflict,) = await entries(journal, "effect_conflict")
    assert conflict.data["what"] == "found_mismatch"
    assert conflict.id == _kinds.conflict_entry_id(
        conflict.data["record_id"], conflict.data["what"], conflict.data["seqs"]
    )


async def test_inquiry_covers_every_attempt_and_request_mutation_cannot_change_binding(db):
    writer = await opened(db[0])
    original = await seed(writer, attempts=2)
    seen = []

    async def find(ctx, inquiry):
        seen.append(inquiry)
        inquiry.request.amount = 999
        return Found(Receipt(id="receipt"), seen={"amount": 999})

    try:
        await lookup(writer, declaration(find))
        (inquiry,) = seen
        assert inquiry.key == "R1" and inquiry.first_use_at == 1000
        assert inquiry.idempotency_keys == ("q1_0", "q1_1")
        assert [(a.number, a.generation, a.outcome) for a in inquiry.attempts] == [
            (1, 0, "unknown"),
            (2, 1, "unknown"),
        ]
        assert original.request["amount"] == 500
        assert writer.view().get("shop", "refund", "R1").state == "X"
    finally:
        await writer.close()


@pytest.mark.parametrize(
    "verdict",
    [
        Absent(),
        Unknown("offline"),
        NotYetVisible(2),
        NotYetVisible(float("nan")),
        NotYetVisible(-1),
        object(),
    ],
)
async def test_observation_keeps_store_start_time_and_never_makes_absence_final(
    journal, db, verdict
):
    async def find(ctx, inquiry):
        db[1][0] += 5
        return verdict

    result = await lookup(journal, declaration(find, authoritative=True))
    (observed,) = await entries(journal, "effect_observed")
    assert observed.data["started_at"] == 1000
    assert observed.data["authoritative"] is True
    assert observed.data["final"] is False
    assert journal.view().get("shop", "refund", "R1").state == "D"
    if isinstance(verdict, NotYetVisible) and verdict.retry_after_s == 2:
        assert observed.data["retry_after"] == 1002
    elif not isinstance(verdict, (Absent, Unknown)):
        assert isinstance(result, Unknown)


async def test_missing_dependencies_propagate_without_an_observation(journal):
    async def find(ctx, inquiry):
        return ctx.deps

    with pytest.raises(ConfigurationError, match="pass deps"):
        await lookup(journal, declaration(find))
    assert not await entries(journal, "effect_observed")
    assert _ports.current_inquiry() is None


async def test_lookup_exception_and_invalid_receipt_are_distinct(journal):
    async def broken(ctx, inquiry):
        raise OSError("read unavailable")

    assert isinstance(await lookup(journal, declaration(broken)), Unknown)
    assert journal.view().get("shop", "refund", "R1").state == "D"

    async def found(ctx, inquiry):
        return Found({"status": "missing identity"}, tag=inquiry.operation_tag)

    await lookup(journal, declaration(found))
    (committed_entry,) = await entries(journal, "effect_committed")
    assert committed_entry.data["receipt"] is None
    assert committed_entry.data["receipt_error"]
    assert journal.view().get("shop", "refund", "R1").state == "C"


async def test_failed_on_commit_preserves_lookup_receipt(journal):
    async def found(ctx, inquiry):
        return Found(Receipt(id="receipt"), tag=inquiry.operation_tag)

    def patch(state, request, receipt):
        state["partial"] = 1
        raise ValueError("cannot update")

    await lookup(journal, declaration(found, patch=patch))
    (entry,) = await entries(journal, "effect_committed")
    assert entry.data["receipt"]["id"] == "receipt"
    assert entry.data["state_patch"] is None
    assert "cannot update" in entry.data["on_commit_error"]


async def test_transport_wraps_actual_async_callback_and_context(journal):
    visits = []

    class Transport:
        async def lookup(self, effect, ctx, inquiry, call_lookup):
            visits.append(("transport", _ports.current_inquiry() is inquiry))
            return await call_lookup()

    async def find(ctx, inquiry):
        visits.append(("callback", _ports.current_inquiry() is inquiry))
        assert ctx.deps == {"local": True}
        return Absent()

    token = _ports.TRANSPORT.set(Transport())
    try:
        await lookup(journal, declaration(find), deps={"local": True})
    finally:
        _ports.TRANSPORT.reset(token)
    assert visits == [("transport", True), ("callback", True)]


async def test_sync_lookup_uses_worker_thread_and_copied_inquiry(journal):
    loop_thread = threading.get_ident()

    def find(ctx, inquiry):
        assert threading.get_ident() != loop_thread
        assert _ports.current_inquiry() is inquiry
        return Found(Receipt(id="receipt"), tag=inquiry.operation_tag)

    await lookup(journal, declaration(find))
    assert journal.view().get("shop", "refund", "R1").state == "C"


async def test_sync_timeout_tracks_raw_thread_until_drain(journal):
    started, release = threading.Event(), threading.Event()
    finished = threading.Event()

    def find(ctx, inquiry):
        started.set()
        assert release.wait(3)
        finished.set()
        return Absent()

    try:
        result = await asyncio.wait_for(lookup(journal, declaration(find, timeout=0.03)), 1)
        assert started.is_set() and isinstance(result, Unknown)
        assert journal.lease.running_sends() > 0
        draining = asyncio.create_task(journal.lease.sends_drained())
        await asyncio.sleep(0)
        assert not draining.done()
    finally:
        release.set()
    await asyncio.wait_for(draining, 2)
    assert finished.is_set()
    assert not await entries(journal, "effect_send_ended")
    assert len(await entries(journal, "effect_observed")) == 1


async def test_cancellation_resistant_async_timeout_is_bounded_and_late_found_not_committed(
    journal,
):
    cancelled, release = asyncio.Event(), asyncio.Event()

    async def find(ctx, inquiry):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        return Found(Receipt(id="late"), tag=inquiry.operation_tag)

    try:
        result = await asyncio.wait_for(lookup(journal, declaration(find, timeout=0.02)), 1)
        await asyncio.wait_for(cancelled.wait(), 1)
        assert isinstance(result, Unknown)
        assert journal.lease.running_sends() > 0
    finally:
        release.set()
    await asyncio.wait_for(journal.lease.sends_drained(), 1)
    assert not await entries(journal, "effect_committed")


async def test_external_cancel_propagates_and_keeps_lookup_in_drain(journal):
    started, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def find(ctx, inquiry):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        return Absent()

    task = asyncio.create_task(lookup(journal, declaration(find)))
    try:
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(cancelled.wait(), 1)
        assert journal.lease.running_sends() > 0
        assert not await entries(journal, "effect_observed")
    finally:
        release.set()
    await asyncio.wait_for(journal.lease.sends_drained(), 1)


async def test_commit_arriving_during_lookup_does_not_apply_patch_twice(journal):
    async def find(ctx, inquiry):
        target = journal.view().get("shop", "refund", "R1").attempts[-1]
        await journal.append_effect(
            [
                Entry(
                    kind="effect_committed",
                    data=committed(
                        intent_seq=target.intent_seq,
                        receipt={"id": "receipt", "status": "complete"},
                    ),
                )
            ]
        )
        return Found(Receipt(id="receipt"), tag=inquiry.operation_tag)

    def patch(*args):
        pytest.fail("lookup must not repeat a completed state patch")

    await lookup(journal, declaration(find, patch=patch))
    assert len(await entries(journal, "effect_committed")) == 1


async def test_existing_conflict_records_receipts_without_patching_until_operator_resolution(db):
    writer = await opened(db[0])
    op = await seed(writer)
    await writer.append_effect(
        [
            Entry(
                kind="effect_conflict",
                data={
                    "namespace": op.namespace,
                    "name": op.name,
                    "key": op.key,
                    "record_id": op.record_id,
                    "what": "descriptor",
                    "seqs": [op.revision],
                },
            )
        ]
    )
    selected = ["first"]
    patches = []

    async def find(ctx, inquiry):
        return Found(Receipt(id=selected[0]), tag=inquiry.operation_tag)

    def patch(state, request, receipt):
        patches.append(receipt.id)
        state["receipt"] = receipt.id

    declared = declaration(find, patch=patch)
    try:
        await lookup(writer, declared)
        await lookup(writer, declared)
        selected[0] = "second"
        await lookup(writer, declared)
        await lookup(writer, declared)
        assert patches == []
        assert writer.view().get("shop", "refund", "R1").state == "X"
        assert len(await entries(writer, "effect_committed")) == 2
    finally:
        await writer.close()

    # A lookup that cannot decide leaves the operator free to resolve reviewed X.
    async def unknown(ctx, inquiry):
        return Unknown()

    declared.lookup_fn = unknown
    result = await _operator.settle(
        Session(db[0], id="session"),
        Settle.committed(
            op.record_id, receipt=Receipt(id="second"), by="operator", evidence="reviewed"
        ),
        effects={"refund": declared},
        permissions=Permissions(),
        lease_ttl_s=30,
        worker="operator",
    )
    assert result.via == "operator" and patches == ["second"]
    writer = await opened(db[0])
    try:
        declared.lookup_fn = find
        selected[0] = "third"
        await lookup(writer, declared)
        assert writer.view().get("shop", "refund", "R1").state == "X"
        assert patches == ["second"]
    finally:
        await writer.close()


async def test_callback_cancel_is_unknown_and_does_not_cancel_caller(journal):
    async def find(ctx, inquiry):
        raise asyncio.CancelledError()

    assert isinstance(await lookup(journal, declaration(find)), Unknown)
    assert len(await entries(journal, "effect_observed")) == 1


async def test_found_evidence_is_snapshotted_before_journal_await(journal, monkeypatch):
    receipt = Receipt(id="original")
    seen = {"amount": 500}

    async def find(ctx, inquiry):
        return Found(receipt, seen=seen)

    original_append = journal.append_effect

    async def append(batch):
        if any(e.kind == "effect_observed" for e in batch):
            receipt.id = "mutated"
            seen["amount"] = -1
        return await original_append(batch)

    monkeypatch.setattr(journal, "append_effect", append)
    await lookup(journal, declaration(find))
    (entry,) = await entries(journal, "effect_committed")
    assert entry.data["receipt"]["id"] == "original"
    assert entry.data["found_seen"] == {"amount": 500}


@pytest.mark.parametrize("answer", ["committed", "not_applied"])
async def test_actual_lookup_during_settlement_keeps_found_evidence(db, answer):
    writer = await opened(db[0])
    op = await seed(writer)
    await writer.close()

    async def find(ctx, inquiry):
        return Found(Receipt(id="receipt"), tag=inquiry.operation_tag)

    declared = declaration(find)
    request = (
        Settle.committed(
            op.record_id, receipt=Receipt(id="receipt"), by="operator", evidence="read"
        )
        if answer == "committed"
        else Settle.not_applied(op.record_id, by="operator", evidence="manual")
    )
    call = _operator.settle(
        Session(db[0], id="session"),
        request,
        effects={"refund": declared},
        permissions=Permissions(),
        lease_ttl_s=30,
        worker="settle",
    )
    if answer == "committed":
        assert (await call).via == "lookup"
    else:
        with pytest.raises(SettlementConflict, match="contradicted"):
            await call
    commits = [e for e in await db[0].load("session") if e.kind == "effect_committed"]
    assert len(commits) == 1 and commits[0].data["via"] == "lookup"


@pytest.mark.parametrize("closure", ["not_applied", "close_unknown"])
async def test_actual_recheck_after_operator_closure_records_late_found(db, closure):
    writer = await opened(db[0])
    op = await seed(writer)
    await writer.close()
    session = Session(db[0], id="session")
    args = dict(
        effects={"refund": declaration(None)},
        permissions=Permissions(),
        lease_ttl_s=30,
        worker="operator",
    )
    answer = (
        Settle.not_applied(op.record_id, by="operator", evidence="manual")
        if closure == "not_applied"
        else Settle.close_unknown(op.record_id, by="operator", reason="manual")
    )
    await _operator.settle(session, answer, **args)

    async def find(ctx, inquiry):
        return Found(Receipt(id="late"), tag=inquiry.operation_tag)

    args["effects"] = {"refund": declaration(find)}
    held = await _operator.recheck(session, **args)
    assert len(held) == 1
    # The first late Found moves closure to X. A second read still cannot apply
    # the receipt's patch or overwrite the operator's closure with a commit.
    assert len(await _operator.recheck(session, **args)) == 1
    persisted = await db[0].load("session")
    assert not any(e.kind == "effect_committed" for e in persisted)
    assert any(
        e.kind == "effect_conflict" and e.data["what"] == "late_found_after_settlement"
        for e in persisted
    )
