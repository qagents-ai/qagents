"""O5 against real stores and permissions; lookup port doubles explicitly isolate O2."""

from __future__ import annotations

import asyncio
import copy
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel

from qagents.effects import (
    Attempt,
    EffectContext,
    Found,
    Inquiry,
    Key,
    Settle,
    _canonical,
    _engine,
    _operator,
    effect,
)
from qagents.effects._types import MISSING, Absent, Unknown
from qagents.errors import ConfigurationError, SettlementConflict
from qagents.permissions import Permissions
from qagents.session import Entry, InMemoryStore, LeasedWriter, Session, SQLiteStore

from ._samples import committed, failed, intent, reference


async def test_inferred_conflict_uses_stable_entry_identity(store):
    from qagents.effects import _kinds

    db, _ = store
    session, _ = await seed(db, state="C")
    writer = LeasedWriter(db, session.id, worker="conflict", ttl_s=30, purpose="settle")
    assert await writer.open()
    await writer.load()
    try:
        op = writer.view().get("shop", "refund", "R1")
        await writer.append_effect(
            [
                Entry(
                    kind="effect_committed",
                    data=committed(
                        intent_seq=op.attempts[-1].intent_seq,
                        receipt={"id": "different", "status": "complete"},
                    ),
                )
            ]
        )
        op = writer.view().get("shop", "refund", "R1")
        (finding,) = op.findings
        await _operator._findings(writer, op)
        persisted = [e for e in await db.load(session.id) if e.kind == "effect_conflict"]
        assert len(persisted) == 1
        assert persisted[0].id == _kinds.conflict_entry_id(
            finding.record_id, finding.what, finding.seqs
        )
    finally:
        await writer.close()


class Request(BaseModel):
    request_id: str
    amount: int


class Receipt(BaseModel):
    id: str
    status: str


def declaration(*, lookup=False, authoritative=False, patch=False):
    @effect(
        "refund",
        key=Key("refund", "request_id"),
        namespace="shop",
        timeout_s=1,
        applies_within_s=2,
        receipt_identity=("id",),
    )
    async def refund(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        pytest.fail("operator paths must never send")

    if lookup:

        @refund.lookup(authoritative=authoritative)
        async def find(ctx: EffectContext, inquiry: Inquiry[Request]):
            return Absent()

    if patch:

        @refund.on_commit
        def remember(state, request: Request, receipt: Receipt):
            state["refund"] = receipt.id
            state["total"] = state.get("total", 0) + request.amount

    return refund


@pytest.fixture(params=["memory", "sqlite"])
async def store(request, tmp_path):
    clock = [1000.0]
    db = (
        InMemoryStore(clock=lambda: clock[0])
        if request.param == "memory"
        else SQLiteStore(tmp_path / "settle.db", clock=lambda: clock[0])
    )
    yield db, clock
    await db.aclose()


async def seed(db, *, key="R1", state="D", bound=2, ended=True, now=1000.0):
    session = Session(db, id="session")
    writer = LeasedWriter(
        db, session.id, worker="seed", ttl_s=30, purpose="run", journal_acquisition=True
    )
    assert await writer.open()
    await writer.load()
    try:
        payload = intent(key=key, store_ts=now, first_use_at=now, epoch=writer.lease.epoch)
        payload["timing"]["applies_within_s"] = bound
        seq = (await writer.append_effect([Entry(kind="effect_intent", data=payload)]))[0]
        await writer.append_effect(
            [
                Entry(
                    kind="effect_failed",
                    data=failed(
                        key=key,
                        intent_seq=seq,
                        ended_at=now if ended else None,
                        send_ended=ended,
                    ),
                )
            ]
        )
        if state == "C":
            await writer.append_effect(
                [
                    Entry(
                        kind="effect_committed",
                        data=committed(
                            key=key,
                            intent_seq=seq,
                            ended_at=now,
                        ),
                    )
                ]
            )
        elif state == "X":
            await writer.append_effect(
                [
                    Entry(
                        kind="effect_conflict",
                        data={
                            **reference(key=key),
                            "what": "descriptor",
                            "seqs": [seq],
                        },
                    )
                ]
            )
    finally:
        await writer.close()
    return session, (await session.effects())[-1]


async def settle(session, answer, declared=None, permissions=None, **kwargs):
    declared = declared or declaration()
    return await _operator.settle(
        session,
        answer,
        effects={declared.name: declared},
        permissions=permissions or Permissions(),
        lease_ttl_s=30,
        worker="operator",
        **kwargs,
    )


async def recheck(session, declared, **kwargs):
    return await _operator.recheck(
        session,
        effects={declared.name: declared},
        permissions=Permissions(),
        lease_ttl_s=30,
        worker="operator",
        **kwargs,
    )


@pytest.fixture
def lookup_port(monkeypatch):
    """A typed O2 boundary double; it is not an engine or real-run integration test."""
    config = {"verdict": Unknown(), "calls": [], "require_deps": False, "wait": None}

    async def lookup(effect, op, *, journal, deps, purpose, state_values):
        config["calls"].append((purpose, deps, copy.deepcopy(state_values())))
        if config["require_deps"]:
            assert EffectContext(deps).deps is deps
        if config["wait"] is not None:
            entered, release = config["wait"]
            entered.set()
            await release.wait()
        verdict = config["verdict"]
        ref = {"record_id": op.record_id, "namespace": op.namespace, "name": op.name, "key": op.key}
        now = await journal.store_now()
        await journal.append_effect(
            [
                Entry(
                    kind="effect_observed",
                    data={
                        **ref,
                        "attempts": [a.number for a in op.attempts],
                        "verdict": "found"
                        if isinstance(verdict, Found)
                        else "absent"
                        if isinstance(verdict, Absent)
                        else "unknown",
                        "authoritative": effect.authoritative,
                        "started_at": config.get("started_at", now),
                    },
                )
            ]
        )
        if isinstance(verdict, Found):
            if op.state in {"NA", "CU"}:
                await journal.append_effect(
                    [
                        Entry(
                            kind="effect_conflict",
                            data={
                                **ref,
                                "what": "late_found_after_settlement",
                                "seqs": [op.revision],
                            },
                        )
                    ]
                )
            else:
                receipt, error = _engine.receipt_json(effect, verdict.receipt)
                assert error is None
                patch, patch_error = _engine.on_commit_patch(
                    effect, state_values(), op.request, receipt
                )
                last = op.attempts[-1]
                await journal.append_effect(
                    [
                        Entry(
                            kind="effect_committed",
                            data={
                                **ref,
                                "attempt": last.number,
                                "intent_seq": last.intent_seq,
                                "via": "lookup",
                                "receipt": receipt,
                                "identity_digest": _canonical.identity_digest(
                                    receipt, effect.receipt_identity
                                ),
                                "state_patch": patch.model_dump() if patch is not None else None,
                                "on_commit_error": patch_error,
                            },
                        )
                    ]
                )
        return verdict

    monkeypatch.setattr(_engine, "lookup_and_record", lookup)
    return config


async def test_settlement_requires_own_lease_and_busy_call_writes_nothing(store):
    db, _ = store
    session, held = await seed(db)
    before = await session.entries()
    assert await db.acquire_lease(session.id, "busy", 30)
    with pytest.raises(SettlementConflict) as failure:
        await settle(session, Settle.not_applied(held, by="ops", evidence="checked"))
    assert failure.value.reason == "session_busy"
    assert await session.entries() == before
    await db.release_lease(session.id, "busy")
    record = await settle(session, Settle.not_applied(held, by="ops", evidence="checked"))
    assert record.status == "not_applied"
    entries = await session.entries()
    assert entries[-2].kind == "lease_acquired" and entries[-2].data["purpose"] == "settle"
    assert entries[-1].run_id is None
    assert (await db.status(session.id)).lease_owner is None


async def test_revision_and_committed_at_entry_are_public_conflicts(store):
    db, _ = store
    session, held = await seed(db)
    answer = Settle.not_applied(held, by="ops", evidence="checked")
    settled = await settle(session, answer)
    with pytest.raises(SettlementConflict) as failure:
        await settle(session, answer)
    assert failure.value.reason == "revision"
    with pytest.raises(SettlementConflict) as failure:
        await settle(session, Settle.not_applied(settled, by="ops", evidence="again"))
    assert failure.value.reason == "already_settled"


async def test_committed_entry_refused_before_lookup(store, lookup_port):
    db, _ = store
    session, committed_record = await seed(db, state="C")
    with pytest.raises(SettlementConflict) as failure:
        await settle(
            session,
            Settle.committed(
                committed_record, {"id": "refund-1", "status": "pending"}, by="ops", evidence="same"
            ),
            declaration(lookup=True),
        )
    assert failure.value.reason == "already_settled" and lookup_port["calls"] == []


async def test_prefix_answer_resolves_fresh_record_and_ambiguous_prefix_refused(store):
    db, _ = store
    session, held = await seed(db)
    await seed(db, key="R2")
    with pytest.raises(SettlementConflict) as failure:
        await settle(session, Settle.close_unknown("ef_", by="ops", reason="review"))
    assert failure.value.reason == "revision"
    record = await settle(session, Settle.close_unknown(held.id[:-1], by="ops", reason="review"))
    assert record.id == held.id and record.status == "closed_unknown"


async def test_directly_built_empty_record_id_never_settles_a_lone_record(store):
    db, _ = store
    session, held = await seed(db)
    before = await db.load(session.id)
    with pytest.raises(ValueError, match="record id"):
        await settle(session, Settle("close_unknown", "", None, None, "ops", None, "review"))
    assert await db.load(session.id) == before
    assert [record.status for record in await session.effects()] == ["held"]


async def test_operator_commit_runs_patch_on_durable_state_and_updates_claim(store):
    db, _ = store
    await db.append("session", [Entry(kind="state_updated", data={"values": {"total": 7}})])
    session, held = await seed(db)
    record = await settle(
        session,
        Settle.committed(
            held, Receipt(id="approved", status="done"), by="ops", evidence="destination"
        ),
        declaration(patch=True),
    )
    assert record.status == "committed" and record.via == "operator"
    assert (await session.state()).values == {"total": 507, "refund": "approved"}
    claim = await db.get_claim("shop", "refund", "R1")
    assert claim.status == "committed" and claim.receipt["id"] == "approved"


@pytest.mark.parametrize("outcome", ["not_applied", "close_unknown", "resend"])
async def test_noncommit_settlements_keep_claim_owned(store, outcome):
    db, clock = store
    session, held = await seed(db)
    clock[0] += 10
    answer = getattr(Settle, outcome)(
        held,
        by="ops",
        **({"reason": "checked"} if outcome == "close_unknown" else {"evidence": "checked"}),
    )
    await settle(session, answer)
    claim = await db.get_claim("shop", "refund", "R1")
    assert claim.status == "held" and claim.session_id == session.id


@pytest.mark.parametrize("outcome", ["committed", "not_applied", "close_unknown"])
async def test_x_rows_resolve_prior_conflicts(store, outcome):
    db, _ = store
    session, held = await seed(db, state="X")
    if outcome == "committed":
        answer = Settle.committed(
            held, Receipt(id="settled", status="done"), by="ops", evidence="reviewed"
        )
    elif outcome == "not_applied":
        answer = Settle.not_applied(held, by="ops", evidence="reviewed")
    else:
        answer = Settle.close_unknown(held, by="ops", reason="reviewed")
    record = await settle(session, answer)
    assert (
        record.status
        == {
            "committed": "committed",
            "not_applied": "not_applied",
            "close_unknown": "closed_unknown",
        }[outcome]
    )
    assert any(entry.kind == "effect_conflict" for entry in await session.entries())


async def test_invalid_receipt_or_permission_refusal_never_settles(store):
    db, _ = store
    session, held = await seed(db)
    with pytest.raises(SettlementConflict) as failure:
        await settle(session, Settle.committed(held, {"invalid": 1}, by="ops", evidence="checked"))
    assert failure.value.reason == "refused"
    seen = []

    async def authorize(answer, record):
        seen.append(record)
        return False

    with pytest.raises(SettlementConflict) as failure:
        await settle(
            session,
            Settle.not_applied(held, by="ops", evidence="checked"),
            permissions=Permissions(authorize_settlement=authorize),
        )
    assert failure.value.reason == "refused" and seen[0].id == held.id
    assert (await session.effects())[0].status == "held"


@pytest.mark.parametrize("commit", [False, True])
async def test_settlement_mac_binds_session_kind_and_every_payload_field(store, commit):
    db, _ = store
    session, held = await seed(db)
    permissions = Permissions(approval_secret="a-secret-of-at-least-sixteen-bytes")
    answer = (
        Settle.committed(held, Receipt(id="one", status="done"), by="ops", evidence="audit")
        if commit
        else Settle.not_applied(held, by="ops", evidence="audit")
    )
    await settle(session, answer, permissions=permissions)
    entry = (await session.entries())[-1]
    payload = {key: value for key, value in entry.data.items() if key != "mac"}
    envelope = {"session_id": session.id, "kind": entry.kind, "data": payload}
    assert entry.data["mac"] == permissions.sign_settlement(envelope)
    for name in ("session_id", "kind"):
        assert entry.data["mac"] != permissions.sign_settlement({**envelope, name: "other"})
    assert entry.data["mac"] != permissions.sign_settlement(
        {**envelope, "data": {**payload, "by": "attacker"}}
    )


async def test_resend_too_early_reports_time_and_covers_existing_attempts(store):
    db, clock = store
    session, held = await seed(db)
    answer = Settle.resend(held, by="ops", evidence="no application")
    with pytest.raises(SettlementConflict) as failure:
        await settle(session, answer)
    assert failure.value.reason == "too_early"
    assert failure.value.earliest == datetime.fromtimestamp(1006, UTC)
    clock[0] = 1006
    record = await settle(session, answer)
    assert record.status == "held" and record.settlement.outcome == "resend"
    assert (await session.entries())[-1].data["covers_attempts"] == [1]


async def test_unbounded_resend_uses_minimum_age(store):
    db, clock = store
    session, held = await seed(db, bound=None, ended=False)
    permissions = Permissions(resend_min_age_s=20)
    answer = Settle.resend(held, by="ops", evidence="checked")
    with pytest.raises(SettlementConflict) as failure:
        await settle(session, answer, permissions=permissions)
    assert failure.value.earliest == datetime.fromtimestamp(1020, UTC)
    clock[0] = 1020
    assert (await settle(session, answer, permissions=permissions)).settlement.outcome == "resend"


async def test_observations_do_not_starve_revision_and_recheck_never_sends(store, lookup_port):
    db, _ = store
    session, held = await seed(db)
    declared = declaration(lookup=True)
    assert (await recheck(session, declared))[0].revision == held.revision
    assert (await recheck(session, declared))[0].revision == held.revision
    result = await settle(session, Settle.not_applied(held, by="ops", evidence="checked"), declared)
    assert result.status == "not_applied"
    assert [call[0] for call in lookup_port["calls"]] == ["recheck", "recheck", "settle"]


@pytest.mark.parametrize("compatible", [True, False])
async def test_lookup_commit_during_settlement_is_not_overwritten_or_patched_twice(
    store, lookup_port, compatible
):
    db, _ = store
    session, held = await seed(db)
    lookup_port["verdict"] = Found(Receipt(id="destination", status="done"), tag="op1_refund")
    declared = declaration(lookup=True, authoritative=True, patch=True)
    answer = Settle.committed(
        held,
        Receipt(id="destination" if compatible else "wrong", status="done"),
        by="ops",
        evidence="checked",
    )
    if compatible:
        assert (await settle(session, answer, declared)).via == "lookup"
    else:
        with pytest.raises(SettlementConflict) as failure:
            await settle(session, answer, declared)
        assert failure.value.reason == "contradicted"
    assert (await session.state()).values["total"] == 500
    commits = [entry for entry in await session.entries() if entry.kind == "effect_committed"]
    assert len(commits) == 1 and commits[0].data["via"] == "lookup"


async def test_authoritative_found_refuses_not_applied_and_preserves_evidence(store, lookup_port):
    db, _ = store
    session, held = await seed(db)
    lookup_port["verdict"] = Found(Receipt(id="actual", status="done"), tag="op1_refund")
    with pytest.raises(SettlementConflict) as failure:
        await settle(
            session,
            Settle.not_applied(held, by="ops", evidence="wrong"),
            declaration(lookup=True, authoritative=True),
        )
    assert failure.value.reason == "contradicted"
    assert (await session.effects())[0].receipt["id"] == "actual"


async def test_later_found_on_closed_answer_becomes_conflict(store, lookup_port):
    db, _ = store
    session, held = await seed(db)
    await settle(session, Settle.close_unknown(held, by="ops", reason="stopped"))
    lookup_port["verdict"] = Found(Receipt(id="late", status="done"), tag="op1_refund")
    assert (await recheck(session, declaration(lookup=True)))[0].reason.value == "CONFLICT"
    assert (await session.effects())[0].status == "conflict"


async def test_missing_dependencies_propagate_before_settlement(store, lookup_port):
    db, _ = store
    session, held = await seed(db)
    lookup_port["require_deps"] = True
    with pytest.raises(ConfigurationError, match="deps"):
        await settle(
            session,
            Settle.not_applied(held, by="ops", evidence="checked"),
            declaration(lookup=True),
        )
    assert lookup_port["calls"][0][1] is MISSING
    assert (await session.effects())[0].status == "held"


async def test_two_operator_answers_cannot_both_write(store, lookup_port):
    db, _ = store
    session, held = await seed(db)
    entered, release = asyncio.Event(), asyncio.Event()
    lookup_port["wait"] = (entered, release)
    answer = Settle.not_applied(held, by="first", evidence="checked")
    first = asyncio.create_task(settle(session, answer, declaration(lookup=True)))
    await asyncio.wait_for(entered.wait(), 1)
    try:
        with pytest.raises(SettlementConflict) as failure:
            await settle(session, Settle.close_unknown(held, by="second", reason="other"))
        assert failure.value.reason == "session_busy"
    finally:
        release.set()
    assert (await first).status == "not_applied"
    assert len([entry for entry in await session.entries() if entry.kind == "effect_settled"]) == 1


@pytest.mark.parametrize("lookup_started", [1001.0, 1010.0])
async def test_authoritative_absence_contradicts_only_when_lookup_started_after_quiescence(
    store, lookup_port, lookup_started
):
    db, clock = store
    session, held = await seed(db)
    clock[0] = 1010
    lookup_port["verdict"] = Absent()
    lookup_port["started_at"] = lookup_started
    answer = Settle.committed(
        held, Receipt(id="operator", status="done"), by="ops", evidence="checked"
    )
    if lookup_started >= 1006:
        with pytest.raises(SettlementConflict) as failure:
            await settle(session, answer, declaration(lookup=True, authoritative=True))
        assert failure.value.reason == "contradicted"
        assert (await session.effects())[0].status == "held"
    else:
        assert (
            await settle(session, answer, declaration(lookup=True, authoritative=True))
        ).status == "committed"


async def test_lease_takeover_during_authorization_refuses_stale_writer(store):
    db, clock = store
    session, held = await seed(db)

    async def authorize(answer, record):
        clock[0] += 40
        assert await db.acquire_lease(session.id, "new-owner", 30)
        return True

    with pytest.raises(SettlementConflict) as failure:
        await settle(
            session,
            Settle.not_applied(held, by="ops", evidence="checked"),
            permissions=Permissions(authorize_settlement=authorize),
        )
    assert failure.value.reason == "session_busy"
    assert (await session.effects())[0].status == "held"
    assert (await db.status(session.id)).lease_owner == "new-owner"
    await db.release_lease(session.id, "new-owner")


async def test_cancelled_lookup_releases_operator_lease_without_an_answer(store, lookup_port):
    db, _ = store
    session, held = await seed(db)
    entered, release = asyncio.Event(), asyncio.Event()
    lookup_port["wait"] = (entered, release)
    task = asyncio.create_task(
        settle(
            session,
            Settle.not_applied(held, by="ops", evidence="checked"),
            declaration(lookup=True),
        )
    )
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (await db.status(session.id)).lease_owner is None
    assert not any(entry.kind == "effect_settled" for entry in await session.entries())


async def test_operator_commit_ack_loss_reads_back_one_atomic_claim_and_entry():
    class LostAck(InMemoryStore):
        lost = False

        async def append_leased(self, session_id, entries, owner):
            seqs = await super().append_leased(session_id, entries, owner)
            if not self.lost and any(entry.kind == "effect_committed" for entry in entries):
                self.lost = True
                raise ConnectionError("commit acknowledgement lost")
            return seqs

    db = LostAck(clock=lambda: 1000.0)
    session, held = await seed(db)
    result = await settle(
        session,
        Settle.committed(held, Receipt(id="done", status="done"), by="ops", evidence="checked"),
    )
    assert result.status == "committed"
    entries = [entry for entry in await session.entries() if entry.kind == "effect_committed"]
    assert len(entries) == 1 and db.lost
    claim = await db.get_claim("shop", "refund", "R1")
    assert claim.status == "committed" and claim.receipt == result.receipt


async def test_authorization_uses_an_isolated_answer_snapshot(store):
    db, _ = store
    session, held = await seed(db)
    receipt = Receipt(id="approved", status="done")
    answer = Settle.committed(held, receipt, by="ops", evidence="checked")
    entered, release = asyncio.Event(), asyncio.Event()

    async def authorize(observed, record):
        assert observed.receipt.id == "approved"
        observed.receipt.id = "callback-edit"
        entered.set()
        await release.wait()
        return True

    task = asyncio.create_task(
        settle(session, answer, permissions=Permissions(authorize_settlement=authorize))
    )
    await asyncio.wait_for(entered.wait(), 1)
    receipt.id = "caller-edit"
    release.set()
    result = await task
    assert result.receipt["id"] == "approved"


async def test_later_attempt_voids_operator_resend(store):
    db, clock = store
    session, held = await seed(db)
    clock[0] = 1010
    record = await settle(session, Settle.resend(held, by="ops", evidence="checked"))
    writer = LeasedWriter(
        db, session.id, worker="recovery", ttl_s=30, purpose="recovery", journal_acquisition=True
    )
    assert await writer.open()
    await writer.load()
    try:
        payload = intent(attempt=2, epoch=writer.lease.epoch, store_ts=clock[0], rule="D3")
        await writer.append_effect([Entry(kind="effect_intent", data=payload)])
    finally:
        await writer.close()
    latest = (await session.effects())[0]
    assert latest.settlement.voided and latest.revision > record.revision
    with pytest.raises(SettlementConflict) as failure:
        await settle(session, Settle.not_applied(record, by="ops", evidence="stale"))
    assert failure.value.reason == "revision"
