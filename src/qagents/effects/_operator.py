"""Fenced operator answers and read-only destination rechecks."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import inspect
from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from qagents import session as _session
from qagents.errors import ConfigurationError, LeaseError, SettlementConflict
from qagents.session import Session

from . import _canonical, _engine, _fold, _kinds, _ports, _quiescence, _table
from ._declare import Effect
from ._fold import Operation
from ._types import MISSING, Absent, EffectRecord, Found, HeldEffect, Settle

if TYPE_CHECKING:
    from qagents.permissions import Permissions

# The record statuses (EffectRecord.status) a forged resend can stand on.
_STATUS = {"N": "never attempted", "R": "rejected", "NA": "not_applied", "CU": "closed_unknown"}


@contextlib.asynccontextmanager
async def _take_lease(
    session: Session, *, ttl_s: float, worker: str, purpose: str
) -> AsyncIterator[_session.LeasedWriter]:
    await session._resolve()
    missing = _session.EFFECTS_REQUIRED - session.store.features
    if missing:
        raise ConfigurationError(
            f"Operator effects require store features: {', '.join(sorted(missing))}."
        )
    writer = _session.LeasedWriter(
        session.store,
        session.id,
        worker=worker,
        ttl_s=ttl_s,
        purpose=purpose,
        journal_acquisition=True,
    )
    if not await writer.open():
        raise SettlementConflict(reason="session_busy")
    # The operator works in its caller's task, so a lost lease cancels that task. Only
    # that one cancel is the lease's own: it is taken back, and a cancel from outside
    # propagates, counted from the caller's cancelling() now (as in the runtime).
    owner = asyncio.current_task()
    mark = 0 if owner is None else owner.cancelling()
    lost = False  # the keeper reported the lease lost and cancelled the owner, once
    owed = False  # that cancel is not taken back yet

    def on_lost(reason: str) -> None:
        nonlocal lost, owed
        if owner is not None and not lost:
            lost = owed = True
            owner.cancel()

    def take_back() -> bool:
        nonlocal owed
        if not owed or owner is None:
            return False
        owed = False
        return _take_back_cancel(owner, mark)

    keeper = asyncio.create_task(writer.keep(on_lost=on_lost))
    try:
        await writer.load()
        yield writer
    except asyncio.CancelledError:
        if take_back():  # the lease's own cancel: another worker took the session over
            raise SettlementConflict(reason="session_busy") from None
        raise  # a cancel from outside, with or without the lease's own
    except LeaseError as error:
        raise SettlementConflict(reason="session_busy") from error
    finally:
        keeper.cancel()
        try:
            await asyncio.wait([keeper])  # unlike awaiting it, lets an outside cancel through
        finally:
            take_back()  # a lease's cancel that the operator's code caught
            await writer.close()
        if not keeper.cancelled():
            keeper.result()  # a keeper that failed fails the call, as before


def _take_back_cancel(owner: asyncio.Task[Any], mark: int) -> bool:
    """Take back the one cancel `_take_lease` gave ``owner`` for a lost lease.

    True when no other cancel request remains (``owner.cancelling()`` is back at
    ``mark``, its count when the lease was taken): the CancelledError is then the
    lease's own and becomes ``SettlementConflict(reason="session_busy")``.
    """
    return owner.uncancel() <= mark


def _resolve(writer: _session.LeasedWriter, reference: str) -> Operation:
    matches = [op for op in writer.view().operations.values() if op.record_id.startswith(reference)]
    if len(matches) != 1:
        raise SettlementConflict(
            "Unknown or ambiguous effect record id; reload the record.", reason="revision"
        )
    return matches[0]


def _record(writer: _session.LeasedWriter, op: Operation) -> EffectRecord:
    return next(
        record
        for record in writer.view().records(writer.session_id)
        if (record.namespace, record.effect, record.key) == (op.namespace, op.name, op.key)
    )


def _declaration(effects: Mapping[str, Effect], op: Operation) -> Effect:
    effect = effects.get(op.name)
    if not isinstance(effect, Effect):
        raise ConfigurationError(f"No effect declaration is available for {op.name!r}.")
    return effect


def _ref(op: Operation) -> dict[str, str]:
    return {"record_id": op.record_id, "namespace": op.namespace, "name": op.name, "key": op.key}


async def _findings(writer: _session.LeasedWriter, op: Operation) -> Operation:
    for finding in list(op.findings):
        await writer.append_effect(
            [
                _session.Entry(
                    id=_kinds.conflict_entry_id(finding.record_id, finding.what, finding.seqs),
                    kind=_kinds.KIND_EFFECT_CONFLICT,
                    data=finding.model_dump(mode="json"),
                )
            ]
        )
    return writer.view().get(op.namespace, op.name, op.key)


def _revision(op: Operation, answer: Settle) -> None:
    if answer.expected_revision is not None and answer.expected_revision != op.revision:
        raise SettlementConflict(reason="revision")


def _answer_state(permissions: Permissions, session_id: str, op: Operation, answer: Settle) -> str:
    """The state a resend is checked against (O3r). With signed pauses, a journaled resend
    that does not verify is no operator answer, so a new resend is checked against the
    record's status without any such resend, the pending one and every earlier one: forged
    resends cannot reopen a not_applied, closed_unknown or rejected record for a real one
    (M25, M67, SPEC 5.3). The other answers send nothing, so they are checked against the
    folded state and can close such a record again."""
    if answer.outcome != "resend":
        return op.state
    kept = [
        settlement
        for settlement in op.settlements
        if settlement.outcome != "resend"
        or _engine._settlement_verified(permissions, session_id, settlement)
    ]
    if len(kept) == len(op.settlements):
        return op.state
    pending = op.pending_resend if any(item is op.pending_resend for item in kept) else None
    return _fold._status_of(replace(op, settlements=kept, pending_resend=pending))


def _eligible(op: Operation, answer: Settle, state: str) -> None:
    if state not in {"D", "X"} or (answer.outcome == "resend" and state != "D"):
        if state != op.state:  # only a resend standing on an unverified one
            raise SettlementConflict(
                "Settlement refused: already_settled. The record's pending resend does not "
                "verify under approval_secret, and without the resends that do not verify "
                f"the record is {_STATUS.get(state, state)}, so it is not resent.",
                reason="already_settled",
            )
        raise SettlementConflict(reason="already_settled" if state != "X" else "refused")


def _reach(writer: _session.LeasedWriter, op: Operation, rule: str, point: str | None) -> None:
    _ports.reached(_ports.ReachEvent(writer.session_id, op.record_id, op.name, rule, point, None))


async def _lookup(
    writer: _session.LeasedWriter, effect: Effect, op: Operation, deps: Any, purpose: str
):
    entries = await writer.load()
    op = writer.view().get(op.namespace, op.name, op.key)
    return await _engine.lookup_and_record(
        effect,
        op,
        journal=writer,
        deps=deps,
        purpose=purpose,
        state_values=lambda: _session.derive_state(entries).values,
    )


async def settle(
    session: Session,
    answer: Settle,
    *,
    effects: Mapping[str, Effect],
    permissions: Permissions,
    deps: Any = MISSING,
    lease_ttl_s: float,
    worker: str,
) -> EffectRecord:
    """O2, O3, O3r, O4 rows under a new lease epoch (a LeasedWriter, purpose settle) with
    revision compare-and-swap; raises SettlementConflict."""
    if not isinstance(answer, Settle):
        raise TypeError("settle requires a Settle answer.")
    # Settle is frozen, but its receipt may be a mutable model or mapping.
    # Neither the caller nor an authorization callback may change the answer
    # while the operator is awaiting a lease or a policy decision.
    answer = copy.deepcopy(answer)
    async with _take_lease(session, ttl_s=lease_ttl_s, worker=worker, purpose="settle") as writer:
        op = _resolve(writer, answer.record_id)
        _revision(op, answer)
        # C at entry is a conflict, even for a compatible receipt
        _eligible(op, answer, _answer_state(permissions, session.id, op, answer))
        op = await _findings(writer, op)
        _revision(op, answer)
        effect = _declaration(effects, op)
        authorize = permissions.authorize_settlement
        if authorize is not None:
            allowed = authorize(copy.deepcopy(answer), copy.deepcopy(_record(writer, op)))
            if inspect.isawaitable(allowed):
                allowed = await allowed
            if allowed is not True:
                raise SettlementConflict(reason="refused")
        receipt = None
        if answer.outcome == "committed":
            receipt, error = _engine.receipt_json(effect, answer.receipt)
            if error is not None:
                raise SettlementConflict(f"Invalid settlement receipt: {error}", reason="refused")
        # The new acquisition is real journal evidence for Q2, even without a lookup.
        await writer.append_effect([])
        verdict = None
        before_lookup = op.revision
        if effect.lookup_fn is not None:
            verdict = await _lookup(writer, effect, op, deps, "settle")
            op = writer.view().get(op.namespace, op.name, op.key)
            op = await _findings(writer, op)
            if op.state == "C":
                if (
                    answer.outcome == "committed"
                    and op.identity_digest
                    == _canonical.identity_digest(receipt, effect.receipt_identity)
                ):
                    return _record(writer, op)
                raise SettlementConflict(reason="contradicted")
            if op.revision != before_lookup:
                # Fresh contradictory evidence must be reviewed, not overwritten by
                # the answer the operator prepared before this lookup.
                raise SettlementConflict(reason="contradicted")
            if isinstance(verdict, Found) and answer.outcome == "committed":
                found_receipt, found_error = _engine.receipt_json(effect, verdict.receipt)
                if found_error is not None or _canonical.identity_digest(
                    found_receipt, effect.receipt_identity
                ) != _canonical.identity_digest(receipt, effect.receipt_identity):
                    raise SettlementConflict(reason="contradicted")
        contradicted = False
        if effect.authoritative:
            contradicted = isinstance(verdict, Found) and answer.outcome != "committed"
            if isinstance(verdict, Absent) and answer.outcome == "committed":
                q = _quiescence._all_quiescent(op, writer.view(), writer.timing)
                started = op.hold.started_at if op.hold is not None else None
                contradicted = started is not None and _quiescence._lookup_after_quiescence(
                    started, q
                )
        earliest = None
        too_early = False
        if answer.outcome == "resend":
            earliest = _quiescence.earliest_resend(
                op, writer.view(), writer.timing, permissions.resend_min_age_s
            )
            too_early = earliest is None or await writer.store_now() < earliest
        situation = _table.OperatorSituation(answer.outcome, contradicted, False, too_early)
        row = _table.decide(_table.TState(op.state), _table.TEvent.OPERATOR, situation)
        if row.action == "settlement_conflict":
            if contradicted:
                raise SettlementConflict(reason="contradicted")
            raise SettlementConflict(
                reason="too_early",
                earliest=None if earliest is None else datetime.fromtimestamp(earliest, UTC),
            )
        entries = await writer.load()
        op = writer.view().get(op.namespace, op.name, op.key)
        _revision(op, answer)
        _eligible(op, answer, _answer_state(permissions, session.id, op, answer))
        if answer.outcome == "committed":
            patch, patch_error = _engine.on_commit_patch(
                effect, _session.derive_state(entries).values, op.request, receipt
            )
            data = _kinds.EffectCommittedData(
                **_ref(op),
                attempt=None,
                intent_seq=None,
                via="operator",
                receipt=receipt,
                identity_digest=_canonical.identity_digest(receipt, effect.receipt_identity),
                state_patch=patch,
                on_commit_error=patch_error,
                by=answer.by,
                evidence=answer.evidence,
                expected_revision=op.revision,
            ).model_dump(mode="json")
            kind = _kinds.KIND_EFFECT_COMMITTED
        else:
            data = _kinds.EffectSettledData(
                **_ref(op),
                outcome=answer.outcome,
                by=answer.by,
                evidence=answer.evidence,
                reason=answer.reason,
                expected_revision=op.revision,
                covers_attempts=[attempt.number for attempt in op.attempts],
            ).model_dump(mode="json")
            kind = _kinds.KIND_EFFECT_SETTLED
        # With approval_secret, runs verify a pending resend against these same fields
        # before any row sends, and settle before it accepts another resend (SPEC 5.3).
        data["mac"] = permissions.sign_settlement(
            _kinds.settlement_mac_fields(session.id, kind, data)
        )
        await writer.append_effect([_session.Entry(kind=kind, data=data)])
        op = writer.view().get(op.namespace, op.name, op.key)
        _reach(writer, op, row.rule, "operator_commit" if answer.outcome == "committed" else None)
        return _record(writer, op)


async def recheck(
    session: Session,
    *,
    effects: Mapping[str, Effect],
    permissions: Permissions,
    deps: Any = MISSING,
    lease_ttl_s: float,
    worker: str,
) -> list[HeldEffect]:
    """O1 row: lookups only for D, X, NA, CU records under a new epoch (purpose recheck);
    nothing is sent; returns what is still held."""
    async with _take_lease(session, ttl_s=lease_ttl_s, worker=worker, purpose="recheck") as writer:
        keys = [
            key
            for key, op in writer.view().operations.items()
            if op.state in {"D", "X", "NA", "CU"}
        ]
        for key in keys:
            op = writer.view().operations[key]
            effect = _declaration(effects, op)
            op = await _findings(writer, op)
            if effect.lookup_fn is None:
                continue
            await writer.append_effect([])
            await _lookup(writer, effect, op, deps, "recheck")
            op = writer.view().operations[key]
            await _findings(writer, op)
            _reach(writer, op, "O1", None)
        return writer.view().held(session.id)
