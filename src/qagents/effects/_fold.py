"""Incremental, journal-only recovery state for every effect operation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from . import _kinds as kinds
from ._canonical import Descriptor, JSONValue
from ._types import (
    HOLD_DETAIL,
    AttemptSummary,
    EffectRecord,
    HeldEffect,
    HoldReason,
    SettlementSummary,
)

if TYPE_CHECKING:
    from qagents.session import Entry

State = Literal["N", "C", "R", "D", "NA", "CU", "X"]

# The keys the fold adds to each folded effect_settled data (Operation.settlements);
# without them the data is what the operator journaled and signed.
SETTLEMENT_FOLD_KEYS = frozenset({"seq", "voided", "prior_revision"})


@dataclass
class AttemptState:
    number: int
    generation: int
    intent_seq: int
    intent: kinds.EffectIntentData
    terminal: Literal["committed", "rejected", "unknown", "not_sent"] | None
    terminal_seq: int | None
    error_kind: str | None
    raised: bool
    send_ended_at: float | None
    overran: bool


@dataclass
class Operation:
    record_id: str
    namespace: str
    name: str
    key: str
    version: str
    state: State
    generation: int
    attempts: list[AttemptState]
    descriptor: Descriptor | None
    request: dict[str, JSONValue] | None
    first_use_at: float | None
    receipt: dict[str, JSONValue] | None
    identity_digest: str | None
    via: str | None
    commit_seq: int | None
    hold: kinds.EffectObservedData | None
    settlements: list[kinds.EffectSettledData]
    pending_resend: kinds.EffectSettledData | None
    conflicts: list[kinds.EffectConflictData]
    findings: list[kinds.EffectConflictData]
    operation_ids: list[str]
    tool_call_ids: list[str]
    revision: int
    resolved_by_rejected_resend: bool
    # [additive] the journaled move of a rejected claim to this record, with the earlier
    # attempts of other owners (SPEC 7.10); its seq is in model_extra["seq"].
    claimed: kinds.EffectClaimedData | None = None
    _resolved_conflicts: set[tuple[str, tuple[int, ...]]] = field(default_factory=set, repr=False)
    _resolution_seq: int = field(default=0, repr=False)
    _operator_settlement: SettlementSummary | None = field(default=None, repr=False)


def _new_operation(ref: kinds.EffectRefData) -> Operation:
    return Operation(
        record_id=ref.record_id,
        namespace=ref.namespace,
        name=ref.name,
        key=ref.key,
        version="",
        state="N",
        generation=0,
        attempts=[],
        descriptor=None,
        request=None,
        first_use_at=None,
        receipt=None,
        identity_digest=None,
        via=None,
        commit_seq=None,
        hold=None,
        settlements=[],
        pending_resend=None,
        conflicts=[],
        findings=[],
        operation_ids=[],
        tool_call_ids=[],
        revision=0,
        resolved_by_rejected_resend=False,
    )


def _ref(op: Operation) -> dict:
    return {"record_id": op.record_id, "namespace": op.namespace, "name": op.name, "key": op.key}


def _in_scope(entry: Entry, finished_runs: set[str]) -> bool:
    """Effect evidence outlives the run that produced it, including a late commit (M10)."""
    return entry.kind in kinds.EFFECT_KINDS or entry.run_id not in finished_runs


def _settled_state(outcome: str) -> State:
    """A not-applied answer never authorizes a resend (M25)."""
    return {"not_applied": "NA", "close_unknown": "CU", "resend": "D"}[outcome]


def _status_of(op: Operation) -> State:
    """Consider every attempt in the generation, while keeping a recorded commit (M32)."""
    if any(
        _conflict_key(item) not in op._resolved_conflicts for item in [*op.findings, *op.conflicts]
    ):
        return "X"
    if op.commit_seq is not None:
        return "C"
    for settlement in reversed(op.settlements):
        if not (settlement.model_extra or {}).get("voided", False):
            if settlement.outcome != "resend":
                return _settled_state(settlement.outcome)
            break
    if op.pending_resend is not None:
        return "D"
    attempts = [attempt for attempt in op.attempts if attempt.generation == op.generation]
    if op.resolved_by_rejected_resend:
        return "R"
    if any(attempt.terminal not in ("committed", "rejected", "not_sent") for attempt in attempts):
        return "D"
    if any(attempt.terminal == "rejected" for attempt in attempts) or _claimed_rejection(op):
        return "R"
    return "N"


def _claimed_rejection(op: Operation) -> bool:
    """A claim taken over as rejected is the rejection of its generation (SPEC 7.10)."""
    return op.claimed is not None and op.claimed.generation == op.generation


def _conflict_key(finding: kinds.EffectConflictData) -> tuple[str, tuple[int, ...]]:
    return finding.what, tuple(finding.seqs)


def _resolve_prefix(
    op: Operation, seq: int, expected_revision: int | None, by: str | None, evidence: str | None
) -> bool:
    # The fold has no secret and checks no MAC: it trusts operator answers as it trusts
    # every effect entry. With approval_secret set, the engine checks a pending resend's
    # MAC and position before any row sends (SPEC 5.3); the other answers authorize no
    # send. Here an answer resolves an X prefix only if it is complete and names its revision.
    if (
        op.state not in ("D", "X")
        or expected_revision != op.revision
        or not by
        or not by.strip()
        or not evidence
        or not evidence.strip()
    ):
        return False
    op._resolved_conflicts.update(_conflict_key(item) for item in [*op.findings, *op.conflicts])
    op._resolution_seq = seq
    op.pending_resend = None
    return True


def _finding(op: Operation, what: str, seqs: Iterable[int], detail: str | None = None) -> None:
    finding = kinds.EffectConflictData(**_ref(op), what=what, seqs=sorted(set(seqs)), detail=detail)
    known = {_conflict_key(item) for item in [*op.findings, *op.conflicts]}
    if _conflict_key(finding) not in known:
        op.findings.append(finding)


def _first_identity_seq(op: Operation, default: int) -> int:
    """The entry that first fixed the record's descriptor: an intent or a claim transfer."""
    seqs = [op.attempts[0].intent_seq] if op.attempts else []
    if op.claimed is not None:
        seqs.append((op.claimed.model_extra or {}).get("seq", default))
    return min(seqs, default=default)


def _terminal_attempt(
    op: Operation, attempt: int | None, intent_seq: int | None, seq: int
) -> AttemptState | None:
    found = next(
        (item for item in op.attempts if item.number == attempt and item.intent_seq == intent_seq),
        None,
    )
    if found is None:
        _finding(op, "orphan_terminal", [seq], "The terminal entry names no matching intent.")
    return found


def _end(attempt: AttemptState, ended_at: float | None, overran: bool = False) -> None:
    # Entry.ts is a client timestamp and never establishes the send's end.
    if ended_at is not None:
        attempt.send_ended_at = (
            ended_at if attempt.send_ended_at is None else min(attempt.send_ended_at, ended_at)
        )
    attempt.overran = attempt.overran or overran


def _quiescent_at(
    attempt: AttemptState, acquisitions: list[tuple[int, float, int]]
) -> float | None:
    """Read the recorded evidence used to explain a rejected resend, not permission to send."""
    if attempt.terminal == "not_sent":
        return attempt.intent.store_ts
    timing = attempt.intent.timing
    if timing.applies_within_s is None:
        return None
    ended = attempt.send_ended_at
    if ended is None:
        if attempt.intent.may_overrun or attempt.overran or attempt.intent.epoch is None:
            return None
        later = [(seq, at) for epoch, at, seq in acquisitions if epoch > attempt.intent.epoch]
        if not later:
            return None
        ended = min(later)[1]
    return ended + timing.applies_within_s + timing.m_s + timing.sigma_s


def _rejected_resend_resolves(op: Operation, acquisitions: list[tuple[int, float, int]]) -> bool:
    attempts = [item for item in op.attempts if item.generation == op.generation]
    for index, last in enumerate(attempts):
        if last.terminal != "rejected" or last.intent.rule not in ("D3", "D4", "D5"):
            continue
        previous = attempts[:index]
        # A later unresolved attempt still holds the operation.
        if any(item.terminal not in ("rejected", "not_sent") for item in attempts[index + 1 :]):
            continue
        if not previous or any(
            item.intent.idempotency_key != last.intent.idempotency_key for item in previous
        ):
            continue
        if last.intent.timing.concurrent == "waits":
            return True
        bounds = [_quiescent_at(item, acquisitions) for item in previous]
        if all(bound is not None and bound <= last.intent.store_ts for bound in bounds):
            return True
    return False


@dataclass
class EffectView:
    operations: dict[tuple[str, str, str], Operation] = field(default_factory=dict)
    by_call: dict[tuple[str, str, int | None], tuple[str, str, str]] = field(default_factory=dict)
    operations_with_intent: set[str] = field(default_factory=set)
    acquisitions: list[tuple[int, float, int]] = field(default_factory=list)
    incompatible_closures: list[tuple[tuple[str, str, str], str, str]] = field(default_factory=list)
    _seen_ids: set[str] = field(default_factory=set, repr=False)
    _finished_runs: set[str] = field(default_factory=set, repr=False)
    _run_requires: dict[str, frozenset[str]] = field(default_factory=dict, repr=False)
    _operation_runs: dict[str, str | None] = field(default_factory=dict, repr=False)
    _call_operations: dict[tuple[str | None, str], str] = field(default_factory=dict, repr=False)
    _closed_operations: set[str] = field(default_factory=set, repr=False)

    def apply(self, entry: Entry) -> None:
        """Fold one entry, preserving unknown kinds and ignoring a duplicate entry id."""
        identity = getattr(entry, "id", None)
        if identity is not None and identity in self._seen_ids:
            return
        if identity is not None:
            self._seen_ids.add(identity)
        if not _in_scope(entry, self._finished_runs):
            return
        if entry.kind in ("run_started", kinds.KIND_RUN_RESUMED):
            if entry.run_id is not None:
                self._run_requires[entry.run_id] = frozenset(entry.data.get("requires", ()))
            return
        if entry.kind == "run_finished":
            if entry.run_id is not None:
                self._finished_runs.add(entry.run_id)
            return
        if entry.kind == kinds.KIND_LEASE_ACQUIRED:
            acquired = kinds.LeaseAcquiredData.model_validate(entry.data)
            self.acquisitions.append((acquired.epoch, acquired.acquired_at, entry.seq))
            for op in self.operations.values():
                self._refresh(op)
            return
        if entry.kind == "tool_started":
            operation_id = entry.data.get("operation_id")
            call_id = entry.data.get("tool_call_id")
            if operation_id and call_id:
                self._call_operations[(entry.run_id, call_id)] = operation_id
            return
        if entry.kind == "tool_result":
            self._close_call(entry)
            return
        if entry.kind not in kinds.EFFECT_KINDS:
            return
        ref = kinds.EffectRefData.model_validate(entry.data)
        key = (ref.namespace, ref.name, ref.key)
        op = self.operations.setdefault(key, _new_operation(ref))
        previous_revision = op.revision
        if entry.kind == kinds.KIND_EFFECT_INTENT:
            self._intent(op, entry)
        elif entry.kind == kinds.KIND_EFFECT_CLAIMED:
            self._claimed(op, entry)
        elif entry.kind == kinds.KIND_EFFECT_COMMITTED:
            self._commit(op, entry)
        elif entry.kind == kinds.KIND_EFFECT_FAILED:
            self._failure(op, entry)
        elif entry.kind == kinds.KIND_EFFECT_SEND_ENDED:
            data = kinds.EffectSendEndedData.model_validate(entry.data)
            target = _terminal_attempt(op, data.attempt, data.intent_seq, entry.seq)
            if target is not None:
                _end(target, data.ended_at, data.overran)
        elif entry.kind == kinds.KIND_EFFECT_OBSERVED:
            # Observations can describe a hold but never change status or revision.
            op.hold = kinds.EffectObservedData.model_validate(entry.data)
            return
        elif entry.kind == kinds.KIND_EFFECT_SETTLED:
            data = kinds.EffectSettledData.model_validate(entry.data)
            # prior_revision: the record's revision just before this answer. A genuine
            # answer names it as expected_revision; a copy appended later names an older one.
            data = data.model_copy(
                update={"seq": entry.seq, "voided": False, "prior_revision": previous_revision}
            )
            if data.outcome != "resend" and _resolve_prefix(
                op,
                entry.seq,
                data.expected_revision,
                data.by,
                data.reason if data.outcome == "close_unknown" else data.evidence,
            ):
                # The answer resolves an X prefix that can contain contradictory commits.
                # Their evidence remains in the journal and conflict history.
                op.receipt = op.identity_digest = op.via = op.commit_seq = None
            op.settlements.append(data)
            op.pending_resend = data if data.outcome == "resend" else None
        elif entry.kind == kinds.KIND_EFFECT_CONFLICT:
            data = kinds.EffectConflictData.model_validate(entry.data)
            if _conflict_key(data) not in {_conflict_key(item) for item in op.conflicts}:
                op.conflicts.append(data)
            op.findings = [
                item for item in op.findings if _conflict_key(item) != _conflict_key(data)
            ]
        if entry.kind in kinds.STATUS_CHANGING_KINDS:
            op.revision = max(previous_revision, entry.seq)
        self._refresh(op)

    def _refresh(self, op: Operation) -> None:
        op.resolved_by_rejected_resend = _rejected_resend_resolves(op, self.acquisitions)
        op.state = _status_of(op)

    def _intent(self, op: Operation, entry: Entry) -> None:
        data = kinds.EffectIntentData.model_validate(entry.data)
        value = Descriptor(data.name, data.version, data.destination_digest, data.request_digest)
        if op.descriptor is None:
            op.descriptor, op.version, op.request = value, data.version, data.request
        elif op.descriptor != value:
            first_seq = _first_identity_seq(op, entry.seq)
            _finding(op, "descriptor", [first_seq, entry.seq])
        op.generation = max(op.generation, data.generation)
        op.first_use_at = (
            data.first_use_at
            if op.first_use_at is None
            else min(op.first_use_at, data.first_use_at)
        )
        op.attempts.append(
            AttemptState(
                data.attempt, data.generation, entry.seq, data, None, None, None, False, None, False
            )
        )
        if data.operation_id not in op.operation_ids:
            op.operation_ids.append(data.operation_id)
            if data.tool_call_id is not None:
                op.tool_call_ids.append(data.tool_call_id)
        key = (data.namespace, data.name, data.key)
        self.by_call.setdefault((data.operation_id, data.name, data.occurrence), key)
        self.operations_with_intent.add(data.operation_id)
        self._operation_runs[data.operation_id] = entry.run_id
        if data.tool_call_id is not None:
            self._call_operations[(entry.run_id, data.tool_call_id)] = data.operation_id
        # O3r covers the attempts present at acceptance, not any later attempt.
        for settlement in op.settlements:
            if settlement.outcome == "resend":
                settlement.__pydantic_extra__["voided"] = True
        op.pending_resend = None

    def _claimed(self, op: Operation, entry: Entry) -> None:
        data = kinds.EffectClaimedData.model_validate(entry.data)
        value = Descriptor(data.name, data.version, data.destination_digest, data.request_digest)
        if op.descriptor is None:
            op.descriptor, op.version, op.request = value, data.version, data.request
        elif op.descriptor != value:
            first_seq = _first_identity_seq(op, entry.seq)
            _finding(op, "descriptor", [first_seq, entry.seq])
        op.generation = max(op.generation, data.generation)
        op.first_use_at = (
            data.first_use_at
            if op.first_use_at is None
            else min(op.first_use_at, data.first_use_at)
        )
        # A later transfer carries the longer history (claims only append attempts).
        op.claimed = data.model_copy(update={"seq": entry.seq})

    def _commit(self, op: Operation, entry: Entry) -> None:
        data = kinds.EffectCommittedData.model_validate(entry.data)
        resolved = (
            data.via == "operator"
            and data.receipt is not None
            and _resolve_prefix(op, entry.seq, data.expected_revision, data.by, data.evidence)
        )
        target = None
        if data.attempt is not None or data.intent_seq is not None:
            target = _terminal_attempt(op, data.attempt, data.intent_seq, entry.seq)
        elif data.via not in ("claim", "operator") and not (
            # A lookup after a claim transfer may find an earlier owner's attempt.
            data.via == "lookup" and op.claimed is not None
        ):
            _finding(op, "orphan_terminal", [entry.seq])
        if target is not None:
            target.terminal, target.terminal_seq = "committed", entry.seq
            if data.send_ended:
                _end(target, data.ended_at)
        if resolved:
            op.receipt, op.identity_digest = data.receipt, data.identity_digest
            op.commit_seq, op.via = entry.seq, data.via
            op._operator_settlement = SettlementSummary(
                "committed", data.by, data.evidence, None, entry.seq, False
            )
            return
        for settlement in op.settlements:
            if settlement.outcome == "resend":
                if data.attempt in settlement.covers_attempts:
                    settlement.__pydantic_extra__["voided"] = True
                    if settlement is op.pending_resend:
                        op.pending_resend = None
            elif (settlement.model_extra or {})["seq"] >= op._resolution_seq and (
                data.via in ("lookup", "operator") or op.commit_seq is None
            ):
                _finding(
                    op,
                    "late_found_after_settlement",
                    [(settlement.model_extra or {})["seq"], entry.seq],
                )
        different = (
            op.identity_digest is not None
            and data.identity_digest is not None
            and op.identity_digest != data.identity_digest
        )
        if different:
            _finding(op, "receipts_differ", [op.commit_seq, entry.seq])
        elif data.receipt is not None:
            op.receipt, op.identity_digest = data.receipt, data.identity_digest
        if op.commit_seq is None:
            op.commit_seq = entry.seq
            op.via = data.via

    def _failure(self, op: Operation, entry: Entry) -> None:
        data = kinds.EffectFailedData.model_validate(entry.data)
        target = _terminal_attempt(op, data.attempt, data.intent_seq, entry.seq)
        if op.commit_seq is not None:
            _finding(op, "late_failure_after_commit", [op.commit_seq, entry.seq])
        if target is not None:
            if target.terminal != "committed":
                target.terminal = (
                    "not_sent"
                    if data.committed is False and data.why == "not_sent"
                    else "rejected"
                    if data.committed is False
                    else "unknown"
                )
                target.terminal_seq = entry.seq
                target.error_kind = data.error_kind
                target.raised = data.error_kind == "raised"
            if data.send_ended:
                _end(target, data.ended_at, data.overran)
            else:
                target.overran = target.overran or data.overran

    def _close_call(self, entry: Entry) -> None:
        call_id = entry.data.get("tool_call_id")
        operation_id = entry.data.get("operation_id") or self._call_operations.get(
            (entry.run_id, call_id)
        )
        if not operation_id:
            return
        self._closed_operations.add(operation_id)
        requires = self._run_requires.get(entry.run_id, frozenset())
        if kinds.REQUIRES_EFFECTS in requires:
            return
        for key, op in self.operations.items():
            if operation_id not in op.operation_ids:
                continue
            closure = (key, operation_id, entry.run_id or "")
            if closure not in self.incompatible_closures:
                self.incompatible_closures.append(closure)
                _finding(
                    op, "closed_by_incompatible_worker", [entry.seq], entry.data.get("content")
                )
                self._refresh(op)

    def get(self, namespace: str, name: str, key: str) -> Operation | None:
        return self.operations.get((namespace, name, key))

    def by_record_id(self, record_id: str) -> Operation | None:
        matches = [op for op in self.operations.values() if op.record_id == record_id]
        if len(matches) > 1:
            raise ValueError("Record id is ambiguous; use the full namespace, effect and key.")
        return matches[0] if matches else None

    def held(self, session_id: str) -> list[HeldEffect]:
        return [record.hold for record in self.records(session_id) if record.hold is not None]

    def records(self, session_id: str) -> list[EffectRecord]:
        """Return journal-derived records for all operations except state N."""
        records = []
        statuses = {
            "C": "committed",
            "R": "rejected",
            "D": "held",
            "NA": "not_applied",
            "CU": "closed_unknown",
            "X": "conflict",
        }
        for op in self.operations.values():
            if op.state == "N":
                continue
            hold = None
            if op.state in ("D", "X"):
                observed = op.hold
                reason = observed.reason if observed is not None else None
                if op.state == "X":
                    reason = (
                        HoldReason.CLOSED_BY_INCOMPATIBLE_WORKER
                        if any(
                            item.what == "closed_by_incompatible_worker"
                            and _conflict_key(item) not in op._resolved_conflicts
                            for item in [*op.conflicts, *op.findings]
                        )
                        else HoldReason.CONFLICT
                    )
                reason = reason or HoldReason.DISPATCH_UNKNOWN
                retry_after = observed.retry_after if observed is not None else None
                detail = observed.detail if observed is not None else None
                active = [
                    item
                    for item in op.attempts
                    if item.intent.operation_id not in self._closed_operations
                ]
                hold = HeldEffect(
                    op.record_id,
                    session_id,
                    op.name,
                    op.key,
                    op.namespace,
                    reason,
                    detail or HOLD_DETAIL[reason],
                    datetime.fromtimestamp(retry_after, UTC) if retry_after is not None else None,
                    op.revision,
                    active[-1].intent.tool_call_id if active else None,
                )
            attempts = tuple(
                AttemptSummary(
                    item.number,
                    item.generation,
                    item.intent.epoch,
                    item.intent.idempotency_key,
                    item.intent.store_ts,
                    item.intent.rule,
                    item.terminal,
                    item.intent.may_overrun,
                    item.overran,
                    item.send_ended_at is not None,
                )
                for item in op.attempts
            )
            settlement = None
            if op.settlements:
                last = op.settlements[-1]
                extra = last.model_extra or {}
                settlement = SettlementSummary(
                    last.outcome,
                    last.by,
                    last.evidence,
                    last.reason,
                    extra["seq"],
                    extra.get("voided", False),
                )
            if op._operator_settlement is not None and (
                settlement is None or op._operator_settlement.seq > settlement.seq
            ):
                settlement = op._operator_settlement
            records.append(
                EffectRecord(
                    op.record_id,
                    session_id,
                    op.namespace,
                    op.name,
                    op.key,
                    statuses[op.state],
                    op.via,
                    op.receipt,
                    hold,
                    attempts,
                    settlement,
                    op.revision,
                )
            )
        return records


def fold_effects(entries: Iterable[Entry]) -> EffectView:
    """Fold all effect evidence across every run; completed runs do not reset operations."""
    view = EffectView()
    for entry in entries:
        view.apply(entry)
    return view
