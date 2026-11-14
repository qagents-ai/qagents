"""Typed, forward-compatible Effects journal data and capability markers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict
from pydantic import JsonValue as JSONValue

from ._canonical import digest
from ._types import HoldReason


class _Data(BaseModel):
    model_config = ConfigDict(extra="allow")


# kind names
KIND_LEASE_ACQUIRED = "lease_acquired"
KIND_EFFECT_INTENT = "effect_intent"
KIND_EFFECT_COMMITTED = "effect_committed"
KIND_EFFECT_FAILED = "effect_failed"
KIND_EFFECT_SEND_ENDED = "effect_send_ended"
KIND_EFFECT_OBSERVED = "effect_observed"
KIND_EFFECT_SETTLED = "effect_settled"
KIND_EFFECT_CONFLICT = "effect_conflict"
KIND_EFFECT_CLAIMED = "effect_claimed"  # [additive] a claim moved to this session (7.10)
KIND_MEMO_RECORDED = "memo_recorded"
KIND_RUN_RESUMED = "run_resumed"
KIND_WORKFLOW_STEP_STARTED = "workflow_step_started"

EFFECT_KINDS = frozenset(
    {
        KIND_EFFECT_INTENT,
        KIND_EFFECT_COMMITTED,
        KIND_EFFECT_FAILED,
        KIND_EFFECT_SEND_ENDED,
        KIND_EFFECT_OBSERVED,
        KIND_EFFECT_SETTLED,
        KIND_EFFECT_CONFLICT,
        KIND_EFFECT_CLAIMED,
    }
)
STATUS_CHANGING_KINDS = frozenset(
    {
        KIND_EFFECT_INTENT,
        KIND_EFFECT_COMMITTED,
        KIND_EFFECT_FAILED,
        KIND_EFFECT_SETTLED,
        KIND_EFFECT_CONFLICT,
        KIND_EFFECT_CLAIMED,
    }
)
LIFECYCLE_KINDS = frozenset(
    {
        "run_started",
        "run_finished",
        "run_paused",
        "run_failed",
        KIND_RUN_RESUMED,
        "run_recovered",
        "message",
        "tool_started",
        "tool_result",
        "tool_deferred",
        "tool_checkpoint",
        "approval_decided",
        "state_updated",
        KIND_MEMO_RECORDED,
        KIND_LEASE_ACQUIRED,
    }
)
# Refuse lifecycle kinds through unleased append once SessionStatus.has_effects.
# Effect kinds are always refused there (SPEC 7.7.7).
REQUIRES_EFFECTS = "effects/1"
REQUIRES_APPROVALS = "approvals/3"
SUPPORTED_REQUIRES = frozenset({REQUIRES_EFFECTS, REQUIRES_APPROVALS})
PROTOCOL = 1
DIGEST_ALG = "jcs-sha256/1"


class EffectRefData(_Data):
    namespace: str
    name: str
    key: str
    record_id: str  # [additive] every effect entry places itself


class LeaseAcquiredData(_Data):
    epoch: int
    acquired_at: float
    worker: str
    purpose: Literal["run", "recovery", "settle", "recheck", "workflow"]


class AuthorizationData(_Data):
    decision: Literal["allow", "ask", "none", "resend"]
    rule: str | None = None
    approval_id: str | None = None
    settlement_seq: int | None = None


class IntentTiming(
    _Data
):  # [additive] the declaration's bounds at intent time (fold, Q, CLI ledger)
    timeout_s: float
    applies_within_s: float | None
    m_s: float
    sigma_s: float
    window_s: float | None
    margin_s: float | None
    unique: bool
    concurrent: Literal["rejects", "waits", "unknown"] = "unknown"


class EffectIntentData(EffectRefData):
    protocol: Literal[1] = 1
    key_spec_digest: str
    version: str
    operation_id: str
    tool_call_id: str | None
    occurrence: int | None
    attempt: int
    generation: int
    idempotency_key: str
    operation_tag: str
    digest_alg: Literal["jcs-sha256/1"] = "jcs-sha256/1"
    request_digest: str
    destination_digest: str
    request: dict[str, JSONValue]  # the frozen request, at most 16 KiB
    destination: JSONValue | None
    epoch: int | None
    store_ts: float
    first_use_at: float
    not_after: float | None
    may_overrun: bool
    authorization: AuthorizationData
    rule: Literal["D1", "D2", "D3", "D4", "D5"]  # [additive] coverage and display
    timing: IntentTiming  # [additive]
    receipt_identity: list[str] | None  # [additive] display; the fold compares identity_digest


class StatePatchData(_Data):
    set: dict[str, JSONValue] = {}
    unset: list[str] = []


class EffectCommittedData(EffectRefData):
    attempt: int | None
    intent_seq: int | None  # None for via=claim and some operator commits
    via: Literal["direct", "dedupe", "lookup", "claim", "resend", "operator"]
    receipt: dict[str, JSONValue] | None
    receipt_error: str | None = None
    identity_digest: str | None = None  # [additive] 3.1 identity_digest; None with receipt_error
    state_patch: StatePatchData | None = None
    on_commit_error: str | None = None
    send_ended: bool | None = None
    ended_at: float | None = None  # [additive] store time read after send ended (Q1)
    found_tag: str | None = None
    found_seen: dict[str, JSONValue] | None = None
    claim_owner: dict[str, str] | None = None  # via=claim: {session_id, record_id}
    by: str | None = None
    evidence: str | None = None  # via=operator
    expected_revision: int | None = None
    mac: str | None = None


class EffectFailedData(EffectRefData):
    attempt: int
    intent_seq: int
    committed: Literal[False] | None  # False: proven or not sent; None: unknown
    why: Literal["rejected", "not_sent", "unknown", "interrupted"]
    error: str | None = None
    error_kind: Literal["raised", "timeout", "interrupted", "duplicate_key"] | None = None
    send_ended: bool
    overran: bool = False
    ended_at: float | None = None  # [additive] with send_ended=True (Q1)


class EffectSendEndedData(EffectRefData):
    attempt: int
    intent_seq: int
    overran: bool
    outcome: Literal["returned", "raised"] | None = None
    ended_at: float  # [additive] (Q1)


class EffectObservedData(EffectRefData):
    attempts: list[int]
    verdict: (
        Literal["found", "not_yet_visible", "absent", "unknown", "error"] | None
    )  # None: a hold without a lookup
    authoritative: bool = False
    final: bool = False
    started_at: float | None = None  # store clock, when the lookup started
    reason: HoldReason | None = None
    detail: str | None = None
    retry_after: float | None = None


class EffectSettledData(EffectRefData):
    outcome: Literal["not_applied", "resend", "close_unknown"]
    by: str
    evidence: str | None = None
    reason: str | None = None
    expected_revision: int
    mac: str | None = None
    covers_attempts: list[int]  # O3r: the attempts that existed when accepted


class EffectConflictData(EffectRefData):
    what: Literal[
        "receipts_differ",
        "descriptor",
        "orphan_terminal",
        "late_failure_after_commit",
        "found_mismatch",
        "late_found_after_settlement",
        "closed_by_incompatible_worker",
    ]
    seqs: list[int]
    detail: str | None = None


class ClaimAttemptData(_Data):  # [additive] one attempt of a claim's history, by any owner
    session_id: str
    number: int
    generation: int
    idempotency_key: str
    epoch: int | None
    store_ts: float
    outcome: Literal["committed", "rejected", "unknown", "not_sent"] | None = None


class EffectClaimedData(EffectRefData):
    """[additive] A rejected claim moves to this record before any lookup or send (SPEC 7.10).

    It carries what the lookup must ask the destination about: the frozen request and its
    operation tag, and every earlier attempt with its idempotency key. The claim index
    accepts it only while the claim still matches ``previous_owner``, ``status``,
    ``generation`` and ``attempts`` exactly. ``status`` is ``"not_sent"`` when a record
    in R(n) takes back a claim whose last owner's attempt was never sent.
    """

    version: str
    request_digest: str
    destination_digest: str
    request: dict[str, JSONValue]
    operation_tag: str
    receipt_identity: list[str] | None
    operation_id: str  # the call that took the claim; it is not one of the record's own attempts
    tool_call_id: str | None
    previous_owner: dict[str, str]  # {session_id, incarnation, record_id}
    status: Literal["rejected", "not_sent"]  # the claim's status when it was read
    generation: int
    first_use_at: float
    attempts: list[ClaimAttemptData]
    store_ts: float


class MemoRecordedData(_Data):
    operation_id: str
    name: str
    value: JSONValue


class RunResumedData(_Data):
    epoch: int | None
    worker: str
    requires: list[str]
    agent_config_digest: str
    build_id: str | None = None


class WorkflowStepStartedData(_Data):
    name: str
    input_digest: str | None = None


# additions to existing kinds (fields with defaults, so 0.2 entries still parse)
class RunStartedV2(_Data):
    requires: list[str] = []
    agent_config_digest: str | None = None
    build_id: str | None = None
    epoch: int | None = None
    kept_held: list[str] = []  # record ids kept under if_held_effects="continue"


class ToolStartedV2(_Data):
    operation_id: str | None = None
    attempt: int = 1
    arguments_digest: str | None = None
    authorization: AuthorizationData | None = None
    arguments_changed_by: str | None = None


class ToolResultV2(_Data):  # [additive] provisional state is matched by operation, not call id
    operation_id: str | None = None


class ToolDeferredV2(
    _Data
):  # [additive] a hold is journaled by record id; PendingInput.held is folded on read
    held_record_id: str | None = None


class StateUpdatedV2(_Data):
    provisional: bool = False
    operation_id: str | None = None


class RunFailedV2(_Data):
    retry_after: float | None = None


class WorkflowAgentStartedV2(_Data):
    run_id: str | None = None


class ApprovalRecord(
    _Data
):  # the approval_decided data, v3; one model, re-exported by qagents.permissions
    v: Literal[3] = 3
    approval_id: str
    session_id: str
    run_id: str
    tool_call_id: str
    operation_id: str  # [additive] durable identity of the call
    name: str
    approved: bool
    reason: str | None
    arguments_digest: str
    tool_spec_digest: str
    by: str | None
    decided_at: float
    expires_at: float
    mac: str | None
    ask: int = 1  # [additive] 1 + earlier decisions of this operation (re-ask, 3.9)
    arguments_changed: bool = False
    arguments: dict[str, Any] | None = None
    arguments_changed_by: str | None = None


APPROVAL_MAC_FIELDS = (
    "v",
    "approval_id",
    "session_id",
    "run_id",
    "tool_call_id",
    "approved",
    "arguments_digest",
    "tool_spec_digest",
    "by",
    "decided_at",
    "expires_at",
)  # SPEC 7.5


def settlement_mac_fields(session_id: str, kind: str, data: dict[str, Any]) -> dict[str, Any]:
    """What an operator answer's MAC signs (SPEC 5.3): the session id, the entry kind
    (``effect_settled``, or ``effect_committed`` for ``via=operator``) and every journaled
    data field except ``mac``. The operator signs these fields. With approval_secret, a
    pending ``resend`` is verified against them before a run sends, and every journaled
    ``resend`` before the operator path accepts another one; the other answers authorize
    no send and are not checked when they are used."""
    return {
        "session_id": session_id,
        "kind": kind,
        "data": {name: value for name, value in data.items() if name != "mac"},
    }


def lease_entry_id(owner_token: str) -> str:
    """A re-appended acquisition uses the same client-generated entry identity."""
    return "la_" + digest(owner_token)[:24]


def conflict_entry_id(record_id: str, what: str, seqs: list[int]) -> str:
    """The same contradiction is journaled once by independent writers."""
    return "cf_" + digest([record_id, what, seqs])[:24]
