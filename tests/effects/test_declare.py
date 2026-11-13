"""Public effect declarations and the errors tool authors may handle."""

from datetime import UTC, datetime
from types import SimpleNamespace

from qagents.errors import (
    ApprovalConflictError,
    ClaimConflict,
    DeferToolCall,
    EffectConflictError,
    EffectHeld,
    EffectReceiptError,
    EffectRejected,
    EntryConflict,
    HeldEffectsError,
    JournalOutcomeUnknown,
    SessionError,
    SettlementConflict,
    ToolError,
    WorkflowBusy,
)


def test_error_tree_preserves_actionable_attributes():
    refusal = ValueError("declined")
    rejected = EffectRejected("declined", effect="refund", key="r:1", error=refusal)
    conflict = EffectConflictError(
        "different request",
        effect="refund",
        key="r:1",
        record_id="ef_one",
        existing_status="committed",
    )
    receipt = EffectReceiptError(
        "invalid receipt", effect="refund", key="r:1", receipt_error=refusal
    )
    assert all(isinstance(error, ToolError) for error in (rejected, conflict, receipt))
    assert (rejected.effect, rejected.key, rejected.error) == ("refund", "r:1", refusal)
    assert (conflict.record_id, conflict.existing_status) == ("ef_one", "committed")
    assert receipt.receipt_error is refusal
    assert EffectRejected("declined").error is None


def test_holds_are_control_flow_not_tool_errors():
    due = datetime(2026, 11, 13, tzinfo=UTC)
    held = SimpleNamespace(detail="await confirmation", retry_after=due)
    error = EffectHeld(held)
    assert isinstance(error, DeferToolCall)
    assert not isinstance(error, ToolError)
    assert error.held is held
    assert error.reason == held.detail
    gate = HeldEffectsError([held], retry_after=due)
    assert isinstance(gate, SessionError)
    assert gate.held == (held,)
    assert gate.retry_after == due


def test_store_and_operator_errors_are_not_tool_errors():
    due = datetime(2026, 11, 13, tzinfo=UTC)
    conflict = SettlementConflict(reason="too_early", earliest=due)
    assert (conflict.reason, conflict.earliest) == ("too_early", due)
    approval = ApprovalConflictError("decision differs", tool_call_id="call-1")
    assert approval.tool_call_id == "call-1"
    claim = ClaimConflict(
        "already claimed", namespace="shop", effect="refund", key="1", owner_session="original"
    )
    assert (claim.namespace, claim.effect, claim.key, claim.owner_session) == (
        "shop",
        "refund",
        "1",
        "original",
    )
    for error in (
        conflict,
        approval,
        claim,
        EntryConflict("partial batch"),
        JournalOutcomeUnknown("unconfirmed"),
        WorkflowBusy("busy"),
    ):
        assert isinstance(error, SessionError)
        assert not isinstance(error, ToolError)
