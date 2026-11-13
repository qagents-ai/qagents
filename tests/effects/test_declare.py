"""Public effect declarations and the errors tool authors may handle."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from qagents.effects import _types
from qagents.effects._declare import Dedupe, Unique
from qagents.effects._types import EffectContext, Found, current_attempt
from qagents.errors import (
    ApprovalConflictError,
    ClaimConflict,
    ConfigurationError,
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


def test_dedupe_pessimistic_defaults_and_margin():
    contract = Dedupe(window_s=3600, contract="Example API v1")
    assert contract.concurrent == "unknown"
    assert contract.replays_errors is True
    assert contract.replays_rejections is False
    assert contract.binds_params is False
    assert contract.margin() == 180
    assert Dedupe(10, "short").margin() == 60
    assert Dedupe(3600, "explicit", margin_s=0).margin() == 0
    assert Dedupe(3600, "explicit", margin_s=20).margin() == 20


@pytest.mark.parametrize("window", [0, -1, True, float("nan"), float("inf"), "3600"])
def test_dedupe_refuses_invalid_retention_window(window):
    with pytest.raises(ConfigurationError, match="window_s"):
        Dedupe(window, "example")


@pytest.mark.parametrize(
    "fields",
    [
        {"contract": ""},
        {"concurrent": "allows"},
        {"margin_s": -1},
        {"margin_s": float("inf")},
        {"replays_errors": "false"},
        {"replays_rejections": 0},
        {"binds_params": 1},
    ],
)
def test_dedupe_refuses_invalid_contract_settings(fields):
    with pytest.raises(ConfigurationError):
        Dedupe(**{"window_s": 100, "contract": "example", **fields})


def test_unique_keeps_its_declared_duplicate_types():
    class Duplicate(Exception):
        pass

    contract = Unique("unique constraint on request_id", (Duplicate,))
    assert contract.on_duplicate == (Duplicate,)
    with pytest.raises(ConfigurationError):
        Unique("", (Duplicate,))
    for bad in ((), [Duplicate], (42,), (object,)):
        with pytest.raises(ConfigurationError):
            Unique("constraint", bad)


def test_found_requires_a_keyword_binding():
    receipt = object()
    assert Found(receipt, tag="op_1").tag == "op_1"
    assert Found(receipt, seen={"request_id": "R1"}).seen == {"request_id": "R1"}
    with pytest.raises(TypeError, match="tag= or seen="):
        Found(receipt)
    with pytest.raises(TypeError):
        Found(receipt, "op_1")
    for invalid in ({"tag": ""}, {"seen": {}}, {"tag": 1}, {"seen": []}):
        with pytest.raises(TypeError):
            Found(receipt, **invalid)


def test_effect_context_distinguishes_missing_and_explicit_none_dependencies():
    with pytest.raises(ConfigurationError, match="pass deps=.*--deps"):
        _ = EffectContext().deps
    assert EffectContext(None).deps is None
    events = []
    context = EffectContext({"tenant": "one"}, session_id="s", emit=events.append)
    assert context.deps == {"tenant": "one"}
    assert context.session_id == "s"
    assert context.logger.name == "qagents.effects"
    context.emit("diagnostic")
    assert events == ["diagnostic"]
    with pytest.raises(AttributeError):
        context.state = {}


async def test_current_attempt_is_task_local_and_restored():
    import asyncio

    assert current_attempt() is None
    first, second = object(), object()

    async def one(value):
        token = _types._ATTEMPT.set(value)
        try:
            await asyncio.sleep(0)
            return current_attempt()
        finally:
            _types._ATTEMPT.reset(token)

    assert await asyncio.gather(one(first), one(second)) == [first, second]
    assert current_attempt() is None
