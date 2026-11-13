"""Public effect declarations and the errors tool authors may handle."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from qagents.effects import _types
from qagents.effects._declare import REACH_NAMES, Dedupe, Key, Unique, effect
from qagents.effects._types import Absent, Attempt, EffectContext, Found, current_attempt
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


class RequestModel(BaseModel):
    business_id: str
    values: list[int] = []


class ReceiptModel(BaseModel):
    receipt_id: str
    status: str = "pending"


async def async_send(ctx: EffectContext, attempt: Attempt[RequestModel]) -> ReceiptModel:
    return ReceiptModel(receipt_id=attempt.request.business_id)


def sync_send(ctx: EffectContext, attempt: Attempt[RequestModel]) -> ReceiptModel:
    return ReceiptModel(receipt_id=attempt.request.business_id)


def declare(**kwargs):
    return effect("example.refund", **{"key": Key("refund", "business_id"), **kwargs})(async_send)


def test_declare_infers_request_receipt_and_exposes_raw_send():
    declared = declare(receipt_identity=("receipt_id",))
    assert declared.request_type is RequestModel
    assert declared.receipt_type is ReceiptModel
    assert declared.send is async_send
    assert declared.is_async is True
    assert declared.klass == "opaque"
    assert declared.lookup_fn is declared.on_commit_fn is None
    assert declared.key_spec_digest == Key("refund", "business_id").spec_digest()
    assert declared.request_schema_digest == declare().request_schema_digest
    with pytest.raises(TypeError, match="await ctx.perform.*example.refund"):
        declared(None, None)


def test_lookup_decorators_and_contract_classes():
    declared = declare()

    @declared.lookup
    def lookup(ctx, inquiry):
        return Absent()

    assert declared.lookup_fn is lookup
    assert declared.klass == "reconcilable"
    bounded = declare(applies_within_s=1)

    @bounded.lookup(authoritative=True)
    async def authoritative(ctx, inquiry):
        return Absent()

    assert bounded.lookup_fn is authoritative and bounded.authoritative
    assert bounded.klass == "authoritative"
    assert declare(dedupe=Dedupe(3600, "example")).klass == "deduplicated"
    unique = declare(dedupe=Unique("constraint", (ValueError,)))
    assert unique.klass == "unique" and unique.lookup_fn is None


def test_authoritative_lookup_needs_async_send_and_application_bound():
    with pytest.raises(ConfigurationError, match="applies_within_s.*async send"):
        declare().lookup(authoritative=True)
    synchronous = effect("sync", key=Key("refund", "business_id"), applies_within_s=1)(sync_send)
    assert synchronous.is_async is False
    with pytest.raises(ConfigurationError, match="async send"):
        synchronous.lookup(authoritative=True)
    assert synchronous.lookup_fn is None and not synchronous.authoritative


def test_commit_callback_and_coverage_point_names():
    declared = declare()

    @declared.on_commit
    def patch(state, request, receipt):
        state["receipt"] = receipt.receipt_id

    assert declared.on_commit_fn is patch
    for name in REACH_NAMES:
        point = getattr(declared.reached, name)
        assert (point.effect, point.name) == (declared.name, name)
    with pytest.raises(AttributeError):
        _ = declared.reached.invented

    async def invalid_patch(state, request, receipt):
        pass

    with pytest.raises(ConfigurationError, match="synchronous"):
        declared.on_commit(invalid_patch)


@pytest.mark.parametrize(
    "invalid",
    [
        {"key": "business_id"},
        {"key": Key("refund", "missing")},
        {"key": Key("refund", "values")},
        {"on_unknown": "tell_model", "key": Key.per_call()},
        {"destination": "secret"},
        {"dedupe": True},
        {"timeout_s": 0},
        {"applies_within_s": -1},
        {"on_unknown": "ignore"},
        {"approval": "effect"},
        {"version": ""},
        {"namespace": ""},
        {"idempotency_key_max_len": 23},
        {"receipt_identity": ()},
        {"receipt_identity": ("missing",)},
        {"receipt_identity": ("receipt_id", "receipt_id")},
    ],
)
def test_invalid_effect_declarations_fail_before_a_run(invalid):
    with pytest.raises(ConfigurationError):
        declare(**invalid)


@pytest.mark.parametrize("kind", [Exception, BaseException, ToolError, TimeoutError, "error"])
def test_rejected_on_refuses_ambiguous_or_broad_exceptions(kind):
    with pytest.raises(ConfigurationError, match="rejected_on"):
        declare(rejected_on=(kind,))


def test_rejected_on_refuses_cancellation_but_accepts_explicit_refusal():
    import asyncio

    with pytest.raises(ConfigurationError, match="rejected_on"):
        declare(rejected_on=(asyncio.CancelledError,))
    assert declare(rejected_on=(ValueError, EffectRejected)).rejected_on == (
        ValueError,
        EffectRejected,
    )


@pytest.mark.parametrize(
    "annotations",
    [
        {},
        {"ctx": dict, "attempt": Attempt[RequestModel], "return": ReceiptModel},
        {"ctx": EffectContext, "attempt": RequestModel, "return": ReceiptModel},
        {"ctx": EffectContext, "attempt": Attempt[RequestModel], "return": str},
    ],
)
def test_invalid_send_annotations_are_configuration_errors(annotations):
    def send(ctx, attempt):
        return None

    send.__annotations__ = annotations
    with pytest.raises(ConfigurationError):
        effect("bad", key=Key("refund", "business_id"))(send)


def test_local_forward_annotations_resolve_at_declaration():
    class LocalRequest(BaseModel):
        key: str

    def send(ctx, attempt):
        return ReceiptModel(receipt_id="local")

    send.__annotations__ = {
        "ctx": "EffectContext",
        "attempt": "Attempt[LocalRequest]",
        "return": "ReceiptModel",
    }
    declared = effect("local", key=Key("local", "key"))(send)
    assert declared.request_type is LocalRequest
