"""Decision signatures bind durable operations, asks, arguments and tool contracts."""

import copy
import hashlib
import hmac
from types import SimpleNamespace

import pytest

from qagents import Permissions, ToolCallPart, tool
from qagents.effects._canonical import digest, jcs
from qagents.effects._kinds import APPROVAL_MAC_FIELDS, ApprovalRecord
from qagents.errors import ConfigurationError
from qagents.permissions import (
    APPROVAL_TOKEN_VERSION,
    Approval,
    approval_token,
    arguments_digest,
    canonical_json,
    tool_spec_digest,
)
from qagents.session import PendingApproval

SECRET = b"public-test-fixture-key-32-bytes!!"


@tool(requires_approval=True)
def save_note(text: str) -> str:
    """Save an approved note."""
    return text


def record(permissions=None, **overrides):
    permissions = permissions or Permissions(approval_secret=SECRET)
    values = dict(
        session_id="session",
        run_id="run",
        operation_id="run:turn:call",
        ask=1,
        tool=save_note,
        call=ToolCallPart(id="repeated-provider-id", name="save_note", arguments={"text": "one"}),
        approved=True,
        reason=None,
        by="reviewer",
        decided_at=100.0,
    )
    values.update(overrides)
    return permissions.decision_record(**values)


def test_record_uses_frozen_mac_fields_and_shared_canonical_encoding():
    permissions = Permissions(approval_secret=SECRET, approval_ttl_s=20)
    decision = record(permissions)
    assert isinstance(decision, ApprovalRecord)
    assert decision.expires_at == 120
    assert decision.approval_id == "ap1:" + digest(
        ["session", "run", "run:turn:call", "repeated-provider-id", 1]
    )
    payload = {field: getattr(decision, field) for field in APPROVAL_MAC_FIELDS}
    assert decision.mac == hmac.new(SECRET, jcs(payload), hashlib.sha256).hexdigest()
    assert permissions.verify_decision(decision)
    assert canonical_json({"b": -0.0, "a": 1e-7}) == jcs({"b": -0.0, "a": 1e-7}).decode()


@pytest.mark.parametrize(
    "field,value",
    [
        ("session_id", "other"),
        ("run_id", "other"),
        ("tool_call_id", "other"),
        ("operation_id", "run:later:call"),
        ("ask", 2),
        ("approval_id", "ap1:forged"),
        ("approved", False),
        ("arguments_digest", "wrong"),
        ("tool_spec_digest", "wrong"),
        ("by", "another"),
        ("decided_at", 101.0),
        ("expires_at", 999.0),
        ("mac", "not-a-valid-mac"),
        ("mac", "é"),
        ("mac", None),
        ("v", 2),
    ],
)
def test_changed_identity_or_signed_value_cannot_reuse_decision(field, value):
    permissions = Permissions(approval_secret=SECRET)
    decision = record(permissions).model_copy(update={field: value})
    assert not permissions.verify_decision(decision)


def test_recomputing_unsigned_identity_fields_does_not_forge_the_mac():
    permissions = Permissions(approval_secret=SECRET)
    decision = record(permissions)
    for operation_id, ask in (("run:later:call", 1), ("run:turn:call", 2)):
        forged = decision.model_copy(
            update={
                "operation_id": operation_id,
                "ask": ask,
                "approval_id": "ap1:"
                + digest(
                    [decision.session_id, decision.run_id, operation_id, decision.tool_call_id, ask]
                ),
            }
        )
        assert not permissions.verify_decision(forged)


def test_repeated_provider_call_id_gets_distinct_signed_operation_and_ask():
    first = record()
    later = record(operation_id="run:later:call")
    reask = record(ask=2)
    assert len({first.approval_id, later.approval_id, reask.approval_id}) == 3
    assert len({first.mac, later.mac, reask.mac}) == 3


def test_unsigned_decisions_never_report_verified_and_other_secret_fails():
    unsigned = Permissions()
    decision = record(unsigned)
    assert decision.mac is None and not unsigned.verify_decision(decision)
    signed = record()
    assert not unsigned.verify_decision(signed)
    assert not Permissions(approval_secret=b"different-public-fixture").verify_decision(signed)


def test_edited_arguments_are_snapshotted_and_bound_to_the_signed_digest():
    permissions = Permissions(approval_secret=SECRET)
    call = ToolCallPart(id="repeated-provider-id", name="save_note", arguments={"text": "edited"})
    decision = record(
        permissions, call=call, arguments_changed=True, arguments_changed_by="reviewer"
    )
    call.arguments["text"] = "later mutation"
    assert decision.arguments == {"text": "edited"}
    assert decision.arguments_digest == arguments_digest({"text": "edited"})
    assert permissions.verify_decision(decision)
    assert not permissions.verify_decision(
        decision.model_copy(update={"arguments": {"text": "forged"}})
    )
    assert not permissions.verify_decision(decision.model_copy(update={"arguments": None}))


def test_expiry_is_bound_but_clock_decision_belongs_to_the_runtime():
    permissions = Permissions(approval_secret=SECRET, approval_ttl_s=1)
    decision = record(permissions, decided_at=0)
    assert decision.expires_at == 1
    # No wall clock reads: replay checks expiry against the store clock at dispatch.
    assert permissions.verify_decision(decision)


def test_tool_digest_covers_schema_flags_description_and_effect_versions():
    original = tool_spec_digest(save_note)
    for field, value in [
        ("name", "different_name"),
        ("description", "Changed behavior"),
        ("parameters", {"type": "object", "properties": {}}),
        ("read_only", True),
        ("idempotent", True),
        ("requires_approval", False),
        ("deferred", True),
        ("strict", True),
        ("timeout", 4),
        ("takes_context", True),
    ]:
        changed = copy.copy(save_note)
        setattr(changed, field, value)
        assert tool_spec_digest(changed) != original, field
    changed = copy.copy(save_note)
    changed.effects = (SimpleNamespace(name="notes.save", version="1"),)
    first = tool_spec_digest(changed)
    changed.effects = (SimpleNamespace(name="notes.save", version="2"),)
    assert first != original and tool_spec_digest(changed) != first


def test_tokens_are_versioned_and_bound_to_reasks():
    common = dict(
        session_id="s",
        run_id="r",
        position=1,
        tool_call_id="c",
        name="save_note",
        arguments={"text": "one"},
        operation_id="operation",
        ask=1,
    )
    assert APPROVAL_TOKEN_VERSION == 3
    permissions = Permissions(approval_secret=SECRET)
    first = approval_token(SECRET, **common)
    assert permissions.verify(first, **common)
    assert not permissions.verify(first, **{**common, "ask": 2})
    assert not permissions.verify(first, **{**common, "operation_id": "later"})
    legacy = approval_token(SECRET, **common, v=2)
    assert legacy != first
    assert permissions.verify(legacy, **common, v=2)
    assert not permissions.verify(legacy, **common)
    assert not permissions.verify(first, **common, v=2)
    payload = {key: value for key, value in common.items() if key not in {"operation_id", "ask"}}
    payload.update(v=2, kind="approval")
    assert legacy == hmac.new(SECRET, jcs(payload), hashlib.sha256).hexdigest()


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), 10**400, "20"])
def test_invalid_approval_ttl_is_refused(value):
    with pytest.raises(ConfigurationError, match="approval_ttl_s"):
        Permissions(approval_ttl_s=value)


@pytest.mark.parametrize("value", [-1, True, float("nan"), float("inf"), 10**400, "20"])
def test_invalid_resend_min_age_is_refused(value):
    with pytest.raises(ConfigurationError, match="resend_min_age_s"):
        Permissions(resend_min_age_s=value)


def test_settlement_callback_and_signing_are_explicit():
    def callback(settlement, effect):
        return True

    permissions = Permissions(
        approval_secret=SECRET, authorize_settlement=callback, resend_min_age_s=0
    )
    assert permissions.authorize_settlement is callback
    fields = {"outcome": "not_applied", "record_id": "ef_1", "by": "reviewer"}
    expected = hmac.new(SECRET, jcs(fields), hashlib.sha256).hexdigest()
    assert permissions.sign_settlement(fields) == expected
    assert Permissions().sign_settlement(fields) is None
    with pytest.raises(ConfigurationError, match="authorize_settlement"):
        Permissions(authorize_settlement="not callable")


def test_approval_carries_application_asserted_identity():
    pending = PendingApproval(tool_call_id="c", name="save_note", arguments={}, token="fixture")
    answer = Approval.for_call(pending, by="reviewer")
    assert answer.by == "reviewer" and answer.token == "fixture" and answer.approved


@pytest.mark.parametrize(
    "changes",
    [
        {"ask": 0},
        {"ask": True},
        {"operation_id": ""},
        {"decided_at": float("nan")},
        {"decided_at": float("inf")},
        {"decided_at": 10**400},
    ],
)
def test_invalid_decision_identity_or_time_is_refused(changes):
    with pytest.raises(ValueError):
        record(**changes)


def test_only_explicit_legacy_tokens_preserve_non_json_fallback():
    class Opaque:
        def __str__(self):
            return "opaque"

    common = dict(session_id="s", run_id="r", position=1, tool_call_id="c", name="save_note")
    with pytest.raises(ValueError, match="strict JSON"):
        approval_token(SECRET, arguments={"object": Opaque()}, **common)
    assert approval_token(SECRET, arguments={"object": Opaque()}, v=2, **common) == approval_token(
        SECRET, arguments={"object": "opaque"}, v=2, **common
    )


@pytest.mark.parametrize("extra", [{"v": 1}, {"v": True}, {"ask": 0}, {"ask": True}])
def test_invalid_token_version_or_ask_is_refused(extra):
    with pytest.raises(ValueError):
        approval_token(
            SECRET,
            session_id="s",
            run_id="r",
            position=1,
            tool_call_id="c",
            name="save_note",
            arguments={},
            **extra,
        )
