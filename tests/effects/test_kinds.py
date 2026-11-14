import inspect

from pydantic import BaseModel

from qagents.effects import _kinds as kinds
from qagents.effects._canonical import digest

from ._samples import claimed, committed, failed, intent


def test_all_journal_models_preserve_unknown_fields():
    models = [
        cls
        for name, cls in vars(kinds).items()
        if inspect.isclass(cls)
        and issubclass(cls, BaseModel)
        and cls.__module__ == kinds.__name__
        and not name.startswith("_")
    ]
    assert len(models) == 25
    assert all(cls.model_config["extra"] == "allow" for cls in models)
    data = kinds.EffectIntentData.model_validate({**intent(), "future_field": {"revision": 2}})
    assert data.model_dump(mode="json", exclude_none=False)["future_field"] == {"revision": 2}
    assert data.model_dump()["authorization"]["settlement_seq"] is None


def test_journal_payloads_round_trip_and_preserve_semantics():
    for cls, data in (
        (kinds.EffectIntentData, intent()),
        (kinds.EffectCommittedData, committed()),
        (kinds.EffectFailedData, failed()),
        (kinds.EffectClaimedData, claimed()),
    ):
        parsed = cls.model_validate(data)
        assert cls.model_validate_json(parsed.model_dump_json()) == parsed
    assert kinds.EffectFailedData.model_validate(failed()).committed is None
    assert kinds.EffectFailedData.model_validate(failed(why="rejected")).committed is False


def test_legacy_additions_default_without_shared_mutable_values():
    first = kinds.RunStartedV2()
    first.requires.append("effects/1")
    assert kinds.RunStartedV2().requires == []
    assert kinds.RunStartedV2().kept_held == []
    assert kinds.ToolStartedV2().attempt == 1
    assert kinds.ToolResultV2().operation_id is None
    assert kinds.StateUpdatedV2().provisional is False
    patch = kinds.StatePatchData()
    patch.set["one"] = 1
    assert kinds.StatePatchData().set == {}
    assert kinds.EffectIntentData.model_validate(intent()).timing.concurrent == "unknown"


def test_kind_sets_and_capability_markers():
    assert len(kinds.EFFECT_KINDS) == 8
    assert kinds.KIND_EFFECT_CLAIMED in kinds.STATUS_CHANGING_KINDS
    assert kinds.STATUS_CHANGING_KINDS < kinds.EFFECT_KINDS
    assert kinds.KIND_EFFECT_OBSERVED not in kinds.STATUS_CHANGING_KINDS
    assert kinds.KIND_EFFECT_SEND_ENDED not in kinds.STATUS_CHANGING_KINDS
    assert {"message", "tool_result", "approval_decided", "lease_acquired"} <= kinds.LIFECYCLE_KINDS
    assert kinds.SUPPORTED_REQUIRES == {"effects/1", "approvals/3"}
    assert (kinds.PROTOCOL, kinds.DIGEST_ALG) == (1, "jcs-sha256/1")


def test_deterministic_entry_ids_bind_the_complete_identity():
    assert kinds.lease_entry_id("worker:token") == "la_" + digest("worker:token")[:24]
    assert kinds.lease_entry_id("worker:token") != kinds.lease_entry_id("worker:next")
    expected = "cf_" + digest(["ef_one", "receipts_differ", [1, 3]])[:24]
    assert kinds.conflict_entry_id("ef_one", "receipts_differ", [1, 3]) == expected
    assert kinds.conflict_entry_id("ef_one", "receipts_differ", [1, 4]) != expected


def test_approval_record_mac_inventory_and_extra_fields():
    assert kinds.APPROVAL_MAC_FIELDS == (
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
    )
    record = kinds.ApprovalRecord(
        approval_id="ap1_test",
        session_id="s",
        run_id="r",
        tool_call_id="call",
        operation_id="r:1:call",
        name="refund",
        approved=True,
        reason=None,
        arguments_digest="args",
        tool_spec_digest="spec",
        by="ops",
        decided_at=100,
        expires_at=200,
        mac=None,
        future_hint="retained",
    )
    assert (record.v, record.ask, record.arguments_changed, record.arguments) == (3, 1, False, None)
    assert record.model_dump()["future_hint"] == "retained"


def test_settlement_mac_fields_bind_session_kind_and_every_field_but_the_mac():
    data = {"record_id": "ef_1", "outcome": "resend", "expected_revision": 4, "mac": "ab"}
    fields = kinds.settlement_mac_fields("s", kinds.KIND_EFFECT_SETTLED, data)
    assert fields == {
        "session_id": "s",
        "kind": "effect_settled",
        "data": {"record_id": "ef_1", "outcome": "resend", "expected_revision": 4},
    }
    assert data["mac"] == "ab"  # the journaled data is not changed
