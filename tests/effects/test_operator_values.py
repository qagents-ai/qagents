from datetime import UTC, datetime

import pytest

from qagents.effects._types import HOLD_DETAIL, EffectRecord, HeldEffect, HoldReason, Settle


def held():
    return HeldEffect(
        "ef_abc",
        "session",
        "refund",
        "R1",
        "shop",
        HoldReason.RESPONSE_LOST,
        HOLD_DETAIL[HoldReason.RESPONSE_LOST],
        datetime(2026, 11, 13, tzinfo=UTC),
        12,
        "call-1",
    )


def test_hold_details_cover_every_reason():
    assert set(HOLD_DETAIL) == set(HoldReason)
    assert all(isinstance(text, str) and text.endswith(".") for text in HOLD_DETAIL.values())
    assert str(HoldReason.RESPONSE_LOST) == "RESPONSE_LOST"


def test_expired_intent_approval_points_to_operator_paths():
    detail = HOLD_DETAIL[HoldReason.APPROVAL_EXPIRED]
    assert "new decision" not in detail
    assert "operator" in detail and "recheck" in detail


def test_settlement_constructors_preserve_revision_and_distinct_meanings():
    original = held()
    receipt = {"id": "refund-1"}
    done = Settle.committed(original, receipt, by="ops:kim", evidence="provider record")
    assert (done.record_id, done.expected_revision, done.receipt) == ("ef_abc", 12, receipt)
    assert (done.by, done.evidence, done.reason) == ("ops:kim", "provider record", None)
    assert (
        Settle.not_applied(original, by="ops:kim", evidence="verified absent").outcome
        == "not_applied"
    )
    assert Settle.resend(original, by="ops:kim", evidence="verified absent").outcome == "resend"
    unknown = Settle.close_unknown(original, by="ops:kim", reason="retention exhausted")
    assert (unknown.outcome, unknown.evidence, unknown.receipt) == ("close_unknown", None, None)


def test_settlement_from_record_or_id_uses_the_intended_revision():
    record = EffectRecord(
        "ef_abc", "session", "shop", "refund", "R1", "held", None, None, held(), (), None, 23
    )
    assert Settle.resend(record, by="operator", evidence="checked").expected_revision == 23
    assert Settle.resend("ef_abc", by="operator", evidence="checked").expected_revision is None


@pytest.mark.parametrize("invalid", [{"by": ""}, {"by": "  "}, {"evidence": ""}])
def test_settlement_requires_operator_and_evidence(invalid):
    with pytest.raises(ValueError):
        Settle.resend(held(), **{"by": "operator", "evidence": "checked", **invalid})


def test_close_unknown_needs_reason_and_record_reference():
    with pytest.raises(ValueError, match="nonempty"):
        Settle.close_unknown(held(), by="operator", reason="")
    with pytest.raises(ValueError, match="record id"):
        Settle.resend("", by="operator", evidence="checked")
    with pytest.raises(TypeError, match="HeldEffect"):
        Settle.resend(42, by="operator", evidence="checked")


@pytest.mark.parametrize(
    "record_id, error",
    [
        ("", ValueError),
        ("   ", ValueError),
        (None, TypeError),
        (42, TypeError),
        (b"ef_", TypeError),
    ],
)
def test_directly_built_answer_refuses_an_empty_or_non_string_record_id(record_id, error):
    with pytest.raises(error, match="record id"):
        Settle("resend", record_id, None, None, "operator", "checked", None)


def test_directly_built_answer_keeps_a_short_record_id_and_no_revision():
    answer = Settle("close_unknown", "ef_a", None, None, "operator", None, "review")
    assert (answer.record_id, answer.expected_revision) == ("ef_a", None)
