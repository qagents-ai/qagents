"""Operator answers resolve a journal prefix, never later contradictory evidence."""

from copy import deepcopy
from dataclasses import asdict

import pytest

from qagents.effects import _kinds as kinds
from qagents.effects._fold import SETTLEMENT_FOLD_KEYS, fold_effects

from ._samples import committed, failed, intent, reference
from .test_fold import entry, settlement


def conflicted():
    return fold_effects(
        [
            entry(1, "effect_intent", intent()),
            entry(2, "effect_committed", committed()),
            entry(3, "effect_committed", committed(receipt={"id": "other"})),
        ]
    )


def answer(seq=4, outcome="committed", **changes):
    if outcome == "committed":
        return entry(
            seq,
            "effect_committed",
            committed(
                attempt=None,
                intent_seq=None,
                via="operator",
                receipt={"id": "verified"},
                by="auditor",
                evidence="provider reconciliation #12",
                expected_revision=3,
                **changes,
            ),
        )
    return entry(
        seq,
        "effect_settled",
        settlement(outcome, expected_revision=3, **changes),
    )


@pytest.mark.parametrize(
    "outcome,state", [("committed", "C"), ("not_applied", "NA"), ("close_unknown", "CU")]
)
def test_operator_answer_resolves_existing_conflict_without_erasing_evidence(outcome, state):
    view = conflicted()
    op = view.get("shop", "refund", "R1")
    original = deepcopy(op.findings)
    view.apply(answer(outcome=outcome))
    assert op.state == state
    assert op.findings == original
    assert op.revision == 4
    record = view.records("session")[0]
    assert record.settlement.outcome == outcome
    assert record.settlement.seq == 4
    assert record.receipt == ({"id": "verified"} if outcome == "committed" else None)


@pytest.mark.parametrize("outcome", ["committed", "not_applied", "close_unknown"])
def test_same_operator_entry_is_idempotent(outcome):
    view = conflicted()
    decision = answer(outcome=outcome)
    view.apply(decision)
    before = deepcopy(asdict(view))
    view.apply(decision)
    assert asdict(view) == before


@pytest.mark.parametrize("outcome", ["committed", "not_applied", "close_unknown"])
def test_new_evidence_for_old_attempt_reopens_resolved_conflict(outcome):
    view = conflicted()
    view.apply(answer(outcome=outcome))
    view.apply(entry(5, "effect_committed", committed(via="lookup", receipt={"id": "late"})))
    op = view.get("shop", "refund", "R1")
    assert op.state == "X"
    expected = "receipts_differ" if outcome == "committed" else "late_found_after_settlement"
    assert any(item.what == expected and 5 in item.seqs for item in op.findings)


@pytest.mark.parametrize(
    "change", [{"by": ""}, {"by": "  "}, {"evidence": ""}, {"expected_revision": 2}]
)
def test_incomplete_or_stale_operator_commit_does_not_resolve_conflict(change):
    view = conflicted()
    decision = answer()
    decision.data.update(change)
    view.apply(decision)
    assert view.get("shop", "refund", "R1").state == "X"


@pytest.mark.parametrize("outcome", ["not_applied", "close_unknown"])
@pytest.mark.parametrize("invalid", ["attribution", "explanation", "revision"])
def test_incomplete_or_stale_closure_does_not_resolve_conflict(outcome, invalid):
    view = conflicted()
    decision = answer(outcome=outcome)
    field = {
        "attribution": "by",
        "explanation": "reason" if outcome == "close_unknown" else "evidence",
        "revision": "expected_revision",
    }[invalid]
    decision.data[field] = 2 if invalid == "revision" else ""
    view.apply(decision)
    assert view.get("shop", "refund", "R1").state == "X"


def test_operator_answer_cannot_hide_its_own_orphan_terminal():
    view = conflicted()
    decision = answer()
    decision.data.update(attempt=999, intent_seq=999)
    view.apply(decision)
    op = view.get("shop", "refund", "R1")
    assert op.state == "X"
    assert any(item.what == "orphan_terminal" and item.seqs == [4] for item in op.findings)


def test_stale_answer_with_new_entry_id_cannot_resolve_later_conflict():
    view = conflicted()
    view.apply(answer())
    view.apply(entry(5, "effect_committed", committed(receipt={"id": "new"})))
    view.apply(answer(seq=6))
    assert view.get("shop", "refund", "R1").state == "X"


def test_operator_commit_summary_is_present_without_an_earlier_settlement():
    view = fold_effects([entry(1, "effect_intent", intent())])
    decision = answer(seq=2)
    decision.data["expected_revision"] = 1
    view.apply(decision)
    record = view.records("session")[0]
    assert record.status == "committed"
    assert record.settlement.outcome == "committed"
    assert record.settlement.by == "auditor"
    assert record.settlement.evidence == "provider reconciliation #12"


def test_journaling_resolved_finding_does_not_reopen_it():
    view = conflicted()
    op = view.get("shop", "refund", "R1")
    finding = op.findings[0].model_dump(mode="json")
    view.apply(answer())
    view.apply(entry(5, "effect_conflict", finding))
    assert op.state == "C"
    assert op.conflicts[0].seqs == [2, 3]


def test_operator_commit_can_resolve_found_after_closed_settlement():
    view = fold_effects(
        [
            entry(1, "effect_intent", intent()),
            entry(2, "effect_settled", settlement("not_applied")),
            entry(3, "effect_committed", committed(via="lookup")),
        ]
    )
    assert view.get("shop", "refund", "R1").state == "X"
    view.apply(answer())
    view.apply(entry(5, "effect_committed", committed(via="lookup", receipt={"id": "verified"})))
    assert view.get("shop", "refund", "R1").state == "C"
    assert view.records("session")[0].settlement.outcome == "committed"


@pytest.mark.parametrize("rule", ["D3", "D4", "D5"])
def test_any_later_attempt_voids_prior_resend_authorization(rule):
    view = fold_effects(
        [
            entry(1, "effect_intent", intent()),
            entry(2, "effect_settled", settlement("resend")),
            entry(3, "effect_intent", intent(attempt=2, rule=rule)),
        ]
    )
    assert view.get("shop", "refund", "R1").pending_resend is None
    assert view.records("session")[0].settlement.voided


def test_each_answer_keeps_the_revision_it_followed_and_a_later_copy_does_not_match():
    genuine = entry(3, "effect_settled", settlement("resend", expected_revision=2))
    copy = entry(5, "effect_settled", settlement("resend", expected_revision=2), identity="copy")
    view = fold_effects(
        [
            entry(1, "effect_intent", intent()),
            entry(2, "effect_failed", failed()),
            genuine,
            entry(4, "effect_observed", {**reference(), "attempts": [1], "verdict": None}),
        ]
    )
    op = view.get("shop", "refund", "R1")
    assert op.pending_resend.model_extra == {"seq": 3, "voided": False, "prior_revision": 2}
    assert op.pending_resend.expected_revision == op.pending_resend.model_extra["prior_revision"]
    view.apply(copy)
    assert op.pending_resend.model_extra["seq"] == 5
    assert (
        op.pending_resend.model_extra["prior_revision"] == 3 != op.pending_resend.expected_revision
    )
    assert set(op.pending_resend.model_extra) == SETTLEMENT_FOLD_KEYS
    signed = op.pending_resend.model_dump(mode="json", exclude=SETTLEMENT_FOLD_KEYS)
    assert signed == kinds.EffectSettledData.model_validate(copy.data).model_dump(mode="json")
