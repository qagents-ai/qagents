"""Journal recovery is based on all evidence, never just the latest attempt."""

from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from qagents.effects._fold import EffectView, fold_effects
from qagents.effects._types import HoldReason

from ._samples import committed, failed, intent, reference


def entry(seq, kind, data=None, *, run="run", identity=None, ts=90000.0):
    return SimpleNamespace(
        seq=seq, kind=kind, data=data or {}, run_id=run, id=identity or f"entry-{seq}", ts=ts
    )


def operation(entries):
    return fold_effects(entries).get("shop", "refund", "R1")


def settlement(outcome, *, covers=None, **changes):
    return {
        **reference(),
        "outcome": outcome,
        "by": "operator",
        "evidence": "service log",
        "reason": "operator closed",
        "expected_revision": 2,
        "covers_attempts": covers or [1],
        **changes,
    }


def test_late_commit_after_run_finished_is_not_dropped():
    op = operation(
        [
            entry(1, "effect_intent", intent()),
            entry(2, "run_finished"),
            entry(3, "effect_committed", committed()),
        ]
    )
    assert op.state == "C"
    assert op.receipt["id"] == "refund-1"
    assert op.revision == 3


@pytest.mark.parametrize("first_terminal", [None, "unknown", "interrupted"])
def test_all_attempts_count_when_latest_attempt_was_not_sent(first_terminal):
    entries = [entry(1, "effect_intent", intent())]
    if first_terminal:
        entries.append(entry(2, "effect_failed", failed(why=first_terminal)))
    entries.extend(
        [
            entry(3, "effect_intent", intent(attempt=2)),
            entry(4, "effect_failed", failed(attempt=2, intent_seq=3, why="not_sent")),
        ]
    )
    assert operation(entries).state == "D"


@pytest.mark.parametrize("why,state", [("not_sent", "N"), ("rejected", "R"), ("unknown", "D")])
def test_single_attempt_outcomes(why, state):
    view = fold_effects(
        [entry(1, "effect_intent", intent()), entry(2, "effect_failed", failed(why=why))]
    )
    assert view.get("shop", "refund", "R1").state == state
    assert bool(view.records("session")) == (state != "N")


def test_conflict_receipts_differ_and_original_commit_is_preserved():
    op = operation(
        [
            entry(1, "effect_intent", intent()),
            entry(2, "effect_committed", committed()),
            entry(3, "effect_committed", committed(receipt={"id": "other", "status": "done"})),
        ]
    )
    assert op.state == "X"
    assert op.receipt["id"] == "refund-1"
    assert op.commit_seq == 2
    assert [(finding.what, finding.seqs) for finding in op.findings] == [
        ("receipts_differ", [2, 3])
    ]


def test_status_only_change_updates_non_identity_fields_without_conflict():
    op = operation(
        [
            entry(1, "effect_intent", intent()),
            entry(2, "effect_committed", committed()),
            entry(3, "effect_committed", committed(receipt={"id": "refund-1", "status": "done"})),
        ]
    )
    assert op.state == "C"
    assert op.receipt == {"id": "refund-1", "status": "done"}
    assert not op.findings


@pytest.mark.parametrize(
    "change", [{"version": "2"}, {"request_digest": "changed"}, {"destination_digest": "other"}]
)
def test_conflict_descriptor(change):
    op = operation(
        [
            entry(1, "effect_intent", intent()),
            entry(2, "effect_intent", intent(attempt=2, **change)),
        ]
    )
    assert op.state == "X"
    assert op.findings[0].what == "descriptor"
    assert op.version == "1"


@pytest.mark.parametrize(
    "kind,data",
    [("effect_committed", committed(intent_seq=999)), ("effect_failed", failed(intent_seq=999))],
)
def test_conflict_orphan_terminal(kind, data):
    op = operation([entry(1, "effect_intent", intent()), entry(2, kind, data)])
    assert op.state == "X"
    assert op.findings[0].what == "orphan_terminal"


def test_conflict_late_failure_after_commit():
    op = operation(
        [
            entry(1, "effect_intent", intent()),
            entry(2, "effect_committed", committed()),
            entry(3, "effect_failed", failed()),
        ]
    )
    assert op.state == "X"
    assert op.attempts[0].terminal == "committed"
    assert op.findings[0].what == "late_failure_after_commit"


def test_journaled_conflict_replaces_derived_finding_without_duplicate():
    view = fold_effects(
        [entry(1, "effect_intent", intent()), entry(2, "effect_failed", failed(intent_seq=8))]
    )
    op = view.get("shop", "refund", "R1")
    finding = op.findings[0].model_dump(mode="json")
    view.apply(entry(3, "effect_conflict", finding))
    view.apply(entry(4, "effect_conflict", finding))
    assert op.state == "X"
    assert not op.findings
    assert len(op.conflicts) == 1


def test_duplicate_entry_id_is_a_complete_noop():
    view = fold_effects([entry(1, "effect_intent", intent())])
    before = deepcopy(asdict(view))
    view.apply(entry(9, "effect_committed", committed(), identity="entry-1"))
    assert asdict(view) == before


def test_observation_changes_hold_detail_but_not_state_or_revision():
    view = fold_effects([entry(1, "effect_intent", intent())])
    view.apply(
        entry(
            2,
            "effect_observed",
            {
                **reference(),
                "attempts": [1],
                "verdict": "unknown",
                "reason": "NOT_YET_VISIBLE",
                "detail": "check later",
                "retry_after": 120.0,
            },
        )
    )
    op = view.get("shop", "refund", "R1")
    assert (op.state, op.revision) == ("D", 1)
    hold = view.held("session")[0]
    assert hold.reason == HoldReason.NOT_YET_VISIBLE
    assert hold.detail == "check later"
    assert hold.retry_after.timestamp() == 120.0
    assert hold.retry_after.tzinfo is not None
    assert hold.tool_call_id == "call"


def test_end_evidence_uses_store_ended_at_not_client_entry_timestamp():
    op = operation(
        [
            entry(1, "effect_intent", intent()),
            entry(2, "effect_failed", failed(ended_at=103.0), ts=-1000000),
        ]
    )
    assert op.attempts[0].send_ended_at == 103.0
    op = operation(
        [
            entry(1, "effect_intent", intent()),
            entry(2, "effect_failed", failed(ended_at=None), ts=103.0),
        ]
    )
    assert op.attempts[0].send_ended_at is None


@pytest.mark.parametrize(
    "contract,at,expected",
    [
        ("waits", 104, "R"),
        ("rejects", 104, "D"),
        ("unknown", 104, "D"),
        (None, 104, "D"),
        ("rejects", 109, "R"),
        ("unknown", 109, "R"),
    ],
)
def test_rejected_resend_uses_recorded_contract_and_prior_quiescence(contract, at, expected):
    retry = intent(attempt=2, store_ts=at)
    if contract is not None:
        retry["timing"]["concurrent"] = contract
    op = operation(
        [
            entry(1, "effect_intent", intent()),
            entry(2, "effect_failed", failed()),
            entry(3, "effect_intent", retry),
            entry(4, "effect_failed", failed(attempt=2, intent_seq=3, why="rejected")),
        ]
    )
    assert op.state == expected
    assert op.resolved_by_rejected_resend == (expected == "R")


def test_rejected_resend_does_not_resolve_a_different_key_or_later_unknown():
    retry = intent(attempt=2, idempotency_key="different")
    retry["timing"]["concurrent"] = "waits"
    entries = [
        entry(1, "effect_intent", intent()),
        entry(2, "effect_intent", retry),
        entry(3, "effect_failed", failed(attempt=2, intent_seq=2, why="rejected")),
    ]
    assert operation(entries).state == "D"
    retry["idempotency_key"] = "q1_0"
    assert operation(entries).state == "R"
    entries.append(entry(4, "effect_intent", intent(attempt=3)))
    assert operation(entries).state == "D"


def test_all_earlier_attempts_must_be_quiescent():
    entries = [
        entry(1, "effect_intent", intent()),
        entry(2, "effect_intent", intent(attempt=2)),
        entry(3, "effect_failed", failed(attempt=2, intent_seq=2)),
        entry(4, "effect_intent", intent(attempt=3, store_ts=200)),
        entry(5, "effect_failed", failed(attempt=3, intent_seq=4, why="rejected")),
    ]
    assert operation(entries).state == "D"


@pytest.mark.parametrize(
    "may_overrun,overran,expected", [(False, False, "R"), (True, False, "D"), (False, True, "D")]
)
def test_takeover_evidence_uses_earliest_higher_epoch(may_overrun, overran, expected):
    entries = [
        entry(1, "effect_intent", intent(may_overrun=may_overrun)),
        entry(2, "effect_failed", failed(send_ended=False, ended_at=None, overran=overran)),
        entry(
            3,
            "lease_acquired",
            {
                "owner_token": "worker#2",
                "epoch": 2,
                "acquired_at": 110.0,
                "worker": "worker",
                "purpose": "run",
            },
            run=None,
        ),
        entry(
            4,
            "lease_acquired",
            {
                "owner_token": "worker#3",
                "epoch": 3,
                "acquired_at": 200.0,
                "worker": "worker",
                "purpose": "run",
            },
            run=None,
        ),
        entry(5, "effect_intent", intent(attempt=2, store_ts=116.0)),
        entry(6, "effect_failed", failed(attempt=2, intent_seq=5, why="rejected")),
    ]
    view = fold_effects(entries)
    assert view.get("shop", "refund", "R1").state == expected
    assert view.acquisitions == [(2, 110.0, 3), (3, 200.0, 4)]


def test_generation_only_counts_its_own_attempts():
    op = operation(
        [
            entry(1, "effect_intent", intent()),
            entry(2, "effect_failed", failed(why="rejected")),
            entry(3, "effect_intent", intent(attempt=2, generation=1)),
            entry(4, "effect_failed", failed(attempt=2, intent_seq=3, why="not_sent")),
        ]
    )
    assert op.state == "N"


@pytest.mark.parametrize(
    "outcome,state", [("not_applied", "NA"), ("close_unknown", "CU"), ("resend", "D")]
)
def test_operator_outcomes_are_distinct(outcome, state):
    op = operation(
        [entry(1, "effect_intent", intent()), entry(2, "effect_settled", settlement(outcome))]
    )
    assert op.state == state
    assert (op.pending_resend is not None) == (outcome == "resend")


def test_resend_is_consumed_only_by_matching_authorized_attempt_and_late_commit_voids_it():
    view = fold_effects(
        [entry(1, "effect_intent", intent()), entry(2, "effect_settled", settlement("resend"))]
    )
    op = view.get("shop", "refund", "R1")
    view.apply(
        entry(
            3,
            "effect_intent",
            intent(
                attempt=2, rule="D5", authorization={"decision": "resend", "settlement_seq": 999}
            ),
        )
    )
    assert op.pending_resend is not None
    view.apply(
        entry(
            4,
            "effect_intent",
            intent(attempt=3, rule="D5", authorization={"decision": "resend", "settlement_seq": 2}),
        )
    )
    assert op.pending_resend is None
    view.apply(entry(5, "effect_committed", committed()))
    assert op.state == "C"
    assert view.records("session")[0].settlement.voided


@pytest.mark.parametrize("outcome", ["not_applied", "close_unknown"])
def test_found_after_closed_settlement_conflicts(outcome):
    op = operation(
        [
            entry(1, "effect_intent", intent()),
            entry(2, "effect_settled", settlement(outcome)),
            entry(3, "effect_committed", committed(via="lookup")),
        ]
    )
    assert op.state == "X"
    assert op.findings[0].what == "late_found_after_settlement"


def test_call_ids_reused_across_turns_and_runs_keep_operation_identity():
    entries = [
        entry(1, "run_started", {"requires": ["effects/1"]}),
        entry(2, "effect_intent", intent(operation_id="run:1:call")),
        entry(3, "tool_result", {"tool_call_id": "call", "operation_id": "run:1:call"}),
        entry(4, "effect_intent", intent(operation_id="run:2:call", key="R2")),
        entry(5, "run_started", {"requires": ["effects/1"]}, run="next"),
        entry(6, "effect_intent", intent(operation_id="next:1:call", key="R3"), run="next"),
        entry(7, "tool_result", {"tool_call_id": "call"}, run="next"),
    ]
    view = fold_effects(entries)
    assert len(view.by_call) == 3
    assert not view.incompatible_closures
    assert {held.key: held.tool_call_id for held in view.held("session")} == {
        "R1": None,
        "R2": "call",
        "R3": None,
    }


def test_old_worker_closure_becomes_conflict_but_plain_old_journal_is_unchanged():
    entries = [
        entry(1, "run_started"),
        entry(2, "effect_intent", intent()),
        entry(3, "tool_result", {"tool_call_id": "call", "content": "interrupted"}),
    ]
    view = fold_effects(entries)
    assert view.get("shop", "refund", "R1").state == "X"
    assert view.held("session")[0].reason == HoldReason.CLOSED_BY_INCOMPATIBLE_WORKER
    assert view.held("session")[0].tool_call_id is None
    assert not fold_effects([entries[0], entries[2]]).records("session")


def test_incremental_and_batch_folds_match_every_prefix():
    entries = [
        entry(1, "effect_intent", intent()),
        entry(2, "effect_failed", failed()),
        entry(3, "effect_settled", settlement("resend")),
        entry(
            4,
            "effect_intent",
            intent(attempt=2, rule="D5", authorization={"decision": "resend", "settlement_seq": 3}),
        ),
        entry(5, "effect_committed", committed(attempt=2, intent_seq=4)),
        entry(6, "run_finished"),
        entry(7, "effect_committed", committed(receipt={"id": "refund-1", "status": "done"})),
    ]
    view = EffectView()
    for index, item in enumerate(entries):
        view.apply(item)
        assert asdict(view) == asdict(fold_effects(entries[: index + 1]))
        assert view.records("session") == fold_effects(entries[: index + 1]).records("session")


def test_record_id_lookup_refuses_collisions_and_does_not_guess():
    view = fold_effects(
        [
            entry(1, "effect_intent", intent()),
            entry(2, "effect_intent", intent(key="R2", record_id=reference()["record_id"])),
        ]
    )
    assert view.by_record_id("missing") is None
    with pytest.raises(ValueError, match="ambiguous"):
        view.by_record_id(reference()["record_id"])
