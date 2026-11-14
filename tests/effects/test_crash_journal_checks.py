"""Independent journal checks reject falsified recovery evidence."""

import pytest

from qagents.session import Entry
from qagents.testing.crash._journal_checks import (
    check_decisions,
    check_patches,
    check_receipts,
    check_starts,
    check_terminal,
)

from ._samples import committed, intent


@pytest.fixture
def journal():
    return [
        Entry(id="first", seq=1, kind="effect_intent", data=intent()),
        Entry(
            id="second",
            seq=2,
            kind="effect_committed",
            data=committed(
                state_patch={"set": {"balance": 10}, "unset": []},
            ),
        ),
    ]


def test_receipt_identity_must_match_earlier_intent(journal):
    assert check_receipts(journal).status == "safe"
    journal[1].data["receipt"]["id"] = "different receipt"
    assert check_receipts(journal).findings[0].code == "receipt_identity_mismatch"
    journal[1].data["intent_seq"] = 999
    assert check_receipts(journal).findings[0].code == "commit_without_intent"


def test_late_replay_cannot_write_another_patch(journal):
    assert check_patches(journal, {"balance": 10}).status == "safe"
    duplicate = journal[1].model_copy(deep=True, update={"id": "third", "seq": 3})
    assert check_patches([*journal, duplicate], {"balance": 10}).findings[0].code == (
        "patch_repeated"
    )
    assert check_patches(journal, {"balance": 20}).findings[0].code == "final_state_differs"


def test_terminal_receipt_or_surfaced_hold_required(journal):
    assert check_terminal(journal, []).status == "safe"
    assert check_terminal(journal[:1], []).findings[0].code == "intent_not_surfaced"


def test_a_completed_call_cannot_start_again_even_when_idempotent():
    common = {"operation_id": "operation", "name": "charge", "tool_call_id": "call"}
    start = Entry(kind="tool_started", data=common)
    result = Entry(
        kind="tool_result",
        data={
            "operation_id": "operation",
            "result": {"tool_call_id": "call"},
        },
    )
    assert check_starts([start, result], {"charge"}).status == "safe"
    assert check_starts([start, result, start], {"charge"}).status == "violated"
    assert check_starts([start, start], set()).status == "violated"


def test_intent_cannot_cite_another_calls_approval(journal):
    data = journal[0].data
    decision = Entry(
        kind="approval_decided",
        data={
            "operation_id": data["operation_id"],
            "tool_call_id": data["tool_call_id"],
            "approval_id": "approval",
            "approved": True,
            "arguments_digest": "args",
        },
    )
    data["authorization"] = {
        "approval_id": "approval",
        "decision": "ask",
        "arguments_digest": "args",
    }
    assert check_decisions([decision, *journal], [], []).status == "safe"
    decision.data["operation_id"] = "another call"
    assert check_decisions([decision, *journal], [], []).findings[0].code == (
        "intent_decision_mismatch"
    )
