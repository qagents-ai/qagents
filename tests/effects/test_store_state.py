"""Replay preserves effect evidence and isolates provisional tool state."""

import pytest

from qagents.errors import SessionError
from qagents.session import (
    Entry,
    InMemoryStore,
    Session,
    SessionState,
    SessionStatus,
    advance_status,
    derive_state,
)

from ._samples import committed, intent


def entry(kind, data=None, run_id="r", seq=0):
    return Entry(kind=kind, data=data or {}, run_id=run_id, seq=seq)


def test_late_effect_patch_survives_finished_run_and_provisional_state_needs_same_operation():
    entries = [
        entry("run_started"),
        entry("state_updated", {"values": {"stable": 1}}),
        entry(
            "state_updated", {"values": {"bad": 2}, "provisional": True, "operation_id": "r:1:call"}
        ),
        entry(
            "tool_result",
            {
                "operation_id": "r:2:call",
                "result": {
                    "type": "tool_result",
                    "tool_call_id": "call",
                    "name": "tool",
                    "content": "ok",
                },
            },
        ),
        entry("effect_intent", intent(), seq=5),
        entry("run_finished"),
        entry(
            "effect_committed",
            committed(
                intent_seq=5, state_patch={"set": {"receipt": "refund-1"}, "unset": ["stable"]}
            ),
            seq=7,
        ),
        entry("state_updated", {"values": {"late": "ignored"}}),
    ]
    state = derive_state(entries, session_id="s")
    assert state.values == {"receipt": "refund-1"}
    assert state.effects.records("s")[0].status == "committed"
    assert "bad" not in state.values


def test_replay_provisional_success_and_operation_indexes():
    entries = [
        entry("run_started", {"requires": ["effects/1"]}),
        entry("message", {"message": {"role": "assistant", "parts": []}}),
        entry("tool_started", {"tool_call_id": "call", "operation_id": "r:1:call"}),
        entry("tool_started", {"tool_call_id": "call", "operation_id": "r:1:call"}),
        entry("approval_decided", {"operation_id": "r:1:call", "approved": False}),
        entry("approval_decided", {"operation_id": "r:1:call", "approved": True}),
        entry("memo_recorded", {"operation_id": "r:1:call", "name": "memo", "value": {"n": 1}}),
        entry(
            "state_updated",
            {"values": {"accepted": 1}, "provisional": True, "operation_id": "r:1:call"},
        ),
        entry(
            "tool_result",
            {
                "operation_id": "r:1:call",
                "result": {
                    "type": "tool_result",
                    "tool_call_id": "call",
                    "name": "tool",
                    "content": "ok",
                },
            },
        ),
        entry("compaction", {"messages": []}),
    ]
    state = derive_state(entries)
    assert state.values == {"accepted": 1} and state.assistant_messages == 1
    assert state.tool_attempts == {"r:1:call": 2}
    assert len(state.open_turn_starts["call"]) == 2
    assert state.decision_counts == {"r:1:call": 2}
    assert state.decisions["r:1:call"]["approved"] is True
    assert state.memos == {"r:1:call": {"memo": {"n": 1}}}
    assert state.run_requires == {"r": ["effects/1"]}
    assert SessionState().effects.operations == {}


def test_lease_and_operator_entries_do_not_resume_a_paused_run():
    status = SessionStatus(
        session_id="s", status="paused", open_run_id="r", failures=2, retry_after=123
    )
    advance_status(status, entry("lease_acquired", run_id=None))
    advance_status(status, entry("effect_committed", run_id=None))
    assert status.has_effects and status.status == "paused" and status.failures == 2
    advance_status(status, entry("run_failed", {"retry_after": 456}))
    assert status.retry_after == 456
    advance_status(status, entry("run_resumed"))
    assert status.retry_after is None and status.status == "running"


async def test_held_pending_input_filters_and_force_tombstone():
    db = InMemoryStore()
    session = Session(db, id="s")
    await db.acquire_lease("s", "owner", 30)
    await db.append_leased(
        "s",
        [
            entry("run_started"),
            entry("effect_intent", intent(), seq=2),
            entry(
                "tool_deferred",
                {
                    "input": {"tool_call_id": "call", "name": "refund"},
                    "held_record_id": intent()["record_id"],
                },
            ),
        ],
        "owner",
    )
    state = await session.state()
    assert state.pending_inputs[0].held.id == intent()["record_id"]
    assert len(await session.effects(status="held", namespace="shop")) == 1
    assert await session.effects(status="committed") == []
    with pytest.raises(SessionError):
        await session.clear()
    await session.clear(force=True)
    assert await session.entries() == []
    assert await db.get_claim("shop", "refund", "R1") is not None


async def test_clear_does_not_pass_new_keyword_to_legacy_store():
    class Legacy(InMemoryStore):
        async def delete(self, session_id):
            await super().delete(session_id)

    await Session(Legacy(), id="s").clear()
