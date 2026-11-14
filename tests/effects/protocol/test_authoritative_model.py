"""Bounded schedules for an authoritative destination using the production decision table."""

from collections import deque
from dataclasses import dataclass, replace

import pytest

from qagents.effects import _quiescence as timing
from qagents.effects import _table as table

BOUND = 2
HORIZON = 9


@dataclass(frozen=True)
class State:
    time: int = 0
    owner: int = 0
    lease_until: int = 3
    issued: tuple[int, ...] = (0,)
    pending: tuple[int, ...] = (0,)
    applied: int = 0
    prior_end: int = 0
    committed: bool = False
    held: bool = False


def successors(current, *, late):
    if current.time < HORIZON and (
        late or all(current.time + 1 <= issued + BOUND for issued in current.pending)
    ):
        yield "tick", replace(current, time=current.time + 1)
    if current.pending:
        yield "apply", replace(current, pending=current.pending[1:], applied=current.applied + 1)
        yield "lost_before_apply", replace(current, pending=current.pending[1:])
    if current.owner == 0 and current.time >= current.lease_until + 1:
        yield "takeover", replace(current, owner=1, lease_until=current.time + 3)
    if current.owner != 1 or current.committed or len(current.issued) > 1:
        return
    if current.time >= current.lease_until:
        return
    if current.applied:
        verdict = "found"
    else:
        quiescent = timing.q_of(
            ended_at=current.prior_end,
            acq_after=None,
            applies_within_s=BOUND,
            m_s=1,
            sigma_s=1,
            may_overrun=False,
            overran=False,
            not_sent=False,
        )
        final = timing._lookup_after_quiescence(current.time, quiescent)
        verdict = "absent_authoritative_after_q" if final else "absent"
    situation = table.PerformSituation(
        allowed=True,
        same_call=True,
        descriptor_differs=False,
        arguments_diverged=False,
        key_changed=False,
        version_changed=False,
        own_rejection=False,
        claim="none",
        lookup=verdict,
        fence_ok=True,
        auth_valid=True,
        dedupe_valid=False,
        resend_accepted=False,
    )
    row = table.decide(table.TState.D, table.TEvent.PERFORM, situation)
    if row.rule == "D4":
        yield (
            row.rule,
            replace(
                current,
                issued=(*current.issued, current.time),
                pending=(*current.pending, current.time),
            ),
        )
    elif row.rule == "L1":
        yield row.rule, replace(current, committed=True)
    elif row.rule == "H1":
        yield row.rule, replace(current, held=True)
    else:
        pytest.fail(f"Unexpected authoritative recovery decision: {row.rule}")


def explore(*, late=False):
    queue = deque([(State(), ())])
    visited = set()
    reached = set()
    while queue:
        state, history = queue.popleft()
        if state in visited:
            continue
        visited.add(state)
        if state.applied > 1:
            return history, reached
        for event, following in successors(state, late=late):
            reached.add(event)
            if following not in visited:
                queue.append((following, (*history, event)))
    return None, reached


def test_authoritative_recovery_with_bounded_application_is_safe_in_model():
    counterexample, reached = explore()
    assert counterexample is None
    assert {"takeover", "L1", "D4"} <= reached


def test_authoritative_recovery_detects_late_application_counterexample():
    counterexample, reached = explore(late=True)
    assert counterexample
    assert "D4" in counterexample and counterexample[-1] == "apply"
    assert "L1" in reached
