"""Timing bounds and task-local harness ports have executable boundary contracts."""

import asyncio
import gc
import weakref
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from qagents.effects import _ports as ports
from qagents.effects._types import Attempt, AttemptDebug, Inquiry


def effect(timeout=20):
    return SimpleNamespace(name="refund", timeout_s=timeout)


def test_timing_preserves_ordinary_short_leases_and_effect_slack():
    assert ports.Timing.for_lease(0.3, 0, effects=False).m_s == pytest.approx(0.03)
    assert ports.Timing.for_lease(0.3, 0, effects=True).m_s == 2
    assert ports.Timing.for_lease(60, 1, effects=True).m_s == 6


def test_lease_ttl_bound_and_actionable_knobs():
    timing = ports.Timing(30, 3, 1, rho=0)
    assert timing.lease_ttl_problem(effect(24)) is None
    problem = timing.lease_ttl_problem(effect(24.001))
    assert "refund" in problem
    assert "lease_ttl_s=30" in problem
    assert "timeout_s=24.001" in problem
    assert "30.001" in problem
    assert ports.Timing(30, 3, 1).lease_ttl_problem(effect(24)) is not None


def test_tool_timeout_bound_distinguishes_maximum_from_sequential_sum():
    timing = ports.Timing(60, 6, 1)
    assert timing.tool_timeout_problem(None, [effect()]) == (None, None)
    assert timing.tool_timeout_problem(1, []) == (None, None)
    error, warning = timing.tool_timeout_problem(57.999, [effect()])
    assert "58" in error and warning is None
    assert timing.tool_timeout_problem(58, [effect()]) == (None, None)
    error, warning = timing.tool_timeout_problem(58, [effect(), effect(10)])
    assert error is None and "96" in warning
    assert timing.tool_timeout_problem(96, [effect(), effect(10)]) == (None, None)


@pytest.mark.parametrize(
    "values", [(0, 0, 0), (float("nan"), 0, 0), (10, -1, 0), (10, 1, float("inf")), (10, 1, 0, 1)]
)
def test_invalid_timing_is_refused(values):
    with pytest.raises(ValueError):
        ports.Timing(*values)


def test_authorization_data_preserves_each_decision_binding():
    auth = ports.Authorization("ask", "refund-policy", "ap1:123", 999.0, False, "args", 12)
    assert auth.data().model_dump() == asdict(auth)


@pytest.mark.asyncio
async def test_model_turn_and_inquiry_are_task_local_and_resettable():
    assert ports.current_inquiry() is None
    assert ports.MODEL_TURN.get() is None

    async def task(number):
        inquiry = Inquiry({}, str(number), "tag", (), (), None)
        inquiry_token = ports._INQUIRY.set(inquiry)
        turn = ports.ModelTurn("session", "run", number)
        turn_token = ports.MODEL_TURN.set(turn)
        try:
            await asyncio.sleep(0)
            assert ports.current_inquiry() is inquiry
            assert ports.MODEL_TURN.get() is turn
        finally:
            ports._INQUIRY.reset(inquiry_token)
            ports.MODEL_TURN.reset(turn_token)

    await asyncio.gather(task(1), task(2))
    assert ports.current_inquiry() is None
    assert ports.MODEL_TURN.get() is None


@pytest.mark.parametrize("error", [RuntimeError("observer"), asyncio.CancelledError()])
def test_observer_cannot_change_engine_outcome(error):
    calls = []

    def observer(event):
        calls.append(event)
        raise error

    event = ports.ReachEvent("session", "record", "refund", "D1", "first_send", None)
    token = ports.REACH_OBSERVER.set(observer)
    try:
        ports.reached(event)
    finally:
        ports.REACH_OBSERVER.reset(token)
    assert calls == [event]
    ports.reached(event)
    assert calls == [event]


def test_attempt_liveness_is_by_object_identity_and_ends_on_abandonment():
    class Request(BaseModel):
        amount: int

    attempt = Attempt(Request(amount=5), "key", "idem", "tag", 10, None, AttemptDebug(1, 0, 1, 1))
    equal_attempt = replace(attempt)
    assert not ports.attempt_live(attempt)
    ports._set_attempt_live(attempt, True)
    try:
        assert ports.attempt_live(attempt)
        assert not ports.attempt_live(equal_attempt)
        attempt.request.amount = 6
        assert ports.attempt_live(attempt)
    finally:
        ports._set_attempt_live(attempt, False)
    assert not ports.attempt_live(attempt)


def test_liveness_registry_does_not_retain_discarded_attempts():
    attempt = Attempt({}, "key", "idem", "tag", 10, None, AttemptDebug(1, 0, 1, 1))
    identity, ref = id(attempt), weakref.ref(attempt)
    ports._set_attempt_live(attempt, True)
    del attempt
    gc.collect()
    assert ref() is None
    assert identity not in ports._LIVE_ATTEMPTS
