"""Quiescence must come from recorded evidence and include every possible arrival."""

import math
from types import SimpleNamespace

import pytest

from qagents.effects import _quiescence as q
from qagents.effects._declare import Dedupe, Unique
from qagents.effects._fold import fold_effects
from qagents.effects._ports import Timing

from ._samples import failed, intent


def core(**changes):
    args = dict(
        ended_at=None,
        acq_after=None,
        applies_within_s=10.0,
        m_s=2.0,
        sigma_s=1.0,
        may_overrun=False,
        overran=False,
        not_sent=False,
    )
    return q.q_of(**(args | changes))


def entry(seq, kind, data):
    return SimpleNamespace(
        seq=seq, kind=kind, data=data, run_id="run", id=f"entry-{seq}", ts=-10000.0
    )


def folded(*entries):
    view = fold_effects(entries)
    return view, view.get("shop", "refund", "R1")


def declaration(*, contract=None, timeout=20):
    return SimpleNamespace(dedupe=contract, timeout_s=timeout)


def test_q1_uses_recorded_end_even_when_send_could_overrun():
    assert core(ended_at=100, acq_after=999, may_overrun=True, overran=True) == 113


def test_q2_requires_a_later_acquisition_and_no_possible_overrun():
    assert core(acq_after=100) == 113
    assert core() is None
    assert core(acq_after=100, may_overrun=True) is None
    assert core(acq_after=100, overran=True) is None


@pytest.mark.parametrize("bound", [None, 10])
def test_q3_is_immediately_quiescent_even_for_unbounded_send(bound):
    assert core(not_sent=True, applies_within_s=bound, may_overrun=True) == -math.inf


def test_no_application_bound_means_no_automatic_quiescence():
    assert core(ended_at=100, acq_after=200, applies_within_s=None) is None


@pytest.mark.parametrize(
    "change",
    [
        {"m_s": -1},
        {"sigma_s": -1},
        {"applies_within_s": -1},
        {"ended_at": math.nan},
        {"acq_after": math.inf},
        {"m_s": True},
    ],
)
def test_invalid_numeric_evidence_is_refused(change):
    with pytest.raises(ValueError):
        core(**change)


def test_window_strict_boundary_includes_latest_arrival_and_margin():
    args = dict(
        store_now=60.0,
        first_use_at=0.0,
        timeout_s=20.0,
        m_s=2.0,
        sigma_s=1.0,
        window_s=100.0,
        margin_s=17.0,
    )
    assert not q.window_covers(**args)
    assert q.window_covers(**(args | {"store_now": math.nextafter(60.0, -math.inf)}))
    assert not q.window_covers(**(args | {"store_now": 61.0}))


def test_window_that_covers_now_but_not_arrival_refuses_resend():
    assert not q.window_covers(
        store_now=90, first_use_at=0, timeout_s=20, m_s=2, sigma_s=1, window_s=100, margin_s=0
    )


def test_q1_wrapper_uses_journaled_timing_and_end_not_entry_or_current_clock():
    view, op = folded(
        entry(1, "effect_intent", intent()), entry(2, "effect_failed", failed(ended_at=400))
    )
    assert q.quiescence_time(op.attempts[0], view.acquisitions, Timing(90, 20, 50)) == 406


def test_q2_earliest_journaled_takeover_survives_two_recoveries_and_clock_step():
    view, op = folded(entry(1, "effect_intent", intent()))
    view.acquisitions = [(4, 400, 9), (2, 300, 4), (3, 299, 7), (1, 100, 1)]
    timing = Timing(30, 2, 1)
    assert q._acquisition_after(view.acquisitions, 1) == 300
    assert q.quiescence_time(op.attempts[0], view.acquisitions, timing) == 306
    assert q.retry_after(op, view, timing, 305) == 306
    view.acquisitions.append((5, 600, 12))
    assert q.retry_after(op, view, timing, 306) is None
    assert q._lookup_after_quiescence(306, q._all_quiescent(op, view, timing))


def test_q2_current_acquisition_can_supply_the_first_later_epoch():
    view, op = folded(entry(1, "effect_intent", intent()))
    view.acquisitions = [(2, 200, 2)]
    assert q.quiescence_time(op.attempts[0], view.acquisitions, Timing(30, 2, 1)) == 206


def test_late_end_enables_q1_for_overrunning_attempt():
    view, op = folded(
        entry(1, "effect_intent", intent(may_overrun=True)),
        entry(2, "effect_failed", failed(send_ended=False, overran=True, ended_at=None)),
    )
    timing = Timing(30, 2, 1)
    assert q._all_quiescent(op, view, timing) is None
    view.apply(
        entry(
            3,
            "effect_send_ended",
            {
                "record_id": op.record_id,
                "namespace": op.namespace,
                "name": op.name,
                "key": op.key,
                "attempt": 1,
                "intent_seq": 1,
                "ended_at": 150,
                "overran": True,
            },
        )
    )
    assert q._all_quiescent(op, view, timing) == 156


def test_all_attempts_must_be_quiescent_even_if_latest_was_not_sent():
    view, op = folded(
        entry(1, "effect_intent", intent(may_overrun=True)),
        entry(2, "effect_intent", intent(attempt=2)),
        entry(3, "effect_failed", failed(attempt=2, intent_seq=2, why="not_sent")),
    )
    assert q._all_quiescent(op, view, Timing(30, 2, 1)) is None


@pytest.mark.parametrize("policy", ["rejects", "waits", "unknown"])
def test_d3_concurrency_contract_or_evidence_required(policy):
    view, op = folded(entry(1, "effect_intent", intent()))
    effect = declaration(contract=Dedupe(1000, "fixture", concurrent=policy))
    timing = Timing(30, 2, 1)
    assert q._concurrency_permits(effect, op, view, timing, 100) is (policy != "unknown")
    view.acquisitions = [(2, 200, 2)]
    assert not q._concurrency_permits(
        declaration(contract=Dedupe(1000, "fixture")), op, view, timing, 205
    )
    assert q._concurrency_permits(effect, op, view, timing, 206)


def test_d3_unique_contract_has_no_retention_window():
    view, op = folded(entry(1, "effect_intent", intent()))
    effect = declaration(contract=Unique("fixture", (ValueError,)))
    timing = Timing(30, 2, 1)
    assert q._dedupe_window_covers(10**10, op.first_use_at, effect, timing)
    assert q._concurrency_permits(effect, op, view, timing, 10**10)


def test_d3_first_use_does_not_move_with_new_intent():
    view, op = folded(
        entry(1, "effect_intent", intent()),
        entry(2, "effect_intent", intent(attempt=2, store_ts=1000, first_use_at=1000)),
    )
    assert q._window_start(op) == 101
    effect = declaration(contract=Dedupe(1000, "fixture", margin_s=0))
    assert not q._dedupe_window_covers(1090, q._window_start(op), effect, Timing(30, 2, 1))


def test_d4_lookup_start_must_be_at_or_after_quiescence():
    assert not q._lookup_after_quiescence(99, 100)
    assert q._lookup_after_quiescence(100, 100)
    assert q._lookup_after_quiescence(101, 100)
    assert not q._lookup_after_quiescence(100, None)


def test_d5_earliest_requires_all_bound_attempts_or_operator_minimum_age():
    assert (
        q.resend_earliest(qs=[100, 200], last_store_ts=10, has_bound=True, resend_min_age_s=900)
        == 200
    )
    assert (
        q.resend_earliest(qs=[100, None], last_store_ts=10, has_bound=True, resend_min_age_s=900)
        is None
    )
    assert (
        q.resend_earliest(qs=[None], last_store_ts=10, has_bound=False, resend_min_age_s=900) == 910
    )
    assert (
        q.resend_earliest(qs=[-math.inf], last_store_ts=10, has_bound=False, resend_min_age_s=900)
        == -math.inf
    )


def test_d5_wrapper_does_not_override_a_bound_unfinished_attempt_with_minimum_age():
    view, op = folded(entry(1, "effect_intent", intent(may_overrun=True)))
    assert q.earliest_resend(op, view, Timing(30, 2, 1), 900) is None


def test_d5_wrapper_uses_last_attempt_age_for_unbounded_contract():
    first = intent()
    first["timing"]["applies_within_s"] = None
    second = intent(attempt=2, store_ts=150)
    second["timing"]["applies_within_s"] = None
    view, op = folded(entry(1, "effect_intent", first), entry(2, "effect_intent", second))
    assert q.earliest_resend(op, view, Timing(30, 2, 1), 900) == 1050


def test_core_overflow_does_not_make_a_retry_safe():
    assert core(ended_at=1e308, applies_within_s=1e308) is None
    assert not q.window_covers(
        store_now=1e308,
        first_use_at=1e308,
        timeout_s=1e308,
        m_s=0,
        sigma_s=0,
        window_s=1e308,
        margin_s=0,
    )
