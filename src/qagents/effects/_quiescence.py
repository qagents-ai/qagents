"""Retry timing derived from journaled send endings and lease acquisitions."""

from __future__ import annotations

import math
from collections.abc import Sequence

from ._declare import Dedupe, Effect, Unique
from ._fold import AttemptState, EffectView, Operation
from ._ports import Timing


def _number(value: float, name: str, *, duration: bool = False) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result) or (duration and result < 0):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if duration else ""))
    return result


def q_of(
    *,
    ended_at: float | None,
    acq_after: float | None,
    applies_within_s: float | None,
    m_s: float,
    sigma_s: float,
    may_overrun: bool,
    overran: bool,
    not_sent: bool,
) -> float | None:
    """Q3 is immediate; Q1 uses the recorded end; Q2 needs a bounded old worker.

    A known end works even for a send that overran. Without an application bound,
    only not_sent establishes automatic quiescence. None means new evidence is needed.
    """
    margin = _number(m_s, "m_s", duration=True)
    skew = _number(sigma_s, "sigma_s", duration=True)
    ended = _number(ended_at, "ended_at") if ended_at is not None else None
    acquired = _number(acq_after, "acq_after") if acq_after is not None else None
    bound = (
        _number(applies_within_s, "applies_within_s", duration=True)
        if applies_within_s is not None
        else None
    )
    if any(type(flag) is not bool for flag in (may_overrun, overran, not_sent)):
        raise ValueError("quiescence flags must be bool")
    if not_sent:
        return -math.inf
    if bound is None:
        return None
    base = ended if ended is not None else acquired if not may_overrun and not overran else None
    if base is None:
        return None
    result = base + bound + margin + skew
    return result if math.isfinite(result) else None


def window_covers(
    *,
    store_now: float,
    first_use_at: float,
    timeout_s: float,
    m_s: float,
    sigma_s: float,
    window_s: float,
    margin_s: float,
) -> bool:
    """Require the whole latest-arrival bound and margin strictly inside retention."""
    now = _number(store_now, "store_now")
    start = _number(first_use_at, "first_use_at")
    timeout = _number(timeout_s, "timeout_s", duration=True)
    slack = _number(m_s, "m_s", duration=True)
    skew = _number(sigma_s, "sigma_s", duration=True)
    window = _number(window_s, "window_s", duration=True)
    margin = _number(margin_s, "margin_s", duration=True)
    # Sum the difference so intermediate rounding does not erase a boundary-sized
    # gap (e.g. now immediately below the last safe instant). Overflow fails closed.
    try:
        difference = math.fsum((now, timeout, slack, skew, margin, -start, -window))
    except OverflowError:
        return False
    return math.isfinite(difference) and difference < 0


def resend_earliest(
    *, qs: Sequence[float | None], last_store_ts: float, has_bound: bool, resend_min_age_s: float
) -> float | None:
    """D5 uses quiescence, or the operator minimum age for an unbounded contract."""
    last = _number(last_store_ts, "last_store_ts")
    age = _number(resend_min_age_s, "resend_min_age_s", duration=True)
    values = [value for value in qs if value is not None]
    if any(
        value != -math.inf and (type(value) not in (int, float) or not math.isfinite(value))
        for value in values
    ):
        raise ValueError("quiescence times must be finite, -inf, or None")
    if all(value is not None for value in qs):
        return max(values, default=-math.inf)
    if has_bound:
        return None
    earliest = max([last + age, *values])
    return earliest if math.isfinite(earliest) else None


def _acquisition_after(
    acquisitions: Sequence[tuple[int, float, int]], epoch: int | None
) -> float | None:
    """The earliest journaled later epoch is the fixed Q2 bound (M34)."""
    if epoch is None:
        return None
    candidates = [item for item in acquisitions if item[0] > epoch]
    return min(candidates, key=lambda item: item[2])[1] if candidates else None


def quiescence_time(
    attempt: AttemptState, acquisitions: Sequence[tuple[int, float, int]], timing: Timing
) -> float | None:
    """Use this attempt's recorded bounds, unaffected by current worker settings."""
    recorded = attempt.intent.timing
    return q_of(
        ended_at=attempt.send_ended_at,
        acq_after=_acquisition_after(acquisitions, attempt.intent.epoch),
        applies_within_s=recorded.applies_within_s,
        m_s=recorded.m_s,
        sigma_s=recorded.sigma_s,
        may_overrun=attempt.intent.may_overrun,
        overran=attempt.overran,
        not_sent=attempt.terminal == "not_sent",
    )


def _all_quiescent(op: Operation, view: EffectView, timing: Timing) -> float | None:
    """Latest Q across every attempt; one missing bound prevents automatic retry (M2)."""
    values = [quiescence_time(attempt, view.acquisitions, timing) for attempt in op.attempts]
    if any(value is None for value in values):
        return None
    return max(values, default=-math.inf)


def _lookup_after_quiescence(started_at: float, q: float | None) -> bool:
    """Lookup must start after the last possible application, not merely end after it (M37)."""
    started = _number(started_at, "started_at")
    return q is not None and started >= q


def _window_start(op: Operation) -> float:
    """Retention begins at first use and is never extended by a resend (M5)."""
    if op.first_use_at is None:
        raise ValueError("operation has no recorded first_use_at")
    return _number(op.first_use_at, "first_use_at")


def _dedupe_window_covers(
    store_now: float, first_use_at: float, effect: Effect, timing: Timing
) -> bool:
    """Unique keys never expire; bounded keys must cover the latest arrival (M36)."""
    if isinstance(effect.dedupe, Unique):
        return True
    if not isinstance(effect.dedupe, Dedupe):
        return False
    return window_covers(
        store_now=store_now,
        first_use_at=first_use_at,
        timeout_s=effect.timeout_s,
        m_s=timing.m_s,
        sigma_s=timing.sigma_s,
        window_s=effect.dedupe.window_s,
        margin_s=effect.dedupe.margin(),
    )


def _concurrency_permits(
    effect: Effect, op: Operation, view: EffectView, timing: Timing, store_now: float
) -> bool:
    """Concurrent key behavior must be declared or every earlier attempt quiescent (M6)."""
    if isinstance(effect.dedupe, Unique):
        return True
    if not isinstance(effect.dedupe, Dedupe):
        return False
    if effect.dedupe.concurrent in ("rejects", "waits"):
        return True
    quiescent = _all_quiescent(op, view, timing)
    return quiescent is not None and _number(store_now, "store_now") >= quiescent


def earliest_resend(
    op: Operation, view: EffectView, timing: Timing, resend_min_age_s: float
) -> float | None:
    """D5/O3r operator resend, including evidence from all prior attempts."""
    if not op.attempts:
        return -math.inf
    return resend_earliest(
        qs=[quiescence_time(attempt, view.acquisitions, timing) for attempt in op.attempts],
        last_store_ts=max(attempt.intent.store_ts for attempt in op.attempts),
        has_bound=any(
            attempt.intent.timing.applies_within_s is not None for attempt in op.attempts
        ),
        resend_min_age_s=resend_min_age_s,
    )


def retry_after(op: Operation, view: EffectView, timing: Timing, store_now: float) -> float | None:
    """A future evidence bound can schedule recovery; missing evidence cannot."""
    quiescent = _all_quiescent(op, view, timing)
    return (
        quiescent if quiescent is not None and quiescent > _number(store_now, "store_now") else None
    )
