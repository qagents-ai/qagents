"""Frozen Effects interfaces; behavior is implemented by the owning stage."""

from __future__ import annotations

from collections.abc import Sequence

from ._fold import AttemptState, EffectView, Operation
from ._ports import Timing


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
    """Q of one attempt (4.4): Q3 not_sent -> -inf; without applies_within_s only Q3; Q1
    ended_at + a + m + sigma; Q2 acq_after + a + m + sigma when neither may_overrun nor
    overran; else None (never quiescent)."""
    raise NotImplementedError("Effects implementation is pending its owning stage.")


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
    """store_now + timeout_s + m + sigma + margin < first_use_at + window_s."""
    raise NotImplementedError("Effects implementation is pending its owning stage.")


def resend_earliest(
    *, qs: Sequence[float | None], last_store_ts: float, has_bound: bool, resend_min_age_s: float
) -> float | None:
    raise NotImplementedError("Effects implementation is pending its owning stage.")


def quiescence_time(
    attempt: AttemptState, acquisitions: Sequence[tuple[int, float, int]], timing: Timing
) -> float | None:
    """q_of with the attempt's journaled timing; acq_after = the earliest journaled
    acquisition with an epoch greater than the attempt's (it may be the current writer's
    own; the latest acquisition is never substituted for it)."""
    raise NotImplementedError("Effects implementation is pending its owning stage.")


def earliest_resend(
    op: Operation, view: EffectView, timing: Timing, resend_min_age_s: float
) -> float | None:
    """D5 and O3r; None = never without new evidence."""
    raise NotImplementedError("Effects implementation is pending its owning stage.")


def retry_after(op: Operation, view: EffectView, timing: Timing, store_now: float) -> float | None:
    raise NotImplementedError("Effects implementation is pending its owning stage.")
