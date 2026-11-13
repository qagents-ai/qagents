"""Frozen Effects interfaces; behavior is implemented by the owning stage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class TState(StrEnum):
    N = "N"
    C = "C"
    R = "R"
    D = "D"
    NA = "NA"
    CU = "CU"
    X = "X"


class TEvent(StrEnum):
    PERFORM = "perform"
    OUTCOME = "outcome"
    OPERATOR = "operator"


@dataclass(frozen=True)
class PerformSituation:
    allowed: bool
    same_call: bool
    descriptor_differs: bool
    arguments_diverged: bool
    key_changed: bool
    version_changed: bool
    own_rejection: bool
    claim: Literal[
        "none",
        "own",
        "committed_same",
        "held_elsewhere",
        "other_descriptor",
        "rejected",
        "not_sent",
    ]
    lookup: Literal[
        "not_run",
        "none_declared",
        "found",
        "found_mismatch",
        "not_yet_visible",
        "absent",
        "absent_authoritative_after_q",
        "unknown",
    ]
    fence_ok: bool
    auth_valid: bool
    dedupe_valid: bool
    resend_accepted: bool


@dataclass(frozen=True)
class OutcomeSituation:
    outcome: Literal[
        "receipt",
        "receipt_unstorable",
        "rejected",
        "duplicate",
        "other_error",
        "timeout",
        "ended_late",
        "cancelled",
        "interrupt",
        "fence2_failed",
    ]
    rejected_resolves: bool
    sync: bool


@dataclass(frozen=True)
class OperatorSituation:
    answer: Literal["recheck", "committed", "not_applied", "resend", "close_unknown"]
    contradicted: bool
    refused: bool
    too_early: bool


Situation = PerformSituation | OutcomeSituation | OperatorSituation


@dataclass(frozen=True)
class Row:
    rule: str
    states: frozenset[TState]
    event: TEvent
    sends: bool
    action: str
    next: str
    reads: frozenset[str]
    condition: Callable[[Situation], bool | None]


ROWS: tuple[Row, ...] = ()
NEED_LOOKUP: Row = Row(
    "NEED_LOOKUP",
    frozenset(),
    TEvent.PERFORM,
    False,
    "lookup",
    "same",
    frozenset(),
    lambda situation: None,
)


def decide(state: TState, event: TEvent, situation: Situation) -> Row:
    """First matching row, or NEED_LOOKUP."""
    raise NotImplementedError("Effects implementation is pending its owning stage.")
