"""Ordered Effects protocol decisions; only D1-D5 authorize a send."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from qagents.errors import ConfigurationError

from ._canonical import Descriptor
from ._declare import Effect
from ._fold import EffectView, Operation
from ._ports import CallScope


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


def _descriptor_matches(op: Operation, frozen_descriptor: Descriptor) -> bool:
    """The sole descriptor comparison guard, excluding only version (M3)."""
    return op.descriptor is not None and op.descriptor.same_without_version(frozen_descriptor)


def _key_changed(
    view: EffectView,
    scope: CallScope,
    effect: Effect,
    key: tuple[str, str, str],
    occurrence: int | None,
) -> bool:
    """Bind a call occurrence to its full resolved namespace/effect/key identity (M40)."""
    if not isinstance(key, tuple) or len(key) != 3:
        raise ConfigurationError("Key guard needs the resolved (namespace, effect, key) tuple.")
    prior = view.by_call.get((scope.operation_id, effect.name, occurrence))
    if prior is None or prior == key:
        return False
    previous = view.operations.get(prior)
    if previous is None:
        raise ConfigurationError("Call-key index references an absent effect operation.")
    return previous.state in ("D", "C")


def _row(rule, states, event, action, next_state, reads, condition, *, sends=False):
    return Row(
        rule,
        frozenset(TState) if states == "*" else frozenset(TState(item) for item in states.split()),
        event,
        sends,
        action,
        next_state,
        frozenset(reads.split()),
        condition,
    )


def _lookup_first(situation):
    return None if situation.lookup == "not_run" else True


def _changed_version(situation):
    return _lookup_first(situation) if situation.version_changed else False


def _found(situation):
    return None if situation.lookup == "not_run" else situation.lookup == "found"


P = TEvent.PERFORM
E = TEvent.OUTCOME
OP = TEvent.OPERATOR

# Rows stay in SPEC7.1 order. Variants sharing a rule make its different outcomes
# explicit without adding an authorization route to the five sending rows.
ROWS: tuple[Row, ...] = (
    _row("P0", "*", P, "configuration_error", "same", "allowed", lambda s: not s.allowed),
    _row(
        "P1",
        "C R D NA CU X",
        P,
        "hold_diverged",
        "same",
        "same_call descriptor_differs arguments_diverged",
        lambda s: s.same_call and (s.descriptor_differs or s.arguments_diverged),
    ),
    _row("P1k", "*", P, "hold_key_changed", "same", "key_changed", lambda s: s.key_changed),
    _row(
        "P2",
        "C R D NA CU X",
        P,
        "descriptor_conflict",
        "same",
        "same_call descriptor_differs",
        lambda s: not s.same_call and s.descriptor_differs,
    ),
    _row("P3", "C", P, "replay_receipt", "C", "", lambda s: True),
    _row("P3r", "R", P, "replay_rejection", "R", "own_rejection", lambda s: s.own_rejection),
    _row("P4", "NA", P, "closed_not_applied", "same_or_X", "lookup", _lookup_first),
    _row("P5", "CU", P, "closed_unknown", "same_or_X", "lookup", _lookup_first),
    _row("P6", "X", P, "hold_conflict", "X", "lookup", _lookup_first),
    _row(
        "P7", "D", P, "found_or_hold_version", "C_or_D", "version_changed lookup", _changed_version
    ),
    _row(
        "C1",
        "N R",
        P,
        "resolve_claim",
        "C_or_same",
        "claim",
        lambda s: s.claim not in ("none", "own"),
    ),
    _row("L1", "D", P, "commit_lookup", "C", "lookup", _found),
    _row("L2", "R", P, "commit_lookup", "C", "lookup", _found),
    _row(
        "D1",
        "N",
        P,
        "send_first",
        "outcome",
        "fence_ok auth_valid",
        lambda s: s.fence_ok and s.auth_valid,
        sends=True,
    ),
    _row(
        "D2",
        "R",
        P,
        "send_after_rejection",
        "outcome",
        "same_call fence_ok auth_valid",
        lambda s: not s.same_call and s.fence_ok and s.auth_valid,
        sends=True,
    ),
    _row(
        "D3",
        "D",
        P,
        "send_deduplicated",
        "outcome",
        "fence_ok auth_valid dedupe_valid",
        lambda s: s.fence_ok and s.auth_valid and s.dedupe_valid,
        sends=True,
    ),
    _row(
        "D4",
        "D",
        P,
        "send_after_absence",
        "outcome",
        "fence_ok auth_valid lookup",
        lambda s: s.fence_ok and s.auth_valid and s.lookup == "absent_authoritative_after_q",
        sends=True,
    ),
    _row(
        "D5",
        "D",
        P,
        "send_operator_resend",
        "outcome",
        "fence_ok resend_accepted",
        lambda s: s.fence_ok and s.resend_accepted,
        sends=True,
    ),
    _row("F1", "N R", P, "lease_error", "same", "fence_ok", lambda s: not s.fence_ok),
    _row("H1", "D", P, "hold", "D", "", lambda s: True),
    _row("E1", "*", E, "commit", "C", "outcome", lambda s: s.outcome == "receipt"),
    _row(
        "E2",
        "*",
        E,
        "commit_unstorable",
        "C",
        "outcome",
        lambda s: s.outcome == "receipt_unstorable",
    ),
    _row(
        "E3",
        "*",
        E,
        "record_rejection",
        "R",
        "outcome rejected_resolves",
        lambda s: s.outcome == "rejected" and s.rejected_resolves,
    ),
    _row(
        "E3",
        "*",
        E,
        "hold_rejected_in_flight",
        "D",
        "outcome rejected_resolves",
        lambda s: s.outcome == "rejected" and not s.rejected_resolves,
    ),
    _row(
        "E3u", "*", E, "lookup_duplicate", "C_or_D", "outcome", lambda s: s.outcome == "duplicate"
    ),
    _row(
        "E4",
        "*",
        E,
        "lookup_after_unknown",
        "C_or_D",
        "outcome",
        lambda s: s.outcome == "other_error",
    ),
    _row(
        "E5",
        "*",
        E,
        "abandon_thread",
        "D",
        "outcome sync",
        lambda s: s.outcome == "timeout" and s.sync,
    ),
    _row(
        "E5",
        "*",
        E,
        "cancel_and_bound",
        "D",
        "outcome sync",
        lambda s: s.outcome == "timeout" and not s.sync,
    ),
    _row(
        "E5o",
        "*",
        E,
        "record_late_end",
        "by_outcome",
        "outcome",
        lambda s: s.outcome == "ended_late",
    ),
    _row(
        "E6", "*", E, "shield_then_cancel", "C_or_D", "outcome", lambda s: s.outcome == "cancelled"
    ),
    _row(
        "E6", "*", E, "interrupt_without_write", "D", "outcome", lambda s: s.outcome == "interrupt"
    ),
    _row(
        "E7",
        "*",
        E,
        "record_not_sent",
        "by_fold",
        "outcome",
        lambda s: s.outcome == "fence2_failed",
    ),
    _row("O1", "D X NA CU", OP, "recheck", "C_or_same", "answer", lambda s: s.answer == "recheck"),
    _row(
        "O2",
        "D X",
        OP,
        "settlement_conflict",
        "same",
        "answer contradicted refused",
        lambda s: s.answer == "committed" and (s.contradicted or s.refused),
    ),
    _row("O2", "D X", OP, "settle_committed", "C", "answer", lambda s: s.answer == "committed"),
    _row(
        "O3",
        "D X",
        OP,
        "settlement_conflict",
        "same",
        "answer contradicted refused",
        lambda s: s.answer == "not_applied" and (s.contradicted or s.refused),
    ),
    _row(
        "O3", "D X", OP, "settle_not_applied", "NA", "answer", lambda s: s.answer == "not_applied"
    ),
    _row(
        "O3r",
        "D",
        OP,
        "settlement_conflict",
        "same",
        "answer contradicted refused too_early",
        lambda s: s.answer == "resend" and (s.contradicted or s.refused or s.too_early),
    ),
    _row("O3r", "D", OP, "settle_resend", "D", "answer", lambda s: s.answer == "resend"),
    _row(
        "O4",
        "D X",
        OP,
        "settlement_conflict",
        "same",
        "answer contradicted refused",
        lambda s: s.answer == "close_unknown" and (s.contradicted or s.refused),
    ),
    _row(
        "O4",
        "D X",
        OP,
        "settle_closed_unknown",
        "CU",
        "answer",
        lambda s: s.answer == "close_unknown",
    ),
)
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
    """First matching row or NEED_LOOKUP; invalid internal events fail closed.

    The host validates fresh authorization before N/R can send. O5 validates public
    settlement revisions and committed records with SettlementConflict before it
    calls this internal table. A mismatched lookup must first refold its conflict.
    """
    expected = {
        TEvent.PERFORM: PerformSituation,
        TEvent.OUTCOME: OutcomeSituation,
        TEvent.OPERATOR: OperatorSituation,
    }
    if (
        not isinstance(state, TState)
        or not isinstance(event, TEvent)
        or not isinstance(situation, expected[event])
    ):
        raise ConfigurationError("Effect state, event and situation do not match the protocol.")
    if (
        event == TEvent.PERFORM
        and state in (TState.D, TState.R)
        and situation.lookup == "found_mismatch"
    ):
        raise ConfigurationError("A mismatched lookup must refold the conflict before deciding.")
    for row in ROWS:
        if state in row.states and event == row.event:
            matched = row.condition(situation)
            if matched is None:
                return NEED_LOOKUP
            if matched:
                return row
    raise ConfigurationError(
        "No valid effect transition; check authorization, same-call status and operator state."
    )
