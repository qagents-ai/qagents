"""The normative transition table is ordered, total on valid states, and fail-closed."""

from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import pytest

from qagents.effects import _table as t
from qagents.effects._canonical import Descriptor
from qagents.errors import ConfigurationError


def perform(**changes):
    base = t.PerformSituation(
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        "none",
        "none_declared",
        True,
        True,
        False,
        False,
    )
    return replace(base, **changes)


def rule(state, **changes):
    return t.decide(t.TState(state), t.TEvent.PERFORM, perform(**changes)).rule


@pytest.mark.parametrize("state", list(t.TState))
def test_p0_refuses_disallowed_call_even_when_a_receipt_exists(state):
    assert rule(state, allowed=False) == "P0"


@pytest.mark.parametrize("state", ["C", "R", "D", "NA", "CU", "X"])
def test_divergence_precedes_replay_and_settlements(state):
    assert rule(state, same_call=True, descriptor_differs=True) == "P1"
    assert rule(state, same_call=True, arguments_diverged=True) == "P1"
    assert rule(state, descriptor_differs=True) == "P2"


def test_named_presend_rows():
    assert rule("N", key_changed=True) == "P1k"
    assert rule("C", version_changed=True) == "P3"
    assert rule("R", same_call=True, own_rejection=True, auth_valid=False) == "P3r"
    for state, expected in [("NA", "P4"), ("CU", "P5"), ("X", "P6")]:
        assert rule(state, lookup="not_run") == "NEED_LOOKUP"
        assert rule(state) == expected
    assert rule("D", version_changed=True, lookup="not_run") == "NEED_LOOKUP"
    assert rule("D", version_changed=True, lookup="found") == "P7"


@pytest.mark.parametrize(
    "claim", ["committed_same", "held_elsewhere", "other_descriptor", "rejected", "not_sent"]
)
def test_c1_precedes_first_send_for_foreign_claims(claim):
    assert rule("N", claim=claim) == "C1"


@pytest.mark.parametrize(
    "claim", ["committed_same", "held_elsewhere", "other_descriptor", "rejected"]
)
def test_c1_reads_a_claim_another_session_took_from_a_rejected_record(claim):
    """SPEC 7.1 C1, 7.10: before L2 or D2, but after the call's own rejection (P3r)."""
    assert rule("R", claim=claim, lookup="not_run") == "C1"
    assert rule("R", claim=claim) == "C1"
    assert rule("R", claim=claim, same_call=True, own_rejection=True) == "P3r"
    assert rule("R", claim="own", lookup="not_run") == "NEED_LOOKUP"
    assert rule("R", claim="own") == "D2"


def test_lookup_precedes_resend_and_a_bound_found_never_sends():
    assert rule("D", lookup="not_run", dedupe_valid=True) == "NEED_LOOKUP"
    assert rule("R", lookup="not_run") == "NEED_LOOKUP"
    assert rule("D", lookup="found", dedupe_valid=True) == "L1"
    assert rule("R", lookup="found") == "L2"


def test_only_d_rows_send_and_their_order_is_fixed():
    assert {row.rule for row in t.ROWS if row.sends} == {"D1", "D2", "D3", "D4", "D5"}
    assert rule("N") == "D1"
    assert rule("R") == "D2"
    assert (
        rule("D", dedupe_valid=True, lookup="absent_authoritative_after_q", resend_accepted=True)
        == "D3"
    )
    assert rule("D", lookup="absent_authoritative_after_q", resend_accepted=True) == "D4"
    assert rule("D", resend_accepted=True, auth_valid=False) == "D5"
    assert rule("D") == "H1"


@pytest.mark.parametrize("state", ["N", "R", "D"])
def test_no_send_when_fence_fails(state):
    assert rule(state, fence_ok=False, dedupe_valid=True, resend_accepted=True) == (
        "H1" if state == "D" else "F1"
    )


@pytest.mark.parametrize("state", ["N", "R"])
def test_missing_new_authority_is_a_host_error_not_an_unlisted_send(state):
    with pytest.raises(ConfigurationError, match="authorization"):
        rule(state, auth_valid=False)


def test_expired_original_authority_holds_without_operator_resend():
    assert (
        rule("D", auth_valid=False, dedupe_valid=True, lookup="absent_authoritative_after_q")
        == "H1"
    )


def test_mismatched_lookup_requires_refolding_the_conflict_before_retry():
    with pytest.raises(ConfigurationError, match="refold"):
        rule("D", lookup="found_mismatch", dedupe_valid=True)
    assert rule("X", lookup="found_mismatch") == "P6"


@pytest.mark.parametrize(
    "outcome,expected",
    [
        ("receipt", "E1"),
        ("receipt_unstorable", "E2"),
        ("rejected", "E3"),
        ("duplicate", "E3u"),
        ("other_error", "E4"),
        ("timeout", "E5"),
        ("ended_late", "E5o"),
        ("cancelled", "E6"),
        ("interrupt", "E6"),
        ("fence2_failed", "E7"),
    ],
)
def test_outcome_rows(outcome, expected):
    for resolves, sync in product([False, True], repeat=2):
        row = t.decide(t.TState.D, t.TEvent.OUTCOME, t.OutcomeSituation(outcome, resolves, sync))
        assert row.rule == expected
        assert not row.sends


@pytest.mark.parametrize(
    "state,answer,expected",
    [
        ("D", "recheck", "O1"),
        ("X", "recheck", "O1"),
        ("NA", "recheck", "O1"),
        ("CU", "recheck", "O1"),
        ("D", "committed", "O2"),
        ("X", "committed", "O2"),
        ("D", "not_applied", "O3"),
        ("X", "not_applied", "O3"),
        ("D", "resend", "O3r"),
        ("D", "close_unknown", "O4"),
        ("X", "close_unknown", "O4"),
    ],
)
def test_operator_rows_never_send(state, answer, expected):
    for contradicted, refused, early in product([False, True], repeat=3):
        row = t.decide(
            t.TState(state),
            t.TEvent.OPERATOR,
            t.OperatorSituation(answer, contradicted, refused, early),
        )
        assert row.rule == expected
        assert not row.sends
        if answer != "recheck" and (contradicted or refused or (answer == "resend" and early)):
            assert row.action == "settlement_conflict"


def test_internal_operator_invalid_state_is_not_public_settlement_validation():
    # O5 validates C/revision first and raises public SettlementConflict. The internal
    # table rejects an impossible direct operator event; it never supplies a send row.
    with pytest.raises(ConfigurationError):
        t.decide(
            t.TState.C, t.TEvent.OPERATOR, t.OperatorSituation("committed", False, False, False)
        )


def test_event_and_situation_types_must_agree():
    with pytest.raises(ConfigurationError):
        t.decide(t.TState.D, t.TEvent.OUTCOME, perform())


def test_descriptor_matches_ignores_only_version():
    previous = Descriptor("refund", "1", "destination", "request")
    op = SimpleNamespace(descriptor=previous)
    assert t._descriptor_matches(op, Descriptor("refund", "2", "destination", "request"))
    assert not t._descriptor_matches(op, Descriptor("refund", "1", "other", "request"))
    assert not t._descriptor_matches(op, Descriptor("refund", "1", "destination", "other"))


@pytest.mark.parametrize("status,changed", [("D", True), ("C", True), ("R", False), ("N", False)])
def test_p1k_uses_full_namespace_key_and_operation_id(status, changed):
    old = ("tenant-a", "refund", "R1")
    view = SimpleNamespace(
        by_call={("r1:1:call", "refund", None): old},
        operations={old: SimpleNamespace(state=status)},
    )
    scope = SimpleNamespace(operation_id="r1:1:call")
    effect = SimpleNamespace(name="refund")
    # A callable namespace can resolve to a different tenant on rerun even with the same key.
    assert t._key_changed(view, scope, effect, ("tenant-b", "refund", "R1"), None) is changed
    assert not t._key_changed(view, scope, effect, old, None)
    assert not t._key_changed(
        view, SimpleNamespace(operation_id="r2:1:call"), effect, ("tenant-b", "refund", "R1"), None
    )


class Reads:
    def __init__(self, value):
        self.value = value
        self.fields = set()

    def __getattr__(self, name):
        self.fields.add(name)
        return getattr(self.value, name)


def test_every_condition_reads_only_its_declared_fields():
    assert t.ROWS
    samples = [
        perform(),
        perform(
            lookup="not_run",
            same_call=True,
            descriptor_differs=True,
            key_changed=True,
            version_changed=True,
            own_rejection=True,
            dedupe_valid=True,
            resend_accepted=True,
        ),
    ]
    samples += [
        t.OutcomeSituation(outcome, resolve, sync)
        for outcome in (
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
        )
        for resolve, sync in product([False, True], repeat=2)
    ]
    samples += [
        t.OperatorSituation(answer, bad, refused, early)
        for answer in ("recheck", "committed", "not_applied", "resend", "close_unknown")
        for bad, refused, early in product([False, True], repeat=3)
    ]
    for row in t.ROWS:
        expected = {
            t.TEvent.PERFORM: t.PerformSituation,
            t.TEvent.OUTCOME: t.OutcomeSituation,
            t.TEvent.OPERATOR: t.OperatorSituation,
        }[row.event]
        for sample in samples:
            if isinstance(sample, expected):
                proxy = Reads(sample)
                row.condition(proxy)
                assert proxy.fields <= row.reads, (row.rule, proxy.fields - row.reads)


def valid_internal_perform(state, s):
    """Independent protocol-domain definition, not derived from Row.condition."""
    if state in (t.TState.D, t.TState.R) and s.lookup == "found_mismatch":
        return False  # a mismatch changes the journal state to X before another decision
    if not s.allowed or s.key_changed:
        return True
    if state != t.TState.N and (
        (s.same_call and (s.descriptor_differs or s.arguments_diverged))
        or (not s.same_call and s.descriptor_differs)
    ):
        return True
    if state == t.TState.N:
        return s.claim not in ("none", "own") or not s.fence_ok or s.auth_valid
    if state == t.TState.R:
        return (
            s.own_rejection
            or s.claim not in ("none", "own")
            or s.lookup in ("not_run", "found")
            or not s.fence_ok
            or (not s.same_call and s.auth_valid)
        )
    return True


def test_exhaustive_valid_domains_row_variants_and_declared_reads():
    bools = (False, True)
    domains = {
        "allowed": bools,
        "same_call": bools,
        "descriptor_differs": bools,
        "arguments_diverged": bools,
        "key_changed": bools,
        "version_changed": bools,
        "own_rejection": bools,
        "fence_ok": bools,
        "auth_valid": bools,
        "dedupe_valid": bools,
        "resend_accepted": bools,
        "claim": (
            "none",
            "own",
            "committed_same",
            "held_elsewhere",
            "other_descriptor",
            "rejected",
            "not_sent",
        ),
        "lookup": (
            "not_run",
            "none_declared",
            "found",
            "found_mismatch",
            "not_yet_visible",
            "absent",
            "absent_authoritative_after_q",
            "unknown",
        ),
        "outcome": (
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
        ),
        "rejected_resolves": bools,
        "sync": bools,
        "answer": ("recheck", "committed", "not_applied", "resend", "close_unknown"),
        "contradicted": bools,
        "refused": bools,
        "too_early": bools,
    }
    seen = set()
    read_union = {id(row): set() for row in t.ROWS}
    checked = 0
    for state, event in product(t.TState, t.TEvent):
        applicable = [row for row in t.ROWS if state in row.states and event == row.event]
        fields = sorted(set().union(*(row.reads for row in applicable)))
        baseline = {
            t.TEvent.PERFORM: perform(),
            t.TEvent.OUTCOME: t.OutcomeSituation("receipt", False, False),
            t.TEvent.OPERATOR: t.OperatorSituation("recheck", False, False, False),
        }[event]
        for values in product(*(domains[name] for name in fields)):
            situation = replace(baseline, **dict(zip(fields, values, strict=True)))
            if event == t.TEvent.PERFORM:
                valid = valid_internal_perform(state, situation)
            elif event == t.TEvent.OPERATOR:
                valid = (
                    state == t.TState.D
                    or (state == t.TState.X and situation.answer != "resend")
                    or (state in (t.TState.NA, t.TState.CU) and situation.answer == "recheck")
                )
            else:
                valid = True
            if not valid:
                with pytest.raises(ConfigurationError):
                    t.decide(state, event, situation)
                continue
            selected = t.decide(state, event, situation)
            seen.add(id(selected))
            checked += 1
            for candidate in applicable:
                proxy = Reads(situation)
                candidate.condition(proxy)
                assert proxy.fields <= candidate.reads
                read_union[id(candidate)].update(proxy.fields)
            if selected.sends:
                assert situation.fence_ok
                assert situation.resend_accepted if selected.rule == "D5" else situation.auth_valid
    assert checked > 1000
    assert {id(row) for row in t.ROWS} <= seen
    assert all(read_union[id(row)] == row.reads for row in t.ROWS)
