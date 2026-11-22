"""Independent oracles must reject seeded repeats, missing intent and lost durable history."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from qagents.effects._canonical import digest
from qagents.session import Entry
from qagents.testing.crash import _invariants as checks
from qagents.testing.crash._world import Application

from ._samples import intent


def snapshot(*entries, independent=True, session="session"):
    return checks.JournalSnapshot(
        session,
        tuple(e.model_dump_json() for e in entries),
        tuple(e.id for e in entries),
        independent,
    )


def application(**changes):
    values = dict(
        fake="refund",
        idempotency_key="q1_first",
        payload_digest=digest({"amount": 10}),
        attempt=1,
        epoch=1,
        leg=0,
        arrival=100.0,
        identity="order-42",
        replayed=False,
        key_passed=True,
    )
    return Application(**(values | changes))


def proof(app, *, index=0, key="order-42", namespace="shop", record="record", **changes):
    data = intent(
        name=app.fake,
        namespace=namespace,
        key=key,
        record_id=record,
        idempotency_key=app.idempotency_key,
        attempt=app.attempt,
        epoch=app.epoch,
        request={"amount": 10},
        request_digest=app.payload_digest,
        **changes,
    )
    entry = Entry(id="intent", seq=1, kind="effect_intent", data=data)
    return checks.ApplicationEvidence(index, key, record, snapshot(entry))


def results(apps, refs=None, proofs=None):
    if refs is None:
        refs = [application()]
    if proofs is None:
        proofs = [proof(app, index=i) for i, app in enumerate(apps)]
    return {c.invariant: c for c in checks.check_applications(apps, refs, proofs)}


def test_positive_application_is_bound_to_earlier_intent():
    assert all(c.status == "safe" for c in results([application()]).values())


def test_same_key_applied_again_is_not_a_deduplicated_resend():
    bad = results([application(), application(arrival=99999)])
    assert bad["I1a"].status == bad["I1b"].status == "violated"
    good = results([application(), application(replayed=True, arrival=120)])
    assert all(c.status == "safe" for c in good.values())


def test_m43_duplicate_under_a_new_generation_is_counted(monkeypatch):
    apps = [application(), application(idempotency_key="q1_generation2")]
    assert results(apps)["I1b"].status == "violated"
    # The mutation seam is actually consumed, not imported by name or duplicated inline.
    monkeypatch.setattr(
        checks,
        "_count_identities",
        lambda items: {(item.fake, item.identity): 1 for item in items if not item.replayed},
    )
    assert results(apps)["I1b"].status == "safe"


def test_reference_multiset_counts_and_destination_isolation():
    apps = [application(), application(idempotency_key="q1_second")]
    assert results(apps, refs=apps)["I1b"].status == "safe"
    other = application(fake="different-service")
    assert results([other])["I1b"].status == "violated"


def test_different_generation_is_same_operation_but_new_business_key_is_not():
    apps = [application(), application(idempotency_key="q1_generation2")]
    assert results(apps)["I1c"].status == "safe"
    bad = results(apps, proofs=[proof(apps[0]), proof(apps[1], index=1, key="retry-42")])
    assert bad["I1c"].status == "violated"


def test_namespace_is_part_of_operation_identity():
    apps = [application(), application(idempotency_key="other")]
    bad = results(apps, proofs=[proof(apps[0]), proof(apps[1], index=1, namespace="other")])
    assert bad["I1c"].status == "violated"


@pytest.mark.parametrize("changes", [{"key_passed": False}, {"idempotency_key": None}])
def test_missing_or_wrong_key_is_reported_even_without_a_repeat(changes):
    result = results([application(**changes)])["I1a"]
    assert result.status == "violated"
    assert result.findings[0].code == "key_not_passed"


def test_missing_and_duplicate_evidence_are_inconclusive():
    app = application()
    for evidence in ([], [proof(app), proof(app)]):
        outcome = results([app], proofs=evidence)
        assert outcome["I11"].status == outcome["I1c"].status == "inconclusive"


def test_later_journal_intent_cannot_retroactively_validate_application():
    app = application()
    empty_at_issue = checks.ApplicationEvidence(0, "order-42", None, snapshot())
    # Even though a valid intent can be constructed later, only the captured prefix counts.
    assert results([app], proofs=[proof(app)])["I11"].status == "safe"
    bad = results([app], proofs=[empty_at_issue])["I11"]
    assert bad.status == "violated"
    assert bad.findings[0].code == "intent_not_before_application"


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "other"),
        ("idempotency_key", "wrong"),
        ("attempt", 2),
        ("epoch", 2),
        ("epoch", True),
        ("attempt", True),
        ("epoch", None),
    ],
)
def test_application_needs_exact_intent_identity(field, value):
    app = application()
    item = proof(app)
    entry = Entry.model_validate_json(item.snapshot.entries_json[0])
    entry.data[field] = value
    assert (
        results([app], proofs=[replace(item, snapshot=snapshot(entry))])["I11"].status == "violated"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"operation_key": "other"},
        {"record_id": "other"},
    ],
)
def test_capture_metadata_must_agree_with_journal_binding(change):
    app = application()
    item = replace(proof(app), **change)
    assert results([app], proofs=[item])["I11"].status == "violated"


@pytest.mark.parametrize(
    "change",
    [
        {"request": {"amount": 20}},
        {"request_digest": "forged"},
        {"request": {"amount": True}},
    ],
)
def test_payload_and_intent_canonical_digest_must_match(change):
    app = application()
    item = proof(app)
    entry = Entry.model_validate_json(item.snapshot.entries_json[0])
    entry.data.update(change)
    assert (
        results([app], proofs=[replace(item, snapshot=snapshot(entry))])["I11"].status == "violated"
    )


def test_application_digest_cannot_be_forged_alongside_intent_digest():
    app = application(payload_digest="forged")
    assert results([app])["I11"].status == "violated"


def test_snapshot_is_immutable_and_memory_independence_is_labeled():
    app = application()
    item = proof(app)
    entry = Entry.model_validate_json(item.snapshot.entries_json[0])
    entry.data["request"]["amount"] = 500
    item = replace(item, snapshot=replace(item.snapshot, independent_read=False))
    outcome = results([app], proofs=[item])
    assert outcome["I11"].status == "safe"
    assert outcome["I11"].independent is False


def test_snapshot_metadata_corruption_is_not_a_pass():
    app = application()
    item = proof(app)
    item = replace(item, snapshot=replace(item.snapshot, entry_ids=("different",)))
    assert results([app], proofs=[item])["I11"].status == "inconclusive"


def state_entry(seq, value):
    return Entry(id=f"entry-{seq}", seq=seq, kind="state_updated", data={"values": {"x": value}})


def test_every_durable_crash_prefix_survives_final_journal():
    one, two = state_entry(1, "first"), state_entry(2, "second")
    outcome = checks.check_journal(snapshot(one, two), [snapshot(), snapshot(one)])
    assert outcome.status == "safe" and outcome.independent


def test_memory_prefix_is_checked_but_not_labeled_independent():
    outcome = checks.check_journal(snapshot(state_entry(1, 1), independent=False))
    assert outcome.status == "safe" and not outcome.independent
    assert "not an independent read" in outcome.notes[0]


@pytest.mark.parametrize("variant", ["changed", "lost", "other_session"])
def test_durable_history_cannot_be_rewritten_or_lost(variant):
    one = state_entry(1, "first")
    before = snapshot(one)
    final = {
        "changed": snapshot(state_entry(1, "changed")),
        "lost": snapshot(),
        "other_session": snapshot(one, session="other"),
    }[variant]
    assert checks.check_journal(final, [before]).status == "violated"


def test_canonical_prefix_comparison_distinguishes_bool_and_number():
    assert (
        checks.check_journal(snapshot(state_entry(1, 1)), [snapshot(state_entry(1, True))]).status
        == "violated"
    )


@pytest.mark.parametrize("variant", ["duplicate_id", "gap", "zero", "broken_derivation"])
def test_journal_identity_sequence_and_every_prefix_are_validated(variant):
    one, two = state_entry(1, "first"), state_entry(2, "second")
    if variant == "duplicate_id":
        two.id = one.id
    elif variant == "gap":
        two.seq = 3
    elif variant == "zero":
        one.seq = 0
    else:
        one.kind, one.data = "message", {"message": {"role": "invalid"}}
    assert checks.check_journal(snapshot(one, two)).status == "violated"


def test_absent_or_corrupt_durable_capture_is_inconclusive():
    assert checks.check_journal(None).status == "inconclusive"
    corrupt = checks.JournalSnapshot("s", ("not JSON",), ("id",), True)
    assert checks.check_journal(corrupt).status == "inconclusive"
    assert checks.check_journal(snapshot(), [corrupt]).status == "inconclusive"


@pytest.mark.parametrize(
    "outcome",
    sorted(
        checks.OUTCOMES
        - {
            "failed_unexpected",
            "no_terminal_state",
        }
    ),
)
def test_allowed_labeled_outcomes_pass_progress(outcome):
    assert checks.check_progress(outcome, legs=3, max_legs=3).status == "safe"


@pytest.mark.parametrize("outcome", ["failed_unexpected", "no_terminal_state", "", "success"])
def test_failed_or_unknown_outcomes_fail_progress(outcome):
    assert checks.check_progress(outcome, legs=1, max_legs=3).status == "violated"


def test_execution_bound_is_enforced_even_if_eventually_completed():
    assert checks.check_progress("completed_as_reference", legs=4, max_legs=3).status == "violated"


@pytest.mark.parametrize("legs,limit", [(True, 3), (0, 3), (2, 0), (2, 2.5)])
def test_progress_rejects_invalid_bound_evidence(legs, limit):
    with pytest.raises(ValueError):
        checks.check_progress("completed_as_reference", legs=legs, max_legs=limit)


@pytest.mark.parametrize(
    "value,status",
    [
        (True, "safe"),
        (False, "violated"),
        (None, "not_applicable"),
        (1, "inconclusive"),
        ("yes", "inconclusive"),
    ],
)
def test_user_predicates_are_strict_and_none_is_not_a_pass(value, status):
    world = SimpleNamespace(value=value)
    result = checks.check_user_invariant(world, lambda w: w.value, name="business-rule")
    assert result.status == status and not result.independent


def test_broken_user_predicate_is_inconclusive():
    def broken(world):
        raise ValueError("cannot evaluate")

    result = checks.check_user_invariant(SimpleNamespace(), broken, name="broken")
    assert result.status == "inconclusive"
    assert "ValueError" in result.findings[0].message


def test_async_predicate_is_closed_and_not_mistaken_for_true():
    async def predicate(world):
        return True

    result = checks.check_user_invariant(SimpleNamespace(), predicate, name="async")
    assert result.status == "inconclusive"
