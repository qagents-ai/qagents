"""Coverage requires actual reached points, counted once per crash point."""

from types import SimpleNamespace

import pytest

from qagents.effects._declare import Dedupe, ReachPoint, Unique
from qagents.effects._ports import ReachEvent
from qagents.errors import ConfigurationError
from qagents.testing.crash import _coverage as coverage


def effect(*, name="refund", dedupe=None, authoritative=False):
    return SimpleNamespace(name=name, dedupe=dedupe, authoritative=authoritative)


def reach(point, *, rule="D1", name="refund"):
    return ReachEvent("session", "record", name, rule, point, None)


def basic():
    return [
        reach("first_send"),
        reach("committed", rule="E1"),
        reach("unknown_outcome", rule="E4"),
        reach("held", rule="H1"),
    ]


def test_required_coverage_is_counted_by_distinct_point():
    result = coverage.build_coverage(
        [effect()], {"reference": basic(), "crash-1": [reach("held", rule="H1")] * 3}
    )
    assert result.complete
    assert result.reached["refund"]["held"] == 2
    assert result.rules["refund"]["H1"] == 2


def test_dedupe_requires_resend_and_expiry_hold():
    result = coverage.build_coverage(
        [effect(dedupe=Dedupe(100, "fixture"))], {"reference": basic()}
    )
    assert result.missing == ["refund.dedupe_resend", "refund.held_after_window"]
    assert not result.complete


def test_authoritative_requires_absence_resend():
    result = coverage.build_coverage([effect(authoritative=True)], {"reference": basic()})
    assert result.missing == ["refund.absence_resend"]


def test_unique_authoritative_has_explicit_shadow_note():
    result = coverage.build_coverage(
        [effect(dedupe=Unique("fixture", (ValueError,)), authoritative=True)],
        {"reference": basic()},
    )
    assert result.complete
    assert result.notes == ["refund: D4 shadowed by Unique"]
    assert "absence_resend" not in result.required["refund"]


def test_operator_answers_require_their_actual_rules_including_unnamed_reaches():
    result = coverage.build_coverage(
        [effect()],
        {"reference": basic(), "operator": [reach(None, rule="O3")]},
        operator_rules={"refund": ["O3", "O4"]},
    )
    assert result.rules["refund"]["O3"] == 1
    assert result.missing == ["refund.rule:O4"]


def test_must_reach_can_require_shadowed_or_optional_points():
    result = coverage.build_coverage(
        [effect(dedupe=Unique("fixture", (ValueError,)), authoritative=True)],
        {"reference": basic()},
        must_reach=[ReachPoint("refund", "absence_resend")],
    )
    assert result.missing == ["refund.absence_resend"]


@pytest.mark.parametrize(
    "required", [ReachPoint("refund", "typo"), ReachPoint("unknown", "held"), "refund.held"]
)
def test_unknown_must_reach_is_usage_error(required):
    with pytest.raises(ConfigurationError):
        coverage.build_coverage([effect()], {}, must_reach=[required])


def test_unknown_effect_event_is_not_silently_counted_for_another_effect():
    with pytest.raises(ConfigurationError):
        coverage.build_coverage([effect()], {"point": [reach("held", name="other")]})


@pytest.mark.parametrize("rules", [{"other": ["O2"]}, {"refund": ["D1"]}])
def test_operator_rule_requirements_are_validated(rules):
    with pytest.raises(ConfigurationError):
        coverage.build_coverage([effect()], {}, operator_rules=rules)


def test_vacuity_requires_reference_commit_fault_trigger_coverage_and_judged_users():
    complete = coverage.build_coverage([effect()], {"reference": basic()})
    args = dict(
        reference_committed=1, faults_triggered=1, coverage=complete, invariant_judgments=[1, 2]
    )
    assert coverage.vacuity_reasons(**args) == []
    assert coverage.vacuity_reasons(**(args | {"reference_committed": 0})) == [
        "reference committed no effect"
    ]
    assert coverage.vacuity_reasons(**(args | {"faults_triggered": 0})) == ["no fault triggered"]
    assert coverage.vacuity_reasons(**(args | {"invariant_judgments": [1, 0]})) == [
        "user invariant 1 was judged at no point"
    ]
    incomplete = coverage.build_coverage([effect()], {})
    assert "required coverage was missed" in coverage.vacuity_reasons(
        **(args | {"coverage": incomplete})
    )


def test_reports_do_not_depend_on_event_mapping_order():
    a = coverage.build_coverage([effect()], {"a": basic(), "b": [reach("held", rule="H1")]})
    b = coverage.build_coverage(
        [effect()], {"b": [reach("held", rule="H1")], "a": list(reversed(basic()))}
    )
    assert a == b
