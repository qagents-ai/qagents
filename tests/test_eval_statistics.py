import math

import pytest

from qagents.errors import ConfigurationError
from qagents.eval_statistics import ReliabilityReport, compare_reliability, reliability
from qagents.evals import CaseResult, EvalReport


def report(outcomes, **kwargs):
    return EvalReport(
        repeat=len(next(iter(outcomes.values()))),
        results=[
            CaseResult(case=case, repeat=i, passed=passed, status="completed")
            for case, values in outcomes.items()
            for i, passed in enumerate(values)
        ],
        **kwargs,
    )


def test_combinatorial_success_estimators():
    result = reliability(report({"task": [True, True, True, False, False]}), ks=(1, 2, 3))
    assert result.estimates[0].pass_at_k == pytest.approx(0.6)
    assert result.estimates[1].pass_at_k == pytest.approx(0.9)
    assert result.estimates[1].pass_power_k == pytest.approx(0.3)
    assert result.estimates[2].pass_at_k == 1
    assert result.estimates[2].pass_power_k == pytest.approx(0.1)
    assert result.cases[0].successes == 3
    assert "Pass^k" in result.to_markdown()
    assert ReliabilityReport.model_validate_json(result.model_dump_json()) == result


def test_equal_weight_cases_and_error_status_are_respected():
    source = report({"first": [True, True], "second": [True, False]})
    source.results[0].error = "request failed"
    source.results[1].status = "paused"
    result = reliability(source, ks=(1, 2))
    assert result.cases[0].successes == 0
    assert result.estimates[0].pass_at_k == 0.25
    assert result.estimates[1].pass_power_k == 0


@pytest.mark.parametrize("ks", [(), (0,), (True,), (1.5,), (1, 1), (math.nan,), (3,)])
def test_invalid_sample_requests_fail(ks):
    with pytest.raises(ConfigurationError):
        reliability(report({"a": [True, False]}), ks=ks)


def test_incomplete_duplicate_and_empty_reports_fail():
    source = report({"a": [True, False]})
    source.results.pop()
    with pytest.raises(ConfigurationError, match="missing or duplicate"):
        reliability(source)
    source.results.append(source.results[0])
    with pytest.raises(ConfigurationError, match="missing or duplicate"):
        reliability(source)
    with pytest.raises(ConfigurationError, match="nonempty"):
        reliability(EvalReport())


def test_incomplete_success_only_allowed_when_explicit():
    source = report({"a": [True]}, require_completed=False)
    source.results[0].status = "paused"
    assert reliability(source).cases[0].successes == 1


def test_paired_bootstrap_matches_cases_not_position_and_keeps_global_rng():
    import random

    baseline = report({"a": [False, True], "b": [False, False], "c": [True, True]})
    current = report({"c": [True, True], "b": [True, True], "a": [True, True]})
    before = random.getstate()
    paired = compare_reliability(baseline, current, k=2, resamples=1000, seed=7)
    assert random.getstate() == before
    assert paired.mean_delta == pytest.approx(2 / 3)
    assert paired.case_deltas == (("a", 1), ("b", 1), ("c", 0))
    assert 0 <= paired.lower <= paired.mean_delta <= paired.upper <= 1
    assert compare_reliability(baseline, current, k=2, resamples=1000, seed=7) == paired
    current.results.reverse()
    assert compare_reliability(baseline, current, k=2, resamples=1000, seed=7) == paired


def test_identical_reports_have_zero_deltas_and_all_failures_have_negative_deltas():
    source = report({"a": [True, True], "b": [True, True]})
    same = compare_reliability(source, source)
    assert same.mean_delta == same.lower == same.upper == 0
    failed = report({"a": [False, False], "b": [False, False]})
    worse = compare_reliability(source, failed)
    assert worse.mean_delta == worse.lower == worse.upper == -1


@pytest.mark.parametrize(
    "change",
    [
        {"dataset": "another"},
        {"repeat": 4},
        {"evaluators": ["different-check"]},
        {"require_completed": False},
        {"metadata": {"dataset_revision": "another"}},
    ],
)
def test_mismatched_contracts_are_rejected(change):
    source = report({"a": [True], "b": [False]})
    with pytest.raises(ConfigurationError, match="same evaluation contract"):
        compare_reliability(source, source.model_copy(update=change))


def test_mismatched_case_keys_metadata_and_single_case_are_rejected():
    source = report({"a": [True], "b": [False]})
    other = report({"a": [True], "c": [False]})
    with pytest.raises(ConfigurationError, match="same set"):
        compare_reliability(source, other)
    other = source.model_copy(deep=True)
    other.results[0].metadata = {"difficulty": "changed"}
    with pytest.raises(ConfigurationError, match="different metadata"):
        compare_reliability(source, other)
    with pytest.raises(ConfigurationError, match="at least two"):
        compare_reliability(report({"a": [True]}), report({"a": [True]}))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"resamples": 99},
        {"seed": True},
        {"confidence": 1},
        {"confidence": math.nan},
        {"confidence": True},
        {"metric": "unknown"},
        {"k": True},
    ],
)
def test_invalid_bootstrap_configuration_is_rejected(kwargs):
    source = report({"a": [True], "b": [False]})
    with pytest.raises(ConfigurationError):
        compare_reliability(source, source, **kwargs)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_trial_target_receives_case_and_repetition_with_concurrent_isolation(asynchronous):
    from qagents import Agent, TrialTarget
    from qagents.evals import Case, ExactMatch, evaluate
    from qagents.models.testing import ScriptedModel

    def make(case, repetition):
        if repetition == 1:
            raise RuntimeError("fixture setup failed")
        return Agent(ScriptedModel([str(repetition)]))

    async def async_make(case, repetition):
        return make(case, repetition)

    results = await evaluate(
        TrialTarget(async_make if asynchronous else make),
        [
            Case(name="first", prompt="x", expected="0"),
            Case(name="second", prompt="y", expected="0"),
        ],
        [ExactMatch()],
        repeat=3,
        concurrency=6,
    )
    assert [(r.case, r.repeat) for r in results.results] == [
        ("first", 0),
        ("first", 1),
        ("first", 2),
        ("second", 0),
        ("second", 1),
        ("second", 2),
    ]
    assert [r.output_text for r in results.results] == ["0", "", "2", "0", "", "2"]
    assert results.summary.errors == 2
    assert reliability(results).estimates[0].pass_at_k == pytest.approx(1 / 3)
