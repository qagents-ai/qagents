"""Reliability estimates from repeated evaluations, without model or service calls."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from .errors import ConfigurationError

if TYPE_CHECKING:
    from .evals import CaseResult, EvalReport


class SuccessEstimate(BaseModel):
    """Without-replacement estimates from the observed repeated trials."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)
    k: int
    pass_at_k: float
    pass_power_k: float


class CaseReliability(BaseModel):
    """Counts and estimates for one case; errors count as unsuccessful trials."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)
    case: str
    trials: int
    successes: int
    pass_rate: float
    estimates: tuple[SuccessEstimate, ...]


class ReliabilityReport(BaseModel):
    """Equal-weight means across cases, with the underlying case counts."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)
    dataset: str
    cases: tuple[CaseReliability, ...]
    estimates: tuple[SuccessEstimate, ...]

    def to_markdown(self) -> str:
        """Render both 'at least one' and 'all attempts' success estimates."""
        rows = ["| k | Pass@k | Pass^k |", "| --- | --- | --- |"]
        rows.extend(
            f"| {item.k} | {item.pass_at_k:.1%} | {item.pass_power_k:.1%} |"
            for item in self.estimates
        )
        return "\n".join(rows)


def _trials(report: EvalReport) -> dict[str, list[CaseResult]]:
    if type(report.repeat) is not int or report.repeat < 1 or not report.results:
        raise ConfigurationError("reliability requires a nonempty report with positive repeat")
    groups: dict[str, list[CaseResult]] = defaultdict(list)
    for item in report.results:
        if not item.case.strip() or type(item.repeat) is not int:
            raise ConfigurationError("each trial needs a case name and integer repetition")
        groups[item.case].append(item)
    expected = set(range(report.repeat))
    for name, values in groups.items():
        indices = [item.repeat for item in values]
        if len(indices) != report.repeat or set(indices) != expected:
            raise ConfigurationError(f"case {name!r} has missing or duplicate repetitions")
    return dict(groups)


def _ks(values: Iterable[int]) -> tuple[int, ...]:
    try:
        result = tuple(values)
    except TypeError as error:
        raise ConfigurationError("ks must be an iterable of positive integers") from error
    if not result or any(type(k) is not int or k < 1 for k in result):
        raise ConfigurationError("ks must contain positive integers")
    if len(set(result)) != len(result):
        raise ConfigurationError("ks must not contain duplicates")
    return result


def _estimate(n: int, successes: int, k: int) -> SuccessEstimate:
    denominator = math.comb(n, k)
    miss = math.comb(n - successes, k) if n - successes >= k else 0
    all_pass = math.comb(successes, k) if successes >= k else 0
    return SuccessEstimate(
        k=k, pass_at_k=1 - miss / denominator, pass_power_k=all_pass / denominator
    )


def reliability(report: EvalReport, *, ks: Iterable[int] = (1,)) -> ReliabilityReport:
    """Estimate pass@k and pass^k from complete repeated case results.

    Pass@k is the probability of at least one success among k attempts.
    Pass^k is the probability that all k attempts succeed. These use
    without-replacement combinatorial estimators, then average cases equally.
    Every requested k must be no larger than the number of trials per case.
    Independent, representative repeated attempts are needed to interpret the
    estimates as future success probabilities. This function makes no runs.
    """
    requested = _ks(ks)
    groups = _trials(report)
    if max(requested) > report.repeat:
        raise ConfigurationError("each k must be no larger than report.repeat")
    cases = []
    for name, values in groups.items():
        successes = sum(
            item.passed
            and item.error is None
            and (not report.require_completed or item.status == "completed")
            for item in values
        )
        cases.append(
            CaseReliability(
                case=name,
                trials=len(values),
                successes=successes,
                pass_rate=successes / len(values),
                estimates=tuple(_estimate(len(values), successes, k) for k in requested),
            )
        )
    means = tuple(
        SuccessEstimate(
            k=k,
            pass_at_k=math.fsum(case.estimates[i].pass_at_k for case in cases) / len(cases),
            pass_power_k=math.fsum(case.estimates[i].pass_power_k for case in cases) / len(cases),
        )
        for i, k in enumerate(requested)
    )
    return ReliabilityReport(dataset=report.dataset, cases=tuple(cases), estimates=means)


class PairedComparison(BaseModel):
    """Percentile bootstrap of paired case deltas; cases are the resampling unit."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)
    dataset: str
    baseline: str
    current: str
    metric: Literal["pass_at_k", "pass_power_k"]
    k: int
    mean_delta: float
    lower: float
    upper: float
    confidence: float
    resamples: int
    seed: int
    case_deltas: tuple[tuple[str, float], ...]


def _quantile(values: list[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def compare_reliability(
    baseline: EvalReport,
    current: EvalReport,
    *,
    k: int = 1,
    metric: Literal["pass_at_k", "pass_power_k"] = "pass_power_k",
    confidence: float = 0.95,
    resamples: int = 10000,
    seed: int = 0,
) -> PairedComparison:
    """Compare the same cases using a reproducible paired case bootstrap.

    Require identical dataset names, case/repetition keys, evaluator names,
    completion requirements and case metadata. If metadata.dataset_revision
    is supplied, it must match too. The caller must keep actual task inputs,
    scoring configuration and trial sampling conditions fixed; names alone
    cannot prove those conditions. No evaluations are run by this function.

    Cases, with all their repeated attempts, are resampled together. This
    avoids treating trials of the same task as independent benchmark cases.
    At least two cases and 100 bootstrap resamples are required. Intervals
    describe variation across the supplied cases, not all possible workloads.
    """
    if metric not in ("pass_at_k", "pass_power_k"):
        raise ConfigurationError("metric must be pass_at_k or pass_power_k")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 < confidence < 1
    ):
        raise ConfigurationError("confidence must be between zero and one")
    if type(resamples) is not int or resamples < 100 or type(seed) is not int:
        raise ConfigurationError("resamples must be at least 100 and seed must be an integer")
    if (
        baseline.dataset != current.dataset
        or baseline.repeat != current.repeat
        or baseline.require_completed != current.require_completed
        or set(baseline.evaluators) != set(current.evaluators)
        or baseline.metadata.get("dataset_revision") != current.metadata.get("dataset_revision")
    ):
        raise ConfigurationError("paired reports must describe the same evaluation contract")
    left, right = _trials(baseline), _trials(current)
    if set(left) != set(right) or len(left) < 2:
        raise ConfigurationError("paired reports need the same set of at least two cases")
    for name in left:
        left_metadata = {item.repeat: item.metadata for item in left[name]}
        right_metadata = {item.repeat: item.metadata for item in right[name]}
        if left_metadata != right_metadata:
            raise ConfigurationError(f"paired case {name!r} has different metadata")
    first = reliability(baseline, ks=(k,))
    second = {item.case: item for item in reliability(current, ks=(k,)).cases}
    deltas = tuple(
        (
            item.case,
            getattr(second[item.case].estimates[0], metric) - getattr(item.estimates[0], metric),
        )
        for item in first.cases
    )
    # Sort by case id so serialization/reordering does not change seeded samples.
    deltas = tuple(sorted(deltas))
    values = [value for _, value in deltas]
    rng = random.Random(seed)
    means = sorted(
        math.fsum(rng.choices(values, k=len(values))) / len(values) for _ in range(resamples)
    )
    tail = (1 - confidence) / 2
    return PairedComparison(
        dataset=baseline.dataset,
        baseline=baseline.name,
        current=current.name,
        metric=metric,
        k=k,
        mean_delta=math.fsum(values) / len(values),
        lower=_quantile(means, tail),
        upper=_quantile(means, 1 - tail),
        confidence=confidence,
        resamples=resamples,
        seed=seed,
        case_deltas=deltas,
    )
