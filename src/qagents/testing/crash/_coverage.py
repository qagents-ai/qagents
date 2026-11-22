"""Declaration-driven reach requirements and explicit vacuity for crash reports."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from qagents.effects._declare import REACH_NAMES, Dedupe, Effect, ReachPoint, Unique
from qagents.effects._ports import ReachEvent
from qagents.errors import ConfigurationError

_ALWAYS = frozenset({"first_send", "committed", "unknown_outcome", "held"})
_OPERATOR_RULES = frozenset({"O1", "O2", "O3", "O3r", "O4"})


@dataclass(frozen=True)
class Coverage:
    """Counts of distinct crash points reaching each decision, grouped by effect."""

    reached: dict[str, dict[str, int]]
    rules: dict[str, dict[str, int]]
    required: dict[str, frozenset[str]]
    notes: list[str]

    @property
    def missing(self) -> list[str]:
        missing = []
        for effect, requirements in self.required.items():
            for requirement in requirements:
                counts = (
                    self.rules[effect] if requirement.startswith("rule:") else self.reached[effect]
                )
                key = requirement.removeprefix("rule:")
                if counts.get(key, 0) == 0:
                    missing.append(f"{effect}.{requirement}")
        return sorted(missing)

    @property
    def complete(self) -> bool:
        return not self.missing


def build_coverage(
    effects: Sequence[Effect],
    events_by_point: Mapping[str, Sequence[ReachEvent]],
    *,
    must_reach: Sequence[ReachPoint] = (),
    operator_rules: Mapping[str, Sequence[str]] | None = None,
) -> Coverage:
    """Build coverage from actual reach events, with one count per distinct point.

    The driver excludes outside-A3 experiments from this normative coverage. An
    untriggered fault contributes no synthetic reaches. Operator requirements come
    from answers actually supplied, including rules without a named reach point.
    """
    declarations = {}
    required: dict[str, set[str]] = {}
    notes = []
    for effect in effects:
        if effect.name in declarations:
            if declarations[effect.name] is effect:
                continue
            raise ConfigurationError(f"Duplicate effect name in crash coverage: {effect.name!r}.")
        declarations[effect.name] = effect
        required[effect.name] = set(_ALWAYS)
        if isinstance(effect.dedupe, Dedupe):
            required[effect.name].update({"dedupe_resend", "held_after_window"})
        if effect.authoritative:
            if isinstance(effect.dedupe, Unique):
                notes.append(f"{effect.name}: D4 shadowed by Unique")
            else:
                required[effect.name].add("absence_resend")
    for point in must_reach:
        if (
            not isinstance(point, ReachPoint)
            or point.effect not in declarations
            or point.name not in REACH_NAMES
        ):
            raise ConfigurationError(f"Unknown must_reach point: {point!r}.")
        required[point.effect].add(point.name)
    for effect_name, rules in (operator_rules or {}).items():
        if effect_name not in declarations or any(rule not in _OPERATOR_RULES for rule in rules):
            raise ConfigurationError(f"Unknown operator coverage requirement for {effect_name!r}.")
        required[effect_name].update(f"rule:{rule}" for rule in rules)

    reached_counts = {name: Counter() for name in declarations}
    rule_counts = {name: Counter() for name in declarations}
    for events in events_by_point.values():
        reached, rules = set(), set()
        for event in events:
            if event.effect not in declarations:
                raise ConfigurationError(f"Reach event names undeclared effect {event.effect!r}.")
            rules.add((event.effect, event.rule))
            if event.point is not None:
                if event.point not in REACH_NAMES:
                    raise ConfigurationError(f"Unknown emitted reach point {event.point!r}.")
                reached.add((event.effect, event.point))
        for effect_name, point in reached:
            reached_counts[effect_name][point] += 1
        for effect_name, rule in rules:
            rule_counts[effect_name][rule] += 1
    return Coverage(
        reached={
            name: dict(sorted(counts.items())) for name, counts in sorted(reached_counts.items())
        },
        rules={name: dict(sorted(counts.items())) for name, counts in sorted(rule_counts.items())},
        required={name: frozenset(points) for name, points in sorted(required.items())},
        notes=sorted(notes),
    )


def vacuity_reasons(
    *,
    reference_committed: int,
    faults_triggered: int,
    coverage: Coverage,
    invariant_judgments: Sequence[int] = (),
) -> list[str]:
    """Missing exercised evidence cannot make a successful report (SPEC9.6)."""
    counts = (reference_committed, faults_triggered, *invariant_judgments)
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("coverage evidence counts must be nonnegative integers")
    reasons = []
    if reference_committed == 0:
        reasons.append("reference committed no effect")
    if faults_triggered == 0:
        reasons.append("no fault triggered")
    if not coverage.complete:
        reasons.append("required coverage was missed")
    reasons.extend(
        f"user invariant {index} was judged at no point"
        for index, count in enumerate(invariant_judgments)
        if count == 0
    )
    return reasons
