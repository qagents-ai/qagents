"""Deterministic point descriptions derived from observed execution boundaries."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from qagents.effects._declare import Dedupe, Effect, Unique
from qagents.session import Entry

from ._faults import FaultPlan, Faults
from ._world import Boundary

_CRASHES = frozenset(
    {
        "kill_before",
        "kill_after",
        "recovery_kill",
        "commit_then_crash",
        "approval_redelivered",
        "stale_answer",
    }
)
_ZOMBIES = frozenset({"freeze_after_intent", "freeze_before_fence2", "freeze_in_send"})
_MODEL = frozenset({"fresh_ids", "retry_effect", "retry_changed", "permuted", "abandon"})
_EFFECT = frozenset(
    {
        "reject",
        "fail_before_commit",
        "commit_then_lose_response",
        "commit_then_error_response",
        "commit_then_reject_shaped_error",
        "commit_then_crash",
        "commit_late",
        "not_yet_visible_then_commit",
        "unstorable_receipt",
        "duplicate_key",
        "rejected_on",
        "slow_arrival",
        "cancel_during",
    }
)
_STORE = frozenset(
    {
        "ack_lost_committed",
        "ack_lost_uncommitted",
        "stall",
        "cancel_during",
        "lease_ack_lost",
        "lease_renew_refused",
        "lease_error",
    }
)
_OBSERVABLE = frozenset(
    {
        "effect_send",
        "effect_lookup",
        "service_call",
        "application",
        "model_request",
        "model_chunk",
        "thread_start",
        "acquire_lease_grant",
        "acquire_lease",
        "renew_lease",
        "release_lease",
    }
)
_OUTCOME = frozenset({"effect_committed", "effect_failed", "effect_send_ended"})
_LEASES = frozenset({"acquire_lease_grant", "acquire_lease", "renew_lease", "release_lease"})


def _scalar(value: Any) -> bool:
    return (
        value is None
        or type(value) in {str, bool, int}
        or (type(value) is float and math.isfinite(value))
    )


@dataclass(frozen=True)
class Point:
    """A rerunnable injection description; its display id is not executable input."""

    id: str
    plans: tuple[FaultPlan, ...] = ()
    aliases: tuple[str, ...] = ()
    in_effect_window: bool = False
    recovery: bool = False
    variant: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("point id must be nonempty text")
        if not isinstance(self.plans, tuple) or not all(
            isinstance(p, FaultPlan) for p in self.plans
        ):
            raise TypeError("plans must be an immutable tuple of FaultPlans")
        if len({plan.id for plan in self.plans}) != len(self.plans):
            raise ValueError("point fault ids must be distinct")
        if any(
            not _scalar(plan.value)
            and not (isinstance(plan.value, type) and issubclass(plan.value, BaseException))
            for plan in self.plans
        ):
            raise TypeError("fault values must be immutable scalars or exception types")
        if not isinstance(self.aliases, tuple) or not all(isinstance(x, str) for x in self.aliases):
            raise TypeError("point aliases must be immutable text")
        if (
            not isinstance(self.variant, tuple)
            or any(
                not isinstance(pair, tuple)
                or len(pair) != 2
                or not isinstance(pair[0], str)
                or not _scalar(pair[1])
                for pair in self.variant
            )
            or len({key for key, _ in self.variant}) != len(self.variant)
        ):
            raise TypeError("point variant must have unique names and immutable JSON scalar values")
        kinds = [plan.kind for plan in self.plans]
        if sum(kind in _CRASHES for kind in kinds) > 2:
            raise ValueError("v1 point permits at most two crashes")
        if sum(kind in _ZOMBIES for kind in kinds) > 1:
            raise ValueError("v1 point permits at most one zombie")
        if sum(kind in _EFFECT | _STORE for kind in kinds) > 1:
            raise ValueError("v1 point permits at most one effect/store fault")
        if sum(kind in _MODEL for kind in kinds) > 1:
            raise ValueError("one model perturbation may be active in a point")

    @property
    def kinds(self) -> frozenset[str]:
        return frozenset(plan.kind for plan in self.plans)

    def to_data(self) -> dict[str, Any]:
        def value(item: Any) -> Any:
            if _scalar(item):
                return item
            if isinstance(item, type) and issubclass(item, BaseException):
                return {"exception_type": f"{item.__module__}:{item.__qualname__}"}
            raise TypeError("fault values need a stable JSON scalar or exception type")

        return {
            "id": self.id,
            "aliases": list(self.aliases),
            "in_effect_window": self.in_effect_window,
            "recovery": self.recovery,
            "variant": dict(self.variant),
            "plans": [
                {
                    "id": p.id,
                    "kind": p.kind,
                    "index": p.index,
                    "leg": p.leg,
                    "phase": p.phase,
                    "method": p.method,
                    "value": value(p.value),
                }
                for p in self.plans
            ],
        }


@dataclass(frozen=True)
class Selection:
    points: tuple[Point, ...]
    total: int
    sampled: bool
    seed: int
    covered_kinds: frozenset[str]


def _entry_parts(boundary: Boundary) -> tuple[tuple[str, str, str, str], ...]:
    if boundary.entries_json:
        entries = [Entry.model_validate_json(raw) for raw in boundary.entries_json]
        return tuple(
            (
                e.kind,
                str(e.data.get("name", "")),
                str(e.data.get("rule", "")),
                str(e.data.get("via", "")),
            )
            for e in entries
        )
    return tuple(
        (text.partition("/")[0], text.partition("/")[2], "", "")
        for text in boundary.detail.split(",")
        if text
    )


def _label(boundary: Boundary, phase: str) -> str:
    detail = quote(boundary.detail or boundary.kind, safe="/._-,")
    prefix = f"leg{boundary.leg}:" if boundary.leg else ""
    return f"{prefix}{phase}({boundary.index})/{detail}"


def _inside_windows(boundaries: Sequence[Boundary]) -> dict[tuple[int, int], bool]:
    active: dict[int, set[str]] = {}
    result = {}
    for b in boundaries:
        if not b.write or b.phase != "before":
            continue
        current = active.setdefault(b.leg, set())
        parts = _entry_parts(b)
        result[b.leg, b.index] = bool(current) or any(
            kind == "effect_intent" or kind in _OUTCOME for kind, *_ in parts
        )
        identities = [
            str(Entry.model_validate_json(raw).data.get("record_id", "")) for raw in b.entries_json
        ]
        for position, (kind, name, _, _) in enumerate(parts):
            identity = identities[position] if position < len(identities) else ""
            identity = identity or name
            if kind == "effect_intent":
                current.add(identity)
            elif kind in _OUTCOME:
                current.discard(identity)
    return result


def _plan_point(
    b: Boundary,
    kind: str,
    *,
    phase: str = "call",
    value: Any = None,
    suffix: str = "",
    inside: bool = False,
) -> Point:
    identity = (
        _label(b, phase)
        if kind in {"kill_before", "kill_after"}
        else (f"{_label(b, phase)}+{kind}{suffix}")
    )
    return Point(
        identity,
        (FaultPlan(identity, kind, b.index, b.leg, phase, b.kind, value),),
        in_effect_window=inside,
    )


def reference_points(
    boundaries: Sequence[Boundary],
    effects: Sequence[Effect],
    faults: Faults,
    *,
    points: str = "all",
    sigma_s: float = 1.0,
) -> tuple[Point, ...]:
    """Enumerate supported reference faults from events which actually occurred.

    Recovery-only time/delivery/model points are added after those observations,
    never invented from declarations alone.
    """
    if points not in {"all", "all-raw", "effects", "recovery"}:
        raise ValueError("unknown point selection")
    if type(sigma_s) not in {int, float} or not math.isfinite(sigma_s) or sigma_s < 0:
        raise ValueError("sigma_s must be finite and nonnegative")
    declared = {effect.name: effect for effect in effects}
    if len(declared) != len(effects):
        raise ValueError("effect names must be distinct")
    inside = _inside_windows(boundaries)
    writes = [
        (position, b) for position, b in enumerate(boundaries) if b.write and b.phase == "before"
    ]
    made: list[Point] = [Point("reference")]
    before_by_write: dict[tuple[int, int], Point] = {}
    after_by_write: dict[tuple[int, int], Point] = {}
    for _, b in writes:
        parts = _entry_parts(b)
        window = inside[b.leg, b.index]
        for kind, phase, lookup in (
            ("kill_before", "before", before_by_write),
            ("kill_after", "after", after_by_write),
        ):
            if kind in faults.kinds:
                point = _plan_point(b, kind, phase=phase, inside=window)
                lookup[b.leg, b.index] = point
                made.append(point)
        if any(kind == "approval_decided" for kind, *_ in parts):
            for kind in sorted(faults.kinds & {"approval_redelivered", "stale_answer"}):
                made.append(_plan_point(b, kind, phase="after", inside=True))
        for kind in sorted(faults.kinds & {"ack_lost_committed", "ack_lost_uncommitted"}):
            made.append(_plan_point(b, kind, inside=window))
        if any(kind == "effect_intent" or kind in _OUTCOME for kind, *_ in parts):
            for kind in sorted(faults.kinds & {"stall", "cancel_during"}):
                made.append(_plan_point(b, kind, inside=True))
        if (
            any(kind == "effect_intent" for kind, *_ in parts)
            and "freeze_after_intent" in faults.kinds
        ):
            made.append(_plan_point(b, "freeze_after_intent", phase="after", inside=True))
        if b.kind in _LEASES and "lease_ack_lost" in faults.kinds:
            for shape in ("committed", "uncommitted"):
                made.append(
                    _plan_point(
                        b, "lease_ack_lost", value=shape, suffix=f"[{shape}]", inside=window
                    )
                )
        if b.kind == "renew_lease":
            for kind in sorted(faults.kinds & {"lease_error", "lease_renew_refused"}):
                made.append(_plan_point(b, kind, inside=window))
        for kind in sorted(faults.kinds & {"store_step", "store_step_beyond_sigma"}):
            amount = sigma_s if kind == "store_step" else sigma_s + 1
            for delta in sorted({-amount, amount}):
                made.append(
                    _plan_point(
                        b,
                        kind,
                        phase="before",
                        value=delta,
                        suffix=f"[{delta:g}]",
                        inside=window,
                    )
                )

    if points != "all-raw":
        replacements: dict[str, Point] = {}
        removed = set()
        for (_, previous), (next_position, following) in zip(writes, writes[1:], strict=False):
            if previous.leg != following.leg:
                continue
            after = after_by_write.get((previous.leg, previous.index))
            before = before_by_write.get((following.leg, following.index))
            positions = [
                i
                for i, b in enumerate(boundaries[:next_position])
                if b.leg == previous.leg
                and b.index == previous.index
                and b.write
                and b.phase == "after"
            ]
            if after is None or before is None or not positions:
                continue
            between = boundaries[positions[-1] + 1 : next_position]
            if any(b.kind in _OBSERVABLE or b.write for b in between):
                continue
            aliases = (after.id, before.id)
            replacements[after.id] = Point(
                "~".join(aliases),
                after.plans,
                aliases,
                after.in_effect_window or before.in_effect_window,
            )
            removed.add(before.id)
        made = [replacements.get(p.id, p) for p in made if p.id not in removed]

    sends: Counter[tuple[int, str]] = Counter()
    for b in boundaries:
        if b.kind != "effect_send" or b.phase != "before":
            continue
        effect = declared.get(b.detail)
        if effect is None:
            raise ValueError(f"observed send has no effect declaration: {b.detail}")
        sends[b.leg, effect.name] += 1
        index = sends[b.leg, effect.name]
        for kind in sorted(faults.kinds & (_EFFECT | {"freeze_in_send"})):
            if kind == "duplicate_key" and not isinstance(effect.dedupe, Unique):
                continue
            if kind in {"commit_late", "not_yet_visible_then_commit"} and not (
                effect.applies_within_s or isinstance(effect.dedupe, Dedupe)
            ):
                continue
            if (
                kind == "slow_arrival"
                and effect.applies_within_s is not None
                and effect.applies_within_s < effect.timeout_s
            ):
                continue
            values = (
                effect.rejected_on
                if kind == "rejected_on"
                else (("oversized", "non_json") if kind == "unstorable_receipt" else (None,))
            )
            for value in values:
                suffix = (
                    f"[{value.__module__}.{value.__qualname__}]"
                    if isinstance(value, type)
                    else (f"[{value}]" if value else "")
                )
                leg_label = f"leg{b.leg}:" if b.leg else ""
                identity = (
                    f"{leg_label}send({index})/{quote(effect.name, safe='._-')}+{kind}{suffix}"
                )
                made.append(
                    Point(
                        identity,
                        (FaultPlan(identity, kind, index, b.leg, method=effect.name, value=value),),
                        in_effect_window=True,
                    )
                )
    if "freeze_before_fence2" in faults.kinds:
        for b in boundaries:
            if b.kind == "before_second_fence" and b.phase == "before":
                label = f"leg{b.leg}:before_fence({b.index})/{b.detail}"
                made.append(
                    Point(
                        label + "+freeze_before_fence2",
                        (FaultPlan(label, "freeze_before_fence2", b.index, b.leg),),
                        in_effect_window=True,
                    )
                )
    if points == "effects":
        made = [point for point in made if point.id == "reference" or point.in_effect_window]
    # Recovery exploration needs depth-1 prefixes even when the final selection is recovery-only.
    if len({point.id for point in made}) != len(made):
        raise ValueError("observed boundaries produced duplicate point ids")
    return tuple(made)


def recovery_points(
    prefix: Point, reference: Sequence[Boundary], recovered: Sequence[Boundary]
) -> tuple[Point, ...]:
    """Depth-2 writes are selected by observed kind/rule/via, not random entry ids."""
    known = {
        (kind, rule, via) for b in reference if b.write for kind, _, rule, via in _entry_parts(b)
    }
    seen = set()
    made = []
    for b in recovered:
        if not b.write or b.phase != "before" or b.leg != 1:
            continue
        entries = [Entry.model_validate_json(raw) for raw in b.entries_json]
        second_attempt = any(
            e.kind == "effect_intent" and e.data.get("attempt", 0) > 1 for e in entries
        )
        signatures = [
            (kind, rule, via)
            for kind, _, rule, via in _entry_parts(b)
            if (kind, rule, via) not in known or kind == "lease_acquired" or second_attempt
        ]
        for signature in signatures:
            for phase in ("before", "after"):
                key = signature, phase
                if key in seen:
                    continue
                seen.add(key)
                identity = (
                    prefix.id
                    + "+"
                    + _label(b, phase)
                    + "/"
                    + quote("|".join(signature), safe="._-|")
                )
                plan = FaultPlan(identity, "recovery_kill", b.index, 1, phase, b.kind)
                made.append(
                    Point(
                        identity,
                        prefix.plans + (plan,),
                        in_effect_window=True,
                        recovery=True,
                        variant=prefix.variant,
                    )
                )
    return tuple(made)


def select_points(
    points: Sequence[Point], *, budget: int | None = None, seed: int = 0, first: Sequence[str] = ()
) -> Selection:
    """Stable ordering; sampling never silently drops an emitted fault kind."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if isinstance(first, str | bytes) or any(kind not in Faults.KINDS for kind in first):
        raise ValueError("first must contain known fault kinds")
    if len({point.id for point in points}) != len(points):
        raise ValueError("point ids must be distinct")
    references = [point for point in points if point.id == "reference"]
    if len(references) != 1:
        raise ValueError("selection needs exactly one reference")
    if budget is not None and (type(budget) is not int or budget < 1):
        raise ValueError("budget must be a positive integer")
    kinds = frozenset(kind for point in points for kind in point.kinds)
    chosen = list(points)
    if budget is not None and budget < len(chosen):
        rng = random.Random(seed)
        keep = {"reference"}
        covered = set()
        for kind in sorted(kinds):
            if kind in covered:
                continue
            candidate = rng.choice([point for point in chosen if kind in point.kinds])
            keep.add(candidate.id)
            covered.update(candidate.kinds)
        if len(keep) > budget:
            raise ValueError(
                f"budget {budget} cannot preserve reference and emitted fault kinds; "
                f"need {len(keep)}"
            )
        remaining = [point for point in chosen if point.id not in keep]
        keep.update(point.id for point in rng.sample(remaining, budget - len(keep)))
        chosen = [point for point in chosen if point.id in keep]
    order = {kind: index for index, kind in enumerate(first)}
    chosen.sort(
        key=lambda point: (
            -1
            if point.id == "reference"
            else min((order.get(kind, len(order)) for kind in point.kinds), default=len(order))
        )
    )
    return Selection(tuple(chosen), len(points), len(chosen) < len(points), seed, kinds)


def boundary_points(
    prefix: Point, values: Mapping[str, float], *, leg: int = 1
) -> tuple[Point, ...]:
    """Targets are absolute store times consumed by the recovery driver before build."""
    made = []
    for name, value in sorted(values.items()):
        if type(value) not in {int, float} or not math.isfinite(value):
            raise ValueError("time boundaries must be finite numbers")
        for offset in (-0.001, 0.001):
            target = value + offset
            if not math.isfinite(target):
                raise ValueError("time boundary overflow")
            identity = prefix.id + f"+advance_boundaries[{quote(name, safe='._-')}:{offset:+g}]"
            plan = FaultPlan(identity, "advance_boundaries", 1, leg, value=target)
            variant = prefix.variant + (("advance_store_to", target), ("advance_leg", leg))
            made.append(
                Point(
                    identity,
                    prefix.plans + (plan,),
                    in_effect_window=prefix.in_effect_window,
                    recovery=True,
                    variant=variant,
                )
            )
    return tuple(made)


def concurrency_variants(points: Sequence[Point], effects: Sequence[Effect]) -> tuple[Point, ...]:
    """Unknown declarations get both conservative uniform scenarios, explicitly labelled."""
    if not any(
        isinstance(effect.dedupe, Dedupe) and effect.dedupe.concurrent == "unknown"
        for effect in effects
    ):
        return tuple(points)
    made = []
    for point in points:
        if point.id == "reference":
            made.append(point)
            continue
        for behavior in ("rejects", "waits"):
            made.append(
                Point(
                    point.id + f"+concurrent[{behavior}]",
                    point.plans,
                    point.aliases,
                    point.in_effect_window,
                    point.recovery,
                    point.variant + (("unknown_concurrent", behavior),),
                )
            )
    return tuple(made)
