"""Recovery reports over recorded journal and destination observations."""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

from qagents.effects import HeldEffect
from qagents.effects._canonical import jcs

from ._coverage import build_coverage, vacuity_reasons
from ._invariants import (
    Check,
    Finding,
    _read,
    check_applications,
    check_journal,
    check_progress,
)
from ._journal_checks import (
    check_business_outcomes,
    check_decisions,
    check_patches,
    check_receipts,
    check_spend,
    check_starts,
    check_terminal,
    check_usage,
    check_usage_totals,
)
from ._model_checks import check_model_calls

if TYPE_CHECKING:
    from ._coverage import Coverage
    from ._execution import Execution


@dataclass(frozen=True)
class PointResult:
    execution: Execution
    checks: tuple[Check, ...]

    @property
    def id(self) -> str:
        return self.execution.point.id

    @property
    def outcome(self) -> str:
        return self.execution.outcome

    @property
    def verdict(self) -> str:
        if any(item.status == "violated" for item in self.checks):
            return "violated"
        if self.execution.inconclusive_reasons or any(
            item.status == "inconclusive" for item in self.checks
        ):
            return "inconclusive"
        return "safe"


def _point_checks(raw, reference, user, max_legs, expect):
    journal_check = check_journal(raw.final_snapshot, raw.crash_snapshots)
    values = [
        *check_applications(raw.applications, reference.applications, raw.application_evidence),
        journal_check,
        *user,
    ]
    if raw.final_snapshot is not None and journal_check.status == "safe":
        entries = _read(raw.final_snapshot)
        values.extend(
            (
                check_starts(entries, raw.replayable_tools),
                check_decisions(
                    entries, raw.application_evidence, raw.applications, raw.approval_macs
                ),
                check_patches(entries, raw.final_state),
                check_receipts(entries),
                check_terminal(entries, raw.held),
                check_business_outcomes(raw, reference, entries),
                check_usage(entries, raw.model_calls),
                check_usage_totals(entries),
                check_spend(raw),
            )
        )
    values.extend(check_model_calls(raw.model_calls, raw.model_evidence, raw.adapters))
    for encoded in raw.diagnostics_json:
        diagnostic = json.loads(encoded)
        kind = diagnostic.get("kind")
        if kind in {"a2_violation", "post_seal_mutation", "shared_between_legs", "swallowed_hold"}:
            values.append(
                Check(
                    "harness",
                    "violated",
                    (Finding("harness", kind, diagnostic.get("detail", kind)),),
                )
            )
        if kind in {"post_seal_call_uncertain", "isolation_check_incomplete"}:
            values.append(
                Check(
                    "harness",
                    "inconclusive",
                    (Finding("harness", kind, diagnostic.get("detail", kind), "inconclusive"),),
                )
            )
    if not raw.inconclusive_reasons:
        values.append(check_progress(raw.outcome, legs=raw.legs, max_legs=max_legs))
    if expect is not None and raw.outcome not in expect and not raw.inconclusive_reasons:
        values.append(
            Check(
                "expect",
                "violated",
                (
                    Finding(
                        "expect", "unexpected_outcome", f"Outcome {raw.outcome!r} was not expected."
                    ),
                ),
            )
        )
    return tuple(values)


class CrashReport:
    ok: bool
    strict_ok: bool
    vacuous: bool
    coverage: Coverage
    fault_model: list[str]
    failures: list[Finding]
    held: list[HeldEffect]
    points: list[PointResult]
    outcomes: dict[str, int]

    def __init__(
        self,
        executions,
        *,
        effects,
        user_checks,
        must_reach,
        expect,
        allow_held,
        allow_undeclared,
        assume,
        probes,
        undeclared,
        strict,
        max_legs,
        selected,
        requested_faults,
        rerun,
        mode,
        store,
        reproduction=None,
    ):
        self._reproduction = reproduction
        self._reference = executions[0]
        self._rerun = rerun
        self._max_legs, self._expect = max_legs, expect
        self._strict = strict
        self.points = [
            PointResult(
                raw,
                _point_checks(raw, self._reference, user_checks[raw.point.id], max_legs, expect),
            )
            for raw in executions
        ]
        normative = [item for item in self.points if not item.execution.outside_a3]
        exercised = [
            item for item in normative if item.id == "reference" or item.execution.triggered
        ]
        self.coverage = build_coverage(
            effects,
            {item.id: item.execution.events for item in exercised},
            must_reach=must_reach,
        )
        self.failures = [
            failure for item in exercised for check in item.checks for failure in check.findings
        ]
        self.held = [held for item in exercised for held in item.execution.held]
        self.outcomes = dict(Counter(item.outcome for item in exercised))
        reference_committed = (
            sum(item.kind == "effect_committed" for item in _read(self._reference.final_snapshot))
            if self._reference.final_snapshot is not None
            else 0
        )
        judged = [
            sum(
                any(
                    check.invariant == user.invariant and check.status in {"safe", "violated"}
                    for check in item.checks
                )
                for item in exercised
            )
            for user in user_checks["reference"]
        ]
        self.vacuity = vacuity_reasons(
            reference_committed=reference_committed,
            faults_triggered=sum(len(item.execution.triggered) for item in exercised),
            coverage=self.coverage,
            invariant_judgments=judged,
        )
        self.vacuous = bool(self.vacuity)
        self.ok = not self.vacuous and all(item.verdict == "safe" for item in exercised)
        not_exercised = sorted({line for raw in executions for line in raw.fault_model})
        observed_faults = {
            plan.kind
            for item in self.points
            for plan in item.execution.point.plans
            if plan.id in item.execution.triggered
        }
        self.fault_model = [
            *not_exercised,
            f"checked against {len(self.points)} points; {mode} mode, virtual clock, {store} store",
            f"triggered faults: {', '.join(sorted(observed_faults)) or 'none'}",
        ]
        if missing := requested_faults - observed_faults:
            self.fault_model.append("untriggered fault kinds: " + ", ".join(sorted(missing)))
        if selected.sampled:
            self.fault_model.append(
                f"sampled {len(selected.points)} of {selected.total} reference points "
                f"(seed {selected.seed})"
            )
        for effect in effects:
            status = probes.get(effect.name)
            if status is None and {"A2", "A4", "A5"}.issubset(assume.get(effect.name, ())):
                status = "explicitly assumed"
            self.fault_model.append(
                f"{effect.name}: destination assumptions A2/A4/A5; {status or 'probes pending'}"
            )
        self.fault_model.append(f"undeclared side-effect tools: {', '.join(undeclared) or 'none'}")
        self.strict_gaps = list(self.vacuity)
        if selected.sampled:
            self.strict_gaps.append("Fault points were sampled.")
        if missing := requested_faults - observed_faults:
            self.strict_gaps.append("Untriggered fault kinds: " + ", ".join(sorted(missing)))
        for effect in effects:
            if probes.get(effect.name) == "failed" or (
                probes.get(effect.name) != "passed"
                and not {"A2", "A4", "A5"}.issubset(assume.get(effect.name, ()))
            ):
                self.strict_gaps.append(f"{effect.name}: destination assumptions lack evidence.")
        if not_exercised:
            self.strict_gaps.append("one or more send/lookup bodies were bypassed")
        if any(item.effect not in allow_held for item in self.held):
            self.strict_gaps.append("held effects require operator attention")
        if set(undeclared) - set(allow_undeclared):
            self.strict_gaps.append("undeclared side-effect tools")
        self.strict_ok = self.ok and not self.strict_gaps

    def _point(self, point: str) -> PointResult:
        for item in self.points:
            if item.id == point or point in item.execution.point.aliases:
                return item
        raise KeyError(f"Unknown crash point {point!r}.")

    def explain(self, point: str | None = None) -> str:
        lines = list(self.fault_model)
        lines.append(f"ok={self.ok} strict_ok={self.strict_ok} vacuous={self.vacuous}")
        lines.extend(self.vacuity)
        lines.extend("Missing reach: " + name for name in self.coverage.missing)
        for item in [self._point(point)] if point is not None else self.points:
            raw = item.execution
            status = "not triggered" if raw.point.plans and not raw.triggered else item.verdict
            lines.append(f"{item.id}: {status}, {item.outcome} ({raw.legs} legs)")
            if raw.outside_a3:
                lines.append(
                    "  Outside A3; use permanent uniqueness or a destination-enforced expiry."
                )
            lines.extend("  " + issue for issue in raw.inconclusive_reasons)
            lines.extend(
                f"  {finding.invariant}: {finding.message}"
                for check in item.checks
                for finding in check.findings
            )
            if point is not None:
                lines.extend(f"  leg-{b.leg} {b.phase} {b.kind} {b.detail}" for b in raw.boundaries)
        return "\n".join(lines)

    def to_json(self) -> str:
        return jcs(
            {
                "ok": self.ok,
                "strict_ok": self.strict_ok,
                "vacuous": self.vacuous,
                "fault_model": self.fault_model,
                "vacuity": self.vacuity,
                "outcomes": self.outcomes,
                "coverage": {
                    "reached": self.coverage.reached,
                    "rules": self.coverage.rules,
                    "missing": self.coverage.missing,
                    "notes": self.coverage.notes,
                },
                "points": [
                    {
                        "point": item.execution.point.to_data(),
                        "outcome": item.outcome,
                        "verdict": item.verdict,
                        "legs": item.execution.legs,
                        "triggered": list(item.execution.triggered),
                        "not_triggered": list(item.execution.not_triggered),
                        "outside_a3": list(item.execution.outside_a3),
                        "checks": [
                            {
                                "invariant": check.invariant,
                                "status": check.status,
                                "independent": check.independent,
                                "notes": list(check.notes),
                                "findings": [
                                    {
                                        key: value
                                        for key, value in asdict(finding).items()
                                        if key != "entry_id"
                                    }
                                    for finding in check.findings
                                ],
                            }
                            for check in item.checks
                        ],
                        "journal_kinds": [
                            entry.kind for entry in _read(item.execution.final_snapshot)
                        ]
                        if item.execution.final_snapshot is not None
                        else [],
                    }
                    for item in self.points
                ],
            }
        ).decode("utf-8")

    def to_junit(self) -> str:
        root = ET.Element("testsuite", name="qagents.crash")
        failures = skipped = 0
        for point in self.points:
            case = ET.SubElement(root, "testcase", name=point.id, classname="qagents.crash")
            raw = point.execution
            if (
                raw.outside_a3
                or (raw.point.plans and not raw.triggered)
                or point.verdict == "inconclusive"
            ):
                skipped += 1
                ET.SubElement(case, "skipped").text = self.explain(point.id)
            elif point.verdict == "violated":
                failures += 1
                ET.SubElement(case, "failure").text = self.explain(point.id)
        if self.vacuous or (self._strict and not self.strict_ok):
            case = ET.SubElement(root, "testcase", name="coverage", classname="qagents.crash")
            ET.SubElement(case, "failure").text = "\n".join([*self.vacuity, *self.strict_gaps])
            failures += 1
        root.set("tests", str(len(root)))
        root.set("failures", str(failures))
        root.set("skipped", str(skipped))
        return ET.tostring(root, encoding="unicode")

    def pin(self, point: str, directory: str | os.PathLike[str]) -> Path:
        item = self._point(point)
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        path = target / ("point-" + sha256(item.id.encode()).hexdigest()[:12] + ".json")
        portable = self._reproduction is not None and all(
            not isinstance(plan.value, type) for plan in item.execution.point.plans
        )
        payload = {
            "format": "qagents-crash-replay/1" if portable else "qagents-crash-evidence/1",
            "point": item.execution.point.to_data(),
            "verdict": item.verdict,
            "outcome": item.outcome,
            "journal": list(item.execution.final_snapshot.entries_json)
            if item.execution.final_snapshot
            else [],
        }
        if portable:
            payload["reproduction"] = self._reproduction
        encoded = jcs(payload).decode("utf-8")
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
        return path

    async def rerun(self, point: str) -> PointResult:
        raw, user = await self._rerun(self._point(point).execution.point)
        return PointResult(
            raw, _point_checks(raw, self._reference, user, self._max_legs, self._expect)
        )
