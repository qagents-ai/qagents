"""Frozen crash-harness interfaces; behavior is pending its owner stage."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from qagents.effects import HeldEffect

if TYPE_CHECKING:
    from ._coverage import Coverage
    from ._invariants import Finding
    from ._world import PointResult


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

    def explain(self, point: str | None = None) -> str:
        raise NotImplementedError("The crash harness is pending its owning stage.")

    def to_json(self) -> str:
        raise NotImplementedError("The crash harness is pending its owning stage.")

    def to_junit(self) -> str:
        raise NotImplementedError("The crash harness is pending its owning stage.")

    def pin(self, point: str, directory: str | os.PathLike[str]) -> Path:
        raise NotImplementedError("The crash harness is pending its owning stage.")

    async def rerun(self, point: str) -> PointResult:
        raise NotImplementedError("The crash harness is pending its owning stage.")
