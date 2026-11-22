"""Copied raw observations passed from the harness to independent report checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qagents.effects._ports import ReachEvent
    from qagents.effects._types import HeldEffect

    from ._explore import Point
    from ._invariants import ApplicationEvidence, JournalSnapshot, ModelEvidence
    from ._model import ModelCall
    from ._world import Application, Boundary


@dataclass(frozen=True)
class Execution:
    """Evidence only; the driver copies mutable payloads before returning this record."""

    point: Point
    outcome: str
    legs: int
    final_snapshot: JournalSnapshot | None
    crash_snapshots: tuple[JournalSnapshot, ...]
    applications: tuple[Application, ...]
    application_evidence: tuple[ApplicationEvidence, ...]
    events: tuple[ReachEvent, ...]
    model_calls: tuple[ModelCall, ...]
    model_evidence: tuple[ModelEvidence, ...]
    diagnostics_json: tuple[str, ...]
    held: tuple[HeldEffect, ...]
    triggered: tuple[str, ...]
    not_triggered: tuple[str, ...]
    outside_a3: tuple[str, ...]
    output: Any = None
    boundaries: tuple[Boundary, ...] = ()
    inconclusive_reasons: tuple[str, ...] = ()
    fault_model: tuple[str, ...] = ()
    final_state: Mapping[str, Any] | None = None
    replayable_tools: frozenset[str] = frozenset()
    adapters: tuple[str, ...] = ()
    approval_macs: tuple[tuple[str, bool], ...] = ()
    usage_limits: tuple[tuple[int, int | None, float | None], ...] = ()
