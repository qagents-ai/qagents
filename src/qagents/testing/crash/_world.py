"""Frozen crash-harness interfaces; behavior is pending its owner stage."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

from qagents.effects import Effect, EffectRecord
from qagents.session import Entry


@dataclass(frozen=True)
class Application:
    fake: str
    idempotency_key: str | None
    payload_digest: str
    attempt: int | None
    epoch: int | None
    leg: int
    arrival: float
    identity: Hashable
    replayed: bool
    key_passed: bool


class World:
    def count_applied(self, effect: Effect, **fields: Any) -> int:
        raise NotImplementedError("The crash harness is pending its owning stage.")

    def applications(self, effect: Effect | None = None) -> list[Application]:
        raise NotImplementedError("The crash harness is pending its owning stage.")

    def service(self, effect: Effect) -> Any:
        raise NotImplementedError("The crash harness is pending its owning stage.")

    journal: list[Entry]
    records: list[EffectRecord]
    outcome: str
