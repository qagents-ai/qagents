"""Frozen crash-harness interfaces; behavior is pending its owner stage."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qagents.effects import Effect
from qagents.models.base import Model
from qagents.session import SessionStore

if TYPE_CHECKING:
    from ._clock import ManualClock


class CrashEnv:
    leg: int
    worker_id: str
    recovering: bool
    clock: ManualClock
    store: SessionStore
    model: Model
    session_id: str
    seed: int
    directory: Path

    def service(
        self,
        effect: Effect,
        impl: type[Any] | Callable[..., Any],
        *,
        key_arg: str = "idempotency_key",
    ) -> Any:
        raise NotImplementedError("The crash harness is pending its owning stage.")
