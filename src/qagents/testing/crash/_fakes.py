"""Frozen crash-harness interfaces; behavior is pending its owner stage."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import TYPE_CHECKING, Any

from qagents.effects import Effect

if TYPE_CHECKING:
    from ._clock import ManualClock


class Fake:
    def __init__(
        self,
        effect: Effect,
        *,
        identity: Callable[[Any], Hashable] | None = None,
        receipt: Callable[[Any, int], Any] | None = None,
    ) -> None:
        raise NotImplementedError("The crash harness is pending its owning stage.")


class FakeStripeRefunds:
    def __init__(self, *, clock: ManualClock | None = None, keys: bool = True) -> None:
        raise NotImplementedError("Stripe fake is pending the harness implementation.")
