"""Two manually advanced time domains shared by a point's legs and worker threads."""

from __future__ import annotations

import math
import threading


def _finite(value: float, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        converted = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    return converted


class ManualClock:
    """Advance elapsed time together; move the store clock independently for clock faults.

    Args:
        store_time: Initial store time, in seconds.
        monotonic: Initial monotonic time, in seconds.
    """

    def __init__(self, *, store_time: float = 0.0, monotonic: float = 0.0) -> None:
        self._store = _finite(store_time, "store_time")
        self._monotonic = _finite(monotonic, "monotonic")
        self._lock = threading.Lock()

    def store_time(self) -> float:
        """Read the store's clock, including injected clock steps."""
        with self._lock:
            return self._store

    def monotonic(self) -> float:
        """Read elapsed virtual time; store-clock steps do not affect it."""
        with self._lock:
            return self._monotonic

    def advance(self, delta: float) -> None:
        """Advance both domains by a finite nonnegative duration, atomically."""
        duration = _finite(delta, "delta")
        if duration < 0:
            raise ValueError("advance cannot move monotonic time backwards")
        with self._lock:
            store = _finite(self._store + duration, "resulting store time")
            monotonic = _finite(self._monotonic + duration, "resulting monotonic time")
            self._store, self._monotonic = store, monotonic

    def step_store(self, delta: float) -> None:
        """Inject a positive or negative store-clock step without changing elapsed time."""
        duration = _finite(delta, "delta")
        with self._lock:
            self._store = _finite(self._store + duration, "resulting store time")
