"""Effect declarations, explicit business keys, and destination contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from qagents.errors import ConfigurationError

from ._canonical import digest, jcs


def _pct(value: str, separator: str = ":") -> str:
    jcs(value)
    return value.replace("%", "%25").replace(separator, f"%{ord(separator):02X}")


@dataclass(frozen=True, init=False)
class Key:
    """A business identity over scalar request fields; use per_call for call-only protection."""

    prefix: str
    fields: tuple[str, ...]
    is_per_call: bool

    def __init__(self, prefix: str, *fields: str) -> None:
        """Render a business key as prefix:v1:v2, escaping delimiters within values."""
        if not isinstance(prefix, str) or not prefix:
            raise ConfigurationError("Key prefix must be a nonempty string.")
        if not fields or any(not isinstance(field, str) or not field for field in fields):
            raise ConfigurationError("Key requires one or more request field names.")
        if len(set(fields)) != len(fields):
            raise ConfigurationError("Key field names must not repeat.")
        jcs(prefix)
        object.__setattr__(self, "prefix", prefix)
        object.__setattr__(self, "fields", tuple(fields))
        object.__setattr__(self, "is_per_call", False)

    @classmethod
    def per_call(cls) -> Key:
        """A call/<operation_id>/<effect>[/<occurrence>] key protects reruns of one call only."""
        result = object.__new__(cls)
        object.__setattr__(result, "prefix", "call")
        object.__setattr__(result, "fields", ())
        object.__setattr__(result, "is_per_call", True)
        return result

    def render(
        self, request: BaseModel, *, effect: str, operation_id: str, occurrence: int | None
    ) -> str:
        """Return an injectively encoded key within this declared key schema."""
        if occurrence is not None and (type(occurrence) is not int or occurrence < 0):
            raise ConfigurationError("occurrence must be a nonnegative integer or None.")
        if self.is_per_call:
            key = f"call/{_pct(operation_id, '/')}/{_pct(effect, '/')}"
            return key if occurrence is None else f"{key}/{occurrence}"
        values = []
        for field in self.fields:
            value = getattr(request, field)
            if type(value) not in (str, int, bool):
                raise ConfigurationError(f"Key field {field!r} must be str, int, or bool.")
            jcs(value)
            text = ("true" if value else "false") if isinstance(value, bool) else str(value)
            values.append(_pct(text))
        return self.prefix + ":" + ":".join(values)

    def spec_digest(self) -> str:
        """Hash the declared field order and key mode for the effect registry."""
        return digest(
            {"prefix": self.prefix, "fields": list(self.fields), "per_call": self.is_per_call}
        )


def _finite_nonnegative(value: float, name: str, *, positive: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        bound = "positive" if positive else "nonnegative"
        raise ConfigurationError(f"{name} must be a finite {bound} number.")


@dataclass(frozen=True)
class Dedupe:
    """A destination's bounded key-retention contract, with pessimistic defaults.

    Concurrency defaults to unknown, errors are presumed replayed, rejection
    replay and parameter binding are not promised. Prefer a probed preset.
    """

    window_s: float
    contract: str
    concurrent: Literal["rejects", "waits", "unknown"] = "unknown"
    replays_errors: bool = True
    replays_rejections: bool = False
    binds_params: bool = False
    margin_s: float | None = None

    def __post_init__(self) -> None:
        _finite_nonnegative(self.window_s, "window_s", positive=True)
        if not isinstance(self.contract, str) or not self.contract.strip():
            raise ConfigurationError(
                "Dedupe contract must name the destination's documented promise."
            )
        if self.concurrent not in ("rejects", "waits", "unknown"):
            raise ConfigurationError("concurrent must be rejects, waits, or unknown.")
        for name in ("replays_errors", "replays_rejections", "binds_params"):
            if type(getattr(self, name)) is not bool:
                raise ConfigurationError(f"{name} must be bool.")
        if self.margin_s is not None:
            _finite_nonnegative(self.margin_s, "margin_s")

    def margin(self) -> float:
        """Return the explicit margin, or max(60 seconds, 5% of the retention window)."""
        return self.margin_s if self.margin_s is not None else max(60.0, 0.05 * self.window_s)


@dataclass(frozen=True)
class Unique:
    """A destination's permanent unique-key promise and its definite duplicate exceptions."""

    contract: str
    on_duplicate: tuple[type[BaseException], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.contract, str) or not self.contract.strip():
            raise ConfigurationError(
                "Unique contract must name the destination's permanent constraint."
            )
        if not isinstance(self.on_duplicate, tuple) or not self.on_duplicate:
            raise ConfigurationError(
                "Unique on_duplicate needs a nonempty tuple of exception types."
            )
        if any(
            not isinstance(kind, type) or not issubclass(kind, BaseException)
            for kind in self.on_duplicate
        ):
            raise ConfigurationError("Unique on_duplicate entries must be exception types.")
