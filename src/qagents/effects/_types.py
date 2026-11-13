"""Typed values crossing effect sends, lookups, and operator boundaries."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Final, Generic, Literal, TypeVar

from pydantic import BaseModel

from qagents.errors import ConfigurationError

from ._canonical import JSONValue

ReqT = TypeVar("ReqT", bound=BaseModel)
RecT = TypeVar("RecT", bound=BaseModel)
DepsT = TypeVar("DepsT")


class _Missing:
    def __repr__(self) -> str:
        return "MISSING"


MISSING: Final[Any] = _Missing()


@dataclass(frozen=True)
class AttemptDebug:
    """Diagnostic attempt identity; these values do not replace the destination key."""

    number: int
    generation: int
    epoch: int | None
    fence_token: int


@dataclass(frozen=True)
class Attempt(Generic[ReqT]):
    """A frozen request and the key, tag, and deadlines of one send."""

    request: ReqT
    key: str
    idempotency_key: str
    operation_tag: str
    deadline: float
    not_after: float | None
    debug: AttemptDebug


@dataclass(frozen=True)
class AttemptInfo:
    """The journaled identity and known outcome of a previous dispatch."""

    number: int
    generation: int
    idempotency_key: str
    epoch: int | None
    store_ts: float
    outcome: Literal["committed", "rejected", "unknown", "not_sent"] | None


@dataclass(frozen=True)
class Inquiry(Generic[ReqT]):
    """The original request and every attempt a lookup must consider."""

    request: ReqT
    key: str
    operation_tag: str
    idempotency_keys: tuple[str, ...]
    attempts: tuple[AttemptInfo, ...]
    first_use_at: float | None


@dataclass(frozen=True)
class Found(Generic[RecT]):
    """A receipt bound to the operation by a destination-read tag or recorded request fields."""

    receipt: RecT
    tag: str | None = field(default=None, kw_only=True)
    seen: Mapping[str, JSONValue] | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.tag is not None and (not isinstance(self.tag, str) or not self.tag):
            raise TypeError("Found tag must be a nonempty string read from the destination.")
        if self.seen is not None and (not isinstance(self.seen, Mapping) or not self.seen):
            raise TypeError("Found seen must contain request fields read from the destination.")
        if self.tag is None and self.seen is None:
            raise TypeError("Found needs tag= or seen= to bind the receipt to the operation.")


@dataclass(frozen=True)
class NotYetVisible:
    """A lookup cannot see the result yet; retry_after_s is only a scheduling hint."""

    retry_after_s: float | None = None


@dataclass(frozen=True)
class Absent:
    """The lookup found no application; only the recovery table can authorize a resend."""


@dataclass(frozen=True)
class Unknown:
    """The lookup could not establish an outcome."""

    reason: str | None = None


class EffectContext(Generic[DepsT]):
    """Trusted dependencies and diagnostics for send/lookup, without mutable session state."""

    __slots__ = ("_deps", "_emit", "logger", "session_id")

    def __init__(
        self,
        deps: DepsT = MISSING,
        *,
        session_id: str | None = None,
        emit: Callable[[Any], None] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._deps = deps
        self.session_id = session_id
        self._emit = emit
        self.logger = logger if logger is not None else logging.getLogger("qagents.effects")

    @property
    def deps(self) -> DepsT:
        """Read supplied dependencies; missing operator dependencies are a configuration error."""
        if self._deps is MISSING:
            raise ConfigurationError(
                "pass deps= to agent.recheck / agent.settle, or --deps to the CLI"
            )
        return self._deps

    def emit(self, event: Any) -> None:
        """Forward an application diagnostic event if an emitter was supplied."""
        if self._emit is not None:
            self._emit(event)


_ATTEMPT: ContextVar[Attempt[Any] | None] = ContextVar("qagents_effect_attempt", default=None)


def current_attempt() -> Attempt[Any] | None:
    """Return the send's Attempt in this task context, or None outside send."""
    return _ATTEMPT.get()
