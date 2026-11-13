"""Typed values crossing effect sends, lookups, and operator boundaries."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
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


class HoldReason(StrEnum):
    RESPONSE_LOST = "RESPONSE_LOST"
    DISPATCH_UNKNOWN = "DISPATCH_UNKNOWN"
    REJECTED_WHILE_IN_FLIGHT = "REJECTED_WHILE_IN_FLIGHT"
    SEND_STILL_RUNNING = "SEND_STILL_RUNNING"
    WINDOW_EXPIRED = "WINDOW_EXPIRED"
    NOT_YET_QUIESCENT = "NOT_YET_QUIESCENT"
    NOT_YET_VISIBLE = "NOT_YET_VISIBLE"
    LOOKUP_INCONCLUSIVE = "LOOKUP_INCONCLUSIVE"
    DIVERGED = "DIVERGED"
    KEY_CHANGED = "KEY_CHANGED"
    VERSION_CHANGED = "VERSION_CHANGED"
    APPROVAL_EXPIRED = "APPROVAL_EXPIRED"
    APPROVAL_UNVERIFIABLE = "APPROVAL_UNVERIFIABLE"
    LEASE = "LEASE"
    CONFLICT = "CONFLICT"
    CLAIMED_ELSEWHERE = "CLAIMED_ELSEWHERE"
    CLOSED_BY_INCOMPATIBLE_WORKER = "CLOSED_BY_INCOMPATIBLE_WORKER"


HOLD_DETAIL: Mapping[HoldReason, str] = MappingProxyType(
    {
        HoldReason.RESPONSE_LOST: "Response lost; recheck the destination's outcome.",
        HoldReason.DISPATCH_UNKNOWN: "Dispatch uncertain; recheck before resending.",
        HoldReason.REJECTED_WHILE_IN_FLIGHT: "An earlier attempt may still apply.",
        HoldReason.SEND_STILL_RUNNING: "The previous send has not ended; wait before rechecking.",
        HoldReason.WINDOW_EXPIRED: "Dedupe window expired; automatic resend refused.",
        HoldReason.NOT_YET_QUIESCENT: "An earlier request may still apply; wait.",
        HoldReason.NOT_YET_VISIBLE: "Outcome not visible yet; recheck later.",
        HoldReason.LOOKUP_INCONCLUSIVE: "Lookup inconclusive; recheck or settle with evidence.",
        HoldReason.DIVERGED: "Request or destination changed; restore or settle the original.",
        HoldReason.KEY_CHANGED: "The same call changed its business key; restore the original key.",
        HoldReason.VERSION_CHANGED: "Effect version changed; operator review needed.",
        HoldReason.APPROVAL_EXPIRED: "Approval expired; recheck or use an operator settlement.",
        HoldReason.APPROVAL_UNVERIFIABLE: "Approval unverifiable; obtain a valid decision.",
        HoldReason.LEASE: "The writer lost its lease; a current owner must recover the operation.",
        HoldReason.CONFLICT: "Conflicting journal evidence; inspect before settlement.",
        HoldReason.CLAIMED_ELSEWHERE: "Another session owns this operation; inspect its record.",
        HoldReason.CLOSED_BY_INCOMPATIBLE_WORKER: (
            "A worker without effect support closed this run; operator review is needed."
        ),
    }
)


@dataclass(frozen=True)
class HeldEffect:
    """An unresolved operation surfaced to the host, without a fabricated tool result."""

    id: str
    session_id: str
    effect: str
    key: str
    namespace: str
    reason: HoldReason
    detail: str
    retry_after: datetime | None
    revision: int
    tool_call_id: str | None


@dataclass(frozen=True)
class AttemptSummary:
    """The dispatch history exposed by Session.effects and the operator CLI."""

    number: int
    generation: int
    epoch: int | None
    idempotency_key: str
    store_ts: float
    rule: Literal["D1", "D2", "D3", "D4", "D5"]
    outcome: Literal["committed", "rejected", "unknown", "not_sent"] | None
    may_overrun: bool
    overran: bool
    send_ended: bool


@dataclass(frozen=True)
class SettlementSummary:
    """One recorded operator answer and whether later evidence voided its resend."""

    outcome: Literal["committed", "not_applied", "resend", "close_unknown"]
    by: str
    evidence: str | None
    reason: str | None
    seq: int
    voided: bool


@dataclass(frozen=True)
class EffectRecord:
    """A journal-derived operation record; revision changes only with status-changing entries."""

    id: str
    session_id: str
    namespace: str
    effect: str
    key: str
    status: Literal["committed", "rejected", "held", "not_applied", "closed_unknown", "conflict"]
    via: str | None
    receipt: dict[str, JSONValue] | None
    hold: HeldEffect | None
    attempts: tuple[AttemptSummary, ...]
    settlement: SettlementSummary | None
    revision: int


@dataclass(frozen=True)
class Settle:
    """An asserted operator decision, checked against fresh evidence under the session lease."""

    outcome: Literal["committed", "not_applied", "resend", "close_unknown"]
    record_id: str
    expected_revision: int | None
    receipt: Any | None
    by: str
    evidence: str | None
    reason: str | None

    def __post_init__(self) -> None:
        if self.outcome not in ("committed", "not_applied", "resend", "close_unknown"):
            raise ValueError("Unknown settlement outcome.")
        if not isinstance(self.by, str) or not self.by.strip():
            raise ValueError("A settlement needs the asserted operator identity in by=.")
        proof = self.reason if self.outcome == "close_unknown" else self.evidence
        if not isinstance(proof, str) or not proof.strip():
            raise ValueError("A settlement needs nonempty evidence, or a close_unknown reason.")

    @staticmethod
    def _reference(held: HeldEffect | EffectRecord | str) -> tuple[str, int | None]:
        if isinstance(held, str):
            if not held.strip():
                raise ValueError("A settlement record id must not be empty.")
            return held, None
        if not isinstance(held, HeldEffect | EffectRecord):
            raise TypeError("Settle needs a HeldEffect, EffectRecord, or record id.")
        return held.id, held.revision

    @classmethod
    def committed(
        cls, held: HeldEffect | EffectRecord | str, receipt: Any, *, by: str, evidence: str
    ) -> Settle:
        """Record evidence of an application and its typed receipt."""
        record, revision = cls._reference(held)
        return cls("committed", record, revision, receipt, by, evidence, None)

    @classmethod
    def not_applied(
        cls, held: HeldEffect | EffectRecord | str, *, by: str, evidence: str
    ) -> Settle:
        """Record evidence of no application; this does not authorize a resend."""
        record, revision = cls._reference(held)
        return cls("not_applied", record, revision, None, by, evidence, None)

    @classmethod
    def resend(cls, held: HeldEffect | EffectRecord | str, *, by: str, evidence: str) -> Settle:
        """Request a new attempt after the operator and quiescence checks pass."""
        record, revision = cls._reference(held)
        return cls("resend", record, revision, None, by, evidence, None)

    @classmethod
    def close_unknown(
        cls, held: HeldEffect | EffectRecord | str, *, by: str, reason: str
    ) -> Settle:
        """Stop tracking an outcome that remains unknown, without asserting rejection."""
        record, revision = cls._reference(held)
        return cls("close_unknown", record, revision, None, by, None, reason)
