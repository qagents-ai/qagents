"""Small runtime contracts shared by the effect engine, its hosts, and the crash harness."""

from __future__ import annotations

import asyncio
import concurrent.futures
import math
import threading
import weakref
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

from ._canonical import JSONValue
from ._kinds import AuthorizationData, StatePatchData
from ._types import Attempt, EffectContext, HeldEffect, Inquiry

if TYPE_CHECKING:
    from qagents.session import Entry, Lease, SessionStore

    from ._declare import Effect
    from ._fold import EffectView


@dataclass(frozen=True)
class Timing:
    ttl_s: float
    m_s: float
    sigma_s: float
    rho: float = 0.001

    def __post_init__(self) -> None:
        if not math.isfinite(self.ttl_s) or self.ttl_s <= 0:
            raise ValueError("Lease ttl_s must be positive and finite.")
        if any(not math.isfinite(value) or value < 0 for value in (self.m_s, self.sigma_s)):
            raise ValueError("Lease slack and clock tolerance must be finite and nonnegative.")
        if not math.isfinite(self.rho) or not 0 <= self.rho < 1:
            raise ValueError("Lease clock drift rho must be finite and in [0, 1).")

    @classmethod
    def for_lease(cls, ttl_s: float, sigma_s: float, *, effects: bool) -> Timing:
        """Keep short ordinary leases; effect leases need at least two seconds of slack."""
        return cls(ttl_s, max(2.0, ttl_s * 0.1) if effects else ttl_s * 0.1, sigma_s)

    def lease_ttl_problem(self, effect: Effect) -> str | None:
        available = self.ttl_s * (1 - self.rho)
        required = effect.timeout_s + 2 * self.m_s
        if available < required:
            return (
                f"Effect {effect.name!r} timeout_s={effect.timeout_s:g} needs "
                f"lease_ttl_s*(1-rho) >= timeout_s + 2*m = {required:g}s; "
                f"lease_ttl_s={self.ttl_s:g} provides {available:g}s. "
                "Increase lease_ttl_s or reduce the effect timeout_s."
            )
        return None

    def tool_timeout_problem(
        self, tool_timeout_s: float | None, effects: Sequence[Effect]
    ) -> tuple[str | None, str | None]:
        if tool_timeout_s is None or not effects:
            return None, None
        bounds = [2 * effect.timeout_s + 3 * self.m_s for effect in effects]
        largest, total = max(bounds), sum(bounds)
        if tool_timeout_s < largest:
            return (
                f"tool timeout_s={tool_timeout_s:g} is below the required effect bound "
                f"max(2*timeout_s + 3*m)={largest:g}s; increase the tool timeout_s "
                "or reduce effect timeout_s.",
                None,
            )
        if tool_timeout_s < total:
            return None, (
                f"tool timeout_s={tool_timeout_s:g} is below the sequential effects bound "
                f"sum(2*timeout_s + 3*m)={total:g}s."
            )
        return None, None


@dataclass(frozen=True)
class Authorization:
    decision: Literal["allow", "ask", "none", "resend"]
    rule: str | None = None
    approval_id: str | None = None
    expires_at: float | None = None
    verifiable: bool = True
    arguments_digest: str | None = None
    settlement_seq: int | None = None

    def data(self) -> AuthorizationData:
        return AuthorizationData(
            decision=self.decision,
            rule=self.rule,
            approval_id=self.approval_id,
            expires_at=self.expires_at,
            verifiable=self.verifiable,
            arguments_digest=self.arguments_digest,
            settlement_seq=self.settlement_seq,
        )


class EffectJournal(Protocol):
    """The writer operations needed by the engine, without a dependency on the run loop."""

    session_id: str
    store: SessionStore
    lease: Lease
    timing: Timing
    worker: str

    def view(self) -> EffectView: ...
    async def store_now(self) -> float: ...
    async def append_effect(self, entries: Sequence[Entry]) -> list[int]: ...
    def poisoned(self) -> bool: ...
    async def salt(self) -> bytes: ...
    def track(self, fut: asyncio.Future[Any] | concurrent.futures.Future[Any]) -> None: ...


@dataclass
class CallScope:
    """Context and outcome accounting for one tool-call attempt.

    When perform returns or raises (including cancellation), every intent it wrote is in
    exactly one of committed, rejected, or held. The engine's shield maintains that
    contract; the host reads these lists for this attempt and the journal for earlier ones.
    """

    session_id: str
    run_id: str | None
    tool_name: str | None
    tool_call_id: str | None
    operation_id: str
    tool_attempt: int
    effects: frozenset[str]
    authorization: Authorization
    original_authorization: Authorization | None
    arguments_digest: str | None
    original_arguments_digest: str | None
    composition_refusal: str | None
    deps: Any
    emit: Callable[[Any], None]
    journal: EffectJournal
    permissions: Any
    memos: dict[str, JSONValue]
    record_memo: Callable[[str, JSONValue], Awaitable[None]]
    committed_values: Callable[[], Mapping[str, JSONValue]]
    apply_patch: Callable[[StatePatchData], None]
    key_override: str | None = None
    committed: list[tuple[str, str]] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)
    held: list[HeldEffect] = field(default_factory=list)
    occurrences: Counter[str] = field(default_factory=Counter)


class Transport(Protocol):
    async def send(
        self,
        effect: Effect,
        ctx: EffectContext,
        attempt: Attempt,
        call_send: Callable[[], Awaitable[Any]],
    ) -> Any: ...

    async def lookup(
        self,
        effect: Effect,
        ctx: EffectContext,
        inquiry: Inquiry,
        call_lookup: Callable[[], Awaitable[Any]],
    ) -> Any: ...


TRANSPORT: ContextVar[Transport | None] = ContextVar("qagents_effect_transport", default=None)


@dataclass(frozen=True)
class ReachEvent:
    session_id: str
    record_id: str
    effect: str
    rule: str
    point: str | None
    detail: str | None


REACH_OBSERVER: ContextVar[Callable[[ReachEvent], None] | None] = ContextVar(
    "qagents_effect_reach_observer", default=None
)


def reached(event: ReachEvent) -> None:
    """Diagnostics cannot alter the engine's outcome, even when an observer raises."""
    observer = REACH_OBSERVER.get()
    if observer is not None:
        try:
            observer(event)
        except BaseException:
            pass


FINDING_OBSERVER: ContextVar[Callable[[str, str], None] | None] = ContextVar(
    "qagents_effect_finding_observer", default=None
)


def report_finding(kind: str, detail: str) -> None:
    """Expose a diagnostic to a test observer without changing run behavior."""
    observer = FINDING_OBSERVER.get()
    if observer is not None:
        try:
            observer(kind, detail)
        except Exception:
            pass


@dataclass(frozen=True)
class ModelTurn:
    session_id: str
    run_id: str
    turn: int


MODEL_TURN: ContextVar[ModelTurn | None] = ContextVar("qagents_model_turn", default=None)
_INQUIRY: ContextVar[Inquiry[Any] | None] = ContextVar("qagents_effect_inquiry", default=None)


def current_inquiry() -> Inquiry[Any] | None:
    return _INQUIRY.get()


_LIVE_ATTEMPTS: dict[int, weakref.ReferenceType[Attempt[Any]]] = {}
_LIVE_LOCK = threading.Lock()


def _set_attempt_live(attempt: Attempt[Any], live: bool) -> None:
    """Engine-owned lifetime; object identity avoids mutable request hashes and key reuse."""
    identity = id(attempt)

    def forget(ref: weakref.ReferenceType[Attempt[Any]]) -> None:
        with _LIVE_LOCK:
            if _LIVE_ATTEMPTS.get(identity) is ref:
                _LIVE_ATTEMPTS.pop(identity, None)

    with _LIVE_LOCK:
        if live:
            _LIVE_ATTEMPTS[identity] = weakref.ref(attempt, forget)
        else:
            _LIVE_ATTEMPTS.pop(identity, None)


def attempt_live(attempt: Attempt[Any]) -> bool:
    """False before dispatch, after an outcome is journaled, or after abandonment."""
    with _LIVE_LOCK:
        ref = _LIVE_ATTEMPTS.get(id(attempt))
        return ref is not None and ref() is attempt


FINDINGS_LOGGER = "qagents.effects"
