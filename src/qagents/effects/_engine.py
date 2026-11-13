"""Frozen Effects interfaces; behavior is implemented by the owning stage."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from qagents.session import SessionStore

from ._canonical import JSONValue
from ._declare import Effect
from ._fold import Operation
from ._kinds import StatePatchData
from ._ports import CallScope, EffectJournal, Timing
from ._types import Absent, Found, NotYetVisible, RecT, ReqT, Unknown

Verdict = Found | NotYetVisible | Absent | Unknown


async def perform(
    effect: Effect[ReqT, RecT], request: ReqT, *, scope: CallScope, occurrence: int | None = None
) -> RecT:
    """SPEC 7.2 steps 1-6. Returns the typed receipt; raises EffectRejected,
    EffectConflictError, EffectReceiptError, EffectHeld ("pause"), HeldEffectsError
    ("fail_run"), ToolError(EFFECT_OUTCOME_UNKNOWN) ("tell_model"), ConfigurationError
    (P0), LeaseError (F1: nothing journaled or sent), JournalOutcomeUnknown (poisoned),
    CancelledError (after the shield). Keeps the perform contract of 3.6."""
    raise NotImplementedError("Effects implementation is pending its owning stage.")


async def lookup_and_record(
    effect: Effect,
    op: Operation,
    *,
    journal: EffectJournal,
    deps: Any,
    purpose: Literal["perform", "recheck", "settle"],
    state_values: Callable[[], Mapping[str, JSONValue]],
) -> Verdict:
    """Run the lookup once under timeout_s (exceptions and timeouts: Unknown;
    ConfigurationError propagates), journal effect_observed with started_at, and commit
    via=lookup on a bound Found (on_commit on state_values()); a mismatched Found
    journals effect_conflict found_mismatch; a Found on NA or CU journals
    late_found_after_settlement."""
    raise NotImplementedError("Effects implementation is pending its owning stage.")


def on_commit_patch(
    effect: Effect,
    state_values: Mapping[str, JSONValue],
    request: dict[str, JSONValue],
    receipt: dict[str, JSONValue],
) -> tuple[StatePatchData | None, str | None]:
    """Run on_commit_fn on a deep copy of state_values with the typed request and receipt,
    then diff: (patch, None), or (None, error text) when it raises. The engine passes
    scope.committed_values(); the operator and the CLI pass
    derive_state(entries).values."""
    raise NotImplementedError("Effects implementation is pending its owning stage.")


def receipt_json(effect: Effect, receipt: Any) -> tuple[dict[str, JSONValue] | None, str | None]:
    """Validate against receipt_type, dump, jcs, 16 KiB; (json, None) or (None,
    receipt_error)."""
    raise NotImplementedError("Effects implementation is pending its owning stage.")


def check_declarations(
    effects: Sequence[Effect], *, timing: Timing, tool_timeout_s: float | None
) -> list[str]:
    """Build-time checks of 5.1 across declarations: lease TTL, tool timeout, "Unique needs
    a lookup"; returns warnings, raises ConfigurationError. Called by Agent.__init__
    (agent TTL) and AgentRuntime.__init__ (runtime TTL)."""
    raise NotImplementedError("Effects implementation is pending its owning stage.")


async def check_run_start(
    effects: Sequence[Effect], *, store: SessionStore, deps: Any, session_id: str
) -> None:
    """Run-start checks: store features (EFFECTS_REQUIRED), the registry key spec
    (ConfigurationError "payments.refund v1 changed its key; bump version"), and
    _require_claims for every shared namespace (the only claims check; perform calls the
    same function for namespaces resolved from deps). The loop calls check_declarations
    first for effects tools that toolkits resolved only at run start."""
    raise NotImplementedError("Effects implementation is pending its owning stage.")


def send_running(store: SessionStore, session_id: str, record_id: str) -> bool:
    """True while an abandoned sync send of that operation still runs in this process
    (SEND_STILL_RUNNING: O4a does not re-run the call)."""
    raise NotImplementedError("Effects implementation is pending its owning stage.")


def _forget_store(store: SessionStore) -> None:
    """Drop the single-flight and send-running entries whose root store is this object
    (harness teardown)."""
    raise NotImplementedError("Effects implementation is pending its owning stage.")
