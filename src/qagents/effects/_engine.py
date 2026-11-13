"""Frozen Effects interfaces; behavior is implemented by the owning stage."""

from __future__ import annotations

import inspect
import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel

from qagents.errors import ConfigurationError
from qagents.session import EFFECTS_REQUIRED, SessionStore

from ._canonical import LIMIT_BYTES, JSONValue, check_model_numbers, jcs, resolve_namespace
from ._declare import Effect, Unique
from ._fold import Operation
from ._kinds import StatePatchData
from ._ports import FINDINGS_LOGGER, CallScope, EffectJournal, Timing
from ._types import MISSING, Absent, Found, NotYetVisible, RecT, ReqT, Unknown

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
    if effect.on_commit_fn is None:
        return None, None
    try:
        before = json.loads(jcs(dict(state_values)))
        after = deepcopy(before)
        typed_request = effect.request_type.model_validate_json(jcs(request))
        typed_receipt = effect.receipt_type.model_validate_json(jcs(receipt))
        result = effect.on_commit_fn(after, typed_request, typed_receipt)
        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()
            raise TypeError("on_commit must finish synchronously; it returned an awaitable.")
        # Snapshot before diffing: the callback cannot mutate a returned patch later.
        after = json.loads(jcs(after))
        changed = {
            key: value
            for key, value in after.items()
            if key not in before or jcs(value) != jcs(before[key])
        }
        removed = [key for key in before if key not in after]
        return StatePatchData(set=changed, unset=removed), None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


def receipt_json(effect: Effect, receipt: Any) -> tuple[dict[str, JSONValue] | None, str | None]:
    """Validate against receipt_type, dump, jcs, 16 KiB; (json, None) or (None,
    receipt_error)."""
    try:
        # Revalidate even a model_construct() instance or an instance mutated after validation.
        values = (
            receipt.model_dump(mode="python", warnings=False)
            if isinstance(receipt, BaseModel)
            else deepcopy(receipt)
        )
        validated = effect.receipt_type.model_validate(values)
        check_model_numbers(validated.model_dump(mode="python", warnings="error"))
        data = validated.model_dump(mode="json", warnings="error")
        if not isinstance(data, dict):
            raise ValueError("An effect receipt must serialize as a JSON object.")
        encoded = jcs(data)
        if len(encoded) > LIMIT_BYTES:
            raise ValueError(
                f"{effect.name} receipt exceeds the {LIMIT_BYTES}-byte canonical JSON limit."
            )
        # A custom serializer may produce JSON that no longer satisfies the receipt schema.
        effect.receipt_type.model_validate_json(encoded)
        return json.loads(encoded), None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


def check_declarations(
    effects: Sequence[Effect], *, timing: Timing, tool_timeout_s: float | None
) -> list[str]:
    """Build-time checks of 5.1 across declarations: lease TTL, tool timeout, "Unique needs
    a lookup"; returns warnings, raises ConfigurationError. Called by Agent.__init__
    (agent TTL) and AgentRuntime.__init__ (runtime TTL)."""
    if not effects:
        return []
    if tool_timeout_s is not None and (
        isinstance(tool_timeout_s, bool)
        or not isinstance(tool_timeout_s, int | float)
        or not math.isfinite(tool_timeout_s)
        or tool_timeout_s <= 0
    ):
        raise ConfigurationError("tool timeout_s must be positive and finite, or None.")
    for effect in effects:
        if isinstance(effect.dedupe, Unique) and effect.lookup_fn is None:
            raise ConfigurationError(
                f"{effect.name}: Unique needs a lookup; attach @effect.lookup."
            )
        problem = timing.lease_ttl_problem(effect)
        if problem is not None:
            raise ConfigurationError(problem)
    problem, warning = timing.tool_timeout_problem(tool_timeout_s, effects)
    if problem is not None:
        raise ConfigurationError(problem)
    return [warning] if warning is not None else []


async def check_run_start(
    effects: Sequence[Effect], *, store: SessionStore, deps: Any, session_id: str
) -> None:
    """Run-start checks: store features (EFFECTS_REQUIRED), the registry key spec
    (ConfigurationError "payments.refund v1 changed its key; bump version"), and
    _require_claims for every shared namespace (the only claims check; perform calls the
    same function for namespaces resolved from deps). The loop calls check_declarations
    first for effects tools that toolkits resolved only at run start."""
    if not effects:
        return
    missing = EFFECTS_REQUIRED - store.features
    if missing:
        raise ConfigurationError(
            f"{type(store).__name__} lacks required Effects features: "
            f"{', '.join(sorted(missing))}. "
            "Use a store implementing the Effects store contract."
        )
    # Resolve all namespaces before writing any registry entries.
    for effect in effects:
        if callable(effect.namespace) and deps is MISSING:
            raise ConfigurationError(f"{effect.name}: pass deps= to resolve the effect namespace.")
        try:
            namespace = resolve_namespace(effect, deps, session_id)
        except Exception as error:
            raise ConfigurationError(f"{effect.name}: cannot resolve namespace: {error}") from error
        _require_claims(effect, namespace, store)
    for effect in effects:
        outcome = await store.register_effect(
            effect.name, effect.version, effect.key_spec_digest, effect.request_schema_digest
        )
        if outcome == "key_changed":
            raise ConfigurationError(
                f"{effect.name} v{effect.version} changed its key; bump version"
            )
        if outcome == "schema_changed":
            logging.getLogger(FINDINGS_LOGGER).warning(
                "%s v%s changed its request schema; bump version if its meaning changed",
                effect.name,
                effect.version,
            )
        elif outcome not in ("registered", "matches"):
            raise ConfigurationError(
                f"{type(store).__name__} returned an invalid effect registry result."
            )


def _require_claims(effect, namespace, store) -> None:
    """M45: every declared shared namespace needs claims, regardless of its spelling."""
    if effect.namespace != "session" and "claims" not in store.features:
        raise ConfigurationError(
            f"{effect.name}: shared namespace {namespace!r} needs the store claims feature. "
            "Use a claims-capable store or namespace='session'."
        )


def send_running(store: SessionStore, session_id: str, record_id: str) -> bool:
    """True while an abandoned sync send of that operation still runs in this process
    (SEND_STILL_RUNNING: O4a does not re-run the call)."""
    raise NotImplementedError("Effects implementation is pending its owning stage.")


def _forget_store(store: SessionStore) -> None:
    """Drop the single-flight and send-running entries whose root store is this object
    (harness teardown)."""
    raise NotImplementedError("Effects implementation is pending its owning stage.")
