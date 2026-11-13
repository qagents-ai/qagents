"""Frozen Effects interfaces; behavior is implemented by the owning stage."""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import json
import logging
import math
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from qagents import events, prompts
from qagents.errors import (
    ConfigurationError,
    EffectConflictError,
    EffectRejected,
    LeaseError,
    ToolError,
)
from qagents.session import EFFECTS_REQUIRED, Entry, SessionStore

from . import _canonical as canonical
from . import _kinds as kinds
from . import _ports as ports
from . import _table as table
from . import _types as types
from ._canonical import LIMIT_BYTES, JSONValue, check_model_numbers, jcs, resolve_namespace
from ._declare import Dedupe, Effect, Unique
from ._fold import Operation
from ._kinds import StatePatchData
from ._ports import FINDINGS_LOGGER, CallScope, EffectJournal, Timing
from ._types import MISSING, Absent, Found, HoldReason, NotYetVisible, RecT, ReqT, Unknown

Verdict = Found | NotYetVisible | Absent | Unknown
_TOOLERROR_WARNED = False


@dataclass
class _Flight:
    descriptor: canonical.Descriptor
    result: concurrent.futures.Future[Any] = field(default_factory=concurrent.futures.Future)
    task: asyncio.Task[Any] | None = None


@dataclass
class _StoreWork:
    root: SessionStore
    flights: dict[tuple[str, str, str, str], _Flight] = field(default_factory=dict)
    sends: dict[tuple[str, str], set[concurrent.futures.Future[Any]]] = field(default_factory=dict)


_WORK: dict[int, _StoreWork] = {}
_WORK_LOCK = threading.Lock()


def _root(store: SessionStore) -> SessionStore:
    from qagents import agent

    return agent._store_root(store)


def _discard_empty(work: _StoreWork) -> None:
    if not work.flights and not work.sends and _WORK.get(id(work.root)) is work:
        _WORK.pop(id(work.root), None)


async def _shield(awaitable: Awaitable[Any], *, scope: CallScope) -> Any:
    """Complete the bounded operation and its durable accounting before cancellation (M39)."""
    task = asyncio.ensure_future(awaitable)
    scope.journal.track(task)
    cancelled = None
    while True:
        try:
            value = await asyncio.shield(task)
            break
        except asyncio.CancelledError as error:
            if task.cancelled():
                raise
            cancelled = error
            if task.done():
                # Retrieve an exception even when caller cancellation takes precedence.
                task.exception()
                raise cancelled from None
        except BaseException:
            if cancelled is not None:
                raise cancelled from None
            raise
    if cancelled is not None:
        raise cancelled
    return value


async def _single_flight(
    prepared: _Prepared, scope: CallScope, work: Callable[[], Awaitable[Any]]
) -> Any:
    """Share one live result across callers and event loops using the same store object."""
    root = _root(scope.journal.store)
    key = (scope.session_id, prepared.namespace, prepared.descriptor.effect, prepared.key)
    with _WORK_LOCK:
        store_work = _WORK.setdefault(id(root), _StoreWork(root))
        flight = store_work.flights.get(key)
        owner = flight is None
        if owner:
            flight = _Flight(prepared.descriptor)
            store_work.flights[key] = flight
        elif flight.descriptor != prepared.descriptor:
            raise EffectConflictError(
                prompts.EFFECT_CONFLICT.format(
                    effect=prepared.descriptor.effect, key=prepared.key, status="held"
                ),
                effect=prepared.descriptor.effect,
                key=prepared.key,
                record_id=canonical.record_id(*key[1:]),
                existing_status="held",
            )

    async def run() -> None:
        try:
            flight.result.set_result(await work())
        except BaseException as error:
            flight.result.set_exception(error)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
        finally:
            with _WORK_LOCK:
                if store_work.flights.get(key) is flight:
                    store_work.flights.pop(key)
                _discard_empty(store_work)

    if owner:
        flight.task = asyncio.create_task(run())
        scope.journal.track(flight.task)
    return await _shield(asyncio.wrap_future(flight.result), scope=scope)


def _track_abandoned(
    store: SessionStore,
    session_id: str,
    record_id: str,
    future: concurrent.futures.Future[Any],
) -> None:
    root = _root(store)
    key = (session_id, record_id)
    with _WORK_LOCK:
        store_work = _WORK.setdefault(id(root), _StoreWork(root))
        store_work.sends.setdefault(key, set()).add(future)

    def ended(done: concurrent.futures.Future[Any]) -> None:
        with _WORK_LOCK:
            active = store_work.sends.get(key)
            if active is not None:
                active.discard(done)
                if not active:
                    store_work.sends.pop(key)
            _discard_empty(store_work)

    future.add_done_callback(ended)


@dataclass(frozen=True)
class _Prepared:
    frozen: canonical.Frozen
    descriptor: canonical.Descriptor
    namespace: str
    key: str
    occurrence: int | None
    destination: JSONValue
    original: BaseModel


_DISPATCH_INPUT: ContextVar[_Prepared | None] = ContextVar("qagents_dispatch_input", default=None)


@dataclass(frozen=True)
class Outcome:
    kind: Literal["committed", "rejected", "held", "not_sent"]
    receipt: dict[str, JSONValue] | None = None
    receipt_error: str | None = None
    error: BaseException | None = None
    reason: HoldReason | None = None
    error_kind: str | None = None


class _DispatchNotSent(Exception):
    """The callback was still queued when its original send deadline expired."""


def _reference(op: Operation) -> dict[str, str]:
    return dict(namespace=op.namespace, name=op.name, key=op.key, record_id=op.record_id)


def _reach(scope: CallScope, op: Operation, rule: str, point: str | None = None) -> None:
    ports.reached(ports.ReachEvent(scope.session_id, op.record_id, op.name, rule, point, None))


async def _append_outcome(
    scope: CallScope, kind: str, payload: BaseModel, *, receipt: dict[str, JSONValue] | None = None
) -> None:
    try:
        await scope.journal.append_effect(
            [Entry(kind=kind, data=payload.model_dump(mode="json"), run_id=scope.run_id)]
        )
    except BaseException:
        if kind == kinds.KIND_EFFECT_COMMITTED:
            logging.getLogger(FINDINGS_LOGGER).error(
                "Effect commit could not be journaled: effect=%s key=%s receipt=%r",
                payload.name,
                payload.key,
                receipt,
            )
        raise


async def _finish_send(
    effect: Effect,
    op: Operation,
    scope: CallScope,
    number: int,
    intent_seq: int,
    rule: str,
    *,
    result: Any = None,
    error: BaseException | None = None,
    timeout: bool = False,
    ended: bool = True,
    overran: bool = False,
) -> Outcome:
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        raise error
    ended_at = await scope.journal.store_now() if ended else None
    ref = _reference(op)
    if isinstance(error, _DispatchNotSent):
        await _append_outcome(
            scope,
            kinds.KIND_EFFECT_FAILED,
            kinds.EffectFailedData(
                **ref,
                attempt=number,
                intent_seq=intent_seq,
                committed=False,
                why="not_sent",
                send_ended=True,
                ended_at=ended_at,
                overran=overran,
            ),
        )
        _reach(scope, op, "E7")
        return Outcome("not_sent", reason=HoldReason.LEASE)
    if error is None and not timeout:
        receipt, receipt_error = receipt_json(effect, result)
        identity = None
        patch, patch_error = None, None
        if receipt is not None:
            try:
                identity = canonical.identity_digest(receipt, effect.receipt_identity)
            except Exception as identity_error:
                receipt, receipt_error = None, str(identity_error)
            else:
                # Lookup or an operator may have committed while this send ran.
                # Its patch has already been applied and must not run a second time.
                if op.commit_seq is None and op.state != "X":
                    patch, patch_error = on_commit_patch(
                        effect, scope.committed_values(), op.request, receipt
                    )
        outcome = "receipt" if receipt_error is None else "receipt_unstorable"
        row = table.decide(
            table.TState(op.state),
            table.TEvent.OUTCOME,
            table.OutcomeSituation(outcome, False, not effect.is_async),
        )
        payload = kinds.EffectCommittedData(
            **ref,
            attempt=number,
            intent_seq=intent_seq,
            via="dedupe" if rule == "D3" else "resend" if rule in ("D4", "D5") else "direct",
            receipt=receipt,
            receipt_error=receipt_error,
            identity_digest=identity,
            state_patch=patch,
            on_commit_error=patch_error,
            send_ended=ended,
            ended_at=ended_at,
        )
        await _append_outcome(scope, kinds.KIND_EFFECT_COMMITTED, payload, receipt=receipt)
        if patch is not None:
            scope.apply_patch(patch)
        current = scope.journal.view().get(op.namespace, op.name, op.key)
        if current.state == "X":
            return Outcome("held", reason=HoldReason.CONFLICT)
        identity_pair = (op.name, op.key)
        if identity_pair not in scope.committed:
            scope.committed.append(identity_pair)
        _reach(scope, op, row.rule, "committed")
        return Outcome("committed", receipt=receipt, receipt_error=receipt_error)
    classification = "unknown" if timeout else _classify_send_error(effect, error)
    rejected = classification == "rejected"
    error_kind = (
        "timeout" if timeout else "duplicate_key" if classification == "duplicate" else "raised"
    )
    if isinstance(error, asyncio.CancelledError) and not timeout:
        error_kind = "interrupted"
    payload = kinds.EffectFailedData(
        **ref,
        attempt=number,
        intent_seq=intent_seq,
        committed=False if rejected else None,
        why="rejected" if rejected else "interrupted" if error_kind == "interrupted" else "unknown",
        error=str(error) if error is not None else None,
        error_kind=error_kind,
        send_ended=ended,
        ended_at=ended_at,
        overran=overran,
    )
    await _append_outcome(scope, kinds.KIND_EFFECT_FAILED, payload)
    current = scope.journal.view().get(op.namespace, op.name, op.key)
    resolved = current.state == "R"
    kind = (
        "timeout"
        if timeout
        else "rejected"
        if rejected
        else "duplicate"
        if classification == "duplicate"
        else "other_error"
    )
    row = table.decide(
        table.TState(current.state),
        table.TEvent.OUTCOME,
        table.OutcomeSituation(kind, resolved, not effect.is_async),
    )
    _reach(scope, current, row.rule, "unknown_outcome" if not resolved else None)
    if rejected and resolved:
        if (op.name, op.key) not in scope.rejected:
            scope.rejected.append((op.name, op.key))
        return Outcome("rejected", error=error, error_kind=error_kind)
    reason = (
        HoldReason.REJECTED_WHILE_IN_FLIGHT
        if rejected
        else HoldReason.LOOKUP_INCONCLUSIVE
        if classification == "duplicate"
        else HoldReason.SEND_STILL_RUNNING
        if not ended
        else HoldReason.RESPONSE_LOST
    )
    return Outcome("held", error=error, reason=reason, error_kind=error_kind)


async def _dispatch(
    effect: Effect,
    op: Operation,
    frozen: canonical.Frozen,
    scope: CallScope,
    rule: str,
    generation: int,
) -> Outcome:
    from qagents import tools

    prepared = _DISPATCH_INPUT.get()
    if prepared is None or prepared.frozen is not frozen:
        raise ConfigurationError("Dispatch requires the current frozen perform input.")
    journal = scope.journal
    store_now = await journal.store_now()
    invalid = _authorization_valid(rule, scope, store_now)
    if invalid is not None:
        if op.state in ("N", "R"):
            raise ConfigurationError("The tool attempt has no valid authorization to send.")
        return Outcome("held", reason=invalid)
    if not await _self_fence(journal, effect):
        if op.state in ("N", "R"):
            raise LeaseError("Effect dispatch refused: no safe lease margin.")
        return Outcome("held", reason=HoldReason.LEASE)
    slot = None
    if not effect.is_async:
        slot = tools.reserve_thread()
        if slot is None:
            if op.state == "D":
                return Outcome("held", reason=HoldReason.SEND_STILL_RUNNING)
            raise ToolError(
                prompts.TOOL_THREADS_EXHAUSTED.format(name=effect.name, count=tools.TOOL_THREADS)
            )
    try:
        number = max((item.number for item in op.attempts), default=0) + 1
        store_now = await journal.store_now()
        salt = await journal.salt()
        idem = canonical.idempotency_key(
            salt, op.namespace, op.name, op.key, generation, effect.idempotency_key_max_len
        )
        tag = canonical.operation_tag(op.namespace, op.name, op.key, frozen.digest)
        not_after = store_now + effect.timeout_s + journal.timing.m_s
        contract = effect.dedupe
        payload = kinds.EffectIntentData(
            **_reference(op),
            key_spec_digest=effect.key_spec_digest,
            version=effect.version,
            operation_id=scope.operation_id,
            tool_call_id=scope.tool_call_id,
            occurrence=prepared.occurrence,
            attempt=number,
            generation=generation,
            idempotency_key=idem,
            operation_tag=tag,
            request_digest=frozen.digest,
            destination_digest=prepared.descriptor.destination_digest,
            request=frozen.json,
            destination=prepared.destination,
            epoch=journal.lease.epoch,
            store_ts=store_now,
            first_use_at=op.first_use_at if op.first_use_at is not None else store_now,
            not_after=not_after,
            may_overrun=_may_overrun(effect),
            authorization=(
                scope.original_authorization if rule in ("D3", "D4") else scope.authorization
            ).data(),
            rule=rule,
            timing=kinds.IntentTiming(
                timeout_s=effect.timeout_s,
                applies_within_s=effect.applies_within_s,
                m_s=journal.timing.m_s,
                sigma_s=journal.timing.sigma_s,
                window_s=contract.window_s if isinstance(contract, Dedupe) else None,
                margin_s=contract.margin() if isinstance(contract, Dedupe) else None,
                unique=isinstance(contract, Unique),
                concurrent=contract.concurrent if isinstance(contract, Dedupe) else "unknown",
            ),
            receipt_identity=list(effect.receipt_identity)
            if effect.receipt_identity is not None
            else None,
        )
        [intent_seq] = await journal.append_effect(
            [
                Entry(
                    kind=kinds.KIND_EFFECT_INTENT,
                    data=payload.model_dump(mode="json"),
                    run_id=scope.run_id,
                )
            ]
        )
        op = journal.view().get(op.namespace, op.name, op.key)
        scope.emit(
            events.EffectStarted(
                record_id=op.record_id,
                effect=op.name,
                namespace=op.namespace,
                key=op.key,
                attempt=number,
                generation=generation,
                rule=rule,
                tool_call_id=scope.tool_call_id,
            )
        )
        invalid = _authorization_valid(rule, scope, await journal.store_now())
        if not await _self_fence(journal, effect) or invalid is not None:
            ended_at = await journal.store_now()
            await _append_outcome(
                scope,
                kinds.KIND_EFFECT_FAILED,
                kinds.EffectFailedData(
                    **_reference(op),
                    attempt=number,
                    intent_seq=intent_seq,
                    committed=False,
                    why="not_sent",
                    send_ended=True,
                    ended_at=ended_at,
                ),
            )
            _reach(scope, op, "E7")
            return Outcome("not_sent", reason=invalid or HoldReason.LEASE)
        loop = asyncio.get_running_loop()
        attempt = types.Attempt(
            _request_for_send(effect, frozen, prepared.original),
            op.key,
            idem,
            tag,
            loop.time() + effect.timeout_s,
            not_after,
            types.AttemptDebug(number, generation, journal.lease.epoch, journal.lease.epoch or 0),
        )
        ctx = types.EffectContext(scope.deps, session_id=scope.session_id, emit=scope.emit)
        raw = None

        async def invoke() -> Any:
            nonlocal raw
            token = types._ATTEMPT.set(attempt)
            ports._set_attempt_live(attempt, True)
            try:

                async def call_send() -> Any:
                    nonlocal raw
                    if loop.time() >= attempt.deadline:
                        raise _DispatchNotSent()
                    if effect.is_async:
                        return await effect.send(ctx, attempt)

                    def call_sync() -> Any:
                        if loop.time() >= attempt.deadline:
                            raise _DispatchNotSent()
                        return effect.send(ctx, attempt)

                    wrapped, raw = tools.start_thread(effect.name, call_sync, slot=slot)
                    journal.track(raw)
                    return await wrapped

                transport = ports.TRANSPORT.get()
                return (
                    await call_send()
                    if transport is None
                    else await transport.send(effect, ctx, attempt, call_send)
                )
            finally:
                ports._set_attempt_live(attempt, False)
                types._ATTEMPT.reset(token)

        send = asyncio.create_task(invoke())
        journal.track(send)
        _reach(
            scope,
            op,
            rule,
            {
                "D1": "first_send",
                "D2": "retry_after_rejection",
                "D3": "dedupe_resend",
                "D4": "absence_resend",
                "D5": "operator_resend",
            }[rule],
        )
        done, _ = await asyncio.wait({send}, timeout=effect.timeout_s)
        timed_out = not done
        if timed_out:
            ports._set_attempt_live(attempt, False)
        if timed_out and effect.is_async:
            send.cancel()
            done, _ = await asyncio.wait({send}, timeout=journal.timing.m_s)
        if not done:
            gate = asyncio.Event()
            if raw is not None:
                tools._abandon(effect.name, raw, "exceeded the effect timeout")
                _track_abandoned(journal.store, scope.session_id, op.record_id, raw)

            async def late() -> None:
                await gate.wait()
                result, error = None, None
                try:
                    result = await send
                except BaseException as caught:
                    error = caught
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise error
                await _finish_send(
                    effect,
                    op,
                    scope,
                    number,
                    intent_seq,
                    rule,
                    result=result,
                    error=error,
                    overran=True,
                )
                await _append_outcome(
                    scope,
                    kinds.KIND_EFFECT_SEND_ENDED,
                    kinds.EffectSendEndedData(
                        **_reference(op),
                        attempt=number,
                        intent_seq=intent_seq,
                        overran=True,
                        outcome="raised" if error is not None else "returned",
                        ended_at=await journal.store_now(),
                    ),
                )
                _reach(scope, op, "E5o")

            watcher = asyncio.create_task(late())
            journal.track(watcher)
            try:
                return await _finish_send(
                    effect,
                    op,
                    scope,
                    number,
                    intent_seq,
                    rule,
                    timeout=True,
                    ended=False,
                    overran=True,
                )
            finally:
                gate.set()
        result, error = None, None
        try:
            result = send.result()
        except BaseException as caught:
            error = caught
        return await _finish_send(
            effect,
            op,
            scope,
            number,
            intent_seq,
            rule,
            result=result,
            error=error,
            timeout=timed_out and isinstance(error, asyncio.CancelledError),
        )
    finally:
        if slot is not None:
            slot.release()


def _prepare(effect: Effect, request: Any, scope: CallScope, occurrence: int | None) -> _Prepared:
    """P0 and the immutable request/destination identity, before the first store write."""
    if not isinstance(effect, Effect) or effect.name not in scope.effects:
        raise ConfigurationError("perform requires an effect listed in this tool's effects=[...].")
    if scope.composition_refusal is not None:
        raise ConfigurationError(scope.composition_refusal)
    journal = scope.journal
    if (
        not scope.session_id
        or journal.session_id != scope.session_id
        or not getattr(journal, "leased", True)
        or not getattr(journal.lease, "owner", None)
        or EFFECTS_REQUIRED - journal.store.features
    ):
        raise ConfigurationError(
            "perform requires a fenced session implementing the Effects contract."
        )
    if scope.key_override is not None and scope.tool_name is not None:
        raise ConfigurationError("An effect key override is only available to Workflow.effect.")
    occurrence = _occurrence_for(scope, effect, occurrence)
    frozen = canonical.freeze(effect, request)
    restored = canonical.rebuild(effect, frozen)
    try:
        namespace = canonical.resolve_namespace(effect, scope.deps, scope.session_id)
        destination = effect.destination(scope.deps) if effect.destination is not None else None
        destination = json.loads(canonical.jcs(destination))
    except Exception as error:
        raise ConfigurationError(
            f"{effect.name}: invalid namespace or destination: {error}"
        ) from error
    _require_claims(effect, namespace, journal.store)
    key = effect.key.render(
        restored, effect=effect.name, operation_id=scope.operation_id, occurrence=occurrence
    )
    if scope.key_override is not None:
        if not isinstance(scope.key_override, str) or not scope.key_override:
            raise ConfigurationError("Workflow effect key must be a nonempty string.")
        canonical.jcs(scope.key_override)
        key = scope.key_override
    return _Prepared(
        frozen,
        canonical.descriptor(effect, destination, frozen.digest),
        namespace,
        key,
        occurrence,
        destination,
        request,
    )


def _occurrence_for(scope: CallScope, effect: Effect, occurrence: int | None) -> int | None:
    """Require an explicit occurrence after the first call-scoped invocation (M12)."""
    if occurrence is not None and (type(occurrence) is not int or occurrence < 0):
        raise ConfigurationError("occurrence must be a nonnegative integer or None.")
    if effect.key.is_per_call and scope.occurrences[effect.name] and occurrence is None:
        raise ConfigurationError("A second per-call perform needs an explicit occurrence=.")
    scope.occurrences[effect.name] += 1
    return occurrence


def _request_for_send(effect: Effect, frozen: canonical.Frozen, original: Any) -> Any:
    """The sender receives a fresh validated snapshot, never the caller's object (M13)."""
    return canonical.rebuild(effect, frozen)


def _may_overrun(effect: Effect) -> bool:
    """A synchronous send cannot be stopped by cancelling its asyncio wrapper (M35)."""
    return not effect.is_async


def _classify_send_error(
    effect: Effect, error: BaseException
) -> Literal["rejected", "duplicate", "unknown"]:
    """Only explicit destination contracts classify a raised response as definite (M4)."""
    global _TOOLERROR_WARNED
    if isinstance(error, EffectRejected) or isinstance(error, effect.rejected_on):
        return "rejected"
    if isinstance(effect.dedupe, Unique) and isinstance(error, effect.dedupe.on_duplicate):
        return "duplicate"
    if isinstance(error, ToolError) and not _TOOLERROR_WARNED:
        _TOOLERROR_WARNED = True
        logging.getLogger("qagents.effects").warning(
            "ToolError from an effect send has an unknown outcome; declare rejected_on only "
            "when the destination guarantees no application."
        )
    return "unknown"


async def _self_fence(journal: EffectJournal, effect: Effect) -> bool:
    """Renew once for a short margin; a lost lease never authorizes a new send (M7)."""
    lease = journal.lease
    if lease.lost():
        return False
    required = effect.timeout_s + journal.timing.m_s
    if lease.remaining() >= required:
        return True
    try:
        if not await lease.renew():
            return False
    except Exception:
        return False
    return not lease.lost() and lease.remaining() >= required


def _authorization_valid(rule: str, scope: CallScope, store_now: float) -> HoldReason | None:
    """Fresh sends use the current decision; automatic retries need its original proof (M17)."""
    if rule == "D5":
        return (
            None
            if scope.authorization.decision == "resend"
            and scope.authorization.settlement_seq is not None
            else HoldReason.APPROVAL_UNVERIFIABLE
        )
    if rule not in {"D1", "D2", "D3", "D4"}:
        raise ConfigurationError("Authorization can only be checked for a sending row.")
    auth = scope.original_authorization if rule in {"D3", "D4"} else scope.authorization
    if auth is None or auth.decision == "resend":
        return HoldReason.APPROVAL_UNVERIFIABLE
    if rule in {"D3", "D4"} and not auth.verifiable:
        return HoldReason.APPROVAL_UNVERIFIABLE
    if auth.expires_at is not None and store_now > auth.expires_at:
        return HoldReason.APPROVAL_EXPIRED
    return None


def _d2_plan(effect: Effect, op: Operation) -> tuple[bool, int]:
    """Look up a rejection before retrying; only replayed rejections advance generation (M33)."""
    changes_key = isinstance(effect.dedupe, Dedupe) and effect.dedupe.replays_rejections
    return effect.lookup_fn is not None, op.generation + int(changes_key)


async def perform(
    effect: Effect[ReqT, RecT], request: ReqT, *, scope: CallScope, occurrence: int | None = None
) -> RecT:
    """SPEC 7.2 steps 1-6. Returns the typed receipt; raises EffectRejected,
    EffectConflictError, EffectReceiptError, EffectHeld ("pause"), HeldEffectsError
    ("fail_run"), ToolError(EFFECT_OUTCOME_UNKNOWN) ("tell_model"), ConfigurationError
    (P0), LeaseError (F1: nothing journaled or sent), JournalOutcomeUnknown (poisoned),
    CancelledError (after the shield). Keeps the perform contract of 3.6."""
    _prepare(effect, request, scope, occurrence)
    raise NotImplementedError("Effect state transitions and dispatch are pending implementation.")


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


def send_running(store: SessionStore, session_id: str, record_id: str) -> bool:
    """True while an abandoned sync send of that operation still runs in this process
    (SEND_STILL_RUNNING: O4a does not re-run the call)."""
    root = _root(store)
    with _WORK_LOCK:
        work = _WORK.get(id(root))
        return work is not None and any(
            not future.done() for future in work.sends.get((session_id, record_id), ())
        )


def _forget_store(store: SessionStore) -> None:
    """Drop the single-flight and send-running entries whose root store is this object
    (harness teardown)."""
    root = _root(store)
    with _WORK_LOCK:
        _WORK.pop(id(root), None)


def _require_claims(effect, namespace, store) -> None:
    """M45: every declared shared namespace needs claims, regardless of its spelling."""
    if effect.namespace != "session" and "claims" not in store.features:
        raise ConfigurationError(
            f"{effect.name}: shared namespace {namespace!r} needs the store claims feature. "
            "Use a claims-capable store or namespace='session'."
        )
