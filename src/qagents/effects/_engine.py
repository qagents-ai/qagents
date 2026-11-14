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
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from functools import partial
from typing import Any, Literal

from pydantic import BaseModel

from qagents import events, prompts
from qagents.errors import (
    ConfigurationError,
    EffectConflictError,
    EffectHeld,
    EffectReceiptError,
    EffectRejected,
    HeldEffectsError,
    JournalOutcomeUnknown,
    LeaseError,
    ToolError,
)
from qagents.session import EFFECTS_REQUIRED, Entry, SessionStore

from . import _canonical as canonical
from . import _declare
from . import _kinds as kinds
from . import _ports as ports
from . import _quiescence as quiescence
from . import _table as table
from . import _types as types
from ._canonical import LIMIT_BYTES, JSONValue, check_model_numbers, jcs, resolve_namespace
from ._declare import Dedupe, Effect, Unique
from ._fold import SETTLEMENT_FOLD_KEYS, Operation, _new_operation
from ._kinds import StatePatchData
from ._ports import FINDINGS_LOGGER, CallScope, EffectJournal, Timing
from ._types import (
    HOLD_DETAIL,
    MISSING,
    Absent,
    Found,
    HeldEffect,
    HoldReason,
    NotYetVisible,
    RecT,
    ReqT,
    Unknown,
)

Verdict = Found | NotYetVisible | Absent | Unknown
_TOOLERROR_WARNED = False
# The hold detail for an operator resend that does not verify under the approval_secret.
_RESEND_UNVERIFIABLE = (
    "Operator resend unverifiable: no valid MAC for this record and revision; settle again."
)


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
    op = scope.journal.view().get(op.namespace, op.name, op.key) or op
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
        number = (
            max((item.number for item in [*op.attempts, *_claimed_attempts(op)]), default=0) + 1
        )
        salt = await journal.salt()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + effect.timeout_s
        store_now = await journal.store_now()
        if rule == "D3" and not _dedupe_allowed(effect, op, scope, store_now):
            return Outcome("held", reason=HoldReason.WINDOW_EXPIRED)
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
        transport = ports.TRANSPORT.get()
        before_fence = getattr(transport, "before_second_fence", None)
        if before_fence is not None:
            await before_fence(effect)
        fenced = await _self_fence(journal, effect)
        current_time = await journal.store_now()
        invalid = _authorization_valid(rule, scope, current_time)
        if rule == "D3" and not quiescence._dedupe_window_covers(
            current_time,
            quiescence._window_start(op),
            effect,
            journal.timing,
        ):
            invalid = HoldReason.WINDOW_EXPIRED
        if not fenced or journal.lease.lost() or invalid is not None or loop.time() >= deadline:
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
        attempt = types.Attempt(
            _request_for_send(effect, frozen, prepared.original),
            op.key,
            idem,
            tag,
            deadline,
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
        done, _ = await asyncio.wait({send}, timeout=max(0, deadline - loop.time()))
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
                    timeout=_cancelled_at_deadline(effect, True, error),
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
            timeout=_cancelled_at_deadline(effect, timed_out, error),
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


def _cancelled_at_deadline(effect: Effect, timed_out: bool, error: BaseException | None) -> bool:
    """Whether a send's error is the engine's own timeout (SPEC 7.1 E5, E5o; mutant seam M53).

    Past the deadline the engine cancels an async send. The cancellation, not an
    answer from the destination, ends it then, so any error it raises, within m or
    later, is that timeout and its outcome is unknown: also a ``rejected_on`` type
    that the send raises in reaction (``except CancelledError: ... raise
    PaymentsError()``). A sync send is never cancelled; its late error is classified
    as it was raised.
    """
    return timed_out and error is not None and effect.is_async


def _classify_send_error(
    effect: Effect, error: BaseException
) -> Literal["rejected", "duplicate", "unknown"]:
    """Only explicit destination contracts classify a raised response as definite (M4).

    A transport-level no answer (a timeout, a lost connection, a cancellation, an
    httpx transport error) is never a rejection, even when its class also derives
    from EffectRejected or a rejected_on type (SPEC 5.1). Its outcome is unknown.
    """
    global _TOOLERROR_WARNED
    if not _declare._no_answer(error) and (
        isinstance(error, EffectRejected) or isinstance(error, effect.rejected_on)
    ):
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
            and scope.authorization.verifiable
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


def _resend_verified(scope: CallScope, op: Operation) -> bool:
    """With an approval_secret, a pending operator resend counts only if its MAC signs the
    journaled answer and it names the revision it followed (M50, SPEC 5.3). A copy of a
    genuine answer appended later names an older revision. Without a secret the journal is
    trusted, as before."""
    settlement = op.pending_resend
    permissions = scope.permissions
    if settlement is None or permissions is None or not permissions.signs_pauses:
        return True
    prior = (settlement.model_extra or {}).get("prior_revision")
    data = settlement.model_dump(mode="json", exclude=set(SETTLEMENT_FOLD_KEYS))
    fields = kinds.settlement_mac_fields(scope.session_id, kinds.KIND_EFFECT_SETTLED, data)
    return prior == settlement.expected_revision and permissions.verify_settlement(
        fields, settlement.mac
    )


def _d2_plan(effect: Effect, op: Operation) -> tuple[bool, int]:
    """Look up a rejection before retrying; only replayed rejections advance generation (M33)."""
    changes_key = isinstance(effect.dedupe, Dedupe) and effect.dedupe.replays_rejections
    return effect.lookup_fn is not None, op.generation + int(changes_key)


def _claimed_attempts(op: Operation) -> list[kinds.ClaimAttemptData]:
    return list(op.claimed.attempts) if op.claimed is not None else []


def _lookup_state(effect: Effect, op: Operation) -> Literal["not_run", "none_declared"]:
    """A declared lookup runs first once the journal holds any earlier attempt (M49).

    The attempts may be this record's own or another owner's, carried over by a
    claim transfer (SPEC 4.4 D2, 7.10).
    """
    if effect.lookup_fn is None or not (op.attempts or op.claimed is not None):
        return "none_declared"
    return "not_run"


async def _take_claim(
    effect: Effect, op: Operation, claim: Any, prepared: _Prepared, scope: CallScope
) -> None:
    """Journal the move of a rejected claim, with its attempt history, before any
    lookup or send (SPEC 7.10). The claim index refuses it if the claim changed."""
    journal = scope.journal
    payload = kinds.EffectClaimedData(
        **_reference(op),
        version=effect.version,
        request_digest=prepared.frozen.digest,
        destination_digest=prepared.descriptor.destination_digest,
        request=prepared.frozen.json,
        operation_tag=canonical.operation_tag(
            op.namespace, op.name, op.key, prepared.frozen.digest
        ),
        receipt_identity=list(effect.receipt_identity)
        if effect.receipt_identity is not None
        else None,
        operation_id=scope.operation_id,
        tool_call_id=scope.tool_call_id,
        previous_owner={
            "session_id": claim.session_id,
            "incarnation": claim.incarnation,
            "record_id": claim.record_id,
        },
        status="rejected",
        generation=claim.generation,
        first_use_at=claim.first_use_at,
        attempts=claim.attempts,
        store_ts=await journal.store_now(),
    )
    await _append_outcome(scope, kinds.KIND_EFFECT_CLAIMED, payload)
    _reach(scope, op, "C1")


async def perform(
    effect: Effect[ReqT, RecT], request: ReqT, *, scope: CallScope, occurrence: int | None = None
) -> RecT:
    """SPEC 7.2 steps 1-6. Returns the typed receipt; raises EffectRejected,
    EffectConflictError, EffectReceiptError, EffectHeld ("pause"), HeldEffectsError
    ("fail_run"), ToolError(EFFECT_OUTCOME_UNKNOWN) ("tell_model"), ConfigurationError
    (P0), LeaseError (F1: nothing journaled or sent), JournalOutcomeUnknown (poisoned),
    CancelledError (after the shield). Keeps the perform contract of 3.6."""
    prepared = _prepare(effect, request, scope, occurrence)
    if scope.journal.poisoned():
        raise JournalOutcomeUnknown("Cannot perform with a poisoned Effects journal.")
    return await _single_flight(prepared, scope, lambda: _perform(effect, prepared, scope))


def _finish_event(
    scope: CallScope,
    op: Operation,
    outcome: Literal["committed", "rejected", "held"],
    *,
    replayed: bool = False,
    reason: HoldReason | None = None,
) -> None:
    scope.emit(
        events.EffectFinished(
            record_id=op.record_id,
            effect=op.name,
            namespace=op.namespace,
            key=op.key,
            outcome=outcome,
            via=op.via,
            replayed=replayed,
            reason=reason,
            tool_call_id=scope.tool_call_id,
        )
    )


def _receipt(effect: Effect, op: Operation, scope: CallScope, *, replayed: bool) -> Any:
    pair = (op.name, op.key)
    if pair not in scope.committed:
        scope.committed.append(pair)
    _finish_event(scope, op, "committed", replayed=replayed)
    if op.receipt is not None:
        try:
            return effect.receipt_type.model_validate_json(jcs(op.receipt))
        except (ValueError, TypeError) as error:
            cause = error
    else:
        cause = ValueError("The committed receipt is unavailable.")
    raise EffectReceiptError(
        prompts.EFFECT_RECEIPT_UNSTORABLE.format(effect=op.name, key=op.key, error=cause),
        effect=op.name,
        key=op.key,
        receipt_error=cause,
    ) from cause


def _reject(scope: CallScope, op: Operation, error: BaseException | None = None) -> None:
    pair = (op.name, op.key)
    if pair not in scope.rejected:
        scope.rejected.append(pair)
    _finish_event(scope, op, "rejected", replayed=error is None)
    if error is None:
        error = EffectRejected("The journal records a definite rejection.")
    raise EffectRejected(
        prompts.EFFECT_REJECTED.format(effect=op.name, key=op.key, error=error),
        effect=op.name,
        key=op.key,
        error=error,
    ) from error


async def _hold(
    effect: Effect,
    op: Operation,
    scope: CallScope,
    reason: HoldReason,
    *,
    rule: str = "H1",
    detail: str | None = None,
) -> None:
    journal = scope.journal
    now = await journal.store_now()
    retry = quiescence.retry_after(op, journal.view(), journal.timing, now)
    observed = op.hold
    if observed is not None and observed.retry_after is not None and observed.retry_after > now:
        retry = max(retry or now, observed.retry_after)
    if reason in {
        HoldReason.CONFLICT,
        HoldReason.DIVERGED,
        HoldReason.KEY_CHANGED,
        HoldReason.VERSION_CHANGED,
        HoldReason.WINDOW_EXPIRED,
        HoldReason.APPROVAL_EXPIRED,
        HoldReason.APPROVAL_UNVERIFIABLE,
        HoldReason.CLAIMED_ELSEWHERE,
    }:
        retry = None
    detail = detail or HOLD_DETAIL[reason]
    if not journal.lease.lost() and reason != HoldReason.CLAIMED_ELSEWHERE:
        try:
            await _append_outcome(
                scope,
                kinds.KIND_EFFECT_OBSERVED,
                kinds.EffectObservedData(
                    **_reference(op),
                    attempts=[item.number for item in op.attempts],
                    verdict=None,
                    reason=reason,
                    detail=detail,
                    retry_after=retry,
                ),
            )
        except LeaseError:
            # A different writer can take over between our clock read and the append.
            # Surface the hold without claiming the observation was persisted.
            reason, detail, retry = HoldReason.LEASE, HOLD_DETAIL[HoldReason.LEASE], None
    held = HeldEffect(
        op.record_id,
        scope.session_id,
        op.name,
        op.key,
        op.namespace,
        reason,
        detail,
        None if retry is None else datetime.fromtimestamp(retry, UTC),
        op.revision,
        scope.tool_call_id,
    )
    scope.held.append(held)
    _finish_event(scope, op, "held", reason=reason)
    _reach(scope, op, rule, "held")
    if reason == HoldReason.WINDOW_EXPIRED:
        _reach(scope, op, rule, "held_after_window")
    if effect.on_unknown == "fail_run":
        raise HeldEffectsError(scope.held, retry_after=held.retry_after)
    if effect.on_unknown == "tell_model":
        error = ToolError(
            prompts.EFFECT_OUTCOME_UNKNOWN.format(
                effect=op.name,
                key=op.key,
                committed=", ".join(f"{name} for {key}" for name, key in scope.committed)
                or prompts.EFFECT_NONE_DONE,
            )
        )
        error.metadata = {"effect_outcome_unknown": True}
        raise error
    raise EffectHeld(held)


def _conflict(op: Operation) -> None:
    status = {
        "C": "committed",
        "R": "rejected",
        "D": "held",
        "NA": "not_applied",
        "CU": "closed_unknown",
        "X": "conflict",
        "N": "claimed",
    }[op.state]
    raise EffectConflictError(
        prompts.EFFECT_CONFLICT.format(effect=op.name, key=op.key, status=status),
        effect=op.name,
        key=op.key,
        record_id=op.record_id,
        existing_status=status,
    )


def _dedupe_allowed(effect: Effect, op: Operation, scope: CallScope, now: float) -> bool:
    journal = scope.journal
    contract = effect.dedupe
    if (
        isinstance(contract, Dedupe)
        and contract.replays_errors
        and any(
            a.generation == op.generation and a.terminal == "unknown" and a.raised
            for a in op.attempts
        )
    ):
        return False
    return quiescence._dedupe_window_covers(
        now,
        quiescence._window_start(op),
        effect,
        journal.timing,
    ) and quiescence._concurrency_permits(effect, op, journal.view(), journal.timing, now)


def _absence_is_final(effect, verdict, op, situation):
    """Require an authoritative absence observed after all possible applications."""
    started = op.hold.started_at if op.hold else None
    return (
        isinstance(verdict, Absent)
        and effect.authoritative
        and started is not None
        and quiescence._lookup_after_quiescence(started, situation)
    )


async def _perform(effect: Effect, prepared: _Prepared, scope: CallScope) -> Any:
    journal = scope.journal
    key = (prepared.namespace, effect.name, prepared.key)
    view = journal.view()
    op = view.get(*key) or _new_operation(
        kinds.EffectRefData(
            namespace=key[0],
            name=key[1],
            key=key[2],
            record_id=canonical.record_id(*key),
        )
    )
    for finding in list(op.findings):
        await _lookup_conflict(journal, _reference(op), finding.what, finding.seqs, finding.detail)

    claim = None
    claim_kind = "none"
    if (
        (op.state == "N" or op.via == "claim")
        and not effect.key.is_per_call
        and ("claims" in journal.store.features)
    ):
        claim = await journal.store.get_claim(*key)
        if op.via == "claim" and (
            claim is None or claim.descriptor_digest != prepared.descriptor.digest()
        ):
            _conflict(op)
        if claim is not None:
            if claim.descriptor_digest != prepared.descriptor.digest():
                claim_kind = "other_descriptor"
            elif claim.status == "committed":
                claim_kind = "committed_same"
            elif claim.status in ("rejected", "not_sent"):
                claim_kind = claim.status
            else:
                claim_kind = "held_elsewhere"

    lookup = _lookup_state(effect, op)
    reason = None
    retry_blocked = False
    while True:
        now = await journal.store_now()
        view = journal.view()
        op = view.get(*key) or op
        same_call = scope.operation_id in op.operation_ids
        args_diverged = (
            scope.original_arguments_digest is not None
            and scope.arguments_digest != scope.original_arguments_digest
        )
        own_rejection = any(
            a.generation == op.generation
            and a.terminal == "rejected"
            and a.intent.operation_id == scope.operation_id
            for a in op.attempts
        )
        pending = op.pending_resend
        resend_seq = (pending.model_extra or {}).get("seq") if pending is not None else None
        # A resend that does not verify authorizes nothing, and no row sends while it stands:
        # it may have reopened a record that was rejected or settled (SPEC 5.3, 7.5).
        resend_unverified = pending is not None and not _resend_verified(scope, op)
        dedupe = op.state == "D" and not retry_blocked and _dedupe_allowed(effect, op, scope, now)
        if op.state == "D" and send_running(journal.store, scope.session_id, op.record_id):
            await _hold(effect, op, scope, HoldReason.SEND_STILL_RUNNING)
        auth_rule = (
            "D3" if dedupe else "D4" if op.state == "D" else "D2" if op.state == "R" else "D1"
        )
        invalid = _authorization_valid(auth_rule, scope, now)
        if resend_unverified:
            invalid = HoldReason.APPROVAL_UNVERIFIABLE
        situation = table.PerformSituation(
            allowed=True,
            same_call=same_call,
            descriptor_differs=op.descriptor is not None
            and not table._descriptor_matches(op, prepared.descriptor),
            arguments_diverged=args_diverged,
            key_changed=table._key_changed(view, scope, effect, key, prepared.occurrence),
            version_changed=bool(op.version and op.version != effect.version),
            own_rejection=own_rejection,
            claim=claim_kind,
            lookup=lookup,
            fence_ok=not journal.lease.lost(),
            auth_valid=invalid is None,
            dedupe_valid=dedupe,
            resend_accepted=resend_seq is not None and not retry_blocked and not resend_unverified,
        )
        if op.state in ("N", "R") and invalid is not None and not own_rejection:
            raise ConfigurationError("The tool attempt has no valid authorization to send.")
        row = table.decide(table.TState(op.state), table.TEvent.PERFORM, situation)

        if row is table.NEED_LOOKUP:
            prior_state, prior_commit = op.state, op.commit_seq
            verdict = await lookup_and_record(
                effect,
                op,
                journal=journal,
                deps=scope.deps,
                purpose="perform",
                state_values=scope.committed_values,
            )
            view = journal.view()
            current = view.get(*key)
            if current.state == "C":
                # The lookup helper journals the patch. Reuse its exact durable delta.
                entries = await journal.store.load(scope.session_id)
                commit = next(
                    (
                        item
                        for item in entries
                        if item.seq == current.commit_seq
                        and item.seq != prior_commit
                        and item.data.get("via") == "lookup"
                    ),
                    None,
                )
                if commit is not None and commit.data.get("state_patch") is not None:
                    scope.apply_patch(StatePatchData.model_validate(commit.data["state_patch"]))
                if commit is not None:
                    _reach(scope, current, "L2" if prior_state == "R" else "L1", "found_by_lookup")
                return _receipt(effect, current, scope, replayed=commit is None)
            lookup = "unknown"
            if isinstance(verdict, NotYetVisible):
                lookup, reason = "not_yet_visible", HoldReason.NOT_YET_VISIBLE
            elif isinstance(verdict, Absent):
                lookup = "absent"
                q = quiescence._all_quiescent(current, view, journal.timing)
                if _absence_is_final(effect, verdict, current, q):
                    observation = current.hold.model_copy(update={"final": True})
                    await _append_outcome(scope, kinds.KIND_EFFECT_OBSERVED, observation)
                    if not retry_blocked:
                        lookup = "absent_authoritative_after_q"
                elif effect.authoritative:
                    reason = HoldReason.NOT_YET_QUIESCENT
            continue

        if row.rule == "P1k":
            prior = view.by_call[(scope.operation_id, effect.name, prepared.occurrence)]
            await _hold(
                effect, view.operations[prior], scope, HoldReason.KEY_CHANGED, rule=row.rule
            )
        if row.rule == "P1":
            await _hold(effect, op, scope, HoldReason.DIVERGED, rule=row.rule)
        if row.rule == "P2":
            _conflict(op)
        if row.rule == "P3":
            if situation.version_changed:
                logging.getLogger(FINDINGS_LOGGER).warning(
                    "%s replayed a receipt written by effect version %s", effect.name, op.version
                )
            _reach(scope, op, "P3", "receipt_replayed")
            return _receipt(effect, op, scope, replayed=True)
        if row.rule == "P3r":
            _reject(scope, op)
        if row.rule == "P4":
            raise EffectRejected(
                prompts.EFFECT_NOT_APPLIED.format(
                    effect=op.name,
                    key=op.key,
                    evidence=op.settlements[-1].evidence,
                ),
                effect=op.name,
                key=op.key,
            )
        if row.rule == "P5":
            raise ToolError(
                prompts.EFFECT_CLOSED_UNKNOWN.format(
                    effect=op.name,
                    key=op.key,
                    reason=op.settlements[-1].reason,
                )
            )
        if row.rule in ("P6", "P7"):
            await _hold(
                effect,
                op,
                scope,
                HoldReason.CONFLICT if row.rule == "P6" else HoldReason.VERSION_CHANGED,
                rule=row.rule,
            )
        if row.rule == "C1":
            if claim_kind == "other_descriptor":
                _conflict(op)
            if claim_kind == "held_elsewhere":
                await _hold(
                    effect,
                    op,
                    scope,
                    HoldReason.CLAIMED_ELSEWHERE,
                    rule="C1",
                    detail=f"Claimed by {claim.session_id}, record {claim.record_id}.",
                )
            if claim_kind == "committed_same":
                receipt, receipt_error = receipt_json(effect, claim.receipt)
                await _append_outcome(
                    scope,
                    kinds.KIND_EFFECT_COMMITTED,
                    kinds.EffectCommittedData(
                        **_reference(op),
                        attempt=None,
                        intent_seq=None,
                        via="claim",
                        receipt=receipt,
                        receipt_error=receipt_error,
                        identity_digest=canonical.identity_digest(receipt, effect.receipt_identity)
                        if receipt is not None
                        else None,
                        claim_owner={"session_id": claim.session_id, "record_id": claim.record_id},
                    ),
                    receipt=claim.receipt,
                )
                _reach(scope, op, "C1", "claim_replayed")
                return _receipt(effect, view.get(*key), scope, replayed=True)
            if claim_kind == "rejected":
                if not situation.fence_ok:
                    raise LeaseError("Effect dispatch refused: no safe lease margin.")
                # Record first: the claim and its earlier attempts move here, then the
                # folded R(n) rows decide, so a declared lookup runs before any D2.
                await _take_claim(effect, op, claim, prepared, scope)
                claim_kind = "none"
                lookup = _lookup_state(effect, journal.view().get(*key))
                continue
            # Never sent: the atomic intent append checks the claim and transfers it (D1).
            op = replace(op, generation=claim.generation, first_use_at=claim.first_use_at)
            row = table.decide(
                table.TState.N,
                table.TEvent.PERFORM,
                replace(situation, claim="none", same_call=False),
            )
        if row.rule == "F1":
            raise LeaseError("Effect dispatch refused: no safe lease margin.")
        if not row.sends:
            if (
                reason is None
                and isinstance(effect.dedupe, Dedupe)
                and not quiescence._dedupe_window_covers(
                    now,
                    quiescence._window_start(op),
                    effect,
                    journal.timing,
                )
            ):
                reason = HoldReason.WINDOW_EXPIRED
            await _hold(
                effect,
                op,
                scope,
                invalid or reason or HoldReason.DISPATCH_UNKNOWN,
                detail=_RESEND_UNVERIFIABLE if resend_unverified else None,
            )

        dispatch_scope = scope
        generation = op.generation
        if row.rule == "D2":
            _, generation = _d2_plan(effect, op)
        if row.rule == "D5":
            dispatch_scope = replace(
                scope,
                authorization=ports.Authorization(
                    "resend",
                    settlement_seq=resend_seq,
                    verifiable=not resend_unverified,
                ),
            )
        token = _DISPATCH_INPUT.set(prepared)
        try:
            outcome = await _dispatch(
                effect, op, prepared.frozen, dispatch_scope, row.rule, generation
            )
        finally:
            _DISPATCH_INPUT.reset(token)
        op = journal.view().get(*key) or op
        if outcome.kind == "committed":
            return _receipt(effect, op, scope, replayed=False)
        if outcome.kind == "rejected":
            _reject(scope, op, outcome.error)
        reason = outcome.reason
        if outcome.kind == "not_sent" or reason in {
            HoldReason.CONFLICT,
            HoldReason.LEASE,
            HoldReason.APPROVAL_EXPIRED,
            HoldReason.APPROVAL_UNVERIFIABLE,
            HoldReason.SEND_STILL_RUNNING,
            HoldReason.REJECTED_WHILE_IN_FLIGHT,
            HoldReason.WINDOW_EXPIRED,
        }:
            await _hold(effect, op, scope, reason or HoldReason.LEASE)
        # A failed response permits one lookup, and at most one automatic re-send.
        retry_blocked = row.rule in ("D3", "D4", "D5") or outcome.error_kind == "duplicate_key"
        lookup = "not_run" if effect.lookup_fn is not None else "none_declared"


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
    if effect.lookup_fn is None:
        raise ConfigurationError(f"{effect.name}: no lookup is declared.")
    # The caller's view and the model in Inquiry can both change while lookup awaits.
    # Binding always uses the original durable JSON, independently of either object.
    original = deepcopy(op)
    if (
        original.request is None
        or original.descriptor is None
        or not (original.attempts or original.claimed is not None)
    ):
        raise ConfigurationError("Lookup requires a journaled effect intent or claim transfer.")
    request = json.loads(jcs(original.request))
    # Earlier owners' attempts come first: the lookup asks about every key ever sent.
    attempts = (
        *(
            types.AttemptInfo(
                a.number, a.generation, a.idempotency_key, a.epoch, a.store_ts, a.outcome
            )
            for a in _claimed_attempts(original)
        ),
        *(
            types.AttemptInfo(
                a.number,
                a.generation,
                a.intent.idempotency_key,
                a.intent.epoch,
                a.intent.store_ts,
                a.terminal,
            )
            for a in original.attempts
        ),
    )
    inquiry = types.Inquiry(
        request=effect.request_type.model_validate_json(jcs(request)),
        key=original.key,
        operation_tag=canonical.operation_tag(
            original.namespace, original.name, original.key, original.descriptor.request_digest
        ),
        idempotency_keys=tuple(dict.fromkeys(a.idempotency_key for a in attempts)),
        attempts=attempts,
        first_use_at=original.first_use_at,
    )
    ctx = types.EffectContext(deps, session_id=journal.session_id)
    started_at = await journal.store_now()
    verdict = await _lookup_verdict(effect, ctx, inquiry, journal)
    receipt = receipt_error = identity = None
    tag = seen = None
    binding_error = None
    if isinstance(verdict, Found):
        # Capture all mutable evidence before the next store await.
        tag = verdict.tag
        try:
            seen = json.loads(jcs(dict(verdict.seen))) if verdict.seen is not None else None
            if tag is not None and tag != inquiry.operation_tag:
                binding_error = "The destination operation tag differs from the frozen operation."
            if seen is not None and any(
                key not in request or jcs(value) != jcs(request[key]) for key, value in seen.items()
            ):
                binding_error = "The destination request fields differ from the frozen request."
        except Exception as error:
            binding_error = f"Invalid destination request binding: {type(error).__name__}: {error}"
        receipt, receipt_error = receipt_json(effect, verdict.receipt)
        if receipt is not None:
            identity = canonical.identity_digest(receipt, effect.receipt_identity)
            if binding_error is None:
                verdict = Found(
                    effect.receipt_type.model_validate_json(jcs(receipt)), tag=tag, seen=seen
                )
    retry_after = None
    if isinstance(verdict, NotYetVisible) and verdict.retry_after_s is not None:
        hint = verdict.retry_after_s
        if type(hint) not in (int, float) or not math.isfinite(hint) or hint < 0:
            verdict = Unknown("Lookup returned an invalid retry_after_s.")
        else:
            retry_after = started_at + hint
            if not math.isfinite(retry_after):
                verdict, retry_after = Unknown("Lookup retry time is not finite."), None
    ref = {
        "record_id": original.record_id,
        "namespace": original.namespace,
        "name": original.name,
        "key": original.key,
    }
    observed = kinds.EffectObservedData(
        **ref,
        attempts=[a.number for a in original.attempts],
        verdict="found"
        if isinstance(verdict, Found)
        else "absent"
        if isinstance(verdict, Absent)
        else "not_yet_visible"
        if isinstance(verdict, NotYetVisible)
        else "unknown",
        authoritative=effect.authoritative,
        started_at=started_at,
        reason=HoldReason.NOT_YET_VISIBLE
        if isinstance(verdict, NotYetVisible)
        else HoldReason.LOOKUP_INCONCLUSIVE
        if isinstance(verdict, (Unknown, Absent))
        else None,
        detail=verdict.reason if isinstance(verdict, Unknown) else None,
        retry_after=retry_after,
    )
    observed_seq = (
        await journal.append_effect(
            [Entry(kind=kinds.KIND_EFFECT_OBSERVED, data=observed.model_dump(mode="json"))]
        )
    )[0]
    if not isinstance(verdict, Found):
        return verdict
    current = journal.view().get(original.namespace, original.name, original.key)
    if binding_error is not None:
        await _lookup_conflict(journal, ref, "found_mismatch", [observed_seq], binding_error)
        return verdict
    closure_seq = next(
        (
            (settlement.model_extra or {})["seq"]
            for settlement in reversed(current.settlements)
            if settlement.outcome != "resend"
            and not (settlement.model_extra or {}).get("voided", False)
            and (settlement.model_extra or {})["seq"] >= current._resolution_seq
        ),
        None,
    )
    if closure_seq is not None:
        if not _lookup_evidence_seen(current, "late_found_after_settlement", identity, closure_seq):
            await _lookup_conflict(
                journal,
                ref,
                "late_found_after_settlement",
                [closure_seq, observed_seq],
                "A bound result was found after operator closure.",
                identity_digest=identity,
                lookup_base_seq=closure_seq,
            )
        return verdict
    # A send may have committed while the read was in flight. Evidence is still
    # compared, but a successful patch must never run a second time.
    if current.commit_seq is not None and (identity is None or identity == current.identity_digest):
        return verdict
    if _lookup_evidence_seen(current, "receipts_differ", identity, current._resolution_seq):
        return verdict
    patch = patch_error = None
    if receipt is not None and current.commit_seq is None and current.state != "X":
        patch, patch_error = on_commit_patch(effect, state_values(), request, receipt)
    # After a claim transfer with no local attempt, the find names no attempt of this record.
    target = original.attempts[-1] if original.attempts else None
    committed = kinds.EffectCommittedData(
        **ref,
        attempt=target.number if target is not None else None,
        intent_seq=target.intent_seq if target is not None else None,
        via="lookup",
        receipt=receipt,
        receipt_error=receipt_error,
        identity_digest=identity,
        state_patch=patch,
        on_commit_error=patch_error,
        found_tag=tag,
        found_seen=seen,
    )
    await journal.append_effect(
        [Entry(kind=kinds.KIND_EFFECT_COMMITTED, data=committed.model_dump(mode="json"))]
    )
    current = journal.view().get(original.namespace, original.name, original.key)
    for finding in list(current.findings):
        await _lookup_conflict(
            journal,
            ref,
            finding.what,
            finding.seqs,
            finding.detail,
            identity_digest=identity if finding.what == "receipts_differ" else None,
            lookup_base_seq=current._resolution_seq if finding.what == "receipts_differ" else None,
        )
    return verdict


def _lookup_evidence_seen(op: Operation, what: str, identity: str | None, base_seq: int) -> bool:
    """Only identical positive evidence in the active resolution interval is replayed."""
    return identity is not None and any(
        finding.what == what
        and (finding.model_extra or {}).get("identity_digest") == identity
        and (finding.model_extra or {}).get("lookup_base_seq") == base_seq
        and (finding.what, tuple(finding.seqs)) not in op._resolved_conflicts
        for finding in op.conflicts
    )


async def _lookup_conflict(
    journal: EffectJournal,
    ref: dict[str, str],
    what: str,
    seqs: list[int],
    detail: str | None,
    *,
    identity_digest: str | None = None,
    lookup_base_seq: int | None = None,
):
    evidence = (
        {}
        if identity_digest is None
        else {
            "identity_digest": identity_digest,
            "lookup_base_seq": lookup_base_seq,
        }
    )
    finding = kinds.EffectConflictData(**ref, what=what, seqs=seqs, detail=detail, **evidence)
    await journal.append_effect(
        [
            Entry(
                id=kinds.conflict_entry_id(finding.record_id, what, seqs),
                kind=kinds.KIND_EFFECT_CONFLICT,
                data=finding.model_dump(mode="json"),
            )
        ]
    )


async def _lookup_verdict(
    effect: Effect, ctx: types.EffectContext, inquiry: types.Inquiry, journal: EffectJournal
) -> Verdict:
    """Bound the read without waiting indefinitely for a cancellation-resistant callback."""
    from qagents import tools

    async def call_lookup():
        fn = effect.lookup_fn
        if inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(type(fn).__call__):
            value = fn(ctx, inquiry)
        else:
            wrapped, raw = tools.start_thread("effect-lookup", partial(fn, ctx, inquiry))
            journal.track(raw)
            value = await wrapped
        return await value if inspect.isawaitable(value) else value

    async def invoke():
        token = ports._INQUIRY.set(inquiry)
        try:
            transport = ports.TRANSPORT.get()
            return (
                await call_lookup()
                if transport is None
                else await transport.lookup(effect, ctx, inquiry, call_lookup)
            )
        finally:
            ports._INQUIRY.reset(token)

    task = asyncio.create_task(invoke())
    journal.track(task)
    # Timed-out reads may finish later. Retrieve their exception without turning a
    # late result into proof, and retain them in the lease's drain accounting.
    task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
    try:
        done, _ = await asyncio.wait({task}, timeout=effect.timeout_s)
        if not done:
            task.cancel()
            return Unknown("Lookup timed out.")
        if task.cancelled():
            return Unknown("Lookup cancelled itself.")
        value = task.result()
        if not isinstance(value, (Found, NotYetVisible, Absent, Unknown)):
            return Unknown("Lookup returned an unsupported verdict.")
        if (
            isinstance(value, Unknown)
            and value.reason is not None
            and not isinstance(value.reason, str)
        ):
            return Unknown("Lookup returned an invalid reason.")
        return value
    except ConfigurationError:
        raise
    except asyncio.CancelledError:
        task.cancel()
        raise
    except Exception as error:
        return Unknown(f"{type(error).__name__}: {error}")


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
