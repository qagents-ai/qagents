"""Destination contract models; application identity is independent of delivery keys."""

from __future__ import annotations

import asyncio
import copy
import types
from collections.abc import AsyncIterator, Awaitable, Callable, Hashable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import Enum
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError

from qagents.effects import _declare
from qagents.effects._canonical import digest
from qagents.effects._declare import Dedupe, Effect, Unique
from qagents.effects._types import Absent, Attempt, Found, Inquiry, NotYetVisible
from qagents.errors import ConfigurationError, EffectRejected, ToolError

from ._clock import ManualClock, _finite
from ._faults import FaultController, FaultPlan
from ._invariants import JournalSnapshot
from ._world import Application, World


class Fake:
    """An effect's destination model, with optional identity and receipt factories."""

    def __init__(
        self,
        effect: Effect,
        *,
        identity: Callable[[Any], Hashable] | None = None,
        receipt: Callable[[Any, int], Any] | None = None,
    ) -> None:
        if not isinstance(effect, Effect):
            raise TypeError("Fake requires a declared Effect")
        if identity is not None and not callable(identity):
            raise TypeError("identity must be callable")
        if receipt is not None and not callable(receipt):
            raise TypeError("receipt must be callable")
        self.effect, self.identity, self.receipt = effect, identity, receipt


class ContractRefusal(EffectRejected):
    """A definite simulated refusal when no client-specific exception was supplied."""


def _field_value(annotation: Any, name: str, sequence: int, depth: int) -> Any:
    if depth > 12:
        raise ConfigurationError("Recursive receipt schema needs Fake(receipt=...)")
    origin, arguments = get_origin(annotation), get_args(annotation)
    if origin is Literal:
        return arguments[0]
    if origin in (Union, types.UnionType):
        choices = [item for item in arguments if item is not type(None)]
        return _field_value(choices[0], name, sequence, depth + 1) if choices else None
    if annotation is str:
        return f"{name}_{sequence}"
    if annotation is int:
        return sequence
    if annotation is float:
        return float(sequence)
    if annotation is bool:
        return True
    if annotation is bytes:
        return f"{name}_{sequence}".encode()
    if annotation is datetime:
        return datetime(2000, 1, 1, tzinfo=UTC)
    if annotation is date:
        return date(2000, 1, 1)
    if origin in (list, set, frozenset, tuple, dict):
        return origin()
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return next(iter(annotation))
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _synthesize(annotation, None, sequence, depth + 1)
    raise ConfigurationError(f"Receipt field {name!r} needs Fake(receipt=...)")


def _synthesize(
    model: type[BaseModel], request: BaseModel | None, sequence: int, depth: int = 0
) -> BaseModel:
    values = {}
    for name, info in model.model_fields.items():
        source = type(request).model_fields.get(name) if request is not None else None
        if source is not None and source.annotation == info.annotation:
            values[name] = copy.deepcopy(getattr(request, name))
        elif not info.is_required():
            continue  # let Pydantic apply defaults/factories
        else:
            values[name] = _field_value(info.annotation, name, sequence, depth)
    try:
        return model.model_validate(values)
    except ValidationError as error:
        raise ConfigurationError(
            "The receipt schema needs explicit values; provide Fake(receipt=...)"
        ) from error


@dataclass
class _Cached:
    first_arrival: float
    payload_digest: str
    done: asyncio.Event = field(default_factory=asyncio.Event)
    receipt: Any = None
    error: BaseException | None = None
    applied: bool = False
    retain_error: bool = False


class ContractFake:
    """One destination world for one effect, shared across simulated process legs."""

    def __init__(
        self,
        world: World,
        fake: Effect | Fake,
        *,
        unknown_concurrent: Literal["rejects", "waits"] = "rejects",
    ) -> None:
        if unknown_concurrent not in {"rejects", "waits"}:
            raise ValueError("unknown concurrency needs an explicit rejects or waits scenario")
        self.world = world
        self.fake = fake if isinstance(fake, Fake) else Fake(fake)
        self.effect = self.fake.effect
        self.unknown_concurrent = unknown_concurrent
        self._keys: dict[str, _Cached] = {}
        self._receipts: list[tuple[str, str, Any]] = []
        self._sequence = 0
        self.fault_model: list[str] = []
        self.pending_destinations: set[asyncio.Task[Any]] = set()
        self._hidden_until: dict[str, float] = {}

    def _refusal(
        self, kind: str, errors: Mapping[str, Callable[[], BaseException]] | None
    ) -> BaseException:
        if errors is not None and kind in errors:
            error = errors[kind]()
            if not isinstance(error, BaseException):
                raise ConfigurationError("contract_errors factories must return exceptions")
            return error
        if errors is not None:
            note = f"{kind} refusal not exercised through the client transport: {self.effect.name}"
            if note not in self.fault_model:
                self.fault_model.append(note)
        if kind == "duplicate" and isinstance(self.effect.dedupe, Unique):
            exception = self.effect.dedupe.on_duplicate[0]
            try:
                return exception("simulated duplicate key")
            except TypeError as error:
                raise ConfigurationError(
                    "Duplicate exception requires a contract_errors['duplicate'] factory"
                ) from error
        return ContractRefusal(f"destination refused {kind}")

    def _identity(self, attempt: Attempt) -> Hashable:
        request = attempt.request.model_dump(mode="json")
        identity = (
            self.fake.identity(attempt.request.model_copy(deep=True))
            if self.fake.identity is not None
            else digest(request)
        )
        try:
            hash(identity)
        except TypeError as error:
            raise ConfigurationError("Fake identity must return a hashable value") from error
        return identity

    def _application(
        self,
        attempt: Attempt,
        *,
        leg: int,
        key: str | None,
        key_passed: bool,
        replayed: bool,
        identity: Hashable,
        issued_at: float,
        issued_store_time: float,
        snapshot: JournalSnapshot | None,
    ) -> None:
        request = attempt.request.model_dump(mode="json")
        if self.world.sealed and not replayed:
            self.world.findings.append(
                {"kind": "post_seal_mutation", "leg": leg, "detail": self.effect.name}
            )
        if (
            not replayed
            and self.effect.applies_within_s is not None
            and self.world.loop.time() > issued_at + self.effect.applies_within_s + 1e-9
        ):
            self.world.findings.append(
                {
                    "kind": "a2_violation",
                    "leg": leg,
                    "detail": f"{self.effect.name} applied after its declared bound",
                }
            )
        application = Application(
            self.effect.name,
            key,
            digest(request),
            attempt.debug.number,
            attempt.debug.epoch,
            leg,
            self.world.loop.time(),
            identity,
            replayed,
            key_passed,
        )
        index = len(self.world._applications)
        self.world._applications.append((application, copy.deepcopy(request)))
        self.world.observe(leg, "application", self.effect.name)
        self.world.record_application_evidence(
            index, self.effect, attempt, snapshot, issued_store_time
        )

    def _receipt(self, request: BaseModel, sequence: int) -> Any:
        if self.fake.receipt is not None:
            return self.fake.receipt(request.model_copy(deep=True), sequence)
        return _synthesize(self.effect.receipt_type, request, sequence)

    async def send(
        self,
        attempt: Attempt,
        *,
        leg: int,
        apply: Callable[[], Awaitable[Any]] | None = None,
        after_apply: Callable[[Any], Awaitable[Any]] | None = None,
        key: str | None,
        key_passed: bool = True,
        errors: Mapping[str, Callable[[], BaseException]] | None = None,
        destination_delay_s: float | None = None,
        on_issue: Callable[[], None] | None = None,
        network_response_loss: bool = False,
        issued_at: float | None = None,
    ) -> Any:
        payload = digest(attempt.request.model_dump(mode="json"))
        issued_at = self.world.loop.time() if issued_at is None else _finite(issued_at, "issued_at")
        if issued_at > self.world.loop.time():
            raise ValueError("issue time cannot be in the future")
        issued_store_time = self.world.clock.store_time()
        identity = self._identity(attempt)  # validate before invoking a mutating client
        snapshot = await self.world.snapshot()
        contract = self.effect.dedupe
        protected = key is not None and isinstance(contract, Dedupe | Unique)
        cached = self._keys.get(key) if protected else None
        while cached is not None:
            now = self.world.loop.time()
            if isinstance(contract, Dedupe):
                if now >= cached.first_arrival + contract.window_s:
                    break
                if contract.binds_params and payload != cached.payload_digest:
                    raise self._refusal("changed_params", errors)
                if not cached.done.is_set():
                    behavior = (
                        self.unknown_concurrent
                        if contract.concurrent == "unknown"
                        else contract.concurrent
                    )
                    if behavior == "rejects":
                        raise self._refusal("concurrent", errors)
                    await cached.done.wait()
                    cached = self._keys.get(key)
                    continue
                if cached.error is not None:
                    if cached.retain_error:
                        raise cached.error
                    break
                self._application(
                    attempt,
                    leg=leg,
                    key=key,
                    key_passed=key_passed,
                    replayed=True,
                    identity=identity,
                    issued_at=issued_at,
                    issued_store_time=issued_store_time,
                    snapshot=snapshot,
                )
                return copy.deepcopy(cached.receipt)
            if not cached.done.is_set():
                await cached.done.wait()
                cached = self._keys.get(key)
                continue
            if cached.applied:
                raise self._refusal("duplicate", errors)
            break

        if on_issue is not None:
            on_issue()
        arrival = self.world.loop.time()
        if (
            cached is not None
            and isinstance(contract, Dedupe)
            and arrival < cached.first_arrival + contract.window_s
        ):
            arrival = cached.first_arrival
        cached = _Cached(arrival, payload)
        if protected:
            self._keys[key] = cached
        self._sequence += 1
        sequence = self._sequence

        async def destination() -> Any:
            try:
                if destination_delay_s is not None:
                    await asyncio.sleep(destination_delay_s)
                receipt = (
                    await apply() if apply is not None else self._receipt(attempt.request, sequence)
                )
                self._application(
                    attempt,
                    leg=leg,
                    key=key,
                    key_passed=key_passed,
                    replayed=False,
                    identity=identity,
                    issued_at=issued_at,
                    issued_store_time=issued_store_time,
                    snapshot=snapshot,
                )
                cached.applied = True
                self._receipts.append((attempt.operation_tag, payload, copy.deepcopy(receipt)))
                cached.receipt = copy.deepcopy(receipt)
                if network_response_loss:
                    # A lost client response does not keep destination execution in flight.
                    cached.done.set()
                if after_apply is not None:
                    receipt = await after_apply(receipt)
                cached.receipt = copy.deepcopy(receipt)
                return receipt
            except BaseException as error:
                if cached.applied and (
                    network_response_loss or isinstance(error, asyncio.CancelledError)
                ):
                    # Client cancellation/loss is not a destination's cached error response.
                    cached.error = None
                    cached.retain_error = False
                    raise
                cached.error = (
                    ConnectionError("destination request interrupted")
                    if isinstance(error, asyncio.CancelledError)
                    else error
                )
                # A transport-level no answer is an error, never a rejection (SPEC 5.1).
                rejected = not _declare._no_answer(error) and isinstance(
                    error, (EffectRejected, *self.effect.rejected_on)
                )
                cached.retain_error = (
                    isinstance(contract, Dedupe)
                    and (contract.replays_rejections if rejected else contract.replays_errors)
                ) or (isinstance(contract, Unique) and cached.applied)
                if isinstance(error, asyncio.CancelledError):
                    cached.retain_error = False
                raise
            finally:
                cached.done.set()

        if destination_delay_s is None:
            return await destination()
        pending = asyncio.create_task(destination())
        self.pending_destinations.add(pending)
        self.world.keep_alive.append(pending)

        def ended(task: asyncio.Task[Any]) -> None:
            self.pending_destinations.discard(task)
            if task.cancelled():
                self.world.loop.inconclusive_reasons.append(
                    "destination application pending at teardown"
                )
            elif task.exception() is not None:
                self.world.loop.inconclusive_reasons.append(
                    "detached destination application failed"
                )

        pending.add_done_callback(ended)
        raise ConnectionError("response lost while the destination request remains in flight")

    async def send_faulted(
        self,
        attempt: Attempt,
        *,
        leg: int,
        key: str | None,
        fault: FaultPlan,
        controller: FaultController,
        apply: Callable[[], Awaitable[Any]] | None = None,
        key_passed: bool = True,
        errors: Mapping[str, Callable[[], BaseException]] | None = None,
    ) -> Any:
        """Exercise one permitted destination fault, retaining its actual application."""
        kind = fault.kind
        issued_at = self.world.loop.time()
        allowed = {
            "reject",
            "rejected_on",
            "fail_before_commit",
            "commit_then_lose_response",
            "commit_then_error_response",
            "commit_then_reject_shaped_error",
            "commit_then_crash",
            "commit_late",
            "not_yet_visible_then_commit",
            "unstorable_receipt",
            "duplicate_key",
            "slow_arrival",
            "cancel_during",
        }
        if kind not in allowed:
            raise ValueError("not an effect fault")
        if kind == "duplicate_key" and not isinstance(self.effect.dedupe, Unique):
            raise ValueError("duplicate_key requires a Unique declaration")
        rejected_error = None
        if kind == "rejected_on":
            rejected = fault.value
            if rejected not in self.effect.rejected_on:
                raise ValueError("rejected_on fault must select an actual declared exception type")
            try:
                rejected_error = rejected("injected declared refusal")
            except TypeError as error:
                raise ConfigurationError(
                    "Cannot construct this declared refusal exception"
                ) from error

        delay = None
        if kind in {"commit_late", "not_yet_visible_then_commit"}:
            bounds = []
            if self.effect.applies_within_s is not None:
                bounds.append(self.effect.applies_within_s)
            if isinstance(self.effect.dedupe, Dedupe):
                bounds.append(self.effect.dedupe.window_s)
            bound = min(bounds) if bounds else None
            delay = _finite(fault.value if fault.value is not None else (bound or 1), "late delay")
            if delay <= 0 or (bound is not None and delay > bound):
                raise ValueError(
                    "late application must stay within the declared application/window bound"
                )
        elif kind == "slow_arrival":
            delay_to_arrival = max(0, attempt.deadline - self.world.loop.time() - 0.001)
            if (
                self.effect.applies_within_s is not None
                and delay_to_arrival > self.effect.applies_within_s
            ):
                raise ValueError("slow arrival would exceed applies_within_s")
            controller.hit(fault)
            await asyncio.sleep(delay_to_arrival)
        elif kind == "cancel_during":
            controller.hit(fault)
            self.world.cancel_driver(leg)

        async def before() -> Any:
            if kind in {"reject", "rejected_on", "fail_before_commit"}:
                controller.hit(fault)
                if kind == "reject":
                    raise EffectRejected("injected definite refusal")
                if kind == "fail_before_commit":
                    raise ConnectionError("injected failure before application")
                assert rejected_error is not None
                raise rejected_error
            return (
                await apply()
                if apply is not None
                else self._receipt(attempt.request, self._sequence)
            )

        def issued() -> None:
            if kind in {"commit_late", "not_yet_visible_then_commit"}:
                controller.hit(fault)
                if kind == "not_yet_visible_then_commit":
                    self._hidden_until[attempt.operation_tag] = self.world.loop.time() + delay

        async def after(receipt: Any) -> Any:
            if kind in {
                "commit_then_lose_response",
                "commit_then_error_response",
                "commit_then_reject_shaped_error",
                "commit_then_crash",
                "unstorable_receipt",
                "duplicate_key",
            }:
                controller.hit(fault)
                if kind == "commit_then_lose_response":
                    await asyncio.sleep(max(0, attempt.deadline - self.world.loop.time()))
                    raise TimeoutError("response lost after application")
                if kind == "commit_then_error_response":
                    raise ConnectionError("error response after application")
                if kind == "commit_then_reject_shaped_error":
                    raise ToolError("refused")
                if kind == "commit_then_crash":
                    self.world.freeze(leg)
                    await self.world._check_gate(leg)
                if kind == "duplicate_key":
                    # The same live send may internally retry a request that applied.
                    raise self._refusal("duplicate", errors)
                if kind == "unstorable_receipt":
                    if fault.value == "non_json" or not isinstance(receipt, BaseModel):
                        return {"not_json": object()}
                    for name, field_info in type(receipt).model_fields.items():
                        if field_info.annotation is str:
                            return receipt.model_copy(update={name: "x" * (17 * 1024)})
                    return {"not_json": object()}
            return receipt

        return await self.send(
            attempt,
            leg=leg,
            apply=before
            if apply is not None or kind in {"reject", "rejected_on", "fail_before_commit"}
            else None,
            after_apply=after,
            key=key,
            key_passed=key_passed,
            errors=errors,
            destination_delay_s=delay,
            on_issue=issued,
            network_response_loss=kind in {"commit_then_lose_response", "commit_then_crash"},
            issued_at=issued_at,
        )

    async def lookup(self, inquiry: Inquiry, *, miss: bool = False) -> Any:
        if miss and self.effect.authoritative:
            raise ValueError("authoritative lookup cannot hide an already visible application")
        hidden = self._hidden_until.get(inquiry.operation_tag)
        if hidden is not None and self.world.loop.time() < hidden:
            return NotYetVisible(retry_after_s=hidden - self.world.loop.time())
        if not miss:
            for tag, _, receipt in reversed(self._receipts):
                if tag == inquiry.operation_tag:
                    return Found(copy.deepcopy(receipt), tag=tag)
        return Absent()


@dataclass(frozen=True)
class _StripeRefund:
    id: str
    charge: str
    amount: int
    currency: str
    status: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class _StripePage:
    data: list[_StripeRefund]
    has_more: bool


@dataclass
class _StripeResponse:
    arrival: float
    parameters: str
    running: bool = True
    value: _StripeRefund | None = None
    error: BaseException | None = None


class FakeStripeRefunds:
    """Offline refund client for the documented Stripe API v1 contract.

    This narrow fake models refunds, not the entire SDK. ``ch_A`` starts with 10000
    EUR minor units; seed_charge supplies explicit fixture charges before use.
    With keys=False, the harness transport alone arbitrates idempotency. Listing
    copies rows newest first in pages of ten. Sources checked 2026-11-13:
    https://docs.stripe.com/api/idempotent_requests
    https://docs.stripe.com/api/refunds/create
    https://docs.stripe.com/api/refunds/list
    """

    class InvalidRequest(Exception):
        """Validation or remaining-amount refusal before execution."""

    class IdempotencyConflict(Exception):
        """The same key supplied different parameters."""

    class ConcurrentRequest(Exception):
        """The same key is still executing; no new response is cached."""

    class ServerError(Exception):
        """A controlled execution-time error used by offline contract probes."""

    mutating = frozenset({"create_refund", "refunds.create"})
    contract_errors = {
        "concurrent": lambda: FakeStripeRefunds.ConcurrentRequest("key is executing"),
        "changed_params": lambda: FakeStripeRefunds.IdempotencyConflict("parameters changed"),
        "duplicate": lambda: FakeStripeRefunds.IdempotencyConflict("key already applied"),
    }

    def __init__(self, *, clock: ManualClock | None = None, keys: bool = True) -> None:
        if type(keys) is not bool:
            raise TypeError("keys must be bool")
        self.clock, self.keys = clock or ManualClock(), keys
        self.refunds = self
        self._charges: dict[str, tuple[int, str]] = {}
        self._refunds: list[_StripeRefund] = []
        self._responses: dict[str, _StripeResponse] = {}
        self._next_execution_error: BaseException | None = None
        self.pages_read = 0
        self.seed_charge("ch_A", amount=10000, currency="eur")

    def seed_charge(self, charge: str, *, amount: int, currency: str = "eur") -> None:
        """Define parent-side fixture data; do not call through an effect client."""
        if not isinstance(charge, str) or not charge or charge in self._charges:
            raise ValueError("charge fixture needs a new nonempty identifier")
        if type(amount) is not int or amount <= 0:
            raise ValueError("charge amount must be positive integer minor units")
        if not isinstance(currency, str) or len(currency) != 3 or not currency.isalpha():
            raise ValueError("currency fixture needs a three-letter code")
        self._charges[charge] = amount, currency.lower()

    def inject_execution_error(self, error: BaseException) -> None:
        """Offline probe control: fail the next validated execution, before mutation."""
        if not isinstance(error, Exception):
            raise TypeError("execution error must be an ordinary Exception")
        if self._next_execution_error is not None:
            raise ValueError("an execution error is already queued")
        self._next_execution_error = error

    def _charge(self, charge: str) -> tuple[int, str]:
        if not isinstance(charge, str) or charge not in self._charges:
            raise self.InvalidRequest("unknown charge")
        return self._charges[charge]

    async def create_refund(
        self,
        *,
        charge: str,
        amount: int | None = None,
        idempotency_key: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> _StripeRefund:
        total, currency = self._charge(charge)
        if idempotency_key is not None and (
            not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key) <= 255
        ):
            raise self.InvalidRequest("idempotency key must contain 1 to 255 characters")
        if metadata is not None and (
            not isinstance(metadata, Mapping)
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items())
        ):
            raise self.InvalidRequest("metadata must map strings to strings")
        if amount is not None and (type(amount) is not int or amount <= 0):
            raise self.InvalidRequest("amount must be positive integer minor units")
        parameters = digest({"charge": charge, "amount": amount, "metadata": dict(metadata or {})})
        now = self.clock.monotonic()
        key = idempotency_key if self.keys else None
        prior = self._responses.get(key) if key is not None else None
        if prior is not None and now < prior.arrival + 86400:
            if parameters != prior.parameters:
                raise self.IdempotencyConflict("parameters changed")
            if prior.running:
                raise self.ConcurrentRequest("key is executing")
            if prior.error is not None:
                raise prior.error
            return copy.deepcopy(prior.value)

        refunded = sum(row.amount for row in self._refunds if row.charge == charge)
        remaining = total - refunded
        actual_amount = remaining if amount is None else amount
        if actual_amount <= 0 or actual_amount > remaining:
            raise self.InvalidRequest("amount exceeds the remaining unrefunded charge")
        response = _StripeResponse(now, parameters)
        if key is not None:
            self._responses[key] = response
        try:
            await asyncio.sleep(0)  # exposes concurrent same-key requests deterministically
            if self._next_execution_error is not None:
                error, self._next_execution_error = self._next_execution_error, None
                raise error
            # Another distinct key may have consumed the balance while this request yielded.
            remaining = total - sum(row.amount for row in self._refunds if row.charge == charge)
            if actual_amount > remaining:
                raise self.InvalidRequest("amount exceeds the remaining unrefunded charge")
            response.value = _StripeRefund(
                f"re_{len(self._refunds) + 1}",
                charge,
                actual_amount,
                currency,
                "succeeded",
                dict(metadata or {}),
            )
            self._refunds.append(response.value)
            return copy.deepcopy(response.value)
        except Exception as error:
            response.error = error
            raise
        except BaseException:
            # No application occurred before our only suspension; cancellation is not a
            # destination response and may be retried with the original key.
            if key is not None and self._responses.get(key) is response:
                self._responses.pop(key)
            raise
        finally:
            response.running = False

    async def create(self, **kwargs: Any) -> _StripeRefund:
        """Nested ``client.refunds.create`` alias for the same client surface."""
        return await self.create_refund(**kwargs)

    async def list_refunds(
        self,
        *,
        charge: str,
        limit: int = 10,
        starting_after: str | None = None,
    ) -> _StripePage:
        self._charge(charge)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise self.InvalidRequest("limit must be an integer from 1 to 100")
        rows = [row for row in reversed(self._refunds) if row.charge == charge]
        if starting_after is not None:
            positions = [index for index, row in enumerate(rows) if row.id == starting_after]
            if not positions:
                raise self.InvalidRequest("unknown pagination cursor")
            rows = rows[positions[0] + 1 :]
        self.pages_read += 1
        return _StripePage(copy.deepcopy(rows[:limit]), len(rows) > limit)

    async def iter_refunds(self, *, charge: str) -> AsyncIterator[_StripeRefund]:
        after = None
        while True:
            page = await self.list_refunds(charge=charge, starting_after=after)
            for row in page.data:
                yield row
            if not page.has_more:
                return
            after = page.data[-1].id
