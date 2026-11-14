"""Actual send/lookup execution over per-leg clients and one destination world."""

from __future__ import annotations

import contextvars
import inspect
import threading
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from qagents.effects import _ports
from qagents.effects._canonical import digest
from qagents.effects._declare import Effect
from qagents.effects._types import Attempt, EffectContext, Inquiry, current_attempt
from qagents.errors import ConfigurationError

from ._fakes import ContractFake, Fake
from ._faults import FaultController, FaultPlan
from ._world import World

_EFFECT_FAULTS = frozenset(
    {
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
)


@dataclass
class _Dispatch:
    leg: int
    effect: Effect
    attempt: Attempt | None
    fault: FaultPlan | None = None
    consumed: bool = False
    inquiry: Inquiry | None = None


_DISPATCH: contextvars.ContextVar[_Dispatch | None] = contextvars.ContextVar(
    "qagents_crash_dispatch", default=None
)


class HarnessTransport:
    """One point's contracts/service instances; every leg receives a fresh wrapper."""

    def __init__(
        self,
        world: World,
        *,
        fakes: Sequence[Effect | Fake] = (),
        mode: str = "auto",
        controller: FaultController | None = None,
        unknown_concurrent: str = "rejects",
    ) -> None:
        if mode not in {"auto", "fake", "client"}:
            raise ValueError("transport must be auto, fake or client")
        self.world, self.mode, self.controller = world, mode, controller
        self.unknown_concurrent = unknown_concurrent
        self.contracts: dict[str, ContractFake] = {}
        self._declarations: dict[str, tuple[Any, ...]] = {}
        self._implementations: dict[str, tuple[Any, str, Any]] = {}
        self._clients: dict[tuple[int, str], ServiceProxy] = {}
        self._sends: dict[tuple[int, str], int] = {}
        self.not_exercised: set[str] = set()
        for fake in fakes:
            wrapped = fake if isinstance(fake, Fake) else Fake(fake)
            if wrapped.effect.name in self.contracts:
                raise ConfigurationError("each fake effect may be declared only once")
            self.contracts[wrapped.effect.name] = ContractFake(
                world, wrapped, unknown_concurrent=unknown_concurrent
            )
            self._declarations[wrapped.effect.name] = self._declaration(wrapped.effect)

    @staticmethod
    def _declaration(effect: Effect) -> tuple[Any, ...]:
        return (
            effect.key_spec_digest,
            effect.request_schema_digest,
            digest(effect.receipt_type.model_json_schema()),
            effect.dedupe,
            effect.version,
            effect.applies_within_s,
            effect.authoritative,
            effect.receipt_identity,
            effect.rejected_on,
        )

    def contract(self, effect: Effect) -> ContractFake:
        found = self.contracts.get(effect.name)
        if found is None:
            found = ContractFake(self.world, effect, unknown_concurrent=self.unknown_concurrent)
            self.contracts[effect.name] = found
            self._declarations[effect.name] = self._declaration(effect)
        if self._declarations[effect.name] != self._declaration(effect):
            raise ConfigurationError(f"effect {effect.name!r} changed between process legs")
        return found

    def for_leg(self, leg: int) -> LegTransport:
        made = LegTransport(self, leg)
        self.world.keep_alive.append(made)
        return made

    def service(
        self, leg: int, effect: Effect, impl: Any, *, key_arg: str = "idempotency_key"
    ) -> ServiceProxy:
        if not isinstance(key_arg, str) or not key_arg:
            raise ConfigurationError("key_arg must be a nonempty argument name")
        self.contract(effect)
        stored = self._implementations.get(effect.name)
        if stored is None:
            try:
                instance = impl(clock=self.world.clock, keys=False)
            except TypeError as error:
                raise ConfigurationError(
                    "service implementation must accept clock= and keys="
                ) from error
            paths = getattr(instance, "mutating", None)
            errors = getattr(instance, "contract_errors", None)
            if not isinstance(paths, frozenset) or not all(
                isinstance(path, str) and path for path in paths
            ):
                raise ConfigurationError("service mutating must be a frozenset of method paths")
            if not isinstance(errors, Mapping) or not all(
                isinstance(name, str) and callable(factory) for name, factory in errors.items()
            ):
                raise ConfigurationError(
                    "service contract_errors must map names to exception factories"
                )
            stored = self._implementations[effect.name] = impl, key_arg, instance
            self.world._services[effect.name] = instance
            self.world.keep_alive.append(instance)
        elif stored[0] is not impl or stored[1] != key_arg:
            raise ConfigurationError("service implementation/key_arg changed between legs")
        identity = leg, effect.name
        if identity not in self._clients:
            self._clients[identity] = ServiceProxy(self, leg, effect, stored[2], "", key_arg)
            self.world.keep_alive.append(self._clients[identity])
        return self._clients[identity]

    def _fault(self, leg: int, effect: Effect) -> FaultPlan | None:
        identity = leg, effect.name
        self._sends[identity] = self._sends.get(identity, 0) + 1
        if self.controller is None:
            return None
        return next(
            (
                plan
                for plan in self.controller.plans
                if plan.leg == leg
                and plan.index == self._sends[identity]
                and plan.method == effect.name
                and plan.kind in _EFFECT_FAULTS
                and plan.id not in self.controller.triggered
            ),
            None,
        )

    async def apply(
        self,
        effect: Effect,
        attempt: Attempt,
        *,
        leg: int,
        key: str | None,
        key_passed: bool = True,
        call: Callable | None = None,
        errors: Mapping[str, Callable] | None = None,
        dispatch: _Dispatch | None = None,
    ) -> Any:
        fake = self.contract(effect)
        if dispatch is not None and dispatch.fault is not None and not dispatch.consumed:
            dispatch.consumed = True
            assert self.controller is not None
            return await fake.send_faulted(
                attempt,
                leg=leg,
                key=key,
                key_passed=key_passed,
                apply=call,
                errors=errors,
                fault=dispatch.fault,
                controller=self.controller,
            )
        return await fake.send(
            attempt, leg=leg, key=key, key_passed=key_passed, apply=call, errors=errors
        )


class LegTransport:
    def __init__(self, shared: HarnessTransport, leg: int) -> None:
        self.shared, self.leg = shared, leg

    def _client(self, effect: Effect) -> bool:
        registered = (self.leg, effect.name) in self.shared._clients
        if self.shared.mode == "client" and not registered:
            raise ConfigurationError(f"client transport requires env.service for {effect.name}")
        return self.shared.mode != "fake" and registered

    async def send(
        self, effect: Effect, ctx: EffectContext, attempt: Attempt, call_send: Callable
    ) -> Any:
        self.shared.contract(effect)
        dispatch = _Dispatch(self.leg, effect, attempt, self.shared._fault(self.leg, effect))
        token = _DISPATCH.set(dispatch)
        try:
            if self._client(effect):
                return await self.shared.world.gate(
                    self.leg, call_send, kind="effect_send", detail=effect.name
                )
            self.shared.not_exercised.add(effect.name)
            return await self.shared.world.gate(
                self.leg,
                lambda: self.shared.apply(
                    effect, attempt, leg=self.leg, key=attempt.idempotency_key, dispatch=dispatch
                ),
                kind="effect_send",
                detail=effect.name,
            )
        finally:
            _DISPATCH.reset(token)

    async def lookup(
        self, effect: Effect, ctx: EffectContext, inquiry: Inquiry, call_lookup: Callable
    ) -> Any:
        token = _DISPATCH.set(_Dispatch(self.leg, effect, None, inquiry=inquiry))
        try:
            if self._client(effect):
                return await self.shared.world.gate(
                    self.leg, call_lookup, kind="effect_lookup", detail=effect.name
                )
            self.shared.not_exercised.add(effect.name)
            return await self.shared.world.gate(
                self.leg,
                lambda: self.shared.contract(effect).lookup(inquiry),
                kind="effect_lookup",
                detail=effect.name,
            )
        finally:
            _DISPATCH.reset(token)


class ServiceProxy:
    """A leg-local facade; nested method paths are matched against the service contract."""

    def __init__(
        self,
        shared: HarnessTransport,
        leg: int,
        effect: Effect,
        target: Any,
        path: str,
        key_arg: str,
    ) -> None:
        self._shared, self._leg, self._effect = shared, leg, effect
        self._target, self._path, self._key_arg = target, path, key_arg

    def __getattr__(self, name: str) -> Any:
        member = getattr(self._target, name)
        path = f"{self._path}.{name}" if self._path else name
        if inspect.isasyncgenfunction(member):

            async def generator(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
                instance = self._shared.world.service(self._effect)
                if path in instance.mutating:
                    raise ConfigurationError("mutating service methods cannot be async generators")
                iterator = member(*args, **kwargs)
                try:
                    while True:
                        try:
                            value = await self._invoke(path, lambda: anext(iterator), (), {})
                        except StopAsyncIteration:
                            return
                        yield value
                finally:
                    await iterator.aclose()

            return generator
        if inspect.iscoroutinefunction(member):

            async def asynchronous(*args: Any, **kwargs: Any) -> Any:
                return await self._invoke(path, member, args, kwargs)

            return asynchronous
        if callable(member):

            def synchronous(*args: Any, **kwargs: Any) -> Any:
                loop = self._shared.world.loop
                if threading.get_ident() == loop._thread_id:
                    raise ConfigurationError(
                        "sync service methods must be called from a sync send worker"
                    )
                return loop.call_from_thread(self._invoke(path, member, args, kwargs))

            return synchronous
        if member is None or isinstance(member, str | int | float | bool | bytes):
            return member
        return ServiceProxy(self._shared, self._leg, self._effect, member, path, self._key_arg)

    async def _invoke(self, path: str, method: Callable, args: tuple, kwargs: dict) -> Any:
        world = self._shared.world
        instance = world.service(self._effect)
        mutating = path in instance.mutating
        attempt, inquiry, dispatch = current_attempt(), _ports.current_inquiry(), _DISPATCH.get()
        active = attempt is not None and _ports.attempt_live(attempt)
        binding_wrong = dispatch is not None and (
            dispatch.leg != self._leg
            or dispatch.attempt is not attempt
            or (dispatch.inquiry is not None and dispatch.inquiry is not inquiry)
        )
        if (
            (mutating and not active)
            or (not mutating and not active and inquiry is None)
            or (attempt is not None and world.loop.time() >= attempt.deadline)
            or (dispatch is not None and dispatch.effect.name != self._effect.name)
            or binding_wrong
        ):
            world.findings.append(
                {
                    "kind": "a2_violation",
                    "leg": self._leg,
                    "detail": (
                        f"service call outside its live effect boundary: {self._effect.name}.{path}"
                    ),
                }
            )
        if mutating and attempt is None:
            raise ConfigurationError(
                "mutating service call requires an Attempt to identify its payload"
            )
        if dispatch is not None and dispatch.effect.name != self._effect.name:
            raise ConfigurationError("service wrapper belongs to a different effect")
        if binding_wrong:
            raise ConfigurationError(
                "service wrapper or context belongs to a different leg/boundary"
            )

        async def call() -> Any:
            result = method(*args, **kwargs)
            return await result if inspect.isawaitable(result) else result

        async def invoke() -> Any:
            if not mutating:
                return await call()
            try:
                key = (
                    inspect.signature(method)
                    .bind_partial(*args, **kwargs)
                    .arguments.get(self._key_arg, kwargs.get(self._key_arg))
                )
            except (TypeError, ValueError):
                key = kwargs.get(self._key_arg)
            passed = isinstance(key, str) and key == attempt.idempotency_key
            if not passed:
                world.findings.append(
                    {
                        "kind": "key_not_passed",
                        "leg": self._leg,
                        "detail": f"{self._effect.name}.{path}",
                    }
                )
            return await self._shared.apply(
                self._effect,
                attempt,
                leg=self._leg,
                key=key if isinstance(key, str) else None,
                key_passed=passed,
                call=call,
                errors=instance.contract_errors,
                dispatch=dispatch,
            )

        return await world.gate(
            self._leg,
            invoke,
            kind="service_call",
            detail=f"{self._effect.name}.{path}",
            mutating=mutating,
        )
