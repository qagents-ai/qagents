"""Parent-side execution of a crash leg in an independent interpreter."""

from __future__ import annotations

import asyncio
import pickle
import sys
from dataclasses import replace
from importlib import import_module
from os.path import abspath
from pathlib import Path

from qagents.effects import _ports, _types
from qagents.effects._declare import Effect, ReachPoints
from qagents.errors import ConfigurationError, HeldEffectsError, JournalOutcomeUnknown, LeaseError
from qagents.results import RunResult

from ._remote import (
    CancelDriver,
    KillWithoutReply,
    OwnedLegProcess,
    RemoteProtocolError,
    WorldServer,
)
from ._transport import _DISPATCH, _Dispatch, _nothing
from ._world import Boundary, LegCrashed


def import_reference(value):
    if value is None:
        return None
    module = getattr(value, "__module__", None)
    name = getattr(value, "__qualname__", "")
    if not module or module == "__main__" or "<" in name:
        raise ConfigurationError("Subprocess build and scenario functions must be importable.")
    address = module, name
    try:
        resolved = resolve(address)
    except (ImportError, AttributeError) as error:
        raise ConfigurationError(
            "Subprocess callable does not resolve to its module name."
        ) from error
    if resolved is not value:
        raise ConfigurationError("Subprocess callable does not resolve to its module name.")
    return address


def resolve(address):
    value = import_module(address[0])
    for segment in address[1].split("."):
        value = getattr(value, segment)
    return value


def describe(effect):
    """Transfer the declaration data needed by the fake destination and report."""
    fields = (
        "name",
        "dedupe",
        "applies_within_s",
        "timeout_s",
        "rejected_on",
        "receipt_identity",
        "authoritative",
        "request_type",
        "receipt_type",
        "key_spec_digest",
        "request_schema_digest",
        "version",
    )
    description = {field: getattr(effect, field) for field in fields}
    description["lookup_fn"] = bool(effect.lookup_fn)
    description["_worker_receipt_schema"] = effect.receipt_type.model_json_schema()
    return description


def _effect(data):
    effect = object.__new__(Effect)
    effect.__dict__.update(data)
    effect.reached = ReachPoints(effect.name)
    return effect


class ProcessWorld:
    """Propagate a kill to the parent RPC dispatcher before it can reply."""

    def __init__(self, world):
        self.world = world
        self.cancellations = set()

    def cancel_driver(self, leg: int) -> None:
        self.cancellations.add(leg)

    def freeze(self, leg, *, zombie=False):
        if zombie:
            raise ConfigurationError("A terminated subprocess cannot be thawed.")
        self.world.legs[leg].dead = True
        self.world.loop.deactivate_leg(leg)
        raise LegCrashed(f"leg-{leg}")

    @property
    def boundary(self):
        return self.world.boundary

    @boundary.setter
    def boundary(self, value):
        self.world.boundary = value

    @property
    def interceptor(self):
        return self.world.interceptor

    @interceptor.setter
    def interceptor(self, value):
        self.world.interceptor = value

    def __getattr__(self, name):
        return getattr(self.world, name)


class ProcessLeg:
    def __init__(self, world, controller, destination, model, number, reached, answer):
        self.world, self.controller, self.destination = world, controller, destination
        self.number = number
        self.leg = world.legs[number]
        self.gated_model = model
        self.reached, self.answer = reached, answer
        self.effects, self.services, self.writes = {}, [], {}
        self.transport = destination.for_leg(number)
        self.active = {}
        self.lookups = {}
        self.service_iterators = {}
        self.iterators = {}
        self.next_iterator = 0
        self.result = None
        self.error = None
        self.paths = [abspath(path or ".") for path in sys.path]
        self.declarations = None

    async def dispatch(self, name, value):
        try:
            result = await self.handle(name, value)
            cancelled = self.destination.world.cancellations
            if self.number in cancelled:
                cancelled.remove(self.number)
                return CancelDriver(result)
            return result
        except LegCrashed as crash:
            return KillWithoutReply(str(crash))

    async def handle(self, name, value):
        if name == "describe":
            descriptions, tools, key, limits = value
            self.effects = {item["name"]: _effect(item) for item in descriptions}
            for effect in self.effects.values():
                self.destination.contract(effect)
            for effect_name, impl, key_arg in self.services:
                self.destination.service(
                    self.number, self.effects[effect_name], impl, key_arg=key_arg
                )
            self.gated_model.model.effect_tools = frozenset(tool[0] for tool in tools if tool[1])
            self.declarations = (self.effects, tools, key, limits)
        elif name == "service.register":
            self.services.append(value)
        elif name == "capabilities":
            return await self.gated_model.capabilities()
        elif name == "model.open":
            request, turn = value
            self.next_iterator += 1
            self.iterators[self.next_iterator] = self.gated_model.stream(request), turn
            return self.next_iterator
        elif name in {"model.next", "model.close"}:
            iterator, turn = self.iterators[value]
            token = _ports.MODEL_TURN.set(turn)
            try:
                if name == "model.close":
                    del self.iterators[value]
                    return await iterator.aclose()
                try:
                    return False, await anext(iterator)
                except StopAsyncIteration:
                    return True, None
            finally:
                _ports.MODEL_TURN.reset(token)
        elif name == "store.before":
            return await self.before_store(value)
        elif name in {"store.after", "store.aborted"}:
            boundary, lost = self.writes.pop(value)
            if name == "store.after":
                if lost:
                    raise OSError("acknowledgement lost")
                await self.at_boundary(replace(boundary, phase="after"))
        elif name == "fence":
            await self.transport.before_second_fence(self.effects[value])
        elif name == "send":
            effect, attempt = value
            return await self.transport.send(self.effects[effect], None, attempt, _nothing)
        elif name == "lookup":
            effect, inquiry = value
            return await self.transport.lookup(self.effects[effect], None, inquiry, _nothing)
        elif name == "send.start":
            name, attempt = value
            effect = self.effects[name]
            self.active[self._attempt_id(name, attempt)] = _Dispatch(
                self.number, effect, attempt, self.destination._fault(self.number, effect)
            )
            await self.edge("effect_send", name, "before")
        elif name == "send.end":
            name, attempt, completed = value
            self.active.pop(self._attempt_id(name, attempt))
            if completed:
                await self.edge("effect_send", name, "after")
        elif name == "lookup.start":
            effect, inquiry = value
            self.lookups[self._inquiry_id(effect, inquiry)] = _Dispatch(
                self.number, self.effects[effect], None, inquiry=inquiry
            )
            await self.edge("effect_lookup", effect, "before")
        elif name == "lookup.end":
            effect, inquiry, completed = value
            self.lookups.pop(self._inquiry_id(effect, inquiry))
            if completed:
                await self.edge("effect_lookup", effect, "after")
        elif name in {"service.call", "service.open"}:
            return await self.service_call(value, opened=name == "service.open")
        elif name == "service.next":
            value, iterator = self.service_iterators[value]
            try:
                return False, await self.service_call(value, iterator=iterator)
            except StopAsyncIteration:
                return True, None
        elif name == "service.close":
            _, iterator = self.service_iterators.pop(value)
            await iterator.aclose()
        elif name == "result":
            self.result = value
        elif name == "error":
            self.error = value
        elif name == "answer":
            return self.answer(value)
        elif name == "held_delivery":
            self.world.observe(self.number, name)
        elif name == "hit":
            self.controller.hit(next(plan for plan in self.controller.plans if plan.id == value))
        elif name == "finding":
            kind, detail = value
            self.world.findings.append({"kind": kind, "detail": detail, "leg": self.number})
        else:
            raise RemoteProtocolError(f"Unknown worker operation: {name}")
        return None

    @staticmethod
    def _attempt_id(name, attempt):
        return name, attempt.operation_tag, attempt.debug.number

    @staticmethod
    def _inquiry_id(name, inquiry):
        return name, inquiry.operation_tag

    async def service_call(self, value, *, opened=False, iterator=None):
        name, path, args, kwargs, (attempt, live), inquiry = value
        proxy = self.destination._clients[self.number, name]
        member = proxy._target
        for segment in path.split("."):
            if segment.startswith("_"):
                raise ConfigurationError("Private service members are not exposed to workers.")
            member = getattr(member, segment)
        if opened:
            if path in proxy._target.mutating:
                raise ConfigurationError("mutating service methods cannot be async generators")
            self.next_iterator += 1
            self.service_iterators[self.next_iterator] = value, member(*args, **kwargs)
            return self.next_iterator
        method = member
        if iterator is not None:
            method, args, kwargs = lambda: anext(iterator), (), {}
        dispatch = None
        if inquiry is not None:
            dispatch = self.lookups.get(self._inquiry_id(name, inquiry))
            if dispatch is not None:
                inquiry = dispatch.inquiry
        if attempt is not None:
            dispatch = self.active.get(self._attempt_id(name, attempt))
            if dispatch is not None:
                attempt = dispatch.attempt
        tokens = (
            (_DISPATCH, _DISPATCH.set(dispatch)),
            (_types._ATTEMPT, _types._ATTEMPT.set(attempt)),
            (_ports._INQUIRY, _ports._INQUIRY.set(inquiry)),
        )
        if live and attempt is not None:
            _ports._set_attempt_live(attempt, True)
        try:
            return await proxy._invoke(path, method, args, kwargs)
        finally:
            if live and attempt is not None:
                _ports._set_attempt_live(attempt, False)
            for var, token in reversed(tokens):
                var.reset(token)

    async def at_boundary(self, event):
        self.world.boundaries.append(event)
        await self.controller.at_boundary(event)

    async def edge(self, kind, detail, phase):
        self.leg.calls += 1
        await self.at_boundary(
            Boundary(
                self.number,
                self.leg.calls,
                phase,
                kind,
                detail,
                False,
                False,
                self.world.loop.time(),
            )
        )

    async def before_store(self, value):
        method, write, detail, entries = value
        self.leg.calls += 1
        if write:
            self.leg.writes += 1
        index = self.leg.writes if write else self.leg.calls
        boundary = Boundary(
            self.number,
            index,
            "before",
            method,
            detail,
            True,
            write,
            self.world.loop.time(),
            entries,
        )
        await self.at_boundary(boundary)
        call = replace(boundary, phase="call")
        proceed, result, lost, cancel, stall_s = True, None, False, False, None
        for plan in self.controller._matches(call):
            kind = plan.kind
            if kind in {"ack_lost_uncommitted", "lease_error"} or (
                kind == "lease_ack_lost" and plan.value == "uncommitted"
            ):
                self.controller.hit(plan)
                if kind == "lease_error":
                    raise LeaseError("injected lease renewal failure")
                raise OSError("acknowledgement lost before commit")
            if kind in {
                "ack_lost_committed",
                "lease_ack_lost",
                "stall",
                "cancel_during",
                "lease_renew_refused",
            }:
                self.controller.hit(plan)
                if kind == "stall":
                    stall_s = self.controller.faults.stall_s
                elif kind == "cancel_during":
                    cancel = True
                elif kind == "lease_renew_refused":
                    proceed, result = False, False
                else:
                    lost = True
        token = self.leg.calls
        self.writes[token] = boundary, lost
        return token, proceed, result, cancel, stall_s

    async def run(self, bootstrap):
        parent = None
        self.world.loop.deactivate_leg(self.number)
        server = WorldServer(
            self.world.loop,
            self.world.clock,
            {},
            reach=self.reached.append,
            verify_pid=lambda pid: parent.verify(pid),
            kill=lambda: parent.kill(),
        )
        for method in (
            "describe",
            "service.register",
            "capabilities",
            "model.open",
            "model.next",
            "model.close",
            "store.before",
            "store.after",
            "store.aborted",
            "fence",
            "send",
            "lookup",
            "send.start",
            "send.end",
            "lookup.start",
            "lookup.end",
            "service.call",
            "service.open",
            "service.next",
            "service.close",
            "result",
            "error",
            "answer",
            "held_delivery",
            "hit",
            "finding",
        ):
            server.handlers[method] = lambda value, method=method: self.dispatch(method, value)
        try:
            payload = {
                **bootstrap,
                "address": server.address,
                "authkey": server.authkey,
                "paths": self.paths,
            }
            try:
                parent = OwnedLegProcess(
                    Path(__file__).with_name("_worker_main.py"),
                    [],
                    directory=bootstrap["directory"] / f"leg-{self.number}",
                    bootstrap=payload,
                )
            except (AttributeError, pickle.PicklingError, TypeError) as error:
                raise ConfigurationError(
                    "Subprocess input must contain data or importable module-level objects."
                ) from error
            server.start()
            await server.wait(parent)
            if parent.killed:
                return None
            code = parent.process.wait(timeout=10)
            if code or not parent.verified or server.errors:
                raise RemoteProtocolError(
                    f"Worker exited without a verified result (exit {code}; {server.errors!r})."
                )
            if self.error:
                name, detail = self.error
                if name in {"LeaseError", "JournalOutcomeUnknown", "CancelledError", "OSError"}:
                    raise {
                        "LeaseError": LeaseError,
                        "JournalOutcomeUnknown": JournalOutcomeUnknown,
                        "CancelledError": asyncio.CancelledError,
                        "OSError": OSError,
                    }[name](detail)
                if name == "UnscriptedRequest":
                    from ._model import UnscriptedRequest

                    raise UnscriptedRequest(detail)
                if name == "ConfigurationError":
                    raise ConfigurationError(detail)
                raise RemoteProtocolError(f"Worker failed: {name}: {detail}")
            if self.result is None:
                raise RemoteProtocolError("Worker exited without returning a run result.")
            kind, value = self.result
            if kind == "held_error":
                raise HeldEffectsError(value)
            return RunResult.model_validate(value) if kind == "run" else value
        finally:
            if parent is not None:
                parent.close()
            await server.aclose()
            for iterator, _ in self.iterators.values():
                await iterator.aclose()
            for _, iterator in self.service_iterators.values():
                await iterator.aclose()
