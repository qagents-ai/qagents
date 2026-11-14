"""Per-leg object construction and evidence of shared process-local state."""

from __future__ import annotations

import inspect
import types
from collections.abc import AsyncIterator, Callable, Iterable
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any
from uuid import UUID

from qagents.agent import Agent
from qagents.effects._declare import Dedupe, Effect, Key, Unique
from qagents.errors import ConfigurationError
from qagents.models.base import Model, ModelRequest, ModelSettings, ModelStreamEvent
from qagents.models.recording import request_fingerprint
from qagents.permissions import Permissions
from qagents.session import SessionStore
from qagents.tools import Tool

from ._clock import ManualClock
from ._invariants import ModelEvidence
from ._world import World

if TYPE_CHECKING:
    from ._transport import HarnessTransport


class CrashEnv:
    """The fresh process-local resources supplied to build(env) for one leg."""

    leg: int
    worker_id: str
    recovering: bool
    clock: ManualClock
    store: SessionStore
    model: Model
    session_id: str
    seed: int
    directory: Path

    def __init__(
        self,
        *,
        world: World,
        transport: HarnessTransport,
        leg: int,
        worker_id: str,
        recovering: bool,
        store: SessionStore,
        model: Model,
        session_id: str,
        seed: int,
        directory: Path,
    ) -> None:
        self.leg, self.worker_id, self.recovering = leg, worker_id, recovering
        self.clock, self.store, self.model = world.clock, store, model
        self.session_id, self.seed, self.directory = session_id, seed, Path(directory)
        self._transport = transport

    def service(
        self,
        effect: Effect,
        impl: type[Any] | Callable[..., Any],
        *,
        key_arg: str = "idempotency_key",
    ) -> Any:
        """Return this leg's facade over the one point's shared destination world."""
        return self._transport.service(self.leg, effect, impl, key_arg=key_arg)


_IMMUTABLE = (
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    range,
    date,
    datetime,
    time,
    timedelta,
    Decimal,
    UUID,
    PurePath,
    Enum,
)
_DECLARATIONS = (
    Tool,
    Effect,
    Permissions,
    Key,
    Dedupe,
    Unique,
    types.ModuleType,
    types.FunctionType,
    types.BuiltinFunctionType,
    type,
)


class BuildIsolation:
    """Retain reachable leg objects and report identities reused by later legs.

    Function globals/closures are intentionally outside this in-process check.
    Exact driver-owned durable roots may be shared, but a wrapper pointing at such
    a root remains a separate process-local object and must be rebuilt.
    """

    def __init__(self, *, infrastructure: Iterable[Any] = (), max_nodes: int = 20000) -> None:
        if type(max_nodes) is not int or max_nodes < 1:
            raise ValueError("max_nodes must be a positive integer")
        self._infrastructure = {id(value): value for value in infrastructure}
        self._objects: dict[int, tuple[int, str, Any]] = {}
        self.max_nodes = max_nodes
        self.findings: list[dict[str, Any]] = []

    def retain_infrastructure(self, *values: Any) -> None:
        """Exclude exact parent-owned services from the leg identity check."""
        self._infrastructure.update((id(value), value) for value in values)

    def record(self, leg: int, agent: Agent, deps: Any) -> list[dict[str, Any]]:
        if not isinstance(agent, Agent):
            raise ConfigurationError("build(env) must return an Agent or (Agent, deps)")
        before = len(self.findings)
        stack = [("agent", agent), ("deps", deps)]
        visited: set[int] = set()
        while stack:
            path, value = stack.pop()
            if isinstance(value, _IMMUTABLE + _DECLARATIONS):
                continue
            ident = id(value)
            if ident in visited or self._infrastructure.get(ident) is value:
                continue
            visited.add(ident)
            if len(visited) > self.max_nodes:
                self.findings.append(
                    {"kind": "isolation_check_incomplete", "leg": leg, "detail": "object limit"}
                )
                break
            if isinstance(value, tuple | frozenset):
                stack.extend((f"{path}[{index}]", item) for index, item in enumerate(value))
                continue
            previous = self._objects.get(ident)
            if previous is not None and previous[0] != leg and previous[2] is value:
                self.findings.append(
                    {
                        "kind": "shared_between_legs",
                        "leg": leg,
                        "first_leg": previous[0],
                        "path": path,
                        "first_path": previous[1],
                        "type": f"{type(value).__module__}.{type(value).__qualname__}",
                    }
                )
                continue
            self._objects[ident] = leg, path, value  # retain identities for the whole test
            if isinstance(value, dict):
                for index, (key, item) in enumerate(value.items()):
                    stack.append((f"{path}[key#{index}]", key))
                    stack.append((f"{path}[value#{index}]", item))
            elif isinstance(value, list | set):
                stack.extend((f"{path}[{index}]", item) for index, item in enumerate(value))
            elif isinstance(value, types.MethodType):
                stack.append((f"{path}.__self__", value.__self__))
            else:
                for name, item in self._attributes(value):
                    # These are generated validation/schema caches, not user clients.
                    # Pydantic may intentionally share their internals by model class.
                    if isinstance(value, Agent) and name in {
                        "_output_adapter",
                        "_output_schema",
                        "_native_schema",
                    }:
                        continue
                    stack.append((f"{path}.{name}", item))
        return self.findings[before:]

    @staticmethod
    def _attributes(value: Any) -> Iterable[tuple[str, Any]]:
        try:
            attributes = object.__getattribute__(value, "__dict__")
        except (AttributeError, TypeError):
            attributes = {}
        if isinstance(attributes, dict):
            yield from attributes.items()
        for base in type(value).__mro__:
            slots = base.__dict__.get("__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for slot in slots:
                if slot in {"__dict__", "__weakref__"}:
                    continue
                if slot.startswith("__") and not slot.endswith("__"):
                    slot = "_" + base.__name__.lstrip("_") + slot
                try:
                    item = object.__getattribute__(value, slot)
                except AttributeError:
                    continue
                yield slot, item

    async def build(self, leg: int, build: Callable[[Any], Any], env: Any) -> tuple[Agent, Any]:
        result = build(env)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, Agent):
            agent, deps = result, None
        elif isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], Agent):
            agent, deps = result
        else:
            raise ConfigurationError("build(env) must return an Agent or (Agent, deps)")
        if isinstance(env, CrashEnv):
            self.retain_infrastructure(*env._transport.world._services.values())
        self.record(leg, agent, deps)
        return agent, deps


class GatedModel(Model):
    """Fresh per-leg model wrapper capturing the durable prefix before consultation."""

    def __init__(self, world: World, leg: int, model: Model) -> None:
        self.world, self.leg, self.model = world, leg, model
        self.name, self.provider = model.name, model.provider
        world.keep_alive.extend((self, model))
        world._model_clients.append(model)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        iterator = self.model.stream(request)
        first = True
        try:
            while True:

                async def advance(start: bool = first) -> ModelStreamEvent:
                    if start:
                        snapshot = await self.world.snapshot()
                        consultation = self.world._model_consultations.get(self.leg, 0)
                        self.world._model_consultations[self.leg] = consultation + 1
                        if snapshot is not None:
                            self.world.model_evidence.append(
                                ModelEvidence(
                                    self.leg, consultation, request_fingerprint(request), snapshot
                                )
                            )
                    return await anext(iterator)

                try:
                    event = await self.world.gate(
                        self.leg, advance, kind="model_request" if first else "model_chunk"
                    )
                except StopAsyncIteration:
                    return
                first = False
                yield event
        finally:
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()

    async def capabilities(self) -> Any:
        return await self.world.gate(self.leg, self.model.capabilities, kind="model_capabilities")

    def validate_settings(self, settings: ModelSettings) -> None:
        self.model.validate_settings(settings)

    async def aclose(self) -> None:
        await self.world.gate(self.leg, self.model.aclose, kind="model_close")
