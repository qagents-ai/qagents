"""A point's shared external world, isolated simulated processes, and crash gates."""

from __future__ import annotations

import asyncio
import contextvars
import copy
import shutil
import threading
import time
from collections.abc import Awaitable, Callable, Hashable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar

from qagents.effects._claims import Claim
from qagents.effects._declare import Effect
from qagents.effects._types import Attempt, EffectRecord
from qagents.session import (
    Entry,
    InMemoryStore,
    LeaseGrant,
    SessionStatus,
    SessionStore,
    SQLiteStore,
)

from ._clock import ManualClock, _finite
from ._invariants import ApplicationEvidence, JournalSnapshot, ModelEvidence
from ._loop import WORLD_THREAD, VirtualLoop

T = TypeVar("T")
LEG: contextvars.ContextVar[int | None] = contextvars.ContextVar("qagents_crash_leg", default=None)


class LegCrashed(BaseException):
    """Exception-mode process loss; deliberately outside ordinary application catches."""


class WorldSealed(BaseException):
    """The verdict is fixed and no further call may change the shared medium."""


@dataclass(frozen=True)
class Application:
    fake: str
    idempotency_key: str | None
    payload_digest: str
    attempt: int | None
    epoch: int | None
    leg: int
    arrival: float
    identity: Hashable
    replayed: bool
    key_passed: bool


@dataclass(frozen=True)
class Boundary:
    leg: int
    index: int
    phase: str
    kind: str
    detail: str
    store: bool
    write: bool
    time: float
    entries_json: tuple[str, ...] = ()


@dataclass
class _Park:
    wake: asyncio.Future[None]
    cancellations: int = 0


@dataclass
class Leg:
    id: int
    worker_id: str
    idle_horizon_s: float
    dead: bool = False
    frozen: bool = False
    task: asyncio.Task[Any] | None = None
    calls: int = 0
    writes: int = 0
    parks: dict[asyncio.Task[Any], _Park] = field(default_factory=dict)


class World:
    """One point's durable medium and observable boundaries, owned by its event loop."""

    journal: list[Entry]
    records: list[EffectRecord]
    outcome: str

    def __init__(
        self,
        loop: VirtualLoop,
        *,
        mode: str = "freeze",
        boundary: Callable[[Boundary], Awaitable[None]] | None = None,
        session_id: str = "crash-test",
    ) -> None:
        if mode not in {"freeze", "exception"}:
            raise ValueError("mode must be freeze or exception")
        self.loop = loop
        self.clock: ManualClock = loop.clock
        self.mode = mode
        self.session_id = session_id
        self.boundary = boundary
        self.interceptor: (
            Callable[[Boundary, Callable[[], Awaitable[Any]]], Awaitable[Any]] | None
        ) = None
        self.medium_lock = asyncio.Lock()
        self.legs: dict[int, Leg] = {}
        self.keep_alive: list[Any] = []
        self.trace: list[tuple[int, str, str, float]] = []
        self.boundaries: list[Boundary] = []
        self.findings: list[dict[str, Any]] = []
        self.sealed = False
        self.crashed = asyncio.Event()
        self.journal: list[Entry] = []
        self.records: list[EffectRecord] = []
        self.outcome = "no_terminal_state"
        self._applications: list[tuple[Application, dict[str, Any]]] = []
        self.application_evidence: list[ApplicationEvidence] = []
        self.model_evidence: list[ModelEvidence] = []
        self._model_consultations: dict[int, int] = {}
        self._model_clients: list[Any] = []
        self._services: dict[str, Any] = {}
        self._store_kind: str | None = None
        self._store_medium: SessionStore | None = None
        self._store_path: Path | None = None
        self._stores: list[WorldStore] = []
        self.loop.thread_started = self._thread_started

    def observe(self, leg: int, kind: str, detail: str = "") -> None:
        """Append an observed event without turning it into another gate or write."""
        self.boundaries.append(
            Boundary(leg, 0, "observed", kind, detail, False, False, self.loop.time())
        )

    def _thread_started(self) -> None:
        leg = LEG.get()
        if leg is not None:
            self.observe(leg, "thread_start")

    def store_for_leg(
        self,
        leg: int,
        *,
        backend: str,
        directory: Path | None = None,
        template: Path | None = None,
        mask_features: frozenset[str] = frozenset(),
    ) -> WorldStore:
        if self.sealed:
            raise WorldSealed()
        if leg not in self.legs:
            raise ValueError("add the leg before creating its store")
        if backend not in {"memory", "sqlite"}:
            raise ValueError("this world needs a memory or sqlite store")
        if self._store_kind not in (None, backend):
            raise ValueError("all legs of one point share the same store medium")
        if any(store._leg == leg for store in self._stores):
            raise ValueError("each leg has one store root")
        if backend == "memory":
            if self._store_medium is None:
                self._store_medium = InMemoryStore(clock=self.clock.store_time)
            native = self._store_medium
        else:
            if directory is None:
                raise ValueError("sqlite worlds require a point directory")
            path = directory.resolve() / "journal.db"
            if self._store_path not in (None, path):
                raise ValueError("all SQLite legs use the same point directory")
            if self._store_path is None:
                directory.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    raise FileExistsError("a new point cannot reuse an existing journal")
                if template is not None:
                    # Never replace another point's file or copy a live WAL database.
                    with template.open("rb") as source, path.open("xb") as destination:
                        shutil.copyfileobj(source, destination)
                self._store_path = path
            native = SQLiteStore(path, clock=self.clock.store_time)
        self._store_kind = backend
        made = WorldStore(self, leg, native, mask_features=mask_features)
        self._stores.append(made)
        self.keep_alive.extend((made, native))
        return made

    async def snapshot(self, session_id: str | None = None) -> JournalSnapshot | None:
        """Read durable evidence outside simulated worker gates, never engine state."""
        session_id = self.session_id if session_id is None else session_id
        if self._store_kind is None:
            return None  # no durable medium; the independent checker must not infer a pass
        async with self.medium_lock:
            if self._store_kind == "sqlite":
                assert self._store_path is not None
                reader = SQLiteStore(self._store_path, clock=self.clock.store_time)
                try:
                    entries = await reader.load(session_id)
                finally:
                    await reader.aclose()
            else:
                assert self._store_medium is not None
                entries = await self._store_medium.load(session_id)
        return JournalSnapshot(
            session_id,
            tuple(entry.model_dump_json() for entry in entries),
            tuple(entry.id for entry in entries),
            self._store_kind == "sqlite",
        )

    def record_application_evidence(
        self,
        index: int,
        effect: Effect,
        attempt: Attempt,
        snapshot: JournalSnapshot | None,
        issued_store_time: float,
    ) -> None:
        if snapshot is None:
            return
        matching = []
        for serialized in snapshot.entries_json:
            entry = Entry.model_validate_json(serialized)
            data = entry.data
            if (
                entry.kind == "effect_intent"
                and data.get("name") == effect.name
                and data.get("key") == attempt.key
                and data.get("idempotency_key") == attempt.idempotency_key
                and data.get("attempt") == attempt.debug.number
                and isinstance(data.get("record_id"), str)
            ):
                matching.append(data["record_id"])
        record_id = matching[0] if len(set(matching)) == 1 else None
        self.application_evidence.append(
            ApplicationEvidence(index, attempt.key, record_id, snapshot, issued_store_time)
        )

    def add_leg(self, ident: int, *, idle_horizon_s: float = 600) -> Leg:
        if self.sealed:
            raise WorldSealed()
        if ident in self.legs:
            raise ValueError(f"leg {ident} already exists")
        leg = Leg(ident, f"leg-{ident}", idle_horizon_s)
        self.legs[ident] = leg
        self.keep_alive.append(leg)
        self.loop.register_leg(ident, idle_horizon_s=idle_horizon_s)
        return leg

    def create_task(self, leg: int, coroutine: Awaitable[T]) -> asyncio.Task[T]:
        context = contextvars.copy_context()
        context.run(LEG.set, leg)
        task = self.loop.create_task(coroutine, context=context)
        self.legs[leg].task = task
        self.keep_alive.append(task)
        return task

    def freeze(self, leg: int, *, zombie: bool = False) -> None:
        state = self.legs[leg]
        state.dead = True
        state.frozen = zombie
        self.loop.deactivate_leg(leg)
        self.crashed.set()
        if self.mode == "exception":
            raise LegCrashed(f"leg-{leg}")

    def thaw(self, leg: int) -> None:
        state = self.legs[leg]
        if not state.frozen:
            raise ValueError("only a frozen zombie leg may be thawed")
        state.dead = state.frozen = False
        self.loop.register_leg(leg, idle_horizon_s=state.idle_horizon_s)
        for park in tuple(state.parks.values()):
            if not park.wake.done():
                park.wake.set_result(None)

    def _post_mortem(self, leg: int, detail: str) -> None:
        self.findings.append({"kind": "post_mortem_call", "leg": leg, "detail": detail})

    async def _park(self, leg: int) -> None:
        state = self.legs[leg]
        task = asyncio.current_task()
        assert task is not None
        park = _Park(self.loop.create_future())
        state.parks[task] = park
        thread = WORLD_THREAD.get()
        if thread is not None:
            self.loop.thread_state(thread, "parked")
        try:
            while state.dead and not self.sealed:
                if park.wake.done():
                    # A thaw followed by a new freeze can precede this task's turn.
                    park.wake = self.loop.create_future()
                try:
                    await asyncio.shield(park.wake)
                except asyncio.CancelledError:
                    park.cancellations += 1
                    self._post_mortem(leg, "cancellation swallowed while parked")
            if self.sealed:
                raise WorldSealed()
            if park.cancellations:
                for _ in range(park.cancellations - 1):
                    task.uncancel()
                raise asyncio.CancelledError()
        finally:
            state.parks.pop(task, None)
            if thread is not None:
                self.loop.thread_state(thread, "running")

    async def _check_gate(self, leg: int) -> None:
        if self.sealed:
            raise WorldSealed()
        if self.legs[leg].dead:
            self._post_mortem(leg, "call from a dead leg")
            await self._park(leg)
        if self.sealed:
            raise WorldSealed()

    async def gate(
        self,
        leg: int,
        call: Callable[[], Awaitable[T]],
        *,
        kind: str,
        detail: str = "",
        store: bool = False,
        write: bool = False,
        renewal: bool = False,
        mutating: bool = False,
        entries_json: tuple[str, ...] = (),
    ) -> T:
        await self._check_gate(leg)
        state = self.legs[leg]
        state.calls += 1
        if write:
            state.writes += 1
        index = state.writes if write else state.calls

        async def boundary(phase: str) -> None:
            item = Boundary(
                leg, index, phase, kind, detail, store, write, self.loop.time(), entries_json
            )
            self.boundaries.append(item)
            if self.boundary is not None:
                await self.boundary(item)
            await self._check_gate(leg)

        await boundary("before")
        self.trace.append((leg, kind, detail, self.loop.time()))
        self.loop.note_world_call(leg, renewal=renewal)

        async def delegate() -> T:
            completed = False
            try:
                if self.interceptor is not None:
                    point = Boundary(
                        leg,
                        index,
                        "call",
                        kind,
                        detail,
                        store,
                        write,
                        self.loop.time(),
                        entries_json,
                    )
                    result = await self.interceptor(point, call)
                else:
                    result = await call()
                completed = True
                return result
            finally:
                if self.sealed and (mutating or write):
                    self.findings.append(
                        {
                            "kind": "post_seal_call_uncertain",
                            "leg": leg,
                            "detail": (
                                f"{kind} {'returned' if completed else 'ended without a result'} "
                                "after sealing; application is not established by return status"
                            ),
                        }
                    )

        if store:
            while True:
                await self.medium_lock.acquire()
                if not self.legs[leg].dead and not self.sealed:
                    break
                self.medium_lock.release()
                await self._check_gate(leg)
            try:
                result = await delegate()
            finally:
                self.medium_lock.release()
        else:
            result = await delegate()
        await boundary("after")
        return result

    def call_from_thread(self, leg: int, call: Callable[[], Awaitable[T]], **options: Any) -> T:
        if self.sealed:
            raise WorldSealed()
        return self.loop.call_from_thread(self.gate(leg, call, **options))

    def seal(self) -> None:
        self.sealed = True
        for state in self.legs.values():
            self.loop.deactivate_leg(state.id)
            for park in tuple(state.parks.values()):
                if not park.wake.done():
                    park.wake.set_result(None)

    async def teardown(self, *, timeout_s: float = 5.0) -> tuple[str, ...]:
        """Seal, release parked calls, and bound task/thread cleanup in real time."""
        timeout_s = _finite(timeout_s, "timeout_s")
        if timeout_s <= 0:
            raise ValueError("teardown timeout must be positive")
        self.seal()
        self.loop.begin_teardown()
        deadline = time.monotonic() + timeout_s
        current = asyncio.current_task()
        seen = {item for item in self.keep_alive if isinstance(item, asyncio.Task)}
        while True:
            tasks = {task for task in asyncio.all_tasks(self.loop) if task is not current}
            seen.update(tasks)
            if not tasks and not self.loop.pending_threads:
                break
            if time.monotonic() >= deadline:
                if tasks:
                    self.loop.inconclusive_reasons.append("tasks remained after teardown bound")
                if self.loop.pending_threads:
                    self.loop.inconclusive_reasons.append("threads remained after teardown bound")
                break
            for task in tasks:
                task.cancel()
            await self._real_pause()
        for task in seen:
            if task.done() and not task.cancelled():
                task.exception()  # retrieve teardown exceptions without hiding the recorded result
        if self.loop.pending_threads:
            if self._stores:
                self.loop.inconclusive_reasons.append("stores could not close while threads remain")
        else:
            closed: set[int] = set()
            for store in self._stores:
                if id(store._backend) not in closed:
                    try:
                        await store._backend.aclose()
                    except BaseException as error:
                        self.loop.inconclusive_reasons.append(
                            f"store close failed: {type(error).__name__}"
                        )
                    closed.add(id(store._backend))
            for client in self._model_clients:
                if id(client) not in closed:
                    try:
                        await client.aclose()
                    except BaseException as error:
                        self.loop.inconclusive_reasons.append(
                            f"model close failed: {type(error).__name__}"
                        )
                    closed.add(id(client))
        return tuple(self.loop.inconclusive_reasons)

    async def _real_pause(self) -> None:
        future = self.loop.create_future()

        def wake() -> None:
            if not future.done():
                future.set_result(None)

        timer = threading.Timer(0.01, lambda: self.loop.call_soon_threadsafe(wake))
        timer.daemon = True
        timer.start()
        try:
            await future
        finally:
            timer.cancel()
            timer.join(0.1)

    def applications(self, effect: Effect | None = None) -> list[Application]:
        return [
            application
            for application, _ in self._applications
            if effect is None or application.fake == effect.name
        ]

    def count_applied(self, effect: Effect, **fields: Any) -> int:
        return sum(
            not application.replayed
            and application.fake == effect.name
            and all(name in request and request[name] == value for name, value in fields.items())
            for application, request in self._applications
        )

    def service(self, effect: Effect) -> Any:
        return self._services[effect.name]


class WorldStore(SessionStore):
    """A distinct per-leg registry root over the one point's durable store medium.

    There is intentionally no ``wrapped`` attribute. Memory reads are deep copies;
    they remain same-process simulations, not independent process persistence checks.
    """

    def __init__(
        self,
        world: World,
        leg: int,
        backend: SessionStore,
        *,
        mask_features: frozenset[str] = frozenset(),
    ) -> None:
        self._world, self._leg, self._backend = world, leg, backend
        self._mask_features = mask_features

    @property
    def features(self) -> frozenset[str]:
        return self._backend.features - self._mask_features

    @property
    def takeover_grace_s(self) -> float:
        return self._backend.takeover_grace_s

    async def _call(
        self,
        method: str,
        *args: Any,
        write: bool = False,
        mutating: bool = False,
        detail: str = "",
        **kwargs: Any,
    ) -> Any:
        async def delegate() -> Any:
            value = await getattr(self._backend, method)(*args, **kwargs)
            return copy.deepcopy(value)

        return await self._world.gate(
            self._leg,
            delegate,
            kind=method,
            detail=detail,
            store=True,
            write=write,
            mutating=mutating,
            renewal=method == "renew_lease",
            entries_json=(
                tuple(entry.model_dump_json() for entry in args[1])
                if method in {"append", "append_leased"}
                else ()
            ),
        )

    async def append(self, session_id: str, entries: list[Entry]) -> list[int] | None:
        return await self._call(
            "append", session_id, entries, write=True, detail=self._entries_detail(entries)
        )

    async def append_leased(
        self, session_id: str, entries: list[Entry], owner: str
    ) -> list[int] | None:
        return await self._call(
            "append_leased",
            session_id,
            entries,
            owner,
            write=True,
            detail=self._entries_detail(entries),
        )

    @staticmethod
    def _entries_detail(entries: list[Entry]) -> str:
        return ",".join(
            entry.kind
            + ("/" + entry.data["name"] if isinstance(entry.data.get("name"), str) else "")
            for entry in entries
        )

    async def load(self, session_id: str) -> list[Entry]:
        return await self._call("load", session_id)

    async def delete(
        self, session_id: str, *, force: bool | Literal["forget_effects"] = False
    ) -> None:
        await self._call("delete", session_id, force=force, mutating=True)

    async def aclose(self) -> None:
        await self._call("aclose")

    async def status(self, session_id: str) -> SessionStatus:
        return await self._call("status", session_id)

    async def list_sessions(
        self,
        status: str | Iterable[str] | None = None,
        prefix: str | None = None,
        limit: int | None = None,
    ) -> list[SessionStatus]:
        return await self._call("list_sessions", status=status, prefix=prefix, limit=limit)

    async def now(self) -> float:
        return await self._call("now")

    async def salt(self) -> bytes:
        return await self._call("salt")

    async def acquire_lease(self, session_id: str, owner: str, ttl_s: float) -> bool:
        return await self._call("acquire_lease", session_id, owner, ttl_s, write=True)

    async def acquire_lease_grant(
        self, session_id: str, owner: str, ttl_s: float
    ) -> LeaseGrant | None:
        return await self._call("acquire_lease_grant", session_id, owner, ttl_s, write=True)

    async def renew_lease(self, session_id: str, owner: str, ttl_s: float) -> bool:
        return await self._call("renew_lease", session_id, owner, ttl_s, write=True)

    async def release_lease(self, session_id: str, owner: str) -> None:
        await self._call("release_lease", session_id, owner, write=True)

    async def get_claim(self, namespace: str, effect: str, key: str) -> Claim | None:
        return await self._call("get_claim", namespace, effect, key)

    async def purge_claims(self, *, older_than: float) -> int:
        return await self._call("purge_claims", older_than=older_than, mutating=True)

    async def register_effect(
        self, effect: str, version: str, key_spec_digest: str, request_schema_digest: str
    ) -> Literal["registered", "matches", "key_changed", "schema_changed"]:
        return await self._call(
            "register_effect",
            effect,
            version,
            key_spec_digest,
            request_schema_digest,
            mutating=True,
        )
