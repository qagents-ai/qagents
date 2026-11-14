"""Named fault choices and executable injections at actual world/store boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar

from qagents.errors import LeaseError

from ._clock import _finite
from ._world import Boundary, World


class Faults:
    kinds: frozenset[str]

    KINDS: ClassVar[frozenset[str]] = frozenset(
        {
            "kill_before",
            "kill_after",
            "recovery_kill",
            "freeze_after_intent",
            "freeze_before_fence2",
            "freeze_in_send",
            "reject",
            "fail_before_commit",
            "commit_then_lose_response",
            "commit_then_error_response",
            "commit_then_reject_shaped_error",
            "commit_then_crash",
            "commit_late",
            "not_yet_visible_then_commit",
            "unstorable_receipt",
            "duplicate_key",
            "rejected_on",
            "slow_arrival",
            "ack_lost_committed",
            "ack_lost_uncommitted",
            "stall",
            "cancel_during",
            "lease_ack_lost",
            "lease_renew_refused",
            "lease_error",
            "advance_boundaries",
            "store_step",
            "store_step_beyond_sigma",
            "approval_redelivered",
            "stale_answer",
            "double_resume",
            "new_run_on_held",
            "operator_race",
            "fresh_ids",
            "retry_effect",
            "retry_changed",
            "permuted",
            "abandon",
        }
    )
    DEFAULT_KINDS: ClassVar[frozenset[str]] = KINDS
    QUICK_KINDS: ClassVar[frozenset[str]] = frozenset(
        {"kill_before", "kill_after", "commit_then_lose_response", "retry_effect"}
    )
    ALIASES: ClassVar[Mapping[str, frozenset[str]]] = MappingProxyType(
        {"zombie": frozenset({"freeze_after_intent", "freeze_before_fence2", "freeze_in_send"})}
    )

    def __init__(
        self,
        *,
        kinds: Iterable[str],
        adapters: Sequence[str] = ("openai_chat", "anthropic"),
        stall_s: float = 3600.0,
    ) -> None:
        if isinstance(kinds, str | bytes):
            raise TypeError("kinds must be an iterable of names; use Faults.parse for text")
        chosen = frozenset(kinds)
        if not all(isinstance(kind, str) for kind in chosen):
            raise TypeError("fault names must be strings")
        unknown = chosen - self.KINDS
        if unknown:
            raise ValueError("unknown fault kinds: " + ", ".join(sorted(unknown)))
        if isinstance(adapters, str | bytes) or any(
            adapter not in {"openai_chat", "anthropic"} for adapter in adapters
        ):
            raise ValueError("adapters must contain openai_chat and/or anthropic")
        self.kinds = chosen
        self.adapters = tuple(dict.fromkeys(adapters))
        self.stall_s = _finite(stall_s, "stall_s")
        if self.stall_s <= 0:
            raise ValueError("stall_s must be positive")

    @classmethod
    def default(cls) -> Faults:
        return cls(kinds=cls.DEFAULT_KINDS)

    @classmethod
    def quick(cls) -> Faults:
        return cls(kinds=cls.QUICK_KINDS)

    @classmethod
    def parse(cls, spec: str) -> Faults:
        if not isinstance(spec, str) or not spec.strip():
            raise ValueError("fault specification must be nonempty text")
        kinds: set[str] = set()
        for name in spec.split(","):
            name = name.strip()
            if not name:
                raise ValueError("fault specification has an empty item")
            if name == "default":
                kinds.update(cls.DEFAULT_KINDS)
            elif name == "quick":
                kinds.update(cls.QUICK_KINDS)
            else:
                kinds.update(cls.ALIASES.get(name, {name}))
        return cls(kinds=kinds)


@dataclass(frozen=True)
class FaultPlan:
    """One explicit target; enumeration supplies its stable point identifier."""

    id: str
    kind: str
    index: int
    leg: int = 0
    phase: str = "call"
    method: str | None = None
    value: Any = None

    def __post_init__(self) -> None:
        if not self.id or self.kind not in Faults.KINDS:
            raise ValueError("fault target needs an id and a known kind")
        if type(self.index) is not int or self.index < 1:
            raise ValueError("fault index must be a positive integer")
        if type(self.leg) is not int or self.leg < 0:
            raise ValueError("fault leg must be a nonnegative integer")
        if self.phase not in {"before", "after", "call"}:
            raise ValueError("fault phase must be before, after or call")
        required_phase = {
            "kill_before": "before",
            "kill_after": "after",
            "freeze_after_intent": "after",
        }.get(self.kind)
        if required_phase is not None and self.phase != required_phase:
            raise ValueError(f"{self.kind} must target its {required_phase} boundary")


class FaultController:
    """Apply a point's selected faults once, preserving evidence of missed targets."""

    def __init__(
        self,
        world: World,
        plans: Sequence[FaultPlan],
        *,
        faults: Faults | None = None,
        sigma_s: float = 1.0,
    ) -> None:
        if len({plan.id for plan in plans}) != len(plans):
            raise ValueError("fault target ids must be distinct")
        self.world, self.plans = world, tuple(plans)
        self.faults = faults if faults is not None else Faults.default()
        if any(plan.kind not in self.faults.kinds for plan in plans):
            raise ValueError("fault target is outside the selected fault set")
        self.sigma_s = _finite(sigma_s, "sigma_s")
        if self.sigma_s < 0:
            raise ValueError("sigma_s must be nonnegative")
        self.triggered: list[str] = []
        self.outside_a3: list[str] = []

    @property
    def not_triggered(self) -> tuple[str, ...]:
        return tuple(plan.id for plan in self.plans if plan.id not in self.triggered)

    def install(self) -> None:
        if self.world.boundary is not None or self.world.interceptor is not None:
            raise ValueError("world already has a fault controller")
        self.world.boundary = self.at_boundary
        self.world.interceptor = self.intercept

    def _matches(self, point: Boundary) -> list[FaultPlan]:
        return [
            plan
            for plan in self.plans
            if plan.id not in self.triggered
            and plan.leg == point.leg
            and plan.index == point.index
            and plan.phase == point.phase
            and (plan.method is None or plan.method == point.kind)
            and point.write
        ]

    def hit(self, plan: FaultPlan) -> None:
        self.triggered.append(plan.id)
        self.world.trace.append((plan.leg, "fault", plan.id, self.world.loop.time()))
        if plan.kind in {"freeze_in_send", "store_step_beyond_sigma"}:
            self.outside_a3.append(plan.id)

    async def at_boundary(self, point: Boundary) -> None:
        for plan in self._matches(point):
            if plan.kind in {"kill_before", "kill_after", "recovery_kill"}:
                self.hit(plan)
                self.world.freeze(point.leg)
            elif plan.kind == "freeze_after_intent" and "effect_intent" in point.detail:
                self.hit(plan)
                self.world.freeze(point.leg, zombie=True)
            elif plan.kind in {"store_step", "store_step_beyond_sigma"}:
                delta = _finite(plan.value, "store step")
                beyond = abs(delta) > self.sigma_s
                if beyond != (plan.kind == "store_step_beyond_sigma"):
                    raise ValueError("store-clock fault does not match its sigma classification")
                self.hit(plan)
                self.world.clock.step_store(delta)
            elif plan.kind == "advance_boundaries":
                delta = _finite(plan.value, "boundary advance")
                if delta < 0:
                    raise ValueError("boundary advances cannot go backwards")
                self.hit(plan)
                self.world.clock.advance(delta)

    async def intercept(self, point: Boundary, call: Callable[[], Awaitable[Any]]) -> Any:
        selected = self._matches(point)

        async def apply(index: int) -> Any:
            if index == len(selected):
                return await call()
            plan = selected[index]

            async def following() -> Any:
                return await apply(index + 1)

            if plan.kind == "ack_lost_committed":
                self.hit(plan)
                await following()
                raise OSError("acknowledgement lost")
            if plan.kind == "ack_lost_uncommitted":
                self.hit(plan)
                raise OSError("acknowledgement lost before commit")
            if plan.kind == "stall":
                self.hit(plan)
                await asyncio.sleep(self.faults.stall_s)
                return await following()
            if plan.kind == "cancel_during":
                task = self.world.legs[point.leg].task
                if task is None:
                    raise RuntimeError("cancel_during requires the leg's run task")
                self.hit(plan)
                task.cancel()
                return await following()
            if plan.kind == "lease_ack_lost" and point.kind in {
                "acquire_lease_grant",
                "renew_lease",
                "release_lease",
            }:
                if plan.value not in {"committed", "uncommitted"}:
                    raise ValueError("lease acknowledgement fault needs its commit shape")
                self.hit(plan)
                if plan.value == "committed":
                    await following()
                raise OSError("lease acknowledgement lost")
            if plan.kind == "lease_renew_refused" and point.kind == "renew_lease":
                self.hit(plan)
                return False
            if plan.kind == "lease_error" and point.kind == "renew_lease":
                self.hit(plan)
                raise LeaseError("injected lease renewal failure")
            return await following()

        return await apply(0)
