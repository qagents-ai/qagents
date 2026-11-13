"""Frozen crash-harness interfaces; behavior is pending its owner stage."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from qagents.effects import Effect, HeldEffect, Settle
from qagents.effects._declare import ReachPoint
from qagents.permissions import Approval
from qagents.session import PendingApproval

from ._env import CrashEnv
from ._fakes import Fake
from ._faults import Faults
from ._model import Cassette
from ._report import CrashReport
from ._world import World

if TYPE_CHECKING:
    pass


async def crash_test(
    build: Callable[[CrashEnv], Any],
    prompt: str | None = None,
    *,
    store: Literal["memory", "sqlite", "postgres"] = "sqlite",
    directory: str | os.PathLike[str] | None = None,
    fakes: Sequence[Effect | Fake] = (),
    model: Sequence[Any] | Cassette | None = None,
    approve: Callable[[PendingApproval], Approval] | None = None,
    answer: Callable[[HeldEffect], Settle | None] | None = None,
    faults: Faults | None = None,
    invariants: Sequence[Callable[[World], bool | None]] = (),
    must_reach: Sequence[ReachPoint] = (),
    expect: Sequence[str] | None = None,
    allow_held: Sequence[str] = (),
    allow_undeclared: Sequence[str] = (),
    assume: Mapping[str, Sequence[str]] | None = None,
    probes: Mapping[str, Literal["passed", "failed"]] | str | os.PathLike[str] | None = None,
    scenario: Callable[[CrashEnv, Any, Any], Awaitable[Any]] | None = None,
    state: Mapping[str, Any] | None = None,
    session_id: str = "crash-test",
    mode: Literal["freeze", "exception"] = "freeze",
    transport: Literal["auto", "fake", "client"] = "auto",
    isolation: Literal["inprocess", "subprocess"] = "inprocess",
    points: Literal["all", "all-raw", "effects", "recovery"] = "all",
    budget: int | None = None,
    seed: int = 0,
    max_legs: int = 6,
    strict: bool = False,
    _first: Sequence[str] = (),
    _stop_on_violation: bool = False,
    _mask_features: frozenset[str] = frozenset(),
) -> CrashReport:
    raise NotImplementedError("The crash harness is pending its owning stage.")


async def assert_recovers(build: Callable[[CrashEnv], Any], path: str | os.PathLike[str]) -> None:
    raise NotImplementedError("The crash harness is pending its owning stage.")
