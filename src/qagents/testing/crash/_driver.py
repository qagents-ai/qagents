"""Run fresh agent instances against one observed destination and journal per point."""

from __future__ import annotations

import asyncio
import copy
import hmac
import json
import os
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from qagents.agent import _forget_store
from qagents.effects import Effect, HeldEffect, Settle, _engine, _ports
from qagents.effects._canonical import jcs
from qagents.effects._declare import ReachPoint
from qagents.effects._kinds import APPROVAL_MAC_FIELDS
from qagents.errors import ConfigurationError, HeldEffectsError, JournalOutcomeUnknown, LeaseError
from qagents.models.base import ModelResponse
from qagents.models.testing import ScriptedModel
from qagents.permissions import Approval
from qagents.session import Entry, PendingApproval, Session, derive_state

from ._clock import ManualClock
from ._env import BuildIsolation, CrashEnv, GatedModel
from ._execution import Execution
from ._explore import (
    Point,
    boundary_points,
    concurrency_variants,
    recovery_points,
    reference_points,
    select_points,
)
from ._fakes import Fake
from ._faults import FaultController, FaultPlan, Faults
from ._loop import NoTerminalState, VirtualLoop
from ._model import PERTURBATIONS, Cassette, ModelLedger, UnscriptedRequest
from ._report import CrashReport, _point_checks
from ._transport import HarnessTransport
from ._world import LegCrashed, World

_EXPLORATION_LOCK = threading.Lock()


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
    """Run an offline scenario and check observed recovery at durable boundaries."""
    if store not in {"memory", "sqlite"} or isolation != "inprocess":
        raise ConfigurationError(
            "This crash driver currently supports in-process memory and SQLite."
        )
    if model is None:
        raise ConfigurationError("Supply an offline model script or cassette to crash_test.")
    if not callable(build) or type(max_legs) is not int or max_legs < 1 or type(seed) is not int:
        raise ConfigurationError("crash_test needs build, an integer seed, and positive max_legs.")
    if not isinstance(session_id, str) or not session_id:
        raise ConfigurationError("crash_test requires a nonempty session_id.")
    if points not in {"all", "all-raw", "effects", "recovery"}:
        raise ConfigurationError("Unknown crash point selection.")
    chosen = faults or Faults.default()
    if isinstance(probes, (str, os.PathLike)):
        probes = json.loads(await asyncio.to_thread(Path(probes).read_text, encoding="utf-8"))
    if probes is not None and (
        not isinstance(probes, Mapping)
        or any(value not in {"passed", "failed"} for value in probes.values())
    ):
        raise ConfigurationError("Probe results must map effect names to passed or failed.")

    def execute(point: Point, reference: Execution | None):
        with _EXPLORATION_LOCK:
            return execute_point(point, reference)

    def execute_point(point: Point, reference: Execution | None):
        return _execute_point(
            point,
            reference,
            directory=directory,
            build=build,
            prompt=prompt,
            store=store,
            model=model,
            faults=chosen,
            fakes=fakes,
            mode=mode,
            transport=transport,
            session_id=session_id,
            seed=seed,
            max_legs=max_legs,
            scenario=scenario,
            state=state,
            approve=approve,
            answer=answer,
            invariants=invariants,
            mask_features=_mask_features,
        )

    if directory is not None:
        await asyncio.to_thread(Path(directory).mkdir, parents=True, exist_ok=True)

    # Each point has its own loop. Run them sequentially on an owned worker
    # because the observed tool-thread producer is process-global.
    def explore():
        reference, effects, undeclared, reference_checks = execute(Point("reference"), None)
        candidates = reference_points(reference.boundaries, effects, chosen, points=points)
        candidates = concurrency_variants(candidates, effects)
        selection = select_points(candidates, budget=budget, seed=seed, first=_first)
        executed = [reference]
        checks = {reference.point.id: reference_checks}
        deeper = {}
        for point in selection.points:
            if point.id == "reference":
                continue
            result, _, _, user = execute(point, reference)
            if points != "recovery":
                executed.append(result)
                checks[point.id] = user
            if "recovery_kill" in chosen.kinds:
                for recovered in recovery_points(point, reference.boundaries, result.boundaries):
                    deeper[recovered.id] = recovered
            for variant in _model_points(point, result, chosen):
                deeper[variant.id] = variant
            if "advance_boundaries" in chosen.kinds and result.crash_snapshots:
                for boundary in boundary_points(point, _recorded_boundaries(result), leg=1):
                    deeper[boundary.id] = boundary
            if _stop_on_violation and any(
                check.status == "violated"
                for check in _point_checks(result, reference, user, max_legs, expect)
            ):
                break
        if budget is None:
            for point in deeper.values():
                result, _, _, user = execute(point, reference)
                executed.append(result)
                checks[point.id] = user
        return reference, effects, undeclared, executed, checks, selection

    reference, effects, undeclared, executions, user_checks, selection = await asyncio.to_thread(
        explore
    )

    async def rerun(point: Point):
        result, _, _, checks = await asyncio.to_thread(execute, point, reference)
        return result, checks

    return CrashReport(
        executions,
        effects=effects,
        user_checks=user_checks,
        must_reach=must_reach,
        expect=expect,
        allow_held=allow_held,
        allow_undeclared=allow_undeclared,
        assume=assume or {},
        probes=probes or {},
        undeclared=undeclared,
        strict=strict,
        max_legs=max_legs,
        selected=selection,
        requested_faults=chosen.kinds,
        rerun=rerun,
        mode=mode,
        store=store,
        reproduction=await asyncio.to_thread(
            _reproduction,
            model,
            prompt,
            store,
            chosen,
            state,
            session_id,
            mode,
            transport,
            seed,
            max_legs,
            supported=not (scenario or fakes or approve or answer or invariants or _mask_features),
        ),
    )


def _recorded_boundaries(result):
    """Recover the observed dedupe cutoff and approval bounds before an injected crash."""
    values = {}
    if not result.crash_snapshots:
        return values
    for raw in result.crash_snapshots[0].entries_json:
        entry = Entry.model_validate_json(raw)
        data = entry.data
        if entry.kind == "effect_intent":
            timing = data.get("timing", {})
            first = data.get("first_use_at")
            window = timing.get("window_s")
            if first is not None and window is not None:
                boundary = (
                    first
                    + window
                    - sum(
                        timing.get(field) or 0.0
                        for field in ("timeout_s", "m_s", "sigma_s", "margin_s")
                    )
                )
                values[f"{data['name']}/dedupe"] = boundary
        elif entry.kind == "approval_decided" and "expires_at" in data:
            values[f"{data['name']}/approval"] = data["expires_at"]
        elif entry.kind == "effect_observed" and data.get("retry_after") is not None:
            values[f"{data['name']}/retry"] = data["retry_after"]
    return values


def _model_points(prefix, result, faults):
    """Choose model variants only where recovery actually consulted a model."""
    if not prefix.plans or not result.triggered:
        return
    calls = [item for item in result.model_calls if item.leg > 0 and item.turn is not None]
    if not calls:
        return
    leg = min(item.leg for item in calls)
    for kind in sorted(faults.kinds & PERTURBATIONS):
        identity = f"{prefix.id}+leg{leg}:model/{kind}"
        yield Point(
            identity,
            (*prefix.plans, FaultPlan(identity, kind, 1, leg)),
            in_effect_window=prefix.in_effect_window,
            recovery=True,
            variant=prefix.variant,
        )


def _reproduction(
    model,
    prompt,
    store,
    faults,
    state,
    session_id,
    mode,
    transport,
    seed,
    max_legs,
    *,
    supported,
):
    if not supported:
        return None
    if isinstance(model, Cassette):
        # Copy the offline recording into the pin. Replaying a pin must not
        # depend on the original cassette's path or modification time.
        model_source = {"cassette": json.loads(model.path.read_text(encoding="utf-8"))}
    else:
        script = ScriptedModel([])
        replies = []
        for reply in model:
            try:
                response = script._as_response(reply)
            except TypeError:
                return None
            replies.append(response.model_dump(mode="json"))
        model_source = {"replies": replies}
    return {
        **model_source,
        "prompt": prompt,
        "store": store,
        "faults": sorted(faults.kinds),
        "adapters": list(faults.adapters),
        "stall_s": faults.stall_s,
        "state": state,
        "session_id": session_id,
        "mode": mode,
        "transport": transport,
        "seed": seed,
        "max_legs": max_legs,
    }


async def assert_recovers(build: Callable[[CrashEnv], Any], path: str | os.PathLike[str]) -> None:
    """Re-run a pinned offline point and assert the observed recovery checks."""
    pin = await asyncio.to_thread(Path(path).read_text, encoding="utf-8")
    record = json.loads(pin)
    if not isinstance(record, dict) or record.get("format") != "qagents-crash-replay/1":
        raise ConfigurationError("The pin does not contain an offline replay configuration.")
    config = record["reproduction"]
    saved = record["point"]
    plans = tuple(FaultPlan(**item) for item in saved["plans"])
    point = Point(
        saved["id"],
        plans,
        tuple(saved["aliases"]),
        saved["in_effect_window"],
        saved["recovery"],
        tuple(sorted(saved["variant"].items())),
    )
    faults = Faults(kinds=config["faults"], adapters=config["adapters"], stall_s=config["stall_s"])
    if config["store"] not in {"memory", "sqlite"} or config["mode"] not in {"freeze", "exception"}:
        raise ConfigurationError("Unsupported store or mode in the replay configuration.")

    def replay():
        with TemporaryDirectory(prefix="qagents-crash-source-") as temp:
            if "cassette" in config:
                cassette = Path(temp) / "model.json"
                cassette.write_text(json.dumps(config["cassette"]), encoding="utf-8")
                model = Cassette(cassette)
            else:
                model = tuple(ModelResponse.model_validate(item) for item in config["replies"])
            return replay_with(model)

    def replay_with(model):
        options = {
            "build": build,
            "prompt": config["prompt"],
            "store": config["store"],
            "model": model,
            "faults": faults,
            "mode": config["mode"],
            "transport": config["transport"],
            "session_id": config["session_id"],
            "seed": config["seed"],
            "max_legs": config["max_legs"],
            "state": config["state"],
        }
        with _EXPLORATION_LOCK:
            reference, _, _, _ = _execute_point(Point("reference"), None, **options)
            result, _, _, user = _execute_point(point, reference, **options)
        if reference.inconclusive_reasons or reference.outcome != "completed_as_reference":
            raise AssertionError("The replay reference failed to complete.")
        return result, _point_checks(result, reference, user, config["max_legs"], None)

    result, checks = await asyncio.to_thread(replay)
    failures = [finding.message for check in checks for finding in check.findings]
    failures.extend(result.inconclusive_reasons)
    if point.plans and set(result.triggered) != {plan.id for plan in point.plans}:
        failures.append("One or more pinned faults did not trigger.")
    if result.outside_a3:
        failures.append("The pinned point is outside the declared store-clock assumptions.")
    if failures:
        raise AssertionError("\n".join([f"{point.id}: recovery checks failed", *failures]))


def _execute_point(
    point,
    reference,
    *,
    build,
    prompt,
    store,
    model,
    faults,
    mode,
    transport,
    session_id,
    seed,
    max_legs,
    state,
    directory=None,
    fakes=(),
    scenario=None,
    approve=None,
    answer=None,
    invariants=(),
    mask_features=frozenset(),
):
    with TemporaryDirectory(dir=directory, prefix="qagents-crash-") as temp:
        loop = VirtualLoop(ManualClock(store_time=1_000_000))
        world = World(loop, mode=mode, session_id=session_id)
        controller = FaultController(world, point.plans, faults=faults)
        controller.install()
        destination = HarnessTransport(
            world,
            fakes=fakes,
            mode=transport,
            controller=controller,
            unknown_concurrent=dict(point.variant).get("unknown_concurrent", "rejects"),
        )
        ledger = ModelLedger(model, seed=seed)
        identity = BuildIsolation(infrastructure=(world, ledger, destination))
        return _execute_loop(
            loop,
            world,
            controller,
            destination,
            ledger,
            identity,
            point,
            build=build,
            prompt=prompt,
            scenario=scenario,
            state=state,
            approve=approve,
            answer=answer,
            reference=reference,
            store=store,
            directory=Path(temp),
            seed=seed,
            max_legs=max_legs,
            invariants=invariants,
            mask_features=mask_features,
        )


def _execute_loop(
    loop,
    world,
    controller,
    destination,
    ledger,
    identity,
    point,
    *,
    build,
    prompt,
    scenario,
    state,
    approve,
    answer,
    reference,
    store,
    directory,
    seed,
    max_legs,
    invariants,
    mask_features,
):
    effects, undeclared, reached, snapshots, checks = {}, set(), [], [], []
    replayable = set()
    output, held = None, []
    errors = []
    approval_keys = {}
    usage_limits = {}
    perturbation = next((p for p in point.plans if p.kind in PERTURBATIONS), None)

    async def run():
        nonlocal output, held
        answers = {}
        needs_recheck = False
        for number in range(max_legs):
            clock_fault = next(
                (
                    plan
                    for plan in point.plans
                    if plan.kind == "advance_boundaries" and plan.leg == number
                ),
                None,
            )
            if clock_fault is not None and clock_fault.value > world.clock.store_time():
                world.clock.advance(clock_fault.value - world.clock.store_time())
                controller.hit(clock_fault)
            world.crashed.clear()
            leg = world.add_leg(number)
            backend = world.store_for_leg(
                number,
                backend=store,
                directory=directory,
                mask_features=mask_features,
            )
            model = ledger.for_leg(
                number,
                recovering=number > 0,
                perturbation=perturbation.kind
                if perturbation and perturbation.leg == number
                else None,
            )
            env = CrashEnv(
                world=world,
                transport=destination,
                leg=number,
                worker_id=leg.worker_id,
                recovering=number > 0,
                store=backend,
                model=GatedModel(world, number, model),
                session_id=world.session_id,
                seed=seed,
                directory=directory,
            )
            session = Session(backend, id=world.session_id)

            async def drive(
                number=number,
                env=env,
                session=session,
                model=model,
                needs_recheck=needs_recheck,
                answers=answers,
            ):
                token = _ports.TRANSPORT.set(destination.for_leg(number))
                observer = _ports.REACH_OBSERVER.set(reached.append)
                try:
                    agent, deps = await identity.build(number, build, env)
                    if agent.model is not env.model:
                        raise ConfigurationError(
                            "build(env) must use env.model for offline replay."
                        )
                    approval_keys[number] = agent.permissions._approval_key
                    usage_limits[number] = (
                        agent.limits.max_total_tokens,
                        agent.limits.max_cost_usd,
                    )
                    tools = await agent._collect_tools()
                    model.effect_tools = frozenset(
                        name for name, tool in tools.items() if tool.effects
                    )
                    effects.update(agent._effect_map(tools.values()))
                    replayable.update(
                        name
                        for name, tool in tools.items()
                        if tool.read_only or tool.idempotent or tool.effects
                    )
                    undeclared.update(
                        name
                        for name, tool in tools.items()
                        if not tool.read_only and not tool.effects
                    )
                    if needs_recheck:
                        pending = await agent.recheck(session, deps=deps)
                        if pending and answer is not None:
                            for item in pending:
                                if (decision := answer(item)) is not None:
                                    await agent.settle(session, decision, deps=deps)
                            pending = [item.hold for item in await session.effects() if item.hold]
                        if pending and not any(item.retry_after for item in pending):
                            return pending
                        if pending:
                            later = min(
                                item.retry_after.timestamp() for item in pending if item.retry_after
                            )
                            world.clock.advance(max(0, later - world.clock.store_time()) + 0.001)
                    if scenario is not None:
                        return await scenario(env, agent, deps)
                    if (await session.state()).open_run_id is not None:
                        return await agent.resume(session, deps=deps, approvals=answers)
                    return await agent.run(prompt, session=session, deps=deps, state=state)
                finally:
                    _ports.REACH_OBSERVER.reset(observer)
                    _ports.TRANSPORT.reset(token)

            task = world.create_task(number, drive())
            crashed = asyncio.create_task(world.crashed.wait())
            try:
                await asyncio.wait({task, crashed}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                crashed.cancel()
                await asyncio.gather(crashed, return_exceptions=True)
            if leg.dead:
                snapshot = await world.snapshot()
                if snapshot is not None:
                    snapshots.append(snapshot)
                status = await backend._backend.status(world.session_id)
                if status.lease_expires_at is not None:
                    target = status.lease_expires_at + backend.takeover_grace_s
                    world.clock.advance(max(0, target - world.clock.store_time()) + 0.001)
                if leg.frozen:
                    world.thaw(number)
                    await asyncio.sleep(0)
                if task.done() and not task.cancelled():
                    task.exception()
                continue
            world.loop.deactivate_leg(number)
            if (
                perturbation is not None
                and perturbation.leg == number
                and any(
                    item.leg == number and item.perturbation == perturbation.kind
                    for item in ledger.calls
                )
            ):
                controller.hit(perturbation)
            try:
                result = task.result()
            except (LeaseError, JournalOutcomeUnknown, asyncio.CancelledError):
                if not controller.triggered:
                    raise
                continue
            except HeldEffectsError as error:
                held = list(error.held)
                needs_recheck = True
                continue
            if isinstance(result, list):
                held = result
                world.outcome = "paused_awaiting_operator"
                return
            held = list(getattr(result, "held_effects", ()))
            output = getattr(result, "output", result)
            if getattr(result, "status", None) == "paused":
                pending = result.pending_approvals
                if pending and approve is not None:
                    answers = {item.tool_call_id: approve(item) for item in pending}
                    continue
                if held and not needs_recheck:
                    needs_recheck, answers = True, {}
                    continue
                world.outcome = "paused_awaiting_operator" if held else "paused_awaiting_approval"
                return
            if getattr(result, "status", "completed") == "completed":
                if (
                    any(
                        "EFFECT_OUTCOME_UNKNOWN:" in part.content
                        for msg in result.messages
                        for part in msg.tool_results
                    )
                    if hasattr(result, "messages")
                    else False
                ):
                    world.outcome = "completed_after_model_told"
                else:
                    world.outcome = (
                        "completed_as_reference"
                        if reference is None or output == reference.output
                        else "completed_alternative"
                    )
                return
            world.outcome = "failed_exhausted"
            return

    with loop.observe_tool_threads():
        try:
            loop.run_until_complete(run())
        except UnscriptedRequest as error:
            errors.append(str(error))
        except NoTerminalState as error:
            errors.append(str(error))
        except ConfigurationError:
            raise
        except (Exception, LegCrashed) as error:
            world.outcome = "failed_unexpected"
            world.findings.append(
                {"kind": "unexpected_failure", "detail": f"{type(error).__name__}: {error}"}
            )
        finally:
            try:
                loop.begin_teardown()
                snapshot = loop.run_until_complete(world.snapshot())
                observed_state = None
                if snapshot is not None:
                    world.journal = [
                        Entry.model_validate_json(item) for item in snapshot.entries_json
                    ]
                    replayed = derive_state(world.journal, session_id=world.session_id)
                    observed_state = replayed.values
                    world.records = replayed.effects.records(world.session_id)
                from ._invariants import check_user_invariant

                checks = [
                    check_user_invariant(world, invariant, name=str(index))
                    for index, invariant in enumerate(invariants)
                ]
                errors.extend(loop.run_until_complete(world.teardown()))
            finally:
                for wrapper in world._stores:
                    _forget_store(wrapper)
                    _engine._forget_store(wrapper)
                loop.close()
    return (
        Execution(
            point,
            world.outcome,
            len(world.legs),
            snapshot,
            tuple(snapshots),
            tuple(world.applications()),
            tuple(world.application_evidence),
            tuple(reached),
            tuple(copy.deepcopy(ledger.calls)),
            tuple(world.model_evidence),
            tuple(
                json.dumps(item, sort_keys=True) for item in [*world.findings, *identity.findings]
            ),
            tuple(held),
            tuple(controller.triggered),
            controller.not_triggered,
            tuple(controller.outside_a3),
            output=copy.deepcopy(output),
            boundaries=tuple(world.boundaries),
            inconclusive_reasons=tuple(errors),
            fault_model=tuple(
                f"{name}: NOT exercised through client transport"
                for name in sorted(destination.not_exercised)
            ),
            final_state=copy.deepcopy(observed_state),
            replayable_tools=frozenset(replayable),
            adapters=controller.faults.adapters,
            approval_macs=_approval_macs(world.journal, approval_keys),
            usage_limits=tuple((leg, *limits) for leg, limits in usage_limits.items()),
        ),
        list(effects.values()),
        sorted(undeclared),
        checks,
    )


def _approval_macs(entries, keys):
    """Evaluate signatures independently of Permissions.verify_decision."""
    if not keys:
        return ()
    key = keys[max(keys)]
    result = []
    for entry in entries:
        if entry.kind != "approval_decided" or "approval_id" not in entry.data:
            continue
        data = entry.data
        mac = data.get("mac")
        if key is None and mac is None:
            continue
        expected = None
        if key is not None:
            try:
                fields = {name: data[name] for name in APPROVAL_MAC_FIELDS}
                expected = hmac.new(key, jcs(fields), sha256).hexdigest()
            except (KeyError, TypeError, ValueError):
                pass
        valid = (
            expected is not None
            and isinstance(mac, str)
            and mac.isascii()
            and hmac.compare_digest(mac, expected)
        )
        result.append((entry.id, valid))
    return tuple(result)
