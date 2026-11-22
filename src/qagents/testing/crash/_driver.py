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
from qagents.errors import (
    ApprovalConflictError,
    ConfigurationError,
    HeldEffectsError,
    JournalOutcomeUnknown,
    LeaseError,
    SessionError,
    SettlementConflict,
)
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
_SUBPROCESS_UNSUPPORTED = {"freeze_after_intent", "freeze_before_fence2", "freeze_in_send"}


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
    if store not in {"memory", "sqlite"} or isolation not in {"inprocess", "subprocess"}:
        raise ConfigurationError(
            "Crash testing supports memory or SQLite and a known isolation mode."
        )
    if isolation == "subprocess" and (store != "sqlite" or mode != "freeze"):
        raise ConfigurationError(
            "Subprocess crash testing requires store='sqlite' and mode='freeze'."
        )
    if model is None:
        raise ConfigurationError("Supply an offline model script or cassette to crash_test.")
    if not callable(build) or type(max_legs) is not int or max_legs < 1 or type(seed) is not int:
        raise ConfigurationError("crash_test needs build, an integer seed, and positive max_legs.")
    if isolation == "subprocess":
        from ._subprocess import import_reference

        import_reference(build)
        import_reference(scenario)
    if not isinstance(session_id, str) or not session_id:
        raise ConfigurationError("crash_test requires a nonempty session_id.")
    if points not in {"all", "all-raw", "effects", "recovery"}:
        raise ConfigurationError("Unknown crash point selection.")
    chosen = faults or Faults.default()
    if isolation == "subprocess":
        unsupported = chosen.kinds & _SUBPROCESS_UNSUPPORTED
        if unsupported and chosen.kinds != Faults.DEFAULT_KINDS:
            raise ConfigurationError(
                "Subprocess crash testing cannot thaw a terminated worker. Unsupported faults: "
                + ", ".join(sorted(unsupported))
            )
        if unsupported:
            chosen = Faults(
                kinds=chosen.kinds - unsupported, adapters=chosen.adapters, stall_s=chosen.stall_s
            )
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
            isolation=isolation,
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
            if scenario is None:
                for variant in _held_delivery_points(point, result, chosen, answer):
                    deeper[variant.id] = variant
            if "advance_boundaries" in chosen.kinds and result.crash_snapshots:
                crashed_legs = {plan.leg for plan in point.plans if plan.id in result.triggered}
                recovery_leg = max(crashed_legs, default=0) + 1
                for boundary in boundary_points(
                    point, _recorded_boundaries(result), leg=recovery_leg
                ):
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
        isolation=isolation,
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
            isolation=isolation,
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
        if kind in {"retry_effect", "retry_changed"} and not any(
            item.request.messages and any(message.tool_calls for message in item.request.messages)
            for item in calls
            if item.leg == leg
        ):
            continue
        identity = f"{prefix.id}+leg{leg}:model/{kind}"
        yield Point(
            identity,
            (*prefix.plans, FaultPlan(identity, kind, 1, leg)),
            in_effect_window=prefix.in_effect_window,
            recovery=True,
            variant=prefix.variant,
        )


def _held_delivery_points(prefix, result, faults, answer):
    """Exercise a competing caller only after a scenario actually hit a hold."""
    if not result.triggered:
        return
    kinds = {"new_run_on_held", "double_resume"}
    if answer is not None:
        kinds.add("operator_race")
    held_leg = next(
        (event.leg for event in result.boundaries if event.kind == "held_delivery"), None
    )
    if held_leg is None:
        return
    for kind in sorted(faults.kinds & kinds):
        identity = f"{prefix.id}+leg{held_leg}:{kind}"
        yield Point(
            identity,
            (*prefix.plans, FaultPlan(identity, kind, 1, held_leg)),
            in_effect_window=True,
            recovery=True,
            variant=prefix.variant,
        )


async def _deliver_while_held(agent, session, deps, answer, held, plan, hit, finding):
    """Try a competing action, then return control to the normal recovery legs."""

    def unexpected(detail):
        finding("unsafe_held_delivery", detail)

    if plan.kind == "new_run_on_held":
        hit(plan)
        try:
            await agent.run("Competing run during recovery", session=session, deps=deps)
        except HeldEffectsError:
            return
        unexpected("A newly submitted run took over a held call.")
        return

    owner = f"harness-recovery-owner-{plan.leg}"
    acquired = await session.store.acquire_lease(session.id, owner, agent.lease_ttl_s)
    if not acquired:
        raise LeaseError("The harness could not obtain the competing recovery lease.")
    try:
        if plan.kind == "double_resume":
            hit(plan)
            try:
                await agent.resume(session, deps=deps)
            except SessionError as error:
                if "in progress elsewhere" in str(error):
                    return
                raise
            unexpected("A competing resume ran while another worker held the lease.")
        elif (decision := answer(held[0])) is not None:
            hit(plan)
            try:
                await agent.settle(session, decision, deps=deps)
            except SettlementConflict as error:
                if error.reason == "session_busy":
                    return
                raise
            unexpected("An operator settlement ran while another worker held the lease.")
    finally:
        await session.store.release_lease(session.id, owner)


async def _drive_agent(
    agent,
    session,
    deps,
    *,
    env,
    prompt,
    state,
    scenario,
    answers,
    needs_recheck,
    stale,
    operator_answer,
    has_answer,
    delivery,
    observe,
    hit,
    finding,
    clock,
):
    """Use identical Agent entry points inside either kind of leg."""
    if needs_recheck:
        durable_holds = (await session.state()).effects.held(session.id)
        if durable_holds:
            observe()
            if delivery is not None:
                await _deliver_while_held(
                    agent, session, deps, operator_answer, durable_holds, delivery, hit, finding
                )
        pending = await agent.recheck(session, deps=deps)
        if pending and has_answer:
            for item in pending:
                if (decision := operator_answer(item)) is not None:
                    await agent.settle(session, decision, deps=deps)
            pending = [item.hold for item in await session.effects() if item.hold]
        if pending and not any(item.retry_after for item in pending):
            return pending
        if pending:
            later = min(item.retry_after.timestamp() for item in pending if item.retry_after)
            clock.advance(max(0, later - clock.store_time()) + 0.001)
    if scenario is not None:
        return await scenario(env, agent, deps)
    if (await session.state()).open_run_id is None:
        return await agent.run(prompt, session=session, deps=deps, state=state)
    if stale is not None and answers:
        changed = {
            key: value.model_copy(update={"approved": not value.approved})
            for key, value in answers.items()
        }
        try:
            result = await agent.resume(session, deps=deps, approvals=changed)
        except ApprovalConflictError:
            pass
        else:
            finding("stale_approval_accepted", "A conflicting approval answer was accepted.")
            return result
    return await agent.resume(session, deps=deps, approvals=answers)


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
    isolation,
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
        "isolation": isolation,
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
            "isolation": config.get("isolation", "inprocess"),
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
    isolation="inprocess",
):
    with TemporaryDirectory(dir=directory, prefix="qagents-crash-") as temp:
        loop = VirtualLoop(
            ManualClock(store_time=1_000_000),
            thread_timeout_s=45 if isolation == "subprocess" else 10,
        )
        world = World(loop, mode=mode, session_id=session_id)
        fault_world = world
        if isolation == "subprocess":
            from ._subprocess import ProcessWorld

            fault_world = ProcessWorld(world)
        controller = FaultController(fault_world, point.plans, faults=faults)
        controller.install()
        destination = HarnessTransport(
            fault_world,
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
            isolation=isolation,
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
    isolation,
):
    effects, undeclared, reached, snapshots, checks = {}, set(), [], [], []
    replayable = set()
    output, held = None, []
    errors = []
    approval_keys = {}
    usage_limits = {}
    perturbation = next((p for p in point.plans if p.kind in PERTURBATIONS), None)
    stale = next((p for p in point.plans if p.kind == "stale_answer"), None)

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
                diagnostics = _ports.FINDING_OBSERVER.set(
                    lambda kind, detail: world.findings.append(
                        {"kind": kind, "detail": detail, "leg": number}
                    )
                )
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
                    delivery = next(
                        (
                            plan
                            for plan in point.plans
                            if plan.leg == number
                            and plan.kind in {"new_run_on_held", "double_resume", "operator_race"}
                        ),
                        None,
                    )
                    return await _drive_agent(
                        agent,
                        session,
                        deps,
                        env=env,
                        prompt=prompt,
                        state=state,
                        scenario=scenario,
                        answers=answers,
                        needs_recheck=needs_recheck,
                        stale=stale
                        if stale and number == stale.leg + 1 and stale.id in controller.triggered
                        else None,
                        operator_answer=answer,
                        has_answer=answer is not None,
                        delivery=delivery,
                        observe=lambda: world.observe(number, "held_delivery"),
                        hit=controller.hit,
                        finding=lambda kind, detail: world.findings.append(
                            {"kind": kind, "detail": detail, "leg": number}
                        ),
                        clock=world.clock,
                    )
                finally:
                    _ports.FINDING_OBSERVER.reset(diagnostics)
                    _ports.REACH_OBSERVER.reset(observer)
                    _ports.TRANSPORT.reset(token)

            process = None
            if isolation == "subprocess":
                from ._subprocess import ProcessLeg, import_reference

                delivery = next(
                    (
                        plan
                        for plan in point.plans
                        if plan.leg == number
                        and plan.kind in {"new_run_on_held", "double_resume", "operator_race"}
                    ),
                    None,
                )
                process = ProcessLeg(
                    world, controller, destination, env.model, number, reached, answer
                )
                task = world.create_task(
                    number,
                    process.run(
                        {
                            "number": number,
                            "directory": directory,
                            "mask_features": mask_features,
                            "build": import_reference(build),
                            "scenario": import_reference(scenario),
                            "prompt": prompt,
                            "state": state,
                            "session_id": world.session_id,
                            "seed": seed,
                            "transport": destination.mode,
                            "answers": answers,
                            "needs_recheck": needs_recheck,
                            "stale": stale
                            if stale
                            and number == stale.leg + 1
                            and stale.id in controller.triggered
                            else None,
                            "delivery": delivery,
                            "has_answer": answer is not None,
                        }
                    ),
                )
                await asyncio.wait({task})
                if process.declarations is not None:
                    declared, tools, key, limits = process.declarations
                    effects.update(declared)
                    approval_keys[number], usage_limits[number] = key, limits
                    replayable.update(
                        name for name, effectful, read, safe in tools if effectful or read or safe
                    )
                    undeclared.update(
                        name for name, effectful, read, _ in tools if not effectful and not read
                    )
            else:
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
                    failure = task.exception()
                    if process is not None and failure is not None:
                        raise failure
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
            except (LeaseError, JournalOutcomeUnknown, asyncio.CancelledError, OSError) as error:
                targeted = [
                    plan
                    for plan in point.plans
                    if plan.leg == number and plan.id in controller.triggered
                ]
                if not targeted:
                    raise
                leases = {"acquire_lease", "acquire_lease_grant"}
                lost_acquisition = any(plan.method in leases for plan in targeted)
                if isinstance(error, OSError) and not lost_acquisition:
                    raise
                if lost_acquisition:
                    status = await backend._backend.status(world.session_id)
                    if status.lease_expires_at is not None:
                        cutoff = status.lease_expires_at + backend.takeover_grace_s
                        world.clock.advance(max(0, cutoff - world.clock.store_time()) + 0.001)
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
            if isolation == "subprocess":
                pending = {
                    task
                    for fake in destination.contracts.values()
                    for task in fake.pending_destinations
                }
                if pending:
                    loop.run_until_complete(asyncio.wait(pending))
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
