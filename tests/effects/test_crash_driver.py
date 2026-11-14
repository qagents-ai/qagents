"""Public crash exploration over a real Agent and destination observations."""

from qagents import Agent, RunContext, tool
from qagents.effects import Attempt, Dedupe, EffectContext, Key, Settle, effect
from qagents.models.recording import RecordingModel
from qagents.models.testing import ScriptedModel, call
from qagents.session import Entry
from qagents.testing.crash import Cassette, FakeStripeRefunds, Faults, assert_recovers, crash_test

from .test_engine_dispatch import Receipt, Request


def build(env):
    @effect(
        "refund",
        key=Key("request", "id"),
        timeout_s=2,
        dedupe=Dedupe(3600, contract="offline fixture", concurrent="waits"),
    )
    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        raise AssertionError("this scenario uses fake transport")

    @tool(effects=[send])
    async def refund(ctx: RunContext, id: str) -> str:
        """Refund the fixture order."""
        return (await ctx.perform(send, Request(id=id))).id

    return Agent(env.model, tools=[refund])


async def test_explorer_uses_a_fresh_writer_and_destination_after_kill(tmp_path):
    report = await crash_test(
        build,
        "Refund",
        directory=tmp_path,
        store="sqlite",
        model=[call("refund", id="R1"), "Complete."],
        faults=Faults.parse("kill_after,commit_then_lose_response"),
        transport="fake",
        points="effects",
    )
    reference = report.points[0]
    assert reference.outcome == "completed_as_reference", report.explain()
    assert len(reference.execution.applications) == 1
    assert all(
        item.status == "safe"
        for item in reference.checks
        if item.invariant in {"I2", "I7a", "I7b", "I12"}
    ), report.explain("reference")
    killed = [p for p in report.points if "kill_after" in p.execution.point.kinds]
    assert killed and any(p.execution.crash_snapshots for p in killed)
    assert any(p.execution.legs > 1 and p.outcome == "completed_as_reference" for p in killed), (
        report.explain()
    )
    assert all(
        sum(not application.replayed for application in p.execution.applications) <= 1
        for p in report.points
    ), report.explain()
    assert report.fault_model[0].endswith("NOT exercised through client transport")
    assert not report.strict_ok

    point = next(p for p in killed if p.execution.triggered and p.verdict == "safe")
    pinned = report.pin(point.id, tmp_path / "pins")
    await assert_recovers(build, pinned)


async def test_recovery_explores_the_recorded_dedupe_boundary(tmp_path):
    report = await crash_test(
        build,
        "Refund",
        directory=tmp_path,
        store="sqlite",
        model=[call("refund", id="R1"), "Complete."],
        faults=Faults.parse("kill_after,advance_boundaries"),
        transport="fake",
        points="effects",
    )
    passes = [
        point
        for point in report.points
        if "advance_boundaries" in point.execution.point.kinds
        and len(point.execution.triggered) == 2
    ]
    assert passes, report.explain()
    assert any(
        item.reason.value == "WINDOW_EXPIRED" for point in passes for item in point.execution.held
    ), report.explain()


async def test_pinned_cassette_keeps_its_offline_model_after_original_is_removed(tmp_path):
    path = tmp_path / "recording.json"
    model = RecordingModel(ScriptedModel(["Reply."]), path)

    def read_only(env):
        return Agent(env.model, name="reader")

    await Agent(model, name="reader").run("Read the fixture")
    report = await crash_test(
        read_only,
        "Read the fixture",
        model=Cassette(path),
        store="sqlite",
        directory=tmp_path,
        faults=Faults.parse("kill_before"),
    )
    point = next(
        item for item in report.points if item.execution.triggered and item.verdict == "safe"
    )
    pin = report.pin(point.id, tmp_path / "pins")
    path.unlink()
    await assert_recovers(read_only, pin)


async def test_worker_thawed_before_second_fence_cannot_send_on_expired_lease(tmp_path):
    report = await crash_test(
        build,
        "Refund",
        directory=tmp_path,
        store="sqlite",
        model=[call("refund", id="R1"), "Complete."],
        points="effects",
        faults=Faults.parse("freeze_before_fence2"),
        transport="fake",
    )
    (point,) = report.points[1:]
    assert point.execution.triggered, report.explain()
    assert point.verdict == "safe", report.explain(point.id)
    assert all(app.leg > 0 for app in point.execution.applications)


async def test_freeze_inside_send_is_reported_outside_the_clock_assumptions(tmp_path):
    report = await crash_test(
        build,
        "Refund",
        directory=tmp_path,
        store="sqlite",
        model=[call("refund", id="R1"), "Complete."],
        points="effects",
        faults=Faults.parse("freeze_in_send"),
        transport="fake",
    )
    (point,) = report.points[1:]
    assert point.execution.outside_a3 == point.execution.triggered
    assert point.execution.triggered
    assert not report.strict_ok


async def test_swallowed_pause_remains_held_and_appears_in_harness_report(tmp_path):
    def swallowing(env):
        @effect("refund", key=Key("request", "id"))
        async def destination(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
            raise AssertionError("Only the fixture transport runs.")

        @tool(effects=[destination])
        async def refund(ctx: RunContext, id: str) -> str:
            """Return a value even when the refund incurs a hold."""
            try:
                return (await ctx.perform(destination, Request(id=id))).id
            except Exception:
                return "swallowed"

        return Agent(env.model, tools=[refund])

    report = await crash_test(
        swallowing,
        "Refund",
        directory=tmp_path,
        store="sqlite",
        model=[call("refund", id="R1"), "Complete."],
        points="effects",
        faults=Faults.parse("commit_then_lose_response"),
        transport="fake",
    )
    (point,) = report.points[1:]
    assert point.execution.held
    assert point.outcome == "paused_awaiting_operator"
    assert point.execution.final_snapshot is not None
    entries = [
        Entry.model_validate_json(raw) for raw in point.execution.final_snapshot.entries_json
    ]
    assert not any(
        entry.kind == "tool_result" and not entry.data["result"]["is_error"] for entry in entries
    )
    assert any(item.code == "swallowed_hold" for item in report.failures), report.explain()


async def test_client_transport_runs_send_body_and_preserves_destination_key(tmp_path):
    def configured(env):
        @effect(
            "stripe.refund",
            key=Key("request", "id"),
            dedupe=Dedupe(86400, contract="offline Stripe refund semantics", concurrent="rejects"),
        )
        async def destination(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
            created = await ctx.deps.create_refund(
                charge="ch_A",
                amount=100,
                idempotency_key=attempt.idempotency_key,
                metadata={"refund_intent": attempt.request.id},
            )
            return Receipt(id=created.id)

        @tool(effects=[destination])
        async def refund(ctx: RunContext, id: str) -> str:
            """Issue a refund to the fixture Stripe destination."""
            return (await ctx.perform(destination, Request(id=id))).id

        return Agent(env.model, tools=[refund]), env.service(destination, FakeStripeRefunds)

    report = await crash_test(
        configured,
        "Refund",
        directory=tmp_path,
        store="sqlite",
        model=[call("refund", id="R1"), "Complete."],
        points="effects",
        transport="client",
        faults=Faults.parse("kill_after,commit_then_lose_response,advance_boundaries"),
    )
    assert report.ok, report.explain()
    assert not report.strict_ok
    assert not any("NOT exercised" in line for line in report.fault_model)
    assert any(
        application.key_passed
        for point in report.points
        for application in point.execution.applications
    )


async def test_held_effect_refuses_a_competing_run_resume_and_settlement(tmp_path):
    def opaque(env):
        @effect("refund", key=Key("request", "id"))
        async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
            raise AssertionError("The fixture transport owns the destination.")

        @tool(effects=[send])
        async def refund(ctx: RunContext, id: str) -> str:
            """Refund the fixture order."""
            return (await ctx.perform(send, Request(id=id))).id

        return Agent(env.model, tools=[refund])

    def settle(item):
        return Settle.committed(item, receipt=Receipt(id="R1"), by="qa", evidence="ledger")

    report = await crash_test(
        opaque,
        "Refund",
        directory=tmp_path,
        store="sqlite",
        model=[call("refund", id="R1"), "Complete."],
        answer=settle,
        points="effects",
        faults=Faults.parse(
            "commit_then_lose_response,new_run_on_held,double_resume,operator_race"
        ),
        transport="fake",
    )
    for kind in ("new_run_on_held", "double_resume", "operator_race"):
        matching = [p for p in report.points if kind in p.execution.point.kinds]
        assert len(matching) == 1, report.explain()
        (point,) = matching
        assert len(point.execution.triggered) == 2, report.explain(point.id)
        assert point.verdict == "safe", report.explain(point.id)
        assert point.outcome == "completed_as_reference", report.explain(point.id)
        assert len(point.execution.applications) == 1
