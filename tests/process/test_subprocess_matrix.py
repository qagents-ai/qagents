"""The public process runner and the complete in-process crash matrix."""

import os

import pytest
from pydantic import BaseModel

from qagents import Agent, RunContext, tool
from qagents.effects import Attempt, Dedupe, EffectContext, Found, Inquiry, Key, Unknown, effect
from qagents.errors import ConfigurationError
from qagents.models.testing import call
from qagents.session import Entry
from qagents.testing.crash import FakeStripeRefunds, Faults, assert_recovers, crash_test


class Request(BaseModel):
    id: str


class Receipt(BaseModel):
    id: str


# Run the exhaustive matrix in the process CI lane.
pytestmark = [
    pytest.mark.process,
    pytest.mark.skipif(
        os.environ.get("QAGENTS_RUN_PROCESS_TESTS") != "1",
        reason="enable the process CI lane to run the full matrix",
    ),
]
_BUILDS = 0


def build_refund(env):
    global _BUILDS
    _BUILDS += 1

    @effect(
        "stripe.refund",
        key=Key("request", "id"),
        dedupe=Dedupe(120, contract="fixture refund key", concurrent="rejects", margin_s=0.1),
        timeout_s=1,
    )
    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        created = await ctx.deps.create_refund(
            charge="ch_A", amount=100, idempotency_key=attempt.idempotency_key
        )
        return Receipt(id=created.id)

    @tool(effects=[send])
    async def refund(ctx: RunContext, id: str) -> str:
        """Refund one fixture payment."""
        return (await ctx.perform(send, Request(id=id))).id

    @tool(read_only=True)
    def build_count() -> int:
        """Return the number of factory calls in this interpreter."""
        return _BUILDS

    return Agent(env.model, tools=[refund, build_count]), env.service(send, FakeStripeRefunds)


def build_streaming_lookup(env):
    @effect(
        "stripe.refund",
        key=Key("request", "id"),
        timeout_s=1,
    )
    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        created = await ctx.deps.create_refund(
            charge="ch_A",
            amount=100,
            idempotency_key=attempt.idempotency_key,
            metadata={"operation": attempt.operation_tag},
        )
        return Receipt(id=created.id)

    @send.lookup
    async def lookup(ctx: EffectContext, inquiry: Inquiry[Request]):
        async for refund in ctx.deps.refunds.iter_refunds(charge="ch_A"):
            if refund.metadata["operation"] == inquiry.operation_tag:
                return Found(Receipt(id=refund.id), tag=refund.metadata["operation"])
        return Unknown()

    @tool(effects=[send])
    async def refund(ctx: RunContext, id: str) -> str:
        """Refund one fixture payment."""
        return (await ctx.perform(send, Request(id=id))).id

    return Agent(env.model, tools=[refund]), env.service(send, FakeStripeRefunds)


def build_provision(env):
    @effect(
        "provision.instance",
        key=Key("request", "id"),
        dedupe=Dedupe(120, contract="fixture provisioning key", concurrent="waits", margin_s=0.1),
        timeout_s=1,
    )
    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        raise AssertionError("This fixture uses the parent-owned destination.")

    @tool(effects=[send])
    async def provision(ctx: RunContext, id: str) -> str:
        """Provision one fixture instance."""
        return (await ctx.perform(send, Request(id=id))).id

    return Agent(env.model, tools=[provision])


def _effect_result(point):
    return (
        point.verdict,
        point.outcome,
        [(app.fake, app.attempt, app.epoch, app.replayed) for app in point.execution.applications],
        [(event.effect, event.rule, event.point) for event in point.execution.events],
    )


@pytest.mark.parametrize(
    "build,name,transport",
    [(build_refund, "refund", "client"), (build_provision, "provision", "fake")],
)
async def test_full_crash_matrix(build, name, transport, tmp_path):
    script = [call(name, id="A1"), "Done."]
    # Enumerate every supported kill boundary at both depths. The in-process
    # tests retain freeze/thaw and the other non-process fault models.
    report = await crash_test(
        build,
        "Apply A1",
        model=script,
        directory=tmp_path,
        isolation="subprocess",
        transport=transport,
        faults=Faults.parse(
            "kill_before,kill_after,recovery_kill,commit_then_crash,"
            "commit_then_lose_response,advance_boundaries,"
            "ack_lost_committed,ack_lost_uncommitted,lease_ack_lost"
        ),
        points="all-raw",
    )
    (tmp_path / f"{name}-subprocess.json").write_text(report.to_json(), encoding="utf-8")
    unsafe = [p for p in report.points if p.verdict != "safe"]
    assert not unsafe, "\n".join(report.explain(p.id) for p in unsafe)
    assert report.points[0].outcome == "completed_as_reference", report.explain("reference")
    assert not report.failures, report.explain()
    assert any("recovery_kill" in p.execution.point.kinds for p in report.points)
    assert all(
        sum(not app.replayed for app in p.execution.applications) <= 1 for p in report.points
    ), report.explain()


async def test_subprocess_uses_fresh_module_state_and_supports_pins(tmp_path):
    report = await crash_test(
        build_refund,
        "Apply A1",
        model=[call("refund", id="A1"), call("build_count"), "Done."],
        directory=tmp_path,
        isolation="subprocess",
        transport="client",
        points="effects",
        faults=Faults.parse("kill_after,commit_then_crash"),
    )
    for point in report.points:
        assert point.verdict == "safe", report.explain(point.id)
        snapshot = point.execution.final_snapshot
        assert snapshot is not None
        entries = [Entry.model_validate_json(raw) for raw in snapshot.entries_json]
        assert any(
            entry.kind == "tool_result"
            and entry.data["result"]["name"] == "build_count"
            and entry.data["result"]["content"] == "1"
            for entry in entries
        ), report.explain(point.id)
    point = next(p for p in report.points if "commit_then_crash" in p.execution.point.kinds)
    assert point.execution.legs == 2
    assert len(point.execution.crash_snapshots) == 1
    assert [app.replayed for app in point.execution.applications] == [False, True]
    await assert_recovers(build_refund, report.pin(point.id, tmp_path / "pins"))


async def test_subprocess_and_inprocess_agree_at_real_kill_points(tmp_path):
    points = []
    for isolation in ("inprocess", "subprocess"):
        report = await crash_test(
            build_provision,
            "Apply A1",
            model=[call("provision", id="A1"), "Done."],
            directory=tmp_path,
            isolation=isolation,
            transport="fake",
            points="all-raw",
            faults=Faults.parse("kill_before,kill_after,commit_then_crash"),
        )
        points.append({p.id: _effect_result(p) for p in report.points})
    assert points[0] == points[1]


async def test_subprocess_refuses_nonimportable_build_and_zombie_faults():
    with pytest.raises(ConfigurationError, match="must be importable"):
        await crash_test(
            lambda env: build_provision(env), "A1", model=["Done."], isolation="subprocess"
        )
    with pytest.raises(ConfigurationError, match="cannot thaw"):
        await crash_test(
            build_provision,
            "A1",
            model=["Done."],
            isolation="subprocess",
            faults=Faults.parse("zombie"),
        )


async def test_subprocess_runs_client_lookup_after_lost_response(tmp_path):
    report = await crash_test(
        build_streaming_lookup,
        "Apply A1",
        model=[call("refund", id="A1"), "Done."],
        directory=tmp_path,
        isolation="subprocess",
        transport="client",
        points="effects",
        faults=Faults.parse("commit_then_crash"),
    )
    point = next(p for p in report.points if p.execution.triggered)
    assert point.verdict == "safe", report.explain(point.id)
    assert point.outcome == "completed_as_reference", report.explain(point.id)
    assert [app.replayed for app in point.execution.applications] == [False]
    assert [e.phase for e in point.execution.boundaries if e.kind == "effect_lookup"] == [
        "before",
        "after",
    ]
    assert any(
        e.kind == "service_call" and e.detail == "stripe.refund.refunds.iter_refunds"
        for e in point.execution.boundaries
    )


async def test_subprocess_recovers_from_cancellation_during_destination_send(tmp_path):
    report = await crash_test(
        build_provision,
        "Apply A1",
        model=[call("provision", id="A1"), "Done."],
        directory=tmp_path,
        isolation="subprocess",
        transport="fake",
        points="effects",
        faults=Faults.parse("cancel_during"),
    )
    send = next(p for p in report.points if p.id.startswith("send("))
    assert send.execution.triggered == (send.id,)
    assert send.verdict == "safe", report.explain(send.id)
    assert send.outcome == "completed_as_reference"
    assert sum(not item.replayed for item in send.execution.applications) == 1


async def test_subprocess_observes_late_destination_and_preserves_the_hold(tmp_path):
    report = await crash_test(
        build_provision,
        "Apply A1",
        model=[call("provision", id="A1"), "Done."],
        directory=tmp_path,
        isolation="subprocess",
        transport="fake",
        points="effects",
        faults=Faults.parse("commit_late,not_yet_visible_then_commit,new_run_on_held"),
    )
    recovered = [point for point in report.points if point.execution.triggered]
    assert len(recovered) == 4, report.explain()
    for point in recovered:
        assert point.verdict == "safe", report.explain(point.id)
        assert point.outcome == "paused_awaiting_operator"
        assert len(point.execution.applications) == 1


async def test_subprocess_reports_definite_rejection_as_missing_required_work(tmp_path):
    report = await crash_test(
        build_provision,
        "Apply A1",
        model=[call("provision", id="A1"), "Done."],
        directory=tmp_path,
        isolation="subprocess",
        transport="fake",
        points="effects",
        faults=Faults.parse("reject"),
    )
    point = next(p for p in report.points if p.execution.triggered)
    assert point.verdict == "violated", report.explain(point.id)
    assert not point.execution.applications
    assert any(
        finding.code == "missing_application_or_hold"
        for check in point.checks
        for finding in check.findings
    )
