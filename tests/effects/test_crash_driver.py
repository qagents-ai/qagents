"""Public crash exploration over a real Agent and destination observations."""

from qagents import Agent, RunContext, tool
from qagents.effects import Attempt, Dedupe, EffectContext, Key, effect
from qagents.models.testing import call
from qagents.testing.crash import Faults, assert_recovers, crash_test

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
