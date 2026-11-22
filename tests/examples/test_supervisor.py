"""Offline tests of examples/patterns/supervisor.py."""

from __future__ import annotations

import json
import runpy
import sys

import pytest

from examples.harness import last_text, main, scripted_context, tool_names
from examples.patterns import supervisor
from examples.patterns.supervisor import COMPANIES, calculate, digits, run, search
from qagents import ToolError, Usage
from qagents.models.testing import FunctionModel, call


def tool_turns(request) -> int:
    return sum(1 for message in request.messages if message.role == "tool")


def policy(request):
    """Plays all three agents; which one is asking is told by the tools it has."""
    tools = tool_names(request)
    answered = request.messages[-1].role == "tool"
    if tools == {"search"}:  # the researcher: three searches at once, then the facts
        if answered:
            return last_text(request)
        return [call("search", query=name) for name in COMPANIES]
    if tools == {"calculate"}:  # the analyst
        if answered:
            return f"The total is {last_text(request).split('=')[-1].strip()}."
        return call("calculate", expression="12,480 + 8,215 + 4,903")
    assert tools == {"researcher", "analyst"}, tools  # the supervisor
    step = tool_turns(request)
    if step == 0:
        return call("researcher", task="Find the 2025 headcount of " + ", ".join(COMPANIES))
    if step == 1:
        return call("analyst", task=f"Add up these headcounts:\n{last_text(request)}")
    return f"Report: the three companies had 25,598 employees in 2025. {last_text(request)}"


async def test_the_supervisor_delegates_and_reports_the_total(tmp_path, capsys):
    ctx = scripted_context(policy, tmp_path)
    outcome = await run(ctx)
    assert outcome.ok, outcome.summary
    # Three requests by the supervisor, two by each worker: all counted in the supervisor's run.
    assert outcome.details["supervisor_requests"] == 3
    assert outcome.details["total_requests"] == 7 == ctx.meter.usage.requests
    events = outcome.details["worker_events"]
    assert set(events) == {"researcher", "analyst"} and min(events.values()) > 0
    printed = capsys.readouterr().out
    assert '[researcher] -> search(query="Borealis Energy")' in printed
    assert '[analyst] -> calculate(expression="12,480 + 8,215 + 4,903")' in printed

    # Each worker saw only the task the supervisor wrote, never the user's question.
    analyst_first = next(r for r in ctx.meter.requests if tool_names(r) == {"calculate"})
    assert [m.role for m in analyst_first.messages] == ["system", "user"]
    assert analyst_first.messages[-1].text.startswith("Add up these headcounts:")
    # And the supervisor saw only the workers' answers, never their searches or calculations.
    supervisor_last = [r for r in ctx.meter.requests if tool_names(r) == {"researcher", "analyst"}]
    results = [r.name for m in supervisor_last[-1].messages for r in m.tool_results]
    assert results == ["researcher", "analyst"]


async def test_a_report_without_the_total_is_not_ok(tmp_path):
    def vague(request):
        if tool_names(request) == {"researcher", "analyst"} and tool_turns(request) == 2:
            return "Report: together they employ roughly 26 thousand people."
        return policy(request)

    outcome = await run(scripted_context(vague, tmp_path))
    assert not outcome.ok and "does not state 25,598" in outcome.summary


async def test_the_supervisors_budget_stops_a_worker_that_has_none(tmp_path, capsys):
    # $0.16 a request against the supervisor's $0.30: the researcher starts with the $0.14
    # left, stops after one request, and the supervisor, now over budget, stops too.
    priced = FunctionModel(policy, usage=Usage(requests=1, cost_usd=0.16))
    ctx = scripted_context(priced, tmp_path)
    outcome = await run(ctx)
    assert not outcome.ok
    assert (outcome.details["status"], outcome.details["stop_reason"]) == (
        "limit_reached",
        "max_cost_usd",
    )
    assert ctx.meter.usage.requests == outcome.details["total_requests"] == 2
    assert outcome.details["cost_usd"] == pytest.approx(0.32)
    assert "analyst" not in outcome.details["worker_events"]
    printed = capsys.readouterr().out
    assert "<- researcher error: researcher stopped before finishing (max_cost_usd)" in printed


def test_search_and_calculate():
    assert search.function("acme robotics") == COMPANIES["Acme Robotics"]
    assert search.function("headcounts").startswith("No results.")
    assert calculate.function("12,480 + 8,215 + 4,903") == "12,480 + 8,215 + 4,903 = 25598"
    assert calculate.function("(10 - 4) / 4") == "(10 - 4) / 4 = 1.5"
    for unsafe in ("__import__('os')", "x + 1", "2 ** 8", "1 / 0", "3 +"):
        with pytest.raises(ToolError):
            calculate.function(unsafe)
    assert digits("25,598 and 25 598 and 25598") == "25598 and 25598 and 25598"


def test_runs_as_a_module_and_replays_from_its_cassette(tmp_path, monkeypatch, capsys):
    common = ["--provider", "anthropic", "--cassette-dir", str(tmp_path / "cassettes"), "--json"]
    with pytest.raises(SystemExit) as recorded:
        main(
            run,
            name=supervisor.NAME,
            description=supervisor.DESCRIPTION,
            inspired_by=supervisor.INSPIRED_BY,
            argv=[*common, "--record", "--workdir", str(tmp_path / "recorded")],
            model_factory=lambda provider, model: FunctionModel(policy),
        )
    assert recorded.value.code == 0
    capsys.readouterr()

    argv = ["supervisor", *common, "--replay", "--workdir", str(tmp_path / "replayed")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as replayed:
        runpy.run_path(supervisor.__file__, run_name="__main__")
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert replayed.value.code == 0, report
    assert report["ok"] and report["name"] == "supervisor" and report["requests"] == 7
