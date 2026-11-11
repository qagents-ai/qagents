"""Offline tests of examples/patterns/hierarchical_teams.py."""

from __future__ import annotations

import asyncio
import json
import runpy
import sys

import pytest

from examples.harness import last_text, main, scripted_context, system_text, tool_names
from examples.patterns import hierarchical_teams
from examples.patterns.hierarchical_teams import (
    BRIEF,
    MARKET,
    SUB_QUESTIONS,
    RunTree,
    build_manager,
    run,
    search,
    uncovered,
)
from qagents import Usage
from qagents.models.testing import FunctionModel, call

EXPECTED_RUNS = [
    "manager@0",
    "research_lead@1",
    "searcher@2",
    "searcher@2",
    "searcher@2",
    "writing_lead@1",
    "writer@2",
    "editor@2",
]


def tool_turns(request) -> int:
    return sum(1 for message in request.messages if message.role == "tool")


def policy(request):
    """Plays all six agents; which one is asking is told by its tools or its instructions."""
    tools = tool_names(request)
    answered = request.messages[-1].role == "tool"
    step = tool_turns(request)
    if tools == {"search"}:  # a searcher: search for the task, then answer with the facts
        return last_text(request) if answered else call("search", query=last_text(request))
    if tools == {"searcher_parallel"}:  # the research lead
        if not answered:
            return call("searcher_parallel", tasks=SUB_QUESTIONS)
        items = json.loads(last_text(request))
        return "Findings:\n" + "\n".join(item.get("answer") or item["error"] for item in items)
    if tools == {"writer", "editor"}:  # the writing lead
        if step == 0:
            return call("writer", task=last_text(request))
        if step == 1:
            return call("editor", task=last_text(request))
        return last_text(request)
    if tools == {"research_lead", "writing_lead"}:  # the manager
        if step == 0:
            return call("research_lead", task=BRIEF)
        if step == 1:
            return call("writing_lead", task=f"{BRIEF}\n\n{last_text(request)}")
        return last_text(request)
    if system_text(request).startswith("You write a briefing"):
        return f"# Heat pumps in 2025\n\n{last_text(request)}"
    assert system_text(request).startswith("You edit"), system_text(request)
    return last_text(request)


class Overlap(FunctionModel):
    """Makes searcher requests slow and records how many were in flight at once."""

    def __init__(self, function) -> None:
        super().__init__(function)
        self.active = self.peak = 0

    async def stream(self, request):
        searcher = tool_names(request) == {"search"}
        self.active += searcher
        self.peak = max(self.peak, self.active)
        try:
            if searcher:
                await asyncio.sleep(0.02)
            async for event in super().stream(request):
                yield event
        finally:
            self.active -= searcher


def tool_result(result, name: str):
    return next(r for m in result.messages for r in m.tool_results if r.name == name)


async def test_the_manager_runs_both_teams_and_covers_every_sub_question(tmp_path, capsys):
    model = Overlap(policy)
    ctx = scripted_context(model, tmp_path)
    outcome = await run(ctx)
    assert outcome.ok, outcome.summary
    assert outcome.details["runs"] == EXPECTED_RUNS and outcome.details["missing"] == []
    # manager 3, research lead 2, searchers 3 x 2, writing lead 3, writer 1, editor 1: all of
    # them counted in the manager's run, because every level adds its children's usage.
    assert outcome.details["requests"] == 16 == ctx.meter.usage.requests
    assert model.peak == 3  # the three searchers worked at the same time

    # Each searcher worked on one sub-question in a conversation of its own.
    firsts = [r for r in ctx.meter.requests if tool_names(r) == {"search"} and len(r.messages) == 2]
    assert sorted(r.messages[-1].text for r in firsts) == sorted(SUB_QUESTIONS)
    # The writing team got the findings only through the manager's task: they share no context.
    writer_task = next(r for r in ctx.meter.requests if system_text(r).startswith("You write"))
    assert uncovered(writer_task.messages[-1].text) == []
    printed = capsys.readouterr().out
    assert "[research_lead] [searcher[1]] -> search(" in printed
    assert "[writing_lead] -> editor(" in printed


async def test_max_depth_stops_the_level_below_it(tmp_path):
    tree = RunTree()
    manager = build_manager(scripted_context(policy, tmp_path).model, hooks=[tree], max_depth=1)
    result = await manager.run(BRIEF)
    assert result.status == "completed"
    assert sorted(tree.runs) == [(0, "manager"), (1, "research_lead"), (1, "writing_lead")]
    # The searchers refused to start at depth 2; the research lead reported it upward.
    assert "Limits.max_depth is 1" in tool_result(result, "research_lead").content


async def test_the_managers_budget_caps_every_level(tmp_path):
    priced = FunctionModel(policy, usage=Usage(requests=1, cost_usd=0.01))
    ctx = scripted_context(priced, tmp_path)
    tree = RunTree()
    result = await build_manager(ctx.model, hooks=[tree], max_cost_usd=0.025).run(BRIEF)
    # The manager spent 0.01, its research lead 0.01 of the remaining 0.015, and each searcher
    # started with the 0.005 left: one request each, then every level stopped at its share.
    # The three searchers started together, each with that snapshot, so the tree spent 0.05:
    # the overshoot of up to max_parallel runs that fan_out documents.
    assert (result.status, result.stop_reason) == ("limit_reached", "max_cost_usd")
    assert ctx.meter.usage.requests == 5 and result.usage.cost_usd == pytest.approx(0.05)
    stopped = tool_result(result, "research_lead")
    assert stopped.is_error and "max_cost_usd" in stopped.content
    assert [name for _, name in tree.runs].count("searcher") == 3
    assert "writing_lead" not in [name for _, name in tree.runs]


async def test_a_document_without_the_research_is_not_ok(tmp_path):
    def skips_research(request):  # the manager sends the brief straight to the writing team
        if tool_names(request) == {"research_lead", "writing_lead"}:
            return (
                call("writing_lead", task=BRIEF) if tool_turns(request) == 0 else last_text(request)
            )
        return policy(request)

    outcome = await run(scripted_context(skips_research, tmp_path))
    # The document names every country (the brief does), but it has none of the figures.
    assert all(country in outcome.output for country in MARKET)
    assert not outcome.ok and outcome.details["missing"] == list(MARKET)
    assert "never ran: research_lead, searcher" in outcome.summary


def test_search_and_coverage():
    assert search.function("Polish heat pump sales") == MARKET["Poland"]
    assert search.function("Spain").startswith("No results.")
    assert uncovered("Norway 112,000; Poland 210 000 units; Italy 380k") == []
    assert uncovered("Norway: 112 thousand. Poland: 210000.") == ["Italy"]
    assert uncovered("Norway, Poland and Italy, 112 or 2,380 units") == list(MARKET)


def test_runs_as_a_module_and_replays_from_its_cassette(tmp_path, monkeypatch, capsys):
    common = ["--provider", "qcloud", "--cassette-dir", str(tmp_path / "cassettes"), "--json"]
    with pytest.raises(SystemExit) as recorded:
        main(
            run,
            name=hierarchical_teams.NAME,
            description=hierarchical_teams.DESCRIPTION,
            inspired_by=hierarchical_teams.INSPIRED_BY,
            argv=[*common, "--record", "--workdir", str(tmp_path / "recorded")],
            model_factory=lambda provider, model: FunctionModel(policy),
        )
    assert recorded.value.code == 0
    capsys.readouterr()

    argv = ["hierarchical_teams", *common, "--replay", "--workdir", str(tmp_path / "replayed")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as replayed:
        runpy.run_path(hierarchical_teams.__file__, run_name="__main__")
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert replayed.value.code == 0, report
    assert report["ok"] and report["name"] == "hierarchical_teams" and report["requests"] == 16
