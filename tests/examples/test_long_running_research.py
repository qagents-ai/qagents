"""Offline tests of examples/patterns/long_running_research.py."""

from __future__ import annotations

import json
import re
import runpy
import sys

import pytest

from examples.harness import last_text, main, scripted_context, system_text
from examples.patterns import long_running_research as example
from examples.patterns.long_running_research import SURVEYS, instructions, run, search
from qagents import RunContext, Usage, prompts
from qagents.models.testing import FunctionModel, call

SITES = list(SURVEYS)  # Larissa, Patras, Kavala, in the order the instructions give


@pytest.fixture(autouse=True)
def short_lease(monkeypatch):
    monkeypatch.setattr(example, "LEASE_TTL_S", 0.5)


def plan(done: int, final: bool = False) -> list[dict[str, str]]:
    items = [
        {"content": f"Research {site}", "status": "completed" if number < done else "pending"}
        for number, site in enumerate(SITES)
    ]
    items.append(
        {"content": "Write the recommendation", "status": "completed" if final else "pending"}
    )
    return items


def calls_so_far(request) -> list[str]:
    return [c.name for m in request.messages if m.role == "assistant" for c in m.tool_calls]


def policy(request):
    """Follows the instructions step by step; where it stands is read from the conversation."""
    names = calls_so_far(request)
    saved = names.count("save_note")
    if not names:
        return call("write_todos", todos=plan(0))
    if names[-1] == "search":  # the survey just arrived: note it, tick the site off
        facts = "\n".join(last_text(request).splitlines()[1:4])
        return [
            call("save_note", name=SITES[saved].lower(), content=facts),
            call("write_todos", todos=plan(saved + 1)),
        ]
    if saved < len(SITES):
        return call("search", query=SITES[saved])
    if names.count("write_todos") == 1 + len(SITES):  # the first plan and one per site
        return call("write_todos", todos=plan(len(SITES), final=True))
    return (
        "Larissa: 92 EUR/MWh, 40 MW spare, 4 ms.\nPatras: 78 EUR/MWh, 18 MW spare, 6 ms.\n"
        "Kavala: 85 EUR/MWh, 55 MW spare, 7 ms.\nRecommendation: Kavala."
    )


def priced(function=policy) -> FunctionModel:
    """Reports usage as a provider would, so the compactor sees requests of 2,500 tokens."""
    usage = Usage(requests=1, input_tokens=2_500, output_tokens=60, cost_usd=0.002)
    return FunctionModel(function, usage=usage)


async def test_worker_b_recovers_the_run_and_finishes_the_plan(tmp_path, capsys):
    ctx = scripted_context(priced(), tmp_path)
    outcome = await run(ctx)
    assert outcome.ok, outcome.summary
    details = outcome.details
    # Worker A wrote the plan, researched Larissa and died in the middle of the Patras search.
    assert details["tools_on_worker_a"] == [
        "write_todos",
        "search",
        "save_note",
        "write_todos",
        "search",
    ]
    # The search worker A died in was the only call without a result. Worker B ran that one
    # again first (it is read-only); no call that had a result ran twice.
    assert details["interrupted_on_worker_a"] == ["search"]
    assert details["run_again_on_worker_b"] == ["search"]
    assert details["worker_a_handle"] == "interrupted"
    on_b = details["tools_on_worker_b"]
    assert on_b[0] == "search" and on_b.count("search") == 2 and on_b.count("save_note") == 2
    assert details["notes"] == ["kavala", "larissa", "patras"]
    assert [item["status"] for item in details["todos"]] == ["completed"] * 4
    assert details["compactions"] >= 1 and details["recommends_expected_site"]

    printed = capsys.readouterr().out
    assert "worker-a -> search" in printed and "worker-b -> search" in printed
    assert "could not release its lease" in printed  # the dead worker's last log line
    assert re.search(r"the journal says 'running'; worker-a#[0-9a-f]{32} holds the lease", printed)
    assert "worker-b recovers session 'research:data-centre-sites'" in printed.lower()
    assert "of its 5 tool calls, search never finished. worker-b ran search again" in printed
    assert "4. [completed] Write the recommendation" in printed

    # Worker B's first request already showed the plan worker A had written (session state).
    patras = SURVEYS["Patras"]
    first_on_b = next(r for r in ctx.meter.requests if patras in last_text(r))
    assert "1. [completed] Research Larissa\n2. [pending] Research Patras" in system_text(
        first_on_b
    )

    # Larissa's survey was cleared from later requests; the plan rode along in the instructions.
    last = ctx.meter.requests[-1]
    results = [result.content for message in last.messages for result in message.tool_results]
    assert prompts.TOOL_RESULT_CLEARED.format(name="search") in results
    assert SURVEYS["Larissa"] not in results
    assert "1. [completed] Research Larissa" in system_text(last)


async def test_parallel_searches_only_the_interrupted_one_runs_again(tmp_path):
    def all_at_once(request):
        """Searches the three sites in one turn (read-only calls run in parallel)."""
        names = calls_so_far(request)
        saved = names.count("save_note")
        if not names:
            return call("write_todos", todos=plan(0))
        if "search" not in names:
            return [call("search", query=site) for site in SITES]
        if saved < len(SITES):
            return [
                call("save_note", name=SITES[saved].lower(), content="facts"),
                call("write_todos", todos=plan(saved + 1)),
            ]
        if names.count("write_todos") == 1 + len(SITES):
            return call("write_todos", todos=plan(len(SITES), final=True))
        return "Recommendation: Kavala."

    outcome = await run(scripted_context(priced(all_at_once), tmp_path))
    assert outcome.ok, outcome.summary
    details = outcome.details
    assert details["tools_on_worker_a"] == ["write_todos", "search", "search", "search"]
    # Larissa's and Kavala's searches finished on worker A and kept their results.
    assert details["interrupted_on_worker_a"] == details["run_again_on_worker_b"] == ["search"]
    assert details["tools_on_worker_b"].count("search") == 1


async def test_a_run_that_ends_before_the_crash_point_is_not_ok(tmp_path):
    def one_search(request):
        if not calls_so_far(request):
            return call("search", query="Larissa, Patras and Kavala")
        return "Kavala."

    outcome = await run(scripted_context(one_search, tmp_path))
    assert not outcome.ok
    assert "ended on worker-a (completed) before the simulated crash" in outcome.summary


def test_search_and_instructions():
    assert search.function("larissa") == SURVEYS["Larissa"]
    assert search.function("Patras and Kavala").count("Site survey") == 2
    assert search.function("Thessaloniki").startswith("No survey found.")
    ctx = RunContext(deps=None, state={})
    assert prompts.TODOS_HEADER not in instructions(ctx)
    ctx.state["todos"] = plan(1)
    assert "1. [completed] Research Larissa\n2. [pending] Research Patras" in instructions(ctx)


def test_runs_as_a_module_and_replays_from_its_cassette(tmp_path, monkeypatch, capsys):
    common = ["--provider", "anthropic", "--cassette-dir", str(tmp_path / "cassettes"), "--json"]
    with pytest.raises(SystemExit) as recorded:
        main(
            run,
            name=example.NAME,
            description=example.DESCRIPTION,
            inspired_by=example.INSPIRED_BY,
            argv=[*common, "--record", "--workdir", str(tmp_path / "recorded")],
            model_factory=lambda provider, model: priced(),
        )
    assert recorded.value.code == 0
    requests = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["requests"]

    argv = [example.__file__, *common, "--replay", "--workdir", str(tmp_path / "replayed")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as replayed:
        runpy.run_path(example.__file__, run_name="__main__")
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert replayed.value.code == 0, report
    assert report["ok"] and report["name"] == example.NAME and report["requests"] == requests
