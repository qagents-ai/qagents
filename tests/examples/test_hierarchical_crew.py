"""Offline tests of examples/patterns/hierarchical_crew.py: delegation, parallel specialists."""

from __future__ import annotations

import asyncio
import json
import runpy
import sys

import pytest

from examples.harness import final_answer, last_text, main, scripted_context, tool_names
from examples.patterns import hierarchical_crew
from examples.patterns.hierarchical_crew import LaunchReport, run
from qagents.models.testing import FunctionModel, call

QUESTIONS = {
    "ask_market": ["How many Lisbon families might pay EUR 89 a month?"],
    "ask_legal": [
        "Does a 250 W cargo bike need registration in Portugal?",
        "What cancellation rights must a monthly subscription give?",
    ],
    "ask_technical": ["Can the bike handle Lisbon's hills with a load?"],
}
SPECIALIST_CALLS = {
    "market_data": call("market_data", city="Lisbon"),
    "regulation": call("regulation", topic="e-bikes"),
    "field_test": call("field_test"),
}
REPORT = {
    "recommendation": "launch with conditions",
    "findings": [
        {"area": "market", "summary": "3.1% would pay EUR 89.", "risks": ["small market"]},
        {"area": "legal", "summary": "250 W bikes need no registration.", "risks": []},
        {"area": "technical", "summary": "12% climbs loaded; 18% of streets are steeper.",
         "risks": ["hills"]},
    ],
    "next_steps": ["Pilot in flat districts"],
}  # fmt: skip


def answer(request, **fields):
    """A structured answer in either output mode: a final_answer call, or JSON text (native)."""
    if "final_answer" in tool_names(request):
        return final_answer(**fields)
    return json.dumps(fields)


def is_manager(request) -> bool:
    return "ask_market" in tool_names(request)


def policy(request, report=REPORT):
    """The manager is recognised by its ask_* tools, each specialist by its data tool."""
    if is_manager(request):
        if request.messages[-1].role == "user":
            return [call(name, tasks=tasks) for name, tasks in QUESTIONS.items()]
        return answer(request, **report)
    (data,) = tool_names(request) & set(SPECIALIST_CALLS)
    if request.messages[-1].role == "user":
        return SPECIALIST_CALLS[data]
    return f"Based on {data}: {last_text(request)[:60]}"


async def test_the_manager_delegates_to_every_specialist_and_reports(tmp_path, capsys):
    ctx = scripted_context(policy, tmp_path)
    outcome = await run(ctx)
    output = capsys.readouterr().out
    assert outcome.ok, outcome.summary
    assert LaunchReport.model_validate(outcome.output).recommendation == "launch with conditions"
    assert "delegated to: legal, market, technical" in outcome.summary

    # One child run per question, each seeing only its own question.
    tasks = [r.messages[1].text for r in ctx.meter.requests if not is_manager(r)]
    asked = [question for questions in QUESTIONS.values() for question in questions]
    assert sorted(set(tasks)) == sorted(asked) and len(tasks) == 2 * len(asked)
    assert '[legal_specialist[1]] -> regulation(topic="e-bikes")' in output
    assert "[technical_specialist[0]] -> field_test()" in output

    # The manager read one answer per question, as the JSON list fan_out returns.
    answers = ctx.meter.requests[-1].messages[-1].tool_results
    legal = json.loads(next(result.content for result in answers if result.name == "ask_legal"))
    assert [entry["task"] for entry in legal] == QUESTIONS["ask_legal"]
    assert all(entry["answer"].startswith("Based on regulation") for entry in legal)

    # Linked child runs: the specialists' requests are counted in the manager's run.
    assert outcome.details["usage"]["requests"] == len(ctx.meter.requests) == 10


async def test_the_specialists_work_in_parallel(tmp_path):
    in_flight, peak = 0, 0

    async def slow_specialist(request):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return policy(request)

    def dispatch(request):
        return policy(request) if is_manager(request) else slow_specialist(request)

    assert (await run(scripted_context(dispatch, tmp_path))).ok
    assert peak == 4  # three ask_* calls of one turn, with two legal questions, all at once


async def test_the_crew_runs_in_native_mode_as_on_anthropic(tmp_path):
    ctx = scripted_context(policy, tmp_path, output_mode="native")
    outcome = await run(ctx)
    assert outcome.ok, outcome.summary
    managers = [request for request in ctx.meter.requests if is_manager(request)]
    assert all(request.output is not None for request in managers)
    assert all("final_answer" not in tool_names(request) for request in managers)


async def test_a_report_without_an_area_fails(tmp_path):
    partial = {**REPORT, "findings": REPORT["findings"][:2]}
    outcome = await run(scripted_context(lambda request: policy(request, partial), tmp_path))
    assert not outcome.ok and "areas reported: legal, market;" in outcome.summary


async def test_a_manager_that_does_not_delegate_fails(tmp_path):
    def lazy(request):
        if is_manager(request):
            return answer(request, **REPORT)
        raise AssertionError("no specialist should run")

    outcome = await run(scripted_context(lazy, tmp_path))
    assert not outcome.ok and "delegated to: nobody" in outcome.summary


async def test_a_delegation_that_never_ran_does_not_count(tmp_path):
    def hasty(request):  # asks the market team and answers in the same turn
        if is_manager(request):
            return [call("ask_market", tasks=QUESTIONS["ask_market"]), final_answer(**REPORT)]
        raise AssertionError("the ask_market call is skipped, so no specialist runs")

    ctx = scripted_context(hasty, tmp_path)
    outcome = await run(ctx)
    assert not outcome.ok and "delegated to: nobody" in outcome.summary
    assert len(ctx.meter.requests) == 1


async def test_a_specialist_that_stops_early_does_not_count(tmp_path):
    def stuck(request):  # the fleet engineer calls field_test until its max_turns
        if "field_test" in tool_names(request):
            return call("field_test")
        return policy(request)

    ctx = scripted_context(stuck, tmp_path)
    outcome = await run(ctx)
    assert not outcome.ok and "delegated to: legal, market" in outcome.summary
    answers = ctx.meter.requests[-1].messages[-1].tool_results
    technical = json.loads(next(r.content for r in answers if r.name == "ask_technical"))
    assert "max_turns" in technical[0]["error"]


async def test_a_malformed_report_is_sent_back_to_the_manager(tmp_path):
    sent_back = []

    def sloppy(request):
        if is_manager(request) and request.messages[-1].role == "tool":
            results = request.messages[-1].tool_results
            if not any(result.name == "final_answer" for result in results):
                return final_answer(**{**REPORT, "recommendation": "maybe"})
            sent_back.append(results[0].content)
        return policy(request)

    outcome = await run(scripted_context(sloppy, tmp_path))
    assert outcome.ok, outcome.summary
    assert sent_back and "recommendation" in sent_back[0]


def test_runs_as_a_module_and_replays_from_its_cassette(tmp_path, capsys, monkeypatch):
    cassettes = tmp_path / "cassettes"
    common = ["--provider", "qcloud", "--cassette-dir", str(cassettes)]
    with pytest.raises(SystemExit) as recorded:
        main(
            run,
            name=hierarchical_crew.NAME,
            description=hierarchical_crew.DESCRIPTION,
            inspired_by=hierarchical_crew.INSPIRED_BY,
            argv=[*common, "--record", "--workdir", str(tmp_path / "recorded")],
            model_factory=lambda provider, model: FunctionModel(policy),
        )
    assert recorded.value.code == 0
    capsys.readouterr()

    argv = ["hierarchical_crew", *common, "--replay", "--workdir", str(tmp_path / "replayed")]
    monkeypatch.setattr(sys, "argv", [*argv, "--json"])
    with pytest.raises(SystemExit) as replayed:
        runpy.run_path(hierarchical_crew.__file__, run_name="__main__")
    output = capsys.readouterr().out
    report = json.loads(output.strip().splitlines()[-1])
    assert replayed.value.code == 0, output
    assert report["ok"] and report["name"] == "hierarchical_crew" and report["requests"] == 10
    assert "Recommendation: launch with conditions" in output
