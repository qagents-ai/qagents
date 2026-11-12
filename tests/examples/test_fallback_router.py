"""Offline tests of examples/patterns/fallback_router.py."""

from __future__ import annotations

import asyncio
import json
import runpy
import sys

import pytest

from examples.harness import final_answer, last_text, main, scripted_context, system_text
from examples.patterns import fallback_router
from examples.patterns.fallback_router import QUESTIONS, AlwaysBusy, answered, run
from qagents import Agent, FallbackModel, Limits
from qagents.errors import ModelRateLimitError
from qagents.models.testing import FunctionModel, ScriptedModel

ANSWERS = {
    QUESTIONS[0][0]: "Lisbon.",
    QUESTIONS[2][0]: "A leap year has 366 days.",
}


def label(question: str) -> str:
    return "complex" if "bakery" in question else "simple"


def policy(request):
    """The classifier labels, the quick agent answers in a line, the expert in steps."""
    system, question = system_text(request), last_text(request)
    if "Classify the user's question" in system:
        return final_answer(value=label(question))
    if "step by step" in system:
        return "3 x 48 = 144 rolls, 144 - 12 = 132, 132 / 7 = 18 rest 6.\nAnswer: 18 bags, 6 left."
    assert "one short sentence" in system, system
    return ANSWERS[question]


async def test_routes_every_question_through_the_fallback(tmp_path, capsys):
    ctx = scripted_context(policy, tmp_path)
    outcome = await run(ctx)
    assert outcome.ok, outcome.summary
    assert outcome.details["routes"] == ["simple", "complex", "simple"]
    # Every request was refused by the busy primary first, then answered by ctx.model.
    assert outcome.details["primary_refusals"] == len(ctx.meter.requests) == 6
    assert outcome.output[1].endswith("Answer: 18 bags, 6 left.")

    experts = [r for r in ctx.meter.requests if "step by step" in system_text(r)]
    assert [last_text(r) for r in experts] == [QUESTIONS[1][0]]
    classifier = [r for r in ctx.meter.requests if "Classify" in system_text(r)]
    assert len(classifier) == 3 and all(
        {t.name for t in r.tools} == {"final_answer"} and r.output is None for r in classifier
    )
    printed = capsys.readouterr().out
    assert "-> complex -> expert" in printed and "refused 6 requests" in printed


async def test_native_mode_sends_the_route_names_as_the_output_schema(tmp_path):
    """The Anthropic path: no final-answer tool, the Literal goes out as the response schema."""

    def native(request):
        if "Classify the user's question" in system_text(request):
            assert not request.tools
            # Native mode wraps a schema whose root is not an object as {"value": ...}.
            schema = request.output.json_schema
            assert schema["properties"]["value"]["enum"] == ["simple", "complex"]
            assert schema["required"] == ["value"]
            return json.dumps(label(last_text(request)))
        return policy(request)

    outcome = await run(scripted_context(native, tmp_path, output_mode="native"))
    assert outcome.ok, outcome.summary
    assert outcome.details["routes"] == ["simple", "complex", "simple"]


async def test_the_rate_limiter_keeps_two_requests_in_flight(tmp_path):
    in_flight = peak = 0

    async def slow(request):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1
        return policy(request)

    outcome = await run(scripted_context(slow, tmp_path))
    assert outcome.ok, outcome.summary
    assert peak == 2  # three questions ran at once, but max_concurrent=2 held them to two


async def test_a_wrong_answer_is_not_ok(tmp_path):
    def wrong(request):
        if "step by step" in system_text(request):
            return "132 / 7 = 18 rest 6, so 19 bags.\nAnswer: 19 bags, 0 left."
        return policy(request)

    outcome = await run(scripted_context(wrong, tmp_path))
    assert not outcome.ok and outcome.details["correct"] == [True, False, True]


async def test_a_wrong_route_is_not_ok(tmp_path):
    def everything_complex(request):
        if "Classify the user's question" in system_text(request):
            return final_answer(value="complex")
        if "step by step" in system_text(request):
            question = last_text(request)
            return ANSWERS.get(question) or policy(request)
        return policy(request)

    outcome = await run(scripted_context(everything_complex, tmp_path))
    assert not outcome.ok and outcome.details["correct"] == [True, True, True]
    assert outcome.details["routes"] == ["complex"] * 3 and "expected" in outcome.summary


async def test_the_busy_model_fails_before_streaming_so_the_fallback_answers():
    busy = AlwaysBusy()
    with pytest.raises(ModelRateLimitError, match="429"):  # alone, the loop gives up at once
        await Agent(busy, limits=Limits(model_retries=0)).run("Hi")
    fallback = ScriptedModel(["Hello."])
    result = await Agent(FallbackModel([busy, fallback])).run("Hi")
    assert result.text == "Hello." and busy.refused == 2 and len(fallback.requests) == 1


def test_answer_check():
    assert answered("It fills 18 bags and 6 rolls are left.", ("18", "6"))
    assert not answered("It fills 18 bags and 16 rolls are left.", ("18", "6"))
    assert answered("lisbon", ("Lisbon",))
    # Only the text after the last "Answer:" counts, not the numbers in the working.
    assert answered("132 / 7 = 18 rest 6\n**Answer:** 18 bags, 6 rolls left", ("18", "6"))
    assert not answered("132 / 7 = 18 rest 6\nAnswer: 19 bags, 0 left", ("18", "6"))


def test_runs_as_a_module_and_replays_from_its_cassette(tmp_path, monkeypatch, capsys):
    common = ["--provider", "qcloud", "--cassette-dir", str(tmp_path / "cassettes"), "--json"]
    with pytest.raises(SystemExit) as recorded:
        main(
            run,
            name=fallback_router.NAME,
            description=fallback_router.DESCRIPTION,
            inspired_by=fallback_router.INSPIRED_BY,
            argv=[*common, "--record", "--workdir", str(tmp_path / "recorded")],
            model_factory=lambda provider, model: FunctionModel(policy),
        )
    assert recorded.value.code == 0
    capsys.readouterr()

    argv = ["fallback_router", *common, "--replay", "--workdir", str(tmp_path / "replayed")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as replayed:
        runpy.run_path(fallback_router.__file__, run_name="__main__")
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert replayed.value.code == 0, report
    assert report["ok"] and report["name"] == "fallback_router" and report["requests"] == 6
