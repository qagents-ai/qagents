"""Offline tests of examples/patterns/debate.py."""

from __future__ import annotations

import json
import re
import runpy
import sys

import pytest

from examples.harness import final_answer, last_text, main, scripted_context, system_text
from examples.patterns import debate
from examples.patterns.debate import MOTION, Verdict, run
from qagents.errors import OutputValidationError
from qagents.models.testing import FunctionModel


def speech(request) -> str:
    side = "proposition" if "You are the proposition" in system_text(request) else "opposition"
    number = re.search(r"speech for round (\d)", last_text(request)).group(1)
    return f"{side} speech {number}."


def policy(request):
    """Debaters answer with a numbered speech; the judge ties first, then decides."""
    if "debate judge" not in system_text(request):
        return speech(request)
    if request.messages[-1].role == "tool":  # the tie was sent back by the validator
        return final_answer(
            winner="opposition",
            reasons=["Costed its rebuttals.", "Answered every point."],
            proposition_score=6,
            opposition_score=8,
        )
    return final_answer(
        winner="proposition", reasons=["Close."], proposition_score=7, opposition_score=7
    )


async def test_two_rounds_then_a_checked_verdict(tmp_path, capsys):
    ctx = scripted_context(policy, tmp_path)
    outcome = await run(ctx)
    assert outcome.ok, outcome.summary
    assert outcome.summary == "opposition won 6-8 after 4 speeches"
    assert outcome.output["winner"] == "opposition" and len(outcome.output["reasons"]) == 2
    assert outcome.details["transcript"] == [
        "proposition 1: proposition speech 1.",
        "opposition 1: opposition speech 1.",
        "proposition 2: proposition speech 2.",
        "opposition 2: opposition speech 2.",
    ]

    debaters = [r for r in ctx.meter.requests if "formal debate" in system_text(r)]
    assert len(debaters) == 4 and all(not r.tools for r in debaters)
    # Each speech is a fresh run whose prompt is the motion and the transcript so far.
    assert all(len(r.messages) == 2 for r in debaters)
    last = last_text(debaters[-1])
    assert MOTION in last
    assert "[Round 1] PROPOSITION: proposition speech 1." in last
    assert "[Round 2] PROPOSITION: proposition speech 2." in last
    assert "opposition speech 2" not in last
    assert "you open the debate" in last_text(debaters[0])

    judge = [r for r in ctx.meter.requests if "debate judge" in system_text(r)]
    assert len(judge) == 2  # the tie was rejected once by the output validator
    assert "Ties are not allowed" in last_text(judge[1])
    assert "opposition speech 2." in judge[0].messages[-1].text
    assert "<- final_answer error: The final answer was not accepted" in capsys.readouterr().out


async def test_native_mode_sends_the_verdict_schema_and_retries_with_a_message(tmp_path):
    """The Anthropic path: the judge answers JSON text constrained by the schema it was sent."""

    def native(request):
        if "debate judge" not in system_text(request):
            return speech(request)
        assert request.output is not None and not request.tools
        assert set(request.output.json_schema["properties"]) == set(Verdict.model_fields)
        if "Ties are not allowed" in last_text(request):  # the validator's correction
            scores = {"proposition_score": 8, "opposition_score": 6}
            return json.dumps({"winner": "proposition", "reasons": ["Clearer costs."], **scores})
        scores = {"proposition_score": 7, "opposition_score": 7}
        return json.dumps({"winner": "opposition", "reasons": ["Close."], **scores})

    ctx = scripted_context(native, tmp_path, output_mode="native")
    outcome = await run(ctx)
    assert outcome.ok, outcome.summary
    assert outcome.summary == "proposition won 8-6 after 4 speeches"
    judge = [r for r in ctx.meter.requests if "debate judge" in system_text(r)]
    assert len(judge) == 2 and judge[1].messages[-1].role == "user"


async def test_a_judge_that_keeps_tying_fails_the_run(tmp_path):
    def stubborn(request):
        if "debate judge" in system_text(request):
            return final_answer(
                winner="opposition", reasons=["Even."], proposition_score=5, opposition_score=5
            )
        return speech(request)

    with pytest.raises(OutputValidationError, match="higher score"):
        await run(scripted_context(stubborn, tmp_path))


async def test_a_missing_speech_is_not_ok(tmp_path):
    def silent(request):
        if "You are the opposition" in system_text(request):
            return ""
        return policy(request)

    outcome = await run(scripted_context(silent, tmp_path))
    assert not outcome.ok and outcome.summary.startswith("the opposition gave no speech")


def test_runs_as_a_module_and_replays_from_its_cassette(tmp_path, monkeypatch, capsys):
    common = ["--provider", "qcloud", "--cassette-dir", str(tmp_path / "cassettes"), "--json"]
    with pytest.raises(SystemExit) as recorded:
        main(
            run,
            name=debate.NAME,
            description=debate.DESCRIPTION,
            inspired_by=debate.INSPIRED_BY,
            argv=[*common, "--record", "--workdir", str(tmp_path / "recorded")],
            model_factory=lambda provider, model: FunctionModel(policy),
        )
    assert recorded.value.code == 0
    capsys.readouterr()

    monkeypatch.setattr(
        sys, "argv", ["debate", *common, "--replay", "--workdir", str(tmp_path / "replayed")]
    )
    with pytest.raises(SystemExit) as replayed:
        runpy.run_path(debate.__file__, run_name="__main__")
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert replayed.value.code == 0, report
    assert report["ok"] and report["name"] == "debate" and report["requests"] == 6
