"""Offline tests of examples/patterns/react_assistant.py."""

from __future__ import annotations

import json
import runpy
import sys

import pytest

from examples.harness import main, scripted_context
from examples.patterns import react_assistant
from examples.patterns.react_assistant import (
    QUESTIONS,
    calculate,
    convert_units,
    get_weather,
    run,
)
from qagents import ToolError
from qagents.models.testing import FunctionModel, call


def results_since_question(request) -> dict[str, str]:
    """Tool results (by tool name) the model received since the last user message."""
    found: dict[str, str] = {}
    for message in reversed(request.messages):
        if message.role == "user":
            break
        for result in message.tool_results:
            found.setdefault(result.name, result.content)
    return found


def policy(request):
    """Plays the model: a ReAct loop per question, reading earlier turns from the history."""
    question = next(m.text for m in reversed(request.messages) if m.role == "user")
    done = results_since_question(request)
    if question == QUESTIONS[0]:
        if "get_weather" not in done:
            return call("get_weather", city="Lisbon")
        if "convert_units" not in done:
            celsius = json.loads(done["get_weather"])["temperature_c"]
            return call("convert_units", value=celsius, from_unit="celsius", to_unit="fahrenheit")
        return f"It is sunny in Lisbon: {done['convert_units']}."

    # Turn 2. "Where I am" and Lisbon's temperature come only from the first turn.
    earlier = [r for m in request.messages for r in m.tool_results if r.name == "get_weather"]
    lisbon = next((json.loads(r.content) for r in earlier if "Lisbon" in r.content), None)
    if lisbon is None:
        return "Where are you?"
    if "get_weather" not in done:
        return call("get_weather", city="Oslo")
    if "calculate" not in done:
        oslo = json.loads(done["get_weather"])["temperature_c"]
        return call("calculate", expression=f"{lisbon['temperature_c']} - {oslo}")
    difference = done["calculate"].split("=")[-1].strip()
    return f"Lisbon is {difference} degrees Celsius warmer than Oslo."


async def test_the_second_turn_uses_the_first_turn(tmp_path, capsys):
    ctx = scripted_context(policy, tmp_path)
    outcome = await run(ctx)
    assert outcome.ok, outcome.summary
    assert outcome.output == [
        "It is sunny in Lisbon: 23 celsius = 73.4 fahrenheit.",
        "Lisbon is 16 degrees Celsius warmer than Oslo.",
    ]
    # The first request of turn 2 carried turn 1, read back from chat.db.
    turn_two = next(r for r in ctx.meter.requests if r.messages[-1].text == QUESTIONS[1])
    assert any(m.role == "user" and m.text == QUESTIONS[0] for m in turn_two.messages)
    assert any(r.name == "convert_units" for m in turn_two.messages for r in m.tool_results)
    # Per turn: the question, two tool calls with their results, and the answer.
    assert outcome.details["messages_in_session"] == 12
    assert outcome.details["tool_calls"] == [
        "get_weather",
        "convert_units",
        "get_weather",
        "calculate",
    ]
    printed = capsys.readouterr().out
    assert '-> get_weather(city="Oslo")' in printed
    assert '-> calculate(expression="23 - 7")' in printed


async def test_without_the_history_the_outcome_is_not_ok(tmp_path):
    def forgetful(request):  # a model that is sent only the current question, no history
        start = max(i for i, m in enumerate(request.messages) if m.role == "user")
        system = [m for m in request.messages if m.role == "system"]
        return policy(request.model_copy(update={"messages": system + request.messages[start:]}))

    outcome = await run(scripted_context(forgetful, tmp_path))
    assert not outcome.ok and "Where are you?" in outcome.summary


def test_the_tools():
    assert calculate.function("(23 - 7) * 1.8") == "(23 - 7) * 1.8 = 28.8"
    assert calculate.function("-(7 - 23) // 3 % 4") == "-(7 - 23) // 3 % 4 = 1"
    assert calculate.function("1 / 3") == "1 / 3 = 0.333333"
    for refused in ("__import__('os').getcwd()", "x + 1", "2 ** 10", "1 / 0", "2 +", "True + 1"):
        with pytest.raises(ToolError):
            calculate.function(refused)
    assert convert_units.function(23.0, "celsius", "fahrenheit") == "23 celsius = 73.4 fahrenheit"
    with pytest.raises(ToolError, match="Cannot convert"):
        convert_units.function(1.0, "km", "celsius")
    assert get_weather.function("Lisbon, Portugal")["temperature_c"] == 23
    with pytest.raises(ToolError, match="Known cities: Lisbon, Oslo"):
        get_weather.function("Paris")
    # Literal arguments reach the model as an enum.
    assert "mph" in convert_units.parameters["properties"]["to_unit"]["enum"]


def test_runs_as_a_module_and_replays_from_its_cassette(tmp_path, monkeypatch, capsys):
    common = ["--provider", "qcloud", "--cassette-dir", str(tmp_path / "cassettes"), "--json"]
    with pytest.raises(SystemExit) as recorded:
        main(
            run,
            name=react_assistant.NAME,
            description=react_assistant.DESCRIPTION,
            inspired_by=react_assistant.INSPIRED_BY,
            argv=[*common, "--record", "--workdir", str(tmp_path / "recorded")],
            model_factory=lambda provider, model: FunctionModel(policy),
        )
    assert recorded.value.code == 0
    capsys.readouterr()

    argv = ["react_assistant", *common, "--replay", "--workdir", str(tmp_path / "replayed")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as replayed:
        runpy.run_path(react_assistant.__file__, run_name="__main__")
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert replayed.value.code == 0, report
    assert report["ok"] and report["name"] == "react_assistant" and report["requests"] == 6
