"""Offline tests of examples/patterns/plan_and_execute.py."""

from __future__ import annotations

import json
import re

import pytest

from examples.harness import (
    final_answer,
    last_text,
    main,
    scripted_context,
    system_text,
    tool_names,
)
from examples.patterns.plan_and_execute import (
    DESCRIPTION,
    EXPECTED_MW,
    INSPIRED_BY,
    NAME,
    calculate,
    lookup,
    run,
)
from qagents import ModelRetry, RunContext, ToolError
from qagents.models.testing import FunctionModel, call


def structured(request, **fields):
    """A structured answer in either output mode: a final_answer call, or JSON text (native)."""
    if "final_answer" in tool_names(request):
        return final_answer(**fields)
    return json.dumps(fields)


def executor(request):
    """Does its step with the tools, like a model would, then reports what the tools said."""
    last = request.messages[-1]
    if last.role == "tool":
        results = [result.content for result in last.tool_results]
        if last.tool_results[0].name == "calculate":
            return f"The total is {results[0].rsplit('= ', 1)[1]} MW."
        return " ".join(results)
    step = last_text(request).rsplit("Your step:", 1)[1]
    if "Northwind" in step:
        return call("lookup", topic="Northwind Energy")
    if "each wind farm" in step:
        return [
            call("lookup", topic=name) for name in ("Orla Ridge", "Brannock Bay", "Kessick Moor")
        ]
    pairs = re.findall(r"(\d+) turbines rated ([\d.]+) MW", last_text(request))
    return call("calculate", expression=" + ".join(f"{count} * {mw}" for count, mw in pairs))


def make_policy(*, replanner_finishes: bool = True):
    def policy(request):
        system = system_text(request)
        if "You plan research" in system:
            # No calculation step: the replanner has to add one.
            return structured(
                request,
                steps=[
                    "Find the wind farms Northwind Energy operates",
                    "Look up the turbines of each wind farm",
                ],
            )
        if "You carry out one step" in system:
            return executor(request)
        if "You review" in system:
            total = re.search(r"The total is ([\d.]+) MW", last_text(request))
            if total and replanner_finishes:
                return structured(
                    request,
                    done=True,
                    answer=float(total.group(1)),
                    explanation="Turbines times rating, summed over the three wind farms.",
                )
            return structured(request, done=False, new_steps=["Calculate the total rated capacity"])
        raise AssertionError(f"Unexpected request: {system[:80]!r}")

    return policy


def requests_of(ctx, marker):
    return [request for request in ctx.meter.requests if marker in system_text(request)]


@pytest.mark.parametrize("output_mode", ["tool", "native"])
async def test_the_replanner_adds_the_missing_step_and_answers(tmp_path, output_mode):
    ctx = scripted_context(make_policy(), tmp_path, output_mode=output_mode)
    outcome = await run(ctx)

    assert outcome.ok, outcome.summary
    assert outcome.output == pytest.approx(143.2)
    assert outcome.details["replans"] == 1
    assert [item["step"] for item in outcome.details["steps"]][-1] == (
        "Calculate the total rated capacity"
    )
    assert outcome.details["workflow_steps"] == [
        "plan",
        "step-1",
        "step-2",
        "review-1",
        "step-3",
        "review-2",
    ]
    # The replanner saw both results, and the step it added was told what they were.
    first_review = requests_of(ctx, "You review")[0]
    assert "2. Look up the turbines of each wind farm" in last_text(first_review)
    added_step = next(
        request
        for request in requests_of(ctx, "You carry out one step")
        if "Your step: Calculate" in last_text(request)
    )
    assert "Orla Ridge, Brannock Bay" in last_text(added_step)  # the result of step 1
    assert "9 turbines rated 4.2 MW" in last_text(added_step)  # the result of step 2
    # The three lookups of step 2 were one turn, and the calculator did the arithmetic.
    tool_turns = [
        request.messages[-1].tool_results
        for request in requests_of(ctx, "You carry out one step")
        if request.messages[-1].role == "tool"
    ]
    assert [[result.name for result in turn] for turn in tool_turns] == [
        ["lookup"],
        ["lookup"] * 3,
        ["calculate"],
    ]
    calculation = requests_of(ctx, "You carry out one step")[-1]
    assert "calculate" in tool_names(calculation)
    assert last_text(calculation).endswith("= 143.2")


async def test_replanning_stops_after_two_replans(tmp_path):
    ctx = scripted_context(make_policy(replanner_finishes=False), tmp_path)
    outcome = await run(ctx)

    assert not outcome.ok
    assert outcome.output is None and "No answer after 4 steps and 2 replans" in outcome.summary
    reviews = requests_of(ctx, "You review")
    assert len(reviews) == 3
    assert "No more steps can be added" in last_text(reviews[-1])
    assert all("No more steps" not in last_text(review) for review in reviews[:-1])


async def test_a_step_that_stops_at_its_limit_is_reported_and_not_recorded(tmp_path):
    policy = make_policy()

    def stuck_on_step_1(request):
        step = next(message.text for message in request.messages if message.role == "user")
        if "You carry out one step" in system_text(request) and "Your step: Find" in step:
            return call("lookup", topic="Northwind Energy")  # again and again, never answers
        return policy(request)

    ctx = scripted_context(stuck_on_step_1, tmp_path)
    outcome = await run(ctx)

    assert outcome.ok, outcome.summary  # step 2 found the wind farms anyway
    assert outcome.details["steps"][0]["result"] == "(stopped: max_turns)"
    assert "(stopped: max_turns)" in last_text(requests_of(ctx, "You review")[0])
    assert "step-1" not in outcome.details["workflow_steps"]  # a rerun would try it again


async def test_a_second_run_replays_the_workflow_without_the_model(tmp_path, capsys):
    first = scripted_context(make_policy(), tmp_path)
    expected = await run(first)
    assert expected.ok and first.meter.usage.requests > 0
    assert "Replaying" not in capsys.readouterr().out

    def no_model(request):
        raise AssertionError("every step should come from the workflow journal")

    second = scripted_context(no_model, tmp_path)  # the same work directory
    outcome = await run(second)
    assert outcome.ok, outcome.summary
    assert second.meter.usage.requests == 0
    assert outcome.details == expected.details  # the same plan, results and review
    assert "Replaying 6 finished step(s) from the workflow journal" in capsys.readouterr().out


async def test_the_tools():
    ctx = RunContext()
    assert EXPECTED_MW == pytest.approx(143.2)
    assert await calculate.run(ctx, {"expression": "14 * 3.6 + (22 * 2.5)"}) == (
        "14 * 3.6 + (22 * 2.5) = 105.4"
    )
    for bad in ("14 MW * 3", "__import__('os')", "2 ** 10", "1 / 0"):
        with pytest.raises(ModelRetry, match="Use only numbers"):
            await calculate.run(ctx, {"expression": bad})
    assert "14 turbines rated 3.6 MW" in await lookup.run(ctx, {"topic": "Orla Ridge wind farm"})
    with pytest.raises(ToolError, match="The knowledge base has: Northwind Energy"):
        await lookup.run(ctx, {"topic": "Atlantis"})


@pytest.mark.parametrize("provider", ["anthropic", "qcloud"])
def test_runs_as_a_module_and_replays_from_its_cassette(provider, tmp_path, capsys):
    """The command line, recorded with the scripted model and replayed in another work
    directory: nothing random or machine-specific reaches the model."""

    def record(provider, model):
        return FunctionModel(make_policy())

    def refuse(provider, model):
        raise AssertionError("a replay must not build the provider's model")

    def run_main(factory, *argv):
        with pytest.raises(SystemExit) as exited:
            main(
                run,
                name=NAME,
                description=DESCRIPTION,
                inspired_by=INSPIRED_BY,
                argv=["--provider", provider, "--cassette-dir", str(tmp_path / "cassettes"), *argv],
                model_factory=factory,
            )
        return exited.value.code

    assert run_main(record, "--record", "--workdir", str(tmp_path / "w1")) == 0
    assert run_main(refuse, "--replay", "--workdir", str(tmp_path / "w2")) == 0
    output = capsys.readouterr().out
    assert output.count("ok: 143.2 MW after 3 steps and 1 replan(s)") == 2
