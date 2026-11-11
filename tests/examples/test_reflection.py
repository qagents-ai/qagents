"""Offline tests of examples/patterns/reflection.py."""

from __future__ import annotations

import json

import pytest

from examples.harness import (
    final_answer,
    last_text,
    main,
    scripted_context,
    system_text,
    tool_names,
)
from examples.patterns.reflection import (
    DESCRIPTION,
    INSPIRED_BY,
    MAX_ROUNDS,
    NAME,
    check_rules,
    run,
)
from qagents.models.testing import FunctionModel

TOO_LONG = (
    "Meet the amazing Tidewell TW-2, the best bottle you will ever own! It keeps your drinks icy "
    "cold for a full 24 hours and piping hot for 12 hours, wherever the trail takes you. Built "
    "from 90% recycled stainless steel with a double wall, it holds 750 ml, weighs just 340 g "
    "and comes with a lifetime warranty."
)
GOOD = (
    "The Tidewell TW-2 keeps drinks cold for 24 hours and hot for 12. Its double-wall body is 90% "
    "recycled stainless steel, and the leak-proof, dishwasher-safe lid makes cleaning easy. It "
    "holds 750 ml and weighs 340 g."
)


def structured(request, **fields):
    """A structured answer in either output mode: a final_answer call, or JSON text (native)."""
    if "final_answer" in tool_names(request):
        return final_answer(**fields)
    return json.dumps(fields)


def make_policy(*, revised=GOOD, critic_passes=True):
    """The writer's first draft breaks the rules; its revision is ``revised``."""

    def policy(request):
        system = system_text(request)
        if "You are a copywriter" in system:
            return revised if last_text(request).startswith("Revise") else TOO_LONG
        if "You are the editor" in system:
            draft = last_text(request).split("Description:\n", 1)[1]
            if draft == GOOD and critic_passes:
                return structured(request, score=9, issues=[], passes=True)
            return structured(
                request,
                score=5,
                issues=["Remove the lifetime warranty: it is not in the facts."],
                passes=False,
            )
        raise AssertionError(f"Unexpected request: {system[:80]!r}")

    return policy


def writer_requests(ctx):
    return [r for r in ctx.meter.requests if "You are a copywriter" in system_text(r)]


def critic_requests(ctx):
    return [r for r in ctx.meter.requests if "You are the editor" in system_text(r)]


@pytest.mark.parametrize("output_mode", ["tool", "native"])
async def test_the_writer_revises_with_the_rules_and_the_critique(tmp_path, output_mode):
    ctx = scripted_context(make_policy(), tmp_path, output_mode=output_mode)
    outcome = await run(ctx)

    assert outcome.ok, outcome.summary
    assert outcome.output == GOOD
    assert "after 2 round(s), accepted by the critic" in outcome.summary
    first, second = outcome.details["rounds"]
    assert first["critique"]["passes"] is False and len(first["rule_problems"]) == 4
    assert second["rule_problems"] == [] and second["critique"]["score"] == 9

    # The revision request continues the writer's session: brief, first draft, then feedback.
    revision = writer_requests(ctx)[1]
    assert [m.role for m in revision.messages] == ["system", "user", "assistant", "user"]
    assert revision.messages[2].text == TOO_LONG
    feedback = last_text(revision)
    assert "It has 58 words; the limit is 50." in feedback
    assert 'exact phrase "dishwasher-safe lid"' in feedback
    assert "exclamation marks" in feedback
    assert "lifetime warranty" in feedback  # the critic's issue

    # The critic starts afresh every round: the brief and the current draft, nothing else.
    reviews = critic_requests(ctx)
    assert len(reviews) == 2
    assert all([m.role for m in review.messages] == ["system", "user"] for review in reviews)
    assert last_text(reviews[1]).endswith(f"Description:\n{GOOD}")


async def test_the_loop_stops_after_three_rounds_with_a_draft_that_meets_the_rules(tmp_path):
    ctx = scripted_context(make_policy(critic_passes=False), tmp_path)
    outcome = await run(ctx)

    assert outcome.ok, outcome.summary  # the hard rules decide, not the critic
    assert len(outcome.details["rounds"]) == MAX_ROUNDS == len(writer_requests(ctx))
    assert "the critic did not pass it" in outcome.summary


async def test_a_draft_that_breaks_the_rules_fails_even_if_the_critic_passes_it(tmp_path):
    ctx = scripted_context(make_policy(revised=TOO_LONG), tmp_path)
    outcome = await run(ctx)

    assert not outcome.ok
    assert len(writer_requests(ctx)) == MAX_ROUNDS
    assert "breaks: It has 58 words" in outcome.summary


async def test_a_failing_verdict_with_nothing_to_fix_ends_the_loop(tmp_path):
    def policy(request):
        if "You are a copywriter" in system_text(request):
            return GOOD
        return structured(request, score=7, issues=[], passes=False)

    ctx = scripted_context(policy, tmp_path)
    outcome = await run(ctx)

    assert outcome.ok, outcome.summary
    assert len(writer_requests(ctx)) == 1  # no revision without something to revise
    assert "after 1 round(s), the critic did not pass it" in outcome.summary


def test_check_rules():
    assert check_rules(GOOD) == []
    assert check_rules(GOOD.replace("dishwasher-safe", "dishwasher‑safe")) == []
    assert check_rules(GOOD.upper()) == []  # phrases match regardless of case
    assert check_rules(GOOD + "!") == ["It must not contain exclamation marks."]
    assert check_rules(" ".join(["word"] * 51) + " " + GOOD)[0] == (
        "It has 88 words; the limit is 50."
    )


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
    assert output.count("ok: 37 words after 2 round(s), accepted by the critic") == 2
