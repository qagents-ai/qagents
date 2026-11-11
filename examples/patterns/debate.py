"""A debate between two agents with opposing briefs, scored by a judge with a typed verdict.

The proposition and the opposition exchange two rounds on a fixed motion. Each speech is a
fresh run whose prompt is the motion plus the transcript so far, so the debaters share nothing
but what was said. A judge agent then reads the transcript and returns a `Verdict`: the
winner, the reasons and a score per side. An output validator sends the judge back when the
winner is not the side with the higher score.

Mirrors: the debate-with-a-judge demo of CrewAI and LangGraph. In CrewAI, a sequential crew
gets one task per speech, each listing the earlier tasks as ``context``, and ends with a judge
task whose ``output_pydantic`` is the verdict. In LangGraph, debater nodes append to a shared
``messages`` state, a conditional edge loops back until the last round, and a judge node calls
``with_structured_output(Verdict)``. In qagents the rounds are a plain ``for`` loop in your
code, the transcript is an explicit string in each prompt (no shared graph state or crew
memory), the second debater is ``Agent.copy`` of the first with the other brief, and the verdict
is checked twice: by its pydantic type, and by an output validator that raises ``ModelRetry``
(like a CrewAI task ``guardrail``), which sends the judge back with the reason.

Run it:

    python -m examples.patterns.debate --provider anthropic
    python -m examples.patterns.debate --provider qcloud
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from examples.harness import ExampleContext, ExampleOutcome, main
from qagents import Agent, Limits, ModelRetry, RunContext

NAME = "debate"
DESCRIPTION = "Two debaters argue for two rounds, then a judge returns a typed, checked verdict"
INSPIRED_BY = "CrewAI / LangGraph debate between two agents with a judge"

MOTION = "This house believes that cities should make public transport free to ride."
SIDES = ("proposition", "opposition")
ROUNDS = 2

Side = Literal["proposition", "opposition"]


class Verdict(BaseModel):
    """The judge's decision."""

    winner: Side = Field(description="The side that argued better: the one with the higher score.")
    reasons: list[str] = Field(
        min_length=1, max_length=4, description="Two or three short reasons for the decision."
    )
    proposition_score: int = Field(ge=1, le=10, description="The proposition's score, 1 to 10.")
    opposition_score: int = Field(ge=1, le=10, description="The opposition's score, 1 to 10.")


def winner_has_the_higher_score(ctx: RunContext[Any], verdict: Verdict) -> None:
    """An output validator: a verdict whose winner scored lower (or the same) is sent back."""
    scores = {"proposition": verdict.proposition_score, "opposition": verdict.opposition_score}
    loser = "opposition" if verdict.winner == "proposition" else "proposition"
    if scores[verdict.winner] <= scores[loser]:
        raise ModelRetry(
            f"The winner must have the higher score, but {verdict.winner} has "
            f"{scores[verdict.winner]} and {loser} has {scores[loser]}. Ties are not allowed: "
            "change the scores or the winner."
        )


BRIEF = (
    "You are the {side} in a formal debate: you argue {stance} the motion, whatever your own "
    "view. Each turn you receive the motion and the transcript so far. Write only your next "
    "speech, at most 120 words: answer the other side's latest point if there is one, then add "
    "one new argument. Do not write the other side's speech or a verdict."
)
JUDGE = (
    "You are an impartial debate judge. Judge only the arguments in the transcript, not your "
    "own view of the motion. Score each side from 1 to 10 for the strength of its reasoning "
    "and rebuttals, give two or three short reasons, and name as the winner the side with the "
    "higher score. Ties are not allowed."
)


def render(transcript: list[tuple[int, str, str]]) -> str:
    if not transcript:
        return "(nothing yet: you open the debate)"
    return "\n\n".join(f"[Round {r}] {side.upper()}: {speech}" for r, side, speech in transcript)


async def run(ctx: ExampleContext) -> ExampleOutcome:
    proposition = Agent(
        ctx.model,
        name="proposition",
        instructions=BRIEF.format(side="proposition", stance="FOR"),
        limits=Limits(max_turns=2, max_cost_usd=0.05),
        hooks=ctx.hooks,
    )
    # The same agent with the opposite brief; everything else (model, limits, hooks) is shared.
    opposition = proposition.copy(
        name="opposition", instructions=BRIEF.format(side="opposition", stance="AGAINST")
    )
    judge = Agent(
        ctx.model,
        name="judge",
        instructions=JUDGE,
        output_type=Verdict,
        output_mode=ctx.output_mode,
        output_validators=[winner_has_the_higher_score],
        limits=Limits(max_turns=4, max_cost_usd=0.10),
        hooks=ctx.hooks,
    )
    debaters = {"proposition": proposition, "opposition": opposition}

    transcript: list[tuple[int, str, str]] = []
    for number in range(1, ROUNDS + 1):
        for side in SIDES:
            ctx.log(f"Round {number}: {side}")
            prompt = (
                f"Motion: {MOTION}\n\nTranscript so far:\n{render(transcript)}\n\n"
                f"Write the {side}'s speech for round {number}."
            )
            result = await ctx.print_stream(debaters[side].stream(prompt))
            if result.status != "completed" or not result.text.strip():
                return ExampleOutcome(
                    ok=False,
                    summary=f"the {side} gave no speech in round {number} ({result.status})",
                    output=transcript,
                )
            transcript.append((number, side, result.text.strip()))

    ctx.log("The judge reads the transcript")
    judged = await ctx.print_stream(
        judge.stream(f"Motion: {MOTION}\n\nTranscript:\n{render(transcript)}")
    )
    verdict = judged.output
    # The type and the validator already checked the winner and the scores.
    ok = (
        judged.status == "completed"
        and isinstance(verdict, Verdict)
        and all(reason.strip() for reason in verdict.reasons)
    )
    if ok:
        summary = (
            f"{verdict.winner} won {verdict.proposition_score}-{verdict.opposition_score} "
            f"after {len(transcript)} speeches"
        )
        ctx.log(f"Verdict: {summary}. " + " ".join(verdict.reasons))
    else:
        summary = f"no valid verdict: the judge's run ended {judged.status} ({judged.stop_reason})"
    return ExampleOutcome(
        ok=ok,
        summary=summary,
        output=verdict.model_dump() if isinstance(verdict, Verdict) else None,
        details={"transcript": [f"{side} {number}: {text}" for number, side, text in transcript]},
    )


if __name__ == "__main__":
    main(run, name=NAME, description=DESCRIPTION, inspired_by=INSPIRED_BY)
