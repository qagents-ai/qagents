"""Reflection: a writer drafts, a critic reviews, and the writer revises until the draft passes.

A writer agent drafts a short product description from a brief. A critic agent answers with a
typed `Critique` (a score, concrete issues, and whether it passes), and plain Python checks the
hard rules: the word limit, the required phrases, no exclamation marks. The writer revises in
the same session with both lists of issues, for at most three rounds.

Mirrors: LangGraph's reflection tutorial and its evaluator-optimizer workflow: a generate node
and a reflect node in a loop, the critique added to the generator's messages as a human
message, and a conditional edge that ends the loop when the evaluator accepts the draft or
after a fixed number of rounds. In qagents:

- the generate node is ``writer.stream(prompt, session=session)``. One `Session` holds the
  brief, every draft and every critique, so a revision request is simply the next user message;
- the reflect node is a fresh ``critic.run`` with ``output_type=Critique``, a validated
  pydantic model instead of free text. It sees only the brief and the current draft, so it
  judges each draft on its own;
- the conditional edge is a ``break`` inside a Python ``for`` over at most ``MAX_ROUNDS``
  rounds: the loop ends when the rules are met and the critic passes the draft (or names
  nothing left to fix);
- what code can check (the word limit, the exact phrases, no exclamation marks) is checked by
  `check_rules`, which never miscounts, so the critic judges only accuracy and clarity.

Run it:

    python -m examples.patterns.reflection --provider anthropic
    python -m examples.patterns.reflection --provider qcloud
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from examples.harness import ExampleContext, ExampleOutcome, main
from qagents import Agent, Limits, Session

NAME = "reflection"
DESCRIPTION = "A writer and a critic in a bounded revise loop, with hard rules checked in Python"
INSPIRED_BY = "LangGraph reflection / evaluator-optimizer workflow"

MAX_ROUNDS = 3
MAX_WORDS = 50
REQUIRED_PHRASES = ("Tidewell TW-2", "keeps drinks cold for 24 hours", "dishwasher-safe lid")
FACTS = """\
- Product: Tidewell TW-2 insulated bottle
- Capacity: 750 ml; weight: 340 g
- Keeps drinks cold for 24 hours and hot for 12 hours
- Body: double-wall, 90% recycled stainless steel
- Lid: leak-proof, dishwasher-safe lid"""
BRIEF = (
    f"Write the shop-page description of this bottle.\n\nFacts:\n{FACTS}\n\nRules: at most "
    f"{MAX_WORDS} words; include these phrases word for word: "
    + ", ".join(f'"{phrase}"' for phrase in REQUIRED_PHRASES)
    + "; use only the facts above; no exclamation marks."
)


class Critique(BaseModel):
    score: int = Field(ge=1, le=10, description="1 (poor) to 10 (ready to publish).")
    issues: list[str] = Field(
        default_factory=list, description="Concrete problems, each one sentence saying what to fix."
    )
    passes: bool = Field(description="True when the description can be published as it is.")


def check_rules(draft: str) -> list[str]:
    """The hard rules a draft breaks, checked in plain Python."""
    text = " ".join(draft.replace("‑", "-").split()).lower()  # U+2011: non-breaking hyphen
    problems = []
    words = len(draft.split())
    if words > MAX_WORDS:
        problems.append(f"It has {words} words; the limit is {MAX_WORDS}.")
    for phrase in REQUIRED_PHRASES:
        if phrase.lower() not in text:
            problems.append(f'It must contain the exact phrase "{phrase}".')
    if "!" in draft:
        problems.append("It must not contain exclamation marks.")
    return problems


async def run(ctx: ExampleContext) -> ExampleOutcome:
    writer = Agent(
        ctx.model,
        name="writer",
        instructions=(
            "You are a copywriter for an outdoor-gear shop. Follow the brief exactly. Reply with "
            "the description only, as plain text: no title, no markdown, no quotes, no notes. "
            "When asked to revise, reply with the complete revised description."
        ),
        limits=Limits(max_turns=2),
        hooks=ctx.hooks,
    )
    critic = Agent(
        ctx.model,
        name="critic",
        instructions=(
            "You are the editor of an outdoor-gear shop. Review a product description against "
            "its brief: every claim must be in the facts, and the text must be clear and plain. "
            "Do not count words or check the required phrases; code does that. List only the "
            "issues that must be fixed, each as one instruction, and set passes to true when "
            "there are none. Style preferences alone are not issues."
        ),
        output_type=Critique,
        output_mode=ctx.output_mode,
        limits=Limits(max_turns=3),
        hooks=ctx.hooks,
    )

    session = Session()  # the writer's conversation: brief, drafts and feedback
    prompt = BRIEF
    draft, critique, rounds = "", None, []
    for round_number in range(1, MAX_ROUNDS + 1):
        ctx.log(f"Round {round_number}: the writer drafts")
        written = await ctx.print_stream(writer.stream(prompt, session=session))
        draft = written.text.strip()
        problems = check_rules(draft)
        reviewed = await critic.run(f"Brief:\n{BRIEF}\n\nDescription:\n{draft}")
        critique = reviewed.output if reviewed.completed else None
        if critique is None:
            ctx.log(f"The critic stopped: {reviewed.stop_reason}")
            break
        ctx.log(
            f"Critic: score {critique.score}, passes={critique.passes}, "
            f"{len(critique.issues)} issue(s). Rule check: {' '.join(problems) or 'all rules met'}"
        )
        rounds.append(
            {"draft": draft, "rule_problems": problems, "critique": critique.model_dump()}
        )
        issues = problems + critique.issues
        if not problems and (critique.passes or not issues):
            break  # the rules are met, and the critic passes it or names nothing to fix
        prompt = "Revise the description. Fix these issues:\n" + "\n".join(f"- {i}" for i in issues)

    problems = check_rules(draft)
    passed = critique is not None and critique.passes
    verdict = "accepted by the critic" if passed else "the critic did not pass it"
    if problems:
        verdict = "breaks: " + " ".join(problems)
    return ExampleOutcome(
        ok=not problems,  # an empty draft lacks the required phrases
        summary=f"{len(draft.split())} words after {len(rounds)} round(s), {verdict}",
        output=draft,
        details={"rounds": rounds},
    )


if __name__ == "__main__":
    main(run, name=NAME, description=DESCRIPTION, inspired_by=INSPIRED_BY)
