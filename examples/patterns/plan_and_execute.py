"""Plan and execute: a planner writes the steps, an executor does them, a replanner checks.

A planner agent turns the objective into a short typed plan. An executor agent with two tools,
a lookup in a small knowledge base and a calculator, carries out the steps one at a time. A
replanner reads the results and either gives the final answer or adds the steps still missing,
at most twice. Every agent run is a durable `Workflow` step.

Mirrors: LangGraph's plan-and-execute tutorial, a StateGraph with a planner node, an agent
node that carries out one step, and a replan node whose conditional edge goes back to the agent
until it returns a response. In qagents the graph is an ordinary async function:

- each node is an agent run as ``Workflow.run_agent``; the graph state is a list of
  (step, result) pairs, and the conditional edge is ``if review.done: break``;
- the tutorial's ``Act = Response | Plan`` is one flat `Review` model (``done``, ``answer``,
  ``new_steps``), which the structured output of every provider accepts, and an output
  validator sends back a review that neither answers nor adds steps;
- the recursion limit is a Python ``for`` over at most ``MAX_REPLANS`` replans, plus `Limits`
  on every agent;
- the checkpointer is the workflow journal, a SQLite file in the work directory. Run the
  example twice with the same ``--workdir`` and the second run replays every finished step
  without a model request.

Two choices differ from the tutorial: the executor also sees the results of earlier steps, and
the replanner reviews after each batch of planned steps rather than after every step, which
saves one model request per step.

Run it:

    python -m examples.patterns.plan_and_execute --provider anthropic
    python -m examples.patterns.plan_and_execute --provider qcloud
    python -m examples.patterns.plan_and_execute --workdir <dir>   # twice: the 2nd replays
"""

from __future__ import annotations

import ast
import contextlib
import operator
from typing import Any

from pydantic import BaseModel, Field

from examples.harness import ExampleContext, ExampleOutcome, main
from qagents import (
    Agent,
    Limits,
    ModelRetry,
    RunContext,
    RunResult,
    SQLiteStore,
    ToolError,
    Workflow,
    tool,
)

NAME = "plan_and_execute"
DESCRIPTION = "A planner, an executor with tools and a bounded replanner, as durable workflow steps"
INSPIRED_BY = "LangGraph plan-and-execute (planner, executor, replanner)"

OBJECTIVE = (
    "What is the total rated capacity, in MW, of the wind farms that Northwind Energy operates?"
)
MAX_REPLANS = 2

# -- the knowledge base: deterministic fixture data -------------------------------------------

WIND_FARMS = {  # name: (turbines, MW per turbine, what it is)
    "Orla Ridge": (14, 3.6, "an onshore wind farm opened in 2019"),
    "Brannock Bay": (22, 2.5, "an offshore wind farm opened in 2021"),
    "Kessick Moor": (9, 4.2, "an onshore wind farm opened in 2016"),
}
KNOWLEDGE = {
    "Northwind Energy": (
        "Northwind Energy is a utility. It operates three wind farms (Orla Ridge, Brannock Bay "
        "and Kessick Moor) and one solar park, Tarn Valley."
    ),
    "Tarn Valley": "Tarn Valley is a solar park with a rated capacity of 48 MW.",
    **{
        name: f"{name} is {kind}. It has {turbines} turbines rated {mw} MW each."
        for name, (turbines, mw, kind) in WIND_FARMS.items()
    },
}
EXPECTED_MW = round(sum(turbines * mw for turbines, mw, _ in WIND_FARMS.values()), 6)


@tool(read_only=True)
def lookup(topic: str) -> str:
    """Look up a company, wind farm or solar park in the knowledge base.

    Args:
        topic: Its name, for example "Orla Ridge".
    """
    wanted = topic.strip().lower()
    found = [
        text
        for name, text in KNOWLEDGE.items()
        if wanted and (name.lower() in wanted or wanted in name.lower())
    ]
    if not found:
        raise ToolError(f"Nothing about {topic!r}. The knowledge base has: {', '.join(KNOWLEDGE)}.")
    return "\n".join(found)


_OPERATORS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate(node: ast.AST) -> float:
    """Numbers, + - * / and parentheses; anything else raises ValueError."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError("unsupported expression")


@tool(read_only=True)
def calculate(expression: str) -> str:
    """Evaluate arithmetic with numbers, + - * / and parentheses.

    Args:
        expression: For example "14 * 3.6 + 22 * 2.5".
    """
    try:
        value = _evaluate(ast.parse(expression, mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError):
        raise ModelRetry(
            f"Cannot evaluate {expression!r}. Use only numbers, + - * / and parentheses, "
            "with no units or names."
        ) from None
    return f"{expression} = {round(value, 6)}"


# -- what the planner and the replanner answer ------------------------------------------------


class Plan(BaseModel):
    steps: list[str] = Field(
        min_length=1, max_length=5, description="The steps in order, each one short instruction."
    )


class Review(BaseModel):
    """LangGraph's ``Response | Plan`` as one flat object: ``done`` says which one it is."""

    done: bool = Field(description="True when the results so far answer the objective.")
    answer: float | None = Field(None, description="The final number, when done is true.")
    explanation: str = Field("", description="When done: how the results give the answer.")
    new_steps: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="When not done: only the steps still needed, in order.",
    )


def check_review(ctx: RunContext, review: Review) -> None:
    """An output validator: a review must either answer or say what is still needed."""
    if review.done and review.answer is None:
        raise ModelRetry("done is true, so give the final number in answer.")
    if not review.done and not review.new_steps:
        raise ModelRetry("done is false, so list the steps still needed in new_steps.")


# -- the flow -----------------------------------------------------------------------------------


def _results(done: list[tuple[str, str]]) -> str:
    lines = [f"{number}. {step}\n   {result}" for number, (step, result) in enumerate(done, 1)]
    return "\n".join(lines) or "(none yet)"


def step_prompt(done: list[tuple[str, str]], step: str) -> str:
    return (
        f"Objective: {OBJECTIVE}\n\nResults of earlier steps:\n{_results(done)}\n\n"
        f"Your step: {step}"
    )


def review_prompt(done: list[tuple[str, str]], *, last: bool) -> str:
    prompt = f"Objective: {OBJECTIVE}\n\nResults of the steps so far:\n{_results(done)}"
    if last:
        prompt += "\n\nNo more steps can be added: set done to true and answer from these results."
    return prompt


def _tool_calls(result: RunResult) -> str:
    """The tool calls of a step, for the narration (``run_agent`` has no event stream)."""
    calls = [
        f"{call.name}({', '.join(map(str, call.arguments.values()))})"
        for message in result.new_messages
        for call in message.tool_calls
    ]
    return ", ".join(calls) or "no tool calls"


async def run(ctx: ExampleContext) -> ExampleOutcome:
    tools_line = (
        "lookup (facts about a company, wind farm or solar park, by name) and calculate "
        "(arithmetic)"
    )
    planner = Agent(
        ctx.model,
        name="planner",
        instructions=(
            f"You plan research for an assistant that has two tools: {tools_line}. Split the "
            "objective into 2 to 5 short steps, in order; the last step calculates the final "
            "number. Do not answer the objective yourself."
        ),
        output_type=Plan,
        output_mode=ctx.output_mode,
        limits=Limits(max_turns=3),
        hooks=ctx.hooks,
    )
    executor = Agent(
        ctx.model,
        name="executor",
        instructions=(
            "You carry out one step of a research plan, using the results of earlier steps. Use "
            "lookup for facts and calculate for every piece of arithmetic; never calculate in "
            "your head. Do only your step, then reply in one or two sentences with what you "
            "found, including every number."
        ),
        tools=[lookup, calculate],
        limits=Limits(max_turns=8, max_tool_calls=10, max_cost_usd=0.08),
        hooks=ctx.hooks,
    )
    replanner = Agent(
        ctx.model,
        name="replanner",
        instructions=(
            "You review a research plan in progress. Its steps are carried out by an assistant "
            f"with two tools: {tools_line}. If the results answer the objective, set done to "
            "true and give the final number in answer; it must be a number that appears in the "
            "results. Otherwise set done to false and list in new_steps only the steps still "
            "needed (at most 3), without repeating steps that have results."
        ),
        output_type=Review,
        output_mode=ctx.output_mode,
        output_validators=[check_review],
        limits=Limits(max_turns=3),
        hooks=ctx.hooks,
    )

    done: list[tuple[str, str]] = []  # the graph state: (step, result) pairs
    review: Review | None = None
    replans = 0
    # The journal lives in the work directory: the same --workdir continues the same job.
    async with contextlib.aclosing(SQLiteStore(ctx.workdir / "plan_and_execute.db")) as store:
        workflow = Workflow(store, id="northwind-capacity")
        if finished := await workflow.completed_steps():
            ctx.log(f"Replaying {len(finished)} finished step(s) from the workflow journal")
        ctx.log(f"Objective: {OBJECTIVE}")
        planned = await workflow.run_agent("plan", planner, OBJECTIVE)
        if not planned.completed:
            return ExampleOutcome(ok=False, summary=f"The planner stopped: {planned.stop_reason}")
        steps = planned.output.steps
        ctx.log("Plan: " + " | ".join(steps))

        for round_number in range(1, MAX_REPLANS + 2):
            for step in steps:
                number = len(done) + 1
                ctx.log(f"Step {number}: {step}")
                result = await workflow.run_agent(
                    f"step-{number}", executor, step_prompt(done, step)
                )
                text = result.text if result.completed else f"(stopped: {result.stop_reason})"
                ctx.log(f"  {_tool_calls(result)} -> {text}")
                done.append((step, text))

            last = round_number > MAX_REPLANS
            reviewed = await workflow.run_agent(
                f"review-{round_number}", replanner, review_prompt(done, last=last)
            )
            review = reviewed.output if reviewed.completed else None
            if review is None or review.done or last:
                break
            replans += 1
            steps = review.new_steps
            ctx.log(f"Replan {replans}: " + " | ".join(steps))
        status = await workflow.status()

    answer = review.answer if review is not None and review.done else None
    ok = answer is not None and abs(answer - EXPECTED_MW) < 0.01
    if answer is None:
        summary = f"No answer after {len(done)} steps and {replans} replans"
    else:
        summary = f"{answer:g} MW after {len(done)} steps and {replans} replan(s)"
    return ExampleOutcome(
        ok=ok,
        summary=f"{summary} (expected {EXPECTED_MW:g} MW)",
        output=answer,
        details={
            "plan": planned.output.steps,
            "steps": [{"step": step, "result": result} for step, result in done],
            "replans": replans,
            "explanation": review.explanation if review is not None else None,
            "workflow_steps": status.completed_steps,
        },
    )


if __name__ == "__main__":
    main(run, name=NAME, description=DESCRIPTION, inspired_by=INSPIRED_BY)
