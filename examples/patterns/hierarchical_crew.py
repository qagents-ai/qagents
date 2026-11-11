"""A manager agent delegates to a pool of specialists and returns one structured report.

The manager reviews a product launch with three specialists (market, legal, technical), each
with its own fixture tool. It has one `fan_out` tool per specialist, writes the questions for
each of them, and the questions are answered in parallel child runs. From the answers it
writes a `LaunchReport`, validated against the pydantic type. The example checks the report
and also reads the manager's conversation: an area only counts as delegated when that
specialist really answered.

Mirrors: CrewAI's hierarchical process (``Crew(process=Process.hierarchical,
manager_llm=...)`` or ``manager_agent=...``), where a manager agent plans the work and hands
it out with its "Delegate work to coworker" and "Ask question to coworker" tools, naming the
coworker by role and passing the context the coworker needs. In qagents the manager is an
agent you write yourself, with one delegation tool per specialist: a `fan_out` tool that
takes several questions and works on them in parallel. Every answer is a linked child run:
it sees only the question the manager wrote (so the question must carry its own context, as
in CrewAI), it spends from the manager's cost budget, and the manager's propagating hooks
trace it (one MLflow trace for the whole crew).

Run it:

    python -m examples.patterns.hierarchical_crew --provider anthropic
    python -m examples.patterns.hierarchical_crew --provider qcloud
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from examples.harness import ExampleContext, ExampleOutcome, main
from qagents import Agent, Limits, RunResult, Tool, ToolError, fan_out, tool

NAME = "hierarchical_crew"
DESCRIPTION = "Manager delegates to market, legal and technical specialists via fan_out"
INSPIRED_BY = "CrewAI hierarchical process (manager agent delegating to specialists)"

PROPOSAL = (
    "We plan to launch an e-cargo-bike subscription for families in Lisbon in spring 2027: "
    "EUR 89 a month, bike, insurance and repairs included. Should we launch?"
)
AREAS = ("market", "legal", "technical")

# -- the specialists' data ------------------------------------------------------------------

MARKET = {
    "lisbon": {
        "households_with_children": 118_000,
        "survey_interest_pct": 9,
        "would_pay_89_eur_pct": 3.1,
        "competitors": [
            {"name": "bike rental shops", "monthly_price_eur": 120, "cargo_bikes": True},
            {"name": "municipal bike share", "monthly_price_eur": 15, "cargo_bikes": False},
        ],
    },
}
RULES = {
    "e-bikes": "Pedal assistance up to 25 km/h and 250 W counts as a bicycle: no licence, "
    "registration or helmet requirement. Faster or stronger models are mopeds.",
    "subscriptions": "Consumers may cancel a distance contract within 14 days without a reason; "
    "a monthly subscription must be cancellable with at most 30 days' notice.",
    "insurance": "Third-party liability insurance is not mandatory for pedal-assisted "
    "bicycles; operators renting them commercially must hold it.",
}
FIELD_TEST = {
    "bike": "Cargo One, 250 W motor, 25 km/h assistance",
    "range_km_city_test": 48,
    "max_climbable_gradient_pct_loaded": 12,
    "steepest_street_gradient_pct": 20,
    "share_of_streets_too_steep_when_loaded_pct": 18,
    "charge_hours": 4.5,
    "breakdowns_per_1000_km": 0.6,
}


@tool(read_only=True)
def market_data(city: str) -> dict:
    """Market figures for a city: households, survey interest, competitors and prices.

    Args:
        city: The city, for example "Lisbon".
    """
    data = MARKET.get(city.strip().lower())
    if data is None:
        raise ToolError(f"No market data for {city!r}. Cities with data: Lisbon.")
    return data


@tool(read_only=True)
def regulation(topic: Literal["e-bikes", "subscriptions", "insurance"]) -> str:
    """The Portuguese rule on one topic.

    Args:
        topic: Which rule to look up.
    """
    return RULES[topic]


@tool(read_only=True)
def field_test() -> dict:
    """Results of the Lisbon field test of the bike: range, hills, charging and breakdowns."""
    return FIELD_TEST


# -- the report ----------------------------------------------------------------------------------


class Finding(BaseModel):
    area: Literal["market", "legal", "technical"]
    summary: str = Field(description="What the specialists found, with their figures.")
    risks: list[str]


class LaunchReport(BaseModel):
    recommendation: Literal["launch", "launch with conditions", "do not launch"]
    findings: list[Finding] = Field(description="One finding per area.")
    next_steps: list[str]


def specialist(ctx: ExampleContext, area: str, role: str, data: Tool) -> Tool:
    """A specialist agent, exposed to the manager as a fan_out tool named ``ask_<area>``."""
    agent = Agent(
        ctx.model,
        name=f"{area}_specialist",
        description=f"The {role}. Answers {area} questions about the launch.",
        instructions=(
            f"You are the {role}. Call {data.name} to get the facts, then answer the question "
            "in at most four sentences, quoting the figures you used. If the data does not "
            "cover the question, say so."
        ),
        tools=[data],
        limits=Limits(max_turns=4),  # its cost is capped by what is left of the manager's
    )
    # Several specialist runs on a reasoning model can outlast Limits.tool_timeout_s (120 s).
    return fan_out(agent, name=f"ask_{area}", max_parallel=3, max_tasks=3, timeout=240)


def build_manager(ctx: ExampleContext) -> Agent:
    return Agent(
        ctx.model,
        name="manager",
        instructions=(
            "You are the manager of a launch review. You never research anything yourself; "
            "your specialists do. First, in one turn, call ask_market, ask_legal and "
            "ask_technical, giving each one or two specific questions about the proposal. "
            "The specialists cannot see the proposal, so put the facts they need (city, "
            "price, what is included) into every question. When the answers are back, write "
            "the report from them alone: one finding per area (market, legal, technical) with "
            "the specialists' figures, your recommendation, and the next steps."
        ),
        tools=[
            specialist(ctx, "market", "market analyst", market_data),
            specialist(ctx, "legal", "legal adviser for Portugal", regulation),
            specialist(ctx, "technical", "fleet engineer", field_test),
        ],
        output_type=LaunchReport,
        output_mode=ctx.output_mode,
        limits=Limits(max_turns=8, max_cost_usd=0.40),  # the whole crew: children share it
        hooks=ctx.hooks,
    )


def answered_areas(result: RunResult) -> set[str]:
    """Areas whose specialist answered at least one question (from the fan_out results)."""
    areas = set()
    for message in result.messages:
        for answer in message.tool_results:
            if answer.name.startswith("ask_") and not answer.is_error:
                if any("answer" in entry for entry in json.loads(answer.content)):
                    areas.add(answer.name.removeprefix("ask_"))
    return areas


async def run(ctx: ExampleContext) -> ExampleOutcome:
    manager = build_manager(ctx)
    ctx.log("The manager reviews the proposal and delegates to its specialists")
    result = await ctx.print_stream(manager.stream(PROPOSAL))
    if not result.completed:
        reason = result.stop_reason or "-"
        return ExampleOutcome(ok=False, summary=f"The manager stopped: {result.status} ({reason})")

    report: LaunchReport = result.output
    print(f"\nRecommendation: {report.recommendation}")
    for finding in report.findings:
        print(f"- {finding.area}: {finding.summary}")
    delegated = answered_areas(result)
    reported = {finding.area for finding in report.findings}
    ok = reported == set(AREAS) and delegated == set(AREAS)
    summary = (
        f"{report.recommendation}; areas reported: {', '.join(sorted(reported))}; "
        f"delegated to: {', '.join(sorted(delegated)) or 'nobody'}"
    )
    return ExampleOutcome(
        ok=ok,
        summary=summary,
        output=report.model_dump(),
        details={"turns": result.turns, "usage": result.usage.model_dump()},
    )


if __name__ == "__main__":
    main(run, name=NAME, description=DESCRIPTION, inspired_by=INSPIRED_BY)
