"""A manager directs two teams, each run by a lead agent with workers of its own.

Three levels of agents, each level a tool of the one above:

    manager                       depth 0
    |-- research_lead             depth 1
    |   `-- searcher x 3          depth 2, one per sub-question, in parallel (fan_out)
    `-- writing_lead              depth 1
        |-- writer                depth 2 (agent.as_tool)
        `-- editor                depth 2 (agent.as_tool)

Only the manager has a cost budget (``Limits.max_cost_usd``); every run below it starts with
what is left of its parent's, so the cap holds for the whole tree. Every agent has
``Limits(max_depth=2)``, so a fourth level would refuse to start. A propagating hook (a `Hooks`
subclass with ``propagate = True``) is attached once, to the manager, and records every run of
the tree with its depth. The sales figures exist only in the searchers' database, so a final
document that states them shows the findings travelled up one team and down the other.

Mirrors: LangGraph hierarchical agent teams. There each team is a graph of its own (a research
team with a search agent and a web scraper, a writing team with a writer, a note-taker and a
chart generator), run by a supervisor node that picks the next worker with structured output
and ``Command(goto=...)``. The top-level supervisor calls a team graph with only its last
message and adds the team's last message back to its own state. In qagents there are no graphs
to compose: a lead is an agent with its workers as tools, and the manager calls each lead as a
tool, which gives the same boundary (a task in, an answer out) by construction. Routing is the
model's tool calls, so a lead can also run workers in parallel: the research lead sends every
sub-question to a searcher in one ``fan_out`` call, where LangGraph's research supervisor routes
to one worker at a time. Budgets, usage and traces roll up through the linked child runs, and
the stops are ``Limits`` on each agent instead of one ``recursion_limit`` per graph run.

Run it:

    python -m examples.patterns.hierarchical_teams --provider anthropic
    python -m examples.patterns.hierarchical_teams --provider qcloud
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from examples.harness import ExampleContext, ExampleOutcome, main
from qagents import Agent, Hooks, Limits, Model, RunContext, fan_out, tool

NAME = "hierarchical_teams"
DESCRIPTION = "A manager with a research team (parallel searchers) and a writing team"
INSPIRED_BY = "LangGraph hierarchical agent teams"

MARKET = {  # fixture data
    "Norway": "Norway: 112,000 heat pumps sold in 2025; 67% of homes already have one; cheap "
    "hydropower keeps running costs low.",
    "Poland": "Poland: 210,000 heat pumps sold in 2025, down 18% on 2024 after a subsidy pause; "
    "30% of homes still heat with coal.",
    "Italy": "Italy: 380,000 heat pumps sold in 2025; hybrid systems were 40% of sales after "
    "the tax credit fell from 65% to 50%.",
}
COUNTRY_WORDS = {
    "Norway": ("norway", "norwegian"),
    "Poland": ("poland", "polish"),
    "Italy": ("italy", "italian"),
}
SOLD_THOUSANDS = {"Norway": 112, "Poland": 210, "Italy": 380}  # the figure the document must state
SUB_QUESTIONS = [
    "How many heat pumps were sold in Norway in 2025, and what drives demand there?",
    "How many heat pumps were sold in Poland in 2025, and why did sales change?",
    "How many heat pumps were sold in Italy in 2025, and which systems sold best?",
]
BRIEF = "Topic: the European heat pump market in 2025.\nSub-questions:\n" + "\n".join(
    f"{number}. {question}" for number, question in enumerate(SUB_QUESTIONS, start=1)
)
MAX_DEPTH = 2
TEAMS = {(1, "research_lead"), (2, "searcher"), (1, "writing_lead"), (2, "writer")}
"""Runs that show both teams did the work: (depth, agent)."""


@tool(read_only=True)
def search(query: str) -> str:
    """Search the 2025 European heat pump market database.

    Args:
        query: What to look for; name the country, for example "Norway heat pump sales 2025".
    """
    text = query.lower()
    hits = [
        fact for country, fact in MARKET.items() if any(w in text for w in COUNTRY_WORDS[country])
    ]
    return "\n".join(hits) or f"No results. The database covers {', '.join(MARKET)}."


class RunTree(Hooks):
    """Records every run of the tree with its depth, in the order the runs start."""

    propagate = True  # attached to the manager, it also runs in every child run below it

    def __init__(self) -> None:
        self.runs: list[tuple[int, str]] = []

    async def on_run_start(self, ctx: RunContext[Any]) -> None:
        self.runs.append((ctx.depth, ctx.agent_name))


def build_manager(
    model: Model,
    *,
    hooks: Sequence[Hooks] = (),
    max_depth: int = MAX_DEPTH,
    max_cost_usd: float = 0.40,
) -> Agent:
    """The manager and its two teams.

    A child run is checked against its own ``max_depth`` and its direct parent's, not the
    manager's, so every agent carries the limit. The budget needs setting once: it is inherited.
    A sub-agent tool runs under its caller's ``Limits.tool_timeout_s`` (120 s by default), so
    the callers of slow sub-agents raise it.
    """

    def limits(turns: int, **more: Any) -> Limits:
        return Limits(max_turns=turns, max_depth=max_depth, **more)

    searcher = Agent(
        model,
        name="searcher",
        description="Answers one research question from the heat pump market database.",
        instructions=(
            "Answer the question with the search tool, then reply in at most three sentences "
            "with the facts and figures found. Use only what the search returns."
        ),
        tools=[search],
        limits=limits(4),
    )
    research_lead = Agent(
        model,
        name="research_lead",
        description="Leads the research team. Give it the topic and every sub-question.",
        instructions=(
            "You lead a research team. Call searcher_parallel once, with one task per "
            "sub-question, each task being the sub-question itself. Then reply with the "
            "findings: one short paragraph per sub-question, keeping every figure. Never "
            "answer from memory."
        ),
        tools=[fan_out(searcher, max_parallel=3, max_tasks=6, timeout=300)],
        limits=limits(4),
    )
    writer = Agent(
        model,
        name="writer",
        description="Writes a briefing. Give it the topic, the sub-questions and all findings.",
        instructions=(
            "You write a briefing in Markdown from the notes you are given: a title, then one "
            "short section per sub-question with its figures. At most 250 words. Use only the "
            "notes."
        ),
        limits=limits(2),
    )
    editor = Agent(
        model,
        name="editor",
        description="Edits a draft. Give it the complete draft.",
        instructions=(
            "You edit the draft you are given: fix mistakes and tighten the wording, keep every "
            "section and every figure, stay under 250 words. Reply with the edited draft only."
        ),
        limits=limits(2),
    )
    writing_lead = Agent(
        model,
        name="writing_lead",
        description=(
            "Leads the writing team. Give it the topic, the sub-questions and the research "
            "findings; it returns the finished document."
        ),
        instructions=(
            "You lead a writing team and never write yourself.\n"
            "1. Call writer once with the topic, the sub-questions and all findings.\n"
            "2. After it has answered, call editor once with the writer's complete draft.\n"
            "3. Reply with the editor's text exactly, and nothing else."
        ),
        tools=[writer.as_tool(), editor.as_tool()],
        limits=limits(5, tool_timeout_s=300),
    )
    return Agent(
        model,
        name="manager",
        instructions=(
            "You manage two teams and never research or write yourself.\n"
            "1. Call research_lead once with the topic and every sub-question.\n"
            "2. After it has answered, call writing_lead once with the topic, the "
            "sub-questions and the research lead's findings copied in full.\n"
            "3. Reply with the writing lead's document exactly as returned. Then stop."
        ),
        tools=[research_lead.as_tool(), writing_lead.as_tool()],
        limits=limits(6, max_cost_usd=max_cost_usd, tool_timeout_s=600),
        hooks=hooks,
    )


def uncovered(document: str) -> list[str]:
    """The countries whose 2025 sales figure the document does not state.

    112,000 may be written 112,000, 112 000, 112000, 112k or 112 thousand.
    """
    return [
        country
        for country, thousands in SOLD_THOUSANDS.items()
        if not re.search(rf"\b{thousands}(?:[,.\s]?000\b|\s?(?:k|thousand)\b)", document, re.I)
    ]


async def run(ctx: ExampleContext) -> ExampleOutcome:
    tree = RunTree()
    manager = build_manager(ctx.model, hooks=[*ctx.hooks, tree])
    ctx.log("Briefing the manager; worker lines are prefixed with the path of agents")
    result = await ctx.print_stream(manager.stream(BRIEF))

    ctx.log("The runs, as the propagating hook saw them start:")
    for depth, name in tree.runs:
        print(f"   {'    ' * depth}{name} (depth {depth})")
    cost, budget = result.usage.cost_usd, manager.limits.max_cost_usd
    spent = "cost unknown" if cost is None else f"${cost:.4f} of the manager's ${budget:.2f}"
    ctx.log(f"{result.usage.requests} requests in the manager's run, all levels included; {spent}")

    missing = uncovered(result.text)
    skipped = sorted(name for _, name in TEAMS - set(tree.runs))
    ok = result.status == "completed" and not missing and not skipped
    summary = (
        f"the document states the figures of {len(SOLD_THOUSANDS) - len(missing)} of "
        f"{len(SOLD_THOUSANDS)} sub-questions; {len(tree.runs)} runs, deepest at depth "
        f"{max(depth for depth, _ in tree.runs)}; {result.usage.requests} requests"
    )
    if skipped:
        summary += f"; never ran: {', '.join(skipped)}"
    if result.status != "completed":
        summary = f"{result.status} ({result.stop_reason}); {summary}"
    return ExampleOutcome(
        ok=ok,
        summary=summary,
        output=result.text,
        details={
            "runs": [f"{name}@{depth}" for depth, name in tree.runs],
            "missing": missing,
            "requests": result.usage.requests,
            "cost_usd": cost,
        },
    )


if __name__ == "__main__":
    main(run, name=NAME, description=DESCRIPTION, inspired_by=INSPIRED_BY)
