"""A sequential crew: a researcher, a writer and an editor hand typed results down the line.

Each agent's instructions are written the CrewAI way, as a role, a goal and a backstory. Each
task is a durable `Workflow.run_agent` step whose answer is a pydantic type (`ResearchNotes`,
`Draft`, `FinalArticle`). The researcher searches a fixture news archive that returns at most
three articles per search, so it has to search more than once; the writer and the editor see
only what the code puts in their prompts: the notes, then the notes and the draft.

Mirrors: CrewAI's sequential process, ``Crew(agents=[...], tasks=[...],
process=Process.sequential)`` with ``Agent(role=, goal=, backstory=)`` and
``Task(description=, expected_output=, output_pydantic=, context=[...])``. In qagents there
is no Crew object: the process is an ordinary async function, and each task is one
``run_agent`` call. ``role``, ``goal`` and ``backstory`` become the agent's instructions,
``description`` and ``expected_output`` its prompt, and ``output_pydantic`` its
``output_type``, which the answer is validated against. CrewAI hands the earlier task outputs
to the next task for you; here ``context`` is JSON you put into the next prompt yourself, so
you can read and test exactly what each agent receives. The steps are journaled in SQLite:
run it again with the same ``--workdir`` and the finished tasks are replayed, not re-run.

Run it:

    python -m examples.patterns.research_crew --provider anthropic
    python -m examples.patterns.research_crew --provider qcloud
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from examples.harness import ExampleContext, ExampleOutcome, main
from qagents import Agent, Limits, RunResult, SQLiteStore, Workflow, tool

NAME = "research_crew"
DESCRIPTION = "Researcher, writer and editor as typed, durable workflow steps"
INSPIRED_BY = "CrewAI sequential crew (role, goal, backstory; tasks with expected_output)"

TOPIC = "the Aegean Green Ferry programme"

# The news archive the researcher searches: source id -> (title, text). src-5 is off topic.
ARCHIVE = {
    "src-1": (
        "Programme launch",
        "The Aegean Green Ferry programme will put 17 battery-electric ferries into service "
        "between 2027 and 2031.",
    ),
    "src-2": (
        "Funding",
        "The ferry programme has a budget of EUR 436 million; 60% comes from the EU Recovery Fund.",
    ),
    "src-3": (
        "The ships",
        "Each of the new ferries carries up to 388 passengers and recharges in 25 minutes at port.",
    ),
    "src-4": (
        "First route",
        "The first electric ferry route, Piraeus to Aegina, opens in May 2027.",
    ),
    "src-5": ("Olive harvest", "The 2026 olive harvest on Crete was 12% smaller than in 2025."),
}
ON_TOPIC = {"src-1", "src-2", "src-3", "src-4"}
KEY_FIGURES = ("17", "436", "388", "2027")
"""Figures the archive states; the final article must keep every one the research found."""


@tool(read_only=True)
def search(query: str) -> list[dict[str, str]] | str:
    """Search the news archive. Returns up to three articles, each with its source_id.

    Args:
        query: A few keywords, for example "ferry funding".
    """
    stems = {word[:4] for word in re.findall(r"\w+", query.lower()) if len(word) > 2}
    hits = []
    for source_id, (title, text) in ARCHIVE.items():
        score = sum(stem in f"{title} {text}".lower() for stem in stems)  # "ferr" finds ferries
        if score:
            hits.append((score, {"source_id": source_id, "title": title, "text": text}))
    hits.sort(key=lambda hit: -hit[0])  # best first; ties keep the archive order
    if not hits:
        return "No articles matched. Try other keywords, such as: ferries, funding, route."
    return [article for _, article in hits[:3]]


# -- what each task returns (CrewAI's output_pydantic) ----------------------------------------


class Fact(BaseModel):
    statement: str = Field(description="One fact, with its figures exactly as the source says.")
    source_id: str = Field(description="The source_id of the search result it comes from.")


class ResearchNotes(BaseModel):
    facts: list[Fact] = Field(min_length=3)


class Draft(BaseModel):
    title: str
    body: str = Field(description="The article, 120 to 200 words.")


class FinalArticle(BaseModel):
    title: str
    body: str
    changes: list[str] = Field(description="What the editor changed, one line each.")


# -- CrewAI's vocabulary, as two small functions ---------------------------------------------


def persona(*, role: str, goal: str, backstory: str, how: str) -> str:
    """Instructions written as CrewAI's role, goal and backstory, plus how to work."""
    return f"Role: {role}\nGoal: {goal}\nBackstory: {backstory}\n\n{how}"


def task_prompt(description: str, expected_output: str, **context: BaseModel) -> str:
    """A task as a prompt. ``context`` is what earlier tasks produced, passed explicitly."""
    parts = [description, f"Expected output: {expected_output}"]
    parts += [f"{name}:\n{value.model_dump_json(indent=2)}" for name, value in context.items()]
    return "\n\n".join(parts)


def build_crew(ctx: ExampleContext) -> tuple[Agent, Agent, Agent]:
    common = {"output_mode": ctx.output_mode, "hooks": ctx.hooks}
    researcher = Agent(
        ctx.model,
        name="researcher",
        instructions=persona(
            role="Senior research analyst",
            goal="Find sourced facts about the topic you are given.",
            backstory="Ten years at a news wire taught you never to report a figure you "
            "cannot source.",
            how="Call search two or three times with different keywords (the programme, its "
            "funding, its ships and routes), then answer. Use only facts the results state, "
            "keep their figures exactly, give each fact the source_id of its result, and "
            "leave out results about other topics.",
        ),
        tools=[search],
        output_type=ResearchNotes,
        limits=Limits(max_turns=6, max_cost_usd=0.15),
        **common,
    )
    writer = Agent(
        ctx.model,
        name="writer",
        instructions=persona(
            role="Science and technology writer",
            goal="Turn research notes into a clear news article for general readers.",
            backstory="You write for a weekly magazine and are known for getting every "
            "number right.",
            how="Write the article in one go from the notes you are given. Use every figure "
            "in the notes, written in digits exactly as the notes give it, and nothing that "
            "is not in them.",
        ),
        output_type=Draft,
        limits=Limits(max_turns=3, max_cost_usd=0.10),
        **common,
    )
    editor = Agent(
        ctx.model,
        name="editor",
        instructions=persona(
            role="Managing editor",
            goal="Publish an accurate, tight article.",
            backstory="You have sent back many drafts that got a figure wrong.",
            how="Check the draft against the research notes: correct any figure that differs, "
            "cut filler and keep every fact, with its figures in digits. Return the final "
            "article and list your changes (an empty list if there were none).",
        ),
        output_type=FinalArticle,
        limits=Limits(max_turns=3, max_cost_usd=0.10),
        **common,
    )
    return researcher, writer, editor


def stopped(task: str, result: RunResult) -> ExampleOutcome:
    reason = result.stop_reason or "-"
    return ExampleOutcome(ok=False, summary=f"The {task} task stopped: {result.status} ({reason})")


def figures_in(text: str) -> list[str]:
    return [figure for figure in KEY_FIGURES if re.search(rf"(?<!\d){figure}(?!\d)", text)]


async def run(ctx: ExampleContext) -> ExampleOutcome:
    researcher, writer, editor = build_crew(ctx)
    store = SQLiteStore(ctx.workdir / "crew.db")
    workflow = Workflow(store, id="article:aegean-green-ferry")
    try:
        ctx.log("Task 1, research: the researcher searches the archive")
        research = await workflow.run_agent(
            "research",
            researcher,
            task_prompt(
                f"Research {TOPIC}: what it is, what it costs, the ships and the first route.",
                "Three to six facts, each with the source_id it comes from.",
            ),
        )
        if not research.completed:
            return stopped("research", research)
        notes: ResearchNotes = research.output
        ctx.log(f"{len(notes.facts)} facts from {sorted({f.source_id for f in notes.facts})}")

        ctx.log("Task 2, write: the writer gets the research notes as context")
        written = await workflow.run_agent(
            "write",
            writer,
            task_prompt(
                f"Write a news article about {TOPIC}.",
                "A title and a body of 120 to 200 words.",
                research_notes=notes,
            ),
        )
        if not written.completed:
            return stopped("write", written)

        ctx.log("Task 3, edit: the editor gets the notes and the draft as context")
        edited = await workflow.run_agent(
            "edit",
            editor,
            task_prompt(
                "Edit the draft into the final article.",
                "The final title and body, and the list of changes you made.",
                research_notes=notes,
                draft=written.output,
            ),
        )
        if not edited.completed:
            return stopped("edit", edited)
    finally:
        await store.aclose()  # SQLiteStore has no `async with`; this releases the file

    article: FinalArticle = edited.output
    print(f"\n# {article.title}\n\n{article.body}\n")
    sourced = [fact for fact in notes.facts if fact.source_id in ON_TOPIC]
    researched = figures_in(notes.model_dump_json())
    covered = figures_in(f"{article.title}\n{article.body}")
    missing = [figure for figure in researched if figure not in covered]
    ok = len(sourced) >= 3 and len(researched) >= 3 and not missing
    summary = (
        f"{len(sourced)} sourced facts; the article keeps {len(researched) - len(missing)} of "
        f"the {len(researched)} researched figures"
        + (f" (missing: {', '.join(missing)})" if missing else "")
    )
    return ExampleOutcome(
        ok=ok,
        summary=summary,
        output=article.model_dump(),
        details={"facts": [fact.model_dump() for fact in notes.facts], "changes": article.changes},
    )


if __name__ == "__main__":
    main(run, name=NAME, description=DESCRIPTION, inspired_by=INSPIRED_BY)
