"""Documents are summarised in parallel, then one agent combines the summaries into a brief.

The job is a `Workflow` journaled in a SQLite file in the work directory. Its first step,
``collect``, reads the documents in the inbox folder, so how many branches the map has is
known only at run time. The map is ``Workflow.map``: one durable step per document
(``summarise[0]``, ``summarise[1]``, ...), at most three at a time. The reduce is
``Workflow.run_agent``, a durable agent step. The example then runs the same workflow id a
second time, as a restarted process would: every step is replayed from the journal, so the
second run sends no model request at all (the harness's meter counts them).

Mirrors: LangGraph map-reduce with ``Send()``. There a conditional edge returns one
``Send("summarise", {...})`` per item, so the number of parallel branches is decided at run
time; every branch writes to a state key whose reducer (``Annotated[list, operator.add]``)
collects the results for the reduce node; and a checkpointer makes the graph durable. In
qagents there is no graph and no state reducer: ``Workflow.map`` takes a list and returns the
results in the same order, and each item is a journaled step of its own. Running the workflow
id again skips every step that finished, so after a failure only the unfinished items run.
Results are matched to items by position, which is why the item list is itself a step: a
later run maps over the same documents in the same order, even if the inbox has changed.

Run it:

    python -m examples.patterns.map_reduce --provider anthropic
    python -m examples.patterns.map_reduce --provider qcloud
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

from examples.harness import ExampleContext, ExampleOutcome, main
from qagents import Agent, Limits, SQLiteStore, Workflow

NAME = "map_reduce"
DESCRIPTION = "Summarise documents in parallel as durable steps, reduce them to one brief"
INSPIRED_BY = "LangGraph map-reduce with Send()"

WORKFLOW_ID = "council-brief:2025-q3"  # running this id again continues the same job
INBOX = "inbox"  # in the work directory: one text file per document, the title on line 1

DOCUMENTS = {  # fixture data: title -> text
    "Ferry Pilot Results": (
        "The six-month electric ferry pilot carried 148,000 passengers, 22% more than the "
        "diesel service a year earlier. Charging problems at the north pier cancelled two days "
        "of service in March. Operating cost per passenger fell from 3.10 to 2.45 euros."
    ),
    "Rooftop Solar Survey": (
        "Of 410 municipal roofs surveyed, 126 can carry solar panels without structural work. "
        "Together they could produce 9.8 GWh a year, about 14% of the city's own electricity "
        "use. The survey recommends starting with the 30 school roofs."
    ),
    "Bike Lane Counts": (
        "Counters on the new Elm Street lanes recorded 3,900 cyclists a day in September, up "
        "from 1,600 before the lanes opened. Car traffic on Elm Street fell 11%. Two collisions "
        "were reported, both at the Market Square junction."
    ),
    "Water Leak Repairs": (
        "Crews repaired 1,240 leaks in the third quarter, cutting estimated water loss from 24% "
        "to 19% of supply. Old cast-iron mains in the port district cause half of all leaks; "
        "replacing them would cost 18 million euros over six years."
    ),
    "Library Opening Hours": (
        "Sunday opening at the central library, trialled for twelve weeks, drew 640 visitors a "
        "day, more than any weekday. The extra staffing cost 52,000 euros. Branch libraries saw "
        "no fall in visits."
    ),
    "Street Lighting Upgrade": (
        "8,300 of the city's 12,000 street lights now use LEDs. Electricity used for street "
        "lighting fell 41% compared with 2022, and residents reported 35% fewer broken lights."
    ),
}


class Document(BaseModel):
    title: str
    text: str


class Summary(BaseModel):
    """The result of one map step, as it is journaled and replayed."""

    title: str
    summary: str


def fill_inbox(folder: Path) -> None:
    """Put the fixture documents in the inbox, one file each."""
    folder.mkdir(parents=True, exist_ok=True)
    for number, (title, text) in enumerate(DOCUMENTS.items(), start=1):
        (folder / f"{number:02d}.txt").write_text(f"{title}\n{text}\n", encoding="utf-8")


def read_inbox(folder: Path) -> list[Document]:
    """Every document in the inbox, in file name order."""
    documents = []
    for path in sorted(folder.glob("*.txt")):
        title, _, text = path.read_text(encoding="utf-8").partition("\n")
        documents.append(Document(title=title.strip(), text=text.strip()))
    return documents


async def write_brief(ctx: ExampleContext, store: SQLiteStore) -> str:
    """The whole job: collect the documents, summarise each one, combine the summaries."""
    workflow = Workflow(store, id=WORKFLOW_ID)
    summariser = Agent(
        ctx.model,
        name="summariser",
        instructions=(
            "You summarise one document for a city council brief in two sentences that keep "
            "its key figures. Use only the document. Reply with the two sentences only."
        ),
        limits=Limits(max_turns=2),
        hooks=ctx.hooks,
    )
    reducer = Agent(
        ctx.model,
        name="reducer",
        instructions=(
            "You write a combined brief for the city council from the document summaries you "
            "are given: one bullet per document, starting with the document's exact title and "
            "giving its key figure. At most 200 words. Use only the summaries."
        ),
        limits=Limits(max_turns=2),
        hooks=ctx.hooks,
    )

    async def summarise(document: Document) -> Summary:  # one map step
        ctx.log(f"summarising {document.title!r}")
        result = await summariser.run(f"Title: {document.title}\n\n{document.text}")
        if result.status != "completed":  # raising leaves the step unfinished, to run again
            raise RuntimeError(f"The summary of {document.title!r} stopped: {result.stop_reason}")
        return Summary(title=document.title, summary=result.text)

    # How many items there are is known only now, like the list a Send() edge fans out over.
    documents = await workflow.step(
        "collect", lambda: read_inbox(ctx.workdir / INBOX), type=list[Document]
    )
    # Map: one step per document, three at a time; finished steps are replayed, not run.
    summaries = await workflow.map("summarise", documents, summarise, concurrency=3, type=Summary)
    # Reduce: an agent step, also replayed once it has finished.
    notes = "\n\n".join(f"## {item.title}\n{item.summary}" for item in summaries)
    brief = await workflow.run_agent("reduce", reducer, f"Summaries:\n\n{notes}")
    return brief.text


def requests_sent(ctx: ExampleContext) -> int:
    """Model requests of the example so far, from the harness's meter."""
    if ctx.meter is None:
        raise RuntimeError("Run this example through examples.harness, which counts requests.")
    return ctx.meter.usage.requests


def missing_titles(brief: str) -> list[str]:
    """Titles the brief does not mention, ignoring case, punctuation and markup."""
    words = " ".join(re.findall(r"[a-z0-9]+", brief.lower()))
    return [title for title in DOCUMENTS if " ".join(title.lower().split()) not in words]


async def run(ctx: ExampleContext) -> ExampleOutcome:
    fill_inbox(ctx.workdir / INBOX)
    store = SQLiteStore(ctx.workdir / "map_reduce.db")
    try:
        earlier = (await Workflow(store, id=WORKFLOW_ID).status()).completed_steps
        if earlier:  # a --workdir that holds this job already
            ctx.log(f"The journal already holds {len(earlier)} finished steps; they are replayed")
        ctx.log(f"Run 1 of workflow {WORKFLOW_ID!r}: collect, map three at a time, reduce")
        start = requests_sent(ctx)
        first = await write_brief(ctx, store)
        first_requests = requests_sent(ctx) - start
        print(first)

        ctx.log("Run 2 of the same workflow id, as after a restart: every step is replayed")
        start = requests_sent(ctx)
        second = await write_brief(ctx, store)
        second_requests = requests_sent(ctx) - start
        steps = (await Workflow(store, id=WORKFLOW_ID).status()).completed_steps
    finally:
        await store.aclose()

    ctx.log(f"Requests: {first_requests} in run 1, {second_requests} in run 2")
    ctx.log(f"Journaled steps: {', '.join(steps)}")
    missing = missing_titles(first)
    ok = not missing and second_requests == 0 and second == first
    summary = (
        f"the brief names {len(DOCUMENTS) - len(missing)} of {len(DOCUMENTS)} documents; "
        f"run 1 sent {first_requests} requests, run 2 replayed {len(steps)} steps "
        f"with {second_requests}"
    )
    return ExampleOutcome(
        ok=ok,
        summary=summary,
        output=first,
        details={
            "first_requests": first_requests,
            "second_requests": second_requests,
            "steps": steps,
            "missing_titles": missing,
        },
    )


if __name__ == "__main__":
    main(run, name=NAME, description=DESCRIPTION, inspired_by=INSPIRED_BY)
