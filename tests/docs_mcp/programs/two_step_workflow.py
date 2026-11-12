"""Dogfood 2: a workflow with two durable agent steps and a plain step, run twice."""

import asyncio
import tempfile
from pathlib import Path

from qagents import Agent, SQLiteStore, Workflow, tool
from qagents.models.testing import ScriptedModel, call

PUBLISHED: list[str] = []


@tool(read_only=True)
def search(query: str) -> str:
    """Search the notes for a topic."""
    return f"Notes on {query}: adoption doubled in 2026."


def build_agents() -> tuple[Agent, Agent]:
    researcher = Agent(
        ScriptedModel([call("search", query="agents"), "Adoption of agents doubled in 2026."]),
        name="researcher",
        instructions="Find facts.",
        tools=[search],
    )
    writer = Agent(
        ScriptedModel(["Report: agent adoption doubled in 2026."]),
        name="writer",
        instructions="Write a one-line report.",
    )
    return researcher, writer


async def report(store: SQLiteStore, topic: str) -> str:
    researcher, writer = build_agents()
    wf = Workflow(store, id=f"report:{topic}")
    notes = await wf.run_agent("research", researcher, f"Research {topic}")
    draft = await wf.run_agent("write", writer, f"Write a report from:\n{notes.text}")

    def publish() -> str:
        PUBLISHED.append(draft.text)
        return "published"

    await wf.step("publish", publish)
    return draft.text


async def main() -> None:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "jobs.db"
        async with SQLiteStore(path) as store:
            first = await report(store, "agents")
        print("first run:", first)
        # A second run with the same id, in a fresh store object, replays every finished
        # step: the scripted models would fail if they were asked again, and publish does
        # not run twice.
        async with SQLiteStore(path) as store:
            second = await report(store, "agents")
            status = await Workflow(store, id="report:agents").status()
        print("second run:", second)
        assert first == second
        assert PUBLISHED == [first]
        print("completed steps:", status.completed_steps)


if __name__ == "__main__":
    asyncio.run(main())
