"""Dogfood 3: a supervisor with two sub-agents, tested with scripted models."""

import asyncio

from qagents import Agent, Limits, tool
from qagents.models.testing import FunctionModel, ScriptedModel, call

HEADCOUNT = {"Acme": 120, "Borealis": 80}


@tool(read_only=True)
def lookup(company: str) -> str:
    """Return the headcount of a company."""
    return f"{company} has {HEADCOUNT[company]} employees."


@tool(read_only=True)
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def analyst_model(request):
    """Answers like an analyst: add the numbers, then report the sum."""
    last = request.messages[-1]
    if last.role == "tool":
        return f"The total is {last.tool_results[0].content}."
    return call("add", a=120, b=80)


async def main() -> None:
    researcher = Agent(
        ScriptedModel(
            [
                call("lookup", company="Acme"),
                call("lookup", company="Borealis"),
                "Acme 120, Borealis 80.",
            ]
        ),
        name="researcher",
        description="Finds company facts.",
        tools=[lookup],
    )
    analyst = Agent(
        FunctionModel(analyst_model),
        name="analyst",
        description="Does arithmetic on numbers it is given.",
        tools=[add],
    )
    supervisor_model = ScriptedModel(
        [
            call("researcher", task="Find the headcount of Acme and Borealis."),
            call("analyst", task="Add 120 and 80."),
            "Acme and Borealis employ 200 people together.",
        ]
    )
    supervisor = Agent(
        supervisor_model,
        name="supervisor",
        instructions="Delegate research and arithmetic, then answer.",
        tools=[researcher, analyst.as_tool(read_only=True)],
        limits=Limits(max_turns=8, max_depth=2),
    )
    result = await supervisor.run("How many people do Acme and Borealis employ?")
    print("status:", result.status, "| output:", result.output)
    assert result.status == "completed"
    researcher_answer = supervisor_model.requests[1].messages[-1].tool_results[0].content
    analyst_answer = supervisor_model.requests[2].messages[-1].tool_results[0].content
    print("researcher said:", researcher_answer)
    print("analyst said:", analyst_answer)
    assert "120" in researcher_answer and analyst_answer == "The total is 200."
    print("requests in the whole tree:", result.usage.requests)


if __name__ == "__main__":
    asyncio.run(main())
