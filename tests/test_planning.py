from __future__ import annotations

import pytest

from qagents import Agent, RunContext, Session, prompts
from qagents.errors import SessionError
from qagents.models.testing import ScriptedModel, call
from qagents.toolkits.planning import TodoItem, read_todo_list, render_todos, todo_tools

PLAN = [
    {"content": "Collect the sources", "status": "completed"},
    {"content": "Draft the report", "status": "in_progress"},
    {"content": "Check the figures", "status": "pending"},
]


def instructions(ctx: RunContext) -> str:
    plan = render_todos(ctx.state)
    return f"Write a report.\n\n{plan}" if plan else "Write a report."


async def test_the_plan_is_kept_in_state_and_shown_on_every_turn():
    model = ScriptedModel([call("write_todos", todos=PLAN), "Working on the draft."])
    agent = Agent(model, instructions=instructions, tools=[todo_tools()])
    session = Session()
    result = await agent.run("Write the solar report", session=session)

    assert result.messages[2].tool_results[0].content == (
        "Saved 3 items: 1 completed, 1 in progress, 1 pending."
    )
    assert (await session.state()).values == {"todos": PLAN}
    assert model.requests[0].messages[0].text == "Write a report."
    assert model.requests[1].messages[0].text == (
        "Write a report.\n\n"
        "Current plan (update it with write_todos as you work):\n"
        "1. [completed] Collect the sources\n"
        "2. [in_progress] Draft the report\n"
        "3. [pending] Check the figures"
    )

    # The next run of the session reads the same plan back.
    model = ScriptedModel([call("read_todos"), "Still drafting."])
    later = await Agent(model, tools=[todo_tools()]).run("Where are we?", session=session)
    assert later.messages[-2].tool_results[0].content == render_todos({"todos": PLAN})


async def test_more_than_one_item_in_progress_is_sent_back_to_the_model():
    busy = [{"content": "a", "status": "in_progress"}, {"content": "b", "status": "in_progress"}]
    model = ScriptedModel([call("write_todos", todos=busy), "Fixed."])
    session = Session()
    result = await Agent(model, tools=[todo_tools()]).run("plan", session=session)
    refused = result.messages[2].tool_results[0]
    assert refused.is_error
    assert refused.content == prompts.TODOS_TOO_MANY_IN_PROGRESS.format(count=2)
    assert (await session.state()).values == {}


async def test_invalid_items_are_reported_as_argument_errors():
    model = ScriptedModel([call("write_todos", todos=[{"content": "a", "status": "doing"}]), "x"])
    result = await Agent(model, tools=[todo_tools()]).run("plan")
    error = result.messages[2].tool_results[0]
    assert error.is_error and "todos.0.status" in error.content


async def test_read_todos_without_a_plan_and_a_custom_key():
    write, read = todo_tools(key="plan")
    assert (write.name, read.name) == ("write_todos", "read_todos")
    assert read.read_only and not write.read_only and write.idempotent
    assert write.description == prompts.WRITE_TODOS_DESCRIPTION
    assert await read.run(RunContext(), {}) == prompts.TODOS_EMPTY

    ctx = RunContext()
    await write.run(ctx, {"todos": PLAN[:1]})
    assert ctx.state == {"plan": PLAN[:1]}
    assert render_todos(ctx.state) == ""  # the default key holds nothing
    assert render_todos(ctx.state, key="plan").endswith("1. [completed] Collect the sources")


def test_the_schema_tells_the_model_the_statuses():
    write, _ = todo_tools()
    item = write.parameters["$defs"]["TodoItem"]
    assert item["properties"]["status"]["enum"] == ["pending", "in_progress", "completed"]
    assert item["required"] == ["content", "status"]
    assert write.parameters["required"] == ["todos"]


def test_reading_a_state_that_is_not_a_todo_list_fails_clearly():
    assert read_todo_list({"todos": PLAN})[1] == TodoItem(
        content="Draft the report", status="in_progress"
    )
    with pytest.raises(SessionError, match="not a todo list"):
        render_todos({"todos": "buy milk"})
