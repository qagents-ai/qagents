"""A todo list the agent keeps for itself, for long multi-step work.

The list lives in the session state (``ctx.state["todos"]`` by default), so it
survives restarts and resumes and stays out of the transcript. Show it to the
model on every turn from an instructions function, so a long-running agent
keeps sight of its plan even after the conversation was compacted:

    from qagents.toolkits.planning import render_todos, todo_tools

    def instructions(ctx: RunContext) -> str:
        return "Research the topic and write a report.\\n\\n" + render_todos(ctx.state)

    agent = Agent("qcloud:<model-id>", instructions=instructions, tools=[todo_tools()])

The tool and parameter descriptions and the rendering are in `qagents.prompts`
(``WRITE_TODOS_DESCRIPTION``, ``WRITE_TODOS_TODOS_DESCRIPTION``,
``TODO_ITEM_DESCRIPTION``, ``TODO_CONTENT_DESCRIPTION``,
``TODO_STATUS_DESCRIPTION``, ``READ_TODOS_DESCRIPTION``, ``TODOS_HEADER``,
``TODO_LINE``, ``TODOS_EMPTY``, ``TODOS_SAVED``,
``TODOS_TOO_MANY_IN_PROGRESS``); they are read when the tools are made or used.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, TypeAdapter, ValidationError

from .. import prompts
from ..errors import SessionError, ToolError
from ..tools import RunContext, Tool, tool

TodoStatus = Literal["pending", "in_progress", "completed"]


class TodoItem(BaseModel):
    """One item of a todo list: ``content`` and ``status``.

    The descriptions the model sees are in `qagents.prompts`.
    """

    content: str
    status: TodoStatus


_TODO_LIST = TypeAdapter(list[TodoItem])


def todo_tools(key: str = "todos") -> list[Tool]:
    """Make the ``write_todos`` and ``read_todos`` tools.

    Args:
        key: The ``ctx.state`` key that holds the list, as a list of
            ``{"content": ..., "status": ...}`` dicts. Use another key to give
            two agents of one session separate lists.

    Returns:
        ``[write_todos, read_todos]``. ``write_todos`` replaces the whole list
        and is idempotent; ``read_todos`` is read-only.
    """

    @tool(idempotent=True, description=prompts.WRITE_TODOS_DESCRIPTION)
    def write_todos(ctx: RunContext[Any], todos: list[TodoItem]) -> str:
        """Replace the todo list."""
        counts = {status: 0 for status in ("pending", "in_progress", "completed")}
        for item in todos:
            counts[item.status] += 1
        if counts["in_progress"] > 1:
            raise ToolError(prompts.TODOS_TOO_MANY_IN_PROGRESS.format(count=counts["in_progress"]))
        ctx.state[key] = [item.model_dump(mode="json") for item in todos]
        return prompts.TODOS_SAVED.format(total=len(todos), **counts)

    parameters = write_todos.parameters
    parameters["properties"]["todos"]["description"] = prompts.WRITE_TODOS_TODOS_DESCRIPTION
    item = parameters["$defs"]["TodoItem"]
    item["description"] = prompts.TODO_ITEM_DESCRIPTION
    item["properties"]["content"]["description"] = prompts.TODO_CONTENT_DESCRIPTION
    item["properties"]["status"]["description"] = prompts.TODO_STATUS_DESCRIPTION

    @tool(read_only=True, description=prompts.READ_TODOS_DESCRIPTION)
    def read_todos(ctx: RunContext[Any]) -> str:
        """Read the todo list."""
        return render_todos(ctx.state, key) or prompts.TODOS_EMPTY

    return [write_todos, read_todos]


def read_todo_list(state: Mapping[str, Any], key: str = "todos") -> list[TodoItem]:
    """The todo list stored under ``key``, validated.

    Args:
        state: The session state: ``ctx.state``, or ``SessionState.values``.
        key: The key given to `todo_tools`.

    Returns:
        The items, or an empty list when there is none yet.

    Raises:
        SessionError: The key holds something that is not a todo list.
    """
    raw = state.get(key)
    if raw is None:
        return []
    try:
        return _TODO_LIST.validate_python(raw)
    except ValidationError as error:
        raise SessionError(
            f"The session state under {key!r} is not a todo list: {error}. "
            "Use another key for todo_tools(), or remove the value."
        ) from error


def render_todos(state: Mapping[str, Any], key: str = "todos") -> str:
    """The todo list as text for the model, or ``""`` when there is none.

    Meant for an instructions function, so the agent sees its plan on every turn:

        1. [completed] Collect the sources
        2. [in_progress] Draft the report
        3. [pending] Check the figures

    Args:
        state: The session state: ``ctx.state``, or ``SessionState.values``.
        key: The key given to `todo_tools`.
    """
    items = read_todo_list(state, key)
    if not items:
        return ""
    lines = [
        prompts.TODO_LINE.format(number=number, status=item.status, content=item.content)
        for number, item in enumerate(items, start=1)
    ]
    return "\n".join([prompts.TODOS_HEADER, *lines])


__all__ = ["TodoItem", "TodoStatus", "read_todo_list", "render_todos", "todo_tools"]
