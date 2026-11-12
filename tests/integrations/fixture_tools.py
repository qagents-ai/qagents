"""qagents tools that the MCP tool server serves in tests.

They are module-level names of an importable module, so a spec file can name them as
``"integrations.fixture_tools:<name>"`` when ``tests/`` is listed in the spec's ``paths``
(the stdio server runs with ``python -I``, which puts neither the current directory nor
``PYTHONPATH`` on ``sys.path``). The same objects are served in process over HTTP.

Nothing here touches the network or the file system, except ``record_call``, which
appends a line to the file named by ``deps["log"]`` (a path the test chose).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Literal

from qagents import RunContext, ToolError, tool

CUSTOMERS = {"maria@example.com": {"name": "Maria", "plan": "pro"}}


@tool(read_only=True)
def lookup_customer(email: str) -> dict[str, Any]:
    """Find a customer by email address.

    Args:
        email: The customer's email address.
    """
    found = CUSTOMERS.get(email)
    if found is None:
        raise ToolError(f"No customer has the email {email}.")
    return {"email": email, **found}


@tool(requires_approval=True)
def create_ticket(
    ctx: RunContext[Any], subject: str, priority: Literal["low", "high"] = "low"
) -> str:
    """Open a support ticket.

    Args:
        subject: One line that says what is wrong.
        priority: How urgent it is.
    """
    ctx.state["tickets"] = [*ctx.state.get("tickets", []), subject]
    return f"Opened ticket {len(ctx.state['tickets'])}: {subject} ({priority})"


@tool
def delete_customer(email: str) -> str:
    """Delete a customer for good."""
    CUSTOMERS.pop(email, None)
    return f"Deleted {email}"


@tool(read_only=True)
def broken() -> str:
    """Always fails."""
    raise ToolError("the fixture tool failed on purpose")


@tool(read_only=True)
async def slow(seconds: float) -> str:
    """Wait, then answer.

    Args:
        seconds: How long to wait.
    """
    await asyncio.sleep(seconds)
    return f"waited {seconds:g} s"


@tool
def record_call(ctx: RunContext[Any], note: str) -> str:
    """Append a note to the log file given in the deps."""
    log = Path(ctx.deps["log"])
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"{ctx.operation_id} {note}\n")
    return f"recorded {note}"


def make_deps() -> dict[str, Any]:
    """Deps for the stdio server: read from the environment of the test."""
    import os

    return {"log": os.environ.get("FIXTURE_TOOLS_LOG", "fixture-tools.log")}


TOOLS = [lookup_customer, create_ticket, delete_customer, broken, slow]
"""Every tool above except `record_call`."""


def not_a_tool() -> None:
    """A plain function: served as a tool with default flags, like ``Agent(tools=[fn])``."""


NOT_SERVABLE = 42
"""Not a tool: a spec that names it is refused."""
