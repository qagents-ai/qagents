"""A small MCP server used by tests/test_mcp.py.

Run it as a script to serve over stdio (``python tests/mcp_fixture_server.py``), or import
``build_server()`` to serve it over streamable HTTP in-process. It has no network access and no
dependencies beyond the ``mcp`` SDK.

Environment variables change what it lists, so tests can simulate a server that changes its
tool definitions between connections:

* ``FIXTURE_ECHO_DESCRIPTION`` replaces the description of ``echo``.
* ``FIXTURE_PID_FILE``: the server writes its process id to this file when it starts.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

READ_ONLY = ToolAnnotations(read_only_hint=True)
DESTRUCTIVE = ToolAnnotations(read_only_hint=False, destructive_hint=True)
ADDITIVE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True)


def build_server() -> MCPServer:
    """Build the fixture server with every test tool, resource and prompt registered."""
    server = MCPServer("qagents-fixture", instructions="A server for qagents tests.")

    echo_description = os.environ.get("FIXTURE_ECHO_DESCRIPTION", "Repeat the text back.")

    @server.tool(description=echo_description, annotations=READ_ONLY)
    def echo(text: str) -> str:
        return text

    @server.tool(annotations=READ_ONLY)
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @server.tool(annotations=DESTRUCTIVE)
    def delete_item(id: str) -> str:  # noqa: A002 - the MCP argument is called "id"
        """Delete an item for good."""
        return f"deleted {id}"

    @server.tool(annotations=ADDITIVE)
    def tag_item(id: str, tag: str) -> str:  # noqa: A002
        """Add a tag to an item. Changes data, but never destroys it."""
        return f"tagged {id} with {tag}"

    @server.tool()
    def unannotated(note: str = "") -> str:
        """A tool without annotations: the MCP defaults say it may be destructive."""
        return f"noted {note}"

    @server.tool(annotations=READ_ONLY)
    def fail() -> str:
        """Always fails."""
        raise ToolError("the fixture tool failed on purpose")

    @server.tool(annotations=READ_ONLY)
    async def slow(seconds: float) -> str:
        """Sleep, then answer."""
        await asyncio.sleep(seconds)
        return f"slept {seconds}"

    @server.tool(annotations=READ_ONLY)
    async def count(n: int, ctx: Context) -> str:
        """Count to n, reporting progress at every step."""
        for step in range(1, n + 1):
            await ctx.report_progress(step, n, f"step {step}")
        return f"counted to {n}"

    @server.tool(annotations=READ_ONLY, structured_output=True)
    def structured(city: str = "Athens") -> dict[str, Any]:
        """Return a structured weather report."""
        return {"city": city, "temperature_c": 21, "conditions": ["sunny", "calm"]}

    @server.tool(annotations=READ_ONLY)
    def picture() -> list[Any]:
        """Return text, an image and an embedded resource."""
        from mcp_types import EmbeddedResource, ImageContent, TextContent, TextResourceContents

        return [
            TextContent(type="text", text="Here is the chart."),
            ImageContent(type="image", data="iVBORw0KGgo=", mime_type="image/png"),
            EmbeddedResource(
                type="resource",
                resource=TextResourceContents(
                    uri="memo://notes/1", mime_type="text/plain", text="Remember the milk."
                ),
            ),
        ]

    @server.resource("memo://notes/1", name="note-1", mime_type="text/plain")
    def note() -> str:
        """The first note."""
        return "Remember the milk."

    @server.prompt()
    def summarize(topic: str) -> str:
        """Ask for a short summary."""
        return f"Summarise {topic} in one sentence."

    return server


def main() -> None:
    pid_file = os.environ.get("FIXTURE_PID_FILE")
    if pid_file:
        with open(pid_file, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
    build_server().run("stdio")


if __name__ == "__main__":
    main()
    sys.exit(0)
