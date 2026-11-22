"""The MCP server, through the official MCP client: in process, and over stdio end to end."""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("mcp")

from mcp import Client, StdioServerParameters  # noqa: E402
from mcp.shared.exceptions import MCPError  # noqa: E402

from qagents.docs_mcp.server import build_server  # noqa: E402

from .conftest import ROOT  # noqa: E402

TOOLS = {
    "search_docs", "read_page", "list_pages", "get_api", "list_examples", "read_example",
    "check_code",
}  # fmt: skip
CALLS = {
    "search_docs": ({"query": "pause and resume", "limit": 3}, "# Search: 'pause and resume'"),
    "read_page": (
        {"path": "concepts/permissions.md", "section": "pause-and-resume"},
        "<!-- source: concepts/permissions.md#pause-and-resume",
    ),
    "list_pages": ({}, "# qagents "),
    "get_api": ({"symbol": "Agent.resume"}, "# qagents.Agent.resume (method)"),
    "list_examples": ({}, "# qagents "),
    "read_example": ({"name": "supervisor"}, "# Example supervisor: "),
    "check_code": (
        {"code": "from qagents import Agent\nAgent('x', system_prompt='hi')\n"},
        "# check_code: 1 problem(s)",
    ),
}


def text_of(result) -> str:
    assert not result.is_error, result
    (content,) = result.content
    return content.text


async def exercise(client: Client, version: str) -> None:
    """Everything a client can ask, with the answers it should get."""
    listed = (await client.list_tools()).tools
    assert {tool.name for tool in listed} == TOOLS
    for tool in listed:
        assert tool.description and len(tool.description) > 60
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.destructive_hint is False
        for name, schema in tool.input_schema.get("properties", {}).items():
            assert schema.get("description"), (tool.name, name)
    for name, (arguments, start) in CALLS.items():
        assert text_of(await client.call_tool(name, arguments)).startswith(start), name

    resources = (await client.list_resources()).resources
    uris = {str(resource.uri) for resource in resources}
    assert "qagents://llms.txt" in uris
    assert "qagents://docs/concepts/permissions.md" in uris
    assert {"qagents://docs/AGENTS.md", "qagents://docs/skills/qagents/SKILL.md"} <= uris
    skill = (await client.read_resource("qagents://docs/skills/qagents/SKILL.md")).contents[0]
    assert skill.text.startswith("---\nname: qagents\n")  # the file as it is, to copy
    templates = {
        t.uri_template for t in (await client.list_resource_templates()).resource_templates
    }
    assert templates == {"qagents://docs/{+path}", "qagents://examples/{+name}"}
    llms = (await client.read_resource("qagents://llms.txt")).contents[0].text
    assert llms.startswith("# QAgents\n\n> ")
    page = (await client.read_resource("qagents://docs/concepts/tools.md")).contents[0].text
    assert page.startswith("# Tools")
    example = (await client.read_resource("qagents://examples/patterns/supervisor")).contents[0]
    assert '"""A supervisor delegates' in example.text
    with pytest.raises(MCPError, match="There is no page"):
        await client.read_resource("qagents://docs/no-such-page.md")

    prompts = {prompt.name: prompt for prompt in (await client.list_prompts()).prompts}
    assert set(prompts) == {"build_agent", "review_code"}
    built = await client.get_prompt("build_agent", {"task": "a refund bot"})
    assert built.messages[0].content.text.startswith(f"Build this with qagents {version}: ")
    reviewed = await client.get_prompt("review_code", {"code": "from qagents import Tools\n"})
    text = reviewed.messages[0].content.text
    assert "`qagents` has no `Tools`" in text and "requires_approval" in text


async def test_in_process(service):
    async with Client(build_server(service)) as client:
        assert client.server_info.name == "qagents-docs"
        assert client.server_info.version == service.version
        assert f"installed qagents {service.version}" in client.instructions
        await exercise(client, service.version)


async def test_over_stdio_with_the_official_client(service):
    parameters = StdioServerParameters(
        command=sys.executable, args=["-m", "qagents.docs_mcp", "--docs-dir", str(ROOT)]
    )
    async with Client(parameters, read_timeout_seconds=60) as client:
        assert client.server_info.name == "qagents-docs"
        await exercise(client, service.version)
