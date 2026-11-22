"""The qagents MCP tool server (`qagents.integrations.mcp_server`), driven by the official
MCP client.

Over stdio the server runs as ``python -I -m qagents.integrations.mcp_server --spec FILE``
with the tools of ``fixture_tools.py``. Over Streamable HTTP it runs in the test's own event
loop on 127.0.0.1, either on its own or bound to a running qagents run, where every call
is a tool call of that run. Nothing leaves the machine.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp", reason="the MCP tool server needs qagents[mcp]")

import httpx2  # noqa: E402 - installed with mcp
from mcp import Client, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402
from mcp.client.streamable_http import streamable_http_client  # noqa: E402

from qagents import (  # noqa: E402
    Agent,
    Approval,
    Hooks,
    InMemoryStore,
    Limits,
    Permissions,
    RunContext,
    Session,
    prompts,
    tool,
)
from qagents import prompts as texts  # noqa: E402
from qagents.errors import ConfigurationError  # noqa: E402
from qagents.events import (  # noqa: E402
    RunStarted,
    SubagentEvent,
    ToolCallFinished,
    ToolCallStarted,
)
from qagents.external import ScriptedExternalAgent  # noqa: E402
from qagents.integrations.mcp_server import (  # noqa: E402
    METADATA_KEY,
    SOURCE,
    ToolRules,
    ToolServer,
    ToolServerSpec,
    acp_stdio_config,
    collect_tools,
    stdio_command,
    tool_reference,
    tool_server,
)
from qagents.models.testing import ScriptedModel, call  # noqa: E402
from qagents.toolkits.mcp import MCPToolkit  # noqa: E402

from . import fixture_tools as fx  # noqa: E402

TESTS = Path(__file__).resolve().parents[1]
"""``tests/``: the directory that makes ``integrations.fixture_tools`` importable."""

SERVED = [fx.lookup_customer, fx.create_ticket, fx.delete_customer, fx.broken, fx.slow]


@contextlib.asynccontextmanager
async def mcp_client(
    url: str, headers: dict[str, str] | None, mode: str = "auto"
) -> AsyncIterator[Client]:
    """The official MCP client over Streamable HTTP, with ``headers`` on every request.

    ``mode="auto"`` negotiates protocol 2026-07-28 with this server; ``"legacy"`` uses the
    2025-11-25 handshake.
    """
    http = httpx2.AsyncClient(headers=headers, timeout=httpx2.Timeout(30.0, read=60.0))
    transport = streamable_http_client(url, http_client=http)
    async with http, Client(transport, mode=mode) as client:  # type: ignore[arg-type]
        yield client


@pytest.fixture(autouse=True)
def no_server_errors(caplog: pytest.LogCaptureFixture) -> Any:
    """Fail every test of this module in which the server stack (uvicorn, Starlette, the SDK,
    this module) logged an error. The MCP client's own logs do not count."""
    yield
    errors = [
        record
        for record in [*caplog.get_records("setup"), *caplog.get_records("call")]
        if record.levelno >= 40 and not record.name.startswith("mcp.client")
    ]
    assert not errors, [f"{record.name}: {record.getMessage()}" for record in errors]


def text_of(result: Any) -> str:
    return "\n".join(block.text for block in result.content if getattr(block, "text", None))


def refused(name: str, reason: str) -> str:
    return texts.TOOL_SERVER_REFUSED.format(name=name, reason=reason)


def served_entries(entries: list[Any]) -> list[Any]:
    return [entry for entry in entries if entry.data.get("source") == SOURCE]


async def eventually(check: Callable[[], Any], seconds: float = 10.0) -> bool:
    """Poll ``check`` (sync or async) until it is true or ``seconds`` pass."""
    for _ in range(int(seconds / 0.02)):
        value = check()
        if asyncio.iscoroutine(value):
            value = await value
        if value:
            return True
        await asyncio.sleep(0.02)
    return False


async def connection_refused(port: int) -> bool:
    """Whether nothing listens on 127.0.0.1:``port`` any more."""
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), 10)
    except ConnectionRefusedError:
        return True
    except (TimeoutError, OSError):
        return False
    writer.close()
    with contextlib.suppress(Exception):
        await writer.wait_closed()
    return False


# -- definitions and the spec ----------------------------------------------------------------


async def test_collect_tools_refuses_what_cannot_be_served() -> None:
    @tool(deferred=True)
    def wait_for_human(question: str) -> str:
        """Ask a human."""
        return ""

    with pytest.raises(ConfigurationError, match="deferred"):
        await collect_tools([wait_for_human])
    with pytest.raises(ConfigurationError, match="Two served tools"):
        await collect_tools([fx.lookup_customer, [fx.lookup_customer]])
    with pytest.raises(ConfigurationError, match="is not a tool"):
        await collect_tools([42])
    plain = await collect_tools([fx.not_a_tool])
    assert [item.name for item in plain] == ["not_a_tool"]


def test_tool_reference_names_module_level_tools() -> None:
    assert tool_reference(fx.lookup_customer) == "integrations.fixture_tools:lookup_customer"

    @tool
    def local_only(x: int) -> int:
        """Not importable from another process."""
        return x

    with pytest.raises(ConfigurationError, match="module-level"):
        tool_reference(local_only)


def test_spec_round_trip_and_validation(tmp_path: Path) -> None:
    spec = ToolServerSpec(
        tools=[tool_reference(fx.lookup_customer), "integrations.fixture_tools:TOOLS"],
        permissions=ToolRules(allow=["create_ticket"], deny=["delete_*"]),
        paths=[str(TESTS)],
    )
    path = spec.write(tmp_path / "spec.json")
    assert ToolServerSpec.load(path) == spec
    assert json.loads(path.read_text(encoding="utf-8"))["permissions"]["deny"] == ["delete_*"]

    bad = tmp_path / "bad.json"
    for document in (
        {"tools": ["not a reference"]},
        {"tools": ["a.b:c"], "paths": ["relative/dir"]},
        {"tools": ["a.b:c"], "unknown": 1},
    ):
        bad.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ConfigurationError, match="not valid"):
            ToolServerSpec.load(bad)

    command = stdio_command(path)
    assert command[1:5] == ["-I", "-m", "qagents.integrations.mcp_server", "--spec"]
    assert Path(command[0]).is_absolute() and command[5] == str(path)
    entry = acp_stdio_config(path, name="tools")
    assert entry == {"name": "tools", "command": command[0], "args": command[1:], "env": []}


async def test_tool_definitions_come_from_the_qagents_tools() -> None:
    async with tool_server(SERVED) as server, mcp_client(server.url, server.headers) as client:
        listed = {item.name: item for item in (await client.list_tools()).tools}
    assert list(listed) == [item.name for item in SERVED]
    lookup = listed["lookup_customer"]
    assert lookup.description == fx.lookup_customer.description
    assert lookup.input_schema == fx.lookup_customer.parameters
    assert lookup.annotations is not None and lookup.annotations.read_only_hint is True
    ticket = listed["create_ticket"]
    assert "ctx" not in ticket.input_schema["properties"]  # the RunContext stays hidden
    assert ticket.input_schema["properties"]["priority"]["enum"] == ["low", "high"]
    assert ticket.annotations is not None and ticket.annotations.read_only_hint is False


# -- Streamable HTTP, on its own -----------------------------------------------------------


async def test_http_calls_tools_and_maps_results_and_errors() -> None:
    async with tool_server(SERVED) as server, mcp_client(server.url, server.headers) as client:
        found = await client.call_tool("lookup_customer", {"email": "maria@example.com"})
        failed = await client.call_tool("broken", {})
        invalid = await client.call_tool("lookup_customer", {"mail": "x"})
        missing = await client.call_tool("lookup_customer", {"email": "nobody@example.com"})
        unknown = await client.call_tool("no_such_tool", {})

    assert not found.is_error
    assert json.loads(text_of(found)) == {
        "email": "maria@example.com",
        "name": "Maria",
        "plan": "pro",
    }
    assert failed.is_error and text_of(failed) == "the fixture tool failed on purpose"
    assert invalid.is_error and text_of(invalid).startswith(
        "Invalid arguments for 'lookup_customer'"
    )
    assert missing.is_error and text_of(missing) == "No customer has the email nobody@example.com."
    assert unknown.is_error
    assert text_of(unknown) == prompts.UNKNOWN_TOOL.format(
        name="no_such_tool", available=", ".join(item.name for item in SERVED)
    )


async def test_http_permission_refusals_are_mcp_error_results() -> None:
    rules = Permissions(deny=["delete_*"])
    async with tool_server(SERVED, permissions=rules) as server:
        async with mcp_client(server.url, server.headers) as client:
            deleted = await client.call_tool("delete_customer", {"email": "maria@example.com"})
            ticket = await client.call_tool("create_ticket", {"subject": "Broken login"})
    assert deleted.is_error
    assert text_of(deleted) == refused("delete_customer", "a deny rule matches this tool.")
    assert ticket.is_error  # requires_approval, and nobody can approve over MCP
    assert text_of(ticket) == refused("create_ticket", texts.TOOL_SERVER_NO_APPROVER)
    assert "maria@example.com" in fx.CUSTOMERS  # the denied call never ran

    allowed = Permissions(allow=["create_ticket"])
    async with tool_server(SERVED, permissions=allowed) as server:
        async with mcp_client(server.url, server.headers) as client:
            ticket = await client.call_tool("create_ticket", {"subject": "Broken login"})
    assert not ticket.is_error and text_of(ticket) == "Opened ticket 1: Broken login (low)"


async def test_an_approval_that_takes_too_long_refuses_the_call() -> None:
    async def never_answers(ctx: Any, tool: Any, call: Any, reason: Any) -> bool:
        await asyncio.sleep(30)
        return True

    rules = Permissions(on_ask=never_answers)
    async with tool_server(SERVED, permissions=rules, approval_timeout_s=0.2) as server:
        async with mcp_client(server.url, server.headers) as client:
            began = time.monotonic()
            ticket = await client.call_tool("create_ticket", {"subject": "Refund"})
    assert time.monotonic() - began < 10
    assert ticket.is_error
    reason = texts.TOOL_SERVER_APPROVAL_TIMEOUT.format(seconds=0.2)
    assert text_of(ticket) == refused("create_ticket", reason)


async def test_requests_without_the_bearer_token_are_refused() -> None:
    hello = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "probe", "version": "0"},
        },
    }
    accept = {"Accept": "application/json, text/event-stream"}
    async with tool_server(SERVED) as server:
        token = server.headers["Authorization"].removeprefix("Bearer ")
        async with httpx2.AsyncClient(timeout=10) as raw:
            missing = await raw.post(server.url, json=hello, headers=accept)
            wrong = await raw.post(
                server.url, json=hello, headers={**accept, "Authorization": "Bearer nope"}
            )
            other_scheme = await raw.post(
                server.url, json=hello, headers={**accept, "Authorization": f"Basic {token}"}
            )
            lower = await raw.post(
                server.url, json=hello, headers={**accept, "Authorization": f"bearer {token}"}
            )
        with pytest.raises(BaseException):  # noqa: B017 - the SDK wraps it in exception groups
            async with mcp_client(server.url, None) as client:
                await client.list_tools()
        async with mcp_client(server.url, server.headers) as client:  # still serving
            assert (await client.list_tools()).tools
    for response in (missing, wrong, other_scheme):
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
    assert lower.status_code == 200  # the scheme is case-insensitive (RFC 6750)


async def test_listens_on_loopback_only_with_a_new_port_and_token_each_time() -> None:
    async with tool_server(SERVED) as first, tool_server(SERVED) as second:
        for server in (first, second):
            assert server.url == f"http://127.0.0.1:{server.port}/mcp"
            assert server._socket is not None
            assert server._socket.getsockname() == ("127.0.0.1", server.port)
            assert server._socket.family == socket.AF_INET
        assert first.port != second.port
        assert first.headers != second.headers
        assert str(first.port) in repr(first) and first.headers["Authorization"] not in repr(first)
        config = first.acp_config()
        assert config["type"] == "http" and config["url"] == first.url
        assert config["headers"] == [
            {"name": "Authorization", "value": first.headers["Authorization"]}
        ]


@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_shutdown_cancels_running_calls_and_closes_the_port(mode: str) -> None:
    server = await tool_server(SERVED, shutdown_timeout_s=5).start()
    port = server.port
    async with mcp_client(server.url, server.headers, mode) as client:
        waiting = asyncio.create_task(client.call_tool("slow", {"seconds": 60}))
        dispatcher = server._dispatcher
        assert dispatcher is not None
        assert await eventually(lambda: dispatcher.inflight), "the call never reached the server"
        began = time.monotonic()
        await server.aclose()
        assert time.monotonic() - began < 5  # measured: about 0.2 s
        assert not dispatcher.inflight
        waiting.cancel()
        with contextlib.suppress(BaseException):
            await waiting
    assert server._owner is not None and server._owner.done()
    assert server._socket is not None and server._socket.fileno() == -1
    assert await connection_refused(port)
    await server.aclose()  # a second close is a no-op


@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_both_protocol_eras_list_and_call(mode: str) -> None:
    async with tool_server(SERVED) as server:
        async with mcp_client(server.url, server.headers, mode) as client:
            version = client.protocol_version
            names = [item.name for item in (await client.list_tools()).tools]
            found = await client.call_tool("lookup_customer", {"email": "maria@example.com"})
    assert version == ("2026-07-28" if mode == "auto" else "2025-11-25")
    assert "lookup_customer" in names and "Maria" in text_of(found)


async def test_a_qagents_agent_uses_the_served_tools_through_mcp_toolkit() -> None:
    async with tool_server(SERVED) as server:
        toolkit = MCPToolkit.http(server.url, headers=server.headers, approval="none")
        async with toolkit:
            model = ScriptedModel([call("lookup_customer", email="maria@example.com"), "Found."])
            result = await Agent(model, tools=[toolkit]).run("Who is maria@example.com?")
    assert result.output == "Found."
    answer = model.requests[1].messages[-1].tool_results[0]
    assert not answer.is_error and "Maria" in answer.content


def test_options_that_conflict_with_a_bound_run_are_refused() -> None:
    with pytest.raises(ConfigurationError, match="bound"):
        tool_server(SERVED, ctx=RunContext(), permissions=Permissions())


async def test_binding_needs_the_context_of_a_running_tool_call() -> None:
    with pytest.raises(ConfigurationError, match="running tool call"):
        await tool_server(SERVED, ctx=RunContext(tool_call_id="x")).start()


# -- Streamable HTTP, bound to a qagents run --------------------------------------------------

Script = Callable[[Client, dict[str, Any]], Awaitable[None]]


async def run_delegating(
    script: Script,
    *,
    permissions: Permissions | None = None,
    limits: Limits | None = None,
    hooks: list[Hooks] | None = None,
    served: list[Any] | None = None,
) -> tuple[Any, list[Any], list[Any], dict[str, Any]]:
    """Run a lead agent whose one tool serves ``served`` over MCP, bound to the lead's run,
    and lets ``script`` call them with the official client (as an external agent would)."""
    seen: dict[str, Any] = {}

    @tool
    async def delegate(ctx: RunContext[Any], task: str) -> str:
        """Hand the task to another agent."""
        seen["caller"] = ctx.tool_call_id
        seen["operation_id"] = ctx.operation_id
        async with tool_server(served or SERVED, ctx=ctx) as server:
            seen["server"] = server
            async with mcp_client(server.url, server.headers) as client:
                await script(client, seen)
        return "delegated"

    session = Session(InMemoryStore(), id="lead")
    seen["session"] = session
    agent = Agent(
        ScriptedModel([call("delegate", task="Handle the customer."), "Done."]),
        tools=[delegate],
        permissions=permissions,
        limits=limits,
        hooks=hooks or [],
    )
    stream = agent.stream("Help maria@example.com", session=session)
    events = [event async for event in stream]
    return stream.result, events, await session.entries(), seen


async def test_bound_calls_are_tool_calls_of_the_parent_run() -> None:
    async def script(client: Client, seen: dict[str, Any]) -> None:
        seen["lookup"] = await client.call_tool("lookup_customer", {"email": "maria@example.com"})
        seen["ticket"] = await client.call_tool("create_ticket", {"subject": "Refund"})
        seen["delete"] = await client.call_tool("delete_customer", {"email": "maria@example.com"})

    rules = Permissions(allow=["create_ticket"], deny=["delete_customer"])
    result, events, entries, seen = await run_delegating(script, permissions=rules)

    assert result.status == "completed" and result.output == "Done."
    assert not seen["lookup"].is_error and "Maria" in text_of(seen["lookup"])
    assert text_of(seen["ticket"]) == "Opened ticket 1: Refund (low)"
    assert seen["delete"].is_error
    assert text_of(seen["delete"]) == refused("delete_customer", "a deny rule matches this tool.")
    assert "maria@example.com" in fx.CUSTOMERS

    # Journaled in the parent session, as tool calls of the parent run, tagged as MCP calls.
    mine = served_entries(entries)
    run_ids = {entry.run_id for entry in entries if entry.kind == "run_started"}
    assert {entry.run_id for entry in mine} <= run_ids
    assert all(entry.data["caller_tool_call_id"] == seen["caller"] for entry in mine)
    assert all(entry.data["server"] == "qagents" for entry in mine)
    started = [entry.data["name"] for entry in mine if entry.kind == "tool_started"]
    assert started == ["lookup_customer", "create_ticket"]  # the denied call never started
    results = {
        entry.data["result"]["name"]: entry.data["result"]
        for entry in mine
        if entry.kind == "tool_result"
    }
    assert set(results) == {"lookup_customer", "create_ticket", "delete_customer"}
    assert results["delete_customer"]["is_error"] is True
    assert results["create_ticket"]["metadata"][METADATA_KEY] == {
        "server": "qagents",
        "caller_tool_call_id": seen["caller"],
    }
    # Each served call is journaled while the delegating call runs: start before, result after.
    kinds = [(entry.kind, entry.data.get("tool_call_id")) for entry in entries]
    first = kinds.index(("tool_started", seen["caller"]))
    assert all(entries.index(entry) > first for entry in mine)
    caller_result = next(
        index
        for index, entry in enumerate(entries)
        if entry.kind == "tool_result" and entry.data["result"]["tool_call_id"] == seen["caller"]
    )
    assert all(entries.index(entry) < caller_result for entry in mine)

    # ctx.state written by a served tool is the run's state, journaled with the run.
    assert (await seen["session"].state()).values["tickets"] == ["Refund"]
    states = [entry.data["values"] for entry in entries if entry.kind == "state_updated"]
    assert {"tickets": ["Refund"]} in states

    # The parent's stream announces them like the model's own calls.
    started_events = [event.call.name for event in events if isinstance(event, ToolCallStarted)]
    assert started_events == ["delegate", "lookup_customer", "create_ticket"]
    finished = {
        event.call.name: event.result for event in events if isinstance(event, ToolCallFinished)
    }
    assert finished["delete_customer"].is_error and not finished["lookup_customer"].is_error


async def test_bound_ask_is_answered_by_the_parent_on_ask() -> None:
    asked: list[tuple[str, dict[str, Any], str | None]] = []

    def on_ask(ctx: RunContext[Any], tool: Any, call: Any, reason: str | None) -> Approval:
        asked.append((call.name, dict(call.arguments), ctx.tool_call_id))
        if call.arguments.get("subject") == "Refund":
            return Approval(approved=True, arguments={"subject": "Refund", "priority": "high"})
        return Approval(approved=False, reason="not today")

    async def script(client: Client, seen: dict[str, Any]) -> None:
        seen["approved"] = await client.call_tool("create_ticket", {"subject": "Refund"})
        seen["declined"] = await client.call_tool("create_ticket", {"subject": "Spam"})

    result, _, entries, seen = await run_delegating(script, permissions=Permissions(on_ask=on_ask))
    assert result.status == "completed"
    assert [name for name, _, _ in asked] == ["create_ticket", "create_ticket"]
    assert all(call_id.startswith("mcp-") for _, _, call_id in asked)
    note = prompts.APPROVAL_ARGUMENTS_CHANGED.format(
        arguments=json.dumps({"subject": "Refund", "priority": "high"})
    )
    assert text_of(seen["approved"]) == f"{note}\nOpened ticket 1: Refund (high)"
    assert text_of(seen["declined"]) == refused("create_ticket", "not today")
    decisions = [
        entry.data for entry in served_entries(entries) if entry.kind == "approval_decided"
    ]
    assert [(item["approved"], item["arguments_changed"]) for item in decisions] == [
        (True, True),
        (False, False),
    ]
    order = [entry.kind for entry in served_entries(entries)]
    assert order.index("approval_decided") < order.index("tool_started")  # record first


async def test_bound_ask_without_on_ask_is_refused() -> None:
    async def script(client: Client, seen: dict[str, Any]) -> None:
        seen["ticket"] = await client.call_tool("create_ticket", {"subject": "Refund"})

    result, _, entries, seen = await run_delegating(script)
    assert result.status == "completed"  # the run did not pause: the call was refused
    assert seen["ticket"].is_error
    assert text_of(seen["ticket"]) == refused("create_ticket", texts.TOOL_SERVER_NO_APPROVER)
    assert not [entry for entry in served_entries(entries) if entry.kind == "tool_started"]
    assert "tickets" not in (await seen["session"].state()).values


async def test_bound_calls_count_toward_max_tool_calls() -> None:
    async def script(client: Client, seen: dict[str, Any]) -> None:
        seen["first"] = await client.call_tool("lookup_customer", {"email": "maria@example.com"})
        seen["second"] = await client.call_tool("lookup_customer", {"email": "maria@example.com"})

    _, _, _, seen = await run_delegating(script, limits=Limits(max_tool_calls=2))
    assert not seen["first"].is_error  # the delegating call is the first of two
    assert seen["second"].is_error
    assert text_of(seen["second"]) == prompts.TOOL_CALL_LIMIT.format(
        name="lookup_customer", limit=2
    )


async def test_bound_calls_go_through_the_run_hooks() -> None:
    class Redact(Hooks):
        def __init__(self) -> None:
            self.before: list[str] = []

        async def before_tool(self, ctx: RunContext[Any], call: Any) -> Any:
            self.before.append(call.name)
            return None

        async def after_tool(self, ctx: RunContext[Any], call: Any, result: Any) -> Any:
            if call.name == "lookup_customer":
                return result.model_copy(update={"content": "[redacted]"})
            return None

    hook = Redact()

    async def script(client: Client, seen: dict[str, Any]) -> None:
        seen["lookup"] = await client.call_tool("lookup_customer", {"email": "maria@example.com"})

    _, _, entries, seen = await run_delegating(script, hooks=[hook])
    assert hook.before == ["delegate", "lookup_customer"]
    assert text_of(seen["lookup"]) == "[redacted]"  # the caller sees what the hooks allow
    stored = [entry for entry in served_entries(entries) if entry.kind == "tool_result"]
    assert stored[0].data["result"]["content"] == "[redacted]"


async def test_closing_a_bound_server_records_the_running_call() -> None:
    async def script(client: Client, seen: dict[str, Any]) -> None:
        server: ToolServer = seen["server"]
        session: Session = seen["session"]
        waiting = asyncio.create_task(client.call_tool("slow", {"seconds": 60}))

        async def slow_started() -> bool:
            return any(
                entry.kind == "tool_started" and entry.data.get("name") == "slow"
                for entry in await session.entries()
            )

        assert await eventually(slow_started), "the slow call never started"
        await server.aclose()
        waiting.cancel()
        with contextlib.suppress(BaseException):
            await waiting

    result, events, entries, _ = await run_delegating(script)
    assert result.status == "completed"
    stored = [entry for entry in served_entries(entries) if entry.kind == "tool_result"]
    assert [item.data["result"]["content"] for item in stored] == [
        prompts.TOOL_CANCELLED.format(name="slow")  # read-only: safe to call again
    ]
    finished = [event for event in events if isinstance(event, ToolCallFinished)]
    assert [event.call.name for event in finished] == ["slow", "delegate"]


async def test_a_bound_server_refuses_calls_once_its_tool_call_ended() -> None:
    kept: list[ToolServer] = []

    @tool
    async def delegate(ctx: RunContext[Any], task: str) -> str:
        """Start a server and leave it running (a mistake the server must survive)."""
        kept.append(await tool_server(SERVED, ctx=ctx).start())
        return "started"

    session = Session(InMemoryStore(), id="late")
    agent = Agent(ScriptedModel([call("delegate", task="x"), "Done."]), tools=[delegate])
    await agent.run("Go", session=session)
    before = len(await session.entries())
    server = kept[0]
    try:
        async with mcp_client(server.url, server.headers) as client:
            late = await client.call_tool("lookup_customer", {"email": "maria@example.com"})
    finally:
        await server.aclose()
    assert late.is_error
    assert text_of(late) == refused("lookup_customer", texts.TOOL_SERVER_RUN_ENDED)
    assert len(await session.entries()) == before  # nothing written after the run


async def test_an_external_agent_turn_calls_qagents_tools_through_the_lead_run() -> None:
    """The Devvi shape: the external agent's turn is the lead's tool call, and the qagents
    tools it calls over MCP are tool calls of the lead's run, under the lead's rules."""
    outcome: dict[str, Any] = {}

    async def use_qagents_tools(turn: Any) -> None:
        async with tool_server(SERVED, ctx=turn.parent) as server:
            async with mcp_client(server.url, server.headers) as client:
                outcome["lookup"] = await client.call_tool(
                    "lookup_customer", {"email": "maria@example.com"}
                )
                outcome["delete"] = await client.call_tool(
                    "delete_customer", {"email": "maria@example.com"}
                )
        turn.text_delta("Looked the customer up; deleting was refused.")

    coder = ScriptedExternalAgent([[use_qagents_tools]], name="coder")
    session = Session(InMemoryStore(), id="devvi-shape")
    lead = Agent(
        ScriptedModel([call("coder", task="Clean up maria@example.com"), "Done."]),
        tools=[coder.as_tool()],
        permissions=Permissions(deny=["delete_*"]),
    )
    async with coder:
        result = await lead.run("Go", session=session)
    assert result.status == "completed"
    assert not outcome["lookup"].is_error
    assert text_of(outcome["delete"]) == refused(
        "delete_customer", "a deny rule matches this tool."
    )
    entries = await session.entries()
    coder_call = next(
        entry.data["tool_call_id"]
        for entry in entries
        if entry.kind == "tool_started" and entry.data["name"] == "coder"
    )
    mine = served_entries(entries)
    assert mine and all(entry.data["caller_tool_call_id"] == coder_call for entry in mine)
    assert [entry.data["name"] for entry in mine if entry.kind == "tool_started"] == [
        "lookup_customer"
    ]


# -- depth: a served call comes from one level below the calling tool call --------------------


@tool(read_only=True)
def depth_probe(ctx: RunContext[Any]) -> str:
    """Report the depth the call runs at."""
    return str(ctx.depth)


async def test_served_calls_run_at_the_depth_of_the_external_turn() -> None:
    """The agent that calls over MCP is the tool call's child: its calls run at
    ``ctx.depth + 1``, the depth of the external turn that makes them."""
    outcome: dict[str, Any] = {}

    async def probe(turn: Any) -> None:
        outcome["turn"] = turn.depth
        async with tool_server([depth_probe], ctx=turn.parent) as server:
            async with mcp_client(server.url, server.headers) as client:
                outcome["probe"] = await client.call_tool("depth_probe", {})
        turn.text_delta("Probed.")

    coder = ScriptedExternalAgent([[probe]], name="coder")
    lead = Agent(ScriptedModel([call("coder", task="Probe."), "Done."]), tools=[coder.as_tool()])
    async with coder:
        result = await lead.run("Go")
    assert result.status == "completed"
    assert outcome["turn"] == 1
    assert not outcome["probe"].is_error
    assert text_of(outcome["probe"]) == "1"  # not 0, the depth of the lead's tool call


async def test_max_depth_counts_the_hop_into_a_served_agent() -> None:
    """lead (0) -> the agent its tool runs (1) -> a served sub-agent (2), beyond max_depth=1."""
    model = ScriptedModel(["Found it."])
    researcher = Agent(model, name="researcher", description="Finds facts.")

    async def script(client: Client, seen: dict[str, Any]) -> None:
        seen["research"] = await client.call_tool("researcher", {"task": "Find it."})

    result, _, _, seen = await run_delegating(
        script, limits=Limits(max_depth=1), served=[researcher]
    )
    assert result.status == "completed"
    assert seen["research"].is_error
    assert text_of(seen["research"]) == prompts.SUBAGENT_TOO_DEEP.format(
        name="researcher", depth=2, limit=1
    )
    assert not model.requests  # the served agent never started


async def test_a_served_agent_within_max_depth_runs_two_levels_below_the_lead() -> None:
    researcher = Agent(ScriptedModel(["Found it."]), name="researcher", description="Finds.")

    async def script(client: Client, seen: dict[str, Any]) -> None:
        seen["research"] = await client.call_tool("researcher", {"task": "Find it."})

    _, events, _, seen = await run_delegating(
        script, limits=Limits(max_depth=2), served=[researcher]
    )
    assert not seen["research"].is_error and text_of(seen["research"]) == "Found it."
    started = [
        event.event
        for event in events
        if isinstance(event, SubagentEvent) and isinstance(event.event, RunStarted)
    ]
    assert [event.depth for event in started] == [2]


async def test_max_depth_counts_the_hop_into_a_served_external_agent() -> None:
    helper = ScriptedExternalAgent([["Helped."]], name="helper")

    async def script(client: Client, seen: dict[str, Any]) -> None:
        seen["help"] = await client.call_tool("helper", {"task": "Help."})

    async with helper:
        result, _, _, seen = await run_delegating(
            script, limits=Limits(max_depth=1), served=[helper.as_tool()]
        )
    assert result.status == "completed"
    assert seen["help"].is_error
    assert text_of(seen["help"]) == texts.EXTERNAL_AGENT_TOO_DEEP.format(
        name="helper", depth=2, limit=1
    )
    assert not helper.calls  # the served external agent never got a turn


# -- concurrency, budgets and shutdown -------------------------------------------------------


async def test_concurrent_calls_never_pass_max_tool_calls() -> None:
    ran: list[str] = []
    asked: list[str] = []

    @tool(requires_approval=True)
    def write_note(text: str) -> str:
        """Write a note."""
        ran.append(text)
        return f"wrote {text}"

    async def approve(ctx: Any, tool: Any, call: Any, reason: Any) -> bool:
        asked.append(call.arguments["text"])
        await asyncio.sleep(0.3)  # the other calls arrive while this one is decided
        return True

    rules = Permissions(on_ask=approve)
    async with tool_server(
        [write_note], permissions=rules, limits=Limits(max_tool_calls=1)
    ) as server:
        async with mcp_client(server.url, server.headers) as client:
            results = await asyncio.gather(
                *(client.call_tool("write_note", {"text": f"n{i}"}) for i in range(3))
            )
    assert len(ran) == 1 and len(asked) == 1  # nobody approves a call that could not run
    limit = prompts.TOOL_CALL_LIMIT.format(name="write_note", limit=1)
    assert sorted(text_of(result) for result in results if result.is_error) == [limit, limit]
    assert [text_of(result) for result in results if not result.is_error] == [f"wrote {ran[0]}"]


async def test_concurrent_bound_calls_never_pass_the_run_max_tool_calls() -> None:
    asked: list[str] = []

    async def on_ask(ctx: Any, tool: Any, call: Any, reason: Any) -> bool:
        asked.append(call.arguments["subject"])
        await asyncio.sleep(0.3)
        return True

    async def script(client: Client, seen: dict[str, Any]) -> None:
        seen["results"] = await asyncio.gather(
            *(client.call_tool("create_ticket", {"subject": f"t{i}"}) for i in range(3))
        )

    result, _, entries, seen = await run_delegating(
        script, permissions=Permissions(on_ask=on_ask), limits=Limits(max_tool_calls=2)
    )
    assert result.status == "completed"
    assert len(asked) == 1  # the delegating call is the first of two
    limit = prompts.TOOL_CALL_LIMIT.format(name="create_ticket", limit=2)
    refusals = [text_of(item) for item in seen["results"] if item.is_error]
    assert refusals == [limit, limit]
    assert len((await seen["session"].state()).values["tickets"]) == 1
    mine = served_entries(entries)
    assert len([entry for entry in mine if entry.kind == "tool_started"]) == 1
    counted = [entry for entry in mine if entry.kind == "tool_result" and entry.data.get("counted")]
    assert len(counted) == 1  # refused calls do not count


def _blocking_tools(events: list[str], release: Any) -> tuple[Any, Any]:
    """``hang`` times out after 0.2 s and its thread runs on until ``release`` is set;
    ``write`` has no timeout of its own. Neither is read-only: their calls run alone."""

    @tool(timeout=0.2)
    def hang(text: str) -> str:
        """Write, and block until the test lets go."""
        events.append(f"start {text}")
        release.wait(10)
        events.append(f"end {text}")
        return text

    @tool
    def write(text: str) -> str:
        """Write at once."""
        events.append(f"start {text}")
        events.append(f"end {text}")
        return f"wrote {text}"

    return hang, write


async def test_a_call_that_runs_alone_waits_for_the_thread_of_a_timed_out_call() -> None:
    import threading

    events: list[str] = []
    release = threading.Event()
    hang, write = _blocking_tools(events, release)
    try:
        async with tool_server([hang, write]) as server:
            async with mcp_client(server.url, server.headers) as client:
                first = await client.call_tool("hang", {"text": "a"})
                assert text_of(first) == prompts.TOOL_TIMEOUT_ABANDONED.format(
                    name="hang", seconds=0.2
                )
                waiting = asyncio.create_task(client.call_tool("write", {"text": "b"}))
                await asyncio.sleep(0.4)
                assert events == ["start a"]  # b waits for a's thread: they never overlap
                release.set()
                second = await waiting
                assert text_of(second) == "wrote b"
                assert events == ["start a", "end a", "start b", "end b"]

                # A call whose own timeout passes while it waits is not run, and says so.
                release.clear()
                events.clear()
                await client.call_tool("hang", {"text": "c"})
                blocked = await client.call_tool("hang", {"text": "d"})
                assert blocked.is_error
                assert text_of(blocked) == prompts.TOOL_WAITED_FOR_EARLIER.format(
                    name="hang", seconds=0.2
                )
                assert events == ["start c"]
    finally:
        release.set()
    assert await eventually(lambda: "end c" in events)


async def test_a_call_that_runs_alone_waits_for_the_thread_of_a_cancelled_call() -> None:
    """A client can cancel one request (notifications/cancelled): a sync tool's thread then
    goes on, and the next call that runs alone waits for it, as for a timed-out call."""
    import threading

    from qagents.integrations.mcp_server import _LocalCalls

    events: list[str] = []
    release = threading.Event()

    @tool
    def hold(text: str) -> str:
        """Block until the test lets go (no timeout of its own)."""
        events.append(f"start {text}")
        release.wait(10)
        events.append(f"end {text}")
        return text

    _, write = _blocking_tools(events, release)
    calls = _LocalCalls(
        "local", permissions=Permissions(), deps=None, limits=Limits(), approval_timeout_s=None
    )
    try:
        first = asyncio.create_task(calls.call(hold, {"text": "a"}))
        assert await eventually(lambda: events == ["start a"])
        first.cancel()  # what the dispatcher does for a cancelled request
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(calls.call(write, {"text": "b"}))
        await asyncio.sleep(0.3)
        assert events == ["start a"]  # b waits for a's thread: they never overlap
        release.set()
        assert (await second).content == "wrote b"
        assert events == ["start a", "end a", "start b", "end b"]
    finally:
        release.set()


async def test_the_run_own_calls_wait_for_the_thread_of_a_timed_out_served_call() -> None:
    import threading

    events: list[str] = []
    release = threading.Event()
    hang, write = _blocking_tools(events, release)
    seen: dict[str, Any] = {}

    @tool
    async def delegate(ctx: RunContext[Any], task: str) -> str:
        """Hand the task to another agent."""
        async with tool_server([hang], ctx=ctx) as server:
            async with mcp_client(server.url, server.headers) as client:
                seen["hang"] = await client.call_tool("hang", {"text": "a"})
        return "delegated"

    async def let_go_later() -> None:
        await asyncio.sleep(0.4)  # long after the model asked for write b
        seen["before release"] = list(events)
        release.set()

    model = ScriptedModel([call("delegate", task="x"), call("write", text="b"), "Done."])
    agent = Agent(model, tools=[delegate, write])
    timer: asyncio.Task[None] | None = None
    try:
        stream = agent.stream("Go", session=Session(InMemoryStore(), id="stuck"))
        async for event in stream:
            if isinstance(event, ToolCallFinished) and event.call.name == "delegate":
                timer = asyncio.create_task(let_go_later())
    finally:
        release.set()
        if timer is not None:
            await timer
    assert text_of(seen["hang"]) == prompts.TOOL_TIMEOUT_ABANDONED.format(name="hang", seconds=0.2)
    assert seen["before release"] == ["start a"]  # the run's write waited for a's thread
    assert events == ["start a", "end a", "start b", "end b"]
    assert stream.result.status == "completed"


async def test_a_bound_call_does_not_run_once_its_tool_call_ended_during_before_tool() -> None:
    ran: list[str] = []
    entered = asyncio.Event()
    go_on = asyncio.Event()
    seen: dict[str, Any] = {}

    @tool(read_only=True)
    def probe(text: str) -> str:
        """Record that it ran."""
        ran.append(text)
        return text

    class SlowBefore(Hooks):
        async def before_tool(self, ctx: RunContext[Any], call: Any) -> Any:
            if call.name == "probe":
                entered.set()
                await go_on.wait()
            return None

    async def client_job(server: ToolServer) -> Any:
        async with mcp_client(server.url, server.headers) as client:
            return await client.call_tool("probe", {"text": "late"})

    @tool
    async def delegate(ctx: RunContext[Any], task: str) -> str:
        """Start a server by hand, and end while a served call waits in before_tool."""
        server = await tool_server([probe], ctx=ctx).start()
        seen["server"] = server
        seen["job"] = asyncio.create_task(client_job(server))
        await entered.wait()
        return "started"

    @tool
    async def release(ctx: RunContext[Any]) -> str:
        """Let the served call go on, after the delegating call ended."""
        go_on.set()
        seen["answer"] = await seen["job"]
        return "released"

    session = Session(InMemoryStore(), id="before-tool")
    model = ScriptedModel([call("delegate", task="x"), call("release"), "Done."])
    agent = Agent(model, tools=[delegate, release], hooks=[SlowBefore()])
    try:
        result = await agent.run("Go", session=session)
    finally:
        go_on.set()
        await seen["server"].aclose()
    assert result.status == "completed"
    assert ran == []  # the tool never ran
    assert text_of(seen["answer"]) == refused("probe", texts.TOOL_SERVER_RUN_ENDED)
    mine = served_entries(await session.entries())
    assert [entry.kind for entry in mine] == ["tool_started", "tool_result"]
    stored = mine[1].data["result"]
    assert stored["is_error"] and stored["content"] == refused("probe", texts.TOOL_SERVER_RUN_ENDED)


async def test_a_cancelled_aclose_still_stops_the_server() -> None:
    @tool(read_only=True)
    async def stubborn() -> str:
        """Wait, and take a while to stop when cancelled."""
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.sleep(0.5)
            raise
        return "done"

    server = await tool_server([stubborn], shutdown_timeout_s=5).start()
    port = server.port
    async with mcp_client(server.url, server.headers) as client:
        waiting = asyncio.create_task(client.call_tool("stubborn", {}))
        dispatcher = server._dispatcher
        assert dispatcher is not None
        assert await eventually(lambda: dispatcher.inflight), "the call never reached the server"
        closing = asyncio.create_task(server.aclose())
        await asyncio.sleep(0.1)  # aclose waits for the stubborn call to stop
        closing.cancel()  # the caller gives up (a deadline)
        with contextlib.suppress(asyncio.CancelledError):
            await closing
        began = time.monotonic()
        await server.aclose()  # waits for the shutdown that is still going on
        assert time.monotonic() - began < 10
        waiting.cancel()
        with contextlib.suppress(BaseException):
            await waiting
    assert server._owner is not None and server._owner.done()
    assert server._socket is not None and server._socket.fileno() == -1
    assert await connection_refused(port)
    assert not server.running


# -- stdio, in a separate process --------------------------------------------------------------


def write_spec(tmp_path: Path, **changes: Any) -> Path:
    spec = ToolServerSpec(
        name="fixture",
        tools=[
            "integrations.fixture_tools:TOOLS",
            tool_reference(fx.record_call),
        ],
        permissions=ToolRules(allow=["create_ticket", "record_call"], deny=["delete_*"]),
        paths=[str(TESTS)],
        deps="integrations.fixture_tools:make_deps",
    )
    return spec.model_copy(update=changes).write(tmp_path / "spec.json")


async def test_stdio_lists_and_calls_tools_with_the_official_client(tmp_path: Path) -> None:
    spec = write_spec(tmp_path)
    log = tmp_path / "calls.log"
    command = stdio_command(spec)
    parameters = StdioServerParameters(
        command=command[0], args=command[1:], env={"FIXTURE_TOOLS_LOG": str(log)}
    )
    errors = tmp_path / "stderr.txt"
    with errors.open("w", encoding="utf-8") as errlog:
        async with Client(stdio_client(parameters, errlog=errlog)) as client:
            names = [item.name for item in (await client.list_tools()).tools]
            found = await client.call_tool("lookup_customer", {"email": "maria@example.com"})
            ticket = await client.call_tool("create_ticket", {"subject": "Refund"})
            deleted = await client.call_tool("delete_customer", {"email": "maria@example.com"})
            failed = await client.call_tool("broken", {})
            noted = await client.call_tool("record_call", {"note": "hello"})
    assert names == [*(item.name for item in fx.TOOLS), "record_call"]
    assert json.loads(text_of(found))["name"] == "Maria"
    assert text_of(ticket) == "Opened ticket 1: Refund (low)"
    assert deleted.is_error
    assert text_of(deleted) == refused("delete_customer", "a deny rule matches this tool.")
    assert failed.is_error and text_of(failed) == "the fixture tool failed on purpose"
    assert text_of(noted) == "recorded hello"
    assert log.read_text(encoding="utf-8").strip().endswith(" hello")  # deps reached the tool
    assert "Traceback" not in errors.read_text(encoding="utf-8")


def _rpc(message: dict[str, Any]) -> bytes:
    return (json.dumps(message) + "\n").encode("utf-8")


async def test_stdio_server_exits_cleanly_when_stdin_closes(tmp_path: Path) -> None:
    spec = write_spec(tmp_path)
    process = await asyncio.create_subprocess_exec(
        *stdio_command(spec),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None
    try:
        process.stdin.write(
            _rpc(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {},
                        "clientInfo": {"name": "probe", "version": "0"},
                    },
                }
            )
        )
        process.stdin.write(_rpc({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        process.stdin.write(_rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}))
        await process.stdin.drain()
        answers: dict[int, Any] = {}
        while len(answers) < 2:
            line = await asyncio.wait_for(process.stdout.readline(), 30)
            assert line, "the server closed stdout early"
            message = json.loads(line)
            if "id" in message:
                answers[message["id"]] = message
        assert answers[1]["result"]["serverInfo"]["name"] == "fixture"
        assert "lookup_customer" in [item["name"] for item in answers[2]["result"]["tools"]]
        process.stdin.close()
        _, stderr = await asyncio.wait_for(process.communicate(), 30)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == 0
    assert b"Traceback" not in stderr


def test_stdio_refuses_a_spec_that_names_something_that_is_not_a_tool(tmp_path: Path) -> None:
    spec = write_spec(tmp_path, tools=["integrations.fixture_tools:NOT_SERVABLE"])
    completed = subprocess.run(
        stdio_command(spec), stdin=subprocess.DEVNULL, capture_output=True, timeout=60
    )
    assert completed.returncode == 2
    assert b"is not a tool" in completed.stderr
    missing = subprocess.run(
        [sys.executable, "-I", "-m", "qagents.integrations.mcp_server", "--spec", "nope.json"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=60,
        cwd=tmp_path,
    )
    assert missing.returncode == 2 and b"Cannot read the tool server spec" in missing.stderr
