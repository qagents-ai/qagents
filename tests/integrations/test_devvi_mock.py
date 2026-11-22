"""Tier 1 of the Devvi test plan (design 3.2): the real Devvi CLI against a local mock model.

No API key and no cost: Devvi's ``deepseek-official`` route talks to `mock_model_server`
(``DEEPSEEK_BASE_URL`` on ``127.0.0.1``, a dummy key), which scripts text and tool-use
answers. Everything else is real: Node, Devvi's ACP and SDK servers, its tools, its
sandbox and its session logs.

Opt-in (marker ``devvi_mock``), Windows PowerShell:

    $env:QAGENTS_DEVVI_BIN = "<devvi-harness>\\apps\\cli\\lib\\bin.js"
    .venv\\Scripts\\python.exe -m pytest -m devvi_mock tests/integrations -q

* ``QAGENTS_DEVVI_BIN`` (required, else the tests skip): Devvi's built CLI.
* ``QAGENTS_NODE`` (optional): the Node to use; default ``node`` on PATH. Devvi needs
  Node >= 24.9.0. With an older Node the tests start the CLI through
  ``devvi_node_compat.mjs`` (a module hook for ``import.meta.main``, which Node 24.2
  added) with ``DevviAgent(command=...)``, and a warning says so.
* ``QAGENTS_DEVVI_TMP`` (optional): a directory for the tests' Devvi homes and
  workspaces; default pytest's ``tmp_path``. Each test gets its own Devvi home,
  workspace, user profile and temp folder (``USERPROFILE``, ``HOME``, ``APPDATA``,
  ``LOCALAPPDATA``, ``TEMP``, ``TMP``), so nothing reads or writes the user's own Devvi
  home or configuration.

Every test starts one or two Devvi processes (3 to 10 seconds each on Windows). On a
Windows machine where Devvi's restricted-token sandbox cannot use the workspace, the two
``workspace-write`` shell tests are reported as xfail with Devvi's own message.
"""

from __future__ import annotations

import asyncio
import gc
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from qagents import Agent, Limits, Permissions, Session, SQLiteStore
from qagents import prompts as devvi_texts
from qagents import prompts as texts
from qagents.events import SubagentEvent
from qagents.external import ExternalAgentConfigurationError, ExternalApproval
from qagents.integrations._jsonrpc import JsonRpcError, JsonRpcPeer
from qagents.integrations._process import ManagedProcess
from qagents.integrations.acp import ACPAgent
from qagents.integrations.devvi import DevviAgent, TokenPricing, logs
from qagents.integrations.devvi.__main__ import doctor
from qagents.integrations.devvi.sdk import BASELINE_CHECKPOINT, MESSAGE_CHECKPOINT
from qagents.models.testing import FunctionModel, ScriptedModel, call

from .mock_model_server import (
    IS_WINDOWS,
    MOCK_MODEL,
    MOCK_ROUTE,
    SHELL,
    Hold,
    MockModel,
    MockRequest,
    Text,
    ToolUse,
    Usage,
    escalated_write_command,
    message_text,
    mock_models_row,
    shell_arguments,
    sleep_command,
    write_file_command,
)

pytestmark = pytest.mark.devvi_mock

HERE = Path(__file__).resolve().parent
COMPAT = HERE / "devvi_node_compat.mjs"
NODE_MIN = (24, 9, 0)
LIMIT_S = 300.0
"""Upper bound of one test's work (a Devvi start takes 3 to 10 seconds)."""

SHELL_NAME = "pwsh" if IS_WINDOWS else "bash"
"""Devvi's shell tool on this platform (``packages/bundle/base``: tool-pwsh on Windows)."""

SANDBOX_DENIED = "[sandbox: file access denied"
"""How Devvi's shell tool reports a write its file sandbox refused."""

DENIED = (SANDBOX_DENIED, "PermissionDenied", "UnauthorizedAccess", "Permission denied")
"""Signs of a refused write in a shell tool's output (Devvi's, PowerShell's, bash's)."""


# --- the Devvi CLI, homes and workspaces ----------------------------------------------------------


@dataclass(frozen=True)
class DevviCli:
    bin: Path
    node: str
    node_version: tuple[int, int, int]

    @property
    def compat(self) -> bool:
        """Node is older than Devvi's engines.node: start through the compat hook."""
        return self.node_version < NODE_MIN

    def options(self) -> dict[str, Any]:
        """How `DevviAgent` and `doctor` start this CLI."""
        if self.compat:
            return {"command": [self.node, "--import", COMPAT.as_uri(), str(self.bin)]}
        return {"devvi_bin": str(self.bin), "node": self.node}


@pytest.fixture(scope="session")
def devvi_cli() -> DevviCli:
    configured = os.environ.get("QAGENTS_DEVVI_BIN")
    if not configured:
        pytest.skip("set QAGENTS_DEVVI_BIN to Devvi's apps/cli/lib/bin.js")
    path = Path(configured)
    if not path.is_file():
        pytest.skip(f"QAGENTS_DEVVI_BIN={configured} does not exist (run `pnpm run build`)")
    node = os.environ.get("QAGENTS_NODE") or shutil.which("node")
    if not node:
        pytest.skip("Node.js is not on PATH (or set QAGENTS_NODE)")
    out = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=60)
    match = re.search(r"v?(\d+)\.(\d+)\.(\d+)", out.stdout)
    if match is None:
        pytest.skip(f"cannot read the version of {node}: {out.stdout!r}")
    major, minor, patch = (int(part) for part in match.groups())
    cli = DevviCli(path.resolve(), node, (major, minor, patch))
    if cli.compat:
        warnings.warn(
            f"Node {major}.{minor}.{patch} is older than Devvi's engines.node (>= 24.9.0): the "
            "devvi_mock tests start Devvi through tests/integrations/devvi_node_compat.mjs "
            "(DevviAgent(command=...), which skips qagents' Node check).",
            stacklevel=1,
        )
    return cli


@dataclass(frozen=True)
class Place:
    """One test's directories: Devvi's home, the workspace, and a folder outside it."""

    root: Path
    home: Path
    workspace: Path
    outside: Path


@pytest.fixture
def place(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Place]:
    tempfile.gettempdir()  # cache this process's temp folder before TEMP changes below
    base = os.environ.get("QAGENTS_DEVVI_TMP")
    if base:
        Path(base).mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="dv-", dir=base))
    else:
        root = tmp_path
    user = root / "user"
    for sub in ("", "AppData/Roaming", "AppData/Local", "temp"):
        (user / sub).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USERPROFILE", str(user))
    monkeypatch.setenv("HOME", str(user))
    monkeypatch.setenv("APPDATA", str(user / "AppData" / "Roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(user / "AppData" / "Local"))
    # Devvi keeps spill files, ACL lock files and private sandbox temp folders in TEMP.
    monkeypatch.setenv("TEMP", str(user / "temp"))
    monkeypatch.setenv("TMP", str(user / "temp"))
    for name in ("DEVVI_HOME", "DSH_HOME", "QAGENTS_DEVVI_HOME"):
        monkeypatch.delenv(name, raising=False)
    found = Place(root, root / "home", root / "ws", root / "outside")
    for folder in (found.home, found.workspace, found.outside):
        folder.mkdir()
    yield found
    if base:
        gc.collect()  # close what still holds a file (a SQLite connection) before deleting
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def model() -> Iterator[MockModel]:
    with MockModel() as server:
        yield server
    assert server.errors == [], server.errors


def coder(cli: DevviCli, place: Place, model: MockModel, **options: Any) -> DevviAgent:
    """A `DevviAgent` named ``coder`` on the mock model, in the test's home and workspace."""
    settings: dict[str, Any] = {
        "workspace": place.workspace,
        "name": "coder",
        "home": place.home,
        "env": model.env(),
        "env_passthrough": (),  # nothing of this process's environment, no real key
        "model": MOCK_ROUTE,
        "patches": [mock_models_row()],
        "track_changes": False,
        **cli.options(),
    }
    settings.update(options)
    return DevviAgent(**settings)


async def run_turn(agent: DevviAgent, task: str, **options: Any) -> tuple[list[Any], Any]:
    async with agent.stream(task, **options) as turn:
        events = [event async for event in turn]
    assert turn.result is not None
    return events, turn.result


def kinds(events: list[Any]) -> list[str]:
    return [event.kind if event.type == "external_event" else event.type for event in events]


def inner(events: list[Any]) -> list[Any]:
    return [event.event for event in events if isinstance(event, SubagentEvent)]


def tool_results(result: Any) -> list[Any]:
    return [part for message in result.messages for part in message.tool_results]


def tool_output(model: MockModel, call_id: str) -> str:
    """What Devvi sent its model as the result of tool call ``call_id``."""
    for request in model.requests:
        for block in request.tool_results:
            if block.get("tool_use_id") == call_id:
                return message_text(block.get("content"))
    raise AssertionError(f"the mock model never got the result of {call_id}")


def xfail_if_sandbox_denied(output: str, permission_mode: str) -> None:
    if permission_mode == "workspace-write" and any(sign in output for sign in DENIED):
        lines = output.splitlines()
        first = next((line for line in lines if any(sign in line for sign in DENIED)), "")
        pytest.xfail(
            "Devvi's Windows sandbox (restricted token) cannot use this workspace on this "
            f"machine: {first.strip()} (likely the ACLs another sandbox tool set on an "
            "ancestor folder; Devvi's sandbox-windows-acl README). Not a qagents issue."
        )


def write_turn(model: MockModel, target: Path, call_id: str = "write-1") -> None:
    command = write_file_command(str(target), "hello from devvi")
    arguments = shell_arguments(command, "Write hello.txt")
    model.on_prompt("WRITE-FILE", ToolUse(SHELL, arguments, id=call_id))
    model.after_tool(call_id, Text("Wrote hello.txt."))


def escalation_turn(model: MockModel, target: Path, call_id: str = "esc-1") -> None:
    arguments = shell_arguments(
        escalated_write_command(str(target), "escalated"),
        "Write a file outside the workspace",
        sandbox_permissions="danger-full-access",
        justification="the user asked to write a file outside the workspace",
    )
    model.on_prompt("ESCALATE", ToolUse(SHELL, arguments, id=call_id))
    model.after_tool(
        call_id, lambda r: Text("DONE" if "escalated" in r.tool_result_text else "REFUSED")
    )


def rules_for(decision: str) -> Permissions:
    name = f"coder.{SHELL_NAME}"
    return Permissions(allow=[name]) if decision == "allow" else Permissions(deny=[name])


def recall(request: MockRequest) -> Text:
    known = "OSPREY" in request.transcript()
    return Text("The code word is OSPREY." if known else "I do not know the code word.")


def json_path(path: Path) -> str:
    """``path`` as Devvi writes it inside a JSON string in its runtime context."""
    return str(path).replace("\\", "\\\\")


def resume_marker() -> str:
    return texts.EXTERNAL_AGENT_RESUME[:40]


# --- both transports: the doctor's handshakes ----------------------------------------------------


async def test_doctor_runs_the_acp_and_sdk_handshakes(
    devvi_cli: DevviCli, place: Place, model: MockModel
) -> None:
    environ = {**os.environ, **model.env()}
    async with asyncio.timeout(LIMIT_S):
        checks = await doctor(
            home=place.home, workspace=place.workspace, environ=environ, **devvi_cli.options()
        )
    report = {check.name: check for check in checks}
    assert [check.line() for check in checks if check.status == "fail"] == []
    assert re.fullmatch(r"0\.2\.\d+(-[0-9A-Za-z.]+)?", report["Devvi"].detail)
    acp = report["ACP"].detail
    assert acp.startswith(
        "deepseek-harness-acp 0.0.1 (ACP v1): MCP over HTTP yes, image prompts no, "
        "session resume yes; ready in "
    )
    assert acp.endswith("shut down with exit code 0")
    assert "current deepseek-official/deepseek-v4-flash" in report["ACP models"].detail
    sdk = report["SDK"].detail
    assert sdk.startswith(
        "deepseek-harness-sdk-runtime 0.0.1 on route deepseek-official/deepseek-v4-flash"
    )
    assert sdk.endswith("shut down with exit code 0")
    assert model.requests == []  # a handshake asks the model nothing


# --- ACP ---------------------------------------------------------------------------------------


async def test_acp_handshake_and_a_text_turn(
    devvi_cli: DevviCli, place: Place, model: MockModel
) -> None:
    model.on_prompt("TEXT-TURN", Text("ACP text OK", thinking="The user wants the mock text."))
    agent = coder(devvi_cli, place, model)
    async with asyncio.timeout(LIMIT_S), agent:
        (connection,) = agent.connections.values()
        assert connection.agent_info == {"name": "deepseek-harness-acp", "version": "0.0.1"}
        capabilities = connection.agent_capabilities
        assert capabilities["mcpCapabilities"] == {"http": True}
        assert capabilities["promptCapabilities"]["image"] is False
        assert {"close", "list", "resume"} <= set(capabilities["sessionCapabilities"])
        offered = await agent.capabilities()
        assert (offered.token_usage, offered.cancel_turn, offered.resume_sessions) == (
            False,
            True,
            True,
        )
        events, result = await run_turn(agent, "TEXT-TURN: reply with the mock text.")
        options = connection.session_config(result.session_id)
    assert result.status == "completed" and result.stop_reason == "end_turn"
    assert result.text == "ACP text OK"
    assert uuid.UUID(result.session_id)  # Devvi chooses the session id on ACP
    assert re.fullmatch(r"devvi-acp@0\.2\.\d+.*", result.backend)
    assert result.usage.requests == 0  # ACP reports no token counts
    assert kinds(events)[0] == "run_started" and kinds(events)[-1] == "turn_finished"
    assert [e.text for e in events if e.type == "reasoning_delta"] == [
        "The user wants the mock text."
    ]
    assert [e.text for e in events if e.type == "text_delta"] == ["ACP text OK"]
    context = [e.data for e in events if e.type == "external_event" and e.kind == "context"]
    assert context and context[-1]["used"] > 0 and context[-1]["size"] > 0
    chosen = next(option for option in options if option.get("id") == "model")
    assert chosen["currentValue"] == '["deepseek-official","smoke-model"]'
    (request,) = model.requests_with("TEXT-TURN")
    assert request.body["model"] == MOCK_MODEL and request.body["stream"] is True
    assert SHELL_NAME in request.tools


@pytest.mark.parametrize("permission_mode", ["workspace-write", "danger-full-access"])
async def test_acp_shell_tool_writes_a_file_in_the_workspace(
    devvi_cli: DevviCli, place: Place, model: MockModel, permission_mode: str
) -> None:
    write_turn(model, place.workspace / "hello.txt")
    agent = coder(devvi_cli, place, model, permission_mode=permission_mode)
    lead = Agent(
        ScriptedModel([call("coder", task="WRITE-FILE: create hello.txt"), "Done."]),
        name="lead",
        tools=[agent.as_tool(timeout=LIMIT_S)],
    )
    async with asyncio.timeout(LIMIT_S), agent:
        stream = lead.stream("Go")
        events = [event async for event in stream]
    result = stream.result
    assert result is not None and result.output == "Done."
    xfail_if_sandbox_denied(tool_output(model, "write-1"), permission_mode)
    assert (place.workspace / "hello.txt").read_text(encoding="utf-8-sig") == "hello from devvi"
    forwarded = inner(events)
    started = [e for e in forwarded if e.type == "tool_call_started"]
    finished = [e for e in forwarded if e.type == "tool_call_finished"]
    assert [e.call.name for e in started] == [f"coder.{SHELL_NAME}"]
    assert started[0].call.id == "write-1"
    assert started[0].call.arguments["description"] == "Write hello.txt"
    assert finished[0].result.is_error is False
    content = tool_results(result)[0].content
    assert content.startswith("Wrote hello.txt.\n\n[coder: session ")
    assert content.endswith(f"; completed; 1 tool call ({SHELL_NAME} 1)]")


@pytest.mark.parametrize("decision", ["allow", "deny"])
async def test_acp_escalation_is_decided_by_qagents_rules(
    devvi_cli: DevviCli, place: Place, model: MockModel, decision: str
) -> None:
    target = place.outside / "escalated.txt"
    escalation_turn(model, target)
    agent = coder(devvi_cli, place, model, permissions=rules_for(decision))
    async with asyncio.timeout(LIMIT_S), agent:
        events, result = await run_turn(agent, "ESCALATE: write the file outside the workspace.")
    (permission,) = result.permissions
    assert permission.tool == f"coder.{SHELL_NAME}" and permission.external_tool == SHELL_NAME
    assert permission.tool_call_id == "esc-1"
    assert permission.decided_by == "rules" and permission.outcome == decision
    assert permission.arguments["sandbox_permissions"] == "danger-full-access"
    assert kinds(events).count("permission_requested") == 1
    assert kinds(events).count("permission_decided") == 1
    (called,) = result.tool_calls
    if decision == "allow":
        assert result.text == "DONE" and target.read_text(encoding="utf-8-sig") == "escalated"
        assert called.is_error is False
    else:
        assert result.text == "REFUSED" and not target.exists()
        assert "permission refused" in result.footer("coder")


async def test_acp_a_pause_grant_survives_a_reworded_escalation(
    devvi_cli: DevviCli, place: Place, model: MockModel
) -> None:
    """``on_permission="pause"``: the person approves the escalation; on the next call Devvi's
    model asks again for the same command with another description and justification, and
    the approval answers it (``grant_ignored_arguments``). A changed command pauses again."""
    target = place.outside / "escalated.txt"
    other = place.outside / "other.txt"

    def escalation(path: Path, text: str, description: str, justification: str) -> Any:
        return shell_arguments(
            escalated_write_command(str(path), text),
            description,
            sandbox_permissions="danger-full-access",
            justification=justification,
        )

    asked = escalation(target, "escalated", "Write a file outside the workspace", "asked for")
    reworded = escalation(target, "escalated", "Write escalated.txt", "the user wants it there")
    changed = escalation(other, "other", "Write other.txt", "the user wants it there")
    model.on_prompt("ESC-PAUSE", ToolUse(SHELL, asked, id="esc-1"))
    model.on_prompt("AGAIN", ToolUse(SHELL, reworded, id="esc-2"))
    model.after_tool("esc-2", ToolUse(SHELL, changed, id="esc-3"))
    model.after_tool("esc-3", Text("DONE"))

    def reply(request: Any) -> Any:
        last = request.messages[-1]
        if last.role != "tool":
            return call("coder", task="ESC-PAUSE: write the file outside the workspace.")
        found = re.search(r"session=([\w-]+)", last.tool_results[0].content)
        if found:
            return call("coder", task="AGAIN: write it now.", session=found.group(1))
        return "Done."

    agent = coder(devvi_cli, place, model, on_permission="pause")
    lead = Agent(FunctionModel(reply), name="lead", tools=[agent.as_tool(timeout=LIMIT_S)])
    async with asyncio.timeout(LIMIT_S), agent:
        paused = await lead.run("Write the file.", session=Session(id="pause"))
        assert paused.status == "paused", paused.output
        first = paused.pending_inputs[0]
        assert first.metadata["tool"] == f"coder.{SHELL_NAME}"
        assert first.metadata["arguments"]["description"] == "Write a file outside the workspace"
        assert not target.exists()  # nothing ran before the person answered
        approval = ExternalApproval.for_input(first, approved=True)
        again = await lead.resume(paused.session, results={first.tool_call_id: approval})
    assert target.read_text(encoding="utf-8-sig") == "escalated"  # the reworded request ran
    assert again.status == "paused"  # the changed command waits for a person again
    second = again.pending_inputs[0]
    assert second.metadata["arguments"]["command"] == changed["command"]
    assert not other.exists()


async def test_acp_cancel_during_a_tool_and_continue_the_session(
    devvi_cli: DevviCli, place: Place, model: MockModel
) -> None:
    model.on_prompt(
        "SLEEP", ToolUse(SHELL, shell_arguments(sleep_command(120), "Wait"), id="sleep-1")
    )
    model.after_tool("sleep-1", Text("The wait finished."))
    model.on_prompt("AFTER-CANCEL", Text("Continued after the cancel."))
    agent = coder(devvi_cli, place, model, cancel_grace_s=20)
    async with asyncio.timeout(LIMIT_S), agent:
        (connection,) = agent.connections.values()
        pid = connection.process.pid if connection.process else None
        began = time.monotonic()
        async with agent.stream("SLEEP: wait two minutes.") as turn:
            async for event in turn:
                if event.type == "tool_call_started":
                    await asyncio.sleep(1.0)  # let the command start
                    turn.cancel()
        elapsed = time.monotonic() - began
        first = turn.result
        assert first is not None
        _, second = await run_turn(agent, "AFTER-CANCEL: go on.", session=first.session_id)
        same = connection.process.pid if connection.process else None
    assert first.status == "cancelled" and first.stop_reason == "cancelled"
    assert not first.raw.get("forced")  # Devvi answered session/cancel within the grace period
    assert elapsed < 60
    assert [call.name for call in first.tool_calls] == [SHELL_NAME]
    assert second.status == "completed" and second.text == "Continued after the cancel."
    assert second.session_id == first.session_id
    assert pid is not None and same == pid  # the cancel kept the Devvi process
    assert all("The wait finished." not in r.transcript() for r in model.requests)


async def test_acp_a_second_agent_resumes_the_session(
    devvi_cli: DevviCli, place: Place, model: MockModel
) -> None:
    model.on_prompt("REMEMBER", Text("Noted."))
    model.on_prompt("RECALL", recall)
    async with asyncio.timeout(LIMIT_S):
        first_agent = coder(devvi_cli, place, model)
        async with first_agent:
            _, first = await run_turn(first_agent, "REMEMBER: the code word is OSPREY.")
        second_agent = coder(devvi_cli, place, model)
        async with second_agent:
            events, second = await run_turn(
                second_agent, "RECALL the code word.", session=first.session_id
            )
    assert first.text == "Noted."
    assert events[0].type == "run_started" and events[0].resumed is True
    assert second.session_id == first.session_id
    assert second.text == "The code word is OSPREY."


async def test_acp_exposed_qagents_tools_are_called_by_devvis_model_over_mcp(
    devvi_cli: DevviCli, place: Place, model: MockModel
) -> None:
    """``expose_tools``: an in-process qagents tool, served over MCP Streamable HTTP on
    127.0.0.1 with a bearer token, called by Devvi's model in two turns of one session (the
    session is closed after each turn and resumed with a new server)."""
    pytest.importorskip("mcp")
    from qagents.integrations.mcp_server import SOURCE

    from . import fixture_tools as fx

    lookup = ToolUse("mcp__qagents__lookup_customer", {"email": "maria@example.com"})
    for marker, call_id, label in (("LOOKUP", "mcp-1", "first"), ("AGAIN", "mcp-2", "second")):
        model.on_prompt(marker, ToolUse(lookup.name, lookup.arguments, id=call_id))
        model.after_tool(
            call_id, lambda r, label=label: Text(f"{label}: {r.tool_result_text.strip()}")
        )

    def lead_reply(request: Any) -> Any:
        last = request.messages[-1]
        if last.role != "tool":
            return call("coder", task="LOOKUP Maria.")
        content = last.tool_results[0].content
        if content.startswith("first:"):
            session = re.search(r"\[coder: session ([^;]+);", content)
            assert session is not None, content
            return call("coder", task="AGAIN, please.", session=session.group(1))
        return "Done."

    agent = coder(devvi_cli, place, model, expose_tools=[fx.lookup_customer])
    lead = Agent(
        FunctionModel(lead_reply),
        tools=[agent.as_tool(timeout=LIMIT_S)],
        limits=Limits(max_turns=6),
    )
    async with asyncio.timeout(LIMIT_S), agent, SQLiteStore(place.root / "runs.db") as store:
        session = Session(store, id="exposed")
        result = await lead.run("Go", session=session)
        entries = await session.entries()
    assert result.output == "Done.", tool_results(result)
    first, second = (part.content for part in tool_results(result))
    assert first.startswith('first: {"email": "maria@example.com", "name": "Maria"')
    assert second.startswith('second: {"email": "maria@example.com", "name": "Maria"')
    assert all("mcp__qagents__lookup_customer" in r.tools for r in model.requests)
    served = [e for e in entries if e.kind == "tool_started" and e.data.get("source") == SOURCE]
    assert [e.data["name"] for e in served] == ["lookup_customer"] * 2  # the lead run's journal
    assert "permission" not in first + second  # Devvi asked nothing for an MCP tool


@pytest.mark.parametrize("transport", ["acp", "sdk"])
async def test_a_missing_api_key_fails_closed(
    devvi_cli: DevviCli, place: Place, model: MockModel, transport: str
) -> None:
    """No ``DEEPSEEK_API_KEY``: ACP offers none of the route's declared models, so the model
    choice is refused; SDK starts, and the turn ends with Devvi's ``MISSING_CREDENTIAL``,
    which opens qagents' circuit breaker. The model is never asked."""
    env = {"DEEPSEEK_BASE_URL": model.url}
    agent = coder(devvi_cli, place, model, transport=transport, env=env)
    async with asyncio.timeout(LIMIT_S), agent:
        if transport == "acp":
            with pytest.raises(ExternalAgentConfigurationError, match="is not offered"):
                await agent.run("Hello without a key.")
        else:
            result = await agent.run("Hello without a key.")
            assert result.status == "error" and "no API key" in (result.stop_reason or "")
            assert result.raw["error_code"] == "MISSING_CREDENTIAL"
            assert result.raw["unavailable"] is True
    assert model.requests == []


# --- SDK ---------------------------------------------------------------------------------------

PRICES = TokenPricing(input=1.0, output=2.0, cache_read=0.1)
"""USD per million tokens; cache writes are charged at the input price."""


async def test_sdk_text_turn_reports_usage_and_cost(
    devvi_cli: DevviCli, place: Place, model: MockModel
) -> None:
    usage = Usage(
        input_tokens=1200, output_tokens=40, cache_read_tokens=800, cache_write_tokens=100
    )
    model.on_prompt("USAGE", Text("SDK text OK", thinking="Counting tokens.", usage=usage))
    agent = coder(devvi_cli, place, model, transport="sdk", pricing=PRICES)
    lead = Agent(
        ScriptedModel([call("coder", task="USAGE: reply with the mock text."), "Done."]),
        name="lead",
        tools=[agent.as_tool(timeout=LIMIT_S)],
        limits=Limits(max_cost_usd=1.0),
    )
    async with asyncio.timeout(LIMIT_S), agent:
        stream = lead.stream("Go")
        events = [event async for event in stream]
    result = stream.result
    assert result is not None and result.output == "Done."
    forwarded = inner(events)
    assert kinds(forwarded) == [
        "run_started", "turn_started", "reasoning_delta", "text_delta", "model_responded",
        "turn_finished",
    ]  # fmt: skip
    (responded,) = [e for e in forwarded if e.type == "model_responded"]
    assert (responded.response.provider, responded.response.model) == MOCK_ROUTE
    counted = responded.response.usage
    assert (counted.requests, counted.input_tokens, counted.cached_input_tokens) == (1, 2100, 800)
    assert counted.output_tokens == 40
    cost = (1200 * 1.0 + 40 * 2.0 + 800 * 0.1 + 100 * 1.0) / 1_000_000
    assert counted.cost_usd == pytest.approx(cost)
    assert result.usage.cost_usd == pytest.approx(cost)  # the lead's scripted model is free
    assert result.usage.input_tokens == 2100
    finished = forwarded[-1].data
    # the same word as ACP for the same outcome; Devvi's own is in raw["agent_stop_reason"]
    assert finished["status"] == "completed" and finished["stop_reason"] == "end_turn"
    assert re.fullmatch(r"devvi-sdk@0\.2\.\d+.*", finished["backend"])
    assert finished["session_id"].startswith("qa-")
    content = tool_results(result)[0].content
    assert content.startswith(f"SDK text OK\n\n[coder: session {finished['session_id']}; completed")
    (request,) = model.requests_with("USAGE")
    assert request.body["model"] == MOCK_MODEL


@pytest.mark.parametrize("permission_mode", ["workspace-write", "danger-full-access"])
async def test_sdk_shell_tool_writes_a_file_in_the_workspace(
    devvi_cli: DevviCli, place: Place, model: MockModel, permission_mode: str
) -> None:
    write_turn(model, place.workspace / "hello.txt")
    agent = coder(devvi_cli, place, model, transport="sdk", permission_mode=permission_mode)
    async with asyncio.timeout(LIMIT_S), agent:
        events, result = await run_turn(agent, "WRITE-FILE: create hello.txt")
    assert result.status == "completed" and result.text == "Wrote hello.txt."
    xfail_if_sandbox_denied(tool_output(model, "write-1"), permission_mode)
    assert (place.workspace / "hello.txt").read_text(encoding="utf-8-sig") == "hello from devvi"
    (called,) = result.tool_calls
    assert (called.id, called.name, called.is_error) == ("write-1", SHELL_NAME, False)
    jobs = [e.data for e in events if e.type == "external_event" and e.kind == "job"]
    assert [job["type"] for job in jobs] == ["job/registered", "job/settled"]
    assert jobs[-1]["status"] == "completed"
    assert result.usage.requests == 2  # the tool call, then the answer


@pytest.mark.parametrize("decision", ["allow", "deny"])
async def test_sdk_escalation_is_decided_by_qagents_rules(
    devvi_cli: DevviCli, place: Place, model: MockModel, decision: str
) -> None:
    target = place.outside / "escalated.txt"
    escalation_turn(model, target)
    agent = coder(devvi_cli, place, model, transport="sdk", permissions=rules_for(decision))
    async with asyncio.timeout(LIMIT_S), agent:
        events, result = await run_turn(agent, "ESCALATE: write the file outside the workspace.")
    (permission,) = result.permissions
    assert permission.tool == f"coder.{SHELL_NAME}" and permission.external_tool == SHELL_NAME
    assert permission.tool_call_id == "esc-1"  # approval/request carries Devvi's callId
    assert permission.decided_by == "rules" and permission.outcome == decision
    assert permission.arguments["sandbox_permissions"] == "danger-full-access"
    audit = [e.kind for e in events if e.type == "external_event"]
    assert audit.count("permission_requested") == 1 and audit.count("permission_decided") == 1
    assert "approval_asked" in audit and "approval_decided" in audit  # Devvi's own records
    if decision == "allow":
        assert result.text == "DONE" and target.read_text(encoding="utf-8-sig") == "escalated"
    else:
        assert result.text == "REFUSED" and not target.exists()


async def test_sdk_cancel_during_a_tool_stops_devvi_and_the_session_continues(
    devvi_cli: DevviCli, place: Place, model: MockModel
) -> None:
    model.on_prompt(
        "SLEEP", ToolUse(SHELL, shell_arguments(sleep_command(120), "Wait"), id="sleep-1")
    )
    model.after_tool("sleep-1", Text("The wait finished."))
    model.on_prompt("AFTER-CANCEL", Text("Continued after the cancel."))
    agent = coder(devvi_cli, place, model, transport="sdk", cancel_grace_s=20)
    async with asyncio.timeout(LIMIT_S), agent:
        began = time.monotonic()
        async with agent.stream("SLEEP: wait two minutes.") as turn:
            async for event in turn:
                if event.type == "tool_call_started":
                    await asyncio.sleep(1.0)
                    turn.cancel()
        elapsed = time.monotonic() - began
        first = turn.result
        assert first is not None
        events, second = await run_turn(agent, "AFTER-CANCEL: go on.", session=first.session_id)
    assert first.status == "cancelled" and first.stop_reason == "cancelled"
    assert not first.raw.get("forced")  # shutdown stopped Devvi within the grace period
    assert elapsed < 60
    assert events[0].type == "run_started" and events[0].resumed is True
    assert second.status == "completed" and second.text == "Continued after the cancel."
    assert second.session_id == first.session_id
    log = logs.read_session_log(Path(second.raw["session_log"]))
    assert [record.end_kind for record in log.turns()][:1] == ["aborted"]


async def test_sdk_a_second_agent_resumes_the_session(
    devvi_cli: DevviCli, place: Place, model: MockModel
) -> None:
    model.on_prompt("REMEMBER", Text("Noted."))
    model.on_prompt("RECALL", recall)
    async with asyncio.timeout(LIMIT_S):
        first_agent = coder(devvi_cli, place, model, transport="sdk")
        async with first_agent:
            _, first = await run_turn(first_agent, "REMEMBER: the code word is OSPREY.")
        second_agent = coder(devvi_cli, place, model, transport="sdk")
        async with second_agent:
            events, second = await run_turn(
                second_agent, "RECALL the code word.", session=first.session_id
            )
    assert first.text == "Noted." and first.session_id.startswith("qa-")
    assert events[0].type == "run_started" and events[0].resumed is True
    assert second.session_id == first.session_id
    assert second.text == "The code word is OSPREY."


# --- SDK: crash re-attach with Devvi's real session log ----------------------------------------------


class _CrashBeforeResult(SQLiteStore):
    """A store whose process "crashes" just before it journals the external result.

    The append of the ``external_result`` checkpoint never returns, so the test can kill
    the run at exactly the moment after Devvi finished the turn and before qagents
    recorded it.
    """

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.reached = asyncio.Event()

    def _stops(self, entries: list[Any]) -> bool:
        return any(
            entry.kind == "tool_checkpoint" and entry.data.get("key") == "external_result"
            for entry in entries
        )

    async def append(self, session_id: str, entries: list[Any]) -> None:
        if self._stops(entries):
            self.reached.set()
            await asyncio.Event().wait()
        await super().append(session_id, entries)

    async def append_leased(self, session_id: str, entries: list[Any], owner: str) -> None:
        if self._stops(entries):
            self.reached.set()
            await asyncio.Event().wait()
        await super().append_leased(session_id, entries, owner)


async def eventually(check: Any, timeout: float = 120.0) -> None:
    """Wait until ``check()`` (sync or async) is true."""
    for _ in range(int(timeout / 0.05)):
        value = check()
        if asyncio.iscoroutine(value):
            value = await value
        if value:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for {check}")


async def _has_checkpoint(db: Path, key: str) -> bool:
    return key in await _checkpoint_keys(db)


async def _crash(task: asyncio.Task[Any]) -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def _checkpoint_keys(db: Path) -> list[str]:
    async with SQLiteStore(db) as store:
        entries = await Session(store, id="crash").entries()
    assert not any(entry.kind == "tool_result" for entry in entries)
    return [entry.data["key"] for entry in entries if entry.kind == "tool_checkpoint"]


async def test_sdk_crash_during_the_turn_resumes_it_from_the_log(
    devvi_cli: DevviCli, place: Place, model: MockModel
) -> None:
    hold = Hold(Text("Too late."), timeout=120)
    model.on_prompt("CRASH-OPEN", hold)
    model.on_prompt(resume_marker(), Text("Finished after the interruption."))
    db = place.root / "runs.db"
    async with asyncio.timeout(LIMIT_S):
        agent = coder(devvi_cli, place, model, transport="sdk")
        lead = Agent(
            ScriptedModel([call("coder", task="CRASH-OPEN: start the long job.")]),
            tools=[agent.as_tool(timeout=LIMIT_S)],
        )
        async with SQLiteStore(db) as store:
            task = asyncio.create_task(lead.run("go", session=Session(store, id="crash")))
            await eventually(hold.started.is_set)  # Devvi's model request is in flight
            await eventually(lambda: _has_checkpoint(db, MESSAGE_CHECKPOINT))
            await _crash(task)  # the calling run dies while Devvi works
        hold.release()
        await agent.aclose()
        assert await _checkpoint_keys(db) == [
            BASELINE_CHECKPOINT, "external_session", MESSAGE_CHECKPOINT
        ]  # fmt: skip

        second = coder(devvi_cli, place, model, transport="sdk")
        lead2 = Agent(ScriptedModel(["Done."]), tools=[second.as_tool(timeout=LIMIT_S)])
        async with second, SQLiteStore(db) as store:
            resumed = await lead2.resume(Session(store, id="crash"))
    assert resumed.output == "Done."
    content = tool_results(resumed)[0].content
    assert content.startswith("Finished after the interruption.\n\n[coder: session qa-")
    (asked,) = [r for r in model.requests if resume_marker() in r.prompt]
    assert "CRASH-OPEN" in asked.transcript()  # the resumed session kept the first prompt
    # Devvi puts the resume prompt into the same user message as the unanswered first one
    assert "CRASH-OPEN" in asked.prompt and len(asked.messages) == 1
    log_file = next(place.home.joinpath("sessions").rglob("session.v4.jsonl"))
    ends = [record.end_kind for record in logs.read_session_log(log_file).turns()]
    assert ends[0] in ("aborted", "interrupted") and ends[-1] == "completed"


async def test_sdk_crash_after_the_turn_is_recovered_from_the_log_without_a_prompt(
    devvi_cli: DevviCli, place: Place, model: MockModel
) -> None:
    usage = Usage(input_tokens=500, output_tokens=7)
    model.on_prompt("CRASH-DONE", Text("Finished before the crash.", usage=usage))
    db = place.root / "runs.db"
    async with asyncio.timeout(LIMIT_S):
        agent = coder(devvi_cli, place, model, transport="sdk")
        lead = Agent(
            ScriptedModel([call("coder", task="CRASH-DONE: a short job.")]),
            tools=[agent.as_tool(timeout=LIMIT_S)],
        )
        crashing = _CrashBeforeResult(db)
        task = asyncio.create_task(lead.run("go", session=Session(crashing, id="crash")))
        await crashing.reached.wait()  # Devvi finished; qagents has not recorded it
        await _crash(task)
        await crashing.aclose()
        await agent.aclose()
        assert await _checkpoint_keys(db) == [
            BASELINE_CHECKPOINT, "external_session", MESSAGE_CHECKPOINT
        ]  # fmt: skip
        asked_before = len(model.requests)

        second = coder(devvi_cli, place, model, transport="sdk")
        lead2 = Agent(ScriptedModel(["Done."]), tools=[second.as_tool(timeout=LIMIT_S)])
        async with second, SQLiteStore(db) as store:
            resumed = await lead2.resume(Session(store, id="crash"))
    assert resumed.output == "Done."
    content = tool_results(resumed)[0].content
    assert content.startswith("Finished before the crash.\n\n[coder: session qa-")
    assert len(model.requests) == asked_before  # re-attached: Devvi was not prompted again
    assert resumed.usage.input_tokens == 500  # the recovered turn's usage is counted


async def test_sdk_a_session_of_another_workspace_is_refused(
    devvi_cli: DevviCli, place: Place, model: MockModel
) -> None:
    """Two coders share one Devvi home and work in two repositories. Coder ``a`` is given
    coder ``b``'s session id: Devvi's ``session/resume`` would restore ``b``'s directory and
    ``a`` would write in ``b``'s repository, outside ``b``'s lock and rules. Refused before
    Devvi opens the session: nothing is written, and ``b``'s log is untouched."""
    model.on_prompt("NOTE", Text("Noted."))
    command = write_file_command("where.txt", "written by a")
    model.on_prompt("WHERE", ToolUse(SHELL, shell_arguments(command, "Write where.txt"), id="w-1"))
    model.after_tool("w-1", Text("Wrote where.txt."))
    other = place.root / "ws_b"
    other.mkdir()
    async with asyncio.timeout(LIMIT_S):
        b = coder(devvi_cli, place, model, transport="sdk", workspace=other)
        async with b:
            _, first = await run_turn(b, "NOTE: you work on repository b.")
        assert first.status == "completed"
        log_file = next(place.home.joinpath("sessions").rglob("session.v4.jsonl"))
        before = log_file.read_bytes()
        a = coder(devvi_cli, place, model, transport="sdk")
        async with a:
            _, refused = await run_turn(a, "WHERE: write where.txt.", session=first.session_id)
    assert refused.status == "error"
    assert refused.stop_reason == devvi_texts.DEVVI_OTHER_WORKSPACE.format(session=first.session_id)
    assert model.requests_with("WHERE") == []  # the task never reached Devvi's model
    assert not (place.workspace / "where.txt").exists() and not (other / "where.txt").exists()
    assert log_file.read_bytes() == before


# --- open questions of the design (4.2), pinned by what Devvi does --------------------------------


async def test_acp_the_file_sandbox_follows_the_session_cwd_not_the_process_cwd(
    devvi_cli: DevviCli, place: Place, model: MockModel
) -> None:
    """Design 4.2 #3: the process starts in the workspace, the session opens in ``outside``."""
    in_session = place.outside / "in-session-cwd.txt"
    in_process = place.workspace / "in-process-cwd.txt"
    model.on_prompt(
        "WHERE", ToolUse("write", {"file_path": str(in_session), "content": "B"}, id="w-b")
    )
    model.after_tool(
        "w-b", ToolUse("write", {"file_path": str(in_process), "content": "A"}, id="w-a")
    )
    model.after_tool("w-a", Text("done"))
    launch = await coder(devvi_cli, place, model).launch()
    agent = ACPAgent(
        launch.argv("acp"),
        env=launch.env,
        cwd=place.workspace,
        workspace=place.outside,
        model=MOCK_ROUTE,
        name="coder",
        track_changes=False,
    )
    async with asyncio.timeout(LIMIT_S), agent:
        result = await agent.run("WHERE: write both files.")
    assert result.status == "completed" and result.text == "done"
    assert in_session.read_text(encoding="utf-8") == "B"
    assert not in_process.exists()
    assert SANDBOX_DENIED in tool_output(model, "w-a")
    (asked,) = model.requests_with("WHERE")[:1]
    assert f'session workspace: "{json_path(place.outside)}"' in asked.prompt


async def test_sdk_a_prompt_for_a_stored_session_that_was_not_resumed_is_refused(
    devvi_cli: DevviCli, place: Place, model: MockModel
) -> None:
    """Design 4.2 #5: ``session/prompt`` on an existing, unopened id neither creates nor forks."""
    model.on_prompt("REMEMBER", Text("Noted."))
    agent = coder(devvi_cli, place, model, transport="sdk")
    async with asyncio.timeout(LIMIT_S):
        async with agent:
            first = await agent.run("REMEMBER: the code word is OSPREY.")
        launch = await agent.launch()
        process = await ManagedProcess.start(
            launch.argv("sdk"), env=launch.env, cwd=place.workspace, name="probe"
        )
        peer = JsonRpcPeer.for_process(process)
        peer.on_notification("session.event", lambda params: None)
        peer.on_notification("session.status", lambda params: None)
        peer.start()
        try:
            await peer.request(
                "initialize",
                {"cwd": str(place.workspace), "provider": MOCK_ROUTE[0], "model": MOCK_ROUTE[1]},
                timeout=120,
            )
            blocks = [{"type": "text", "text": "RECALL the code word."}]
            with pytest.raises(JsonRpcError) as refused:
                await peer.request(
                    "session/prompt",
                    {"sessionId": first.session_id, "contentBlocks": blocks},
                    timeout=60,
                )
            resumed = await peer.request(
                "session/resume", {"sessionId": first.session_id}, timeout=60
            )
            await peer.request("shutdown", timeout=10)
        finally:
            await process.close(grace=5)
            await peer.aclose()
    assert refused.value.code == -32603
    assert f'session "{first.session_id}" already exists' in refused.value.message
    assert resumed == {}
    assert len(list(place.home.joinpath("sessions").rglob("session.v4.jsonl"))) == 1
    assert model.requests_with("RECALL") == []


async def test_sdk_the_question_tool_fails_closed_without_an_answerer(
    devvi_cli: DevviCli, place: Place, model: MockModel
) -> None:
    """Design 4.2 #6: qagents does not offer ``userQuestions``; Devvi's question tool then
    fails at once (it does not hang). With the capability, an empty ``answers`` list is
    refused by Devvi too (``BAD_ANSWER``), and an unanswered question waits until
    ``shutdown`` cancels it (``user-question.cancelled``); see devvi-observed.md."""
    question = {"questions": [{"id": "q1", "question": "Which color?"}]}
    model.on_prompt("QUESTION", ToolUse("ask_user_question", question, id="ask-1"))
    model.after_tool("ask-1", Text("The question tool failed."))
    agent = coder(devvi_cli, place, model, transport="sdk")
    async with asyncio.timeout(LIMIT_S), agent:
        result = await agent.run("QUESTION: ask me which color.")
    assert result.status == "completed" and result.text == "The question tool failed."
    (called,) = result.tool_calls
    assert called.name == "ask_user_question" and called.is_error is True
    assert tool_output(model, "ask-1").startswith("Error: ")
