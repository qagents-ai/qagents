"""The ACP v1 client against a scripted fake agent that replays Devvi's recorded transcripts.

The fake (``fakes/fake_acp_agent.py``) checks every client message against the fixture and
writes the recorded agent lines; the fixtures in ``tests/fixtures/devvi/acp`` are copied
from Devvi's ``snapshots/acp`` (MIT). Tests that need other shapes write a small script.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from qagents import Agent, Approval, Permissions
from qagents import prompts as acp_texts
from qagents import prompts as texts
from qagents.errors import ConfigurationError
from qagents.events import SubagentEvent, TextDelta, ToolCallStarted
from qagents.external import (
    ExternalAgentConfigurationError,
    ExternalAgentConnectionError,
    ExternalAgentEvent,
    ExternalResult,
    ExternalTurnContext,
)
from qagents.integrations.acp import (
    CLIENT_CAPABILITIES,
    TITLES_ARE_TOOL_NAMES,
    ACPAgent,
    MCPServerHTTP,
    MCPServerStdio,
    default_tool_name,
    title_tool_name,
)
from qagents.integrations.acp import client as acp_client
from qagents.messages import ImagePart, TextPart
from qagents.models.testing import ScriptedModel, call

from .fakes import script

pytestmark = pytest.mark.devvi_fake

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "devvi" / "acp"
WINDOWS = sys.platform == "win32"
PIXEL = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


# -- helpers ----------------------------------------------------------------------------------


def lines_of(name: str) -> list[str]:
    return (FIXTURES / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()


HANDSHAKE = lines_of("handshake")[1:]
"""expect initialize, its answer, expect session/new, its answer (session fake-session-1)."""

INIT_ANSWER = json.loads(HANDSHAKE[1])

PROMPT = {
    "expect": {
        "id": 3,
        "method": "session/prompt",
        "params": {"sessionId": "{{sessionId}}", "prompt": "$list"},
    }
}


def update(session_update: str, **fields: Any) -> dict[str, Any]:
    body = {"sessionUpdate": session_update, **fields}
    return {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {"sessionId": "{{sessionId}}", "update": body},
    }


def chunk(text: str) -> dict[str, Any]:
    content = {"type": "text", "text": text}
    return update("agent_message_chunk", messageId="{{messageId}}", content=content)


def stop(request_id: int = 3, reason: str = "end_turn") -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": {"stopReason": reason}}


ALLOW_OR_REJECT = [
    {"optionId": "allow-once", "name": "Allow once", "kind": "allow_once"},
    {"optionId": "reject-once", "name": "Reject", "kind": "reject_once"},
]
REJECTED = {"outcome": {"outcome": "selected", "optionId": "reject-once"}}


def ask_permission(request_id: int, call_id: str, accept: dict[str, Any]) -> list[Any]:
    """The agent asks about ``call_id`` (only its id, as Devvi does); expects ``accept``."""
    params = {
        "sessionId": "{{sessionId}}",
        "toolCall": {"toolCallId": call_id},
        "options": ALLOW_OR_REJECT,
    }
    request = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "session/request_permission",
        "params": params,
    }
    return [request, {"answer": {"id": request_id, "accept": [accept]}}]


def new_session(request_id: int, session: str) -> list[Any]:
    """``session/new``, answered with the session id ``session``."""
    params = {"cwd": "$absolute", "mcpServers": "$list"}
    return [
        {"expect": {"id": request_id, "method": "session/new", "params": params}},
        {"jsonrpc": "2.0", "id": request_id, "result": {"sessionId": session}},
    ]


def prompted(request_id: int, session: str, text: str) -> list[Any]:
    """``session/prompt`` in ``session``, answered with ``text`` and ``end_turn``."""
    body = {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": text}}
    params = {"sessionId": session, "prompt": "$list"}
    return [
        {"expect": {"id": request_id, "method": "session/prompt", "params": params}},
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {"sessionId": session, "update": body},
        },
        stop(request_id),
    ]


def resumed(request_id: int, session: str) -> list[Any]:
    params = {"sessionId": session, "cwd": "$absolute", "mcpServers": "$list"}
    return [
        {"expect": {"id": request_id, "method": "session/resume", "params": params}},
        {"jsonrpc": "2.0", "id": request_id, "result": {}},
    ]


def closed(request_id: int, session: str, answer: dict[str, Any] | None = None) -> list[Any]:
    """``session/close`` of ``session``, answered ``{}`` (or ``answer``, an error object)."""
    reply: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
    if answer is None:
        reply["result"] = {}
    else:
        reply["error"] = answer
    params = {"sessionId": session}
    return [{"expect": {"id": request_id, "method": "session/close", "params": params}}, reply]


def initialize_answer(**changes: Any) -> dict[str, Any]:
    answer = json.loads(HANDSHAKE[1])
    result = answer["result"]
    for key, value in changes.items():
        if key in ("loadSession", "mcpCapabilities", "promptCapabilities", "sessionCapabilities"):
            result["agentCapabilities"][key] = value
        else:
            result[key] = value
    return answer


def write_script(tmp_path: Path, rows: list[Any], name: str = "script.jsonl") -> Path:
    path = tmp_path / name
    text = "\n".join(row if isinstance(row, str) else json.dumps(row) for row in rows)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def make(tmp_path: Path, fixture: str | Path, *extra: str, **options: Any) -> ACPAgent:
    path = fixture if isinstance(fixture, Path) else FIXTURES / f"{fixture}.jsonl"
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    options.setdefault("name", "coder")
    options.setdefault("workspace", workspace)
    options.setdefault("track_changes", False)
    options.setdefault("close_grace_s", 2.0)
    command = script("fake_acp_agent", "--script", str(path), "--log", str(log_path(tmp_path)))
    return ACPAgent([*command, *extra], **options)


def log_path(tmp_path: Path) -> Path:
    return tmp_path / "client.jsonl"


def sent(tmp_path: Path) -> list[dict[str, Any]]:
    path = log_path(tmp_path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def methods(tmp_path: Path) -> list[str]:
    return [message.get("method", "<answer>") for message in sent(tmp_path)]


async def play(agent: ACPAgent, task: Any = "Do the task.", **options: Any) -> tuple[list, Any]:
    turn = agent.stream(task, **options)
    async with turn:
        events = [event async for event in turn]
    return events, turn.result


def describe(event: Any) -> str:
    kind = event.type
    if kind == "run_started":
        return f"run_started {event.session_id} resumed={event.resumed} depth={event.depth}"
    if kind == "text_delta":
        return f"text {event.text[:40]!r}"
    if kind == "reasoning_delta":
        return f"reasoning {event.text[:40]!r}"
    if kind == "tool_call_started":
        return f"tool_call_started {event.call.name} {event.call.id}"
    if kind == "tool_call_finished":
        flag = "failed" if event.result.is_error else "ok"
        return f"tool_call_finished {event.call.name} {flag} {event.result.content[:40]!r}"
    if kind == "external_event":
        data = event.data
        if event.kind == "permission_requested":
            return f"permission_requested {data['tool']}"
        if event.kind == "permission_decided":
            return f"permission_decided {data['tool']} {data['outcome']} by {data['rule']}"
        if event.kind == "turn_finished":
            return f"turn_finished {data['status']} {data['stop_reason']}"
        if event.kind == "context":
            return f"context {data['used']}/{data['size']}"
        return f"external_event {event.kind}"
    return kind


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _pid_of(heartbeat: Path) -> int:
    return int(Path(f"{heartbeat}.pid").read_text(encoding="ascii"))


async def _beating(path: Path) -> bool:
    before = _size(path)
    await asyncio.sleep(0.4)
    return _size(path) > before


def _alive(pid: int) -> bool:
    if WINDOWS:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel32.OpenProcess(0x00100000 | 0x1000, False, pid)
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == 0x102
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


BASH_COMMAND = (
    "printf 'escalated\\n' > /tmp/dsh-escalated.txt && cat /tmp/dsh-escalated.txt "
    "&& rm /tmp/dsh-escalated.txt"
)
APPROVED_CALL = "call_00_d0sAHpJ9mYOJi0z7KNy30441"
REJECTED_CALL = "call_00_WB1vnPomi8yr6MlcFKTj7912"


# -- 1. handshake --------------------------------------------------------------------------------


async def test_handshake_reads_the_capabilities_and_offers_none(tmp_path: Path) -> None:
    agent = make(tmp_path, "handshake", auth_method="unused")
    async with agent:
        caps = await agent.capabilities()
        assert agent.agent_info == {"name": "deepseek-harness-acp", "version": "0.0.1"}
        assert agent.backend_name == "acp:deepseek-harness-acp@0.0.1"
    assert caps.model_dump() == {
        "token_usage": False,
        "live_tokens": False,
        "cancel_turn": True,
        "resume_sessions": True,
        "client_session_ids": False,
        "permission_requests": True,
        "mcp_servers": True,
    }
    first = sent(tmp_path)[0]
    assert first["method"] == "initialize"
    assert first["params"]["protocolVersion"] == 1
    assert first["params"]["clientCapabilities"] == CLIENT_CAPABILITIES
    assert first["params"]["clientCapabilities"]["fs"] == {
        "readTextFile": False,
        "writeTextFile": False,
    }
    assert first["params"]["clientCapabilities"]["terminal"] is False
    assert first["params"]["clientInfo"]["name"] == "qagents"
    assert "authenticate" not in methods(tmp_path)  # authMethods is empty: never authenticate


async def test_protocol_version_2_is_refused(tmp_path: Path) -> None:
    rows = [HANDSHAKE[0], initialize_answer(protocolVersion=2)]
    agent = make(tmp_path, write_script(tmp_path, rows))
    with pytest.raises(ExternalAgentConfigurationError, match="protocol version 2") as raised:
        await agent.capabilities()
    assert "qagents speaks version 1" in str(raised.value).splitlines()[0]
    assert agent._process is None and not agent.connected


async def test_a_chosen_auth_method_is_used_only_when_offered(tmp_path: Path) -> None:
    methods_offered = [{"id": "api-key", "name": "API key"}, {"id": "browser", "name": "Log in"}]
    rows = [
        HANDSHAKE[0],
        initialize_answer(authMethods=methods_offered),
        {"expect": {"id": 2, "method": "authenticate", "params": {"methodId": "api-key"}}},
        {"jsonrpc": "2.0", "id": 2, "result": {}},
    ]
    agent = make(tmp_path, write_script(tmp_path, rows), auth_method="api-key")
    async with agent:
        await agent.capabilities()
    assert methods(tmp_path) == ["initialize", "authenticate"]


async def test_offered_auth_without_a_choice_names_the_methods(tmp_path: Path) -> None:
    rows = [
        HANDSHAKE[0],
        initialize_answer(authMethods=[{"id": "api-key", "name": "API key"}]),
        HANDSHAKE[2],
        {
            "jsonrpc": "2.0",
            "id": 2,
            "error": {"code": -32000, "message": "Authentication required"},
        },
    ]
    agent = make(tmp_path, write_script(tmp_path, rows))
    async with agent:
        with pytest.raises(ExternalAgentConfigurationError, match="auth_method") as raised:
            await play(agent)
    assert "api-key" in str(raised.value)
    assert "authenticate" not in methods(tmp_path)


async def test_an_unknown_or_interactive_auth_method_is_refused(tmp_path: Path) -> None:
    offered = [{"id": "login", "name": "Log in", "type": "terminal", "args": ["login"]}]
    rows = [HANDSHAKE[0], initialize_answer(authMethods=offered)]
    agent = make(tmp_path, write_script(tmp_path, rows), auth_method="api-key")
    with pytest.raises(ExternalAgentConfigurationError, match="offers no authentication method"):
        await agent.capabilities()
    agent = make(tmp_path, write_script(tmp_path, rows), auth_method="login")
    with pytest.raises(ExternalAgentConfigurationError, match="interactively"):
        await agent.capabilities()


async def test_an_agent_that_exits_during_the_handshake(tmp_path: Path) -> None:
    rows = [{"exit": 1, "stderr": "no provider route is configured"}]
    agent = make(tmp_path, write_script(tmp_path, rows))
    with pytest.raises(ExternalAgentConfigurationError, match="during the ACP handshake") as raised:
        await agent.capabilities()
    assert raised.value.stderr_tail is not None
    assert "no provider route" in raised.value.stderr_tail
    assert "no provider route" not in str(raised.value).splitlines()[0]  # the model-safe line


async def test_a_missing_program_is_a_configuration_error(tmp_path: Path) -> None:
    agent = ACPAgent(["qagents-no-such-acp-agent"], name="coder", workspace=tmp_path)
    with pytest.raises(ExternalAgentConfigurationError, match="not found"):
        await agent.capabilities()


def test_the_command_must_be_a_list() -> None:
    with pytest.raises(ConfigurationError, match="list of strings"):
        ACPAgent("node agent.js --acp")
    with pytest.raises(ConfigurationError):
        ACPAgent([sys.executable], env={"A": 1})  # type: ignore[dict-item]


# -- 2. event mapping, fixture by fixture ----------------------------------------------------------


async def test_escalation_approved_event_sequence(tmp_path: Path) -> None:
    agent = make(tmp_path, "escalation-approved", permissions=Permissions(allow=["coder.bash"]))
    async with agent:
        events, result = await play(agent, "Write the file outside the workspace.")
    assert [describe(event) for event in events] == [
        "run_started fake-session-1 resumed=False depth=0",
        "reasoning 'The user wants me to run a command with '",
        f"tool_call_started coder.bash {APPROVED_CALL}",
        "permission_requested coder.bash",
        "permission_decided coder.bash allow by rules",
        "tool_call_finished coder.bash ok 'escalated\\n'",
        "reasoning 'The command succeeded — it wrote the fil'",
        "text 'DONE'",
        "turn_finished completed end_turn",
    ]
    started = events[2]
    assert started.call.arguments["command"] == BASH_COMMAND
    assert started.call.arguments["sandbox_permissions"] == "danger-full-access"
    finished = events[5]
    assert finished.result.metadata == {"title": "bash", "kind": "other"}
    assert isinstance(result, ExternalResult)
    assert (result.status, result.stop_reason, result.text) == ("completed", "end_turn", "DONE")
    assert result.session_id == "fake-session-1"
    assert result.backend == "acp:deepseek-harness-acp@0.0.1"
    assert result.usage.total_tokens == 0
    assert [(c.name, c.finished, c.is_error) for c in result.tool_calls] == [("bash", True, False)]
    assert [(p.tool, p.outcome, p.decided_by) for p in result.permissions] == [
        ("coder.bash", "allow", "rules")
    ]
    assert result.permissions[0].arguments["command"] == BASH_COMMAND
    assert result.raw["acp_stop_reason"] == "end_turn"
    messages = sent(tmp_path)
    assert [m.get("method", "<answer>") for m in messages] == [
        "initialize",
        "session/new",
        "session/prompt",
        "<answer>",
    ]
    new = messages[1]["params"]
    assert new == {"cwd": str(tmp_path / "workspace"), "mcpServers": []}
    assert messages[2]["params"] == {
        "sessionId": "fake-session-1",
        "prompt": [{"type": "text", "text": "Write the file outside the workspace."}],
    }
    assert messages[3] == {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"outcome": {"outcome": "selected", "optionId": "allow-once"}},
    }


async def test_escalation_rejected_event_sequence(tmp_path: Path) -> None:
    agent = make(tmp_path, "escalation-rejected", permissions=Permissions(deny=["coder.bash"]))
    async with agent:
        events, result = await play(agent)
    assert [describe(event) for event in events] == [
        "run_started fake-session-1 resumed=False depth=0",
        "reasoning 'The user wants me to run a specific comm'",
        f"tool_call_started coder.bash {REJECTED_CALL}",
        "permission_requested coder.bash",
        "permission_decided coder.bash deny by rules",
        "tool_call_finished coder.bash failed 'Error: the user rejected escalating this'",
        "reasoning 'The user rejected the escalation. As ins'",
        "text 'The user rejected the permission escalat'",
        "turn_finished completed end_turn",
    ]
    assert (
        result.text == "The user rejected the permission escalation, so this command cannot be run."
    )
    assert result.tool_calls[0].is_error
    assert sent(tmp_path)[-1]["result"] == {
        "outcome": {"outcome": "selected", "optionId": "reject-once"}
    }
    assert 'bash: "printf' in result.footer("coder")  # the refusal is in the footer


async def test_fs_escalation_fills_in_the_workspace(tmp_path: Path) -> None:
    agent = make(tmp_path, "fs-escalation-approved", permissions=Permissions(allow=["coder.*"]))
    async with agent:
        events, result = await play(agent)
    finished = [e for e in events if e.type == "tool_call_finished"]
    assert finished[0].call.name == "coder.write"
    assert finished[0].call.arguments["file_path"] == "escalated.md"
    workspace = str(tmp_path / "workspace")
    assert finished[0].result.content.startswith(f"<path>{workspace}/escalated.md</path>")
    assert result.text == "DONE" and result.permissions[0].approved


async def test_two_turns_in_one_session_with_images_and_context(tmp_path: Path) -> None:
    agent = make(tmp_path, "image-compaction")
    task = [TextPart(text="Here are screenshots: "), ImagePart(data=PIXEL, media_type="image/png")]
    async with agent:
        events, first = await play(agent, task)
        second_events, second = await play(agent, "Now reply DONE.", session=first.session_id)
    assert [describe(event) for event in events] == [
        "run_started fake-session-1 resumed=False depth=0",
        "text 'Received six dashboard frames; ready to '",
        "context 1234/10022",
        "turn_finished completed end_turn",
    ]
    assert [describe(event) for event in second_events] == [
        "run_started fake-session-1 resumed=True depth=0",
        "text 'DONE'",
        "context 1234/10022",
        "turn_finished completed end_turn",
    ]
    assert events[1].text == "Received six dashboard frames; ready to discuss."
    assert (first.text, second.text) == (first.text, "DONE")
    run_ids = [events[0].run_id, second_events[0].run_id]
    assert all(run_id.startswith("ext:fake-session-1:") for run_id in run_ids)
    assert run_ids[0] != run_ids[1]  # one run per turn of the session
    messages = sent(tmp_path)
    assert [m["method"] for m in messages] == [
        "initialize",
        "session/new",
        "session/prompt",
        "session/prompt",  # a session this connection opened is not resumed again
    ]
    assert messages[2]["params"]["prompt"] == [
        {"type": "text", "text": "Here are screenshots: "},
        {"type": "image", "mimeType": "image/png", "data": PIXEL},
    ]


async def test_images_are_refused_when_the_agent_takes_none(tmp_path: Path) -> None:
    agent = make(tmp_path, "handshake")
    async with agent:
        with pytest.raises(ExternalAgentConfigurationError, match="does not accept images"):
            await play(agent, [TextPart(text="look"), ImagePart(data=PIXEL)])
    assert "session/new" not in methods(tmp_path)  # refused before a session was opened


async def test_other_update_kinds(tmp_path: Path) -> None:
    rows = [
        *HANDSHAKE,
        PROMPT,
        update("user_message_chunk", content={"type": "text", "text": "echo"}),
        update("available_commands_update", availableCommands=[]),
        update(
            "agent_message_chunk", content={"type": "image", "mimeType": "image/png", "data": PIXEL}
        ),
        update("plan", entries=[{"content": "Read", "priority": "high", "status": "pending"}]),
        update("config_option_update", configOptions=[{"id": "model", "currentValue": "x"}]),
        update(
            "tool_call",
            toolCallId="c1",
            title="Run ls -la",
            kind="execute",
            status="pending",
            rawInput="not json {",
        ),
        update("tool_call_update", toolCallId="c1", rawInput={"command": "ls -la"}),
        update(
            "tool_call_update",
            toolCallId="c1",
            status="completed",
            content=[
                {"type": "content", "content": {"type": "text", "text": "a.py"}},
                {"type": "diff", "path": "/w/a.py", "oldText": None, "newText": "x"},
                {"type": "terminal", "terminalId": "t1"},
            ],
        ),
        update("tool_call_update", toolCallId="c1", status="completed"),  # finished once only
        {"raw": "this line is not JSON"},
        chunk("Done."),
        stop(),
    ]
    path = write_script(tmp_path, rows)
    agent = make(tmp_path, path)
    async with agent:
        events, result = await play(agent)
        assert agent.session_config("fake-session-1") == [{"id": "model", "currentValue": "x"}]
    assert [describe(event) for event in events] == [
        "run_started fake-session-1 resumed=False depth=0",
        "external_event image",
        "external_event plan",
        "external_event config",
        "tool_call_started coder.execute c1",
        "external_event tool_arguments",  # the update brought the input as JSON
        "tool_call_finished coder.execute ok 'a.py\\n[diff: /w/a.py]\\n[terminal: t1]'",
        "text 'Done.'",
        "turn_finished completed end_turn",
    ]
    image = events[1]
    assert isinstance(image, ExternalAgentEvent) and image.data == {"mimeType": "image/png"}
    started = events[4]
    assert started.call.raw_arguments == "not json {" and started.call.parse_error
    assert events[5].data["arguments"] == {"command": "ls -la"}
    finished = events[6]
    assert finished.call.arguments == {"command": "ls -la"} and finished.call.parse_error is None
    assert result.tool_calls[0].arguments == {"command": "ls -la"}  # updated by the later update
    assert result.raw["ignored_events"] == {"user_message_chunk": 1, "available_commands_update": 1}

    raw = make(tmp_path, path, raw_events=True)
    async with raw:
        events, result = await play(raw)
    unknown = [e for e in events if getattr(e, "kind", None) == "unknown"]
    assert [e.data["sessionUpdate"] for e in unknown] == ["available_commands_update"]
    assert result.raw["ignored_events"] == {"user_message_chunk": 1}


def _text_rows(*chunks: tuple[str, str | None]) -> list[Any]:
    """A turn of text chunks, each ``(text, messageId or None)``."""
    rows: list[Any] = [*HANDSHAKE, PROMPT]
    for text, message_id in chunks:
        fields: dict[str, Any] = {"content": {"type": "text", "text": text}}
        if message_id is not None:
            fields["messageId"] = message_id
        rows.append(update("agent_message_chunk", **fields))
    return [*rows, stop()]


async def test_chunks_of_one_message_are_joined_and_a_new_message_id_starts_a_block(
    tmp_path: Path,
) -> None:
    """A token stream keeps its chunks together; two messages read as two paragraphs."""
    rows = _text_rows(("Fixed", "m1"), (" it.", "m1"), ("Tests pass.", "m2"), (" All", None))
    agent = make(tmp_path, write_script(tmp_path, rows))
    async with agent:
        events, result = await play(agent)
    deltas = [event.text for event in events if isinstance(event, TextDelta)]
    assert deltas == ["Fixed", " it.", "\n\nTests pass.", " All"]
    assert result.text == "Fixed it.\n\nTests pass. All"


async def test_an_agent_whose_chunks_are_blocks_separates_every_chunk(tmp_path: Path) -> None:
    """Devvi sends a whole committed block per chunk, with the message's id on each."""

    class BlockAgent(ACPAgent):
        chunks_are_blocks = True

    rows = _text_rows(("Fixed it.", "m1"), ("Tests pass.", "m1"))
    path = write_script(tmp_path, rows)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = script("fake_acp_agent", "--script", str(path))
    agent = BlockAgent(command, name="coder", workspace=workspace, track_changes=False)
    async with agent:
        _, result = await play(agent)
    assert result.text == "Fixed it.\n\nTests pass."
    plain = make(tmp_path, path)  # the default: chunks with one id are one message
    async with plain:
        _, joined = await play(plain)
    assert joined.text == "Fixed it.Tests pass."


async def test_a_cancelled_disconnect_still_stops_the_process(tmp_path: Path) -> None:
    """The references stay until the process is closed, so a disconnect cancelled half way
    (the loop shutting down) still ends the process and forgets it."""
    agent = make(tmp_path, "handshake")
    await agent.ensure_connected()
    process = agent._process
    assert process is not None and process.returncode is None
    closing = asyncio.create_task(agent.disconnect())
    await asyncio.sleep(0)  # it is closing the connection now
    closing.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await closing
    assert process.returncode is not None  # stopped, not left running
    assert agent._process is None and agent._peer is None
    await agent.aclose()


def test_default_tool_names_never_use_the_title() -> None:
    # display text the model can steer: a deny rule on the kind must still match
    assert default_tool_name({"title": "make", "kind": "execute"}) == "execute"
    assert default_tool_name({"title": "rm", "kind": "execute"}) == "execute"
    assert default_tool_name({"title": "setup.py", "kind": "edit"}) == "edit"
    assert default_tool_name({"title": "bash", "kind": "other"}) == "other"
    assert default_tool_name({"title": "Run `ls`", "kind": "execute"}) == "execute"
    assert default_tool_name({"name": "read_file", "title": "Read a.py"}) == "read_file"
    assert default_tool_name({"title": "Thinking hard"}) == "unknown"
    # agents whose code puts the tool's name in the title (Devvi) opt in
    assert title_tool_name({"title": "bash", "kind": "other"}) == "bash"
    assert title_tool_name({"title": "mcp__db__query"}) == "mcp__db__query"
    assert title_tool_name({"title": "Run `ls`", "kind": "execute"}) == "execute"
    assert title_tool_name({"name": "read_file", "title": "read"}) == "read_file"
    assert title_tool_name({"title": "Thinking hard"}) == "unknown"
    assert "deepseek-harness-acp" in TITLES_ARE_TOOL_NAMES  # the fixtures' agentInfo.name


def _other_agent_asks(*calls: list[Any]) -> list[Any]:
    """An agent that is not Devvi (its own agentInfo) runs ``calls`` in one prompt."""
    other = initialize_answer(agentInfo={"name": "other-agent", "version": "2.0"})
    rows = [HANDSHAKE[0], other, HANDSHAKE[2], HANDSHAKE[3], PROMPT]
    for call_rows in calls:
        rows += call_rows
    return [*rows, chunk("Stopped."), stop()]


ACP_TOOL_KINDS = {
    "read",
    "edit",
    "delete",
    "move",
    "search",
    "execute",
    "think",
    "fetch",
    "switch_mode",
    "other",
}


def test_the_examples_name_a_generic_agents_tools_by_kind() -> None:
    """The examples coding agents read (get_api, search) use names `default_tool_name` makes."""
    import qagents.integrations.acp as package

    for doc in (package.__doc__, ACPAgent.__doc__):
        assert doc is not None
        rules = re.findall(r"Permissions\((.*?)\)\)?\n", doc, flags=re.DOTALL)
        names = re.findall(r'"[\w-]+\.([\w*]+)"', " ".join(rules))
        assert names, doc
        assert set(names) <= ACP_TOOL_KINDS | {"*"}, names


async def test_a_title_does_not_get_past_a_deny_rule_on_the_kind(tmp_path: Path) -> None:
    rows = _other_agent_asks(
        [
            update("tool_call", toolCallId="c1", title="rm", kind="execute", status="pending",
                   rawInput={"command": "rm -rf build"}),
            *ask_permission(1, "c1", REJECTED),
        ],
        [
            update("tool_call", toolCallId="c2", title="setup.py", kind="edit", status="pending",
                   rawInput={"path": "setup.py"}),
            *ask_permission(2, "c2", REJECTED),
        ],
        [  # described first without a kind; the update that brings it names the call
            update("tool_call", toolCallId="c3", title="git:push", status="pending",
                   rawInput={"command": "git push"}),
            update("tool_call_update", toolCallId="c3", kind="execute"),
            *ask_permission(3, "c3", REJECTED),
        ],
    )  # fmt: skip
    rules = Permissions(mode="allow_all", deny=["coder.execute", "coder.edit"])
    agent = make(tmp_path, write_script(tmp_path, rows), permissions=rules)
    async with agent:
        events, result = await play(agent)
    assert result.completed, result.raw
    assert [(p.tool, p.outcome, p.decided_by) for p in result.permissions] == [
        ("coder.execute", "deny", "rules"),
        ("coder.edit", "deny", "rules"),
        ("coder.execute", "deny", "rules"),
    ]
    assert [call.name for call in result.tool_calls] == ["execute", "edit", "execute"]
    started = [describe(event) for event in events if event.type == "tool_call_started"]
    assert started[2] == "tool_call_started coder.unknown c3"  # the event keeps its first name


async def test_a_later_input_that_is_not_json_is_refused(tmp_path: Path) -> None:
    cancelled = {"outcome": {"outcome": "cancelled"}}
    rows = _other_agent_asks(
        [
            update("tool_call", toolCallId="c1", title="Run ls", kind="execute", status="pending",
                   rawInput={"command": "ls"}),
            update("tool_call_update", toolCallId="c1", rawInput="rm -rf / {"),
            *ask_permission(1, "c1", cancelled),
        ]
    )  # fmt: skip
    agent = make(tmp_path, write_script(tmp_path, rows), permissions=Permissions(mode="allow_all"))
    async with agent:
        _, result = await play(agent)
    assert result.completed, result.raw
    assert [(p.outcome, p.decided_by) for p in result.permissions] == [
        ("cancel", "unreadable_arguments")
    ]


@pytest.mark.parametrize(
    ("tool_name", "permission"),
    [(None, ("coder.other", "no_approver")), (title_tool_name, ("coder.bash", "rules"))],
    ids=["titles ignored", "titles opted in"],
)
async def test_titles_name_tools_only_when_opted_in(
    tmp_path: Path, tool_name: Any, permission: tuple[str, str]
) -> None:
    rows = _other_agent_asks(
        [
            update("tool_call", toolCallId="c1", title="bash", kind="other", status="pending",
                   rawInput={"command": "ls"}),
            *ask_permission(1, "c1", REJECTED),
        ]
    )  # fmt: skip
    rules = Permissions(deny=["coder.bash"])  # anything else asks: no approver, refused
    agent = make(tmp_path, write_script(tmp_path, rows), permissions=rules, tool_name=tool_name)
    async with agent:
        _, result = await play(agent)
    assert [(p.tool, p.decided_by) for p in result.permissions] == [permission]


async def test_as_tool_wraps_the_events_and_writes_the_footer(tmp_path: Path) -> None:
    agent = make(tmp_path, "escalation-approved")
    model = ScriptedModel([call("coder", task="Write the file."), "Done."])
    lead = Agent(
        model, name="lead", tools=[agent.as_tool()], permissions=Permissions(allow=["coder.bash"])
    )
    async with agent:
        stream = lead.stream("Go")
        events = [event async for event in stream]
    result = stream.result
    assert result is not None and result.output == "Done."
    forwarded = [event for event in events if isinstance(event, SubagentEvent)]
    assert {event.agent for event in forwarded} == {"coder"}
    inner = [describe(event.event) for event in forwarded]
    assert inner[0] == "run_started fake-session-1 resumed=False depth=1"
    assert inner[1:] == [
        "reasoning 'The user wants me to run a command with '",
        f"tool_call_started coder.bash {APPROVED_CALL}",
        "permission_requested coder.bash",
        "permission_decided coder.bash allow by rules",
        "tool_call_finished coder.bash ok 'escalated\\n'",
        "reasoning 'The command succeeded — it wrote the fil'",
        "text 'DONE'",
        "turn_finished completed end_turn",
    ]
    assert forwarded[0].event.parent_run_id == result.run_id
    content = [part for m in result.messages for part in m.tool_results][0].content
    assert content == "DONE\n\n[coder: session fake-session-1; completed; 1 tool call (bash 1)]"


async def test_the_turn_projects_onto_ag_ui(tmp_path: Path) -> None:
    from qagents.integrations.agui import to_agui_events

    async def kinds(mode: str) -> list[dict[str, Any]]:
        agent = make(tmp_path, "escalation-approved")
        lead = Agent(
            ScriptedModel([call("coder", task="t"), "ok"]),
            tools=[agent.as_tool()],
            permissions=Permissions(allow=["coder.*"]),
        )
        async with agent:
            stream = to_agui_events(lead.stream("go"), thread_id="t", run_id="r", subagents=mode)
            return [event async for event in stream]

    def check_rules(events: list[dict[str, Any]]) -> None:
        """The AG-UI event rules, on the whole stream: the run starts first and ends last,
        exactly once; every tool call's ARGS, END and RESULT follow its START (same id),
        ARGS before END, START and END once each; every text message is START, CONTENT...,
        END under one id; every reasoning phase is REASONING_START, its reasoning messages
        (START, CONTENT..., END under the phase's id), REASONING_END. Text, reasoning
        message and reasoning phase ids are tracked apart, so an event of one family never
        passes on an open id of another, and nothing is left open."""
        types = [event["type"] for event in events]
        assert types[0] == "RUN_STARTED" and types.count("RUN_STARTED") == 1
        ends = [kind for kind in types if kind in ("RUN_FINISHED", "RUN_ERROR")]
        assert ends == ["RUN_FINISHED"] and types[-1] == "RUN_FINISHED"
        calls: dict[str, str] = {}  # id -> "started" | "ended"
        opened: dict[tuple[str, str], str] = {}  # (family, id) -> "open" | "closed"
        families = {"TEXT_MESSAGE": "text", "REASONING_MESSAGE": "reasoning message"}
        for event in events:
            kind = event["type"]
            if kind == "TOOL_CALL_START":
                assert event["toolCallId"] not in calls, event
                calls[event["toolCallId"]] = "started"
            elif kind == "TOOL_CALL_ARGS":
                assert calls.get(event["toolCallId"]) == "started", event
            elif kind == "TOOL_CALL_END":
                assert calls.get(event["toolCallId"]) == "started", event
                calls[event["toolCallId"]] = "ended"
            elif kind == "TOOL_CALL_RESULT":
                assert calls.get(event["toolCallId"]) == "ended", event
            elif kind == "REASONING_START":
                assert ("reasoning", event["messageId"]) not in opened, event
                opened["reasoning", event["messageId"]] = "open"
            elif kind == "REASONING_END":
                assert opened.get(("reasoning", event["messageId"])) == "open", event
                message = opened.get(("reasoning message", event["messageId"]))
                assert message in (None, "closed"), event  # its message ended first
                opened["reasoning", event["messageId"]] = "closed"
            elif (family := families.get(kind.rsplit("_", 1)[0])) is not None:
                key = (family, event["messageId"])
                if family == "reasoning message":  # only inside its open reasoning phase
                    assert opened.get(("reasoning", event["messageId"])) == "open", event
                if kind.endswith("_START"):
                    assert key not in opened, event
                    opened[key] = "open"
                else:
                    assert opened.get(key) == "open", event
                    if kind.endswith("_END"):
                        opened[key] = "closed"
        assert set(calls.values()) <= {"ended"} and set(opened.values()) <= {"closed"}

    native = await kinds("native")
    check_rules(native)
    started = next(event for event in native if event["type"] == "SUBAGENT_STARTED")
    assert started["subagentRunId"].startswith("ext:fake-session-1:")
    tagged = [e for e in native if e.get("subagentRunId") == started["subagentRunId"]]
    reasoning = [
        "REASONING_START",
        "REASONING_MESSAGE_START",
        "REASONING_MESSAGE_CONTENT",
        "REASONING_MESSAGE_END",
        "REASONING_END",
    ]
    # The agent's own tool call arrives without a model response (ACP has none): the
    # projector announces it from tool_call_started.
    assert [e["type"] for e in tagged] == [
        "SUBAGENT_STARTED",
        *reasoning,
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
        "TOOL_CALL_RESULT",
        *reasoning,
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "SUBAGENT_FINISHED",
    ]
    start = next(e for e in tagged if e["type"] == "TOOL_CALL_START")
    assert (start["toolCallId"], start["toolCallName"]) == (APPROVED_CALL, "coder.bash")
    result = next(e for e in tagged if e["type"] == "TOOL_CALL_RESULT")
    assert (result["toolCallId"], result["content"]) == (APPROVED_CALL, "escalated\n")
    text = next(e for e in tagged if e["type"] == "TEXT_MESSAGE_CONTENT")
    assert text["delta"] == "DONE"

    custom = await kinds("custom")
    check_rules(custom)
    values = [e["value"] for e in custom if e["type"] == "CUSTOM" and e["name"] == "subagent_event"]
    inner_types = [value["event"]["type"] for value in values]
    assert inner_types[0] == "run_started"
    assert inner_types.count("external_event") == 3  # requested, decided, turn_finished
    assert "tool_call_started" in inner_types and "tool_call_finished" in inner_types


# -- 3. permissions --------------------------------------------------------------------------------


async def test_on_ask_sees_the_correlated_arguments_and_approves(tmp_path: Path) -> None:
    seen: list[tuple[str, dict, str]] = []

    def on_ask(ctx: Any, tool: Any, call: Any, reason: Any) -> bool:
        seen.append((tool.name, call.arguments, call.id))
        return True

    agent = make(tmp_path, "escalation-approved", permissions=Permissions(on_ask=on_ask))
    async with agent:
        _, result = await play(agent)
    assert seen[0][0] == "coder.bash" and seen[0][2] == APPROVED_CALL
    assert seen[0][1]["command"] == BASH_COMMAND  # rawInput of the earlier tool_call update
    assert (result.permissions[0].outcome, result.permissions[0].decided_by) == ("allow", "on_ask")
    assert result.text == "DONE"


@pytest.mark.parametrize(
    ("permissions", "outcome", "decided_by", "answer"),
    [
        (
            Permissions(on_ask=lambda *_: Approval(approved=False, reason="no")),
            "deny",
            "on_ask",
            "reject-once",
        ),
        (Permissions(), "deny", "no_approver", "reject-once"),
        (Permissions(can_use_tool=lambda *_: 1 / 0), "deny", "rules", "reject-once"),
        (
            Permissions(on_ask=lambda *_: Approval(approved=True, arguments={})),
            "deny",
            "edited_arguments",
            "reject-once",
        ),
    ],
    ids=["on_ask declines", "no approver", "raising can_use_tool", "edited arguments"],
)
async def test_refusals(
    tmp_path: Path, permissions: Permissions, outcome: str, decided_by: str, answer: str
) -> None:
    agent = make(tmp_path, "escalation-approved", permissions=permissions)
    async with agent:
        _, result = await play(agent)
    assert (result.permissions[0].outcome, result.permissions[0].decided_by) == (
        outcome,
        decided_by,
    )
    assert sent(tmp_path)[-1]["result"] == {"outcome": {"outcome": "selected", "optionId": answer}}
    # the rejected recording was played (its tool call id differs from the approved one)
    assert result.completed and result.text.startswith("The user rejected")
    assert result.tool_calls[-1].is_error


async def test_a_failing_decision_or_a_timeout_answers_cancelled(tmp_path: Path) -> None:
    class Broken(Permissions):
        async def decide(self, ctx: Any, tool: Any, call: Any) -> Any:
            raise RuntimeError("rules store unreachable")

    async def slow(*_: Any) -> bool:
        await asyncio.sleep(30)
        return True

    for permissions, decided_by in ((Broken(), "error"), (Permissions(on_ask=slow), "timeout")):
        agent = make(
            tmp_path, "escalation-approved", permissions=permissions, approval_timeout_s=0.3
        )
        log_path(tmp_path).unlink(missing_ok=True)
        async with agent:
            _, result = await play(agent)
        assert (result.permissions[0].outcome, result.permissions[0].decided_by) == (
            "cancel",
            decided_by,
        )
        assert sent(tmp_path)[-1]["result"] == {"outcome": {"outcome": "cancelled"}}


async def test_allow_always_is_never_chosen(tmp_path: Path) -> None:
    rows = lines_of("escalation-approved")[1:]
    rows[7] = rows[7].replace(
        '"optionId":"allow-once","name":"Allow once","kind":"allow_once"',
        ('"optionId":"always","name":"Always","kind":"allow_always"'),
    )
    rows[8] = json.dumps(
        {
            "answer": {"id": 1, "accept": [{"outcome": {"outcome": "cancelled"}}]},
            "otherwise": str(FIXTURES / "escalation-rejected.jsonl"),
        }
    )
    agent = make(tmp_path, write_script(tmp_path, rows), permissions=Permissions(allow=["coder.*"]))
    async with agent:
        _, result = await play(agent)
    assert (result.permissions[0].outcome, result.permissions[0].decided_by) == (
        "cancel",
        "no_option",
    )
    assert sent(tmp_path)[-1]["result"] == {"outcome": {"outcome": "cancelled"}}


async def test_requests_without_a_turn_or_a_tool_call_are_cancelled(tmp_path: Path) -> None:
    def ask(request_id: int, params: dict[str, Any]) -> list[Any]:
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "session/request_permission",
            "params": params,
        }
        accept = [{"outcome": {"outcome": "cancelled"}}]
        return [request, {"answer": {"id": request_id, "accept": accept}}]

    options = [{"optionId": "allow-once", "name": "Allow", "kind": "allow_once"}]
    rows = [
        *HANDSHAKE,
        PROMPT,
        *ask(1, {"sessionId": "another", "toolCall": {"toolCallId": "x"}, "options": options}),
        *ask(2, {"sessionId": "{{sessionId}}", "toolCall": {}, "options": options}),
        *ask(3, {"sessionId": "{{sessionId}}", "options": options}),
        chunk("Stopped."),
        stop(),
    ]
    agent = make(tmp_path, write_script(tmp_path, rows), permissions=Permissions(allow=["coder.*"]))
    async with agent:
        _, result = await play(agent)
    assert result.completed and result.permissions == []
    assert result.raw["malformed_permission_requests"] == 2


async def test_a_request_that_describes_its_tool_call(tmp_path: Path) -> None:
    tool_call = {
        "toolCallId": "c9",
        "title": "Run ls",
        "kind": "execute",
        "rawInput": {"command": "ls"},
    }
    options = [
        {"optionId": "yes", "name": "Allow", "kind": "allow_once"},
        {"optionId": "no", "name": "Reject", "kind": "reject_once"},
    ]
    rows = [
        *HANDSHAKE,
        PROMPT,
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "session/request_permission",
            "params": {"sessionId": "{{sessionId}}", "toolCall": tool_call, "options": options},
        },
        {"answer": {"id": 7, "accept": [{"outcome": {"outcome": "selected", "optionId": "yes"}}]}},
        update("tool_call_update", toolCallId="c9", status="completed", content=[]),
        stop(),
    ]
    rules = Permissions(allow=["coder.execute"])
    agent = make(tmp_path, write_script(tmp_path, rows), permissions=rules)
    async with agent:
        events, result = await play(agent)
    assert describe(events[1]) == "tool_call_started coder.execute c9"
    assert result.permissions[0].arguments == {"command": "ls"}
    assert result.permissions[0].approved


async def test_a_pending_request_is_answered_cancelled_when_the_turn_is_cancelled(
    tmp_path: Path,
) -> None:
    never = asyncio.Event()

    async def waits(*_: Any) -> bool:
        await never.wait()
        return True

    rows = [
        *lines_of("escalation-approved")[1:9],  # up to the permission request
        {"answer": {"id": 1, "accept": [{"outcome": {"outcome": "cancelled"}}]}},
        {"expect": {"method": "session/cancel", "params": {"sessionId": "{{sessionId}}"}}},
        stop(reason="cancelled"),
    ]
    agent = make(tmp_path, write_script(tmp_path, rows), permissions=Permissions(on_ask=waits))
    async with agent:
        turn = agent.stream("t")
        async with turn:
            async for event in turn:
                if isinstance(event, ExternalAgentEvent) and event.kind == "permission_requested":
                    turn.cancel()
        result = turn.result
    assert result is not None
    assert (result.status, result.stop_reason) == ("cancelled", "cancelled")
    assert (result.permissions[0].outcome, result.permissions[0].decided_by) == (
        "cancel",
        "cancelling",
    )
    assert "<answer>" in methods(tmp_path) and "session/cancel" in methods(tmp_path)


async def test_an_approval_that_comes_with_the_cancel_is_answered_cancelled(
    tmp_path: Path,
) -> None:
    turns: list[Any] = []

    def approves_as_the_turn_is_cancelled(*_: Any) -> bool:
        turns[0].cancel()  # say the as_tool timeout fires just as the human approves
        return True

    rows = [
        *lines_of("escalation-approved")[1:9],  # up to the permission request
        {"answer": {"id": 1, "accept": [{"outcome": {"outcome": "cancelled"}}]}},
        {"expect": {"method": "session/cancel", "params": {"sessionId": "{{sessionId}}"}}},
        stop(reason="cancelled"),
    ]
    rules = Permissions(on_ask=approves_as_the_turn_is_cancelled)
    agent = make(tmp_path, write_script(tmp_path, rows), permissions=rules)
    async with agent:
        turn = agent.stream("t")
        turns.append(turn)
        result = await turn.wait()
        assert agent.connected
    assert (result.status, result.stop_reason) == ("cancelled", "cancelled"), result.raw
    assert (result.permissions[0].outcome, result.permissions[0].decided_by) == (
        "cancel",
        "cancelling",
    )
    answers = [message["result"] for message in sent(tmp_path) if "result" in message]
    assert answers == [{"outcome": {"outcome": "cancelled"}}]  # never "selected allow-once"


# -- 5. cancellation -----------------------------------------------------------------------------------


async def test_cancel_sends_session_cancel_and_ends_cancelled(tmp_path: Path) -> None:
    agent = make(tmp_path, "cancel")
    async with agent:
        turn = agent.stream("Start a long task.")
        async with turn:
            async for event in turn:
                if isinstance(event, TextDelta):
                    turn.cancel()
        assert agent.connected  # the agent stopped by itself: the process is kept
    result = turn.result
    assert result is not None
    assert (result.status, result.stop_reason, result.text) == ("cancelled", "cancelled", "partial")
    assert "forced" not in result.raw
    cancels = [m for m in sent(tmp_path) if m.get("method") == "session/cancel"]
    assert cancels == [
        {"jsonrpc": "2.0", "method": "session/cancel", "params": {"sessionId": "fake-session-1"}}
    ]


async def test_a_cancel_during_the_prompt_checkpoint_sends_no_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    turns: list[Any] = []
    journal = ExternalTurnContext.checkpoint

    async def slow_journal(self: ExternalTurnContext, key: str, value: Any) -> bool:
        if key == "acp_prompt_sent":  # a SQLite append with an fsync, say
            turns[0].cancel()  # the cancel comes in while the write is in flight
            await asyncio.sleep(0.2)
        return await journal(self, key, value)

    monkeypatch.setattr(ExternalTurnContext, "checkpoint", slow_journal)
    agent = make(tmp_path, "handshake", resumable=True, cancel_grace_s=1.0)
    turn = agent._open_turn(
        "Fix the parser.",
        session=None,
        cwd=None,
        parent=None,
        permissions=None,
        label=None,
        resumable=True,
    )
    turns.append(turn)
    async with agent:
        result = await turn.wait()
        assert agent.connected  # stopped without a kill
    assert (result.status, result.stop_reason) == ("cancelled", "cancelled")
    assert "forced" not in result.raw
    # neither a session/cancel for a session with no prompt, nor the prompt after it
    assert methods(tmp_path) == ["initialize", "session/new"]


async def test_a_cancel_during_session_setup_keeps_the_process(tmp_path: Path) -> None:
    rows = [
        *HANDSHAKE[:3],  # session/new (request 2) ...
        {"sleep": 1.5},  # ... is slow: the agent starts the session's MCP servers
        HANDSHAKE[3],  # its answer comes after qagents gave the request up: dropped
        {"expect": {"method": "$/cancel_request", "params": {"requestId": 2}}},
        *new_session(3, "s2"),  # the same process serves the next turn
        *prompted(4, "s2", "second"),
    ]
    agent = make(tmp_path, write_script(tmp_path, rows), cancel_grace_s=0.5)
    async with agent:
        process = agent._process
        turn = agent.stream("First.")

        async def cancel_once_sent() -> None:
            for _ in range(500):  # until the agent has read session/new (10 s at most)
                if "session/new" in methods(tmp_path):
                    break
                await asyncio.sleep(0.02)
            turn.cancel()

        canceller = asyncio.create_task(cancel_once_sent())
        result = await turn.wait()
        await canceller
        assert (result.status, result.stop_reason) == ("cancelled", "cancelled")
        assert "forced" not in result.raw  # it stopped at once, not after cancel_grace_s
        assert agent.connected and process is not None and process.returncode is None
        second = await agent.run("Second.")
        assert agent._process is process
    assert (second.status, second.text) == ("completed", "second")
    assert "session/prompt" in methods(tmp_path)  # only the second turn's
    assert methods(tmp_path).count("session/prompt") == 1


async def test_cancel_with_open_tool_calls(tmp_path: Path) -> None:
    agent = make(tmp_path, "cancel-tool-calls", permissions=Permissions(allow=["coder.*"]))
    async with agent:
        turn = agent.stream("Run two commands.")
        async with turn:
            events = []
            async for event in turn:
                events.append(event)
                if isinstance(event, ToolCallStarted):
                    turn.cancel()
    assert [describe(event) for event in events] == [
        "run_started fake-session-1 resumed=False depth=0",
        "tool_call_started coder.bash call_wait",
        "tool_call_finished coder.bash failed 'Error: tool call aborted'",
        "tool_call_started coder.bash call_skipped",
        "tool_call_finished coder.bash failed 'Error: tool call aborted before dispatch'",
        "turn_finished cancelled cancelled",
    ]
    assert turn.result is not None and len(turn.result.tool_calls) == 2


async def test_an_agent_that_does_not_stop_is_killed_with_its_process_tree(
    tmp_path: Path,
) -> None:
    heartbeat = tmp_path / "heartbeat"
    agent = make(
        tmp_path,
        "cancel",
        "--hang-after-cancel",
        "--child",
        str(heartbeat),
        cancel_grace_s=0.5,
        close_grace_s=0.5,
    )
    async with agent:
        turn = agent.stream("Start a long task.")
        async with turn:
            async for event in turn:
                if isinstance(event, TextDelta):
                    assert await _beating(heartbeat), "the agent's child should be running"
                    turn.cancel()
        result = turn.result
        assert result is not None
        assert (result.status, result.raw.get("forced")) == ("cancelled", True)
        assert not agent.connected  # the next turn starts a new process
        child = _pid_of(heartbeat)
        assert not await _beating(heartbeat), "the agent's child outlived the kill"
        assert not _alive(child)


async def test_cancelling_the_calling_run_cancels_the_turn(tmp_path: Path) -> None:
    agent = make(tmp_path, "cancel")
    lead = Agent(ScriptedModel([call("coder", task="t")]), tools=[agent.as_tool()])
    async with agent:
        stream = lead.stream("go")
        async for event in stream:
            if isinstance(event, SubagentEvent) and isinstance(event.event, TextDelta):
                stream.cancel()
    assert stream.result is not None and stream.result.status == "cancelled"
    assert "session/cancel" in methods(tmp_path)


async def test_as_tool_timeout_cancels_the_turn(tmp_path: Path) -> None:
    agent = make(tmp_path, "cancel")
    lead = Agent(ScriptedModel([call("coder", task="t"), "ok"]), tools=[agent.as_tool(timeout=1)])
    async with agent:
        result = await lead.run("go")
    content = [part for m in result.messages for part in m.tool_results][0].content
    assert content.startswith(texts.EXTERNAL_AGENT_STOPPED.format(name="coder", reason="timeout"))
    assert "session fake-session-1" in content
    assert "session/cancel" in methods(tmp_path)


# -- 8. errors ----------------------------------------------------------------------------------------


async def test_a_failed_turn_is_an_error_with_the_agents_message(tmp_path: Path) -> None:
    failure = {"code": -32603, "message": "Internal error: turn failed: model route unavailable"}
    rows = [*HANDSHAKE, PROMPT, chunk("Trying."), {"jsonrpc": "2.0", "id": 3, "error": failure}]
    path = write_script(tmp_path, rows)
    agent = make(tmp_path, path)
    async with agent:
        _, result = await play(agent)
        assert agent.connected
    assert (result.status, result.stop_reason) == ("error", "turn failed: model route unavailable")
    assert result.raw["error_code"] == -32603

    agent = make(tmp_path, path)
    lead = Agent(ScriptedModel([call("coder", task="t"), "ok"]), tools=[agent.as_tool()])
    async with agent:
        run = await lead.run("go")
    content = [part for m in run.messages for part in m.tool_results][0].content
    assert content.startswith("coder failed: turn failed: model route unavailable")


async def test_session_new_refused_with_invalid_params(tmp_path: Path) -> None:
    agent = make(tmp_path, "reject-extra-dirs")
    async with agent:
        with pytest.raises(ExternalAgentConfigurationError, match="refused session/new"):
            await play(agent)
    agent = make(tmp_path, "reject-extra-dirs")
    lead = Agent(ScriptedModel([call("coder", task="t"), "ok"]), tools=[agent.as_tool()])
    async with agent:
        run = await lead.run("go")
    content = [part for m in run.messages for part in m.tool_results][0].content
    assert content.startswith("coder is unavailable: coder refused session/new: Invalid params")


async def test_end_of_stream_mid_turn_is_interrupted_and_ambiguous(tmp_path: Path) -> None:
    rows = [*HANDSHAKE, PROMPT, chunk("partial"), {"exit": 7, "stderr": "segfault in the loop"}]
    agent = make(tmp_path, write_script(tmp_path, rows))
    async with agent:
        _, result = await play(agent)
        assert not agent.connected
    assert (result.status, result.stop_reason) == ("interrupted", "the process exited with code 7")
    assert result.raw["ambiguous"] is True and result.raw["returncode"] == 7
    assert "segfault" in result.raw["stderr_tail"]
    assert result.text == "partial"

    agent = make(tmp_path, write_script(tmp_path, rows))
    lead = Agent(ScriptedModel([call("coder", task="t"), "ok"]), tools=[agent.as_tool()])
    async with agent:
        run = await lead.run("go")
    content = [part for m in run.messages for part in m.tool_results][0].content
    assert "outcome is unknown" in content and "session=fake-session-1" in content
    assert "segfault" not in content  # stderr never reaches the model


async def test_a_few_malformed_lines_are_skipped_three_in_a_row_end_the_turn(
    tmp_path: Path,
) -> None:
    rows = [*HANDSHAKE, PROMPT, {"raw": "not json"}, {"raw": "[]"}, chunk("still here"), stop()]
    agent = make(tmp_path, write_script(tmp_path, rows))
    async with agent:
        _, result = await play(agent)
    assert (result.status, result.text) == ("completed", "still here")

    garbage = [{"raw": "not json"}, {"raw": '{"no": "jsonrpc"}'}, {"raw": "{"}]
    rows = [*HANDSHAKE, PROMPT, *garbage, chunk("late"), stop()]
    agent = make(tmp_path, write_script(tmp_path, rows))
    async with agent:
        _, result = await play(agent)
    assert (result.status, result.stop_reason) == (
        "interrupted",
        "the agent sent malformed messages",
    )


async def test_a_20_mib_line_is_accepted_and_a_line_over_the_limit_is_not(tmp_path: Path) -> None:
    size = 20 * 1024 * 1024
    rows = [*HANDSHAKE, PROMPT, {"big_chunk": size}, stop()]
    agent = make(tmp_path, write_script(tmp_path, rows))
    async with agent:
        _, result = await play(agent)
    assert result.status == "completed" and len(result.text) == size

    rows = [*HANDSHAKE, PROMPT, {"big_chunk": 2 * 1024 * 1024}, stop()]
    agent = make(tmp_path, write_script(tmp_path, rows), max_line_bytes=1024 * 1024)
    async with agent:
        _, result = await play(agent)
    assert (result.status, result.stop_reason) == (
        "interrupted",
        "the agent sent a message that is too large",
    )


async def test_an_answer_of_the_wrong_shape(tmp_path: Path) -> None:
    rows = [*HANDSHAKE, PROMPT, {"jsonrpc": "2.0", "id": 3, "result": {"stop": "end_turn"}}]
    agent = make(tmp_path, write_script(tmp_path, rows))
    async with agent:
        _, result = await play(agent)
    assert (result.status, result.stop_reason) == ("interrupted", acp_texts.ACP_BAD_ANSWER)


async def test_a_setup_request_without_an_answer(tmp_path: Path) -> None:
    rows = [HANDSHAKE[0], HANDSHAKE[1], HANDSHAKE[2], {"hang": True}]
    agent = make(tmp_path, write_script(tmp_path, rows), request_timeout_s=0.3, close_grace_s=0.3)
    async with agent:
        _, result = await play(agent)
    # the task was never sent: the outcome is known, not "interrupted, check what it changed"
    assert (result.status, "ambiguous" in result.raw) == ("error", False)
    assert result.stop_reason == acp_texts.ACP_NO_ANSWER.format(method="session/new", seconds=0.3)


async def test_an_agent_that_exits_during_session_new_ends_the_turn_as_error(
    tmp_path: Path,
) -> None:
    rows = [*HANDSHAKE[:3], {"exit": 1, "stderr": "mcp server failed to start"}]
    agent = make(tmp_path, write_script(tmp_path, rows))
    async with agent:
        _, result = await play(agent)
        assert not agent.connected  # the next turn starts a new process
    assert (result.status, result.stop_reason) == ("error", "the process exited with code 1")
    assert "ambiguous" not in result.raw

    agent = make(tmp_path, write_script(tmp_path, rows))
    lead = Agent(ScriptedModel([call("coder", task="t"), "ok"]), tools=[agent.as_tool()])
    async with agent:
        run = await lead.run("go")
    content = [part for m in run.messages for part in m.tool_results][0].content
    assert content.startswith("coder failed: the process exited with code 1")
    assert "outcome is unknown" not in content


async def test_not_connected_is_not_ambiguous_unless_a_repeated_call(tmp_path: Path) -> None:
    agent = make(tmp_path, "handshake")
    for checkpoints, ambiguous in (
        (None, False),
        ({"external_session": {"session_id": "old", "cwd": None}}, True),  # a repeated call
    ):
        turn = agent._open_turn(
            "Fix the parser.",
            session=None,
            cwd=None,
            parent=None,
            permissions=None,
            label=None,
            checkpoints=checkpoints,
            resumable=checkpoints is not None,
        )
        with pytest.raises(ExternalAgentConnectionError, match="not connected") as caught:
            await agent.run_turn(turn.context)  # never connected
        assert caught.value.ambiguous is ambiguous


@pytest.mark.parametrize(
    ("stop_reason", "status", "mapped"),
    [
        ("max_tokens", "limit_reached", "max_tokens"),
        ("max_turn_requests", "limit_reached", "max_turn_requests"),
        ("refusal", "completed", "content_filter"),
        ("something_new", "completed", "something_new"),
    ],
)
async def test_stop_reasons(tmp_path: Path, stop_reason: str, status: str, mapped: str) -> None:
    rows = [*HANDSHAKE, PROMPT, chunk("x"), stop(reason=stop_reason)]
    agent = make(tmp_path, write_script(tmp_path, rows))
    async with agent:
        _, result = await play(agent)
    assert (result.status, result.stop_reason) == (status, mapped)


# -- sessions, configuration, MCP servers ---------------------------------------------------------


def _resume_rows(method: str, **init: Any) -> list[Any]:
    return [
        HANDSHAKE[0],
        initialize_answer(**init),
        {
            "expect": {
                "id": 2,
                "method": method,
                "params": {"sessionId": "{{sessionId}}", "cwd": "$absolute", "mcpServers": []},
            }
        },
    ]


async def test_an_older_session_is_resumed(tmp_path: Path) -> None:
    rows = [
        *_resume_rows("session/resume"),
        {"jsonrpc": "2.0", "id": 2, "result": {"configOptions": []}},
        PROMPT,
        chunk("Continued."),
        stop(),
    ]
    agent = make(tmp_path, write_script(tmp_path, rows), "--session-id", "old-session")
    async with agent:
        events, result = await play(agent, session="old-session")
    assert describe(events[0]) == "run_started old-session resumed=True depth=0"
    assert (result.session_id, result.text) == ("old-session", "Continued.")


async def test_session_load_is_used_without_resume_and_its_replay_is_ignored(
    tmp_path: Path,
) -> None:
    rows = [
        *_resume_rows("session/load", loadSession=True, sessionCapabilities={}),
        update("user_message_chunk", content={"type": "text", "text": "old question"}),
        chunk("old answer"),
        {"jsonrpc": "2.0", "id": 2, "result": {}},
        PROMPT,
        chunk("new answer"),
        stop(),
    ]
    agent = make(tmp_path, write_script(tmp_path, rows), "--session-id", "old-session")
    async with agent:
        caps = await agent.capabilities()
        events, result = await play(agent, session="old-session")
    assert caps.resume_sessions
    assert result.text == "new answer"
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["new answer"]


async def test_a_session_that_cannot_be_continued(tmp_path: Path) -> None:
    rows = [HANDSHAKE[0], initialize_answer(sessionCapabilities={})]
    agent = make(tmp_path, write_script(tmp_path, rows))
    async with agent:
        assert not (await agent.capabilities()).resume_sessions
        _, result = await play(agent, session="elsewhere")
    assert result.status == "error"
    assert result.stop_reason == acp_texts.ACP_CANNOT_CONTINUE.format(
        session="elsewhere", reason=acp_texts.ACP_NO_SESSION_RESUME
    )

    refused = {"code": -32602, "message": "Invalid params: session is not resumable: gone"}
    rows = [*_resume_rows("session/resume"), {"jsonrpc": "2.0", "id": 2, "error": refused}]
    agent = make(tmp_path, write_script(tmp_path, rows), "--session-id", "gone")
    async with agent:
        _, result = await play(agent, session="gone")
    assert result.status == "error"
    assert result.stop_reason == (
        "cannot continue session gone: Invalid params: session is not resumable: gone"
    )


async def test_the_model_is_selected_by_name_or_value(tmp_path: Path) -> None:
    pro = '["deepseek-official","deepseek-v4-pro"]'
    options = json.loads(HANDSHAKE[3])["result"]["configOptions"]
    options[0]["currentValue"] = pro
    rows = [
        *HANDSHAKE,
        {
            "expect": {
                "id": 3,
                "method": "session/set_config_option",
                "params": {"sessionId": "{{sessionId}}", "configId": "model", "value": pro},
            }
        },
        {"jsonrpc": "2.0", "id": 3, "result": {"configOptions": options}},
        {"expect": {"id": 4, "method": "session/prompt", "params": {"prompt": "$list"}}},
        chunk("ok"),
        stop(4),
    ]
    path = write_script(tmp_path, rows)
    for model in ("deepseek-v4-pro", ("deepseek-official", "deepseek-v4-pro"), pro):
        log_path(tmp_path).unlink(missing_ok=True)
        agent = make(tmp_path, path, model=model)
        async with agent:
            _, result = await play(agent)
            assert agent.session_config(result.session_id)[0]["currentValue"] == pro
        assert result.text == "ok", result.raw

    rows = [*HANDSHAKE, PROMPT, chunk("ok"), stop()]
    log_path(tmp_path).unlink(missing_ok=True)
    agent = make(tmp_path, write_script(tmp_path, rows), model="deepseek-v4-flash")
    async with agent:
        _, result = await play(agent)  # already selected: nothing is sent
    assert "session/set_config_option" not in methods(tmp_path)


async def test_an_option_that_is_not_offered_is_refused(tmp_path: Path) -> None:
    agent = make(tmp_path, "handshake", model="gpt-9")
    async with agent:
        with pytest.raises(ExternalAgentConfigurationError, match="deepseek-v4-pro"):
            await play(agent)
    agent = make(tmp_path, "handshake", config_options={"reasoning_effort": "high"})
    async with agent:
        with pytest.raises(ExternalAgentConfigurationError, match="reasoning_effort"):
            await play(agent)


async def test_mcp_servers_are_sent_with_each_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QAGENTS_ACP_TEST_VALUE", "from-parent")
    stdio = MCPServerStdio(
        "tools",
        command=sys.executable,
        args=["-m", "tools"],
        env={"MODE": "test"},
        passthrough=["QAGENTS_ACP_TEST_VALUE", "QAGENTS_ACP_NOT_SET"],
    )
    http = MCPServerHTTP("qagents", url="http://127.0.0.1:9/mcp", headers={"Authorization": "x"})
    rows = [*HANDSHAKE, PROMPT, chunk("ok"), stop()]
    seen: list[Any] = []

    def servers(turn: Any) -> list[Any]:
        seen.append(turn.cwd)
        return [stdio, http]

    agent = make(tmp_path, write_script(tmp_path, rows), mcp_servers=servers)
    async with agent:
        await play(agent)
    assert seen == [str(tmp_path / "workspace")]
    new = next(m for m in sent(tmp_path) if m.get("method") == "session/new")
    assert new["params"]["mcpServers"] == [
        {
            "name": "tools",
            "command": sys.executable,
            "args": ["-m", "tools"],
            "env": [
                {"name": "MODE", "value": "test"},
                {"name": "QAGENTS_ACP_TEST_VALUE", "value": "from-parent"},
            ],
        },
        {
            "type": "http",
            "name": "qagents",
            "url": "http://127.0.0.1:9/mcp",
            "headers": [{"name": "Authorization", "value": "x"}],
        },
    ]
    assert "Authorization" not in repr(http) and "test" not in repr(stdio)


async def test_a_servers_function_runs_only_when_servers_are_sent(tmp_path: Path) -> None:
    rows = [
        HANDSHAKE[0],
        HANDSHAKE[1],
        *new_session(2, "s1"),
        *prompted(3, "s1", "one"),
        *prompted(4, "s1", "two"),  # continued directly: nothing is mounted
    ]
    calls: list[str] = []

    def servers(turn: ExternalTurnContext) -> list[Any]:
        if calls:
            raise RuntimeError("would start a server that is never mounted")
        calls.append(turn.prompt_text)
        return []

    agent = make(tmp_path, write_script(tmp_path, rows), mcp_servers=servers)
    async with agent:
        first = await agent.run("One.")
        second = await agent.run("Two.", session=first.session_id)
    assert (first.text, second.status, second.text) == ("one", "completed", "two")
    assert calls == ["One."]


def _turn_servers(log: list[Any], tmp_path: Path) -> Any:
    """``mcp_servers=`` that gives each turn its own server, as an async context manager."""

    @contextlib.asynccontextmanager
    async def servers(turn: ExternalTurnContext) -> Any:
        number = sum(1 for entry in log if entry[0] == "start") + 1
        log.append(("start", number))
        try:
            yield [MCPServerHTTP("tools", url=f"http://127.0.0.1:9/mcp/{number}")]
        finally:  # how many closes the agent had read when the server stopped
            log.append(("stop", number, methods(tmp_path).count("session/close")))

    return servers


async def test_servers_that_stop_with_the_turn_get_a_fresh_session_each_turn(
    tmp_path: Path,
) -> None:
    rows = [
        HANDSHAKE[0],
        HANDSHAKE[1],  # Devvi's: sessionCapabilities close and resume
        *new_session(2, "s1"),
        *prompted(3, "s1", "one"),
        *closed(4, "s1"),  # right after the turn, before its server stops
        *resumed(5, "s1"),
        *prompted(6, "s1", "two"),
        *closed(7, "s1"),
    ]
    log: list[Any] = []
    agent = make(tmp_path, write_script(tmp_path, rows), mcp_servers=_turn_servers(log, tmp_path))
    async with agent:
        first = await agent.run("One.")
        second = await agent.run("Two.", session=first.session_id)
    assert (first.text, second.text) == ("one", "two")
    assert log == [("start", 1), ("stop", 1, 1), ("start", 2), ("stop", 2, 2)]
    opened = [m for m in sent(tmp_path) if m.get("method") in ("session/new", "session/resume")]
    assert [m["params"]["mcpServers"][0]["url"] for m in opened] == [
        "http://127.0.0.1:9/mcp/1",
        "http://127.0.0.1:9/mcp/2",
    ]


async def test_servers_that_stop_with_the_turn_on_an_agent_that_cannot_close(
    tmp_path: Path,
) -> None:
    rows = [
        HANDSHAKE[0],
        initialize_answer(sessionCapabilities={"resume": {}}),  # no close
        *new_session(2, "s1"),
        *prompted(3, "s1", "one"),
        *resumed(4, "s1"),  # not prompted directly: the first turn's server is gone
        *prompted(5, "s1", "two"),
    ]
    log: list[Any] = []
    agent = make(tmp_path, write_script(tmp_path, rows), mcp_servers=_turn_servers(log, tmp_path))
    async with agent:
        first = await agent.run("One.")
        second = await agent.run("Two.", session=first.session_id)
    assert (first.text, second.text) == ("one", "two")
    assert [entry[:2] for entry in log] == [("start", 1), ("stop", 1), ("start", 2), ("stop", 2)]
    assert "session/close" not in methods(tmp_path)


async def test_mcp_servers_the_agent_cannot_mount_are_refused(tmp_path: Path) -> None:
    rows = [HANDSHAKE[0], initialize_answer(mcpCapabilities={})]
    http = MCPServerHTTP("qagents", url="http://127.0.0.1:9/mcp")
    agent = make(tmp_path, write_script(tmp_path, rows), mcp_servers=[http])
    with pytest.raises(ExternalAgentConfigurationError, match="mcpCapabilities.http"):
        await agent.capabilities()
    with pytest.raises(ConfigurationError, match="absolute path"):
        MCPServerStdio("tools", command="python")
    with pytest.raises(ConfigurationError, match="http"):
        MCPServerHTTP("tools", url="ftp://host/mcp")
    with pytest.raises(ConfigurationError, match="MCPServerStdio"):
        ACPAgent([sys.executable], mcp_servers=["tools"])  # type: ignore[list-item]


async def test_a_turn_without_a_workspace_is_refused(tmp_path: Path) -> None:
    agent = make(tmp_path, "handshake", workspace=None)
    agent.workspace = None
    async with agent:
        with pytest.raises(ExternalAgentConfigurationError, match="working directory"):
            await play(agent)


# -- idle sessions are closed (keep_sessions) ------------------------------------------------------


async def test_idle_sessions_beyond_keep_sessions_are_closed(tmp_path: Path) -> None:
    rows = [
        HANDSHAKE[0],
        HANDSHAKE[1],  # Devvi's: sessionCapabilities close and resume
        *new_session(2, "s1"),
        *prompted(3, "s1", "one"),
        *new_session(4, "s2"),
        *prompted(5, "s2", "two"),
        *closed(6, "s1"),  # two idle sessions, one is kept: the older one is closed
        *resumed(7, "s1"),
        *prompted(8, "s1", "three"),
        *closed(9, "s2"),  # s1 is now the last one used
    ]
    agent = make(tmp_path, write_script(tmp_path, rows))
    assert agent.keep_sessions == 1
    async with agent:
        first = await agent.run("One.")
        second = await agent.run("Two.")
        third = await agent.run("Three.", session=first.session_id)
        assert agent.connected
    assert [(r.status, r.session_id, r.text) for r in (first, second, third)] == [
        ("completed", "s1", "one"),
        ("completed", "s2", "two"),
        ("completed", "s1", "three"),
    ]
    assert methods(tmp_path) == [
        "initialize",
        "session/new",
        "session/prompt",
        "session/new",
        "session/prompt",
        "session/close",
        "session/resume",
        "session/prompt",
        "session/close",
    ]


async def test_keep_sessions_0_closes_each_session_and_a_failed_close_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    failure = {"code": -32603, "message": "Internal error: session close failed: disk full"}
    rows = [
        HANDSHAKE[0],
        HANDSHAKE[1],
        *new_session(2, "s1"),
        *prompted(3, "s1", "one"),
        *closed(4, "s1", failure),
        *resumed(5, "s1"),
        *prompted(6, "s1", "two"),
        *closed(7, "s1"),
    ]
    agent = make(tmp_path, write_script(tmp_path, rows), keep_sessions=0)
    with caplog.at_level(logging.WARNING, logger="qagents.integrations.acp"):
        async with agent:
            first = await agent.run("One.")
            second = await agent.run("Two.", session=first.session_id)
            assert agent.connected  # a failed close is logged, nothing more
    assert (first.status, second.status, second.text) == ("completed", "completed", "two")
    assert "session close failed: disk full" in caplog.text
    assert methods(tmp_path).count("session/close") == 2


@pytest.mark.parametrize(
    "capabilities",
    [{"resume": {}}, {"close": {}}],
    ids=["cannot close", "cannot reopen"],
)
async def test_sessions_stay_open_when_the_agent_cannot_close_and_reopen_them(
    tmp_path: Path, capabilities: dict[str, Any]
) -> None:
    rows = [
        HANDSHAKE[0],
        initialize_answer(sessionCapabilities=capabilities),
        *new_session(2, "s1"),
        *prompted(3, "s1", "one"),
        *new_session(4, "s2"),
        *prompted(5, "s2", "two"),
    ]
    agent = make(tmp_path, write_script(tmp_path, rows), keep_sessions=0)
    async with agent:
        first = await agent.run("One.")
        second = await agent.run("Two.")
    assert (first.text, second.text) == ("one", "two")
    assert "session/close" not in methods(tmp_path)


async def test_a_slow_close_is_waited_for_before_the_session_is_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(acp_client, "CLOSE_TIMEOUT_S", 0.3)
    rows = [
        HANDSHAKE[0],
        HANDSHAKE[1],
        *new_session(2, "s1"),
        *prompted(3, "s1", "one"),
        {"expect": {"id": 4, "method": "session/close", "params": {"sessionId": "s1"}}},
        {"sleep": 1.0},  # the agent drains the session; it refuses to resume it until done
        {"jsonrpc": "2.0", "id": 4, "result": {}},
        *resumed(5, "s1"),
        *prompted(6, "s1", "two"),
        *closed(7, "s1"),
    ]
    agent = make(tmp_path, write_script(tmp_path, rows), keep_sessions=0)
    with caplog.at_level(logging.WARNING, logger="qagents.integrations.acp"):
        async with agent:
            first = await agent.run("One.")
            second = await agent.run("Two.", session=first.session_id)
    assert (second.status, second.text) == ("completed", "two")
    assert "did not answer session/close for s1 within 0.3 seconds" in caplog.text
    # the close was not given up ($/cancel_request), and the resume waited for its answer
    assert methods(tmp_path) == [
        "initialize",
        "session/new",
        "session/prompt",
        "session/close",
        "session/resume",
        "session/prompt",
        "session/close",
    ]


async def test_a_session_still_closing_after_request_timeout_s_is_not_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(acp_client, "CLOSE_TIMEOUT_S", 0.2)
    rows = [
        HANDSHAKE[0],
        HANDSHAKE[1],
        *new_session(2, "s1"),
        *prompted(3, "s1", "one"),
        {"expect": {"id": 4, "method": "session/close", "params": {"sessionId": "s1"}}},
        {"sleep": 1.5},
        {"jsonrpc": "2.0", "id": 4, "result": {}},
    ]
    agent = make(tmp_path, write_script(tmp_path, rows), keep_sessions=0, request_timeout_s=0.4)
    async with agent:
        first = await agent.run("One.")
        second = await agent.run("Two.", session=first.session_id)
        assert agent.connected
    assert second.status == "error"
    assert second.stop_reason == acp_texts.ACP_CANNOT_CONTINUE.format(
        session="s1", reason=acp_texts.ACP_STILL_CLOSING
    )
    assert "session/resume" not in methods(tmp_path)


async def test_after_a_failed_close_a_refused_resume_prompts_the_session_directly(
    tmp_path: Path,
) -> None:
    failure = {"code": -32603, "message": "Internal error: session close failed: flush timed out"}
    refused = {"code": -32602, "message": "Invalid params: session is already active: s1"}
    rows = [
        HANDSHAKE[0],
        HANDSHAKE[1],
        *new_session(2, "s1"),
        *prompted(3, "s1", "one"),
        *closed(4, "s1", failure),
        {"expect": {"id": 5, "method": "session/resume", "params": {"sessionId": "s1"}}},
        {"jsonrpc": "2.0", "id": 5, "error": refused},
        *prompted(6, "s1", "two"),  # the agent still had it open
        *closed(7, "s1"),
    ]
    agent = make(tmp_path, write_script(tmp_path, rows), keep_sessions=0)
    async with agent:
        first = await agent.run("One.")
        second = await agent.run("Two.", session=first.session_id)
        assert agent.connected
    assert (second.status, second.session_id, second.text) == ("completed", "s1", "two")


def test_keep_sessions_must_be_a_whole_number() -> None:
    for wrong in (-1, True, 1.5):
        with pytest.raises(ConfigurationError, match="keep_sessions"):
            ACPAgent([sys.executable], keep_sessions=wrong)  # type: ignore[arg-type]


# -- re-attaching after a crash (resumable=True) ----------------------------------------------------


async def test_a_repeated_call_resumes_the_journaled_session(tmp_path: Path) -> None:
    rows = [
        *_resume_rows("session/resume"),
        {"jsonrpc": "2.0", "id": 2, "result": {}},
        PROMPT,
        chunk("Finished."),
        stop(),
    ]
    agent = make(
        tmp_path, write_script(tmp_path, rows), "--session-id", "old-session", resumable=True
    )
    assert agent.as_tool().idempotent
    checkpoints = {
        "external_session": {"session_id": "old-session", "cwd": str(tmp_path / "workspace")},
        "acp_prompt_sent": True,
    }
    turn = agent._open_turn(
        "Fix the parser.",
        session=None,
        cwd=None,
        parent=None,
        permissions=None,
        label=None,
        checkpoints=checkpoints,
        resumable=True,
    )
    async with agent:
        result = await turn.wait()
    assert (result.status, result.session_id, result.text) == (
        "completed",
        "old-session",
        "Finished.",
    )
    prompt = next(m for m in sent(tmp_path) if m.get("method") == "session/prompt")
    text = prompt["params"]["prompt"][0]["text"]
    assert text == acp_texts.ACP_RESUME_TASK.format(
        resume=texts.EXTERNAL_AGENT_RESUME, task="Fix the parser."
    )


async def test_a_repeated_call_that_cannot_re_attach_is_not_started_over(tmp_path: Path) -> None:
    rows = [HANDSHAKE[0], initialize_answer(sessionCapabilities={})]
    agent = make(tmp_path, write_script(tmp_path, rows), resumable=True)
    turn = agent._open_turn(
        "Fix the parser.",
        session=None,
        cwd=None,
        parent=None,
        permissions=None,
        label=None,
        checkpoints={"external_session": {"session_id": "old-session", "cwd": None}},
        resumable=True,
    )
    async with agent:
        result = await turn.wait()
    assert result.status == "interrupted" and result.raw["ambiguous"] is True
    assert result.stop_reason == acp_texts.ACP_NOT_REATTACHED.format(session="old-session")
    assert "session/new" not in methods(tmp_path)
