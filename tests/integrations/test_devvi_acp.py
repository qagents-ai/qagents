"""`DevviAgent` on its default transport, ACP (``devvi --profile acp``), against the fake peer.

The fake Devvi CLI (``fakes/fake_devvi_acp.py``) reads the launcher's flags and plays an ACP
fixture with ``fakes/fake_acp_agent.py``: Devvi's recorded transcripts from
``tests/fixtures/devvi/acp`` or small scripts written here. A ``--plan`` file gives each
workspace, and each restart of a workspace's process, its own fixture. No Devvi is needed.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from qagents import Agent, InMemoryStore, Limits, Permissions, Session, SQLiteStore
from qagents import prompts as acp_texts
from qagents import prompts as devvi_texts
from qagents import prompts as texts
from qagents.errors import ConfigurationError
from qagents.events import SubagentEvent
from qagents.external import ExternalAgentConfigurationError, ExternalApproval
from qagents.integrations.acp import CLIENT_CAPABILITIES
from qagents.integrations.devvi import DevviAgent, ExposedTools
from qagents.models.testing import FunctionModel, ScriptedModel, call

from . import fixture_tools as fx
from .fakes.devvi import (
    ACP_FIXTURES,
    FAKE_NODE,
    acp_agent,
    acp_command,
    records,
    sent,
    sent_methods,
    wait_until_gone,
)

pytestmark = pytest.mark.devvi_fake

TESTS = Path(__file__).resolve().parents[1]
"""``tests/``: puts ``integrations.fixture_tools`` on the stdio tool server's path."""

PRO = '["deepseek-official","deepseek-v4-pro"]'

PYTHON = os.path.abspath(sys.executable)
"""What the stdio tool server entries run (not resolved: a venv's python is a link)."""


# -- fixture rows ---------------------------------------------------------------------------------


def lines_of(name: str) -> list[str]:
    return (ACP_FIXTURES / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()


HANDSHAKE = lines_of("handshake")[1:]
"""expect initialize, its answer, expect session/new, its answer (session fake-session-1)."""

OPTIONS: list[dict[str, Any]] = json.loads(HANDSHAKE[3])["result"]["configOptions"]


def initialize_answer(**capabilities: Any) -> dict[str, Any]:
    answer = json.loads(HANDSHAKE[1])
    answer["result"]["agentCapabilities"].update(capabilities)
    return answer


def new_answer(session: str = "{{sessionId}}", options: Any = None) -> dict[str, Any]:
    answer = json.loads(HANDSHAKE[3])
    answer["result"]["sessionId"] = session
    if options is not None:
        answer["result"]["configOptions"] = options
    return answer


def expect(request_id: int | None, method: str, **params: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"method": method, "params": params}
    if request_id is not None:
        body["id"] = request_id
    return {"expect": body}


def answer(request_id: int, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def prompt(request_id: int, session: str = "{{sessionId}}", **params: Any) -> dict[str, Any]:
    return expect(request_id, "session/prompt", sessionId=session, prompt="$list", **params)


def update(update_kind: str, session: str = "{{sessionId}}", **fields: Any) -> dict[str, Any]:
    body = {"sessionUpdate": update_kind, **fields}
    return {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {"sessionId": session, "update": body},
    }


def chunk(text: str, session: str = "{{sessionId}}") -> dict[str, Any]:
    content = {"type": "text", "text": text}
    return update("agent_message_chunk", session, messageId="{{messageId}}", content=content)


def stop(request_id: int, reason: str = "end_turn") -> dict[str, Any]:
    return answer(request_id, {"stopReason": reason})


def bash_with_permission(call_id: str = "call-1", request: int = 1) -> list[dict[str, Any]]:
    """A ``bash`` tool call that asks for permission and runs once allowed."""
    return [
        update("tool_call", toolCallId=call_id, title="bash", kind="other", status="in_progress",
               rawInput={"command": "pytest -q"}),
        {"jsonrpc": "2.0", "id": request, "method": "session/request_permission", "params": {
            "sessionId": "{{sessionId}}", "toolCall": {"toolCallId": call_id},
            "options": [{"optionId": "allow-once", "name": "Allow once", "kind": "allow_once"},
                        {"optionId": "reject-once", "name": "Reject", "kind": "reject_once"}]}},
        {"answer": {"id": request, "accept": [
            {"outcome": {"outcome": "selected", "optionId": "allow-once"}}]}},
        update("tool_call_update", toolCallId=call_id, status="completed",
               content=[{"type": "content", "content": {"type": "text", "text": "2 passed"}}]),
    ]  # fmt: skip


def write_script(path: Path, rows: list[Any]) -> Path:
    text = "\n".join(row if isinstance(row, str) else json.dumps(row) for row in rows)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def write_plan(tmp_path: Path, plan: dict[Path, list[list[Any]]]) -> Path:
    """A ``--plan`` file: for each workspace, the fixture of each process started there."""
    entries: dict[str, list[str]] = {}
    for directory, scripts in plan.items():
        directory.mkdir(parents=True, exist_ok=True)
        names = []
        for index, rows in enumerate(scripts):
            name = f"plan-{directory.name}-{index}.jsonl"
            write_script(tmp_path / name, rows)
            names.append(name)
        entries[str(directory)] = names
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def starts(tmp_path: Path) -> list[dict[str, Any]]:
    return [item["start"] for item in records(tmp_path / "start.jsonl")]


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tool_results(result: Any) -> list[Any]:
    return [part for message in result.messages for part in message.tool_results]


async def eventually(check: Any, seconds: float = 10.0) -> bool:
    for _ in range(int(seconds / 0.02)):
        if check():
            return True
        await asyncio.sleep(0.02)
    return False


# -- the launch and the wire --------------------------------------------------------------------


async def test_acp_is_the_default_and_runs_devvi_with_the_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in {
        "DEEPSEEK_API_KEY": "fake-deepseek",
        "OPENAI_API_KEY": "fake-openai",
        "GITHUB_TOKEN": "fake-gh",
        "DEVVI_HOME": str(tmp_path / "users-own"),
    }.items():
        monkeypatch.setenv(name, value)
    agent = acp_agent(
        tmp_path,
        "escalation-approved",
        permissions=Permissions(allow=["coder.bash"]),
        permission_mode="read-only",
    )
    assert agent.transport == "acp" and agent.model is None
    assert agent.resumable_turns and agent.as_tool().idempotent
    async with agent:
        caps = await agent.capabilities()
        result = await agent.run("Write the file outside the workspace.")
    assert not caps.token_usage and caps.cancel_turn and caps.resume_sessions
    assert caps.permission_requests and caps.mcp_servers and not caps.client_session_ids
    assert (result.status, result.stop_reason, result.text) == ("completed", "end_turn", "DONE")
    assert result.raw["agent_stop_reason"] == "end_turn"  # the SDK transport's key too
    assert result.session_id == "fake-session-1"
    assert result.backend == "devvi-acp@0.2.1-rc.3"
    assert [(p.tool, p.outcome, p.decided_by) for p in result.permissions] == [
        ("coder.bash", "allow", "rules")
    ]
    messages = sent(tmp_path)
    assert sent_methods(tmp_path) == ["initialize", "session/new", "session/prompt", "<answer>"]
    assert messages[0]["params"]["protocolVersion"] == 1
    assert messages[0]["params"]["clientCapabilities"] == CLIENT_CAPABILITIES
    workspace = tmp_path / "workspace"
    assert messages[1]["params"] == {"cwd": str(workspace), "mcpServers": []}
    assert messages[3]["result"] == {"outcome": {"outcome": "selected", "optionId": "allow-once"}}
    [start] = starts(tmp_path)
    assert start["profile"] == "acp" and start["cwd"] == str(workspace)
    home = tmp_path / "devvi-home"
    assert start["home"] == {"DEVVI_HOME": str(home), "DSH_HOME": str(home)}
    assert Path(start["patches"][0]).parent == home / "qagents" / "patches"
    assert start["permission_mode"] == "read-only" and start["telemetry_disabled"] == "1"
    assert "DEEPSEEK_API_KEY" in start["env_names"]
    assert "OPENAI_API_KEY" not in start["env_names"] and "GITHUB_TOKEN" not in start["env_names"]


async def test_a_turn_through_node_and_bin_js(tmp_path: Path) -> None:
    script = tmp_path / "devvi-cli" / "lib" / "bin.js"
    script.parent.mkdir(parents=True)
    acp_args = [
        "--script", str(ACP_FIXTURES / "escalation-rejected.jsonl"),
        "--start-record", str(tmp_path / "start.jsonl"),
    ]  # fmt: skip
    script.write_text(json.dumps({"version": "0.2.1-rc.3", "acp_args": acp_args}), "utf-8")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    agent = DevviAgent(
        workspace=workspace,
        name="coder",
        devvi_bin=script,
        node=[sys.executable, "-I", str(FAKE_NODE)],
        home=tmp_path / "home",
        permissions=Permissions(deny=["coder.bash"]),
        track_changes=False,
    )
    async with agent:
        result = await agent.run("Write outside the workspace.")
    assert result.completed and "rejected" in result.text
    assert [(p.tool, p.outcome, p.decided_by) for p in result.permissions] == [
        ("coder.bash", "deny", "rules")
    ]
    [start] = starts(tmp_path)
    assert start["profile"] == "acp" and start["cwd"] == str(workspace)
    argv = start["argv"]
    assert argv[argv.index("--profile") + 1] == "acp" and "--patch" in argv


async def test_devvis_committed_blocks_are_separated_by_a_blank_line(tmp_path: Path) -> None:
    """Devvi sends one chunk per committed block, all with the message's id."""
    rows = [*HANDSHAKE, prompt(3), chunk("Fixed it."), chunk("Tests pass."), stop(3)]
    coder = acp_agent(tmp_path, write_script(tmp_path / "blocks.jsonl", rows))
    async with coder:
        result = await coder.run("Fix it.")
    assert result.text == "Fixed it.\n\nTests pass."


async def test_as_tool_forwards_the_events_and_writes_the_footer(tmp_path: Path) -> None:
    coder = acp_agent(tmp_path, "escalation-approved")
    approved: list[str] = []

    def on_ask(ctx: Any, tool: Any, call_part: Any, reason: Any) -> bool:
        approved.append(f"{tool.name} {call_part.arguments['command'][:6]}")
        return True

    lead = Agent(
        ScriptedModel([call("coder", task="Write the file."), "Done."]),
        name="lead",
        tools=[coder.as_tool(timeout=600)],
        permissions=Permissions(ask=["coder.bash"], on_ask=on_ask),
    )
    async with coder:
        stream = lead.stream("Go")
        events = [event async for event in stream]
    result = stream.result
    assert result is not None and result.output == "Done."
    forwarded = [event.event for event in events if isinstance(event, SubagentEvent)]
    kinds = [event.type if event.type != "external_event" else event.kind for event in forwarded]
    assert kinds == [
        "run_started", "reasoning_delta", "tool_call_started", "permission_requested",
        "permission_decided", "tool_call_finished", "reasoning_delta", "text_delta", "turn_finished",
    ]  # fmt: skip
    assert forwarded[0].depth == 1 and forwarded[2].call.name == "coder.bash"
    assert approved == ["coder.bash printf"]
    content = tool_results(result)[0].content
    assert content == "DONE\n\n[coder: session fake-session-1; completed; 1 tool call (bash 1)]"


async def test_a_cost_budget_needs_allow_unmetered(tmp_path: Path) -> None:
    def lead(coder: DevviAgent) -> Agent:
        return Agent(
            ScriptedModel([call("coder", task="Write the file."), "Done."]),
            tools=[coder.as_tool()],
            permissions=Permissions(allow=["coder.bash"]),
            limits=Limits(max_turns=4, max_cost_usd=1.0),
        )

    unmetered = acp_agent(tmp_path, "escalation-approved")
    async with unmetered:
        refused = await lead(unmetered).run("Go")
    assert tool_results(refused)[0].content == texts.EXTERNAL_AGENT_UNMETERED.format(name="coder")
    coder = acp_agent(tmp_path, "escalation-approved", allow_unmetered=True)
    async with coder:
        accepted = await lead(coder).run("Go")
    assert tool_results(accepted)[0].content.startswith("DONE")


def test_sdk_only_options_are_refused_on_acp(tmp_path: Path) -> None:
    from qagents.integrations.devvi import TokenPricing

    with pytest.raises(ConfigurationError, match="pricing needs transport='sdk'"):
        DevviAgent(workspace=tmp_path, pricing=TokenPricing(input=1.0, output=1.0))
    with pytest.raises(ConfigurationError, match="max_tokens needs transport='sdk'"):
        DevviAgent(workspace=tmp_path, max_tokens=1000)
    sdk = DevviAgent(workspace=tmp_path, transport="sdk")
    assert sdk.model == ("deepseek-official", "deepseek-v4-flash")  # the SDK needs a route


async def test_cancel_sends_session_cancel(tmp_path: Path) -> None:
    agent = acp_agent(tmp_path, "cancel")
    async with agent:
        async with agent.stream("Count slowly.") as turn:
            async for event in turn:
                if event.type == "text_delta":
                    turn.cancel()
    assert turn.result is not None and turn.result.status == "cancelled"
    assert "session/cancel" in sent_methods(tmp_path)


# -- the model, from the session's options ------------------------------------------------------


async def test_the_model_is_selected_from_the_session_options(tmp_path: Path) -> None:
    options = json.loads(json.dumps(OPTIONS))
    options[0]["currentValue"] = PRO
    rows = [
        *HANDSHAKE,
        expect(
            3, "session/set_config_option", sessionId="{{sessionId}}", configId="model", value=PRO
        ),
        answer(3, {"configOptions": options}),
        prompt(4),
        chunk("on pro"),
        stop(4),
    ]
    agent = acp_agent(
        tmp_path,
        write_script(tmp_path / "pro.jsonl", rows),
        model=("deepseek-official", "deepseek-v4-pro"),
    )
    async with agent:
        result = await agent.run("Go")
        [connection] = agent.connections.values()
        assert connection.session_config(result.session_id)[0]["currentValue"] == PRO
    assert result.text == "on pro"


async def test_a_model_the_session_does_not_offer_is_refused(tmp_path: Path) -> None:
    agent = acp_agent(tmp_path, "handshake", model=("deepseek-official", "deepseek-v9"))
    async with agent:
        with pytest.raises(ExternalAgentConfigurationError, match="it offers: deepseek-v4-flash"):
            await agent.run("Go")

    log = tmp_path / "client.jsonl"
    log.unlink()
    coder = acp_agent(tmp_path, "handshake", model=("deepseek-official", "deepseek-v9"))
    lead = Agent(
        ScriptedModel([call("coder", task="a"), call("coder", task="b"), "Gave up."]),
        tools=[coder.as_tool()],
        limits=Limits(max_turns=5),
    )
    async with coder:
        result = await lead.run("Go")
    first, second = (part.content for part in tool_results(result))
    assert first.startswith(
        "coder is unavailable: coder: model ('deepseek-official', 'deepseek-v9')"
    )
    assert second == first  # the circuit breaker: no second session
    assert sent_methods(tmp_path).count("session/new") == 1


async def test_the_reasoning_effort_is_an_option_too(tmp_path: Path) -> None:
    agent = acp_agent(tmp_path, "handshake", reasoning_effort="high")
    async with agent:
        with pytest.raises(
            ExternalAgentConfigurationError, match="no config option 'reasoning_effort'"
        ):
            await agent.run("Go")
    effort = {
        "id": "reasoning_effort", "name": "Reasoning effort", "category": "thought_level",
        "type": "select", "currentValue": "",
        "options": [{"value": "", "name": "Provider default"}, {"value": "high", "name": "High"}],
    }  # fmt: skip
    rows = [
        *HANDSHAKE[:3],
        new_answer(options=[*OPTIONS, effort]),
        expect(3, "session/set_config_option", configId="reasoning_effort", value="high"),
        answer(3, {"configOptions": [*OPTIONS, {**effort, "currentValue": "high"}]}),
        prompt(4),
        chunk("thinking hard"),
        stop(4),
    ]
    agent = acp_agent(
        tmp_path, write_script(tmp_path / "effort.jsonl", rows), reasoning_effort="high"
    )
    async with agent:
        assert (await agent.run("Go")).text == "thinking hard"


# -- one process per workspace ------------------------------------------------------------------


def _other_session(session: str, *texts_: str) -> list[Any]:
    """A fixture for a second workspace: its own session id, one prompt per text."""
    rows: list[Any] = [*HANDSHAKE[:3], new_answer(session)]
    for index, text in enumerate(texts_):
        rows += [prompt(3 + index, session), chunk(text, session), stop(3 + index)]
    return rows


async def test_each_workspace_has_its_own_devvi_process(tmp_path: Path) -> None:
    workspace, other = tmp_path / "workspace", tmp_path / "other"
    plan = write_plan(
        tmp_path,
        {
            workspace: [
                [*HANDSHAKE, prompt(3), chunk("one"), stop(3), prompt(4), chunk("two"), stop(4)]
            ],
            other: [_other_session("other-session", "elsewhere")],
        },
    )
    agent = acp_agent(tmp_path, None, "--plan", str(plan))
    async with agent:
        first = await agent.run("First.")
        second = await agent.run("Second.", session=first.session_id)
        third = await agent.run("Elsewhere.", cwd=other)
        assert len(agent.connections) == 2
    assert (first.text, second.text, third.text) == ("one", "two", "elsewhere")
    assert (second.session_id, third.session_id) == ("fake-session-1", "other-session")
    assert third.workspace == str(other)
    begun = starts(tmp_path)
    assert [item["cwd"] for item in begun] == [str(workspace), str(other)]
    assert begun[0]["pid"] != begun[1]["pid"]
    news = [m["params"]["cwd"] for m in sent(tmp_path) if m.get("method") == "session/new"]
    assert news == [str(workspace), str(other)]
    assert "session/resume" not in sent_methods(tmp_path)  # a live session is prompted directly


async def test_a_lost_process_restarts_and_other_workspaces_keep_theirs(tmp_path: Path) -> None:
    workspace, other = tmp_path / "workspace", tmp_path / "other"
    resumed = [
        *HANDSHAKE[:2],
        expect(
            2, "session/resume", sessionId="fake-session-1", cwd=str(workspace), mcpServers="$list"
        ),
        answer(2, {"configOptions": OPTIONS}),
        prompt(3),
        chunk("back"),
        stop(3),
    ]
    plan = write_plan(
        tmp_path,
        {
            workspace: [[*HANDSHAKE, prompt(3), chunk("one"), stop(3), {"exit": 0}], resumed],
            other: [_other_session("other-session", "o1", "o2")],
        },
    )
    agent = acp_agent(tmp_path, None, "--plan", str(plan))
    async with agent:
        o1 = await agent.run("o1", cwd=other)
        first = await agent.run("First.")
        connection = agent.connections[os.path.normcase(str(workspace))]
        assert await eventually(lambda: not connection.connected)  # Devvi exited after the turn
        again = await agent.run("Again.", session=first.session_id)
        o2 = await agent.run("o2", cwd=other, session=o1.session_id)
        assert agent.connected  # the agent never disconnected as a whole
    assert (first.text, again.text, o1.text, o2.text) == ("one", "back", "o1", "o2")
    begun = starts(tmp_path)
    # the agent's own workspace starts on connect; the other one with its first turn
    assert [item["cwd"] for item in begun] == [str(workspace), str(other), str(workspace)]
    resumes = [m for m in sent(tmp_path) if m.get("method") == "session/resume"]
    assert [m["params"]["sessionId"] for m in resumes] == ["fake-session-1"]
    for item in begun:
        await wait_until_gone(item["pid"])  # every process closed with the agent


# -- re-attaching after a crash of the calling run ----------------------------------------------


async def test_a_call_repeated_after_a_crash_continues_its_session(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    resume_text = acp_texts.ACP_RESUME_TASK.format(
        resume=texts.EXTERNAL_AGENT_RESUME, task="Fix the parser."
    )
    resumed = [
        *HANDSHAKE[:2],
        expect(2, "session/resume", sessionId="fake-session-1", cwd=str(workspace)),
        answer(2, {"configOptions": OPTIONS}),
        expect(3, "session/prompt", sessionId="fake-session-1",
               prompt=[{"type": "text", "text": resume_text}]),
        chunk("Finished after the interruption."),
        stop(3),
    ]  # fmt: skip
    plan = write_plan(tmp_path, {workspace: [[*HANDSHAKE, prompt(3), {"hang": True}], resumed]})
    db = tmp_path / "runs.db"
    coder = acp_agent(tmp_path, None, "--plan", str(plan), cancel_grace_s=0.5)
    lead = Agent(ScriptedModel([call("coder", task="Fix the parser.")]), tools=[coder.as_tool()])
    task = asyncio.create_task(lead.run("go", session=Session(SQLiteStore(db), id="crash")))
    try:
        assert await eventually(lambda: "session/prompt" in sent_methods(tmp_path))
    finally:
        task.cancel()  # the calling run crashes while Devvi works
    with pytest.raises(asyncio.CancelledError):
        await task
    await coder.aclose()
    store = SQLiteStore(db)
    entries = await Session(store, id="crash").entries()
    keys = [entry.data["key"] for entry in entries if entry.kind == "tool_checkpoint"]
    assert keys == ["external_session", "acp_prompt_sent"]

    second = acp_agent(tmp_path, None, "--plan", str(plan))
    async with second:
        done = await Agent(ScriptedModel(["Done."]), tools=[second.as_tool()]).resume(
            Session(store, id="crash")
        )
    assert done.output == "Done."
    content = tool_results(done)[0].content
    assert content.startswith("Finished after the interruption.\n\n[coder: session fake-session-1;")
    assert sent_methods(tmp_path).count("session/new") == 1  # never started over


# -- a human answers across a qagents pause (on_permission="pause") -------------------------------


async def test_the_pause_posture_round_trips_through_a_store(tmp_path: Path) -> None:
    rm = {"command": "rm -rf build"}

    def ask(call_id: str, request: int, outcome: dict[str, Any]) -> list[Any]:
        return [
            update("tool_call", toolCallId=call_id, title="bash", kind="other", rawInput=rm),
            {"jsonrpc": "2.0", "id": request, "method": "session/request_permission", "params": {
                "sessionId": "{{sessionId}}", "toolCall": {"toolCallId": call_id},
                "options": [{"optionId": "allow-once", "name": "Allow once", "kind": "allow_once"},
                            {"optionId": "reject-once", "name": "Reject", "kind": "reject_once"}]}},
            {"answer": {"id": request, "accept": [outcome]}},
        ]  # fmt: skip

    rows = [
        *HANDSHAKE,
        prompt(3),
        *ask("c1", 1, {"outcome": {"outcome": "cancelled"}}),
        expect(None, "session/cancel", sessionId="{{sessionId}}"),
        stop(3, "cancelled"),
        prompt(4),  # the session is still open in this process: prompted directly
        *ask("c2", 2, {"outcome": {"outcome": "selected", "optionId": "allow-once"}}),
        update("tool_call_update", toolCallId="c2", status="completed",
               content=[{"type": "content", "content": {"type": "text", "text": "removed"}}]),
        chunk("Cleaned."),
        stop(4),
    ]  # fmt: skip
    coder = acp_agent(tmp_path, write_script(tmp_path / "pause.jsonl", rows), on_permission="pause")

    def reply(request: Any) -> Any:
        last = request.messages[-1]
        if last.role != "tool":
            return call("coder", task="Clean the build.")
        found = re.search(r"session=([\w-]+)", last.tool_results[0].content)
        return call("coder", task="Go on.", session=found.group(1)) if found else "Done."

    lead = Agent(FunctionModel(reply), tools=[coder.as_tool()])
    db = tmp_path / "runs.db"
    async with coder:
        paused = await lead.run("Clean up", session=Session(SQLiteStore(db), id="s1"))
        assert paused.status == "paused"
        pending = paused.pending_inputs[0]
        assert pending.metadata["tool"] == "coder.bash" and pending.metadata["arguments"] == rm
        assert pending.metadata["external_session"] == "fake-session-1"
        session = Session(SQLiteStore(db), id="s1")
        restored = (await session.state()).pending_inputs[0]
        again = await lead.resume(
            session,
            results={restored.tool_call_id: ExternalApproval.for_input(restored, approved=True)},
        )
    assert again.output == "Done."
    answered = [m["result"]["outcome"] for m in sent(tmp_path) if "result" in m]
    assert answered == [{"outcome": "cancelled"}, {"outcome": "selected", "optionId": "allow-once"}]
    prompts = [
        m["params"]["prompt"][0]["text"]
        for m in sent(tmp_path)
        if m.get("method") == "session/prompt"
    ]
    assert prompts[1].startswith(texts.EXTERNAL_AGENT_CONTINUE.splitlines()[0])
    assert tool_results(again)[-1].content.startswith("Cleaned.")


async def test_a_pause_grant_survives_reworded_free_text_but_not_another_command(
    tmp_path: Path,
) -> None:
    """Devvi's model writes ``description`` and ``justification`` anew on each attempt: the
    approval still answers the same escalation, and a changed ``command`` is asked again."""
    asked = {
        "command": "Remove-Item -Recurse C:/build",
        "description": "Remove the build folder",
        "sandbox_permissions": "danger-full-access",
        "justification": "the build folder is outside the workspace",
    }
    reworded = {**asked, "description": "Delete C:/build", "justification": "stale output"}
    other = {**reworded, "command": "Remove-Item -Recurse C:/src"}
    cancelled = {"outcome": {"outcome": "cancelled"}}
    allowed = {"outcome": {"outcome": "selected", "optionId": "allow-once"}}

    def ask(call_id: str, request: int, arguments: dict[str, Any], outcome: Any) -> list[Any]:
        return [
            update("tool_call", toolCallId=call_id, title="pwsh", kind="other",
                   rawInput=arguments),
            {"jsonrpc": "2.0", "id": request, "method": "session/request_permission", "params": {
                "sessionId": "{{sessionId}}", "toolCall": {"toolCallId": call_id},
                "options": [{"optionId": "allow-once", "name": "Allow once", "kind": "allow_once"},
                            {"optionId": "reject-once", "name": "Reject", "kind": "reject_once"}]}},
            {"answer": {"id": request, "accept": [outcome]}},
        ]  # fmt: skip

    rows = [
        *HANDSHAKE,
        prompt(3),
        *ask("c1", 1, asked, cancelled),  # paused for a person
        expect(None, "session/cancel", sessionId="{{sessionId}}"),
        stop(3, "cancelled"),
        prompt(4),
        *ask("c2", 2, reworded, allowed),  # the same escalation in other words: granted
        update("tool_call_update", toolCallId="c2", status="completed",
               content=[{"type": "content", "content": {"type": "text", "text": "removed"}}]),
        *ask("c3", 3, other, cancelled),  # another command: paused again
        expect(None, "session/cancel", sessionId="{{sessionId}}"),
        stop(4, "cancelled"),
    ]  # fmt: skip
    script_path = write_script(tmp_path / "reworded.jsonl", rows)
    coder = acp_agent(tmp_path, script_path, on_permission="pause")
    assert coder.grant_arguments("coder.pwsh", reworded) == {
        "command": "Remove-Item -Recurse C:/build",
        "sandbox_permissions": "danger-full-access",
    }

    def reply(request: Any) -> Any:
        last = request.messages[-1]
        if last.role != "tool":
            return call("coder", task="Clean the build.")
        found = re.search(r"session=([\w-]+)", last.tool_results[0].content)
        return call("coder", task="Go on.", session=found.group(1)) if found else "Done."

    lead = Agent(FunctionModel(reply), tools=[coder.as_tool()])
    async with coder:
        paused = await lead.run("Clean up", session=Session(InMemoryStore(), id="s1"))
        assert paused.status == "paused"
        first = paused.pending_inputs[0]
        assert first.metadata["arguments"] == asked  # the person sees the whole request
        approval = ExternalApproval.for_input(first, approved=True)
        again = await lead.resume(paused.session, results={first.tool_call_id: approval})
    assert again.status == "paused"
    assert again.pending_inputs[0].metadata["arguments"] == other
    answered = [m["result"]["outcome"] for m in sent(tmp_path) if "result" in m]
    assert answered == [cancelled["outcome"], allowed["outcome"], cancelled["outcome"]]


# -- qagents tools inside the Devvi session (expose_tools) --------------------------------------


@contextlib.asynccontextmanager
async def mcp_client(url: str, headers: dict[str, str]) -> AsyncIterator[Any]:
    """The official MCP client over Streamable HTTP: what Devvi's MCP client does."""
    httpx2 = pytest.importorskip("httpx2")
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client

    http = httpx2.AsyncClient(headers=headers, timeout=httpx2.Timeout(30.0, read=60.0))
    async with http, Client(streamable_http_client(url, http_client=http)) as client:
        yield client


def http_entry(servers: list[dict[str, Any]]) -> tuple[str, dict[str, str]]:
    [entry] = [server for server in servers if server.get("type") == "http"]
    return entry["url"], {item["name"]: item["value"] for item in entry["headers"]}


def text_of(result: Any) -> str:
    return "\n".join(block.text for block in result.content if getattr(block, "text", None))


def stdio_spec() -> Any:
    from qagents.integrations.mcp_server import ToolServerSpec

    return ToolServerSpec(
        name="fixture", tools=["integrations.fixture_tools:lookup_customer"], paths=[str(TESTS)]
    )


def test_expose_tools_are_sorted_and_checked(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    spec = stdio_spec()
    exposed = ExposedTools.build(
        [fx.lookup_customer, "integrations.fixture_tools:create_ticket", spec], name="qagents"
    )
    assert exposed is not None and exposed.in_process == (fx.lookup_customer,)
    assert exposed.server_names == ["qagents", "qagents_stdio", "fixture"]
    assert exposed.specs[0].tools == ["integrations.fixture_tools:create_ticket"]
    assert ExposedTools.build([]) is None
    with pytest.raises(ConfigurationError, match="package.module:attr"):
        ExposedTools.build(["not a reference"])
    with pytest.raises(ConfigurationError, match="two MCP servers are named 'fixture'"):
        ExposedTools.build([spec, spec])
    with pytest.raises(ConfigurationError, match="1 to 32 letters"):
        ExposedTools.build([fx.lookup_customer], name="has space")
    with pytest.raises(ConfigurationError, match="needs transport='acp'"):
        DevviAgent(workspace=tmp_path, transport="sdk", expose_tools=[fx.lookup_customer])
    paths = exposed.write_specs(tmp_path / "home")
    assert paths == exposed.write_specs(tmp_path / "home")  # content-addressed, written once
    assert all(path.parent == tmp_path / "home" / "qagents" / "tool-servers" for path in paths)
    assert json.loads(paths[1].read_text("utf-8"))["paths"] == [str(TESTS)]


def _exposed_script() -> list[Any]:
    """Two turns of one session; each mounts the servers, uses bash, and closes the session."""
    return [
        *HANDSHAKE,
        prompt(3),
        *bash_with_permission("call-1", 1),
        chunk("Looked Maria up."),
        stop(3),
        expect(4, "session/close", sessionId="{{sessionId}}"),
        answer(4, {}),
        expect(5, "session/resume", sessionId="{{sessionId}}", cwd="$absolute", mcpServers="$list"),
        answer(5, {"configOptions": OPTIONS}),
        prompt(6),
        *bash_with_permission("call-2", 2),
        chunk("Again."),
        stop(6),
        expect(7, "session/close", sessionId="{{sessionId}}"),
        answer(7, {}),
    ]


async def test_exposed_tools_are_tool_calls_of_the_lead_run(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    from qagents.integrations.mcp_server import SOURCE

    coder = acp_agent(
        tmp_path,
        write_script(tmp_path / "exposed.jsonl", _exposed_script()),
        expose_tools=[fx.lookup_customer, fx.delete_customer, stdio_spec()],
    )
    seen: dict[str, Any] = {}

    async def on_ask(ctx: Any, tool: Any, call_part: Any, reason: Any) -> bool:
        if tool.name != "coder.bash":
            return False
        opened = [m for m in sent(tmp_path) if m.get("method") in ("session/new", "session/resume")]
        url, headers = http_entry(opened[-1]["params"]["mcpServers"])
        async with mcp_client(url, headers) as client:  # as Devvi's model would call them
            seen.setdefault(
                "lookup", await client.call_tool("lookup_customer", {"email": "maria@example.com"})
            )
            seen.setdefault(
                "delete", await client.call_tool("delete_customer", {"email": "maria@example.com"})
            )
        seen.setdefault("ports", []).append(int(url.rsplit(":", 1)[1].split("/")[0]))
        return True

    session = Session(InMemoryStore(), id="exposed")
    lead = Agent(
        ScriptedModel([
            call("coder", task="Look Maria up."),
            call("coder", task="Again.", session="fake-session-1"),
            "Done.",
        ]),
        tools=[coder.as_tool()],
        permissions=Permissions(ask=["coder.bash"], deny=["delete_*"], on_ask=on_ask),
        limits=Limits(max_turns=6),
    )  # fmt: skip
    async with coder:
        result = await lead.run("Go", session=session)
    assert result.output == "Done.", tool_results(result)
    assert not seen["lookup"].is_error and json.loads(text_of(seen["lookup"]))["name"] == "Maria"
    assert seen["delete"].is_error
    assert text_of(seen["delete"]) == texts.TOOL_SERVER_REFUSED.format(
        name="delete_customer", reason="a deny rule matches this tool."
    )
    entries = await session.entries()
    coder_calls = [
        e.data["tool_call_id"]
        for e in entries
        if e.kind == "tool_started" and e.data["name"] == "coder"
    ]
    served = [entry for entry in entries if entry.data.get("source") == SOURCE]
    assert [e.data["name"] for e in served if e.kind == "tool_started"] == ["lookup_customer"] * 2
    assert {e.data["caller_tool_call_id"] for e in served} == set(coder_calls)
    # every session open mounts the stdio server and a new in-process server
    opened = [m for m in sent(tmp_path) if m.get("method") in ("session/new", "session/resume")]
    assert [m["method"] for m in opened] == ["session/new", "session/resume"]
    stdio = [[s for s in m["params"]["mcpServers"] if s.get("type") != "http"] for m in opened]
    assert stdio[0] == stdio[1] and len(stdio[0]) == 1
    entry = stdio[0][0]
    assert entry["name"] == "fixture" and entry["command"] == PYTHON
    assert entry["args"][:4] == ["-I", "-m", "qagents.integrations.mcp_server", "--spec"]
    assert Path(entry["args"][4]).parent == tmp_path / "devvi-home" / "qagents" / "tool-servers"
    first, second = (http_entry(m["params"]["mcpServers"]) for m in opened)
    assert first[0].startswith("http://127.0.0.1:") and first[0].endswith("/mcp")
    assert first[1]["Authorization"].startswith("Bearer ") and first != second
    assert sent_methods(tmp_path).count("session/close") == 2
    for port in seen["ports"]:  # each turn's server is gone once the turn ended
        with pytest.raises(OSError):
            _, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), 5)
            writer.close()


async def test_a_direct_run_serves_exposed_tools_under_the_turns_rules(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    rows = [*HANDSHAKE, prompt(3), *bash_with_permission(), chunk("ok"), stop(3),
            expect(4, "session/close", sessionId="{{sessionId}}"), answer(4, {})]  # fmt: skip
    agent = acp_agent(
        tmp_path,
        write_script(tmp_path / "direct.jsonl", rows),
        expose_tools=[fx.lookup_customer, fx.delete_customer],
    )
    seen: dict[str, Any] = {}

    async def on_ask(ctx: Any, tool: Any, call_part: Any, reason: Any) -> bool:
        new = next(m for m in sent(tmp_path) if m.get("method") == "session/new")
        async with mcp_client(*http_entry(new["params"]["mcpServers"])) as client:
            seen["lookup"] = await client.call_tool(
                "lookup_customer", {"email": "maria@example.com"}
            )
            seen["delete"] = await client.call_tool(
                "delete_customer", {"email": "maria@example.com"}
            )
        return True

    async with agent:
        result = await agent.run(
            "Go", permissions=Permissions(ask=["coder.bash"], deny=["delete_*"], on_ask=on_ask)
        )
    assert result.completed
    assert not seen["lookup"].is_error and seen["delete"].is_error


async def test_the_stdio_entry_starts_a_working_tool_server(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    from mcp import Client, StdioServerParameters
    from mcp.client.stdio import stdio_client

    exposed = ExposedTools.build([stdio_spec()])
    assert exposed is not None
    [entry] = exposed.acp_entries(tmp_path / "home")
    parameters = StdioServerParameters(command=entry["command"], args=entry["args"])
    with (tmp_path / "stderr.txt").open("w", encoding="utf-8") as errlog:
        async with Client(stdio_client(parameters, errlog=errlog)) as client:
            names = [item.name for item in (await client.list_tools()).tools]
            found = await client.call_tool("lookup_customer", {"email": "maria@example.com"})
    assert names == ["lookup_customer"]
    assert json.loads(text_of(found))["name"] == "Maria"


async def test_in_process_tools_need_mcp_over_http(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    rows = [HANDSHAKE[0], initialize_answer(mcpCapabilities={})]
    agent = acp_agent(
        tmp_path, write_script(tmp_path / "nohttp.jsonl", rows), expose_tools=[fx.lookup_customer]
    )
    with pytest.raises(ExternalAgentConfigurationError, match="mcpCapabilities.http"):
        await agent.capabilities()
    await agent.aclose()


async def test_the_sdk_transport_mounts_importable_tools_with_a_patch_row(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    from .fakes.devvi import fake_command

    agent = DevviAgent(
        workspace=tmp_path,
        transport="sdk",
        command=fake_command(None),
        home=tmp_path / "home",
        expose_tools=["integrations.fixture_tools:lookup_customer"],
    )
    launch = await agent.launch()
    rows = read_json(launch.patch_files[0])
    [mounted] = [row for row in rows if "insert" in row]
    config = mounted["insert"][0]["config"]
    assert mounted["insert"][0]["name"] == "@deepseek-ai/dsh-mcp-client"
    assert config["serverName"] == "qagents_stdio" and config["transport"] == "stdio"
    assert config["command"] == PYTHON
    spec = read_json(config["args"][-1])
    assert spec["tools"] == ["integrations.fixture_tools:lookup_customer"]


def test_the_acp_command_helper_names_fixtures() -> None:
    command = acp_command("handshake")
    assert command[command.index("--script") + 1] == str(ACP_FIXTURES / "handshake.jsonl")
    assert devvi_texts.DEVVI_DESCRIPTION.startswith("Delegate a coding task")


# -- review round 2 --------------------------------------------------------------------------------


async def test_tool_names_come_from_titles_whatever_devvi_calls_itself(tmp_path: Path) -> None:
    """Devvi sends ``kind: "other"`` for every tool and its name in the title. The rules
    keep matching ``coder.bash`` when a Devvi release reports another ``agentInfo``."""
    renamed = json.loads(HANDSHAKE[1])
    renamed["result"]["agentInfo"] = {"name": "devvi-acp", "version": "0.3.0"}
    rows = [HANDSHAKE[0], renamed, *HANDSHAKE[2:], prompt(3), *bash_with_permission(),
            chunk("2 passed."), stop(3)]  # fmt: skip
    agent = acp_agent(
        tmp_path,
        write_script(tmp_path / "renamed.jsonl", rows),
        permissions=Permissions(allow=["coder.bash"]),
    )
    async with agent:
        result = await agent.run("Run the tests.")
    assert result.completed and result.text == "2 passed."
    assert [(p.tool, p.outcome, p.decided_by) for p in result.permissions] == [
        ("coder.bash", "allow", "rules")
    ]
    assert [call.name for call in result.tool_calls] == ["bash"]


async def test_a_cancel_during_the_session_close_keeps_the_finished_turn(tmp_path: Path) -> None:
    """With in-process tools the session is closed after the turn. A cancel that comes in
    while Devvi closes it neither turns the finished turn into a cancelled one nor kills
    Devvi: the close finishes in the background, and the next turn waits for it."""
    pytest.importorskip("mcp")
    rows = [
        *HANDSHAKE,
        prompt(3),
        chunk("Done."),
        stop(3),
        expect(4, "session/close", sessionId="{{sessionId}}"),
        {"sleep": 1.5},
        answer(4, {}),
        expect(5, "session/resume", sessionId="{{sessionId}}", cwd="$absolute", mcpServers="$list"),
        answer(5, {"configOptions": OPTIONS}),
        prompt(6),
        chunk("Again."),
        stop(6),
        expect(7, "session/close", sessionId="{{sessionId}}"),
        answer(7, {}),
    ]
    agent = acp_agent(
        tmp_path,
        write_script(tmp_path / "close.jsonl", rows),
        expose_tools=[fx.lookup_customer],
        cancel_grace_s=0.2,
    )
    async with agent:
        async with agent.stream("Go") as turn:

            async def cancel_while_closing() -> None:
                assert await eventually(lambda: "session/close" in sent_methods(tmp_path))
                turn.cancel()

            watcher = asyncio.create_task(cancel_while_closing())
            _ = [event async for event in turn]
            await watcher
        result = turn.result
        assert result is not None
        assert (result.status, result.text) == ("completed", "Done.")
        assert not result.raw.get("forced")
        [connection] = agent.connections.values()
        process = connection.process
        assert process is not None and process.returncode is None and connection.connected
        again = await agent.run("Again.", session=result.session_id)  # after the close
        assert again.completed and again.text == "Again."
        assert connection.process is process  # the same Devvi: it was never killed
    methods = sent_methods(tmp_path)
    assert methods.index("session/resume") > methods.index("session/close")


async def test_an_abandoned_turn_refuses_tool_calls_before_it_closes_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A turn that does not stop is abandoned. Its tool server keeps its MCP session until
    Devvi's session is closed, but from the moment the turn is abandoned it refuses calls:
    one that Devvi makes while qagents closes the session does not run."""
    pytest.importorskip("mcp")
    from qagents.integrations.devvi import acp

    rows = [*HANDSHAKE, prompt(3), chunk("Working."),
            expect(None, "session/cancel", sessionId="{{sessionId}}"), {"hang": True}]  # fmt: skip
    agent = acp_agent(
        tmp_path,
        write_script(tmp_path / "abandoned.jsonl", rows),
        expose_tools=[fx.lookup_customer],
        cancel_grace_s=0.2,
    )
    late: list[Any] = []
    original = acp.DevviACPAgent._release_session

    async def release(self: Any, turn: Any, *, kill_on_cancel: bool = False) -> None:
        new = next(m for m in sent(tmp_path) if m.get("method") == "session/new")

        async def call_as_devvi() -> Any:  # in a task of its own, as Devvi's request is
            async with mcp_client(*http_entry(new["params"]["mcpServers"])) as client:
                return await client.call_tool("lookup_customer", {"email": "maria@example.com"})

        late.append(await asyncio.shield(asyncio.create_task(call_as_devvi())))
        await original(self, turn, kill_on_cancel=kill_on_cancel)

    monkeypatch.setattr(acp.DevviACPAgent, "_release_session", release)
    async with agent:
        async with agent.stream("Go", permissions=Permissions(allow=["lookup_customer"])) as turn:
            async for event in turn:
                if event.type == "text_delta":
                    turn.cancel()
    assert turn.result is not None and turn.result.status == "cancelled"
    [result] = late
    assert result.is_error
    assert text_of(result) == texts.TOOL_SERVER_REFUSED.format(
        name="lookup_customer", reason=texts.TOOL_SERVER_CLOSING
    )


def test_the_module_says_the_sandbox_follows_the_session_cwd() -> None:
    from qagents.integrations.devvi import acp

    doc = " ".join((acp.__doc__ or "").split())
    assert "roots its file sandbox at ``process.cwd()``" not in doc
    assert "Devvi roots each session's file sandbox at the session's ``cwd``" in doc
