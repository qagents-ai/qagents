"""External agents: the contract, the permission bridge, child runs through as_tool(),
cancellation, budgets, pauses across approvals, idempotency after a crash."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from qagents import (
    Agent,
    Approval,
    Hooks,
    Limits,
    Permissions,
    RunContext,
    Session,
    SQLiteStore,
    Usage,
)
from qagents import prompts as texts
from qagents import tool as qagents_tool
from qagents.errors import ConfigurationError
from qagents.events import RunStarted, SubagentEvent, TextDelta, ToolCallFinished, ToolCallStarted
from qagents.external import (
    CHECKPOINT_KIND,
    STATE_GRANTS,
    STATE_SESSIONS,
    EndStep,
    EventStep,
    ExternalAgent,
    ExternalAgentConfigurationError,
    ExternalAgentConnectionError,
    ExternalAgentEvent,
    ExternalAgentUnavailableError,
    ExternalApproval,
    ExternalCapabilities,
    ExternalOutcome,
    ExternalPermission,
    ExternalResult,
    ExternalToolCall,
    ExternalTurnContext,
    PermissionAnswer,
    PermissionBridge,
    PermissionStep,
    ReasoningStep,
    ScriptedExternalAgent,
    ToolStep,
    WaitStep,
    is_permanent_failure,
    permission_fingerprint,
)
from qagents.models.testing import ScriptedModel, call
from qagents.session import PendingInput


def coder(*turns: Any, **options: Any) -> ScriptedExternalAgent:
    options.setdefault("name", "coder")
    return ScriptedExternalAgent(list(turns), **options)


def tool_results(result: Any) -> list[Any]:
    return [part for message in result.messages for part in message.tool_results]


# -- running directly ----------------------------------------------------------------------


async def test_run_collects_text_tools_usage_and_permissions() -> None:
    agent = coder(
        [
            ReasoningStep("Let me look."),
            "Reading the code. ",
            ToolStep("read", {"path": "a.py"}, result="print(1)"),
            ToolStep("bash", {"command": "pytest -q"}, result="1 failed", is_error=True),
            Usage(requests=1, input_tokens=120, output_tokens=40),
            "Fixed it.",
        ]
    )
    result = await agent.run("Fix a.py", permissions=Permissions(mode="allow_all"))
    assert result.status == "completed" and result.stop_reason == "end_turn"
    assert result.text == "Fixed it."  # the last message: text after the last tool call
    assert result.session_id == "coder-1"
    assert result.backend == "scripted"
    assert result.usage.input_tokens == 120 and result.usage.output_tokens == 40
    assert [(c.name, c.is_error, c.finished) for c in result.tool_calls] == [
        ("read", False, True),
        ("bash", True, True),
    ]
    assert all(call.duration_ms is not None for call in result.tool_calls)
    assert result.permissions == []
    assert agent.prompts == ["Fix a.py"]
    assert agent.connects == 1


async def test_text_before_the_last_tool_call_is_kept_when_none_follows() -> None:
    result = await coder(["I will run the tests.", ToolStep("bash", {"command": "pytest"})]).run(
        "t"
    )
    assert result.text == "I will run the tests."


async def test_events_are_qagents_events_with_prefixed_tool_names() -> None:
    agent = coder(["Hi. ", ToolStep("bash", {"command": "ls"}, result="a.py"), "Bye."])
    async with agent.stream("t") as turn:
        events = [event async for event in turn]
    assert isinstance(events[0], RunStarted)
    assert events[0].session_id == "coder-1" and events[0].agent == "coder"
    assert events[0].depth == 0 and events[0].run_id == turn.run_id
    assert re.fullmatch(r"ext:coder-1:[0-9a-f]{12}", events[0].run_id)
    started = [e for e in events if isinstance(e, ToolCallStarted)]
    finished = [e for e in events if isinstance(e, ToolCallFinished)]
    assert started[0].call.name == "coder.bash" and started[0].call.arguments == {"command": "ls"}
    assert finished[0].result.content == "a.py" and finished[0].duration_ms >= 0
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Hi. ", "Bye."]
    last = events[-1]
    assert isinstance(last, ExternalAgentEvent) and last.kind == "turn_finished"
    assert last.data["status"] == "completed" and last.data["tool_calls"] == {"bash": 1}
    assert turn.result is not None and turn.result.text == "Bye."


async def test_custom_events_and_reasoning() -> None:
    agent = coder([EventStep("context", {"used": 10, "size": 100}), ReasoningStep("hmm")])
    async with agent.stream("t") as turn:
        events = [event async for event in turn]
    context = [e for e in events if isinstance(e, ExternalAgentEvent) and e.kind == "context"]
    assert context[0].data == {"used": 10, "size": 100}
    assert context[0].session_id == "coder-1"
    assert any(e.type == "reasoning_delta" and e.text == "hmm" for e in events)
    payload = json.loads(context[0].model_dump_json())
    assert payload["type"] == "external_event"


async def test_end_step_and_outcome_override() -> None:
    agent = coder(["partial", EndStep("limit_reached", "max_tokens", text="cut")])
    result = await agent.run("t")
    assert (result.status, result.stop_reason, result.text) == (
        "limit_reached",
        "max_tokens",
        "cut",
    )


async def test_an_adapter_bug_is_raised_from_the_stream() -> None:
    agent = coder(["a", ValueError("bug in the adapter")])
    with pytest.raises(ValueError, match="bug in the adapter"):
        await agent.run("t")


async def test_a_broken_connection_ends_the_turn_as_interrupted() -> None:
    crash = ExternalAgentConnectionError(
        "devvi closed the connection: devvi exited with code 1; stderr: secret path C:/x",
        reason="the process exited with code 1",
        returncode=1,
        stderr_tail="secret path C:/x",
    )
    result = await coder(["working", crash]).run("t")
    assert result.status == "interrupted"
    assert result.stop_reason == "the process exited with code 1"
    assert result.raw["ambiguous"] is True and result.raw["returncode"] == 1
    assert result.raw["stderr_tail"] == "secret path C:/x"


async def test_a_broken_connection_reconnects_on_the_next_turn() -> None:
    agent = coder([ExternalAgentConnectionError("gone")], ["back"])
    first = await agent.run("a")
    assert first.status == "interrupted" and not agent.connected
    second = await agent.run("b")
    assert second.status == "completed"
    assert (agent.connects, agent.disconnects) == (2, 1)  # the broken one was released first
    await agent.aclose()
    assert agent.disconnects == 2


async def test_an_abandoned_turn_marks_the_agent_disconnected() -> None:
    agent = coder(["working", WaitStep(ignore_cancel=True)], cancel_grace_s=0.1)
    async with agent.stream("t") as turn:
        async for event in turn:
            if isinstance(event, TextDelta):
                asyncio.get_running_loop().call_later(0.05, turn.cancel)
    assert turn.result is not None and turn.result.raw["forced"] is True
    assert not agent.connected


async def test_continuing_a_session() -> None:
    agent = coder(["one"], ["two"])
    first = await agent.run("a")
    second = await agent.run("b", session=first.session_id)
    assert second.session_id == first.session_id
    assert agent.calls[1].resumed is True and agent.calls[1].session == "coder-1"


async def test_task_parts_and_bad_parts() -> None:
    from qagents import ImagePart, TextPart

    agent = coder(["ok"])
    await agent.run([TextPart(text="look"), ImagePart(url="https://example.invalid/x.png")])
    assert agent.prompts == ["look"]
    with pytest.raises(ConfigurationError):
        agent.stream([object()])  # type: ignore[list-item]


def test_configuration_is_checked() -> None:
    with pytest.raises(ConfigurationError):
        ScriptedExternalAgent([], on_permission="ask")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        ScriptedExternalAgent([], name="")


# -- the permission bridge -------------------------------------------------------------------


async def test_rules_decide_on_prefixed_names() -> None:
    agent = coder(
        [
            ToolStep("read", {"path": "a.py"}, ask=True, result="text"),
            ToolStep("web_fetch", {"url": "https://x.invalid"}, ask=True),
            ToolStep("bash", {"command": "rm -rf build"}, ask=True),
            ToolStep("edit", {"path": "b.py"}, ask=True),
        ]
    )
    rules = Permissions(allow=["coder.read"], ask=["coder.bash"], deny=["coder.web_fetch"])
    result = await agent.run("t", permissions=rules)
    by_tool = {p.external_tool: p for p in result.permissions}
    assert by_tool["read"].outcome == "allow" and by_tool["read"].decision == "allow"
    assert by_tool["web_fetch"].outcome == "deny" and by_tool["web_fetch"].decided_by == "rules"
    # ask without on_ask under the default "rules" posture: refused, and said so
    assert by_tool["bash"].outcome == "deny" and by_tool["bash"].decided_by == "no_approver"
    assert by_tool["bash"].reason == texts.EXTERNAL_PERMISSION_NO_APPROVER
    # no rule: the synthetic tool requires approval, so it asks, and is refused too
    assert by_tool["edit"].decision == "ask" and by_tool["edit"].outcome == "deny"
    assert [c.is_error for c in result.tool_calls] == [False, True, True, True]
    # the arguments came from the started tool call (correlated by its id)
    assert by_tool["bash"].arguments == {"command": "rm -rf build"}
    assert by_tool["bash"].tool_call_id == "coder-1-call-3"
    assert by_tool["bash"].fingerprint == permission_fingerprint(
        "coder.bash", {"command": "rm -rf build"}
    )
    footer = result.footer("coder")
    assert "3 permissions refused" in footer and 'bash: "rm -rf build"' in footer


async def test_every_decision_is_an_event() -> None:
    agent = coder([PermissionStep("bash", {"command": "ls"})])
    async with agent.stream("t", permissions=Permissions(allow=["coder.*"])) as turn:
        events = [e async for e in turn if isinstance(e, ExternalAgentEvent)]
    kinds = [e.kind for e in events]
    assert kinds == ["permission_requested", "permission_decided", "turn_finished"]
    decided = events[1].data
    assert decided["approved"] is True and decided["rule"] == "rules"
    assert decided["tool"] == "coder.bash" and decided["arguments"] == {"command": "ls"}


async def test_on_ask_answers_approve_decline_and_edits_are_refused() -> None:
    answers = iter(
        [True, Approval(approved=False, reason="not now"), Approval(approved=True, arguments={})]
    )
    seen: list[tuple[str, dict, Any]] = []

    def on_ask(ctx: Any, tool: Any, call: Any, reason: Any) -> Any:
        seen.append((tool.name, call.arguments, tool.requires_approval))
        return next(answers)

    agent = coder(
        [
            ToolStep("bash", {"command": "a"}, ask=True),
            ToolStep("bash", {"command": "b"}, ask=True),
            ToolStep("bash", {"command": "c"}, ask=True),
        ]
    )
    result = await agent.run("t", permissions=Permissions(on_ask=on_ask))
    assert [p.outcome for p in result.permissions] == ["allow", "deny", "deny"]
    assert [p.decided_by for p in result.permissions] == ["on_ask", "on_ask", "edited_arguments"]
    assert result.permissions[1].reason == "not now"
    assert seen[0] == ("coder.bash", {"command": "a"}, True)


async def test_on_ask_timeout_answers_cancelled() -> None:
    async def slow(*_: Any) -> bool:
        await asyncio.sleep(5)
        return True

    agent = coder([ToolStep("bash", {"command": "a"}, ask=True)], approval_timeout_s=0.05)
    result = await agent.run("t", permissions=Permissions(on_ask=slow))
    assert result.permissions[0].outcome == "cancel"
    assert result.permissions[0].decided_by == "timeout"
    assert agent.answers[0].outcome == "cancel"


async def test_a_failing_decision_answers_cancelled() -> None:
    class Broken(Permissions):
        async def decide(self, ctx: Any, tool: Any, call: Any) -> Any:
            raise RuntimeError("rules store unreachable")

    def raising_check(ctx: Any, tool: Any, call: Any) -> Any:
        raise RuntimeError("bug")

    agent = coder(
        [ToolStep("bash", {"command": "a"}, ask=True)],
        [ToolStep("bash", {"command": "a"}, ask=True)],
    )
    first = await agent.run("t", permissions=Broken())
    assert (first.permissions[0].outcome, first.permissions[0].decided_by) == ("cancel", "error")
    second = await agent.run("t", permissions=Permissions(can_use_tool=raising_check))
    assert second.permissions[0].outcome == "deny"  # Permissions refuses a raising check itself


async def test_can_use_tool_sees_the_external_tool() -> None:
    seen: list[Any] = []

    def check(ctx: Any, tool: Any, call: Any) -> bool:
        seen.append(tool)
        return tool.external_name == "read"

    agent = coder([PermissionStep("read"), PermissionStep("bash")])
    result = await agent.run("t", permissions=Permissions(can_use_tool=check))
    assert [p.outcome for p in result.permissions] == ["allow", "deny"]
    assert seen[0].name == "coder.read" and seen[0].agent == "coder"
    assert not seen[0].read_only and seen[0].requires_approval
    with pytest.raises(Exception, match="never runs"):
        seen[0].function()


def test_select_option_never_picks_allow_always() -> None:
    options = [
        {"optionId": "always", "kind": "allow_always"},
        {"optionId": "once", "kind": "allow_once"},
        {"optionId": "never", "kind": "reject_always"},
        {"optionId": "no", "kind": "reject_once"},
    ]
    assert PermissionAnswer(outcome="allow").select_option(options) == "once"
    assert PermissionAnswer(outcome="deny").select_option(options) == "no"
    assert PermissionAnswer(outcome="cancel").select_option(options) is None
    only_always = [{"optionId": "always", "kind": "allow_always"}]
    assert PermissionAnswer(outcome="allow").select_option(only_always) is None  # fail closed
    assert PermissionAnswer(outcome="deny").select_option(options[2:3]) == "never"


async def test_options_are_chosen_before_the_decision_is_recorded() -> None:
    bridge = PermissionBridge(prefix="coder", permissions=Permissions(mode="allow_all"))
    offered = [
        {"optionId": "allow-once", "kind": "allow_once"},
        {"optionId": "reject-once", "kind": "reject_once"},
    ]
    answer = await bridge.decide("bash", options=offered)
    assert (answer.outcome, answer.option_id) == ("allow", "allow-once")
    only_always = [{"optionId": "always", "kind": "allow_always"}]
    answer = await bridge.decide("bash", options=only_always)
    assert (answer.outcome, answer.decided_by, answer.option_id) == ("cancel", "no_option", None)
    assert bridge.records[-1].outcome == "cancel"
    assert bridge.records[-1].reason == texts.EXTERNAL_PERMISSION_NO_OPTION
    refusing = PermissionBridge(prefix="coder", permissions=Permissions(deny=["*"]))
    answer = await refusing.decide("bash", options=offered)
    assert (answer.outcome, answer.option_id) == ("deny", "reject-once")


async def test_bridge_used_directly() -> None:
    records: list[tuple[str, str]] = []
    bridge = PermissionBridge(
        prefix="other",
        permissions=Permissions(allow=["other.read"]),
        on_record=lambda record, kind: records.append((kind, record.tool)),
    )
    assert (await bridge.decide("read", {"path": "x"})).approved
    refused = await bridge.decide("bash")
    assert refused.outcome == "deny" and refused.decided_by == "no_approver"
    assert [r.outcome for r in bridge.records] == ["allow", "deny"]
    assert records[0] == ("permission_requested", "other.read")
    with pytest.raises(ConfigurationError):
        PermissionBridge(prefix="x", posture="maybe")  # type: ignore[arg-type]


async def test_a_decision_that_ends_while_the_turn_is_cancelled_is_recorded_as_cancelled() -> None:
    cancelling = False
    kinds: list[tuple[str, str]] = []

    def on_ask(ctx: Any, tool: Any, call: Any, reason: Any) -> bool:
        nonlocal cancelling
        cancelling = True  # the turn is cancelled while the human decides
        return True

    bridge = PermissionBridge(
        prefix="coder",
        permissions=Permissions(on_ask=on_ask),
        is_cancelling=lambda: cancelling,
        on_record=lambda record, kind: kinds.append((kind, record.outcome)),
    )
    answer = await bridge.decide("bash", {"command": "ls"}, options=[allow_once()])
    assert (answer.outcome, answer.decided_by, answer.option_id) == ("cancel", "cancelling", None)
    assert kinds[-1] == ("permission_decided", "cancel")  # the event says what was answered


def allow_once() -> dict[str, str]:
    return {"optionId": "a1", "kind": "allow_once"}


def test_fingerprints_depend_on_name_and_arguments() -> None:
    one = permission_fingerprint("coder.bash", {"command": "ls", "cwd": "."})
    assert one == permission_fingerprint("coder.bash", {"cwd": ".", "command": "ls"})
    assert one != permission_fingerprint("coder.bash", {"command": "ls -a", "cwd": "."})
    assert one != permission_fingerprint("other.bash", {"command": "ls", "cwd": "."})
    assert len(permission_fingerprint("x", {"n": float("nan")})) == 64


# -- cancellation --------------------------------------------------------------------------------


async def test_cancel_mid_turn() -> None:
    agent = coder(["working", WaitStep()])
    async with agent.stream("t") as turn:
        async for event in turn:
            if isinstance(event, TextDelta):  # cancel once the agent waits
                asyncio.get_running_loop().call_later(0.05, turn.cancel)
    assert turn.result is not None
    assert turn.result.status == "cancelled" and turn.result.stop_reason == "cancelled"
    assert agent.cancel_requests == ["cancelled"]
    assert agent.calls[0].cancelled == "cancelled"
    assert "forced" not in turn.result.raw


async def test_an_unresponsive_turn_is_abandoned_after_the_grace_period() -> None:
    agent = coder(["working", WaitStep(ignore_cancel=True)], cancel_grace_s=0.2)
    began = time.monotonic()
    async with agent.stream("t") as turn:
        async for event in turn:
            if isinstance(event, TextDelta):
                asyncio.get_running_loop().call_later(0.05, turn.cancel)
    assert time.monotonic() - began < 3
    assert turn.result is not None and turn.result.status == "cancelled"
    assert turn.result.raw["forced"] is True


async def test_leaving_the_stream_early_cancels_the_turn() -> None:
    agent = coder(["working", WaitStep()])
    async with agent.stream("t") as turn:
        async for event in turn:
            if isinstance(event, TextDelta):
                break
    assert turn.result is not None and turn.result.stop_reason == "closed"
    assert agent.calls[0].cancelled == "closed"


async def test_a_turn_cancelled_before_it_starts_never_runs() -> None:
    agent = coder(["never"])
    turn = agent.stream("t")
    turn.cancel()
    result = await turn.wait()
    assert result.status == "cancelled" and agent.calls == []


async def test_max_external_turns_stops_the_turn() -> None:
    agent = coder([ToolStep("a"), ToolStep("b"), ToolStep("c"), "never"], max_external_turns=2)
    result = await agent.run("t")
    assert result.status == "limit_reached" and result.stop_reason == "max_external_turns"
    assert [c.name for c in result.tool_calls] == ["a", "b", "c"]


async def test_aclose_cancels_live_turns_and_disconnects() -> None:
    agent = coder(["working", WaitStep()])
    turn = agent.stream("t")
    first = await turn.__anext__()
    assert isinstance(first, RunStarted)
    await agent.aclose()
    assert turn.result is not None and turn.result.status == "cancelled"
    assert agent.disconnects == 1 and not agent.connected
    await agent.aclose()
    assert agent.disconnects == 1


async def test_one_turn_at_a_time_per_workspace(tmp_path: Path) -> None:
    spans: list[tuple[str, float, float]] = []

    def stamp(label: str) -> Any:
        async def step(turn: ExternalTurnContext) -> None:
            began = time.monotonic()
            await asyncio.sleep(0.15)
            spans.append((label, began, time.monotonic()))

        return step

    one = coder([stamp("one")], name="one", workspace=tmp_path, track_changes=False)
    two = coder([stamp("two")], name="two", workspace=tmp_path, track_changes=False)
    await asyncio.gather(one.run("a"), two.run("b"))
    (_, _, first_end), (_, second_start, _) = sorted(spans, key=lambda span: span[1])
    assert second_start >= first_end


async def test_checkpoint_without_a_calling_run_is_a_no_op() -> None:
    results: list[bool] = []

    async def step(turn: ExternalTurnContext) -> None:
        results.append(await turn.checkpoint("k", {"v": 1}))
        assert turn.checkpoints == {"k": {"v": 1}}

    await coder([step]).run("t")
    assert results == [False]


# -- as a tool of a qagents agent ------------------------------------------------------------------


async def test_as_tool_schema_and_description() -> None:
    agent = coder(description="Edits code in the repository.")
    tool = agent.as_tool()
    assert tool.name == "coder" and not tool.read_only and not tool.idempotent
    assert tool.description.endswith(texts.EXTERNAL_AGENT_CONTINUE_HINT)
    assert set(tool.parameters["properties"]) == {"task", "session"}
    assert tool.parameters["required"] == ["task"]
    plain = coder(name="my coder").as_tool(continuable=False, requires_approval=True, timeout=60)
    assert plain.name == "my_coder" and plain.requires_approval
    assert set(plain.parameters["properties"]) == {"task"}
    assert plain.timeout == 60 + 5.0 + 5.0  # room for a graceful cancel inside the call
    assert plain.description == texts.EXTERNAL_AGENT_DESCRIPTION.format(name="my_coder")
    assert coder(resumable_turns=True).as_tool().idempotent


async def test_as_tool_child_run_events_usage_and_footer() -> None:
    external = coder(
        [
            "Looking. ",
            ToolStep("read", {"path": "a.py"}, result="..."),
            ToolStep("bash", {"command": "pytest"}, ask=True, result="1 passed"),
            Usage(requests=1, input_tokens=100, output_tokens=20),
            "All tests pass.",
        ]
    )
    model = ScriptedModel([call("coder", task="Make the tests pass."), "Done."])
    lead = Agent(
        model,
        name="lead",
        tools=[external.as_tool()],
        permissions=Permissions(allow=["coder.bash"]),
    )
    stream = lead.stream("Fix it")
    events = [event async for event in stream]
    result = stream.result
    assert result is not None and result.output == "Done."
    forwarded = [e for e in events if isinstance(e, SubagentEvent)]
    assert forwarded and all(e.agent == "coder" for e in forwarded)
    inner = [e.event for e in forwarded]
    assert isinstance(inner[0], RunStarted)
    assert inner[0].depth == 1 and inner[0].parent_run_id == result.run_id
    assert {e.tool_call_id for e in forwarded} == {tool_results(result)[0].tool_call_id}
    assert any(isinstance(e, ToolCallStarted) and e.call.name == "coder.bash" for e in inner)
    content = tool_results(result)[0].content
    assert content.startswith("All tests pass.\n\n[coder: session coder-1; completed; ")
    assert "2 tool calls (read 1, bash 1)" in content
    # The loop keeps the result the tool built: the footer's data is in its metadata.
    metadata = tool_results(result)[0].metadata
    assert metadata["external"]["session_id"] == "coder-1"
    assert metadata["external"]["status"] == "completed" and "ambiguous" not in metadata
    assert result.usage.input_tokens == 100 and result.usage.output_tokens == 20
    assert result.session is not None
    state = (await result.session.state()).values
    assert state[STATE_SESSIONS]["coder"][0]["session_id"] == "coder-1"


async def test_the_model_continues_a_session_with_the_session_argument() -> None:
    external = coder(["first"], ["second"])
    model = ScriptedModel(
        [
            call("coder", task="Start."),
            call("coder", task="Go on.", session="coder-1"),
            "Done.",
        ]
    )
    result = await Agent(model, tools=[external.as_tool()]).run("go")
    assert result.output == "Done."
    assert [c.session for c in external.calls] == [None, "coder-1"]
    assert external.calls[1].resumed


async def test_the_callers_rules_and_on_ask_apply() -> None:
    asked: list[str] = []

    def on_ask(ctx: Any, tool: Any, call: Any, reason: Any) -> bool:
        asked.append(call.name)
        return call.arguments.get("command") == "pytest"

    external = coder(
        [
            ToolStep("read", {"path": "a"}, ask=True),
            ToolStep("web_fetch", {"url": "u"}, ask=True),
            ToolStep("bash", {"command": "pytest"}, ask=True),
            ToolStep("bash", {"command": "rm -rf /"}, ask=True),
        ]
    )
    lead = Agent(
        ScriptedModel([call("coder", task="t"), "ok"]),
        tools=[external.as_tool()],
        permissions=Permissions(
            allow=["coder.read"], ask=["coder.bash"], deny=["coder.web_fetch"], on_ask=on_ask
        ),
    )
    result = await lead.run("go")
    assert [a.outcome for a in external.answers] == ["allow", "deny", "allow", "deny"]
    assert asked == ["coder.bash", "coder.bash"]
    assert "2 permissions refused" in tool_results(result)[0].content


async def test_an_agent_with_own_rules_inherits_the_callers_on_ask() -> None:
    external = coder(
        [ToolStep("bash", {"command": "ls"}, ask=True)],
        permissions=Permissions(ask=["coder.*"]),
    )
    lead = Agent(
        ScriptedModel([call("coder", task="t"), "ok"]),
        tools=[external.as_tool()],
        permissions=Permissions(on_ask=lambda *_: True),
    )
    await lead.run("go")
    assert external.answers[0].decided_by == "on_ask" and external.answers[0].approved


async def test_depth_limit_refuses_to_start() -> None:
    external = coder(["never"])
    lead = Agent(
        ScriptedModel([call("coder", task="t"), "ok"]),
        tools=[external.as_tool()],
        limits=Limits(max_depth=0),
    )
    result = await lead.run("go")
    content = tool_results(result)[0].content
    assert content == texts.EXTERNAL_AGENT_TOO_DEEP.format(name="coder", depth=1, limit=0)
    assert external.calls == []


async def test_cancelling_the_caller_cancels_the_turn() -> None:
    external = coder(["working", WaitStep()])
    lead = Agent(ScriptedModel([call("coder", task="t")]), tools=[external.as_tool()])
    stream = lead.stream("go")
    async for event in stream:
        if isinstance(event, SubagentEvent) and isinstance(event.event, TextDelta):
            stream.cancel()
    assert stream.result is not None and stream.result.status == "cancelled"
    assert external.cancel_requests == ["cancelled"]
    assert external.calls[0].cancelled == "cancelled"
    assert "outcome is unknown" in tool_results(stream.result)[0].content


async def test_the_callers_token_budget_stops_the_turn() -> None:
    external = coder(["a", Usage(requests=1, input_tokens=150), WaitStep(), "never"])
    lead = Agent(
        ScriptedModel([call("coder", task="t")]),
        tools=[external.as_tool()],
        limits=Limits(max_total_tokens=100),
    )
    result = await lead.run("go")
    assert result.status == "limit_reached" and result.stop_reason == "max_total_tokens"
    content = tool_results(result)[0].content
    assert content.startswith(
        texts.EXTERNAL_AGENT_STOPPED.format(name="coder", reason="max_total_tokens")
    )
    assert "session coder-1" in content
    assert result.usage.input_tokens == 150


async def test_an_unmetered_agent_is_refused_under_a_budget_unless_allowed() -> None:
    caps = ExternalCapabilities(token_usage=False, cancel_turn=True)
    refused = coder(["never"], capabilities=caps)
    lead = Agent(
        ScriptedModel([call("coder", task="t"), "ok"]),
        tools=[refused.as_tool()],
        limits=Limits(max_cost_usd=1.0),
    )
    result = await lead.run("go")
    assert tool_results(result)[0].content == texts.EXTERNAL_AGENT_UNMETERED.format(name="coder")
    assert refused.calls == []

    allowed = coder(["fine"], capabilities=caps, allow_unmetered=True)
    lead = Agent(
        ScriptedModel([call("coder", task="t"), "ok"]),
        tools=[allowed.as_tool()],
        limits=Limits(max_cost_usd=1.0),
    )
    result = await lead.run("go")
    assert tool_results(result)[0].content.startswith("fine")


async def test_as_tool_timeout_cancels_the_turn_and_names_the_session() -> None:
    external = coder(["working", WaitStep()], cancel_grace_s=0.5)
    lead = Agent(
        ScriptedModel([call("coder", task="t"), "ok"]), tools=[external.as_tool(timeout=0.2)]
    )
    result = await lead.run("go")
    content = tool_results(result)[0].content
    assert content.startswith(texts.EXTERNAL_AGENT_STOPPED.format(name="coder", reason="timeout"))
    assert "session coder-1" in content and "limit reached (timeout)" in content
    assert external.cancel_requests == ["timeout"]


async def test_an_interrupted_turn_is_an_error_with_unknown_outcome() -> None:
    crash = ExternalAgentConnectionError(
        "devvi closed the connection; stderr: C:/Users/secret",
        reason="the process exited with code 1",
        stderr_tail="C:/Users/secret",
    )
    external = coder(["working", crash])
    lead = Agent(ScriptedModel([call("coder", task="t"), "ok"]), tools=[external.as_tool()])
    result = await lead.run("go")
    part = tool_results(result)[0]
    assert part.is_error
    assert part.content.startswith(
        texts.EXTERNAL_AGENT_INTERRUPTED.format(
            name="coder", reason="the process exited with code 1", session="coder-1"
        )
    )
    assert "secret" not in part.content
    assert part.metadata["ambiguous"] is True  # copied from the ExternalToolError
    assert part.metadata["external"]["session_id"] == "coder-1"


async def test_failures_that_repeat_open_the_circuit_breaker() -> None:
    external = coder([EndStep("error", "401 Unauthorized: invalid API key")])
    model = ScriptedModel([call("coder", task="a"), call("coder", task="b"), "ok"])
    result = await Agent(model, tools=[external.as_tool()]).run("go")
    first, second = tool_results(result)
    assert first.content.startswith("coder is unavailable: 401 Unauthorized: invalid API key.")
    assert second.content == first.content  # the second call never reached the agent
    assert len(external.calls) == 1


async def test_an_ordinary_error_is_reported_and_can_be_retried() -> None:
    external = coder([EndStep("error", "turn failed: tool crashed")], ["ok now"])
    model = ScriptedModel([call("coder", task="a"), call("coder", task="b"), "done"])
    result = await Agent(model, tools=[external.as_tool()]).run("go")
    first, second = tool_results(result)
    assert first.is_error and first.content.startswith("coder failed: turn failed: tool crashed")
    assert not second.is_error and second.content.startswith("ok now")


class _CannotStart(ScriptedExternalAgent):
    async def connect(self) -> None:
        raise ExternalAgentConfigurationError("Cannot start 'devvi': it was not found on PATH.")


async def test_a_setup_failure_makes_the_agent_unavailable_for_the_run() -> None:
    external = _CannotStart([], name="coder")
    model = ScriptedModel([call("coder", task="a"), call("coder", task="b"), "ok"])
    result = await Agent(model, tools=[external.as_tool()]).run("go")
    first, second = tool_results(result)
    assert "is unavailable: Cannot start 'devvi'" in first.content
    assert second.content == first.content


async def test_unavailable_error_from_the_adapter() -> None:
    external = coder([ExternalAgentUnavailableError("MISSING_CREDENTIAL: no DEEPSEEK_API_KEY")])
    result = await external.run("t")
    assert result.status == "error" and result.raw["unavailable"] is True


def test_permanent_failures_are_recognised() -> None:
    for text in ("HTTP 401", "Invalid API key", "insufficient balance", "MISSING_CREDENTIAL"):
        assert is_permanent_failure(text), text
    for text in ("turn failed: tool crashed", "timeout after 30s", ""):
        assert not is_permanent_failure(text), text


# -- pausing for a human across a qagents pause (on_permission="pause") ------------------------------


async def test_pause_posture_end_to_end(tmp_path: Path) -> None:
    path = tmp_path / "sessions.db"
    external = coder(
        [ToolStep("bash", {"command": "rm -rf build"}, ask=True, result="removed")],
        [
            ToolStep("bash", {"command": "rm -rf build"}, ask=True, result="removed"),
            ToolStep("bash", {"command": "rm -rf dist"}, ask=True, result="removed dist"),
        ],
        on_permission="pause",
    )
    model = ScriptedModel(
        [call("coder", task="Clean the build."), call("coder", task="Go on.", session="coder-1")]
    )
    lead = Agent(model, tools=[external.as_tool()])

    paused = await lead.run("Clean up", session=Session(SQLiteStore(path), id="s1"))
    assert paused.status == "paused"
    pending = paused.pending_inputs[0]
    assert pending.reason == "permission"
    assert pending.metadata["agent"] == "coder"
    assert pending.metadata["external_session"] == "coder-1"
    assert pending.metadata["tool"] == "coder.bash"
    assert pending.metadata["arguments"] == {"command": "rm -rf build"}
    assert pending.metadata["fingerprint"] == permission_fingerprint(
        "coder.bash", {"command": "rm -rf build"}
    )
    assert external.answers[0].decided_by == "deferred"
    assert external.calls[0].cancelled == "permission"

    # A new process: the pause and its metadata come back from the store.
    session = Session(SQLiteStore(path), id="s1")
    restored = (await session.state()).pending_inputs[0]
    assert restored.metadata == pending.metadata
    approval = ExternalApproval.for_input(restored, approved=True)
    again = await lead.resume(session, results={restored.tool_call_id: approval})
    seen = model.requests[1].messages[-1].tool_results[0]
    assert seen.content.startswith('Approved: coder may run bash "rm -rf build" once.')
    # The continuation: the grant allows the same request once; a changed one pauses again.
    assert external.calls[1].session == "coder-1"
    assert external.calls[1].task.startswith(texts.EXTERNAL_AGENT_CONTINUE.split("\n")[0])
    assert [a.decided_by for a in external.answers] == ["deferred", "grant", "deferred"]
    assert again.status == "paused"
    assert again.pending_inputs[0].metadata["arguments"] == {"command": "rm -rf dist"}


async def test_a_declined_pause_refuses_the_request_on_continue() -> None:
    external = coder(
        [ToolStep("bash", {"command": "rm -rf build"}, ask=True)],
        [ToolStep("bash", {"command": "rm -rf build"}, ask=True), "I did not delete it."],
        on_permission="pause",
    )
    model = ScriptedModel(
        [call("coder", task="Clean."), call("coder", task="Go on.", session="coder-1"), "ok"]
    )
    lead = Agent(model, tools=[external.as_tool()])
    paused = await lead.run("Clean up")
    pending = paused.pending_inputs[0]
    done = await lead.resume(
        paused.session,
        results={pending.tool_call_id: ExternalApproval(approved=False, reason="keep it")},
    )
    assert done.status == "completed"
    assert external.answers[1].decided_by == "declined"
    assert (await done.session.state()).values.get("qagents.external.grants") is None


async def test_pause_posture_with_on_ask_does_not_pause() -> None:
    external = coder([ToolStep("bash", {"command": "ls"}, ask=True), "ok"], on_permission="pause")
    lead = Agent(
        ScriptedModel([call("coder", task="t"), "done"]),
        tools=[external.as_tool()],
        permissions=Permissions(on_ask=lambda *_: True),
    )
    result = await lead.run("go")
    assert result.status == "completed" and external.answers[0].decided_by == "on_ask"


async def test_pause_posture_without_a_calling_run_refuses() -> None:
    agent = coder([ToolStep("bash", {"command": "ls"}, ask=True), "after"], on_permission="pause")
    result = await agent.run("t")
    assert result.status == "completed"
    assert agent.answers[0].decided_by == "no_approver"


async def test_a_plain_answer_to_a_pause_grants_nothing() -> None:
    external = coder(
        [ToolStep("bash", {"command": "rm -rf build"}, ask=True)],
        [ToolStep("bash", {"command": "rm -rf build"}, ask=True)],
        on_permission="pause",
    )
    model = ScriptedModel(
        [call("coder", task="Clean."), call("coder", task="Go on.", session="coder-1")]
    )
    lead = Agent(model, tools=[external.as_tool()])
    paused = await lead.run("Clean up")
    pending = paused.pending_inputs[0]
    again = await lead.resume(paused.session, results={pending.tool_call_id: "yes, go ahead"})
    assert [a.decided_by for a in external.answers] == ["deferred", "deferred"]
    assert again.status == "paused"


def test_external_approval_shapes() -> None:
    approved = ExternalApproval(approved=True)
    assert approved.approved and not approved.is_error
    assert approved.metadata["qagents.external.approval"] == {"approved": True, "reason": None}
    declined = ExternalApproval(False, reason="no")
    assert not declined.approved and declined.is_error and "(no)" in declined.content
    assert json.loads(declined.model_dump_json())["type"] == "tool_result"


# -- journaling and repeats after a crash -------------------------------------------------------------


class _CrashAfterTool(Hooks):
    """Fails the run after the first tool returned, before its result is journaled."""

    def __init__(self) -> None:
        self.crashed = False

    async def after_tool(self, ctx: Any, call: Any, result: Any) -> Any:
        if not self.crashed:
            self.crashed = True
            raise RuntimeError("the process died")
        return result


async def test_a_finished_turn_never_runs_again_after_a_crash(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "s.db")
    external = coder(["The fix is in."], resumable_turns=True)
    model = ScriptedModel([call("coder", task="Fix it."), "Done."])
    lead = Agent(model, tools=[external.as_tool()], hooks=[_CrashAfterTool()])
    session = Session(store, id="crash")
    with pytest.raises(RuntimeError, match="the process died"):
        await lead.run("go", session=session)
    entries = await session.entries()
    checkpoints = [e.data for e in entries if e.kind == CHECKPOINT_KIND]
    assert [c["key"] for c in checkpoints] == ["external_session", "external_result"]
    assert checkpoints[0]["value"]["session_id"] == "coder-1"
    assert not any(e.kind == "tool_result" for e in entries)

    resumed = await lead.resume(Session(SQLiteStore(tmp_path / "s.db"), id="crash"))
    assert resumed.output == "Done."
    assert len(external.calls) == 1  # the idempotent call was repeated, the turn was not
    assert tool_results(resumed)[0].content.startswith("The fix is in.\n\n[coder: session coder-1")


async def test_a_turn_that_cannot_reattach_is_reported_as_unknown(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "s.db")
    external = coder(["The fix is in."])  # not resumable: the tool is not idempotent
    model = ScriptedModel([call("coder", task="Fix it."), "Checked."])
    lead = Agent(model, tools=[external.as_tool()], hooks=[_CrashAfterTool()])
    session = Session(store, id="crash")
    with pytest.raises(RuntimeError):
        await lead.run("go", session=session)
    assert not any(e.kind == CHECKPOINT_KIND for e in await session.entries())
    resumed = await lead.resume(session)
    assert resumed.output == "Checked."
    assert len(external.calls) == 1
    assert "outcome is unknown" in tool_results(resumed)[0].content


async def test_a_repeated_call_gets_the_checkpoints_of_the_first_attempt(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    started = asyncio.Event()

    def mark(turn: ExternalTurnContext) -> None:
        started.set()

    external = coder(["working", mark, WaitStep()], ["resumed fine"], resumable_turns=True)
    lead = Agent(ScriptedModel([call("coder", task="t"), "ok"]), tools=[external.as_tool()])
    task = asyncio.create_task(lead.run("go", session=Session(SQLiteStore(path), id="mid")))
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()  # the process "dies" while the external turn runs
    with pytest.raises(asyncio.CancelledError):
        await task

    resumed = await lead.resume(Session(SQLiteStore(path), id="mid"))
    assert resumed.output == "ok"
    assert len(external.calls) == 2  # repeated (idempotent), with what the first attempt saved
    assert external.calls[1].checkpoints == {
        "external_session": {"session_id": "coder-1", "cwd": None}
    }
    assert tool_results(resumed)[0].content.startswith("resumed fine")


# -- results and footers ---------------------------------------------------------------------------------


def test_footer_matches_the_design() -> None:
    result = ExternalResult(
        status="completed",
        stop_reason="end_turn",
        text="Patched.",
        session_id="5f0c",
        tool_calls=[
            ExternalToolCall(id=str(n), name=name)
            for n, name in enumerate(["bash", "edit", "bash", "read", "edit", "bash"])
        ],
        permissions=[
            ExternalPermission(
                tool="coder.bash",
                external_tool="bash",
                arguments={"command": "rm -rf build"},
                outcome="deny",
                decided_by="rules",
            )
        ],
        changed_files=["src/parser.py", "tests/test_parser.py"],
    )
    assert result.for_model("coder") == (
        "Patched.\n\n[coder: session 5f0c; completed; 6 tool calls (bash 3, edit 2, read 1); "
        '1 permission refused (bash: "rm -rf build"); '
        "changed: src/parser.py, tests/test_parser.py]"
    )
    empty = ExternalResult(status="cancelled", stop_reason="timeout", session_id="s")
    assert empty.for_model("coder") == (
        "(coder wrote no final message.)\n\n[coder: session s; cancelled (timeout); no tool calls]"
    )
    assert ExternalResult.model_validate_json(result.model_dump_json()) == result


def test_outcome_defaults() -> None:
    outcome = ExternalOutcome()
    assert outcome.status == "completed" and outcome.text is None and outcome.raw == {}


async def test_forwarded_events_project_onto_ag_ui() -> None:
    from qagents.integrations.agui import to_agui_events

    external = coder(["Looking. ", ToolStep("bash", {"command": "ls"}, result="a.py"), "Done."])
    lead = Agent(
        ScriptedModel([call("coder", task="t"), "ok"]),
        tools=[external.as_tool()],
        permissions=Permissions(allow=["coder.*"]),
    )
    stream = to_agui_events(lead.stream("go"), thread_id="t", run_id="r", subagents="native")
    kinds = [event["type"] async for event in stream]
    assert kinds[0] == "RUN_STARTED" and kinds[-1] == "RUN_FINISHED"
    assert "SUBAGENT_STARTED" in kinds and "SUBAGENT_FINISHED" in kinds


# -- git change tracking -------------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
async def test_changed_files_are_tracked_with_git(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-c", "user.email=t@example.invalid", "-c", "user.name=t", *args],
            cwd=repo,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    (repo / "keep.txt").write_text("same\n")
    (repo / "dirty.txt").write_text("v1\n")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    (repo / "dirty.txt").write_text("v2\n")  # dirty before the turn

    async def work(turn: ExternalTurnContext) -> None:
        (repo / "dirty.txt").write_text("v3, changed again during the turn\n")
        (repo / "new.txt").write_text("new\n")

    external = coder([work, ToolStep("write", {"path": "sub/written.txt"}), "done"], workspace=repo)
    (repo / "sub").mkdir()
    (repo / "sub" / "written.txt").write_text("x\n")  # also dirty before; reported via the tool
    result = await external.run("t")
    assert result.changed_files == ["dirty.txt", "new.txt", "sub/written.txt"]
    assert "v3, changed again" in await result.diff()
    assert result.workspace == str(repo)


# -- review fixes: the workspace's git config cannot make qagents run a program ------------------------


def _hook(tmp_path: Path, kind: str) -> tuple[str, Path]:
    """A command line that appends a line to a marker file each time it runs. As a filter
    (``kind="filter"``) it also copies stdin to stdout, as a clean filter must."""
    marker = tmp_path / f"{kind}-ran.txt"
    script = tmp_path / f"{kind}_hook.py"
    script.write_text(
        "import sys\n"
        "open(sys.argv[1], 'a').write('ran\\n')\n"
        "if sys.argv[2:] == ['filter']:\n"
        "    sys.stdout.write(sys.stdin.read())\n"
    )
    command = f'"{sys.executable}" "{script}" "{marker}"'.replace("\\", "/")
    return (f"{command} filter" if kind == "filter" else command), marker


def _committed_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-c", "user.email=t@example.invalid", "-c", "user.name=t", *args],
            cwd=repo,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    (repo / "a.txt").write_bytes(b"hello\n")
    git("add", "a.txt")
    git("commit", "-q", "-m", "init")
    return repo


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
async def test_a_workspace_git_config_cannot_make_qagents_run_a_program(tmp_path: Path) -> None:
    repo = _committed_repo(tmp_path)
    fsmonitor, fsmonitor_ran = _hook(tmp_path, "fsmonitor")
    clean, filter_ran = _hook(tmp_path, "filter")
    # What a file-write tool of the agent can do inside its workspace (.git included):
    (repo / ".gitattributes").write_text("*.txt filter=evil diff=evil\n")
    config = repo / ".git" / "config"
    config.write_text(
        config.read_text()
        + f"[core]\n\tfsmonitor = {fsmonitor}\n"
        + f'[filter "evil"]\n\tclean = {clean}\n\trequired = true\n'
        + f'[diff "evil"]\n\ttextconv = {clean}\n'
    )

    async def edit(turn: ExternalTurnContext) -> None:
        (repo / "a.txt").write_bytes(b"HELLO\n")  # the same size: git compares the content

    external = coder([edit, "done"], workspace=repo)
    result = await external.run("t")
    assert result.changed_files is not None and "a.txt" in result.changed_files
    diff = await result.diff()
    assert "+HELLO" in diff
    assert not fsmonitor_ran.exists(), "git status ran the fsmonitor hook from .git/config"
    assert not filter_ran.exists(), "git ran the clean filter or textconv from .git/config"


def test_git_gets_a_small_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    import qagents.external.changes as ext

    monkeypatch.setenv("QAGENTS_TEST_FAKE_API_KEY", "not-a-real-key")
    monkeypatch.setenv("GIT_DIR", str(Path("elsewhere").resolve()))
    env = {name.upper(): value for name, value in ext._git_env().items()}
    assert "QAGENTS_TEST_FAKE_API_KEY" not in env
    assert "GIT_DIR" not in env  # the parent's git variables do not redirect qagents' git
    assert "PATH" in env
    assert env["GIT_OPTIONAL_LOCKS"] == "0"
    assert env["GIT_ALLOW_PROTOCOL"] == ""  # no transport: no fetch, no remote helper
    assert env["GIT_NO_LAZY_FETCH"] == "1" and env["GIT_TERMINAL_PROMPT"] == "0"


@pytest.mark.parametrize("trusted_git", [False, True])
def test_git_is_never_taken_from_the_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trusted_git: bool
) -> None:
    import qagents.external.changes as ext

    workspace = tmp_path / "repo"
    (workspace / "bin").mkdir(parents=True)
    name = "git.exe" if sys.platform == "win32" else "git"
    for planted in (workspace / name, workspace / "bin" / name):
        planted.write_text("planted")
        planted.chmod(0o755)
    monkeypatch.chdir(workspace)  # Windows looks in the current directory first
    path = [".", str(workspace / "bin")]
    expected = None
    if trusted_git:
        trusted_directory = tmp_path / "trusted"
        trusted_directory.mkdir()
        expected = trusted_directory / name
        expected.write_text("trusted")
        expected.chmod(0o755)
        path.append(str(trusted_directory))
    monkeypatch.setenv("PATH", os.pathsep.join(path))

    found = ext._git_program(str(workspace))
    assert (Path(found) if found is not None else None) == expected


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
async def test_a_cancelled_or_timed_out_git_call_stops_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import qagents.external.changes as ext

    started: list[asyncio.subprocess.Process] = []
    running = asyncio.Event()
    real = asyncio.create_subprocess_exec

    async def slow_git(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        process = await real(sys.executable, "-c", "import time; time.sleep(60)", **kwargs)
        started.append(process)
        running.set()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", slow_git)
    task = asyncio.create_task(ext._git(str(tmp_path), "status"))
    await asyncio.wait_for(running.wait(), 20)
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert started[0].returncode is not None, "git was left running"

    assert await ext._git(str(tmp_path), "status", timeout=0.2) is None
    assert started[1].returncode is not None


# -- review fixes: process lifecycle and turn ordering -----------------------------------------------


class _NeedsConnection(ScriptedExternalAgent):
    """Like ACPAgent: a turn on a dead connection fails before anything is sent."""

    async def run_turn(self, turn: ExternalTurnContext) -> ExternalOutcome:
        if not self.connected:
            raise ExternalAgentConnectionError(f"{self.name} is not connected.", ambiguous=False)
        return await super().run_turn(turn)


async def test_a_turn_queued_behind_a_dying_one_reconnects() -> None:
    crash = ExternalAgentConnectionError("died", reason="the process exited with code 1")
    agent = _NeedsConnection([[WaitStep(0.2), crash], ["second"]], name="coder")
    first = asyncio.create_task(agent.run("a", session="s"))
    await asyncio.sleep(0.05)
    second = asyncio.create_task(agent.run("b", session="s"))  # waits for the session lock
    one, two = await asyncio.gather(first, second)
    assert one.status == "interrupted"
    assert (two.status, two.text) == ("completed", "second")
    assert agent.connects == 2 and agent.disconnects == 1


async def test_a_turn_that_was_never_sent_is_an_error_not_an_unknown_outcome() -> None:
    class _Gone(ScriptedExternalAgent):
        async def run_turn(self, turn: ExternalTurnContext) -> ExternalOutcome:
            raise ExternalAgentConnectionError("coder is not connected.", ambiguous=False)

    agent = _Gone([], name="coder")
    result = await agent.run("t")
    assert (result.status, result.stop_reason) == ("error", texts.EXTERNAL_REASON_CONNECTION_LOST)
    assert "ambiguous" not in result.raw


def test_an_agent_used_from_a_new_event_loop_connects_again() -> None:
    agent = coder(["one"], ["two"])
    assert asyncio.run(agent.run("a")).text == "one"
    assert asyncio.run(agent.run("b")).text == "two"  # Agent.run_sync does this
    assert (agent.connects, agent.disconnects) == (2, 1)


async def test_aclose_during_a_connect_leaves_the_agent_closed() -> None:
    class _SlowStart(ScriptedExternalAgent):
        async def connect(self) -> None:
            await asyncio.sleep(0.3)
            await super().connect()

    agent = _SlowStart([], name="coder")
    connecting = asyncio.create_task(agent.ensure_connected())
    await asyncio.sleep(0.05)
    await agent.aclose()
    await connecting
    assert not agent.connected
    assert (agent.connects, agent.disconnects) == (1, 1)


async def test_per_session_state_stays_small() -> None:
    import qagents.external.agent as ext

    agent = coder(*(["ok"] for _ in range(5)))
    for number in range(5):
        await agent.run("t", session=f"s{number}")
    assert len(agent._session_locks) == 0  # a lock nobody holds is dropped
    assert len(ext._WORKSPACE_LOCKS) == 0
    assert not hasattr(agent, "_turn_counts")  # run ids need no per-session counter


async def test_run_ids_are_unique_across_agents_and_stable_for_a_repeated_tool_call() -> None:
    # Two agents (or two processes) continuing one session never repeat a run id.
    first, second = coder(["one"]), coder(["two"])
    one = first.stream("a", session="shared")
    two = second.stream("b", session="shared")
    await one.wait()
    await two.wait()
    assert one.run_id and two.run_id and one.run_id != two.run_id
    assert one.run_id.startswith("ext:shared:") and two.run_id.startswith("ext:shared:")
    # A call of as_tool() derives its id from the operation id: the same on a repeat.
    ctx = RunContext(agent_name="lead", tool_call_id="c1", operation_id="run-1:0:c1")
    turns = [
        coder(["x"])._open_turn(
            "t", session="s", cwd=None, parent=ctx, permissions=None, label="coder"
        )
        for _ in range(2)
    ]
    for turn in turns:
        await turn.wait()
    assert turns[0].run_id == turns[1].run_id and turns[0].run_id.startswith("ext:s:")


# -- review fixes: grants, sessions and usage across cancels and crashes -------------------------------


async def test_a_grant_is_used_up_even_when_its_continuation_is_cancelled() -> None:
    request = PermissionStep("bash", {"command": "rm -rf build"})
    external = coder(
        [request],  # asks: the lead pauses
        [request, WaitStep()],  # the continuation uses the grant, then hangs until cancelled
        [request, "done"],  # a later call in the same session asks again
        on_permission="pause",
        cancel_grace_s=0.2,
    )
    tool = external.as_tool(timeout=60)
    session = Session()
    model = ScriptedModel(
        [call("coder", task="clean"), call("coder", task="clean", session="coder-1"), "stopped"]
    )
    lead = Agent(model, tools=[tool], limits=Limits(timeout_s=1.0))
    paused = await lead.run("go", session=session)
    assert paused.status == "paused"
    pending = paused.pending_inputs[0]
    answer = ExternalApproval.for_input(pending, approved=True)
    second = await lead.resume(session, results={pending.tool_call_id: answer})
    assert second.status == "limit_reached"
    assert STATE_GRANTS not in (await session.state()).values

    again = Agent(
        ScriptedModel([call("coder", task="again", session="coder-1"), "ok"]), tools=[tool]
    )
    third = await again.run("again", session=session)
    assert third.status == "paused", "the approval must not allow the request a second time"
    decided = [[(a.outcome, a.decided_by) for a in turn.answers] for turn in external.calls]
    assert decided == [[("cancel", "deferred")], [("allow", "grant")], [("cancel", "deferred")]]


async def test_the_session_is_remembered_when_the_call_is_cancelled() -> None:
    external = coder(["working", WaitStep()], resumable_turns=True)
    session = Session()
    lead = Agent(
        ScriptedModel([call("coder", task="t"), "ok"]),
        tools=[external.as_tool()],
        limits=Limits(timeout_s=0.5),
    )
    result = await lead.run("go", session=session)
    assert result.status == "limit_reached"
    remembered = (await session.state()).values[STATE_SESSIONS]["coder"]
    assert [entry["session_id"] for entry in remembered] == ["coder-1"]
    # Idempotent (re-attachable) is not harmless when cut short: the turn may have changed
    # files, so the calling model is told to check (TOOL_INTERRUPTED), not "cancelled".
    assert "outcome is unknown" in tool_results(result)[0].content


async def test_a_replayed_result_still_counts_the_turns_usage(tmp_path: Path) -> None:
    turn_usage = Usage(requests=1, input_tokens=120, output_tokens=40)
    external = coder([turn_usage, "The fix is in."], resumable_turns=True)
    model = ScriptedModel([call("coder", task="Fix it."), "Done."])
    lead = Agent(model, tools=[external.as_tool()], hooks=[_CrashAfterTool()])
    session = Session(SQLiteStore(tmp_path / "s.db"), id="crash")
    with pytest.raises(RuntimeError, match="the process died"):
        await lead.run("go", session=session)

    entries = await session.entries()
    kinds = [entry.kind for entry in entries]
    assert kinds[-2:] == [CHECKPOINT_KIND, "run_failed"]
    carried = Usage.model_validate(entries[-2].data["added_usage"])
    assert carried.input_tokens == 120  # the turn's usage rides on the result checkpoint
    resumed = await lead.resume(Session(SQLiteStore(tmp_path / "s.db"), id="crash"))
    assert len(external.calls) == 1  # replayed from the checkpoint, not run again
    assert resumed.usage.input_tokens == 120  # counted once (the scripted lead reports none)

    # The process died right after the checkpoint: nothing after it reached the journal.
    died = Session(SQLiteStore(tmp_path / "died.db"), id="crash")
    await died.append_entries(entries[:-1])
    again = Agent(ScriptedModel(["Done."]), tools=[external.as_tool()])
    resumed = await again.resume(died)
    assert len(external.calls) == 1
    assert resumed.usage.input_tokens == 120, "the replayed turn's usage was lost"


async def test_a_normal_result_counts_the_turns_usage_once(tmp_path: Path) -> None:
    turn_usage = Usage(requests=1, input_tokens=120, output_tokens=40)
    external = coder([turn_usage, "Done there."], resumable_turns=True)
    model = ScriptedModel([call("coder", task="Fix it."), "Done."])
    lead = Agent(model, tools=[external.as_tool()])
    session = Session(SQLiteStore(tmp_path / "s.db"), id="once")
    result = await lead.run("go", session=session)
    assert result.usage.input_tokens == 120
    entries = await session.entries()
    journaled = sum(
        Usage.model_validate(entry.data["added_usage"]).input_tokens
        for entry in entries
        if "added_usage" in entry.data
    )
    assert journaled == 120  # in the journal exactly once: a resume restores the same total


async def test_child_turns_started_directly_get_the_depth_and_budget_checks() -> None:
    unmetered = coder(["never"], capabilities=ExternalCapabilities(token_usage=False))

    @qagents_tool
    async def delegate(ctx: RunContext, task: str) -> str:
        """Delegate a task."""
        return (await unmetered.run(task, parent=ctx)).text

    lead = Agent(
        ScriptedModel([call("delegate", task="t"), "ok"]),
        tools=[delegate],
        limits=Limits(max_cost_usd=1.0),
    )
    result = await lead.run("go")
    assert tool_results(result)[0].content == texts.EXTERNAL_AGENT_UNMETERED.format(name="coder")
    assert unmetered.calls == []

    deep = coder(["never"])

    @qagents_tool
    async def nested(ctx: RunContext, task: str) -> str:
        """Delegate a task."""
        return (await deep.run(task, parent=ctx)).text

    lead = Agent(
        ScriptedModel([call("nested", task="t"), "ok"]), tools=[nested], limits=Limits(max_depth=0)
    )
    result = await lead.run("go")
    assert tool_results(result)[0].content == texts.EXTERNAL_AGENT_TOO_DEEP.format(
        name="coder", depth=1, limit=0
    )
    assert deep.calls == []


# -- review fixes: what the calling model reads ------------------------------------------------------


async def test_a_turn_that_broke_before_a_session_says_to_start_a_new_one() -> None:
    class _DiesAtStart(ScriptedExternalAgent):
        async def run_turn(self, turn: ExternalTurnContext) -> ExternalOutcome:
            raise ExternalAgentConnectionError("died", reason="the process exited with code 1")

    external = _DiesAtStart([], name="coder")
    lead = Agent(ScriptedModel([call("coder", task="t"), "ok"]), tools=[external.as_tool()])
    content = tool_results(await lead.run("go"))[0].content
    assert content == (
        texts.EXTERNAL_AGENT_INTERRUPTED_NO_SESSION.format(
            name="coder", reason="the process exited with code 1"
        )
        + "\n\n[coder: no session; interrupted (the process exited with code 1); no tool calls]"
    )
    assert "session=" not in content and "none" not in content

    pending = PendingInput(
        tool_call_id="c1",
        name="coder",
        reason="permission",
        metadata={"agent": "coder", "tool": "coder.bash", "external_tool": "bash"},
    )
    approved = ExternalApproval.for_input(pending, approved=True)
    assert approved.content == texts.EXTERNAL_AGENT_APPROVED_NO_SESSION.format(
        name="coder", tool="bash", detail="(no arguments)"
    )
    declined = ExternalApproval.for_input(pending, approved=False, reason="no")
    assert "session=" not in declined.content and declined.content.startswith("Declined:")


async def test_an_unavailable_message_has_one_period() -> None:
    class _BadVersion(ScriptedExternalAgent):
        async def connect(self) -> None:
            raise ExternalAgentConfigurationError("coder: it speaks protocol v2, not v1.")

    agent = _BadVersion([], name="coder")
    lead = Agent(ScriptedModel([call("coder", task="t"), "ok"]), tools=[agent.as_tool()])
    content = tool_results(await lead.run("go"))[0].content
    assert content == texts.EXTERNAL_AGENT_UNAVAILABLE.format(
        name="coder", error="coder: it speaks protocol v2, not v1"
    )
    assert ".." not in content


async def test_new_tool_arguments_are_an_event_and_never_change_an_emitted_part() -> None:
    def started(turn: ExternalTurnContext) -> None:
        turn.tool_started("c1", "bash", {"command": "ls"})

    def updated(turn: ExternalTurnContext) -> None:
        turn.tool_started("c1", "bash", {"command": "rm -rf build"})
        turn.tool_started("c1", "bash")  # an update without input changes nothing
        turn.tool_finished("c1", "done")

    events: list[Any] = []
    async with coder([started, updated, "ok"]).stream("t") as turn:
        async for event in turn:
            events.append(event)
    first = next(e for e in events if isinstance(e, ToolCallStarted))
    assert first.call.arguments == {"command": "ls"}  # what consumers already handled
    changed = [
        e for e in events if isinstance(e, ExternalAgentEvent) and e.kind == "tool_arguments"
    ]
    assert [e.data for e in changed] == [
        {"tool_call_id": "c1", "name": "coder.bash", "arguments": {"command": "rm -rf build"}}
    ]
    finished = next(e for e in events if isinstance(e, ToolCallFinished))
    assert finished.call.arguments == {"command": "rm -rf build"}
    assert turn.result is not None
    assert turn.result.tool_calls[0].arguments == {"command": "rm -rf build"}


async def test_permission_requests_see_the_latest_arguments_and_unreadable_ones_are_refused() -> (
    None
):
    seen: list[dict[str, Any]] = []
    answers: list[PermissionAnswer] = []

    async def can_use_tool(ctx: Any, tool: Any, call: Any) -> bool:
        seen.append(dict(call.arguments))
        return True

    async def steps(turn: ExternalTurnContext) -> None:
        turn.tool_started("c1", "bash", {"dry_run": True})
        turn.tool_started("c1", "bash", {})  # the input became empty
        answers.append(await turn.ask_permission("bash", tool_call_id="c1"))
        turn.tool_started("c2", "bash", {"dry_run": True})
        turn.tool_started("c2", "bash", {}, raw_arguments="rm -rf {", parse_error="not JSON")
        answers.append(await turn.ask_permission("bash", tool_call_id="c2"))

    agent = coder([steps, "ok"], permissions=Permissions(can_use_tool=can_use_tool))
    await agent.run("t")
    assert seen == [{}]  # the rules saw the updated, empty input; never the stale one
    assert answers[0].approved
    assert (answers[1].outcome, answers[1].decided_by) == ("cancel", "unreadable_arguments")
    assert answers[1].reason == texts.EXTERNAL_PERMISSION_UNREADABLE


async def test_the_suggested_session_id_is_the_same_on_every_read() -> None:
    ids: list[str] = []

    def read(turn: ExternalTurnContext) -> None:
        ids.extend([turn.suggested_session_id, turn.suggested_session_id])

    await coder([read, "ok"]).run("t")
    assert len(ids) == 2 and ids[0] == ids[1] and ids[0].startswith("qa-")


def test_the_graceful_cancel_threshold_is_the_documented_one() -> None:
    import qagents.external.tool as ext

    assert ext._graceful_deadline(120, 10) == 110  # cancelled 10 s early, with the session
    assert ext._graceful_deadline(20, 10) is None  # at most 2 x margin: the loop's timeout
    assert ext._graceful_deadline(9, 6) is None
    doc = " ".join((ExternalAgent.as_tool.__doc__ or "").split())
    assert "more than twice" in doc and "cancel_grace_s + 5" in doc


# -- review fixes, round 2 ------------------------------------------------------------------------------


async def test_a_grant_still_applies_when_the_agent_rewords_its_free_text() -> None:
    asked = {"command": "rm -rf build", "description": "Remove the build", "justification": "old"}
    reworded = {**asked, "description": "Clean the build output", "justification": "stale files"}
    other = {**reworded, "command": "rm -rf src"}
    external = coder(
        [ToolStep("bash", asked, ask=True)],
        [ToolStep("bash", reworded, ask=True, result="removed"), ToolStep("bash", other, ask=True)],
        on_permission="pause",
        grant_ignored_arguments=("description", "justification"),
    )
    model = ScriptedModel(
        [call("coder", task="Clean."), call("coder", task="Go on.", session="coder-1")]
    )
    lead = Agent(model, tools=[external.as_tool()])
    paused = await lead.run("Clean up")
    pending = paused.pending_inputs[0]
    assert pending.metadata["arguments"] == asked  # the human sees the whole request
    approval = ExternalApproval.for_input(pending, approved=True)
    again = await lead.resume(paused.session, results={pending.tool_call_id: approval})
    # The reworded request is the approved one; a changed command is decided again.
    assert [a.decided_by for a in external.answers] == ["deferred", "grant", "deferred"]
    assert again.status == "paused"


def test_by_default_a_grant_must_match_every_argument() -> None:
    request = {"command": "ls", "description": "list"}
    assert coder().grant_arguments("bash", request) == request
    fingerprints = []
    for grant_arguments in (None, lambda tool, args: {"command": args["command"]}):
        bridge = PermissionBridge(prefix="coder", grant_arguments=grant_arguments)
        fingerprints.append(bridge._matched_arguments("bash", request))
    assert fingerprints == [request, {"command": "ls"}]

    def broken(tool: str, arguments: Any) -> Any:
        raise RuntimeError("bug")

    bridge = PermissionBridge(prefix="coder", grant_arguments=broken)
    assert bridge._matched_arguments("bash", request) == request  # the stricter match


async def test_a_pause_continues_by_itself_for_a_tool_without_the_session_parameter() -> None:
    external = coder(
        [ToolStep("bash", {"command": "rm -rf build"}, ask=True)],
        [ToolStep("bash", {"command": "rm -rf build"}, ask=True, result="removed"), "Done."],
        on_permission="pause",
    )
    model = ScriptedModel([call("coder", task="Clean."), call("coder", task="Go on."), "ok"])
    lead = Agent(model, tools=[external.as_tool(continuable=False)])
    paused = await lead.run("Clean up")
    pending = paused.pending_inputs[0]
    assert pending.metadata["continuable"] is False
    approval = ExternalApproval.for_input(pending, approved=True)
    assert "session=" not in approval.content
    assert approval.content == texts.EXTERNAL_AGENT_APPROVED_NO_SESSION.format(
        name="coder", tool="bash", detail='"rm -rf build"'
    )
    done = await lead.resume(paused.session, results={pending.tool_call_id: approval})
    assert done.status == "completed"
    assert external.calls[1].session == "coder-1"  # the paused session, continued
    assert [a.decided_by for a in external.answers] == ["deferred", "grant"]
    assert (await done.session.state()).values.get(STATE_GRANTS) is None


async def test_an_interrupted_turn_of_a_tool_without_sessions_offers_no_session() -> None:
    crash = ExternalAgentConnectionError("gone", reason="the process exited with code 1")
    external = coder(["working", crash])
    lead = Agent(
        ScriptedModel([call("coder", task="t"), "ok"]),
        tools=[external.as_tool(continuable=False)],
    )
    content = tool_results(await lead.run("go"))[0].content
    assert content.startswith(
        texts.EXTERNAL_AGENT_INTERRUPTED_NOT_CONTINUABLE.format(
            name="coder", reason="the process exited with code 1"
        )
    )
    assert "session=" not in content


async def test_the_usage_of_a_child_turn_is_not_counted_twice_when_added_again() -> None:
    external = coder(["x", Usage(requests=1, input_tokens=100), "done"])
    worker = Agent(ScriptedModel(["sub"], usage=Usage(requests=1, input_tokens=100)), name="w")

    @qagents_tool
    async def delegate(ctx: RunContext, task: str) -> str:
        """Delegate the task."""
        turn = await external.run(task, parent=ctx)
        ctx.add_usage(turn.usage)  # the documented child-run pattern: recognised, not added
        answer = await worker.run(task, parent=ctx)
        ctx.add_usage(answer.usage)
        return turn.text + answer.output

    lead = Agent(ScriptedModel([call("delegate", task="b"), "Done."]), tools=[delegate])
    result = await lead.run("go")
    assert result.usage.input_tokens == 200


async def test_a_direct_turn_has_a_run_id_and_streams_its_text() -> None:
    agent = coder(["Hello ", ToolStep("read", {"path": "a"}), "world"])
    turn = agent.stream("t")
    assert turn.run_id is None  # known once run_started is emitted
    pieces = [piece async for piece in turn.text()]
    assert pieces == ["Hello ", "world"]
    assert turn.run_id is not None and turn.run_id.startswith("ext:coder-1:")
    assert turn.result is not None and turn.result.status == "completed"


async def test_breaking_out_of_the_events_cancels_the_turn() -> None:
    ran: list[str] = []
    agent = coder(["start", WaitStep(0.3), lambda turn: ran.append("edit after break"), "done"])
    turn = agent.stream("go")  # kept in a variable, but its loop is left: no async with
    async for event in turn:
        if event.type == "text_delta":
            break
    await asyncio.sleep(0.6)
    assert ran == []  # the agent was stopped, not left working unobserved
    assert turn.result is not None and turn.result.status == "cancelled"
    assert agent.calls[0].cancelled == "closed"


async def test_a_direct_turn_projects_onto_ag_ui_as_a_finished_run() -> None:
    from qagents.integrations.agui import to_agui_events

    agent = coder(["x", ToolStep("read", {"path": "a"}), "done"])
    async with agent.stream("go") as turn:
        kinds = [event["type"] async for event in to_agui_events(turn)]
    assert turn.result is not None and turn.result.status == "completed"
    assert kinds[0] == "RUN_STARTED" and kinds[-1] == "RUN_FINISHED"


async def test_committed_blocks_are_separated_and_token_streams_are_not() -> None:
    result = await coder(["one", WaitStep(0.01), "first done"]).run("t")
    assert result.text == "one\n\nfirst done"  # two committed messages, not "onefirst done"

    def blocks(turn: ExternalTurnContext) -> None:
        turn.text_delta("Hel", message_id="m1")  # one message streamed in pieces
        turn.text_delta("lo.", message_id="m1")
        turn.text_delta("Second block.", message_id="m2")
        turn.text_delta(" More.")  # no id: joined as it comes

    async with coder([blocks]).stream("t") as turn:
        deltas = [e.text async for e in turn if isinstance(e, TextDelta)]
    assert turn.result is not None and turn.result.text == "Hello.\n\nSecond block. More."
    assert "".join(deltas) == turn.result.text  # the events carry the separator too
    plain = await coder([lambda t: (t.text_delta("a"), t.text_delta("b"))]).run("t")
    assert plain.text == "ab"


async def test_a_turn_cancelled_before_it_starts_does_not_wait_for_the_workspace(
    tmp_path: Path,
) -> None:
    holding = asyncio.Event()

    async def hold(turn: ExternalTurnContext) -> None:
        holding.set()
        await asyncio.sleep(2.0)

    busy = coder([hold], name="busy", workspace=tmp_path, track_changes=False)
    queued = coder(["never"], name="queued", workspace=tmp_path, track_changes=False)
    first = asyncio.create_task(busy.run("a"))
    await asyncio.wait_for(holding.wait(), 5)
    began = time.monotonic()
    before = queued.stream("b")
    before.cancel()  # before it is first iterated
    early = await before.wait()
    waiting = queued.stream("c")
    asyncio.get_running_loop().call_later(0.1, waiting.cancel)  # while it waits for the lock
    late = await waiting.wait()
    assert time.monotonic() - began < 1.0
    for result in (early, late):
        assert (result.status, result.stop_reason) == ("cancelled", "cancelled")
        assert result.raw["not_started"] is True and "forced" not in result.raw
    assert queued.calls == [] and queued.connects == 0 and queued.cancel_requests == []
    await first


async def test_aclose_ends_turns_that_were_not_iterated_yet_without_connecting() -> None:
    agent = coder(["never"])
    turn = agent.stream("t")
    await agent.aclose()
    result = await turn.wait()
    assert (result.status, result.stop_reason) == ("cancelled", "closed")
    assert agent.connects == 0 and agent.calls == []


async def test_a_call_whose_time_runs_out_in_the_queue_says_it_did_not_start(
    tmp_path: Path,
) -> None:
    holding = asyncio.Event()

    async def hold(turn: ExternalTurnContext) -> None:
        holding.set()
        await asyncio.sleep(0.8)

    external = coder([hold, "A done"], ["B"], workspace=tmp_path, track_changes=False)
    first = Agent(ScriptedModel([call("coder", task="A"), "ok"]), tools=[external.as_tool()])
    second = Agent(
        ScriptedModel([call("coder", task="B"), "ok"]), tools=[external.as_tool(timeout=0.2)]
    )
    running = asyncio.create_task(first.run("go"))
    await asyncio.wait_for(holding.wait(), 5)
    queued = await second.run("go")
    content = tool_results(queued)[0].content
    assert content.startswith(texts.EXTERNAL_AGENT_BUSY.format(name="coder"))
    assert "cancelled (busy)" in content
    assert [call.task for call in external.calls] == ["A"]  # nothing was sent for B
    assert (await running).status == "completed"


class _SlowDisconnect(ScriptedExternalAgent):
    """A disconnect that waits (an idle session close, a slow process exit)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.started = asyncio.Event()
        self.gate = asyncio.Event()
        self.finished = 0

    async def disconnect(self) -> None:
        self.disconnects += 1
        self.started.set()
        await self.gate.wait()
        self.finished += 1


async def test_a_cancelled_aclose_still_finishes_the_disconnect() -> None:
    agent = _SlowDisconnect([], name="coder")
    await agent.ensure_connected()
    closing = asyncio.create_task(agent.aclose())
    await asyncio.wait_for(agent.started.wait(), 5)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert agent._opened  # not forgotten: the disconnect is still running
    again = asyncio.create_task(agent.aclose())  # waits for the same disconnect
    await asyncio.sleep(0.05)
    assert not again.done()
    agent.gate.set()
    await asyncio.wait_for(again, 5)
    assert (agent.disconnects, agent.finished) == (1, 1) and not agent._opened


async def test_a_reconnect_cancelled_during_its_disconnect_is_finished_by_the_next() -> None:
    agent = _SlowDisconnect([], name="coder")
    await agent.ensure_connected()
    agent.mark_disconnected()
    reconnecting = asyncio.create_task(agent.ensure_connected())
    await asyncio.wait_for(agent.started.wait(), 5)
    reconnecting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reconnecting
    agent.gate.set()
    await asyncio.wait_for(agent.ensure_connected(), 5)
    assert agent.connected and (agent.connects, agent.disconnects, agent.finished) == (2, 1, 1)


def test_the_summary_names_the_first_changed_files_and_counts_them() -> None:
    import qagents.external.contract as ext
    from qagents.external.tool import _tool_outcome

    files = [f"node_modules/pkg{n}/index.js" for n in range(5000)]
    result = ExternalResult(status="completed", session_id="s", changed_files=files)
    summary = ext._result_summary(result)
    assert summary["changed_files"] == files[: ext._SUMMARY_FILES]
    assert summary["changed_files_count"] == 5000
    assert len(json.dumps(summary)) < 10_000
    _, _, metadata = _tool_outcome("coder", result)
    assert metadata["external"]["changed_files_count"] == 5000
    assert "4980 more" in result.footer("coder")  # the footer names 20


def test_the_error_result_layout_comes_from_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    from qagents.external.tool import _tool_outcome

    monkeypatch.setattr(texts, "EXTERNAL_AGENT_ERROR_RESULT", "<{message}|{footer}>")
    monkeypatch.setattr(texts, "EXTERNAL_AGENT_STOPPED_WITH_TEXT", "{message} / {text}")
    stopped = ExternalResult(
        status="cancelled", stop_reason="timeout", text="  half done  \n", session_id="s-1"
    )
    content, is_error, _ = _tool_outcome("coder", stopped)
    message = texts.EXTERNAL_AGENT_STOPPED.format(name="coder", reason="timeout")
    assert is_error and content == f"<{message} / half done|{stopped.footer('coder')}>"
    failed = ExternalResult(status="error", stop_reason="disk full", session_id="s-1")
    content, is_error, _ = _tool_outcome("coder", failed)
    message = texts.EXTERNAL_AGENT_FAILED.format(name="coder", error="disk full")
    assert is_error and content == f"<{message}|{failed.footer('coder')}>"


def test_a_long_error_is_shortened_with_the_marker_from_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qagents.external.tool import _tool_outcome

    failed = ExternalResult(status="error", stop_reason="x" * 400, session_id="s-1")
    content, _, _ = _tool_outcome("coder", failed)
    default = content.split("\n\n")[0]
    assert default == texts.EXTERNAL_AGENT_FAILED.format(name="coder", error="x" * 297 + "...")
    monkeypatch.setattr(texts, "EXTERNAL_AGENT_LINE_SHORTENED", " [cut]")
    content, _, _ = _tool_outcome("coder", failed)
    error = content.split("\n\n")[0].removeprefix("coder failed: ")
    assert error == "x" * 294 + " [cut]" and len(error) == 300


def test_the_continue_hint_is_joined_by_the_template_in_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default = ScriptedExternalAgent([["done"]], name="coder", description="Codes.").as_tool()
    assert default.description == f"Codes. {texts.EXTERNAL_AGENT_CONTINUE_HINT}"
    monkeypatch.setattr(texts, "EXTERNAL_AGENT_DESCRIPTION_WITH_HINT", "{description}\n{hint}")
    built = ScriptedExternalAgent([["done"]], name="coder", description="Codes.").as_tool()
    assert built.description == f"Codes.\n{texts.EXTERNAL_AGENT_CONTINUE_HINT}"
    given = ScriptedExternalAgent([["done"]], name="coder").as_tool(description="Mine.")
    assert given.description == "Mine."  # a description you pass is used as it is


def test_diff_paths_are_split_to_fit_the_command_line() -> None:
    import qagents.external.changes as ext

    paths = [f"dir/file{n:05}.txt" for n in range(3000)]
    batches = ext._path_batches(paths, 1000)
    assert [path for batch in batches for path in batch] == paths
    assert all(sum(len(p) + 3 for p in batch) <= 1000 for batch in batches)
    assert ext._path_batches(["x" * 5000], 1000) == [["x" * 5000]]


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
async def test_change_tracking_stats_off_the_event_loop_and_diffs_in_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    import qagents.external.changes as ext

    repo = _committed_repo(tmp_path)
    tracked = sorted(path.name for path in repo.iterdir() if path.is_file())
    threads: set[int] = set()
    stat = ext._stat

    def recording_stat(path: str) -> tuple[int, int]:
        threads.add(threading.get_ident())
        return stat(path)

    async def work(turn: ExternalTurnContext) -> None:
        for name in tracked:
            (repo / name).write_text("changed\n")
        for number in range(30):
            (repo / f"new{number:02}.txt").write_text("new\n")

    monkeypatch.setattr(ext, "_stat", recording_stat)
    monkeypatch.setattr(ext, "_DIFF_ARGS_CHARS", 40)  # a few paths per git diff call
    result = await coder([work, "done"], workspace=repo).run("t")
    assert result.changed_files is not None and len(result.changed_files) == 30 + len(tracked)
    assert threads and threading.get_ident() not in threads  # never on the event loop
    diff = await result.diff()
    assert all(f"a/{name}" in diff for name in tracked)


# -- integration with the core: sessions the model may continue, footers, errors -------------------


async def test_the_model_continues_only_sessions_this_tool_opened() -> None:
    external = coder(["first"], ["second"])
    model = ScriptedModel(
        [
            call("coder", task="Go on.", session="made-up-1"),  # never opened by this tool
            call("coder", task="Start."),
            call("coder", task="Go on.", session="coder-1"),  # opened by the call before
            "Done.",
        ]
    )
    session = Session()
    result = await Agent(model, tools=[external.as_tool()]).run("go", session=session)
    refused, started, continued = tool_results(result)
    unknown = texts.EXTERNAL_AGENT_UNKNOWN_SESSION.format(session="made-up-1", name="coder")
    assert refused.is_error
    assert refused.content == texts.EXTERNAL_AGENT_FAILED.format(name="coder", error=unknown)
    assert refused.metadata == {"unknown_session": "made-up-1"}
    assert not started.is_error and not continued.is_error
    assert [c.session for c in external.calls] == [None, "coder-1"]  # never reached for it

    # Another tool's session is not this tool's, even in the same qagents session.
    other = coder(["other"], name="reviewer")
    later = Agent(
        ScriptedModel([call("reviewer", task="Look.", session="coder-1"), "ok"]),
        tools=[other.as_tool()],
    )
    (foreign,) = tool_results(await later.run("again", session=session))[-1:]
    assert foreign.is_error
    assert foreign.content == texts.EXTERNAL_AGENT_FAILED.format(
        name="reviewer",
        error=texts.EXTERNAL_AGENT_UNKNOWN_SESSION.format(session="coder-1", name="reviewer"),
    )
    assert other.calls == []

    # Code that continues a session itself may pass any id.
    direct = await coder(["mine"]).run("t", session="made-up-2")
    assert direct.status == "completed" and direct.session_id == "made-up-2"


async def test_a_session_dropped_from_the_full_list_is_not_called_never_opened() -> None:
    """The tool keeps its latest 50 sessions. Once the list is full, an older session the
    tool did open may have been dropped, so its refusal must not say it was never opened."""

    @qagents_tool
    def fill(ctx: RunContext) -> str:
        """Stand for 50 sessions the coder tool opened earlier in this conversation."""
        old = [{"session_id": f"old-{i}", "cwd": None, "backend": None} for i in range(50)]
        ctx.state[STATE_SESSIONS] = {"coder": old}
        return "filled"

    external = coder(["new"], ["kept"])
    model = ScriptedModel(
        [
            call("fill"),
            call("coder", task="Start."),  # opens coder-1: old-0 drops out of the list
            call("coder", task="Go on.", session="old-0"),  # this tool opened it, earlier
            call("coder", task="Go on.", session="old-1"),  # still kept: it continues
            "Done.",
        ]
    )
    result = await Agent(model, tools=[fill, external.as_tool()]).run("go")
    _, started, dropped, kept = tool_results(result)
    assert not started.is_error and not kept.is_error
    forgotten = texts.EXTERNAL_AGENT_FORGOTTEN_SESSION.format(
        session="old-0", name="coder", limit=50
    )
    assert dropped.is_error
    assert dropped.content == texts.EXTERNAL_AGENT_FAILED.format(name="coder", error=forgotten)
    assert "was not opened by" not in dropped.content
    assert dropped.metadata == {"unknown_session": "old-0"}
    assert [c.session for c in external.calls] == [None, "old-1"]  # never reached for old-0


async def test_the_footer_of_a_turn_that_never_began_names_no_session() -> None:
    class _RefusesToContinue(ScriptedExternalAgent):
        async def run_turn(self, turn: ExternalTurnContext) -> ExternalOutcome:
            # Refused before the agent confirmed the session: begin() is never called.
            return ExternalOutcome(status="error", stop_reason="that session cannot continue")

    agent = _RefusesToContinue([], name="coder")
    result = await agent.run("t", session="coder-9")
    assert result.status == "error" and result.session_id == ""
    assert result.footer("coder") == (
        "coder: no session; error (that session cannot continue); no tool calls"
    )


def test_the_external_errors_live_in_errors_and_keep_their_paths() -> None:
    from qagents import errors, external

    for name in (
        "ExternalAgentError",
        "ExternalAgentConfigurationError",
        "ExternalAgentConnectionError",
        "ExternalAgentProtocolError",
        "ExternalAgentTimeoutError",
        "ExternalAgentUnavailableError",
        "ExternalToolError",
    ):
        assert getattr(external, name) is getattr(errors, name), name
        assert name in external.__all__
    assert issubclass(errors.ExternalAgentConfigurationError, errors.ConfigurationError)
    assert issubclass(errors.ExternalToolError, errors.ToolError)
    lost = errors.ExternalAgentConnectionError("gone")
    assert lost.reason == texts.EXTERNAL_REASON_CONNECTION_LOST and lost.ambiguous


def test_no_stand_in_for_the_run_internals_is_left() -> None:
    import importlib
    import inspect
    import pkgutil

    from qagents import external
    from qagents.integrations import mcp_server

    private = re.compile(
        r"\._(run|journal|save_state|counted|absorbed|usage_to_journal|usage_lock|stuck_threads)\b"
    )
    package = [
        importlib.import_module(f"qagents.external.{info.name}")
        for info in pkgutil.iter_modules(external.__path__)
    ]
    assert len(package) >= 7  # the modules of the qagents.external package
    for module in (external, *package, mcp_server):
        found = private.findall(inspect.getsource(module))
        assert not found, f"{module.__name__} uses run internals: {found}"
    assert not hasattr(external, "write_checkpoint") and not hasattr(mcp_server, "_RunCalls")


# -- the test markers --------------------------------------------------------------------------------


def test_opt_in_markers_need_an_explicit_selection() -> None:
    from .conftest import selected_markers

    assert selected_markers("devvi_mock") == {"devvi_mock"}
    assert selected_markers("devvi_mock or devvi_live") == {"devvi_mock", "devvi_live"}
    assert selected_markers("not devvi_live") == set()
    assert selected_markers("devvi_fake and not devvi_mock") == {"devvi_fake"}
    assert selected_markers("") == set()


# -- texts of the second literals pass ------------------------------------------------------------


def test_the_three_cutters_agree_when_the_marker_is_longer_than_the_limit(monkeypatch) -> None:
    """``ExternalPermission.detail``, the tool's ``_first_line`` and Devvi's ``_first_line``
    cut with the same marker; with a limit shorter than the marker, all three leave the
    marker alone (a negative slice would keep almost the whole text)."""
    from qagents.external.tool import _first_line as tool_first_line
    from qagents.integrations.devvi import sdk as devvi_sdk

    marker = " [shortened]"
    monkeypatch.setattr(texts, "EXTERNAL_AGENT_LINE_SHORTENED", marker)
    record = ExternalPermission(
        tool="c.bash",
        external_tool="bash",
        arguments={"command": "x" * 40},
        outcome="deny",
        decided_by="r",
    )
    assert record.detail(10) == marker
    assert tool_first_line("y" * 40, 10) == marker
    assert devvi_sdk._first_line("z" * 40, 10) == marker
    assert record.detail(20) == '"xxxxxxx' + marker  # the marker counts toward the limit
    assert devvi_sdk._first_line("z" * 40, 20) == "z" * 8 + marker


@pytest.mark.devvi_fake
async def test_new_values_of_the_moved_texts_reach_their_readers(
    monkeypatch, tmp_path: Path
) -> None:
    """A sample of the texts the second pass moved to the prompts module, at least one from
    every module it changed: a new value reaches what the calling model, the external agent
    or the MCP client reads. Where that is cheap, the check is on the end text: the calling
    model's tool result, or the MCP client's call result."""
    from qagents.external.tool import _tool_outcome
    from qagents.integrations.acp import mapping
    from qagents.integrations.devvi import DevviAgent, launcher, patches
    from qagents.integrations.devvi import events as devvi_events
    from qagents.integrations.mcp_server import collect_tools, tool_server

    from . import fixture_tools as fx
    from .fakes.devvi import acp_agent, sdk_agent
    from .test_devvi_acp import HANDSHAKE, initialize_answer
    from .test_devvi_acp import write_script as acp_script
    from .test_mcp_server import mcp_client, text_of

    for name, value in {
        "EXTERNAL_AGENT_TEXT_BLOCK_SEPARATOR": " // ",  # external/turn.py
        "EXTERNAL_AGENT_FOOTER_LIST_SEPARATOR": " | ",  # external/contract.py
        "EXTERNAL_AGENT_LINE_SHORTENED": "~",  # external/permissions.py
        "EXTERNAL_AGENT_DECLINED_REASON": " because {reason}",
        "DEVVI_ERROR_WITHOUT_MESSAGE": "failed quietly",  # devvi/events.py
        "ACP_TERMINAL_PLACEHOLDER": "<t {terminal}>",  # acp/mapping.py
        "ACP_MEDIA_PLACEHOLDER": "<{kind} {mime_type}>",
        "ACP_RESOURCE_LINK_PLACEHOLDER": "<link {uri}>",
        "ACP_RESOURCE_PLACEHOLDER": "<res {uri}>",
        "ACP_SETUP_FAILED": "{name} would not start ({summary})",  # acp/client.py
        "ACP_PROTOCOL_VERSION": "ACP {version} is not {supported}",
        "DEVVI_EXPOSE_NEEDS_HTTP": "{name} has no MCP over HTTP",  # devvi/acp.py
        "DEVVI_DID_NOT_START": "Devvi stayed down: {why}",  # devvi/sdk.py
        "DEVVI_INITIALIZE_FAILED": "init said {message}",
        "DEVVI_CLI_DID_NOT_START": "no CLI ({how})",  # devvi/launcher.py
        "DEVVI_PROBE_EXIT": "code {code}",
        "EXTERNAL_SETUP_LIST_SEPARATOR": " + ",
        "DEVVI_PERMISSION_MODE_UNKNOWN": "modes: {modes}",
        "DEVVI_PATCH_ROW_TYPE": "row {index} is a {type}",  # devvi/patches.py
        "TOOL_SERVER_DUPLICATE_NAME": "twice: {name}",  # mcp_server.py
        "UNKNOWN_TOOL_SEPARATOR": " / ",
        "UNKNOWN_TOOL_NONE": "nothing",
        "TOOL_FAILED_EXCEPTION": "<{type}> {error}",
    }.items():
        monkeypatch.setattr(texts, name, value)

    async def first_result(agent: Any, name: str) -> str:
        """What the calling model reads from one call of ``agent.as_tool()``."""
        model = ScriptedModel([call(name, task="Go."), "Done."])
        lead = Agent(model, tools=[agent.as_tool()], limits=Limits(max_turns=4))
        try:
            return tool_results(await lead.run("go"))[0].content
        finally:
            await agent.aclose()

    # The calling model's tool result: committed text blocks, refused requests (one cut
    # short), and the footer's lists.
    external = coder(
        [
            "First.",
            "Second.",
            PermissionStep("bash", {"command": "y" * 200}),
            PermissionStep("edit"),
        ]
    )
    content = await first_result(external, "coder")
    assert content.startswith("First. // Second.")
    refused = content.split("2 permissions refused (")[1]
    assert refused.startswith('bash: "' + "y" * 78 + "~ | edit: (no arguments))")
    footer = ExternalResult(status="completed", changed_files=["a.py", "b.py"]).footer("coder")
    assert "changed: a.py | b.py" in footer

    # A paused request declined with a reason, answered through the pending input.
    pending = PendingInput(
        tool_call_id="c1",
        name="coder",
        reason="permission",
        metadata={"agent": "coder", "tool": "coder.bash", "external_session": "s-1"},
    )
    declined = ExternalApproval.for_input(pending, False, reason="too risky").content
    assert "because too risky." in declined

    # Devvi: a turn end of kind "error" without a message, as the calling model reads it.
    outcome = devvi_events.turn_end_outcome({"kind": "error", "error": {}})
    result = ExternalResult(status=outcome.status, stop_reason=outcome.stop_reason)
    assert _tool_outcome("coder", result)[0].startswith("coder failed: failed quietly")

    # ACP: the notes in a tool call's result content.
    pieces = [
        {"type": "terminal", "terminalId": "t1"},
        {"type": "content", "content": {"type": "image", "mimeType": "image/png"}},
        {"type": "content", "content": {"type": "resource_link", "uri": "file:///a"}},
        {"type": "content", "content": {"type": "resource", "resource": {"uri": "file:///b"}}},
    ]
    assert mapping.content_text(pieces).split("\n") == [
        "<t t1>",
        "<image image/png>",
        "<link file:///a>",
        "<res file:///b>",
    ]

    # Setup errors, as the calling model reads them through as_tool(). The ACP client: an
    # agent that speaks another protocol version.
    answer = initialize_answer()
    answer["result"]["protocolVersion"] = 2
    for folder in ("v2", "http", "sdk", "ws"):
        (tmp_path / folder).mkdir()
    old = acp_agent(tmp_path / "v2", acp_script(tmp_path / "v2.jsonl", [HANDSHAKE[0], answer]))
    assert (await first_result(old, "coder")).startswith(
        "coder is unavailable: coder would not start (ACP 2 is not 1)"
    )
    # Devvi over ACP, with in-process tools and no MCP over HTTP.
    rows = [HANDSHAKE[0], initialize_answer(mcpCapabilities={})]
    no_http = acp_agent(
        tmp_path / "http",
        acp_script(tmp_path / "http.jsonl", rows),
        expose_tools=[fx.lookup_customer],
    )
    assert (await first_result(no_http, "coder")).startswith(
        "coder is unavailable: coder has no MCP over HTTP"
    )
    # Devvi's SDK transport: initialize refused.
    sdk, _ = sdk_agent(tmp_path / "sdk", None, "--fail-initialize", "no adapter", name="coder")
    assert (await first_result(sdk, "coder")).startswith(
        "coder is unavailable: Devvi stayed down: init said no adapter"
    )
    # The launcher: `devvi --version` exits with an error.
    broken_cli = DevviAgent(
        workspace=tmp_path / "ws",
        transport="sdk",
        command=[sys.executable, "-c", "import sys; sys.exit(2)"],
        home=tmp_path / "home",
        name="coder",
    )
    assert (await first_result(broken_cli, "coder")).startswith(
        "coder is unavailable: no CLI (code 2)"
    )
    with pytest.raises(ExternalAgentConfigurationError, match=r"^modes: read-only \+ "):
        launcher.build_environment(tmp_path, permission_mode="none", environ={})
    # Patch rows, and the tools a server is asked to serve.
    with pytest.raises(ExternalAgentConfigurationError, match="^row 0 is a int$"):
        patches.check_rows([5])
    with pytest.raises(ConfigurationError, match="^twice: lookup_customer$"):
        await collect_tools([fx.lookup_customer, fx.lookup_customer])

    # The MCP tool server: an unknown tool (with tools and without), and a tool that raised.
    @qagents_tool
    def explode() -> str:
        """Fail with an exception that is not a ToolError."""
        raise ValueError("bad input")

    async with (
        tool_server([fx.lookup_customer, explode]) as server,
        mcp_client(server.url, server.headers) as client,
    ):
        unknown = await client.call_tool("nope", {})
        raised = await client.call_tool("explode", {})
    assert text_of(unknown).endswith("Available tools: lookup_customer / explode.")
    assert text_of(raised) == "'explode' failed: <ValueError> bad input"
    async with tool_server([]) as server, mcp_client(server.url, server.headers) as client:
        empty = await client.call_tool("nope", {})
    assert text_of(empty).endswith("Available tools: nothing.")
