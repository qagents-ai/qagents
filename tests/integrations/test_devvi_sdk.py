"""DevviAgent on the SDK transport, against a fake Devvi that replays Devvi's own snapshots.

Covers the event mapping (design 2.5), approvals through qagents' rules, limits and usage
(test plan 6), crash re-attach with a SQLiteStore (test plan 7), errors, and the read-only
session log reader.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from qagents import Agent, Approval, Limits, Permissions, Session, SQLiteStore
from qagents import prompts as devvi_texts
from qagents import prompts as texts
from qagents.events import ModelResponded, RunStarted, SubagentEvent, TextDelta
from qagents.external import ExternalAgentConfigurationError, ExternalAgentEvent, ExternalApproval
from qagents.integrations.devvi import TokenPricing, logs, turn_end_outcome, usage_from_devvi
from qagents.integrations.devvi.sdk import (
    BASELINE_CHECKPOINT,
    MESSAGE_CHECKPOINT,
    SdkSettings,
    SdkTransport,
    content_blocks,
)
from qagents.messages import ImagePart, TextPart
from qagents.models.testing import FunctionModel, ScriptedModel, call

from .fakes.devvi import (
    CHILD,
    FAKE_HOME,
    ROOT,
    SDK_FIXTURES,
    answers,
    assistant,
    fake_command,
    fixture_lines,
    methods,
    notification,
    requests,
    sdk_agent,
    starts,
    step,
    tool_call,
    tool_result,
    turn,
    wait_for_log_text,
    wait_until_gone,
    write_script,
)

pytestmark = pytest.mark.devvi_fake

PRICES = TokenPricing(input=1.0, output=2.0, cache_read=0.1)


def tool_results(result: Any) -> list[Any]:
    return [part for message in result.messages for part in message.tool_results]


def inner(events: list[Any]) -> list[Any]:
    return [event.event for event in events if isinstance(event, SubagentEvent)]


async def run_turn(agent: Any, task: str = "Do the task.", **options: Any) -> tuple[list[Any], Any]:
    async with agent.stream(task, **options) as stream:
        events = [event async for event in stream]
    return events, stream.result


def expected(name: str) -> dict[str, Any]:
    return json.loads((SDK_FIXTURES / name).read_text(encoding="utf-8"))


# -- the event mapping, fixture by fixture ------------------------------------------------------


async def test_text_turn_events_usage_and_wire(tmp_path: Path) -> None:
    agent, record = sdk_agent(tmp_path, "text-turn.notifications.jsonl")
    async with agent:
        events, result = await run_turn(agent, "Reply with exactly: SDK snapshot OK")
    assert [event.type for event in events] == [
        "run_started", "turn_started", "reasoning_delta", "text_delta", "model_responded", "external_event",
    ]  # fmt: skip
    assert events[-1].kind == "turn_finished"
    assert result.status == "completed" and result.stop_reason == "end_turn"
    assert result.raw["agent_stop_reason"] == "completed"  # Devvi's own word
    assert result.text == expected("text-turn.result.json")["finalResponse"] == "SDK snapshot OK"
    assert result.backend == "devvi-sdk@0.2.1-rc.3"
    assert result.session_id.startswith("qa-") and len(result.session_id) == 35
    usage = result.usage
    assert (usage.requests, usage.input_tokens, usage.output_tokens, usage.reasoning_tokens) == (
        1,
        1769,
        24,
        19,
    )
    assert usage.cost_usd is None and usage.unpriced_requests == 1
    response = next(event for event in events if isinstance(event, ModelResponded)).response
    assert (response.provider, response.model) == ("deepseek-official", "deepseek-v4-flash")
    assert methods(record) == [
        "initialize",
        "session/resume",
        "session/prompt",
        "session/wait",
        "shutdown",
    ]
    init = requests(record, "initialize")[0]["params"]
    assert init == {
        "cwd": str(tmp_path / "workspace"), "provider": "deepseek-official",
        "model": "deepseek-v4-flash", "approvalRequests": True,
    }  # fmt: skip
    prompt = requests(record, "session/prompt")[0]["params"]
    assert prompt == {
        "sessionId": result.session_id,
        "contentBlocks": [{"type": "text", "text": "Reply with exactly: SDK snapshot OK"}],
    }
    argv = starts(record)[0]["argv"]
    assert argv[argv.index("--profile") + 1] == "sdk" and "--patch" in argv
    assert result.raw["message_id"].startswith("msg-")


async def test_bash_tool_turn_maps_tool_calls_jobs_and_priced_usage(tmp_path: Path) -> None:
    agent, _ = sdk_agent(tmp_path, "bash-tool.notifications.jsonl", pricing=PRICES)
    async with agent:
        events, result = await run_turn(agent)
    started = [event for event in events if event.type == "tool_call_started"]
    assert len(started) == 1 and started[0].call.name == "devvi.bash"
    assert started[0].call.arguments == {
        "command": "echo dsh-sdk-proof-7391",
        "description": "Run the echo command as requested",
    }
    assert started[0].call.raw_arguments.startswith('{"command"')
    finished = next(event for event in events if event.type == "tool_call_finished")
    assert finished.result.content == "dsh-sdk-proof-7391\n" and not finished.result.is_error
    jobs = [
        event for event in events if isinstance(event, ExternalAgentEvent) and event.kind == "job"
    ]
    assert [job.data["type"] for job in jobs] == ["job/registered", "job/settled"]
    assert [event.turn for event in events if event.type == "turn_started"] == [1, 2]
    assert result.text == expected("bash-tool.result.json")["finalResponse"]
    assert result.tool_calls[0].name == "bash" and result.tool_calls[0].finished
    usage = result.usage
    assert (usage.requests, usage.input_tokens, usage.cached_input_tokens, usage.output_tokens) == (
        2,
        3684,
        3328,
        113,
    )
    assert usage.cost_usd == pytest.approx((356 * 1.0 + 113 * 2.0 + 3328 * 0.1) / 1e6)
    assert usage.unpriced_requests == 0
    responded = [event for event in events if isinstance(event, ModelResponded)]
    assert responded[0].response.finish_reason == "tool_calls"
    assert responded[0].response.message.tool_calls[0].name == "devvi.bash"
    assert "devvi: session" in result.footer("devvi") and "1 tool call (bash 1)" in result.footer(
        "devvi"
    )


async def test_session_log_turns_and_approvals_decided_by_rules(tmp_path: Path) -> None:
    agent, record = sdk_agent(tmp_path, "approval.session.v4.jsonl")
    async with agent:
        allow = Permissions(allow=["devvi.fixture-write"])
        events, first = await run_turn(
            agent, "Reply with exactly the word: ONE. No tools.", permissions=allow
        )
        deny = Permissions(deny=["devvi.fixture-*"])
        later, second = await run_turn(
            agent,
            "Reply with exactly the word: TWO. No tools.",
            session=first.session_id,
            permissions=deny,
        )
    assert (first.text, second.text) == ("ONE", "TWO")
    assert [item["outcome"] for item in answers(record)] == ["allowed-once", "rejected"]
    assert [(p.outcome, p.decided_by) for p in first.permissions] == [("allow", "rules")]
    assert [(p.outcome, p.decided_by) for p in second.permissions] == [("deny", "rules")]
    kinds = [event.kind for event in events if isinstance(event, ExternalAgentEvent)]
    assert kinds[:4] == [
        "approval_asked",
        "permission_requested",
        "permission_decided",
        "approval_decided",
    ]
    decided = [
        event
        for event in later
        if isinstance(event, ExternalAgentEvent) and event.kind == "approval_decided"
    ]
    assert decided[0].data["outcome"] == "rejected"
    run_started = [event for event in later if isinstance(event, RunStarted)][0]
    assert run_started.resumed and run_started.session_id == first.session_id
    assert "1 permission refused (fixture-write" in second.footer("devvi")
    assert len(starts(record)) == 2  # one process per call


async def test_approvals_use_the_arguments_of_the_requesting_sessions_own_call(
    tmp_path: Path,
) -> None:
    """Call ids are unique only within one session: each request is decided on its own
    session's call, and a call its session never started is refused without a decision."""
    rm, ls, ls_la = {"command": "rm -rf build"}, {"command": "ls"}, {"command": "ls -la"}
    other = "{{session:1}}-child-2"
    started = "subagent.started"

    def ask(call_id: str, session: str = ROOT) -> dict[str, Any]:
        return {"$approval": {"toolName": "bash", "callId": call_id, "sessionId": session}}

    script = write_script(
        tmp_path / "ids.jsonl",
        turn(
            step(),
            tool_call("c1", "bash", ls),
            {"method": started, "params": {"parentSessionId": ROOT, "childSessionId": CHILD}},
            {"method": started, "params": {"parentSessionId": ROOT, "childSessionId": other}},
            tool_call("c1", "bash", rm, session=CHILD),
            tool_call("c1", "bash", ls, session=other),  # a later call with the same id
            tool_call("c2", "bash", ls_la),
            ask("c1", CHILD),
            ask("c1"),
            ask("c1", other),
            ask("c2", CHILD),  # only the root has a call c2
            assistant("Done."),
        ),
    )
    seen: list[dict[str, Any]] = []

    def reviewer(ctx: Any, tool: Any, request: Any, reason: Any) -> Approval:
        seen.append(dict(request.arguments))
        return Approval(approved=request.arguments.get("command") == "ls")

    agent, record = sdk_agent(
        tmp_path, script, permissions=Permissions(ask=["devvi.bash"], on_ask=reviewer)
    )
    async with agent:
        events, result = await run_turn(agent)
    assert [item["outcome"] for item in answers(record)] == [
        "rejected",  # the child's own c1 (rm -rf build), not the root's or the other child's
        "allowed-once",
        "allowed-once",
        "cancelled",
    ]
    assert seen == [rm, ls, ls]  # the unknown call never reached the rules
    assert [(p.arguments, p.outcome) for p in result.permissions] == [
        (rm, "deny"),
        (ls, "allow"),
        (ls, "allow"),
    ]
    refused = [
        event
        for event in events
        if isinstance(event, ExternalAgentEvent) and event.kind == "approval_refused"
    ]
    assert len(refused) == 1 and refused[0].data == {
        "toolName": "bash",
        "callId": "c2",
        "reason": "unknown_call",
    }
    assert refused[0].session_id == f"{result.session_id}-child-1"


async def test_max_tokens_is_limit_reached_and_the_session_continues(tmp_path: Path) -> None:
    agent, _ = sdk_agent(tmp_path, "max-tokens-continue.session.v4.jsonl")
    async with agent:
        _, first = await run_turn(
            agent, "This turn is cut off at the output limit while calling a tool."
        )
        _, second = await run_turn(agent, "Continue.", session=first.session_id)
    assert (first.status, first.stop_reason, first.text) == (
        "limit_reached",
        "max_tokens",  # what ACP reports for the same outcome
        "Starting the write now.",
    )
    assert first.raw["agent_stop_reason"] == "max-tokens"
    assert second.status == "completed" and second.text.startswith(
        "The previous reply hit the output limit"
    )


async def test_subagent_events_and_child_usage(tmp_path: Path) -> None:
    agent, _ = sdk_agent(tmp_path, "subagent-spawn-in-process.tokenized.notifications.jsonl")
    async with agent:
        events, result = await run_turn(agent)
    external = [event for event in events if isinstance(event, ExternalAgentEvent)]
    child = result.raw["subagents"][0]
    assert child == f"{result.session_id}-child-1"
    started = next(event for event in external if event.kind == "subagent_started")
    assert started.session_id == child and started.data["parent_session_id"] == result.session_id
    assert started.data["depth"] == 1
    inner_types = [event.data["type"] for event in external if event.kind == "subagent_event"]
    assert inner_types == ["turn/start", "step/start", "assistant/message", "turn/end"]
    message = next(
        event
        for event in external
        if event.kind == "subagent_event" and event.data["type"] == "assistant/message"
    )
    assert message.data["text"] == "child answer 42."
    # The child's message carries no usage in this snapshot: one request of unknown cost.
    assert (
        message.data["usage"]["requests"] == 1 and message.data["usage"]["unpriced_requests"] == 1
    )
    finished = next(event for event in external if event.kind == "subagent_finished")
    assert finished.data["status"] == "ok" and finished.data["text"] == "child answer 42."
    assert result.text == expected("subagent-spawn-in-process.result.json")["finalResponse"]
    assert result.usage.requests == 3  # the child's model request counts too
    assert (
        result.usage.input_tokens == (135 + 1664) + (19 + 1920)
        and result.usage.unpriced_requests == 3
    )
    texts_seen = [event.text for event in events if isinstance(event, TextDelta)]
    assert texts_seen == [
        "child answer 42."
    ]  # only the root's text; the child's is in subagent_event


async def test_unknown_event_types_are_ignored_and_counted(tmp_path: Path) -> None:
    script = write_script(
        tmp_path / "unknown.jsonl",
        turn(
            step(),
            notification("future/thing", {"x": 1}),
            notification("todo/write", {}),
            assistant("Done.", usage={"inputTokens": 5, "outputTokens": 1}),
        ),
        turn(
            step(),
            notification("future/thing", {"x": 2}),
            assistant("Again."),
            token="{{message:2}}",
            number=2,
        ),
    )
    agent, _ = sdk_agent(tmp_path, script)
    async with agent:
        events, result = await run_turn(agent)
        raw_events, again = await run_turn(agent, session=result.session_id)
    assert result.raw["ignored_events"] == 1 and result.text == "Done."
    assert not any(
        isinstance(event, ExternalAgentEvent) and event.kind in ("raw", "unknown")
        for event in events
    )
    assert again.usage.unpriced_requests == 1  # an assistant message without usage is not free

    (tmp_path / "raw").mkdir()
    agent, _ = sdk_agent(tmp_path / "raw", script, raw_events=True)
    async with agent:
        events, _ = await run_turn(agent)
    kinds = [event.kind for event in events if isinstance(event, ExternalAgentEvent)]
    assert "unknown" in kinds and "raw" in kinds


# -- limits and usage through as_tool (test plan 6) -----------------------------------------------


async def test_usage_reaches_the_calling_run_per_step(tmp_path: Path) -> None:
    agent, _ = sdk_agent(tmp_path, "bash-tool.notifications.jsonl", pricing=PRICES)
    lead = Agent(
        ScriptedModel([call("devvi", task="Run the echo."), "Done."]),
        tools=[agent.as_tool()],
        limits=Limits(max_cost_usd=1.0),
    )
    stream = lead.stream("go")
    events = [event async for event in stream]
    result = stream.result
    assert result is not None and result.output == "Done."
    responded = [event for event in inner(events) if isinstance(event, ModelResponded)]
    assert len(responded) == 2
    assert result.usage.input_tokens == 3684 and result.usage.output_tokens == 113
    assert result.usage.cost_usd == pytest.approx((356 + 226 + 332.8) / 1e6)
    content = tool_results(result)[0].content
    assert (
        content.startswith("dsh-sdk-proof-7391\n\n[devvi: session qa-")
        and "1 tool call (bash 1)" in content
    )


async def test_an_exhausted_token_budget_cancels_the_turn_and_stops_devvi(tmp_path: Path) -> None:
    script = write_script(
        tmp_path / "budget.jsonl",
        turn(
            step(),
            assistant("Thinking.", usage={"inputTokens": 1769, "outputTokens": 24}),
            {"$hang": True},
            close=False,
        ),
    )
    agent, record = sdk_agent(tmp_path, script)
    lead = Agent(
        ScriptedModel([call("devvi", task="t")]),
        tools=[agent.as_tool()],
        limits=Limits(max_total_tokens=100),
    )
    result = await lead.run("go")
    assert result.status == "limit_reached" and result.stop_reason == "max_total_tokens"
    content = tool_results(result)[0].content
    assert content.startswith(
        texts.EXTERNAL_AGENT_STOPPED.format(name="devvi", reason="max_total_tokens")
    )
    assert "session qa-" in content
    assert result.usage.input_tokens == 1769
    assert methods(record)[-1] == "shutdown"


async def test_max_external_turns_counts_devvi_steps(tmp_path: Path) -> None:
    script = write_script(
        tmp_path / "steps.jsonl",
        turn(step(1), assistant("One.", step=1), step(2), {"$hang": True}, close=False),
    )
    agent, record = sdk_agent(tmp_path, script, max_external_turns=1)
    async with agent:
        _, result = await run_turn(agent)
    assert (result.status, result.stop_reason) == ("limit_reached", "max_external_turns")
    assert methods(record)[-1] == "shutdown"


async def test_a_cost_budget_without_prices_refuses_before_starting_devvi(tmp_path: Path) -> None:
    agent, record = sdk_agent(tmp_path, "text-turn.notifications.jsonl")
    lead = Agent(
        ScriptedModel([call("devvi", task="t"), "ok"]),
        tools=[agent.as_tool()],
        limits=Limits(max_cost_usd=1.0),
    )
    result = await lead.run("go")
    assert tool_results(result)[0].content == texts.EXTERNAL_AGENT_UNAVAILABLE.format(
        name="devvi", error=devvi_texts.DEVVI_NEEDS_PRICING
    )
    assert starts(record) == []  # no process was started


# -- crash re-attach with SQLiteStore (test plan 7) ------------------------------------------------


async def _crash_during(tmp_path: Path, script: Path, marker: str, *fake: str) -> tuple[Path, Path]:
    """Run a lead agent until Devvi's log shows ``marker``, then kill the run (a crash)."""
    agent, record = sdk_agent(tmp_path, script, *fake)
    lead = Agent(
        ScriptedModel([call("devvi", task="Reply with exactly: SDK snapshot OK")]),
        tools=[agent.as_tool()],
    )
    db = tmp_path / "runs.db"
    task = asyncio.create_task(lead.run("go", session=Session(SQLiteStore(db), id="crash")))
    try:
        await wait_for_log_text(tmp_path / "devvi-home" / "sessions", marker)
    finally:
        task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    return db, record


def _hang_before_idle() -> list[dict[str, Any]]:
    lines = fixture_lines("text-turn.notifications.jsonl")
    assert lines[-1]["method"] == "session.status"
    return [*lines[:-1], {"$hang": True}]


async def test_a_finished_turn_is_recovered_from_the_log_and_not_prompted_again(
    tmp_path: Path,
) -> None:
    script = write_script(tmp_path / "finished.jsonl", _hang_before_idle())
    db, record = await _crash_during(tmp_path, script, '"turn/end"')
    entries = await Session(SQLiteStore(db), id="crash").entries()
    keys = [entry.data["key"] for entry in entries if entry.kind == "tool_checkpoint"]
    assert keys == [BASELINE_CHECKPOINT, "external_session", MESSAGE_CHECKPOINT]
    assert not any(entry.kind == "tool_result" for entry in entries)

    second, record2 = sdk_agent(
        tmp_path, write_script(tmp_path / "none.jsonl", []), record=tmp_path / "record2.jsonl"
    )
    lead = Agent(ScriptedModel(["Done."]), tools=[second.as_tool()])
    resumed = await lead.resume(Session(SQLiteStore(db), id="crash"))
    assert resumed.output == "Done."
    content = tool_results(resumed)[0].content
    assert content.startswith("SDK snapshot OK\n\n[devvi: session qa-")
    assert "session/prompt" not in methods(
        record2
    )  # re-attached: the finished turn was not repeated
    assert methods(record2)[:2] == ["initialize", "session/resume"]
    first_id = requests(record, "session/prompt")[0]["params"]["sessionId"]
    assert requests(record2, "session/resume")[0]["params"]["sessionId"] == first_id
    assert resumed.usage.input_tokens == 1769  # the recovered turn's usage is counted


@pytest.mark.parametrize("ignore_shutdown", [False, True], ids=["shutdown", "killed"])
async def test_an_open_turn_gets_the_resume_prompt_after_a_crash(
    tmp_path: Path, ignore_shutdown: bool
) -> None:
    script = write_script(
        tmp_path / "open.jsonl",
        turn(
            step(),
            assistant("Working.", calls=[("c1", "bash", {"command": "make"})]),
            tool_call("c1", "bash", {"command": "make"}),
            {"$hang": True},
            prompt="Reply with exactly: SDK snapshot OK",
            close=False,
        ),
        turn(
            step(),
            assistant("Finished after the interruption."),
            prompt="(resume)",
            token="{{message:2}}",
            number=2,
        ),
    )
    fake = ("--ignore-shutdown",) if ignore_shutdown else ()
    db, _ = await _crash_during(tmp_path, script, '"tool/call"', *fake)

    second, record2 = sdk_agent(tmp_path, script, record=tmp_path / "record2.jsonl")
    lead = Agent(ScriptedModel(["Done."]), tools=[second.as_tool()])
    resumed = await lead.resume(Session(SQLiteStore(db), id="crash"))
    assert resumed.output == "Done."
    sent = requests(record2, "session/prompt")[0]["params"]["contentBlocks"]
    assert sent == [{"type": "text", "text": texts.EXTERNAL_AGENT_RESUME}]
    assert tool_results(resumed)[0].content.startswith("Finished after the interruption.")
    log = logs.read_session_log(
        next((tmp_path / "devvi-home" / "sessions").rglob("session.v4.jsonl"))
    )
    first = log.turns()[0]
    assert first.end_kind == ("interrupted" if ignore_shutdown else "aborted")


async def test_a_log_that_cannot_be_read_sends_the_task_with_a_warning(tmp_path: Path) -> None:
    rows = [
        {
            "id": "session-persistence-jsonl",
            "config": {"root": str(tmp_path / "devvi-home" / "sessions"), "compression": "zstd"},
        }
    ]
    script = write_script(
        tmp_path / "finished.jsonl",
        _hang_before_idle(),
        turn(assistant("Checked."), token="{{message:2}}", number=2),
    )
    agent, _ = sdk_agent(tmp_path, script, patches=rows)
    lead = Agent(
        ScriptedModel([call("devvi", task="Reply with exactly: SDK snapshot OK")]),
        tools=[agent.as_tool()],
    )
    db = tmp_path / "runs.db"
    task = asyncio.create_task(lead.run("go", session=Session(SQLiteStore(db), id="crash")))
    try:
        await wait_for_log_text(tmp_path / "devvi-home" / "sessions", '"turn/end"')
    finally:
        task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    second, record2 = sdk_agent(tmp_path, script, record=tmp_path / "record2.jsonl", patches=rows)
    resumed = await Agent(ScriptedModel(["Done."]), tools=[second.as_tool()]).resume(
        Session(SQLiteStore(db), id="crash")
    )
    sent = requests(record2, "session/prompt")[0]["params"]["contentBlocks"][0]["text"]
    assert sent == devvi_texts.DEVVI_RESUME_UNKNOWN.format(
        task="Reply with exactly: SDK snapshot OK"
    )
    assert tool_results(resumed)[0].content.startswith("Checked.")


@pytest.mark.parametrize("answered", [True, False], ids=["prompt-taken", "prompt-maybe-sent"])
async def test_a_repeat_whose_session_devvi_lost_is_unknown_and_not_prompted_again(
    tmp_path: Path, answered: bool
) -> None:
    """The journaled session is gone (here: Devvi now uses another home). Whether or not
    the first attempt saw Devvi take its prompt, the task is not sent again."""
    if answered:  # crash after the prompt's answer: both checkpoints are journaled
        script = write_script(tmp_path / "taken.jsonl", _hang_before_idle())
        marker = '"turn/end"'
    else:  # crash while waiting for the answer: only the session id is journaled
        script = write_script(
            tmp_path / "sent.jsonl",
            turn({"$no_answer": True}, prompt="Reply with exactly: SDK snapshot OK", close=False),
        )
        marker = '"agent/inbox/spliced"'
    db, _ = await _crash_during(tmp_path, script, marker)
    entries = await Session(SQLiteStore(db), id="crash").entries()
    keys = [entry.data["key"] for entry in entries if entry.kind == "tool_checkpoint"]
    first = [BASELINE_CHECKPOINT, "external_session"]
    assert keys == ([*first, MESSAGE_CHECKPOINT] if answered else first)

    second, record2 = sdk_agent(
        tmp_path, script, record=tmp_path / "record2.jsonl", home=tmp_path / "moved-home"
    )
    lead = Agent(ScriptedModel(["Checked."]), tools=[second.as_tool()])
    resumed = await lead.resume(Session(SQLiteStore(db), id="crash"))
    assert resumed.output == "Checked."
    assert methods(record2) == ["initialize", "session/resume", "shutdown"]  # no prompt
    session = requests(record2, "session/resume")[0]["params"]["sessionId"]
    reason = devvi_texts.DEVVI_NOT_REATTACHED.format(session=session)
    content = tool_results(resumed)[0].content
    assert content.startswith(  # no session to continue: Devvi no longer has it
        texts.EXTERNAL_AGENT_INTERRUPTED_NO_SESSION.format(name="devvi", reason=reason)
    )
    assert "outcome is unknown" in content


# -- pausing for a human across a qagents pause (on_permission="pause") ----------------------------


async def test_the_pause_posture_round_trips_through_a_store(tmp_path: Path) -> None:
    rm = {"command": "rm -rf build"}
    script = write_script(
        tmp_path / "pause.jsonl",
        turn(
            step(),
            assistant("Cleaning.", calls=[("c1", "bash", rm)]),
            tool_call("c1", "bash", rm),
            {
                "$approval": {"toolName": "bash", "callId": "c1", "reason": "deletes files"},
                "$on": {"cancelled": [{"$hang": True}]},
            },
            prompt="Clean the build.",
            close=False,
        ),
        turn(
            step(),
            tool_call("c2", "bash", rm),
            {
                "$approval": {"toolName": "bash", "callId": "c2"},
                "$on": {
                    "allowed-once": [tool_result("c2", "removed"), assistant("Cleaned.", step=2)]
                },
            },
            prompt="(continue)",
            token="{{message:2}}",
            number=2,
        ),
    )
    agent, record = sdk_agent(tmp_path, script, on_permission="pause")

    def reply(request: Any) -> Any:
        last = request.messages[-1]
        if last.role != "tool":
            return call("devvi", task="Clean the build.")
        found = re.search(r"session=(qa-[0-9a-f]+)", last.tool_results[0].content)
        return call("devvi", task="Go on.", session=found.group(1)) if found else "Done."

    lead = Agent(FunctionModel(reply), tools=[agent.as_tool()])
    db = tmp_path / "runs.db"
    paused = await lead.run("Clean up", session=Session(SQLiteStore(db), id="s1"))
    assert paused.status == "paused"
    pending = paused.pending_inputs[0]
    assert pending.metadata["tool"] == "devvi.bash" and pending.metadata["arguments"] == rm
    assert pending.metadata["external_session"].startswith("qa-")
    session = Session(SQLiteStore(db), id="s1")
    restored = (await session.state()).pending_inputs[0]
    again = await lead.resume(
        session,
        results={restored.tool_call_id: ExternalApproval.for_input(restored, approved=True)},
    )
    assert again.output == "Done."
    assert [item["outcome"] for item in answers(record)] == ["cancelled", "allowed-once"]
    prompts = [
        item["params"]["contentBlocks"][0]["text"] for item in requests(record, "session/prompt")
    ]
    assert prompts[1].startswith(texts.EXTERNAL_AGENT_CONTINUE.splitlines()[0])
    assert (
        requests(record, "session/resume")[1]["params"]["sessionId"]
        == pending.metadata["external_session"]
    )
    assert tool_results(again)[-1].content.startswith("Cleaned.")


# -- cancellation and errors ----------------------------------------------------------------------


@pytest.mark.parametrize("ignore_shutdown", [False, True], ids=["shutdown", "killed"])
async def test_cancel_stops_the_process(tmp_path: Path, ignore_shutdown: bool) -> None:
    script = write_script(
        tmp_path / "hang.jsonl", turn(step(), assistant("Working."), {"$hang": True}, close=False)
    )
    fake = ("--ignore-shutdown",) if ignore_shutdown else ()
    agent, record = sdk_agent(tmp_path, script, *fake)
    async with agent:
        async with agent.stream("t") as stream:
            async for event in stream:
                if isinstance(event, TextDelta):
                    stream.cancel()
        result = stream.result
    assert result.status == "cancelled" and result.stop_reason == "cancelled"
    assert not result.raw.get("forced")
    assert methods(record)[-1] == "shutdown"


async def test_an_unresponsive_devvi_is_killed_when_the_grace_period_ends(tmp_path: Path) -> None:
    script = write_script(
        tmp_path / "hang.jsonl", turn(step(), assistant("Working."), {"$hang": True}, close=False)
    )
    agent, record = sdk_agent(tmp_path, script, "--ignore-shutdown", cancel_grace_s=0.5)
    async with agent:
        async with agent.stream("t") as stream:
            async for event in stream:
                if isinstance(event, TextDelta):
                    stream.cancel()
        result = stream.result
    assert result.status == "cancelled" and result.raw.get("forced")
    await wait_until_gone(starts(record)[0]["pid"])


async def test_a_process_exit_mid_turn_is_interrupted_and_stderr_stays_out(tmp_path: Path) -> None:
    script = write_script(
        tmp_path / "exit.jsonl",
        turn(
            step(),
            tool_call("c1", "bash", {"command": "make"}),
            {"$exit": 3, "$stderr": "boom at C:/Users/secret"},
            close=False,
        ),
    )
    agent, _ = sdk_agent(tmp_path, script)
    lead = Agent(ScriptedModel([call("devvi", task="t"), "ok"]), tools=[agent.as_tool()])
    result = await lead.run("go")
    content = tool_results(result)[0].content
    assert "outcome is unknown" in content and "the process exited with code 3" in content
    assert "secret" not in content and "session=qa-" in content
    (tmp_path / "direct").mkdir()
    direct, _ = sdk_agent(tmp_path / "direct", script)
    async with direct:
        _, outcome = await run_turn(direct)
    assert (
        outcome.status == "interrupted"
        and outcome.raw["ambiguous"]
        and "secret" in outcome.raw["stderr_tail"]
    )


async def test_a_permanent_turn_error_opens_the_circuit_breaker(tmp_path: Path) -> None:
    failure = {"kind": "error", "error": {"message": "the API key was rejected", "code": "AUTH"}}
    script = write_script(tmp_path / "auth.jsonl", turn(step(), end=failure))
    agent, record = sdk_agent(tmp_path, script)
    lead = Agent(
        ScriptedModel([call("devvi", task="t"), call("devvi", task="again"), "ok"]),
        tools=[agent.as_tool()],
    )
    result = await lead.run("go")
    first, second = (part.content for part in tool_results(result))
    assert first.startswith(
        texts.EXTERNAL_AGENT_UNAVAILABLE.format(name="devvi", error="the API key was rejected")
    )
    assert second == first  # the breaker answers at once, with the same text
    assert len(starts(record)) == 1


async def test_an_ordinary_turn_error_is_reported(tmp_path: Path) -> None:
    failure = {"kind": "error", "error": {"message": "tool scheduler failed", "code": "UNKNOWN"}}
    agent, _ = sdk_agent(tmp_path, write_script(tmp_path / "err.jsonl", turn(step(), end=failure)))
    async with agent:
        _, result = await run_turn(agent)
    assert (result.status, result.stop_reason) == ("error", "tool scheduler failed")
    assert not result.raw.get("unavailable")


async def test_initialize_failure_is_a_configuration_error(tmp_path: Path) -> None:
    agent, record = sdk_agent(
        tmp_path,
        None,
        "--fail-initialize",
        'no adapter registered for provider "qcloud"',
        "--stderr",
        "loader trace",
    )
    with pytest.raises(
        ExternalAgentConfigurationError, match="initialize failed: no adapter registered"
    ) as caught:
        await agent.run("t")
    assert caught.value.stderr_tail == "loader trace"
    assert methods(record) == ["initialize"]


async def test_continuing_an_unknown_session_is_an_error_without_a_prompt(tmp_path: Path) -> None:
    agent, record = sdk_agent(tmp_path, "text-turn.notifications.jsonl")
    async with agent:
        _, result = await run_turn(agent, session="qa-nope")
    assert result.status == "error"
    assert result.stop_reason == devvi_texts.DEVVI_UNKNOWN_SESSION.format(session="qa-nope")
    assert "session/prompt" not in methods(record)


async def test_a_session_that_cannot_be_reopened_is_an_error(tmp_path: Path) -> None:
    agent, _ = sdk_agent(tmp_path, "text-turn.notifications.jsonl", "--missing")
    async with agent:
        _, result = await run_turn(agent, session="qa-gone")
    assert result.status == "error" and "qa-gone" in (result.stop_reason or "")


async def test_a_shared_process_serves_several_calls(tmp_path: Path) -> None:
    agent, record = sdk_agent(
        tmp_path,
        "approval.session.v4.jsonl",
        process_per_call=False,
        permissions=Permissions(allow=["devvi.*"]),
    )
    async with agent:
        _, first = await run_turn(agent, "one")
        _, second = await run_turn(agent, "two", session=first.session_id)
    assert (first.text, second.text) == ("ONE", "TWO")
    assert len(starts(record)) == 1 and methods(record).count("initialize") == 1
    assert methods(record)[-1] == "shutdown"


async def test_a_broken_turn_stops_only_its_own_pooled_process(tmp_path: Path) -> None:
    """``process_per_call=False``: Devvi dying in workspace ``a`` neither disconnects the
    agent nor stops the process of workspace ``b``, which serves its next call."""
    script = write_script(
        tmp_path / "pool.jsonl",
        turn(step(), {"$if_cwd": "a", "$then": [{"$exit": 3}], "$else": [assistant("One.")]}),
        turn(step(), assistant("Two."), token="{{message:2}}", number=2),
    )
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    agent, record = sdk_agent(tmp_path, script, process_per_call=False)
    async with agent:
        _, first = await run_turn(agent, "one", cwd=b)
        _, broken = await run_turn(agent, "boom", cwd=a)
        assert broken.status == "interrupted" and broken.raw["ambiguous"]
        assert agent.connected  # a lost process is the pool's business, not the agent's
        _, second = await run_turn(agent, "two", session=first.session_id, cwd=b)
    assert (first.text, second.text) == ("One.", "Two.")
    assert [start["cwd"] for start in starts(record)] == [str(b), str(a)]  # b was not restarted
    assert methods(record).count("initialize") == 2


async def test_a_pooled_process_that_died_between_calls_is_replaced(tmp_path: Path) -> None:
    script = write_script(
        tmp_path / "died.jsonl",
        turn(step(), assistant("One.")),
        turn(step(), assistant("Two."), token="{{message:2}}", number=2),
    )
    agent, record = sdk_agent(tmp_path, script, process_per_call=False)
    async with agent:
        _, first = await run_turn(agent, "one")
        (dead,) = agent._transport._pool.values()
        dead.kill()
        await dead.process.wait()
        _, second = await run_turn(agent, "two", session=first.session_id)
        assert dead.peer.closed  # what was left of the dead process was released
    assert (first.text, second.text) == ("One.", "Two.")
    assert len(starts(record)) == 2


def test_images_become_inline_blocks_and_urls_are_refused() -> None:
    blocks = content_blocks(
        [
            TextPart(text="look"),
            ImagePart(data="aGk=", media_type="image/png"),
            ImagePart(url="data:image/jpeg;base64,eHk="),
        ]
    )
    assert blocks == [
        {"type": "text", "text": "look"},
        {"type": "image", "data": "aGk=", "mimeType": "image/png"},
        {"type": "image", "data": "eHk=", "mimeType": "image/jpeg"},
    ]
    with pytest.raises(Exception):  # noqa: B017 - the private _BadInput
        content_blocks([ImagePart(url="https://example.com/a.png")])


# -- events and usage, without a process ----------------------------------------------------------


async def test_committed_text_blocks_are_separated_by_a_blank_line() -> None:
    """Two text blocks of one message, or two messages with no tool call between them, read
    as two paragraphs, not ``"Fixed it.Tests pass."``."""
    from qagents.external import ScriptedExternalAgent
    from qagents.integrations.devvi.events import SessionEventMapper

    def message(*texts: str) -> dict[str, Any]:
        content = [{"type": "text", "text": text} for text in texts]
        return {"type": "assistant/message", "data": {"message": {"content": content}}}

    def devvi_turn(turn: Any) -> None:  # the scripted agent has begun session coder-1
        mapper = SessionEventMapper(turn, root="coder-1")
        mapper.handle_event("coder-1", message("Fixed it.", "Tests pass."))
        mapper.handle_event("coder-1", message("All done."))

    agent = ScriptedExternalAgent([[devvi_turn]], name="coder", track_changes=False)
    async with agent.stream("go") as turn:
        deltas = [event.text async for event in turn if isinstance(event, TextDelta)]
    assert deltas == ["Fixed it.", "\n\nTests pass.", "\n\nAll done."]
    assert turn.result is not None
    assert turn.result.text == "Fixed it.\n\nTests pass.\n\nAll done."


def test_usage_mapping_and_prices() -> None:
    usage = usage_from_devvi(
        {
            "inputTokens": 100,
            "outputTokens": 10,
            "cacheReadTokens": 50,
            "cacheWriteTokens": 5,
            "reasoningTokens": 4,
        },
        TokenPricing(input=2.0, output=8.0),
    )
    assert (
        usage.requests,
        usage.input_tokens,
        usage.cached_input_tokens,
        usage.output_tokens,
        usage.reasoning_tokens,
    ) == (1, 155, 50, 10, 4)
    assert usage.cost_usd == pytest.approx(
        (155 * 2.0 + 10 * 8.0) / 1e6
    )  # cache priced as input by default
    assert usage_from_devvi(None).unpriced_requests == 1
    assert usage_from_devvi({"inputTokens": 3, "outputTokens": 1}).cost_usd is None
    with pytest.raises(ValueError):
        TokenPricing(input=-1, output=1)


@pytest.mark.parametrize(
    ("reason", "status", "stop"),
    [
        ({"kind": "completed"}, "completed", "end_turn"),
        ({"kind": "forked"}, "completed", "forked"),
        ({"kind": "max-tokens"}, "limit_reached", "max_tokens"),
        ({"kind": "aborted", "reason": {}}, "cancelled", "cancelled"),
        ({"kind": "blocked"}, "error", "blocked"),
        ({"kind": "error", "error": {"message": "boom", "code": "X"}}, "error", "boom"),
        ({"kind": "interrupted"}, "interrupted", "interrupted"),
        ({"kind": "something-new"}, "error", "something-new"),
    ],
)
def test_turn_end_reasons(reason: dict[str, Any], status: str, stop: str) -> None:
    outcome = turn_end_outcome(reason)
    assert (outcome.status, outcome.stop_reason) == (status, stop)
    assert outcome.raw["agent_stop_reason"] == reason["kind"]


@pytest.mark.parametrize(
    ("acp", "devvi"),
    [("end_turn", "completed"), ("max_tokens", "max-tokens"), ("cancelled", "aborted")],
)
def test_the_same_outcome_has_the_same_stop_reason_on_both_transports(acp: str, devvi: str) -> None:
    from qagents.integrations.acp.mapping import outcome_for_stop_reason

    over_acp = outcome_for_stop_reason({"stopReason": acp})
    over_sdk = turn_end_outcome({"kind": devvi})
    assert (over_sdk.status, over_sdk.stop_reason) == (over_acp.status, over_acp.stop_reason)
    assert over_sdk.raw["agent_stop_reason"] == devvi


# -- the session log reader ------------------------------------------------------------------------

LOG = FAKE_HOME / "sessions" / "--work-repo--" / "qa-fixture" / "session.v4.jsonl"


def test_logs_find_and_read_the_fake_home() -> None:
    root = logs.sessions_dir(FAKE_HOME)
    assert logs.find_session_log(root, "qa-fixture", cwd="/work/repo") == LOG
    assert logs.find_session_log(root, "qa-fixture") == LOG
    assert logs.find_session_log(root, "qa-missing") is None
    log = logs.read_session_log(LOG)
    assert log.session_id == "qa-fixture" and log.cwd == "/work/repo" and log.torn_tail
    first, second = log.turns()
    assert (first.number, first.end_kind, first.finished, first.final_text) == (
        1,
        "completed",
        True,
        "dsh-sdk-proof-7391",
    )
    assert first.message_ids == ["msg-1"] and first.tool_calls == ["bash"] and len(first.usage) == 2
    assert (second.number, second.end_reason, second.finished, second.tool_calls) == (
        2,
        None,
        False,
        ["write"],
    )
    assert logs.find_prompt_turn(log, message_id="msg-1") == first
    assert logs.find_prompt_turn(log, message_id="msg-9") is None
    assert logs.find_prompt_turn(log, prompts=["Now add a test."]) == second
    assert logs.find_prompt_turn(log, prompts=[first.prompt_text]) is None  # not the latest prompt
    assert log.turn_events(first)[0]["type"] == "agent/inbox/spliced"
    assert log.turn_events(first)[-1]["type"] == "turn/end"
    sessions = logs.list_sessions(root)
    assert [item.session_id for item in sessions] == ["qa-fixture"] and not sessions[0].compressed
    text = logs.to_markdown(log)
    assert "## User" in text and "dsh-sdk-proof-7391" in text and "Turn 1 ended: completed" in text


def test_the_log_reader_refuses_damage_and_other_formats(tmp_path: Path) -> None:
    good = LOG.read_bytes().split(b"\n")
    damaged = tmp_path / "damaged" / LOG.name
    damaged.parent.mkdir()
    damaged.write_bytes(b"\n".join([good[0], b"{not json", *good[1:3]]) + b"\n")
    with pytest.raises(logs.DevviLogError, match="line 2 is not JSON"):
        logs.read_session_log(damaged)
    old = tmp_path / "old" / LOG.name
    old.parent.mkdir()
    old.write_text(
        json.dumps({"type": "session", "version": 3, "id": "x"}) + "\n", encoding="utf-8"
    )
    with pytest.raises(logs.DevviLogError, match="not a session format v4 log"):
        logs.read_session_log(old)
    compressed = tmp_path / "zstd" / (LOG.name + ".zstd")
    compressed.parent.mkdir()
    compressed.write_bytes(b"\x28\xb5\x2f\xfd")
    with pytest.raises(logs.DevviLogError, match="compressed"):
        logs.read_session_log(compressed)
    with pytest.raises(logs.DevviLogError, match="Cannot read"):
        logs.read_session_log(tmp_path / "missing.jsonl")


def test_log_paths_follow_devvi() -> None:
    assert logs.project_key("C:\\Users\\me\\repo") == "--C-Users-me-repo--"
    assert logs.project_key("/work/repo") == "--work-repo--"
    assert logs.project_key("/a b") == "--a~0020b--"
    assert logs.encode_segment("qa-1") == "qa-1"
    assert logs.encode_segment("a/b") == "a~002Fb"
    assert logs.encode_segment("..") == "~002E~002E"
    assert logs.encode_segment("x~") == "x~007E"
    assert logs.encode_segment("\U0001f600") == "~D83D~DE00"
    path = logs.session_log_path("/root", "qa-1", "/work/repo")
    assert path.parts[-3:] == ("--work-repo--", "qa-1", "session.v4.jsonl")


# -- review round 2: workspaces, re-attach baselines, auxiliary usage, cleanup ---------------------


async def eventually(check: Any, seconds: float = 15.0) -> None:
    for _ in range(int(seconds / 0.02)):
        if check():
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"timed out waiting for {check}")


def agent_in(
    tmp_path: Path, name: str, script: Path, *fake: str, **options: Any
) -> tuple[Any, Path]:
    """A `DevviAgent` (SDK) in workspace ``ws_<name>``; every one shares one Devvi home."""
    from qagents.integrations.devvi import DevviAgent

    workspace = tmp_path / f"ws_{name}"
    workspace.mkdir(exist_ok=True)
    record = tmp_path / f"record_{name}.jsonl"
    options.setdefault("home", tmp_path / "devvi-home")
    options.setdefault("track_changes", False)
    agent = DevviAgent(
        workspace=workspace,
        transport="sdk",
        command=fake_command(script, *fake, record=record),
        **options,
    )
    return agent, record


async def test_a_session_of_another_workspace_is_refused_before_devvi_opens_it(
    tmp_path: Path,
) -> None:
    """Coder ``a`` is given the session of coder ``b`` (same home, another repository). It
    must not work in ``b``'s workspace outside ``b``'s lock and rules: refused, and Devvi
    never opens, repairs or prompts that session."""
    script = write_script(tmp_path / "b.jsonl", turn(step(), assistant("B was here.")))
    b, _ = agent_in(tmp_path, "b", script)
    a, record_a = agent_in(tmp_path, "a", script)
    async with b:
        _, first = await run_turn(b, "Note it.")
    assert first.status == "completed"
    log = next((tmp_path / "devvi-home" / "sessions").rglob("session.v4.jsonl"))
    before = log.read_bytes()
    reason = devvi_texts.DEVVI_OTHER_WORKSPACE.format(session=first.session_id)
    async with a:
        _, refused = await run_turn(a, "Write where.txt.", session=first.session_id)
        lead = Agent(
            ScriptedModel([call("coder", task="Write where.txt.", session=first.session_id), "ok"]),
            tools=[a.as_tool(name="coder")],
        )
        through_tool = await lead.run("go")
    assert (refused.status, refused.stop_reason) == ("error", reason)
    assert refused.raw["session_cwd"] == str(tmp_path / "ws_b")
    # Through the tool, the model may continue only sessions that tool opened: refused even
    # before the workspace check, since this tool never opened b's session.
    unknown = devvi_texts.DEVVI_UNKNOWN_SESSION.format(session=first.session_id)
    assert tool_results(through_tool)[0].content == texts.EXTERNAL_AGENT_FAILED.format(
        name="coder", error=unknown
    )
    assert starts(record_a) == []  # refused before any Devvi process started
    assert log.read_bytes() == before


@pytest.mark.parametrize("devvi_says", [True, False], ids=["devvi-answers", "devvi-cannot-say"])
async def test_a_resumed_session_whose_log_qagents_cannot_see_is_checked_with_devvi(
    tmp_path: Path, devvi_says: bool
) -> None:
    """A user patch moved Devvi's logs where qagents does not look: Devvi still finds the
    session on resume, so qagents asks Devvi where it works before it prompts."""
    moved = tmp_path / "elsewhere"
    patch = tmp_path / "moved.patch.yml"
    row = {"id": "session-persistence-jsonl", "config": {"root": str(moved), "compression": "none"}}
    patch.write_text(json.dumps([row]), encoding="utf-8")
    script = write_script(
        tmp_path / "b.jsonl",
        turn(step(), assistant("B was here.")),
        turn(step(), assistant("Again."), token="{{message:2}}", number=2),
    )
    fake = () if devvi_says else ("--no-working-directory",)
    b, _ = agent_in(tmp_path, "b", script, *fake, patches=[patch])
    a, record_a = agent_in(tmp_path, "a", script, *fake, patches=[patch], process_per_call=False)
    async with b:
        _, first = await run_turn(b, "Note it.")
    assert next(moved.rglob("session.v4.jsonl"))  # not under <home>/sessions
    async with a:
        _, refused = await run_turn(a, "Write where.txt.", session=first.session_id)
        assert a._transport._pool == {}  # the process that opened b's session was stopped
    assert refused.status == "error" and "session/prompt" not in methods(record_a)
    assert methods(record_a)[-2:] == ["session/working-directory/get", "shutdown"]
    if devvi_says:
        assert refused.stop_reason == devvi_texts.DEVVI_OTHER_WORKSPACE.format(
            session=first.session_id
        )
        async with b:  # its own workspace: Devvi confirms it, and the turn runs
            _, again = await run_turn(b, "Again.", session=first.session_id)
        assert again.status == "completed" and again.text == "Again."
    else:
        assert refused.stop_reason == devvi_texts.DEVVI_WORKSPACE_UNKNOWN.format(
            session=first.session_id
        )


async def test_a_repeated_continuation_is_not_matched_to_an_earlier_calls_turn(
    tmp_path: Path,
) -> None:
    """Call 2 continues call 1's session with the same task and crashes before Devvi logs
    its prompt. Its repeat sends the task again: call 1's finished turn, whose prompt has
    the same text, is older than call 2's baseline, so it is not replayed as call 2's."""
    first_script = write_script(
        tmp_path / "first.jsonl",
        turn(step(), assistant("First result."), prompt="Continue."),
        [{"$no_answer": True}],  # call 2: no receipt, no answer (Devvi dies with qagents)
    )
    agent, record = sdk_agent(tmp_path, first_script)

    def reply(request: Any) -> Any:
        results = [part for message in request.messages for part in message.tool_results]
        if not results:
            return call("devvi", task="Continue.")
        if len(results) == 1:
            session = re.search(r"session (qa-[0-9a-f]+)", results[0].content)
            assert session is not None
            return call("devvi", task="Continue.", session=session.group(1))
        return "Done."

    db = tmp_path / "runs.db"
    lead = Agent(FunctionModel(reply), tools=[agent.as_tool()])
    task = asyncio.create_task(lead.run("go", session=Session(SQLiteStore(db), id="crash")))
    try:
        await eventually(lambda: len(requests(record, "session/prompt")) == 2)
    finally:
        task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    entries = await Session(SQLiteStore(db), id="crash").entries()
    checkpoints = [entry.data for entry in entries if entry.kind == "tool_checkpoint"]
    second_call = checkpoints[-1]["tool_call_id"]
    keys = [item["key"] for item in checkpoints if item["tool_call_id"] == second_call]
    assert keys == [BASELINE_CHECKPOINT, "external_session"]  # no message id: no answer came
    baselines = [item["value"] for item in checkpoints if item["key"] == BASELINE_CHECKPOINT]
    assert baselines[0] == {"events": 0}  # call 1 started a new session
    assert baselines[-1]["events"] > 0  # call 2 started after call 1's whole turn

    second_script = write_script(
        tmp_path / "second.jsonl",
        turn(step(), assistant("First result."), prompt="Continue."),
        turn(step(), assistant("Second result."), prompt="Continue.", token="{{message:2}}"),
    )
    second, record2 = sdk_agent(tmp_path, second_script, record=tmp_path / "record2.jsonl")
    resumed = await Agent(FunctionModel(reply), tools=[second.as_tool()]).resume(
        Session(SQLiteStore(db), id="crash")
    )
    assert resumed.output == "Done."
    contents = [part.content for part in tool_results(resumed)]
    assert contents[0].startswith("First result.") and contents[1].startswith("Second result.")
    sent = [item["params"]["contentBlocks"] for item in requests(record2, "session/prompt")]
    assert sent == [[{"type": "text", "text": "Continue."}]]  # sent as new


def _log_with_turn(root: Path, session_id: str, cwd: str, prompt_text: str) -> Path:
    """A Devvi log holding one finished turn started by ``prompt_text``."""
    path = logs.session_log_path(root, session_id, cwd)
    path.parent.mkdir(parents=True)
    header = {"type": "session", "version": 4, "id": session_id, "cwd": cwd, "createdAt": 1}
    items = turn(step(), assistant("Earlier result."), prompt=prompt_text, token="msg-old")
    events = [item["params"]["event"] for item in items if item.get("method") == "session.event"]
    rows = [header, *events]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("session", "baseline", "expected"),
    [
        ("qa-continued", None, "unknown"),  # a continued session without a baseline
        ("qa-continued", "after", "not_sent"),  # the matching prompt is an earlier call's
        ("qa-continued", 0, "finished"),  # this call's own prompt, received after the baseline
        ("qa-own", None, "finished"),  # the call's own session: only its attempts prompt it
    ],
)
def test_reattach_matches_a_prompt_by_text_only_after_the_baseline(
    tmp_path: Path, session: str, baseline: Any, expected: str
) -> None:
    root, cwd = tmp_path / "sessions", str(tmp_path / "ws")
    path = _log_with_turn(root, session, cwd, "Continue.")
    _, events = logs.peek_session_log(path)
    assert events == len(logs.read_session_log(path).events)
    checkpoints: dict[str, Any] = {"external_session": {"session_id": session, "cwd": cwd}}
    if baseline is not None:
        checkpoints[BASELINE_CHECKPOINT] = {"events": events if baseline == "after" else baseline}

    class Turn:
        prompt_text = "Continue."
        suggested_session_id = "qa-own"

        def __init__(self) -> None:
            self.checkpoints = checkpoints
            self.raw: dict[str, Any] = {}

        @contextlib.contextmanager
        def replaying(self, *, since: Any = None) -> Any:
            yield

    class Mapper:
        def __init__(self) -> None:
            self.replayed: list[str] = []

        def handle_event(self, session_id: str, event: dict[str, Any]) -> None:
            self.replayed.append(event["type"])

    turn_, mapper = Turn(), Mapper()
    transport = SdkTransport(SdkSettings(), launch=None, sessions_root=lambda _: root)  # type: ignore[arg-type]
    blocks = [{"type": "text", "text": "Continue."}]
    plan = transport._reattach(turn_, session, cwd, root, mapper, blocks)  # type: ignore[arg-type]
    assert turn_.raw["reattach"] == expected
    if expected == "unknown":
        text = devvi_texts.DEVVI_RESUME_UNKNOWN.format(task="Continue.")
        assert plan == [{"type": "text", "text": text}]
    elif expected == "not_sent":
        assert plan == blocks and mapper.replayed == []
    else:
        assert plan is None and mapper.replayed[-1] == "turn/end"


TEXT_TURN = SDK_FIXTURES / "text-turn.session.v4.jsonl"
"""A finished turn as Devvi logged it: one model step of 1769 input tokens."""


async def _replayed_after_a_crash(mid_turn: str) -> Any:
    """A first attempt reports the whole logged turn (1769 input tokens), then ``mid_turn``
    writes an entry that carries that usage (or none does), and the process dies before
    the result is journaled. The repeat re-attaches through `SdkTransport._reattach`,
    which finds the turn finished in the log and replays it."""
    from qagents import RunContext, ScriptedExternalAgent, tool
    from qagents.integrations.devvi.events import SessionEventMapper

    events = [json.loads(line) for line in TEXT_TURN.read_text("utf-8").splitlines()][1:]
    task_text = "Reply with exactly: SDK snapshot OK"
    crashed = asyncio.Event()

    @tool(read_only=True)
    def lookup(q: str) -> str:
        """Look something up."""
        return "found"

    async def first_attempt(turn: Any) -> None:
        await turn.checkpoint(MESSAGE_CHECKPOINT, "{{message:1}}")
        mapper = SessionEventMapper(turn, root=turn.session_id)
        for event in events:
            mapper.handle_event(turn.session_id, event)
        if mid_turn == "checkpoint":
            await turn.checkpoint("note", 1)
        elif mid_turn == "served call":
            parent: RunContext[Any] = turn.parent
            await parent.call_tool(lookup, {"q": "x"}, source="mcp")
        crashed.set()
        await asyncio.Event().wait()  # the process dies before external_result is journaled

    async def recover(turn: Any) -> None:
        mapper = SessionEventMapper(turn, root=turn.session_id)
        transport = object.__new__(SdkTransport)
        with patch("qagents.integrations.devvi.sdk.logs.find_session_log", return_value=TEXT_TURN):
            blocks = [{"type": "text", "text": task_text}]
            plan = transport._reattach(
                turn, turn.session_id, "unused", TEXT_TURN.parent, mapper, blocks
            )
        assert plan is None and turn.raw["reattach"] == "finished"
        assert turn.usage.input_tokens == 1769  # the turn itself reports all of it

    session = Session()
    coder = ScriptedExternalAgent(
        [[first_attempt], [recover]],
        name="coder",
        resumable_turns=True,
        track_changes=False,
        cancel_grace_s=0.1,
    )
    lead = Agent(ScriptedModel([call("coder", task=task_text), "done"]), tools=[coder.as_tool()])
    task = asyncio.create_task(lead.run("go", session=session))
    await asyncio.wait_for(crashed.wait(), 10)
    task.cancel()  # a crash: no run_failed is written
    with pytest.raises(asyncio.CancelledError):
        await task
    restored = (await session.state()).run_usage.input_tokens
    assert restored == (0 if mid_turn == "nothing" else 1769)
    return await lead.resume(session)


@pytest.mark.parametrize("mid_turn", ["checkpoint", "served call", "nothing"])
async def test_a_replayed_turn_counts_usage_the_journal_held_once(mid_turn: str) -> None:
    resumed = await _replayed_after_a_crash(mid_turn)
    assert resumed.status == "completed"
    assert resumed.usage.input_tokens == 1769  # Devvi used 1769, whatever carried them


def test_replaying_gives_the_run_only_what_its_journal_does_not_hold() -> None:
    """``since``: usage journaled for older turns of the call (before a new prompt)."""
    from qagents import RunContext
    from qagents.external import ExternalTurnContext
    from qagents.messages import Usage

    given: list[Usage] = []

    class Run:
        journaled_usage = {"op": Usage(requests=2, input_tokens=300, cost_usd=0.3)}
        limits = None

        def _remaining(self) -> dict[str, Any]:
            return {"max_total_tokens": None, "max_cost_usd": None, "timeout_s": None}

    parent = RunContext(tool_call_id="c", operation_id="op", _add_usage=given.append, _run=Run())
    turn = ExternalTurnContext.__new__(ExternalTurnContext)
    turn.parent, turn.usage, turn._replayed = parent, Usage(), None
    step = Usage(requests=1, input_tokens=250, cost_usd=0.25)
    with turn.replaying(since=Usage(requests=1, input_tokens=100, cost_usd=0.1)):
        turn.add_usage(step)
        turn.add_usage(step)
        assert given == []  # nothing reaches the run while it replays
    assert turn.usage.input_tokens == 500
    # 500 replayed; the journal holds 300, of which 100 belong to an older turn: 200 of
    # this turn are counted already, so 300 are owed.
    [owed] = given
    assert (owed.requests, owed.input_tokens) == (1, 300)
    assert owed.cost_usd == pytest.approx(0.3)
    given.clear()
    with turn.replaying():  # all of it journaled already: nothing more
        turn.add_usage(Usage(requests=2, input_tokens=300, cost_usd=0.3))
    assert given == []


def _summary(usage: dict[str, int] | None = None, *, session: str = ROOT) -> dict[str, Any]:
    data: dict[str, Any] = {
        "compactionId": "c1", "summary": [{"type": "text", "text": "so far"}],
        "shadowedRange": {"start": 0, "end": 5}, "shadowedSeqs": [1, 2],
        "shadowedTokenCount": 9000, "provider": "deepseek-official",
        "model": "deepseek-v4-flash", "rawOutput": [], "llmStreamCall": True,
    }  # fmt: skip
    if usage is not None:
        data["usage"] = usage
    return notification("compaction/summary", data, session=session)


def _web_search(session: str = ROOT) -> dict[str, Any]:
    data = {"endpoint": "https://api.deepseek.com/anthropic", "apiVersion": "1", "body": {}}
    return notification("web/deepseek-search-llm-request", data, session=session)


async def test_compactions_and_web_searches_count_as_model_requests(tmp_path: Path) -> None:
    started = {
        "method": "subagent.started",
        "params": {"parentSessionId": ROOT, "childSessionId": CHILD},
    }
    script = write_script(
        tmp_path / "aux.jsonl",
        turn(
            step(),
            assistant("Working.", usage={"inputTokens": 100, "outputTokens": 10}),
            notification("compaction/start", {"compactionId": "c1", "turn": 1}),
            _summary({"inputTokens": 90000, "outputTokens": 3000}),
            notification("compaction/end", {"compactionId": "c1", "turn": 1}),
            _summary(None),  # a model call whose usage Devvi did not get
            _web_search(),
            started,
            _summary({"inputTokens": 500, "outputTokens": 50}, session=CHILD),
            _web_search(CHILD),
        ),
    )
    agent, _ = sdk_agent(tmp_path, script, pricing=PRICES)
    async with agent:
        events, result = await run_turn(agent)
    usage = result.usage
    assert usage.requests == 6  # the message, three compactions, two web searches
    assert (usage.input_tokens, usage.output_tokens) == (100 + 90000 + 500, 10 + 3000 + 50)
    assert usage.unpriced_requests == 3  # the summary without usage and the two searches
    priced = (100 * 1.0 + 10 * 2.0) + (90000 * 1.0 + 3000 * 2.0) + (500 * 1.0 + 50 * 2.0)
    assert usage.cost_usd == pytest.approx(priced / 1e6)
    external = [event for event in events if isinstance(event, ExternalAgentEvent)]
    compactions = [event.data for event in external if event.kind == "compaction"]
    assert [item["type"] for item in compactions] == [
        "compaction/start", "compaction/summary", "compaction/end", "compaction/summary",
    ]  # fmt: skip
    assert compactions[1]["usage"]["input_tokens"] == 90000
    assert compactions[3]["usage"]["unpriced_requests"] == 1
    searches = [event.data for event in external if event.kind == "web_search"]
    assert searches[0]["usage"]["unpriced_requests"] == 1 and "body" not in searches[0]
    child = [event.data for event in external if event.kind == "subagent_event"]
    assert [item["type"] for item in child] == [
        "compaction/summary", "web/deepseek-search-llm-request",
    ]  # fmt: skip


async def test_a_compaction_counts_against_the_token_budget(tmp_path: Path) -> None:
    script = write_script(
        tmp_path / "compaction.jsonl",
        turn(
            step(),
            assistant("Working.", usage={"inputTokens": 100, "outputTokens": 10}),
            _summary({"inputTokens": 90000, "outputTokens": 3000}),
            {"$hang": True},
            close=False,
        ),
    )
    agent, record = sdk_agent(tmp_path, script)
    lead = Agent(
        ScriptedModel([call("devvi", task="t")]),
        tools=[agent.as_tool()],
        limits=Limits(max_total_tokens=50000),
    )
    result = await lead.run("go")
    assert result.status == "limit_reached" and result.stop_reason == "max_total_tokens"
    assert result.usage.input_tokens == 90100
    assert methods(record)[-1] == "shutdown"


async def test_a_web_search_stops_a_turn_under_a_cost_budget(tmp_path: Path) -> None:
    """Devvi records no usage for its web search's model request: under a cost budget the
    spend is then unknown, so the turn stops (fail closed)."""
    script = write_script(
        tmp_path / "search.jsonl",
        turn(
            step(),
            assistant("Searching.", usage={"inputTokens": 100, "outputTokens": 10}),
            _web_search(),
            {"$hang": True},
            close=False,
        ),
    )
    agent, record = sdk_agent(tmp_path, script, pricing=PRICES)
    lead = Agent(
        ScriptedModel([call("devvi", task="t")]),
        tools=[agent.as_tool()],
        limits=Limits(max_cost_usd=1.0),
    )
    result = await lead.run("go")
    assert result.status == "limit_reached" and result.stop_reason == "max_cost_usd"
    assert result.usage.unpriced_requests == 1
    assert methods(record)[-1] == "shutdown"


async def test_a_cancel_while_devvi_stops_keeps_the_finished_turn(tmp_path: Path) -> None:
    """The turn finished; Devvi is slow to stop (it ignores ``shutdown``) and a cancel comes
    in meanwhile. The finished turn stays finished: the stop goes on in the background."""
    agent, record = sdk_agent(
        tmp_path,
        "text-turn.notifications.jsonl",
        "--ignore-shutdown",
        shutdown_timeout_s=3.0,
        cancel_grace_s=0.2,
    )
    async with agent:
        async with agent.stream("Reply with exactly: SDK snapshot OK") as stream:

            async def cancel_while_stopping() -> None:
                await eventually(lambda: "shutdown" in methods(record))
                stream.cancel()

            watcher = asyncio.create_task(cancel_while_stopping())
            _ = [event async for event in stream]
            await watcher
        result = stream.result
        assert result is not None
        assert (result.status, result.text) == ("completed", "SDK snapshot OK")
        assert not result.raw.get("forced")
    await wait_until_gone(starts(record)[0]["pid"])  # the agent waited for the stop on exit


def test_connection_reasons_come_from_the_prompts_module() -> None:
    import inspect

    from qagents.integrations.devvi import sdk

    source = inspect.getsource(sdk)
    for constant in ("EXTERNAL_REASON_BAD_ANSWER", "EXTERNAL_REASON_NO_ANSWER"):
        assert f"base_prompts.{constant}" in source
        assert f'"{getattr(texts, constant)}"' not in source  # no copy of the text


async def test_a_protocol_violation_is_reported_with_the_shared_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(texts, "EXTERNAL_REASON_BAD_ANSWER", "a reason set at run time")
    agent, _ = sdk_agent(tmp_path, "text-turn.notifications.jsonl", "--no-message-id")
    async with agent:
        _, result = await run_turn(agent)
    assert result.status == "interrupted" and result.stop_reason == "a reason set at run time"


def test_the_documented_log_recipe_works() -> None:
    from qagents.integrations.devvi import find_session_log, read_session_log, sessions_dir

    assert "read_session_log(find_session_log(sessions_dir(home)," in (logs.__doc__ or "")
    home = FAKE_HOME
    log = read_session_log(find_session_log(sessions_dir(home), "qa-fixture"))
    assert log.turns()[0].final_text == "dsh-sdk-proof-7391"
    assert find_session_log(home, "qa-fixture") is None  # a home is not a sessions folder
