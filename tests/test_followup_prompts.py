"""Prompt overrides reach the model through every core tool execution path."""

from __future__ import annotations

import asyncio
import json

import pytest

from qagents import Agent, Approval, Permissions, RunContext, Session, fan_out, prompts, tool
from qagents.models.testing import FunctionModel, ScriptedModel, call


def results(result):
    return [part for message in result.messages for part in message.tool_results]


@pytest.mark.parametrize("has_tools", [False, True])
async def test_unknown_tool_list_uses_prompt_constants(monkeypatch, has_tools):
    monkeypatch.setattr(prompts, "UNKNOWN_TOOL_SEPARATOR", " / ")
    monkeypatch.setattr(prompts, "UNKNOWN_TOOL_NONE", "nothing")

    @tool(read_only=True)
    def one() -> str:
        """Return one."""
        return "one"

    @tool(read_only=True)
    def two() -> str:
        """Return two."""
        return "two"

    model = ScriptedModel([call("missing"), "done"])
    result = await Agent(model, tools=[one, two] if has_tools else []).run("go")
    assert result.status == "completed"
    assert results(result)[0].content == prompts.UNKNOWN_TOOL.format(
        name="missing", available="one / two" if has_tools else "nothing"
    )


@pytest.mark.parametrize("path", ["inline", "resume", "served"])
async def test_edited_argument_result_uses_prompt_constant(monkeypatch, path):
    monkeypatch.setattr(prompts, "APPROVAL_ARGUMENTS_CHANGED_RESULT", "{content} [{note}]")

    @tool(requires_approval=True)
    def write(value: str) -> str:
        """Return the approved value."""
        return value

    @tool
    async def delegate(ctx: RunContext) -> str:
        """Serve the write tool through the running call."""
        return (await ctx.call_tool(write, {"value": "original"}, source="test")).content

    approval = Approval(approved=True, arguments={"value": "edited"})
    permissions = Permissions(on_ask=None if path == "resume" else lambda *args: approval)
    model = ScriptedModel(
        [call("delegate") if path == "served" else call("write", value="original"), "done"]
    )
    agent = Agent(model, tools=[delegate, write], permissions=permissions)
    session = Session()
    result = await agent.run("go", session=session)
    if path == "resume":
        assert result.status == "paused"
        result = await agent.resume(
            session, approvals={result.pending_approvals[0].tool_call_id: approval}
        )
    assert result.status == "completed"
    note = prompts.APPROVAL_ARGUMENTS_CHANGED.format(arguments='{"value": "edited"}')
    assert results(result)[0].content == f"edited [{note}]"


@pytest.mark.parametrize("cancel", [False, True])
async def test_tool_exception_detail_uses_prompt_constant(monkeypatch, cancel):
    monkeypatch.setattr(prompts, "TOOL_FAILED_EXCEPTION", "<{type}> {error}")

    @tool
    async def fail() -> str:
        """Raise a tool failure, without cancelling the caller."""
        if cancel:
            raise asyncio.CancelledError("stopped")
        raise ValueError("broken")

    result = await Agent(ScriptedModel([call("fail"), "done"]), tools=[fail]).run("go")
    assert result.status == "completed"
    expected = "<CancelledError> stopped" if cancel else "<ValueError> broken"
    assert results(result)[0].content == prompts.TOOL_FAILED.format(name="fail", error=expected)


async def test_fan_out_exception_detail_uses_prompt_constant(monkeypatch):
    monkeypatch.setattr(prompts, "TOOL_FAILED_EXCEPTION", "<{type}> {error}")

    def broken(request):
        raise ValueError("broken model")

    worker = Agent(FunctionModel(broken), name="worker")
    lead = Agent(
        ScriptedModel([call("workers", tasks=["go"]), "done"]),
        tools=[fan_out(worker, name="workers")],
    )
    result = await lead.run("go")
    assert result.status == "completed"
    assert json.loads(results(result)[0].content) == [
        {
            "task": "go",
            "error": prompts.FAN_OUT_TASK_FAILED.format(
                name="worker", error="<ValueError> broken model"
            ),
        }
    ]
