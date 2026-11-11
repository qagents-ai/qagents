"""Linked child runs (sub-agents), inherited budgets and hooks, Agent.copy, max_tool_calls."""

from __future__ import annotations

from typing import Any

import pytest

from qagents import Agent, Hooks, Limits, Permissions, RunContext, Session, Usage, prompts, tool
from qagents.errors import ConfigurationError
from qagents.models.testing import FunctionModel, ScriptedModel, call


@tool(read_only=True)
def search(query: str) -> str:
    """Search the knowledge base."""
    return f"results for {query}"


class Recorder(Hooks):
    """Records the runs it sees and their model requests."""

    def __init__(self) -> None:
        self.runs: list[tuple[str, str | None, str | None, int]] = []
        self.requests: list[str] = []

    async def on_run_start(self, ctx: RunContext[Any]) -> None:
        self.runs.append((ctx.agent_name, ctx.run_id, ctx.parent_run_id, ctx.depth))

    async def before_model(self, ctx: RunContext[Any], request: Any) -> None:
        self.requests.append(ctx.agent_name)


class Tracer(Recorder):
    propagate = True


async def test_a_subagent_run_is_linked_to_the_calling_tool_call():
    seen: list[RunContext[Any]] = []

    @tool(read_only=True)
    def inspect_context(ctx: RunContext[Any]) -> str:
        """Report the run context."""
        seen.append(ctx)
        return "ok"

    worker = Agent(
        ScriptedModel([call("inspect_context"), "done"]), name="worker", tools=[inspect_context]
    )
    lead_model = ScriptedModel([call("worker", task="look"), "finished"])
    lead = Agent(lead_model, name="lead", tools=[worker])
    result = await lead.run("go")

    (child,) = seen
    delegated = result.messages[1].tool_calls[0]
    assert child.parent_run_id == result.run_id
    assert child.parent_tool_call_id == delegated.id
    assert child.depth == 1 and child.run_id != result.run_id
    assert result.output == "finished"


async def test_a_run_can_be_started_as_a_child_of_any_context_and_journals_the_link():
    session = Session()
    parent = RunContext(run_id="parent-run", tool_call_id="call-1", depth=2)
    result = await Agent(ScriptedModel(["hi"])).run("hello", session=session, parent=parent)
    assert result.completed
    started = (await session.entries())[0]
    assert started.kind == "run_started"
    assert started.data == {
        "agent": "agent",
        "parent_run_id": "parent-run",
        "parent_tool_call_id": "call-1",
        "depth": 3,
    }


async def test_child_runs_refuse_to_start_beyond_max_depth():
    with pytest.raises(ConfigurationError, match="depth 6: Limits.max_depth is 5"):
        await Agent(ScriptedModel([])).run("go", parent=RunContext(depth=5))
    with pytest.raises(ConfigurationError, match="max_depth is 1"):
        Agent(ScriptedModel([]), limits=Limits(max_depth=1)).stream(
            "go", parent=RunContext(depth=1)
        )

    # An agent that keeps delegating to itself is stopped by the depth limit.
    def model(request):
        last = request.messages[-1]
        if last.role == "tool":
            return f"depth reported: {last.tool_results[0].content[:40]}"
        return call("recurse", task="again")

    recursive = Agent(FunctionModel(model), name="recurse", limits=Limits(max_depth=2))
    recursive.tools.append(recursive.as_tool())
    recorder = Tracer()
    recursive.hooks.append(recorder)
    result = await recursive.run("go")
    assert [depth for *_, depth in recorder.runs] == [0, 1, 2]
    assert result.completed


async def test_a_child_stops_at_the_parents_remaining_token_budget():
    worker_model = FunctionModel(
        lambda request: call("search", query="more"),
        usage=Usage(requests=1, input_tokens=25, output_tokens=5),
    )
    worker = Agent(worker_model, name="worker", tools=[search])  # no limit of its own
    lead_model = ScriptedModel(
        [call("worker", task="dig")], usage=Usage(requests=1, input_tokens=50, output_tokens=10)
    )
    lead = Agent(lead_model, name="lead", tools=[worker], limits=Limits(max_total_tokens=100))

    result = await lead.run("go")

    # 40 tokens were left: the worker stops after two requests of 30.
    assert len(worker_model.requests) == 2
    failure = result.messages[2].tool_results[0]
    assert failure.is_error
    assert failure.content == prompts.SUBAGENT_STOPPED.format(
        name="worker", reason="max_total_tokens"
    )
    assert (result.status, result.stop_reason) == ("limit_reached", "max_total_tokens")
    assert result.usage.total_tokens == 120


async def test_a_child_inherits_the_remaining_cost_budget_even_when_it_is_spent():
    worker_model = ScriptedModel([])  # never asked: nothing is left to spend
    worker = Agent(worker_model, name="worker")
    lead_model = ScriptedModel(
        [call("worker", task="dig"), "unreachable"],
        usage=Usage(requests=1, input_tokens=1, output_tokens=1, cost_usd=0.05),
    )
    lead = Agent(lead_model, name="lead", tools=[worker], limits=Limits(max_cost_usd=0.05))
    result = await lead.run("go")

    assert worker_model.requests == []
    assert result.messages[2].tool_results[0].content == prompts.SUBAGENT_STOPPED.format(
        name="worker", reason="max_cost_usd"
    )
    assert result.stop_reason == "max_cost_usd"

    cheap = Usage(requests=1, input_tokens=1, output_tokens=1, cost_usd=0.015)
    worker_model = FunctionModel(lambda request: call("search", query="x"), usage=cheap)
    worker = Agent(worker_model, name="worker", tools=[search], limits=Limits(max_cost_usd=1.0))
    lead_model = ScriptedModel([call("worker", task="dig"), "done"], usage=cheap)
    lead = Agent(lead_model, name="lead", tools=[worker], limits=Limits(max_cost_usd=0.05))
    await lead.run("go")
    assert len(worker_model.requests) == 3  # 0.035 was left: stopped at 0.045, not at 1.0


async def test_a_propagating_hook_sees_every_run_of_the_tree_once():
    tracer = Tracer()
    local = Recorder()
    researcher = Agent(
        ScriptedModel([call("search", query="solar"), "facts"]),
        name="researcher",
        tools=[search],
        hooks=[tracer],  # attached here too: still seen once
    )
    lead = Agent(
        ScriptedModel([call("researcher", task="find"), "report"]),
        name="lead",
        tools=[researcher],
        hooks=[tracer, local],
    )
    result = await lead.run("go")

    (lead_run, child_run) = tracer.runs
    assert lead_run == ("lead", result.run_id, None, 0)
    assert child_run[0] == "researcher" and child_run[2:] == (result.run_id, 1)
    assert tracer.requests == ["lead", "researcher", "researcher", "lead"]
    assert [name for name, *_ in local.runs] == ["lead"]  # not propagating: lead only
    assert Hooks.propagate is False


async def test_a_child_without_on_ask_uses_the_parents():
    asked: list[tuple[str, int]] = []
    log: list[str] = []

    @tool(requires_approval=True)
    def refund(amount: float) -> str:
        """Refund the customer."""
        log.append(f"refund {amount}")
        return "refunded"

    def approve(ctx, target, call_part, reason):
        asked.append((target.name, ctx.depth))
        return True

    def billing() -> Agent:
        return Agent(
            ScriptedModel([call("refund", amount=15), "Refunded."]), name="billing", tools=[refund]
        )

    lead = Agent(
        ScriptedModel([call("billing", task="refund 15"), "Done."]),
        tools=[billing()],
        permissions=Permissions(on_ask=approve),
    )
    result = await lead.run("go")
    assert result.output == "Done." and asked == [("refund", 1)] and log == ["refund 15.0"]

    # Without an on_ask anywhere, the child pauses and the caller gets an error result.
    lead = Agent(ScriptedModel([call("billing", task="refund 15"), "Not done."]), tools=[billing()])
    result = await lead.run("go")
    stopped = result.messages[2].tool_results[0]
    assert stopped.is_error
    assert stopped.content == prompts.SUBAGENT_STOPPED.format(name="billing", reason="paused")
    assert log == ["refund 15.0"]

    # A child's own on_ask wins over the parent's.
    own: list[str] = []
    child = billing()
    child.permissions = Permissions(on_ask=lambda ctx, t, c, r: own.append(t.name) or True)
    lead = Agent(
        ScriptedModel([call("billing", task="refund 15"), "Done."]),
        tools=[child],
        permissions=Permissions(on_ask=approve),
    )
    await lead.run("go")
    assert own == ["refund"] and asked == [("refund", 1)]


async def test_copy_replaces_some_arguments_and_keeps_the_rest():
    original = Agent(
        ScriptedModel(["from the original"]),
        name="writer",
        instructions="Be brief.",
        tools=[search],
        limits=Limits(max_turns=4),
    )
    other_model = ScriptedModel(["from the copy"])
    copy = original.copy(model=other_model, name="drafter")

    assert (copy.name, copy.instructions, copy.limits.max_turns) == ("drafter", "Be brief.", 4)
    assert copy.model is other_model and original.name == "writer"
    assert copy.tools == original.tools and copy.tools is not original.tools
    assert (await copy.run("go")).output == "from the copy"
    assert other_model.requests[0].messages[0].text == "Be brief."
    with pytest.raises(ConfigurationError, match="not Agent arguments"):
        original.copy(modle="x")


async def test_max_tool_calls_stops_the_run_and_closes_the_open_calls():
    log: list[str] = []

    @tool(read_only=True)
    def lookup(key: str) -> str:
        """Look something up."""
        log.append(key)
        return key

    model = ScriptedModel(
        [[call("lookup", key="a"), call("lookup", key="b")], [call("lookup", key="c")], "unused"]
    )
    session = Session()
    result = await Agent(model, tools=[lookup], limits=Limits(max_tool_calls=2)).run(
        "go", session=session
    )
    # Two calls ran; the third was not started, and the run stopped there.
    assert (result.status, result.stop_reason) == ("limit_reached", "max_tool_calls")
    assert log == ["a", "b"] and len(model.requests) == 2
    refused = result.messages[-1].tool_results[0]
    assert refused.is_error
    assert refused.content == prompts.TOOL_CALL_LIMIT.format(name="lookup", limit=2)
    assert not (await session.state()).unresolved_tool_calls

    log.clear()
    model = ScriptedModel([[call("lookup", key="a"), call("lookup", key="b")]])
    result = await Agent(model, tools=[lookup], limits=Limits(max_tool_calls=1)).run("go")
    assert log == ["a"] and result.stop_reason == "max_tool_calls"
    ran, refused = result.messages[-1].tool_results
    assert (ran.content, refused.is_error) == ("a", True)
