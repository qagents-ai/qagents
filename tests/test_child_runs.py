"""Linked child runs (sub-agents), inherited budgets and hooks, Agent.copy, max_tool_calls."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel

from qagents import (
    Agent,
    Hooks,
    Limits,
    Message,
    Permissions,
    RunContext,
    Session,
    Usage,
    prompts,
    tool,
)
from qagents.errors import ConfigurationError, ModelServerError
from qagents.models.base import ModelResponse
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
    assert sorted(log) == ["a", "b"] and len(model.requests) == 2  # read-only: in parallel
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


# -- 0.2 review fixes: what a sub-agent returns, budgets shared live, depth ------------------


async def test_as_tool_returns_the_accepted_output_not_the_last_message_text():
    # A sub-agent in output_mode="tool" answers through final_answer: its text is empty.
    geo = Agent(
        ScriptedModel([call("final_answer", value="Paris")]), name="geo", output_mode="tool"
    )
    lead_model = ScriptedModel([call("geo", task="capital of France"), "done"])
    await Agent(lead_model, tools=[geo]).run("go")
    assert lead_model.requests[1].messages[-1].tool_results[0].content == "Paris"

    # An output validator's replacement is what the caller sees.
    def redact(ctx, text: str) -> str:
        return text.replace("4111-1111", "[redacted]")

    finder = Agent(
        ScriptedModel(["card 4111-1111 found"]), name="finder", output_validators=[redact]
    )
    lead_model = ScriptedModel([call("finder", task="look"), "done"])
    await Agent(lead_model, tools=[finder]).run("go")
    assert lead_model.requests[1].messages[-1].tool_results[0].content == "card [redacted] found"


async def test_a_refused_sub_agent_is_an_error_result_never_ok():
    class Report(BaseModel):
        summary: str

    refusal = ModelResponse(
        message=Message.assistant("I cannot help with that."), finish_reason="content_filter"
    )
    for output_type in (Report, None):
        analyst = Agent(ScriptedModel([refusal]), name="analyst", output_type=output_type)
        lead_model = ScriptedModel([call("analyst", task="analyse X"), "ok"])
        await Agent(lead_model, tools=[analyst]).run("go")
        result = lead_model.requests[1].messages[-1].tool_results[0]
        assert result.is_error
        assert result.content == prompts.SUBAGENT_STOPPED.format(
            name="analyst", reason="content_filter"
        )


async def test_a_failed_or_timed_out_child_still_counts_its_usage_in_the_parent():
    billed = Usage(requests=1, input_tokens=4000, output_tokens=1000)
    worker = Agent(
        ScriptedModel(
            [call("search", query="x"), ModelServerError("down", retryable=False)], usage=billed
        ),
        name="worker",
        tools=[search],
    )
    lead = Agent(
        ScriptedModel([call("worker", task="dig"), "unreachable"]),
        tools=[worker],
        limits=Limits(max_total_tokens=3000),
    )
    result = await lead.run("go")
    assert result.usage.total_tokens == 5000  # the failed child's request counts
    assert (result.status, result.stop_reason) == ("limit_reached", "max_total_tokens")

    async def reply(request):
        if len(request.messages) < 3:
            return call("search", query="x")
        await asyncio.sleep(5)  # the second request never ends in time
        return "late"

    sleeper = Agent(FunctionModel(reply, usage=billed), name="sleeper", tools=[search])
    lead = Agent(
        ScriptedModel([call("sleeper", task="dig"), "done"]),
        tools=[sleeper.as_tool(timeout=0.3)],
    )
    result = await lead.run("go")
    assert "did not finish within 0.3 seconds" in result.messages[2].tool_results[0].content
    assert result.usage.total_tokens == 5000  # its first request was billed before the timeout


async def test_children_running_in_parallel_share_the_parents_budget_live():
    def worker(name: str) -> Agent:
        model = FunctionModel(
            lambda request: call("search", query="more"),
            usage=Usage(requests=1, input_tokens=25, output_tokens=5),
        )
        return Agent(model, name=name, tools=[search])

    a, b = worker("a"), worker("b")
    lead = Agent(
        ScriptedModel([[call("a", task="dig"), call("b", task="dig")], "done"]),
        tools=[a.as_tool(read_only=True), b.as_tool(read_only=True)],
        limits=Limits(max_total_tokens=100),
    )
    result = await lead.run("go")
    # 100 tokens are shared: four requests of 30 between the two children, not four each.
    assert len(a.model.requests) + len(b.model.requests) == 4
    assert result.usage.total_tokens == 120
    assert result.stop_reason == "max_total_tokens"


async def test_adding_a_childs_result_usage_again_does_not_count_it_twice():
    child = Agent(ScriptedModel(["hi"], usage=Usage(requests=1, input_tokens=7)), name="child")

    @tool
    async def ask_child(ctx: RunContext[Any]) -> str:
        """Ask the child."""
        result = await child.run("hello", parent=ctx)
        ctx.add_usage(result.usage)  # what code written for 0.2 does
        return result.output

    lead = Agent(ScriptedModel([call("ask_child"), "done"]), tools=[ask_child])
    result = await lead.run("go")
    assert result.usage.input_tokens == 7 and result.usage.requests == 3


async def test_max_depth_is_inherited_by_every_level_and_reported_to_the_model():
    leaf_model = ScriptedModel(["leaf answer"])
    leaf = Agent(leaf_model, name="leaf")
    mid_model = ScriptedModel([call("leaf", task="go deeper"), "mid answer"])
    mid = Agent(mid_model, name="mid", tools=[leaf])  # default limits: max_depth=5
    root = Agent(
        ScriptedModel([call("mid", task="delegate"), "root answer"]),
        name="root",
        tools=[mid],
        limits=Limits(max_depth=1),
    )
    tracer = Tracer()
    root.hooks.append(tracer)
    result = await root.run("go")

    assert [depth for *_, depth in tracer.runs] == [0, 1]  # the leaf never started
    assert leaf_model.requests == []
    refused = mid_model.requests[1].messages[-1].tool_results[0]
    assert refused.is_error
    assert refused.content == prompts.SUBAGENT_TOO_DEEP.format(name="leaf", depth=2, limit=1)
    assert result.output == "root answer"


async def test_as_tool_takes_a_timeout_and_tools_can_read_their_limits():
    helper = Agent(ScriptedModel([]), name="helper")
    assert helper.as_tool(timeout=600).timeout == 600

    seen: list[tuple[Any, dict[str, Any]]] = []

    @tool(read_only=True)
    def budget(ctx: RunContext[Any]) -> str:
        """Report the budget."""
        seen.append((ctx.limits, ctx.remaining_budget()))
        return "ok"

    worker = Agent(
        ScriptedModel([call("budget"), "done"], usage=Usage(requests=1, input_tokens=10)),
        name="worker",
        tools=[budget],
    )
    lead = Agent(
        ScriptedModel([call("worker", task="t"), "done"], usage=Usage(requests=1, input_tokens=30)),
        tools=[worker],
        limits=Limits(max_total_tokens=100, max_depth=3),
    )
    await lead.run("go")
    ((limits, remaining),) = seen
    assert limits.max_total_tokens == 70 and limits.max_depth == 3
    assert remaining["max_total_tokens"] == 60  # live: the worker's own first request counts
    assert RunContext().limits is None


async def test_a_cost_that_is_not_reported_stops_a_run_with_a_cost_budget():
    unpriced = FunctionModel(
        lambda request: call("search", query="x"), usage=Usage(requests=1, input_tokens=200_000)
    )
    result = await Agent(unpriced, tools=[search], limits=Limits(max_cost_usd=0.01)).run("go")
    assert (result.status, result.stop_reason) == ("limit_reached", "max_cost_usd")
    assert len(unpriced.requests) == 1
    assert result.usage.unpriced_requests == 1 and result.usage.cost_usd is None

    priced = Usage(requests=1, input_tokens=10, cost_usd=0.001)
    mixed = priced.add(Usage(requests=1, input_tokens=10))
    assert mixed.cost_usd == 0.001 and mixed.unpriced_requests == 1 and not mixed.cost_complete
    assert "unpriced_requests" not in priced.add(priced).model_dump()  # 0 is left out


async def test_a_response_marked_unpriced_stops_a_run_with_a_cost_budget():
    # What a model reports when its server sent no usage at all: no tokens, no cost, unpriced.
    unreported = Usage(requests=1, unpriced_requests=1)
    silent = FunctionModel(lambda request: call("search", query="x"), usage=unreported)
    limits = Limits(max_cost_usd=0.01, max_turns=6)
    session = Session()
    result = await Agent(silent, tools=[search], limits=limits).run("go", session=session)
    assert (result.status, result.stop_reason) == ("limit_reached", "max_cost_usd")
    assert len(silent.requests) == 1
    assert result.usage.unpriced_requests == 1 and not result.usage.cost_complete
    assert (await session.state()).run_usage.unpriced_requests == 1  # journaled for a resume

    # A test model's usage (one request, nothing else) counts as free.
    free = FunctionModel(lambda request: "done", usage=Usage(requests=1))
    plain = await Agent(free, limits=limits).run("go")
    assert plain.completed and plain.usage.cost_complete
