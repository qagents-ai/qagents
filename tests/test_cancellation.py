"""Cancel, timeouts and the tasks of the loop: nothing starts after a stop, nothing hangs.

Each test here is a regression test for a defect found in 0.2.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest

from qagents import Agent, Hooks, Limits, Message, Permissions, Session, prompts, tool
from qagents import tools as tools_module
from qagents.context import Compactor
from qagents.errors import ModelRateLimitError
from qagents.events import ToolCallFinished
from qagents.messages import ToolCallPart
from qagents.models.base import Model, ModelRequest, ModelResponse, ResponseDone, TextDelta
from qagents.models.testing import ScriptedModel, call


def transfer_tool(log: list[str]):
    @tool
    async def transfer(amount: int) -> str:
        """Move money."""
        log.append(f"transfer {amount}")
        return "done"

    return transfer


async def entries_of(session: Session) -> list[str]:
    return [entry.kind for entry in await session.entries()]


# -- nothing starts after a stop (F16, F3, F85, F13) -----------------------------------------


async def test_a_cancel_while_on_ask_decides_runs_nothing():
    log: list[str] = []
    holder: dict[str, Any] = {}

    def approve(ctx, target, call_part, reason):
        holder["stream"].cancel()  # the user stops the run while the approval is pending
        return True

    agent = Agent(
        ScriptedModel([call("transfer", amount=7)]),
        tools=[transfer_tool(log)],
        permissions=Permissions(mode="ask_all", on_ask=approve),
    )
    session = Session()
    stream = agent.stream("pay", session=session)
    holder["stream"] = stream
    result = await stream.wait()

    assert log == []  # the approved side effect never ran
    assert "tool_started" not in await entries_of(session)
    assert result.status == "cancelled"
    assert result.messages[-1].tool_results[0].content == prompts.TOOL_NOT_EXECUTED.format(
        name="transfer"
    )


async def test_a_timeout_while_a_human_decides_runs_nothing():
    log: list[str] = []

    async def slow_approval(ctx, target, call_part, reason):
        await asyncio.sleep(0.3)
        return True

    agent = Agent(
        ScriptedModel([call("transfer", amount=10)]),
        tools=[transfer_tool(log)],
        permissions=Permissions(mode="ask_all", on_ask=slow_approval),
        limits=Limits(timeout_s=0.1),
    )
    result = await agent.run("pay")
    assert log == []
    assert (result.status, result.stop_reason) == ("limit_reached", "timeout")


async def test_a_cancel_from_the_consumer_stops_the_rest_of_the_turn():
    log: list[str] = []
    agent = Agent(
        ScriptedModel([[call("nope"), call("transfer", amount=3)]]), tools=[transfer_tool(log)]
    )
    stream = agent.stream("go")  # the default in-memory session: its writes never suspend
    async for event in stream:
        if isinstance(event, ToolCallFinished) and event.call.name == "nope":
            stream.cancel()
    assert log == []
    assert stream.result is not None and stream.result.status == "cancelled"


async def test_a_timeout_in_a_turn_that_hit_max_tool_calls_is_reported_as_a_timeout():
    @tool
    async def slow(x: int) -> str:
        """Take long."""
        await asyncio.sleep(5)
        return "late"

    agent = Agent(
        ScriptedModel([[call("slow", x=1), call("slow", x=2)]]),
        tools=[slow],
        limits=Limits(max_tool_calls=1, timeout_s=0.2),
    )
    session = Session()
    result = await agent.run("go", session=session)

    assert (result.status, result.stop_reason) == ("limit_reached", "timeout")
    last = result.messages[-1]
    assert last.role == "tool"
    assert [r.content for r in last.tool_results] == [
        prompts.TOOL_INTERRUPTED.format(name="slow"),
        prompts.TOOL_CALL_LIMIT.format(name="slow", limit=1),
    ]
    assert not (await session.state()).unresolved_tool_calls


async def test_calls_closed_by_a_cancel_get_their_tool_call_finished_event():
    @tool(read_only=True)
    async def slow(x: int) -> str:
        """Take long."""
        await asyncio.sleep(5)
        return "late"

    agent = Agent(ScriptedModel([[call("slow", x=1), call("slow", x=2)]]), tools=[slow])
    stream = agent.stream("go")
    events = []
    async for event in stream:
        events.append(event)
        if event.type == "tool_call_started":
            stream.cancel()
    assert stream.result is not None
    finished = {e.call.id: e.result.content for e in events if e.type == "tool_call_finished"}
    closing = stream.result.messages[-1].tool_results
    assert finished == {r.tool_call_id: r.content for r in closing}
    assert set(finished.values()) == {prompts.TOOL_CANCELLED.format(name="slow")}


async def test_a_cancelled_non_idempotent_call_has_an_unknown_outcome():
    started = asyncio.Event()

    @tool
    async def pay(amount: int) -> str:
        """Pay."""
        started.set()
        await asyncio.sleep(5)
        return "paid"

    stream = Agent(ScriptedModel([call("pay", amount=1)]), tools=[pay]).stream("go")
    async for event in stream:
        if event.type == "tool_call_started":
            stream.cancel()
    assert stream.result is not None
    assert stream.result.messages[-1].tool_results[0].content == prompts.TOOL_INTERRUPTED.format(
        name="pay"
    )


async def test_a_cancel_during_the_retry_backoff_sends_no_other_request():
    model = ScriptedModel([ModelRateLimitError("busy", retryable=True, retry_after=1.0), "never"])
    agent = Agent(model, limits=Limits(model_retries=2))
    stream = agent.stream("go")

    async def stop_soon() -> None:
        await asyncio.sleep(0.2)
        stream.cancel()

    stopper = asyncio.create_task(stop_soon())
    result = await stream.wait()
    await stopper
    assert result.status == "cancelled"
    assert len(model.requests) == 1


# -- tool tasks that end strangely never hang the run (F4, F19, F11, F96) -------------------


async def test_a_tool_that_raises_cancelled_error_by_itself_gets_an_error_result():
    @tool(read_only=True)
    async def lookup(key: str) -> str:
        """Look up."""
        future = asyncio.get_running_loop().create_future()
        future.cancel()
        return await future  # someone else cancelled what it waits for

    agent = Agent(ScriptedModel([call("lookup", key="k"), "done"]), tools=[lookup])
    result = await asyncio.wait_for(agent.run("go"), 3)
    failed = result.messages[2].tool_results[0]
    assert failed.is_error and failed.metadata["exception"] == "CancelledError"
    assert result.output == "done"


async def test_a_hook_that_raises_cancelled_error_stops_the_run_instead_of_hanging_it():
    class Broken(Hooks):
        async def before_tool(self, ctx, call_part):
            raise asyncio.CancelledError

    @tool(read_only=True)
    def lookup(key: str) -> str:
        """Look up."""
        return "x"

    agent = Agent(ScriptedModel([call("lookup", key="k")]), tools=[lookup], hooks=[Broken()])
    with pytest.raises(RuntimeError, match="raised CancelledError"):
        await asyncio.wait_for(agent.run("go"), 3)


async def test_an_emitted_tool_call_finished_does_not_end_the_group_early():
    @tool(read_only=True)
    async def relay(ctx) -> str:
        """Forward an inner event."""
        inner = ToolCallPart(id="inner", name="other")
        ctx.emit(
            ToolCallFinished(
                call=inner,
                result=prompts_result(inner),
                agent="inner",
            )
        )
        await asyncio.sleep(0.05)
        return "relayed"

    agent = Agent(ScriptedModel([call("relay"), "done"]), tools=[relay])
    result = await agent.run("go")
    assert result.messages[2].tool_results[0].content == "relayed"
    assert result.output == "done"


def prompts_result(inner: ToolCallPart):
    from qagents.messages import ToolResultPart

    return ToolResultPart(tool_call_id=inner.id, name=inner.name, content="inner done")


async def test_cancelling_the_task_during_a_tool_leaves_no_pending_queue_task():
    @tool(read_only=True)
    async def slow() -> str:
        """Take long."""
        await asyncio.sleep(30)
        return "late"

    agent = Agent(ScriptedModel([call("slow")]), tools=[slow])
    task = asyncio.create_task(agent.run("go"))
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    leftovers = [
        t for t in asyncio.all_tasks() if "Queue.get" in repr(t.get_coro()) and not t.done()
    ]
    assert leftovers == []


# -- the model stream runs in the run's task (F93, F139, F95) --------------------------------


class CustomStream(Model):
    """A model whose stream body is a test function."""

    provider = "test"
    name = "custom"

    def __init__(self, body) -> None:
        self.body = body
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[Any]:
        self.requests.append(request)
        async for event in self.body():
            yield event


def done(text: str) -> ResponseDone:
    return ResponseDone(response=ModelResponse(message=Message.assistant(text)))


async def test_a_timeout_inside_a_model_stream_fires_under_the_agent():
    async def body():
        async with asyncio.timeout(0.3):  # the provider's own deadline
            for _ in range(10):
                await asyncio.sleep(0.1)
                yield TextDelta(text="x")
            yield done("x" * 10)

    with pytest.raises(TimeoutError):
        await Agent(CustomStream(body)).run("go")


async def test_context_variables_set_in_a_model_stream_survive_its_yields():
    var: contextvars.ContextVar[str] = contextvars.ContextVar("var", default="unset")
    seen: list[str] = []

    async def body():
        var.set("set")
        yield TextDelta(text="a")
        seen.append(var.get())
        yield done("a")

    await Agent(CustomStream(body)).run("go")
    assert seen == ["set"]


async def test_an_outer_cancellation_is_not_swallowed_while_the_stream_closes():
    async def body():
        try:
            while True:
                await asyncio.sleep(0.05)
                yield TextDelta(text="x")
        finally:
            await asyncio.sleep(0.5)  # closing the HTTP response takes a while

    agent = Agent(CustomStream(body), limits=Limits(timeout_s=0.2))
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.4):
            await agent.run("go")


async def test_a_timeout_stops_compaction_and_sends_no_request():
    class SlowSummary(Compactor):
        async def compact(self, ctx, messages, last_response, model):
            await asyncio.sleep(3)  # a summary request that takes long
            return None

    model = ScriptedModel(["never"])
    agent = Agent(model, compactor=SlowSummary(), limits=Limits(timeout_s=0.2))
    began = time.monotonic()
    result = await agent.run("go")
    assert time.monotonic() - began < 1.5
    assert (result.status, result.stop_reason) == ("limit_reached", "timeout")
    assert model.requests == [] and result.turns == 0


# -- sync tools: events from their thread arrive at once (F94) ------------------------------


async def test_an_event_a_sync_tool_emits_arrives_while_it_still_works():
    release = threading.Event()

    @tool(read_only=True)
    def work(ctx) -> str:
        """Work in steps."""
        time.sleep(0.3)  # the event loop is idle, waiting for the tool
        ctx.emit({"type": "progress", "step": 1})
        return "released" if release.wait(2) else "never released"

    stream = Agent(ScriptedModel([call("work"), "done"]), tools=[work]).stream("go")
    order: list[str] = []
    async for event in stream:
        if isinstance(event, dict):
            order.append("progress")
            release.set()  # the tool is still running while its event arrives
        elif event.type == "tool_call_finished":
            order.append(event.result.content)
    assert order == ["progress", "released"]


def test_a_sync_tool_that_emits_does_not_hang_in_debug_mode():
    @tool(read_only=True)
    def work(ctx) -> str:
        """Work."""
        time.sleep(0.05)
        ctx.emit({"type": "progress"})
        time.sleep(0.05)
        return "finished"

    async def main():
        agent = Agent(ScriptedModel([call("work"), "done"]), tools=[work])
        return await asyncio.wait_for(agent.run("go"), 5)

    result = asyncio.run(main(), debug=True)
    assert result.messages[2].tool_results[0].content == "finished"


async def test_a_sync_thread_left_running_by_a_run_timeout_counts_as_abandoned(monkeypatch):
    monkeypatch.setattr(tools_module, "_ABANDONED", set())
    release = threading.Event()

    @tool(read_only=True)
    def hang() -> str:
        """Hang."""
        release.wait(5)
        return "late"

    try:
        agent = Agent(ScriptedModel([call("hang")]), tools=[hang], limits=Limits(timeout_s=0.2))
        result = await agent.run("go")
        assert result.stop_reason == "timeout"
        # Its thread still runs, so it counts toward the bound on abandoned worker threads.
        assert len(tools_module._ABANDONED) == 1
    finally:
        release.set()
    for _ in range(100):  # released: the thread returns and is no longer counted
        if not tools_module._ABANDONED:
            break
        await asyncio.sleep(0.02)
    assert not tools_module._ABANDONED


@pytest.mark.parametrize("steal", [False, True])
async def test_a_direct_run_stops_active_work_when_renewal_refuses(monkeypatch, steal):
    from qagents import InMemoryStore
    from qagents.errors import LeaseError
    from qagents.models.testing import FunctionModel

    monkeypatch.setattr("qagents.agent.RUN_LEASE_TTL_S", 0.09)
    cancelled = asyncio.Event()

    class RefusingStore(InMemoryStore):
        async def renew_lease(self, session_id, owner, ttl):
            if steal:
                await self.release_lease(session_id, owner)
                assert await self.acquire_lease(session_id, "other", 10)
            return False

    async def reply(request):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    store = RefusingStore()
    session = Session(store)
    running = asyncio.create_task(Agent(FunctionModel(reply)).run("go", session=session))
    try:
        if steal:
            with pytest.raises(LeaseError):
                await asyncio.wait_for(running, timeout=1)
        else:
            result = await asyncio.wait_for(running, timeout=1)
            assert result.status == "cancelled" and result.stop_reason == "lease_lost"
        assert cancelled.is_set()
    finally:
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
