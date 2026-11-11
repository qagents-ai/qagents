from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402
from opentelemetry.trace import StatusCode  # noqa: E402

from qagents import Agent, ToolError, Usage, tool  # noqa: E402
from qagents.errors import DeferToolCall, ModelServerError  # noqa: E402
from qagents.messages import ToolCallPart  # noqa: E402
from qagents.models.base import ModelRequest  # noqa: E402
from qagents.models.testing import ScriptedModel, call  # noqa: E402
from qagents.tools import RunContext  # noqa: E402
from qagents.tracing import UNFINISHED, OTelHooks  # noqa: E402


@pytest.fixture
def spans():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, provider.get_tracer("test")


@tool(read_only=True)
def lookup(key: str) -> str:
    """Look something up."""
    if key == "missing":
        raise ToolError("No such key.")
    return "value"


async def test_one_run_is_one_trace_with_model_and_tool_spans(spans):
    exporter, tracer = spans
    model = ScriptedModel(
        [[call("lookup", key="a"), call("lookup", key="missing")], "Done."],
        usage=Usage(requests=1, input_tokens=20, output_tokens=5, cost_usd=0.001),
    )
    agent = Agent(model, name="helper", tools=[lookup], hooks=[OTelHooks(tracer)])
    await agent.run("go")

    finished = exporter.get_finished_spans()
    names = sorted(span.name for span in finished)
    assert names == [
        "chat scripted", "chat scripted", "execute_tool lookup", "execute_tool lookup",
        "invoke_agent helper",
    ]  # fmt: skip
    root = next(span for span in finished if span.name == "invoke_agent helper")
    assert all(span.parent.span_id == root.context.span_id for span in finished if span is not root)
    assert len({span.context.trace_id for span in finished}) == 1
    assert root.attributes["gen_ai.usage.input_tokens"] == 40
    assert root.attributes["qagents.status"] == "completed"
    assert root.attributes["qagents.cost_usd"] == pytest.approx(0.002)

    chat = next(span for span in finished if span.name == "chat scripted")
    assert chat.attributes["gen_ai.operation.name"] == "chat"
    assert chat.attributes["gen_ai.provider.name"] == "test"
    assert chat.attributes["gen_ai.usage.output_tokens"] == 5
    assert "gen_ai.input.messages" not in chat.attributes  # content is off by default

    tools = [span for span in finished if span.name == "execute_tool lookup"]
    statuses = sorted(span.status.status_code.name for span in tools)
    assert statuses == ["ERROR", "UNSET"]  # the failed lookup is marked, the other is not


async def test_content_is_recorded_only_on_request_and_failures_close_spans(spans):
    exporter, tracer = spans
    model = ScriptedModel(["Hi."])
    await Agent(model, hooks=[OTelHooks(tracer, record_content=True)]).run("hello")
    chat = next(span for span in exporter.get_finished_spans() if span.name.startswith("chat"))
    assert '"text": "hello"' in chat.attributes["gen_ai.input.messages"]
    assert '"text": "Hi."' in chat.attributes["gen_ai.output.messages"]

    exporter.clear()
    failing = ScriptedModel([ModelServerError("upstream down", ambiguous=True)])
    with pytest.raises(ModelServerError):
        await Agent(failing, hooks=[OTelHooks(tracer)]).run("hello")
    finished = {span.name: span for span in exporter.get_finished_spans()}
    assert set(finished) == {"chat", "invoke_agent agent"}  # nothing left open
    assert finished["invoke_agent agent"].status.status_code == StatusCode.ERROR
    assert finished["invoke_agent agent"].events[0].name == "exception"


async def test_a_sub_agent_run_nests_under_the_tool_span_that_started_it(spans):
    exporter, tracer = spans
    hooks = OTelHooks(tracer)
    worker = Agent(
        ScriptedModel([call("lookup", key="a"), "Found."]), name="worker", tools=[lookup]
    )
    lead = Agent(
        ScriptedModel([call("worker", task="find a"), "Done."]),
        name="lead",
        tools=[worker],
        hooks=[hooks],  # only on the lead: OTelHooks propagates
    )
    result = await lead.run("go")

    finished = {span.name: span for span in exporter.get_finished_spans()}
    assert sorted(span.name for span in exporter.get_finished_spans()) == [
        "chat scripted", "chat scripted", "chat scripted", "chat scripted",
        "execute_tool lookup", "execute_tool worker", "invoke_agent lead", "invoke_agent worker",
    ]  # fmt: skip
    assert len({span.context.trace_id for span in exporter.get_finished_spans()}) == 1
    lead_span, delegation = finished["invoke_agent lead"], finished["execute_tool worker"]
    worker_span = finished["invoke_agent worker"]
    assert delegation.parent.span_id == lead_span.context.span_id
    assert worker_span.parent.span_id == delegation.context.span_id
    assert finished["execute_tool lookup"].parent.span_id == worker_span.context.span_id
    assert worker_span.attributes["qagents.parent_run_id"] == result.run_id
    assert (
        worker_span.attributes["qagents.parent_tool_call_id"]
        == (delegation.attributes["gen_ai.tool.call.id"])
    )
    assert worker_span.attributes["qagents.depth"] == 1

    # The same hook object on both agents is used once per run: no duplicate spans.
    exporter.clear()
    worker = Agent(ScriptedModel(["Found."]), name="worker", hooks=[hooks])
    lead = Agent(
        ScriptedModel([call("worker", task="t"), "Done."]),
        name="lead",
        tools=[worker],
        hooks=[hooks],
    )
    await lead.run("go")
    assert sorted(span.name for span in exporter.get_finished_spans()) == [
        "chat scripted", "chat scripted", "chat scripted",
        "execute_tool worker", "invoke_agent lead", "invoke_agent worker",
    ]  # fmt: skip


async def test_a_deferred_tool_call_ends_its_span_without_an_error(spans):
    exporter, tracer = spans

    @tool(read_only=True)
    def ask_expert(question: str) -> str:
        """Ask a human expert."""
        raise DeferToolCall("waiting for the expert")

    model = ScriptedModel([call("ask_expert", question="q")])
    result = await Agent(model, tools=[ask_expert], hooks=[OTelHooks(tracer)]).run("ask")
    assert result.status == "paused"
    finished = {span.name: span for span in exporter.get_finished_spans()}
    deferred = finished["execute_tool ask_expert"]
    assert deferred.status.status_code == StatusCode.UNSET
    assert deferred.attributes["qagents.tool.deferred"] is True
    root = finished["invoke_agent agent"]
    assert root.attributes["qagents.status"] == "paused"
    assert root.attributes["qagents.pending_inputs"] == (result.pending_inputs[0].tool_call_id,)


async def test_child_runs_whose_end_was_never_reported_are_closed_with_the_parent(spans):
    exporter, tracer = spans
    hooks = OTelHooks(tracer)
    parent = RunContext(run_id="parent", session_id="s", agent_name="lead")
    await hooks.on_run_start(parent)
    await hooks.before_tool(parent, ToolCallPart(id="c1", name="worker"))
    child = RunContext(
        run_id="child", agent_name="worker", parent_run_id="parent", parent_tool_call_id="c1"
    )
    await hooks.on_run_start(child)
    await hooks.before_model(child, ModelRequest(messages=[]))
    await hooks.on_run_end(parent, None, RuntimeError("cancelled from outside"))

    finished = {span.name: span for span in exporter.get_finished_spans()}
    assert set(finished) == {
        "invoke_agent lead",
        "execute_tool worker",
        "invoke_agent worker",
        "chat",
    }
    for name in ("execute_tool worker", "invoke_agent worker", "chat"):
        assert finished[name].status.description == UNFINISHED
    assert finished["invoke_agent worker"].parent.span_id == (
        finished["execute_tool worker"].context.span_id
    )
