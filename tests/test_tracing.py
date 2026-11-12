from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402
from opentelemetry.trace import StatusCode  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from qagents import Agent, Limits, Message, SummarizingCompactor, ToolError, Usage, tool  # noqa: E402
from qagents.errors import (  # noqa: E402
    DeferToolCall,
    ModelServerError,
    OutputValidationError,
)
from qagents.messages import ToolCallPart  # noqa: E402
from qagents.models.base import ModelRequest  # noqa: E402
from qagents.models.testing import ScriptedModel, call  # noqa: E402
from qagents.tools import RunContext  # noqa: E402
from qagents.tracing import TOOL_FAILED, UNFINISHED, OTelHooks  # noqa: E402

SECRET = "patient John Doe, SSN 123-45-6789"


def texts(span) -> list[str]:
    """Every string a span exports: name, status, attributes and event attributes."""
    found = [span.name, span.status.description or ""]
    values = list(span.attributes.values())
    for event in span.events:
        found.append(event.name)
        values.extend(event.attributes.values())
    for value in values:
        found.extend(str(item) for item in (value if isinstance(value, tuple) else (value,)))
    return found


def leaks(exporter, marker: str = "123-45-6789") -> list[str]:
    return [
        text for span in exporter.get_finished_spans() for text in texts(span) if marker in text
    ]


class Refund(BaseModel):
    amount: int


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


# -- no content in traces unless asked -----------------------------------------------------


@tool(read_only=True)
def lookup_customer(email: str) -> str:
    """Look up a customer."""
    raise ToolError(f"No account for {email}; {SECRET} is on file.")


@tool(read_only=True)
def charge_card(card: str) -> str:
    """Charge a card."""
    raise RuntimeError(f"gateway refused {card} for {SECRET}")


async def test_without_record_content_no_error_text_reaches_the_spans(spans):
    exporter, tracer = spans
    # Failed tool calls: a fixed status description, and the type of an unexpected exception.
    model = ScriptedModel(
        [
            [call("lookup_customer", email="jane@example.com"), call("charge_card", card="4111")],
            "Sorry.",
        ]
    )
    tools = [lookup_customer, charge_card]
    await Agent(model, tools=tools, hooks=[OTelHooks(tracer)]).run("find me")
    finished = {span.name: span for span in exporter.get_finished_spans()}
    for name in ("execute_tool lookup_customer", "execute_tool charge_card"):
        assert finished[name].status.status_code == StatusCode.ERROR
        assert finished[name].status.description == TOOL_FAILED
    assert finished["execute_tool charge_card"].attributes["error.type"] == "RuntimeError"
    assert leaks(exporter) == []

    # A run that fails on the model's answer: the exception's type only, no message or stack.
    exporter.clear()
    answer = f'{{"amount": "{SECRET}"}}'
    agent = Agent(
        ScriptedModel([answer]),
        output_type=Refund,
        output_mode="prompted",
        limits=Limits(output_retries=0),
        hooks=[OTelHooks(tracer)],
    )
    with pytest.raises(OutputValidationError):
        await agent.run("refund")
    root = next(s for s in exporter.get_finished_spans() if s.name == "invoke_agent agent")
    assert root.status.status_code == StatusCode.ERROR
    assert root.status.description == "OutputValidationError"
    (event,) = root.events
    assert event.name == "exception"
    assert dict(event.attributes) == {"exception.type": "qagents.errors.OutputValidationError"}

    # A provider error whose message echoes the request.
    failing = ScriptedModel([ModelServerError(f"upstream rejected: {SECRET}")])
    with pytest.raises(ModelServerError):
        await Agent(failing, hooks=[OTelHooks(tracer)]).run("hello")
    assert leaks(exporter) == []


async def test_with_record_content_error_text_is_recorded(spans):
    exporter, tracer = spans
    model = ScriptedModel([call("lookup_customer", email="jane@example.com"), "Sorry."])
    await Agent(model, tools=[lookup_customer], hooks=[OTelHooks(tracer, record_content=True)]).run(
        "find me"
    )
    failing = ScriptedModel([ModelServerError(f"upstream rejected: {SECRET}")])
    with pytest.raises(ModelServerError):
        await Agent(failing, hooks=[OTelHooks(tracer, record_content=True)]).run("hello")
    finished = {span.name: span for span in exporter.get_finished_spans()}
    assert SECRET in finished["execute_tool lookup_customer"].status.description
    root = finished["invoke_agent agent"]
    assert SECRET in root.status.description
    assert SECRET in root.events[0].attributes["exception.message"]


async def test_a_summary_request_has_a_chat_span_and_counts_in_the_run(spans):
    exporter, tracer = spans
    usage = Usage(requests=1, input_tokens=100, output_tokens=10)
    summarizer = ScriptedModel(["Earlier: the user asked twice."], name="summarizer", usage=usage)
    agent = Agent(
        ScriptedModel(["Third answer."], usage=usage),
        compactor=SummarizingCompactor(trigger_tokens=0, keep_last=1, model=summarizer),
        hooks=[OTelHooks(tracer)],
    )
    history = [
        Message.user("q1"),
        Message.assistant("a1"),
        Message.user("q2"),
        Message.assistant("a2"),
    ]
    await agent.run("q3", history=history)
    names = sorted(span.name for span in exporter.get_finished_spans())
    assert names == ["chat scripted", "chat summarizer", "invoke_agent agent"]
    root = next(s for s in exporter.get_finished_spans() if s.name == "invoke_agent agent")
    assert root.attributes["gen_ai.usage.input_tokens"] == 200
