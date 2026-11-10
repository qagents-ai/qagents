from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402
from opentelemetry.trace import StatusCode  # noqa: E402

from qagents import Agent, ToolError, Usage, tool  # noqa: E402
from qagents.errors import ModelServerError  # noqa: E402
from qagents.models.testing import ScriptedModel, call  # noqa: E402
from qagents.tracing import OTelHooks  # noqa: E402


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
