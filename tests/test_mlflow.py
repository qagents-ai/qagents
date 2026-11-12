"""MLflowHooks against a real MLflow tracking store (SQLite in a temporary directory)."""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from typing import Any

import pytest
from pydantic import BaseModel

mlflow = pytest.importorskip("mlflow")
pytest.importorskip("sqlalchemy")

import mlflow.tracking._tracking_service.utils as tracking_utils  # noqa: E402
import mlflow.tracking.fluent as fluent  # noqa: E402
from mlflow.entities import SpanStatusCode, SpanType  # noqa: E402
from mlflow.tracing.constant import SpanAttributeKey, TraceMetadataKey  # noqa: E402
from mlflow.types.chat import ChatMessage  # noqa: E402

from qagents import (  # noqa: E402
    Agent,
    Hooks,
    Limits,
    Message,
    Session,
    SummarizingCompactor,
    ToolError,
    Usage,
    tool,
)
from qagents.errors import (  # noqa: E402
    ConfigurationError,
    DeferToolCall,
    GuardrailTripped,
    ModelBadRequestError,
    ModelServerError,
    OutputValidationError,
)
from qagents.integrations.mlflow import (  # noqa: E402
    TOOL_FAILED,
    UNFINISHED,
    MLflowHooks,
    chat_messages,
    enable,
)
from qagents.messages import ImagePart, ReasoningPart, TextPart, ToolResultPart  # noqa: E402
from qagents.models.base import ModelRequest, ModelResponse  # noqa: E402
from qagents.models.testing import ScriptedModel, call  # noqa: E402
from qagents.tools import RunContext  # noqa: E402

# -- an isolated tracking store per test ------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _broken_pandas_counts_as_missing():
    """MLflow's tracking store imports pandas when it is installed. Where pandas is
    installed but its native libraries cannot load (a Windows application-control
    policy, for one), the import fails differently each time and breaks span export.
    Make it fail cleanly instead, as if pandas were not installed."""
    try:
        import pandas  # noqa: F401
    except Exception:
        saved = {
            name: sys.modules.pop(name)
            for name in list(sys.modules)
            if name.split(".")[0] in ("pandas", "pyarrow")
        }
        sys.modules["pandas"] = None  # type: ignore[assignment]
        yield
        del sys.modules["pandas"]
        sys.modules.update(saved)
    else:
        yield


@pytest.fixture(scope="module")
def database(tmp_path_factory: pytest.TempPathFactory):
    """One SQLite store for the module, since creating its tables takes seconds.

    Traces are exported synchronously here, which is much faster to read back;
    `test_traces_exported_in_the_background_are_flushed` covers the default. The
    tracking URI, the environment and MLflow's tracer provider are restored when
    the module's tests are done.
    """
    uri = f"sqlite:///{(tmp_path_factory.mktemp('mlflow') / 'mlflow.db').as_posix()}"
    previous_uri = tracking_utils._tracking_uri
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("MLFLOW_TRACKING_URI", raising=False)
        patch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")
        mlflow.set_tracking_uri(uri)
        yield uri
        mlflow.flush_trace_async_logging()
        mlflow.set_tracking_uri(previous_uri)  # also resets MLflow's tracer provider


@pytest.fixture
def experiment(database: str, tmp_path: Any, monkeypatch: pytest.MonkeyPatch):
    """A fresh experiment, active through MLFLOW_EXPERIMENT_ID for this test only."""
    previous_experiment = fluent._active_experiment_id
    experiment_id = mlflow.create_experiment(
        f"qagents-{uuid.uuid4().hex}", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", experiment_id)
    yield experiment_id
    mlflow.flush_trace_async_logging()
    fluent._active_experiment_id = previous_experiment
    assert mlflow.get_tracking_uri() == database  # a test that changed it must restore it


def read_trace(hooks: MLflowHooks, trace_id: str | None = None) -> Any:
    hooks.flush()
    trace = mlflow.get_trace(trace_id or hooks.last_trace_id)
    assert trace is not None
    return trace


def by_name(trace: Any) -> dict[str, list[Any]]:
    spans: dict[str, list[Any]] = {}
    for span in trace.data.spans:
        spans.setdefault(span.name, []).append(span)
    return spans


@tool(read_only=True)
def lookup(key: str) -> str:
    """Look something up."""
    if key == "missing":
        raise ToolError("No such key: missing.")
    return f"value of {key}"


USAGE = Usage(requests=1, input_tokens=20, output_tokens=5, cached_input_tokens=8, cost_usd=0.001)


# -- one run --------------------------------------------------------------------------


async def test_a_run_is_one_trace_with_typed_spans_usage_and_session(experiment):
    hooks = MLflowHooks(user_id="user-7", tags={"team": "support"})
    model = ScriptedModel(
        [[call("lookup", key="a"), call("lookup", key="missing")], "Done."], usage=USAGE
    )
    agent = Agent(model, name="helper", tools=[lookup], hooks=[hooks])
    result = await agent.run("go")
    trace = read_trace(hooks)

    spans = by_name(trace)
    assert sorted(spans) == ["chat scripted", "invoke_agent helper", "lookup"]
    (root,) = spans["invoke_agent helper"]
    assert root.parent_id is None and root.span_type == SpanType.AGENT
    children = spans["chat scripted"] + spans["lookup"]
    assert len(children) == 4 and all(span.parent_id == root.span_id for span in children)
    assert {span.span_type for span in spans["chat scripted"]} == {SpanType.CHAT_MODEL}
    assert {span.span_type for span in spans["lookup"]} == {SpanType.TOOL}

    # Run span: identity, outcome and the run's usage.
    attributes = root.attributes
    assert attributes["qagents.run_id"] == result.run_id
    assert attributes["qagents.session_id"] == result.session_id
    assert attributes["qagents.agent"] == "helper"
    assert attributes["qagents.depth"] == 0
    assert attributes["qagents.status"] == "completed"
    assert attributes["qagents.turns"] == 2
    assert attributes["qagents.usage"]["input_tokens"] == 40
    assert attributes["qagents.cost_usd"] == pytest.approx(0.002)
    assert root.status.status_code == SpanStatusCode.OK

    # Model spans: MLflow's usage, model and cost keys, and qagents' finish reason.
    first = spans["chat scripted"][0].attributes
    assert first[SpanAttributeKey.CHAT_USAGE] == {
        "input_tokens": 20, "output_tokens": 5, "total_tokens": 25, "cache_read_input_tokens": 8,
    }  # fmt: skip
    assert first[SpanAttributeKey.MODEL] == "scripted"
    assert first[SpanAttributeKey.MODEL_PROVIDER] == "test"
    assert first[SpanAttributeKey.LLM_COST] == {"total_cost": 0.001}
    assert {s.attributes["qagents.finish_reason"] for s in spans["chat scripted"]} == {
        "tool_calls", "stop",
    }  # fmt: skip

    # Tool spans: the failed lookup has an error status, without its text.
    statuses = sorted(
        (span.status.status_code, span.status.description) for span in spans["lookup"]
    )
    assert statuses == [(SpanStatusCode.ERROR, TOOL_FAILED), (SpanStatusCode.OK, "")]

    # The trace: summed usage for the UI, session, user, tags and the outcome.
    assert trace.info.token_usage == {
        "input_tokens": 40, "output_tokens": 10, "total_tokens": 50, "cache_read_input_tokens": 16,
    }  # fmt: skip
    assert trace.info.trace_metadata[TraceMetadataKey.TRACE_SESSION] == result.session_id
    assert trace.info.trace_metadata[TraceMetadataKey.TRACE_USER] == "user-7"
    assert trace.info.trace_metadata["qagents.run_id"] == result.run_id
    assert trace.info.tags["team"] == "support"
    assert trace.info.tags["qagents.status"] == "completed"
    assert trace.info.state.value == "OK"
    assert trace.info.experiment_id == experiment

    # No content unless asked.
    for span in trace.data.spans:
        assert span.inputs is None and span.outputs is None
        assert SpanAttributeKey.CHAT_TOOLS not in span.attributes
        assert SpanAttributeKey.MESSAGE_FORMAT not in span.attributes


async def test_record_content_writes_the_conversation_in_the_chat_format(experiment):
    hooks = MLflowHooks(record_content=True)
    model = ScriptedModel([call("lookup", key="missing"), "Not found."])
    await Agent(model, instructions="Be brief.", tools=[lookup], hooks=[hooks]).run("find it")
    spans = by_name(read_trace(hooks))

    first, second = spans["chat scripted"]
    assert first.attributes[SpanAttributeKey.MESSAGE_FORMAT] == "openai"
    assert first.inputs["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "find it"},
    ]
    (definition,) = first.attributes[SpanAttributeKey.CHAT_TOOLS]
    assert definition["type"] == "function"
    assert definition["function"]["name"] == "lookup"
    assert definition["function"]["parameters"]["properties"]["key"]["type"] == "string"
    (choice,) = first.outputs["choices"]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "lookup"
    # The second request carries the tool call and its result, as MLflow's ChatMessage reads them.
    sent = [ChatMessage.model_validate(message) for message in second.inputs["messages"]]
    assert [message.role for message in sent] == ["system", "user", "assistant", "tool"]
    assert sent[3].tool_call_id == sent[2].tool_calls[0].id
    assert sent[3].content == "No such key: missing."
    assert second.outputs["choices"][0]["message"] == {"role": "assistant", "content": "Not found."}

    (lookup_span,) = spans["lookup"]
    assert lookup_span.inputs == {"key": "missing"}
    assert lookup_span.outputs == "No such key: missing."
    assert lookup_span.status.description == "No such key: missing."

    (root,) = spans["invoke_agent agent"]
    assert root.inputs == {"messages": [{"role": "user", "content": "find it"}]}
    assert root.outputs["choices"][0]["message"]["content"] == "Not found."


# -- several agents ---------------------------------------------------------------------


async def test_a_sub_agent_run_nests_under_the_tool_span_that_started_it(experiment):
    hooks = MLflowHooks()
    worker = Agent(
        ScriptedModel([call("lookup", key="a"), "Found a."], usage=USAGE),
        name="worker",
        tools=[lookup],
    )
    lead = Agent(
        ScriptedModel([call("worker", task="find a"), "All done."], usage=USAGE),
        name="lead",
        tools=[worker],
        hooks=[hooks],  # only on the lead: it propagates
    )
    result = await lead.run("go")
    trace = read_trace(hooks)
    assert len(mlflow.search_traces(locations=[experiment], return_type="list")) == 1

    spans = by_name(trace)
    (lead_span,) = spans["invoke_agent lead"]
    (delegation,) = spans["worker"]
    (worker_span,) = spans["invoke_agent worker"]
    assert delegation.parent_id == lead_span.span_id and delegation.span_type == SpanType.TOOL
    assert worker_span.parent_id == delegation.span_id
    assert worker_span.attributes["qagents.parent_run_id"] == result.run_id
    assert (
        worker_span.attributes["qagents.parent_tool_call_id"]
        == (delegation.attributes["qagents.tool_call_id"])
    )
    assert worker_span.attributes["qagents.depth"] == 1
    (worker_lookup,) = spans["lookup"]
    assert worker_lookup.parent_id == worker_span.span_id
    assert len(spans["chat scripted"]) == 4
    # Every model request of the tree counts once in the trace's usage.
    assert trace.info.token_usage["input_tokens"] == 80
    assert lead_span.attributes["qagents.usage"]["input_tokens"] == 80  # includes the worker's


async def test_one_hook_on_both_agents_records_each_run_once(experiment):
    hooks = MLflowHooks()
    worker = Agent(ScriptedModel(["Found."]), name="worker", hooks=[hooks])
    lead = Agent(
        ScriptedModel([call("worker", task="t"), "Done."]),
        name="lead",
        tools=[worker],
        hooks=[hooks],
    )
    await lead.run("go")
    names = sorted(span.name for span in read_trace(hooks).data.spans)
    assert names == [
        "chat scripted", "chat scripted", "chat scripted",
        "invoke_agent lead", "invoke_agent worker", "worker",
    ]  # fmt: skip


async def test_a_run_joins_the_active_application_span(experiment):
    hooks = MLflowHooks(tags={"team": "support"})
    agent = Agent(ScriptedModel(["one", "two"]), hooks=[hooks])
    with mlflow.start_span("handle_request") as outer:
        mlflow.update_current_trace(session_id="app-session")
        await agent.run("hi")
        await agent.run("again")
    trace = read_trace(hooks, outer.trace_id)
    runs = by_name(trace)["invoke_agent agent"]
    assert [span.parent_id for span in runs] == [outer.span_id, outer.span_id]
    assert hooks.last_trace_id is None  # the trace belongs to the application
    assert trace.info.trace_metadata[TraceMetadataKey.TRACE_SESSION] == "app-session"
    assert "team" not in trace.info.tags and "qagents.status" not in trace.info.tags


async def test_experiment_id_sends_this_hook_s_traces_elsewhere(experiment, tmp_path):
    other = mlflow.create_experiment(
        f"other-{uuid.uuid4().hex}", artifact_location=(tmp_path / "other").as_uri()
    )
    hooks = MLflowHooks(experiment_id=other)
    await Agent(ScriptedModel(["hi"]), hooks=[hooks]).run("hello")
    assert read_trace(hooks).info.experiment_id == other
    assert mlflow.search_traces(locations=[experiment], return_type="list") == []


async def test_traces_exported_in_the_background_are_flushed(experiment, monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "true")  # MLflow's default
    mlflow.tracing.reset()  # the next span builds a provider with background export
    try:
        hooks = MLflowHooks()
        result = await Agent(ScriptedModel(["hi"]), hooks=[hooks]).run("hello")
        hooks.flush()
        (trace,) = mlflow.search_traces(locations=[experiment], return_type="list")
        assert trace.info.trace_metadata["qagents.run_id"] == result.run_id
        assert trace.info.trace_id == hooks.last_trace_id
    finally:
        mlflow.flush_trace_async_logging()
        monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")
        mlflow.tracing.reset()


# -- every way a run can end ---------------------------------------------------------------


async def test_a_failing_model_ends_every_span_with_an_error(experiment):
    hooks = MLflowHooks()
    failing = ScriptedModel([ModelServerError("upstream down", ambiguous=True)])
    with pytest.raises(ModelServerError):
        await Agent(failing, hooks=[hooks]).run("hello")
    trace = read_trace(hooks)
    assert trace.info.state.value == "ERROR"
    assert trace.info.tags["qagents.status"] == "failed"
    spans = by_name(trace)
    (chat,) = spans["chat scripted"]
    (root,) = spans["invoke_agent agent"]
    for span in (chat, root):
        assert span.status.status_code == SpanStatusCode.ERROR
        assert span.end_time_ns is not None
        assert [event.name for event in span.events] == ["exception"]


async def test_a_hook_that_raises_mid_tool_still_closes_the_tool_span(experiment):
    class Broken(Hooks):
        async def before_tool(self, ctx: RunContext[Any], call: Any) -> None:
            raise RuntimeError("broken hook")

    hooks = MLflowHooks()
    agent = Agent(ScriptedModel([call("lookup", key="a")]), tools=[lookup], hooks=[hooks, Broken()])
    with pytest.raises(RuntimeError, match="broken hook"):
        await agent.run("go")
    spans = by_name(read_trace(hooks))
    (tool_span,) = spans["lookup"]
    assert tool_span.status.status_code == SpanStatusCode.ERROR
    assert tool_span.status.description == UNFINISHED
    assert spans["invoke_agent agent"][0].status.status_code == SpanStatusCode.ERROR


@tool(requires_approval=True)
def send_email(to: str) -> str:
    """Send an email."""
    return f"sent to {to}"


@tool(read_only=True)
def ask_expert(question: str) -> str:
    """Ask a human expert."""
    raise DeferToolCall("waiting for the expert")


async def test_a_paused_run_ends_its_trace_as_paused_and_resume_starts_a_new_one(experiment):
    hooks = MLflowHooks()
    session = Session()
    model = ScriptedModel(
        [[call("send_email", to="a@example.com"), call("ask_expert", question="q")], "Sent."]
    )
    agent = Agent(model, tools=[send_email, ask_expert], hooks=[hooks])
    paused = await agent.run("mail and ask", session=session)
    assert paused.status == "paused"
    trace = read_trace(hooks)
    assert trace.info.tags["qagents.status"] == "paused"
    assert trace.info.state.value == "OK"
    spans = by_name(trace)
    (root,) = spans["invoke_agent agent"]
    assert root.attributes["qagents.pending_approvals"] == [
        {"tool_call_id": paused.pending_approvals[0].tool_call_id, "name": "send_email"}
    ]
    assert root.attributes["qagents.pending_inputs"][0]["name"] == "ask_expert"
    assert sorted(event.name for event in root.events) == ["approval_required", "input_required"]
    (deferred,) = spans["ask_expert"]  # it ran and deferred; it is not an error
    assert deferred.status.status_code == SpanStatusCode.OK
    assert deferred.attributes["qagents.tool.status"] == "deferred"
    assert "send_email" not in spans  # waiting for approval, never started
    first_trace = hooks.last_trace_id

    resumed = await agent.resume(
        session,
        approvals={paused.pending_approvals[0].tool_call_id: True},
        results={paused.pending_inputs[0].tool_call_id: "expert says yes"},
    )
    assert resumed.status == "completed"
    second = read_trace(hooks)
    assert hooks.last_trace_id != first_trace
    assert second.info.tags["qagents.status"] == "completed"
    assert second.info.trace_metadata[TraceMetadataKey.TRACE_SESSION] == session.id
    (root,) = by_name(second)["invoke_agent agent"]
    assert root.attributes["qagents.resumed"] is True
    assert "send_email" in by_name(second)


async def test_limits_cancellation_and_guardrails_end_the_trace_cleanly(experiment):
    hooks = MLflowHooks()
    looping = ScriptedModel([call("lookup", key="a")])
    limited = await Agent(looping, tools=[lookup], limits=Limits(max_turns=1), hooks=[hooks]).run(
        "go"
    )
    assert limited.status == "limit_reached"
    trace = read_trace(hooks)
    (root,) = by_name(trace)["invoke_agent agent"]
    assert root.status.status_code == SpanStatusCode.ERROR
    assert root.status.description == "limit_reached: max_turns"
    assert trace.info.tags["qagents.stop_reason"] == "max_turns"

    slow = ScriptedModel(["word " * 200], delay=0.01)
    stream = Agent(slow, hooks=[hooks]).stream("talk")
    async for event in stream:
        if event.type == "text_delta":
            stream.cancel()
    assert stream.result is not None and stream.result.status == "cancelled"
    trace = read_trace(hooks)
    assert trace.info.tags["qagents.status"] == "cancelled"
    assert all(span.end_time_ns is not None for span in trace.data.spans)
    (chat,) = by_name(trace)["chat scripted"]
    assert chat.status.description == UNFINISHED

    def no_secrets(ctx: RunContext[Any], prompt: Message) -> None:
        if "password" in prompt.text:
            raise GuardrailTripped("the prompt contains a password: hunter2")

    blocked = await Agent(ScriptedModel([]), input_guardrails=[no_secrets], hooks=[hooks]).run(
        "my password is hunter2"
    )
    assert blocked.status == "blocked"
    trace = read_trace(hooks)
    (root,) = by_name(trace)["invoke_agent agent"]
    assert trace.info.tags["qagents.status"] == "blocked"
    assert root.attributes["qagents.guardrail"] == {"name": "no_secrets", "stage": "input"}
    (event,) = root.events
    assert event.name == "guardrail_tripped" and "reason" not in event.attributes


async def test_child_runs_whose_end_was_never_reported_are_closed_with_the_parent(experiment):
    hooks = MLflowHooks()
    parent = RunContext(run_id="parent", session_id="s", agent_name="lead")
    await hooks.on_run_start(parent)
    await hooks.before_tool(parent, call("worker", task="t").model_copy(update={"id": "c1"}))
    child = RunContext(
        run_id="child", session_id="s2", agent_name="worker",
        parent_run_id="parent", parent_tool_call_id="c1", depth=1,
    )  # fmt: skip
    await hooks.on_run_start(child)
    await hooks.before_model(child, ModelRequest(messages=[]))
    await hooks.on_run_end(parent, None, asyncio.CancelledError())

    spans = by_name(read_trace(hooks))
    assert spans["invoke_agent worker"][0].parent_id == spans["worker"][0].span_id
    for name in ("invoke_agent worker", "chat", "worker"):
        (span,) = spans[name]
        assert span.status.description == UNFINISHED
    assert spans["invoke_agent lead"][0].status.status_code == SpanStatusCode.ERROR


# -- configuration and formats --------------------------------------------------------------


async def test_enable_configures_mlflow_only_when_asked(
    experiment, database, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)  # a new experiment keeps its artifacts under ./mlruns
    with pytest.raises(ConfigurationError, match="not both"):
        enable(experiment_name="a", experiment_id="1")
    hooks = enable()
    assert isinstance(hooks, MLflowHooks)
    assert mlflow.get_tracking_uri() == database

    name = f"enabled-{uuid.uuid4().hex}"
    hooks = enable(tracking_uri=database, experiment_name=name, record_content=True)
    await Agent(ScriptedModel(["hi"]), hooks=[hooks]).run("hello")
    trace = read_trace(hooks)
    assert trace.info.experiment_id == mlflow.get_experiment_by_name(name).experiment_id
    assert hooks.record_content  # the fixture restores the active experiment


def test_messages_are_converted_to_the_chat_format_mlflow_validates():
    messages = [
        Message.user([TextPart(text="What is this?"), ImagePart(url="https://example.com/a.png")]),
        Message.user([ImagePart(data="aGVsbG8=", media_type="image/png")]),
        Message(
            role="assistant",
            parts=[
                ReasoningPart(text="Think."),
                TextPart(text="Calling."),
                call("lookup", key="x"),
            ],
        ),
        Message(
            role="tool",
            parts=[
                ToolResultPart(tool_call_id="c1", name="a", content="one"),
                ToolResultPart(tool_call_id="c2", name="b", content="two", is_error=True),
            ],
        ),
    ]
    converted = chat_messages(messages)
    for message in converted:
        ChatMessage.model_validate(message)
    assert converted[0]["content"][1] == {
        "type": "image_url", "image_url": {"url": "https://example.com/a.png"},
    }  # fmt: skip
    assert converted[1]["content"] == [{"type": "text", "text": "[image/png, 8 base64 characters]"}]
    assert converted[2]["content"][0]["type"] == "reasoning"
    assert converted[2]["tool_calls"][0]["function"] == {
        "name": "lookup",
        "arguments": '{"key": "x"}',
    }
    assert [m["tool_call_id"] for m in converted[3:]] == ["c1", "c2"]


async def test_cache_writes_reported_by_the_provider_reach_the_usage(experiment):
    response = ModelResponse(
        message=Message(
            role="assistant",
            parts=[TextPart(text="ok")],
            metadata={"cache_creation_input_tokens": 30},
        ),
        usage=Usage(requests=1, input_tokens=50, output_tokens=2),
        model="claude-x",
        provider="anthropic",
    )
    hooks = MLflowHooks()
    await Agent(ScriptedModel([response]), hooks=[hooks]).run("hi")
    (chat,) = by_name(read_trace(hooks))["chat scripted"]
    usage = chat.attributes[SpanAttributeKey.CHAT_USAGE]
    assert usage["cache_creation_input_tokens"] == 30
    assert chat.attributes[SpanAttributeKey.MODEL] == "claude-x"


# -- no content in traces unless asked ----------------------------------------------------

SECRET = "patient John Doe, SSN 123-45-6789"


class Refund(BaseModel):
    amount: int


def failing_runs(hooks: MLflowHooks) -> list[tuple[Agent, type[Exception]]]:
    """A run that fails on the model's answer, and one whose provider error echoes the input."""
    invalid = Agent(
        ScriptedModel([f'{{"amount": "{SECRET}"}}']),
        output_type=Refund,
        output_mode="prompted",
        limits=Limits(output_retries=0),
        hooks=[hooks],
    )
    echoing = Agent(
        ScriptedModel([ModelBadRequestError(f"invalid request: {SECRET}")]), hooks=[hooks]
    )
    return [(invalid, OutputValidationError), (echoing, ModelBadRequestError)]


async def test_without_record_content_exception_text_stays_out_of_the_trace(experiment):
    hooks = MLflowHooks()
    for agent, expected in failing_runs(hooks):
        with pytest.raises(expected):
            await agent.run("refund me")
        trace = read_trace(hooks)
        assert "123-45-6789" not in trace.to_json()
        assert trace.info.tags["qagents.status"] == "failed"
        failed = [span for span in trace.data.spans if span.events]
        assert failed  # the run span, and the chat span of a failed request
        for span in failed:
            assert span.status.status_code == SpanStatusCode.ERROR
            assert span.status.description == expected.__name__
            (event,) = span.events
            assert (event.name, event.attributes) == (
                "exception",
                {"exception.type": expected.__name__},
            )


async def test_with_record_content_exception_text_is_recorded(experiment):
    hooks = MLflowHooks(record_content=True)
    for agent, expected in failing_runs(hooks):
        with pytest.raises(expected):
            await agent.run("refund me")
        (root,) = by_name(read_trace(hooks))["invoke_agent agent"]
        assert "123-45-6789" in root.events[0].attributes["exception.message"]


async def test_a_text_answer_given_with_the_final_answer_tool_is_the_run_output(experiment):
    hooks = MLflowHooks(record_content=True)
    model = ScriptedModel([call("final_answer", value="It is sunny in Athens.")])
    result = await Agent(model, output_mode="tool", hooks=[hooks]).run("Weather?")
    assert result.output == "It is sunny in Athens." and result.text == ""
    (root,) = by_name(read_trace(hooks))["invoke_agent agent"]
    assert root.outputs["choices"][0]["message"]["content"] == "It is sunny in Athens."


@pytest.mark.parametrize("record_content", [False, True])
async def test_a_summary_request_is_traced_and_counted(experiment, record_content):
    hooks = MLflowHooks(record_content=record_content)
    summarizer = ScriptedModel(["Earlier: two questions."], name="summarizer", usage=USAGE)
    agent = Agent(
        ScriptedModel(["Third answer."], usage=USAGE),
        compactor=SummarizingCompactor(trigger_tokens=0, keep_last=1, model=summarizer),
        hooks=[hooks],
    )
    history = [
        Message.user("q1"),
        Message.assistant("a1"),
        Message.user("q2"),
        Message.assistant("a2"),
    ]
    await agent.run("q3", history=history)
    trace = read_trace(hooks)
    chats = [span for span in trace.data.spans if span.span_type == SpanType.CHAT_MODEL]
    assert sorted(span.attributes[SpanAttributeKey.MODEL] for span in chats) == [
        "scripted", "summarizer",
    ]  # fmt: skip
    # Each span is named after the model the request went to, not the agent's model.
    assert [span.name for span in chats] == ["chat summarizer", "chat scripted"]
    if record_content:
        assert [span.inputs["model"] for span in chats] == ["summarizer", "scripted"]
    assert trace.info.token_usage["input_tokens"] == 40  # both requests


def test_data_url_images_are_replaced_by_a_placeholder():
    payload = "A" * 5_000
    converted = chat_messages(
        [
            Message.user(
                [
                    TextPart(text="What is this?"),
                    ImagePart(url=f"data:image/png;base64,{payload}"),
                    ImagePart(url="DATA:image/svg+xml,<svg></svg>"),
                ]
            )
        ]
    )
    assert converted[0]["content"][1:] == [
        {"type": "text", "text": "[image/png, 5000 base64 characters]"},
        {"type": "text", "text": "[image/svg+xml, 11 characters]"},
    ]
    assert payload not in json.dumps(converted)
