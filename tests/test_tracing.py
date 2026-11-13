from __future__ import annotations

import asyncio
import json
import time

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402
from opentelemetry.trace import StatusCode  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from qagents import (  # noqa: E402
    Agent,
    Approval,
    Hooks,
    Limits,
    Message,
    Permissions,
    SummarizingCompactor,
    ToolError,
    Usage,
    tool,
)
from qagents.errors import (  # noqa: E402
    DeferToolCall,
    ModelServerError,
    OutputValidationError,
)
from qagents.events import ExternalAgentEvent, SubagentEvent, ToolCallFinished  # noqa: E402
from qagents.external import (  # noqa: E402
    EndStep,
    EventStep,
    PermissionStep,
    ScriptedExternalAgent,
    ToolStep,
)
from qagents.integrations.acp.mapping import STOP_REASONS as ACP_STOP_REASONS  # noqa: E402
from qagents.integrations.devvi.events import TURN_END_KINDS  # noqa: E402
from qagents.messages import ToolCallPart, ToolResultPart  # noqa: E402
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


# -- an external agent's permission decisions and the end of its turn ------------------------


def external_events(span) -> list[tuple[str, dict]]:
    return [
        (event.name.removeprefix("qagents.external."), dict(event.attributes))
        for event in span.events
    ]


def delegation_span(exporter):
    return next(s for s in exporter.get_finished_spans() if s.name == "execute_tool coder")


def lead_of(coder: ScriptedExternalAgent, hooks: OTelHooks, **options) -> Agent:
    model = ScriptedModel([call("coder", task="Fix it"), "Done."])
    return Agent(model, name="lead", tools=[coder.as_tool()], hooks=[hooks], **options)


async def test_an_external_turn_records_who_decided_what_and_how_it_ended(spans):
    exporter, tracer = spans
    coder = ScriptedExternalAgent(
        [
            [
                "Running the tests.",
                ToolStep("bash", {"command": "pytest -q"}, ask=True, result="1 passed"),
                PermissionStep("rm", {"path": "build"}),
                Usage(requests=2, input_tokens=120, output_tokens=40, cost_usd=0.01),
                "All tests pass.",
            ]
        ],
        name="coder",
    )
    rules = Permissions(allow=["coder.bash"], deny=["coder.rm"])
    result = await lead_of(coder, OTelHooks(tracer), permissions=rules).run("go")
    assert result.status == "completed"

    finished = {span.name: span for span in exporter.get_finished_spans()}
    delegation = finished["execute_tool coder"]
    assert delegation.status.status_code == StatusCode.UNSET
    assert external_events(delegation) == [
        (
            "permission_decided",
            {
                "gen_ai.agent.name": "coder",
                "qagents.external.permission.tool": "coder.bash",
                "qagents.external.permission.tool_call_id": "coder-1-call-1",
                "qagents.external.permission.outcome": "allow",
                "qagents.external.permission.approved": True,
                "qagents.external.permission.decision": "allow",
                "qagents.external.permission.decided_by": "rules",
            },
        ),
        (
            "permission_decided",
            {
                "gen_ai.agent.name": "coder",
                "qagents.external.permission.tool": "coder.rm",
                "qagents.external.permission.outcome": "deny",
                "qagents.external.permission.approved": False,
                "qagents.external.permission.decision": "deny",
                "qagents.external.permission.decided_by": "rules",
            },
        ),
        (
            "turn_finished",
            {
                "gen_ai.agent.name": "coder",
                "qagents.external.status": "completed",
                "qagents.external.stop_reason": "end_turn",
                "qagents.external.session_id": "coder-1",
                "qagents.external.backend": "scripted",
                "gen_ai.usage.input_tokens": 120,
                "gen_ai.usage.output_tokens": 40,
                "qagents.cost_usd": 0.01,
                "qagents.external.tool_calls": 1,
            },
        ),
    ]
    # The loop hands the turn's events to the hook after the tool returned; they still land
    # on its span, which ends at the time the tool returned, with every event inside it.
    assert all(event.timestamp <= delegation.end_time for event in delegation.events)
    assert delegation.end_time <= finished["invoke_agent lead"].end_time
    assert finished["invoke_agent lead"].events == ()  # on the tool's span only


async def test_without_record_content_no_argument_or_reason_text_reaches_the_spans(spans):
    exporter, tracer = spans

    def refuse(ctx, tool, call, reason):
        return Approval(approved=False, reason=f"not on {SECRET}'s record")

    coder = ScriptedExternalAgent(
        [
            [
                ToolStep("bash", {"command": f"grep '{SECRET}' notes.txt"}, ask=True),
                PermissionStep("write", {"path": "notes.txt", "text": SECRET}, reason=SECRET),
                EndStep(status="error", stop_reason=f"crashed on {SECRET}"),
            ]
        ],
        name="coder",
    )
    rules = Permissions(ask=["coder.*"], on_ask=refuse)
    await lead_of(coder, OTelHooks(tracer), permissions=rules).run("go")
    assert [answer.reason for answer in coder.answers] == [f"not on {SECRET}'s record"] * 2

    events = external_events(delegation_span(exporter))
    assert [name for name, _ in events] == [
        "permission_decided", "permission_decided", "turn_finished",
    ]  # fmt: skip
    for _, attributes in events[:2]:
        assert attributes["qagents.external.permission.outcome"] == "deny"
        assert attributes["qagents.external.permission.decided_by"] == "on_ask"
        assert "qagents.external.permission.arguments" not in attributes
        assert "qagents.external.permission.reason" not in attributes
    turn = events[2][1]
    assert turn["qagents.external.status"] == "error"
    assert "qagents.external.stop_reason" not in turn  # an error turn's reason is its message
    assert leaks(exporter) == []


async def test_with_record_content_the_arguments_and_reasons_are_recorded(spans):
    exporter, tracer = spans
    coder = ScriptedExternalAgent(
        [
            [
                PermissionStep("write", {"path": "notes.txt", "text": SECRET}),
                EndStep(status="error", stop_reason=f"crashed on {SECRET}"),
            ]
        ],
        name="coder",
    )
    rules = Permissions(deny=["coder.write"])
    await lead_of(coder, OTelHooks(tracer, record_content=True), permissions=rules).run("go")

    (_, decided), (_, turn) = external_events(delegation_span(exporter))
    arguments = json.loads(decided["qagents.external.permission.arguments"])
    assert arguments == {"path": "notes.txt", "text": SECRET}
    assert decided["qagents.external.permission.reason"]  # the deny rule's reason
    assert turn["qagents.external.stop_reason"] == f"crashed on {SECRET}"


async def test_a_paused_external_turn_records_the_deferred_decision(spans):
    exporter, tracer = spans
    coder = ScriptedExternalAgent(
        [[ToolStep("bash", {"command": "rm -rf build"}, ask=True)]],
        name="coder",
        on_permission="pause",
    )
    result = await lead_of(coder, OTelHooks(tracer)).run("go")
    assert result.status == "paused"

    delegation = delegation_span(exporter)
    assert delegation.attributes["qagents.tool.deferred"] is True
    (_, decided), (_, turn) = external_events(delegation)
    assert decided["qagents.external.permission.decided_by"] == "deferred"
    assert decided["qagents.external.permission.outcome"] == "cancel"
    assert turn["qagents.external.status"] == "cancelled"
    assert turn["qagents.external.stop_reason"] == "permission"


async def test_an_external_agent_in_a_sub_agent_records_its_events_once(spans):
    exporter, tracer = spans
    coder = ScriptedExternalAgent(
        [[PermissionStep("rm", {"path": "build"}), "Done."]], name="coder"
    )
    worker = Agent(
        ScriptedModel([call("coder", task="Clean"), "Cleaned."]),
        name="worker",
        tools=[coder.as_tool()],
        permissions=Permissions(allow=["coder.rm"]),
    )
    lead = Agent(
        ScriptedModel([call("worker", task="Clean up"), "Done."]),
        name="lead",
        tools=[worker],
        hooks=[OTelHooks(tracer)],
    )
    await lead.run("go")

    finished = {span.name: span for span in exporter.get_finished_spans()}
    delegation = finished["execute_tool coder"]
    assert delegation.parent.span_id == finished["invoke_agent worker"].context.span_id
    names = [name for name, _ in external_events(delegation)]
    assert names == ["permission_decided", "turn_finished"]
    others = [span for span in exporter.get_finished_spans() if span is not delegation]
    assert all(not span.events for span in others)  # not repeated on the outer spans


async def test_a_tool_span_left_finishing_ends_with_the_run_without_an_error(spans):
    exporter, tracer = spans
    hooks = OTelHooks(tracer)
    ctx = RunContext(run_id="r", session_id="s", agent_name="lead")
    call_part = ToolCallPart(id="c1", name="lookup")
    await hooks.on_run_start(ctx)
    await hooks.before_tool(ctx, call_part)
    returned = ToolResultPart(tool_call_id="c1", name="lookup", content="value")
    await hooks.after_tool(ctx, call_part, returned)
    assert exporter.get_finished_spans() == ()  # waits for its tool_call_finished
    await hooks.on_run_end(ctx, None, None)
    span = next(s for s in exporter.get_finished_spans() if s.name == "execute_tool lookup")
    assert span.status.status_code == StatusCode.UNSET


async def drive_external_call(hooks: OTelHooks, data: dict, *, pause: float = 0.0):
    """Run one tool call ``coder`` by hand whose turn sends ``turn_finished`` with ``data``
    after the tool returned, as the loop does. Returns the times around ``after_tool``."""
    ctx = RunContext(run_id="r", session_id="s", agent_name="lead")
    call_part = ToolCallPart(id="c1", name="coder")
    result = ToolResultPart(tool_call_id="c1", name="coder", content="Done.")
    await hooks.on_run_start(ctx)
    await hooks.before_tool(ctx, call_part)
    before = time.time_ns()
    await hooks.after_tool(ctx, call_part, result)
    after = time.time_ns()
    await asyncio.sleep(pause)
    inner = ExternalAgentEvent(agent="coder", kind="turn_finished", data=data)
    await hooks.on_event(ctx, SubagentEvent(agent="coder", tool_call_id="c1", event=inner))
    await hooks.on_event(ctx, ToolCallFinished(agent="lead", call=call_part, result=result))
    await hooks.on_run_end(ctx, None, None)
    return before, after


@pytest.mark.parametrize("record_content", [False, True])
async def test_the_changed_paths_are_content_and_their_count_is_not(spans, record_content):
    exporter, tracer = spans
    path = f"notes/{SECRET}.txt"
    data = {"status": "completed", "changed_files": [path], "changed_files_count": 1}
    await drive_external_call(OTelHooks(tracer, record_content=record_content), data)

    [(_, turn)] = external_events(delegation_span(exporter))
    assert turn["qagents.external.changed_files_count"] == 1
    if record_content:
        assert turn["qagents.external.changed_files"] == (path,)
    else:
        assert "qagents.external.changed_files" not in turn
        assert leaks(exporter) == []


async def test_a_tool_span_ends_at_the_time_its_tool_returned(spans):
    exporter, tracer = spans
    before, after = await drive_external_call(OTelHooks(tracer), {}, pause=0.1)

    delegation = delegation_span(exporter)
    # Not at the time of its tool_call_finished event, a tenth of a second later.
    assert before <= delegation.end_time <= after
    [event] = delegation.events  # arrived after the tool returned: stamped with that time
    assert event.timestamp == delegation.end_time


ADAPTER_STOP_WORDS = sorted(
    {(status, word) for status, word in [*ACP_STOP_REASONS.values(), *TURN_END_KINDS.values()]}
)


@pytest.mark.parametrize(("status", "word"), ADAPTER_STOP_WORDS)
async def test_every_stop_word_an_adapter_reports_is_recorded_without_content(spans, status, word):
    exporter, tracer = spans
    coder = ScriptedExternalAgent([[EndStep(status=status, stop_reason=word)]], name="coder")
    await lead_of(coder, OTelHooks(tracer)).run("go")

    [(_, turn)] = external_events(delegation_span(exporter))
    assert turn["qagents.external.status"] == status
    assert turn["qagents.external.stop_reason"] == word


MALFORMED = {
    "cost overflows a float": {"usage": {"cost_usd": 10**400}},
    "usage is text": {"usage": "n/a"},
    "tool calls are a list": {"tool_calls": ["bash"]},
    "a count is text": {"changed_files_count": "many"},
    "tokens and cost are text": {"usage": {"input_tokens": "lots", "cost_usd": "free"}},
    "tokens are not a number": {"usage": {"input_tokens": float("nan"), "output_tokens": True}},
    "a call count is text": {"tool_calls": {"bash": "twice"}},
    "the stop reason is a mapping": {"stop_reason": {"text": "no"}},
    "the changed paths are text": {"changed_files": "a.txt"},
}


class Undecidable:
    """A value with no truth value, as a numpy array has."""

    def __bool__(self) -> bool:
        raise ValueError("the truth value is ambiguous")


@pytest.mark.parametrize(
    "kind, field",
    [
        ("permission_decided", "tool"),
        ("permission_decided", "tool_call_id"),
        ("permission_decided", "outcome"),
        ("permission_decided", "decision"),
        ("permission_decided", "rule"),
        ("permission_decided", "arguments"),
        ("permission_decided", "reason"),
        ("turn_finished", "status"),
        ("turn_finished", "session_id"),
        ("turn_finished", "backend"),
    ],
)
async def test_malformed_external_fields_are_not_coerced_or_tested_for_truth(spans, kind, field):
    exporter, tracer = spans
    coder = ScriptedExternalAgent(
        [[EventStep(kind, {field: Undecidable()}), "Done."]], name="coder"
    )
    result = await lead_of(coder, OTelHooks(tracer, record_content=True)).run("go")
    assert result.status == "completed"
    first = external_events(delegation_span(exporter))[0][1]
    assert not any("Undecidable" in str(value) for value in first.values())


@pytest.mark.parametrize(
    "kind,field",
    [
        ("permission_decided", "tool"),
        ("permission_decided", "tool_call_id"),
        ("permission_decided", "outcome"),
        ("permission_decided", "decision"),
        ("permission_decided", "rule"),
        ("permission_decided", "arguments"),
        ("permission_decided", "reason"),
        ("turn_finished", "status"),
        ("turn_finished", "session_id"),
        ("turn_finished", "backend"),
    ],
)
async def test_non_string_external_fields_are_ignored_without_coercion(spans, kind, field):
    exporter, tracer = spans
    ctx = RunContext(run_id="r", session_id="s", agent_name="lead")
    hooks = OTelHooks(tracer, record_content=True)
    await hooks.on_run_start(ctx)
    try:
        inner = ExternalAgentEvent(agent="coder", kind=kind, data={field: Undecidable()})
        await hooks.on_event(ctx, SubagentEvent(agent="coder", event=inner))
    finally:
        await hooks.on_run_end(ctx, None, None)
    [root] = exporter.get_finished_spans()
    assert len(root.events) == 1
    assert "Undecidable" not in str(dict(root.events[0].attributes))


@pytest.mark.parametrize("data", MALFORMED.values(), ids=MALFORMED.keys())
async def test_a_custom_adapters_malformed_turn_data_never_fails_the_run(spans, data):
    exporter, tracer = spans
    decided = {"tool": "coder.x", "approved": Undecidable(), "arguments": {("a", 1): 1}}

    def coder() -> ScriptedExternalAgent:
        steps = [
            EventStep("permission_decided", decided),
            EventStep("turn_finished", data),
            "Done.",
        ]
        return ScriptedExternalAgent([steps], name="coder")

    assert (await lead_of(coder(), Hooks()).run("go")).status == "completed"  # untraced
    result = await lead_of(coder(), OTelHooks(tracer, record_content=True)).run("go")
    assert result.status == "completed"

    events = external_events(delegation_span(exporter))
    assert [name for name, _ in events] == [
        "permission_decided", "turn_finished", "turn_finished",
    ]  # fmt: skip
    assert "qagents.external.permission.arguments" not in events[0][1]
    assert events[0][1]["qagents.external.permission.approved"] is False  # only True is True
    turn = events[1][1]  # the malformed one: the fields of the wrong type are left out
    assert turn["gen_ai.usage.input_tokens"] == 0
    assert turn["gen_ai.usage.output_tokens"] == 0
    assert turn["qagents.external.tool_calls"] == 0
    for name in ("qagents.cost_usd", "qagents.external.stop_reason"):
        assert name not in turn
    for name in ("qagents.external.changed_files", "qagents.external.changed_files_count"):
        assert name not in turn
