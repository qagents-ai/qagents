"""The AG-UI projection of the event stream, driven by scripted models."""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import AsyncIterator

import pytest
from pydantic import BaseModel

from qagents import Agent, Hooks, Message, RunContext, Session, prompts, tool
from qagents.errors import ConfigurationError, ModelServerError, SessionError, ToolError
from qagents.integrations.agui import (
    APPROVAL_RESPONSE_SCHEMA,
    CUSTOM_EVENTS,
    RUN_ERROR_MESSAGE,
    STREAM_ENDED_MESSAGE,
    AGUIProjector,
    agui_resume,
    agui_sse,
    encode_sse,
    to_agui_events,
    to_agui_messages,
)
from qagents.messages import ImagePart, ReasoningPart, TextPart, ToolCallPart, Usage
from qagents.models.base import (
    Model,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ReasoningDelta,
    ResponseDone,
    TextDelta,
    ToolCallDelta,
)
from qagents.models.testing import ScriptedModel, call
from qagents.permissions import Approval


@tool(read_only=True)
def get_weather(city: str) -> str:
    """Look up the weather for a city."""
    return f"Sunny in {city}"


@tool(requires_approval=True)
def refund(order_id: str) -> str:
    """Refund an order."""
    return f"Refunded {order_id}"


class StreamingModel(Model):
    """Plays the given stream events, one list per request; an exception item is raised."""

    provider = "test"

    def __init__(self, *responses: list) -> None:
        self.name = "streaming"
        self._responses = list(responses)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        for item in self._responses.pop(0):
            if isinstance(item, BaseException):
                raise item
            yield item


def done(message: Message) -> ResponseDone:
    finish = "tool_calls" if message.tool_calls else "stop"
    return ResponseDone(response=ModelResponse(message=message, finish_reason=finish))


async def collect(stream, **options) -> list[dict]:
    events = [event async for event in to_agui_events(stream, **options)]
    check_protocol(events)
    return events


def check_protocol(events: list[dict]) -> None:
    """The ordering rules strict AG-UI clients enforce, per subagent run."""
    assert events, "no events"
    assert events[0]["type"] in ("RUN_STARTED", "RUN_ERROR")
    assert events[-1]["type"] in ("RUN_FINISHED", "RUN_ERROR")
    assert [e["type"] for e in events].count("RUN_STARTED") <= 1
    open_text: dict[str | None, str] = {}
    open_reasoning: dict[str | None, str] = {}
    open_call: dict[str | None, str] = {}
    steps: dict[str | None, list[str]] = {}
    started_calls: set[str] = set()
    message_ids: set[str] = set()
    for position, event in enumerate(events):
        kind, scope = event["type"], event.get("subagentRunId")
        if kind in ("RUN_FINISHED", "RUN_ERROR"):
            assert position == len(events) - 1, f"{kind} is not last"
        busy = scope in open_text or scope in open_reasoning or scope in open_call
        if kind == "TEXT_MESSAGE_START":
            assert not busy, f"interleaved at {position}: {event}"
            assert event["messageId"] not in message_ids, "message id reused"
            message_ids.add(event["messageId"])
            open_text[scope] = event["messageId"]
        elif kind == "TEXT_MESSAGE_CONTENT":
            assert open_text.get(scope) == event["messageId"] and event["delta"]
        elif kind == "TEXT_MESSAGE_END":
            assert open_text.pop(scope) == event["messageId"]
        elif kind == "REASONING_START":
            assert not busy
            open_reasoning[scope] = event["messageId"]
        elif kind in ("REASONING_MESSAGE_START", "REASONING_MESSAGE_CONTENT"):
            assert open_reasoning.get(scope) == event["messageId"]
        elif kind == "REASONING_MESSAGE_END":
            assert open_reasoning.get(scope) == event["messageId"]
        elif kind == "REASONING_END":
            assert open_reasoning.pop(scope) == event["messageId"]
        elif kind == "TOOL_CALL_START":
            assert not busy, f"interleaved at {position}: {event}"
            assert event["toolCallId"] not in started_calls, "tool call started twice"
            started_calls.add(event["toolCallId"])
            open_call[scope] = event["toolCallId"]
        elif kind == "TOOL_CALL_ARGS":
            assert open_call.get(scope) == event["toolCallId"]
        elif kind == "TOOL_CALL_END":
            assert open_call.pop(scope) == event["toolCallId"]
        elif kind == "TOOL_CALL_RESULT":
            assert not busy
            assert event["messageId"] not in message_ids, "message id reused"
            message_ids.add(event["messageId"])
        elif kind == "STEP_STARTED":
            assert not steps.get(scope), "steps overlap"
            steps.setdefault(scope, []).append(event["stepName"])
        elif kind == "STEP_FINISHED":
            assert steps.get(scope) == [event["stepName"]]
            steps[scope] = []
        elif kind in ("RUN_FINISHED", "SUBAGENT_FINISHED", "SUBAGENT_ERROR"):
            assert scope not in open_text and scope not in open_call and not steps.get(scope)
    assert not open_text and not open_reasoning and not open_call


# --- the tool loop ------------------------------------------------------------------------


async def test_tool_loop_exact_sequence():
    weather = call("get_weather", city="Athens")
    agent = Agent(ScriptedModel([weather, "It is sunny."]), tools=[get_weather])

    events = await collect(agent.stream("Weather in Athens?"), thread_id="t-1", run_id="r-1")

    assert events == [
        {"type": "RUN_STARTED", "threadId": "t-1", "runId": "r-1"},
        {"type": "STEP_STARTED", "stepName": "agent: turn 1"},
        {
            "type": "TOOL_CALL_START",
            "toolCallId": weather.id,
            "toolCallName": "get_weather",
            "parentMessageId": "r-1-msg-1",
        },
        {"type": "TOOL_CALL_ARGS", "toolCallId": weather.id, "delta": '{"city": "Athens"}'},
        {"type": "TOOL_CALL_END", "toolCallId": weather.id},
        {
            "type": "TOOL_CALL_RESULT",
            "messageId": "r-1-result-2",
            "toolCallId": weather.id,
            "content": "Sunny in Athens",
            "role": "tool",
        },
        {"type": "STEP_FINISHED", "stepName": "agent: turn 1"},
        {"type": "STEP_STARTED", "stepName": "agent: turn 2"},
        {"type": "TEXT_MESSAGE_START", "messageId": "r-1-msg-3", "role": "assistant"},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": "r-1-msg-3", "delta": "It "},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": "r-1-msg-3", "delta": "is "},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": "r-1-msg-3", "delta": "sunny."},
        {"type": "TEXT_MESSAGE_END", "messageId": "r-1-msg-3"},
        {"type": "STEP_FINISHED", "stepName": "agent: turn 2"},
        {
            "type": "RUN_FINISHED",
            "threadId": "t-1",
            "runId": "r-1",
            "result": {"status": "completed", "stop_reason": None, "output": "It is sunny."},
        },
    ]


async def test_ids_default_to_the_session_and_run():
    agent = Agent(ScriptedModel(["Hello."]))
    session = Session(id="thread-42")
    stream = agent.stream("Hi", session=session)

    events = await collect(stream)

    assert events[0] == {"type": "RUN_STARTED", "threadId": "thread-42", "runId": stream.run_id}
    assert events[-1]["threadId"] == "thread-42" and events[-1]["runId"] == stream.run_id
    text_ids = {e["messageId"] for e in events if e["type"].startswith("TEXT_MESSAGE")}
    assert text_ids == {f"{stream.run_id}-msg-1"}


async def test_text_and_tool_calls_of_one_response_share_the_parent_message():
    weather = call("get_weather", city="Rome")
    reply = Message.assistant("Let me check.", tool_calls=[weather])
    agent = Agent(ScriptedModel([reply, "Sunny."]), tools=[get_weather])

    events = await collect(agent.stream("Rome?"), run_id="r")

    start = next(e for e in events if e["type"] == "TEXT_MESSAGE_START")
    call_start = next(e for e in events if e["type"] == "TOOL_CALL_START")
    assert call_start["parentMessageId"] == start["messageId"]
    kinds = [e["type"] for e in events]
    assert kinds.index("TEXT_MESSAGE_END") < kinds.index("TOOL_CALL_START")


async def test_streamed_tool_call_arguments_are_forwarded_as_they_arrive():
    final = ToolCallPart(id="call_1", name="get_weather", arguments={"city": "Oslo"})
    final.raw_arguments = '{"city": "Oslo"}'
    model = StreamingModel(
        [
            ToolCallDelta(index=0, id="call_1", name="get_weather", arguments_delta=""),
            ToolCallDelta(index=0, id="call_1", name="get_weather", arguments_delta='{"city": '),
            ToolCallDelta(index=0, id="call_1", name="get_weather", arguments_delta='"Oslo"}'),
            done(Message(role="assistant", parts=[final])),
        ],
        [TextDelta(text="Cold."), done(Message.assistant("Cold."))],
    )
    agent = Agent(model, tools=[get_weather])

    events = await collect(agent.stream("Oslo?"), run_id="r")

    calls = [e for e in events if e["type"].startswith("TOOL_CALL")]
    assert calls == [
        {
            "type": "TOOL_CALL_START",
            "toolCallId": "call_1",
            "toolCallName": "get_weather",
            "parentMessageId": "r-msg-1",
        },
        {"type": "TOOL_CALL_ARGS", "toolCallId": "call_1", "delta": '{"city": '},
        {"type": "TOOL_CALL_ARGS", "toolCallId": "call_1", "delta": '"Oslo"}'},
        {"type": "TOOL_CALL_END", "toolCallId": "call_1"},
        {
            "type": "TOOL_CALL_RESULT",
            "messageId": "r-result-2",
            "toolCallId": "call_1",
            "content": "Sunny in Oslo",
            "role": "tool",
        },
    ]


async def test_tool_calls_streamed_without_ids_are_announced_with_the_final_id():
    first = ToolCallPart(id="call_a", name="get_weather", arguments={"city": "Rome"})
    second = ToolCallPart(id="call_b", name="get_weather", arguments={"city": "Nice"})
    model = StreamingModel(
        [
            ToolCallDelta(index=0, name="get_weather", arguments_delta='{"city": "Rome"}'),
            ToolCallDelta(index=1, name="get_weather", arguments_delta='{"city": "Nice"}'),
            done(Message(role="assistant", parts=[first, second])),
        ],
        [done(Message.assistant("Both sunny."))],
    )
    agent = Agent(model, tools=[get_weather])

    events = await collect(agent.stream("Weather?"), run_id="r")

    starts = [e["toolCallId"] for e in events if e["type"] == "TOOL_CALL_START"]
    results = [e["toolCallId"] for e in events if e["type"] == "TOOL_CALL_RESULT"]
    assert starts == ["call_a", "call_b"] and sorted(results) == ["call_a", "call_b"]
    # A response that was not streamed as text is sent as one whole message.
    text = [e for e in events if e["type"] == "TEXT_MESSAGE_CONTENT"]
    assert [e["delta"] for e in text] == ["Both sunny."]


async def test_reasoning_is_forwarded_and_can_be_left_out():
    def model() -> StreamingModel:
        answer = Message(
            role="assistant", parts=[ReasoningPart(text="Think."), TextPart(text="Done.")]
        )
        return StreamingModel(
            [
                ReasoningDelta(text="Thi"),
                ReasoningDelta(text="nk."),
                TextDelta(text="Done."),
                done(answer),
            ]
        )

    events = await collect(Agent(model()).stream("Go"), run_id="r", steps=False)
    assert [(e["type"], e.get("delta")) for e in events[1:-1]] == [
        ("REASONING_START", None),
        ("REASONING_MESSAGE_START", None),
        ("REASONING_MESSAGE_CONTENT", "Thi"),
        ("REASONING_MESSAGE_CONTENT", "nk."),
        ("REASONING_MESSAGE_END", None),
        ("REASONING_END", None),
        ("TEXT_MESSAGE_START", None),
        ("TEXT_MESSAGE_CONTENT", "Done."),
        ("TEXT_MESSAGE_END", None),
    ]
    assert events[2] == {
        "type": "REASONING_MESSAGE_START",
        "messageId": "r-reasoning-1",
        "role": "reasoning",
    }

    hidden = await collect(Agent(model()).stream("Go"), reasoning=False, steps=False)
    assert not [e for e in hidden if e["type"].startswith("REASONING")]


async def test_reasoning_that_was_not_streamed_is_sent_whole():
    answer = Message(
        role="assistant", parts=[ReasoningPart(text="Because."), TextPart(text="Yes.")]
    )
    events = await collect(Agent(StreamingModel([done(answer)])).stream("Why?"), steps=False)

    assert [e["type"] for e in events[1:-1]] == [
        "REASONING_START", "REASONING_MESSAGE_START", "REASONING_MESSAGE_CONTENT",
        "REASONING_MESSAGE_END", "REASONING_END",
        "TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT", "TEXT_MESSAGE_END",
    ]  # fmt: skip


# --- errors ---------------------------------------------------------------------------


async def test_a_model_error_ends_with_run_error(caplog):
    model = StreamingModel([TextDelta(text="Partial "), ModelServerError("upstream failed")])
    with caplog.at_level(logging.ERROR, logger="qagents.agui"):
        events = await collect(Agent(model).stream("Hi"), run_id="r")

    assert [e["type"] for e in events] == [
        "RUN_STARTED",
        "STEP_STARTED",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "STEP_FINISHED",
        "RUN_ERROR",
    ]
    assert events[-1] == {
        "type": "RUN_ERROR",
        "message": "upstream failed",
        "code": "ModelServerError",
    }
    assert "RUN_ERROR" in caplog.text


async def test_other_exceptions_are_not_shown_to_the_client_unless_asked():
    class Broken(Hooks):
        async def before_model(self, ctx, request):
            raise RuntimeError("database password is hunter2")

    def stream():
        return Agent(ScriptedModel(["never"]), hooks=[Broken()]).stream("Hi")

    hidden = await collect(stream())
    assert hidden[-1] == {"type": "RUN_ERROR", "message": RUN_ERROR_MESSAGE, "code": "RuntimeError"}

    shown = await collect(stream(), expose_errors=True)
    assert shown[-1]["message"] == "database password is hunter2"


async def test_an_error_before_the_run_starts_is_the_only_event():
    agent = Agent(ScriptedModel([]))
    events = await collect(agent.resume_stream(Session()))

    assert len(events) == 1 and events[0]["type"] == "RUN_ERROR"
    assert events[0]["code"] == SessionError.__name__
    assert "no unfinished run" in events[0]["message"]


async def test_a_stream_that_stops_early_ends_with_run_error():
    async def truncated():
        yield {"type": "run_started", "run_id": "r", "session_id": "t"}
        yield {"type": "text_delta", "text": "Hel"}

    events = await collect(truncated())

    assert [e["type"] for e in events] == [
        "RUN_STARTED", "TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT", "TEXT_MESSAGE_END", "RUN_ERROR"
    ]  # fmt: skip
    assert events[-1] == {
        "type": "RUN_ERROR",
        "message": STREAM_ENDED_MESSAGE,
        "code": "stream_ended",
    }


async def test_closing_the_projection_stops_the_run():
    model = ScriptedModel(["Hello."])
    stream = Agent(model).stream("Hi")
    events = to_agui_events(stream)

    assert (await events.__anext__())["type"] == "RUN_STARTED"
    await events.aclose()

    assert model.requests == []  # the run never got to the model
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()


def test_options_are_checked_when_called():
    with pytest.raises(TypeError):
        to_agui_events(Agent(ScriptedModel([])).stream("Hi"), colour="blue")
    with pytest.raises(ConfigurationError, match="'custom', 'native' or 'ignore'"):
        AGUIProjector(subagents="nested")  # type: ignore[arg-type]


# --- pauses, state and custom events -----------------------------------------------------


async def test_a_paused_run_ends_with_interrupts_and_resumes_from_the_client_answer():
    refund_call = call("refund", order_id="A-1")
    agent = Agent(ScriptedModel([refund_call, "Refunded."]), tools=[refund])
    session = Session()

    events = await collect(agent.stream("Refund A-1", session=session), run_id="r1")

    custom = [e for e in events if e["type"] == "CUSTOM"]
    assert [e["name"] for e in custom] == ["approval_required"]
    assert custom[0]["value"]["approval"]["tool_call_id"] == refund_call.id
    assert custom[0]["value"]["agent"] == "agent"
    assert not [e for e in events if e["type"] == "TOOL_CALL_RESULT"]
    finished = events[-1]
    assert finished["result"]["status"] == "paused"
    assert finished["outcome"] == {
        "type": "interrupt",
        "interrupts": [
            {
                "id": f"approval:{refund_call.id}",
                "reason": "confirmation",
                "message": "this tool requires approval.",
                "toolCallId": refund_call.id,
                "responseSchema": APPROVAL_RESPONSE_SCHEMA,
                "metadata": {"name": "refund", "arguments": {"order_id": "A-1"}},
            }
        ],
    }

    answer = [
        {
            "interruptId": f"approval:{refund_call.id}",
            "status": "resolved",
            "payload": {"approved": True},
        }
    ]
    resumed = await collect(agent.resume_stream(session, **agui_resume(answer)), run_id="r2")

    assert resumed[0] == {"type": "RUN_STARTED", "threadId": session.id, "runId": "r2"}
    result = next(e for e in resumed if e["type"] == "TOOL_CALL_RESULT")
    assert result["toolCallId"] == refund_call.id and result["content"] == "Refunded A-1"
    assert resumed[-1]["result"]["status"] == "completed" and "outcome" not in resumed[-1]


async def test_a_resumed_run_without_a_client_run_id_gets_a_new_one():
    refund_call = call("refund", order_id="A-2")
    agent = Agent(ScriptedModel([refund_call, "Done."]), tools=[refund])
    session = Session()
    first = await collect(agent.stream("Refund A-2", session=session))
    second = await collect(agent.resume_stream(session, approvals={refund_call.id: False}))

    assert second[0]["runId"] != first[0]["runId"]
    assert second[0]["runId"].startswith(f"{first[0]['runId']}-")
    assert second[0]["threadId"] == first[0]["threadId"] == session.id


@pytest.mark.skipif(
    "deferred" not in inspect.signature(tool).parameters, reason="needs deferred tools"
)
async def test_deferred_tool_calls_become_input_interrupts():
    @tool(deferred=True)
    def pick_city(question: str) -> str:
        """Ask the user to pick a city."""

    question = call("pick_city", question="Which city?")
    agent = Agent(ScriptedModel([question, "Athens it is."]), tools=[pick_city])
    session = Session()

    events = await collect(agent.stream("Plan a trip", session=session))

    assert [e["name"] for e in events if e["type"] == "CUSTOM"] == ["input_required"]
    (interrupt,) = events[-1]["outcome"]["interrupts"]
    assert interrupt["id"] == f"input:{question.id}" and interrupt["reason"] == "input_required"
    assert interrupt["toolCallId"] == question.id
    assert interrupt["metadata"] == {"name": "pick_city", "arguments": {"question": "Which city?"}}

    answer = [{"interruptId": interrupt["id"], "status": "resolved", "payload": "Athens"}]
    resumed = await collect(agent.resume_stream(session, **agui_resume(answer)))

    result = next(e for e in resumed if e["type"] == "TOOL_CALL_RESULT")
    assert result["toolCallId"] == question.id and result["content"] == "Athens"
    assert resumed[-1]["result"]["output"] == "Athens it is."


def test_state_updates_become_snapshots_and_custom_events():
    projector = AGUIProjector(thread_id="t", run_id="r")
    projector.feed({"type": "run_started", "run_id": "r", "session_id": "t"})
    projector.feed({"type": "text_delta", "text": "Working"})

    events = projector.feed({"type": "state_updated", "values": {"todos": [1, 2]}, "agent": "a"})

    assert events == [
        {"type": "TEXT_MESSAGE_END", "messageId": "r-msg-1"},
        {"type": "STATE_SNAPSHOT", "snapshot": {"todos": [1, 2]}},
        {
            "type": "CUSTOM",
            "name": "state_updated",
            "value": {"values": {"todos": [1, 2]}, "agent": "a"},
        },
    ]
    quiet = AGUIProjector(state_snapshots=False, custom_events=())
    assert quiet.feed({"type": "state_updated", "values": {"a": 1}}) == []


@pytest.mark.skipif(
    "state" not in inspect.signature(Agent.run).parameters, reason="needs session state"
)
async def test_state_changed_by_a_tool_is_snapshotted():
    @tool
    def remember(ctx: RunContext, city: str) -> str:
        """Remember the city."""
        ctx.state["city"] = city
        return "ok"

    agent = Agent(ScriptedModel([call("remember", city="Athens"), "Noted."]), tools=[remember])
    events = await collect(agent.stream("Remember Athens"))

    snapshots = [e["snapshot"] for e in events if e["type"] == "STATE_SNAPSHOT"]
    assert snapshots and snapshots[-1] == {"city": "Athens"}


async def test_unknown_events_are_skipped_and_chosen_ones_forwarded():
    class Progress(BaseModel):
        type: str = "progress"
        percent: int

    @tool(read_only=True)
    def crunch(ctx: RunContext) -> str:
        """Crunch numbers."""
        ctx.emit(Progress(percent=50))
        return "42"

    def stream():
        return Agent(ScriptedModel([call("crunch"), "42."]), tools=[crunch]).stream("Go")

    plain = await collect(stream())
    assert not [e for e in plain if e["type"] == "CUSTOM"]

    forwarded = await collect(stream(), custom_events=CUSTOM_EVENTS | {"progress"})
    assert [e for e in forwarded if e["type"] == "CUSTOM"] == [
        {"type": "CUSTOM", "name": "progress", "value": {"percent": 50}}
    ]
    assert AGUIProjector().feed({"type": "event_from_the_future", "x": 1}) == []


# --- several agents -----------------------------------------------------------------------


def subagent_setup() -> tuple[Agent, ToolCallPart]:
    researcher = Agent(
        ScriptedModel([call("get_weather", city="Athens"), "Athens is sunny."]),
        name="researcher",
        description="Finds facts.",
        tools=[get_weather],
    )
    delegate = call("researcher", task="Find the weather in Athens")
    lead = Agent(ScriptedModel([delegate, "It is sunny."]), name="lead", tools=[researcher])
    return lead, delegate


async def test_subagent_events_are_custom_events_by_default():
    lead, delegate = subagent_setup()
    events = await collect(lead.stream("Weather?"))

    inner = [e for e in events if e["type"] == "CUSTOM" and e["name"] == "subagent_event"]
    assert inner and all(e["value"]["agent"] == "researcher" for e in inner)
    assert all(e["value"]["tool_call_id"] == delegate.id for e in inner)
    assert "text_delta" in [e["value"]["event"]["type"] for e in inner]
    assert not [e for e in events if e["type"].startswith("SUBAGENT")]

    ignored = await collect(subagent_setup()[0].stream("Weather?"), subagents="ignore")
    assert not [e for e in ignored if e["type"] == "CUSTOM"]


async def test_subagents_can_use_the_native_events():
    lead, delegate = subagent_setup()
    events = await collect(lead.stream("Weather?"), run_id="r", subagents="native")

    started = next(e for e in events if e["type"] == "SUBAGENT_STARTED")
    child = started["subagentRunId"]
    assert started["name"] == "researcher" and started["parentToolCallId"] == delegate.id
    tagged = [e for e in events if e.get("subagentRunId") == child]
    kinds = [e["type"] for e in tagged]
    assert {"TOOL_CALL_START", "TOOL_CALL_RESULT", "TEXT_MESSAGE_CONTENT"} <= set(kinds)
    assert kinds[0] == "SUBAGENT_STARTED" and kinds[-1] == "SUBAGENT_FINISHED"
    assert tagged[-1]["result"] == {"status": "completed", "output": "Athens is sunny."}
    # The subagent finishes before the parent's tool call gets its result.
    order = [e["type"] for e in events]
    parent_result = next(
        i
        for i, e in enumerate(events)
        if e["type"] == "TOOL_CALL_RESULT" and e["toolCallId"] == delegate.id
    )
    assert order.index("SUBAGENT_FINISHED") < parent_result
    assert "subagentRunId" not in events[parent_result]


async def test_a_handoff_call_gets_its_result():
    billing = Agent(ScriptedModel(["Refund sent."]), name="billing", description="Refunds.")
    triage = Agent(ScriptedModel([call("transfer_to_billing")]), name="triage", handoffs=[billing])

    events = await collect(triage.stream("I want a refund"))

    start = next(e for e in events if e["type"] == "TOOL_CALL_START")
    result = next(e for e in events if e["type"] == "TOOL_CALL_RESULT")
    assert start["toolCallName"] == "transfer_to_billing"
    assert result["toolCallId"] == start["toolCallId"]
    assert result["content"] == prompts.HANDOFF_RESULT.format(name="billing")
    switched = next(e for e in events if e["type"] == "CUSTOM")
    assert switched["name"] == "agent_switched"
    assert switched["value"] == {"from_agent": "triage", "to_agent": "billing"}
    assert [e["stepName"] for e in events if e["type"] == "STEP_STARTED"] == [
        "triage: turn 1",
        "billing: turn 2",
    ]


# --- encoding, messages and resume -------------------------------------------------------


async def test_sse_encoding():
    agent = Agent(ScriptedModel(["Γεια σου."]))
    chunks = [chunk async for chunk in agui_sse(agent.stream("Hi"), thread_id="t", run_id="r")]

    assert all(chunk.startswith("data: ") and chunk.endswith("\n\n") for chunk in chunks)
    decoded = [json.loads(chunk.removeprefix("data: ")) for chunk in chunks]
    check_protocol(decoded)
    assert decoded[0] == {"type": "RUN_STARTED", "threadId": "t", "runId": "r"}
    assert "Γεια" in chunks[3]  # not escaped
    assert encode_sse({"type": "CUSTOM", "name": "x", "value": 1}) == (
        'data: {"type":"CUSTOM","name":"x","value":1}\n\n'
    )


async def test_messages_snapshot_before_run_finished():
    weather = call("get_weather", city="Athens")
    agent = Agent(ScriptedModel([weather, "Sunny."]), tools=[get_weather])
    events = await collect(agent.stream("Weather?"), thread_id="t", messages_snapshot=True)

    assert events[-2]["type"] == "MESSAGES_SNAPSHOT"
    assert events[-2]["messages"] == [
        {"id": "t-msg-0", "role": "user", "content": "Weather?"},
        {
            "id": "t-msg-1",
            "role": "assistant",
            "name": "agent",
            "toolCalls": [
                {
                    "id": weather.id,
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Athens"}'},
                }
            ],
        },
        {"id": "t-msg-2-0", "role": "tool", "content": "Sunny in Athens", "toolCallId": weather.id},
        {"id": "t-msg-3", "role": "assistant", "content": "Sunny.", "name": "agent"},
    ]


def test_to_agui_messages_converts_images_and_system_text():
    messages = [
        Message.system("Be brief."),
        Message.user(
            [TextPart(text="What is this?"), ImagePart(url="https://example.com/cat.png")]
        ),
        Message.user([ImagePart(data="aGk=", media_type="image/jpeg")]),
        Message(role="assistant", parts=[ReasoningPart(text="hidden")]),
    ]

    assert to_agui_messages(messages, id_prefix="m") == [
        {"id": "m-0", "role": "system", "content": "Be brief."},
        {
            "id": "m-1",
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "binary", "mimeType": "image/png", "url": "https://example.com/cat.png"},
            ],
        },
        {
            "id": "m-2",
            "role": "user",
            "content": [{"type": "binary", "mimeType": "image/jpeg", "data": "aGk="}],
        },
    ]


def test_agui_resume_reads_approvals_and_results():
    resumed = agui_resume(
        [
            {
                "interruptId": "approval:c1",
                "status": "resolved",
                "payload": {"approved": True, "arguments": {"x": 1}},
            },
            {"interruptId": "approval:c2", "status": "resolved", "payload": False},
            {"interruptId": "approval:c3", "status": "cancelled"},
            {"interruptId": "input:c4", "status": "resolved", "payload": {"city": "Athens"}},
            {"interruptId": "input:c5", "status": "cancelled"},
        ]
    )

    assert resumed["approvals"] == {
        "c1": Approval(approved=True, arguments={"x": 1}),
        "c2": Approval(approved=False),
        "c3": Approval(approved=False),
    }
    assert resumed["results"]["c4"] == {"city": "Athens"}
    cancelled = resumed["results"]["c5"]
    assert isinstance(cancelled, ToolError) and str(cancelled) == prompts.INPUT_CANCELLED


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"interruptId": "other:1", "status": "resolved"}, "not issued by qagents"),
        ({"interruptId": "approval:1", "status": "maybe"}, "'resolved' or 'cancelled'"),
        (
            {"interruptId": "approval:1", "status": "resolved", "payload": "yes"},
            "must be true, false",
        ),
        (
            {"interruptId": "approval:1", "status": "resolved", "payload": {"ok": 1}},
            "not an approval",
        ),
    ],
)
def test_agui_resume_rejects_what_it_cannot_use(entry, message):
    with pytest.raises(ValueError, match=message):
        agui_resume([entry])


def test_usage_is_not_sent_to_the_client():
    projector = AGUIProjector(run_id="r", thread_id="t")
    result = {"status": "completed", "output": "x", "usage": Usage(cost_usd=1.0)}
    (finished,) = projector.feed({"type": "run_finished", "result": result})

    assert finished["result"] == {"status": "completed", "stop_reason": None, "output": "x"}
    assert projector.finished and projector.feed({"type": "text_delta", "text": "late"}) == []
