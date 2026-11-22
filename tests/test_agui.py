"""The AG-UI projection of the event stream, driven by scripted models."""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import AsyncIterator

import pytest
from pydantic import BaseModel

from qagents import Agent, Hooks, Message, RunContext, Session, prompts, tool
from qagents.errors import (
    ApprovalTokenError,
    ConfigurationError,
    GuardrailTripped,
    ModelConnectionError,
    ModelServerError,
    SessionError,
    ToolError,
)
from qagents.external import EndStep, ScriptedExternalAgent, ToolStep
from qagents.integrations.agui import (
    APPROVAL_RESPONSE_SCHEMA,
    CUSTOM_EVENTS,
    RUN_ERROR_MESSAGE,
    STREAM_ENDED_MESSAGE,
    TOOL_ERROR_MESSAGE,
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
from qagents.permissions import Approval, Permissions


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


async def collect(stream, earlier: list[dict] = (), **options) -> list[dict]:
    events = [event async for event in to_agui_events(stream, **options)]
    check_protocol(events, earlier=earlier)
    return events


def check_protocol(events: list[dict], *, earlier: list[dict] = ()) -> None:
    """The ordering rules strict AG-UI clients enforce, per subagent run.

    Every tool call result has a start: in ``events``, or in ``earlier``, the events
    of the run that paused, for the calls a resumed run goes on with.
    """
    assert events, "no events"
    assert events[0]["type"] in ("RUN_STARTED", "RUN_ERROR")
    assert events[-1]["type"] in ("RUN_FINISHED", "RUN_ERROR")
    assert [e["type"] for e in events].count("RUN_STARTED") <= 1
    open_text: dict[str | None, str] = {}
    open_reasoning: dict[str | None, str] = {}
    open_call: dict[str | None, str] = {}
    steps: dict[str | None, list[str]] = {}
    started_calls: set[str] = set()
    known_calls = {e["toolCallId"] for e in earlier if e["type"] == "TOOL_CALL_START"}
    ended_calls: set[str] = set()
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
            assert event["toolCallId"] not in known_calls, "the paused run started it"
            started_calls.add(event["toolCallId"])
            open_call[scope] = event["toolCallId"]
        elif kind == "TOOL_CALL_ARGS":
            assert open_call.get(scope) == event["toolCallId"]
        elif kind == "TOOL_CALL_END":
            assert open_call.pop(scope) == event["toolCallId"]
            ended_calls.add(event["toolCallId"])
        elif kind == "TOOL_CALL_RESULT":
            assert not busy
            call_id = event["toolCallId"]
            assert call_id in ended_calls or call_id in known_calls, f"no start for {call_id}"
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


async def test_interleaved_tool_call_deltas_reach_the_client_whole():
    first = ToolCallPart(id="c0", name="get_weather", arguments={"city": "a"})
    second = ToolCallPart(id="c1", name="get_weather", arguments={"city": "b"})
    model = StreamingModel(
        [
            ToolCallDelta(index=0, id="c0", name="get_weather", arguments_delta='{"ci'),
            ToolCallDelta(index=1, id="c1", name="get_weather", arguments_delta='{"city"'),
            ToolCallDelta(index=0, id="c0", name="get_weather", arguments_delta='ty": "a"}'),
            ToolCallDelta(index=1, id="c1", name="get_weather", arguments_delta=': "b"}'),
            done(Message(role="assistant", parts=[first, second])),
        ],
        [done(Message.assistant("Both sunny."))],
    )

    events = await collect(Agent(model, tools=[get_weather]).stream("Weather?"))

    sent: dict[str, str] = {}
    for event in events:
        if event["type"] == "TOOL_CALL_ARGS":
            sent[event["toolCallId"]] = sent.get(event["toolCallId"], "") + event["delta"]
    assert {key: json.loads(value) for key, value in sent.items()} == {
        "c0": {"city": "a"},
        "c1": {"city": "b"},
    }


async def test_a_streamed_call_id_the_provider_repeats_is_started_once():
    # An OpenAI-compatible server gave two calls one id; the response renames the second.
    first = ToolCallPart(id="call_0", name="get_weather", arguments={"city": "Rome"})
    second = ToolCallPart(id="call_x", name="get_weather", arguments={"city": "Nice"})
    model = StreamingModel(
        [
            ToolCallDelta(
                index=0, id="call_0", name="get_weather", arguments_delta='{"city": "Rome"}'
            ),
            ToolCallDelta(
                index=1, id="call_0", name="get_weather", arguments_delta='{"city": "Nice"}'
            ),
            done(Message(role="assistant", parts=[first, second])),
        ],
        [done(Message.assistant("Both sunny."))],
    )

    events = await collect(Agent(model, tools=[get_weather]).stream("Weather?"))

    starts = [e["toolCallId"] for e in events if e["type"] == "TOOL_CALL_START"]
    results = [e["toolCallId"] for e in events if e["type"] == "TOOL_CALL_RESULT"]
    assert starts == ["call_0", "call_x"] and sorted(results) == ["call_0", "call_x"]
    nice = [e["delta"] for e in events if e.get("toolCallId") == "call_x" and "delta" in e]
    assert nice == ['{"city": "Nice"}']


async def test_events_fed_as_dicts_are_projected_like_the_event_models():
    weather = call("get_weather", city="Athens")
    agent = Agent(ScriptedModel([weather, "It is sunny."]), tools=[get_weather])
    events = [event async for event in agent.stream("Weather?")]

    def project(items: list) -> list[dict]:
        projector = AGUIProjector(thread_id="t", run_id="r", messages_snapshot=True)
        return [out for item in items for out in projector.feed(item)]

    as_models = project(events)
    as_dicts = project([event.model_dump(mode="json") for event in events])

    assert as_dicts == as_models
    kinds = [event["type"] for event in as_dicts]
    assert {"TOOL_CALL_START", "TEXT_MESSAGE_CONTENT", "MESSAGES_SNAPSHOT"} <= set(kinds)


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
        "message": "The model request failed.",
        "code": "ModelServerError",
    }
    assert "RUN_ERROR" in caplog.text and "upstream failed" in caplog.text


async def test_qagents_error_text_is_not_shown_to_the_client_unless_asked():
    def stream():
        url = "http://svc:Pa55w0rd@10.0.0.7:9/internal-vllm/v1"
        failure = ModelConnectionError(f"Could not connect to {url}: refused", provider="test")
        return Agent(StreamingModel([failure])).stream("Hi")

    hidden = await collect(stream())
    assert hidden[-1] == {
        "type": "RUN_ERROR",
        "message": "The model provider could not be reached.",
        "code": "ModelConnectionError",
    }
    assert "Pa55w0rd" not in json.dumps(hidden)

    shown = await collect(stream(), expose_errors=True)
    assert "internal-vllm" in shown[-1]["message"]


async def test_a_sub_agent_error_does_not_show_the_error_text_either():
    projector = AGUIProjector(run_id="r", subagents="native")
    projector.feed({"type": "run_started", "run_id": "r", "session_id": "t"})
    projector.feed({"type": "subagent_event", "agent": "a", "tool_call_id": "c", "event": {}})

    events = projector.fail(SessionError("database /srv/secret/app.db is locked"))

    assert [e["type"] for e in events] == ["SUBAGENT_ERROR", "RUN_ERROR"]
    assert "/srv/secret" not in json.dumps(events)
    assert events[-1]["code"] == "SessionError"


@tool
def lookup(key: str) -> str:
    """Look up a key."""
    raise RuntimeError("cannot open /srv/secret/keys.db")


async def test_the_text_of_an_exception_a_tool_raised_is_not_shown_unless_asked():
    url = "http://svc:Pa55w0rd@10.0.0.7:9/v1"

    def lead() -> Agent:
        # An agent used as a tool whose model cannot be reached: the tool raises.
        failure = ModelConnectionError(f"Could not connect to {url}: refused", provider="test")
        inner = Agent(StreamingModel([failure]), name="inner", description="Inner.")
        return Agent(ScriptedModel([call("inner", task="t"), "Done."]), name="lead", tools=[inner])

    for mode in ("custom", "native"):
        events = await collect(lead().stream("Go"), subagents=mode, messages_snapshot=True)
        assert "Pa55w0rd" not in json.dumps(events)
        result = next(e for e in events if e["type"] == "TOOL_CALL_RESULT")
        assert result["content"] == TOOL_ERROR_MESSAGE.format(name="inner")
    error = next(e for e in events if e["type"] == "SUBAGENT_ERROR")
    assert error["message"] == TOOL_ERROR_MESSAGE.format(name="inner")

    # A tool of a sub-agent raises: its result is in the sub-agent's events.
    researcher = Agent(
        ScriptedModel([call("lookup", key="k"), "No luck."]),
        name="researcher",
        description="Finds facts.",
        tools=[lookup],
    )
    nested = Agent(ScriptedModel([call("researcher", task="t"), "Done."]), tools=[researcher])
    events = await collect(nested.stream("Go"))
    assert "/srv/secret" not in json.dumps(events)
    assert TOOL_ERROR_MESSAGE.format(name="lookup") in json.dumps(events)

    shown = await collect(lead().stream("Go"), expose_errors=True)
    assert "Pa55w0rd" in next(e for e in shown if e["type"] == "TOOL_CALL_RESULT")["content"]


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
    assert events[0]["message"] == "The conversation could not be loaded, resumed or saved."
    shown = await collect(agent.resume_stream(Session()), expose_errors=True)
    assert "no unfinished run" in shown[0]["message"]


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
    resumed = await collect(
        agent.resume_stream(session, **agui_resume(answer)), earlier=events, run_id="r2"
    )

    assert resumed[0] == {"type": "RUN_STARTED", "threadId": session.id, "runId": "r2"}
    result = next(e for e in resumed if e["type"] == "TOOL_CALL_RESULT")
    assert result["toolCallId"] == refund_call.id and result["content"] == "Refunded A-1"
    # The paused run announced the call; the resumed run sends only its result.
    assert [e for e in resumed if e["type"].startswith("TOOL_CALL_")] == [result]
    assert resumed[-1]["result"]["status"] == "completed" and "outcome" not in resumed[-1]


async def test_a_resumed_run_without_a_client_run_id_gets_a_new_one():
    refund_call = call("refund", order_id="A-2")
    agent = Agent(ScriptedModel([refund_call, "Done."]), tools=[refund])
    session = Session()
    first = await collect(agent.stream("Refund A-2", session=session))
    second = await collect(
        agent.resume_stream(session, approvals={refund_call.id: False}), earlier=first
    )

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
    resumed = await collect(agent.resume_stream(session, **agui_resume(answer)), earlier=events)

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


async def test_subagent_events_carry_no_usage_or_replay_data_and_no_reasoning_unless_asked():
    def lead() -> Agent:
        answer = Message(
            role="assistant",
            parts=[
                ReasoningPart(text="SECRET CHAIN", opaque={"signature": "SIGNED-BLOB"}),
                TextPart(text="Athens is sunny."),
            ],
        )
        billed = Usage(requests=1, input_tokens=7, output_tokens=3, cost_usd=0.5)
        reply = ResponseDone(response=ModelResponse(message=answer, usage=billed))
        researcher = Agent(
            StreamingModel([ReasoningDelta(text="SECRET CHAIN"), TextDelta(text="Athens."), reply]),
            name="researcher",
            description="Finds facts.",
        )
        delegate = call("researcher", task="weather")
        return Agent(ScriptedModel([delegate, "Sunny."]), name="lead", tools=[researcher])

    hidden = await collect(lead().stream("Weather?"), reasoning=False)
    inner = [e["value"]["event"]["type"] for e in hidden if e["type"] == "CUSTOM"]
    assert "text_delta" in inner and "reasoning_delta" not in inner
    assert "SECRET" not in json.dumps(hidden)

    shown = await collect(lead().stream("Weather?"))
    assert "SECRET CHAIN" in json.dumps(shown)
    for text in ("SIGNED-BLOB", "input_tokens", "cost_usd"):
        assert text not in json.dumps(shown), text


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
    assert switched["value"]["from_agent"] == "triage"
    assert switched["value"]["to_agent"] == "billing"
    assert [e["stepName"] for e in events if e["type"] == "STEP_STARTED"] == [
        "triage: turn 1",
        "billing: turn 2",
    ]


def test_a_handoff_call_the_event_names_gets_its_result_in_a_resumed_run():
    # A resumed projection never saw the response with the handoff call.
    projector = AGUIProjector(thread_id="t", run_id="r2", steps=False, custom_events=())
    projector.feed({"type": "run_started", "run_id": "r", "session_id": "t", "resumed": True})

    events = projector.feed(
        {
            "type": "agent_switched",
            "from_agent": "triage",
            "to_agent": "billing",
            "tool_call_id": "call_handoff",
        }
    )

    assert events == [
        {
            "type": "TOOL_CALL_RESULT",
            "messageId": "r2-result-1",
            "toolCallId": "call_handoff",
            "content": prompts.HANDOFF_RESULT.format(name="billing"),
            "role": "tool",
        }
    ]


async def test_a_handoff_call_recorded_after_a_pause_gets_its_result():
    # The turn holds a call that needs approval and a handoff. The handoff is
    # made by the resumed run, whose projection never saw that response.
    billing = Agent(ScriptedModel(["Billing here."]), name="billing", description="Billing.")
    triage = Agent(
        ScriptedModel([[call("refund", order_id="A"), call("transfer_to_billing")]]),
        name="triage",
        tools=[refund],
        handoffs=[billing],
    )
    session = Session()
    first = await collect(triage.stream("Refund A", session=session))
    transfer = next(
        e["toolCallId"]
        for e in first
        if e["type"] == "TOOL_CALL_START" and e["toolCallName"] == "transfer_to_billing"
    )
    interrupts = first[-1]["outcome"]["interrupts"]
    answers = [{"interruptId": i["id"], "status": "resolved", "payload": True} for i in interrupts]

    second = await collect(triage.resume_stream(session, **agui_resume(answers)), earlier=first)

    results = [e for e in second if e["type"] == "TOOL_CALL_RESULT"]
    assert [e["toolCallId"] for e in results].count(transfer) == 1
    handoff = next(e for e in results if e["toolCallId"] == transfer)
    assert handoff["content"] == prompts.HANDOFF_RESULT.format(name="billing")


async def test_an_answer_blocked_by_an_output_guardrail_is_taken_back():
    def no_secrets(ctx, answer):
        if "hunter2" in str(answer):
            raise GuardrailTripped("the answer contains the password hunter2")

    def stream():
        model = ScriptedModel(["The admin password is hunter2."])
        return Agent(model, output_guardrails=[no_secrets]).stream("pw?")

    events = await collect(stream(), thread_id="t")

    tripped = next(e for e in events if e["type"] == "CUSTOM" and e["name"] == "guardrail_tripped")
    assert tripped["value"]["stage"] == "output" and "reason" not in tripped["value"]
    assert events[-2] == {
        "type": "MESSAGES_SNAPSHOT",
        "messages": [{"id": "t-msg-0", "role": "user", "content": "pw?"}],
    }
    assert events[-1]["result"]["status"] == "blocked"
    after_trip = events[events.index(tripped) :]
    assert "hunter2" not in json.dumps(after_trip)

    shown = await collect(stream(), guardrail_reasons=True)
    tripped = next(e for e in shown if e["type"] == "CUSTOM" and e["name"] == "guardrail_tripped")
    assert "hunter2" in tripped["value"]["reason"]


async def test_hold_final_answer_never_sends_an_answer_an_output_guardrail_blocks():
    class Secret(BaseModel):
        secret: str

    def no_secrets(ctx, answer):
        if "hunter2" in str(answer):
            raise GuardrailTripped("the answer contains the password hunter2")

    text = Agent(ScriptedModel(["The admin password is hunter2."]), output_guardrails=[no_secrets])
    answer_tool = Agent(
        ScriptedModel([call("final_answer", secret="hunter2")]),
        output_type=Secret,
        output_mode="tool",
        output_guardrails=[no_secrets],
    )
    researcher = Agent(
        ScriptedModel(["The admin password is hunter2."]),
        name="researcher",
        description="Finds facts.",
        output_guardrails=[no_secrets],
    )
    lead = Agent(ScriptedModel([call("researcher", task="pw"), "Sorry."]), tools=[researcher])
    for agent, options in ((text, {}), (answer_tool, {}), (lead, {"subagents": "native"})):
        events = await collect(agent.stream("pw?"), hold_final_answer=True, **options)
        assert "hunter2" not in json.dumps(events)
        tripped = [e for e in events if e.get("name") == "guardrail_tripped"]
        assert tripped and tripped[0]["value"]["stage"] == "output"
    assert events[-1]["result"]["status"] == "completed"  # the lead answered "Sorry."


async def test_hold_final_answer_sends_what_passes_in_the_same_order():
    weather = call("get_weather", city="Athens")

    def stream():
        return Agent(ScriptedModel([weather, "Sunny."]), tools=[get_weather]).stream("Weather?")

    held = await collect(stream(), thread_id="t", run_id="r", hold_final_answer=True)
    assert held == await collect(stream(), thread_id="t", run_id="r")


def test_hold_final_answer_sends_a_response_once_the_run_goes_on():
    projector = AGUIProjector(thread_id="t", run_id="r", steps=False, hold_final_answer=True)
    projector.feed({"type": "run_started", "run_id": "r", "session_id": "t"})

    def respond(*parts) -> list[dict]:
        message = Message(role="assistant", parts=list(parts)).model_dump(mode="json")
        return projector.feed({"type": "model_responded", "response": {"message": message}})

    weather = ToolCallPart(id="c1", name="get_weather", arguments={"city": "Athens"})
    assert projector.feed({"type": "text_delta", "text": "Checking."}) == []
    assert respond(TextPart(text="Checking."), weather) == []
    started = projector.feed({"type": "tool_call_started", "call": weather.model_dump()})
    assert [e["type"] for e in started] == [
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
    ]

    assert projector.feed({"type": "text_delta", "text": "Sunny."}) == []
    assert respond(TextPart(text="Sunny.")) == []
    state = projector.feed({"type": "state_updated", "values": {"city": "Athens"}})
    assert [e["type"] for e in state] == ["STATE_SNAPSHOT", "CUSTOM"]  # not held
    finished = projector.feed({"type": "run_finished", "result": {"status": "completed"}})
    assert [e["type"] for e in finished] == [
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "RUN_FINISHED",
    ]

    # A run that raises drops what waits, and ends nothing it never started.
    failing = AGUIProjector(thread_id="t", run_id="r", steps=False, hold_final_answer=True)
    failing.feed({"type": "run_started", "run_id": "r", "session_id": "t"})
    assert failing.feed({"type": "text_delta", "text": "Partial"}) == []
    assert [e["type"] for e in failing.fail(RuntimeError("down"))] == ["RUN_ERROR"]


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
    assert [e["type"] for e in events].count("MESSAGES_SNAPSHOT") == 1
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
        ],
        allow_argument_edits=True,
    )

    assert resumed["approvals"] == {
        "c1": Approval(approved=True, arguments={"x": 1}),
        "c2": Approval(approved=False),
        "c3": Approval(approved=False),
    }
    assert resumed["results"]["c4"] == {"city": "Athens"}
    cancelled = resumed["results"]["c5"]
    assert isinstance(cancelled, ToolError) and str(cancelled) == prompts.INPUT_CANCELLED


async def test_a_client_cannot_change_the_arguments_of_an_approved_call():
    refunds: list[int] = []

    @tool
    def pay_back(order: str, amount: int) -> str:
        """Refund an order."""
        refunds.append(amount)
        return f"refunded {amount}"

    def policy(ctx, tool, call):
        amount = call.arguments.get("amount", 0)
        return "deny" if amount > 100 else "ask" if amount > 10 else "allow"

    agent = Agent(
        ScriptedModel([call("pay_back", order="A-1", amount=50), "done"]),
        tools=[pay_back],
        permissions=Permissions(can_use_tool=policy, approval_secret="k" * 32),
    )
    session = Session()
    events = await collect(agent.stream("refund A-1", session=session))
    (interrupt,) = events[-1]["outcome"]["interrupts"]
    assert "arguments" not in interrupt["responseSchema"]["properties"]
    assert "arguments" not in APPROVAL_RESPONSE_SCHEMA["properties"]

    payload = {
        "approved": True,
        "token": interrupt["metadata"]["token"],
        "arguments": {"order": "A-1", "amount": 100_000},
    }
    entry = {"interruptId": interrupt["id"], "status": "resolved", "payload": payload}
    with pytest.raises(ValueError, match="allow_argument_edits=True"):
        agui_resume([entry])
    assert refunds == []


async def test_cancelled_interrupts_carry_the_token_of_a_signed_pause():
    refunds: list[str] = []

    @tool(requires_approval=True)
    def refund_order(order: str) -> str:
        """Refund an order."""
        refunds.append(order)
        return "refunded"

    @tool(deferred=True)
    def fetch_report(report_id: str) -> str:
        """Fetch a report another system produces."""
        raise AssertionError("never runs")

    def agent() -> Agent:
        calls = [call("refund_order", order="A-1"), call("fetch_report", report_id="r")]
        return Agent(
            ScriptedModel([calls, "ok"]),
            tools=[refund_order, fetch_report],
            permissions=Permissions(approval_secret="s3cret-for-tests-only-0123456789"),
        )

    async def cancel_all(with_token: bool) -> Session:
        session = Session()
        first = agent()
        events = await collect(first.stream("go", session=session))
        answers = [
            {
                "interruptId": interrupt["id"],
                "status": "cancelled",
                "payload": {"token": interrupt["metadata"]["token"]} if with_token else None,
            }
            for interrupt in events[-1]["outcome"]["interrupts"]
        ]
        result = await first.resume(session, **agui_resume(answers))
        assert result.status == "completed"
        return session

    session = await cancel_all(with_token=True)
    assert refunds == []
    results = [part for message in await session.messages() for part in message.tool_results]
    contents = sorted(result.content for result in results)
    assert prompts.INPUT_CANCELLED in contents and all(result.is_error for result in results)

    with pytest.raises(ApprovalTokenError):
        await cancel_all(with_token=False)
    assert refunds == []


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


async def test_a_later_snapshot_never_brings_back_an_earlier_blocked_answer():
    def no_secrets(ctx, answer):
        if "hunter" in str(answer):
            raise GuardrailTripped("blocked")

    for hold in (False, True):
        model = ScriptedModel(["The password is hunter2.", "Fine: hunter3."])
        agent = Agent(model, output_guardrails=[no_secrets])
        session = Session()
        await collect(agent.stream("pw?", session=session), hold_final_answer=hold)
        second = await collect(agent.stream("again?", session=session), hold_final_answer=hold)
        assert "hunter2" not in json.dumps(second), hold
    state = await session.state()
    assert [m.metadata.get("blocked") for m in state.messages if m.role == "assistant"] == [
        True,
        True,
    ]


async def test_hold_final_answer_sends_the_answer_of_a_native_sub_agent_that_passed():
    from qagents.multi import fan_out

    for kind in ("as_tool", "fan_out"):
        child = Agent(ScriptedModel(["Athens has 3 million people."]), name="researcher")
        built = child.as_tool() if kind == "as_tool" else fan_out(child)
        arguments = {"task": "pop"} if kind == "as_tool" else {"tasks": ["pop"]}
        lead = Agent(ScriptedModel([call(built.name, **arguments), "Done."]), tools=[built])
        events = await collect(lead.stream("q"), subagents="native", hold_final_answer=True)
        text = "".join(
            e["delta"]
            for e in events
            if e["type"] == "TEXT_MESSAGE_CONTENT" and e.get("subagentRunId")
        )
        assert text == "Athens has 3 million people.", kind


# --- external agents ----------------------------------------------------------------------


def coder(*steps) -> ScriptedExternalAgent:
    return ScriptedExternalAgent([list(steps)], name="coder")


async def test_a_direct_external_turn_is_a_finished_run():
    for hold in (False, True):
        agent = coder("Reading.", ToolStep("read", {"path": "a.py"}, result="print(1)"), "Done.")
        async with agent.stream("Fix it") as turn:
            events = await collect(turn, thread_id="t", run_id="r", hold_final_answer=hold)
        assert turn.result is not None
        call_id = turn.result.tool_calls[0].id

        assert events == [
            {"type": "RUN_STARTED", "threadId": "t", "runId": "r"},
            {"type": "TEXT_MESSAGE_START", "messageId": "r-msg-1", "role": "assistant"},
            {"type": "TEXT_MESSAGE_CONTENT", "messageId": "r-msg-1", "delta": "Reading."},
            {"type": "TEXT_MESSAGE_END", "messageId": "r-msg-1"},
            {
                "type": "TOOL_CALL_START",
                "toolCallId": call_id,
                "toolCallName": "coder.read",
                "parentMessageId": "r-msg-1",
            },
            {"type": "TOOL_CALL_ARGS", "toolCallId": call_id, "delta": '{"path": "a.py"}'},
            {"type": "TOOL_CALL_END", "toolCallId": call_id},
            {
                "type": "TOOL_CALL_RESULT",
                "messageId": "r-result-2",
                "toolCallId": call_id,
                "content": "print(1)",
                "role": "tool",
            },
            {"type": "TEXT_MESSAGE_START", "messageId": "r-msg-3", "role": "assistant"},
            {"type": "TEXT_MESSAGE_CONTENT", "messageId": "r-msg-3", "delta": "Done."},
            {"type": "TEXT_MESSAGE_END", "messageId": "r-msg-3"},
            {
                "type": "RUN_FINISHED",
                "threadId": "t",
                "runId": "r",
                "result": {"status": "completed", "stop_reason": "end_turn", "output": None},
            },
        ], hold


async def test_a_direct_external_turn_ends_as_its_status_says():
    ends = {
        "completed": "RUN_FINISHED",
        "cancelled": "RUN_FINISHED",
        "limit_reached": "RUN_FINISHED",
        "error": "RUN_ERROR",
        "interrupted": "RUN_ERROR",
    }
    for status, end in ends.items():
        async with coder("Working.", EndStep(status=status, stop_reason="why")).stream("t") as turn:
            events = await collect(turn)
        assert [e["type"] for e in events].count(end) == 1, status
        if end == "RUN_FINISHED":
            assert events[-1]["result"] == {"status": status, "stop_reason": "why", "output": None}
        else:
            assert events[-1] == {
                "type": "RUN_ERROR",
                "message": RUN_ERROR_MESSAGE,
                "code": f"external_{status}",
            }

    async with coder(EndStep(status="error", stop_reason="the agent crashed")).stream("t") as turn:
        shown = await collect(turn, expose_errors=True)
    assert shown[-1]["message"] == "the agent crashed"


async def test_the_tool_calls_of_an_external_sub_agent_are_started_before_their_results():
    for hold in (False, True):
        external = coder("Looking. ", ToolStep("bash", {"command": "ls"}, result="a.py"), "Done.")
        lead = Agent(
            ScriptedModel([call("coder", task="t"), "ok"]),
            tools=[external.as_tool()],
            permissions=Permissions(allow=["coder.*"]),
        )
        events = await collect(lead.stream("go"), subagents="native", hold_final_answer=hold)

        child = next(e for e in events if e["type"] == "SUBAGENT_STARTED")["subagentRunId"]
        inner = [e for e in events if e.get("subagentRunId") == child]
        start = next(e for e in inner if e["type"] == "TOOL_CALL_START")
        assert start["toolCallName"] == "coder.bash"
        calls = [e["type"] for e in inner if e.get("toolCallId") == start["toolCallId"]]
        assert calls == ["TOOL_CALL_START", "TOOL_CALL_ARGS", "TOOL_CALL_END", "TOOL_CALL_RESULT"]
        assert events[-1]["type"] == "RUN_FINISHED"


def test_a_call_that_only_finished_is_started_before_its_result():
    # A bound tool server reports a call it refused with tool_call_finished alone.
    projector = AGUIProjector(thread_id="t", run_id="r", steps=False)
    projector.feed({"type": "run_started", "run_id": "r", "session_id": "t"})
    served = ToolCallPart(id="mcp-1", name="lookup", arguments={"q": "x"})
    refused = {
        "type": "tool_call_finished",
        "call": served.model_dump(),
        "result": {"tool_call_id": "mcp-1", "name": "lookup", "content": "No.", "is_error": True},
    }

    assert projector.feed(refused) == [
        {
            "type": "TOOL_CALL_START",
            "toolCallId": "mcp-1",
            "toolCallName": "lookup",
            "parentMessageId": "r-msg-1",
        },
        {"type": "TOOL_CALL_ARGS", "toolCallId": "mcp-1", "delta": '{"q": "x"}'},
        {"type": "TOOL_CALL_END", "toolCallId": "mcp-1"},
        {
            "type": "TOOL_CALL_RESULT",
            "messageId": "r-result-2",
            "toolCallId": "mcp-1",
            "content": "No.",
            "role": "tool",
        },
    ]
    started = {"type": "tool_call_started", "call": served.model_dump()}
    assert projector.feed(started) == []  # announced once


def test_a_resumed_run_does_not_start_the_calls_of_the_paused_response_again():
    started = {"type": "tool_call_started", "call": {"id": "c1", "name": "refund"}}
    finished = {
        "type": "tool_call_finished",
        "call": {"id": "c1", "name": "refund"},
        "result": {"tool_call_id": "c1", "name": "refund", "content": "Refunded."},
    }
    resumed = AGUIProjector(thread_id="t", run_id="r2", steps=False)
    resumed.feed({"type": "run_started", "run_id": "r", "session_id": "t", "resumed": True})
    assert resumed.feed(started) == []
    assert [e["type"] for e in resumed.feed(finished)] == ["TOOL_CALL_RESULT"]

    # An external turn that continues a session is not a resumed run: its calls are new.
    continued = AGUIProjector(thread_id="t", run_id="r3", steps=False)
    continued.feed(
        {
            "type": "run_started",
            "run_id": "ext:s1:0123456789ab",
            "session_id": "s1",
            "resumed": True,
        }
    )
    assert [e["type"] for e in continued.feed(started)] == [
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
    ]


async def test_a_resumed_run_starts_the_calls_that_its_approved_call_serves():
    # An approved call that serves calls (ctx.call_tool, as a bound tool server does for an
    # external agent) runs before the resumed run's first model request: its calls are new.
    @tool(read_only=True)
    def lookup(q: str) -> str:
        """Look up q."""
        return f"found {q}"

    @tool
    async def delegate(ctx: RunContext, task: str) -> str:
        """Delegate a task; the worker uses lookup."""
        refused = await ctx.call_tool(refund, {"order_id": "A-1"}, source="mcp")
        found = await ctx.call_tool(lookup, {"q": task}, source="mcp")
        return f"{refused.is_error} {found.content}"

    delegated = call("delegate", task="x")
    agent = Agent(
        ScriptedModel([delegated, "done"]),
        tools=[delegate, lookup, refund],
        permissions=Permissions(ask=["delegate"]),
    )
    session = Session()
    first = await collect(agent.stream("go", session=session))
    assert first[-1]["result"]["status"] == "paused"

    second = await collect(
        agent.resume_stream(session, approvals={delegated.id: True}), earlier=first
    )

    results = [e["toolCallId"] for e in second if e["type"] == "TOOL_CALL_RESULT"]
    served = [call_id for call_id in results if call_id.startswith("mcp-")]
    assert len(served) == 2  # the refused call (no tool_call_started) and the one that ran
    for call_id in served:
        kinds = [e["type"] for e in second if e.get("toolCallId") == call_id]
        assert kinds == ["TOOL_CALL_START", "TOOL_CALL_ARGS", "TOOL_CALL_END", "TOOL_CALL_RESULT"]
    # The approved call itself was started by the paused run, not again.
    assert [e["type"] for e in second if e.get("toolCallId") == delegated.id] == [
        "TOOL_CALL_RESULT"
    ]
    assert second[-1]["result"]["status"] == "completed"


def test_a_resumed_run_starts_only_calls_made_while_a_call_of_this_run_runs():
    def started(call_id: str, **extra) -> dict:
        return {"type": "tool_call_started", "call": {"id": call_id, "name": "t"}, **extra}

    def finished(call_id: str) -> dict:
        result = {"tool_call_id": call_id, "name": "t", "content": "ok"}
        return {
            "type": "tool_call_finished",
            "call": {"id": call_id, "name": "t"},
            "result": result,
        }

    served_id = "mcp-" + "0" * 32
    resumed = AGUIProjector(thread_id="t", run_id="r2", steps=False)
    resumed.feed({"type": "run_started", "run_id": "r", "session_id": "t", "resumed": True})
    # A provider id of the served form, first of the run: no call runs, so it is not served.
    assert resumed.feed(started("chatcmpl-tool-" + "1" * 32)) == []
    # Calls of the paused response run in parallel: a model's id is never served.
    assert resumed.feed(started("call_2")) == []
    # A served id while calls of this run run: a new call.
    assert [e["type"] for e in resumed.feed(started(served_id))] == [
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
    ]
    assert [e["type"] for e in resumed.feed(finished(served_id))] == ["TOOL_CALL_RESULT"]
    # An event that names its caller is served, whatever its id.
    named = resumed.feed(started("x-1", caller_tool_call_id="call_2"))
    assert [e["type"] for e in named][0] == "TOOL_CALL_START"
    assert [e["type"] for e in resumed.feed(finished("call_2"))] == ["TOOL_CALL_RESULT"]
    assert [e["type"] for e in resumed.feed(finished("chatcmpl-tool-" + "1" * 32))] == [
        "TOOL_CALL_RESULT"
    ]
