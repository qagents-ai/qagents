from __future__ import annotations

import json
import logging

import httpx
import pytest

from qagents import Agent, ImagePart, Message, ModelSettings, OpenAIChatModel, TextPart, tool
from qagents.errors import (
    ConfigurationError,
    ModelAuthenticationError,
    ModelBadRequestError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelResponseError,
    ModelServerError,
    ModelTimeoutError,
)
from qagents.messages import ToolCallPart, ToolResultPart
from qagents.models._openai_wire import (
    TRUNCATED_ARGUMENTS,
    ChatAccumulator,
    build_chat_body,
    to_wire_messages,
)
from qagents.models.base import ModelCapabilities, ModelRequest, OutputSpec, ToolSpec


def sse(*chunks: dict | str, done: bool = True) -> bytes:
    lines = [f"data: {c if isinstance(c, str) else json.dumps(c)}\n\n" for c in chunks]
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def delta(**fields) -> dict:
    finish = fields.pop("finish_reason", None)
    return {
        "id": "chatcmpl-1",
        "model": "m",
        "choices": [{"index": 0, "delta": fields, "finish_reason": finish}],
    }


def streaming(body: bytes, status: int = 200, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(
        status, headers={"content-type": "text/event-stream", **(headers or {})}, content=body
    )


def model_with(handler, **options) -> OpenAIChatModel:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAIChatModel(
        "m", base_url="http://llm.test/v1", api_key="k", http_client=client, **options
    )


def test_messages_translate_to_chat_completions():
    conversation = [
        Message.system("Be brief."),
        Message.user([TextPart(text="What is this?"), ImagePart(url="https://x.test/a.png")]),
        Message.assistant(
            "Checking.",
            tool_calls=[
                ToolCallPart(id="c1", name="look", arguments={"q": "a"}, raw_arguments='{"q":"a"}'),
                ToolCallPart(id="c2", name="look", raw_arguments="{bad", parse_error="bad"),
            ],
        ),
        Message(
            role="tool",
            parts=[
                ToolResultPart(tool_call_id="c1", name="look", content="found"),
                ToolResultPart(tool_call_id="c2", name="look", content="invalid", is_error=True),
            ],
        ),
    ]
    assert to_wire_messages(conversation) == [
        {"role": "system", "content": "Be brief."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}},
            ],
        },
        {
            "role": "assistant",
            "content": "Checking.",
            "tool_calls": [
                # valid arguments are replayed byte for byte; invalid ones are replaced
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "look", "arguments": '{"q":"a"}'},
                },
                {"id": "c2", "type": "function", "function": {"name": "look", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "found"},
        {"role": "tool", "tool_call_id": "c2", "content": "invalid"},
    ]


def test_request_body_only_contains_what_was_set():
    request = ModelRequest(
        messages=[Message.user("hi")],
        tools=[ToolSpec(name="look", description="Look.", parameters={"type": "object"})],
        settings=ModelSettings(max_tokens=64, tool_choice="look", parallel_tool_calls=False),
        output=OutputSpec(json_schema={"type": "object"}),
    )
    body = build_chat_body("m", request, stream=True)
    assert set(body) == {
        "model", "messages", "max_tokens", "tools", "tool_choice", "parallel_tool_calls",
        "response_format", "stream", "stream_options",
    }  # fmt: skip
    assert body["tool_choice"] == {"type": "function", "function": {"name": "look"}}
    assert body["stream_options"] == {"include_usage": True}
    assert body["response_format"]["json_schema"]["schema"] == {"type": "object"}

    plain = build_chat_body("m", ModelRequest(messages=[Message.user("hi")]), stream=False)
    assert set(plain) == {"model", "messages"}


def test_accumulator_handles_the_ways_servers_differ():
    acc = ChatAccumulator(provider="p", model="m")
    chunks = [
        delta(role="assistant", reasoning_content="Let me "),
        delta(reasoning="think."),  # vLLM's newer field name
        delta(content="Sure"),
        delta(tool_calls=[{"index": 0, "id": "call_a", "function": {"name": "get_", "arguments": ""}}]),
        delta(tool_calls=[{"index": 0, "function": {"name": "weather", "arguments": '{"city":'}}]),
        delta(tool_calls=[{"index": 0, "function": {"arguments": ' "Athens"}'}}]),
        # no index, repeated full name, arguments as an object, no id
        delta(tool_calls=[{"function": {"name": "get_time", "arguments": {"tz": "UTC"}}}]),
        delta(tool_calls=[{"function": {"name": "get_time"}}]),
        delta(tool_calls=[{"index": 2, "id": "call_a", "function": {"name": "bad", "arguments": "[1]"}}]),
        delta(finish_reason="stop"),
        {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 7,
                                  "prompt_tokens_details": {"cached_tokens": 4}}},
    ]  # fmt: skip
    events = [event for chunk in chunks for event in acc.feed(chunk)]
    assert [e.type for e in events[:3]] == ["reasoning_delta", "reasoning_delta", "text_delta"]

    response = acc.result()
    reasoning, text, weather, clock, bad = response.message.parts
    assert reasoning.text == "Let me think." and text.text == "Sure"
    assert (weather.id, weather.name, weather.arguments) == (
        "call_a",
        "get_weather",
        {"city": "Athens"},
    )
    assert weather.raw_arguments == '{"city": "Athens"}'
    assert (clock.name, clock.arguments) == ("get_time", {"tz": "UTC"}) and clock.id.startswith(
        "call_"
    )
    assert (
        bad.id != "call_a" and "expected a JSON object" in bad.parse_error
    )  # duplicate id replaced
    assert response.finish_reason == "tool_calls"  # "stop" with tool calls is normalised
    assert (response.usage.input_tokens, response.usage.output_tokens) == (12, 7)
    assert response.usage.cached_input_tokens == 4 and response.response_id == "chatcmpl-1"


async def test_streaming_request_end_to_end():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["auth"] = str(request.url), request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return streaming(
            sse(
                delta(content="Hel"),
                delta(content="lo"),
                delta(finish_reason="stop"),
                {"choices": [], "usage": {"prompt_tokens": 1000, "completion_tokens": 500}},
            )
        )

    prices = ModelCapabilities(input_price_per_million=1.0, output_price_per_million=2.0)
    model = model_with(handler, capabilities=prices)
    response = await model.request(ModelRequest(messages=[Message.user("hi")]))

    assert seen["url"] == "http://llm.test/v1/chat/completions" and seen["auth"] == "Bearer k"
    assert seen["body"]["stream"] is True and seen["body"]["messages"] == [
        {"role": "user", "content": "hi"}
    ]
    assert response.message.text == "Hello" and response.finish_reason == "stop"
    assert response.usage.cost_usd == pytest.approx(0.002)


async def test_agent_runs_against_an_openai_compatible_server():
    @tool(read_only=True)
    def get_weather(city: str) -> str:
        """Look up the weather."""
        return f"Sunny in {city}"

    replies = [
        sse(
            delta(tool_calls=[{"index": 0, "id": "c1", "function": {"name": "get_weather", "arguments": '{"city": "Athens"}'}}]),
            delta(finish_reason="tool_calls"),
        ),
        sse(delta(content="Sunny."), delta(finish_reason="stop")),
    ]  # fmt: skip
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return streaming(replies[len(bodies) - 1])

    result = await Agent(model_with(handler), tools=[get_weather]).run("Weather in Athens?")
    assert result.output == "Sunny."
    assert bodies[0]["tools"][0]["function"]["name"] == "get_weather"
    assert bodies[1]["messages"][-1] == {
        "role": "tool", "tool_call_id": "c1", "content": "Sunny in Athens",
    }  # fmt: skip


async def test_adapter_waits_for_long_provider_retry_after(monkeypatch):
    waits = []

    async def record_wait(seconds):
        waits.append(seconds)

    def reply(request: httpx.Request):
        if not waits:
            return httpx.Response(429, headers={"retry-after": "95"}, json={"error": {}})
        return streaming(sse(delta(content="ok"), delta(finish_reason="stop")))

    monkeypatch.setattr("qagents.models.openai_chat.asyncio.sleep", record_wait)
    await model_with(reply).request(ModelRequest(messages=[Message.user("x")]))
    assert waits == [95]


async def test_only_requests_that_never_ran_are_sent_again():
    attempts = {"n": 0}

    def rate_limited_once(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(
                429, headers={"retry-after": "0"}, json={"error": {"message": "slow"}}
            )
        return streaming(sse(delta(content="ok"), delta(finish_reason="stop")))

    response = await model_with(rate_limited_once).request(
        ModelRequest(messages=[Message.user("x")])
    )
    assert response.message.text == "ok" and attempts["n"] == 2

    def always_limited(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "0", "x-request-id": "req_9"},
                              json={"error": {"message": "slow", "code": "rate_limited"}})  # fmt: skip

    with pytest.raises(ModelRateLimitError) as limited:
        await model_with(always_limited, max_retries=1).request(
            ModelRequest(messages=[Message.user("x")])
        )
    assert (limited.value.retryable, limited.value.ambiguous) == (True, False)
    assert (limited.value.request_id, limited.value.code) == ("req_9", "rate_limited")

    calls = {"n": 0}

    def broken(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(502, text="bad gateway")

    with pytest.raises(ModelServerError) as failed:
        await model_with(broken).request(ModelRequest(messages=[Message.user("x")]))
    assert calls["n"] == 1 and failed.value.ambiguous  # a 5xx may have run: not replayed
    assert "[502] bad gateway" in str(failed.value)


async def test_http_errors_map_to_typed_errors():
    def unauthorized(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    with pytest.raises(ModelAuthenticationError, match="bad key"):
        await model_with(unauthorized).request(ModelRequest(messages=[Message.user("x")]))

    def rejected(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "unknown field 'store'"}})

    with pytest.raises(ModelBadRequestError, match="unknown field"):
        await model_with(rejected).request(ModelRequest(messages=[Message.user("x")]))


async def test_a_cut_or_failed_stream_is_an_error_not_a_short_answer():
    def cut(request: httpx.Request) -> httpx.Response:
        return streaming(sse(delta(content="Hel"), done=False))

    with pytest.raises(ModelResponseError, match="ended before the model finished") as error:
        await model_with(cut).request(ModelRequest(messages=[Message.user("x")]))
    assert error.value.ambiguous

    def failed(request: httpx.Request) -> httpx.Response:
        return streaming(sse(delta(content="Hel"), {"error": {"message": "upstream died"}}))

    with pytest.raises(ModelServerError, match="upstream died"):
        await model_with(failed).request(ModelRequest(messages=[Message.user("x")]))


def ask(text: str = "x") -> ModelRequest:
    return ModelRequest(messages=[Message.user(text)])


def replying(*chunks: dict | str, done: bool = True):
    return lambda request: streaming(sse(*chunks, done=done))


def feed_all(*chunks: dict):
    acc = ChatAccumulator(provider="p", model="m")
    for chunk in chunks:
        acc.feed(chunk)
    return acc


def test_tool_calls_without_an_index_are_told_apart():
    def calls(*chunks: dict) -> list[tuple[str | None, str, dict]]:
        response = feed_all(*chunks, delta(finish_reason="tool_calls")).result()
        return [(c.id, c.name, c.arguments) for c in response.message.tool_calls]

    weather = {"name": "get_weather", "arguments": '{"city": "Athens"}'}
    clock = {"name": "get_time", "arguments": '{"tz": "UTC"}'}
    # each call whole in its own chunk, with ids
    assert calls(
        delta(tool_calls=[{"id": "call_a", "function": weather}]),
        delta(tool_calls=[{"id": "call_b", "function": clock}]),
    ) == [("call_a", "get_weather", {"city": "Athens"}), ("call_b", "get_time", {"tz": "UTC"})]
    # without ids: a different name, or the same name with new arguments, is a new call
    no_ids = calls(
        delta(tool_calls=[{"function": weather}]),
        delta(tool_calls=[{"function": clock}]),
        delta(tool_calls=[{"function": {"name": "get_time", "arguments": {"tz": "CET"}}}]),
    )
    assert [(name, arguments) for _, name, arguments in no_ids] == [
        ("get_weather", {"city": "Athens"}),
        ("get_time", {"tz": "UTC"}),
        ("get_time", {"tz": "CET"}),
    ]
    # one call in fragments, the id only in the first, still is one call
    assert calls(
        delta(tool_calls=[{"id": "call_a", "function": {"name": "get_weather", "arguments": ""}}]),
        delta(tool_calls=[{"function": {"arguments": '{"city":'}}]),
        delta(tool_calls=[{"function": {"arguments": ' "Athens"}'}}]),
    ) == [("call_a", "get_weather", {"city": "Athens"})]


async def test_index_less_parallel_calls_both_run():
    ran: list[str] = []

    @tool(read_only=True)
    def get_weather(city: str) -> str:
        """Weather."""
        ran.append(city)
        return "sunny"

    @tool(read_only=True)
    def get_time(tz: str) -> str:
        """Time."""
        ran.append(tz)
        return "noon"

    replies = [
        sse(
            delta(tool_calls=[{"id": "call_a", "function": {"name": "get_weather", "arguments": '{"city": "Athens"}'}}]),
            delta(tool_calls=[{"id": "call_b", "function": {"name": "get_time", "arguments": '{"tz": "UTC"}'}}]),
            delta(finish_reason="tool_calls"),
        ),
        sse(delta(content="Done."), delta(finish_reason="stop")),
    ]  # fmt: skip
    served: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        served.append(1)
        return streaming(replies[len(served) - 1])

    agent = Agent(model_with(handler), tools=[get_weather, get_time])
    assert (await agent.run("Weather and time?")).output == "Done."
    assert sorted(ran) == ["Athens", "UTC"]


def test_finish_reasons_from_other_servers_are_understood():
    for raw in ("model_length", "max_tokens", "MAX_TOKENS", "max_output_tokens"):
        response = feed_all(delta(content="Half an ans"), delta(finish_reason=raw)).result()
        assert response.finish_reason == "length", raw
    acc = feed_all(delta(content="ok"), delta(finish_reason={"type": "stop"}))
    assert acc.result().finish_reason == "stop"  # an object, as some gateways send it

    unknown = feed_all(delta(content="ok"), delta(finish_reason="eos_token")).result()
    assert unknown.finish_reason == "stop"
    assert unknown.message.metadata == {"p": {"finish_reason": "eos_token"}}  # kept, not lost
    plain = feed_all(delta(content="ok"), delta(finish_reason="stop")).result()
    assert plain.message.metadata == {}


async def test_a_generation_that_failed_is_an_error():
    for reason in ("error", "abort"):
        model = model_with(replying(delta(content="Hal"), delta(finish_reason=reason)))
        with pytest.raises(ModelResponseError, match=f"finish_reason '{reason}'") as raised:
            await model.request(ask())
        assert raised.value.ambiguous


async def test_missing_usage_is_unknown_not_free(caplog):
    prices = ModelCapabilities(input_price_per_million=10.0, output_price_per_million=30.0)
    model = model_with(
        replying(delta(content="ok"), delta(finish_reason="stop")),
        include_usage=False,
        capabilities=prices,
    )
    with caplog.at_level(logging.WARNING, logger="qagents.openai_chat"):
        first = await model.request(ask())
        await model.request(ask())
    assert first.usage.cost_usd is None and first.usage.input_tokens == 0
    assert first.usage.unpriced_requests == 1 and not first.usage.cost_complete
    assert caplog.text.count("did not report token usage") == 1  # once per model


async def test_a_cost_limit_stops_a_run_whose_server_reports_no_usage():
    from qagents import Limits

    sent: list[int] = []

    @tool(read_only=True)
    def ping() -> str:
        """Ping."""
        return "pong"

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(1)
        if len(sent) % 2:
            first = {
                "index": 0,
                "id": f"c{len(sent)}",
                "function": {"name": "ping", "arguments": "{}"},
            }
            return streaming(sse(delta(tool_calls=[first]), delta(finish_reason="tool_calls")))
        return streaming(sse(delta(content="done"), delta(finish_reason="stop")))

    model = model_with(handler, include_usage=False)
    result = await Agent(model, tools=[ping], limits=Limits(max_cost_usd=1.0)).run("hi")
    assert (result.status, result.stop_reason, len(sent)) == ("limit_reached", "max_cost_usd", 1)
    assert (await Agent(model, tools=[ping]).run("hi")).status == "completed"  # no cost limit


def test_text_parts_are_never_run_together():
    two = [TextPart(text="Translate to French:"), TextPart(text="Good morning")]
    wire = to_wire_messages(
        [Message(role="system", parts=[TextPart(text="Be brief."), TextPart(text="No lists.")]),
         Message.user(two), Message.system("One part."), Message.user("One part.")]
    )  # fmt: skip
    assert [entry["content"] for entry in wire] == [
        "Be brief.\nNo lists.",
        "Translate to French:\nGood morning",
        "One part.",  # a single part keeps exactly the same bytes
        "One part.",
    ]


async def test_redirects_and_responses_that_are_not_streams():
    def redirect(request: httpx.Request) -> httpx.Response:
        return httpx.Response(308, headers={"location": "https://llm.test/v1/chat/completions"})

    primary = model_with(redirect)
    with pytest.raises(ModelBadRequestError, match="redirected the request to https://") as raised:
        await primary.request(ask())
    assert (raised.value.retryable, raised.value.ambiguous) == (False, False)  # it never ran

    def json_error(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": {"message": "model not loaded", "code": "nl"}})

    with pytest.raises(ModelServerError, match="model not loaded") as raised:
        await model_with(json_error).request(ask())
    assert raised.value.code == "nl" and raised.value.ambiguous

    def whole_completion(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "id": "cmpl-1", "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Whole."},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        })  # fmt: skip

    events = [event async for event in model_with(whole_completion).stream(ask())]
    assert [event.type for event in events] == ["text_delta", "response_done"]
    assert (
        events[-1].response.message.text == "Whole."
        and events[-1].response.usage.output_tokens == 1
    )

    for incomplete in (
        {"id": "c", "choices": []},
        {"choices": [{"message": {"content": "Half"}}]},  # no finish_reason
    ):
        model = model_with(lambda request, body=incomplete: httpx.Response(200, json=body))
        with pytest.raises(ModelResponseError, match="how it finished") as raised:
            await model.request(ask())
        assert raised.value.ambiguous  # never taken as a complete answer

    def html_page(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, text="<h1>Login</h1>")

    with pytest.raises(ModelResponseError, match="'text/html'.*Login") as raised:
        await model_with(html_page).request(ask())
    assert raised.value.ambiguous


async def test_local_mistakes_are_configuration_errors_not_ambiguous_failures():
    with pytest.raises(ConfigurationError, match="must start with http"):
        OpenAIChatModel("m", base_url="127.0.0.1:8000/v1")
    with pytest.raises(ConfigurationError, match="api_key contains a line break") as raised:
        OpenAIChatModel("m", base_url="http://llm.test/v1", api_key="sk-a\nb")
    assert "sk-a" not in str(raised.value)
    with pytest.raises(ConfigurationError, match="'X-Team'"):
        OpenAIChatModel("m", base_url="http://llm.test/v1", headers={"X-Team": "a\r\nb"})

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        return streaming(sse(delta(content="ok"), delta(finish_reason="stop")))

    model = OpenAIChatModel(
        "m",
        base_url="http://llm.test/v1",
        api_key="sk-from-a-file\n",  # a trailing newline, as read from a file
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    await model.request(ask())
    assert seen == ["Bearer sk-from-a-file"]

    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.LocalProtocolError("Illegal header value b'Bearer sk-secret-value '")

    with pytest.raises(ConfigurationError, match="not valid HTTP") as raised:
        await model_with(refused).request(ask())
    assert "sk-secret-value" not in str(raised.value) and raised.value.__cause__ is None


@pytest.mark.parametrize(
    ("status", "expected", "retryable", "ambiguous"),
    [
        (405, ModelBadRequestError, False, False),
        (413, ModelBadRequestError, False, False),
        (415, ModelBadRequestError, False, False),
        (408, ModelTimeoutError, True, True),
        (409, ModelServerError, True, True),
        (500, ModelServerError, True, True),
    ],
)
async def test_client_errors_are_bad_requests(status, expected, retryable, ambiguous):
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(status)
        return httpx.Response(status, text="Request Entity Too Large")

    with pytest.raises(expected) as raised:
        await model_with(handler).request(ask())
    assert type(raised.value) is expected
    assert (raised.value.retryable, raised.value.ambiguous) == (retryable, ambiguous)
    assert len(calls) == 1


def test_a_cut_off_tool_call_is_described_by_the_prompt_constant(monkeypatch):
    from qagents import prompts

    assert TRUNCATED_ARGUMENTS == prompts.TOOL_CALL_TRUNCATED  # the old name stays an alias
    monkeypatch.setattr(prompts, "TOOL_CALL_TRUNCATED", "cut off here")
    response = feed_all(
        delta(tool_calls=[{"index": 0, "id": "c1", "function": {"name": "search", "arguments": '{"q'}}]),
        delta(finish_reason="length"),
    ).result()  # fmt: skip
    assert response.message.tool_calls[0].parse_error == "cut off here"


def test_a_tool_call_cut_off_by_max_tokens_is_not_run_with_empty_arguments():
    response = feed_all(
        delta(tool_calls=[{"index": 0, "id": "c1", "function": {"name": "ping", "arguments": ""}}]),
        delta(tool_calls=[{"index": 1, "id": "c2", "function": {"name": "search", "arguments": ""}}]),
        delta(finish_reason="length"),
    ).result()  # fmt: skip
    ping, search = response.message.tool_calls
    assert ping.parse_error is None and ping.arguments == {}  # finished: a call without arguments
    assert search.parse_error == TRUNCATED_ARGUMENTS and search.raw_arguments == ""
    assert to_wire_messages([response.message])[0]["tool_calls"][1]["function"]["arguments"] == "{}"

    partial = feed_all(
        delta(tool_calls=[{"index": 0, "id": "c1", "function": {"name": "search", "arguments": '{"q": "a'}}]),
        delta(finish_reason="length"),
    ).result()  # fmt: skip
    assert partial.message.tool_calls[0].parse_error == TRUNCATED_ARGUMENTS

    complete = feed_all(
        delta(tool_calls=[{"index": 0, "id": "c1", "function": {"name": "search", "arguments": ""}}]),
        delta(finish_reason="tool_calls"),
    ).result()  # fmt: skip
    assert complete.message.tool_calls[0].parse_error is None


async def test_an_agent_does_not_run_a_call_whose_arguments_were_cut_off():
    ran: list[tuple[str, int]] = []

    @tool(read_only=True)
    def search_logs(query: str = "*", days: int = 30) -> str:
        """Search the logs."""
        ran.append((query, days))
        return "found"

    replies = [
        sse(
            delta(tool_calls=[{"index": 0, "id": "call_1", "function": {"name": "search_logs", "arguments": ""}}]),
            delta(finish_reason="length"),
        ),
        sse(delta(content="Stopping."), delta(finish_reason="stop")),
    ]  # fmt: skip
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return streaming(replies[len(bodies) - 1])

    await Agent(model_with(handler), tools=[search_logs]).run("Search")
    assert ran == []
    refusal = bodies[1]["messages"][-1]
    assert refusal["role"] == "tool" and "output token limit" in refusal["content"]


async def test_a_refusal_is_kept_and_ends_the_turn():
    model = model_with(
        replying(
            delta(refusal="I can't help "),
            delta(refusal="with that."),
            delta(tool_calls=[{"index": 0, "id": "c1", "function": {"name": "wipe", "arguments": "{}"}}]),
            delta(finish_reason="stop"),
        )
    )  # fmt: skip
    events = [event async for event in model.stream(ask())]
    assert [e.text for e in events if e.type == "text_delta"] == ["I can't help ", "with that."]
    response = events[-1].response
    assert response.message.text == "I can't help with that."
    assert response.finish_reason == "content_filter" and response.message.tool_calls == []
    assert response.message.metadata == {"openai-compatible": {"refusal": True}}

    acc = ChatAccumulator(provider="p", model="m")
    acc.feed_completion(
        {"choices": [{"index": 0, "finish_reason": "stop",
                      "message": {"role": "assistant", "content": None, "refusal": "No."}}]}
    )  # fmt: skip
    assert (acc.result().message.text, acc.result().finish_reason) == ("No.", "content_filter")


async def test_an_agent_reports_a_refusal_instead_of_an_empty_answer():
    model = model_with(
        replying(delta(refusal="I can't help with that."), delta(finish_reason="stop"))
    )
    result = await Agent(model).run("Plan it")
    assert result.stop_reason == "content_filter" and result.output == "I can't help with that."


async def test_timeout_applies_to_a_client_passed_in():
    timeouts: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        timeouts.append(request.extensions["timeout"])
        return streaming(sse(delta(content="ok"), delta(finish_reason="stop")))

    def client(**options) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), **options)

    base = "http://llm.test/v1"
    await OpenAIChatModel("m", base_url=base, http_client=client()).request(ask())
    await OpenAIChatModel("m", base_url=base, http_client=client(), timeout=30.0).request(ask())
    configured = client(timeout=httpx.Timeout(42.0))
    await OpenAIChatModel("m", base_url=base, http_client=configured).request(ask())
    # httpx's 5 s default is too short for generation; a timeout chosen on purpose is kept
    assert [entry["read"] for entry in timeouts] == [600.0, 30.0, 42.0]


def test_strict_schemas_close_every_object():
    nested = {
        "type": "object",
        "properties": {"address": {"$ref": "#/$defs/Address"}, "express": {"type": "boolean"}},
        "required": ["address", "express"],
        "$defs": {"Address": {"type": "object", "properties": {"street": {"type": "string"}},
                              "required": ["street"]}},
    }  # fmt: skip
    request = ModelRequest(
        messages=[Message.user("hi")],
        tools=[ToolSpec(name="ship", parameters=nested, strict=True), ToolSpec(name="loose", parameters=nested)],
        output=OutputSpec(json_schema=nested, strict=True),
    )  # fmt: skip
    body = build_chat_body("m", request, stream=False)
    strict, loose = (tool["function"]["parameters"] for tool in body["tools"])
    output = body["response_format"]["json_schema"]["schema"]
    for schema in (strict, output):
        assert schema["additionalProperties"] is False
        assert schema["$defs"]["Address"]["additionalProperties"] is False
        assert schema["properties"]["express"] == {"type": "boolean"}
    assert loose == nested and "additionalProperties" not in nested  # only strict ones change


def test_index_less_fragments_go_on_with_the_call_at_their_position():
    def calls(*chunks: dict) -> list[tuple[str, dict, str | None]]:
        response = feed_all(*chunks, delta(finish_reason="tool_calls")).result()
        return [(c.name, c.arguments, c.parse_error) for c in response.message.tool_calls]

    # two calls opened in one delta, both continued by position in the next
    assert calls(
        delta(tool_calls=[{"id": "a", "function": {"name": "f", "arguments": '{"x":'}},
                          {"id": "b", "function": {"name": "g", "arguments": '{"y":'}}]),
        delta(tool_calls=[{"function": {"arguments": "1}"}}, {"function": {"arguments": "2}"}}]),
    ) == [("f", {"x": 1}, None), ("g", {"y": 2}, None)]  # fmt: skip
    # the name repeated with only whitespace after complete arguments: still one call
    assert calls(
        delta(tool_calls=[{"function": {"name": "f", "arguments": '{"a": 1}'}}]),
        delta(tool_calls=[{"function": {"name": "f", "arguments": "\n"}}]),
    ) == [("f", {"a": 1}, None)]


async def test_index_less_calls_continued_by_position_run_with_their_own_arguments():
    ran: list[tuple[str, int]] = []

    @tool(read_only=True)
    def f(x: int) -> str:
        """F."""
        ran.append(("f", x))
        return "ok"

    @tool(read_only=True)
    def g(y: int) -> str:
        """G."""
        ran.append(("g", y))
        return "ok"

    replies = [
        sse(
            delta(tool_calls=[{"id": "a", "function": {"name": "f", "arguments": '{"x":'}},
                              {"id": "b", "function": {"name": "g", "arguments": '{"y":'}}]),
            delta(tool_calls=[{"function": {"arguments": "1}"}}, {"function": {"arguments": "2}"}}]),
            delta(finish_reason="tool_calls"),
        ),
        sse(delta(content="Done."), delta(finish_reason="stop")),
    ]  # fmt: skip
    served: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        served.append(1)
        return streaming(replies[len(served) - 1])

    await Agent(model_with(handler), tools=[f, g]).run("Both")
    assert sorted(ran) == [("f", 1), ("g", 2)]


async def test_a_stream_labelled_as_json_is_still_read():
    def mislabelled(request: httpx.Request) -> httpx.Response:
        body = sse(delta(content="Hello"), delta(finish_reason="stop"))
        return httpx.Response(200, headers={"content-type": "application/json"}, content=body)

    response = await model_with(mislabelled).request(ask())
    assert response.message.text == "Hello" and response.finish_reason == "stop"


async def test_an_event_stream_without_data_is_described_as_such():
    def keepalive_only(request: httpx.Request) -> httpx.Response:
        return streaming(b": keepalive\n\n")

    with pytest.raises(ModelResponseError, match="before the server sent any data") as raised:
        await model_with(keepalive_only).request(ask())
    assert "instead of an event stream" not in str(raised.value) and raised.value.ambiguous


def test_a_header_value_that_is_not_ascii_is_refused_without_quoting_it():
    with pytest.raises(
        ConfigurationError, match="api_key contains a character that is not ASCII"
    ) as raised:
        OpenAIChatModel("m", base_url="http://llm.test/v1", api_key="sk-secret\u201d")
    assert "sk-secret" not in str(raised.value)
    with pytest.raises(ConfigurationError, match="'X-Team'.*not ASCII") as raised:
        OpenAIChatModel("m", base_url="http://llm.test/v1", headers={"X-Team": "caf\u00e9-team"})
    assert "caf" not in str(raised.value)


async def test_connection_errors_never_quote_credentials_written_into_base_url():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(refuse))
    model = OpenAIChatModel(
        "m",
        base_url="http://svc:Pa55w0rd@llm.test:8080/v1?token=t0p",
        api_key="k",
        http_client=client,
        max_retries=0,
    )
    with pytest.raises(ModelConnectionError) as raised:
        await model.request(ask())
    assert "Pa55w0rd" not in str(raised.value) and "t0p" not in str(raised.value)
    assert "http://llm.test:8080/v1" in str(raised.value)


async def test_header_values_given_as_bytes_are_checked_and_sent():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["x-org"])
        return streaming(sse(delta(content="ok"), delta(finish_reason="stop")))

    model = model_with(handler, headers={"X-Org": b"acme"})
    assert (await model.request(ask())).message.text == "ok" and seen == ["acme"]
    with pytest.raises(ConfigurationError, match="control character"):
        OpenAIChatModel("m", base_url="http://llm.test/v1", headers={"X-Org": b"a\nb"})
