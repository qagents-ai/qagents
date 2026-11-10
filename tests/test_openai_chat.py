from __future__ import annotations

import json

import httpx
import pytest

from qagents import Agent, ImagePart, Message, ModelSettings, OpenAIChatModel, TextPart, tool
from qagents.errors import (
    ModelAuthenticationError,
    ModelBadRequestError,
    ModelRateLimitError,
    ModelResponseError,
    ModelServerError,
)
from qagents.messages import ToolCallPart, ToolResultPart
from qagents.models._openai_wire import ChatAccumulator, build_chat_body, to_wire_messages
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
        delta(tool_calls=[{"index": 1, "function": {"name": "get_time", "arguments": {"tz": "UTC"}}}]),
        delta(tool_calls=[{"index": 1, "function": {"name": "get_time"}}]),
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
