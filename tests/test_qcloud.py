"""The QCloud provider, exercised through the real ``qcloud_ai`` SDK.

The SDK's HTTP transport is replaced by a local one, so these tests cover the
SDK's own request building, stream validation and error classes without any
network access or paid model call.
"""

from __future__ import annotations

import json

import httpx
import pytest

qcloud_ai = pytest.importorskip("qcloud_ai")

from qagents import Agent, ImagePart, Limits, Message, ModelSettings, QCloudModel, TextPart, tool  # noqa: E402
from qagents.errors import (  # noqa: E402
    ModelAuthenticationError,
    ModelBadRequestError,
    ModelBillingError,
    ModelRateLimitError,
    ModelResponseError,
    ModelServerError,
)
from qagents.models.base import ModelRequest, ToolSpec  # noqa: E402
from qagents.models.qcloud import ALLOWED_PARAMS  # noqa: E402

CATALOG_ENTRY = {
    "id": "qc-chat",
    "object": "model",
    "capabilities": ["tools", "vision"],
    "context_length": 128000,
    "max_output_tokens": 8192,
    "pricing": {"input_per_million": 0.5, "output_per_million": 1.5},
}


def sse(*chunks: dict, done: bool = True) -> bytes:
    lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def delta(**fields) -> dict:
    finish = fields.pop("finish_reason", None)
    return {
        "id": "chatcmpl-q",
        "model": "qc-chat",
        "choices": [{"index": 0, "delta": fields, "finish_reason": finish}],
    }


USAGE = {
    "choices": [],
    "usage": {"prompt_tokens": 2000, "completion_tokens": 1000, "total_tokens": 3000},
}


class FakeQCloud:
    """Plays the QCloud API behind the real SDK client."""

    def __init__(
        self,
        *streams: bytes,
        catalog: dict | None = CATALOG_ENTRY,
        error: httpx.Response | None = None,
    ):
        self.streams = list(streams)
        self.catalog = catalog
        self.error = error
        self.posts: list[dict] = []
        self.gets: list[str] = []
        self.headers: list[httpx.Headers] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            self.gets.append(request.url.path)
            if self.catalog is None:
                return httpx.Response(503, json={"error": {"message": "catalog offline"}})
            return httpx.Response(200, json=self.catalog)
        self.posts.append(json.loads(request.content))
        self.headers.append(request.headers)
        if self.error is not None:
            return self.error
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=self.streams.pop(0)
        )

    def model(self, **options) -> QCloudModel:
        client = qcloud_ai.AsyncQCloud(
            api_key="qc-test-key",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self)),
        )
        return QCloudModel("qc-chat", client=client, **options)


async def test_streams_through_the_sdk_and_prices_the_response():
    api = FakeQCloud(
        sse(delta(content="Γεια "), delta(content="σου"), delta(finish_reason="stop"), USAGE)
    )
    model = api.model()
    request = ModelRequest(
        messages=[Message.system("Be brief."), Message.user("Πες γεια")],
        settings=ModelSettings(temperature=0.2, max_tokens=64, seed=7),
    )
    events = [event async for event in model.stream(request)]

    assert [e.type for e in events] == ["text_delta", "text_delta", "response_done"]
    response = events[-1].response
    assert response.message.text == "Γεια σου"
    assert (response.provider, response.model, response.response_id) == (
        "qcloud",
        "qc-chat",
        "chatcmpl-q",
    )
    assert (response.usage.input_tokens, response.usage.output_tokens) == (2000, 1000)
    assert response.usage.cost_usd == pytest.approx(2000 * 0.5 / 1e6 + 1000 * 1.5 / 1e6)

    (body,) = api.posts
    assert set(body) <= ALLOWED_PARAMS  # nothing QCloud would reject with a 400
    assert body == {
        "model": "qc-chat",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Πες γεια"},
        ],
        "max_tokens": 64,
        "temperature": 0.2,
        "seed": 7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    assert api.headers[0]["authorization"] == "Bearer qc-test-key"
    assert api.headers[0]["user-agent"].startswith("qcloud-ai-python/")

    # The catalog entry is read once per model object, not once per request.
    api.streams.append(sse(delta(content="ok"), delta(finish_reason="stop"), USAGE))
    await model.request(request)
    assert api.gets == ["/api/v1/models/qc-chat"]


async def test_an_agent_runs_a_tool_loop_on_qcloud():
    @tool(read_only=True)
    def get_weather(city: str) -> str:
        """Look up the weather for a city."""
        return f"Sunny in {city}"

    api = FakeQCloud(
        sse(
            delta(tool_calls=[{"index": 0, "id": "call_1", "type": "function",
                               "function": {"name": "get_weather", "arguments": ""}}]),
            delta(tool_calls=[{"index": 0, "function": {"arguments": '{"city": "Athens"}'}}]),
            delta(finish_reason="tool_calls"),
            USAGE,
        ),
        sse(delta(content="It is sunny in Athens."), delta(finish_reason="stop"), USAGE),
    )  # fmt: skip
    agent = Agent(api.model(), instructions="Be brief.", tools=[get_weather],
                  limits=Limits(max_cost_usd=1.0))  # fmt: skip
    result = await agent.run("Weather in Athens?")

    assert result.output == "It is sunny in Athens."
    assert result.usage.requests == 2 and result.usage.cost_usd == pytest.approx(0.005)
    first, second = api.posts
    assert first["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Look up the weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    assert second["messages"][2:] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "get_weather", "arguments": '{"city": "Athens"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "Sunny in Athens"},
    ]  # fmt: skip


async def test_unknown_request_fields_fail_before_anything_is_sent():
    api = FakeQCloud()
    request = ModelRequest(
        messages=[Message.user("hi")],
        settings=ModelSettings(extra={"reasoning_effort": "high", "store": True}),
    )
    with pytest.raises(ModelBadRequestError, match="reasoning_effort, store") as error:
        await api.model().request(request)
    assert "Accepted fields:" in str(error.value)
    assert api.posts == [] and api.gets == []


async def test_rejected_settings_fail_when_the_agent_is_built():
    api = FakeQCloud()
    with pytest.raises(
        ModelBadRequestError, match="does not accept the request field\\(s\\) store"
    ):
        Agent(api.model(), settings=ModelSettings(extra={"store": True}))
    agent = Agent(api.model(), settings=ModelSettings(extra={"user": "u-1"}))  # allowed
    with pytest.raises(ModelBadRequestError, match="reasoning_effort"):
        agent.stream("hi", settings=ModelSettings(extra={"reasoning_effort": "high"}))
    assert api.posts == [] and api.gets == []


async def test_catalog_capabilities_fail_fast_with_a_clear_message():
    text_only = {**CATALOG_ENTRY, "capabilities": ["chat"]}
    api = FakeQCloud(catalog=text_only)
    with_tools = ModelRequest(messages=[Message.user("hi")], tools=[ToolSpec(name="look")])
    with pytest.raises(ModelBadRequestError, match="does not list the 'tools' capability"):
        await api.model().request(with_tools)

    with_image = ModelRequest(
        messages=[
            Message.user([TextPart(text="What is this?"), ImagePart(url="https://x.test/a.png")])
        ]
    )
    with pytest.raises(ModelBadRequestError, match="'vision' capability"):
        await api.model().request(with_image)
    assert api.posts == []


async def test_an_unreachable_catalog_does_not_block_inference():
    api = FakeQCloud(sse(delta(content="ok"), delta(finish_reason="stop"), USAGE), catalog=None)
    response = await api.model().request(ModelRequest(messages=[Message.user("hi")]))
    assert response.message.text == "ok" and response.usage.cost_usd is None

    api = FakeQCloud(sse(delta(content="ok"), delta(finish_reason="stop"), USAGE))
    await api.model(use_catalog=False).request(ModelRequest(messages=[Message.user("hi")]))
    assert api.gets == []


@pytest.mark.parametrize(
    ("status", "expected", "retryable", "ambiguous"),
    [
        (400, ModelBadRequestError, False, False),
        (401, ModelAuthenticationError, False, False),
        (402, ModelBillingError, False, False),
        (403, ModelAuthenticationError, False, False),
        (429, ModelRateLimitError, True, False),
        (503, ModelServerError, True, True),
    ],
)
async def test_qcloud_errors_keep_their_meaning_and_request_id(
    status, expected, retryable, ambiguous
):
    failure = httpx.Response(
        status,
        headers={"x-request-id": "req_42", "retry-after": "3"},
        json={"error": {"message": "explained", "type": "t", "code": "some_code"}},
    )
    api = FakeQCloud(error=failure)
    with pytest.raises(expected) as error:
        await api.model().request(ModelRequest(messages=[Message.user("hi")]))
    raised = error.value
    assert (raised.status_code, raised.code, raised.request_id) == (status, "some_code", "req_42")
    assert raised.retry_after == 3.0
    assert (raised.retryable, raised.ambiguous) == (retryable, ambiguous)
    assert "explained" in str(raised) and raised.provider == "qcloud"


async def test_a_truncated_stream_is_ambiguous_and_never_replayed():
    api = FakeQCloud(sse(delta(content="Hel"), done=False))
    agent = Agent(api.model())
    with pytest.raises(ModelResponseError) as error:
        await agent.run("hi")
    assert error.value.ambiguous and len(api.posts) == 1
