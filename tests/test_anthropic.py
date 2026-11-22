"""The Anthropic provider, exercised through the real ``anthropic`` SDK.

The SDK's HTTP transport (``httpx2``, which the SDK is built on) is replaced by
a local one that plays recorded-style SSE streams, so these tests cover the
SDK's own request building, event parsing and error classes without network
access or a paid model call.
"""

from __future__ import annotations

import json
import logging
import re

import pytest
from pydantic import BaseModel

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")

from qagents import Agent, ImagePart, Limits, Message, ModelSettings, TextPart, tool  # noqa: E402
from qagents.errors import (  # noqa: E402
    ConfigurationError,
    ModelAuthenticationError,
    ModelBadRequestError,
    ModelBillingError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelResponseError,
    ModelServerError,
)
from qagents.messages import ReasoningPart, ToolCallPart, ToolResultPart  # noqa: E402
from qagents.models.anthropic import (  # noqa: E402
    ALLOWED_EXTRA,
    DEFAULT_MODEL,
    FALLBACK_BETA,
    PRICING,
    THINKING_UPDATES_BETA,
    AnthropicModel,
    AnthropicPricing,
)
from qagents.models.base import (  # noqa: E402
    ModelRequest,
    OutputSpec,
    ReasoningDelta,
    ResponseDone,
    TextDelta,
    ToolCallDelta,
    ToolSpec,
)

API = "https://api.anthropic.com"


# --- a fake Messages API behind the real SDK -------------------------------------------------


def sse(*events: tuple[str, dict]) -> bytes:
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


def message_start(
    *,
    model: str = "claude-opus-5-5",
    input_tokens: int = 10,
    cache_read: int = 0,
    cache_write: int = 0,
    usage_extra: dict | None = None,
) -> tuple[str, dict]:
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": 1,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_write,
        **(usage_extra or {}),
    }
    return (
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_01",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": usage,
            },
        },
    )


def block_start(index: int, block: dict) -> tuple[str, dict]:
    return (
        "content_block_start",
        {"type": "content_block_start", "index": index, "content_block": block},
    )


def delta(index: int, payload: dict) -> tuple[str, dict]:
    return (
        "content_block_delta",
        {"type": "content_block_delta", "index": index, "delta": payload},
    )


def block_stop(index: int) -> tuple[str, dict]:
    return ("content_block_stop", {"type": "content_block_stop", "index": index})


def message_delta(
    stop_reason: str,
    *,
    output_tokens: int = 20,
    stop_details: dict | None = None,
    thinking_tokens: int | None = None,
    usage_extra: dict | None = None,
) -> tuple[str, dict]:
    usage: dict = {"output_tokens": output_tokens, **(usage_extra or {})}
    if thinking_tokens is not None:
        usage["output_tokens_details"] = {"thinking_tokens": thinking_tokens}
    return (
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "stop_details": stop_details,
            },
            "usage": usage,
        },
    )


MESSAGE_STOP = ("message_stop", {"type": "message_stop"})


def text_events(index: int, *pieces: str) -> list[tuple[str, dict]]:
    return [
        block_start(index, {"type": "text", "text": ""}),
        *[delta(index, {"type": "text_delta", "text": piece}) for piece in pieces],
        block_stop(index),
    ]


def thinking_events(index: int, text: str, signature: str) -> list[tuple[str, dict]]:
    return [
        block_start(index, {"type": "thinking", "thinking": "", "signature": ""}),
        delta(index, {"type": "thinking_delta", "thinking": text}),
        delta(index, {"type": "signature_delta", "signature": signature}),
        block_stop(index),
    ]


def tool_events(index: int, call_id: str, name: str, *fragments: str) -> list[tuple[str, dict]]:
    return [
        block_start(index, {"type": "tool_use", "id": call_id, "name": name, "input": {}}),
        *[delta(index, {"type": "input_json_delta", "partial_json": piece}) for piece in fragments],
        block_stop(index),
    ]


def reply(
    *blocks: list[tuple[str, dict]],
    stop_reason: str = "end_turn",
    start: tuple[str, dict] | None = None,
    **delta_options,
) -> bytes:
    events = [start or message_start()]
    for group in blocks:
        events.extend(group)
    events.append(message_delta(stop_reason, **delta_options))
    events.append(MESSAGE_STOP)
    return sse(*events)


class FakeAnthropic:
    """Plays the Messages API: records each request and answers with the next reply."""

    def __init__(self, *replies: bytes | httpx2.Response | Exception):
        self.replies = list(replies)
        self.bodies: list[dict] = []
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        self.bodies.append(json.loads(request.content))
        if not self.replies:
            raise AssertionError(f"unexpected request #{len(self.bodies)}")
        answer = self.replies.pop(0)
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, httpx2.Response):
            return answer
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream", "request-id": "req_ok"},
            content=answer,
        )

    @property
    def body(self) -> dict:
        return self.bodies[-1]


def make_model(fake: FakeAnthropic, model: str = DEFAULT_MODEL, **options) -> AnthropicModel:
    client = anthropic.AsyncAnthropic(
        api_key="test-key",
        base_url=API,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(fake)),
    )
    return AnthropicModel(model, client=client, **options)


def error_response(
    status: int, kind: str, message: str = "nope", headers: dict | None = None
) -> httpx2.Response:
    return httpx2.Response(
        status,
        headers={"request-id": "req_err", **(headers or {})},
        json={"type": "error", "error": {"type": kind, "message": message}},
    )


async def collect(model: AnthropicModel, request: ModelRequest) -> tuple[list, object]:
    events = [event async for event in model.stream(request)]
    done = events[-1]
    assert isinstance(done, ResponseDone)
    return events[:-1], done.response


WEATHER = ToolSpec(
    name="get_weather",
    description="Look up the weather.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)


# --- request translation ---------------------------------------------------------------------


async def test_request_json_is_exactly_what_the_messages_api_expects():
    fake = FakeAnthropic(reply(text_events(0, "ok")))
    model = make_model(fake, effort="high", thinking_display="summarized")
    request = ModelRequest(
        messages=[
            Message.system("Be brief."),
            Message.user(
                [
                    TextPart(text="What is in these?"),
                    ImagePart(url="https://img.test/a.png"),
                    ImagePart(url="data:image/jpeg;base64,QUJD"),
                    ImagePart(data="WFla", media_type="image/webp"),
                ]
            ),
            Message.assistant(
                "Checking.",
                tool_calls=[
                    ToolCallPart(id="toolu_1", name="get_weather", arguments={"city": "Athens"})
                ],
            ),
            Message(
                role="tool",
                parts=[
                    ToolResultPart(
                        tool_call_id="toolu_1", name="get_weather", content="boom", is_error=True
                    )
                ],
            ),
            Message.user("And now?"),
        ],
        tools=[WEATHER],
        settings=ModelSettings(stop=["END"]),
        output=OutputSpec(
            json_schema={"type": "object", "properties": {"answer": {"type": "string"}}}
        ),
    )
    await model.request(request)

    headers = fake.requests[0].headers
    assert fake.requests[0].url.path == "/v1/messages"
    assert headers["anthropic-version"] == "2026-10-21"  # sent by the SDK 2.x
    assert headers["anthropic-beta"] == FALLBACK_BETA
    assert headers["authorization"] == "Bearer test-key"
    assert headers["x-stainless-retry-count"] == "0"
    assert fake.body == {
        "model": "claude-opus-5-5",
        "max_tokens": 64000,
        "stream": True,
        "system": [{"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in these?"},
                    {"type": "image", "source": {"type": "url", "url": "https://img.test/a.png"}},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"},
                    },
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/webp", "data": "WFla"},
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Checking."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"city": "Athens"},
                    },
                ],
            },
            {
                # the tool result and the next user turn are one user message
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "boom",
                        "is_error": True,
                    },
                    {"type": "text", "text": "And now?"},
                ],
            },
        ],
        "tools": [
            {
                "name": "get_weather",
                "description": "Look up the weather.",
                "input_schema": WEATHER.parameters,
                "eager_input_streaming": True,
            }
        ],
        "cache_control": {"type": "ephemeral"},
        "stop_sequences": ["END"],
        "output_config": {
            "effort": "high",
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "additionalProperties": False,
                },
            },
        },
        "thinking": {"type": "adaptive", "display": "summarized"},
        "fallbacks": "default",
    }


async def test_without_system_the_cache_breakpoint_goes_on_the_last_tool():
    fake = FakeAnthropic(reply(text_events(0, "ok")))
    model = make_model(fake, cache_ttl="1h")
    second = ToolSpec(
        name="b_tool",
        description="B.",
        strict=True,
        parameters={"type": "object", "properties": {"n": {"type": "integer", "minimum": 1}}},
    )
    await model.request(ModelRequest(messages=[Message.user("hi")], tools=[WEATHER, second]))
    tools = fake.body["tools"]
    assert "cache_control" not in tools[0]
    assert tools[1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert fake.body["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    # strict tools get the schema structured outputs need; constraints move into descriptions
    assert tools[1]["strict"] is True
    assert tools[1]["input_schema"]["additionalProperties"] is False
    assert tools[1]["input_schema"]["properties"]["n"] == {
        "type": "integer",
        "description": "{minimum: 1}",
    }


async def test_caching_and_fallbacks_can_be_turned_off():
    fake = FakeAnthropic(reply(text_events(0, "ok")))
    model = make_model(fake, cache=False, fallbacks=None, max_tokens=1000)
    await model.request(ModelRequest(messages=[Message.system("S"), Message.user("hi")]))
    assert fake.body["system"] == [{"type": "text", "text": "S"}]
    assert "cache_control" not in fake.body and "fallbacks" not in fake.body
    assert fake.body["max_tokens"] == 1000
    assert "anthropic-beta" not in fake.requests[0].headers


async def test_fallbacks_default_only_goes_to_models_that_support_it():
    fake = FakeAnthropic(reply(text_events(0, "ok")), reply(text_events(0, "ok")))
    await make_model(fake, "claude-haiku-5-5").request(ModelRequest(messages=[Message.user("hi")]))
    assert "fallbacks" not in fake.body and "anthropic-beta" not in fake.requests[-1].headers
    explicit = make_model(fake, "claude-haiku-4-5-20251001", fallbacks=[{"model": "claude-opus-5"}])
    await explicit.request(ModelRequest(messages=[Message.user("hi")]))
    assert fake.body["fallbacks"] == [{"model": "claude-opus-5"}]
    assert fake.body["max_tokens"] == 64000  # Claude Haiku 4.5's maximum output
    assert fake.requests[-1].headers["anthropic-beta"] == FALLBACK_BETA
    updates = AnthropicModel("claude-sonnet-5-5", thinking_display="updates", fallbacks=None)
    assert updates.build_params(ModelRequest(messages=[Message.user("hi")]))["betas"] == [
        THINKING_UPDATES_BETA
    ]


async def test_settings_extra_reaches_the_request_and_wins():
    fake = FakeAnthropic(reply(text_events(0, "ok")))
    model = make_model(
        fake,
        effort="low",
        request_options={"metadata": {"user_id": "u1"}, "extra_headers": {"x-trace": "t"}},
    )
    settings = ModelSettings(
        extra={
            "output_config": {"effort": "max"},
            "service_tier": "auto",
            "betas": ["context-1m"],
            "fallbacks": None,
        }
    )
    await model.request(ModelRequest(messages=[Message.user("hi")], settings=settings))
    assert fake.body["output_config"] == {"effort": "max"}
    assert fake.body["service_tier"] == "auto" and fake.body["metadata"] == {"user_id": "u1"}
    assert "fallbacks" not in fake.body
    assert fake.requests[0].headers["anthropic-beta"] == "context-1m"
    assert fake.requests[0].headers["x-trace"] == "t"


async def test_tools_named_in_history_are_declared_when_the_agent_has_none():
    fake = FakeAnthropic(reply(text_events(0, "ok")))
    history = [
        Message.user("Hand me over."),
        Message.assistant(tool_calls=[ToolCallPart(id="toolu_9", name="transfer_to_billing")]),
        Message(
            role="tool",
            parts=[
                ToolResultPart(tool_call_id="toolu_9", name="transfer_to_billing", content="ok")
            ],
        ),
    ]
    await make_model(fake).request(ModelRequest(messages=history))
    assert fake.body["tools"][0]["name"] == "transfer_to_billing"
    assert fake.body["tool_choice"] == {"type": "none"}


async def test_reasoning_from_other_providers_is_not_replayed():
    fake = FakeAnthropic(reply(text_events(0, "ok")))
    history = [
        Message.user("hi"),
        Message(
            role="assistant",
            parts=[ReasoningPart(text="hmm", provider="qcloud"), TextPart(text="Hello")],
        ),
        Message.user("again"),
    ]
    await make_model(fake).request(ModelRequest(messages=history))
    assert fake.body["messages"][1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello"}],
    }


def test_conversation_must_start_with_a_user_message():
    model = AnthropicModel()
    with pytest.raises(ModelBadRequestError, match="start with a user message"):
        model.build_params(ModelRequest(messages=[Message.assistant("hi")]))
    with pytest.raises(ModelBadRequestError, match="image/bmp"):
        model.build_params(
            ModelRequest(messages=[Message.user([ImagePart(data="x", media_type="image/bmp")])])
        )


# --- tool_choice -------------------------------------------------------------------------------


def test_forced_tool_choice_is_downgraded_for_models_that_reject_it(caplog):
    model = AnthropicModel("claude-opus-5-5")
    request = ModelRequest(
        messages=[Message.user("hi")],
        tools=[WEATHER],
        settings=ModelSettings(tool_choice="required"),
    )
    with caplog.at_level(logging.WARNING, logger="qagents.anthropic"):
        assert model.build_params(request)["tool_choice"] == {"type": "auto"}
        named = request.model_copy(
            update={"settings": ModelSettings(tool_choice="get_weather", parallel_tool_calls=False)}
        )
        assert model.build_params(named)["tool_choice"] == {
            "type": "auto",
            "disable_parallel_tool_use": True,
        }
    assert len([r for r in caplog.records if "forced tool choice" in r.message]) == 1  # warned once

    older = AnthropicModel("claude-opus-5")
    assert older.build_params(request)["tool_choice"] == {"type": "any"}
    named = request.model_copy(update={"settings": ModelSettings(tool_choice="get_weather")})
    assert older.build_params(named)["tool_choice"] == {"type": "tool", "name": "get_weather"}
    for choice in ("auto", "none"):
        plain = request.model_copy(update={"settings": ModelSettings(tool_choice=choice)})
        assert model.build_params(plain)["tool_choice"] == {"type": choice}
    assert "tool_choice" not in model.build_params(
        request.model_copy(update={"settings": ModelSettings()})
    )


# --- validation ----------------------------------------------------------------------------------


def test_validate_settings_rejects_what_the_api_rejects_early():
    model = AnthropicModel()
    for settings, needle in [
        (ModelSettings(temperature=0.2), "temperature"),
        (ModelSettings(top_p=0.9), "top_p"),
        (ModelSettings(seed=1), "no seed"),
        (ModelSettings(extra={"top_k": 5}), "top_k"),
        (ModelSettings(extra={"frequency_penalty": 1}), "frequency_penalty"),
        (ModelSettings(extra={"betas": "x"}), "betas"),
        (ModelSettings(max_tokens=200_000), "at most 128000"),
    ]:
        with pytest.raises(ModelBadRequestError, match=needle):
            model.validate_settings(settings)
    model.validate_settings(
        ModelSettings(max_tokens=1000, stop=["x"], extra={"metadata": {"user_id": "u"}})
    )
    # an agent with such settings fails when it is built, not mid-run
    with pytest.raises(ModelBadRequestError, match="temperature"):
        Agent(model, settings=ModelSettings(temperature=0.0))
    assert {"thinking", "metadata", "fallbacks", "betas"} <= ALLOWED_EXTRA


def test_constructor_fails_early_on_bad_options():
    with pytest.raises(ConfigurationError, match="effort"):
        AnthropicModel(effort="extreme")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="request_options"):
        AnthropicModel(request_options={"temperature": 1})
    with pytest.raises(ConfigurationError, match="fallbacks"):
        AnthropicModel(fallbacks="always")  # type: ignore[arg-type]


def test_structured_output_is_sent_to_the_4_x_models_too():
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    request = ModelRequest(messages=[Message.user("hi")], output=OutputSpec(json_schema=schema))
    for model in (
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5-20250929",
    ):
        params = AnthropicModel(model).build_params(request)
        assert params["output_config"]["format"] == {
            "type": "json_schema",
            "schema": {**schema, "additionalProperties": False},
        }


class Scores(BaseModel):
    summary: str
    per_language: dict[str, float]


async def test_an_open_map_in_a_structured_schema_is_refused_before_sending():
    fake = FakeAnthropic(reply(text_events(0, "{}")))
    with pytest.raises(ModelBadRequestError, match=r"\$\.properties\.per_language"):
        await Agent(make_model(fake), output_type=Scores).run("Score it.")
    loose = ToolSpec(
        name="configure",
        strict=True,
        parameters={
            "type": "object",
            "properties": {"options": {"type": "object", "additionalProperties": True}},
        },
    )
    with pytest.raises(ModelBadRequestError, match="tool 'configure'"):
        await make_model(fake).request(ModelRequest(messages=[Message.user("hi")], tools=[loose]))
    assert fake.bodies == []
    # an object with properties that only omits additionalProperties is closed, not refused
    declared = {"type": "object", "properties": {"a": {"type": "string"}}}
    closed = AnthropicModel().build_params(
        ModelRequest(messages=[Message.user("hi")], output=OutputSpec(json_schema=declared))
    )
    assert closed["output_config"]["format"]["schema"]["additionalProperties"] is False


def test_a_bare_object_in_a_structured_schema_is_an_open_map():
    model = AnthropicModel()
    with pytest.raises(ModelBadRequestError, match=r"open map at \$[^.]"):
        model.build_params(
            ModelRequest(
                messages=[Message.user("hi")], output=OutputSpec(json_schema={"type": "object"})
            )
        )
    nested = ToolSpec(
        name="configure",
        strict=True,
        parameters={"type": "object", "properties": {"options": {"type": ["object", "null"]}}},
    )
    with pytest.raises(ModelBadRequestError, match=r"\$\.properties\.options"):
        model.build_params(ModelRequest(messages=[Message.user("hi")], tools=[nested]))
    empty = ToolSpec(name="ping", strict=True, parameters={"type": "object", "properties": {}})
    model.build_params(ModelRequest(messages=[Message.user("hi")], tools=[empty]))  # no args: ok


async def test_capabilities_come_from_the_table():
    caps = await AnthropicModel().capabilities()
    assert (caps.tools, caps.vision, caps.context_length, caps.max_output_tokens) == (
        True,
        True,
        1_000_000,
        128_000,
    )
    assert (caps.input_price_per_million, caps.output_price_per_million) == (4.0, 20.0)
    assert (
        await AnthropicModel("claude-haiku-4-5-20251001").capabilities()
    ).max_output_tokens == 64_000
    unknown = await AnthropicModel("claude-future-9").capabilities()
    assert unknown.context_length is None and unknown.input_price_per_million is None


@pytest.mark.parametrize("config_dir", ["missing", "."])
async def test_missing_credentials_fail_on_the_first_request(monkeypatch, tmp_path, config_dir):
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_PROFILE",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_IDENTITY_TOKEN",
        "ANTHROPIC_IDENTITY_TOKEN_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    # no config directory at all, and one without a profile: both fail before sending
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path / config_dir))
    model = AnthropicModel(base_url="http://127.0.0.1:9")  # building it needs no key
    with pytest.raises(ModelAuthenticationError, match="ANTHROPIC_API_KEY"):
        await model.request(ModelRequest(messages=[Message.user("hi")]))
    await model.aclose()


# --- streaming, usage and cost -------------------------------------------------------------------


async def test_stream_events_usage_and_cost():
    fake = FakeAnthropic(
        reply(
            thinking_events(0, "Let me think.", "sig-abc"),
            text_events(1, "Look", "ing."),
            tool_events(2, "toolu_7", "get_weather", '{"city"', ': "Athens"}'),
            stop_reason="tool_use",
            start=message_start(input_tokens=1000, cache_read=4000, cache_write=2000),
            output_tokens=500,
            thinking_tokens=120,
        )
    )
    model = make_model(fake)
    events, response = await collect(
        model, ModelRequest(messages=[Message.user("Weather?")], tools=[WEATHER])
    )

    assert [type(e) for e in events] == [
        ReasoningDelta,
        TextDelta,
        TextDelta,
        ToolCallDelta,
        ToolCallDelta,
        ToolCallDelta,
    ]
    assert events[0].text == "Let me think."
    assert [(e.index, e.id, e.name, e.arguments_delta) for e in events[3:]] == [
        (0, "toolu_7", "get_weather", ""),
        (0, "toolu_7", "get_weather", '{"city"'),
        (0, "toolu_7", "get_weather", ': "Athens"}'),
    ]
    reasoning, text, call = response.message.parts
    assert reasoning.text == "Let me think." and reasoning.provider == "anthropic"
    assert reasoning.opaque["block"] == {
        "type": "thinking",
        "thinking": "Let me think.",
        "signature": "sig-abc",
    }
    assert reasoning.opaque["model"] == "claude-opus-5-5"
    assert text.text == "Looking."
    assert (call.id, call.name, call.arguments, call.raw_arguments) == (
        "toolu_7",
        "get_weather",
        {"city": "Athens"},
        '{"city": "Athens"}',
    )
    assert response.finish_reason == "tool_calls"
    assert (response.model, response.provider, response.response_id) == (
        "claude-opus-5-5",
        "anthropic",
        "msg_01",
    )

    usage = response.usage
    assert (
        usage.input_tokens,
        usage.cached_input_tokens,
        usage.output_tokens,
        usage.reasoning_tokens,
    ) == (7000, 4000, 500, 120)
    # $4 input, $0.20 cache read, $5 cache write (5 minutes), $20 output per million
    expected = (1000 * 4.0 + 4000 * 0.20 + 2000 * 5.0 + 500 * 20.0) / 1_000_000
    assert usage.cost_usd == pytest.approx(expected)
    assert response.message.metadata["anthropic"]["cache_creation_input_tokens"] == 2000


async def test_cost_uses_overridden_prices_and_is_none_for_unknown_models():
    custom = AnthropicPricing(input=1, output=2, cache_read=0, cache_write_5m=0, cache_write_1h=0)
    fake = FakeAnthropic(
        reply(
            text_events(0, "ok"),
            start=message_start(model="claude-x", input_tokens=1000),
            output_tokens=1000,
        ),
        reply(text_events(0, "ok"), start=message_start(model="claude-x")),
    )
    priced = make_model(fake, "claude-x", pricing=custom)
    priced_usage = (await priced.request(ModelRequest(messages=[Message.user("hi")]))).usage
    assert priced_usage.cost_usd == pytest.approx(0.003)
    assert priced_usage.cost_complete
    unpriced = make_model(fake, "claude-x")
    unpriced_usage = (await unpriced.request(ModelRequest(messages=[Message.user("hi")]))).usage
    assert unpriced_usage.cost_usd is None
    assert not unpriced_usage.cost_complete
    haiku = PRICING["claude-haiku-5-5"]
    assert haiku.cost(input_tokens=200_000, output_tokens=0) == pytest.approx(
        0.1
    )  # long-context tier


async def test_the_agent_continues_after_a_priced_claude_tool_turn_under_budget():
    @tool(read_only=True)
    def count_characters(text: str) -> int:
        """Count characters in the draft."""
        return len(text)

    fake = FakeAnthropic(
        reply(
            tool_events(0, "toolu_1", "count_characters", '{"text":"A draft"}'),
            stop_reason="tool_use",
        ),
        reply(text_events(0, "A draft")),
    )
    agent = Agent(make_model(fake), tools=[count_characters], limits=Limits(max_cost_usd=0.5))

    result = await agent.run("Return a measured draft")

    assert (result.status, result.output) == ("completed", "A draft")
    assert result.usage.cost_complete and 0 < result.usage.cost_usd < 0.5
    assert len(fake.bodies) == 2


async def test_invalid_tool_arguments_are_reported_not_guessed():
    fake = FakeAnthropic(
        reply(tool_events(0, "toolu_1", "get_weather", '{"city": "Ath'), stop_reason="max_tokens")
    )
    _, response = await collect(
        make_model(fake), ModelRequest(messages=[Message.user("hi")], tools=[WEATHER])
    )
    (call,) = response.message.tool_calls
    assert call.parse_error and call.arguments == {} and call.raw_arguments == '{"city": "Ath'
    assert response.finish_reason == "length"


async def test_a_cut_off_tool_call_is_described_by_the_prompt_constant(monkeypatch):
    from qagents import prompts

    monkeypatch.setattr(prompts, "TOOL_CALL_TRUNCATED", "cut off here")
    fake = FakeAnthropic(
        reply(tool_events(0, "toolu_1", "get_weather", '{"city": "Ath'), stop_reason="max_tokens")
    )
    _, response = await collect(
        make_model(fake), ModelRequest(messages=[Message.user("hi")], tools=[WEATHER])
    )
    assert response.message.tool_calls[0].parse_error == "cut off here"


async def test_stop_reasons_map_to_finish_reasons():
    cases = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "model_context_window_exceeded": "length",
    }
    fake = FakeAnthropic(*[reply(text_events(0, "x"), stop_reason=reason) for reason in cases])
    model = make_model(fake)
    for expected in cases.values():
        assert (
            await model.request(ModelRequest(messages=[Message.user("hi")]))
        ).finish_reason == expected


async def test_refusal_ends_with_content_filter_and_never_runs_the_cut_off_tool():
    details = {"type": "refusal", "category": "cyber", "explanation": None}
    fake = FakeAnthropic(
        reply(
            text_events(0, "I can"),
            tool_events(1, "toolu_x", "get_weather", '{"ci'),
            stop_reason="refusal",
            stop_details=details,
        )
    )
    _, response = await collect(
        make_model(fake), ModelRequest(messages=[Message.user("hi")], tools=[WEATHER])
    )
    assert response.finish_reason == "content_filter"
    assert response.message.tool_calls == [] and response.message.text == "I can"
    meta = response.message.metadata["anthropic"]
    assert meta["stop_reason"] == "refusal" and meta["stop_details"]["category"] == "cyber"


async def test_refusal_through_the_agent_runs_no_tool():
    ran: list[str] = []

    @tool(read_only=True)
    def get_weather(city: str) -> str:
        """Look up the weather."""
        ran.append(city)
        return "sunny"

    fake = FakeAnthropic(
        reply(tool_events(0, "toolu_x", "get_weather", '{"city": "A"}'), stop_reason="refusal")
    )
    result = await Agent(make_model(fake), tools=[get_weather]).run("hi")
    assert ran == [] and result.status == "completed" and len(fake.bodies) == 1


async def test_a_mid_output_fallback_keeps_only_text_before_the_switch():
    fallback = {
        "type": "fallback",
        "from": {"model": "claude-opus-5-5"},
        "to": {"model": "claude-opus-5"},
        "trigger": {"type": "refusal"},
    }
    fake = FakeAnthropic(
        reply(
            thinking_events(0, "declined reasoning", "sig-old"),
            text_events(1, "Partial "),
            tool_events(2, "toolu_old", "get_weather", '{"city": "X"}'),
            [block_start(3, fallback), block_stop(3)],
            thinking_events(4, "", "sig-new"),
            text_events(5, "answer."),
            output_tokens=100,
        )
    )
    _, response = await collect(
        make_model(fake), ModelRequest(messages=[Message.user("hi")], tools=[WEATHER])
    )
    kinds = [(type(p).__name__, getattr(p, "text", None)) for p in response.message.parts]
    assert kinds == [
        ("TextPart", "Partial "),
        ("ReasoningPart", ""),
        ("ReasoningPart", ""),
        ("TextPart", "answer."),
    ]
    assert response.message.parts[1].opaque["block"]["type"] == "fallback"
    assert response.message.parts[2].opaque["model"] == "claude-opus-5"
    assert response.model == "claude-opus-5" and response.message.tool_calls == []
    assert response.message.metadata["anthropic"]["served_by"] == "claude-opus-5"
    # billed at the fallback model's rates
    assert response.usage.cost_usd == pytest.approx((10 * 5.0 + 100 * 25.0) / 1_000_000)


async def test_pause_turn_is_continued_within_one_model_request():
    server_use = {
        "type": "server_tool_use",
        "id": "srvtoolu_1",
        "name": "web_search",
        "input": {"query": "q"},
    }
    fake = FakeAnthropic(
        reply(
            text_events(0, "Searching. "),
            [block_start(1, server_use), block_stop(1)],
            stop_reason="pause_turn",
        ),
        reply(text_events(0, "Done."), start=message_start(input_tokens=30)),
    )
    events, response = await collect(make_model(fake), ModelRequest(messages=[Message.user("hi")]))
    assert len(fake.bodies) == 2
    assert fake.bodies[1]["messages"][-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "Searching. "}, server_use],
    }
    assert response.message.text == "Searching. Done." and response.finish_reason == "stop"
    assert response.usage.requests == 2 and response.usage.input_tokens == 40
    assert response.usage.cost_complete


async def test_a_stream_that_stops_early_is_an_ambiguous_error():
    truncated = sse(message_start(), *text_events(0, "Hel"))
    fake = FakeAnthropic(truncated)
    with pytest.raises(ModelResponseError) as caught:
        await make_model(fake).request(ModelRequest(messages=[Message.user("hi")]))
    assert caught.value.ambiguous


# --- thinking replay -------------------------------------------------------------------------------


@tool(read_only=True)
def get_weather(city: str) -> str:
    """Look up the weather."""
    return f"Sunny in {city}"


async def test_tool_loop_end_to_end_replays_thinking_byte_for_byte():
    signature = "EqQBCkYIBxgCKkDm1+/=="
    fake = FakeAnthropic(
        reply(
            thinking_events(0, "I should call the tool.", signature),
            tool_events(1, "toolu_1", "get_weather", '{"city": "Athens"}'),
            stop_reason="tool_use",
        ),
        reply(text_events(0, "It is sunny in Athens.")),
    )
    agent = Agent(make_model(fake), instructions="Be brief.", tools=[get_weather])
    result = await agent.run("Weather in Athens?")

    assert result.output == "It is sunny in Athens."
    second = fake.bodies[1]
    assert (
        second["system"] == fake.bodies[0]["system"] and second["tools"] == fake.bodies[0]["tools"]
    )
    assert second["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Weather in Athens?"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "I should call the tool.", "signature": signature},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {"city": "Athens"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Sunny in Athens"}
            ],
        },
    ]
    assert result.usage.requests == 2


async def test_thinking_is_left_out_once_the_prefix_it_was_bound_to_changed():
    fake = FakeAnthropic(
        reply(thinking_events(0, "plan", "sig-1"), text_events(1, "First answer.")),
        reply(text_events(0, "ok")),
        reply(text_events(0, "ok")),
    )
    model = make_model(fake)
    first = [Message.system("Version 1"), Message.user("one")]
    response = await model.request(ModelRequest(messages=first))
    history = [*first, response.message, Message.user("two")]

    await model.request(ModelRequest(messages=history))  # unchanged prefix: replayed
    assert fake.body["messages"][1]["content"][0] == {
        "type": "thinking",
        "thinking": "plan",
        "signature": "sig-1",
    }

    edited = [Message.system("Version 2"), *history[1:]]  # instructions changed
    await model.request(ModelRequest(messages=edited))
    assert fake.body["messages"][1]["content"] == [{"type": "text", "text": "First answer."}]


async def test_keep_tail_compaction_drops_thinking_of_the_retained_turns():
    fake = FakeAnthropic(
        reply(thinking_events(0, "plan", "sig-1"), text_events(1, "Answer.")),
        reply(text_events(0, "ok")),
    )
    model = make_model(fake)
    early = [Message.user("one"), Message.assistant("a"), Message.user("two")]
    response = await model.request(ModelRequest(messages=early))
    compacted = [
        Message.user("Summary of the earlier conversation: ..."),
        Message.user("two"),
        response.message,
        Message.user("three"),
    ]
    await model.request(ModelRequest(messages=compacted))
    assert fake.body["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Summary of the earlier conversation: ..."},
                {"type": "text", "text": "two"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Answer."}]},
        {"role": "user", "content": [{"type": "text", "text": "three"}]},
    ]


async def test_a_rejected_thinking_block_is_stripped_and_the_request_sent_once_more(caplog):
    rejected = error_response(
        400,
        "invalid_request_error",
        "messages.1.content.0: Invalid `signature` in `thinking` block. The block is bound to a different conversation.",
    )
    fake = FakeAnthropic(rejected, reply(text_events(0, "ok")), reply(text_events(0, "ok")))
    model = make_model(fake)
    block = {"type": "thinking", "thinking": "t", "signature": "sig-bad"}
    history = [
        Message.user("one"),
        Message(
            role="assistant",
            parts=[
                ReasoningPart(
                    provider="anthropic", opaque={"provider": "anthropic", "block": block}
                ),
                TextPart(text="A"),
            ],
        ),
        Message.user("two"),
    ]
    with caplog.at_level(logging.WARNING, logger="qagents.anthropic"):
        response = await model.request(ModelRequest(messages=history))
    assert response.message.text == "ok" and len(fake.bodies) == 2
    assert fake.bodies[0]["messages"][1]["content"][0] == block
    assert fake.bodies[1]["messages"][1]["content"] == [{"type": "text", "text": "A"}]
    await model.request(ModelRequest(messages=history))  # remembered: not sent again
    assert fake.bodies[2]["messages"][1]["content"] == [{"type": "text", "text": "A"}]


# --- errors ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("response", "kind", "retryable", "ambiguous"),
    [
        (
            error_response(429, "rate_limit_error", headers={"retry-after": "7"}),
            ModelRateLimitError,
            True,
            False,
        ),
        (error_response(529, "overloaded_error"), ModelServerError, True, False),
        (error_response(500, "api_error"), ModelServerError, True, True),
        (error_response(401, "authentication_error"), ModelAuthenticationError, False, False),
        (error_response(403, "permission_error"), ModelAuthenticationError, False, False),
        (error_response(402, "billing_error"), ModelBillingError, False, False),
        (
            error_response(400, "invalid_request_error", "bad field"),
            ModelBadRequestError,
            False,
            False,
        ),
        (error_response(404, "not_found_error"), ModelBadRequestError, False, False),
        (error_response(413, "request_too_large"), ModelBadRequestError, False, False),
    ],
)
async def test_http_errors_map_to_typed_errors(response, kind, retryable, ambiguous):
    fake = FakeAnthropic(response)
    with pytest.raises(kind) as caught:
        await make_model(fake).request(ModelRequest(messages=[Message.user("hi")]))
    error = caught.value
    assert type(error) is kind
    assert (error.retryable, error.ambiguous) == (retryable, ambiguous)
    assert error.status_code == response.status_code and error.request_id == "req_err"
    assert error.provider == "anthropic" and error.model == "claude-opus-5-5"
    assert len(fake.bodies) == 1  # the SDK did not retry on its own
    if response.status_code == 429:
        assert error.retry_after == 7.0 and error.code == "rate_limit_error"


async def test_an_error_event_inside_a_started_stream_is_ambiguous():
    body = sse(
        message_start(),
        *text_events(0, "Hel"),
        (
            "error",
            {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
        ),
    )
    fake = FakeAnthropic(body)
    with pytest.raises(ModelServerError) as caught:
        await make_model(fake).request(ModelRequest(messages=[Message.user("hi")]))
    assert caught.value.ambiguous and "Overloaded" in str(caught.value)


async def test_connection_errors_say_whether_the_request_was_sent():
    fake = FakeAnthropic(httpx2.ConnectError("refused"), httpx2.ReadError("reset"))
    model = make_model(fake)
    with pytest.raises(ModelConnectionError) as never_sent:
        await model.request(ModelRequest(messages=[Message.user("hi")]))
    assert never_sent.value.retryable and not never_sent.value.ambiguous
    with pytest.raises(ModelConnectionError) as after_send:
        await model.request(ModelRequest(messages=[Message.user("hi")]))
    assert after_send.value.ambiguous


async def test_the_agent_retries_a_rate_limit_but_not_an_ambiguous_error():
    fake = FakeAnthropic(
        error_response(429, "rate_limit_error", headers={"retry-after": "0"}),
        reply(text_events(0, "ok")),
    )
    result = await Agent(make_model(fake)).run("hi")
    assert result.output == "ok" and len(fake.bodies) == 2

    fake = FakeAnthropic(error_response(500, "api_error"), reply(text_events(0, "ok")))
    with pytest.raises(ModelServerError):
        await Agent(make_model(fake)).run("hi")
    assert len(fake.bodies) == 1


async def test_a_client_passed_in_keeps_its_own_settings_but_does_not_retry():
    fake = FakeAnthropic(error_response(529, "overloaded_error"))
    client = anthropic.AsyncAnthropic(
        api_key="k",
        base_url=API,
        max_retries=5,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(fake)),
    )
    model = AnthropicModel(client=client)
    with pytest.raises(ModelServerError):
        await model.request(ModelRequest(messages=[Message.user("hi")]))
    assert len(fake.bodies) == 1 and client.max_retries == 5
    await model.aclose()  # does not close a client it does not own
    assert not client.is_closed()


def test_eager_input_streaming_is_off_behind_a_custom_base_url():
    proxied = AnthropicModel(base_url="https://gateway.internal/anthropic")
    params = proxied.build_params(ModelRequest(messages=[Message.user("hi")], tools=[WEATHER]))
    assert "eager_input_streaming" not in params["tools"][0]
    forced = AnthropicModel(
        base_url="https://gateway.internal/anthropic", eager_input_streaming=True
    )
    assert (
        forced.build_params(ModelRequest(messages=[Message.user("hi")], tools=[WEATHER]))["tools"][
            0
        ]["eager_input_streaming"]
        is True
    )


# --- regressions from the 0.2 review -------------------------------------------------------------


def server_tool_events(
    index: int, call_id: str, kind: str, raw_input: str
) -> list[tuple[str, dict]]:
    """A server-side tool call as the API streams it: ``input: {}``, then input_json_delta."""
    start = {"type": kind, "id": call_id, "name": "web_search", "input": {}}
    if kind == "mcp_tool_use":
        start["server_name"] = "docs"
    return [
        block_start(index, start),
        delta(index, {"type": "input_json_delta", "partial_json": raw_input}),
        block_stop(index),
    ]


async def test_a_failed_continuation_is_not_retried_from_scratch():
    fake = FakeAnthropic(
        reply(
            server_tool_events(0, "srvtoolu_1", "server_tool_use", "{}"), stop_reason="pause_turn"
        ),
        error_response(429, "rate_limit_error", headers={"retry-after": "0"}),
    )
    with pytest.raises(ModelRateLimitError) as caught:
        await Agent(make_model(fake)).run("Research this.")
    # the billed first segment emitted nothing, yet the model request is not repeated
    assert caught.value.ambiguous and len(fake.bodies) == 2


async def test_server_tool_input_streamed_as_json_deltas_is_kept_and_replayed():
    fake = FakeAnthropic(
        reply(
            text_events(0, "Searching. "),
            server_tool_events(1, "srvtoolu_1", "server_tool_use", '{"query": "athens weather"}'),
            stop_reason="pause_turn",
        ),
        reply(text_events(0, "Done.")),
    )
    events, response = await collect(make_model(fake), ModelRequest(messages=[Message.user("hi")]))
    assert not any(isinstance(event, ToolCallDelta) for event in events)
    sent = fake.bodies[1]["messages"][-1]["content"][1]
    assert sent["type"] == "server_tool_use" and sent["input"] == {"query": "athens weather"}
    journaled = response.message.parts[1].opaque["block"]
    assert journaled["input"] == {"query": "athens weather"} and response.message.tool_calls == []

    broken = FakeAnthropic(reply(server_tool_events(0, "mcptoolu_1", "mcp_tool_use", '{"q": ')))
    with pytest.raises(ModelResponseError) as caught:
        await make_model(broken).request(ModelRequest(messages=[Message.user("hi")]))
    assert caught.value.ambiguous


def iteration(kind: str, model: str | None, input_tokens: int, output_tokens: int) -> dict:
    entry = {
        "type": kind,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    if model is not None:
        entry["model"] = model
    return entry


async def test_usage_and_cost_come_from_the_iterations_when_reported():
    fallback = {
        "type": "fallback",
        "from": {"model": "claude-opus-5-5"},
        "to": {"model": "claude-opus-5"},
        "trigger": {"type": "refusal"},
    }
    declined = iteration("message", "claude-opus-5-5", 50_000, 2000)
    served = iteration("fallback_message", "claude-opus-5", 50_000, 300)
    fake = FakeAnthropic(
        reply(
            text_events(0, "Partial "),
            [block_start(1, fallback), block_stop(1)],
            text_events(2, "answer."),
            start=message_start(input_tokens=50_000),
            output_tokens=300,
            usage_extra={"input_tokens": 50_000, "iterations": [declined, served]},
        ),
        reply(
            text_events(0, "ok"),
            start=message_start(input_tokens=10_000),
            output_tokens=100,
            usage_extra={
                "iterations": [
                    iteration("compaction", None, 160_000, 4000),
                    iteration("message", "claude-opus-5-5", 10_000, 100),
                ]
            },
        ),
    )
    model = make_model(fake)
    response = await model.request(ModelRequest(messages=[Message.user("hi")]))
    # the declined attempt at Claude Opus 5.5 prices, the serving one at Claude Opus 5 prices
    expected = (50_000 * 4.0 + 2000 * 20.0 + 50_000 * 5.0 + 300 * 25.0) / 1_000_000
    assert response.usage.cost_usd == pytest.approx(expected)
    assert (response.usage.input_tokens, response.usage.output_tokens) == (100_000, 2300)
    assert response.message.metadata["anthropic"]["context_tokens"] == 50_300

    compacted = await model.request(ModelRequest(messages=[Message.user("hi")]))
    expected = (160_000 * 4.0 + 4000 * 20.0 + 10_000 * 4.0 + 100 * 20.0) / 1_000_000
    assert compacted.usage.cost_usd == pytest.approx(expected)
    assert compacted.usage.input_tokens == 170_000
    # the compaction entry is not the size of the conversation
    assert compacted.message.metadata["anthropic"]["context_tokens"] == 10_100
    # and compactors read the size, not the billed sum (F151)
    from qagents.context import next_request_tokens

    assert compacted.context_tokens == 10_100 and response.context_tokens == 50_300
    assert next_request_tokens([Message.user("hi"), compacted.message], compacted) == 10_100
    assert next_request_tokens([Message.user("hi"), response.message], response) == 50_300


@pytest.mark.parametrize(("ttl", "price"), [("5m", 5.0), ("1h", 8.0)])
async def test_cache_writes_beyond_the_message_start_breakdown_are_priced(ttl, price):
    breakdown = {"ephemeral_5m_input_tokens": 1000, "ephemeral_1h_input_tokens": 0}
    fake = FakeAnthropic(
        reply(
            text_events(0, "ok"),
            start=message_start(cache_write=1000, usage_extra={"cache_creation": breakdown}),
            usage_extra={"cache_creation_input_tokens": 1_000_000},
        )
    )
    response = await make_model(fake, cache_ttl=ttl).request(
        ModelRequest(messages=[Message.user("hi")])
    )
    expected = (10 * 4.0 + 1000 * 5.0 + 999_000 * price + 20 * 20.0) / 1_000_000
    assert response.usage.cost_usd == pytest.approx(expected)


async def test_context_tokens_report_the_last_segment_not_the_sum():
    fake = FakeAnthropic(
        reply(
            server_tool_events(0, "srvtoolu_1", "server_tool_use", "{}"), stop_reason="pause_turn"
        ),
        reply(text_events(0, "Done."), start=message_start(input_tokens=30)),
    )
    response = await make_model(fake).request(ModelRequest(messages=[Message.user("hi")]))
    assert response.usage.input_tokens == 40  # billed: both segments
    assert response.message.metadata["anthropic"]["context_tokens"] == 50  # 30 in + 20 out
    from qagents.context import next_request_tokens

    assert next_request_tokens([Message.user("hi"), response.message], response) == 50


def test_eager_input_streaming_follows_the_base_url_from_the_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.internal/anthropic")
    model = AnthropicModel(api_key="test-key")
    request = ModelRequest(messages=[Message.user("hi")], tools=[WEATHER])
    first = model.build_params(request)["tools"]
    assert "eager_input_streaming" not in first[0]
    assert model.build_params(request)["tools"] == first


async def test_thinking_after_a_pause_is_replayed_with_the_tool_result():
    search_result = {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": []}
    fake = FakeAnthropic(
        reply(
            thinking_events(0, "search first", "SIG-A"),
            server_tool_events(1, "srvtoolu_1", "server_tool_use", '{"query": "q"}'),
            stop_reason="pause_turn",
        ),
        reply(
            [block_start(0, search_result), block_stop(0)],
            thinking_events(1, "now the weather", "SIG-B"),
            tool_events(2, "toolu_1", "get_weather", '{"city": "Athens"}'),
            stop_reason="tool_use",
        ),
        reply(text_events(0, "Sunny.")),
    )
    result = await Agent(make_model(fake), instructions="Be brief.", tools=[get_weather]).run("hi")
    assert result.output == "Sunny."
    sent = [block.get("signature") for block in fake.bodies[2]["messages"][1]["content"]]
    assert "SIG-A" in sent and "SIG-B" in sent


async def test_an_exhausted_pause_is_incomplete_and_its_server_calls_are_not_replayed():
    fake = FakeAnthropic(
        reply(
            text_events(0, "Let me search more. "),
            server_tool_events(1, "srvtoolu_1", "server_tool_use", "{}"),
            stop_reason="pause_turn",
        ),
        reply(
            text_events(0, "Let me search more. "),
            server_tool_events(1, "srvtoolu_2", "server_tool_use", "{}"),
            stop_reason="pause_turn",
        ),
        reply(text_events(0, "ok")),
    )
    model = make_model(fake, max_continuations=1)
    response = await model.request(ModelRequest(messages=[Message.user("hi")]))
    assert response.finish_reason == "length"
    assert response.message.metadata["anthropic"]["incomplete"] == "max_continuations"
    history = [Message.user("hi"), response.message, Message.user("Go on.")]
    await model.request(ModelRequest(messages=history))
    assert [block["type"] for block in fake.body["messages"][1]["content"]] == ["text", "text"]


async def test_a_refusal_after_an_mcp_call_leaves_no_unpaired_call_in_history():
    fake = FakeAnthropic(
        reply(
            text_events(0, "Let me look. "),
            server_tool_events(1, "mcptoolu_1", "mcp_tool_use", '{"q": "x"}'),
            stop_reason="refusal",
        ),
        reply(text_events(0, "ok")),
    )
    model = make_model(fake)
    response = await model.request(ModelRequest(messages=[Message.user("hi")]))
    history = [Message.user("hi"), response.message, Message.user("Something else, then.")]
    await model.request(ModelRequest(messages=history))
    assert fake.body["messages"][1]["content"] == [{"type": "text", "text": "Let me look. "}]
    # a refusal can cut the call's input off mid-JSON: still a refusal, not an error
    cut = FakeAnthropic(
        reply(server_tool_events(0, "mcptoolu_2", "mcp_tool_use", '{"q": '), stop_reason="refusal")
    )
    refused = await make_model(cut).request(ModelRequest(messages=[Message.user("hi")]))
    assert refused.finish_reason == "content_filter"


def test_a_fallbacks_list_fails_early_on_a_model_without_server_side_fallback():
    with pytest.raises(ConfigurationError, match="no server-side fallback"):
        AnthropicModel("claude-haiku-5-5", fallbacks=[{"model": "claude-opus-5"}])
    model = AnthropicModel("claude-haiku-5-5")
    with pytest.raises(ModelBadRequestError, match="no server-side fallback"):
        Agent(model, settings=ModelSettings(extra={"fallbacks": [{"model": "claude-opus-5"}]}))
    model.validate_settings(ModelSettings(extra={"fallbacks": None}))


async def test_a_cut_off_tool_call_without_input_is_never_run():
    ran: list[str] = []

    @tool(read_only=True)
    def purge(scope: str = "everything") -> str:
        """Purge data."""
        ran.append(scope)
        return "purged"

    fake = FakeAnthropic(
        reply(
            text_events(0, "Purging only the temp table."),
            tool_events(1, "toolu_1", "purge"),
            stop_reason="max_tokens",
        ),
        reply(text_events(0, "Stopped.")),
    )
    result = await Agent(make_model(fake), tools=[purge]).run("Purge the temp table.")
    assert ran == []
    answer = fake.bodies[1]["messages"][-1]["content"][0]
    assert answer["type"] == "tool_result" and answer.get("is_error") is True
    assert "max_tokens" in answer["content"] and result.output == "Stopped."


def test_tool_call_ids_from_other_providers_are_made_valid_and_stay_paired():
    search = ToolSpec(name="search", parameters={"type": "object", "properties": {}})
    history = [
        Message.user("Find it."),
        Message.assistant(tool_calls=[ToolCallPart(id="functions.search:0", name="search")]),
        Message(
            role="tool",
            parts=[ToolResultPart(tool_call_id="functions.search:0", name="search", content="r")],
        ),
    ]
    params = AnthropicModel().build_params(ModelRequest(messages=history, tools=[search]))
    call_id = params["messages"][1]["content"][0]["id"]
    assert re.fullmatch(r"[A-Za-z0-9_-]+", call_id)
    assert params["messages"][2]["content"][0]["tool_use_id"] == call_id
    again = AnthropicModel().build_params(ModelRequest(messages=history, tools=[search]))
    assert again["messages"][1]["content"][0]["id"] == call_id  # stable across requests
    valid = [
        Message.user("x"),
        Message.assistant(tool_calls=[ToolCallPart(id="toolu_1", name="search")]),
    ]
    kept = AnthropicModel().build_params(ModelRequest(messages=valid, tools=[search]))
    assert kept["messages"][1]["content"][0]["id"] == "toolu_1"


async def test_thinking_is_replayed_unchanged_on_models_without_the_history_check():
    manual = ModelSettings(extra={"thinking": {"type": "enabled", "budget_tokens": 2000}})
    fake = FakeAnthropic(
        reply(
            thinking_events(0, "call step2", "sig-1"),
            tool_events(1, "toolu_1", "step2", "{}"),
            stop_reason="tool_use",
            start=message_start(model="claude-haiku-4-5"),
        ),
        reply(text_events(0, "ok"), start=message_start(model="claude-haiku-4-5")),
    )
    model = make_model(fake, "claude-haiku-4-5")
    step = ToolSpec(name="step2", parameters={"type": "object", "properties": {}})
    first = [Message.system("Steps so far: 0"), Message.user("go")]
    response = await model.request(ModelRequest(messages=first, tools=[step], settings=manual))
    result = Message(
        role="tool", parts=[ToolResultPart(tool_call_id="toolu_1", name="step2", content="done")]
    )
    changed = [Message.system("Steps so far: 1"), Message.user("go"), response.message, result]
    await model.request(ModelRequest(messages=changed, tools=[step], settings=manual))
    # the open tool round still starts with its thinking block, as manual thinking requires
    assert fake.body["messages"][1]["content"][0]["signature"] == "sig-1"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("model_id", ["claude-sonnet-4-5", "claude-opus-4-1"])
async def test_manual_thinking_keeps_its_blocks_on_models_not_in_the_table(model_id):
    manual = ModelSettings(extra={"thinking": {"type": "enabled", "budget_tokens": 2000}})
    fake = FakeAnthropic(
        reply(
            thinking_events(0, "call step2", "sig-1"),
            tool_events(1, "toolu_1", "step2", "{}"),
            stop_reason="tool_use",
        ),
        reply(text_events(0, "ok")),
    )
    model = make_model(fake, model_id)
    step = ToolSpec(name="step2", parameters={"type": "object", "properties": {}})
    first = [Message.system("Steps so far: 0"), Message.user("go")]
    response = await model.request(ModelRequest(messages=first, tools=[step], settings=manual))
    result = Message(
        role="tool", parts=[ToolResultPart(tool_call_id="toolu_1", name="step2", content="done")]
    )
    changed = [Message.system("Steps so far: 1"), Message.user("go"), response.message, result]
    await model.request(ModelRequest(messages=changed, tools=[step], settings=manual))
    assert [block["type"] for block in fake.body["messages"][1]["content"]] == [
        "thinking",
        "tool_use",
    ]
    # without manual thinking an unknown id still fails closed and drops the stale block
    adaptive = make_model(FakeAnthropic(reply(text_events(0, "ok"))), model_id)
    params = adaptive.build_params(ModelRequest(messages=changed, tools=[step]))
    assert [block["type"] for block in params["messages"][1]["content"]] == ["tool_use"]


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_a_rejected_block_under_manual_thinking_is_not_stripped_and_resent():
    rejected = error_response(
        400,
        "invalid_request_error",
        "messages.1.content.0: Invalid `signature` in `thinking` block.",
    )
    fake = FakeAnthropic(rejected, reply(text_events(0, "ok")))
    model = make_model(fake, "claude-sonnet-4-5")
    block = {"type": "thinking", "thinking": "t", "signature": "sig-bad"}
    history = [
        Message.user("one"),
        Message(
            role="assistant",
            parts=[
                ReasoningPart(
                    provider="anthropic", opaque={"provider": "anthropic", "block": block}
                ),
                ToolCallPart(id="toolu_1", name="step2"),
            ],
        ),
        Message(
            role="tool", parts=[ToolResultPart(tool_call_id="toolu_1", name="step2", content="d")]
        ),
    ]
    step = ToolSpec(name="step2", parameters={"type": "object", "properties": {}})
    manual = ModelSettings(extra={"thinking": {"type": "enabled", "budget_tokens": 2000}})
    with pytest.raises(ModelBadRequestError, match="signature"):
        await model.request(ModelRequest(messages=history, tools=[step], settings=manual))
    assert len(fake.bodies) == 1 and model._rejected_signatures == set()


async def test_a_rejected_block_under_manual_thinking_is_stripped_outside_a_tool_round():
    rejected = error_response(
        400,
        "invalid_request_error",
        "messages.1.content.0: Invalid `signature` in `thinking` block.",
    )
    fake = FakeAnthropic(rejected, reply(text_events(0, "ok")))
    model = make_model(fake, "claude-haiku-4-5")
    block = {"type": "thinking", "thinking": "t", "signature": "sig-bad"}
    history = [
        Message.user("one"),
        Message(
            role="assistant",
            parts=[
                ReasoningPart(
                    provider="anthropic", opaque={"provider": "anthropic", "block": block}
                ),
                TextPart(text="first answer"),
            ],
        ),
        Message.user("two"),
    ]
    manual = ModelSettings(extra={"thinking": {"type": "enabled", "budget_tokens": 2000}})
    response = await model.request(ModelRequest(messages=history, settings=manual))
    assert response.message.text == "ok" and len(fake.bodies) == 2
    resent = [b["type"] for m in fake.bodies[1]["messages"] for b in m["content"]]
    assert "thinking" not in resent and model._rejected_signatures


def test_repeated_tool_call_ids_get_unique_wire_ids_that_stay_paired():
    search = ToolSpec(name="search", parameters={"type": "object", "properties": {}})

    def turn(question: str, content: str) -> list[Message]:
        return [
            Message.user(question),
            Message.assistant(tool_calls=[ToolCallPart(id="functions.search:0", name="search")]),
            Message(
                role="tool",
                parts=[
                    ToolResultPart(
                        tool_call_id="functions.search:0", name="search", content=content
                    )
                ],
            ),
            Message.assistant("Found."),
        ]

    history = [*turn("one", "r1"), *turn("two", "r2")]
    params = AnthropicModel().build_params(ModelRequest(messages=history, tools=[search]))
    calls = [
        block
        for message in params["messages"]
        for block in message["content"]
        if block.get("type") == "tool_use"
    ]
    results = {
        block["content"]: block["tool_use_id"]
        for message in params["messages"]
        for block in message["content"]
        if block.get("type") == "tool_result"
    }
    first, second = calls[0]["id"], calls[1]["id"]
    assert first != second and all(re.fullmatch(r"[A-Za-z0-9_-]+", i) for i in (first, second))
    assert results == {"r1": first, "r2": second}
    again = AnthropicModel().build_params(ModelRequest(messages=history, tools=[search]))
    assert [m["content"] for m in again["messages"]] == [m["content"] for m in params["messages"]]


def test_thinking_display_updates_fails_on_a_known_model_without_it():
    with pytest.raises(ConfigurationError, match="updates"):
        AnthropicModel("claude-sonnet-5", thinking_display="updates")
    AnthropicModel("claude-opus-5-5", thinking_display="updates")
    AnthropicModel("claude-future-9", thinking_display="updates")  # unknown: unchecked


async def test_an_anthropic_beta_in_extra_headers_is_merged_with_the_adapters_betas():
    fake = FakeAnthropic(reply(text_events(0, "ok")))
    headers = {"Anthropic-Beta": "context-management-2025-06-27", "x-trace": "t"}
    model = make_model(fake, request_options={"extra_headers": headers})
    await model.request(ModelRequest(messages=[Message.user("hi")]))
    sent = fake.requests[0].headers
    assert sent["anthropic-beta"] == f"context-management-2025-06-27,{FALLBACK_BETA}"
    assert sent["x-trace"] == "t" and fake.body["fallbacks"] == "default"


def test_options_a_known_model_rejects_fail_when_it_is_built():
    for options, needle in [
        ({"effort": "low"}, "no effort"),
        ({"thinking_display": "summarized"}, "no adaptive thinking"),
        ({"max_tokens": 100_000}, "between 1 and 64000"),
    ]:
        with pytest.raises(ConfigurationError, match=needle):
            AnthropicModel("claude-haiku-4-5", **options)
    with pytest.raises(ConfigurationError, match="xhigh"):
        AnthropicModel("claude-opus-4-6", effort="xhigh")
    haiku = AnthropicModel("claude-haiku-4-5")
    for extra, needle in [
        ({"output_config": {"effort": "high"}}, "no effort"),
        ({"thinking": {"type": "adaptive"}}, "no adaptive thinking"),
    ]:
        with pytest.raises(ModelBadRequestError, match=needle):
            Agent(haiku, settings=ModelSettings(extra=extra))
    AnthropicModel("claude-future-9", effort="xhigh", max_tokens=500_000)  # unknown: unchecked


def test_thread_and_mcp_servers_are_refused_with_a_reason():
    model = AnthropicModel()
    with pytest.raises(ModelBadRequestError, match="cannot be continued"):
        model.validate_settings(ModelSettings(extra={"thread": {"type": "create"}}))
    with pytest.raises(ModelBadRequestError, match="MCPToolkit"):
        model.validate_settings(ModelSettings(extra={"mcp_servers": []}))
    with pytest.raises(ConfigurationError, match="cannot be continued"):
        AnthropicModel(request_options={"thread": {"type": "create"}})


async def test_a_continuation_after_a_stripped_retry_builds_on_what_was_sent():
    rejected = error_response(
        400,
        "invalid_request_error",
        "messages.1.content.0: Invalid `signature` in `thinking` block. The block is bound to "
        "a different conversation.",
    )
    fake = FakeAnthropic(
        rejected,
        reply(
            thinking_events(0, "fresh", "sig-new"),
            text_events(1, "Looking. "),
            server_tool_events(2, "srvtoolu_1", "server_tool_use", "{}"),
            stop_reason="pause_turn",
        ),
        reply(text_events(0, "done")),
    )
    model = make_model(fake)
    stale = {"type": "thinking", "thinking": "t", "signature": "sig-bad"}
    opaque = {"provider": "anthropic", "block": stale}
    history = [
        Message.user("one"),
        Message(
            role="assistant",
            parts=[ReasoningPart(provider="anthropic", opaque=opaque), TextPart(text="A")],
        ),
        Message.user("two"),
    ]
    response = await model.request(ModelRequest(messages=history))
    assert response.message.text == "Looking. done" and len(fake.bodies) == 3
    continued = json.dumps(fake.bodies[2]["messages"])
    assert "sig-bad" not in continued and "sig-new" in continued
    assert model._rejected_signatures == {"sig-bad"}
