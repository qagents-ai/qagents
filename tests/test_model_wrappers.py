from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator

import httpx
import pytest

from qagents import Agent, Message, ModelSettings, OpenAIChatModel, Usage, tool
from qagents.errors import (
    ConfigurationError,
    ModelAuthenticationError,
    ModelBadRequestError,
    ModelBillingError,
    ModelConnectionError,
    ModelError,
    ModelRateLimitError,
    ModelResponseError,
    ModelServerError,
    ModelTimeoutError,
)
from qagents.models.base import (
    Model,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    TextDelta,
)
from qagents.models.fallback import FallbackModel
from qagents.models.qcloud import QCloudModel
from qagents.models.ratelimit import RateLimitedModel
from qagents.models.recording import (
    CASSETTE_MODE_ENV,
    REDACTED,
    CassetteMissError,
    RecordingModel,
    ReplayModel,
    cassette,
    request_fingerprint,
)
from qagents.models.testing import FunctionModel, ScriptedModel, call


class Failing(Model):
    """Streams ``events``, then raises ``error``. Counts its requests."""

    def __init__(
        self,
        error: Exception,
        *,
        name: str = "primary-model",
        provider: str = "qcloud",
        events: tuple[ModelStreamEvent, ...] = (),
    ) -> None:
        self.name = name
        self.provider = provider
        self.error = error
        self.events = events
        self.calls = 0
        self.closed = False

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.calls += 1
        for event in self.events:
            yield event
        raise self.error

    async def aclose(self) -> None:
        self.closed = True


class Described(ScriptedModel):
    """A scripted model whose catalog entry is known."""

    def __init__(self, replies, *, capabilities: ModelCapabilities, **options) -> None:
        super().__init__(replies, **options)
        self._known = capabilities

    async def capabilities(self) -> ModelCapabilities:
        return self._known


@tool(read_only=True)
def get_weather(city: str) -> str:
    """Look up the weather for a city."""
    return f"Sunny in {city}"


def ask(text: str = "Hi", **settings) -> ModelRequest:
    return ModelRequest(messages=[Message.user(text)], settings=ModelSettings(**settings))


# -- FallbackModel -------------------------------------------------------------------------


async def test_fallback_tries_models_in_order_and_names_the_one_that_answered(caplog):
    primary = Failing(ModelRateLimitError("slow down", retryable=True))
    backup = ScriptedModel(["Hello from the backup."], name="backup-model")
    model = FallbackModel([primary, backup])
    assert (model.name, model.provider) == ("primary-model", "qcloud")  # the primary's

    with caplog.at_level(logging.WARNING, logger="qagents.fallback"):
        stream = Agent(model).stream("Hi")
        events = [event async for event in stream]

    assert stream.result is not None and stream.result.output == "Hello from the backup."
    assert primary.calls == 1 and len(backup.requests) == 1
    responded = next(event for event in events if event.type == "model_responded")
    assert (responded.response.model, responded.response.provider) == ("backup-model", "test")
    assert "qcloud:primary-model failed with ModelRateLimitError" in caplog.text
    assert "falling back to test:backup-model" in caplog.text


async def test_fallback_fills_in_the_answering_model_when_the_adapter_did_not():
    bare = ModelResponse(message=Message.assistant("ok"))  # no model or provider
    backup = ScriptedModel([bare], name="backup-model")
    model = FallbackModel([Failing(ModelServerError("down", retryable=True)), backup])

    response = await model.request(ask())

    assert (response.model, response.provider) == ("backup-model", "test")


async def test_the_primary_answers_when_it_can():
    primary = ScriptedModel(["From the primary."], name="primary")
    backup = ScriptedModel([], name="backup")  # fails the test if asked

    response = await FallbackModel([primary, backup]).request(ask())

    assert response.message.text == "From the primary." and response.model == "primary"


async def test_no_fallback_once_output_was_streamed():
    primary = Failing(
        ModelServerError("connection reset", retryable=True), events=(TextDelta(text="Hel"),)
    )
    backup = ScriptedModel([], name="backup")
    agent = Agent(FallbackModel([primary, backup]))

    with pytest.raises(ModelServerError, match="connection reset"):
        await agent.run("Hi")
    assert primary.calls == 1 and backup.requests == []  # the loop did not retry either


async def test_ambiguous_errors_fall_back_only_when_allowed():
    def chain(allow: bool) -> tuple[FallbackModel, ScriptedModel]:
        unsure = ModelServerError("upstream reset", retryable=True, ambiguous=True)
        backup = ScriptedModel(["Backup answer."], name="backup")
        return FallbackModel([Failing(unsure), backup], allow_ambiguous=allow), backup

    model, backup = chain(allow=False)
    with pytest.raises(ModelServerError, match="upstream reset"):
        await Agent(model).run("Hi")
    assert backup.requests == []  # a possibly billed request is not followed by another

    model, backup = chain(allow=True)
    assert (await Agent(model).run("Hi")).output == "Backup answer."


@pytest.mark.parametrize(
    ("error", "falls_back"),
    [
        (ModelRateLimitError("429", retryable=True), True),
        (ModelServerError("503 at admission", retryable=True), True),
        (ModelConnectionError("could not connect", retryable=True), True),
        (ModelBillingError("402"), True),
        (ModelAuthenticationError("401"), True),
        (ModelBadRequestError("400"), False),
        (ModelServerError("500 after sending", retryable=True, ambiguous=True), False),
        (ModelTimeoutError("timed out after sending", ambiguous=True), False),
        (ModelResponseError("truncated stream", ambiguous=True), False),
        (CassetteMissError("miss", fingerprint="f", path=None), False),  # type: ignore[arg-type]
    ],
    ids=lambda value: type(value).__name__ if isinstance(value, Exception) else str(value),
)
async def test_which_errors_fall_back_by_default(error, falls_back):
    backup = ScriptedModel(["Backup answer."], name="backup")
    model = FallbackModel([Failing(error), backup])

    assert model.falls_back_on(error) is falls_back
    if falls_back:
        assert (await model.request(ask())).message.text == "Backup answer."
    else:
        with pytest.raises(type(error)):
            await model.request(ask())
        assert backup.requests == []


async def test_bad_requests_never_fall_back_whatever_fallback_on_says():
    with pytest.raises(ConfigurationError, match="cannot contain ModelBadRequestError"):
        FallbackModel([ScriptedModel([]), ScriptedModel([])], fallback_on=(ModelBadRequestError,))
    with pytest.raises(ConfigurationError, match="subclasses of qagents.errors.ModelError"):
        FallbackModel([ScriptedModel([]), ScriptedModel([])], fallback_on=(ValueError,))  # type: ignore[arg-type]

    backup = ScriptedModel([], name="backup")
    model = FallbackModel([Failing(ModelBadRequestError("bad")), backup], fallback_on=(ModelError,))
    with pytest.raises(ModelBadRequestError):
        await model.request(ask())
    assert backup.requests == []


async def test_a_narrower_fallback_on_is_respected():
    backup = ScriptedModel([], name="backup")
    model = FallbackModel(
        [Failing(ModelServerError("down", retryable=True)), backup],
        fallback_on=(ModelRateLimitError,),
    )
    with pytest.raises(ModelServerError):
        await model.request(ask())
    assert backup.requests == []


async def test_when_every_model_fails_the_last_error_is_raised_with_the_earlier_ones_noted():
    first = Failing(ModelRateLimitError("first is busy", retryable=True), name="one")
    second = Failing(ModelBillingError("second has no credit"), name="two", provider="other")
    third = Failing(ModelRateLimitError("third is busy", retryable=True), name="three")

    with pytest.raises(ModelRateLimitError, match="third is busy") as raised:
        await FallbackModel([first, second, third]).request(ask())

    (note,) = raised.value.__notes__
    assert "qcloud:one: ModelRateLimitError: first is busy" in note
    assert "other:two: ModelBillingError: second has no credit" in note
    assert raised.value.retryable  # the loop may retry the whole chain later


async def test_fallback_reports_the_primary_capabilities_and_closes_every_model():
    primary = Described(["x"], capabilities=ModelCapabilities(tools=True, context_length=8000))
    backup = Failing(ModelServerError("unused"))
    model = FallbackModel([primary, backup])

    assert (await model.capabilities()).context_length == 8000
    await model.aclose()
    assert backup.closed


def test_fallback_checks_settings_against_every_model():
    model = FallbackModel([ScriptedModel([]), QCloudModel("q-model", use_catalog=False)])
    thinking = ModelSettings(extra={"thinking": {"type": "enabled"}})

    with pytest.raises(ConfigurationError) as raised:
        Agent(model, settings=thinking)
    message = str(raised.value)
    assert "Model 2 of the FallbackModel (qcloud:q-model)" in message
    assert "thinking" in message and "do not mix" in message
    assert isinstance(raised.value.__cause__, ModelBadRequestError)

    Agent(model, settings=ModelSettings(temperature=0.2))  # accepted by both


def test_fallback_needs_a_list_of_models():
    with pytest.raises(ConfigurationError, match="needs a list of models"):
        FallbackModel([])
    with pytest.raises(ConfigurationError, match="needs a list of models"):
        FallbackModel(ScriptedModel([]))  # type: ignore[arg-type]


async def test_traces_name_the_model_that_answered():
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from qagents.tracing import OTelHooks

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    primary = Failing(ModelRateLimitError("busy", retryable=True))
    backup = ScriptedModel(["ok"], name="backup-model")
    agent = Agent(FallbackModel([primary, backup]), hooks=[OTelHooks(provider.get_tracer("t"))])

    await agent.run("Hi")

    chat = next(span for span in exporter.get_finished_spans() if span.name.startswith("chat"))
    assert chat.name == "chat backup-model"
    assert chat.attributes["gen_ai.provider.name"] == "test"
    assert chat.attributes["gen_ai.response.model"] == "backup-model"


# -- record and replay ----------------------------------------------------------------------


def weather_script() -> ScriptedModel:
    return ScriptedModel(
        [
            [call("get_weather", city="Athens"), call("get_weather", city="Paris")],
            call("get_weather", city="Rome"),
            "Sunny in Athens, Paris and Rome.",
        ],
        name="recorded-model",
        usage=Usage(requests=1, input_tokens=40, output_tokens=8, cost_usd=0.0004),
    )


async def run_and_collect(agent: Agent, prompt: str):
    stream = agent.stream(prompt)
    deltas = [event.text async for event in stream if event.type == "text_delta"]
    assert stream.result is not None
    return stream.result, deltas


async def test_record_then_replay_a_tool_loop_gives_the_same_result(tmp_path):
    path = tmp_path / "cassettes" / "weather.json"
    recorder = cassette(weather_script(), path)
    assert isinstance(recorder, RecordingModel)
    recorded, recorded_deltas = await run_and_collect(
        Agent(recorder, instructions="Be brief.", tools=[get_weather]), "Weather?"
    )
    assert recorded.output == "Sunny in Athens, Paris and Rome." and recorded.turns == 3

    replayer = cassette(ScriptedModel([]), path)  # this inner model would fail if asked
    assert isinstance(replayer, ReplayModel)
    replayed, replayed_deltas = await run_and_collect(
        Agent(replayer, instructions="Be brief.", tools=[get_weather]), "Weather?"
    )

    assert replayed.output == recorded.output
    assert replayed.messages == recorded.messages  # tool call ids included
    assert replayed.usage == recorded.usage and replayed.usage.cost_usd == pytest.approx(0.0012)
    assert replayed.turns == recorded.turns == 3
    assert replayed_deltas == recorded_deltas and len(replayed_deltas) > 1  # same chunks
    assert replayer.remaining == 0
    assert (replayer.name, replayer.provider) == ("recorded-model", "test")


async def test_a_cassette_is_readable_json_with_a_stable_key_order(tmp_path):
    path = tmp_path / "weather.json"
    await Agent(cassette(weather_script(), path), tools=[get_weather]).run("Weather?")

    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    assert text == json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    assert (data["format"], data["version"]) == ("qagents-cassette", 1)
    assert (data["model"], data["provider"]) == ("recorded-model", "test")
    first = data["interactions"][0]
    assert sorted(first) == ["events", "fingerprint", "kind", "request", "response"]
    assert first["request"]["messages"] == [
        {"digest": first["request"]["messages"][0]["digest"], "excerpt": "Weather?", "role": "user"}
    ]
    assert [tool["name"] for tool in first["request"]["tools"]] == ["get_weather"]
    last = data["interactions"][-1]
    assert [event["type"] for event in last["events"]] == ["text_delta"] * len(last["events"])
    assert last["response"]["message"]["parts"][0]["text"] == "Sunny in Athens, Paris and Rome."
    assert not list(tmp_path.glob("*.tmp"))  # atomic writes leave nothing behind


async def test_a_request_that_was_not_recorded_fails_with_the_closest_recording(tmp_path):
    path = tmp_path / "weather.json"
    await Agent(
        cassette(weather_script(), path), instructions="Be brief.", tools=[get_weather]
    ).run("Weather?")
    replayer = ReplayModel(path)

    with pytest.raises(CassetteMissError) as raised:
        await Agent(replayer, instructions="Be brief.", tools=[get_weather]).run("Weather in Oslo?")

    error = raised.value
    assert isinstance(error, ModelError) and not error.retryable and not error.ambiguous
    assert error.fingerprint == request_fingerprint(replayer.requests[0])
    assert error.fingerprint in str(error)
    assert error.closest_fingerprint is not None and error.closest_fingerprint in str(error)
    assert "closest recorded request is #1" in str(error)
    assert 'message 2 was user "Weather?" and is now user "Weather in Oslo?"' in str(error)
    assert "mode='record'" in str(error)
    assert len(replayer.requests) == 1  # a miss is never retried


async def test_a_miss_names_changed_tools_and_settings(tmp_path):
    path = tmp_path / "weather.json"
    await Agent(cassette(weather_script(), path), tools=[get_weather]).run("Weather?")

    @tool(read_only=True)
    def get_time(city: str) -> str:
        """Current time in a city."""
        return "noon"

    agent = Agent(ReplayModel(path), tools=[get_weather, get_time])
    with pytest.raises(CassetteMissError, match="tools added: get_time"):
        await agent.run("Weather?", settings=ModelSettings(temperature=0.5))
    agent = Agent(ReplayModel(path), tools=[get_weather])
    with pytest.raises(CassetteMissError, match="settings changed: temperature"):
        await agent.run("Weather?", settings=ModelSettings(temperature=0.5))


async def test_duplicate_requests_are_served_in_recorded_order_each_once(tmp_path):
    path = tmp_path / "twice.json"
    recorder = RecordingModel(ScriptedModel(["first", "second"]), path)
    assert (await recorder.request(ask())).message.text == "first"
    assert (await recorder.request(ask())).message.text == "second"

    replayer = ReplayModel(path)
    assert (await replayer.request(ask())).message.text == "first"
    assert (await replayer.request(ask())).message.text == "second"
    with pytest.raises(CassetteMissError, match="recorded 2 time.*served already"):
        await replayer.request(ask())


async def test_non_strict_replay_serves_the_next_recording(tmp_path, caplog):
    path = tmp_path / "loose.json"
    recorder = RecordingModel(ScriptedModel(["one", "two"]), path)
    await recorder.request(ask("a"))
    await recorder.request(ask("b"))

    replayer = ReplayModel(path, strict=False)
    with caplog.at_level(logging.WARNING, logger="qagents.recording"):
        assert (await replayer.request(ask("b"))).message.text == "two"  # exact match first
        assert (await replayer.request(ask("changed"))).message.text == "one"
    assert "serving recording #1 instead" in caplog.text
    with pytest.raises(CassetteMissError):
        await replayer.request(ask("more"))


async def test_concurrent_requests_are_all_recorded(tmp_path):
    path = tmp_path / "parallel.json"

    async def echo(request: ModelRequest) -> str:
        return f"Answer to {request.messages[-1].text}"

    # Two recording models share the file, and requests interleave while streaming.
    one = RecordingModel(FunctionModel(echo, delay=0.001), path)
    two = RecordingModel(FunctionModel(echo, delay=0.001), path)
    prompts = [f"question {n}" for n in range(20)]
    await asyncio.gather(
        *((one if n % 2 else two).request(ask(text)) for n, text in enumerate(prompts))
    )

    # Two more event loops in two threads write to the same file at the same time.
    def record_in_thread(batch: list[str]) -> None:
        model = RecordingModel(FunctionModel(echo, delay=0.001), path)

        async def main() -> None:
            await asyncio.gather(*(model.request(ask(text)) for text in batch))

        asyncio.run(main())

    threaded = [f"threaded {n}" for n in range(20)]
    await asyncio.gather(
        asyncio.to_thread(record_in_thread, threaded[:10]),
        asyncio.to_thread(record_in_thread, threaded[10:]),
    )

    replayer = ReplayModel(path)
    assert replayer.remaining == 40
    everything = prompts + threaded
    answers = await asyncio.gather(*(replayer.request(ask(text)) for text in everything))
    assert [answer.message.text for answer in answers] == [f"Answer to {t}" for t in everything]


async def test_failed_and_abandoned_requests_are_not_recorded(tmp_path):
    path = tmp_path / "failures.json"
    recorder = RecordingModel(
        ScriptedModel([ModelServerError("down", retryable=True), "one two three"]), path
    )
    with pytest.raises(ModelServerError):
        await recorder.request(ask())
    assert not path.exists()

    stream = recorder.stream(ask())
    assert (await anext(stream)).type == "text_delta"
    await stream.aclose()  # the consumer stopped half way
    assert not path.exists()


def sse_reply(text: str) -> bytes:
    chunks = [
        {"id": "r1", "model": "served-model", "choices": [{"index": 0, "delta": {"content": text}}]},
        {"id": "r1", "model": "served-model", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 7, "completion_tokens": 2}},
    ]  # fmt: skip
    return (
        "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks).encode() + b"data: [DONE]\n\n"
    )


async def test_cassettes_never_hold_keys_or_headers(tmp_path):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=sse_reply("Hello.")
        )

    live = OpenAIChatModel(
        "served-model",
        base_url="http://llm.test/v1",
        api_key="sk-test-not-a-real-key",
        headers={"X-Org-Token": "org-header-value"},
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    path = tmp_path / "keys.json"
    extra = {
        "api_key": "sk-extra-value",
        "options": {"Authorization": "Bearer extra-bearer", "max_completion_tokens": 50},
    }
    recorded = await RecordingModel(live, path).request(ask(extra=extra))

    assert seen[0].headers["authorization"] == "Bearer sk-test-not-a-real-key"  # it was sent
    text = path.read_text(encoding="utf-8")
    for secret in ("sk-test-not-a-real-key", "org-header-value", "sk-extra-value", "extra-bearer"):
        assert secret not in text
    assert "X-Org-Token" not in text and REDACTED in text
    assert '"max_completion_tokens": 50' in text  # settings that are not secrets stay readable

    other_secrets = {
        "api_key": "sk-another-value",
        "options": {"Authorization": "Bearer another", "max_completion_tokens": 50},
    }
    replayer = ReplayModel(path)
    replayed = await replayer.request(ask(extra=other_secrets))  # the same fingerprint
    assert replayed == recorded
    assert (replayer.name, replayer.provider) == ("served-model", "openai-compatible")
    assert (replayed.model, replayed.provider) == ("served-model", "openai-compatible")


async def test_capabilities_are_recorded_and_replayed(tmp_path):
    path = tmp_path / "caps.json"
    known = ModelCapabilities(tools=True, context_length=8000, input_price_per_million=1.0)
    recorder = RecordingModel(Described(["ok"], capabilities=known), path)
    assert await recorder.capabilities() == known
    await recorder.request(ask())

    assert await ReplayModel(path).capabilities() == known


def test_fingerprints_cover_what_the_provider_receives():
    plain = ask("Hello")
    assert len(request_fingerprint(plain)) == 64
    assert request_fingerprint(plain) == request_fingerprint(ask("Hello"))

    tagged = plain.model_copy(deep=True)
    tagged.messages[0].metadata["interrupted"] = True  # never sent to a provider
    assert request_fingerprint(tagged) == request_fingerprint(plain)

    assert request_fingerprint(ask("Hello", temperature=0.1)) != request_fingerprint(plain)
    assert request_fingerprint(ask("Hello!")) != request_fingerprint(plain)
    assert request_fingerprint(ask("Hello", extra={"api_key": "a"})) == request_fingerprint(
        ask("Hello", extra={"api_key": "b"})
    )


async def test_cassette_modes(tmp_path, monkeypatch):
    monkeypatch.delenv(CASSETTE_MODE_ENV, raising=False)
    path = tmp_path / "modes.json"
    with pytest.raises(ConfigurationError, match="no cassette at"):
        cassette(ScriptedModel([]), path, mode="replay")
    with pytest.raises(ConfigurationError, match="needs the real model"):
        cassette(None, path)
    with pytest.raises(ConfigurationError, match="must be 'auto', 'record' or 'replay'"):
        cassette(ScriptedModel([]), path, mode="rewind")  # type: ignore[arg-type]

    await cassette(ScriptedModel(["old"]), path).request(ask())
    assert isinstance(cassette(None, path), ReplayModel)  # auto: the file exists

    rerecorder = cassette(ScriptedModel(["new"]), path, mode="record")  # starts a new cassette
    await rerecorder.request(ask())
    assert (await ReplayModel(path).request(ask())).message.text == "new"
    assert ReplayModel(path).remaining == 1

    monkeypatch.setenv(CASSETTE_MODE_ENV, "replay")  # what CI sets
    with pytest.raises(ConfigurationError, match="no cassette at"):
        cassette(ScriptedModel([]), tmp_path / "missing.json")
    with pytest.raises(ConfigurationError, match="forbids recording"):  # no paid calls in CI
        cassette(ScriptedModel([]), path, mode="record")
    monkeypatch.setenv(CASSETTE_MODE_ENV, "record")
    again = cassette(ScriptedModel(["newer"]), path)
    assert isinstance(again, RecordingModel)
    assert ReplayModel(path).remaining == 1  # the old cassette stays until a response is recorded
    await again.request(ask())
    replayer = ReplayModel(path)  # then recording again starts from an empty cassette
    assert replayer.remaining == 1 and (await replayer.request(ask())).message.text == "newer"
    monkeypatch.setenv(CASSETTE_MODE_ENV, "sometimes")
    with pytest.raises(ConfigurationError, match=CASSETTE_MODE_ENV):
        cassette(ScriptedModel([]), path)


async def test_unknown_entries_are_ignored_and_foreign_files_are_refused(tmp_path):
    path = tmp_path / "edited.json"
    await RecordingModel(ScriptedModel(["one two"]), path).request(ask())
    data = json.loads(path.read_text(encoding="utf-8"))
    data["interactions"][0]["events"].insert(0, {"type": "future_delta", "x": 1})
    data["interactions"].insert(0, {"kind": "error", "fingerprint": "f", "error": "later kind"})
    data["written_by"] = "a later version"
    path.write_text(json.dumps(data), encoding="utf-8")

    stream = ReplayModel(path).stream(ask())
    events = [event async for event in stream]
    assert [event.type for event in events] == ["text_delta", "text_delta", "response_done"]

    data["version"] = 99
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Upgrade qagents"):
        ReplayModel(path)

    foreign = tmp_path / "package.json"
    foreign.write_text('{"name": "web-app"}', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="not a qagents cassette"):
        ReplayModel(foreign)
    with pytest.raises(ConfigurationError, match="not a qagents cassette"):
        RecordingModel(ScriptedModel([]), foreign)  # refused before any paid request

    broken = tmp_path / "broken.json"
    broken.write_text('{"format": "qagents-cassette", "version": 1, "interactions": [{}]}', "utf-8")
    with pytest.raises(ConfigurationError, match="Interaction 1 .* is damaged"):
        ReplayModel(broken)


async def test_recording_wraps_a_fallback_and_keeps_which_model_answered(tmp_path):
    path = tmp_path / "fallback.json"
    chain = FallbackModel(
        [Failing(ModelRateLimitError("busy", retryable=True)), ScriptedModel(["ok"], name="b")]
    )
    await RecordingModel(chain, path).request(ask())

    replayer = ReplayModel(path)
    assert (replayer.name, replayer.provider) == ("primary-model", "qcloud")  # the chain's
    response = await replayer.request(ask())
    assert (response.model, response.provider) == ("b", "test")  # who really answered


# -- RateLimitedModel -----------------------------------------------------------------------


async def test_max_concurrent_caps_requests_in_flight():
    active = peak = 0

    async def slow(request: ModelRequest) -> str:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1
        return "ok"

    model = RateLimitedModel(FunctionModel(slow), max_concurrent=2)
    responses = await asyncio.gather(*(model.request(ask()) for _ in range(6)))

    assert peak == 2 and len(responses) == 6


async def test_requests_per_minute_spaces_requests():
    inner = ScriptedModel(["a", "b", "c", "d"])
    model = RateLimitedModel(inner, requests_per_minute=600, burst_s=0.1)  # 1 every 0.1 s

    began = time.perf_counter()
    for _ in range(4):
        await model.request(ask())
    elapsed = time.perf_counter() - began

    assert 0.27 <= elapsed < 1.5


async def test_tokens_per_minute_counts_reported_usage():
    usage = Usage(requests=1, input_tokens=100, output_tokens=50)
    model = RateLimitedModel(
        ScriptedModel(["a", "b", "c"], usage=usage), tokens_per_minute=60_000, burst_s=0.1
    )  # 1000 tokens a second, 100 at most in the bucket

    began = time.perf_counter()
    for _ in range(3):
        await model.request(ask())
    elapsed = time.perf_counter() - began

    # The first request leaves the bucket 50 tokens in debt (0.05 s), the second 150 (0.15 s).
    assert 0.18 <= elapsed < 1.5


async def test_a_request_cancelled_while_waiting_takes_nothing():
    inner = ScriptedModel(["a", "b", "c"])
    model = RateLimitedModel(inner, requests_per_minute=300, burst_s=0.2)  # 1 every 0.2 s
    await model.request(ask())  # takes the only request in the bucket

    waiting = asyncio.create_task(model.request(ask()))
    await asyncio.sleep(0.02)
    waiting.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await waiting

    began = time.perf_counter()
    await model.request(ask())
    # Served when the bucket refills (about 0.18 s), not one refill later (0.38 s).
    assert time.perf_counter() - began < 0.3
    assert len(inner.requests) == 2


async def test_concurrency_slots_come_back_after_cancel_and_early_close():
    gate = asyncio.Event()

    async def blocked(request: ModelRequest) -> str:
        await gate.wait()
        return "one two three"

    model = RateLimitedModel(FunctionModel(blocked, delay=0.001), max_concurrent=1)
    running = asyncio.create_task(model.request(ask()))
    await asyncio.sleep(0.01)
    queued = asyncio.create_task(model.request(ask()))  # waits for the slot
    await asyncio.sleep(0.01)
    queued.cancel()
    running.cancel()
    await asyncio.gather(running, queued, return_exceptions=True)

    gate.set()
    stream = model.stream(ask())
    assert (await anext(stream)).type == "text_delta"
    await stream.aclose()  # the consumer stops early; the slot is given back

    response = await asyncio.wait_for(model.request(ask()), timeout=1.0)
    assert response.message.text == "one two three"


async def test_max_wait_fails_fast_instead_of_waiting():
    inner = ScriptedModel(["a", "b"])
    model = RateLimitedModel(inner, requests_per_minute=300, burst_s=0.2, max_wait_s=0.0)
    await model.request(ask())

    with pytest.raises(ModelRateLimitError, match="was not sent") as raised:
        await model.request(ask())
    assert raised.value.code == "local_rate_limit" and raised.value.retryable
    assert raised.value.retry_after == pytest.approx(0.2, abs=0.05)
    assert len(inner.requests) == 1

    gate = asyncio.Event()

    async def blocked(request: ModelRequest) -> str:
        await gate.wait()
        return "done"

    busy = RateLimitedModel(FunctionModel(blocked), max_concurrent=1, max_wait_s=0.05)
    running = asyncio.create_task(busy.request(ask()))
    await asyncio.sleep(0.01)
    with pytest.raises(ModelRateLimitError, match="slot"):
        await busy.request(ask())
    gate.set()
    assert (await running).message.text == "done"


async def test_a_rate_limited_primary_falls_back_without_waiting():
    primary = RateLimitedModel(
        ScriptedModel(["from the primary"], name="primary"),
        requests_per_minute=1,
        max_wait_s=0.0,
    )
    backup = ScriptedModel(["from the backup"], name="backup")
    model = FallbackModel([primary, backup])

    assert (await model.request(ask())).message.text == "from the primary"
    began = time.perf_counter()
    assert (await model.request(ask())).message.text == "from the backup"
    assert time.perf_counter() - began < 0.5


async def test_a_provider_retry_after_holds_every_request():
    inner = ScriptedModel([ModelRateLimitError("429", retryable=True, retry_after=0.15), "ok"])
    model = RateLimitedModel(inner, max_concurrent=4)

    with pytest.raises(ModelRateLimitError):
        await model.request(ask())
    began = time.perf_counter()
    assert (await model.request(ask())).message.text == "ok"
    assert time.perf_counter() - began >= 0.12


def test_one_rate_limited_model_serves_several_event_loops():
    model = RateLimitedModel(FunctionModel(lambda request: "ok", delay=0.001), max_concurrent=1)

    async def contended() -> None:
        await asyncio.gather(*(model.request(ask()) for _ in range(3)))

    asyncio.run(contended())
    asyncio.run(contended())  # the primitives of the first loop are not reused


@pytest.mark.parametrize(
    "options",
    [
        {"max_concurrent": 0},
        {"max_concurrent": 1.5},
        {"requests_per_minute": -1},
        {"tokens_per_minute": 0},
        {"burst_s": 0},
        {"max_wait_s": -1},
    ],
)
def test_rate_limits_must_be_positive(options):
    with pytest.raises(ConfigurationError):
        RateLimitedModel(ScriptedModel([]), **options)


async def test_rate_limited_model_passes_through_identity_and_checks():
    inner = QCloudModel("q-model", use_catalog=False)
    model = RateLimitedModel(inner, max_concurrent=2)
    assert (model.name, model.provider) == ("q-model", "qcloud")
    with pytest.raises(ModelBadRequestError, match="thinking"):
        Agent(model, settings=ModelSettings(extra={"thinking": True}))

    agent = Agent(RateLimitedModel(weather_script(), max_concurrent=1), tools=[get_weather])
    assert (await agent.run("Weather?")).output == "Sunny in Athens, Paris and Rome."


async def test_max_wait_counts_the_slot_and_the_bucket_together():
    async def slow(request: ModelRequest) -> str:
        await asyncio.sleep(0.3)
        return "done"

    # One slot, busy for 0.3 s; one request per 0.6 s. Each wait alone is under 0.4 s.
    model = RateLimitedModel(
        FunctionModel(slow), max_concurrent=1, requests_per_minute=100, burst_s=0.6, max_wait_s=0.4
    )
    running = asyncio.create_task(model.request(ask()))
    await asyncio.sleep(0.01)
    began = time.perf_counter()
    with pytest.raises(ModelRateLimitError, match="rate limit is used up"):
        await model.request(ask())
    assert time.perf_counter() - began < 0.45  # not 0.3 s for the slot plus 0.4 s more
    assert (await running).message.text == "done"


# -- secrets, safety switches and overwrites in cassettes -----------------------------------


async def test_secret_settings_never_reach_a_cassette_whatever_their_spelling(tmp_path):
    def extra(variant: str) -> dict:
        return {
            "aws_secret_access_key": f"wJalrXUtnFEMI-{variant}",
            "vertex_credentials": f'{{"type": "service_account", "private_key": "{variant}"}}',
            "metadata": {"authToken": f"tok-{variant}", "clientSecret": f"cs-{variant}", "user": "u-1"},
            "Ocp-Apim-Subscription-Key": f"sub-{variant}",
            "key": f"bare-{variant}",
            "innocent_name": f"sk-{variant}",  # looks like a key whatever it is called
            "deployment_pass": f"custom-{variant}",  # named by the caller
            "prompt_cache_key": "cache-1",  # a public request field
            "max_completion_tokens": 50,
        }  # fmt: skip

    path = tmp_path / "secrets.json"
    names = ["deploymentPass"]
    await RecordingModel(ScriptedModel(["ok"]), path, redact_keys=names).request(
        ask(extra=extra("first"))
    )
    text = path.read_text(encoding="utf-8")
    assert "first" not in text  # not one of the secret values
    assert '"cache-1"' in text and '"u-1"' in text and '"max_completion_tokens": 50' in text

    replayer = ReplayModel(path, redact_keys=names)
    assert (await replayer.request(ask(extra=extra("rotated")))).message.text == "ok"
    assert request_fingerprint(ask(extra=extra("a")), redact_keys=names) == request_fingerprint(
        ask(extra=extra("b")), redact_keys=names
    )
    assert request_fingerprint(ask(extra={"prompt_cache_key": "a"})) != request_fingerprint(
        ask(extra={"prompt_cache_key": "b"})
    )


async def test_recording_again_never_deletes_a_file_that_is_not_a_cassette(tmp_path):
    fixture = tmp_path / "fixture.json"
    fixture.write_text('{"customers": [1, 2, 3]}', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="not a qagents cassette"):
        RecordingModel(ScriptedModel([]), fixture, overwrite=True)
    with pytest.raises(ConfigurationError, match="not a qagents cassette"):
        cassette(ScriptedModel([]), fixture, mode="record")
    assert fixture.read_text(encoding="utf-8") == '{"customers": [1, 2, 3]}'


async def test_a_failed_recording_run_keeps_the_previous_cassette(tmp_path):
    path = tmp_path / "good.json"
    await RecordingModel(ScriptedModel(["old"]), path).request(ask())

    failing = RecordingModel(
        ScriptedModel([ModelAuthenticationError("bad key")]), path, overwrite=True
    )
    with pytest.raises(ModelAuthenticationError):
        await failing.request(ask())
    assert (await ReplayModel(path).request(ask())).message.text == "old"  # still there

    fresh = RecordingModel(ScriptedModel(["new", "newer"]), path, overwrite=True)
    await fresh.request(ask())
    await fresh.request(ask("again"))
    replayer = ReplayModel(path)
    assert replayer.remaining == 2 and (await replayer.request(ask())).message.text == "new"


async def test_an_unused_overwrite_never_wipes_what_another_model_records(tmp_path):
    path = tmp_path / "shared.json"
    await RecordingModel(ScriptedModel(["old"]), path).request(ask())
    RecordingModel(ScriptedModel([]), path, overwrite=True)  # made, never used
    appender = RecordingModel(ScriptedModel(["second"]), path)
    await appender.capabilities()
    await appender.request(ask("two"))
    assert ReplayModel(path).remaining == 2  # appended, not replaced

    # two models re-recording one path: the first response starts the new
    # cassette, and the other model's responses are added to it
    first = RecordingModel(ScriptedModel(["a"]), path, overwrite=True)
    second = RecordingModel(ScriptedModel(["b"]), path, overwrite=True)
    await first.request(ask("one"))
    await second.request(ask("two"))
    replayer = ReplayModel(path)
    assert replayer.remaining == 2
    assert (await replayer.request(ask("one"))).message.text == "a"
    assert (await replayer.request(ask("two"))).message.text == "b"


# -- FallbackModel: per-model settings, cooldown, and who answered ---------------------------


async def test_each_model_in_a_chain_can_have_its_own_settings():
    primary = ScriptedModel([ModelRateLimitError("busy", retryable=True)], name="primary")
    backup = ScriptedModel(["ok"], name="backup")
    thinking = {"type": "enabled", "budget_tokens": 1024}
    model = FallbackModel(
        [primary, backup],
        model_settings=[None, ModelSettings(temperature=None, extra={"thinking": thinking})],
    )
    await model.request(ask(temperature=0.3, extra={"user": "u-1"}))

    assert primary.requests[0].settings.temperature == 0.3
    sent = backup.requests[0].settings
    assert sent.temperature is None and sent.extra == {"user": "u-1", "thinking": thinking}

    mixed = FallbackModel(
        [QCloudModel("q-model", use_catalog=False), ScriptedModel([])],
        model_settings=[
            ModelSettings(temperature=0.2),
            ModelSettings(extra={"thinking": thinking}),
        ],
    )
    mixed.validate_settings(ModelSettings())  # QCloud never receives "thinking"
    with pytest.raises(ConfigurationError, match="Model 1 .*model_settings"):
        FallbackModel(
            [QCloudModel("q-model", use_catalog=False), ScriptedModel([])],
            model_settings=[ModelSettings(extra={"thinking": thinking}), None],
        ).validate_settings(ModelSettings())
    with pytest.raises(ConfigurationError, match="one ModelSettings or None per model"):
        FallbackModel([primary, backup], model_settings=[None])
    with pytest.raises(ConfigurationError, match="cooldown_s"):
        FallbackModel([primary, backup], cooldown_s=-1)


async def test_a_failing_model_is_tried_last_while_it_cools_down(caplog):
    primary = Failing(ModelServerError("down", retryable=True))
    backup = ScriptedModel(["one", "two", "three"], name="backup")
    model = FallbackModel([primary, backup], cooldown_s=0.2)

    with caplog.at_level(logging.WARNING, logger="qagents.fallback"):
        first = await model.request(ask())
        second = await model.request(ask())
    assert primary.calls == 1 and caplog.text.count("falling back") == 1  # not asked again
    assert first.message.metadata["fallback"] == {
        "answered_by": "test:backup",
        "position": 1,
        "failed": ["qcloud:primary-model: ModelServerError"],
    }
    assert second.message.metadata["fallback"]["failed"] == []

    await asyncio.sleep(0.25)
    await model.request(ask())
    assert primary.calls == 2  # the cooldown is over: first choice again


async def test_without_a_cooldown_the_order_never_changes():
    primary = Failing(ModelServerError("down", retryable=True))
    model = FallbackModel([primary, ScriptedModel(["a", "b"], name="backup")])
    await model.request(ask())
    response = await model.request(ask())
    assert primary.calls == 2
    assert response.message.metadata["fallback"]["answered_by"] == "test:backup"
