from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from qagents import (
    Agent,
    Hooks,
    Limits,
    Message,
    ModelSettings,
    OutputValidationError,
    RunContext,
    ToolCallPart,
    ToolError,
    ToolResultPart,
    Usage,
    prompts,
    tool,
)
from qagents.errors import ConfigurationError, ModelRateLimitError, ModelServerError
from qagents.models.testing import FunctionModel, ScriptedModel, call


@tool(read_only=True)
def get_weather(city: str) -> str:
    """Look up the weather for a city."""
    return f"Sunny in {city}"


async def test_plain_answer():
    model = ScriptedModel(["Hello there."])
    result = await Agent(model, instructions="Be brief.").run("Hi")

    assert result.status == "completed" and result.completed
    assert result.output == result.text == "Hello there."
    assert result.turns == 1
    assert [m.role for m in result.messages] == ["user", "assistant"]
    sent = model.requests[0]
    assert [m.role for m in sent.messages] == ["system", "user"]
    assert sent.messages[0].text == "Be brief."  # nothing else is added to the instructions
    assert sent.tools == []


async def test_tool_loop_sends_results_back():
    model = ScriptedModel([call("get_weather", city="Athens"), "It is sunny."])
    result = await Agent(model, tools=[get_weather]).run("Weather in Athens?")

    assert result.output == "It is sunny."
    assert [m.role for m in result.messages] == ["user", "assistant", "tool", "assistant"]
    assert [spec.name for spec in model.requests[0].tools] == ["get_weather"]
    tool_message = model.requests[1].messages[-1]
    assert tool_message.tool_results[0].content == "Sunny in Athens"
    assert tool_message.tool_results[0].is_error is False
    assert result.usage.requests == 2


async def test_mistakes_of_the_model_are_reported_back_not_raised():
    broken = ToolCallPart(id="c1", name="get_weather", raw_arguments="{city:", parse_error="bad")
    model = ScriptedModel(
        [
            [broken, call("no_such_tool"), call("get_weather", town="Athens")],
            "Sorry about that.",
        ]
    )
    result = await Agent(model, tools=[get_weather]).run("Weather?")

    json_error, unknown, invalid = result.messages[2].tool_results
    assert all(part.is_error for part in (json_error, unknown, invalid))
    assert "not a valid JSON object" in json_error.content
    assert "no tool named 'no_such_tool'" in unknown.content and "get_weather" in unknown.content
    assert "city" in invalid.content and "town" in invalid.content
    assert result.output == "Sorry about that."


async def test_tool_failures_become_error_results():
    @tool
    def refund(order: str) -> str:
        """Refund an order."""
        raise ToolError(f"Order {order} was already refunded.")

    @tool
    def crash() -> str:
        """Break."""
        raise RuntimeError("database is down")

    model = ScriptedModel([[call("refund", order="A1"), call("crash")], "Done."])
    result = await Agent(model, tools=[refund, crash]).run("go")

    refund_result, crash_result = result.messages[2].tool_results
    assert refund_result.content == "Order A1 was already refunded." and refund_result.is_error
    assert crash_result.content == "'crash' failed: RuntimeError: database is down"
    assert crash_result.metadata == {"exception": "RuntimeError"}


async def test_read_only_tools_run_together_and_the_rest_in_order():
    order: list[str] = []
    lookups_started = set()
    all_lookups_started = asyncio.Event()

    @tool(read_only=True)
    async def lookup(key: str) -> str:
        """Look something up."""
        lookups_started.add(key)
        if len(lookups_started) == 3:
            all_lookups_started.set()
        await asyncio.wait_for(all_lookups_started.wait(), timeout=3)
        return key

    @tool
    async def write(key: str) -> str:
        """Change something."""
        order.append(f"start {key}")
        await asyncio.sleep(0.05)
        order.append(f"end {key}")
        return key

    model = ScriptedModel(
        [
            [call("lookup", key="a"), call("lookup", key="b"), call("lookup", key="c")],
            [call("write", key="x"), call("write", key="y")],
            "Done.",
        ]
    )
    result = await Agent(model, tools=[lookup, write]).run("go")
    assert lookups_started == {"a", "b", "c"}
    assert order == ["start x", "end x", "start y", "end y"]
    assert [part.content for part in result.messages[2].tool_results] == ["a", "b", "c"]


async def test_limits_stop_the_run_and_say_why():
    looping = FunctionModel(lambda request: call("get_weather", city="Athens"))
    result = await Agent(looping, tools=[get_weather], limits=Limits(max_turns=3)).run("go")
    assert (result.status, result.stop_reason, result.turns) == ("limit_reached", "max_turns", 3)
    assert result.output is None
    assert result.messages[-1].role == "tool"  # the history is still valid to continue from

    costly = FunctionModel(
        lambda request: call("get_weather", city="Athens"),
        usage=Usage(requests=1, input_tokens=600, output_tokens=100, cost_usd=0.02),
    )
    by_tokens = await Agent(costly, tools=[get_weather], limits=Limits(max_total_tokens=1000)).run(
        "go"
    )
    assert (by_tokens.stop_reason, by_tokens.turns) == ("max_total_tokens", 2)
    costly = FunctionModel(
        lambda request: call("get_weather", city="Athens"),
        usage=Usage(requests=1, input_tokens=600, output_tokens=100, cost_usd=0.02),
    )
    by_cost = await Agent(costly, tools=[get_weather], limits=Limits(max_cost_usd=0.05)).run("go")
    assert (by_cost.stop_reason, by_cost.turns) == ("max_cost_usd", 3)
    assert by_cost.usage.cost_usd == pytest.approx(0.06)


async def test_run_timeout_stops_a_slow_tool():
    @tool
    async def slow() -> str:
        """Take a while."""
        await asyncio.sleep(5)
        return "finished"

    model = ScriptedModel([call("slow")])
    result = await Agent(model, tools=[slow], limits=Limits(timeout_s=0.1)).run("go")
    assert (result.status, result.stop_reason) == ("limit_reached", "timeout")
    # It had started and is not safe to repeat: the model is told the outcome is unknown.
    assert result.messages[-1].tool_results[0].content == prompts.TOOL_INTERRUPTED.format(
        name="slow"
    )


class Invoice(BaseModel):
    number: str
    total: float


async def test_structured_output_is_validated_and_retried():
    model = ScriptedModel(
        [
            'Here it is: {"number": "INV-7"}',
            '```json\n{"number": "INV-7", "total": 12.5}\n```',
        ]
    )
    result = await Agent(model, output_type=Invoice).run("Extract the invoice")

    assert result.output == Invoice(number="INV-7", total=12.5)
    assert model.requests[0].output.json_schema["required"] == ["number", "total"]
    retry_prompt = model.requests[1].messages[-1]
    assert retry_prompt.role == "user" and "total" in retry_prompt.text
    assert retry_prompt.text.startswith("Your reply could not be used as the final answer")


async def test_structured_output_gives_up_after_the_retries():
    model = ScriptedModel(["nope", "still nope", "no"])
    with pytest.raises(OutputValidationError, match="after 3 attempts"):
        await Agent(model, output_type=Invoice).run("Extract")


async def test_prompted_output_mode_writes_the_schema_into_the_instructions():
    model = ScriptedModel(['{"number": "1", "total": 2}'])
    agent = Agent(model, instructions="Extract.", output_type=Invoice, output_mode="prompted")
    result = await agent.run("go")

    assert result.output.total == 2
    request = model.requests[0]
    assert request.output is None
    assert request.messages[0].text.startswith("Extract.\n\nReply with a single JSON value")
    assert '"number"' in request.messages[0].text


async def test_stream_yields_events_in_order():
    model = ScriptedModel([call("get_weather", city="Athens"), "Sunny today."])
    stream = Agent(model, name="helper", tools=[get_weather]).stream("Weather?")
    events = [event async for event in stream]

    assert [event.type for event in events] == [
        "run_started",
        "turn_started",
        "model_responded",
        "tool_call_started",
        "tool_call_finished",
        "turn_started",
        "text_delta",
        "text_delta",
        "model_responded",
        "run_finished",
    ]
    assert events[0].agent == "helper"
    assert events[4].result.content == "Sunny in Athens"
    assert "".join(e.text for e in events if e.type == "text_delta") == "Sunny today."
    assert stream.result is events[-1].result
    assert events[-1].model_dump(mode="json")["result"]["status"] == "completed"


async def test_text_helper_streams_only_the_answer():
    stream = Agent(ScriptedModel(["One two three."])).stream("count")
    assert [piece async for piece in stream.text()] == ["One ", "two ", "three."]


async def test_cancel_keeps_the_partial_answer_marked_as_interrupted():
    model = ScriptedModel(["one two three four five six seven eight"], delay=0.02)
    agent = Agent(model)
    stream = agent.stream("count")
    seen = 0
    async for event in stream:
        if event.type == "text_delta":
            seen += 1
            if seen == 3:
                stream.cancel("barge_in")
    result = stream.result

    assert (result.status, result.stop_reason) == ("cancelled", "barge_in")
    last = result.messages[-1]
    assert last.role == "assistant" and last.metadata == {"interrupted": True}
    assert last.text.startswith("one two three") and "eight" not in last.text

    # The conversation continues from the interrupted answer.
    model._replies.append("Sure, stopping.")
    follow_up = await agent.run("stop please", session=result.session)
    assert [m.role for m in follow_up.messages] == ["user", "assistant", "user", "assistant"]


async def test_hooks_see_and_can_change_each_step():
    seen: list[str] = []

    class Audit(Hooks):
        async def on_run_start(self, ctx):
            seen.append("start")

        async def before_model(self, ctx, request):
            seen.append(f"model:{len(request.messages)}")
            settings = request.settings.model_copy(update={"temperature": 0.0})
            return request.model_copy(update={"settings": settings})

        async def before_tool(self, ctx, call):
            seen.append(f"tool:{call.name}:{ctx.tool_call_id == call.id}")
            if call.arguments["city"] == "Atlantis":
                return ToolResultPart(
                    tool_call_id=call.id, name=call.name, content="No such place.", is_error=True
                )
            return call.model_copy(update={"arguments": {"city": call.arguments["city"].title()}})

        async def after_tool(self, ctx, call, result):
            return result.model_copy(update={"content": result.content + "!"})

        async def on_event(self, ctx, event):
            if event.type == "run_finished":
                seen.append("event:run_finished")

        async def on_run_end(self, ctx, result, error):
            seen.append(f"end:{result.status}")

    model = ScriptedModel(
        [[call("get_weather", city="athens"), call("get_weather", city="Atlantis")], "ok"]
    )
    result = await Agent(model, tools=[get_weather], hooks=[Audit()]).run("go")

    assert model.requests[0].settings.temperature == 0.0
    first, second = result.messages[2].tool_results
    assert first.content == "Sunny in Athens!"
    assert second.content == "No such place.!" and second.is_error
    assert seen == [
        "start", "model:1", "tool:get_weather:True", "tool:get_weather:True",
        "model:3", "end:completed", "event:run_finished",
    ]  # fmt: skip


async def test_a_failing_hook_stops_the_run():
    class Broken(Hooks):
        async def before_tool(self, ctx, call):
            raise PermissionError("policy service unreachable")

    model = ScriptedModel([call("get_weather", city="Athens")])
    with pytest.raises(PermissionError, match="policy service unreachable"):
        await Agent(model, tools=[get_weather], hooks=[Broken()]).run("go")


async def test_instructions_can_depend_on_the_run():
    async def instructions(ctx: RunContext[dict]) -> str:
        return f"The user is {ctx.deps['name']}."

    model = ScriptedModel(["Hi Maria."])
    await Agent(model, instructions=instructions).run("hello", deps={"name": "Maria"})
    assert model.requests[0].messages[0].text == "The user is Maria."


async def test_settings_are_merged_per_run():
    model = ScriptedModel(["a"])
    agent = Agent(model, settings=ModelSettings(temperature=0.2, max_tokens=100, extra={"n": 1}))
    await agent.run("go", settings=ModelSettings(max_tokens=50, extra={"user": "u1"}))
    sent = model.requests[0].settings
    assert (sent.temperature, sent.max_tokens) == (0.2, 50)
    assert sent.extra == {"n": 1, "user": "u1"}


async def test_safe_model_errors_are_retried_and_ambiguous_ones_are_not():
    limited = ModelRateLimitError("slow down", retryable=True, retry_after=0)
    model = ScriptedModel([limited, "ok"])
    assert (await Agent(model).run("go")).output == "ok"
    assert len(model.requests) == 2

    unsure = ModelServerError("upstream reset", retryable=True, ambiguous=True)
    model = ScriptedModel([unsure, "ok"])
    with pytest.raises(ModelServerError):
        await Agent(model).run("go")
    assert len(model.requests) == 1

    model = ScriptedModel([unsure, "ok"])
    opted_in = Agent(model, limits=Limits(retry_ambiguous=True))
    assert (await opted_in.run("go")).output == "ok"


async def test_long_tool_results_are_shortened_in_the_middle():
    @tool
    def dump() -> str:
        """Return a lot."""
        return "A" * 500 + "B" * 500

    model = ScriptedModel([call("dump"), "ok"])
    result = await Agent(model, tools=[dump], limits=Limits(max_tool_result_chars=90)).run("go")
    content = result.messages[2].tool_results[0].content
    assert content.startswith("A" * 60) and content.endswith("B" * 30)
    assert "[... 910 characters omitted ...]" in content


async def test_history_seeds_a_conversation():
    model = ScriptedModel(["Athens."])
    earlier = [Message.user("My city is Athens."), Message.assistant("Noted.")]
    result = await Agent(model).run("Which city?", history=earlier)
    assert [m.role for m in model.requests[0].messages] == ["user", "assistant", "user"]
    assert len(result.new_messages) == 2 and len(result.messages) == 4


async def test_configuration_mistakes_are_caught_early():
    with pytest.raises(ConfigurationError, match="provider:model-id"):
        Agent("just-a-name")
    with pytest.raises(ConfigurationError, match="Unknown model provider"):
        Agent("nowhere:model")
    with pytest.raises(ConfigurationError, match="not a tool"):
        Agent(ScriptedModel([]), tools=[42])
    with pytest.raises(ConfigurationError, match="Nothing to run"):
        await Agent(ScriptedModel([])).run()
    with pytest.raises(ConfigurationError, match="two tools named"):
        await Agent(
            ScriptedModel([]), tools=[get_weather, tool(lambda: "x", name="get_weather")]
        ).run("go")


def test_run_sync():
    assert Agent(ScriptedModel(["hi"])).run_sync("hello").output == "hi"


def test_every_framework_string_is_a_plain_attribute():
    assert "{name" in prompts.TOOL_FAILED and isinstance(prompts.OUTPUT_INSTRUCTIONS, str)
