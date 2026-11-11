"""Final answers: the final-answer tool mode, output validators, provider refusals."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from qagents import Agent, Limits, Message, Session, prompts, tool
from qagents.errors import ConfigurationError, ModelRetry, OutputValidationError
from qagents.models.base import ModelResponse
from qagents.models.testing import ScriptedModel, call


class Invoice(BaseModel):
    number: str
    total: float


def weather_tool(log: list[str]):
    @tool(read_only=True)
    def get_weather(city: str) -> str:
        """Look up the weather for a city."""
        log.append(city)
        return f"Sunny in {city}"

    return get_weather


# -- output_mode="tool" ------------------------------------------------------------


async def test_the_final_answer_tool_ends_the_run_with_the_validated_output():
    model = ScriptedModel([call("final_answer", number="INV-7", total=12.5)])
    agent = Agent(model, instructions="Extract.", output_type=Invoice, output_mode="tool")
    session = Session()

    stream = agent.stream("Extract the invoice", session=session)
    events = [event async for event in stream]
    result = stream.result

    assert (result.status, result.stop_reason) == ("completed", None)
    assert result.output == Invoice(number="INV-7", total=12.5)
    assert len(model.requests) == 1
    request = model.requests[0]
    assert request.output is None and request.settings.tool_choice is None  # never forced
    spec = request.tools[-1]
    assert spec.name == "final_answer" and spec.description == prompts.OUTPUT_TOOL_DESCRIPTION
    assert spec.parameters["required"] == ["number", "total"]
    assert "title" not in spec.parameters
    assert request.messages[0].text == "Extract.\n\n" + prompts.OUTPUT_TOOL_INSTRUCTIONS.format(
        name="final_answer"
    )
    # The tool turn is closed, so the conversation stays valid for the next run.
    assert [m.role for m in result.messages] == ["user", "assistant", "tool"]
    accepted = result.messages[-1].tool_results[0]
    assert (accepted.content, accepted.is_error) == (prompts.OUTPUT_TOOL_ACCEPTED, False)
    finished = [e for e in events if e.type == "tool_call_finished"]
    assert [e.result.content for e in finished] == [prompts.OUTPUT_TOOL_ACCEPTED]
    assert "tool_call_started" not in [e.type for e in events]  # not a tool: no tool hooks
    state = await session.state()
    assert state.open_run_id is None and not state.unresolved_tool_calls


async def test_other_calls_in_the_final_answer_turn_are_not_run():
    log: list[str] = []
    model = ScriptedModel(
        [[call("get_weather", city="Athens"), call("final_answer", number="1", total=2)]]
    )
    agent = Agent(model, tools=[weather_tool(log)], output_type=Invoice, output_mode="tool")
    result = await agent.run("go")

    assert result.output == Invoice(number="1", total=2) and log == []
    skipped, accepted = result.messages[-1].tool_results
    assert skipped.is_error and skipped.content == prompts.OUTPUT_TOOL_SKIPPED.format(
        name="get_weather"
    )
    assert accepted.content == prompts.OUTPUT_TOOL_ACCEPTED


async def test_an_invalid_final_answer_becomes_an_error_result_and_the_turn_goes_on():
    log: list[str] = []
    model = ScriptedModel(
        [
            [call("get_weather", city="Athens"), call("final_answer", number="INV-7")],
            call("final_answer", number="INV-7", total=3),
        ]
    )
    agent = Agent(model, tools=[weather_tool(log)], output_type=Invoice, output_mode="tool")
    result = await agent.run("go")

    assert result.output == Invoice(number="INV-7", total=3)
    assert log == ["Athens"]  # the answer was rejected, so the other call ran
    weather, rejected = model.requests[1].messages[-1].tool_results
    assert weather.content == "Sunny in Athens"
    assert rejected.is_error
    assert rejected.content.startswith("The final answer was not accepted: - total: Field required")
    assert rejected.content.endswith("Call final_answer again with a corrected answer.")


async def test_a_plain_text_answer_is_sent_back_with_a_request_for_the_call():
    model = ScriptedModel(["The total is 3.", call("final_answer", number="A", total=3)])
    agent = Agent(model, output_type=Invoice, output_mode="tool")
    result = await agent.run("go")

    assert result.output == Invoice(number="A", total=3)
    retry = model.requests[1].messages[-1]
    assert retry.role == "user"
    assert retry.text == prompts.OUTPUT_TOOL_REQUIRED.format(name="final_answer")


async def test_tool_mode_gives_up_after_the_output_retries():
    model = ScriptedModel(["no", call("final_answer", number="A")])
    agent = Agent(model, output_type=Invoice, output_mode="tool", limits=Limits(output_retries=1))
    with pytest.raises(OutputValidationError, match="did not match the output type after 2"):
        await agent.run("go")

    model = ScriptedModel(["no", "still no"])
    agent = Agent(model, output_type=Invoice, output_mode="tool", limits=Limits(output_retries=1))
    with pytest.raises(OutputValidationError, match="was not given with final_answer after 2"):
        await agent.run("go")


async def test_a_non_object_output_is_the_value_argument_and_the_name_can_change():
    model = ScriptedModel([call("submit", value=[1, 2, 3])])
    agent = Agent(model, output_type=list[int], output_mode="tool", output_tool_name="submit")
    result = await agent.run("go")

    assert result.output == [1, 2, 3]
    parameters = model.requests[0].tools[-1].parameters
    assert parameters["required"] == ["value"] and parameters["additionalProperties"] is False
    assert parameters["properties"]["value"] == {"items": {"type": "integer"}, "type": "array"}

    text = Agent(ScriptedModel([call("final_answer", value="Hello.")]), output_mode="tool")
    assert (await text.run("go")).output == "Hello."  # plain text through the tool works too

    wrong = ScriptedModel([call("final_answer", answer="x"), call("final_answer", value="ok")])
    retried = await Agent(wrong, output_mode="tool").run("go")
    rejected = wrong.requests[1].messages[-1].tool_results[0]
    assert (
        "value: Field required" in rejected.content and "answer: Extra inputs" in rejected.content
    )
    assert retried.output == "ok"


async def test_a_second_final_answer_in_the_same_turn_is_ignored():
    model = ScriptedModel(
        [[call("final_answer", value="first"), call("final_answer", value="second")]]
    )
    result = await Agent(model, output_mode="tool").run("go")
    assert result.output == "first"
    first, second = result.messages[-1].tool_results
    assert first.content == prompts.OUTPUT_TOOL_ACCEPTED
    assert second.content == prompts.OUTPUT_TOOL_SKIPPED.format(name="final_answer")


async def test_after_a_handoff_the_target_agents_output_settings_apply():
    billing_model = ScriptedModel([call("final_answer", number="INV-9", total=40)])
    billing = Agent(billing_model, name="billing", output_type=Invoice, output_mode="tool")
    triage_model = ScriptedModel([call("transfer_to_billing")])
    triage = Agent(triage_model, name="triage", handoffs=[billing])

    result = await triage.run("What do I owe?")

    assert (result.agent, result.output) == ("billing", Invoice(number="INV-9", total=40))
    assert [spec.name for spec in triage_model.requests[0].tools] == ["transfer_to_billing"]
    assert [spec.name for spec in billing_model.requests[0].tools] == ["final_answer"]


async def test_final_answer_tool_configuration_mistakes_fail_early():
    with pytest.raises(ConfigurationError, match="output_mode must be"):
        Agent(ScriptedModel([]), output_mode="json")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="not a valid tool name"):
        Agent(ScriptedModel([]), output_mode="tool", output_tool_name="final answer")
    clash = tool(lambda: "x", name="final_answer")
    with pytest.raises(ConfigurationError, match="has the name of another tool"):
        await Agent(ScriptedModel([]), tools=[clash], output_mode="tool").run("go")


async def test_resume_after_an_accepted_answer_finishes_without_asking_again():
    failures = ["the classifier timed out"]

    def flaky_guardrail(ctx, output: Invoice) -> None:
        if failures:
            raise RuntimeError(failures.pop())

    model = ScriptedModel([call("final_answer", number="INV-7", total=1)])
    agent = Agent(
        model, output_type=Invoice, output_mode="tool", output_guardrails=[flaky_guardrail]
    )
    session = Session()
    with pytest.raises(RuntimeError, match="timed out"):
        await agent.run("go", session=session)

    result = await agent.resume(session)
    assert (result.status, result.output) == ("completed", Invoice(number="INV-7", total=1))
    assert len(model.requests) == 1


# -- output validators -----------------------------------------------------------------


async def test_validators_replace_the_output_in_order_sync_and_async():
    seen: list[str] = []

    def strip(ctx, output: str) -> str:
        seen.append(f"strip {ctx.agent_name}")
        return output.strip()

    async def shout(ctx, output: str) -> str:
        return output.upper()

    def check_only(ctx, output: str) -> None:
        seen.append(f"check {output}")

    agent = Agent(ScriptedModel(["  hello  "]), output_validators=[strip, shout, check_only])
    result = await agent.run("go")
    assert result.output == "HELLO" and seen == ["strip agent", "check HELLO"]
    assert result.text == "  hello  "  # the message itself is unchanged


async def test_model_retry_in_a_validator_sends_the_message_back_in_every_mode():
    def total_must_be_positive(ctx, invoice: Invoice) -> Invoice:
        if invoice.total <= 0:
            raise ModelRetry("The total must be positive; read it again.")
        return invoice

    def no_drafts(ctx, text: str) -> None:
        if "draft" in text:
            raise ModelRetry("Remove the word draft.")

    # Plain text.
    model = ScriptedModel(["a draft", "a reply"])
    result = await Agent(model, output_validators=[no_drafts]).run("go")
    assert result.output == "a reply"
    assert model.requests[1].messages[-1].text == prompts.OUTPUT_TEXT_RETRY.format(
        error="Remove the word draft."
    )

    # Native structured output.
    model = ScriptedModel(['{"number": "A", "total": 0}', '{"number": "A", "total": 5}'])
    agent = Agent(model, output_type=Invoice, output_validators=[total_must_be_positive])
    assert (await agent.run("go")).output.total == 5
    assert model.requests[1].messages[-1].text == prompts.OUTPUT_RETRY.format(
        error="The total must be positive; read it again."
    )

    # The final-answer tool: the rejection is the call's error result.
    model = ScriptedModel(
        [call("final_answer", number="A", total=0), call("final_answer", number="A", total=5)]
    )
    agent = Agent(
        model, output_type=Invoice, output_mode="tool", output_validators=[total_must_be_positive]
    )
    assert (await agent.run("go")).output.total == 5
    rejected = model.requests[1].messages[-1].tool_results[0]
    assert rejected.is_error and rejected.content == prompts.OUTPUT_TOOL_RETRY.format(
        error="The total must be positive; read it again.", name="final_answer"
    )


async def test_validator_rejections_count_toward_the_output_retries():
    def never(ctx, output):
        raise ModelRetry("no")

    agent = Agent(
        ScriptedModel(["a", "b"]), output_validators=[never], limits=Limits(output_retries=1)
    )
    with pytest.raises(OutputValidationError, match="rejected by an output validator after 2"):
        await agent.run("go")


async def test_other_exceptions_in_a_validator_fail_the_run():
    def broken(ctx, output):
        raise RuntimeError("validator bug")

    session = Session()
    with pytest.raises(RuntimeError, match="validator bug"):
        await Agent(ScriptedModel(["x"]), output_validators=[broken]).run("go", session=session)
    assert (await session.entries())[-1].kind == "run_failed"


async def test_model_retry_inside_a_tool_asks_for_a_corrected_call():
    @tool
    def book_room(day: str) -> str:
        """Book the meeting room."""
        if day != "friday":
            raise ModelRetry(f"{day} is taken. Free days: friday.")
        return "booked"

    model = ScriptedModel([call("book_room", day="monday"), call("book_room", day="friday"), "ok"])
    result = await Agent(model, tools=[book_room]).run("Book the room")
    first = model.requests[1].messages[-1].tool_results[0]
    assert first.is_error and first.content == "monday is taken. Free days: friday."
    assert model.requests[2].messages[-1].tool_results[0].content == "booked"
    assert result.output == "ok"


# -- provider refusals --------------------------------------------------------------------


def refusal(text: str = "I can't help with that.") -> ModelResponse:
    return ModelResponse(message=Message.assistant(text), finish_reason="content_filter")


async def test_a_content_filter_response_ends_the_run_without_retries():
    model = ScriptedModel([refusal()])
    result = await Agent(model).run("go")
    assert (result.status, result.stop_reason) == ("completed", "content_filter")
    assert result.output == "I can't help with that."

    for mode in ("native", "tool"):
        model = ScriptedModel([refusal()])
        result = await Agent(model, output_type=Invoice, output_mode=mode).run("go")
        assert (result.status, result.stop_reason, result.output) == (
            "completed",
            "content_filter",
            None,
        )
        assert len(model.requests) == 1  # no retry against a refusal


async def test_a_refusal_with_tool_calls_closes_them_without_running():
    log: list[str] = []
    message = Message.assistant(tool_calls=[call("get_weather", city="Athens")])
    model = ScriptedModel([ModelResponse(message=message, finish_reason="content_filter")])
    session = Session()
    result = await Agent(model, tools=[weather_tool(log)]).run("go", session=session)
    assert result.stop_reason == "content_filter" and log == []
    assert result.messages[-1].tool_results[0].content == prompts.TOOL_NOT_EXECUTED.format(
        name="get_weather"
    )
    assert not (await session.state()).unresolved_tool_calls
