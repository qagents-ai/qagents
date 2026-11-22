"""Input and output guardrails."""

from __future__ import annotations

import re

import pytest
from pydantic import BaseModel

from qagents import Agent, Message, Session
from qagents.errors import ConfigurationError, GuardrailTripped
from qagents.models.testing import ScriptedModel, call
from qagents.results import TrippedGuardrail

ACCOUNT = re.compile(r"\bGR\d{25}\b")


def no_account_numbers(ctx, message: Message) -> None:
    if ACCOUNT.search(message.text):
        raise GuardrailTripped("the prompt contains an account number")


async def test_an_input_guardrail_blocks_the_prompt_before_any_model_request():
    model = ScriptedModel([])  # any request would fail the test
    agent = Agent(model, input_guardrails=[no_account_numbers])
    session = Session()

    stream = agent.stream("Pay GR1601101250000000012300695 now", session=session)
    events = [event async for event in stream]
    result = stream.result

    assert (result.status, result.stop_reason) == ("blocked", "no_account_numbers")
    assert result.guardrail == TrippedGuardrail(
        name="no_account_numbers", reason="the prompt contains an account number", stage="input"
    )
    assert result.output is None and result.text == "" and result.turns == 0
    assert model.requests == []
    assert [e.type for e in events] == ["run_started", "guardrail_tripped", "run_finished"]
    tripped = events[1]
    assert (tripped.name, tripped.stage, tripped.agent) == ("no_account_numbers", "input", "agent")

    # The prompt never joined the conversation; the journal records the trip.
    assert await session.messages() == []
    entries = [(entry.kind, entry.data) for entry in await session.entries()]
    assert entries[1] == (
        "guardrail_tripped",
        {
            "name": "no_account_numbers",
            "reason": "the prompt contains an account number",
            "stage": "input",
        },
    )
    assert entries[-1] == (
        "run_finished",
        {"status": "blocked", "stop_reason": "no_account_numbers", "output": None, "text": ""},
    )


async def test_the_session_continues_after_a_blocked_prompt():
    model = ScriptedModel(["Hello Maria."])
    agent = Agent(model, input_guardrails=[no_account_numbers])
    session = Session()

    blocked = await agent.run("My IBAN is GR1601101250000000012300695", session=session)
    assert blocked.status == "blocked"
    done = await agent.run("My name is Maria.", session=session)

    assert (done.status, done.output) == ("completed", "Hello Maria.")
    assert [m.text for m in model.requests[0].messages] == ["My name is Maria."]
    assert (await session.state()).open_run_id is None


async def test_passing_guardrails_run_in_order_and_may_be_async_and_named():
    seen: list[str] = []

    async def log_prompt(ctx, message: Message) -> None:
        seen.append(f"{ctx.agent_name}: {message.text}")

    def topic_check(ctx, message: Message) -> None:
        if "weather" not in message.text:
            raise GuardrailTripped("off topic", name="topic")

    agent = Agent(ScriptedModel(["Sunny."]), input_guardrails=[log_prompt, topic_check])
    assert (await agent.run("the weather?")).output == "Sunny."
    blocked = await agent.run("tell me a joke")
    assert (blocked.stop_reason, blocked.guardrail.reason) == ("topic", "off topic")
    assert seen == ["agent: the weather?", "agent: tell me a joke"]


async def test_input_guardrails_check_only_a_new_prompt():
    calls: list[str] = []

    def record(ctx, message: Message) -> None:
        calls.append(message.text)

    model = ScriptedModel(["one", "two"])
    agent = Agent(model, input_guardrails=[record])
    await agent.run(history=[Message.user("seeded")])  # no new prompt: nothing to check
    assert calls == []
    await agent.run("new prompt")
    assert calls == ["new prompt"]


async def test_a_guardrail_that_breaks_fails_the_run_closed():
    def broken(ctx, message: Message) -> None:
        raise RuntimeError("classifier unavailable")

    model = ScriptedModel([])
    session = Session()
    with pytest.raises(RuntimeError, match="classifier unavailable"):
        await Agent(model, input_guardrails=[broken]).run("hello", session=session)
    assert model.requests == [] and await session.messages() == []
    # The start of a run is written in one piece after the input guardrails ran, so a
    # guardrail that breaks leaves nothing behind: no open run without its prompt.
    assert await session.entries() == []

    def returns_false(ctx, message: Message) -> bool:
        return False  # looks like "block", but guardrails block by raising

    with pytest.raises(ConfigurationError, match="raises GuardrailTripped"):
        await Agent(ScriptedModel([]), input_guardrails=[returns_false]).run("hello")


async def test_an_output_guardrail_blocks_the_answer_and_keeps_it_for_audit():
    def no_secrets(ctx, output: str) -> None:
        if "password" in output:
            raise GuardrailTripped("the answer reveals a password")

    model = ScriptedModel(["The password is hunter2."])
    session = Session()
    stream = Agent(model, output_guardrails=[no_secrets]).stream("What is it?", session=session)
    events = [event async for event in stream]
    result = stream.result

    assert (result.status, result.stop_reason) == ("blocked", "no_secrets")
    assert result.output is None and result.text == ""
    assert result.guardrail.stage == "output"
    assert [e.type for e in events][-3:] == ["model_responded", "guardrail_tripped", "run_finished"]
    # The answer stays in the journal, where an auditor can find it.
    assert (await session.messages())[-1].text == "The password is hunter2."
    kinds = [entry.kind for entry in await session.entries()]
    assert kinds[-2:] == ["guardrail_tripped", "run_finished"]


async def test_output_guardrails_see_the_validated_output_and_survive_a_handoff():
    class Refund(BaseModel):
        amount: float

    seen: list[object] = []

    def at_most_100(ctx, output: Refund) -> None:
        seen.append(output)
        if output.amount > 100:
            raise GuardrailTripped("refund above the limit")

    billing = Agent(
        ScriptedModel(['{"amount": 250}']), name="billing", output_type=Refund
    )  # has no guardrail of its own
    triage = Agent(
        ScriptedModel([call("transfer_to_billing")]),
        name="triage",
        handoffs=[billing],
        output_guardrails=[at_most_100],
    )
    result = await triage.run("Refund me")
    assert (result.status, result.agent, result.stop_reason) == (
        "blocked",
        "billing",
        "at_most_100",
    )
    assert seen == [Refund(amount=250)]


async def test_a_guardrail_listed_by_two_agents_runs_once():
    calls: list[str] = []

    def count(ctx, output: str) -> None:
        calls.append(output)

    worker = Agent(ScriptedModel(["done"]), name="worker", output_guardrails=[count])
    lead = Agent(
        ScriptedModel([call("transfer_to_worker")]),
        name="lead",
        handoffs=[worker],
        output_guardrails=[count],
    )
    assert (await lead.run("go")).output == "done"
    assert calls == ["done"]  # listed by both agents, run once
