from __future__ import annotations

import asyncio

import pytest

from qagents import Agent, InMemoryStore, Limits, Session, Usage, Workflow, tool
from qagents.errors import ConfigurationError
from qagents.models.testing import FunctionModel, ScriptedModel, call


@tool(read_only=True)
def search(query: str) -> str:
    """Search the knowledge base."""
    return f"3 results for {query}"


async def test_an_agent_as_a_tool_works_in_its_own_conversation():
    one_request = Usage(requests=1, input_tokens=10, output_tokens=5)
    researcher_model = ScriptedModel(
        [call("search", query="solar"), "Solar grew 20% in 2025."], usage=one_request
    )
    researcher = Agent(
        researcher_model,
        name="researcher",
        description="Finds facts in the knowledge base.",
        instructions="Research thoroughly.",
        tools=[search],
    )
    lead_model = ScriptedModel(
        [call("researcher", task="How fast did solar grow in 2025?"), "It grew 20%."],
        usage=one_request,
    )
    lead = Agent(lead_model, name="lead", tools=[researcher])

    stream = lead.stream("Tell me about solar growth.")
    events = [event async for event in stream]
    result = stream.result

    assert result.output == "It grew 20%."
    spec = lead_model.requests[0].tools[0]
    assert (spec.name, spec.description) == ("researcher", "Finds facts in the knowledge base.")
    assert list(spec.parameters["properties"]) == ["task"]
    # The sub-agent saw only the task, not the lead's conversation.
    assert [m.text for m in researcher_model.requests[0].messages] == [
        "Research thoroughly.",
        "How fast did solar grow in 2025?",
    ]
    # The lead saw only the sub-agent's answer.
    assert result.messages[2].tool_results[0].content == "Solar grew 20% in 2025."
    assert (
        result.usage.requests == 4 and result.usage.input_tokens == 40
    )  # all four requests counted
    nested = [e.event.type for e in events if e.type == "subagent_event"]
    assert nested[:4] == ["run_started", "turn_started", "model_responded", "tool_call_started"]
    assert "run_finished" not in nested  # its result is the tool result
    assert {e.agent for e in events if e.type == "subagent_event"} == {"researcher"}


async def test_a_subagent_that_stops_early_is_reported_as_a_tool_error():
    looping = FunctionModel(lambda request: call("search", query="x"))
    worker = Agent(looping, name="worker", tools=[search], limits=Limits(max_turns=2))
    lead = Agent(ScriptedModel([call("worker", task="dig"), "It did not finish."]), tools=[worker])
    result = await lead.run("go")
    failure = result.messages[2].tool_results[0]
    assert failure.is_error and failure.content == "worker stopped before finishing (max_turns)."


async def test_handoff_switches_the_active_agent():
    billing_model = ScriptedModel(["Your invoice total is 40 EUR."])
    billing = Agent(
        billing_model,
        name="billing",
        description="Handles invoices and payments.",
        instructions="You are the billing specialist.",
    )
    triage_model = ScriptedModel([call("transfer_to_billing")])
    triage = Agent(
        triage_model, name="triage", instructions="Route the customer.", handoffs=[billing]
    )

    session = Session()
    stream = triage.stream("How much is my invoice?", session=session)
    events = [event async for event in stream]
    result = stream.result

    handoff_spec = triage_model.requests[0].tools[0]
    assert handoff_spec.name == "transfer_to_billing"
    assert (
        handoff_spec.description
        == "Hand the conversation over to billing. Handles invoices and payments."
    )
    switched = next(e for e in events if e.type == "agent_switched")
    assert (switched.from_agent, switched.to_agent) == ("triage", "billing")
    handoff_call = next(m for m in result.messages if m.role == "assistant").tool_calls[0]
    assert switched.tool_call_id == handoff_call.id  # the call that made the switch
    assert result.agent == "billing" and result.output == "Your invoice total is 40 EUR."
    # The specialist continues the same conversation under its own instructions.
    sent = billing_model.requests[0].messages
    assert sent[0].text == "You are the billing specialist."
    assert [m.role for m in sent] == ["system", "user", "assistant", "tool"]
    assert sent[-1].tool_results[0].content == "The conversation is now handled by billing."
    assert [m.name for m in result.messages if m.role == "assistant"] == ["triage", "billing"]
    assert "agent_switched" in [entry.kind for entry in await session.entries()]


async def test_a_run_resumes_with_the_agent_it_was_handed_to():
    @tool(requires_approval=True)
    def issue_refund(amount: float) -> str:
        """Refund the customer."""
        return f"refunded {amount}"

    billing = Agent(
        ScriptedModel([call("issue_refund", amount=15), "Refund issued."]),
        name="billing",
        tools=[issue_refund],
    )
    triage = Agent(ScriptedModel([call("transfer_to_billing")]), name="triage", handoffs=[billing])

    session = Session()
    paused = await triage.run("I want a refund", session=session)
    assert (paused.status, paused.agent) == ("paused", "billing")

    done = await triage.resume(session, approvals={paused.pending_approvals[0].tool_call_id: True})
    assert (done.status, done.agent, done.output) == ("completed", "billing", "Refund issued.")


async def test_only_one_handoff_per_turn_and_no_name_clashes():
    a = Agent(ScriptedModel(["from a"]), name="a")
    b = Agent(ScriptedModel(["from b"]), name="b")
    router = Agent(ScriptedModel([[call("transfer_to_a"), call("transfer_to_b")]]), handoffs=[a, b])
    result = await router.run("go")
    first, second = result.messages[2].tool_results
    assert first.content == "The conversation is now handled by a."
    assert second.is_error and "Only one handoff" in second.content
    assert result.output == "from a"

    clash = tool(lambda: "x", name="transfer_to_a")
    with pytest.raises(ConfigurationError, match="collide with handoffs"):
        await Agent(ScriptedModel([]), tools=[clash], handoffs=[a]).run("go")


async def test_workflow_steps_run_once_per_workflow_id():
    store = InMemoryStore()
    runs: list[str] = []

    async def fetch() -> dict:
        runs.append("fetch")
        return {"rows": 3}

    def publish() -> str:
        runs.append("publish")
        if runs.count("publish") == 1:
            raise RuntimeError("the process died here")
        return "published"

    async def job() -> str:
        workflow = Workflow(store, id="nightly-report")
        data = await workflow.step("fetch", fetch)
        return f"{data['rows']} rows {await workflow.step('publish', publish)}"

    with pytest.raises(RuntimeError):
        await job()
    assert await job() == "3 rows published"
    assert runs == ["fetch", "publish", "publish"]  # fetch was not repeated
    assert await job() == "3 rows published"
    assert len(runs) == 3  # a finished workflow replays entirely from the journal
    assert await Workflow(store, id="another").completed_steps() == {}


async def test_workflow_steps_can_run_in_parallel_and_restore_types():
    store = InMemoryStore()
    workflow = Workflow(store, id="fan-out")

    async def measure(name: str) -> Usage:
        await asyncio.sleep(0.05)
        return Usage(requests=1, input_tokens=len(name))

    first, second = await asyncio.gather(
        workflow.step("a", lambda: measure("alpha")), workflow.step("b", lambda: measure("be"))
    )
    assert (first.input_tokens, second.input_tokens) == (5, 2)

    replay = Workflow(store, id="fan-out")
    assert await replay.step("a", lambda: 1 / 0) == {
        "requests": 1, "input_tokens": 5, "output_tokens": 0,
        "cached_input_tokens": 0, "reasoning_tokens": 0, "cost_usd": None,
    }  # fmt: skip
    assert await replay.step("b", lambda: 1 / 0, type=Usage) == Usage(requests=1, input_tokens=2)


async def test_workflow_agent_steps_resume_instead_of_restarting():
    store = InMemoryStore()
    log: list[str] = []

    @tool(requires_approval=True)
    def send_newsletter(subject: str) -> str:
        """Send the newsletter to all subscribers."""
        log.append(subject)
        return "sent to 1200 subscribers"

    def writer() -> Agent:
        return Agent(
            ScriptedModel([call("send_newsletter", subject="November news"), "Newsletter sent."]),
            name="writer",
            tools=[send_newsletter],
        )

    workflow = Workflow(store, id="newsletter-2026-11")
    paused = await workflow.run_agent("send", writer(), "Send the November newsletter")
    assert paused.status == "paused" and await workflow.completed_steps() == {}

    # Later, in another process: the same call continues the paused run.
    later = Workflow(store, id="newsletter-2026-11")
    approvals = {paused.pending_approvals[0].tool_call_id: True}
    agent = Agent(ScriptedModel(["Newsletter sent."]), name="writer", tools=[send_newsletter])
    done = await later.run_agent("send", agent, "Send the November newsletter", approvals=approvals)
    assert done.status == "completed" and log == ["November news"]

    # Once finished, the step is answered from the journal without touching the agent.
    again = await Workflow(store, id="newsletter-2026-11").run_agent(
        "send", Agent(ScriptedModel([]), name="writer"), "Send the November newsletter"
    )
    assert again.output == "Newsletter sent." and log == ["November news"]
