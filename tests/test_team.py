"""Team communication through real linked child agent runs."""

import asyncio
import json

import pytest

from qagents import Agent, RunContext, tool
from qagents.errors import ConfigurationError, ToolError
from qagents.models.testing import FunctionModel, call
from qagents.team import TeamMailbox


async def test_coordinator_and_two_child_agents_exchange_messages():
    box = TeamMailbox(["lead", "researcher", "reviewer"])
    lead, research, review = (box.endpoint(name) for name in ("lead", "researcher", "reviewer"))

    def transcript(request):
        return [part for message in request.messages for part in message.tool_results]

    def research_reply(request):
        results = transcript(request)
        if not results:
            return call("receive_messages", wait_s=1)
        if len(results) == 1:
            assert json.loads(results[0].content)[0]["sender"] == "lead"
            return call("send_message", recipient="reviewer", text="Evidence A")
        return "Research finished."

    def review_reply(request):
        results = transcript(request)
        if not results:
            return call("receive_messages", wait_s=1)
        if len(results) == 1:
            assert json.loads(results[0].content)[0]["sender"] == "researcher"
            return call("send_message", recipient="lead", text="Evidence A verified")
        return "Review finished."

    researcher = Agent(FunctionModel(research_reply), name="researcher", tools=research.tools())
    reviewer = Agent(FunctionModel(review_reply), name="reviewer", tools=review.tools())

    @tool
    async def run_team(ctx: RunContext) -> None:
        """Start the two linked child agents."""
        async with asyncio.TaskGroup() as group:
            group.create_task(researcher.run("Research the question", parent=ctx))
            group.create_task(reviewer.run("Review the evidence", parent=ctx))

    def lead_reply(request):
        results = transcript(request)
        if not results:
            return call("send_message", recipient="researcher", text="Find one fact")
        if len(results) == 1:
            return call("run_team")
        if len(results) == 2:
            return call("receive_messages")
        message = json.loads(results[-1].content)[0]
        assert message["sender"] == "reviewer"
        return message["text"]

    parent = Agent(
        FunctionModel(lead_reply),
        name="lead",
        tools=[run_team, *lead.tools()],
    )
    assert (await parent.run("Check the evidence")).text == "Evidence A verified"


async def test_team_mailbox_enforces_bounds_and_fifo():
    box = TeamMailbox(["lead", "worker"], capacity=2, max_text_chars=4)
    lead, worker = box.endpoint("lead"), box.endpoint("worker")
    assert await worker.receive(wait_s=0.01) == []
    first, second = await lead.send("worker", "one"), await lead.send("worker", "two")
    with pytest.raises(ToolError, match="full"):
        await lead.send("worker", "more")
    assert await lead.receive() == []
    assert await worker.receive(limit=1) == [first]
    assert await worker.receive() == [second]
    with pytest.raises(ToolError, match="Unknown teammate"):
        await worker.send("stranger", "hi")
    with pytest.raises(ToolError, match="characters"):
        await worker.send("lead", "longer")
    with pytest.raises(ConfigurationError, match="unique"):
        TeamMailbox(["lead", "lead"])


async def test_many_agents_exchange_messages_without_crossing_team_boundaries():
    workers = [f"worker-{index}" for index in range(20)]
    mailbox = TeamMailbox(["lead", *workers])
    lead = mailbox.endpoint("lead")
    other_lead = TeamMailbox(["lead", *workers]).endpoint("lead")

    async def work(member):
        endpoint = mailbox.endpoint(member)
        (task,) = await endpoint.receive(wait_s=1)
        assert task.sender == "lead"
        await endpoint.send("lead", task.text)

    async with asyncio.TaskGroup() as group:
        for worker in workers:
            group.create_task(work(worker))
        for worker in workers:
            await lead.send(worker, worker)
    replies = await lead.receive(limit=len(workers))
    assert {message.sender: message.text for message in replies} == {
        worker: worker for worker in workers
    }
    assert await other_lead.receive() == []
