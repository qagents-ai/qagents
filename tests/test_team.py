"""Team communication through real linked child agent runs."""

import asyncio
import json

import pytest

from qagents import Agent, Hooks, RunContext, Session, SQLiteStore, tool
from qagents.errors import ConfigurationError, ToolError
from qagents.models.testing import FunctionModel, ScriptedModel, call
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


async def test_sqlite_mailbox_reopens_with_unread_messages_and_the_same_sequence(tmp_path):
    path = tmp_path / "team.db"
    first = TeamMailbox(["lead", "worker"], db_path=path, capacity=2)
    lead = first.endpoint("lead")
    sent = await lead.send("worker", "first")

    reopened = TeamMailbox(["worker", "lead"], db_path=path, capacity=2)
    worker = reopened.endpoint("worker")
    assert await worker.receive() == [sent]
    assert await first.endpoint("worker").receive() == []
    answer = await worker.send("lead", "reply")

    fresh = TeamMailbox(["lead", "worker"], db_path=path, capacity=2)
    assert await fresh.endpoint("lead").receive() == [answer]
    assert answer.sequence == sent.sequence + 1
    with pytest.raises(ConfigurationError, match="configuration"):
        TeamMailbox(["lead", "worker"], db_path=path, capacity=1)


@pytest.mark.parametrize("name", ["send_message", "receive_messages"])
async def test_interrupted_mailbox_tool_replays_its_recorded_result(tmp_path, name):
    class Interrupted(Exception):
        pass

    class InterruptAfterTool(Hooks):
        async def after_tool(self, ctx, call, result):
            assert not result.is_error
            raise Interrupted

    db_path = tmp_path / "team.db"
    members = ["lead", "worker"]
    mailbox = TeamMailbox(members, db_path=db_path)
    lead, worker = (mailbox.endpoint(member) for member in members)
    if name == "send_message":
        original = call(name, recipient="worker", text="first")
        active = lead
    else:
        await lead.send("worker", "first")
        original = call(name)
        active = worker
    async with SQLiteStore(tmp_path / "run.db") as store:
        session = Session(store, id="resume")
        first = Agent(
            ScriptedModel([original]),
            name=active.member,
            tools=active.tools(),
            hooks=[InterruptAfterTool()],
        )
        with pytest.raises(Interrupted):
            await first.run("Use the mailbox", session=session)

        # Consume the original send before retrying, or append a new message
        # after the successful receive. A retry must not touch the new inbox.
        if name == "send_message":
            (delivered,) = await worker.receive()
        else:
            await lead.send("worker", "second")

        reopened = TeamMailbox(members, db_path=db_path)
        resumed = Agent(
            ScriptedModel(["Finished"]),
            name=active.member,
            tools=reopened.endpoint(active.member).tools(),
        )
        result = await resumed.resume(session)
        assert result.completed
        (reply,) = [
            part
            for message in result.messages
            for part in message.tool_results
            if part.name == name
        ]
        assert not reply.is_error
        saved = json.loads(reply.content)
        if name == "send_message":
            assert saved["text"] == "first" and saved["sequence"] == delivered.sequence
            assert await worker.receive() == []
        else:
            assert [message["text"] for message in saved] == ["first"]
            assert [message.text for message in await worker.receive()] == ["second"]


async def test_sqlite_mailbox_arbitrates_capacity_and_concurrent_reads(tmp_path):
    members = ["lead", "worker"]
    first = TeamMailbox(members, db_path=tmp_path / "team.db", capacity=4)
    second = TeamMailbox(members, db_path=tmp_path / "team.db", capacity=4)
    leads = [first.endpoint("lead"), second.endpoint("lead")]
    sent = await asyncio.gather(
        *(leads[i % 2].send("worker", f"message {i}") for i in range(10)),
        return_exceptions=True,
    )
    full = [item for item in sent if isinstance(item, ToolError)]
    assert len(full) == 6 and all("full" in str(error) for error in full)
    queued = [item for item in sent if not isinstance(item, BaseException)]
    workers = [mailbox.endpoint("worker") for mailbox in (first, second)]
    batches = await asyncio.gather(*(worker.receive(limit=2) for worker in workers))
    assert [len(batch) for batch in batches] == [2, 2]
    assert all(batch[0].sequence < batch[1].sequence for batch in batches)
    assert sorted(batches[0] + batches[1], key=lambda item: item.sequence) == sorted(
        queued, key=lambda item: item.sequence
    )
    assert await first.endpoint("worker").receive() == []


async def test_resume_returns_recorded_empty_receive_without_waiting_again(tmp_path):
    class Interrupted(Exception):
        pass

    class InterruptAfterTool(Hooks):
        async def after_tool(self, ctx, call, result):
            assert not result.is_error and json.loads(result.content) == []
            raise Interrupted

    class UseLongWaitOnResume(Hooks):
        async def before_tool(self, ctx, call):
            return call.model_copy(update={"arguments": {"wait_s": 30}})

    members = ["lead", "worker"]
    db_path = tmp_path / "team.db"
    box = TeamMailbox(members, db_path=db_path)
    async with SQLiteStore(tmp_path / "run.db") as store:
        session = Session(store, id="empty")
        first = Agent(
            ScriptedModel([call("receive_messages")]),
            tools=box.endpoint("worker").tools(),
            hooks=[InterruptAfterTool()],
        )
        with pytest.raises(Interrupted):
            await first.run("Check your inbox", session=session)
        reopened = TeamMailbox(members, db_path=db_path)
        resumed = Agent(
            ScriptedModel(["Finished"]),
            tools=reopened.endpoint("worker").tools(),
            hooks=[UseLongWaitOnResume()],
        )
        result = await asyncio.wait_for(resumed.resume(session), 5)
        (empty,) = [part for message in result.messages for part in message.tool_results]
        assert not empty.is_error and json.loads(empty.content) == []
