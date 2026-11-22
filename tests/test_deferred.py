from __future__ import annotations

import pytest

from qagents import Agent, InMemoryStore, Session, SQLiteStore, prompts, tool
from qagents.errors import DeferToolCall, SessionError
from qagents.messages import Message, ToolCallPart, ToolResultPart
from qagents.models.testing import ScriptedModel, call
from qagents.session import Entry, PendingApproval, PendingInput, derive_state
from qagents.toolkits.human import ask_user


def job_tools(log: list[str]):
    @tool(deferred=True)
    def fetch_report(report_id: str) -> dict:
        """Fetch a report that an external system produces."""
        log.append(f"fetch {report_id}")  # must never run
        return {}

    @tool
    async def start_render(script: str) -> str:
        """Start rendering a video."""
        log.append(f"render {script}")
        raise DeferToolCall("rendering", metadata={"job_id": f"J-{script}"})

    @tool(read_only=True)
    def lookup(key: str) -> str:
        """Look something up."""
        log.append(f"lookup {key}")
        return f"value of {key}"

    @tool(requires_approval=True)
    def send_email(to: str) -> str:
        """Send an email."""
        log.append(f"email {to}")
        return "sent"

    return [fetch_report, start_render, lookup, send_email]


def tool_results(result) -> list[ToolResultPart]:
    return next(m for m in result.messages if m.role == "tool").tool_results


async def test_a_deferred_tool_pauses_the_run_and_is_never_executed():
    log: list[str] = []
    model = ScriptedModel([call("fetch_report", report_id="R-1"), "The report is ready."])
    agent = Agent(model, tools=job_tools(log))
    session = Session()

    stream = agent.stream("Get report R-1", session=session)
    events = [event async for event in stream]
    paused = stream.result

    assert paused.status == "paused" and paused.output is None
    (waiting,) = paused.pending_inputs
    assert (waiting.name, waiting.arguments, waiting.reason) == (
        "fetch_report",
        {"report_id": "R-1"},
        None,
    )
    assert waiting.metadata == {} and waiting.token is None
    assert [e.type for e in events][-2:] == ["input_required", "run_finished"]
    assert next(e for e in events if e.type == "input_required").input == waiting
    assert "tool_call_started" not in [e.type for e in events]
    assert (await session.state()).pending_inputs == [waiting]

    resumed = await agent.resume(session)  # nothing supplied: still waiting, still not run
    assert resumed.status == "paused" and resumed.pending_inputs == [waiting]

    done = await agent.resume(session, results={waiting.tool_call_id: {"rows": 3}})
    assert done.status == "completed" and done.output == "The report is ready."
    assert done.run_id == paused.run_id
    assert model.requests[1].messages[-1].tool_results[0].content == '{"rows": 3}'
    assert log == []
    state = await session.state()
    assert state.pending_inputs == [] and state.open_run_id is None


async def test_defer_tool_call_keeps_reason_and_metadata_and_other_calls_still_run():
    log: list[str] = []
    model = ScriptedModel(
        [[call("start_render", script="intro"), call("lookup", key="a")], "Rendered."]
    )
    agent = Agent(model, tools=job_tools(log))
    session = Session()

    paused = await agent.run("Render the intro", session=session)
    assert paused.status == "paused"
    (waiting,) = paused.pending_inputs
    assert (waiting.reason, waiting.metadata) == ("rendering", {"job_id": "J-intro"})
    assert log == ["render intro", "lookup a"]

    kinds = [entry.kind for entry in await session.entries()]
    assert "tool_deferred" in kinds and kinds[-1] == "run_paused"
    paused_entry = (await session.entries())[-1]
    assert paused_entry.data["pending"] == []
    assert paused_entry.data["inputs"] == [waiting.model_dump(mode="json")]
    state = await session.state()
    assert state.started_tools == []  # waiting, not interrupted

    done = await agent.resume(session, results={waiting.tool_call_id: "s3://videos/intro.mp4"})
    assert done.status == "completed"
    assert log == ["render intro", "lookup a"]  # the deferring tool did not run again
    rendered, looked_up = tool_results(done)
    assert rendered.content == "s3://videos/intro.mp4" and not rendered.is_error
    assert looked_up.content == "value of a"


async def test_supplying_some_results_keeps_the_run_paused_with_the_rest():
    log: list[str] = []
    first, second = call("fetch_report", report_id="A"), call("fetch_report", report_id="B")
    model = ScriptedModel([[first, second], "Both reports are in."])
    agent = Agent(model, tools=job_tools(log))
    session = Session()

    paused = await agent.run("Get A and B", session=session)
    assert [item.tool_call_id for item in paused.pending_inputs] == [first.id, second.id]

    still = await agent.resume(session, results={first.id: "report A"})
    assert still.status == "paused"
    assert [item.tool_call_id for item in still.pending_inputs] == [second.id]
    assert [item.tool_call_id for item in (await session.state()).pending_inputs] == [second.id]

    done = await agent.resume(session, results={second.id: "report B"})
    assert done.status == "completed"
    assert [part.content for part in tool_results(done)] == ["report A", "report B"]
    assert len(model.requests) == 2 and log == []


async def test_approvals_and_inputs_can_be_answered_together():
    log: list[str] = []
    email, report = call("send_email", to="a@example.com"), call("fetch_report", report_id="R")
    model = ScriptedModel([[email, report], "Sent the report."])
    agent = Agent(model, tools=job_tools(log))
    session = Session()

    paused = await agent.run("Send the report", session=session)
    assert [item.tool_call_id for item in paused.pending_approvals] == [email.id]
    assert [item.tool_call_id for item in paused.pending_inputs] == [report.id]
    paused_entry = (await session.entries())[-1]
    assert paused_entry.kind == "run_paused"
    assert [item["tool_call_id"] for item in paused_entry.data["pending"]] == [email.id]
    assert [item["tool_call_id"] for item in paused_entry.data["inputs"]] == [report.id]

    done = await agent.resume(session, approvals={email.id: True}, results={report.id: "PDF"})
    assert done.status == "completed"
    assert log == ["email a@example.com"]
    assert [part.content for part in tool_results(done)] == ["sent", "PDF"]


async def test_a_deferred_tool_that_needs_approval_asks_first():
    log: list[str] = []

    @tool(deferred=True, requires_approval=True)
    def wire_transfer(amount: int) -> str:
        """Ask the bank to wire money; the bank confirms later."""
        log.append("ran")
        return ""

    transfer = call("wire_transfer", amount=100)
    model = ScriptedModel([transfer, "The transfer went through."])
    agent = Agent(model, tools=[wire_transfer])
    session = Session()

    paused = await agent.run("Wire 100", session=session)
    assert [p.tool_call_id for p in paused.pending_approvals] == [transfer.id]
    assert paused.pending_inputs == []
    with pytest.raises(SessionError, match="approvals=, not results="):
        await agent.resume(session, results={transfer.id: "done"})

    approved = await agent.resume(session, approvals={transfer.id: True})
    assert approved.pending_approvals == []
    assert [p.tool_call_id for p in approved.pending_inputs] == [transfer.id]

    done = await agent.resume(session, results={transfer.id: "confirmed"})
    assert done.status == "completed" and log == []


async def test_exceptions_become_error_results_and_tool_results_are_used_as_given():
    first, second = call("fetch_report", report_id="A"), call("fetch_report", report_id="B")
    model = ScriptedModel([[first, second], "One report failed."])
    agent = Agent(model, tools=job_tools([]))
    session = Session()
    await agent.run("Get A and B", session=session)

    given = ToolResultPart(tool_call_id="x", name="x", content="partial", metadata={"pages": 2})
    done = await agent.resume(
        session, results={first.id: ValueError("the job failed"), second.id: given}
    )
    failed, partial = tool_results(done)
    assert failed.is_error and failed.content == "the job failed"
    assert (partial.tool_call_id, partial.name) == (second.id, "fetch_report")
    assert (partial.content, partial.metadata) == ("partial", {"pages": 2})


async def test_long_supplied_results_are_shortened():
    report = call("fetch_report", report_id="A")
    model = ScriptedModel([report, "ok"])
    agent = Agent(model, tools=job_tools([]))
    session = Session()
    await agent.run("Get A", session=session)
    done = await agent.resume(session, results={report.id: "x" * 70_000})
    assert "characters omitted" in tool_results(done)[0].content


async def test_results_for_calls_that_do_not_wait_fail_before_anything_runs():
    report = call("fetch_report", report_id="A")
    model = ScriptedModel([report, "ok"])
    agent = Agent(model, tools=job_tools([]))
    session = Session()
    await agent.run("Get A", session=session)
    before = await session.entries()

    with pytest.raises(SessionError, match=r"\['nope'\] that do not wait for a result"):
        await agent.resume(session, results={"nope": "value"})
    assert await session.entries() == before  # nothing was journaled
    assert (await agent.resume(session, results={report.id: "A"})).status == "completed"


async def test_a_new_run_closes_pending_inputs_with_an_explanation():
    log: list[str] = []
    report = call("fetch_report", report_id="A")
    model = ScriptedModel([report, "Never mind then."])
    agent = Agent(model, tools=job_tools(log))
    session = Session()
    await agent.run("Get A", session=session)

    result = await agent.run("Forget the report.", session=session)
    assert result.status == "completed"
    closed = tool_results(result)[0]
    assert closed.is_error
    assert closed.content == prompts.TOOL_INPUT_ABANDONED.format(name="fetch_report")
    assert [m.role for m in model.requests[1].messages] == ["user", "assistant", "tool", "user"]
    state = await session.state()
    assert state.pending_inputs == [] and state.open_run_id is None
    statuses = [e.data.get("status") for e in await session.entries() if e.kind == "run_finished"]
    assert statuses == ["abandoned", "completed"]
    assert log == []


async def test_pending_inputs_survive_a_restart_with_sqlite(tmp_path):
    database = tmp_path / "agents.db"
    log: list[str] = []
    store = SQLiteStore(database)
    model = ScriptedModel([[call("start_render", script="intro"), call("lookup", key="k")]])
    paused = await Agent(model, tools=job_tools(log)).run(
        "Render", session=Session(store, id="job-1")
    )
    (waiting,) = paused.pending_inputs
    await store.aclose()

    store = SQLiteStore(database)  # "another process", hours later
    session = Session(store, id="job-1")
    state = await session.state()
    assert state.pending_inputs == [waiting]
    assert state.open_run_id == paused.run_id and state.started_tools == []

    model = ScriptedModel(["The video is at the link."])
    done = await Agent(model, tools=job_tools(log)).resume(
        session, results={waiting.tool_call_id: "https://cdn.example/intro.mp4"}
    )
    assert done.status == "completed" and done.run_id == paused.run_id
    assert log == ["render intro", "lookup k"]
    assert [p.content for p in model.requests[0].messages[-1].tool_results] == [
        "https://cdn.example/intro.mp4",
        "value of k",
    ]
    await store.aclose()


async def test_a_run_paused_entry_without_inputs_still_loads_and_resumes():
    """A journal written by 0.1: run_paused has only ``pending``."""
    log: list[str] = []
    email = ToolCallPart(id="call_1", name="send_email", arguments={"to": "a@example.com"})
    approval = {"tool_call_id": "call_1", "name": "send_email", "arguments": email.arguments}
    approval["reason"] = "this tool requires approval."
    entries = [
        Entry(kind="run_started", run_id="r1", data={"agent": "agent"}),
        Entry(kind="message", run_id="r1", data={"message": Message.user("email a").model_dump()}),
        Entry(
            kind="message",
            run_id="r1",
            data={"message": Message.assistant(tool_calls=[email]).model_dump()},
        ),
        Entry(kind="run_paused", run_id="r1", data={"pending": [approval]}),
    ]
    state = derive_state(entries)
    assert state.pending_inputs == [] and state.values == {}
    assert state.pending_approvals == [PendingApproval.model_validate(approval)]
    assert state.pending_approvals[0].token is None

    store = InMemoryStore()
    await store.append("old", entries)
    done = await Agent(ScriptedModel(["Sent."]), tools=job_tools(log)).resume(
        Session(store, id="old"), approvals={"call_1": True}
    )
    assert done.status == "completed" and log == ["email a@example.com"]


def test_unknown_entry_kinds_are_ignored_and_tool_results_clear_waiting_inputs():
    waiting = PendingInput(tool_call_id="c1", name="fetch_report", arguments={"report_id": "A"})
    call_part = ToolCallPart(id="c1", name="fetch_report", arguments={"report_id": "A"})
    entries = [
        Entry(kind="run_started", run_id="r1"),
        Entry(
            kind="message",
            run_id="r1",
            data={"message": Message.assistant(tool_calls=[call_part]).model_dump()},
        ),
        Entry(kind="tool_deferred", run_id="r1", data={"input": waiting.model_dump()}),
        Entry(kind="something_from_the_future", run_id="r1", data={"x": 1}),
    ]
    assert derive_state(entries).pending_inputs == [waiting]
    result = ToolResultPart(tool_call_id="c1", name="fetch_report", content="A")
    entries.append(Entry(kind="tool_result", run_id="r1", data={"result": result.model_dump()}))
    assert derive_state(entries).pending_inputs == []


async def test_metadata_that_is_not_json_stops_the_run_with_a_clear_error():
    class Handle:
        pass

    @tool
    def submit(job: str) -> str:
        """Submit a job."""
        raise DeferToolCall(metadata={"handle": Handle()})

    agent = Agent(ScriptedModel([call("submit", job="x")]), tools=[submit])
    with pytest.raises(SessionError, match="plain JSON values"):
        await agent.run("go")


async def test_ask_user_end_to_end():
    model = ScriptedModel(
        [
            call("ask_user", question="Which city?", choices=["Athens", "Lisbon"]),
            "Booked a table in Lisbon.",
        ]
    )
    agent = Agent(model, tools=[ask_user])
    assert ask_user.deferred and ask_user.read_only
    spec = ask_user.spec
    assert spec.description == prompts.ASK_USER_DESCRIPTION
    assert list(spec.parameters["properties"]) == ["question", "choices"]
    assert spec.parameters["required"] == ["question"]

    session = Session()
    result = await agent.run("Book a table for Friday", session=session)
    asked = 0
    while result.status == "paused" and result.pending_inputs:
        question = result.pending_inputs[0]
        assert question.arguments == {"question": "Which city?", "choices": ["Athens", "Lisbon"]}
        asked += 1
        result = await agent.resume(session, results={question.tool_call_id: "Lisbon"})

    assert asked == 1
    assert result.output == "Booked a table in Lisbon."
    assert model.requests[1].messages[-1].tool_results[0].content == "Lisbon"


async def test_cancelling_while_inputs_wait_closes_them():
    report = call("fetch_report", report_id="A")
    agent = Agent(ScriptedModel([report, "ok"]), tools=job_tools([]))
    session = Session()
    await agent.run("Get A", session=session)

    stream = agent.resume_stream(session)
    stream.cancel()
    result = await stream.wait()
    assert result.status == "cancelled"
    closed = tool_results(result)[0]
    assert closed.content == prompts.TOOL_INPUT_ABANDONED.format(name="fetch_report")
