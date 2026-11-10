from __future__ import annotations

import asyncio

import pytest

from qagents import (
    Agent,
    InMemoryStore,
    RunContext,
    Session,
    SessionError,
    SQLiteStore,
    SummarizingCompactor,
    Usage,
    tool,
)
from qagents.errors import ConfigurationError, ModelServerError
from qagents.messages import Message
from qagents.models.testing import ScriptedModel, call


async def test_sqlite_session_survives_a_restart(tmp_path):
    database = tmp_path / "agents.db"

    store = SQLiteStore(database)
    first = await Agent(ScriptedModel(["Noted."])).run(
        "My name is Maria.", session=Session(store, id="user-1")
    )
    assert first.session_id == "user-1"
    await store.aclose()

    store = SQLiteStore(database)  # "another process"
    model = ScriptedModel(["You are Maria."])
    second = await Agent(model).run("What is my name?", session=Session(store, id="user-1"))
    assert [m.text for m in model.requests[0].messages] == [
        "My name is Maria.",
        "Noted.",
        "What is my name?",
    ]
    assert len(second.messages) == 4 and len(second.new_messages) == 2

    kinds = [entry.kind for entry in await Session(store, id="user-1").entries()]
    assert kinds == [
        "run_started", "message", "message", "run_finished",
        "run_started", "message", "message", "run_finished",
    ]  # fmt: skip
    assert [e.seq for e in await store.load("user-1")] == list(range(1, 9))
    assert await Session(store, id="someone-else").messages() == []
    await store.aclose()


async def test_sessions_are_isolated_and_can_be_cleared():
    store = InMemoryStore()
    await Agent(ScriptedModel(["a"])).run("one", session=Session(store, id="a"))
    await Agent(ScriptedModel(["b"])).run("two", session=Session(store, id="b"))
    assert [m.text for m in await Session(store, id="a").messages()] == ["one", "a"]
    await Session(store, id="a").clear()
    assert await Session(store, id="a").messages() == []
    assert len(await Session(store, id="b").messages()) == 2


def crashing_tools(log: list[str], gate: asyncio.Event, started: asyncio.Event, **flags):
    @tool
    def charge_card(amount: int) -> str:
        """Charge the customer's card."""
        log.append(f"charge {amount}")
        return "charged"

    @tool(**flags)
    async def send_receipt(to: str) -> str:
        """Email the receipt."""
        log.append(f"receipt {to}")
        started.set()
        await gate.wait()  # the process "dies" while this is in flight
        return "sent"

    return charge_card, send_receipt


async def crash_mid_tool(session: Session, log: list[str], **flags) -> asyncio.Event:
    """Start a run and kill it while the second tool call is executing."""
    gate, started = asyncio.Event(), asyncio.Event()
    model = ScriptedModel([[call("charge_card", amount=40), call("send_receipt", to="m@x.gr")]])
    agent = Agent(model, tools=crashing_tools(log, gate, started, **flags))
    task = asyncio.create_task(agent.run("charge and send the receipt", session=session))
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    return gate


async def test_resume_after_a_crash_never_repeats_a_finished_tool(tmp_path):
    store = SQLiteStore(tmp_path / "runs.db")
    session = Session(store, id="order-7")
    log: list[str] = []
    await crash_mid_tool(session, log)

    state = await session.state()
    assert state.open_run_id is not None
    assert list(state.tool_results) == [state.messages[-1].tool_calls[0].id]
    assert state.started_tools == [state.messages[-1].tool_calls[1].id]

    # A new process: new agent object, same journal.
    gate, started = asyncio.Event(), asyncio.Event()
    gate.set()
    model = ScriptedModel(["The card was charged; I could not confirm the receipt."])
    agent = Agent(model, tools=crashing_tools(log, gate, started))
    result = await agent.resume(Session(store, id="order-7"))

    assert result.status == "completed"
    assert log == ["charge 40", "receipt m@x.gr"]  # neither tool ran a second time
    charged, receipt = result.messages[2].tool_results
    assert charged.content == "charged" and not charged.is_error
    assert receipt.is_error and "Its outcome is unknown" in receipt.content
    assert model.requests[0].messages[-1].role == "tool"
    await store.aclose()


async def test_resume_reruns_an_interrupted_tool_only_when_it_is_safe():
    session = Session()
    log: list[str] = []
    await crash_mid_tool(session, log, idempotent=True)

    gate, started = asyncio.Event(), asyncio.Event()
    gate.set()
    agent = Agent(
        ScriptedModel(["All done."]), tools=crashing_tools(log, gate, started, idempotent=True)
    )
    result = await agent.resume(session)
    assert log == ["charge 40", "receipt m@x.gr", "receipt m@x.gr"]
    assert result.messages[2].tool_results[1].content == "sent"


async def test_operation_id_is_the_same_when_a_call_is_repeated_after_a_resume():
    seen: list[str] = []
    gate, started = asyncio.Event(), asyncio.Event()

    @tool(idempotent=True)
    async def pay(ctx: RunContext, amount: int) -> str:
        """Pay an invoice."""
        seen.append(ctx.operation_id)
        started.set()
        await gate.wait()
        return "paid"

    session = Session()
    scripted = [call("pay", amount=5), call("pay", amount=5), "Both paid."]
    task = asyncio.create_task(
        Agent(ScriptedModel(scripted[:1]), tools=[pay]).run("pay", session=session)
    )
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    gate.set()
    result = await Agent(ScriptedModel(scripted[1:]), tools=[pay]).resume(session)
    first_try, retried, second_call = seen
    assert first_try == retried  # the payment API can deduplicate the repeat
    assert second_call != first_try  # a new call later in the run gets its own id
    assert first_try.startswith(f"{result.run_id}:") and result.output == "Both paid."


async def test_a_new_message_closes_tool_calls_left_open():
    session = Session()
    log: list[str] = []
    await crash_mid_tool(session, log)

    model = ScriptedModel(["Let me check what happened."])
    gate, started = asyncio.Event(), asyncio.Event()
    agent = Agent(model, tools=crashing_tools(log, gate, started))
    result = await agent.run("Hello? Did it work?", session=session)

    roles = [m.role for m in model.requests[0].messages]
    assert roles == ["user", "assistant", "tool", "user"]  # still a valid conversation
    charged, receipt = result.messages[2].tool_results
    assert charged.content == "charged"
    assert "Its outcome is unknown" in receipt.content
    statuses = [e.data.get("status") for e in await session.entries() if e.kind == "run_finished"]
    assert statuses == ["abandoned", "completed"]


async def test_a_failed_model_request_can_be_resumed():
    session = Session()
    failing = ScriptedModel([ModelServerError("gateway timeout", retryable=True, ambiguous=True)])
    with pytest.raises(ModelServerError):
        await Agent(failing).run("Summarise the report", session=session)
    assert (await session.state()).open_run_id is not None

    result = await Agent(ScriptedModel(["Here is the summary."])).resume(session)
    assert result.output == "Here is the summary."
    assert [m.role for m in result.messages] == ["user", "assistant"]


async def test_resume_needs_an_unfinished_run():
    session = Session()
    with pytest.raises(SessionError, match="no unfinished run"):
        await Agent(ScriptedModel([])).resume(session)
    await Agent(ScriptedModel(["done"])).run("go", session=session)
    with pytest.raises(SessionError, match="no unfinished run"):
        await Agent(ScriptedModel([])).resume(session)


async def test_history_and_an_existing_session_do_not_mix():
    session = Session()
    await Agent(ScriptedModel(["a"])).run("one", session=session)
    with pytest.raises(ConfigurationError, match="already has messages"):
        await Agent(ScriptedModel(["b"])).run("two", session=session, history=[Message.user("x")])


async def test_compaction_replaces_old_messages_and_is_journaled():
    @tool(read_only=True)
    def lookup(key: str) -> str:
        """Look something up."""
        return f"value of {key}"

    big = Usage(requests=1, input_tokens=900, output_tokens=200)
    model = ScriptedModel(
        [call("lookup", key="a"), call("lookup", key="b"), "Final answer."], usage=big
    )
    summarizer = ScriptedModel(["The user asked about a and b; a is known."] * 2)
    compactor = SummarizingCompactor(trigger_tokens=1000, keep_last=2, model=summarizer)
    session = Session()
    result = await Agent(model, tools=[lookup], compactor=compactor).run(
        "Compare a and b", session=session
    )

    assert result.output == "Final answer."
    # The summariser saw a plain transcript, with no tool definitions needed.
    transcript = summarizer.requests[0].messages[0].text
    assert (
        transcript.startswith("User: Compare a and b")
        and "Summarise the conversation" in transcript
    )
    # The request after compaction starts with the summary and keeps the call/result pair whole.
    after = model.requests[1].messages
    assert after[0].text.startswith("Summary of the earlier conversation:")
    assert [m.role for m in after] == ["user", "assistant", "tool"]
    assert result.usage.requests == 5  # three agent requests and two summaries are all counted
    assert "compaction" in [entry.kind for entry in await session.entries()]
    assert (await session.messages())[0].text.startswith("Summary of the earlier conversation:")
