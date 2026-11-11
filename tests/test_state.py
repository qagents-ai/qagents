from __future__ import annotations

import asyncio
from datetime import date

import pytest

from qagents import Agent, Hooks, RunContext, Session, SQLiteStore, tool
from qagents.errors import SessionError
from qagents.events import StateUpdated
from qagents.models.testing import FunctionModel, ScriptedModel, call
from qagents.session import Entry, derive_state


@tool
def remember(ctx: RunContext, key: str, value: str) -> str:
    """Remember a fact for later."""
    ctx.state[key] = value
    return "ok"


@tool(read_only=True)
def recall(ctx: RunContext, key: str) -> str:
    """Recall a fact."""
    return ctx.state.get(key, "unknown")


def state_entries(entries: list[Entry]) -> list[dict]:
    return [entry.data["values"] for entry in entries if entry.kind == "state_updated"]


async def test_state_persists_across_runs_of_a_session():
    session = Session()
    model = ScriptedModel([call("remember", key="city", value="Athens"), "Noted."])
    agent = Agent(model, tools=[remember, recall])
    first = await agent.run("I live in Athens.", session=session)
    assert first.status == "completed"
    assert (await session.state()).values == {"city": "Athens"}

    model = ScriptedModel([call("recall", key="city"), "You live in Athens."])
    second = await Agent(model, tools=[remember, recall]).run("Where do I live?", session=session)
    assert model.requests[1].messages[-1].tool_results[0].content == "Athens"
    assert second.output == "You live in Athens."


async def test_state_persists_across_a_pause_and_a_resume():
    @tool(requires_approval=True)
    def book(ctx: RunContext, flight: str) -> str:
        """Book a flight."""
        ctx.state["booked"] = [*ctx.state.get("booked", []), flight]
        return "booked"

    session = Session()
    model = ScriptedModel(
        [[call("remember", key="step", value="searching"), call("book", flight="A3 600")], "Done."]
    )
    agent = Agent(model, tools=[remember, book])
    paused = await agent.run("Book A3 600", session=session)
    assert paused.status == "paused"
    assert (await session.state()).values == {"step": "searching"}

    approval = {paused.pending_approvals[0].tool_call_id: True}
    done = await agent.resume(session, approvals=approval, state={"approved_by": "maria"})
    assert done.status == "completed"
    assert (await session.state()).values == {
        "step": "searching",
        "approved_by": "maria",
        "booked": ["A3 600"],
    }


async def test_state_seeds_the_session_and_instructions_see_it():
    seen: list[dict] = []

    def instructions(ctx: RunContext) -> str:
        seen.append(dict(ctx.state))
        return f"The user is {ctx.state['user']}. Plan: {ctx.state.get('plan', 'none')}."

    model = ScriptedModel(["Hello Maria."])
    agent = Agent(model, instructions=instructions)
    session = Session()

    stream = agent.stream("Hi", session=session, state={"user": "Maria"})
    events = [event async for event in stream]
    assert [e.type for e in events][:2] == ["run_started", "state_updated"]
    assert events[1].values == {"user": "Maria"}
    assert model.requests[0].messages[0].text == "The user is Maria. Plan: none."

    model2 = ScriptedModel(["Still Maria."])
    agent.model = model2
    await agent.run("Again", session=session, state={"plan": "rest"})
    assert model2.requests[0].messages[0].text == "The user is Maria. Plan: rest."
    assert state_entries(await session.entries()) == [
        {"user": "Maria"},
        {"user": "Maria", "plan": "rest"},
    ]
    assert seen == [{"user": "Maria"}, {"user": "Maria", "plan": "rest"}]


async def test_unchanged_state_writes_nothing():
    session = Session()
    agent = Agent(ScriptedModel(["a", "b"]), tools=[recall])
    await agent.run("one", session=session, state={"x": 1})
    await agent.run("two", session=session, state={"x": 1})  # the same value again
    assert state_entries(await session.entries()) == [{"x": 1}]


async def test_changes_are_journaled_before_the_tool_result_and_announced():
    session = Session()
    agent = Agent(
        ScriptedModel([call("remember", key="a", value="1"), "ok"]), tools=[remember, recall]
    )
    stream = agent.stream("go", session=session)
    events = [event async for event in stream]
    types = [event.type for event in events]
    assert types.index("state_updated") < types.index("tool_call_finished")
    updated = next(e for e in events if isinstance(e, StateUpdated))
    assert (updated.values, updated.agent) == ({"a": "1"}, "agent")

    kinds = [entry.kind for entry in await session.entries()]
    assert kinds.index("state_updated") == kinds.index("tool_result") - 1


async def test_hooks_share_the_same_state_and_their_changes_are_saved():
    contexts: list[int] = []

    class CountTurns(Hooks):
        async def before_model(self, ctx, request):
            contexts.append(id(ctx.state))
            ctx.state["turns"] = ctx.state.get("turns", 0) + 1

        async def after_tool(self, ctx, call, result):
            contexts.append(id(ctx.state))
            ctx.state["last_tool"] = call.name

    session = Session()
    agent = Agent(
        ScriptedModel([call("recall", key="x"), "done"]), tools=[recall], hooks=[CountTurns()]
    )
    await agent.run("go", session=session)
    assert len(set(contexts)) == 1  # one dict object for the whole run
    assert state_entries(await session.entries()) == [
        {"turns": 1},
        {"turns": 1, "last_tool": "recall"},
        {"turns": 2, "last_tool": "recall"},
    ]


@pytest.mark.parametrize(
    ("value", "kind"), [(date(2026, 1, 2), "date"), ((1, 2), "tuple"), ({1, 2}, "set")]
)
async def test_values_that_are_not_plain_json_are_refused_with_the_key(value, kind):
    @tool
    def store(ctx: RunContext) -> str:
        """Store something."""
        ctx.state["ok"] = "fine"
        ctx.state["when"] = value
        return "stored"

    session = Session()
    agent = Agent(ScriptedModel([call("store")]), tools=[store])
    with pytest.raises(SessionError, match=rf"ctx.state\['when'\] holds a {kind}"):
        await agent.run("go", session=session)
    assert state_entries(await session.entries()) == []
    assert (await session.state()).open_run_id is not None  # failed runs stay resumable


async def test_seeded_values_and_keys_are_checked_too():
    with pytest.raises(SessionError, match="model_dump"):
        await Agent(ScriptedModel(["x"])).run("go", state={"nan": float("nan")})
    with pytest.raises(SessionError, match="keys must be strings"):
        await Agent(ScriptedModel(["x"])).run("go", state={1: "one"})


async def test_state_survives_a_restart_with_sqlite(tmp_path):
    database = tmp_path / "agents.db"
    store = SQLiteStore(database)
    agent = Agent(
        ScriptedModel([call("remember", key="name", value="Maria"), "Noted."]), tools=[remember]
    )
    await agent.run("I am Maria", session=Session(store, id="u1"), state={"visits": 1})
    await store.aclose()

    store = SQLiteStore(database)  # "another process"
    session = Session(store, id="u1")
    assert (await session.state()).values == {"visits": 1, "name": "Maria"}
    model = ScriptedModel([call("recall", key="name"), "You are Maria."])
    await Agent(model, tools=[recall]).run("Who am I?", session=session)
    assert model.requests[1].messages[-1].tool_results[0].content == "Maria"
    await store.aclose()


async def test_concurrent_writes_from_parallel_tools_leave_a_consistent_journal():
    @tool(read_only=True)  # misuse on purpose: parallel tools that write state
    async def mark(ctx: RunContext, key: str) -> str:
        """Mark a key."""
        await asyncio.sleep(0)
        ctx.state[key] = True
        return "marked"

    model = ScriptedModel([[call("mark", key=f"k{i}") for i in range(5)], "done"])
    session = Session()
    await Agent(model, tools=[mark]).run("go", session=session)
    snapshots = state_entries(await session.entries())
    assert snapshots[-1] == {f"k{i}": True for i in range(5)}
    for earlier, later in zip(snapshots, snapshots[1:], strict=False):
        assert earlier.items() <= later.items()  # entries never go back in time
    assert (await session.state()).values == snapshots[-1]


async def test_session_state_values_are_a_copy_of_the_journal():
    session = Session()
    await Agent(ScriptedModel(["x"])).run("go", session=session, state={"items": [1]})
    values = (await session.state()).values
    values["items"].append(2)
    assert (await session.state()).values == {"items": [1]}

    # A tool changing a nested value is a change, and the journal keeps the old entry intact.
    @tool
    def add(ctx: RunContext) -> str:
        """Add an item."""
        ctx.state["items"].append(3)
        return "ok"

    await Agent(ScriptedModel([call("add"), "ok"]), tools=[add]).run("add", session=session)
    assert state_entries(await session.entries()) == [{"items": [1]}, {"items": [1, 3]}]


def test_derive_state_takes_the_latest_values():
    entries = [
        Entry(kind="state_updated", data={"values": {"a": 1}}),
        Entry(kind="state_updated", data={"values": {"a": 2, "b": 3}}),
    ]
    assert derive_state(entries).values == {"a": 2, "b": 3}
    assert derive_state([]).values == {}


def test_run_context_has_placeholders_for_child_runs():
    ctx = RunContext()
    assert ctx.state == {} and ctx.depth == 0
    assert ctx.parent_run_id is None and ctx.parent_tool_call_id is None


async def test_instructions_functions_that_change_state_are_saved():
    def instructions(ctx: RunContext) -> str:
        ctx.state["rendered"] = ctx.state.get("rendered", 0) + 1
        return "Be brief."

    session = Session()
    model = FunctionModel(lambda request: "ok")
    await Agent(model, instructions=instructions).run("go", session=session)
    assert (await session.state()).values == {"rendered": 1}
