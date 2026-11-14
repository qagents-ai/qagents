"""The journal stays the truth when a run stops at any point: resume, failures and hooks."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from qagents import (
    Agent,
    Hooks,
    InMemoryStore,
    Limits,
    RunContext,
    Session,
    SessionError,
    SQLiteStore,
    tool,
)
from qagents.errors import ConfigurationError, ModelServerError
from qagents.messages import Message, Usage
from qagents.models.base import ModelResponse
from qagents.models.testing import ScriptedModel, call
from qagents.session import Entry
from qagents.tools import Toolkit


def email_tool(log: list[str]):
    @tool
    def send_email(to: str) -> str:
        """Send an email."""
        log.append(f"email {to}")
        return "sent"

    return send_email


class FailOn(Hooks):
    """An on_event hook that raises on the first event of a type (an exporter that is down)."""

    def __init__(self, event_type: str, when: Any = None) -> None:
        self.event_type = event_type
        self.when = when or (lambda event: True)
        self.ends: list[tuple[str | None, str | None]] = []
        self.starts = 0

    async def on_run_start(self, ctx) -> None:
        self.starts += 1

    async def on_event(self, ctx, event) -> None:
        if event.type == self.event_type and self.when(event):
            raise RuntimeError(f"exporter down at {event.type}")

    async def on_run_end(self, ctx, result, error) -> None:
        self.ends.append(
            (result.status if result else None, type(error).__name__ if error else None)
        )


class FlakyStore(InMemoryStore):
    """Definitely refuses the first append selected by ``fail(entry)``."""

    def __init__(self, fail) -> None:
        super().__init__()
        self.fail = fail
        self.failed = False
        self.appends: list[list[str]] = []

    async def append(self, session_id: str, entries: list[Entry]) -> None:
        if not self.failed and any(self.fail(entry) for entry in entries):
            self.failed = True
            raise SessionError("fixture refused the batch before writing")
        self.appends.append([entry.kind for entry in entries])
        await super().append(session_id, entries)

    async def append_leased(self, session_id: str, entries: list[Entry], owner: str) -> list[int]:
        if not self.failed and any(self.fail(entry) for entry in entries):
            self.failed = True
            raise SessionError("fixture refused the batch before writing")
        self.appends.append([entry.kind for entry in entries])
        return await super().append_leased(session_id, entries, owner)


# -- an accepted final answer (F2, F77) ------------------------------------------------


def answer_agent(model: ScriptedModel, log: list[str], hooks=()) -> Agent:
    return Agent(model, tools=[email_tool(log)], output_mode="tool", hooks=list(hooks))


async def test_resume_after_an_accepted_answer_interrupted_by_a_hook_never_runs_the_rest():
    log: list[str] = []
    turn = [call("final_answer", value="done"), call("send_email", to="ceo@x")]
    model = ScriptedModel([turn])
    session = Session()
    hook = FailOn("tool_call_finished", lambda event: event.call.name == "final_answer")
    with pytest.raises(RuntimeError, match="exporter down"):
        await answer_agent(model, log, [hook]).run("finish", session=session)

    later = ScriptedModel([])  # any request would fail the test
    result = await answer_agent(later, log).resume(session)
    assert result.status == "completed" and result.output == "done"
    assert log == [] and later.requests == []


async def test_resume_of_a_half_written_answer_turn_closes_it_without_running_the_rest():
    """A crash between the accepted answer's result and the rest of its turn (older journals)."""
    log: list[str] = []
    session = Session()
    answer, other = call("final_answer", value="done"), call("send_email", to="ceo@x")
    run_id = "r1"
    accepted = {
        "tool_call_id": answer.id,
        "name": "final_answer",
        "content": "Final answer accepted.",
    }
    await session.store.append(
        session.id,
        [
            Entry(kind="run_started", run_id=run_id, data={"agent": "agent"}),
            Entry(kind="message", run_id=run_id, data={"message": _dump(Message.user("finish"))}),
            Entry(
                kind="message",
                run_id=run_id,
                data={"message": _dump(Message.assistant(tool_calls=[answer, other]))},
            ),
            Entry(kind="tool_result", run_id=run_id, data={"result": accepted}),
        ],
    )
    later = ScriptedModel([])
    result = await answer_agent(later, log).resume(session)
    assert result.output == "done" and log == [] and later.requests == []
    skipped = result.messages[-1].tool_results[1]
    assert skipped.is_error and "was not run because the final answer" in skipped.content


async def test_a_store_failure_inside_an_answer_turn_writes_none_of_it():
    log: list[str] = []
    # The skipped call's result fails: the accepted answer must not be written without it.
    store = FlakyStore(
        lambda e: e.kind == "tool_result" and e.data["result"]["name"] == "send_email"
    )
    session = Session(store, id="s")
    turn = [call("final_answer", value="done"), call("send_email", to="ceo@x")]
    with pytest.raises(SessionError, match="refused the batch"):
        await answer_agent(ScriptedModel([turn]), log).run("finish", session=session)
    assert "tool_result" not in [entry.kind for entry in await session.entries()]

    later = ScriptedModel([])
    result = await answer_agent(later, log).resume(session)
    assert result.output == "done" and log == [] and later.requests == []


# -- handoffs (F7, F132) ---------------------------------------------------------------


def handoff_agents(billing_model: ScriptedModel) -> Agent:
    billing = Agent(billing_model, name="billing")
    return Agent(ScriptedModel([call("transfer_to_billing")]), name="triage", handoffs=[billing])


async def test_a_handoff_interrupted_before_the_switch_continues_with_the_target():
    store = FlakyStore(lambda entry: entry.kind == "agent_switched")
    session = Session(store, id="s")
    with pytest.raises(SessionError, match="refused the batch"):
        await handoff_agents(ScriptedModel([])).run("refund A1", session=session)
    # One write holds the handoff's result, the tool message and the switch: none was written.
    assert "tool_result" not in [entry.kind for entry in await session.entries()]

    billing_model = ScriptedModel(["billing here"])
    triage_model = ScriptedModel([call("transfer_to_billing")])
    billing = Agent(billing_model, name="billing")
    triage = Agent(triage_model, name="triage", handoffs=[billing])
    result = await triage.resume(session)
    assert (result.agent, result.output) == ("billing", "billing here")
    assert triage_model.requests == []


async def test_a_journaled_handoff_without_its_switch_is_finished_on_resume():
    """A journal written in separate appends that stopped after the tool message."""
    session = Session()
    handoff = call("transfer_to_billing")
    result = {
        "tool_call_id": handoff.id,
        "name": handoff.name,
        "content": "The conversation is now handled by billing.",
    }
    assistant = Message.assistant(tool_calls=[handoff]).model_copy(update={"name": "triage"})
    tool_message = Message(role="tool", parts=[_tool_result(result)])
    await session.store.append(
        session.id,
        [
            Entry(kind="run_started", run_id="r1", data={"agent": "triage"}),
            Entry(kind="message", run_id="r1", data={"message": _dump(Message.user("refund"))}),
            Entry(kind="message", run_id="r1", data={"message": _dump(assistant)}),
            Entry(kind="tool_result", run_id="r1", data={"result": result}),
            Entry(kind="message", run_id="r1", data={"message": _dump(tool_message)}),
        ],
    )
    billing_model = ScriptedModel(["billing here"])
    billing = Agent(billing_model, name="billing")
    triage_model = ScriptedModel([])
    resumed = await Agent(triage_model, name="triage", handoffs=[billing]).resume(session)
    assert (resumed.agent, resumed.output) == ("billing", "billing here")
    assert triage_model.requests == []
    assert "agent_switched" in [entry.kind for entry in await session.entries()]


async def test_agents_reachable_through_handoffs_need_unique_names():
    billing = Agent(ScriptedModel([]))  # both keep the default name "agent"
    triage = Agent(ScriptedModel([]), handoffs=[billing])
    session = Session()
    with pytest.raises(ConfigurationError, match="unique name"):
        await triage.run("refund", session=session)
    assert await session.entries() == []

    same = Agent(ScriptedModel(["ok"]), name="loop")
    same.handoffs.append(same)  # the same agent reached twice is fine
    assert (await same.run("hi")).output == "ok"


# -- failures are journaled and reported once (F9, F17, F12, F89, L21) -----------------


@pytest.mark.parametrize("event_type", ["run_started", "text_delta", "model_responded"])
async def test_an_on_event_hook_that_raises_fails_the_run_in_the_journal(event_type):
    store = InMemoryStore()
    hook = FailOn(event_type)
    session = Session(store, id="s")
    with pytest.raises(RuntimeError, match="exporter down"):
        await Agent(ScriptedModel(["hello there"]), hooks=[hook]).run("hi", session=session)

    last = (await session.entries())[-1]
    assert last.kind == "run_failed"
    assert last.data["error_type"] == "RuntimeError"
    assert last.data["message"] == f"exporter down at {event_type}"
    assert (last.data["retryable"], last.data["ambiguous"]) == (False, False)
    status = await store.status("s")
    assert (status.status, status.failures) == ("failed", 1)
    if event_type == "run_started":
        assert hook.starts == 0 and hook.ends == []  # on_run_end pairs with on_run_start
    else:
        assert hook.ends == [(None, "RuntimeError")]  # the hook's error, not GeneratorExit


async def test_an_on_event_hook_that_raises_on_run_finished_does_not_reopen_the_run():
    hook = FailOn("run_finished")
    session = Session()
    with pytest.raises(RuntimeError):
        await Agent(ScriptedModel(["hello"]), hooks=[hook]).run("hi", session=session)
    assert [entry.kind for entry in await session.entries()][-1] == "run_finished"
    assert hook.ends == [("completed", None)]


async def test_on_run_end_that_raises_after_success_is_called_once_and_writes_no_failure():
    calls: list[Any] = []

    class Broken(Hooks):
        async def on_run_end(self, ctx, result, error):
            calls.append((result is not None, error))
            raise RuntimeError("tracer down")

    session = Session()
    with pytest.raises(RuntimeError, match="tracer down"):
        await Agent(ScriptedModel(["ok"]), hooks=[Broken()]).run("hi", session=session)
    assert calls == [(True, None)]
    assert [entry.kind for entry in await session.entries()][-1] == "run_finished"


async def test_a_consumer_that_stops_at_run_finished_gets_one_on_run_end():
    hook = FailOn("never")
    stream = Agent(ScriptedModel(["ok"]), hooks=[hook]).stream("hi")
    async with stream:
        async for event in stream:
            if event.type == "run_finished":
                break
    assert hook.ends == [("completed", None)]


async def test_run_failed_and_run_finished_carry_what_rebuilds_the_result():
    session = Session()
    failing = ScriptedModel([ModelServerError("gateway timeout", retryable=True, ambiguous=True)])
    with pytest.raises(ModelServerError):
        await Agent(failing).run("hi", session=session)
    failed = (await session.entries())[-1]
    assert failed.kind == "run_failed"
    assert failed.data["error_type"] == "ModelServerError"
    assert (failed.data["retryable"], failed.data["ambiguous"]) == (True, True)

    result = await Agent(ScriptedModel(["Hello."])).resume(session)
    finished = (await session.entries())[-1]
    assert finished.data == {
        "status": "completed",
        "stop_reason": None,
        "output": "Hello.",
        "text": "Hello.",
    }
    assert result.output == "Hello."


# -- the start of a run is one write (F28, E3) -----------------------------------------


async def test_a_new_run_is_started_in_one_write_with_its_state_and_prompt():
    store = FlakyStore(lambda entry: False)
    session = Session(store, id="s")
    await Agent(ScriptedModel(["one"])).run("first", session=session)
    store.appends.clear()
    await Agent(ScriptedModel(["two"])).run("second", session=session, state={"plan": "gold"})
    assert store.appends[0] == ["run_started", "state_updated", "message"]


async def test_a_crash_while_a_run_starts_leaves_the_previous_run_finished():
    store = FlakyStore(lambda entry: entry.kind == "message" and "Transfer" in str(entry.data))
    session = Session(store, id="s")
    await Agent(ScriptedModel(["Athens."])).run("Capital of Greece?", session=session)
    with pytest.raises(SessionError, match="refused the batch"):
        await Agent(ScriptedModel([])).run("Transfer 500 EUR to Maria", session=session)
    state = await session.state()
    assert state.open_run_id is None  # nothing for recovery to "complete" with the old answer


async def test_an_open_run_without_its_prompt_is_not_completed_with_the_previous_answer():
    """A journal from before the fix: run_started written, its prompt lost."""
    session = Session()
    await Agent(ScriptedModel(["Athens."])).run("Capital of Greece?", session=session)
    await session.append("run_started", {"agent": "agent"}, run_id="r2")
    model = ScriptedModel([])
    with pytest.raises(SessionError, match="no prompt in the journal"):
        await Agent(model).resume(session)
    state = await session.state()
    assert state.open_run_id is None and model.requests == []


async def test_an_invalid_state_value_fails_before_anything_is_journaled():
    import datetime

    session = Session()
    with pytest.raises(SessionError, match="would not come back unchanged"):
        await Agent(ScriptedModel(["ok"])).run(
            "transfer", session=session, state={"day": datetime.date(2026, 1, 1)}
        )
    assert await session.entries() == []


# -- failures before a resume opens the journal count (F31, F75) -----------------------


class DownToolkit(Toolkit):
    async def tools(self):
        raise ConnectionError("MCP server down")


async def test_a_resume_that_cannot_load_its_tools_counts_as_a_failure():
    store = InMemoryStore()
    session = Session(store, id="s")
    failing = ScriptedModel([ModelServerError("timeout", retryable=True, ambiguous=True)])
    with pytest.raises(ModelServerError):
        await Agent(failing).run("hi", session=session)
    assert (await store.status("s")).failures == 1

    for attempt in (2, 3):
        with pytest.raises(ConnectionError):
            await Agent(ScriptedModel([]), tools=[DownToolkit()]).resume(session)
        assert (await store.status("s")).failures == attempt


async def test_a_paused_run_whose_tools_cannot_load_stays_paused():
    store = InMemoryStore()
    session = Session(store, id="s")

    @tool(requires_approval=True)
    def refund(order: str) -> str:
        """Refund."""
        return "ok"

    paused = await Agent(ScriptedModel([call("refund", order="A")]), tools=[refund]).run(
        "refund A", session=session
    )
    assert paused.status == "paused"
    with pytest.raises(ConnectionError):
        await Agent(ScriptedModel([]), tools=[DownToolkit()]).resume(session)
    assert (await store.status("s")).status == "paused"


# -- limits count the whole run (F79, L2) ----------------------------------------------


async def test_max_tool_calls_and_usage_count_across_resumes():
    @tool(deferred=True)
    def ask_user(question: str) -> str:
        """Ask the user."""
        raise AssertionError("never runs")

    replies = [call("ask_user", question=f"q{n}") for n in range(5)] + ["done"]
    model = ScriptedModel(replies, usage=Usage(requests=1, input_tokens=10, output_tokens=5))
    agent = Agent(model, tools=[ask_user], limits=Limits(max_tool_calls=2))
    session = Session()
    result = await agent.run("plan", session=session)
    while result.status == "paused":
        waiting = result.pending_inputs[0]
        result = await agent.resume(session, results={waiting.tool_call_id: "yes"})
    assert (result.status, result.stop_reason) == ("limit_reached", "max_tool_calls")
    assert len(model.requests) == 3
    assert result.turns == 3 and result.usage.input_tokens == 30  # the whole run
    state = await session.state()
    assert state.open_run_id is None


async def test_max_turns_counts_the_turns_before_a_crash():
    session = Session()
    model = ScriptedModel([call("look"), ModelServerError("down", retryable=False)])

    @tool(read_only=True)
    def look() -> str:
        """Look."""
        return "seen"

    agent = Agent(model, tools=[look], limits=Limits(max_turns=2))
    with pytest.raises(ModelServerError):
        await agent.run("go", session=session)
    assert (await session.state()).run_turns == 1

    later = ScriptedModel([call("look"), "never asked"])
    result = await Agent(later, tools=[look], limits=Limits(max_turns=2)).resume(session)
    assert (result.status, result.stop_reason, result.turns) == ("limit_reached", "max_turns", 2)
    assert len(later.requests) == 1


# -- refusals and cut-off answers survive a stop (F83, F128) ---------------------------


async def test_a_refusal_with_tool_calls_is_not_run_after_a_stop_at_model_responded():
    log: list[str] = []
    refusal = ModelResponse(
        message=Message.assistant(
            "I can't help with that.", tool_calls=[call("send_email", to="x")]
        ),
        finish_reason="content_filter",
    )
    session = Session()
    stream = Agent(ScriptedModel([refusal]), tools=[email_tool(log)]).stream("go", session=session)
    async with stream:
        async for event in stream:
            if event.type == "model_responded":
                break  # a shutdown at its safe point, or a client that disconnects

    later = ScriptedModel([])
    result = await Agent(later, tools=[email_tool(log)]).resume(session)
    assert (result.status, result.stop_reason) == ("completed", "content_filter")
    assert result.output == "I can't help with that."
    assert log == [] and later.requests == []


async def test_a_cut_off_answer_keeps_its_stop_reason_after_a_stop():
    cut = ModelResponse(message=Message.assistant("The answer is"), finish_reason="length")
    session = Session()
    stream = Agent(ScriptedModel([cut])).stream("go", session=session)
    async with stream:
        async for event in stream:
            if event.type == "model_responded":
                break
    result = await Agent(ScriptedModel([])).resume(session)
    assert (result.output, result.stop_reason) == ("The answer is", "max_output_tokens")


# -- one run per session (F135, F136) ---------------------------------------------------


def slow_tool(log: list[str], gate: asyncio.Event, started: asyncio.Event):
    @tool(requires_approval=True)
    async def send_email(to: str) -> str:
        """Send an email."""
        log.append(f"email {to}")
        started.set()
        await gate.wait()
        return "sent"

    return send_email


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_two_resumes_of_one_paused_run_run_the_approved_call_once(backend, tmp_path):
    log: list[str] = []
    gate, started = asyncio.Event(), asyncio.Event()

    def store_for() -> Any:
        return InMemoryStore() if backend == "memory" else SQLiteStore(tmp_path / "s.db")

    first_store = store_for()
    second_store = first_store if backend == "memory" else store_for()  # another process
    agent = Agent(
        ScriptedModel([call("send_email", to="x"), "done"]), tools=[slow_tool(log, gate, started)]
    )
    paused = await agent.run("email x", session=Session(first_store, id="chat"))
    approvals = {paused.pending_approvals[0].tool_call_id: True}

    first = asyncio.create_task(agent.resume(Session(first_store, id="chat"), approvals=approvals))
    await asyncio.wait_for(started.wait(), 2)
    with pytest.raises(SessionError, match="run in progress"):
        second = agent.resume(Session(second_store, id="chat"), approvals=approvals)
        await asyncio.wait_for(second, 2)
    gate.set()
    assert (await first).output == "done"
    assert log == ["email x"]
    for store in {id(first_store): first_store, id(second_store): second_store}.values():
        await store.aclose()


async def test_a_second_run_while_one_runs_is_refused_and_the_journal_stays_valid():
    log: list[str] = []
    gate, started = asyncio.Event(), asyncio.Event()

    @tool
    async def lookup(key: str) -> str:
        """Look up."""
        log.append(key)
        started.set()
        await gate.wait()
        return "value"

    session = Session(InMemoryStore(), id="chat")
    agent = Agent(ScriptedModel([call("lookup", key="a"), "first"]), tools=[lookup])
    first = asyncio.create_task(agent.run("one", session=session))
    await asyncio.wait_for(started.wait(), 2)
    with pytest.raises(SessionError, match="run in progress"):
        await Agent(ScriptedModel(["second"])).run("two", session=session)
    gate.set()
    assert (await first).output == "first"
    roles = [message.role for message in await session.messages()]
    assert roles == ["user", "assistant", "tool", "assistant"]


async def test_a_run_refuses_a_session_whose_lease_another_worker_holds():
    store = InMemoryStore()
    assert await store.acquire_lease("s", "worker-b", 30)
    session = Session(store, id="s")
    with pytest.raises(SessionError, match="holds its lease"):
        await Agent(ScriptedModel(["hi"])).run("hello", session=session)
    assert await session.entries() == []


async def test_a_run_holds_the_lease_while_it_runs_and_releases_it():
    store = InMemoryStore()
    seen: list[str | None] = []

    @tool
    async def peek(ctx) -> str:
        """Peek at the lease."""
        seen.append((await store.status("s")).lease_owner)
        return "ok"

    await Agent(ScriptedModel([call("peek"), "done"]), tools=[peek]).run(
        "go", session=Session(store, id="s")
    )
    assert seen[0] is not None and seen[0].startswith("qagents-run#")
    assert (await store.status("s")).lease_owner is None


async def test_entries_of_a_run_written_after_it_was_closed_are_ignored():
    session = Session()
    await session.store.append(
        session.id,
        [
            Entry(kind="run_started", run_id="old", data={"agent": "agent"}),
            Entry(kind="message", run_id="old", data={"message": _dump(Message.user("one"))}),
            Entry(kind="run_finished", run_id="old", data={"status": "abandoned"}),
            Entry(kind="run_started", run_id="new", data={"agent": "agent"}),
            Entry(kind="message", run_id="new", data={"message": _dump(Message.user("two"))}),
            # the old run, still alive in another process, writes on
            Entry(kind="message", run_id="old", data={"message": _dump(Message.assistant("x"))}),
        ],
    )
    state = await session.state()
    assert [m.text for m in state.messages] == ["one", "two"]
    assert state.agent == "agent" and state.open_run_id == "new"


async def test_session_state_agent_is_cleared_when_the_run_finishes():
    session = Session()
    await Agent(ScriptedModel(["hi"]), name="greeter").run("hello", session=session)
    assert (await session.state()).agent is None


# -- helpers ---------------------------------------------------------------------------


def _dump(message: Message) -> dict[str, Any]:
    return message.model_dump(mode="json")


def _tool_result(data: dict[str, Any]):
    from qagents.messages import ToolResultPart

    return ToolResultPart.model_validate(data)


class BillingCompactor:
    """A compactor whose summary request costs 1000 input tokens and changes nothing."""

    async def compact(self, ctx, messages, last_response, model):
        ctx.add_usage(Usage(requests=1, input_tokens=1000))
        return None


async def test_usage_added_before_a_pause_is_restored_on_resume():
    @tool(requires_approval=True)
    def act() -> str:
        """Act."""
        return "ok"

    model = ScriptedModel([call("act"), "done"], usage=Usage(requests=1, input_tokens=10))
    agent = Agent(model, tools=[act], compactor=BillingCompactor())
    session = Session()
    paused = await agent.run("go", session=session)
    assert paused.usage.input_tokens == 1010
    assert (await session.state()).run_usage.input_tokens == 1010  # was 10: the summary was lost
    pending = paused.pending_approvals[0]
    result = await agent.resume(session, approvals={pending.tool_call_id: True})
    assert result.status == "completed" and result.usage.input_tokens == 2020  # as uninterrupted


async def test_usage_added_before_a_crash_is_restored_on_resume():
    model = ScriptedModel([ModelServerError("down", retryable=False)])
    session = Session()
    with pytest.raises(ModelServerError):
        await Agent(model, compactor=BillingCompactor()).run("go", session=session)
    assert (await session.state()).run_usage.input_tokens == 1000  # carried by run_failed

    later = ScriptedModel(["done"], usage=Usage(requests=1, input_tokens=10))
    result = await Agent(later, limits=Limits(max_total_tokens=1000)).resume(session)
    assert (result.status, result.stop_reason) == ("limit_reached", "max_total_tokens")
    assert later.requests == []  # the summary already used the budget


async def test_concurrent_batches_are_atomic_and_do_not_inherit_a_parent_batch():
    from qagents import InMemoryStore

    class RecordingStore(InMemoryStore):
        def __init__(self):
            super().__init__()
            self.batches = []

        async def append(self, session_id, entries):
            entries = list(entries)
            self.batches.append([entry.data["label"] for entry in entries])
            await super().append(session_id, entries)

    store = RecordingStore()
    session = Session(store)
    run = Agent(ScriptedModel([])).stream("go", session=session)._run
    started = asyncio.Event()
    release = asyncio.Event()

    async def child():
        async with run._batched():
            await run._journal("probe", {"label": "child-1"})
            started.set()
            await release.wait()
            async with run._batched():
                await run._journal("probe", {"label": "child-2"})

    async with run._batched():
        await run._journal("probe", {"label": "parent-1"})
        task = asyncio.create_task(child())
        await started.wait()
        await run._journal("probe", {"label": "parent-2"})
    release.set()
    await task
    assert store.batches == [["parent-1", "parent-2"], ["child-1", "child-2"]]
    assert len(await session.entries()) == 4


@pytest.mark.parametrize("outcome", ["committed", "unknown", "cancelled", "refused"])
async def test_added_usage_is_restored_only_after_a_definite_append_refusal(outcome):
    import asyncio

    from qagents import InMemoryStore, Session, Usage
    from qagents.errors import LeaseError

    class FailingStore(InMemoryStore):
        fail = True

        async def append(self, session_id, entries):
            if self.fail:
                self.fail = False
                if outcome == "committed":
                    await super().append(session_id, entries)
                if outcome == "refused":
                    raise LeaseError("nothing written")
                if outcome == "cancelled":
                    raise asyncio.CancelledError
                raise ConnectionError("the acknowledgement was lost")
            await super().append(session_id, entries)

    store = FailingStore()
    session = Session(store)
    run = Agent(ScriptedModel([])).stream("go", session=session)._run
    run._add_usage(Usage(input_tokens=7), operation_id="call-op")
    with pytest.raises((ConnectionError, LeaseError, asyncio.CancelledError)):
        await run._journal("tool_checkpoint", {"operation_id": "call-op", "key": "one"})
    assert run._usage_to_journal.input_tokens == (7 if outcome == "refused" else 0)
    await run._journal("run_failed", {"message": "stopped"})
    entries = await session.entries()
    tokens = sum(e.data.get("added_usage", {}).get("input_tokens", 0) for e in entries)
    assert tokens == (7 if outcome in {"committed", "refused"} else 0)


@pytest.mark.parametrize("path", ["normal", "supplied", "defer"])
async def test_crash_at_the_final_state_boundary_does_not_double_state_on_resume(path):
    from qagents import DeferToolCall

    class CrashAtBoundary(InMemoryStore):
        saw_state = False
        crashed = False

        async def append(self, session_id, entries):
            kinds = {entry.kind for entry in entries}
            terminal = bool(kinds & {"tool_result", "tool_deferred"})
            if not self.crashed and self.saw_state and terminal:
                self.crashed = True
                raise ConnectionError("crash before the separate terminal append")
            await super().append(session_id, entries)
            if "state_updated" in kinds:
                self.saw_state = True
                if terminal and not self.crashed:
                    self.crashed = True
                    raise ConnectionError("crash after the atomic append")

    applications = set()

    class ResultState(Hooks):
        async def after_tool(self, ctx, call, result):
            if path == "supplied":
                ctx.state["count"] = ctx.state.get("count", 0) + 1

    @tool(idempotent=True, deferred=path == "supplied")
    async def issue(ctx: RunContext) -> str:
        """Apply an externally deduplicated action and update local state."""
        applications.add(ctx.operation_id)
        ctx.state["count"] = ctx.state.get("count", 0) + 1
        if path == "defer":
            raise DeferToolCall("waiting")
        return "ok"

    store = CrashAtBoundary()
    session = Session(store)
    agent = Agent(ScriptedModel([call("issue"), "done"]), tools=[issue], hooks=[ResultState()])
    if path == "supplied":
        paused = await agent.run("go", session=session)
        with pytest.raises(ConnectionError):
            await agent.resume(session, results={paused.pending_inputs[0].tool_call_id: "ok"})
    else:
        with pytest.raises(ConnectionError):
            await agent.run("go", session=session)
    pending = (await session.state()).pending_inputs
    answers = {item.tool_call_id: "ok" for item in pending}
    result = await agent.resume(session, results=answers)
    if result.status == "paused":
        result = await agent.resume(
            session, results={item.tool_call_id: "ok" for item in result.pending_inputs}
        )
    assert result.completed and store.crashed
    assert (await session.state()).values == {"count": 1}
    assert len(applications) == (0 if path == "supplied" else 1)
