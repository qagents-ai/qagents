"""Workflow 0.2: map, durable signals and timers, status, and run_agent with results and state."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from pydantic import BaseModel

from qagents import Agent, InMemoryStore, RunContext, Session, SQLiteStore, Usage, tool
from qagents.errors import ConfigurationError, SessionError
from qagents.models.testing import ScriptedModel, call
from qagents.workflow import Workflow, WorkflowStatus, WorkflowSuspended


class Review(BaseModel):
    approved: bool
    note: str = ""


# -- map ----------------------------------------------------------------------------------


async def test_map_runs_items_concurrently_within_the_bound_and_keeps_input_order():
    workflow = Workflow(InMemoryStore(), id="lengths")
    running = 0
    peak = 0

    async def measure(word: str) -> int:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01 * (6 - len(word)))  # shorter words finish later
        running -= 1
        return len(word)

    words = ["a", "bb", "ccc", "dddd", "eeeee"]
    assert await workflow.map("len", words, measure, concurrency=2) == [1, 2, 3, 4, 5]
    assert peak == 2
    assert set(await workflow.completed_steps()) == {f"len[{index}]" for index in range(5)}

    running = peak = 0
    assert await Workflow(InMemoryStore(), id="all").map("len", words, measure) == [1, 2, 3, 4, 5]
    assert peak == 5  # no bound: every item at once
    assert await workflow.map("nothing", [], measure) == []
    with pytest.raises(ConfigurationError, match="at least 1"):
        await workflow.map("bad", words, measure, concurrency=0)


async def test_map_runs_only_unfinished_items_on_a_later_run():
    store = InMemoryStore()
    calls: list[int] = []
    failed: list[int] = []

    def flaky(number: int) -> Usage:
        calls.append(number)
        if number == 3 and not failed:
            failed.append(number)
            raise RuntimeError("the network went away")
        return Usage(requests=1, input_tokens=number)

    with pytest.raises(RuntimeError, match="network"):
        await Workflow(store, id="batch").map("measure", [1, 2, 3, 4], flaky, concurrency=1)
    # The failure did not stop the items after it.
    assert sorted(calls) == [1, 2, 3, 4]
    assert set(await Workflow(store, id="batch").completed_steps()) == {
        "measure[0]",
        "measure[1]",
        "measure[3]",
    }

    calls.clear()
    results = await Workflow(store, id="batch").map("measure", [1, 2, 3, 4], flaky, type=Usage)
    assert calls == [3]  # only the item that had not finished ran again
    assert [usage.input_tokens for usage in results] == [1, 2, 3, 4]
    assert all(isinstance(usage, Usage) for usage in results)


async def test_map_raises_a_real_failure_before_a_suspension():
    workflow = Workflow(InMemoryStore(), id="mixed")

    async def item(kind: str) -> str:
        if kind == "wait":
            await workflow.wait_for_signal("go")
        if kind == "fail":
            raise ValueError("bad item")
        return kind

    with pytest.raises(ValueError, match="bad item"):
        await workflow.map("items", ["wait", "ok", "fail"], item)
    with pytest.raises(WorkflowSuspended):
        await workflow.map("others", ["ok", "wait"], item)
    assert "items[1]" in await workflow.completed_steps()


async def test_cancelling_map_cancels_the_items_and_records_nothing_unfinished():
    workflow = Workflow(InMemoryStore(), id="slow")
    stopped: list[int] = []

    async def slow(number: int) -> int:
        try:
            await asyncio.sleep(10)
        finally:
            stopped.append(number)
        return number

    task = asyncio.create_task(workflow.map("slow", [1, 2, 3], slow))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sorted(stopped) == [1, 2, 3]
    assert await Workflow(workflow.store, id="slow").completed_steps() == {}


# -- signals and timers -------------------------------------------------------------------


async def test_signals_reach_a_workflow_across_instances_on_sqlite(tmp_path):
    database = tmp_path / "workflows.db"
    store = SQLiteStore(database)
    first = Workflow(store, id="launch-42")
    with pytest.raises(WorkflowSuspended) as suspended:
        await first.wait_for_signal("review", type=Review)
    assert suspended.value.signal == "review" and suspended.value.workflow_id == "launch-42"
    assert "Workflow.send_signal(store, 'launch-42', 'review', value)" in str(suspended.value)
    status = await first.status()
    assert (status.pending_signals, status.signals) == (["review"], [])

    # A web handler in another process records the reviewer's decision.
    other = SQLiteStore(database)
    await Workflow.send_signal(other, "launch-42", "review", Review(approved=True, note="ship"))
    await Workflow.send_signal(other, "launch-42", "review", {"approved": False})  # ignored
    await other.aclose()

    # The same object sees it now, and so does a new one in a new process.
    assert await first.wait_for_signal("review", type=Review) == Review(approved=True, note="ship")
    await store.aclose()
    later = Workflow(SQLiteStore(database), id="launch-42")
    assert await later.wait_for_signal("review") == {"approved": True, "note": "ship"}
    assert await later.signals() == {"review": {"approved": True, "note": "ship"}}
    status = await later.status()
    assert (status.pending_signals, status.signals) == ([], ["review"])
    # The wait was recorded once, however often the workflow reached it.
    kinds = [entry.kind for entry in await later._journal.entries()]
    assert kinds.count("workflow_waiting") == 1
    await later.store.aclose()


async def test_a_workflow_function_suspends_and_continues_where_it_stopped():
    store = InMemoryStore()
    published: list[str] = []

    async def publish_after_review(id: str) -> str:
        wf = Workflow(store, id=id)
        draft = await wf.step("draft", lambda: "Draft v1")
        verdict = await wf.wait_for_signal("review")
        if verdict != "approved":
            return "rejected"
        await wf.step("publish", lambda: published.append(draft))
        return "published"

    with pytest.raises(WorkflowSuspended):
        await publish_after_review("news")
    with pytest.raises(WorkflowSuspended):  # nothing sent yet: still waiting
        await publish_after_review("news")
    await Workflow.send_signal(store, "news", "review", "approved")
    assert await publish_after_review("news") == "published"
    assert await publish_after_review("news") == "published"
    assert published == ["Draft v1"]

    with pytest.raises(ConfigurationError, match="cannot be stored as JSON"):
        await Workflow.send_signal(store, "news", "odd", object())


async def test_a_durable_timer_suspends_until_it_is_due(monkeypatch):
    store = InMemoryStore()
    now = [1_000_000.0]
    monkeypatch.setattr(time, "time", lambda: now[0])

    with pytest.raises(WorkflowSuspended) as suspended:
        await Workflow(store, id="follow-up").wait_for_timer("grace", 3600)
    assert (suspended.value.timer, suspended.value.resume_at) == ("grace", 1_003_600.0)
    assert "due at 1970-01-12T14:46:40+00:00" in str(suspended.value)

    now[0] += 1800  # half way: the recorded due time holds, whatever seconds says now
    workflow = Workflow(store, id="follow-up")
    with pytest.raises(WorkflowSuspended):
        await workflow.wait_for_timer("grace", 1)
    assert (await workflow.status()).pending_timers == {"grace": 1_003_600.0}

    now[0] += 1800
    await Workflow(store, id="follow-up").wait_for_timer("grace", 3600)
    assert (await workflow.status()).pending_timers == {}
    await workflow.wait_for_timer("now", 0)  # a timer of zero seconds is due at once
    with pytest.raises(ConfigurationError, match="finite number of seconds"):
        await workflow.wait_for_timer("never", float("inf"))


# -- run_agent ----------------------------------------------------------------------------


def report_tools(log: list[str]) -> list[Any]:
    @tool(deferred=True)
    def fetch_report(report_id: str) -> dict:
        """Fetch a report that an external system produces."""
        log.append(f"fetch {report_id}")  # must never run
        return {}

    @tool
    def remember(ctx: RunContext[Any], fact: str) -> str:
        """Remember a fact."""
        log.append(f"state {sorted(ctx.state.items())}")
        ctx.state["fact"] = fact
        return "saved"

    return [fetch_report, remember]


async def test_run_agent_passes_results_to_a_deferred_tool_and_state_to_the_session():
    store = InMemoryStore()
    log: list[str] = []
    model = ScriptedModel(
        [call("fetch_report", report_id="R-7"), call("remember", fact="revenue up"), "Done."]
    )
    agent = Agent(model, name="analyst", tools=report_tools(log))

    workflow = Workflow(store, id="quarterly")
    paused = await workflow.run_agent("analyse", agent, "Analyse R-7", state={"quarter": "Q3"})
    assert paused.status == "paused" and await workflow.completed_steps() == {}
    (waiting,) = paused.pending_inputs

    status = await workflow.status()
    (step,) = status.open_agent_steps
    assert (step.name, step.status, step.run_id) == ("analyse", "paused", paused.run_id)
    assert step.session_id == "workflow:quarterly:analyse"
    assert [item.tool_call_id for item in step.pending_inputs] == [waiting.tool_call_id]

    # Later, in another process, the external system's result arrives.
    later = Workflow(store, id="quarterly")
    done = await later.run_agent(
        "analyse",
        agent,
        "Analyse R-7",
        results={waiting.tool_call_id: {"revenue": 120}},
        state={"source": "erp"},
    )
    assert done.status == "completed" and done.output == "Done."
    assert model.requests[1].messages[-1].tool_results[0].content == '{"revenue": 120}'
    assert log == ["state [('quarter', 'Q3'), ('source', 'erp')]"]  # fetch_report never ran
    assert (await later.session("analyse").state()).values == {
        "quarter": "Q3",
        "source": "erp",
        "fact": "revenue up",
    }
    status = await later.status()
    assert (status.completed_steps, status.open_agent_steps) == (["analyse"], [])

    # Finished: answered from the journal, with results= ignored.
    again = await Workflow(store, id="quarterly").run_agent(
        "analyse", Agent(ScriptedModel([]), name="analyst"), "Analyse R-7", results={"x": 1}
    )
    assert again.output == "Done." and again.session.id == "workflow:quarterly:analyse"


async def test_run_agent_refuses_what_it_cannot_pass_on():
    workflow = Workflow(InMemoryStore(), id="strict")
    agent = Agent(ScriptedModel(["hi"]), name="a")
    with pytest.raises(SessionError, match="has no open run"):
        await workflow.run_agent("greet", agent, "hello", results={"call_1": "x"})
    with pytest.raises(ConfigurationError, match=r"\['parent'\].*deps, limits, settings"):
        await workflow.run_agent("greet", agent, "hello", parent=RunContext())
    with pytest.raises(ConfigurationError, match="history"):
        await workflow.run_agent("greet", agent, "hello", history=[])
    assert (await workflow.run_agent("greet", agent, "hello", deps={"user": 1})).output == "hi"


async def test_two_agent_steps_with_one_name_cannot_run_at_once():
    workflow = Workflow(InMemoryStore(), id="clash")
    agent = Agent(ScriptedModel(["one", "two"], delay=0.01), name="a")
    first = asyncio.create_task(workflow.run_agent("same", agent, "go"))
    await asyncio.sleep(0)
    with pytest.raises(ConfigurationError, match="two steps named 'same'"):
        await workflow.run_agent("same", agent, "go")
    assert (await first).output == "one"


async def test_a_replayed_agent_step_returns_its_output_in_the_output_type():
    store = InMemoryStore()

    def reviewer() -> Agent:
        return Agent(
            ScriptedModel(['{"approved": true, "note": "fine"}']),
            name="reviewer",
            output_type=Review,
        )

    first = await Workflow(store, id="review").run_agent("check", reviewer(), "Review it")
    replayed = await Workflow(store, id="review").run_agent("check", reviewer(), "Review it")
    assert first.output == replayed.output == Review(approved=True, note="fine")

    class Stricter(BaseModel):
        score: int

    changed = Agent(ScriptedModel([]), name="reviewer", output_type=Stricter)
    with pytest.raises(SessionError, match="give the step a new name"):
        await Workflow(store, id="review").run_agent("check", changed, "Review it")


async def test_status_lists_paused_and_failed_agent_steps():
    store = InMemoryStore()

    @tool(requires_approval=True)
    def send(to: str) -> str:
        """Send a message."""
        return "sent"

    workflow = Workflow(store, id="ops")
    await workflow.step("prepare", lambda: 1)
    mailer = Agent(ScriptedModel([call("send", to="team")]), name="mailer", tools=[send])
    paused = await workflow.run_agent("mail", mailer, "Mail the team")

    broken = Agent(ScriptedModel([RuntimeError("provider down")]), name="writer")
    with pytest.raises(RuntimeError):
        await workflow.run_agent("write", broken, "Write")

    status = await Workflow(store, id="ops").status()
    assert isinstance(status, WorkflowStatus) and status.completed_steps == ["prepare"]
    by_name = {step.name: step for step in status.open_agent_steps}
    assert by_name["mail"].status == "paused"
    assert [item.name for item in by_name["mail"].pending_approvals] == ["send"]
    assert (
        paused.pending_approvals[0].tool_call_id
        == by_name["mail"].pending_approvals[0].tool_call_id
    )
    assert by_name["write"].status == "failed" and by_name["write"].pending_approvals == []

    # A step whose run was closed by hand is no longer open.
    session = Session(store, id=by_name["write"].session_id)
    await session.append("run_finished", {"status": "abandoned"}, run_id=by_name["write"].run_id)
    assert [step.name for step in (await workflow.status()).open_agent_steps] == ["mail"]
