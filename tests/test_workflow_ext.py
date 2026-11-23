"""Workflow 0.2: map, durable signals and timers, status, and run_agent with results and state."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from pydantic import BaseModel

from qagents import (
    Agent,
    InMemoryStore,
    Limits,
    Message,
    RunContext,
    Session,
    SQLiteStore,
    Usage,
    tool,
)
from qagents.errors import ConfigurationError, SessionError
from qagents.models.base import ModelResponse
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
    assert step.session_id == "workflow-step:quarterly:analyse"
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
    assert again.output == "Done." and again.session.id == "workflow-step:quarterly:analyse"


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


async def test_a_truncated_agent_step_can_be_retried_after_reopening_the_workflow():
    store = InMemoryStore()
    cut = ModelResponse(message=Message.assistant("The draft starts"), finish_reason="length")
    model = ScriptedModel([cut, "The finished draft"])
    writer = Agent(model, name="writer")
    workflow = Workflow(store, id="proposal")

    partial = await workflow.run_agent("draft", writer, "Write a proposal")
    assert partial.stop_reason == "max_output_tokens"
    assert partial.output == "The draft starts"
    assert await workflow.completed_steps() == {}

    restarted = Workflow(store, id="proposal")
    result = await restarted.run_agent("draft", writer, "Write a proposal")
    assert result.output == "The finished draft"
    assert result.run_id != partial.run_id
    assert list(await restarted.completed_steps()) == ["draft"]
    again = await Workflow(store, id="proposal").run_agent("draft", writer, "Write a proposal")
    assert again.run_id == result.run_id
    assert len(model.requests) == 2


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


# -- fixes ----------------------------------------------------------------------------------


class DiesBeforeTheStepIsRecorded(InMemoryStore):
    """Fails the first ``workflow_step`` write, like a process that dies right before it."""

    def __init__(self) -> None:
        super().__init__()
        self.dead = True

    async def append(self, session_id, entries):
        if self.dead and any(entry.kind == "workflow_step" for entry in entries):
            self.dead = False
            raise ConnectionError("the process died")
        await super().append(session_id, entries)


async def test_an_agent_step_that_completed_but_was_not_recorded_is_not_run_again():
    store = DiesBeforeTheStepIsRecorded()
    refunds: list[str] = []

    @tool
    def refund(order: str) -> str:
        """Refund an order."""
        refunds.append(order)
        return "refunded"

    model = ScriptedModel(
        [call("refund", order="A-1"), '{"approved": true, "note": "refunded A-1"}'],
        usage=Usage(requests=1, input_tokens=5),
    )
    agent = Agent(model, name="support", tools=[refund], output_type=Review)
    with pytest.raises(ConnectionError):
        await Workflow(store, id="ticket").run_agent("refund", agent, "Refund order A-1")

    again = Workflow(store, id="ticket")
    result = await again.run_agent("refund", agent, "Refund order A-1")
    assert refunds == ["A-1"] and len(model.requests) == 2  # nothing ran again
    assert result.status == "completed" and result.output == Review(
        approved=True, note="refunded A-1"
    )
    assert result.usage.input_tokens == 10 and result.turns == 2 and result.agent == "support"
    assert [m.role for m in result.new_messages] == ["user", "assistant", "tool", "assistant"]
    prompts = [m for m in await again.session("refund").messages() if m.role == "user"]
    assert len(prompts) == 1
    assert "refund" in await again.completed_steps()  # recorded now


async def test_an_unrecorded_step_from_an_older_journal_fails_closed():
    store = InMemoryStore()
    workflow = Workflow(store, id="old")
    session = workflow.session("write")
    await session.append("run_started", {"agent": "writer"}, run_id="r")
    await session.append("message", {"message": Message.user("Write").model_dump()}, run_id="r")
    answer = Message.assistant("Written.").model_dump()
    await session.append("message", {"message": answer}, run_id="r")
    await session.append("run_finished", {"status": "completed", "stop_reason": None}, run_id="r")
    model = ScriptedModel(["Written again."])
    with pytest.raises(SessionError, match="give the step a new name"):
        await workflow.run_agent("write", Agent(model, name="writer"), "Write")
    assert model.requests == []


async def test_a_step_whose_run_ended_without_completing_runs_again():
    store = InMemoryStore()

    @tool(read_only=True)
    def search(query: str) -> str:
        """Search."""
        return "more"

    looping = Agent(
        ScriptedModel([call("search", query="q")] * 2),
        name="r",
        tools=[search],
        limits=Limits(max_turns=1),
    )
    first = await Workflow(store, id="w").run_agent("research", looping, "Research")
    assert first.status == "limit_reached"
    done = await Workflow(store, id="w").run_agent(
        "research", Agent(ScriptedModel(["Found it."]), name="r"), "Research"
    )
    assert done.output == "Found it."


async def test_map_runs_sync_functions_in_threads_without_blocking_the_loop():
    workflow = Workflow(InMemoryStore(), id="sync-map")
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    def summarise(text: str) -> str:
        time.sleep(0.3)
        return text.upper()

    beat = asyncio.create_task(ticker())
    began = time.perf_counter()
    assert await workflow.map("sum", ["a", "b", "c", "d"], summarise) == ["A", "B", "C", "D"]
    took = time.perf_counter() - began
    beat.cancel()
    assert took < 1.0  # four items of 0.3 s ran at once, not one after another
    assert ticks >= 5  # and the loop kept running meanwhile


async def test_completed_steps_returns_a_copy():
    workflow = Workflow(InMemoryStore(), id="copy")
    await workflow.step("one", lambda: 1)
    steps = await workflow.completed_steps()
    steps["two"] = 2
    assert await workflow.completed_steps() == {"one": 1}
    assert await workflow.step("two", lambda: "ran") == "ran"


async def test_run_agent_hands_every_event_to_on_event():
    seen: list[str] = []

    async def narrate(event: Any) -> None:
        seen.append(event.type)

    @tool(read_only=True)
    def lookup(key: str) -> str:
        """Look something up."""
        return "value"

    agent = Agent(ScriptedModel([call("lookup", key="k"), "Done."]), name="a", tools=[lookup])
    result = await Workflow(InMemoryStore(), id="live").run_agent(
        "step", agent, "go", on_event=narrate
    )
    assert result.output == "Done."
    assert seen[0] == "run_started" and seen[-1] == "run_finished"
    assert "tool_call_started" in seen and "tool_call_finished" in seen


@pytest.mark.parametrize("later_open", [False, True])
async def test_unrecorded_agent_step_recovers_its_bound_run_not_a_later_run(later_open):
    store = DiesBeforeTheStepIsRecorded()
    wf = Workflow(store, id="bound")
    original = Agent(ScriptedModel(["original"], usage=Usage(input_tokens=13)), name="writer")
    with pytest.raises(ConnectionError):
        await wf.run_agent("write", original, "first prompt")
    session = wf.session("write")
    journal = await wf._journal.entries()
    binding = next(e for e in journal if e.kind == "workflow_agent_started")
    expected_run = next(e.run_id for e in await session.entries() if e.kind == "run_started")
    if later_open:
        await session.append("run_started", {"agent": "outsider"}, run_id="unrelated")
        await session.append(
            "message", {"message": Message.user("other prompt").model_dump()}, run_id="unrelated"
        )
    else:
        await Agent(ScriptedModel(["other"], usage=Usage(input_tokens=99)), name="outsider").run(
            "other prompt", session=session
        )
    again = Workflow(store, id="bound")
    result = await again.run_agent("write", original, "ignored")
    assert result.output == "original" and result.run_id == expected_run
    assert result.usage.input_tokens == 13 and result.agent == "writer"
    assert [m.text for m in result.messages] == ["first prompt", "original"]
    assert binding.data["run_id"] == expected_run
    assert len(original.model.requests) == 1


async def test_workflow_ids_are_injective_and_never_share_a_step_or_journal():
    from qagents.runtime import WORKFLOW_PREFIX

    store = InMemoryStore()
    pairs = [("a:b", "c"), ("a", "b:c"), ("a/b", "#%"), ("a%2Fb", "#%"), ("", "")]
    workflows = [Workflow(store, id=workflow_id) for workflow_id, _ in pairs]
    ids = [wf.session(name).id for wf, (_, name) in zip(workflows, pairs, strict=True)]
    ids += [wf._journal.id for wf in workflows]
    assert len(ids) == len(set(ids))
    assert all(session_id.startswith(WORKFLOW_PREFIX) for session_id in ids)
    parent = Workflow(store, id="a")
    child = Workflow(store, id="a:b")
    await parent.run_agent("b", Agent(ScriptedModel(["step answer"])), "go")
    await child.step("value", lambda: "workflow value")
    assert set(await parent.completed_steps()) == {"b"}
    assert await child.completed_steps() == {"value": "workflow value"}
    assert all(e.kind.startswith("workflow_") for e in await child._journal.entries())


async def test_legacy_workflow_sessions_and_signals_remain_readable_and_resumable():
    store = InMemoryStore()
    journal = Session(store, id="workflow:old:id")
    legacy_step = Session(store, id="workflow:old:id:task")
    await journal.append("workflow_step", {"name": "saved", "result": 7})
    await journal.append("workflow_agent_started", {"name": "task", "session_id": legacy_step.id})
    await Agent(ScriptedModel(["old answer"])).run("old prompt", session=legacy_step)
    wf = Workflow(store, id="old:id")
    session = wf.session("task")
    assert [m.text for m in await session.messages()] == ["old prompt", "old answer"]
    assert session.id == legacy_step.id
    assert await wf.step("saved", lambda: pytest.fail("replayed step ran")) == 7
    await Workflow.send_signal(store, "old:id", "review", "yes")
    assert await wf.wait_for_signal("review") == "yes"
    await Agent(ScriptedModel(["new answer"])).run(
        "new prompt", session=Workflow(store, "old:id").session("task")
    )
    assert len(await legacy_step.messages()) == 4


async def test_a_legacy_step_collision_is_refused_before_journal_writes():
    store = InMemoryStore()
    session = Session(store, id="workflow:a:b")
    await Agent(ScriptedModel(["step answer"])).run("go", session=session)
    before = await session.entries()
    with pytest.raises(SessionError, match="already an agent-step session"):
        await Workflow(store, "a:b").step("value", lambda: 1)
    assert await session.entries() == before


async def test_workflow_replay_and_live_cache_both_keep_the_first_result():
    store = InMemoryStore()
    workflow = Workflow(store, "first-wins")
    assert await workflow.completed_steps() == {}  # warm an empty cache
    await workflow._journal.append("workflow_step", {"name": "old", "result": "first"})
    await workflow._journal.append("workflow_step", {"name": "old", "result": "second"})
    assert await workflow.step("old", lambda: pytest.fail("cached step ran")) == "first"
    assert await Workflow(store, "first-wins").step("old", lambda: "third") == "first"

    async def race():
        await workflow._journal.append("workflow_step", {"name": "race", "result": "winner"})
        return "later"

    assert await workflow.step("race", race) == "winner"
    assert (await workflow.completed_steps())["race"] == "winner"
    assert (await Workflow(store, "first-wins").completed_steps())["race"] == "winner"


def test_workflow_docstrings_keep_external_changes_out_of_plain_steps():
    import inspect

    import qagents.workflow as workflow_module

    text = inspect.getdoc(workflow_module)
    assert 'wf.step("publish"' not in text
    assert "idempotency key" in text
    assert "external changes" in inspect.getdoc(Workflow.step)
    timer = inspect.getdoc(Workflow.wait_for_timer)
    assert 'wf.step("remind"' not in timer and 'wf.step("close"' not in timer


@pytest.mark.parametrize("legacy", [False, True])
async def test_workflow_identity_checks_ignore_unknown_journal_kinds(legacy):
    store = InMemoryStore()
    journal_id = "workflow:extensions" if legacy else "workflow-journal:extensions"
    journal = Session(store, id=journal_id)
    await journal.append("future_extension", {"value": "preserved"})
    await journal.append("workflow_step", {"name": "recorded", "result": 7})
    before = await journal.entries()
    assert await Workflow(store, "extensions").completed_steps() == {"recorded": 7}
    assert await journal.entries() == before


async def test_legacy_standalone_step_session_resolves_without_a_workflow_journal():
    store = InMemoryStore()
    legacy = Session(store, id="workflow:standalone:write")
    await Agent(ScriptedModel(["old answer"])).run("old prompt", session=legacy)
    step = Workflow(store, "standalone").session("write")
    assert [message.text for message in await step.messages()] == ["old prompt", "old answer"]
    assert step.id == legacy.id
    await Agent(ScriptedModel(["new answer"])).run(
        "new prompt", session=Workflow(store, "standalone").session("write")
    )
    assert [message.text for message in await legacy.messages()] == [
        "old prompt",
        "old answer",
        "new prompt",
        "new answer",
    ]


async def test_map_with_stable_keys_reorders_saved_results_and_rejects_duplicates():
    store = InMemoryStore()
    active = Workflow(store, id="keyed")
    items = [{"id": "a]%", "body": "first"}, {"id": "b", "body": "second"}]

    def render(item):
        return item["body"]

    def get_id(item):
        return item["id"]

    assert await active.map("items", items, render, key=get_id) == ["first", "second"]
    rebuilt = Workflow(store, id="keyed")

    def unexpected(item):
        pytest.fail("Completed work must not run again.")

    assert await rebuilt.map("items", reversed(items), unexpected, key=get_id) == [
        "second",
        "first",
    ]
    with pytest.raises(ConfigurationError, match="unique keys"):
        await rebuilt.map("items", [items[0], items[0]], unexpected, key=get_id)
    kinds = [entry.kind for entry in await active._journal.entries()]
    assert kinds.count("workflow_step_started") == 2
    assert kinds.count("workflow_step") == 2
