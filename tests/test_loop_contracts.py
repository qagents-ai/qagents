"""What the loop promises about tools, outputs, hooks and state.

Each test here is a regression test for a defect found in 0.2.
"""

from __future__ import annotations

import asyncio
import functools
import threading
import time
from typing import Any, Literal

import pytest
from pydantic import BaseModel

from qagents import Agent, Hooks, Limits, Permissions, RunContext, Session, prompts, tool
from qagents import tools as tools_module
from qagents.errors import ModelBadRequestError, ModelRetry
from qagents.hooks import HookChain
from qagents.models.base import ModelSettings
from qagents.models.testing import ScriptedModel, call
from qagents.permissions import Decision
from qagents.toolkits.human import ask_user
from qagents.toolkits.planning import todo_tools

# -- arguments are checked before anyone is asked (F131, F174, L1) --------------------------


async def test_a_deferred_call_with_invalid_arguments_gets_an_error_and_never_pauses():
    model = ScriptedModel([call("ask_user", q="Which city?"), call("ask_user", question="Which?")])
    result = await Agent(model, tools=[ask_user]).run("go")

    rejected = model.requests[1].messages[-1].tool_results[0]
    assert rejected.is_error and rejected.content.startswith("Invalid arguments for 'ask_user'")
    assert result.status == "paused"
    (question,) = result.pending_inputs
    assert question.arguments == {"question": "Which?"}


async def test_approvers_never_see_a_call_whose_arguments_are_invalid():
    asked: list[dict[str, Any]] = []
    checked: list[dict[str, Any]] = []

    @tool(requires_approval=True)
    def refund(order_id: str, amount: int) -> str:
        """Refund an order."""
        return "refunded"

    def approve(ctx, target, call_part, reason):
        asked.append(call_part.arguments)
        return True

    def check(ctx, target, call_part):
        checked.append(call_part.arguments)
        return None

    model = ScriptedModel([call("refund", order_id="A", amount="a lot"), "Sorry."])
    permissions = Permissions(on_ask=approve, can_use_tool=check)
    result = await Agent(model, tools=[refund], permissions=permissions).run("go")
    assert asked == [] and checked == []
    assert result.messages[2].tool_results[0].content.startswith("Invalid arguments for 'refund'")

    paused = await Agent(
        ScriptedModel([call("refund", order_id="A", amount="a lot"), "Sorry."]), tools=[refund]
    ).run("go")
    assert paused.status == "completed" and paused.pending_approvals == []


async def test_refused_calls_do_not_count_toward_max_tool_calls():
    log: list[str] = []

    @tool
    def write(text: str) -> str:
        """Write."""
        log.append(text)
        return "written"

    @tool(read_only=True)
    def read() -> str:
        """Read."""
        log.append("read")
        return "content"

    model = ScriptedModel([call("write", text="a"), call("write", text="b"), call("read"), "done"])
    agent = Agent(
        model,
        tools=[write, read],
        permissions=Permissions(mode="read_only"),
        limits=Limits(max_tool_calls=2),
    )
    result = await agent.run("go")
    assert log == ["read"] and result.completed


async def test_a_call_declined_after_a_pause_does_not_count_toward_max_tool_calls():
    log: list[str] = []

    @tool(requires_approval=True)
    def delete(path: str) -> str:
        """Delete a file."""
        log.append(f"delete {path}")
        return "deleted"

    @tool
    def write(text: str) -> str:
        """Write."""
        log.append(f"write {text}")
        return "written"

    session = Session()
    model = ScriptedModel([call("delete", path="a"), call("write", text="b"), "done"])
    agent = Agent(model, tools=[delete, write], limits=Limits(max_tool_calls=1))
    paused = await agent.run("go", session=session)
    (approval,) = paused.pending_approvals
    assert (await session.state()).run_tool_calls == 0  # waiting, not yet allowed to run
    resumed = await agent.resume(session, approvals={approval.tool_call_id: False})
    assert log == ["write b"] and resumed.completed


async def test_an_approved_call_that_started_before_a_crash_still_counts():
    log: list[str] = []
    crash = [True]

    @tool(requires_approval=True)
    def pay(amount: int) -> str:
        """Pay."""
        log.append(f"pay {amount}")
        if crash[0]:
            raise KeyboardInterrupt  # the process dies while the call runs
        return "paid"

    @tool
    def send(text: str) -> str:
        """Send."""
        log.append(f"send {text}")
        return "sent"

    session = Session()
    model = ScriptedModel([call("pay", amount=5), call("send", text="x"), "done"])
    agent = Agent(model, tools=[pay, send], limits=Limits(max_tool_calls=1))
    (approval,) = (await agent.run("go", session=session)).pending_approvals
    with pytest.raises(KeyboardInterrupt):
        await agent.resume(session, approvals={approval.tool_call_id: True})
    crash[0] = False
    resumed = await agent.resume(session)
    assert (resumed.status, resumed.stop_reason) == ("limit_reached", "max_tool_calls")
    assert log == ["pay 5"]  # send would be a second call under max_tool_calls=1


# -- results supplied on resume go through after_tool (E2) ----------------------------------


async def test_a_result_supplied_on_resume_goes_through_after_tool():
    class Redact(Hooks):
        async def after_tool(self, ctx, call_part, result):
            return result.model_copy(update={"content": result.content.replace("1234", "****")})

    session = Session()
    agent = Agent(
        ScriptedModel([call("ask_user", question="PIN?"), "Thanks."]),
        tools=[ask_user],
        hooks=[Redact()],
    )
    paused = await agent.run("go", session=session)
    (question,) = paused.pending_inputs
    done = await agent.resume(session, results={question.tool_call_id: "it is 1234"})
    assert done.messages[2].tool_results[0].content == "it is ****"


# -- tool timeouts tell the truth (F18, L18, F90, F138, F92) --------------------------------


async def test_a_sync_tool_that_times_out_is_reported_as_still_running():
    finished = threading.Event()

    @tool(timeout=0.2)
    def transfer(amount: int) -> str:
        """Move money."""
        time.sleep(0.5)
        finished.set()
        return "sent"

    result = await Agent(ScriptedModel([call("transfer", amount=100), "ok"]), tools=[transfer]).run(
        "go"
    )
    reported = result.messages[2].tool_results[0]
    assert reported.content == prompts.TOOL_TIMEOUT_ABANDONED.format(name="transfer", seconds=0.2)
    assert reported.metadata == {"timeout": 0.2}
    assert finished.wait(2)  # the thread did finish its work


async def test_a_timeout_raised_by_the_tool_itself_is_a_failure_not_the_tool_timeout():
    @tool(read_only=True)
    async def fetch() -> str:
        """Fetch."""
        await asyncio.wait_for(asyncio.sleep(10), 0.01)
        return "never"

    result = await Agent(ScriptedModel([call("fetch"), "ok"]), tools=[fetch]).run("go")
    reported = result.messages[2].tool_results[0]
    assert reported.content.startswith("'fetch' failed: TimeoutError")
    assert reported.metadata["exception"] == "TimeoutError"


async def test_the_timeout_covers_an_awaitable_returned_by_a_sync_wrapper():
    async def slow_lookup(key: str) -> str:
        await asyncio.sleep(3)
        return f"found {key}"

    @functools.wraps(slow_lookup)
    def logged(*args: Any, **kwargs: Any) -> Any:
        return slow_lookup(*args, **kwargs)

    lookup = tool(logged, read_only=True)
    began = time.monotonic()
    result = await Agent(
        ScriptedModel([call("slow_lookup", key="x"), "ok"]),
        tools=[lookup],
        limits=Limits(tool_timeout_s=0.3),
    ).run("go")
    assert time.monotonic() - began < 2
    assert "did not finish within 0.3 seconds" in result.messages[2].tool_results[0].content


async def test_sync_tools_run_on_their_own_threads_and_abandoned_ones_are_bounded(monkeypatch):
    monkeypatch.setattr(tools_module, "_EXECUTOR", None)  # a pool of one, for this test only
    monkeypatch.setattr(tools_module, "_EXECUTOR_SIZE", 0)
    monkeypatch.setattr(tools_module, "TOOL_THREADS", 1)
    monkeypatch.setattr(tools_module, "_ABANDONED", set())
    names: list[str] = []
    release = threading.Event()

    @tool(read_only=True, timeout=0.1)
    def hang() -> str:
        """Hang."""
        names.append(threading.current_thread().name)
        release.wait(5)
        return "late"

    try:
        model = ScriptedModel([call("hang"), call("hang"), "done"])
        result = await Agent(model, tools=[hang]).run("go")
        first, second = (m.tool_results[0].content for m in result.messages if m.role == "tool")
        assert first == prompts.TOOL_TIMEOUT_ABANDONED.format(name="hang", seconds=0.1)
        # The only worker thread is still held by the first call: refused, not queued.
        assert second == prompts.TOOL_THREADS_EXHAUSTED.format(name="hang", count=1)
        assert names and names[0].startswith("qagents-tool")
    finally:
        release.set()
        executor = tools_module._EXECUTOR
        if executor is not None:
            executor.shutdown(wait=True)


async def test_a_sync_call_that_never_got_a_thread_says_it_did_not_run(monkeypatch):
    monkeypatch.setattr(tools_module, "_EXECUTOR", None)  # a pool of one, for this test only
    monkeypatch.setattr(tools_module, "_EXECUTOR_SIZE", 0)
    monkeypatch.setattr(tools_module, "TOOL_THREADS", 1)
    monkeypatch.setattr(tools_module, "_ABANDONED", set())
    ran: list[str] = []

    @tool(read_only=True)
    def busy(name: str) -> str:
        """Work a while."""
        ran.append(name)
        time.sleep(0.4)
        return name

    @tool(read_only=True, timeout=0.1)
    def quick(name: str) -> str:
        """Work briefly."""
        ran.append(name)
        return name

    try:
        model = ScriptedModel([[call("busy", name="a"), call("quick", name="b")], "done"])
        result = await Agent(model, tools=[busy, quick]).run("go")
        a, b = result.messages[2].tool_results
        assert a.content == "a"
        assert b.content == prompts.TOOL_NOT_STARTED.format(name="quick", seconds=0.1)
        assert ran == ["a"]
    finally:
        executor = tools_module._EXECUTOR
        if executor is not None:
            executor.shutdown(wait=True)


async def test_calls_that_run_one_at_a_time_never_overlap_an_abandoned_thread():
    events: list[str] = []
    active = peak = 0
    lock = threading.Lock()
    pauses = {"a": 0.5, "b": 0.0, "c": 1.0, "d": 0.0}

    @tool(timeout=0.3)
    def write(text: str) -> str:
        """Write (not read-only: calls run one at a time)."""
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        events.append(f"start {text}")
        time.sleep(pauses[text])
        events.append(f"end {text}")
        with lock:
            active -= 1
        return f"wrote {text}"

    # a times out at 0.3 s and ends at 0.5 s: b waits for it, then runs.
    model = ScriptedModel([[call("write", text="a"), call("write", text="b")], "done"])
    result = await Agent(model, tools=[write]).run("go")
    a, b = result.messages[2].tool_results
    assert a.content == prompts.TOOL_TIMEOUT_ABANDONED.format(name="write", seconds=0.3)
    assert b.content == "wrote b"
    assert events == ["start a", "end a", "start b", "end b"] and peak == 1

    # c still runs when d's own timeout has passed: d is not run, and the model is told why.
    events.clear()
    model = ScriptedModel([[call("write", text="c"), call("write", text="d")], "done"])
    result = await Agent(model, tools=[write]).run("go")
    c, d = result.messages[2].tool_results
    assert d.is_error
    assert d.content == prompts.TOOL_WAITED_FOR_EARLIER.format(name="write", seconds=0.3)
    assert "start d" not in events and peak == 1
    await asyncio.sleep(0.6)  # let c's thread finish before the next test


async def test_a_call_refused_while_it_waited_for_an_earlier_thread_does_not_count():
    release = threading.Event()
    ran: list[str] = []

    @tool(timeout=0.25)
    def write(text: str) -> str:
        """Write (not read-only: calls run one at a time)."""
        if text == "a":
            release.wait(5)  # times out, and its thread keeps running
        ran.append(text)
        return f"wrote {text}"

    class Release(Hooks):
        async def before_model(self, ctx, request):
            if ctx.turn == 2:
                release.set()  # b was refused by now; let a's thread end
            return request

    model = ScriptedModel(
        [[call("write", text="a"), call("write", text="b")], call("write", text="c"), "done"]
    )
    agent = Agent(model, tools=[write], hooks=[Release()], limits=Limits(max_tool_calls=2))
    result = await agent.run("go")
    b = result.messages[2].tool_results[1]
    assert b.content == prompts.TOOL_WAITED_FOR_EARLIER.format(name="write", seconds=0.25)
    assert result.completed and ran[-1] == "c"  # a and c count; b never ran


# -- outputs (L5, L7, L8, L23, L25) ---------------------------------------------------------


class Invoice(BaseModel):
    number: str
    total: float


async def test_a_rejected_final_answer_emits_output_rejected_in_every_mode():
    def positive(ctx, invoice: Invoice) -> None:
        if invoice.total <= 0:
            raise ModelRetry("The total must be positive.")

    native = Agent(
        ScriptedModel(['{"number": "A", "total": 0}', '{"number": "A", "total": 5}']),
        output_type=Invoice,
        output_validators=[positive],
    )
    events = [event async for event in native.stream("go")]
    (rejected,) = [e for e in events if e.type == "output_rejected"]
    assert (rejected.reason, rejected.attempt) == ("The total must be positive.", 1)

    in_tool_mode = Agent(
        ScriptedModel(["plain text", call("final_answer", number="A", total=5)]),
        output_type=Invoice,
        output_mode="tool",
    )
    events = [event async for event in in_tool_mode.stream("go")]
    assert [e.type for e in events].count("output_rejected") == 1


async def test_tool_mode_without_an_output_type_accepts_a_plain_text_answer():
    model = ScriptedModel(["Paris."])
    result = await Agent(model, output_mode="tool").run("Capital of France?")
    assert result.output == "Paris." and len(model.requests) == 1
    system = model.requests[0].messages[0].text
    assert system == prompts.OUTPUT_TOOL_TEXT_INSTRUCTIONS.format(name="final_answer")
    assert [spec.name for spec in model.requests[0].tools] == ["final_answer"]


async def test_native_mode_wraps_a_schema_that_is_not_an_object():
    model = ScriptedModel(['{"value": "billing"}'])
    agent = Agent(model, output_type=Literal["billing", "technical"])
    result = await agent.run("My invoice is wrong")
    assert result.output == "billing"
    schema = model.requests[0].output.json_schema
    assert schema["type"] == "object" and schema["required"] == ["value"]
    assert schema["properties"]["value"]["enum"] == ["billing", "technical"]

    bare = await Agent(
        ScriptedModel(['"technical"']), output_type=Literal["billing", "technical"]
    ).run("My router is down")
    assert bare.output == "technical"  # a model that answers without the wrapper still works


async def test_assistant_messages_say_which_model_answered():
    result = await Agent(ScriptedModel(["hi"], name="small")).run("hello")
    metadata = result.messages[-1].metadata
    assert (metadata["model"], metadata["provider"], metadata["finish_reason"]) == (
        "small",
        "test",
        "stop",
    )


# -- hooks and state (F23, F15, F86) ---------------------------------------------------------


async def test_every_on_run_end_hook_runs_even_when_one_raises():
    ended: list[str] = []

    class Broken(Hooks):
        async def on_run_end(self, ctx, result, error):
            raise ConnectionError("metrics backend down")

    class Tracer(Hooks):
        async def on_run_end(self, ctx, result, error):
            ended.append("closed")

    with pytest.raises(ConnectionError):
        await HookChain([Broken(), Tracer()]).on_run_end(RunContext(), None, None)
    assert ended == ["closed"]

    with pytest.raises(ConnectionError):
        await Agent(ScriptedModel(["hi"]), hooks=[Broken(), Tracer()]).run("go")
    assert ended == ["closed", "closed"]  # once per run, also for the hook after the broken one


async def test_state_changed_in_before_tool_is_journaled_before_the_tool_runs():
    session = Session()
    seen: list[list[str]] = []

    class Reserve(Hooks):
        async def before_tool(self, ctx, call_part):
            ctx.state["reserved"] = "A"

    @tool
    async def book() -> str:
        """Book."""
        entries = await session.entries()
        seen.append([e.kind for e in entries if e.kind in ("tool_started", "state_updated")])
        return "booked"

    await Agent(ScriptedModel([call("book"), "done"]), tools=[book], hooks=[Reserve()]).run(
        "go", session=session
    )
    assert seen == [["tool_started", "state_updated"]]


async def test_state_changed_in_on_event_for_the_last_state_update_is_saved():
    class Seen(Hooks):
        async def on_event(self, ctx, event):
            ctx.state.setdefault("seen", []).append(event.type)

    session = Session()
    await Agent(ScriptedModel(["hi"]), hooks=[Seen()]).run("go", session=session)
    saved = (await session.state()).values["seen"]
    assert saved[-1] == "state_updated"  # the change made for the final state_updated is kept


# -- settings, handoffs, prompts (F108, F87, L22, F91, F179) --------------------------------


class Picky(ScriptedModel):
    def validate_settings(self, settings: ModelSettings) -> None:
        if settings.extra:
            raise ModelBadRequestError(f"does not accept {sorted(settings.extra)}")


async def test_per_run_settings_are_checked_against_every_handoff_target():
    billing = Agent(Picky([]), name="billing")
    lookup_log: list[str] = []

    @tool(read_only=True)
    def lookup(customer: str) -> str:
        """Look up a customer."""
        lookup_log.append(customer)
        return "found"

    front = Agent(
        ScriptedModel([call("lookup", customer="acme"), call("transfer_to_billing")]),
        name="front",
        tools=[lookup],
        handoffs=[billing],
    )
    with pytest.raises(ModelBadRequestError, match="does not accept"):
        await front.run("invoice", settings=ModelSettings(extra={"store": True}))
    assert lookup_log == []  # nothing ran


async def test_agents_can_hand_off_to_each_other():
    triage = Agent(ScriptedModel([call("transfer_to_billing"), "Back at triage."]), name="triage")
    billing = Agent(ScriptedModel([call("transfer_to_triage")]), name="billing")
    triage.handoffs.append(billing)
    billing.handoffs.append(triage)
    result = await triage.run("help")
    assert result.agent == "triage" and result.output == "Back at triage."


async def test_permission_refusals_and_toolkit_schemas_use_prompt_constants(monkeypatch):
    @tool
    def delete(path: str) -> str:
        """Delete."""
        return "deleted"

    class Silent(Permissions):
        async def decide(self, ctx, target, call_part):
            return Decision(behavior="deny")  # a refusal without a reason

    monkeypatch.setattr(prompts, "PERMISSION_REASON_DEFAULT", "nope.")
    result = await Agent(
        ScriptedModel([call("delete", path="x"), "ok"]), tools=[delete], permissions=Silent()
    ).run("go")
    content = result.messages[2].tool_results[0].content
    assert content == prompts.PERMISSION_DENIED.format(name="delete", reason="nope.")

    properties = ask_user.parameters["properties"]
    assert properties["question"]["description"] == prompts.ASK_USER_QUESTION_DESCRIPTION
    assert properties["choices"]["description"] == prompts.ASK_USER_CHOICES_DESCRIPTION
    monkeypatch.setattr(prompts, "TODO_STATUS_DESCRIPTION", "Where the item stands.")
    write_todos = todo_tools()[0]
    item = write_todos.parameters["$defs"]["TodoItem"]
    assert item["properties"]["status"]["description"] == "Where the item stands."
    assert item["description"] == prompts.TODO_ITEM_DESCRIPTION
    todos = write_todos.parameters["properties"]["todos"]
    assert todos["description"] == prompts.WRITE_TODOS_TODOS_DESCRIPTION
