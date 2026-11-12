"""What a tool can ask of its run: checkpoints, the run's permissions, an early state save,
counted child usage, its own result part, and tool calls that do not come from the model
(``RunContext.call_tool``). Each test here fails without the API it names."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from qagents import (
    Agent,
    Approval,
    Hooks,
    InMemoryStore,
    Limits,
    Permissions,
    RunContext,
    Session,
    SessionError,
    prompts,
    tool,
)
from qagents.errors import ToolError
from qagents.events import StateUpdated, ToolCallFinished, ToolCallStarted
from qagents.messages import ToolResultPart, Usage
from qagents.models.testing import ScriptedModel, call
from qagents.session import derive_state


def results_of(result: Any) -> list[ToolResultPart]:
    return [part for message in result.messages for part in message.tool_results]


def entries_of_kind(entries: list[Any], kind: str) -> list[Any]:
    return [entry for entry in entries if entry.kind == kind]


# -- C1: RunContext.checkpoint() and the tool_checkpoint entry ------------------------------


async def test_a_checkpoint_is_journaled_before_the_tool_goes_on() -> None:
    seen: dict[str, Any] = {}
    session = Session(InMemoryStore(), id="cp")

    @tool
    async def start_job(ctx: RunContext[Any], name: str) -> str:
        """Start a job somewhere else."""
        await ctx.checkpoint("job", {"id": f"job-{name}"})
        # Record first, act second: the fact is in the journal before the tool acts on it.
        seen["entries"] = entries_of_kind(await session.entries(), "tool_checkpoint")
        seen["checkpoints"] = dict(ctx.checkpoints)
        seen["ids"] = (ctx.tool_call_id, ctx.operation_id)
        return "started"

    agent = Agent(ScriptedModel([call("start_job", name="a"), "ok"]), tools=[start_job])
    await agent.run("go", session=session)
    (entry,) = seen["entries"]
    assert entry.data == {
        "tool_call_id": seen["ids"][0],
        "operation_id": seen["ids"][1],
        "key": "job",
        "value": {"id": "job-a"},
    }
    assert seen["checkpoints"] == {"job": {"id": "job-a"}}


async def test_a_call_repeated_after_a_crash_gets_its_checkpoints() -> None:
    log: list[str] = []
    crash = [True]

    @tool(idempotent=True)
    async def upload(ctx: RunContext[Any], path: str) -> str:
        """Upload a file once."""
        if "upload" in ctx.checkpoints:  # the earlier attempt started it: re-attach
            log.append(f"re-attach {ctx.checkpoints['upload']}")
            return "uploaded"
        log.append("start upload")
        await ctx.checkpoint("upload", "u-1")
        if crash[0]:
            raise KeyboardInterrupt  # the process dies after the upload started
        return "uploaded"

    session = Session()
    agent = Agent(ScriptedModel([call("upload", path="a.txt"), "done"]), tools=[upload])
    with pytest.raises(KeyboardInterrupt):
        await agent.run("go", session=session)
    assert (await session.state()).checkpoints  # it survived in the journal
    crash[0] = False
    resumed = await agent.resume(session)
    assert resumed.status == "completed"
    assert log == ["start upload", "re-attach u-1"]  # the upload was not started twice


async def test_checkpoints_belong_to_their_call_and_a_finished_call_never_runs_again() -> None:
    log: list[str] = []
    crash = [True]

    @tool(idempotent=True)
    async def job(ctx: RunContext[Any], name: str) -> str:
        """Run a job."""
        log.append(f"{name} sees {sorted(ctx.checkpoints)}")
        await ctx.checkpoint(f"started-{name}", True)
        if name == "b" and crash[0]:
            raise KeyboardInterrupt
        return f"{name} done"

    session = Session()
    model = ScriptedModel([[call("job", name="a"), call("job", name="b")], "done"])
    agent = Agent(model, tools=[job])
    with pytest.raises(KeyboardInterrupt):
        await agent.run("go", session=session)
    crash[0] = False
    resumed = await agent.resume(session)
    assert resumed.status == "completed"
    # a had a result: it never runs again, whatever it checkpointed. b sees only its own.
    assert log == ["a sees []", "b sees []", "b sees ['started-b']"]


async def test_checkpoints_of_a_closed_turn_are_dropped() -> None:
    @tool(idempotent=True)
    async def job(ctx: RunContext[Any]) -> str:
        """Run a job."""
        await ctx.checkpoint("k", 1)
        return "ok"

    session = Session()
    await Agent(ScriptedModel([call("job"), "done"]), tools=[job]).run("go", session=session)
    entries = await session.entries()
    assert entries_of_kind(entries, "tool_checkpoint")
    assert derive_state(entries).checkpoints == {}  # no call of a finished run repeats


async def test_checkpoint_refuses_values_that_are_not_json_and_calls_that_ended() -> None:
    kept: list[RunContext[Any]] = []
    errors: list[str] = []

    @tool
    async def first(ctx: RunContext[Any]) -> str:
        """Keep the context."""
        kept.append(ctx)
        try:
            await ctx.checkpoint("bad", {1, 2})
        except SessionError as error:
            errors.append(f"json: {error}")
        return "ok"

    @tool
    async def second(ctx: RunContext[Any]) -> str:
        """Use the first call's context after it ended."""
        try:
            await kept[0].checkpoint("late", 1)
        except RuntimeError as error:
            errors.append(f"ended: {error}")
        return "ok"

    session = Session()
    model = ScriptedModel([call("first"), call("second"), "done"])
    await Agent(model, tools=[first, second]).run("go", session=session)
    assert [text.split(":")[0] for text in errors] == ["json", "ended"]
    assert not entries_of_kind(await session.entries(), "tool_checkpoint")
    with pytest.raises(RuntimeError, match="inside a tool call"):
        await RunContext().checkpoint("k", 1)
    assert RunContext().checkpoints == {}


# -- C2, C3, C5: permissions, save_state, counted -------------------------------------------


async def test_permissions_are_the_runs_with_the_inherited_on_ask() -> None:
    seen: list[Any] = []

    def on_ask(ctx: Any, tool: Any, call: Any, reason: Any) -> bool:
        return True

    @tool
    async def look(ctx: RunContext[Any]) -> str:
        """Report the rules."""
        seen.append(ctx.permissions)
        return "ok"

    rules = Permissions(deny=["x"], on_ask=on_ask)
    child = Agent(ScriptedModel([call("look"), "child done"]), name="child", tools=[look])
    lead = Agent(ScriptedModel([call("child", task="t"), "done"]), tools=[child], permissions=rules)
    await lead.run("go")
    (inherited,) = seen
    assert inherited is not rules and inherited.on_ask is on_ask  # the child's, with on_ask
    assert RunContext().permissions is None


async def test_save_state_journals_the_state_before_the_tool_acts() -> None:
    seen: list[list[str]] = []
    session = Session(InMemoryStore(), id="state")

    @tool
    async def grant(ctx: RunContext[Any]) -> str:
        """Use a one-shot grant."""
        ctx.state["grant"] = "used"
        await ctx.save_state()
        seen.append([entry.kind for entry in await session.entries()])
        return "ok"

    stream = Agent(ScriptedModel([call("grant"), "done"]), tools=[grant]).stream(
        "go", session=session
    )
    events = [event async for event in stream]
    assert seen[0][-1] == "state_updated"  # journaled before the tool went on
    kinds = [type(event) for event in events]
    assert kinds.index(StateUpdated) < kinds.index(ToolCallFinished)
    await RunContext().save_state()  # outside a run: nothing to do, no error


async def test_counted_usage_is_not_added_twice() -> None:
    @tool
    async def child(ctx: RunContext[Any]) -> str:
        """Report a child's usage piece by piece, then its total."""
        piece = Usage(requests=1, input_tokens=7, cost_usd=0.0)
        ctx.add_usage(piece)
        total = piece  # the child's result.usage: the same object
        ctx.counted(total)
        ctx.add_usage(total)  # code written for 0.2 adds it again: recognised
        return "ok"

    result = await Agent(ScriptedModel([call("child"), "done"]), tools=[child]).run("go")
    assert result.usage.input_tokens == 7
    RunContext().counted(Usage())  # outside a run: nothing to do


# -- C4: usage rides on tool_checkpoint entries ----------------------------------------------


async def test_usage_added_before_a_checkpoint_survives_a_crash() -> None:
    crash = [True]

    @tool(idempotent=True)
    async def turn(ctx: RunContext[Any]) -> str:
        """Use a model elsewhere, then record that it finished."""
        if "finished" in ctx.checkpoints:
            return "finished earlier"
        ctx.add_usage(Usage(requests=1, input_tokens=100, cost_usd=0.0))
        await ctx.checkpoint("finished", True)
        if crash[0]:
            raise KeyboardInterrupt  # dies before the result is journaled
        return "finished"

    session = Session()
    agent = Agent(ScriptedModel([call("turn"), "done"]), tools=[turn])
    with pytest.raises(KeyboardInterrupt):
        await agent.run("go", session=session)
    assert (await session.state()).run_usage.input_tokens == 100  # carried by the checkpoint
    crash[0] = False
    resumed = await agent.resume(session)
    assert resumed.usage.input_tokens == 100  # once


# -- C6: a returned ToolResultPart and ToolError.metadata -----------------------------------


async def test_a_tool_can_return_its_own_result_part() -> None:
    @tool
    async def check() -> ToolResultPart:
        """Check something."""
        part = ToolResultPart(tool_call_id="", name="", content="2 problems", is_error=True)
        part.metadata["problems"] = 2
        return part

    result = await Agent(ScriptedModel([call("check"), "done"]), tools=[check]).run("go")
    (part,) = results_of(result)
    assert (part.content, part.is_error, part.metadata) == ("2 problems", True, {"problems": 2})
    assert part.tool_call_id and part.name == "check"  # filled in by the loop


class Failed(ToolError):
    def __init__(self, message: str, metadata: dict[str, Any]) -> None:
        super().__init__(message)
        self.metadata = metadata


async def test_tool_error_metadata_is_copied_and_bad_metadata_is_dropped() -> None:
    sent: list[str] = []

    @tool
    async def fails() -> str:
        """Fail with data."""
        raise Failed("it broke", {"session": "s-1"})

    @tool
    async def odd() -> ToolResultPart:
        """Return metadata that is not JSON."""
        return ToolResultPart(tool_call_id="", name="", content="x", metadata={"o": object()})

    @tool
    def send(to: str) -> ToolResultPart:
        """Send a message (it did its work), with metadata that is not JSON."""
        sent.append(to)
        return ToolResultPart(
            tool_call_id="", name="", content="sent", metadata={"s": float("nan")}
        )

    @tool
    async def fails_oddly() -> str:
        """Fail with data that is not JSON."""
        raise Failed("quota exceeded", {"at": object()})

    model = ScriptedModel(
        [[call("fails"), call("odd"), call("send", to="a"), call("fails_oddly")], "done"]
    )
    session = Session()
    result = await Agent(model, tools=[fails, odd, send, fails_oddly]).run("go", session=session)
    assert result.status == "completed"  # the run goes on: it was the tool's failure
    failed, bad, delivered, refused_part = results_of(result)
    assert (failed.content, failed.is_error, failed.metadata) == (
        "it broke",
        True,
        {"session": "s-1"},
    )
    # The tool ran: its content and error flag stay, only the metadata is dropped.
    assert (bad.content, bad.is_error) == ("x", False)
    assert list(bad.metadata) == ["metadata_error"]
    assert bad.metadata["metadata_error"].startswith("PydanticSerializationError")
    assert (delivered.content, delivered.is_error, sent) == ("sent", False, ["a"])
    assert delivered.metadata["metadata_error"].startswith("ValueError")
    assert (refused_part.content, refused_part.is_error) == ("quota exceeded", True)
    assert list(refused_part.metadata) == ["metadata_error"]
    journaled = [e.data["result"] for e in entries_of_kind(await session.entries(), "tool_result")]
    assert [item["content"] for item in journaled] == ["it broke", "x", "sent", "quota exceeded"]


# -- D1: only a read-only call is harmless when it is cut short ------------------------------


async def test_a_cancelled_idempotent_call_has_an_unknown_outcome() -> None:
    @tool(idempotent=True)
    async def sync_files() -> str:
        """Copy files (safe to repeat, but it changes things)."""
        await asyncio.sleep(5)
        return "synced"

    stream = Agent(ScriptedModel([call("sync_files")]), tools=[sync_files]).stream("go")
    async for event in stream:
        if event.type == "tool_call_started":
            stream.cancel()
    assert stream.result is not None
    (part,) = results_of(stream.result)
    assert part.content == prompts.TOOL_INTERRUPTED.format(name="sync_files")


# -- D2: RunContext.call_tool, one test per point of its specification -----------------------

Script = Callable[[RunContext[Any], dict[str, Any]], Awaitable[None]]


async def delegating(
    script: Script,
    *,
    permissions: Permissions | None = None,
    limits: Limits | None = None,
    hooks: list[Hooks] | None = None,
    then: list[Any] | None = None,
    extra: list[Any] | None = None,
) -> tuple[Any, list[Any], list[Any], dict[str, Any], ScriptedModel]:
    """A lead whose tool ``delegate`` runs ``script`` (which makes calls through call_tool)."""
    seen: dict[str, Any] = {}

    @tool
    async def delegate(ctx: RunContext[Any], task: str) -> str:
        """Hand the task to another agent."""
        seen["caller"] = ctx
        await script(ctx, seen)
        return "delegated"

    session = Session(InMemoryStore(), id="lead")
    seen["session"] = session
    model = ScriptedModel([call("delegate", task="t"), *(then or []), "Done."])
    agent = Agent(
        model,
        tools=[delegate, *(extra or [])],
        permissions=permissions,
        limits=limits,
        hooks=hooks or [],
    )
    stream = agent.stream("go", session=session)
    events = [event async for event in stream]
    return stream.result, events, await session.entries(), seen, model


def served(entries: list[Any]) -> list[Any]:
    return [entry for entry in entries if entry.data.get("source") == "test"]


def refused(name: str, reason: str) -> str:
    return prompts.TOOL_SERVER_REFUSED.format(name=name, reason=reason)


@tool(read_only=True)
def lookup(email: str) -> str:
    """Look a customer up."""
    return f"found {email}"


@tool
def broken() -> str:
    """Always fail."""
    raise RuntimeError("down")


@tool(requires_approval=True)
def ticket(subject: str) -> str:
    """Open a ticket."""
    return f"opened {subject}"


async def test_call_tool_raises_only_outside_a_running_tool_call() -> None:
    with pytest.raises(RuntimeError, match="inside a tool call"):
        await RunContext().call_tool(lookup, {"email": "a"}, source="test")

    class Probe(Hooks):
        async def before_model(self, ctx: RunContext[Any], request: Any) -> Any:
            try:
                await ctx.call_tool(lookup, {"email": "a"}, source="test")
            except RuntimeError as error:
                seen.append(str(error))
            return None

    seen: list[str] = []
    await Agent(ScriptedModel(["done"]), hooks=[Probe()]).run("go")
    assert seen and "inside a tool call" in seen[0]

    async def script(ctx: RunContext[Any], seen: dict[str, Any]) -> None:
        seen["broken"] = await ctx.call_tool(broken, {}, source="test")  # never raises

    result, *_, seen_calls, _ = await delegating(script)
    assert result.status == "completed"
    assert seen_calls["broken"].is_error
    assert seen_calls["broken"].content == "'broken' failed: RuntimeError: down"


async def test_call_tool_journals_an_ordinary_call_with_its_source_and_tags() -> None:
    async def script(ctx: RunContext[Any], seen: dict[str, Any]) -> None:
        seen["found"] = await ctx.call_tool(
            lookup,
            {"email": "a@x"},
            source="test",
            tags={"server": "s1"},
            metadata={"served": {"by": "s1"}},
        )

    result, events, entries, seen, model = await delegating(script)
    found = seen["found"]
    assert found.content == "found a@x" and found.tool_call_id.startswith("test-")
    assert found.metadata == {"served": {"by": "s1"}}
    caller = seen["caller"].tool_call_id
    mine = served(entries)
    assert [entry.kind for entry in mine] == ["tool_started", "tool_result"]
    assert all(
        entry.data["caller_tool_call_id"] == caller and entry.data["server"] == "s1"
        for entry in mine
    )
    assert mine[1].data["counted"] is True
    assert mine[1].data["result"]["metadata"] == {"served": {"by": "s1"}}
    # Not part of any assistant message: the model's next request has only its own result.
    assert [part.name for part in model.requests[1].messages[-1].tool_results] == ["delegate"]
    assert [part.name for part in results_of(result)] == ["delegate"]
    started = [event.call.name for event in events if isinstance(event, ToolCallStarted)]
    assert started == ["delegate", "lookup"]

    async def bad(ctx: RunContext[Any], seen: dict[str, Any]) -> None:
        for options in ({"source": "no spaces"}, {"source": "t", "tags": {"result": 1}}):
            with pytest.raises(ValueError):
                await ctx.call_tool(lookup, {"email": "a"}, **options)

    await delegating(bad)


async def test_call_tool_refuses_and_journals_nothing_once_the_run_or_its_caller_ended() -> None:
    kept: list[RunContext[Any]] = []
    answers: list[ToolResultPart] = []
    cancelled = asyncio.Event()

    @tool
    async def keep(ctx: RunContext[Any]) -> str:
        """Keep the context for later."""
        kept.append(ctx)
        return "kept"

    @tool
    async def late(ctx: RunContext[Any]) -> str:
        """Use a context whose call has ended."""
        answers.append(await kept[0].call_tool(lookup, {"email": "a"}, source="test"))
        return "ok"

    session = Session()
    model = ScriptedModel([call("keep"), call("late"), "done"])
    await Agent(model, tools=[keep, late]).run("go", session=session)
    assert answers[0].content == refused("lookup", prompts.TOOL_SERVER_RUN_ENDED)
    assert not served(await session.entries())

    @tool
    async def waits(ctx: RunContext[Any]) -> str:
        """Wait; when the run is cancelled, try one more call."""
        try:
            cancelled.set()
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            answers.append(await ctx.call_tool(lookup, {"email": "a"}, source="test"))
            raise
        return "never"

    session = Session()
    stream = Agent(ScriptedModel([call("waits")]), tools=[waits]).stream("go", session=session)
    async for event in stream:
        if event.type == "tool_call_started":
            await cancelled.wait()
            stream.cancel()
    assert answers[1].content == refused("lookup", prompts.TOOL_SERVER_RUN_ENDED)
    assert not served(await session.entries())


async def test_call_tool_takes_the_budget_slot_before_deciding_and_gives_it_back() -> None:
    asked: list[str] = []

    async def on_ask(ctx: Any, tool: Any, call: Any, reason: Any) -> bool:
        asked.append(call.arguments["subject"])
        await asyncio.sleep(0.2)  # the other calls arrive while this one is decided
        return True

    async def script(ctx: RunContext[Any], seen: dict[str, Any]) -> None:
        seen["invalid"] = await ctx.call_tool(ticket, {"title": "x"}, source="test")
        seen["denied"] = await ctx.call_tool(lookup, {"email": "a"}, source="test")
        seen["together"] = await asyncio.gather(
            *(ctx.call_tool(ticket, {"subject": f"t{i}"}, source="test") for i in range(3))
        )

    rules = Permissions(deny=["lookup"], on_ask=on_ask)
    result, _, entries, seen, _ = await delegating(
        script, permissions=rules, limits=Limits(max_tool_calls=2)
    )
    assert result.status == "completed"
    assert seen["invalid"].content.startswith("Invalid arguments for 'ticket'")
    assert seen["denied"].content == refused("lookup", prompts.PERMISSION_REASON_DENY_RULE)
    assert len(asked) == 1  # refused calls gave their slot back; nobody approved a call over it
    limit = prompts.TOOL_CALL_LIMIT.format(name="ticket", limit=2)
    assert sorted(part.content for part in seen["together"] if part.is_error) == [limit, limit]
    counted = [e for e in served(entries) if e.kind == "tool_result" and e.data.get("counted")]
    assert len(counted) == 1


async def test_a_call_cancelled_while_it_is_decided_gives_its_slot_back() -> None:
    deciding = asyncio.Event()

    async def on_ask(ctx: Any, tool: Any, call: Any, reason: Any) -> bool:
        if call.arguments["subject"] == "first":
            deciding.set()
            await asyncio.sleep(10)  # cancelled before it answers
        return True

    async def script(ctx: RunContext[Any], seen: dict[str, Any]) -> None:
        job = asyncio.create_task(ctx.call_tool(ticket, {"subject": "first"}, source="test"))
        await deciding.wait()
        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job
        seen["second"] = await ctx.call_tool(ticket, {"subject": "second"}, source="test")

    _, _, entries, seen, _ = await delegating(
        script, permissions=Permissions(on_ask=on_ask), limits=Limits(max_tool_calls=2)
    )
    assert seen["second"].content == "opened second"  # the cancelled call left its slot
    assert [entry.kind for entry in served(entries)] == [
        "approval_decided",
        "tool_started",
        "tool_result",
    ]  # nothing was journaled for the cancelled one


async def test_call_tool_cannot_pause_and_bounds_on_ask() -> None:
    async def script(ctx: RunContext[Any], seen: dict[str, Any]) -> None:
        seen["asked"] = await ctx.call_tool(ticket, {"subject": "a"}, source="test")

    result, _, entries, seen, _ = await delegating(script)
    assert result.status == "completed"  # the run did not pause for it
    assert seen["asked"].content == refused("ticket", prompts.TOOL_SERVER_NO_APPROVER)
    assert not [entry for entry in served(entries) if entry.kind == "tool_started"]

    async def slow(ctx: Any, tool: Any, call: Any, reason: Any) -> bool:
        await asyncio.sleep(5)
        return True

    async def bounded(ctx: RunContext[Any], seen: dict[str, Any]) -> None:
        seen["slow"] = await ctx.call_tool(
            ticket, {"subject": "a"}, source="test", approval_timeout_s=0.1
        )

    _, _, _, seen, _ = await delegating(bounded, permissions=Permissions(on_ask=slow))
    reason = prompts.TOOL_SERVER_APPROVAL_TIMEOUT.format(seconds=0.1)
    assert seen["slow"].content == refused("ticket", reason)

    def edit(ctx: Any, tool: Any, call: Any, reason: Any) -> Approval:
        return Approval(approved=True, arguments={"subject": "edited"})

    async def edited(ctx: RunContext[Any], seen: dict[str, Any]) -> None:
        seen["edited"] = await ctx.call_tool(ticket, {"subject": "a"}, source="test")

    _, _, entries, seen, _ = await delegating(edited, permissions=Permissions(on_ask=edit))
    assert seen["edited"].content.endswith("opened edited")
    kinds = [entry.kind for entry in served(entries)]
    assert kinds == ["approval_decided", "tool_started", "tool_result"]  # recorded first


async def test_call_tool_checks_again_right_before_the_tool_runs() -> None:
    ran: list[str] = []
    entered = asyncio.Event()
    go_on = asyncio.Event()
    seen: dict[str, Any] = {}

    @tool(read_only=True)
    def probe(text: str) -> str:
        """Record that it ran."""
        ran.append(text)
        return text

    class SlowBefore(Hooks):
        async def before_tool(self, ctx: RunContext[Any], call: Any) -> Any:
            if call.name == "probe":
                entered.set()
                await go_on.wait()
            return None

    @tool
    async def delegate(ctx: RunContext[Any]) -> str:
        """Start a call and end while it waits in before_tool."""
        seen["job"] = asyncio.create_task(ctx.call_tool(probe, {"text": "x"}, source="test"))
        await entered.wait()
        return "started"

    @tool
    async def release() -> str:
        """Let the call go on, after the delegating call ended."""
        go_on.set()
        seen["answer"] = await seen["job"]
        return "released"

    session = Session()
    model = ScriptedModel([call("delegate"), call("release"), "done"])
    result = await Agent(model, tools=[delegate, release], hooks=[SlowBefore()]).run(
        "go", session=session
    )
    assert result.status == "completed" and ran == []
    assert seen["answer"].content == refused("probe", prompts.TOOL_SERVER_RUN_ENDED)
    assert [entry.kind for entry in served(await session.entries())] == [
        "tool_started",
        "tool_result",
    ]


async def test_call_tool_shares_the_runs_abandoned_threads() -> None:
    events: list[str] = []
    release = threading.Event()

    @tool(timeout=0.2)
    def hang(text: str) -> str:
        """Block until the test lets go."""
        events.append(f"start {text}")
        release.wait(10)
        events.append(f"end {text}")
        return text

    @tool
    def write(text: str) -> str:
        """Write at once."""
        events.append(f"start {text}")
        events.append(f"end {text}")
        return text

    async def script(ctx: RunContext[Any], seen: dict[str, Any]) -> None:
        seen["hang"] = await ctx.call_tool(hang, {"text": "a"}, source="test")
        asyncio.get_running_loop().call_later(0.4, release.set)

    try:
        _, _, _, seen, _ = await delegating(
            script, then=[call("write", text="b")], extra=[write, hang]
        )
    finally:
        release.set()
    assert seen["hang"].content == prompts.TOOL_TIMEOUT_ABANDONED.format(name="hang", seconds=0.2)
    assert events == ["start a", "end a", "start b", "end b"]  # write waited for a's thread


async def test_call_tool_writes_nothing_once_the_journal_is_closed() -> None:
    started = asyncio.Event()
    finish = asyncio.Event()
    jobs: list[asyncio.Task[ToolResultPart]] = []

    @tool
    async def slow() -> str:
        """Finish when the test says so."""
        started.set()
        await finish.wait()
        return "late"

    async def script(ctx: RunContext[Any], seen: dict[str, Any]) -> None:
        jobs.append(asyncio.create_task(ctx.call_tool(slow, {}, source="test")))
        await started.wait()  # it runs; the delegating call ends and so does the run

    result, _, entries, seen, _ = await delegating(script)
    assert result.status == "completed"
    finish.set()
    answer = await jobs[0]
    assert answer.content == "late"
    after = await seen["session"].entries()
    assert after == entries and entries[-1].kind == "run_finished"  # nothing written after


async def test_call_tool_writes_nothing_while_the_run_writes_its_last_entry() -> None:
    finishing = asyncio.Event()
    release = asyncio.Event()

    class SlowClose(InMemoryStore):
        """Holds the write of run_finished, as a slow database would."""

        async def _hold(self, entries: list[Any]) -> None:
            if any(entry.kind == "run_finished" for entry in entries):
                finishing.set()
                await release.wait()

        async def append(self, session_id: str, entries: list[Any]) -> None:
            await self._hold(entries)
            await super().append(session_id, entries)

        async def append_leased(self, session_id: str, entries: list[Any], owner: str) -> None:
            await self._hold(entries)
            await super().append_leased(session_id, entries, owner)

    started = asyncio.Event()

    @tool
    async def slow() -> str:
        """Finish while the run writes its end."""
        started.set()
        await finishing.wait()
        return "late"

    jobs: list[asyncio.Task[ToolResultPart]] = []

    @tool
    async def delegate(ctx: RunContext[Any]) -> str:
        """Start a call and end before it does."""
        jobs.append(asyncio.create_task(ctx.call_tool(slow, {}, source="test")))
        await started.wait()
        return "started"

    session = Session(SlowClose(), id="closing")
    run = asyncio.create_task(
        Agent(ScriptedModel([call("delegate"), "done"]), tools=[delegate]).run(
            "go", session=session
        )
    )
    await asyncio.wait_for(started.wait(), 10)
    answer = await asyncio.wait_for(jobs[0], 10)  # it finishes while run_finished waits
    release.set()
    await run
    assert answer.content == "late"
    entries = await session.entries()
    assert entries[-1].kind == "run_finished"
    assert [entry.kind for entry in served(entries)] == ["tool_started"]  # no late result


async def test_call_tool_runs_one_level_deeper() -> None:
    @tool(read_only=True)
    def depth(ctx: RunContext[Any]) -> str:
        """Report the depth."""
        return str(ctx.depth)

    async def script(ctx: RunContext[Any], seen: dict[str, Any]) -> None:
        seen["depth"] = await ctx.call_tool(depth, {}, source="test")

    _, _, _, seen, _ = await delegating(script)
    assert seen["caller"].depth == 0 and seen["depth"].content == "1"


async def test_a_hook_that_raises_in_call_tool_fails_the_run() -> None:
    ran: list[str] = []

    @tool
    def probe() -> str:
        """Record that it ran."""
        ran.append("probe")
        return "ok"

    class Broken(Hooks):
        async def before_tool(self, ctx: RunContext[Any], call: Any) -> Any:
            if call.name == "probe":
                raise RuntimeError("audit hook down")
            return None

    async def script(ctx: RunContext[Any], seen: dict[str, Any]) -> None:
        try:
            await ctx.call_tool(probe, {}, source="test")
        except RuntimeError as error:
            seen["raised"] = str(error)  # swallowing it does not save the run

    with pytest.raises(RuntimeError, match="audit hook down"):
        await delegating(script, hooks=[Broken()])
    assert ran == []


async def test_call_tool_keeps_calls_apart_as_parallel_tools_says() -> None:
    active = {"write": 0, "read": 0}
    peak = {"write": 0, "read": 0}

    async def busy(kind: str) -> str:
        active[kind] += 1
        peak[kind] = max(peak[kind], active[kind])
        await asyncio.sleep(0.05)
        active[kind] -= 1
        return kind

    @tool
    async def write_note(text: str) -> str:
        """Write."""
        return await busy("write")

    @tool(read_only=True)
    async def read_note(text: str) -> str:
        """Read."""
        return await busy("read")

    async def script(ctx: RunContext[Any], seen: dict[str, Any]) -> None:
        await asyncio.gather(
            *(ctx.call_tool(write_note, {"text": str(i)}, source="test") for i in range(3)),
            *(ctx.call_tool(read_note, {"text": str(i)}, source="test") for i in range(3)),
        )

    await delegating(script)
    assert peak == {"write": 1, "read": 3}


async def test_a_cancelled_call_tool_records_its_outcome() -> None:
    started = asyncio.Event()

    @tool
    async def pay(amount: int) -> str:
        """Pay."""
        started.set()
        await asyncio.sleep(10)
        return "paid"

    @tool(read_only=True)
    async def look() -> str:
        """Look."""
        started.set()
        await asyncio.sleep(10)
        return "seen"

    async def script(ctx: RunContext[Any], seen: dict[str, Any]) -> None:
        for served_tool, arguments in ((pay, {"amount": 1}), (look, {})):
            started.clear()
            job = asyncio.create_task(ctx.call_tool(served_tool, arguments, source="test"))
            await started.wait()
            job.cancel()
            with pytest.raises(asyncio.CancelledError):
                await job

    _, _, entries, _, _ = await delegating(script)
    stored = [e.data["result"]["content"] for e in served(entries) if e.kind == "tool_result"]
    assert stored == [
        prompts.TOOL_INTERRUPTED.format(name="pay"),  # it may have paid: outcome unknown
        prompts.TOOL_CANCELLED.format(name="look"),  # read-only: harmless
    ]


# -- review fixes ----------------------------------------------------------------------------


def saving(ran: list[str]) -> Any:
    @tool
    def save(path: str) -> str:
        """Save a file (it changes something)."""
        ran.append(path)
        return f"saved {path}"

    return save


def serving_while_cancelled(
    save: Any, answers: list[ToolResultPart], started: asyncio.Event
) -> Any:
    @tool
    async def delegate(ctx: RunContext[Any]) -> str:
        """Wait; while the call is cancelled, serve one more call (as an MCP request can)."""
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            job = asyncio.create_task(ctx.call_tool(save, {"path": "x"}, source="test"))
            answers.append(await asyncio.shield(job))
            raise
        return "never"

    return delegate


async def test_call_tool_refuses_while_the_run_is_cancelled_from_outside() -> None:
    ran: list[str] = []
    answers: list[ToolResultPart] = []
    save = saving(ran)
    delegate = serving_while_cancelled(save, answers, asyncio.Event())
    session = Session()
    agent = Agent(ScriptedModel([call("delegate"), "done"]), tools=[delegate, save])
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.3):  # the documented way to bound a run
            await agent.run("go", session=session)
    assert ran == []
    assert answers[0].content == refused("save", prompts.TOOL_SERVER_RUN_ENDED)
    assert not served(await session.entries())


async def test_call_tool_refuses_once_the_stream_is_closed_early() -> None:
    ran: list[str] = []
    answers: list[ToolResultPart] = []
    started = asyncio.Event()
    save = saving(ran)
    delegate = serving_while_cancelled(save, answers, started)
    session = Session()
    agent = Agent(ScriptedModel([call("delegate"), "done"]), tools=[delegate, save])
    stream = agent.stream("go", session=session)
    async for event in stream:
        if event.type == "tool_call_started":
            await started.wait()
            break
    await stream.aclose()
    assert ran == []
    assert answers[0].content == refused("save", prompts.TOOL_SERVER_RUN_ENDED)
    assert not served(await session.entries())


async def test_a_child_run_takes_no_calls_from_outside_once_its_parent_is_cancelled() -> None:
    ran: list[str] = []
    answers: list[ToolResultPart] = []
    started = asyncio.Event()
    save = saving(ran)
    delegate = serving_while_cancelled(save, answers, started)
    child = Agent(ScriptedModel([call("delegate"), "done"]), name="child", tools=[delegate, save])
    parent = Agent(
        ScriptedModel([call("child", task="go"), "done"]), tools=[child.as_tool(name="child")]
    )
    stream = parent.stream("go")

    async def cancel_once_started() -> None:
        await started.wait()
        stream.cancel()

    watcher = asyncio.create_task(cancel_once_started())
    async for _ in stream:
        pass
    await watcher
    assert stream.result is not None and stream.result.status == "cancelled"
    assert ran == []
    assert answers[0].content == refused("save", prompts.TOOL_SERVER_RUN_ENDED)


async def test_a_context_kept_after_the_run_returned_writes_nothing() -> None:
    ran: list[str] = []
    kept: list[RunContext[Any]] = []
    save = saving(ran)

    @tool
    async def delegate(ctx: RunContext[Any]) -> str:
        """Keep the context, as a tool server that outlives its call would."""
        kept.append(ctx)
        await asyncio.sleep(10)
        return "never"

    session = Session()
    agent = Agent(ScriptedModel([call("delegate"), "done"]), tools=[delegate, save])
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.3):
            await agent.run("go", session=session)
    before = await session.entries()
    answer = await kept[0].call_tool(save, {"path": "late"}, source="test")
    assert answer.content == refused("save", prompts.TOOL_SERVER_RUN_ENDED) and ran == []
    with pytest.raises(RuntimeError, match="works only while its tool call runs"):
        await kept[0].checkpoint("k", 1)
    kept[0].state["x"] = 1
    await kept[0].save_state()
    assert await session.entries() == before  # the run released its session: nothing more


async def test_a_context_kept_after_the_run_failed_writes_nothing() -> None:
    ran: list[str] = []
    kept: list[RunContext[Any]] = []
    save = saving(ran)

    @tool
    async def keep(ctx: RunContext[Any]) -> str:
        """Keep the context."""
        kept.append(ctx)
        return "kept"

    class Broken(Hooks):
        async def after_tool(self, ctx: RunContext[Any], call: Any, result: Any) -> Any:
            raise RuntimeError("audit hook down")

    session = Session()
    agent = Agent(ScriptedModel([call("keep"), "done"]), tools=[keep, save], hooks=[Broken()])
    with pytest.raises(RuntimeError, match="audit hook down"):
        await agent.run("go", session=session)
    before = await session.entries()
    assert before[-1].kind == "run_failed"
    answer = await kept[0].call_tool(save, {"path": "late"}, source="test")
    assert answer.content == refused("save", prompts.TOOL_SERVER_RUN_ENDED) and ran == []
    assert await session.entries() == before


async def test_call_tool_refuses_while_its_calling_call_times_out() -> None:
    ran: list[str] = []
    answers: list[ToolResultPart] = []
    save = saving(ran)

    @tool(timeout=0.2)
    async def delegate(ctx: RunContext[Any]) -> str:
        """Time out; while the timeout unwinds the call, serve one more call."""
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            job = asyncio.create_task(ctx.call_tool(save, {"path": "x"}, source="test"))
            answers.append(await asyncio.shield(job))
            raise
        return "never"

    session = Session()
    agent = Agent(ScriptedModel([call("delegate"), "done"]), tools=[delegate, save])
    result = await agent.run("go", session=session)
    assert result.status == "completed"  # the run goes on: only the calling call ended
    (part,) = results_of(result)
    assert part.content == prompts.TOOL_TIMEOUT.format(name="delegate", seconds=0.2)
    assert ran == []
    assert answers[0].content == refused("save", prompts.TOOL_SERVER_RUN_ENDED)
    assert not served(await session.entries())


async def test_a_run_started_by_cleanup_code_still_serves_calls() -> None:
    """A task that is being cancelled may still run an agent (cleanup code): only a cancel
    that comes after the run took the task counts as cancelling the run."""
    ready = asyncio.Event()

    async def script(ctx: RunContext[Any], seen: dict[str, Any]) -> None:
        seen["found"] = await ctx.call_tool(lookup, {"email": "a@x"}, source="test")

    async def cleanup() -> Any:
        try:
            ready.set()
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            return await delegating(script)
        return None

    task = asyncio.create_task(cleanup())
    await ready.wait()
    task.cancel()
    result, _, _, seen, _ = await task
    assert result.status == "completed"
    assert seen["found"].content == "found a@x"


async def test_call_tool_refuses_the_usage_tag() -> None:
    """derive_state counts ``usage`` on a tool_result as model usage after a resume."""

    async def script(ctx: RunContext[Any], seen: dict[str, Any]) -> None:
        for key in ("usage", "added_usage", "added_usage_by_operation"):
            with pytest.raises(ValueError, match="cannot set"):
                await ctx.call_tool(
                    lookup, {"email": "a"}, source="test", tags={key: {"input_tokens": 10**6}}
                )

    result, _, entries, _, _ = await delegating(script)
    assert result.status == "completed"
    assert not served(entries)
    assert derive_state(entries).run_usage.input_tokens == 0


async def test_a_cancelled_sync_call_tool_keeps_the_next_write_waiting() -> None:
    """A served call can be cancelled on its own (an MCP client cancel): its sync thread
    goes on, and the next call that runs alone waits for it."""
    log: list[str] = []
    active: set[str] = set()
    lock = threading.Lock()
    entered = threading.Event()

    def work(name: str, seconds: float) -> str:
        with lock:
            log.append(f"{name} start {sorted(active)}")
            active.add(name)
        entered.set()
        threading.Event().wait(seconds)
        with lock:
            active.discard(name)
            log.append(f"{name} end")
        return name

    @tool
    def write_a() -> str:
        """Write A (sync, not read-only)."""
        return work("A", 0.6)

    @tool
    def write_b() -> str:
        """Write B (sync, not read-only)."""
        return work("B", 0.05)

    async def script(ctx: RunContext[Any], seen: dict[str, Any]) -> None:
        first = asyncio.create_task(ctx.call_tool(write_a, {}, source="test"))
        await asyncio.to_thread(entered.wait, 5)
        first.cancel()  # notifications/cancelled for that request
        with pytest.raises(asyncio.CancelledError):
            await first
        seen["second"] = await ctx.call_tool(write_b, {}, source="test")

    _, _, _, seen, _ = await delegating(script)
    assert seen["second"].content == "B"
    assert log == ["A start []", "A end", "B start []", "B end"]  # B waited for A's thread


async def test_checkpoints_are_a_deep_copy() -> None:
    seen: dict[str, Any] = {}

    @tool(idempotent=True)
    async def job(ctx: RunContext[Any]) -> str:
        """Start a job."""
        await ctx.checkpoint("job", {"id": "original", "parts": [1]})
        ctx.checkpoints["job"]["id"] = "changed"  # changing the copy
        ctx.checkpoints["job"]["parts"].append(2)
        seen["again"] = ctx.checkpoints["job"]
        return "ok"

    session = Session()
    await Agent(ScriptedModel([call("job"), "done"]), tools=[job]).run("go", session=session)
    (entry,) = entries_of_kind(await session.entries(), "tool_checkpoint")
    assert entry.data["value"] == {"id": "original", "parts": [1]}
    assert seen["again"] == {"id": "original", "parts": [1]}


async def test_journaled_usage_tells_a_repeated_call_what_the_journal_holds() -> None:
    seen: list[Usage] = []
    crash = [True]

    @tool(idempotent=True)
    async def work(ctx: RunContext[Any]) -> str:
        """Report usage, journal a step, report more, then crash (once)."""
        seen.append(ctx.journaled_usage)
        if not crash[0]:
            return "done"
        ctx.add_usage(Usage(requests=1, input_tokens=100))
        await ctx.checkpoint("step", 1)  # the usage rides on this entry
        seen.append(ctx.journaled_usage)  # this attempt's, journaled now
        ctx.add_usage(Usage(requests=1, input_tokens=7))  # no entry carries it: lost
        raise KeyboardInterrupt

    session = Session()
    agent = Agent(ScriptedModel([call("work"), "ok"]), tools=[work])
    with pytest.raises(KeyboardInterrupt):
        await agent.run("go", session=session)
    state = await session.state()
    assert [usage.input_tokens for usage in state.operation_usage.values()] == [100]
    crash[0] = False
    resumed = await agent.resume(session)
    assert resumed.status == "completed"
    assert [usage.input_tokens for usage in seen] == [0, 100, 100]
    assert RunContext().journaled_usage == Usage()


async def test_usage_a_served_call_adds_is_journaled_as_its_own() -> None:
    inside: dict[str, Any] = {}

    @tool
    def costly(ctx: RunContext[Any]) -> str:
        """Use a model somewhere (from a worker thread)."""
        ctx.add_usage(Usage(requests=1, input_tokens=9))
        inside["operation"] = ctx.operation_id
        return "ok"

    async def script(ctx: RunContext[Any], seen: dict[str, Any]) -> None:
        seen["costly"] = await ctx.call_tool(costly, {}, source="test")

    _, _, entries, seen, _ = await delegating(script)
    caller = seen["caller"]
    assert inside["operation"] == f"{caller.operation_id}:{seen['costly'].tool_call_id}"
    (result,) = [entry for entry in served(entries) if entry.kind == "tool_result"]
    [(operation, usage)] = result.data["added_usage_by_operation"].items()
    assert operation == inside["operation"]
    assert (usage["requests"], usage["input_tokens"]) == (1, 9)
