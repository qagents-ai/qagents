"""Independent observations for mutations without an existing regression test."""

import asyncio

import pytest

from qagents import Agent, Limits, RunContext, Session, SQLiteStore, tool
from qagents.effects import Attempt, EffectContext, Key, effect
from qagents.effects import _engine as engine
from qagents.effects._types import Absent
from qagents.errors import EffectHeld, SessionError
from qagents.messages import Message, ReasoningPart
from qagents.models.testing import FunctionModel, ScriptedModel, call
from qagents.permissions import Approval, Permissions
from qagents.session import Entry, InMemoryStore, append_fenced, derive_state
from qagents.testing.crash import Faults, crash_test

from ..test_effects import opened
from ..test_engine_dispatch import Receipt, Request
from ..test_lifecycle import make_agent
from . import scenarios


async def test_plain_absence_keeps_unknown_attempt_held():
    now = [1000.0]
    sends = []

    @effect("refund", key=Key("request", "id"), applies_within_s=1)
    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        sends.append(attempt.idempotency_key)
        if len(sends) == 1:
            raise ConnectionError("ack lost")
        return Receipt(id="duplicate")

    @send.lookup
    async def find(ctx, inquiry):
        return Absent()

    db = InMemoryStore(clock=lambda: now[0])
    async with opened(db) as (scope, _, _):
        with pytest.raises(EffectHeld):
            await engine.perform(send, Request(id="R1"), scope=scope)
    now[0] += 30
    async with opened(db) as (scope, _, _):
        with pytest.raises(EffectHeld):
            await engine.perform(send, Request(id="R1"), scope=scope)
    assert len(sends) == 1


async def test_sqlite_acknowledgement_means_an_independent_reader_sees_entries(tmp_path):
    path = tmp_path / "refunds.db"
    entry = Entry(kind="run_started", run_id="R")
    async with SQLiteStore(path) as writer:
        assert await writer.append("session", [entry]) == [1]
        async with SQLiteStore(path) as reader:
            assert [record.id for record in await reader.load("session")] == [entry.id]


async def test_lost_ack_preserves_entry_identity_before_retry():
    class LoseFirstAck(InMemoryStore):
        def __init__(self):
            super().__init__()
            self.batches = []

        async def append(self, session_id, entries):
            self.batches.append([entry.id for entry in entries])
            seqs = await super().append(session_id, entries)
            if len(self.batches) == 1:
                raise OSError("commit happened before the acknowledgement was lost")
            return seqs

    store = LoseFirstAck()
    entry = Entry(kind="run_started", run_id="R")
    assert await append_fenced(store, "session", [entry], None, fence=lambda: True) == [1]
    assert store.batches == [[entry.id], [entry.id]]
    assert [item.id for item in await store.load("session")] == [entry.id]


async def test_effect_calls_always_run_in_separate_groups():
    events = []
    second_entered = asyncio.Event()

    @effect("refund", key=Key("request", "id"))
    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        return Receipt(id=attempt.request.id)

    @tool(effects=[send])
    async def refund(ctx: RunContext, id: str) -> str:
        """Record a refund."""
        events.append(("start", id))
        if id == "R1":
            try:
                await asyncio.wait_for(second_entered.wait(), 0.05)
            except TimeoutError:
                pass
        else:
            second_entered.set()
        receipt = await ctx.perform(send, Request(id=id))
        events.append(("end", id))
        return receipt.id

    model = ScriptedModel(
        [
            Message(role="assistant", parts=[call("refund", id="R1"), call("refund", id="R2")]),
            "Done",
        ]
    )
    result = await Agent(model, tools=[refund], limits=Limits(parallel_tools="all")).run(
        "Refund both", session=Session()
    )
    assert result.status == "completed"
    assert events == [("start", "R1"), ("end", "R1"), ("start", "R2"), ("end", "R2")]


async def test_verified_decision_is_reused_after_restart():
    class StopBeforeStart(InMemoryStore):
        stopped = False

        async def append_leased(self, session_id, entries, owner):
            if not self.stopped and any(item.kind == "tool_started" for item in entries):
                self.stopped = True
                raise SessionError("worker stopped before tool start")
            return await super().append_leased(session_id, entries, owner)

    store = StopBeforeStart()
    destination = []
    session = Session(store, id="refund")
    paused = await make_agent(destination, approval=True).run("Refund", session=session)
    (pending,) = paused.pending_approvals
    with pytest.raises(SessionError, match="worker stopped"):
        await make_agent(destination, approval=True).resume(
            session, approvals={pending.tool_call_id: Approval.for_call(pending)}
        )
    reopened = Session(store, id=session.id)
    result = await make_agent(destination, approval=True).resume(reopened)
    assert result.status == "completed" and len(destination) == 1
    assert sum(entry.kind == "approval_decided" for entry in await reopened.entries()) == 1


async def test_durable_tool_result_precedes_the_followup_model_request():
    session = Session()
    seen = []

    @tool
    async def read_balance() -> str:
        """Read the fixture balance."""
        return "14"

    async def reply(request):
        if request.messages[-1].role == "tool":
            seen.extend(entry.kind for entry in await session.entries())
            return "Done"
        return call("read_balance")

    result = await Agent(FunctionModel(reply), tools=[read_balance]).run("Balance", session=session)
    assert result.status == "completed"
    assert "tool_result" in seen


def test_replay_preserves_opaque_provider_reasoning():
    reasoning = ReasoningPart(provider="fixture", opaque={"signature": "fixture-only"})
    assistant = Message(role="assistant", parts=[reasoning, call("read_balance")])
    state = derive_state(
        [Entry(kind="message", run_id="R", data={"message": assistant.model_dump()})]
    )
    assert state.messages[0].parts[0] == reasoning


async def test_approved_reference_exercises_an_effect(tmp_path):
    report = await crash_test(
        scenarios.approved_refund,
        "Refund",
        directory=tmp_path,
        store="sqlite",
        model=[call("refund", id="R1"), "Done."],
        approve=scenarios.approve_refund,
        transport="fake",
        faults=Faults.parse("kill_after,commit_then_lose_response"),
        points="effects",
    )
    assert report.ok, report.explain()
    assert len(report.points[0].execution.applications) == 1
    assert not report.vacuous


async def test_sync_attempt_records_possible_overrun():
    @effect("refund", key=Key("request", "id"))
    def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        return Receipt(id=attempt.request.id)

    async with opened(InMemoryStore()) as (scope, _, _):
        assert (await engine.perform(send, Request(id="R1"), scope=scope)).id == "R1"
        (intent,) = [
            item
            for item in await scope.journal.store.load(scope.session_id)
            if item.kind == "effect_intent"
        ]
        assert intent.data["may_overrun"] is True


async def test_held_unsigned_call_is_not_asked_again():
    sends = []
    session = Session()
    agent = make_agent(sends, fail=True, approval=True).copy(permissions=Permissions())
    initial = await agent.run("Refund", session=session)
    (pending,) = initial.pending_approvals
    held = await agent.resume(session, approvals={pending.tool_call_id: Approval.for_call(pending)})
    assert held.held_effects and not held.pending_approvals
    resumed = await agent.resume(session)
    assert resumed.held_effects and not resumed.pending_approvals
    assert len(sends) == 1


async def test_signed_approval_delivery_recovers_and_refuses_conflicting_answers(tmp_path):
    report = await crash_test(
        scenarios.approved_refund,
        "Refund",
        directory=tmp_path,
        store="sqlite",
        model=[call("refund", id="R1"), "Done."],
        approve=scenarios.approve_refund,
        transport="fake",
        faults=Faults.parse("approval_redelivered,stale_answer,commit_then_lose_response"),
        points="effects",
    )
    for fault in ("approval_redelivered", "stale_answer"):
        (point,) = [item for item in report.points if fault in item.execution.point.kinds]
        assert point.execution.triggered, report.explain(point.id)
        assert point.verdict == "safe", report.explain(point.id)
        assert point.outcome == "completed_as_reference", report.explain(point.id)
        assert sum(not app.replayed for app in point.execution.applications) == 1
        assert point.execution.final_snapshot is not None
        entries = [
            Entry.model_validate_json(raw) for raw in point.execution.final_snapshot.entries_json
        ]
        assert sum(item.kind == "approval_decided" for item in entries) == 1
