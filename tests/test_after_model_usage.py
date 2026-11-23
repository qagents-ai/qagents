from __future__ import annotations

import asyncio

import pytest

from qagents import Agent, Hooks, Limits, Message, Session, SQLiteStore, Usage
from qagents.errors import ModelServerError
from qagents.models.base import ModelResponse
from qagents.models.testing import ScriptedModel, call
from qagents.testing.crash._model import ModelLedger

MEASURED = Usage(requests=1, input_tokens=13, output_tokens=7, cost_usd=0.42)


class RejectResponse(Hooks):
    def __init__(self):
        self.ending = []

    async def after_model(self, ctx, response):
        raise ValueError("Response rejected by a hook")

    async def on_run_end(self, ctx, result, error):
        self.ending.append((ctx.usage, result, error))


async def test_rejected_response_remains_billable_in_sqlite_after_reopen(tmp_path):
    path = tmp_path / "model-usage.sqlite"
    hook = RejectResponse()
    model = ScriptedModel(["Rejected answer"], usage=MEASURED)
    agent = Agent(model, hooks=[hook], limits=Limits(max_cost_usd=0.40))

    async with SQLiteStore(path) as store:
        with pytest.raises(ValueError, match="rejected by a hook"):
            await agent.run("Hello", session=Session(store, id="rejected"))

    [(usage, result, error)] = hook.ending
    assert usage == MEASURED
    assert result is None and isinstance(error, ValueError)

    async with SQLiteStore(path) as store:
        session = Session(store, id="rejected")
        state = await session.state()
        assert state.run_usage == MEASURED
        assert state.run_turns == 1
        assert all(message.role != "assistant" for message in state.messages)
        [receipt] = [entry.data for entry in await session.entries() if entry.kind == "model_usage"]
        assert receipt == {
            "usage": MEASURED.model_dump(mode="json"),
            "turn": 1,
            "model_turn": 1,
            "model": "scripted",
            "provider": "test",
        }
        resumed = await agent.resume(session)

    assert resumed.status == "limit_reached"
    assert resumed.stop_reason == "max_cost_usd"
    assert resumed.usage == MEASURED
    assert len(model.requests) == 1


async def test_recovery_replays_the_same_logical_turn_and_counts_both_responses():
    first = ModelResponse(message=Message.assistant("First logical answer"), usage=MEASURED)
    ledger = ModelLedger([first, "The later logical answer"])
    session = Session()
    failed = Agent(ledger.for_leg(0), hooks=[RejectResponse()])

    with pytest.raises(ValueError, match="rejected by a hook"):
        await failed.run("Hello", session=session)

    recovered = Agent(ledger.for_leg(1, recovering=True), limits=Limits(max_turns=2))
    result = await recovered.resume(session)

    assert result.output == "First logical answer"
    assert result.turns == 2
    assert result.usage == MEASURED.add(MEASURED)
    assert [call.turn for call in ledger.calls] == [0, 0]
    receipts = [entry.data for entry in await session.entries() if entry.kind == "model_usage"]
    assert [(receipt["turn"], receipt["model_turn"]) for receipt in receipts] == [(1, 1), (2, 1)]
    state = await session.state()
    assert state.run_turns == 2 and state.run_model_turns == 1


async def test_cancelling_after_model_preserves_its_already_settled_usage():
    entered = asyncio.Event()

    class HoldResponse(RejectResponse):
        async def after_model(self, ctx, response):
            entered.set()
            await asyncio.Future()

    session = Session()
    hook = HoldResponse()
    agent = Agent(ScriptedModel(["Complete provider response"], usage=MEASURED), hooks=[hook])
    run = asyncio.create_task(agent.run("Hello", session=session))
    try:
        await asyncio.wait_for(entered.wait(), timeout=10)
        assert (await session.state()).run_usage == MEASURED
    finally:
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run

    [(usage, result, error)] = hook.ending
    assert usage == MEASURED
    assert result is None and isinstance(error, asyncio.CancelledError)
    assert (await session.state()).run_usage == MEASURED


async def test_rejected_child_response_counts_against_parent_budget():
    child = Agent(
        ScriptedModel(["Already billed"], usage=MEASURED), name="child", hooks=[RejectResponse()]
    )
    parent_model = ScriptedModel([call("child", task="Measure"), "Unreachable"])
    session = Session()
    parent = Agent(parent_model, tools=[child], limits=Limits(max_cost_usd=0.40))

    result = await parent.run("Delegate", session=session)

    assert result.status == "limit_reached" and result.stop_reason == "max_cost_usd"
    assert len(parent_model.requests) == 1
    assert result.usage == Usage(requests=1).add(MEASURED)
    assert (await session.state()).run_usage == result.usage


async def test_replacing_response_content_does_not_change_measured_provider_usage():
    class ReplaceResponse(Hooks):
        async def after_model(self, ctx, response):
            return response.model_copy(
                update={"message": Message.assistant("Sanitised"), "usage": Usage()}
            )

    session = Session()
    model = ScriptedModel(["Original"], usage=MEASURED)
    result = await Agent(model, hooks=[ReplaceResponse()]).run("Hello", session=session)

    assert result.output == "Sanitised"
    assert result.usage == MEASURED
    assert (await session.state()).run_usage == MEASURED


async def test_ambiguous_model_failure_without_a_response_has_no_measured_usage():
    session = Session()
    hook = RejectResponse()
    ambiguous = ModelServerError("Connection lost after dispatch", ambiguous=True)
    agent = Agent(ScriptedModel([ambiguous]), hooks=[hook])

    with pytest.raises(ModelServerError, match="Connection lost"):
        await agent.run("Hello", session=session)

    assert (await session.state()).run_usage == Usage()
    assert hook.ending == [(Usage(), None, ambiguous)]
