"""Exercise the public Agent boundary with an offline model and a real Effects writer."""

import pytest

from qagents import (
    Agent,
    AgentRuntime,
    InMemoryStore,
    Limits,
    RunContext,
    Session,
    SQLiteStore,
    tool,
)
from qagents.effects import Attempt, EffectContext, Key, Settle, effect
from qagents.errors import ConfigurationError, HeldEffectsError, SessionError
from qagents.messages import Message
from qagents.models.testing import FunctionModel, call
from qagents.permissions import Approval, Permissions
from qagents.session import Entry

from .test_engine_dispatch import Receipt, Request


def make_agent(destination, *, fail=False, approval=False, on_unknown="pause"):
    @effect("refund", key=Key("request", "id"), on_unknown=on_unknown)
    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        destination.append(attempt.idempotency_key)
        if fail:
            raise ConnectionError("ack lost")
        return Receipt(id="receipt")

    @send.on_commit
    def patch(state, request, receipt):
        state["receipt"] = receipt.id

    @tool(effects=[send], requires_approval=approval)
    async def refund(ctx: RunContext, id: str) -> str:
        """Record a local refund in the test destination."""
        ctx.state["provisional"] = True
        await ctx.save_state()
        return (await ctx.perform(send, Request(id=id))).id

    return Agent(
        FunctionModel(
            lambda request: (
                "Complete." if request.messages[-1].role == "tool" else call("refund", id="R1")
            )
        ),
        tools=[refund],
        permissions=Permissions(approval_secret="test-approval-secret"),
    )


async def test_effect_tool_requires_explicit_session():
    destination = []
    with pytest.raises(ConfigurationError, match="explicit"):
        await make_agent(destination).run("Refund")
    assert not destination


async def test_public_approval_intent_receipt_and_state_order(tmp_path):
    destination = []
    async with SQLiteStore(tmp_path / "agent.sqlite") as store:
        session = Session(store, id="test")
        agent = make_agent(destination, approval=True)
        paused = await agent.run("Refund", session=session)
        assert paused.status == "paused" and not destination
        pending = paused.pending_approvals[0]
        result = await agent.resume(
            session,
            approvals={pending.tool_call_id: Approval.for_call(pending, by="reviewer")},
        )
        assert result.status == "completed"
        entries = await session.entries()
        indexed = {e.kind: e for e in entries}
        assert indexed["approval_decided"].seq < indexed["tool_started"].seq
        assert indexed["tool_started"].seq < indexed["effect_intent"].seq
        assert indexed["effect_intent"].seq < indexed["effect_committed"].seq
        assert indexed["effect_committed"].seq < indexed["tool_result"].seq
        assert (
            indexed["effect_intent"].data["authorization"]["approval_id"]
            == (indexed["approval_decided"].data["approval_id"])
        )
        assert (await session.state()).values == {"receipt": "receipt", "provisional": True}
    assert len(destination) == 1


async def test_held_call_survives_reopen_and_settlement(tmp_path):
    destination = []
    path = tmp_path / "agent.sqlite"
    async with SQLiteStore(path) as store:
        session = Session(store, id="test")
        paused = await make_agent(destination, fail=True).run("Refund", session=session)
        assert paused.status == "paused" and paused.stop_reason == "effect_held"
        assert len(paused.held_effects) == 1
        call_id = paused.pending_inputs[0].tool_call_id
        assert not (await session.state()).values
        assert not any(e.kind == "tool_result" for e in await session.entries())

    async with SQLiteStore(path) as store:
        session = Session(store, id="test")
        agent = make_agent(destination)
        with pytest.raises(HeldEffectsError):
            await agent.run("Another refund", session=session)
        with pytest.raises(SessionError, match="do not wait"):
            await agent.resume(session, results={call_id: "pretend success"})
        paused_again = await agent.resume(session)
        assert paused_again.status == "paused" and len(destination) == 1
        await agent.settle(
            session,
            Settle.committed(
                paused.held_effects[0].id,
                receipt=Receipt(id="verified"),
                by="reviewer",
                evidence="checked destination",
            ),
        )
        result = await agent.resume(session)
        assert result.status == "completed" and not result.held_effects
        assert len(destination) == 1
        starts = [e.data for e in await session.entries() if e.kind == "tool_started"]
        assert len({item["operation_id"] for item in starts}) == 1
        assert [item["attempt"] for item in starts] == [1, 2, 3]
        assert (await session.state()).values["receipt"] == "verified"


@pytest.mark.parametrize("policy", ["tell_model", "fail_run"])
async def test_unresolved_policies_do_not_persist_provisional_state(policy):
    session = Session()
    agent = make_agent([], fail=True, on_unknown=policy)
    if policy == "fail_run":
        with pytest.raises(HeldEffectsError):
            await agent.run("Refund", session=session)
        assert (await session.entries())[-1].kind == "run_failed"
    else:
        result = await agent.run("Refund", session=session)
        assert result.status == "completed" and result.held_effects
        entries = await session.entries()
        report = next(e.data["result"] for e in entries if e.kind == "tool_result")
        assert report["is_error"] and "EFFECT_OUTCOME_UNKNOWN" in report["content"]
    assert (await session.state()).values == {}


async def test_continue_surfaces_prior_hold_without_repeating_its_action():
    destination = []
    session = Session()
    first = await make_agent(destination, fail=True).run("Refund", session=session)
    assert first.status == "paused"
    new = Agent(FunctionModel(lambda _: "A separate answer."))
    result = await new.run("Another question", session=session, if_held_effects="continue")
    assert result.status == "completed" and result.held_effects[0].id == first.held_effects[0].id
    assert len(destination) == 1
    results = [e.data["result"] for e in await session.entries() if e.kind == "tool_result"]
    assert len(results) == 1 and "EFFECT_HELD_BY_NEW_RUN" in results[0]["content"]


async def test_runtime_workers_share_effect_journal_and_operator_api(tmp_path):
    destination = []
    path = tmp_path / "runtime.sqlite"
    async with SQLiteStore(path) as store:
        agent = make_agent(destination, fail=True)
        runtime = AgentRuntime(store, [agent], worker_id="first")
        async with runtime:
            handle = await runtime.submit(agent.name, "Refund", session_id="refund")
            paused = await handle.wait()
            assert paused.status == "paused" and paused.stop_reason == "effect_held"
            entries = await store.load("refund")
            assert entries[0].kind == "lease_acquired"
            assert entries[0].data["purpose"] == "run"
            assert (await store.status("refund")).lease_owner is None

    async with SQLiteStore(path) as store:
        agent = make_agent(destination)
        runtime = AgentRuntime(store, [agent], worker_id="second")
        async with runtime:
            before = await store.load("refund")
            with pytest.raises(HeldEffectsError):
                await runtime.start(
                    agent.name, "Another refund", session_id="refund", limits=Limits()
                )
            assert await store.load("refund") == before
            (held,) = await runtime.recheck("refund")
            assert held.id == paused.held_effects[0].id
            await runtime.settle(
                "refund",
                Settle.committed(
                    held, receipt=Receipt(id="verified"), by="reviewer", evidence="provider log"
                ),
            )
            result = await (await runtime.resume("refund")).wait()
            assert result.status == "completed" and not result.held_effects
            assert (await result.session.state()).values["receipt"] == "verified"
    assert len(destination) == 1


async def test_runtime_recovery_obeys_store_clock_and_future_retry():
    now = 100.0
    store = InMemoryStore(clock=lambda: now)
    agent = Agent(FunctionModel(lambda _: "done"), name="worker")
    await store.append(
        "session",
        [
            Entry(kind="run_started", run_id="R", data={"agent": agent.name}),
            Entry(kind="message", run_id="R", data={"message": Message.user("Go").model_dump()}),
            Entry(
                kind="run_failed",
                run_id="R",
                data={"ambiguous": False, "retry_after": now + 5},
            ),
        ],
    )
    async with AgentRuntime(store, [agent]) as runtime:
        assert await runtime.recover() == []
        assert await store.acquire_lease("session", "other", 10)
        now += 5
        assert await runtime.recover() == []
        await store.release_lease("session", "other")
        (handle,) = await runtime.recover()
        assert (await handle.wait()).text == "done"


async def test_runtime_recovery_refuses_unrecognized_capability_before_writes():
    store = InMemoryStore()
    agent = Agent(FunctionModel(lambda _: "unexpected"), name="worker")
    await store.append(
        "session",
        [
            Entry(
                kind="run_started",
                run_id="R",
                data={"agent": agent.name, "requires": ["future-runtime-feature"]},
            ),
            Entry(kind="message", run_id="R", data={"message": Message.user("Go").model_dump()}),
        ],
    )
    original = await store.load("session")
    async with AgentRuntime(store, [agent]) as runtime:
        assert await runtime.recover() == []
    assert await store.load("session") == original
    assert (await store.status("session")).lease_owner is None


async def test_effect_tools_are_refused_outside_fenced_agent_runs():
    from qagents.integrations.mcp_server import tool_server
    from qagents.multi import fan_out, route

    agent = make_agent([])
    with pytest.raises(ConfigurationError, match="separate durable session"):
        fan_out(agent)
    classifier = Agent(FunctionModel(lambda _: pytest.fail("Classifier must not run.")))
    with pytest.raises(ConfigurationError, match="separate durable session"):
        await route(classifier, {"refund": agent}, "Refund")
    with pytest.raises(ConfigurationError, match="fenced agent run"):
        await tool_server(agent.tools).start()
