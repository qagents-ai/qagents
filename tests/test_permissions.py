from __future__ import annotations

import pytest

from qagents import Agent, Approval, Permissions, RunContext, Session, tool
from qagents.messages import ToolCallPart
from qagents.models.testing import ScriptedModel, call


def make_tools(log: list[str]):
    @tool(read_only=True)
    def read_order(order: str) -> str:
        """Read an order."""
        log.append(f"read {order}")
        return f"order {order}: 2 items"

    @tool(requires_approval=True)
    def refund_order(order: str, amount: float) -> str:
        """Refund an order."""
        log.append(f"refund {order} {amount}")
        return f"refunded {amount}"

    @tool
    def send_email(to: str) -> str:
        """Send an email."""
        log.append(f"email {to}")
        return "sent"

    return read_order, refund_order, send_email


async def decide(permissions: Permissions, target, **arguments):
    return await permissions.decide(
        RunContext(), target, ToolCallPart(id="c", name=target.name, arguments=arguments)
    )


async def test_rule_order_most_restrictive_wins():
    read_order, refund_order, send_email = make_tools([])

    default = Permissions()
    assert (await decide(default, read_order)).behavior == "allow"
    assert (await decide(default, refund_order)).behavior == "ask"

    rules = Permissions(allow=["refund_*", "send_email"], ask=["send_*"], deny=["refund_order"])
    assert (await decide(rules, refund_order)).behavior == "deny"  # deny beats allow
    assert (await decide(rules, send_email)).behavior == "ask"  # ask beats allow
    assert (await decide(Permissions(allow=["refund_*"]), refund_order)).behavior == "allow"

    assert (await decide(Permissions(mode="read_only"), read_order)).behavior == "allow"
    assert (await decide(Permissions(mode="read_only"), send_email)).behavior == "deny"
    assert (await decide(Permissions(mode="ask_all"), read_order)).behavior == "ask"
    assert (await decide(Permissions(mode="allow_all"), refund_order)).behavior == "allow"
    assert (await decide(Permissions(mode="allow_all", deny=["*"]), read_order)).behavior == "deny"


async def test_callback_decides_and_fails_closed():
    read_order, refund_order, _ = make_tools([])

    def small_refunds_only(ctx, target, call):
        if target.name == "refund_order":
            return "allow" if call.arguments["amount"] <= 20 else "deny"
        return None

    permissions = Permissions(can_use_tool=small_refunds_only)
    assert (await decide(permissions, refund_order, amount=5)).behavior == "allow"
    assert (await decide(permissions, refund_order, amount=500)).behavior == "deny"
    assert (await decide(permissions, read_order)).behavior == "allow"  # None = no opinion

    async def broken(ctx, target, call):
        raise RuntimeError("policy service down")

    assert (await decide(Permissions(can_use_tool=broken), read_order)).behavior == "deny"
    assert (
        await decide(Permissions(can_use_tool=lambda *a: "maybe"), read_order)
    ).behavior == "deny"


async def test_denied_calls_are_reported_to_the_model():
    log: list[str] = []
    model = ScriptedModel([call("send_email", to="a@example.com"), "I could not send it."])
    agent = Agent(model, tools=make_tools(log), permissions=Permissions(deny=["send_*"]))
    result = await agent.run("email them")

    denied = result.messages[2].tool_results[0]
    assert denied.is_error and denied.content == (
        "'send_email' was not run: a deny rule matches this tool."
    )
    assert log == []


async def test_on_ask_resolves_approvals_inline():
    log: list[str] = []
    asked: list[tuple[str, str | None]] = []

    async def ask(ctx, target, call, reason):
        asked.append((target.name, reason))
        if call.arguments["amount"] > 100:
            return Approval(approved=False, reason="Refunds over 100 need a manager.")
        return Approval(approved=True, arguments={**call.arguments, "amount": 10.0})

    model = ScriptedModel(
        [
            [
                call("refund_order", order="A", amount=50),
                call("refund_order", order="B", amount=500),
            ],
            "Done.",
        ]
    )
    agent = Agent(model, tools=make_tools(log), permissions=Permissions(on_ask=ask))
    result = await agent.run("refund both")

    assert result.status == "completed"
    assert asked == [("refund_order", "this tool requires approval.")] * 2
    assert log == ["refund A 10.0"]  # the approver's edited arguments were used
    approved, declined = result.messages[2].tool_results
    assert approved.content == "refunded 10.0"
    assert declined.content == "'refund_order' was not run: Refunds over 100 need a manager."


async def test_without_on_ask_the_run_pauses_and_resumes():
    log: list[str] = []
    model = ScriptedModel(
        [
            [call("read_order", order="A"), call("refund_order", order="A", amount=30)],
            "Refunded.",
        ]
    )
    agent = Agent(model, tools=make_tools(log))
    session = Session()

    paused = await agent.run("refund order A", session=session)
    assert paused.status == "paused" and paused.output is None
    assert log == ["read A"]  # what needed no approval already ran
    (pending,) = paused.pending_approvals
    assert (pending.name, pending.arguments) == ("refund_order", {"order": "A", "amount": 30})
    assert (await session.state()).pending_approvals == [pending]

    done = await agent.resume(session, approvals={pending.tool_call_id: True})
    assert done.status == "completed" and done.output == "Refunded."
    assert done.run_id == paused.run_id
    assert log == ["read A", "refund A 30.0"]  # read_order was not run a second time
    assert [part.content for part in done.messages[2].tool_results] == [
        "order A: 2 items",
        "refunded 30.0",
    ]
    assert (await session.state()).open_run_id is None


async def test_resume_can_decline_and_can_stay_paused():
    log: list[str] = []
    model = ScriptedModel(
        [
            [call("refund_order", order="A", amount=1), call("refund_order", order="B", amount=2)],
            "Only one was refunded.",
        ]
    )
    agent = Agent(model, tools=make_tools(log))
    session = Session()
    paused = await agent.run("refund both", session=session)
    first, second = paused.pending_approvals

    still_paused = await agent.resume(session, approvals={first.tool_call_id: True})
    assert still_paused.status == "paused"
    assert [p.tool_call_id for p in still_paused.pending_approvals] == [second.tool_call_id]
    assert log == ["refund A 1.0"]

    done = await agent.resume(
        session, approvals={second.tool_call_id: Approval(approved=False, reason="Not this one.")}
    )
    assert done.status == "completed"
    assert log == ["refund A 1.0"]
    assert done.messages[2].tool_results[1].content == "'refund_order' was not run: Not this one."


async def test_a_paused_run_result_carries_its_session():
    model = ScriptedModel([call("refund_order", order="A", amount=1), "ok"])
    agent = Agent(model, tools=make_tools([]))
    paused = await agent.run("refund")  # no session given: one is created
    done = await agent.resume(
        paused.session, approvals={paused.pending_approvals[0].tool_call_id: True}
    )
    assert done.status == "completed"


async def test_on_ask_failure_refuses_the_call():
    log: list[str] = []

    def ask(ctx, target, call, reason):
        raise TimeoutError("nobody answered")

    model = ScriptedModel([call("refund_order", order="A", amount=1), "ok"])
    agent = Agent(model, tools=make_tools(log), permissions=Permissions(on_ask=ask))
    result = await agent.run("refund")
    assert log == []
    assert "the approval request failed" in result.messages[2].tool_results[0].content


@pytest.mark.parametrize("answer", [True, False])
async def test_on_ask_may_answer_with_a_bool(answer):
    log: list[str] = []
    model = ScriptedModel([call("refund_order", order="A", amount=1), "ok"])
    permissions = Permissions(on_ask=lambda ctx, target, call, reason: answer)
    await Agent(model, tools=make_tools(log), permissions=permissions).run("refund")
    assert bool(log) is answer
