from __future__ import annotations

import pytest

from qagents import Agent, Approval, Permissions, RunContext, Session, tool
from qagents.errors import ConfigurationError
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


@pytest.mark.parametrize(
    "permissions",
    [
        Permissions(),
        Permissions(mode="allow_all"),
        Permissions(mode="ask_all"),
        Permissions(mode="read_only"),
        Permissions(ask=["send_*"]),
        Permissions(allow=["refund_*"]),
        Permissions(ask=["refund_*"], allow=["refund_*"]),
        Permissions(mode="ask_all", allow=["read_*"]),
        Permissions(mode="ask_all", deny=["send_*"]),
        Permissions(ask=["*"], deny=["refund_order"]),
    ],
)
async def test_asks_by_rule_agrees_with_decide_without_a_callback(permissions):
    """The agent's build-time check (an approval no secret signs) uses the same rule order."""
    for target in make_tools([]):
        asks = (await decide(permissions, target, order="A", amount=1)).behavior == "ask"
        assert permissions._asks_by_rule(target) is asks, target.name


def test_asks_by_rule_leaves_out_can_use_tool():
    _, refund_order, send_email = make_tools([])
    permissions = Permissions(can_use_tool=lambda *args: "allow")
    assert permissions._asks_by_rule(refund_order) is True  # requires_approval, by the rules
    assert Permissions(can_use_tool=lambda *args: "ask")._asks_by_rule(send_email) is False


def test_callback_may_ask_for_any_call_no_deny_rule_refuses_first():
    """can_use_tool decides per call, so the build check counts every call it may see."""
    _, refund_order, _ = make_tools([])
    callback = Permissions(can_use_tool=lambda *args: "allow", deny=["send_*"])
    assert [callback._callback_may_ask(item) for item in make_tools([])] == [True, True, False]
    assert not Permissions(mode="ask_all")._callback_may_ask(refund_order)  # no callback


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


async def test_refusal_reasons_come_from_prompts_at_decision_time(monkeypatch):
    from qagents import prompts

    read_order, _, send_email = make_tools([])
    monkeypatch.setattr(prompts, "PERMISSION_REASON_DENY_RULE", "deny rule hit.")
    monkeypatch.setattr(prompts, "PERMISSION_REASON_READ_ONLY", "read-only mode.")
    monkeypatch.setattr(prompts, "PERMISSION_REASON_CHECK_FAILED", "check crashed.")
    monkeypatch.setattr(prompts, "PERMISSION_REASON_CHECK_REFUSED", "check said no.")

    def broken(*args):
        raise RuntimeError("boom")

    reasons = [
        (await decide(Permissions(deny=["send_*"]), send_email, to="a")).reason,
        (await decide(Permissions(mode="read_only"), send_email, to="a")).reason,
        (await decide(Permissions(can_use_tool=broken), read_order, order="1")).reason,
        (await decide(Permissions(can_use_tool=lambda *a: False), read_order, order="1")).reason,
    ]
    assert reasons == ["deny rule hit.", "read-only mode.", "check crashed.", "check said no."]


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
    # The model is told the call ran with the approver's arguments, not its own.
    assert approved.content == (
        "The approver changed the arguments of this call before it ran. It ran with: "
        '{"order": "A", "amount": 10.0}\nrefunded 10.0'
    )
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


# -- approvals given to resume (F52, F53, F88, F134, L3) ---------------------------------


async def test_an_approval_never_overrides_a_deny_rule():
    """A call left open by a stop after model_responded, then 'approved' by a client."""
    log: list[str] = []
    model = ScriptedModel([call("send_email", to="ceo@x")])
    agent = Agent(model, tools=make_tools(log), permissions=Permissions(deny=["send_*"]))
    session = Session()
    stream = agent.stream("email", session=session)
    async with stream:
        async for event in stream:
            if event.type == "model_responded":
                break
    call_id = (await session.state()).messages[-1].tool_calls[0].id

    later = Agent(
        ScriptedModel(["Not sent."]),
        tools=make_tools(log),
        permissions=Permissions(deny=["send_*"]),
    )
    result = await later.resume(session, approvals={call_id: True})
    assert log == []
    assert "a deny rule matches" in result.messages[2].tool_results[0].content


async def test_a_deny_rule_added_while_paused_refuses_the_approved_call():
    log: list[str] = []
    model = ScriptedModel([call("refund_order", order="A", amount=5), "ok"])
    session = Session()
    paused = await Agent(model, tools=make_tools(log)).run("refund", session=session)
    pending = paused.pending_approvals[0]

    incident = Agent(model, tools=make_tools(log), permissions=Permissions(deny=["refund_*"]))
    result = await incident.resume(session, approvals={pending.tool_call_id: True})
    assert log == [] and result.status == "completed"
    assert "a deny rule matches" in result.messages[2].tool_results[0].content


async def test_edited_arguments_are_checked_journaled_and_shown_to_the_model():
    log: list[str] = []

    def policy(ctx, target, call):
        if target.name == "refund_order" and call.arguments["amount"] > 100:
            return "deny"
        return None

    permissions = Permissions(can_use_tool=policy)
    model = ScriptedModel(
        [
            call("refund_order", order="A", amount=10),
            call("refund_order", order="B", amount=1),
            "ok",
        ]
    )
    agent = Agent(model, tools=make_tools(log), permissions=permissions)
    session = Session()
    paused = await agent.run("refund", session=session)
    pending = paused.pending_approvals[0]

    too_much = Approval(approved=True, arguments={"order": "SOMEONE-ELSE", "amount": 100000})
    second = await agent.resume(session, approvals={pending.tool_call_id: too_much})
    assert log == []  # the edited call is checked like any other, and denied
    refused = second.messages[2].tool_results[0]
    assert (
        refused.is_error and refused.content == "'refund_order' was not run: it is not permitted."
    )
    decided = [e.data for e in await session.entries() if e.kind == "approval_decided"]
    assert decided == [
        {
            "tool_call_id": pending.tool_call_id,
            "name": "refund_order",
            "approved": True,
            "reason": None,
            "arguments_changed": True,
            "arguments": {"order": "SOMEONE-ELSE", "amount": 100000},
        }
    ]

    pending = second.pending_approvals[0]
    edit = Approval(approved=True, arguments={"order": "B", "amount": 2})
    done = await agent.resume(session, approvals={pending.tool_call_id: edit})
    assert log == ["refund B 2.0"]
    assert done.messages[4].tool_results[0].content.startswith("The approver changed the arguments")


@pytest.mark.parametrize("answer", ["false", 1, None, "yes"])
async def test_an_approval_that_is_not_a_bool_or_an_approval_is_refused(answer):
    log: list[str] = []
    model = ScriptedModel([call("refund_order", order="A", amount=5), "ok"])
    agent = Agent(model, tools=make_tools(log))
    session = Session()
    pending = (await agent.run("refund", session=session)).pending_approvals[0]
    with pytest.raises(ConfigurationError, match="Give True, False or an Approval"):
        await agent.resume(session, approvals={pending.tool_call_id: answer})
    assert log == [] and (await session.state()).pending_approvals == [pending]


async def test_an_approval_for_a_call_that_is_not_open_never_approves_a_later_call():
    log: list[str] = []
    # A provider that numbers calls per turn: ids repeat across turns.
    turn_one = [
        ToolCallPart(id="call_0", name="refund_order", arguments={"order": "A", "amount": 1})
    ]
    turn_two = [
        ToolCallPart(id="call_0", name="refund_order", arguments={"order": "B", "amount": 1}),
        ToolCallPart(id="call_1", name="refund_order", arguments={"order": "M", "amount": 9}),
    ]
    model = ScriptedModel([turn_one, turn_two, "ok"])
    agent = Agent(model, tools=make_tools(log))
    session = Session()
    await agent.run("refund", session=session)
    result = await agent.resume(session, approvals={"call_0": True, "call_1": True})
    assert log == ["refund A 1.0"]  # call_1 of the next turn was never reviewed
    assert sorted(p.tool_call_id for p in result.pending_approvals) == ["call_0", "call_1"]


@pytest.mark.parametrize("sign", [1, -1])
def test_signing_refuses_every_integer_outside_the_safe_json_range(sign):
    from qagents.permissions import approval_token, canonical_json

    common = dict(session_id="s", run_id="r", position=1, tool_call_id="c", name="send")
    safe = sign * (2**53 - 1)
    assert canonical_json(safe) == str(safe)
    assert approval_token("test-key-at-least-sixteen", arguments={"n": safe}, **common)
    for value in (sign * 2**53, sign * (2**53 + 1), sign * 10**16):
        with pytest.raises(ValueError):
            approval_token("test-key-at-least-sixteen", arguments={"n": value}, **common)


@pytest.mark.parametrize("n", [2**53, 2**53 + 1])
async def test_unsafe_integer_call_fails_closed_as_unsignable(n):
    from qagents import prompts

    executed = []

    @tool(requires_approval=True)
    def send(n: int) -> str:
        """Send the numbered item."""
        executed.append(n)
        return "sent"

    agent = Agent(
        ScriptedModel([call("send", n=n), "refused"]),
        tools=[send],
        permissions=Permissions(approval_secret="test-key-at-least-sixteen"),
    )
    result = await agent.run("send")
    refused = result.messages[2].tool_results[0]
    assert refused.is_error
    assert prompts.UNSIGNABLE_CALL in refused.content
    assert executed == [] and result.pending_approvals == []
