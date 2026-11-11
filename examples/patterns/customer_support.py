"""Customer support with handoffs: a triage agent passes the customer to a specialist.

A triage agent reads the first message and hands the conversation to billing, technical
support or sales. Billing can refund an order, but ``issue_refund`` requires approval: the run
pauses, ``Permissions(approval_secret=...)`` signs the pause, and the example approves it with
the pause's token and resumes it from a fresh `Session` on the same SQLite file. An approval
without the token is refused before anything runs. The customer's second message goes straight
to the specialist that answered the first one.

Mirrors: LangGraph's swarm (``create_handoff_tool`` on each agent, and ``create_swarm`` keeping
``active_agent`` in the checkpointed state, so the next message on the thread goes to the agent
that spoke last) and the OpenAI Agents SDK's handoffs (``Agent(handoffs=[...])``, then
``result.last_agent`` for the next turn). In qagents ``handoffs=[...]`` gives the agent one
``transfer_to_<name>`` tool per target and the loop switches agents itself: the target takes
over the same history, in the same run. There is no swarm object: ``result.agent`` names the
agent that answered, and your code sends the next message to it, as with ``last_agent``. Where
LangGraph pauses with ``interrupt()`` inside the tool and resumes with ``Command(resume=...)``,
here the tool declares ``requires_approval=True``, the pause is written to the session journal,
and ``resume`` accepts only an answer that carries the HMAC token of that exact call.

Run it:

    python -m examples.patterns.customer_support --provider anthropic
    python -m examples.patterns.customer_support --provider qcloud --interactive
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Any

from examples.harness import ExampleContext, ExampleOutcome, main
from qagents import (
    Agent,
    Approval,
    ApprovalTokenError,
    Limits,
    Permissions,
    RunContext,
    Session,
    SQLiteStore,
    ToolError,
    tool,
)

NAME = "customer_support"
DESCRIPTION = "Triage hands off to billing, a signed approval gates the refund, turn 2 stays there"
INSPIRED_BY = "LangGraph swarm / OpenAI Agents SDK handoffs (customer support)"

SESSION_ID = "customer-4417"
MESSAGES = (
    "Hello, my order A-1001 (wireless headphones) arrived with a cracked case. "
    "Please refund the full amount to my card.",
    "Thanks! How many business days until the money is back on my card?",
)

ORDERS = {
    "A-1001": {"item": "wireless headphones", "total_eur": 129.0, "status": "delivered"},
    "A-1002": {"item": "USB-C charger", "total_eur": 24.5, "status": "in transit"},
}
PLANS = {"basic": "2-year warranty, 4.99 EUR/month", "plus": "3-year warranty, 7.99 EUR/month"}


@dataclass
class Desk:
    """What the tools change, passed as ``deps``: the refunds issued in this example."""

    refunds: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Refunds by order id; an order is refunded at most once."""


def find_order(order_id: str) -> dict[str, Any]:
    order = ORDERS.get(order_id.strip().upper())
    if order is None:
        raise ToolError(f"There is no order {order_id!r}. Ask the customer for the order id.")
    return order


# -- billing -----------------------------------------------------------------------------------


@tool(read_only=True)
def lookup_order(order_id: str) -> dict[str, Any]:
    """Look up an order: the item, the total in EUR and the delivery status.

    Args:
        order_id: The order id, for example "A-1001".
    """
    return {"order_id": order_id.strip().upper(), **find_order(order_id)}


@tool(requires_approval=True)
def issue_refund(ctx: RunContext[Desk], order_id: str, amount_eur: float, reason: str) -> str:
    """Refund an order to the customer's card. A supervisor approves every refund before it
    is made, so do not ask the customer to confirm first.

    Args:
        order_id: The order to refund.
        amount_eur: The amount in EUR, at most the order total.
        reason: Why the order is refunded, in a few words.
    """
    order, key = find_order(order_id), order_id.strip().upper()
    if key in ctx.deps.refunds:
        raise ToolError(f"Order {key} was already refunded. Never refund it twice.")
    if not 0 < amount_eur <= order["total_eur"]:
        raise ToolError(f"The amount must be above 0 and at most {order['total_eur']:.2f} EUR.")
    # ctx.operation_id stays the same if this call is repeated after a crash: it is the key a
    # real payment API would deduplicate on. It holds a run id, so it never reaches the model.
    refund = {"refund_id": f"RF-{key}", "amount_eur": amount_eur, "reason": reason}
    ctx.deps.refunds[key] = {**refund, "operation_id": ctx.operation_id}
    return f"Refund RF-{key} of {amount_eur:.2f} EUR issued for order {key}."


@tool(read_only=True)
def refund_status(ctx: RunContext[Desk], order_id: str) -> str:
    """The status of the refund of an order and when the money reaches the card.

    Args:
        order_id: The refunded order.
    """
    refund = ctx.deps.refunds.get(order_id.strip().upper())
    if refund is None:
        return f"Order {order_id} has not been refunded."
    return (
        f"Refund {refund['refund_id']} of {refund['amount_eur']:.2f} EUR is being processed by "
        "the card network; it reaches the card within 3 to 5 business days."
    )


# -- the other specialists ---------------------------------------------------------------------


@tool(read_only=True)
def troubleshooting_guide(product: str) -> str:
    """Troubleshooting steps for a product that does not work.

    Args:
        product: The product, for example "wireless headphones".
    """
    return "1. Charge it fully. 2. Hold the power button for 15 seconds. 3. Pair it again."


@tool(read_only=True)
def list_protection_plans() -> dict[str, str]:
    """The protection plans on sale and their monthly prices."""
    return PLANS


TRIAGE = (
    "You are the triage agent of an electronics shop's support desk. You never solve the "
    "customer's problem yourself. Read the message and call exactly one transfer tool: "
    "billing for refunds, payments and invoices; technical_support for devices that do not "
    "work; sales for prices, plans and new purchases. Write no text of your own."
)
BILLING = (
    "You are the billing specialist of an electronics shop. To refund an order, call "
    "lookup_order, then call issue_refund once with the order total unless the customer asked "
    "for less. issue_refund gets a supervisor's approval by itself: do not ask the customer to "
    "confirm. If the refund is declined, tell the customer and do not try again. For a question "
    "about a refund that was already issued, call refund_status and answer from it; never "
    "issue a second refund. Reply in at most three sentences."
)


async def run(ctx: ExampleContext) -> ExampleOutcome:
    # Signs every pause with HMAC-SHA256. Load it from your secret store; never log it.
    permissions = Permissions(approval_secret=secrets.token_hex(32))
    # Any agent may start a run (turn 2 starts with billing), and the agent a run starts with
    # owns its limits, permissions and hooks, so every agent gets the same ones.
    shared: dict[str, Any] = {
        "limits": Limits(max_turns=8, max_cost_usd=0.20),
        "permissions": permissions,
        "hooks": ctx.hooks,
    }
    billing = Agent(
        ctx.model,
        name="billing",
        description="Refunds, payments and invoices.",
        instructions=BILLING,
        tools=[lookup_order, issue_refund, refund_status],
        **shared,
    )
    technical = Agent(
        ctx.model,
        name="technical_support",
        description="Devices that do not work, setup and repairs.",
        instructions="You are technical support. Use troubleshooting_guide, then reply briefly.",
        tools=[troubleshooting_guide],
        **shared,
    )
    sales = Agent(
        ctx.model,
        name="sales",
        description="Prices, protection plans and new purchases.",
        instructions="You are sales. Use list_protection_plans, then reply briefly.",
        tools=[list_protection_plans],
        **shared,
    )
    triage = Agent(
        ctx.model,
        name="triage",
        instructions=TRIAGE,
        handoffs=[billing, technical, sales],
        **shared,
    )
    agents = {agent.name: agent for agent in (triage, billing, technical, sales)}

    desk = Desk()
    store = SQLiteStore(ctx.workdir / "support.db")
    try:
        ctx.log(f"Customer: {MESSAGES[0]}")
        first = await ctx.print_stream(
            triage.stream(MESSAGES[0], session=Session(store, id=SESSION_ID), deps=desk)
        )

        approved = forged_refused = False
        if first.status == "paused":
            # The pause lives in support.db; a new Session object (or another process) resumes it.
            session = Session(store, id=SESSION_ID)
            pending = first.pending_approvals
            try:  # an approval that does not carry the pause's token is refused
                await triage.resume(session, approvals={p.tool_call_id: True for p in pending})
            except ApprovalTokenError:
                forged_refused = True
                ctx.log("An approval without the pause's token was refused; nothing ran")
            approvals = {}
            for call in pending:
                arguments = ", ".join(f"{key}={value!r}" for key, value in call.arguments.items())
                question = f"Approve {call.name}({arguments})?"
                approved = (await ctx.ask(question, default="yes")).lower().startswith("y")
                approvals[call.tool_call_id] = Approval.for_call(
                    call, approved, reason=None if approved else "The supervisor declined."
                )
            ctx.log(f"Resuming with {first.agent}, approvals signed with the pause's token")
            # resume() goes to the agent that started the run; the run continues with billing.
            first = await ctx.print_stream(
                triage.resume_stream(session, approvals=approvals, deps=desk)
            )

        # Like a swarm's active_agent: the next message goes to whoever answered the last one.
        active = agents.get(first.agent, triage)
        ctx.log(f"Customer: {MESSAGES[1]} (sent to {active.name})")
        second = await ctx.print_stream(
            active.stream(MESSAGES[1], session=Session(store, id=SESSION_ID), deps=desk)
        )
    finally:
        await store.aclose()

    refunds = len(desk.refunds)
    handed_off = first.agent == "billing"
    stayed = second.agent == "billing" and second.status == "completed" and bool(second.text)
    expected = 1 if approved else 0
    ok = (
        first.status == "completed"
        and handed_off
        and stayed
        and refunds == expected
        and forged_refused
    )
    if ok:
        summary = (
            f"billing took over, {refunds} refund issued after a signed approval, "
            "and turn 2 was answered by billing"
            if approved
            else "billing took over, the refund was declined, and turn 2 was answered by billing"
        )
    else:
        summary = (
            f"turn 1: {first.status} by {first.agent}, turn 2: {second.status} by "
            f"{second.agent}, refunds: {refunds} (expected {expected}), "
            f"unsigned approval refused: {forged_refused}"
        )
    return ExampleOutcome(
        ok=ok,
        summary=summary,
        output=[first.text, second.text],
        details={
            "agents": [first.agent, second.agent],
            "refunds": {key: refund["refund_id"] for key, refund in desk.refunds.items()},
            "approved": approved,
            "unsigned_approval_refused": forged_refused,
            "turn_two_tools": [c.name for m in second.new_messages for c in m.tool_calls],
        },
    )


if __name__ == "__main__":
    main(run, name=NAME, description=DESCRIPTION, inspired_by=INSPIRED_BY)
