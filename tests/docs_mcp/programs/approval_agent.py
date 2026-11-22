"""Dogfood 1: an agent with a tool that needs approval, paused and resumed."""

import asyncio

from qagents import Agent, Limits, RunContext, Session, tool
from qagents.models.testing import ScriptedModel, call

REFUNDS: list[tuple[str, float, str]] = []


@tool(read_only=True)
def find_order(order_id: str) -> str:
    """Look up an order by id."""
    return f"Order {order_id}: 2 items, paid 30.00 EUR"


@tool(requires_approval=True)
def refund_order(ctx: RunContext, order_id: str, amount: float) -> str:
    """Refund an order. Changes money: a human approves it first."""
    REFUNDS.append((order_id, amount, ctx.operation_id))
    return f"Refunded {amount:.2f} EUR on {order_id}"


async def main() -> None:
    model = ScriptedModel(
        [
            call("find_order", order_id="A-1001"),
            call("refund_order", order_id="A-1001", amount=30.0),
            "Order A-1001 was refunded: 30.00 EUR.",
        ]
    )
    agent = Agent(
        model,
        instructions="You handle refunds.",
        tools=[find_order, refund_order],
        limits=Limits(max_turns=6),
    )
    session = Session()
    paused = await agent.run("Refund order A-1001", session=session)
    assert paused.status == "paused", paused.status
    (pending,) = paused.pending_approvals
    print("paused for:", pending.name, pending.arguments)
    assert REFUNDS == []  # nothing refunded before the approval

    done = await agent.resume(session, approvals={pending.tool_call_id: True})
    print("status:", done.status, "| output:", done.output)
    assert done.status == "completed"
    assert len(REFUNDS) == 1 and REFUNDS[0][0] == "A-1001"
    print("refund journaled with operation id", REFUNDS[0][2])


if __name__ == "__main__":
    asyncio.run(main())
