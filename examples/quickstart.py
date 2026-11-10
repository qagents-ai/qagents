"""A support assistant on QCloud: tools, streaming, an approval and a durable session.

Needs the qcloud-ai SDK and two environment variables:

    QCLOUD_API_KEY   your QCloud key
    QCLOUD_MODEL     an active model id from the catalog that lists the "tools" capability

Run it twice: the second run continues the same conversation from agents.db.
This makes real, billable model requests.
"""

from __future__ import annotations

import asyncio
import os

from qagents import Agent, Approval, Limits, Permissions, RunContext, Session, SQLiteStore, tool

ORDERS = {"A-1001": {"status": "delivered", "total": 42.50}}


@tool(read_only=True)
def get_order(order_id: str) -> dict:
    """Look up an order by its id.

    Args:
        order_id: The order id, for example "A-1001".
    """
    order = ORDERS.get(order_id)
    return order or {"error": f"No order {order_id}"}


@tool(requires_approval=True)
def refund_order(ctx: RunContext, order_id: str, amount: float) -> str:
    """Refund part or all of an order to the customer's card.

    Args:
        order_id: The order to refund.
        amount: Amount in EUR.
    """
    # ctx.operation_id stays the same if this call is repeated after a crash:
    # hand it to the payment provider as the idempotency key.
    return f"Refunded {amount:.2f} EUR for {order_id} (operation {ctx.operation_id})"


def ask_on_terminal(ctx, tool, call, reason) -> Approval:
    answer = input(f"\nAllow {tool.name}({call.arguments})? [y/N] ")
    return Approval(approved=answer.strip().lower() == "y")


async def main() -> None:
    store = SQLiteStore("agents.db")
    agent = Agent(
        f"qcloud:{os.environ['QCLOUD_MODEL']}",
        instructions="You are a support assistant for an online shop. Be brief and factual.",
        tools=[get_order, refund_order],
        permissions=Permissions(on_ask=ask_on_terminal),
        limits=Limits(max_turns=8, max_cost_usd=0.05),
    )
    stream = agent.stream(
        "Order A-1001 arrived damaged. Please refund 10 euros.",
        session=Session(store, id="demo-customer-1"),
    )
    async for event in stream:
        if event.type == "text_delta":
            print(event.text, end="", flush=True)
        elif event.type == "tool_call_finished":
            print(f"\n[{event.call.name} -> {event.result.content}]")
    result = stream.result
    print(f"\n\nstatus={result.status} turns={result.turns} cost_usd={result.usage.cost_usd}")
    await agent.model.aclose()
    await store.aclose()


if __name__ == "__main__":
    asyncio.run(main())
