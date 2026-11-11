"""An agent works with the tools of a local MCP server; stock changes are approved in code.

``MCPToolkit.stdio`` starts the inventory server in ``examples/patterns/_mcp_server.py`` with
this Python interpreter and exposes its tools as qagents tools. The server's annotations decide
how each one is treated: ``list_items`` and ``get_item`` are read-only and run at once;
``update_stock`` is destructive, so every call asks for approval first (``approval=
"destructive"``, the default). An ``on_ask`` policy answers in code: a change of up to 10
units of an item goes through, a larger one is declined for a manager, and so is every later
change of that item. The agent books a delivery of 5 cables and 40 monitors: the cables are
booked, the monitors wait.

Mirrors: langchain-mcp-adapters (``MultiServerMCPClient({...: {"transport": "stdio", ...}})``
and ``await client.get_tools()`` given to a LangGraph agent, with a human-in-the-loop interrupt
for review) and CrewAI's ``MCPServerAdapter(StdioServerParameters(...))``. In qagents the
toolkit keeps one server session for all its calls (``async with`` starts and stops the
server), it reads the server's annotations itself (read-only tools may run in parallel,
possibly destructive ones ask first),
the question is answered by ``Permissions(on_ask=...)`` inside the run (without an ``on_ask``
the run pauses for a person), tool definitions are pinned on first use, and server failures
reach the model as error results.

Run it:

    python -m examples.patterns.mcp_tools --provider anthropic
    python -m examples.patterns.mcp_tools --provider qcloud
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from examples.harness import ExampleContext, ExampleOutcome, main
from examples.patterns._mcp_server import STOCK
from qagents import Agent, Approval, Limits, Permissions, RunContext, Tool, ToolCallPart
from qagents.toolkits.mcp import MCPToolkit

NAME = "mcp_tools"
DESCRIPTION = "Tools of a local MCP server, with destructive calls approved by a policy in code"
INSPIRED_BY = "langchain-mcp-adapters / CrewAI MCPServerAdapter (MCP tools)"

SERVER = Path(__file__).with_name("_mcp_server.py")
LIMIT = 10
"""Units of one item the agent may book without a manager."""
DELIVERY = {"CB-01": 5, "MN-27": 40}
PROMPT = (
    "A delivery just arrived: 5 USB-C cables (CB-01) and 40 27-inch monitors (MN-27). Book "
    "both into the stock, then tell me how many of each are in stock now."
)
INSTRUCTIONS = (
    "You keep the stock of a small shop with the inventory tools. To book a delivery, call "
    "update_stock once per item, with the number of units received as a positive change. "
    "Some changes need a manager: when update_stock is declined, do not retry, split or change "
    "it; say that it waits for a manager. Then call get_item for each delivered item and "
    "finish with one line per item: its SKU and the quantity in stock now. Then stop."
)


class StockPolicy:
    """Answers the approval requests of ``update_stock`` (the ``on_ask`` of the permissions).

    A change goes through while the units booked for its item stay within ``limit``. A larger
    one is declined, and the item is held: every later change of it is declined too, so the
    agent cannot get around the limit by splitting the change.
    """

    def __init__(self, limit: int, log: Callable[[str], None] = print) -> None:
        self.limit = limit
        self.log = log
        self.booked: Counter[str] = Counter()
        self.held: set[str] = set()
        self.decisions: list[tuple[str, Any, bool]] = []

    def __call__(
        self, ctx: RunContext[Any], tool: Tool, call: ToolCallPart, reason: str | None
    ) -> Approval:
        if tool.name != "update_stock":  # the only tool of this server that asks
            return Approval(approved=False, reason="only stock changes can be approved here.")
        sku, change = str(call.arguments.get("sku")), call.arguments.get("change")
        if type(change) is not int:  # a malformed call is refused, but holds nothing
            self.decisions.append((sku, change, False))
            return Approval(approved=False, reason="change must be a whole number, like 5 or -2.")
        held = sku in self.held
        approved = not held and self.booked[sku] + abs(change) <= self.limit
        self.decisions.append((sku, change, approved))
        self.log(f"approval for update_stock({sku}, {change:+d}): {'yes' if approved else 'no'}")
        if approved:
            self.booked[sku] += abs(change)
            return Approval(approved=True)
        self.held.add(sku)
        # The reason is the tool result the model reads, so it says what to do next.
        why = (
            f"{sku} waits for a manager's approval of an earlier change"
            if held
            else f"{abs(change)} units of {sku} are over the {self.limit}-unit limit and need "
            "a manager's approval"
        )
        return Approval(
            approved=False,
            reason=f"{why}. Do not retry or split it; tell the user it waits for a manager.",
        )


async def run(ctx: ExampleContext) -> ExampleOutcome:
    stock_file = ctx.workdir / "stock.json"
    stock_file.write_text(json.dumps(STOCK, indent=2), encoding="utf-8")
    policy = StockPolicy(LIMIT, log=ctx.log)

    # The server's stderr goes to a file, so its logging does not mix with the run's output.
    with (ctx.workdir / "mcp_server.log").open("w", encoding="utf-8") as server_log:
        inventory = MCPToolkit.stdio(
            sys.executable,
            [SERVER.as_posix(), stock_file.as_posix()],
            name="inventory",
            errlog=server_log,
        )
        async with inventory:  # starts the server and lists its tools; stops it on the way out
            ctx.log(f"connected to the inventory server (MCP {inventory.protocol_version})")
            for listed in await inventory.tools():
                ctx.log(
                    f"  {listed.name}: read_only={listed.read_only}, "
                    f"needs approval={listed.requires_approval}"
                )
            clerk = Agent(
                ctx.model,
                name="stock_clerk",
                instructions=INSTRUCTIONS,
                tools=[inventory],
                permissions=Permissions(on_ask=policy),
                limits=Limits(max_turns=8, max_cost_usd=0.20),
                hooks=ctx.hooks,
            )
            result = await ctx.print_stream(clerk.stream(PROMPT))

    stock = {
        sku: item["quantity"]
        for sku, item in json.loads(stock_file.read_text(encoding="utf-8")).items()
    }
    expected = {sku: STOCK[sku]["quantity"] for sku in STOCK}
    expected["CB-01"] += DELIVERY["CB-01"]  # within the limit: booked; MN-27 waits for a manager
    ctx.log(f"stock now: {stock}")
    # The answer reports the cables' new quantity (the monitors' stays 4).
    answered = result.status == "completed" and str(expected["CB-01"]) in result.text
    ok = answered and stock == expected
    summary = (
        f"CB-01 booked ({expected['CB-01']} in stock); MN-27 held for a manager "
        f"({stock['MN-27']} in stock)"
        if ok
        else f"{result.status}: stock {stock}, expected {expected}; the answer "
        f"{'gives' if answered else 'does not give'} CB-01's new quantity"
    )
    return ExampleOutcome(
        ok=ok,
        summary=summary,
        output=result.text,
        details={"stock": stock, "decisions": policy.decisions},
    )


if __name__ == "__main__":
    main(run, name=NAME, description=DESCRIPTION, inspired_by=INSPIRED_BY)
