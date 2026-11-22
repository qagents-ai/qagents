"""A support agent that queues an approved case using QCloud, OpenAI or Anthropic.

Model calls use your configured provider. The orders are explicit demo fixtures.
The review queue is a real local SQLite database with operation-key deduplication.

    python examples/support_review.py --provider qcloud
    python examples/support_review.py --provider openai
    python examples/support_review.py --provider anthropic

Set QCLOUD_MODEL/QCLOUD_API_KEY, OPENAI_MODEL/OPENAI_API_KEY or
ANTHROPIC_MODEL/ANTHROPIC_API_KEY. No provider is silently substituted.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
from contextlib import closing
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from qagents import Agent, Approval, Limits, ModelSettings, RunContext, Session, SQLiteStore, tool
from qagents.errors import ToolError
from qagents.models.base import Model

DEMO_ORDERS = {
    "A-1001": {"status": "delivered", "amount_minor": 4250, "currency": "EUR"},
    "A-1002": {"status": "in_transit", "amount_minor": 1900, "currency": "EUR"},
}


class ReviewQueue:
    """A local destination with a durable operation key and payload binding."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with closing(sqlite3.connect(path)) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS cases ("
                "operation_id TEXT PRIMARY KEY, case_id TEXT NOT NULL, payload TEXT NOT NULL)"
            )

    def enqueue(self, operation_id: str, order_id: str, reason: str) -> dict:
        payload = json.dumps({"order_id": order_id, "reason": reason}, sort_keys=True)
        case_id = "case-" + hashlib.sha256(operation_id.encode()).hexdigest()
        with closing(sqlite3.connect(self.path, timeout=10)) as db, db:
            db.execute(
                "INSERT OR IGNORE INTO cases VALUES (?, ?, ?)", (operation_id, case_id, payload)
            )
            row = db.execute(
                "SELECT case_id, payload FROM cases WHERE operation_id = ?", (operation_id,)
            ).fetchone()
            if row[1] != payload:
                raise ToolError("This operation already holds a different review request.")
        return {"case_id": row[0], "status": "awaiting_review", "order_id": order_id}


@dataclass
class SupportDeps:
    orders: dict[str, dict]
    reviews: ReviewQueue


@tool(read_only=True)
def inspect_order(ctx: RunContext[SupportDeps], order_id: str) -> dict:
    """Read an order's delivery status, paid amount and currency."""
    if order_id not in ctx.deps.orders:
        raise ToolError("Order not found.")
    return {"order_id": order_id, **deepcopy(ctx.deps.orders[order_id])}


@tool(requires_approval=True, idempotent=True)
def queue_review(ctx: RunContext[SupportDeps], order_id: str, reason: str) -> dict:
    """Queue a support case for human review. This does not issue a refund."""
    if order_id not in ctx.deps.orders:
        raise ToolError("Order not found.")
    if not reason.strip() or len(reason) > 2000:
        raise ToolError("Give a review reason of 1 to 2000 characters.")
    return ctx.deps.reviews.enqueue(ctx.operation_id, order_id, reason)


def build_agent(model: Model) -> Agent:
    return Agent(
        model,
        name="support-review",
        instructions=(
            "Help the customer using the order tools. Inspect the order before proposing a case. "
            "Queue a review only when asked. "
            "Never claim a refund or external notification was sent. "
            "After queueing, report the returned case id and status."
        ),
        tools=[inspect_order, queue_review],
        settings=ModelSettings(max_tokens=1200),
        limits=Limits(max_turns=6, max_total_tokens=12000, timeout_s=90, tool_timeout_s=15),
    )


def configured_model(provider: str) -> Model:
    """Require an explicit model and credential for the selected provider."""
    prefix = {"qcloud": "QCLOUD", "openai": "OPENAI", "anthropic": "ANTHROPIC"}[provider]
    model_id, api_key = os.environ.get(prefix + "_MODEL"), os.environ.get(prefix + "_API_KEY")
    if not model_id or not api_key:
        raise ValueError(f"Set {prefix}_MODEL and {prefix}_API_KEY before running this example.")
    if provider == "qcloud":
        from qagents import QCloudModel

        return QCloudModel(model_id, api_key=api_key, timeout=30)
    if provider == "anthropic":
        from qagents import AnthropicModel

        return AnthropicModel(model_id, api_key=api_key, timeout=30)
    from qagents import OpenAIChatModel

    return OpenAIChatModel(
        model_id,
        base_url="https://api.openai.com/v1",
        api_key=api_key,
        provider="openai",
        max_tokens_param="max_completion_tokens",
        timeout=30,
    )


async def run_demo(provider: str, directory: Path, prompt: str, session_id: str) -> None:
    model = configured_model(provider)
    try:
        deps = SupportDeps(deepcopy(DEMO_ORDERS), ReviewQueue(directory / "reviews.db"))
        agent = build_agent(model)
        async with SQLiteStore(directory / "sessions.db") as store:
            session = Session(store, id=session_id)
            result = await agent.run(prompt, session=session, deps=deps)
            while result.status == "paused" and result.pending_approvals:
                decisions = {}
                for pending in result.pending_approvals:
                    arguments = json.dumps(pending.arguments, ensure_ascii=False)
                    print(f"\nProposed {pending.name}: {arguments}")
                    answer = await asyncio.to_thread(input, "Approve this exact action? [y/N] ")
                    approved = answer.strip().lower() == "y"
                    decisions[pending.tool_call_id] = Approval.for_call(
                        pending, approved=approved, by="terminal-user"
                    )
                result = await agent.resume(session, approvals=decisions, deps=deps)
            print(f"\nStatus: {result.status}\n{result.text}")
    finally:
        await model.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, choices=("qcloud", "openai", "anthropic"))
    parser.add_argument("--directory", type=Path, default=Path(".work/support-review"))
    parser.add_argument("--session", default="support-demo")
    parser.add_argument(
        "--prompt",
        default="Order A-1001 arrived damaged. Check it and open a case for human review.",
    )
    args = parser.parse_args()
    asyncio.run(run_demo(args.provider, args.directory, args.prompt, args.session))
