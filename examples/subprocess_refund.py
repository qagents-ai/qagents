"""Recover a simulated refund after terminating the worker that submitted it."""

import asyncio
from pathlib import Path

from pydantic import BaseModel

from qagents import Agent, RunContext, tool
from qagents.effects import Attempt, Dedupe, EffectContext, Key, effect
from qagents.models.testing import call
from qagents.testing.crash import CrashEnv, FakeStripeRefunds, Faults, crash_test


class RefundRequest(BaseModel):
    payment: str


class RefundReceipt(BaseModel):
    id: str


@effect(
    "stripe.refund",
    key=Key("payment", "payment"),
    dedupe=Dedupe(120, contract="simulated refund key", concurrent="rejects", margin_s=0.1),
    timeout_s=1,
)
async def send(ctx: EffectContext, attempt: Attempt[RefundRequest]) -> RefundReceipt:
    refund = await ctx.deps.create_refund(
        charge=attempt.request.payment,
        amount=100,
        idempotency_key=attempt.idempotency_key,
    )
    return RefundReceipt(id=refund.id)


@tool(effects=[send])
async def refund_payment(ctx: RunContext, payment: str) -> str:
    """Refund one euro from the simulated payment."""
    receipt = await ctx.perform(send, RefundRequest(payment=payment))
    return receipt.id


def build(env: CrashEnv):
    return Agent(env.model, tools=[refund_payment]), env.service(send, FakeStripeRefunds)


async def main() -> None:
    # Import the same named module the new interpreters will load.
    from examples.subprocess_refund import build

    report = await crash_test(
        build,
        "Refund one euro on ch_A.",
        model=[call("refund_payment", payment="ch_A"), "Refund complete."],
        faults=Faults.parse("commit_then_crash"),
        points="effects",
        isolation="subprocess",
        transport="client",
    )
    await asyncio.to_thread(
        Path("refund-recovery.json").write_text, report.to_json(), encoding="utf-8"
    )
    point = next(item for item in report.points if item.execution.triggered)
    print(report.explain(point.id))
    if point.verdict != "safe":
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
