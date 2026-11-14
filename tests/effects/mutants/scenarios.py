"""Small public Agent fixtures shared by the Effects mutation oracles."""

from qagents import Agent, RunContext, tool
from qagents.effects import Attempt, EffectContext, Key, effect
from qagents.permissions import Approval, Permissions

from ..test_engine_dispatch import Receipt, Request


def approve_refund(pending):
    return Approval.for_call(pending, by="fixture reviewer")


def approved_refund(env):
    @effect("refund", key=Key("request", "id"))
    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        raise AssertionError("the fixture transport owns the destination")

    @tool(effects=[send], requires_approval=True)
    async def refund(ctx: RunContext, id: str) -> str:
        """Refund the fixture order after review."""
        return (await ctx.perform(send, Request(id=id))).id

    return Agent(
        env.model,
        tools=[refund],
        permissions=Permissions(approval_secret="fixture-only-review-secret"),
    )
