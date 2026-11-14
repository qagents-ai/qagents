"""Recover an acknowledged-lost credit using two workers and a durable local destination."""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from qagents import Agent, AgentRuntime, RunContext, SQLiteStore, tool
from qagents.effects import Attempt, EffectContext, Key, Settle, effect
from qagents.models.testing import FunctionModel, call


class Credit(BaseModel):
    order: str
    amount: int


class Receipt(BaseModel):
    order: str
    amount: int


@dataclass
class Destination:
    path: Path
    lose_acknowledgement: bool

    def credit(self, attempt: Attempt[Credit]) -> Receipt:
        request = attempt.request
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS credits "
                "(token TEXT PRIMARY KEY, order_id TEXT NOT NULL, amount INTEGER NOT NULL)"
            )
            db.execute(
                "INSERT OR IGNORE INTO credits VALUES (?, ?, ?)",
                (attempt.idempotency_key, request.order, request.amount),
            )
            row = db.execute(
                "SELECT order_id, amount FROM credits WHERE token = ?",
                (attempt.idempotency_key,),
            ).fetchone()
            if row != (request.order, request.amount):
                raise ValueError("The existing key is bound to a different request.")
        if self.lose_acknowledgement:
            self.lose_acknowledgement = False
            raise ConnectionError("The destination committed before its response was lost.")
        return Receipt(order=request.order, amount=request.amount)


@effect(
    "example.credit",
    key=Key("request", "order"),
    timeout_s=2,
)
async def credit(ctx: EffectContext, attempt: Attempt[Credit]) -> Receipt:
    """Send the durable attempt key with the credit request."""
    return await asyncio.to_thread(ctx.deps.credit, attempt)


@tool(effects=[credit])
async def credit_order(ctx: RunContext[Destination], order: str) -> Receipt:
    """Credit the test order for ten units."""
    return await ctx.perform(credit, Credit(order=order, amount=10))


def build_agent() -> Agent:
    def reply(request):
        if request.messages[-1].role == "tool":
            return "Credit recorded."
        return call("credit_order", order="DEMO-1")

    return Agent(FunctionModel(reply), name="credit-review", tools=[credit_order])


async def main(directory: Path) -> None:
    await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
    journal, ledger = directory / "runs.db", directory / "credits.db"
    if journal.exists() or ledger.exists():
        raise FileExistsError("Choose a fresh directory for the demonstration.")
    async with SQLiteStore(journal) as store:
        destination = Destination(ledger, lose_acknowledgement=True)
        async with AgentRuntime(store, [build_agent()], worker_id="initial") as runtime:
            first = await runtime.start(
                "credit-review", "Credit DEMO-1", session_id="example", deps=destination
            )
            paused = await first.wait()
            assert paused.stop_reason == "effect_held"
    with closing(sqlite3.connect(ledger)) as db:
        (row,) = db.execute("SELECT order_id, amount FROM credits").fetchall()
    verified = Receipt(order=row[0], amount=row[1])
    async with SQLiteStore(journal) as store:
        destination = Destination(ledger, lose_acknowledgement=False)
        async with AgentRuntime(store, [build_agent()], worker_id="recovery") as runtime:
            await runtime.settle(
                "example",
                Settle.committed(
                    paused.held_effects[0],
                    receipt=verified,
                    by="demo-operator",
                    evidence="single matching row in the destination ledger",
                ),
            )
            resumed = await runtime.resume("example", deps=destination)
            assert (await resumed.wait()).text == "Credit recorded."
    with closing(sqlite3.connect(ledger)) as db:
        assert db.execute("SELECT COUNT(*) FROM credits").fetchone() == (1,)
    print("One durable credit recorded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    asyncio.run(main(parser.parse_args().directory))
