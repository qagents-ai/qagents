"""Credit a local ledger, exposing a barrier after commit and before acknowledgement."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

from pydantic import BaseModel

from qagents import Agent, RunContext, Session, SQLiteStore, tool
from qagents.effects import Attempt, Dedupe, EffectContext, Key, effect
from qagents.models.testing import FunctionModel, call


class Credit(BaseModel):
    order_id: str
    amount: int


def persist_credit(ledger: Path, attempt: Attempt[Credit]):
    request = attempt.request
    with closing(sqlite3.connect(ledger)) as db, db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("CREATE TABLE IF NOT EXISTS credits (token TEXT PRIMARY KEY, amount INTEGER)")
        db.execute("CREATE TABLE IF NOT EXISTS requests (token TEXT)")
        db.execute("INSERT INTO requests VALUES (?)", (attempt.idempotency_key,))
        db.execute(
            "INSERT OR IGNORE INTO credits VALUES (?, ?)",
            (attempt.idempotency_key, request.amount),
        )
        return db.execute(
            "SELECT amount FROM credits WHERE token = ?", (attempt.idempotency_key,)
        ).fetchone()[0]


async def run(root: Path, first: bool):
    @effect(
        "process.credit",
        key=Key("request", "order_id"),
        timeout_s=2,
        dedupe=Dedupe(3600, contract="SQLite primary key", concurrent="waits"),
    )
    async def send(ctx: EffectContext, attempt: Attempt[Credit]) -> Credit:
        amount = await asyncio.to_thread(persist_credit, root / "destination.db", attempt)
        if first:
            print(
                json.dumps({"pid": os.getpid(), "effect_key": attempt.idempotency_key}), flush=True
            )
            await asyncio.Event().wait()
        return Credit(order_id=attempt.request.order_id, amount=amount)

    @tool(effects=[send])
    async def credit(ctx: RunContext, order_id: str) -> int:
        """Credit the local test order."""
        result = await ctx.perform(send, Credit(order_id=order_id, amount=10))
        return result.amount

    agent = Agent(
        FunctionModel(
            lambda request: (
                "Credit done."
                if request.messages[-1].role == "tool"
                else call("credit", order_id="R1")
            )
        ),
        name="settlement",
        tools=[credit],
        lease_ttl_s=6.1,
    )
    async with SQLiteStore(root / "journal.db") as store:
        session = Session(store, id="case")
        if first:
            await agent.run("Credit R1", session=session)
        else:
            await asyncio.sleep(
                max(
                    0,
                    (await store.status(session.id)).lease_expires_at
                    + store.takeover_grace_s
                    - await store.now()
                    + 0.01,
                )
            )
            result = await agent.resume(session)
            print(json.dumps({"pid": os.getpid(), "result": result.text}), flush=True)


if __name__ == "__main__":
    asyncio.run(run(Path(sys.argv[1]), sys.argv[2] == "first"))
