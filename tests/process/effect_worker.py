"""Run an Agent against a parent-owned destination and a real SQLite journal."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from pydantic import BaseModel

from qagents import Agent, RunContext, Session, SQLiteStore, tool
from qagents.effects import Attempt, Dedupe, EffectContext, Key, effect
from qagents.effects._ports import REACH_OBSERVER
from qagents.messages import ToolCallPart
from qagents.models.recording import request_fingerprint
from qagents.models.testing import FunctionModel
from qagents.session import Entry


def request(kind, **fields):
    # The private stdin/stdout pipes connect the worker solely to its parent.
    print(json.dumps({"kind": kind, **fields}), flush=True)
    line = sys.stdin.readline()
    if not line:
        raise RuntimeError("Parent closed the control pipe")
    return json.loads(line)


class Credit(BaseModel):
    order_id: str
    amount: int


class ObservedSQLiteStore(SQLiteStore):
    def __init__(self, path: Path, *, now: float):
        super().__init__(path, clock=lambda: now)
        self._writes = 0

    async def _observe(self, append, session_id, entries: list[Entry], *extra):
        self._writes += 1
        index = self._writes
        kinds = [entry.kind for entry in entries]
        request("write", phase="before", index=index, entries=kinds)
        seqs = await append(session_id, entries, *extra)
        request("write", phase="after", index=index, entries=kinds)
        return seqs

    async def append(self, session_id, entries):
        return await self._observe(super().append, session_id, entries)

    async def append_leased(self, session_id, entries, owner):
        return await self._observe(super().append_leased, session_id, entries, owner)


def build_agent(model, effect_name: str):

    @effect(
        effect_name,
        key=Key("request", "order_id"),
        timeout_s=2,
        dedupe=Dedupe(3600, contract="parent ledger", concurrent="waits"),
    )
    async def send(ctx: EffectContext, attempt: Attempt[Credit]) -> Credit:
        response = request(
            "send",
            token=attempt.idempotency_key,
            tag=attempt.operation_tag,
            attempt=attempt.debug.number,
            epoch=attempt.debug.epoch,
            **attempt.request.model_dump(),
        )
        return Credit.model_validate(response)

    @tool(effects=[send])
    async def execute(ctx: RunContext, order_id: str) -> int:
        """Apply an approved amount to the test destination."""
        return (await ctx.perform(send, Credit(order_id=order_id, amount=10))).amount

    return Agent(model, name="settlement", tools=[execute], lease_ttl_s=60)


async def run(root: Path, now: float, effect_name: str):
    request("hello", pid=os.getpid())

    def model(model_request):
        reply = request(
            "model",
            role=model_request.messages[-1].role,
            fingerprint=request_fingerprint(model_request),
        )
        return reply if isinstance(reply, str) else ToolCallPart.model_validate(reply)

    agent = build_agent(FunctionModel(model), effect_name)
    observer = REACH_OBSERVER.set(lambda event: request("reach", **asdict(event)))
    try:
        async with ObservedSQLiteStore(root / "journal.db", now=now) as store:
            session = Session(store, id="case")
            if (await session.state()).open_run_id:
                result = await agent.resume(session)
            else:
                result = await agent.run("Apply R1", session=session)
            request("done", text=result.text, status=result.status)
    finally:
        REACH_OBSERVER.reset(observer)


if __name__ == "__main__":
    asyncio.run(run(Path(sys.argv[1]), float(sys.argv[2]), sys.argv[3]))
