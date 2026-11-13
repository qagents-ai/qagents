"""Approved preferences in durable, tenant-scoped session state, with metadata-only hooks.

Run offline: python -m examples.scoped_memory --database preferences.db
No external memory service or model is called. QAgent Gym has separate Mem0 adapters.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

from qagents import Agent, Approval, Hooks, Limits, RunContext, Session, SQLiteStore, tool
from qagents.models.testing import FunctionModel, ScriptedModel, call


@dataclass(frozen=True)
class Identity:
    """Trusted application identity, never taken from a model's tool arguments."""

    tenant_id: str
    user_id: str

    @property
    def session_id(self) -> str:
        # A structured tuple avoids collisions between ids containing delimiters.
        return "preferences:" + json.dumps([self.tenant_id, self.user_id], ensure_ascii=True)


@tool(requires_approval=True, idempotent=True)
def remember_preference(ctx: RunContext, key: str, value: str) -> str:
    """Store an approved preference for the current session."""
    ctx.state.setdefault("preferences", {})[key] = value
    return "Preference stored."


@tool(read_only=True)
def read_preference(ctx: RunContext, key: str) -> str:
    """Read a preference from the current session, or return unset."""
    return ctx.state.get("preferences", {}).get(key, "unset")


@dataclass
class RunAudit(Hooks):
    """Collect run metadata; no prompts, tool arguments or preference values."""

    rows: list[dict] = field(default_factory=list)

    async def on_run_end(self, ctx, result, error):
        usage = result.usage if result is not None else ctx.usage
        self.rows.append(
            {
                "run_id": ctx.run_id,
                "status": result.status if result is not None else "failed",
                "requests": usage.requests,
                "total_tokens": usage.total_tokens,
            }
        )


def preference_agent(model, audit: RunAudit) -> Agent:
    return Agent(
        model,
        tools=[remember_preference, read_preference],
        hooks=[audit],
        limits=Limits(max_turns=4, timeout_s=10),
    )


async def recall(store: SQLiteStore, identity: Identity, audit: RunAudit) -> str:
    def reply(request):
        if request.messages[-1].role == "tool":
            return request.messages[-1].tool_results[0].content
        return call("read_preference", key="language")

    result = await preference_agent(FunctionModel(reply), audit).run(
        "Read my language preference.", session=Session(store, id=identity.session_id)
    )
    if not result.completed:
        raise RuntimeError(f"Preference read stopped: {result.status}")
    return result.text


async def demo(database: Path) -> dict:
    identity = Identity("quentin", "alice")
    audit = RunAudit()
    model = ScriptedModel([call("remember_preference", key="language", value="el"), "Saved."])
    writer = preference_agent(model, audit)
    async with SQLiteStore(database) as store:
        session = Session(store, id=identity.session_id)
        paused = await writer.run("Remember that I prefer Greek.", session=session)
        # In an application, an authenticated person supplies this decision.
        # The offline demonstration intentionally approves its scripted request.
        pending = paused.pending_approvals[0]
        completed = await writer.resume(
            session, approvals={pending.tool_call_id: Approval.for_call(pending)}
        )
        if not completed.completed:
            raise RuntimeError(f"Preference write stopped: {completed.status}")
    async with SQLiteStore(database) as store:
        return {
            "same_user_after_reopen": await recall(store, identity, audit),
            "another_user": await recall(store, Identity("quentin", "bob"), audit),
            "another_tenant": await recall(store, Identity("other", "alice"), audit),
            "audit": audit.rows,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("preferences.db"))
    args = parser.parse_args()
    print(json.dumps(asyncio.run(demo(args.database)), indent=2))


if __name__ == "__main__":
    main()
