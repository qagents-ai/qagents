"""A tour of qagents that needs no network and no API key.

A scripted model stands in for the LLM, so you can see the loop, an approval
pause, a resume from the journal and a handoff exactly as they happen:

    python examples/offline_tour.py
"""

from __future__ import annotations

import asyncio

from qagents import Agent, InMemoryStore, Session, tool
from qagents.models.testing import ScriptedModel, call


@tool(read_only=True)
def get_invoice(number: str) -> dict:
    """Look up an invoice."""
    return {"number": number, "total_eur": 120.0, "status": "unpaid"}


@tool(requires_approval=True)
def send_reminder(number: str) -> str:
    """Email a payment reminder for an invoice."""
    return f"Reminder for {number} sent."


async def main() -> None:
    store = InMemoryStore()

    print("1) A tool loop, streamed")
    model = ScriptedModel(
        [call("get_invoice", number="INV-7"), "Invoice INV-7 is unpaid: 120 EUR."]
    )
    async for event in Agent(model, tools=[get_invoice]).stream("What about invoice INV-7?"):
        if event.type == "tool_call_finished":
            print(f"   tool {event.call.name} -> {event.result.content}")
        elif event.type == "run_finished":
            print(f"   answer: {event.result.output}")

    print("2) A pause for approval, then a resume by a different agent object")
    session = Session(store, id="billing-run")
    model = ScriptedModel([call("send_reminder", number="INV-7")])
    paused = await Agent(model, tools=[send_reminder]).run(
        "Remind them about INV-7", session=session
    )
    pending = paused.pending_approvals[0]
    print(f"   status={paused.status}, waiting for: {pending.name}({pending.arguments})")

    later = Agent(ScriptedModel(["The reminder is on its way."]), tools=[send_reminder])
    done = await later.resume(
        Session(store, id="billing-run"), approvals={pending.tool_call_id: True}
    )
    print(f"   status={done.status}, answer: {done.output}")
    print(f"   journal: {[entry.kind for entry in await session.entries()]}")

    print("3) A handoff to a specialist")
    billing = Agent(
        ScriptedModel(["Your invoice total is 120 EUR."]),
        name="billing",
        description="Answers questions about invoices.",
    )
    triage = Agent(ScriptedModel([call("transfer_to_billing")]), name="triage", handoffs=[billing])
    result = await triage.run("How much do I owe?")
    print(f"   answered by {result.agent}: {result.output}")


if __name__ == "__main__":
    asyncio.run(main())
