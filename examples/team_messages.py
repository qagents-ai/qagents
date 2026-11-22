"""Send evidence from a researcher to a reviewer and back to the coordinator."""

import asyncio

from qagents import Agent, RunContext, TeamMailbox, tool
from qagents.models.testing import ScriptedModel, call


async def main() -> str:
    mailbox = TeamMailbox(["coordinator", "researcher", "reviewer"])
    coordinator = mailbox.endpoint("coordinator")
    researcher = Agent(
        ScriptedModel(
            [
                call("receive_messages", wait_s=3),
                call("send_message", recipient="reviewer", text="Journal shows one refund."),
                "Evidence sent.",
            ]
        ),
        name="researcher",
        tools=mailbox.endpoint("researcher").tools(),
    )
    reviewer = Agent(
        ScriptedModel(
            [
                call("receive_messages", wait_s=3),
                call("send_message", recipient="coordinator", text="Verified one refund."),
                "Review complete.",
            ]
        ),
        name="reviewer",
        tools=mailbox.endpoint("reviewer").tools(),
    )

    @tool
    async def coordinate(ctx: RunContext) -> str:
        """Ask both specialists to review a test payment."""
        async with asyncio.TaskGroup() as group:
            group.create_task(researcher.run("Collect the evidence.", parent=ctx))
            group.create_task(reviewer.run("Review the evidence.", parent=ctx))
        return "Both specialists finished."

    lead = Agent(
        ScriptedModel(
            [
                call("send_message", recipient="researcher", text="Check the destination ledger."),
                call("coordinate"),
                call("receive_messages"),
                "The specialist review is complete.",
            ]
        ),
        name="coordinator",
        tools=[coordinate, *coordinator.tools()],
    )
    result = await lead.run("Check the refund.")
    (answer,) = [
        part.content
        for message in result.messages
        for part in message.tool_results
        if part.name == "receive_messages"
    ]
    return answer


if __name__ == "__main__":
    print(asyncio.run(main()))
