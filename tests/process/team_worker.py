"""Run one real agent against a mailbox shared with other test processes."""

import asyncio
import json
import os
import sys
from pathlib import Path

from qagents import Agent, Hooks, Session, SQLiteStore, TeamMailbox
from qagents.models.testing import ScriptedModel, call

MEMBERS = ["lead", "researcher", "reviewer"]


def send(recipient, text):
    return call("send_message", recipient=recipient, text=text)


def receive():
    return call("receive_messages", wait_s=15)


class Boundary(Hooks):
    def __init__(self, report_wait, cut):
        self.report_wait = report_wait
        self.cut = cut

    async def before_tool(self, ctx, call):
        if self.report_wait and call.name == "receive_messages":
            print(json.dumps({"pid": os.getpid(), "ready": True}), flush=True)

    async def after_tool(self, ctx, call, result):
        if self.cut and not result.is_error:
            print(
                json.dumps({"pid": os.getpid(), "tool": call.name, "content": result.content}),
                flush=True,
            )
            await asyncio.to_thread(sys.stdin.buffer.readline)


async def run(directory, role, action, now):
    mailbox = TeamMailbox(MEMBERS, db_path=directory / "mailbox.db")
    endpoint = mailbox.endpoint(role)
    if action == "exchange":
        replies = {
            "lead": [send("researcher", "Collect the evidence"), receive(), "Lead done"],
            "researcher": [receive(), send("reviewer", "Evidence A"), "Research done"],
            "reviewer": [receive(), send("lead", "Evidence A verified"), "Review done"],
        }[role]
    elif action == "cut-send":
        replies = [send("researcher", "Survives termination")]
    elif action == "cut-receive":
        replies = [receive()]
    else:
        replies = ["Resumed"]
    agent = Agent(
        ScriptedModel(replies),
        name=role,
        tools=endpoint.tools(),
        hooks=[Boundary(action == "exchange" and role != "lead", action.startswith("cut-"))],
    )
    async with SQLiteStore(directory / f"{role}.db", clock=lambda: now) as store:
        session = Session(store, id=role)
        if action == "resume":
            result = await agent.resume(session)
        else:
            result = await agent.run("Coordinate the task", session=session)
    assert result.completed
    results = [part for message in result.messages for part in message.tool_results]
    assert all(not part.is_error for part in results), results
    print(
        json.dumps(
            {
                "pid": os.getpid(),
                "received": [
                    item
                    for part in results
                    if part.name == "receive_messages"
                    for item in json.loads(part.content)
                ],
                "sent": [
                    json.loads(part.content) for part in results if part.name == "send_message"
                ],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    path, role, action, now = sys.argv[1:]
    asyncio.run(run(Path(path), role, action, float(now)))
