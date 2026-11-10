"""Serve an agent over HTTP with FastAPI.

    POST /chat                      start or continue a conversation; answers as server-sent events
    GET  /sessions/{id}/pending     tool calls waiting for a human
    POST /approve                   decide them; the run continues, again as server-sent events

Run it against QCloud (needs QCLOUD_API_KEY and QCLOUD_MODEL):

    uvicorn --factory examples.serve_fastapi:create_app

Authentication is left out: in a real service, check who is calling and that
the session id belongs to them before touching the session.
"""

import os
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from qagents import Agent, RunStream, Session, SessionStore, SQLiteStore, tool


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ApprovalRequest(BaseModel):
    session_id: str
    #: tool call id -> approved?
    approvals: dict[str, bool]


async def server_sent_events(stream: RunStream) -> AsyncIterator[str]:
    """One SSE message per run event. Closing the stream stops the run."""
    async with stream:
        async for event in stream:
            yield f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"


def build_app(agent: Agent, store: SessionStore) -> FastAPI:
    app = FastAPI(title="qagents example")

    @app.post("/chat")
    async def chat(body: ChatRequest) -> StreamingResponse:
        stream = agent.stream(body.message, session=Session(store, id=body.session_id))
        return StreamingResponse(server_sent_events(stream), media_type="text/event-stream")

    @app.get("/sessions/{session_id}/pending")
    async def pending(session_id: str) -> list[dict]:
        state = await Session(store, id=session_id).state()
        return [item.model_dump() for item in state.pending_approvals]

    @app.post("/approve")
    async def approve(body: ApprovalRequest) -> StreamingResponse:
        stream = agent.resume_stream(Session(store, id=body.session_id), approvals=body.approvals)
        return StreamingResponse(server_sent_events(stream), media_type="text/event-stream")

    return app


@tool(requires_approval=True)
def send_email(to: str, subject: str) -> str:
    """Send an email.

    Args:
        to: Recipient address.
        subject: Subject line.
    """
    return f"Email to {to} queued."


def create_app() -> FastAPI:
    agent = Agent(
        f"qcloud:{os.environ['QCLOUD_MODEL']}",
        instructions="You are a helpful assistant. Be brief.",
        tools=[send_email],
    )
    return build_app(agent, SQLiteStore("agents.db"))
