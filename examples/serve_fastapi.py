"""Serve an agent over HTTP with FastAPI, with approvals and questions to the user.

    POST /chat                      start or continue a conversation; answers as server-sent events
    GET  /sessions/{id}/pending     what the paused run waits for: approvals and inputs
    POST /resume                    send decisions and answers; the run continues, again as events

A run pauses for two reasons. ``send_email`` requires approval, so its calls wait in
``approvals``; answer them with ``{"approvals": {tool_call_id: true}}``. ``ask_user`` is a
deferred tool: the model asks the user a question and the call waits in ``inputs``; answer it
with ``{"results": {tool_call_id: "the answer"}}``. Both can go in one ``/resume``, and a
partial answer keeps the run paused with the rest.

Run it from the project root (the provider comes from QAGENTS_EXAMPLE_PROVIDER, default
anthropic, with its usual key; see examples/harness.py):

    uvicorn --factory examples.serve_fastapi:create_app

Authentication is left out: in a real service, check who is calling and that the session id
belongs to them before touching the session, and sign pauses with
``Permissions(approval_secret=...)`` so a stored pause cannot be forged.
"""

import os
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from qagents import Agent, Limits, RunStream, Session, SessionStore, SQLiteStore, tool
from qagents.toolkits import ask_user


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ResumeRequest(BaseModel):
    session_id: str
    #: tool call id -> approved?
    approvals: dict[str, bool] = Field(default_factory=dict)
    #: tool call id -> the result of a call that waits for input
    results: dict[str, Any] = Field(default_factory=dict)


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
    async def pending(session_id: str) -> dict[str, list[dict]]:
        state = await Session(store, id=session_id).state()
        return {
            "approvals": [item.model_dump() for item in state.pending_approvals],
            "inputs": [item.model_dump() for item in state.pending_inputs],
        }

    @app.post("/resume")
    async def resume(body: ResumeRequest) -> StreamingResponse:
        stream = agent.resume_stream(
            Session(store, id=body.session_id),
            approvals=body.approvals or None,
            results=body.results or None,
        )
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
    from examples.harness import make_model

    agent = Agent(
        make_model(os.environ.get("QAGENTS_EXAMPLE_PROVIDER", "anthropic")),
        instructions=(
            "You are a helpful assistant. Be brief. When you need a detail only the user "
            "knows, such as an email address, ask with ask_user instead of guessing."
        ),
        tools=[send_email, ask_user],
        limits=Limits(max_turns=10, max_cost_usd=0.50),
    )
    return build_app(agent, SQLiteStore("agents.db"))
