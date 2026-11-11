"""The examples and the documentation are part of the product: keep them working."""

from __future__ import annotations

import ast
import json
import re
import runpy
import sys
from pathlib import Path

import httpx
import pytest

from qagents import Agent, InMemoryStore
from qagents.models.testing import ScriptedModel, call

ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"^```python[^\n]*\n(.*?)^```", re.DOTALL | re.MULTILINE)


def python_blocks() -> list[tuple[str, str]]:
    pages = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    blocks = []
    for page in pages:
        for index, match in enumerate(FENCE.finditer(page.read_text(encoding="utf-8"))):
            if "--8<--" not in match.group(1):  # included files are checked as examples
                blocks.append((f"{page.relative_to(ROOT).as_posix()}#{index}", match.group(1)))
    return blocks


@pytest.mark.parametrize(("where", "source"), python_blocks(), ids=lambda value: value[:60])
def test_documentation_code_is_valid_python(where, source):
    # Snippets use top-level await for brevity, as in a notebook.
    compile(source, where, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


def test_offline_tour_runs(capsys):
    runpy.run_path(str(ROOT / "examples" / "offline_tour.py"), run_name="__main__")
    output = capsys.readouterr().out
    assert "status=paused" in output and "status=completed" in output
    assert "answered by billing" in output


async def test_custom_provider_example():
    module = runpy.run_path(str(ROOT / "examples" / "custom_provider.py"))
    assert await module["main"]() == "You said: hello"


def sse_events(body: str) -> list[dict]:
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]


async def test_fastapi_example_streams_and_resumes_after_approval():
    pytest.importorskip("fastapi")
    sys.path.insert(0, str(ROOT))
    try:
        from examples.serve_fastapi import build_app, send_email
    finally:
        sys.path.remove(str(ROOT))

    model = ScriptedModel(
        [call("send_email", to="maria@example.com", subject="Hello"), "The email is on its way."]
    )
    app = build_app(Agent(model, tools=[send_email]), InMemoryStore())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        chat = await client.post("/chat", json={"session_id": "s1", "message": "Email Maria"})
        assert chat.headers["content-type"].startswith("text/event-stream")
        events = sse_events(chat.text)
        assert [e["type"] for e in events] == [
            "run_started", "turn_started", "model_responded", "approval_required", "run_finished",
        ]  # fmt: skip
        assert events[-1]["result"]["status"] == "paused"

        pending = (await client.get("/sessions/s1/pending")).json()
        assert pending["inputs"] == []
        (waiting,) = pending["approvals"]
        assert waiting["name"] == "send_email" and waiting["arguments"]["to"] == "maria@example.com"

        approved = await client.post(
            "/resume", json={"session_id": "s1", "approvals": {waiting["tool_call_id"]: True}}
        )
        events = sse_events(approved.text)
        finished = events[-1]["result"]
        assert (
            finished["status"] == "completed" and finished["output"] == "The email is on its way."
        )
        assert "tool_call_finished" in [e["type"] for e in events]
        assert (await client.get("/sessions/s1/pending")).json() == {"approvals": [], "inputs": []}


async def test_fastapi_example_asks_the_user_and_resumes_with_the_answer():
    pytest.importorskip("fastapi")
    sys.path.insert(0, str(ROOT))
    try:
        from examples.serve_fastapi import ask_user, build_app, send_email
    finally:
        sys.path.remove(str(ROOT))

    model = ScriptedModel(
        [
            call("ask_user", question="What is Maria's email address?"),
            call("send_email", to="maria@example.com", subject="Hello"),
            "Sent.",
        ]
    )
    app = build_app(Agent(model, tools=[send_email, ask_user]), InMemoryStore())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        chat = await client.post("/chat", json={"session_id": "s2", "message": "Email Maria"})
        events = sse_events(chat.text)
        assert "input_required" in [e["type"] for e in events]
        assert events[-1]["result"]["status"] == "paused"

        pending = (await client.get("/sessions/s2/pending")).json()
        (question,) = pending["inputs"]
        assert pending["approvals"] == [] and question["name"] == "ask_user"

        answered = await client.post(
            "/resume",
            json={"session_id": "s2", "results": {question["tool_call_id"]: "maria@example.com"}},
        )
        events = sse_events(answered.text)
        assert events[-1]["result"]["status"] == "paused"  # now send_email waits for approval
        assert model.requests[1].messages[-1].tool_results[0].content == "maria@example.com"

        (waiting,) = (await client.get("/sessions/s2/pending")).json()["approvals"]
        done = await client.post(
            "/resume", json={"session_id": "s2", "approvals": {waiting["tool_call_id"]: True}}
        )
        assert sse_events(done.text)[-1]["result"]["output"] == "Sent."


async def test_a_limited_run_continues_on_its_session_without_a_new_prompt():
    from qagents import Limits, Session, tool

    @tool(read_only=True)
    def step(n: int) -> str:
        """Do one step."""
        return f"step {n} done"

    model = ScriptedModel([call("step", n=1), call("step", n=2), "All steps done."])
    agent = Agent(model, tools=[step])
    session = Session()
    first = await agent.run("Do the steps", session=session, limits=Limits(max_turns=1))
    assert (first.status, first.stop_reason) == ("limit_reached", "max_turns")
    second = await agent.run(session=session, limits=Limits(max_turns=5))
    assert second.output == "All steps done."
    assert [m.role for m in second.messages] == [
        "user", "assistant", "tool", "assistant", "tool", "assistant",
    ]  # fmt: skip
