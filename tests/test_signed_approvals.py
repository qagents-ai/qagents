"""Signed pauses: approvals and results must carry the token of the pause they answer."""

from __future__ import annotations

import json
import sqlite3

import pytest

from qagents import Agent, Approval, Permissions, Session, SQLiteStore, tool
from qagents.errors import ApprovalTokenError, ConfigurationError
from qagents.models.testing import ScriptedModel, call
from qagents.permissions import InputResult, approval_token

SECRET = "s3cret-for-tests-only-0123456789"


def refund_tool(log: list[str]):
    @tool(requires_approval=True)
    def refund(order: str, amount: float) -> str:
        """Refund an order."""
        log.append(f"refund {order} {amount}")
        return f"refunded {amount}"

    return refund


def lookup_tool():
    @tool(deferred=True)
    def fetch_report(report_id: str) -> str:
        """Fetch a report that another system produces."""
        raise AssertionError("a deferred tool never runs")

    return fetch_report


def refund_agent(log: list[str], *, secret: str | None = SECRET) -> Agent:
    return Agent(
        ScriptedModel([call("refund", order="A-1", amount=15), "Refund issued."]),
        tools=[refund_tool(log)],
        permissions=Permissions(approval_secret=secret),
    )


def tamper(path, kind: str, change) -> None:
    """Rewrite the data of every ``kind`` entry, as someone with database access could."""
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            "SELECT seq, data FROM entries WHERE kind = ?", (kind,)
        ).fetchall()
        for seq, data in rows:
            connection.execute(
                "UPDATE entries SET data = ? WHERE seq = ?",
                (json.dumps(change(json.loads(data))), seq),
            )
        connection.commit()
    finally:
        connection.close()


async def test_without_a_secret_nothing_changes():
    log: list[str] = []
    agent = refund_agent(log, secret=None)
    session = Session()
    paused = await agent.run("Refund A-1", session=session)
    pending = paused.pending_approvals[0]
    assert pending.token is None
    done = await agent.resume(session, approvals={pending.tool_call_id: True})
    assert done.output == "Refund issued." and log == ["refund A-1 15.0"]


async def test_a_valid_token_lets_the_approval_through():
    log: list[str] = []
    agent = refund_agent(log)
    session = Session()
    paused = await agent.run("Refund A-1", session=session)

    pending = paused.pending_approvals[0]
    assert pending.token == approval_token(
        SECRET,
        run_id=paused.run_id,
        tool_call_id=pending.tool_call_id,
        name="refund",
        arguments={"order": "A-1", "amount": 15},
    )
    assert (await session.state()).pending_approvals[0].token == pending.token  # journaled

    done = await agent.resume(session, approvals={pending.tool_call_id: Approval.for_call(pending)})
    assert done.output == "Refund issued." and log == ["refund A-1 15.0"]


async def test_a_missing_or_wrong_token_is_refused_before_anything_runs():
    log: list[str] = []
    agent = refund_agent(log)
    session = Session()
    paused = await agent.run("Refund A-1", session=session)
    pending = paused.pending_approvals[0]
    before = len(await session.entries())

    with pytest.raises(
        ApprovalTokenError, match=r"needs the pause's token. Pass Approval.for_call"
    ):
        await agent.resume(session, approvals={pending.tool_call_id: True})
    with pytest.raises(ApprovalTokenError, match="does not match the call recorded"):
        await agent.resume(
            session, approvals={pending.tool_call_id: Approval(approved=True, token="0" * 64)}
        )
    other_secret = Permissions(approval_secret="another-secret-of-32-characters!")
    forged = other_secret.sign(
        run_id=paused.run_id,
        tool_call_id=pending.tool_call_id,
        name=pending.name,
        arguments=pending.arguments,
    )
    with pytest.raises(ApprovalTokenError):
        await agent.resume(
            session, approvals={pending.tool_call_id: Approval(approved=True, token=forged)}
        )

    assert log == [] and len(await session.entries()) == before  # nothing ran or was written
    done = await agent.resume(session, approvals={pending.tool_call_id: Approval.for_call(pending)})
    assert done.completed and log == ["refund A-1 15.0"]


async def test_a_declined_approval_needs_the_token_too():
    log: list[str] = []
    agent = refund_agent(log)
    session = Session()
    pending = (await agent.run("Refund A-1", session=session)).pending_approvals[0]
    with pytest.raises(ApprovalTokenError):
        await agent.resume(session, approvals={pending.tool_call_id: False})
    declined = Approval.for_call(pending, approved=False, reason="Not this one.")
    result = await agent.resume(session, approvals={pending.tool_call_id: declined})
    assert log == []
    assert result.messages[2].tool_results[0].content == "'refund' was not run: Not this one."


async def test_tampered_journal_arguments_are_detected(tmp_path):
    path = tmp_path / "agents.db"
    log: list[str] = []
    agent = refund_agent(log)
    session = Session(SQLiteStore(path), id="user-1")
    paused = await agent.run("Refund A-1", session=session)
    pending = paused.pending_approvals[0]
    await session.store.aclose()

    def raise_amount(data: dict) -> dict:
        for part in data["message"]["parts"]:
            if part["type"] == "tool_call":
                part["arguments"]["amount"] = 15000
        return data

    tamper(path, "message", raise_amount)  # the call that would run
    session = Session(SQLiteStore(path), id="user-1")
    with pytest.raises(ApprovalTokenError, match="journal was changed"):
        await agent.resume(session, approvals={pending.tool_call_id: Approval.for_call(pending)})
    assert log == []
    await session.store.aclose()


async def test_a_tampered_pause_record_is_detected(tmp_path):
    path = tmp_path / "agents.db"
    log: list[str] = []
    agent = refund_agent(log)
    session = Session(SQLiteStore(path), id="user-1")
    pending = (await agent.run("Refund A-1", session=session)).pending_approvals[0]
    await session.store.aclose()

    def hide_amount(data: dict) -> dict:  # what a reviewer would be shown
        data["pending"][0]["arguments"]["amount"] = 1
        return data

    tamper(path, "run_paused", hide_amount)
    session = Session(SQLiteStore(path), id="user-1")
    with pytest.raises(ApprovalTokenError):
        await agent.resume(session, approvals={pending.tool_call_id: Approval.for_call(pending)})
    assert log == []
    await session.store.aclose()


async def test_results_for_signed_inputs_need_the_token(tmp_path):
    path = tmp_path / "agents.db"
    model = ScriptedModel([call("fetch_report", report_id="R-1"), "Ready.", "Ready again."])
    agent = Agent(model, tools=[lookup_tool()], permissions=Permissions(approval_secret=SECRET))
    session = Session(SQLiteStore(path), id="user-1")
    paused = await agent.run("Get R-1", session=session)
    waiting = paused.pending_inputs[0]
    assert waiting.token == approval_token(
        SECRET,
        run_id=paused.run_id,
        tool_call_id=waiting.tool_call_id,
        name="fetch_report",
        arguments={"report_id": "R-1"},
    )

    with pytest.raises(ApprovalTokenError, match="InputResult.for_call"):
        await agent.resume(session, results={waiting.tool_call_id: "rows: 3"})
    with pytest.raises(ApprovalTokenError, match="does not match"):
        await agent.resume(
            session, results={waiting.tool_call_id: InputResult(value="rows: 3", token="bad")}
        )

    done = await agent.resume(
        session, results={waiting.tool_call_id: InputResult.for_call(waiting, "rows: 3")}
    )
    assert done.output == "Ready."
    assert model.requests[1].messages[-1].tool_results[0].content == "rows: 3"
    await session.store.aclose()


async def test_a_token_and_value_mapping_is_read_as_a_signed_result():
    model = ScriptedModel([call("fetch_report", report_id="R-1"), "Ready."])
    agent = Agent(model, tools=[lookup_tool()], permissions=Permissions(approval_secret=SECRET))
    session = Session()
    waiting = (await agent.run("Get R-1", session=session)).pending_inputs[0]
    answer = {"token": waiting.token, "value": {"rows": 3}}  # as it arrives in a JSON body
    done = await agent.resume(session, results={waiting.tool_call_id: answer})
    assert done.output == "Ready."
    assert model.requests[1].messages[-1].tool_results[0].content == '{"rows": 3}'


async def test_tampered_input_arguments_are_detected(tmp_path):
    path = tmp_path / "agents.db"
    model = ScriptedModel([call("fetch_report", report_id="R-1"), "Ready."])
    agent = Agent(model, tools=[lookup_tool()], permissions=Permissions(approval_secret=SECRET))
    session = Session(SQLiteStore(path), id="user-1")
    waiting = (await agent.run("Get R-1", session=session)).pending_inputs[0]
    await session.store.aclose()

    def change_report(data: dict) -> dict:
        for waiting_input in [data["input"]] if "input" in data else data["inputs"]:
            waiting_input["arguments"]["report_id"] = "R-2"
        return data

    tamper(path, "tool_deferred", change_report)
    tamper(path, "run_paused", change_report)
    session = Session(SQLiteStore(path), id="user-1")
    assert (await session.state()).pending_inputs[0].arguments == {"report_id": "R-2"}
    with pytest.raises(ApprovalTokenError, match="journal was changed"):
        await agent.resume(
            session, results={waiting.tool_call_id: InputResult.for_call(waiting, 3)}
        )
    assert len(model.requests) == 1
    await session.store.aclose()


async def test_without_a_secret_a_token_value_mapping_is_an_ordinary_value():
    model = ScriptedModel([call("fetch_report", report_id="R-1"), "Ready."])
    agent = Agent(model, tools=[lookup_tool()])
    session = Session()
    waiting = (await agent.run("Get R-1", session=session)).pending_inputs[0]
    assert waiting.token is None
    await agent.resume(session, results={waiting.tool_call_id: {"token": "t", "value": 1}})
    content = model.requests[1].messages[-1].tool_results[0].content
    assert json.loads(content) == {"token": "t", "value": 1}


def test_a_short_or_wrongly_typed_secret_fails_early():
    with pytest.raises(ConfigurationError, match="at least 16 random bytes"):
        Permissions(approval_secret="short")
    with pytest.raises(ConfigurationError, match="must be str or bytes"):
        Permissions(approval_secret=12345)  # type: ignore[arg-type]
    assert Permissions(approval_secret=b"\x00" * 32).signs_pauses
    assert not Permissions().signs_pauses
