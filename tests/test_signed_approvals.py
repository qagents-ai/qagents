"""Signed pauses: approvals and results must carry the token of the pause they answer."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import uuid

import pytest

from qagents import Agent, Approval, Hooks, InMemoryStore, Permissions, Session, SQLiteStore, tool
from qagents.errors import ApprovalTokenError, ConfigurationError, DeferToolCall
from qagents.models.testing import ScriptedModel, call
from qagents.permissions import InputResult, approval_token, canonical_json

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
        session_id=session.id,
        run_id=paused.run_id,
        position=2,  # the prompt, then the assistant message with the call
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
        session_id=session.id,
        run_id=paused.run_id,
        position=2,
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
        session_id="user-1",
        run_id=paused.run_id,
        position=2,
        tool_call_id=waiting.tool_call_id,
        name="fetch_report",
        arguments={"report_id": "R-1"},
        kind="input",
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


# -- what the token binds (F55, F163) ----------------------------------------------------


async def test_a_signed_pause_copied_into_another_session_is_refused():
    log: list[str] = []
    agent = refund_agent(log)
    store = InMemoryStore()
    attacker = Session(store, id="tenant-attacker")
    pending = (await agent.run("Refund A-1", session=attacker)).pending_approvals[0]
    victim = Session(store, id="tenant-victim")
    await store.append(victim.id, await attacker.entries())  # a store writer copies the pause

    with pytest.raises(ApprovalTokenError, match="does not match"):
        await agent.resume(victim, approvals={pending.tool_call_id: Approval.for_call(pending)})
    assert log == []


async def test_a_copy_of_an_answered_pause_appended_again_is_refused():
    log: list[str] = []
    model = ScriptedModel(
        [call("refund", order="A-1", amount=15), call("refund", order="A-2", amount=1), "ok"]
    )
    agent = Agent(model, tools=[refund_tool(log)], permissions=Permissions(approval_secret=SECRET))
    session = Session()
    first = (await agent.run("Refund", session=session)).pending_approvals[0]
    second = await agent.resume(session, approvals={first.tool_call_id: Approval.for_call(first)})
    assert log == ["refund A-1 15.0"] and second.status == "paused"

    entries = await session.entries()
    assistant = next(
        e
        for e in entries
        if e.kind == "message"
        and e.data["message"]["parts"]
        and e.data["message"]["parts"][0].get("type") == "tool_call"
    )
    pause = next(e for e in entries if e.kind == "run_paused")
    await session.store.append(
        session.id,  # replay the first, already answered, pause in the open run
        [assistant.model_copy(update={"seq": 0}), pause.model_copy(update={"seq": 0})],
    )
    with pytest.raises(ApprovalTokenError):
        await agent.resume(session, approvals={first.tool_call_id: Approval.for_call(first)})
    assert log == ["refund A-1 15.0"]


async def test_changed_metadata_of_a_deferred_call_is_detected():
    @tool
    def start_job(name: str) -> str:
        """Start a job."""
        raise DeferToolCall("waiting for the job", metadata={"job_id": "J1"})

    model = ScriptedModel([call("start_job", name="report"), "Ready."])
    agent = Agent(model, tools=[start_job], permissions=Permissions(approval_secret=SECRET))
    store = InMemoryStore()
    session = Session(store, id="s")
    waiting = (await agent.run("run the job", session=session)).pending_inputs[0]
    for entry in store._sessions["s"]:  # someone with write access points it at another job
        if entry.kind == "tool_deferred":
            entry.data["input"]["metadata"]["job_id"] = "J2"
        if entry.kind == "run_paused":
            entry.data["inputs"][0]["metadata"]["job_id"] = "J2"
    with pytest.raises(ApprovalTokenError, match="journal was changed"):
        await agent.resume(
            session, results={waiting.tool_call_id: InputResult.for_call(waiting, 3)}
        )


@pytest.mark.parametrize("token", ["\u00e9", "\udcff", "0" * 63 + "\u00e9"])
async def test_a_non_ascii_token_is_refused_with_the_documented_error(token):
    log: list[str] = []
    agent = refund_agent(log)
    session = Session()
    pending = (await agent.run("Refund A-1", session=session)).pending_approvals[0]
    with pytest.raises(ApprovalTokenError):
        await agent.resume(
            session, approvals={pending.tool_call_id: Approval(approved=True, token=token)}
        )

    model = ScriptedModel([call("fetch_report", report_id="R-1")])
    deferred = Agent(model, tools=[lookup_tool()], permissions=Permissions(approval_secret=SECRET))
    other = Session()
    waiting = (await deferred.run("Get R-1", session=other)).pending_inputs[0]
    with pytest.raises(ApprovalTokenError):
        await deferred.resume(
            other, results={waiting.tool_call_id: InputResult(value=1, token=token)}
        )
    assert log == []


async def test_a_refused_token_never_reaches_on_run_end():
    events: list[str] = []

    class Tracer(Hooks):
        async def on_run_start(self, ctx):
            events.append("start")

        async def on_run_end(self, ctx, result, error):
            events.append("end")

    log: list[str] = []
    agent = refund_agent(log)
    session = Session()
    pending = (await agent.run("Refund A-1", session=session)).pending_approvals[0]
    agent.hooks.append(Tracer())
    with pytest.raises(ApprovalTokenError):
        await agent.resume(session, approvals={pending.tool_call_id: True})
    assert events == []  # no on_run_end without its on_run_start


# -- the canonical form (F165, F181, F121) -----------------------------------------------


def test_the_token_payload_is_rfc_8785_canonical_json():
    assert canonical_json({"b": [1.0, -0.0, 1e16, 1e-7, 0.5], "a": "é", "c": None}) == (
        '{"a":"é","b":[1,0,10000000000000000,1e-7,0.5],"c":null}'
    )
    assert canonical_json({"\u20ac": 1, "\U0001f600": 2, "z": 3}) == '{"z":3,"€":1,"😀":2}'
    common = dict(session_id="s", run_id="r", position=2, tool_call_id="c", name="refund")
    as_float = approval_token(SECRET, arguments={"amount": 15.0, "big": 1e16}, **common)
    as_int = approval_token(SECRET, arguments={"amount": 15, "big": 10**16}, **common)
    assert as_float == as_int  # what a jsonb store or a JavaScript service gives back
    for bad in (float("nan"), float("inf"), "half \ud83d"):
        with pytest.raises(ValueError):
            approval_token(SECRET, arguments={"x": bad}, **common)


async def test_a_call_whose_arguments_cannot_be_signed_is_refused_not_crashed():
    log: list[str] = []
    model = ScriptedModel([call("refund", order="A-\ud83d", amount=15), "I could not do it."])
    agent = Agent(model, tools=[refund_tool(log)], permissions=Permissions(approval_secret=SECRET))
    result = await agent.run("Refund A-1")
    assert result.status == "completed" and log == []
    refused = result.messages[2].tool_results[0]
    assert refused.is_error and "cannot be written as standard JSON" in refused.content


async def test_a_call_with_an_integer_beyond_a_double_is_refused_not_crashed():
    with pytest.raises(ValueError):
        canonical_json({"n": 10**400})  # float(10**400) raises OverflowError
    assert canonical_json(2**53) == str(2**53) and canonical_json(10**16) == "10000000000000000"
    for beyond in (2**53 + 1, -(2**53) - 1, 1234567890123456789):
        with pytest.raises(ValueError):  # a double would sign 1234567890123456829 the same
            canonical_json({"message_id": beyond})
    log: list[str] = []

    @tool(requires_approval=True)
    def send(n: int) -> str:
        """Send."""
        log.append(f"send {n}")
        return "sent"

    model = ScriptedModel([call("send", n=10**400), "I could not send it."])
    agent = Agent(model, tools=[send], permissions=Permissions(approval_secret=SECRET))
    session = Session()
    result = await agent.run("Send", session=session)
    assert result.status == "completed" and log == []
    refused = result.messages[2].tool_results[0]
    assert refused.is_error and "cannot be written as standard JSON" in refused.content
    assert "run_failed" not in [entry.kind for entry in await session.entries()]


@pytest.mark.skipif(
    not os.environ.get("QAGENTS_TEST_POSTGRES_DSN"),
    reason="set QAGENTS_TEST_POSTGRES_DSN to test PostgresStore",
)
def test_a_signed_pause_with_floats_jsonb_rewrites_resumes_on_postgres():
    dsn = os.environ["QAGENTS_TEST_POSTGRES_DSN"]
    schema = f"qagents_test_{uuid.uuid4().hex[:12]}"

    async def check() -> None:
        import psycopg

        from qagents.stores.postgres import PostgresStore

        store = PostgresStore(dsn, schema=schema, pool_size=2)
        log: list[str] = []
        model = ScriptedModel([call("refund", order="A-1", amount=1e16), "Refund issued."])
        agent = Agent(
            model, tools=[refund_tool(log)], permissions=Permissions(approval_secret=SECRET)
        )
        try:
            session = Session(store, id="user-1")
            pending = (await agent.run("Refund A-1", session=session)).pending_approvals[0]
            # jsonb gives 1e16 back as the integer 10000000000000000
            assert (await session.state()).pending_approvals[0].arguments["amount"] == 10**16
            approval = Approval.for_call(pending)
            done = await agent.resume(session, approvals={pending.tool_call_id: approval})
            assert done.output == "Refund issued." and len(log) == 1
        finally:
            await store.aclose()
            async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as connection:
                await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')

    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
        runner.run(check())
