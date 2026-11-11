"""PostgresStore, and the agent's durable behaviour on every store.

The generic tests run against InMemoryStore, SQLiteStore and PostgresStore.
The Postgres cases need a server: set QAGENTS_TEST_POSTGRES_DSN, for example to
postgresql://postgres:qagents@127.0.0.1:55432/qagents for

    docker run -d --rm --name qagents-test-pg -e POSTGRES_PASSWORD=qagents \\
        -e POSTGRES_DB=qagents -p 55432:5432 postgres:17-alpine

Without it they are skipped. Each test uses its own schema and drops it.
Tests are plain functions that drive a selector event loop, which psycopg's
async mode needs on Windows.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from collections.abc import Awaitable, Callable

import pytest

from qagents import Agent, RunContext, Session, SQLiteStore, tool
from qagents.errors import ConfigurationError, SessionError
from qagents.models.testing import ScriptedModel, call
from qagents.session import Entry, InMemoryStore, SessionStore

DSN = os.environ.get("QAGENTS_TEST_POSTGRES_DSN")
needs_postgres = pytest.mark.skipif(
    not DSN, reason="set QAGENTS_TEST_POSTGRES_DSN to test PostgresStore"
)
BACKENDS = ["memory", "sqlite", pytest.param("postgres", marks=needs_postgres)]


def run(coroutine: Awaitable[None]) -> None:
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
        runner.run(coroutine)  # type: ignore[arg-type]


def new_schema() -> str:
    return f"qagents_test_{uuid.uuid4().hex[:12]}"


def drop_schemas(*schemas: str) -> None:
    async def drop() -> None:
        import psycopg

        async with await psycopg.AsyncConnection.connect(DSN, autocommit=True) as connection:
            for schema in schemas:
                await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')

    run(drop())


class Backend:
    """Makes store instances over one backing database, like separate processes would."""

    def __init__(self, kind: str, path: str) -> None:
        self.kind = kind
        self.path = path
        self.schema = new_schema()
        self._memory: InMemoryStore | None = None
        self._opened: list[SessionStore] = []

    def store(self) -> SessionStore:
        if self.kind == "memory":
            self._memory = self._memory or InMemoryStore()
            return self._memory
        if self.kind == "sqlite":
            store: SessionStore = SQLiteStore(self.path)
        else:
            from qagents.stores.postgres import PostgresStore

            store = PostgresStore(DSN, schema=self.schema, pool_size=4)
        self._opened.append(store)
        return store

    def run(self, test: Callable[[], Awaitable[None]]) -> None:
        async def main() -> None:
            try:
                await test()
            finally:
                for store in self._opened:
                    await store.aclose()
                self._opened.clear()

        run(main())


@pytest.fixture(params=BACKENDS)
def backend(request: pytest.FixtureRequest, tmp_path) -> Backend:
    made = Backend(request.param, str(tmp_path / "agents.db"))
    yield made
    if made.kind == "postgres":
        drop_schemas(made.schema)


# -- the agent on every store ------------------------------------------------------------


def test_a_conversation_survives_a_new_store_instance(backend):
    async def check() -> None:
        first = await Agent(ScriptedModel(["Noted."])).run(
            "My name is Maria.", session=Session(backend.store(), id="user-1")
        )
        assert first.status == "completed"
        model = ScriptedModel(["You are Maria."])
        await Agent(model).run("What is my name?", session=Session(backend.store(), id="user-1"))
        assert [m.text for m in model.requests[0].messages] == [
            "My name is Maria.",
            "Noted.",
            "What is my name?",
        ]
        entries = await backend.store().load("user-1")
        assert [entry.seq for entry in entries] == list(range(1, 9))
        assert [entry.kind for entry in entries].count("run_finished") == 2

    backend.run(check)


def test_resume_after_a_crash_never_repeats_a_finished_tool(backend):
    async def check() -> None:
        log: list[str] = []
        started, gate = asyncio.Event(), asyncio.Event()

        def tools():
            @tool
            def charge_card(amount: int) -> str:
                """Charge the customer's card."""
                log.append(f"charge {amount}")
                return "charged"

            @tool
            async def send_receipt(to: str) -> str:
                """Email the receipt."""
                log.append(f"receipt {to}")
                started.set()
                await gate.wait()
                return "sent"

            return [charge_card, send_receipt]

        model = ScriptedModel([[call("charge_card", amount=40), call("send_receipt", to="m@x.gr")]])
        task = asyncio.create_task(
            Agent(model, tools=tools()).run("charge", session=Session(backend.store(), id="o"))
        )
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()  # the process dies while the receipt is being sent
        with pytest.raises(asyncio.CancelledError):
            await task

        gate.set()
        model = ScriptedModel(["Charged; the receipt is unconfirmed."])
        result = await Agent(model, tools=tools()).resume(Session(backend.store(), id="o"))
        assert result.status == "completed"
        assert log == ["charge 40", "receipt m@x.gr"]
        charged, receipt = result.messages[2].tool_results
        assert charged.content == "charged" and not charged.is_error
        assert receipt.is_error and "outcome is unknown" in receipt.content

    backend.run(check)


def test_an_approval_pause_resumes_from_another_store_instance(backend):
    async def check() -> None:
        refunds: list[str] = []

        @tool(requires_approval=True)
        def refund(ctx: RunContext, order: str) -> str:
            """Refund an order."""
            refunds.append(order)
            ctx.state["refunded"] = order
            return "refunded"

        model = ScriptedModel([call("refund", order="A-1"), "Refunded A-1."])
        agent = Agent(model, tools=[refund])
        paused = await agent.run("Refund A-1", session=Session(backend.store(), id="r"))
        assert paused.status == "paused" and refunds == []

        state = await Session(backend.store(), id="r").state()
        assert [item.tool_call_id for item in state.pending_approvals] == [
            paused.pending_approvals[0].tool_call_id
        ]
        done = await agent.resume(
            Session(backend.store(), id="r"),
            approvals={paused.pending_approvals[0].tool_call_id: True},
        )
        assert done.status == "completed" and refunds == ["A-1"]
        assert (await Session(backend.store(), id="r").state()).values == {"refunded": "A-1"}

    backend.run(check)


def test_a_deferred_input_is_answered_later(backend):
    async def check() -> None:
        @tool(deferred=True)
        def ask_manager(question: str) -> str:
            """Ask the manager."""
            raise AssertionError("never executed")

        model = ScriptedModel([call("ask_manager", question="Discount?"), "Approved 10%."])
        agent = Agent(model, tools=[ask_manager])
        paused = await agent.run("Can we discount?", session=Session(backend.store(), id="q"))
        waiting = paused.pending_inputs[0]
        assert (await backend.store().status("q")).status == "paused"
        done = await agent.resume(
            Session(backend.store(), id="q"), results={waiting.tool_call_id: "yes, 10%"}
        )
        assert done.text == "Approved 10%."
        assert model.requests[1].messages[-1].tool_results[0].content == "yes, 10%"

    backend.run(check)


# -- PostgresStore specifics -------------------------------------------------------------


def test_invalid_arguments_fail_before_connecting():
    from qagents.stores.postgres import PostgresStore

    with pytest.raises(ConfigurationError, match="plain identifier"):
        PostgresStore("postgresql://localhost/x", schema="bad; DROP TABLE x")
    with pytest.raises(ConfigurationError, match="pool_size"):
        PostgresStore("postgresql://localhost/x", pool_size=0)


@needs_postgres
def test_concurrent_writers_keep_seq_gap_free_and_batches_whole():
    from qagents.stores.postgres import PostgresStore

    schema = new_schema()

    async def check() -> None:
        stores = [PostgresStore(DSN, schema=schema, pool_size=5) for _ in range(3)]
        try:

            async def writer(index: int) -> None:
                store = stores[index % 3]
                for batch in range(5):
                    await store.append(
                        "busy",
                        [
                            Entry(kind="note", data={"writer": index, "batch": batch, "n": n})
                            for n in range(3)
                        ],
                    )

            await asyncio.gather(*[writer(index) for index in range(8)])
            entries = await stores[0].load("busy")
            assert [entry.seq for entry in entries] == list(range(1, 121))
            for start in range(0, 120, 3):  # each append is contiguous and in order
                batch = [entry.data for entry in entries[start : start + 3]]
                assert [item["n"] for item in batch] == [0, 1, 2]
                assert len({(item["writer"], item["batch"]) for item in batch}) == 1
            for index in range(8):  # one writer's batches keep their order
                mine = [e.data["batch"] for e in entries if e.data["writer"] == index]
                assert mine == sorted(mine)
        finally:
            for store in stores:
                await store.aclose()

    try:
        run(check())
    finally:
        drop_schemas(schema)


@needs_postgres
def test_tables_are_created_once_by_workers_starting_together():
    from qagents.stores.postgres import PostgresStore

    schema = new_schema()

    async def check() -> None:
        stores = [PostgresStore(DSN, schema=schema, pool_size=1) for _ in range(4)]
        try:
            await asyncio.gather(*[store.load("x") for store in stores])
            await stores[0].append("x", [Entry(kind="note")])
            assert len(await stores[3].load("x")) == 1
        finally:
            for store in stores:
                await store.aclose()

    try:
        run(check())
    finally:
        drop_schemas(schema)


@needs_postgres
def test_schemas_keep_applications_apart_and_a_newer_schema_is_refused():
    from qagents.stores.postgres import PostgresStore

    one, two = new_schema(), new_schema()

    async def check() -> None:
        first = PostgresStore(DSN, schema=one)
        second = PostgresStore(DSN, schema=two)
        try:
            await first.append("s", [Entry(kind="note", data={"app": 1})])
            assert await second.load("s") == []

            import psycopg

            async with await psycopg.AsyncConnection.connect(DSN, autocommit=True) as connection:
                await connection.execute(f'UPDATE "{one}".schema_version SET version = 99')
            newer = PostgresStore(DSN, schema=one)
            with pytest.raises(SessionError, match="newer qagents .*schema version 99"):
                await newer.load("s")
            await newer.aclose()
        finally:
            await first.aclose()
            await second.aclose()

    try:
        run(check())
    finally:
        drop_schemas(one, two)


@needs_postgres
def test_a_nul_character_is_refused_with_advice_and_nothing_is_written():
    from qagents.stores.postgres import PostgresStore

    schema = new_schema()

    async def check() -> None:
        store = PostgresStore(DSN, schema=schema)
        try:
            await store.append("n", [Entry(kind="note", data={"ok": True})])
            bad = [Entry(kind="note"), Entry(kind="tool_result", data={"text": "a\x00b"})]
            with pytest.raises(SessionError, match=r"NUL character .*note, tool_result.*base64"):
                await store.append("n", bad)
            assert [entry.data for entry in await store.load("n")] == [{"ok": True}]
            await store.append("n", [Entry(kind="note")])
            assert [entry.seq for entry in await store.load("n")] == [1, 2]  # still gap-free
        finally:
            await store.aclose()

    try:
        run(check())
    finally:
        drop_schemas(schema)


@needs_postgres
def test_data_round_trips_through_jsonb_and_the_pool_reopens_after_aclose():
    from qagents.stores.postgres import PostgresStore

    schema = new_schema()

    async def check() -> None:
        store = PostgresStore(DSN, schema=schema)
        data = {
            "text": "Καλημέρα ✓",
            "n": 1.5,
            "big": 2**53,
            "list": [None, True, {"z": 1, "a": 2}],
        }
        await store.append("j", [Entry(kind="note", data=data, run_id="r", ts=123.25)])
        await store.aclose()
        (entry,) = await store.load("j")  # opens a new pool
        assert entry.data == data and (entry.ts, entry.run_id, entry.seq) == (123.25, "r", 1)
        assert "postgresql" not in repr(store) and "qagents" in repr(store)
        await store.aclose()

    try:
        run(check())
    finally:
        drop_schemas(schema)


@needs_postgres
def test_an_unreachable_server_fails_with_a_clear_error():
    from qagents.stores.postgres import PostgresStore

    async def check() -> None:
        store = PostgresStore("postgresql://nobody:secret@127.0.0.1:1/none", connect_timeout_s=1.0)
        with pytest.raises(SessionError, match="Could not connect to PostgreSQL") as raised:
            await store.load("x")
        assert "secret" not in str(raised.value)
        await store.aclose()

    run(check())


@needs_postgres
@pytest.mark.skipif(sys.platform != "win32", reason="the proactor loop exists only on Windows")
def test_the_windows_proactor_loop_is_refused_with_the_fix():
    from qagents.stores.postgres import PostgresStore

    async def check() -> None:
        store = PostgresStore(DSN)
        with pytest.raises(ConfigurationError, match="SelectorEventLoop"):
            await store.load("x")

    with asyncio.Runner(loop_factory=asyncio.ProactorEventLoop) as runner:
        runner.run(check())
