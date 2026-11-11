"""Offline tests of examples/patterns/sql_agent.py."""

from __future__ import annotations

import contextlib
import sqlite3
import time

import pytest

from examples.harness import main, scripted_context
from examples.patterns import sql_agent
from examples.patterns.sql_agent import (
    DESCRIPTION,
    INSPIRED_BY,
    MAX_ROWS,
    NAME,
    ORDERS,
    build_database,
    describe_table,
    expected_answer,
    list_tables,
    numbers_in,
    run,
    run_query,
)
from qagents import ModelRetry, RunContext
from qagents.models.testing import FunctionModel, call

QUERY = (
    "SELECT c.country, SUM(o.quantity * p.price_eur) AS spent FROM orders o "
    "JOIN customers c ON c.id = o.customer_id JOIN products p ON p.id = o.product_id "
    "WHERE o.status = 'completed' GROUP BY c.country ORDER BY spent DESC LIMIT 1"
)


def policy(request):
    """List, describe every table, try a write (refused), then the right query, then answer."""
    last = request.messages[-1]
    if last.role == "user":
        return call("list_tables")
    first = last.tool_results[0]
    if first.name == "list_tables":
        return [call("describe_table", table=name) for name in first.content.split(", ")]
    if first.name == "describe_table":
        return call("run_query", sql="DELETE FROM orders WHERE status = 'cancelled'")
    if first.is_error:
        return call("run_query", sql=QUERY)
    country, spent = first.content.splitlines()[1].split(" | ")
    return f"{country} spent the most on completed orders: {float(spent):,.2f} euros."


def order_count(path) -> int:
    with contextlib.closing(sqlite3.connect(path)) as db:
        return db.execute("SELECT COUNT(*) FROM orders").fetchone()[0]


async def test_the_agent_answers_from_the_database(tmp_path):
    ctx = scripted_context(policy, tmp_path)
    outcome = await run(ctx)

    assert outcome.ok, outcome.summary
    assert outcome.output == "Italy spent the most on completed orders: 564.50 euros."
    assert outcome.details["named"] and outcome.details["amount"]
    tools = [
        r.name for m in ctx.meter.requests[-1].messages if m.role == "tool" for r in m.tool_results
    ]
    assert tools == ["list_tables", *["describe_table"] * 3, "run_query", "run_query"]
    refused = ctx.meter.requests[-2].messages[-1].tool_results[0]
    assert refused.is_error and "Only a single SELECT statement may run" in refused.content
    assert "starts with 'DELETE'" in refused.content
    assert order_count(ctx.workdir / "shop.db") == len(ORDERS)  # nothing was deleted


async def test_a_wrong_number_fails_the_outcome(tmp_path):
    def wrong(request):
        reply = policy(request)
        if isinstance(reply, str):  # the final answer, as if cancelled orders were counted
            return "Sweden spent the most: 1,036.00 euros."
        return reply

    outcome = await run(scripted_context(wrong, tmp_path))
    assert not outcome.ok and "expected Italy, 564.50 EUR" in outcome.summary


async def test_the_right_amount_without_the_country_fails_the_outcome(tmp_path):
    def no_country(request):
        reply = policy(request)
        return "They spent 564.50 euros." if isinstance(reply, str) else reply

    outcome = await run(scripted_context(no_country, tmp_path))
    assert not outcome.ok
    assert outcome.details["amount"] and not outcome.details["named"]


def test_the_expected_answer_is_computed_from_the_rows():
    assert expected_answer() == ("Italy", 564.5)


@pytest.fixture
def ctx(tmp_path):
    return RunContext(deps=build_database(tmp_path / "shop.db"))


async def test_list_and_describe(ctx):
    assert await list_tables.run(ctx, {}) == "customers, orders, products"
    described = await describe_table.run(ctx, {"table": "orders"})
    assert "CREATE TABLE orders" in described and "completed, cancelled or refunded" in described
    assert described.endswith("103 | 5 | 4 | 6 | completed | 2026-09-05")
    with pytest.raises(ModelRetry, match="There is no table 'order'. The tables are: customers"):
        await describe_table.run(ctx, {"table": "order"})


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("DROP TABLE orders", "starts with 'DROP'"),
        ("  update orders set quantity = 0", "starts with 'update'"),
        ("-- a comment first\nSELECT 1", "starts with '--'. Send bare SQL"),
        ("```sql\nSELECT 1\n```", "starts with '```sql'"),
        ("  ", "starts with 'nothing'"),
        ("SELECT 1; DELETE FROM orders", "one statement at a time"),
        ("WITH gone AS (SELECT 1) DELETE FROM orders", "readonly database"),
        ("SELECT nope FROM orders", "no such column: nope"),
    ],
)
async def test_run_query_refuses_anything_but_one_select(ctx, sql, reason):
    with pytest.raises(ModelRetry, match=reason):
        await run_query.run(ctx, {"sql": sql})
    assert order_count(ctx.deps) == len(ORDERS)


async def test_run_query_stops_a_runaway_query(ctx, monkeypatch):
    monkeypatch.setattr(sql_agent, "QUERY_SECONDS", 0.2)
    endless = (
        "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) SELECT COUNT(*) FROM n"
    )
    started = time.monotonic()
    with pytest.raises(ModelRetry, match="stopped after 0.2 seconds; write a simpler one"):
        await run_query.run(ctx, {"sql": endless})
    assert time.monotonic() - started < 5


async def test_run_query_caps_the_rows(ctx):
    rows = (await run_query.run(ctx, {"sql": "SELECT id, status FROM orders;"})).splitlines()
    assert rows[0] == "id | status" and rows[1] == "101 | completed"
    assert len(rows) == 1 + MAX_ROWS + 1 and "only the first 20 rows" in rows[-1]
    small = await run_query.run(ctx, {"sql": "select count(*) AS n from orders"})
    assert small == f"n\n{len(ORDERS)}"


def test_numbers_in():
    assert numbers_in("Italy: €1,234.50, or 564.5 EUR.") == [1234.5, 564.5]


@pytest.mark.parametrize("provider", ["anthropic", "qcloud"])
def test_runs_as_a_module_and_replays_from_its_cassette(provider, tmp_path, capsys):
    """The command line, recorded with the scripted model and replayed in another work
    directory: the database path never reaches the model."""

    def record(provider, model):
        return FunctionModel(policy)

    def refuse(provider, model):
        raise AssertionError("a replay must not build the provider's model")

    def run_main(factory, *argv):
        with pytest.raises(SystemExit) as exited:
            main(
                run,
                name=NAME,
                description=DESCRIPTION,
                inspired_by=INSPIRED_BY,
                argv=["--provider", provider, "--cassette-dir", str(tmp_path / "cassettes"), *argv],
                model_factory=factory,
            )
        return exited.value.code

    assert run_main(record, "--record", "--workdir", str(tmp_path / "w1")) == 0
    assert run_main(refuse, "--replay", "--workdir", str(tmp_path / "w2")) == 0
    output = capsys.readouterr().out
    assert output.count("ok: Italy spent the most on completed orders: 564.50 euros.") == 2
