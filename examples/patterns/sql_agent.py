"""SQL agent: answer a question from a SQLite database with list, describe and query tools.

The example builds a small shop database (customers, products, orders) in the work directory.
The agent lists the tables, reads the schemas it needs and answers a question that takes a
join and an aggregate. ``run_query`` runs only a single SELECT: anything else is refused with
`ModelRetry`, whose message tells the model what to fix.

Mirrors: LangGraph's SQL agent tutorial: list_tables, get_schema (the CREATE statement and
three sample rows), generate_query, a check_query node in which the model double-checks the
query, and run_query, with an edge back to generate_query until the model answers. In qagents
it is one agent with three tools, and the order comes from its instructions, not from forced
tool calls:

- ``list_tables`` and ``describe_table`` are the list and schema tools;
- check_query is code inside ``run_query``. Only a statement that starts with SELECT or WITH
  runs, SQLite refuses a second statement, the connection is read-only (so even
  ``WITH ... DELETE`` cannot write), a query is stopped after ``QUERY_SECONDS`` and at most
  ``MAX_ROWS`` rows come back. Instead of a second model call that looks for mistakes,
  SQLite's own error goes back to the model as the tool result, and the model fixes the query
  in the same loop;
- the database path reaches the tools as ``deps`` and is never shown to the model.

Run it:

    python -m examples.patterns.sql_agent --provider anthropic
    python -m examples.patterns.sql_agent --provider qcloud
"""

from __future__ import annotations

import contextlib
import re
import sqlite3
import time
from pathlib import Path

from examples.harness import ExampleContext, ExampleOutcome, main
from qagents import Agent, Limits, ModelRetry, RunContext, tool

NAME = "sql_agent"
DESCRIPTION = "An agent answers a join-and-aggregate question with list, describe and query tools"
INSPIRED_BY = "LangGraph SQL agent (list tables, get schema, check query, run query)"

QUESTION = (
    "Which country's customers spent the most on completed orders, and how much did they "
    "spend in total, in euros?"
)
MAX_ROWS = 20
QUERY_SECONDS = 2.0

# -- the shop: deterministic rows -------------------------------------------------------------

CUSTOMERS = [  # id, name, country
    (1, "Ana Lima", "Portugal"),
    (2, "Bruno Costa", "Portugal"),
    (3, "Clara Weiss", "Germany"),
    (4, "Dieter Braun", "Germany"),
    (5, "Elena Rossi", "Italy"),
    (6, "Fabio Conti", "Italy"),
    (7, "Greta Nilsson", "Sweden"),
    (8, "Hugo Berg", "Sweden"),
]
PRODUCTS = [  # id, name, category, price_eur
    (1, "Trail Backpack", "bags", 89.50),
    (2, "Rain Shell Jacket", "clothing", 145.00),
    (3, "Camp Stove", "gear", 62.50),
    (4, "Merino Socks", "clothing", 18.00),
    (5, "Headlamp", "gear", 34.50),
]
ORDERS = [  # id, customer_id, product_id, quantity, status, ordered_on
    (101, 1, 2, 1, "completed", "2026-09-02"),
    (102, 3, 1, 2, "completed", "2026-09-03"),
    (103, 5, 4, 6, "completed", "2026-09-05"),
    (104, 7, 2, 2, "cancelled", "2026-09-06"),
    (105, 2, 3, 1, "completed", "2026-09-08"),
    (106, 4, 5, 3, "completed", "2026-09-09"),
    (107, 6, 2, 1, "completed", "2026-09-11"),
    (108, 8, 1, 1, "completed", "2026-09-12"),
    (109, 3, 2, 1, "cancelled", "2026-09-14"),
    (110, 1, 4, 4, "completed", "2026-09-15"),
    (111, 5, 3, 3, "completed", "2026-09-17"),
    (112, 7, 5, 2, "completed", "2026-09-18"),
    (113, 2, 1, 1, "refunded", "2026-09-20"),
    (114, 4, 2, 1, "completed", "2026-09-21"),
    (115, 6, 1, 1, "completed", "2026-09-23"),
    (116, 8, 2, 3, "cancelled", "2026-09-24"),
    (117, 3, 4, 2, "completed", "2026-09-26"),
    (118, 5, 5, 1, "completed", "2026-09-27"),
    (119, 7, 3, 1, "completed", "2026-09-29"),
    (120, 1, 5, 2, "completed", "2026-09-30"),
    (121, 8, 4, 5, "completed", "2026-10-01"),
    (122, 2, 2, 1, "completed", "2026-10-02"),
]
SCHEMA = """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, country TEXT NOT NULL);
CREATE TABLE products (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, category TEXT NOT NULL, price_eur REAL NOT NULL);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers (id),
    product_id INTEGER NOT NULL REFERENCES products (id),
    quantity INTEGER NOT NULL,
    status TEXT NOT NULL,  -- completed, cancelled or refunded
    ordered_on TEXT NOT NULL);
"""


def build_database(path: Path) -> Path:
    """Write the shop to a new SQLite file at ``path``."""
    path.unlink(missing_ok=True)
    with contextlib.closing(sqlite3.connect(path)) as db:
        db.executescript(SCHEMA)
        db.executemany("INSERT INTO customers VALUES (?, ?, ?)", CUSTOMERS)
        db.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", PRODUCTS)
        db.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)", ORDERS)
        db.commit()
    return path


def expected_answer() -> tuple[str, float]:
    """The country that spent the most on completed orders, and how much: plain Python."""
    price = {product_id: price for product_id, _, _, price in PRODUCTS}
    country = {customer_id: country for customer_id, _, country in CUSTOMERS}
    spent: dict[str, float] = {}
    for _, customer_id, product_id, quantity, status, _ in ORDERS:
        if status == "completed":
            where = country[customer_id]
            spent[where] = spent.get(where, 0.0) + quantity * price[product_id]
    best = max(spent, key=spent.__getitem__)
    return best, spent[best]


# -- the tools ----------------------------------------------------------------------------------


def _connect(ctx: RunContext[Path]) -> contextlib.closing[sqlite3.Connection]:
    """A read-only connection to the database whose path is the run's deps."""
    return contextlib.closing(sqlite3.connect(f"{ctx.deps.as_uri()}?mode=ro", uri=True))


def _tables(db: sqlite3.Connection) -> list[str]:
    rows = db.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
    return [name for (name,) in rows]


def _table(columns: list[str], rows: list[tuple]) -> str:
    return "\n".join([" | ".join(columns), *(" | ".join(map(str, row)) for row in rows)])


@tool(read_only=True)
def list_tables(ctx: RunContext[Path]) -> str:
    """List the tables in the database."""
    with _connect(ctx) as db:
        return ", ".join(_tables(db))


@tool(read_only=True)
def describe_table(ctx: RunContext[Path], table: str) -> str:
    """Show a table's CREATE statement and its first 3 rows.

    Args:
        table: A table name from list_tables.
    """
    with _connect(ctx) as db:
        found = db.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone()
        if found is None:
            raise ModelRetry(
                f"There is no table {table!r}. The tables are: {', '.join(_tables(db))}."
            )
        cursor = db.execute(f'SELECT * FROM "{table}" LIMIT 3')  # the name exists: see above
        columns = [column[0] for column in cursor.description]
        return f"{found[0]}\n\nFirst rows:\n{_table(columns, cursor.fetchall())}"


@tool(read_only=True)
def run_query(ctx: RunContext[Path], sql: str) -> str:
    """Run one SQLite SELECT statement and return its rows (at most 20).

    Args:
        sql: A single SELECT statement, as bare SQL: no code fence and no comments.
    """
    statement = sql.strip().rstrip(";").strip()
    if re.match(r"\w*", statement).group().upper() not in ("SELECT", "WITH"):
        start = (statement.split() or ["nothing"])[0][:20]
        raise ModelRetry(
            f"Only a single SELECT statement may run, and this one starts with {start!r}. "
            "Send bare SQL that starts with SELECT or WITH: no code fence, no comment, nothing "
            "that changes the database."
        )
    deadline = time.monotonic() + QUERY_SECONDS
    with _connect(ctx) as db:
        # A tool timeout cannot stop the thread a sync tool runs in, so SQLite itself
        # interrupts a runaway query (an endless recursive CTE, say) at the deadline.
        db.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)
        try:
            cursor = db.execute(statement)
            rows = cursor.fetchmany(MAX_ROWS + 1)
        except sqlite3.Error as error:  # a syntax error, an unknown column, a second statement
            problem = str(error)
            if time.monotonic() > deadline:  # the progress handler interrupted it
                problem = f"it was stopped after {QUERY_SECONDS:g} seconds; write a simpler one"
            raise ModelRetry(
                f"The query failed ({problem}). Fix it and call run_query again; describe_table "
                "shows the columns."
            ) from None
        columns = [column[0] for column in cursor.description]
    text = _table(columns, rows[:MAX_ROWS])
    if len(rows) > MAX_ROWS:
        text += f"\n(only the first {MAX_ROWS} rows are shown: aggregate, filter or use LIMIT)"
    return text


def numbers_in(text: str) -> list[float]:
    """Every number in ``text``, with thousands separators removed ("1,234.50" -> 1234.5)."""
    return [float(match.replace(",", "")) for match in re.findall(r"\d[\d,]*(?:\.\d+)?", text)]


async def run(ctx: ExampleContext) -> ExampleOutcome:
    database = build_database(ctx.workdir / "shop.db")
    agent = Agent(
        ctx.model,
        name="sql_agent",
        instructions=(
            "You answer questions about a shop's SQLite database. First call list_tables, then "
            "describe_table for each table you need, then run_query with one SELECT that "
            "computes the answer; let the query do the arithmetic. If run_query reports an "
            "error, fix the query and run it again. Finish with one sentence that answers the "
            "question with the values the query returned; write amounts with a decimal point "
            "and two decimals, for example 1234.50."
        ),
        tools=[list_tables, describe_table, run_query],
        limits=Limits(max_turns=10, max_tool_calls=12, max_cost_usd=0.15),
        hooks=ctx.hooks,
    )
    ctx.log(f"Question: {QUESTION}")
    result = await ctx.print_stream(agent.stream(QUESTION, deps=database))

    country, spent = expected_answer()
    named = country.lower() in result.text.lower()
    amount = any(abs(number - spent) < 0.005 for number in numbers_in(result.text))
    answer = " ".join(result.text.split()) or f"no answer ({result.status}: {result.stop_reason})"
    return ExampleOutcome(
        ok=result.completed and named and amount,
        summary=f"{answer[:120]} (expected {country}, {spent:.2f} EUR)",
        output=result.text,
        details={"country": country, "spent_eur": spent, "named": named, "amount": amount},
    )


if __name__ == "__main__":
    main(run, name=NAME, description=DESCRIPTION, inspired_by=INSPIRED_BY)
