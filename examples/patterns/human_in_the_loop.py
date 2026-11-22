"""A flight booking that stops twice for a human: once for a missing detail, once for approval.

The agent asks for the passenger's name with ``ask_user``, a deferred tool: the run pauses
with the question in ``result.pending_inputs``. It then books with ``book_flight``, which
requires approval: the run pauses again, with the call in ``result.pending_approvals``. Each
step after the first is done the way a separate process would do it: open the SQLiteStore
again, build a new Agent, and resume the session with ``results=`` for the answer and
``approvals=`` for the booking. The simulated user answers from a script; pass
``--interactive`` to answer on the terminal yourself.

Mirrors: LangGraph's human in the loop: ``interrupt(question)`` inside a tool, and an
approve/edit/reject interrupt before a tool runs, each resumed with ``Command(resume=...)`` on
the same ``thread_id`` and checkpointer. In qagents: a question is a call of a deferred tool,
answered with ``agent.resume(session, results={call_id: answer})``; an approval is a tool
declared ``requires_approval=True``, answered with ``approvals={call_id: True}`` (``False``
rejects, ``Approval(approved=True, arguments=...)`` edits). The pause is a tool call recorded
in the session journal, not a suspended node. Where LangGraph runs the interrupted node again
from its start on resume, qagents continues at the waiting call, and no call that already has
a result runs again. Any process that can read the store resumes the run, with an agent of the
same name and tools.

Run it:

    python -m examples.patterns.human_in_the_loop --provider anthropic
    python -m examples.patterns.human_in_the_loop --provider qcloud --interactive
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.harness import ExampleContext, ExampleOutcome, main
from qagents import Agent, Limits, RunContext, RunResult, Session, SQLiteStore, ToolError, tool
from qagents.toolkits import ask_user

NAME = "human_in_the_loop"
DESCRIPTION = "Booking agent that pauses for a question and an approval and is resumed twice"
INSPIRED_BY = "LangGraph interrupt() / human-in-the-loop with Command(resume=...)"

TRIP_ID = "trip-ath-lis"  # the session id: LangGraph's thread_id
PASSENGER = "Maria Silva"  # what the simulated user answers when asked for the name
MAX_PAUSES = 4  # the example gives up after this many pauses

PROMPT = "Please book me the cheapest flight from Athens to Lisbon on 4 December 2026."

INSTRUCTIONS = (
    "You book flights. A booking needs the route, the date and the passenger's full name. "
    "Ask for each missing detail with ask_user, never in a plain reply, and never guess a "
    "name. Then call search_flights, pick the flight the user asked for, and call book_flight "
    "once. book_flight asks the user for approval itself, so do not ask for confirmation "
    "first. Finally reply in one sentence: the booking reference, or that nothing was booked "
    "if the user declined."
)

ROUTE = ("ATH", "LIS")
AIRPORTS = {"athens": "ATH", "ath": "ATH", "lisbon": "LIS", "lis": "LIS"}
FLIGHTS = [
    {"flight_id": "QA-210", "departs": "07:15", "arrives": "09:40", "price_eur": 189},
    {"flight_id": "QA-212", "departs": "13:05", "arrives": "15:30", "price_eur": 149},
    {"flight_id": "QA-216", "departs": "19:50", "arrives": "22:15", "price_eur": 172},
]


@dataclass
class Deps:
    bookings: Path
    """The booking system: a SQLite file every process can reach."""


@tool(read_only=True)
def search_flights(origin: str, destination: str) -> list[dict[str, Any]]:
    """Flights on 4 December 2026 between two cities, with prices in EUR.

    Args:
        origin: Departure city or airport code, for example "Athens".
        destination: Arrival city or airport code, for example "Lisbon".
    """
    route = (AIRPORTS.get(origin.strip().lower()), AIRPORTS.get(destination.strip().lower()))
    if route != ROUTE:
        raise ToolError("This example only has flights from Athens (ATH) to Lisbon (LIS).")
    return FLIGHTS


# idempotent=True lets the loop run the call again if a process dies while it runs; the
# operation id as the booking system's key makes that repeat find the first booking.
@tool(requires_approval=True, idempotent=True)
def book_flight(ctx: RunContext[Deps], flight_id: str, passenger_name: str) -> str:
    """Book one seat on a flight. The user is asked to approve every booking.

    Args:
        flight_id: The flight_id from search_flights, for example "QA-210".
        passenger_name: The passenger's full name, exactly as the user gave it.
    """
    if flight_id not in {flight["flight_id"] for flight in FLIGHTS}:
        raise ToolError(f"There is no flight {flight_id!r}; use a flight_id from search_flights.")
    with closing(sqlite3.connect(ctx.deps.bookings)) as db:
        with db:  # one transaction
            db.execute(
                "INSERT OR IGNORE INTO bookings VALUES (?, ?, ?)",
                (ctx.operation_id, flight_id, passenger_name),
            )
        (number,) = db.execute(
            "SELECT rowid FROM bookings WHERE operation_id = ?", (ctx.operation_id,)
        ).fetchone()
    return f"Booked {passenger_name} on {flight_id}. Booking reference BK-{1000 + number}."


def create_bookings(path: Path) -> None:
    with closing(sqlite3.connect(path)) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS bookings"
            " (operation_id TEXT PRIMARY KEY, flight_id TEXT, passenger TEXT)"
        )


def bookings(path: Path) -> list[tuple[str, str]]:
    """(flight_id, passenger) of every booking, in order."""
    with closing(sqlite3.connect(path)) as db:
        return db.execute("SELECT flight_id, passenger FROM bookings ORDER BY rowid").fetchall()


def travel_agent(ctx: ExampleContext) -> Agent:
    """The agent as every process builds it.

    A resume needs an agent with the name recorded in the journal (or one that reaches it
    through handoffs) and the same tools. Limits apply to each process's part of the run:
    they start again on every resume, so ``MAX_PAUSES`` is what bounds the whole run.
    """
    return Agent(
        ctx.model,
        name="travel_agent",
        instructions=INSTRUCTIONS,
        tools=[ask_user, search_flights, book_flight],
        limits=Limits(max_turns=8, max_cost_usd=0.25),
        hooks=ctx.hooks,
    )


async def process(
    ctx: ExampleContext,
    deps: Deps,
    *,
    prompt: str | None = None,
    results: dict[str, Any] | None = None,
    approvals: dict[str, bool] | None = None,
) -> RunResult:
    """What one process does: open the store, build a new agent, start or resume the run."""
    store = SQLiteStore(ctx.workdir / "sessions.db")
    session = Session(store, id=TRIP_ID)
    agent = travel_agent(ctx)
    try:
        if prompt is not None:
            stream = agent.stream(prompt, session=session, deps=deps)
        else:
            stream = agent.resume_stream(session, results=results, approvals=approvals, deps=deps)
        return await ctx.print_stream(stream)
    finally:
        await store.aclose()


def scripted_answer(question: str) -> str:
    """What the simulated user says to a question (with --interactive, you answer)."""
    if re.search(r"\b(name|passenger|who|travell?er)", question, re.IGNORECASE):
        return PASSENGER
    return f"No other preferences: book the cheapest flight. The passenger is {PASSENGER}."


def words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


async def run(ctx: ExampleContext) -> ExampleOutcome:
    deps = Deps(bookings=ctx.workdir / "bookings.db")
    create_bookings(deps.bookings)

    ctx.log(f"Process 1: {PROMPT}")
    result = await process(ctx, deps, prompt=PROMPT)
    answers: list[str] = []
    pauses: list[dict[str, int]] = []
    while result.status == "paused" and len(pauses) < MAX_PAUSES:
        results: dict[str, Any] = {}
        for waiting in result.pending_inputs:  # LangGraph: the __interrupt__ values
            # The loop does not check a deferred call's arguments, so read them defensively.
            question = str(waiting.arguments.get("question") or "Anything to add?")
            choices = waiting.arguments.get("choices")
            if isinstance(choices, list) and choices:
                question += f" ({' / '.join(map(str, choices))})"
            answer = await ctx.ask(question, default=scripted_answer(question))
            results[waiting.tool_call_id] = answer
            answers.append(answer)
        approvals: dict[str, bool] = {}
        for pending in result.pending_approvals:
            call = ", ".join(f"{key}={value!r}" for key, value in pending.arguments.items())
            reply = await ctx.ask(f"Allow {pending.name}({call})? yes/no", default="yes")
            approvals[pending.tool_call_id] = reply.strip().lower() in ("y", "yes")
        pauses.append({"inputs": len(results), "approvals": len(approvals)})
        ctx.log(f"Process {len(pauses) + 1}: a new agent resumes the session")
        result = await process(ctx, deps, results=results or None, approvals=approvals or None)

    booked = bookings(deps.bookings)
    # Asked for, not guessed: every word of the booked name is one the user said.
    name = words(booked[0][1]) if len(booked) == 1 else set()
    from_user = bool(name) and name <= words(" ".join(answers))
    ok = result.status == "completed" and len(booked) == 1 and from_user
    if ok:
        summary = f"booked {booked[0][0]} once for {booked[0][1]} after {len(pauses)} pauses"
    else:
        summary = f"run {result.status}, bookings {booked}, answers {answers}: {result.text!r}"
    return ExampleOutcome(
        ok=ok,
        summary=summary,
        output=result.text,
        details={"pauses": pauses, "answers": answers, "bookings": booked},
    )


if __name__ == "__main__":
    main(run, name=NAME, description=DESCRIPTION, inspired_by=INSPIRED_BY)
