"""Offline tests of examples/patterns/human_in_the_loop.py."""

from __future__ import annotations

import json
import runpy
import sys
from collections import Counter

import pytest

from examples.harness import main, scripted_context
from examples.patterns import human_in_the_loop
from examples.patterns.human_in_the_loop import (
    TRIP_ID,
    Deps,
    book_flight,
    bookings,
    create_bookings,
    run,
    scripted_answer,
)
from qagents import RunContext, Session, SQLiteStore
from qagents.models.testing import FunctionModel, call


def policy(request, *, spell=lambda name: name):
    """Plays the model: ask for the name, search, book the cheapest flight, confirm."""
    results = {r.name: r for m in request.messages for r in m.tool_results}
    if "ask_user" not in results:
        return call("ask_user", question="What is the passenger's full name?")
    if "search_flights" not in results:
        return call("search_flights", origin="Athens", destination="Lisbon")
    if "book_flight" not in results:
        cheapest = min(json.loads(results["search_flights"].content), key=lambda f: f["price_eur"])
        name = spell(results["ask_user"].content)  # the answer given to resume(results=...)
        return call("book_flight", flight_id=cheapest["flight_id"], passenger_name=name)
    return f"Done. {results['book_flight'].content}"


async def journal(ctx) -> list:
    store = SQLiteStore(ctx.workdir / "sessions.db")
    try:
        return await Session(store, id=TRIP_ID).entries()
    finally:
        await store.aclose()


async def test_pauses_twice_and_books_once_with_the_answer(tmp_path, capsys):
    ctx = scripted_context(policy, tmp_path)
    outcome = await run(ctx)
    assert outcome.ok, outcome.summary
    assert outcome.details["pauses"] == [
        {"inputs": 1, "approvals": 0},
        {"inputs": 0, "approvals": 1},
    ]
    assert outcome.details["bookings"] == [("QA-212", "Maria Silva")]
    assert outcome.output == "Done. Booked Maria Silva on QA-212. Booking reference BK-1001."
    # One run, paused twice and resumed twice by new agents: never restarted from the prompt.
    entries = await journal(ctx)
    kinds = Counter(entry.kind for entry in entries)
    assert kinds["run_started"] == 1 and kinds["run_paused"] == 2 and kinds["run_finished"] == 1
    assert kinds["tool_deferred"] == 1
    assert len({entry.run_id for entry in entries if entry.run_id}) == 1
    # Only the first request followed the prompt; every later one continued from a tool result.
    prompts = [r for r in ctx.meter.requests if r.messages[-1].role == "user"]
    assert len(prompts) == 1 and len(ctx.meter.requests) == 4
    printed = capsys.readouterr().out
    assert "ask_user(" in printed and "needs input" in printed
    assert 'book_flight(flight_id="QA-212", passenger_name="Maria Silva") needs approval' in printed


async def test_interactive_answers_come_from_the_terminal(tmp_path, monkeypatch):
    typed = iter(["Ana Costa", "yes"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(typed))
    outcome = await run(scripted_context(policy, tmp_path, interactive=True))
    assert outcome.ok, outcome.summary
    assert outcome.details["bookings"] == [("QA-212", "Ana Costa")]


async def test_a_declined_booking_is_not_made(tmp_path, monkeypatch):
    typed = iter(["Ana Costa", "no"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(typed))
    outcome = await run(scripted_context(policy, tmp_path, interactive=True))
    assert not outcome.ok
    assert outcome.details["bookings"] == []
    assert "declined" in outcome.output


async def test_a_guessed_name_is_not_ok(tmp_path):
    def guesser(request):  # books without asking, so the name did not come from the user
        if not any(m.tool_results for m in request.messages):
            return call("book_flight", flight_id="QA-212", passenger_name="John Doe")
        return policy(request)

    outcome = await run(scripted_context(guesser, tmp_path))
    assert not outcome.ok
    assert outcome.details["bookings"] == [("QA-212", "John Doe")]


async def test_the_users_name_in_another_order_is_still_ok(tmp_path):
    def reorder(name):
        first, last = name.split()
        return f"{last.upper()}, {first}"

    outcome = await run(scripted_context(lambda r: policy(r, spell=reorder), tmp_path))
    assert outcome.ok, outcome.summary
    assert outcome.details["bookings"] == [("QA-212", "SILVA, Maria")]


def test_a_repeated_booking_call_books_once(tmp_path):
    deps = Deps(bookings=tmp_path / "bookings.db")
    create_bookings(deps.bookings)
    first = RunContext(deps=deps, operation_id="run-1:3:call-a")
    again = book_flight.function(first, "QA-212", "Maria Silva")
    assert again == book_flight.function(first, "QA-212", "Maria Silva")
    assert again.endswith("BK-1001.")
    other = RunContext(deps=deps, operation_id="run-1:7:call-b")
    assert book_flight.function(other, "QA-210", "Ana Costa").endswith("BK-1002.")
    assert bookings(deps.bookings) == [("QA-212", "Maria Silva"), ("QA-210", "Ana Costa")]
    assert book_flight.idempotent and book_flight.requires_approval


def test_the_simulated_user():
    assert scripted_answer("Who is travelling?") == "Maria Silva"
    assert scripted_answer("What is the passenger's full name?") == "Maria Silva"
    assert "cheapest" in scripted_answer("Do you need checked luggage?")


def test_runs_as_a_module_and_replays_from_its_cassette(tmp_path, monkeypatch, capsys):
    common = ["--provider", "qcloud", "--cassette-dir", str(tmp_path / "cassettes"), "--json"]
    with pytest.raises(SystemExit) as recorded:
        main(
            run,
            name=human_in_the_loop.NAME,
            description=human_in_the_loop.DESCRIPTION,
            inspired_by=human_in_the_loop.INSPIRED_BY,
            argv=[*common, "--record", "--workdir", str(tmp_path / "recorded")],
            model_factory=lambda provider, model: FunctionModel(policy),
        )
    assert recorded.value.code == 0
    capsys.readouterr()

    argv = ["human_in_the_loop", *common, "--replay", "--workdir", str(tmp_path / "replayed")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as replayed:
        runpy.run_path(human_in_the_loop.__file__, run_name="__main__")
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert replayed.value.code == 0, report
    assert report["ok"] and report["requests"] == 4
