"""Offline tests of examples/patterns/trip_planner.py: parallel tasks and the budget check."""

from __future__ import annotations

import asyncio
import json
import runpy
import sys

import pytest

from examples.harness import (
    last_text,
    main,
    scripted_context,
    structured_answer,
    system_text,
    tool_names,
)
from examples.patterns import trip_planner
from examples.patterns.trip_planner import TRIP, run
from qagents.models.testing import FunctionModel, call

GUIDE = {
    "activities": [
        {"name": "Jeronimos Monastery", "cost_eur": 12},
        {"name": "Fado dinner show", "cost_eur": 55},
        {"name": "Sintra day trip", "cost_eur": 48},
        {"name": "Viewpoints walk", "cost_eur": 0},
    ],
    "tips": ["Buy a Viva Viagem card."],
}
LODGING = {"hotel": "Casa Alfama Guesthouse", "price_per_night_eur": 92, "food_per_day_eur": 42}


def day(number, *activities, lodging=92, food=42):
    names = {a["name"]: a for a in GUIDE["activities"]}
    return {
        "day": number,
        "activities": [names[name] for name in activities],
        "lodging_eur": lodging,
        "food_eur": food,
    }


PLAN = [  # 4 x (92 + 42) = 536, plus 115 of activities: 651
    day(1, "Jeronimos Monastery", "Viewpoints walk"),
    day(2, "Sintra day trip"),
    day(3, "Fado dinner show"),
    day(4, "Viewpoints walk"),
]
LUXURY = [day(number, "Fado dinner show", lodging=175) for number in range(1, 5)]  # 1088
MESSY = [  # three days, a day without a hotel, no food, a wrong price, an unlisted activity
    day(1, "Jeronimos Monastery"),
    day(2, "Sintra day trip", lodging=0),
    {**day(3, food=0), "activities": [{"name": "Sintra day trip", "cost_eur": 0},
                                      {"name": "Lunch in Alfama", "cost_eur": 20}]},
]  # fmt: skip


def task_of(request) -> str:
    """Which agent and task sent ``request``: its role, plus the task line for the concierge."""
    system, prompt = system_text(request), request.messages[1].text
    for role in ("city selection expert", "local expert"):
        if role in system:
            return role
    if "travel concierge" in system:
        return "lodging" if prompt.startswith("Task: find lodging") else "plan"
    raise AssertionError(f"unexpected request: {system[:80]!r}")


def policy(request, plans=(PLAN,), city="Lisbon"):
    """Plays the crew. ``plans`` are the concierge's itineraries, one per attempt."""
    task, last = task_of(request), request.messages[-1]
    if task == "city selection expert":
        if last.role == "user":
            return [call(tool, city=name) for name in ("Lisbon", "Porto", "Seville")
                    for tool in ("weather", "daily_cost")]  # fmt: skip
        return structured_answer(request, city=city, reason="23 °C and only 5 rainy days.")
    if task == "local expert":
        if last.role == "user":
            return call("attractions", city=city)
        return structured_answer(request, **GUIDE)
    if task == "lodging":
        if last.role == "user":
            return [call("hotels", city=city), call("daily_cost", city=city)]
        return structured_answer(request, **LODGING)
    attempt = sum(message.role == "assistant" for message in request.messages)
    plan = plans[min(attempt, len(plans) - 1)]
    return structured_answer(
        request, days=plan, total_cost_eur=1.0
    )  # a wrong total, recomputed by code


def planner_requests(ctx):
    return [request for request in ctx.meter.requests if task_of(request) == "plan"]


async def test_trip_planner_plans_within_the_budget(tmp_path, capsys):
    ctx = scripted_context(policy, tmp_path)
    outcome = await run(ctx)
    output = capsys.readouterr().out
    assert outcome.ok, outcome.summary
    assert outcome.summary == "Lisbon, 4 days, EUR 651 of EUR 750"
    assert outcome.output["total_cost_eur"] == 651  # the validator replaced the model's total
    assert "Day 2: Sintra day trip (EUR 182)" in output

    (planner,) = planner_requests(ctx)
    prompt = planner.messages[1].text
    assert "City: Lisbon" in prompt and "Casa Alfama Guesthouse" in prompt
    assert "Sintra day trip" in prompt and f"Plan exactly {TRIP.days} days" in prompt
    # The concierge does two tasks: the same instructions and tools, another output type.
    lodging = next(r for r in ctx.meter.requests if task_of(r) == "lodging")
    assert system_text(lodging) == system_text(planner)
    assert {"hotels", "daily_cost"} <= tool_names(lodging) & tool_names(planner)
    assert len(ctx.meter.requests) == 7


async def test_the_local_expert_and_the_concierge_work_at_the_same_time(tmp_path):
    running, overlap = set(), []

    async def slow(request):
        task = task_of(request)
        running.add(task)
        overlap.append(len(running))
        await asyncio.sleep(0.05)
        running.discard(task)
        return policy(request)

    def dispatch(request):
        return slow(request) if task_of(request) in ("local expert", "lodging") else policy(request)

    assert (await run(scripted_context(dispatch, tmp_path))).ok
    assert max(overlap) == 2


async def test_the_crew_runs_in_native_mode_as_on_anthropic(tmp_path):
    plans = (LUXURY, PLAN)
    ctx = scripted_context(lambda r: policy(r, plans=plans), tmp_path, output_mode="native")
    outcome = await run(ctx)
    assert outcome.ok, outcome.summary
    assert all("final_answer" not in tool_names(r) for r in ctx.meter.requests)
    retry = planner_requests(ctx)[1].messages[-1]  # native mode: the correction is a message
    assert retry.role == "user" and "EUR 338 over the budget" in retry.text


async def test_an_over_budget_plan_is_sent_back_until_it_fits(tmp_path):
    ctx = scripted_context(lambda request: policy(request, plans=(LUXURY, PLAN)), tmp_path)
    outcome = await run(ctx)
    assert outcome.ok, outcome.summary
    planner = planner_requests(ctx)
    assert len(planner) == 2
    assert "EUR 1088, EUR 338 over the budget of EUR 750" in last_text(planner[1])


async def test_every_problem_of_a_plan_is_sent_back_at_once(tmp_path):
    ctx = scripted_context(lambda r: policy(r, plans=(MESSY, PLAN)), tmp_path)
    assert (await run(ctx)).ok
    planner = planner_requests(ctx)
    assert len(planner) == 2
    correction = last_text(planner[1])
    for problem in (
        "The trip is 4 days, not 3",
        "Day 2: a hotel night costs at least EUR 92.",
        "Day 3: food costs EUR 42 a day.",
        "Day 3: Sintra day trip costs EUR 48.",
        "Day 3: 'Lunch in Alfama' is not a listed activity.",
        "The listed activities in Lisbon: Jeronimos Monastery (EUR 12), Belem Tower (EUR 9)",
    ):
        assert problem in correction


async def test_a_plan_that_never_fits_fails_the_example(tmp_path):
    ctx = scripted_context(lambda request: policy(request, plans=(LUXURY,)), tmp_path)
    outcome = await run(ctx)
    assert not outcome.ok
    assert outcome.summary.startswith("No itinerary passed the checks")
    assert "over the budget" in outcome.summary
    assert len(planner_requests(ctx)) == 4  # the first answer and Limits(output_retries=3)


async def test_a_city_outside_the_weather_preference_fails(tmp_path):
    seville = [day(n, lodging=78, food=38) for n in range(1, 5)]
    outcome = await run(
        scripted_context(lambda r: policy(r, plans=(seville,), city="Seville"), tmp_path)
    )
    assert not outcome.ok and "Seville is outside the 20 to 26 °C" in outcome.summary


def test_runs_as_a_module_and_replays_from_its_cassette(tmp_path, capsys, monkeypatch):
    cassettes = tmp_path / "cassettes"
    common = ["--provider", "qcloud", "--cassette-dir", str(cassettes)]
    with pytest.raises(SystemExit) as recorded:
        main(
            run,
            name=trip_planner.NAME,
            description=trip_planner.DESCRIPTION,
            inspired_by=trip_planner.INSPIRED_BY,
            argv=[*common, "--record", "--workdir", str(tmp_path / "recorded")],
            model_factory=lambda provider, model: FunctionModel(policy),
        )
    assert recorded.value.code == 0
    capsys.readouterr()

    argv = ["trip_planner", *common, "--replay", "--workdir", str(tmp_path / "replayed")]
    monkeypatch.setattr(sys, "argv", [*argv, "--json"])
    with pytest.raises(SystemExit) as replayed:
        runpy.run_path(trip_planner.__file__, run_name="__main__")
    output = capsys.readouterr().out
    report = json.loads(output.strip().splitlines()[-1])
    assert replayed.value.code == 0, output
    assert report["ok"] and report["name"] == "trip_planner" and report["requests"] == 7
    assert "Lisbon, 4 days, EUR 651 of EUR 750" in report["summary"]
