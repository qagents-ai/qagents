"""A trip-planning crew: pick a city, research it in two parallel tasks, then plan within budget.

Three agents, as in CrewAI's trip planner. A city selector compares three candidate cities
with weather and cost tools. A local expert lists sights with their prices while, at the same
time, a travel concierge finds lodging: two independent `Workflow.run_agent` steps under
``asyncio.gather``. The concierge then turns the guide and the lodging into an `Itinerary`.
An output validator checks the itinerary in code against the city's real prices (the number
of days, a hotel night and food every day, each activity at its listed price, the total
within the budget) and, when something is off, sends the model every problem at once
(`ModelRetry`) so it answers again.

Mirrors: CrewAI's trip planner crew (city selection agent, local expert, travel concierge,
with the tasks identify, gather and plan), using CrewAI's ``Task(async_execution=True)`` for
two tasks that do not depend on each other and ``context=[...]`` for the task that needs
both. In qagents the parallel tasks are plain ``asyncio.gather``; an agent with two tasks of
different output types is ``concierge.copy(output_type=...)``; every task returns a pydantic
type instead of prose; the context each agent gets is written into its prompt by the code;
and the budget is enforced by a validator rather than left to the model's own arithmetic.

Run it:

    python -m examples.patterns.trip_planner --provider anthropic
    python -m examples.patterns.trip_planner --provider qcloud
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Literal

from pydantic import BaseModel, Field

from examples.harness import ExampleContext, ExampleOutcome, main
from qagents import (
    Agent,
    InMemoryStore,
    Limits,
    ModelRetry,
    OutputValidationError,
    RunContext,
    Workflow,
    tool,
)

NAME = "trip_planner"
DESCRIPTION = "City selector, then local expert and concierge in parallel, then a budgeted plan"
INSPIRED_BY = "CrewAI trip planner crew (city selection, local expert, travel concierge)"

City = Literal["Lisbon", "Porto", "Seville"]


@dataclasses.dataclass(frozen=True)
class Trip:
    """The traveller's request; the itinerary's validator reads it as ``ctx.deps``."""

    days: int
    budget_eur: float
    interests: str
    city: City | None = None
    """Set once the city selector has chosen."""


TRIP = Trip(days=4, budget_eur=750, interests="food, history and long walks")
REQUEST = (
    f"One traveller, {TRIP.days} days and {TRIP.days} hotel nights in May, total budget EUR "
    f"{TRIP.budget_eur:.0f} for lodging, food and activities. Interests: {TRIP.interests}. "
    "Prefers daytime highs between 20 and 26 °C and few rainy days."
)

WEATHER_MAY = {
    "Lisbon": {"avg_high_c": 23, "rain_days": 5},
    "Porto": {"avg_high_c": 20, "rain_days": 9},
    "Seville": {"avg_high_c": 29, "rain_days": 3},
}
FOOD_PER_DAY_EUR = {"Lisbon": 42, "Porto": 36, "Seville": 38}
HOTELS = {
    "Lisbon": [("Casa Alfama Guesthouse", "Alfama", 92), ("Hotel Baixa Central", "Baixa", 128),
               ("LX Riverside Suites", "Cais do Sodre", 175)],
    "Porto": [("Ribeira Rooms", "Ribeira", 68), ("Hotel Aliados", "Baixa", 96),
              ("Douro View Hotel", "Gaia", 140)],
    "Seville": [("Triana Patio Rooms", "Triana", 78), ("Hotel Giralda", "Santa Cruz", 115),
                ("Palacio Guadalquivir", "Arenal", 190)],
}  # fmt: skip
ATTRACTIONS = {
    "Lisbon": [("Jeronimos Monastery", 12), ("Belem Tower", 9), ("Tram 28 ride", 3),
               ("Fado dinner show", 55), ("Sintra day trip", 48), ("Food tour", 75),
               ("Sao Jorge Castle", 15), ("Viewpoints walk", 0)],
    "Porto": [("Livraria Lello", 8), ("Port cellar tour", 25), ("Douro valley day trip", 95),
              ("Clerigos Tower", 10), ("Six bridges cruise", 18), ("Serralves Museum", 20),
              ("Riverside walk", 0)],
    "Seville": [("Real Alcazar", 15), ("Cathedral and Giralda", 13), ("Flamenco show", 30),
                ("Plaza de Espana walk", 0), ("Tapas tour", 60), ("Cordoba day trip", 70),
                ("Metropol Parasol", 15)],
}  # fmt: skip


@tool(read_only=True)
def weather(city: City) -> dict:
    """Average weather in May: daytime high in °C and rainy days in the month.

    Args:
        city: One of the candidate cities.
    """
    return WEATHER_MAY[city]


@tool(read_only=True)
def daily_cost(city: City) -> dict:
    """Typical costs for one traveller: the cheapest hotel night and food for a day, in EUR.

    Args:
        city: One of the candidate cities.
    """
    cheapest = min(price for _, _, price in HOTELS[city])
    return {"cheapest_hotel_night_eur": cheapest, "food_per_day_eur": FOOD_PER_DAY_EUR[city]}


@tool(read_only=True)
def hotels(city: City) -> list[dict]:
    """Hotels in the city with their area and price per night in EUR.

    Args:
        city: One of the candidate cities.
    """
    return [{"name": n, "area": a, "price_per_night_eur": p} for n, a, p in HOTELS[city]]


@tool(read_only=True)
def attractions(city: City) -> list[dict]:
    """Sights and activities in the city with their price per person in EUR.

    Args:
        city: One of the candidate cities.
    """
    return [{"name": name, "cost_eur": cost} for name, cost in ATTRACTIONS[city]]


# -- what each task returns ------------------------------------------------------------------


class CityChoice(BaseModel):
    city: City
    reason: str


class Activity(BaseModel):
    name: str
    cost_eur: float = Field(ge=0)


class LocalGuide(BaseModel):
    activities: list[Activity] = Field(description="Six to eight activities, with prices.")
    tips: list[str]


class Lodging(BaseModel):
    hotel: str
    price_per_night_eur: float
    food_per_day_eur: float


class Day(BaseModel):
    day: int
    activities: list[Activity]
    lodging_eur: float = Field(ge=0, description="The hotel night of this day.")
    food_eur: float = Field(ge=0)

    @property
    def cost(self) -> float:
        return self.lodging_eur + self.food_eur + sum(a.cost_eur for a in self.activities)


class Itinerary(BaseModel):
    days: list[Day]
    total_cost_eur: float = Field(description="Every day's lodging, food and activities.")


def check_itinerary(ctx: RunContext[Trip], plan: Itinerary) -> Itinerary:
    """The budget, checked by code against the city's real prices.

    Raises `ModelRetry` listing every problem, so the model can fix them in one answer.
    Returns the plan with the total recomputed.
    """
    trip = ctx.deps
    prices = {name.casefold(): cost for name, cost in ATTRACTIONS[trip.city]}
    cheapest = min(price for _, _, price in HOTELS[trip.city])
    food = FOOD_PER_DAY_EUR[trip.city]
    problems, unlisted = [], False
    if len(plan.days) != trip.days:
        problems.append(f"The trip is {trip.days} days, not {len(plan.days)}: plan every day.")
    for day in plan.days:
        if day.lodging_eur < cheapest:
            problems.append(f"Day {day.day}: a hotel night costs at least EUR {cheapest}.")
        if day.food_eur < food:
            problems.append(f"Day {day.day}: food costs EUR {food} a day.")
        for activity in day.activities:
            listed = prices.get(activity.name.strip().casefold())
            if listed is None:
                unlisted = True
                problems.append(f"Day {day.day}: {activity.name!r} is not a listed activity.")
            elif activity.cost_eur != listed:
                problems.append(f"Day {day.day}: {activity.name} costs EUR {listed}.")
    total = sum(day.cost for day in plan.days)
    if total > trip.budget_eur:
        problems.append(
            f"The plan costs EUR {total:.0f}, EUR {total - trip.budget_eur:.0f} over the budget "
            f"of EUR {trip.budget_eur:.0f}: drop or swap paid activities, or pick a cheaper hotel."
        )
    if unlisted:
        names = ", ".join(f"{name} (EUR {cost})" for name, cost in ATTRACTIONS[trip.city])
        problems.append(f"The listed activities in {trip.city}: {names}.")
    if problems:
        raise ModelRetry("\n".join(problems))
    return plan.model_copy(update={"total_cost_eur": round(total, 2)})


def build_crew(ctx: ExampleContext) -> dict[str, Agent]:
    def member(name: str, instructions: str, tools: list, output_type: type, **options) -> Agent:
        return Agent(
            ctx.model,
            name=name,
            instructions=instructions,
            tools=tools,
            output_type=output_type,
            output_mode=ctx.output_mode,
            hooks=ctx.hooks,
            **options,
        )

    selector = member(
        "city_selector",
        "You are a city selection expert. Call weather and daily_cost for each candidate city "
        "(Lisbon, Porto, Seville), all six calls in one turn, then choose the city that best "
        "fits the traveller's weather preference and budget, and give the reason in one "
        "sentence.",
        [weather, daily_cost],
        CityChoice,
        limits=Limits(max_turns=8, max_cost_usd=0.10),  # room for one lookup per turn
    )
    local_expert = member(
        "local_expert",
        "You are a local expert. Call attractions once for the city you are given, then pick "
        "six to eight activities that suit the traveller, with their names and prices exactly "
        "as attractions lists them, and add three short insider tips.",
        [attractions],
        LocalGuide,
        limits=Limits(max_turns=4, max_cost_usd=0.08),
    )
    concierge = member(
        "concierge",
        "You are a travel concierge. You find lodging and turn a local guide into a "
        "day-by-day plan that fits the traveller's budget. Do the task you are given.",
        [hotels, daily_cost],
        Lodging,
        limits=Limits(max_turns=5, max_cost_usd=0.08),
    )
    # The concierge's second task: the same agent, with another output type and a validator.
    planner = concierge.copy(
        output_type=Itinerary,
        output_validators=[check_itinerary],
        limits=Limits(max_turns=6, max_cost_usd=0.20, output_retries=3),
    )
    return {
        "selector": selector,
        "local_expert": local_expert,
        "concierge": concierge,
        "planner": planner,
    }


async def run(ctx: ExampleContext) -> ExampleOutcome:
    crew = build_crew(ctx)
    workflow = Workflow(InMemoryStore(), id="trip:may")  # SQLiteStore(path) makes it durable

    ctx.log("Task 1, identify: the city selector compares the candidates")
    chosen = await workflow.run_agent("identify", crew["selector"], REQUEST)
    if not chosen.completed:
        return ExampleOutcome(ok=False, summary=f"City selection stopped: {chosen.stop_reason}")
    choice: CityChoice = chosen.output
    ctx.log(f"{choice.city}: {choice.reason}")

    ctx.log("Tasks 2 and 3 at the same time: the local expert's guide, the concierge's lodging")
    traveller = f"City: {choice.city}\nTraveller: {REQUEST}"
    guide_run, lodging_run = await asyncio.gather(
        workflow.run_agent("gather", crew["local_expert"], traveller),
        workflow.run_agent(
            "lodging",
            crew["concierge"],
            "Task: find lodging. Call hotels and daily_cost once each, then choose the hotel "
            f"that leaves the most budget for activities.\n\n{traveller}",
        ),
    )
    for step in (guide_run, lodging_run):
        if not step.completed:
            return ExampleOutcome(ok=False, summary=f"{step.agent} stopped: {step.stop_reason}")

    ctx.log("Task 4, plan: the concierge writes the itinerary; a validator checks the prices")
    plan_task = (
        f"Task: write the itinerary. Plan exactly {TRIP.days} days. Every day has one hotel "
        "night at the lodging price, the food cost per day, and activities from the local "
        "guide only, with their names and prices exactly as listed (meals are covered by the "
        f"food cost). The total must be at most EUR {TRIP.budget_eur:.0f}.\n\n{traveller}\n\n"
        f"Local guide:\n{guide_run.output.model_dump_json(indent=2)}\n\n"
        f"Lodging:\n{lodging_run.output.model_dump_json(indent=2)}"
    )
    trip = dataclasses.replace(TRIP, city=choice.city)
    try:
        planned = await workflow.run_agent("plan", crew["planner"], plan_task, deps=trip)
    except OutputValidationError as error:  # every retry was rejected: an exception, not a result
        return ExampleOutcome(ok=False, summary=f"No itinerary passed the checks: {error}"[:300])
    if not planned.completed:
        return ExampleOutcome(ok=False, summary=f"Planning stopped: {planned.stop_reason}")

    plan: Itinerary = planned.output
    for day in plan.days:
        names = ", ".join(activity.name for activity in day.activities) or "free day"
        print(f"  Day {day.day}: {names} (EUR {day.cost:.0f})")
    total = sum(day.cost for day in plan.days)  # recomputed here, not taken from the model
    weather_fits = 20 <= WEATHER_MAY[choice.city]["avg_high_c"] <= 26
    ok = weather_fits and len(plan.days) == TRIP.days and total <= TRIP.budget_eur
    summary = f"{choice.city}, {len(plan.days)} days, EUR {total:.0f} of EUR {TRIP.budget_eur:.0f}"
    if not weather_fits:
        summary += f" ({choice.city} is outside the 20 to 26 °C the traveller asked for)"
    return ExampleOutcome(ok=ok, summary=summary, output=plan.model_dump())


if __name__ == "__main__":
    main(run, name=NAME, description=DESCRIPTION, inspired_by=INSPIRED_BY)
