"""A tool-calling assistant that remembers the conversation across turns.

The assistant has three tools (weather from fixture data, unit conversion and a calculator)
and talks to the user in two turns on one SQLite-backed session. The second question only
makes sense with the first one in memory: "where I am" is the city the user named a turn
earlier. Both turns are streamed as they happen.

Mirrors: LangGraph's prebuilt ReAct agent, ``create_react_agent(model, tools, prompt=...,
checkpointer=SqliteSaver(...))``, called with ``config={"configurable": {"thread_id": ...}}``.
In qagents: there is no graph to compile. The agent node, the ``ToolNode`` and the
``tools_condition`` edge are the agent loop itself: ask the model, run the tool calls it made
(read-only ones in parallel), repeat until it answers without calling a tool. The
checkpointer plus thread id is ``Session(SQLiteStore(path), id=...)``. Where LangGraph saves
a snapshot of the graph state after each step, qagents appends every message and tool result
to a journal of plain JSON entries, so any process with the same store and id continues the
conversation.

Run it:

    python -m examples.patterns.react_assistant --provider anthropic
    python -m examples.patterns.react_assistant --provider qcloud
"""

from __future__ import annotations

import ast
import operator
import re
from typing import Literal

from examples.harness import ExampleContext, ExampleOutcome, main
from qagents import Agent, Limits, Session, SQLiteStore, ToolError, tool

NAME = "react_assistant"
DESCRIPTION = "Tool-calling assistant with a two-turn conversation kept on a SQLite session"
INSPIRED_BY = "LangGraph prebuilt ReAct agent with a checkpointer (memory)"

THREAD_ID = "weather-chat"  # the session id: LangGraph's thread_id

WEATHER = {
    "lisbon": {"city": "Lisbon", "condition": "sunny", "temperature_c": 23, "wind_kmh": 14},
    "oslo": {"city": "Oslo", "condition": "light rain", "temperature_c": 7, "wind_kmh": 22},
    "athens": {"city": "Athens", "condition": "clear", "temperature_c": 19, "wind_kmh": 9},
    "tokyo": {"city": "Tokyo", "condition": "cloudy", "temperature_c": 12, "wind_kmh": 11},
}


@tool(read_only=True)
def get_weather(city: str) -> dict:
    """Current weather for a city: condition, temperature in Celsius and wind in km/h.

    Args:
        city: City name, for example "Lisbon".
    """
    found = WEATHER.get(city.split(",")[0].strip().lower())  # "Lisbon, Portugal" works too
    if found is None:
        known = ", ".join(entry["city"] for entry in WEATHER.values())
        raise ToolError(f"No weather data for {city!r}. Known cities: {known}.")
    return found


def number(value: float) -> str:
    """A number as a person writes it: 16, 73.4, 0.333333."""
    if isinstance(value, int):
        return str(value)
    value = round(value, 6)
    return str(int(value)) if value.is_integer() else str(value)


Unit = Literal["celsius", "fahrenheit", "km", "miles", "km/h", "mph"]

CONVERSIONS = {
    ("celsius", "fahrenheit"): lambda value: value * 9 / 5 + 32,
    ("fahrenheit", "celsius"): lambda value: (value - 32) * 5 / 9,
    ("km", "miles"): lambda value: value / 1.609344,
    ("miles", "km"): lambda value: value * 1.609344,
    ("km/h", "mph"): lambda value: value / 1.609344,
    ("mph", "km/h"): lambda value: value * 1.609344,
}


@tool(read_only=True)
def convert_units(value: float, from_unit: Unit, to_unit: Unit) -> str:
    """Convert a value from one unit to another.

    Args:
        value: The number to convert.
        from_unit: The unit of the value.
        to_unit: The unit to convert to.
    """
    convert = CONVERSIONS.get((from_unit, to_unit))
    if convert is None:
        raise ToolError(f"Cannot convert {from_unit} to {to_unit}.")
    return f"{number(value)} {from_unit} = {number(round(convert(value), 2))} {to_unit}"


OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def evaluate(node: ast.AST) -> float:
    """Evaluate an arithmetic syntax tree; anything but numbers and operators is refused."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](evaluate(node.left), evaluate(node.right))
    raise ToolError("Only numbers, + - * / // % and parentheses are allowed.")


@tool(read_only=True)
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression exactly, for example "(23 - 7) * 1.8".

    Args:
        expression: Numbers, + - * / // % and parentheses only.
    """
    try:
        value = evaluate(ast.parse(expression, mode="eval").body)
    except SyntaxError as error:
        raise ToolError(f"{expression!r} is not an arithmetic expression: {error.msg}.") from error
    except ZeroDivisionError as error:
        raise ToolError("Division by zero.") from error
    return f"{expression} = {number(value)}"


INSTRUCTIONS = (
    "You are a weather assistant. Use get_weather for current conditions, convert_units for "
    "unit conversions and calculate for any other arithmetic; do not work out numbers "
    "yourself. When the user refers to something said earlier in the conversation, use it "
    "instead of asking again. Answer in one or two sentences that include the numbers."
)

QUESTIONS = (
    "I'm in Lisbon this week. What's the weather like here, and what is the temperature "
    "in Fahrenheit?",
    "My sister is in Oslo. How many degrees Celsius warmer is it where I am than where she is?",
)


async def run(ctx: ExampleContext) -> ExampleOutcome:
    agent = Agent(
        ctx.model,
        name="weather_assistant",
        instructions=INSTRUCTIONS,
        tools=[get_weather, convert_units, calculate],
        limits=Limits(max_turns=8, max_cost_usd=0.20),
        hooks=ctx.hooks,
    )
    store = SQLiteStore(ctx.workdir / "chat.db")
    answers = []
    try:
        for turn, question in enumerate(QUESTIONS, start=1):
            ctx.log(f"Turn {turn}: {question}")
            # The same session id every turn, like LangGraph's thread_id: the run starts from
            # the whole conversation so far, read back from the journal in chat.db.
            session = Session(store, id=THREAD_ID)
            answers.append(await ctx.print_stream(agent.stream(question, session=session)))
        history = await Session(store, id=THREAD_ID).messages()
    finally:
        await store.aclose()

    first, second = answers
    fahrenheit = re.search(r"\b73(\.4)?\b", first.text) is not None  # 23 °C = 73.4 °F
    remembered = re.search(r"\b16(\.0)?\b", second.text) is not None  # Lisbon 23 - Oslo 7
    ok = first.status == second.status == "completed" and fahrenheit and remembered
    summary = (
        "the second answer used Lisbon from the first turn: 16 degrees warmer than Oslo"
        if ok
        else f"turn 1: {first.status} {first.text!r}; turn 2: {second.status} {second.text!r}"
    )
    tools_used = [c.name for m in history if m.role == "assistant" for c in m.tool_calls]
    return ExampleOutcome(
        ok=ok,
        summary=summary,
        output=[first.text, second.text],
        details={"messages_in_session": len(history), "tool_calls": tools_used},
    )


if __name__ == "__main__":
    main(run, name=NAME, description=DESCRIPTION, inspired_by=INSPIRED_BY)
