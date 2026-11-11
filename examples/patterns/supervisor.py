"""A supervisor delegates to a researcher and an analyst, then writes the report.

The two workers are tools of the supervisor (``agent.as_tool()``): a researcher with a search
tool over fixture data and an analyst with a calculator. Every worker call is a child run of
the supervisor's tool call, in a fresh conversation: the worker sees only the task the
supervisor wrote, and the supervisor sees only the worker's answer. The workers' usage is added
to the supervisor's run, they may spend only what is left of its budget, and their events reach
the supervisor's stream as ``subagent_event`` (printed with a ``[researcher]`` or ``[analyst]``
prefix, and counted here by a hook on the supervisor).

Mirrors: the LangGraph supervisor (``create_supervisor`` in ``langgraph-supervisor``). There the
supervisor and the workers are nodes of one graph that share one message list: a handoff tool
(``transfer_to_research_expert``) moves control to a worker, which by default reads the whole
list, and control returns to the supervisor when the worker ends (``output_mode`` keeps either
the worker's full history or only its last message). qagents has no graph and no shared list.
A worker is a tool, as in LangGraph's tool-calling supervisor variant: control comes back
because the tool call returns, and each side sees only the task and the answer. (A qagents
handoff, ``handoffs=[...]``, gives the conversation to the other agent with no automatic
return: the swarm pattern, not a supervisor.) Usage, budget and tracing roll up through the
linked child runs.

Run it:

    python -m examples.patterns.supervisor --provider anthropic
    python -m examples.patterns.supervisor --provider qcloud
"""

from __future__ import annotations

import ast
import operator
import re
from collections import Counter
from typing import Any

from examples.harness import ExampleContext, ExampleOutcome, main
from qagents import Agent, Hooks, Limits, RunContext, ToolError, tool

NAME = "supervisor"
DESCRIPTION = "A supervisor delegates to a researcher and an analyst, then writes a short report"
INSPIRED_BY = "LangGraph supervisor multi-agent (langgraph-supervisor)"

COMPANIES = {  # fixture data
    "Acme Robotics": "Acme Robotics (Lyon) had 12,480 employees at the end of 2025.",
    "Borealis Energy": "Borealis Energy (Oslo) had 8,215 employees at the end of 2025.",
    "Cobalt Health": "Cobalt Health (Porto) had 4,903 employees at the end of 2025.",
}
TOTAL = 12_480 + 8_215 + 4_903  # 25,598: the figure the report must state

QUESTION = (
    "What was the combined headcount of Acme Robotics, Borealis Energy and Cobalt Health at "
    "the end of 2025? Write a short report."
)


@tool(read_only=True)
def search(query: str) -> str:
    """Search the company database. Returns the facts about every company the query names.

    Args:
        query: One or more company names, for example "Acme Robotics".
    """
    words = set(re.findall(r"[a-z]+", query.lower()))
    hits = [fact for name, fact in COMPANIES.items() if words & set(name.lower().split())]
    return "\n".join(hits) or f"No results. The database knows {', '.join(COMPANIES)}."


OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def evaluate(node: ast.AST) -> float:
    """Evaluate an arithmetic syntax tree; anything but numbers and + - * / is refused."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](evaluate(node.left), evaluate(node.right))
    raise ToolError("Only numbers, + - * / and parentheses are allowed.")


@tool(read_only=True)
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression exactly, for example "(1200 + 340) * 2".

    Args:
        expression: Numbers, + - * / and parentheses. Thousands separators are ignored.
    """
    try:
        value = evaluate(ast.parse(expression.replace(",", ""), mode="eval").body)
    except SyntaxError as error:
        raise ToolError(f"{expression!r} is not an arithmetic expression: {error.msg}.") from error
    except ZeroDivisionError as error:
        raise ToolError("Division by zero.") from error
    value = round(value, 6)
    return f"{expression} = {int(value) if float(value).is_integer() else value}"


class WorkerActivity(Hooks):
    """Counts the events each worker's run sends up to the supervisor's stream."""

    def __init__(self) -> None:
        self.events: Counter[str] = Counter()

    async def on_event(self, ctx: RunContext[Any], event: Any) -> None:
        if event.type == "subagent_event":  # a worker's event, wrapped by as_tool()
            self.events[event.agent] += 1


def digits(text: str) -> str:
    """``text`` with thousands separators dropped, so 25,598 and 25 598 both read 25598."""
    return re.sub(r"(?<=\d)[,.\s](?=\d{3}(?!\d))", "", text)


async def run(ctx: ExampleContext) -> ExampleOutcome:
    researcher = Agent(
        ctx.model,
        name="researcher",
        description="Looks up company facts. Give it the company names and what to find.",
        instructions=(
            "You look up facts with the search tool. Search for every company you are asked "
            "about, then reply with one line per company giving the facts exactly as found. "
            "Do not compute anything."
        ),
        tools=[search],
        limits=Limits(max_turns=6),
    )
    analyst = Agent(
        ctx.model,
        name="analyst",
        description="Does exact arithmetic. Give it the numbers and what to compute.",
        instructions=(
            "You compute with the calculate tool. Call it once with one expression for the "
            "whole computation, then reply with the result. Never compute in your head."
        ),
        tools=[calculate],
        limits=Limits(max_turns=4),
    )
    activity = WorkerActivity()
    supervisor = Agent(
        ctx.model,
        name="supervisor",
        instructions=(
            "You supervise two workers and never research or compute anything yourself.\n"
            "1. Call researcher once, asking for the end-of-2025 headcount of every company in "
            "the question.\n"
            "2. After it has answered, call analyst once with those headcounts and ask for "
            "their exact sum.\n"
            "3. Write a report of at most four sentences that gives each company's headcount "
            "and the analyst's total, in digits. Then stop."
        ),
        tools=[researcher.as_tool(), analyst.as_tool()],
        # The workers have no budget of their own: each starts with what is left of this one.
        limits=Limits(max_turns=8, max_cost_usd=0.30),
        hooks=[*ctx.hooks, activity],
    )

    ctx.log("Asking the supervisor; lines marked [researcher] or [analyst] come from workers")
    result = await ctx.print_stream(supervisor.stream(QUESTION))

    own, total = result.turns, result.usage.requests  # as_tool() adds each worker's usage
    ctx.log(f"The supervisor sent {own} requests; its run counts {total}, workers included")
    ctx.log(f"Worker events seen by the supervisor: {dict(activity.events)}")
    stated = str(TOTAL) in digits(result.text)
    ok = result.status == "completed" and stated
    if ok:
        summary = (
            f"the report states the total {TOTAL:,}; {total} requests ({own} by the supervisor)"
        )
    elif result.status != "completed":
        summary = f"{result.status} ({result.stop_reason}) after {total} requests"
    else:
        summary = f"the report does not state {TOTAL:,}: {result.text[:120]!r}"
    return ExampleOutcome(
        ok=ok,
        summary=summary,
        output=result.text,
        details={
            "status": result.status,
            "stop_reason": result.stop_reason,
            "supervisor_requests": own,
            "total_requests": total,
            "cost_usd": result.usage.cost_usd,
            "worker_events": dict(activity.events),
        },
    )


if __name__ == "__main__":
    main(run, name=NAME, description=DESCRIPTION, inspired_by=INSPIRED_BY)
