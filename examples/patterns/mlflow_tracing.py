"""A supervisor and two workers traced in MLflow, then read back and printed as a span tree.

The supervisor delegates to a weather desk and a money desk, two sub-agents used as tools. One
``MLflowHooks(record_content=True)`` on the supervisor traces the whole tree, because it
propagates to child runs: an AGENT span per run, a CHAT_MODEL span per model request (messages,
token usage, cost) and a TOOL span per tool call, with each worker's run nested under the tool
call that started it. The trace goes to a SQLite tracking store in the work directory, which
needs no server. After the run the example flushes the exporter, finds the trace with
``mlflow.search_traces``, loads it with ``mlflow.get_trace``, prints the tree, and prints the
command that opens the MLflow UI on that store.

Mirrors: tracing a LangGraph or CrewAI multi-agent run in MLflow with autologging
(``mlflow.langchain.autolog()``, ``mlflow.crewai.autolog()``). In qagents nothing is patched:
tracing is one hook object on the top agent, the spans follow qagents' own structure (runs,
model requests, tool calls, sub-agent links, usage and cost from the provider), and message
content is recorded only when you ask for it.

Run it (with ``--mlflow`` the harness's tracer and store are used instead of a new store):

    python -m examples.patterns.mlflow_tracing --provider anthropic
    python -m examples.patterns.mlflow_tracing --provider qcloud
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

from examples.harness import MLFLOW_EXPERIMENT, ExampleContext, ExampleOutcome, SetupError, main
from qagents import Agent, Limits, ToolError, tool
from qagents.integrations.mlflow import ATTR, MLflowHooks  # MLflow is imported when used

NAME = "mlflow_tracing"
DESCRIPTION = "A supervisor with two workers traced in MLflow, read back as a span tree"
INSPIRED_BY = "LangGraph / CrewAI multi-agent runs traced with MLflow autolog"

FORECASTS = {"lisbon": "Lisbon, Friday: 19 C, light rain in the afternoon, wind 15 km/h."}
RATES = {"EUR": 1.0, "USD": 1.08, "GBP": 0.86}  # units per euro
QUESTION = (
    "I fly to Lisbon on Friday with 300 EUR in cash. What weather should I expect there, and "
    "how much is my cash in US dollars?"
)


@tool(read_only=True)
def forecast(city: str) -> str:
    """The weather forecast for a city.

    Args:
        city: The city, for example "Lisbon".
    """
    known = [text for name, text in FORECASTS.items() if name in city.lower()]  # "Lisbon, PT"
    if not known:
        raise ToolError(f"No forecast for {city!r}. Known cities: Lisbon.")
    return known[0]


@tool(read_only=True)
def convert(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount of money at today's rate.

    Args:
        amount: The amount to convert.
        from_currency: ISO code of its currency, for example "EUR".
        to_currency: ISO code to convert to, for example "USD".
    """
    codes = from_currency.upper(), to_currency.upper()
    if not all(code in RATES for code in codes):
        raise ToolError(f"Unknown currency; use one of {', '.join(RATES)}.")
    converted = amount / RATES[codes[0]] * RATES[codes[1]]
    return f"{amount:.2f} {codes[0]} = {converted:.2f} {codes[1]}"


def build_supervisor(ctx: ExampleContext, hooks: list[Any]) -> Agent:
    weather = Agent(
        ctx.model,
        name="weather_desk",
        description="Gives the weather forecast for a city. Name the city and the day.",
        instructions="Call forecast once for the city you are asked about, then reply with "
        "the forecast in one sentence.",
        tools=[forecast],
        limits=Limits(max_turns=3),
    )
    money = Agent(
        ctx.model,
        name="money_desk",
        description="Converts an amount of money. Give the amount and both currencies.",
        instructions="Call convert once, then reply with the converted amount.",
        tools=[convert],
        limits=Limits(max_turns=3),
    )
    return Agent(
        ctx.model,
        name="supervisor",
        instructions=(
            "You run a travel desk and never answer from your own knowledge. Ask weather_desk "
            "for the forecast and money_desk for the conversion, both in the same turn. Then "
            "answer in two sentences with the forecast and the amount in US dollars."
        ),
        tools=[weather.as_tool(read_only=True), money.as_tool(read_only=True)],
        limits=Limits(max_turns=4, max_cost_usd=0.20),
        hooks=hooks,  # the workers' runs inherit the tracer: it propagates to child runs
    )


def import_mlflow() -> Any:
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")  # no calls home
    os.environ.setdefault("DO_NOT_TRACK", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    try:
        import mlflow
    except ImportError as error:
        message = f'This example needs MLflow: pip install -e ".[mlflow]" ({error}).'
        raise SetupError(message) from error
    return mlflow


def describe(span: Any) -> str:
    """One line for a span: its name, its type, and the usage and cost it recorded."""
    from mlflow.tracing.constant import SpanAttributeKey

    line = f"{span.name}  [{span.span_type}]"
    tokens = span.get_attribute(SpanAttributeKey.CHAT_USAGE)  # CHAT_MODEL spans: one request
    if tokens:
        line += f"  {tokens['input_tokens']:,} in / {tokens['output_tokens']:,} out tokens"
    usage = span.get_attribute(ATTR["usage"])  # AGENT spans: the whole run, workers included
    if usage:
        line += f"  {usage['requests']} requests, {usage['input_tokens']:,} in tokens"
    cost = span.get_attribute(ATTR["cost"])
    if cost is not None:
        line += f"  ${cost:.4f}"
    status = span.get_attribute(ATTR["tool_status"])  # TOOL spans: ok or error
    if status:
        line += f"  {status}"
    return line


def print_tree(spans: list[Any]) -> None:
    children: dict[str | None, list[Any]] = defaultdict(list)
    for span in spans:
        children[span.parent_id].append(span)

    def show(span: Any, depth: int) -> None:
        print(f"{'    ' * depth}{describe(span)}")
        for child in sorted(children[span.span_id], key=lambda item: item.start_time_ns):
            show(child, depth + 1)

    for root in children[None]:
        show(root, 0)


def own_tracer(mlflow: Any, ctx: ExampleContext) -> MLflowHooks:
    """Point MLflow at a SQLite file in the work directory and trace into an experiment there."""
    mlflow.set_tracking_uri(f"sqlite:///{(ctx.workdir / 'mlflow.db').as_posix()}")
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else mlflow.create_experiment(
            MLFLOW_EXPERIMENT, artifact_location=(ctx.workdir / "artifacts").as_uri()
        )
    )
    return MLflowHooks(
        record_content=True, experiment_id=experiment_id, tags={"qagents.example": NAME}
    )


def shown(uri: str) -> str:
    """The tracking URI, unless it is a server address (which may carry credentials)."""
    return uri if uri.startswith(("sqlite:", "file:")) else "your MLflow tracking server"


async def run(ctx: ExampleContext) -> ExampleOutcome:
    mlflow = import_mlflow()
    from mlflow.entities import SpanType

    previous = mlflow.get_tracking_uri()
    try:
        # With --mlflow the harness has a tracer and a store already; otherwise make both.
        tracer = next((hook for hook in ctx.hooks if isinstance(hook, MLflowHooks)), None)
        if tracer is None:
            tracer = own_tracer(mlflow, ctx)
        store = mlflow.get_tracking_uri()
        supervisor = build_supervisor(ctx, hooks=[*ctx.hooks, tracer])  # each hook runs once

        ctx.log(f"Tracing into {shown(store)}")
        result = await ctx.print_stream(supervisor.stream(QUESTION))
        tracer.flush()  # MLflow exports traces in a background thread

        location = [tracer.experiment_id] if tracer.experiment_id else None
        found = mlflow.search_traces(locations=location, return_type="list")
        ids = [trace.info.trace_id for trace in found]
        if tracer.last_trace_id not in ids:
            return ExampleOutcome(ok=False, summary=f"the trace is not in the store: {ids}")
        trace = mlflow.get_trace(tracer.last_trace_id)
    finally:
        mlflow.set_tracking_uri(previous)  # leave MLflow pointing where it was

    spans = trace.data.spans
    ctx.log(f"Trace {trace.info.trace_id} has {len(spans)} spans:")
    print_tree(spans)
    tokens = (trace.info.token_usage or {}).get("total_tokens", 0)
    print(f"Trace total: {tokens:,} tokens; status tag {trace.info.tags.get(ATTR['status'])!r}")
    if store.startswith("sqlite:"):
        print(f"Open it in the MLflow UI:\n    mlflow ui --backend-store-uri {store}")

    by_id = {span.span_id: span for span in spans}
    roots = [span for span in spans if span.parent_id is None]
    top = roots[0] if len(roots) == 1 else None
    supervisor_span = top is not None and top.span_type == SpanType.AGENT
    supervisor_span = supervisor_span and top.name == "invoke_agent supervisor"
    chats = sum(span.span_type == SpanType.CHAT_MODEL for span in spans)
    nested = sorted(  # worker runs whose parent span is the supervisor's tool call
        span.name.removeprefix("invoke_agent ")
        for span in spans
        if span.span_type == SpanType.AGENT
        and span.parent_id in by_id
        and by_id[span.parent_id].span_type == SpanType.TOOL
    )
    both = {"weather_desk", "money_desk"} <= set(nested)
    ok = result.status == "completed" and supervisor_span and chats > 0 and both
    summary = (
        f"{len(spans)} spans: the supervisor's AGENT span, {chats} CHAT_MODEL spans, and "
        f"{' and '.join(nested)} nested under their TOOL spans"
        if ok
        else f"{result.status}: supervisor span {supervisor_span}, {chats} chat spans, "
        f"nested workers {nested}"
    )
    return ExampleOutcome(
        ok=ok,
        summary=summary,
        output=result.text,
        details={
            "trace_id": trace.info.trace_id,
            "tracking_uri": shown(store),
            "spans": [(span.name, span.span_type) for span in spans],
            "answer_has_amount": "324" in result.text,
        },
    )


if __name__ == "__main__":
    main(run, name=NAME, description=DESCRIPTION, inspired_by=INSPIRED_BY)
