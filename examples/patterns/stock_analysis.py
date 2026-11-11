"""Stock analysis: two analyst sub-agents and an advisor with a checked, structured answer.

A financial analyst computes returns, volatility and ratios with tools over fixture prices and
statements; a research analyst reads fixture news. Both are sub-agents of an investment
advisor, which rates each ticker by a stated rule and returns a typed `Recommendation`. Two
output validators guard the answer: one demands the risk disclaimer, one a pick per ticker.
The example then checks the recommendation against metrics it computes itself.

Mirrors: the CrewAI stock analysis crew. The pieces map like this:

- the financial analyst, research analyst and investment advisor (role, goal, backstory)
  -> three `Agent`s with short instructions
- their web, SEC filing and calculator tools -> `tool` functions over fixtures, so every
  number is computed in code, not by the model
- sequential tasks that hand each output to the next as context -> the advisor calls the
  analysts as tools (``agent.as_tool()``); they run in parallel as child runs, each in its
  own short conversation, and the advisor sees only their answers
- the last task's ``output_pydantic`` and its task guardrail -> ``output_type=Recommendation``
  and ``output_validators`` that raise `ModelRetry` with the correction

In qagents the advisor, not a fixed task list, decides what to ask, and each analyst run
stops at what remains of the advisor's cost and time budget.

Not financial advice: the tickers, prices, statements and news are invented.

Run it:

    python -m examples.patterns.stock_analysis --provider anthropic
    python -m examples.patterns.stock_analysis --provider qcloud
"""

from __future__ import annotations

import math
import statistics
from itertools import pairwise
from typing import Literal

from pydantic import BaseModel, Field

from examples.harness import ExampleContext, ExampleOutcome, main
from qagents import Agent, Limits, ModelRetry, RunContext, tool

NAME = "stock_analysis"
DESCRIPTION = "Analyst sub-agents feed an advisor whose structured recommendation is validated"
INSPIRED_BY = "CrewAI stock analysis crew (financial analyst, research analyst, investment advisor)"

Ticker = Literal["HLX", "MRDN"]
TICKERS: tuple[str, ...] = ("HLX", "MRDN")
ANALYSTS = ("financial_analyst", "research_analyst")
DISCLAIMER = (
    "This is not financial advice. Past performance does not guarantee future results, "
    "and you can lose money investing."
)

# -- fixtures: two fictional companies ---------------------------------------------------------

MONTHLY_CLOSES = {  # 13 month-end closes, oldest first
    "HLX": [50.0, 52.1, 51.3, 54.8, 57.2, 55.9, 58.4, 61.0, 59.7, 63.5, 65.2, 66.8, 69.0],
    "MRDN": [40.0, 41.2, 39.5, 38.1, 39.0, 36.2, 35.4, 37.0, 34.8, 33.9, 35.1, 34.0, 35.2],
}
STATEMENTS = {  # last two fiscal years, in millions (shares: millions of shares)
    "HLX": dict(revenue=1200, revenue_prior=1000, net_income=174, shares=60, debt=300, equity=900),
    "MRDN": dict(revenue=2050, revenue_prior=2100, net_income=82, shares=40, debt=1400, equity=700),
}
NEWS = {
    "HLX": [
        "2026-09-14 Helix Robotics wins a five-year warehouse automation contract",
        "2026-10-02 Helix opens a second factory to double output",
        "2026-10-28 A supplier delay may slow Helix deliveries next quarter",
    ],
    "MRDN": [
        "2026-08-30 Meridian Foods recalls one snack line over labelling",
        "2026-10-05 Meridian cuts its full-year revenue guidance",
        "2026-10-21 Meridian names a new chief financial officer",
    ],
}
NEWS_TERMS = {  # words of each company's headlines: a risk taken from the news uses one
    "HLX": ("supplier", "supply", "delay", "deliver", "factory", "contract"),
    "MRDN": ("recall", "label", "guidance", "revenue", "outlook", "financial officer", "cfo"),
}


def compute_price_metrics(ticker: str) -> dict[str, float]:
    closes = MONTHLY_CLOSES[ticker]
    monthly = [later / earlier - 1 for earlier, later in pairwise(closes)]
    return {
        "last_close": closes[-1],
        "total_return_pct": round((closes[-1] / closes[0] - 1) * 100, 1),
        "volatility_pct": round(statistics.stdev(monthly) * math.sqrt(12) * 100, 1),
    }


def compute_ratios(ticker: str) -> dict[str, float]:
    s = STATEMENTS[ticker]
    eps = s["net_income"] / s["shares"]
    return {
        "pe_ratio": round(MONTHLY_CLOSES[ticker][-1] / eps, 1),
        "net_margin_pct": round(s["net_income"] / s["revenue"] * 100, 1),
        "revenue_growth_pct": round((s["revenue"] / s["revenue_prior"] - 1) * 100, 1),
        "debt_to_equity": round(s["debt"] / s["equity"], 2),
    }


RATING_RULE = (
    "sell when the total return is below 0; buy when the total return divided by the "
    "volatility is at least 1 and the P/E ratio is below 30; hold otherwise"
)


def expected_rating(ticker: str) -> str:
    """`RATING_RULE` in code, for checking the advisor."""
    prices, ratios = compute_price_metrics(ticker), compute_ratios(ticker)
    if prices["total_return_pct"] < 0:
        return "sell"
    risk_adjusted = prices["total_return_pct"] / prices["volatility_pct"]
    return "buy" if risk_adjusted >= 1 and ratios["pe_ratio"] < 30 else "hold"


def expected_preference() -> str:
    def risk_adjusted(ticker: str) -> float:
        prices = compute_price_metrics(ticker)
        return prices["total_return_pct"] / prices["volatility_pct"]

    return max(TICKERS, key=risk_adjusted)


# -- tools ---------------------------------------------------------------------------------------


@tool(read_only=True)
def price_metrics(ticker: Ticker) -> dict[str, float]:
    """Price metrics over the last 12 months: last close, total return and annualised
    volatility of monthly returns, both in percent.

    Args:
        ticker: The stock ticker.
    """
    return compute_price_metrics(ticker)


@tool(read_only=True)
def financial_ratios(ticker: Ticker) -> dict[str, float]:
    """Ratios from the latest annual statements: P/E, net margin and revenue growth in
    percent, and debt to equity.

    Args:
        ticker: The stock ticker.
    """
    return compute_ratios(ticker)


@tool(read_only=True)
def company_news(ticker: Ticker) -> list[str]:
    """Recent headlines about the company, newest last, each starting with its date.

    Args:
        ticker: The stock ticker.
    """
    return NEWS[ticker]


# -- the answer and its validators ------------------------------------------------------------


class Pick(BaseModel):
    ticker: Ticker
    rating: Literal["buy", "hold", "sell"]
    total_return_pct: float
    volatility_pct: float
    pe_ratio: float
    main_risk: str = Field(description="The main risk from the news, in one sentence.")


class Recommendation(BaseModel):
    picks: list[Pick] = Field(description="One pick per ticker.")
    preferred: Ticker = Field(description="The ticker with the higher return / volatility.")
    summary: str = Field(description="Two sentences for the investor.")
    disclaimer: str


def require_disclaimer(ctx: RunContext, answer: Recommendation) -> None:
    text = answer.disclaimer.lower()
    if "not financial advice" not in text or "past performance" not in text:
        raise ModelRetry(f"The disclaimer field must contain this text word for word: {DISCLAIMER}")


def one_pick_per_ticker(ctx: RunContext, answer: Recommendation) -> None:
    given = sorted(pick.ticker for pick in answer.picks)
    if given != sorted(TICKERS):
        raise ModelRetry(f"Give exactly one pick for each of {', '.join(TICKERS)}, not {given}.")


# -- the crew -------------------------------------------------------------------------------------


def build_advisor(ctx: ExampleContext) -> Agent:
    analyst_limits = Limits(max_turns=6)  # children also stop at the advisor's remaining budget
    financial = Agent(
        ctx.model,
        name="financial_analyst",
        description="Reports price metrics and financial ratios of the tickers in the task.",
        instructions="You are a financial analyst. For every ticker in the task, call "
        "price_metrics and financial_ratios. Then list every number exactly as the tools "
        "returned it, one line per ticker. Give no advice.",
        tools=[price_metrics, financial_ratios],
        limits=analyst_limits,
    )
    research = Agent(
        ctx.model,
        name="research_analyst",
        description="Summarises recent news of the tickers in the task.",
        instructions="You are a research analyst. For every ticker in the task, call "
        "company_news once. Then give, per ticker, the main risk and the main positive in "
        "the news, in one short sentence each.",
        tools=[company_news],
        limits=analyst_limits,
    )
    return Agent(
        ctx.model,
        name="advisor",
        instructions="You are an investment advisor. First ask financial_analyst for the "
        "metrics of HLX and MRDN in one task, and research_analyst for their news in one task. "
        "Then write the recommendation: one pick per ticker with the numbers copied exactly "
        "from the financial analyst and the main risk from the research analyst. Rate each "
        f"ticker by this rule: {RATING_RULE}. Prefer the ticker with the higher total return "
        "divided by volatility. Put this disclaimer in the disclaimer field, word for word: "
        f"{DISCLAIMER}",
        tools=[financial.as_tool(read_only=True), research.as_tool(read_only=True)],
        output_type=Recommendation,
        output_mode=ctx.output_mode,
        output_validators=[require_disclaimer, one_pick_per_ticker],
        # A sub-agent call is one tool call of the advisor: allow it more than the 120 s default.
        limits=Limits(max_turns=8, max_cost_usd=0.40, tool_timeout_s=300),
        hooks=ctx.hooks,
    )


def disagreements(answer: Recommendation, consulted: set[str]) -> list[str]:
    """Where the recommendation contradicts the computed metrics or skipped an analyst."""
    problems = [f"{name} did not answer" for name in ANALYSTS if name not in consulted]
    for pick in answer.picks:
        prices, ratios = compute_price_metrics(pick.ticker), compute_ratios(pick.ticker)
        for field, actual in (
            ("total_return_pct", prices["total_return_pct"]),
            ("volatility_pct", prices["volatility_pct"]),
            ("pe_ratio", ratios["pe_ratio"]),
        ):
            if abs(getattr(pick, field) - actual) > 0.25:
                problems.append(f"{pick.ticker} {field} {getattr(pick, field)} != {actual}")
        if pick.rating != expected_rating(pick.ticker):
            problems.append(
                f"{pick.ticker} rated {pick.rating}, not {expected_rating(pick.ticker)}"
            )
        if not any(term in pick.main_risk.lower() for term in NEWS_TERMS[pick.ticker]):
            problems.append(f"{pick.ticker} main_risk is not from the news: {pick.main_risk!r}")
    if answer.preferred != expected_preference():
        problems.append(f"preferred {answer.preferred}, expected {expected_preference()}")
    return problems


async def run(ctx: ExampleContext) -> ExampleOutcome:
    advisor = build_advisor(ctx)
    ctx.log("The advisor delegates to both analysts, then answers with a Recommendation")
    result = await ctx.print_stream(
        advisor.stream("Compare HLX and MRDN for a long-term investor and recommend one.")
    )
    if not result.completed or not isinstance(result.output, Recommendation):
        return ExampleOutcome(ok=False, summary=f"No recommendation: {result.stop_reason}")
    answer = result.output
    # An analyst counts once a call to it returned an answer, not an error (a limit, a timeout).
    consulted = {
        part.name
        for message in result.new_messages
        for part in message.tool_results
        if not part.is_error
    }
    problems = disagreements(answer, consulted)
    ratings = ", ".join(f"{pick.ticker} {pick.rating}" for pick in answer.picks)
    summary = f"{ratings}; prefers {answer.preferred}" + (
        f"; problems: {'; '.join(problems)}" if problems else "; consistent with the metrics"
    )
    return ExampleOutcome(
        ok=not problems,  # the validators already guarantee the disclaimer and the picks
        summary=summary,
        output=answer.model_dump(),
        details={"problems": problems, "consulted": sorted(consulted), "turns": result.turns},
    )


if __name__ == "__main__":
    main(run, name=NAME, description=DESCRIPTION, inspired_by=INSPIRED_BY)
