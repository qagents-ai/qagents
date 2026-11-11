"""Offline tests of examples/patterns/stock_analysis.py."""

from __future__ import annotations

import asyncio
import json
import runpy
import sys

import pytest

from examples.harness import (
    final_answer,
    last_text,
    main,
    scripted_context,
    system_text,
    tool_names,
)
from examples.patterns import stock_analysis
from examples.patterns.stock_analysis import (
    ANALYSTS,
    DISCLAIMER,
    TICKERS,
    Recommendation,
    compute_price_metrics,
    compute_ratios,
    run,
)
from qagents.models.testing import FunctionModel, call

RISKS = {"HLX": "A supplier delay may slow deliveries.", "MRDN": "Lower revenue guidance."}
CORRECTIONS = ("disclaimer field must contain", "exactly one pick for each")
"""Parts of the corrections the two output validators send the model."""


def pick(ticker: str, rating: str, risk: str | None = None) -> dict:
    prices, ratios = compute_price_metrics(ticker), compute_ratios(ticker)
    return {
        "ticker": ticker,
        "rating": rating,
        "total_return_pct": prices["total_return_pct"],
        "volatility_pct": prices["volatility_pct"],
        "pe_ratio": ratios["pe_ratio"],
        "main_risk": risk or RISKS[ticker],
    }


def corrections(request) -> list[str]:
    """Validator corrections the advisor received: tool errors ("tool" mode) or user
    messages ("native" mode)."""
    texts = [m.text for m in request.messages if m.role == "user"]
    texts += [r.content for m in request.messages for r in m.tool_results if r.is_error]
    return [text for text in texts if any(phrase in text for phrase in CORRECTIONS)]


def crew_policy(
    ratings: dict[str, str],
    seen: dict,
    *,
    native: bool = False,
    flaw: str = "disclaimer",
    consult: tuple[str, ...] = ANALYSTS,
    risks: dict[str, str] | None = None,
):
    """The analysts call their tools for both tickers; the advisor delegates, then answers.

    The advisor's first answer has a ``flaw`` ("disclaimer" or "one_pick") that an output
    validator rejects; it answers correctly once it has the correction. When both analysts
    are consulted, each one's first request waits until the other's has arrived, which
    only works if the two sub-agent runs are in flight at the same time.
    """
    arrived: set[str] = set()
    both = asyncio.Event()

    async def barrier(name: str) -> None:
        arrived.add(name)
        if arrived == set(ANALYSTS):
            both.set()
        await asyncio.wait_for(both.wait(), timeout=2)  # raises when the runs are sequential
        seen["overlapped"].add(name)

    def answer(**fields):
        return json.dumps(fields) if native else final_answer(**fields)

    async def policy(request):
        system, last = system_text(request), request.messages[-1]
        if system.startswith("You are a financial analyst"):
            if last.role == "user":
                seen["financial_task"].append(last.text)
                if consult == ANALYSTS:
                    await barrier("financial_analyst")
                names = ("price_metrics", "financial_ratios")
                return [call(name, ticker=ticker) for ticker in TICKERS for name in names]
            return "Metrics: " + last_text(request)
        if system.startswith("You are a research analyst"):
            if last.role == "user":
                if consult == ANALYSTS:
                    await barrier("research_analyst")
                return [call("company_news", ticker=ticker) for ticker in TICKERS]
            return "HLX: supplier delay. MRDN: recall and lower guidance."
        assert system.startswith("You are an investment advisor")
        seen["advisor_tools"].append(tool_names(request))
        if len(request.messages) == 2:  # system and prompt
            return [call(name, task=f"Analyse HLX and MRDN ({name})") for name in consult]
        seen["corrections"] = corrections(request)
        picks = [
            pick(ticker, rating, (risks or {}).get(ticker)) for ticker, rating in ratings.items()
        ]
        disclaimer = DISCLAIMER
        if not seen["corrections"]:
            if flaw == "disclaimer":
                disclaimer = "Invest wisely."
            else:
                picks = picks[:1]
        return answer(
            picks=picks,
            preferred="HLX",
            summary="HLX has the stronger trend. MRDN is falling.",
            disclaimer=disclaimer,
        )

    return policy


def new_seen() -> dict:
    return {"financial_task": [], "advisor_tools": [], "corrections": [], "overlapped": set()}


@pytest.mark.parametrize("mode", ["tool", "native"])
@pytest.mark.parametrize("flaw", ["disclaimer", "one_pick"])
async def test_the_advisor_recommends_by_the_metrics_after_one_correction(tmp_path, mode, flaw):
    seen = new_seen()
    policy = crew_policy({"HLX": "buy", "MRDN": "sell"}, seen, native=mode == "native", flaw=flaw)
    ctx = scripted_context(policy, tmp_path, output_mode=mode)

    outcome = await run(ctx)

    assert outcome.ok, outcome.summary
    answer = Recommendation.model_validate(outcome.output)
    assert {p.ticker: p.rating for p in answer.picks} == {"HLX": "buy", "MRDN": "sell"}
    assert answer.disclaimer == DISCLAIMER
    # the validator rejected the first answer and said what to correct
    assert len(seen["corrections"]) == 1
    expected = DISCLAIMER if flaw == "disclaimer" else "HLX, MRDN, not ['HLX']"
    assert expected in seen["corrections"][0]
    tools = set(ANALYSTS) | ({"final_answer"} if mode == "tool" else set())
    assert seen["advisor_tools"][0] == tools
    assert seen["financial_task"] == ["Analyse HLX and MRDN (financial_analyst)"]
    assert seen["overlapped"] == set(ANALYSTS)  # both sub-agent runs were in flight at once
    assert set(ANALYSTS) <= set(outcome.details["consulted"])
    # advisor: delegate, rejected answer, answer; each analyst: tool calls, answer
    assert ctx.meter.usage.requests == 7


async def test_a_rating_that_breaks_the_rule_fails_the_example(tmp_path):
    ctx = scripted_context(crew_policy({"HLX": "buy", "MRDN": "hold"}, new_seen()), tmp_path)

    outcome = await run(ctx)

    assert not outcome.ok
    assert outcome.details["problems"] == ["MRDN rated hold, not sell"]


async def test_skipping_the_research_analyst_fails_the_example(tmp_path):
    policy = crew_policy(
        {"HLX": "buy", "MRDN": "sell"},
        new_seen(),
        consult=("financial_analyst",),
        risks={"MRDN": "The share price keeps falling."},
    )

    outcome = await run(scripted_context(policy, tmp_path))

    assert not outcome.ok
    assert outcome.details["problems"] == [
        "research_analyst did not answer",
        "MRDN main_risk is not from the news: 'The share price keeps falling.'",
    ]


def test_the_tools_report_the_fixture_metrics():
    assert compute_price_metrics("HLX") == {
        "last_close": 69.0,
        "total_return_pct": 38.0,
        "volatility_pct": 10.9,
    }
    assert compute_price_metrics("MRDN")["total_return_pct"] == -12.0
    assert compute_ratios("MRDN")["pe_ratio"] == 17.2
    assert compute_ratios("HLX")["debt_to_equity"] == 0.33


def test_runs_as_a_module_and_replays_from_its_cassette(tmp_path, monkeypatch, capsys):
    common = ["--provider", "anthropic", "--cassette-dir", str(tmp_path / "cassettes"), "--json"]
    with pytest.raises(SystemExit) as recorded:
        main(
            run,
            name=stock_analysis.NAME,
            description=stock_analysis.DESCRIPTION,
            inspired_by=stock_analysis.INSPIRED_BY,
            argv=[*common, "--record", "--workdir", str(tmp_path / "recorded")],
            model_factory=lambda provider, model: FunctionModel(
                crew_policy({"HLX": "buy", "MRDN": "sell"}, new_seen(), native=True)
            ),
        )
    assert recorded.value.code == 0
    capsys.readouterr()

    replay = [*common, "--replay", "--workdir", str(tmp_path / "replayed")]
    monkeypatch.setattr(sys, "argv", ["stock_analysis", *replay])
    with pytest.raises(SystemExit) as replayed:
        runpy.run_path(stock_analysis.__file__, run_name="__main__")
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert replayed.value.code == 0, report
    assert report["ok"] and report["name"] == "stock_analysis" and report["requests"] == 7
