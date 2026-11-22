"""Offline tests of examples/patterns/evals_compare.py."""

from __future__ import annotations

import json
import runpy
import sys

import pytest

from examples import harness
from examples.harness import final_answer, main, scripted_context, system_text, tool_names
from examples.patterns import evals_compare as example
from examples.patterns.evals_compare import DATASET, lookup, run
from qagents import ToolError, Usage, prompts
from qagents.evals import EvalReport
from qagents.models.testing import FunctionModel, call

LOOKUPS = {  # what a careful assistant looks up for each case
    "stock": ["KB-200"],
    "price": ["HD-550"],
    "status": ["A-1002"],
    "tracking": ["A-1002"],
    "restock": ["MS-310"],
    "order-in-stock": ["A-1001", "MS-310"],
}


def assistant_policy(reply=lambda case: case.expected):
    """The judge always agrees; the assistant looks things up, then answers with ``reply``."""

    def policy(request):
        if prompts.JUDGE_INSTRUCTIONS in system_text(request):
            verdict = {"reason": "The same value as the reference.", "score": 1.0, "passed": True}
            if "final_answer" in tool_names(request):  # output_mode="tool"
                return final_answer(**verdict)
            return json.dumps(verdict)  # output_mode="native", as on Anthropic
        prompt = next(m.text for m in request.messages if m.role == "user")
        case = next(case for case in DATASET.cases if case.prompt == prompt)
        done = sum(1 for m in request.messages if m.role == "assistant")
        refs = LOOKUPS[case.name]
        return call("lookup", ref=refs[done]) if done < len(refs) else reply(case)

    return policy


def priced(policy) -> FunctionModel:
    return FunctionModel(
        policy, usage=Usage(requests=1, input_tokens=300, output_tokens=5, cost_usd=0.001)
    )


@pytest.fixture
def results(tmp_path, monkeypatch):
    folder = tmp_path / "results"
    monkeypatch.setattr(example, "RESULTS_DIR", folder)
    return folder


async def test_every_case_passes_and_the_report_is_saved(tmp_path, results, capsys):
    ctx = scripted_context(priced(assistant_policy()), tmp_path, provider="qcloud")
    outcome = await run(ctx)
    assert outcome.ok, outcome.summary
    # The judge read the case, the reference and the answer, and answered with its tool.
    judged = [r for r in ctx.meter.requests if prompts.JUDGE_INSTRUCTIONS in system_text(r)]
    assert len(judged) == 6 and all("final_answer" in tool_names(r) for r in judged)
    stock = next(r for r in judged if "KB-200 keyboards" in r.messages[-1].text).messages[-1].text
    assert "<reference>\n14\n</reference>" in stock and "<answer>\n14\n</answer>" in stock
    assert outcome.output["pass_rate"] == 1.0 and outcome.details["regressions"] == {}
    report = EvalReport.load(results / "evals-qcloud.json")
    assert [result.case for result in report.results] == [case.name for case in DATASET.cases]
    assert report.evaluators == [
        "exact_match",
        "contains",
        "tool_called:lookup",
        "never_orders",
        "max_turns",
        "max_cost_usd",
        "llm_judge",
    ]
    two = next(result for result in report.results if result.case == "order-in-stock")
    assert [call.arguments["ref"] for call in two.tool_calls] == ["A-1001", "MS-310"]
    assert two.turns == 3 and two.judge_usage.requests == 1
    assert "## Evaluation: shop assistant (qcloud:function)" in (
        results / "evals-qcloud.md"
    ).read_text(encoding="utf-8")
    printed = capsys.readouterr().out
    assert "  order-in-stock: pass; answered 'no'" in printed


async def test_the_second_provider_prints_the_comparison(tmp_path, results, capsys):
    await run(scripted_context(priced(assistant_policy()), tmp_path / "q", provider="qcloud"))
    wordy = assistant_policy(lambda case: f"The answer is {case.expected}.")
    outcome = await run(scripted_context(priced(wordy), tmp_path / "a", provider="anthropic"))
    # Every value is in the answer, but not alone: ExactMatch fails each case.
    assert not outcome.ok and outcome.output["pass_rate"] == 0.0
    assert outcome.output["mean_scores"]["exact_match"] == 0.0
    assert outcome.output["mean_scores"]["contains"] == 1.0
    assert outcome.details["regressions"] == {"qcloud": [case.name for case in DATASET.cases]}
    printed = capsys.readouterr().out
    assert "Regressions against qcloud: stock, price, status, tracking, restock" in printed
    assert "| Metric | qcloud | anthropic | Change |" in printed
    assert "| Pass rate | 100.0% | 0.0% | -100.0 pp |" in printed
    assert "| stock | pass | fail | regressed |" in printed
    assert "  stock: fail (exact_match); answered 'The answer is 14.'" in printed


async def test_ordering_instead_of_answering_fails_the_case(tmp_path, results):
    def eager(request):
        prompt = next(m.text for m in request.messages if m.role == "user")
        if "KB-200" in prompt and not any(m.role == "assistant" for m in request.messages):
            return call("place_order", sku="KB-200", quantity=1)  # pauses for an approval
        return assistant_policy()(request)

    outcome = await run(scripted_context(priced(eager), tmp_path))
    assert outcome.ok and outcome.details["failed"] == ["stock"]  # 5 of 6 still pass
    report = EvalReport.load(results / "evals-scripted.json")
    stock = report.results[0]
    assert stock.status == "paused" and stock.scores["never_orders"] == 0.0


def test_lookup():
    assert lookup.function("kb-200") == "KB-200: Mechanical keyboard, price 89.00 EUR, 14 in stock"
    assert lookup.function("MS-310").endswith("0 in stock, back in stock on 2026-12-01")
    assert lookup.function("A-1002") == (
        "Order A-1002: 1 x KB-200, status shipped, tracking number TRK-55821"
    )
    assert lookup.function("order a-1002") == lookup.function("A-1002")
    with pytest.raises(ToolError, match="No product or order 'ZZ-1'"):
        lookup.function("ZZ-1")


def test_runs_as_a_module_and_replays_from_its_cassette(tmp_path, monkeypatch, capsys):
    folder = tmp_path / "results"
    monkeypatch.setattr(harness, "RESULTS_DIR", folder)  # read when runpy imports the module
    monkeypatch.setattr(example, "RESULTS_DIR", folder)
    common = ["--provider", "anthropic", "--cassette-dir", str(tmp_path / "cassettes"), "--json"]
    with pytest.raises(SystemExit) as recorded:
        main(
            run,
            name=example.NAME,
            description=example.DESCRIPTION,
            inspired_by=example.INSPIRED_BY,
            argv=[*common, "--record", "--workdir", str(tmp_path / "recorded")],
            model_factory=lambda provider, model: priced(assistant_policy()),
        )
    assert recorded.value.code == 0
    recorded_report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    argv = [example.__file__, *common, "--replay", "--workdir", str(tmp_path / "replayed")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as replayed:
        runpy.run_path(example.__file__, run_name="__main__")
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert replayed.value.code == 0, report
    assert report["ok"] and report["requests"] == recorded_report["requests"] == 19  # 13 + 6 judge
    # The replay wrote its own report and compared nothing: no other provider's report exists.
    assert sorted(path.name for path in folder.iterdir()) == [
        "evals-anthropic-replay.json",
        "evals-anthropic-replay.md",
        "evals-anthropic.json",
        "evals-anthropic.md",
    ]
