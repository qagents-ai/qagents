"""Offline tests of examples/patterns/mlflow_tracing.py: a scripted model, a real MLflow store."""

from __future__ import annotations

import json
import runpy
import sys

import pytest

mlflow = pytest.importorskip("mlflow")
pytest.importorskip("sqlalchemy")

from examples.harness import last_text, main, scripted_context, tool_names  # noqa: E402
from examples.patterns import mlflow_tracing as example  # noqa: E402
from examples.patterns.mlflow_tracing import convert, forecast, run  # noqa: E402
from qagents import ToolError, Usage  # noqa: E402
from qagents.models.testing import FunctionModel, call  # noqa: E402


def policy(request):
    """Plays the supervisor and both desks; the tools offered tell which one is asking."""
    tools = tool_names(request)
    answered = request.messages[-1].role == "tool"
    if tools == {"forecast"}:
        return last_text(request) if answered else call("forecast", city="Lisbon")
    if tools == {"convert"}:
        if answered:
            return last_text(request)
        return call("convert", amount=300, from_currency="EUR", to_currency="USD")
    assert tools == {"weather_desk", "money_desk"}, tools
    if answered:
        return "Expect 19 C and light rain in Lisbon on Friday. Your 300 EUR are 324.00 USD."
    return [
        call("weather_desk", task="The forecast for Lisbon on Friday."),
        call("money_desk", task="Convert 300 EUR to USD."),
    ]


def priced() -> FunctionModel:
    return FunctionModel(
        policy, usage=Usage(requests=1, input_tokens=200, output_tokens=20, cost_usd=0.001)
    )


@pytest.fixture(autouse=True)
def keep_mlflow_configuration():
    """The example must leave MLflow's tracking URI as it found it."""
    before = mlflow.get_tracking_uri()
    yield
    assert mlflow.get_tracking_uri() == before


async def test_the_trace_tree_nests_each_worker_under_its_tool_call(tmp_path, capsys):
    ctx = scripted_context(priced(), tmp_path)
    outcome = await run(ctx)
    assert outcome.ok, outcome.summary
    assert outcome.details["answer_has_amount"]
    assert (tmp_path / "work" / "mlflow.db").is_file()
    spans = outcome.details["spans"]
    assert ("invoke_agent supervisor", "AGENT") in spans
    assert [kind for _, kind in spans].count("CHAT_MODEL") == 6  # 2 by the supervisor, 2 per desk
    assert {("weather_desk", "TOOL"), ("invoke_agent money_desk", "AGENT")} <= set(spans)

    printed = capsys.readouterr().out
    lines = printed.splitlines()
    root = next(i for i, line in enumerate(lines) if line.startswith("invoke_agent supervisor"))
    assert "[AGENT]  6 requests, 1,200 in tokens  $0.0060" in lines[root]
    tool_line = next(
        i for i, line in enumerate(lines) if line.startswith("    weather_desk  [TOOL]")
    )
    assert lines[tool_line + 1].startswith("        invoke_agent weather_desk  [AGENT]")
    assert "            forecast  [TOOL]  ok" in printed
    assert "chat function  [CHAT_MODEL]  200 in / 20 out tokens  $0.0010" in printed
    assert "Trace total: 1,320 tokens; status tag 'completed'" in printed
    assert f"mlflow ui --backend-store-uri {outcome.details['tracking_uri']}" in printed


async def test_with_the_harness_tracer_the_trace_goes_to_its_store(tmp_path, capsys):
    """--mlflow: the harness's MLflowHooks and tracking URI are used, no second store is made."""
    from qagents.integrations.mlflow import MLflowHooks

    uri = f"sqlite:///{(tmp_path / 'harness.db').as_posix()}"
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    try:
        experiment_id = mlflow.create_experiment(
            "harness", artifact_location=(tmp_path / "artifacts").as_uri()
        )
        tracer = MLflowHooks(
            record_content=True, experiment_id=experiment_id, tags={"qagents.example": "x"}
        )
        ctx = scripted_context(priced(), tmp_path)
        ctx.hooks.append(tracer)
        outcome = await run(ctx)
        assert outcome.ok, outcome.summary
        assert outcome.details["tracking_uri"] == uri
        assert outcome.details["trace_id"] == tracer.last_trace_id
        assert not (tmp_path / "work" / "mlflow.db").exists()
        (trace,) = mlflow.search_traces(locations=[experiment_id], return_type="list")
        assert trace.info.tags["qagents.example"] == "x"
        # The tracer ran once per run although the supervisor lists it twice.
        assert [span.name for span in trace.data.spans].count("invoke_agent supervisor") == 1
    finally:
        mlflow.set_tracking_uri(previous)
    assert f"mlflow ui --backend-store-uri {uri}" in capsys.readouterr().out


def test_desk_tools():
    assert forecast.function("lisbon").startswith("Lisbon, Friday: 19 C")
    assert forecast.function("Lisbon, Portugal") == forecast.function("Lisbon")
    with pytest.raises(ToolError, match="Known cities"):
        forecast.function("Oslo")
    assert convert.function(300, "eur", "usd") == "300.00 EUR = 324.00 USD"
    with pytest.raises(ToolError, match="Unknown currency"):
        convert.function(1, "EUR", "JPY")


def test_runs_as_a_module_and_replays_from_its_cassette(tmp_path, monkeypatch, capsys):
    common = ["--provider", "anthropic", "--cassette-dir", str(tmp_path / "cassettes"), "--json"]
    with pytest.raises(SystemExit) as recorded:
        main(
            run,
            name=example.NAME,
            description=example.DESCRIPTION,
            inspired_by=example.INSPIRED_BY,
            argv=[*common, "--record", "--workdir", str(tmp_path / "recorded")],
            model_factory=lambda provider, model: priced(),
        )
    assert recorded.value.code == 0
    capsys.readouterr()

    argv = [example.__file__, *common, "--replay", "--workdir", str(tmp_path / "replayed")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as replayed:
        runpy.run_path(example.__file__, run_name="__main__")
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert replayed.value.code == 0, report
    assert report["ok"] and report["name"] == example.NAME and report["requests"] == 6
