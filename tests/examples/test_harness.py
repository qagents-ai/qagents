"""The example harness: models, the usage meter and cost cap, cassettes, main() and run_all."""

from __future__ import annotations

import importlib.util
import json
import runpy
import sys
import uuid

import pytest

from examples import harness, run_all
from examples.harness import (
    DEFAULT_QCLOUD_MODEL,
    BudgetExceeded,
    ExampleContext,
    ExampleOutcome,
    MissingCredentials,
    RunOptions,
    UsageMeter,
    final_answer,
    last_text,
    make_model,
    run_example,
    scripted_context,
    system_text,
    tool_names,
)
from qagents import Agent, Limits, OpenAIChatModel, QCloudModel, Usage, tool
from qagents.errors import ConfigurationError, ModelServerError
from qagents.models.testing import FunctionModel, ScriptedModel, call

CREDENTIAL_VARIABLES = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "QCLOUD_API_KEY",
    "QCLOUD_MODEL",
    "OPENAI_COMPATIBLE_BASE_URL",
    "OPENAI_COMPATIBLE_API_KEY",
    "OPENAI_COMPATIBLE_MODEL",
    "QAGENTS_EXAMPLE_PROVIDER",
    "QAGENTS_CASSETTE_MODE",
)


@pytest.fixture
def no_credentials(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in CREDENTIAL_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    if importlib.util.find_spec("anthropic") is not None:  # no `ant auth login` profile either
        from anthropic.lib.credentials import _constants

        monkeypatch.setattr(_constants, "_has_auto_discoverable_credentials", lambda: False)
    return monkeypatch


# -- a tiny example used by several tests ------------------------------------------------------


@tool(read_only=True)
def lookup(key: str) -> str:
    """Look up a value.

    Args:
        key: The key.
    """
    return {"answer": "42"}.get(key, "unknown")


def lookup_policy(request):
    """Call lookup once, then answer with what it returned."""
    if request.messages[-1].role == "tool":
        return f"The answer is {last_text(request)}."
    return call("lookup", key="answer")


async def lookup_example(ctx: ExampleContext) -> ExampleOutcome:
    agent = Agent(
        ctx.model,
        instructions="Call lookup with key 'answer', then reply with the value.",
        tools=[lookup],
        limits=Limits(max_turns=4),
        hooks=ctx.hooks,
    )
    ctx.log("Looking up the answer")
    result = await ctx.print_stream(agent.stream("What is the answer?"))
    return ExampleOutcome(ok="42" in result.text, summary=result.text, output=result.output)


def priced(policy, cost: float = 0.1) -> FunctionModel:
    return FunctionModel(
        policy, usage=Usage(requests=1, input_tokens=100, output_tokens=10, cost_usd=cost)
    )


def factory(policy=lookup_policy, cost: float = 0.1):
    def build(provider: str, model: str | None) -> FunctionModel:
        return priced(policy, cost)

    return build


def no_factory(provider: str, model: str | None):
    raise AssertionError("a replay must not build the provider's model")


# -- make_model -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "variable"),
    [
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("qcloud", "QCLOUD_API_KEY"),
        ("openai", "OPENAI_COMPATIBLE_BASE_URL"),
    ],
)
def test_make_model_without_credentials_names_the_variable(no_credentials, provider, variable):
    sdk = {"anthropic": "anthropic", "qcloud": "qcloud_ai"}.get(provider)
    if sdk is not None and importlib.util.find_spec(sdk) is None:
        pytest.skip(f"{sdk} is not installed")
    with pytest.raises(MissingCredentials) as caught:
        make_model(provider)
    message = str(caught.value)
    assert variable in message and "\n" not in message


def test_make_model_for_openai_needs_a_model_id(no_credentials):
    no_credentials.setenv("OPENAI_COMPATIBLE_BASE_URL", "http://127.0.0.1:9/v1")
    with pytest.raises(MissingCredentials, match="OPENAI_COMPATIBLE_MODEL"):
        make_model("openai")
    model = make_model("openai", "local-model")
    assert isinstance(model, OpenAIChatModel) and model.name == "local-model"


def test_make_model_picks_the_default_models(no_credentials):
    pytest.importorskip("anthropic")
    pytest.importorskip("qcloud_ai")
    from qagents.models.anthropic import DEFAULT_MODEL, AnthropicModel

    no_credentials.setenv("ANTHROPIC_API_KEY", "test-key")
    no_credentials.setenv("QCLOUD_API_KEY", "test-key")
    anthropic = make_model("anthropic")
    assert isinstance(anthropic, AnthropicModel) and anthropic.name == DEFAULT_MODEL
    assert make_model("anthropic", "claude-sonnet-5-5").name == "claude-sonnet-5-5"

    qcloud = make_model("qcloud")
    assert isinstance(qcloud, QCloudModel) and qcloud.name == DEFAULT_QCLOUD_MODEL
    no_credentials.setenv("QCLOUD_MODEL", "from-env")
    assert make_model("qcloud").name == "from-env"
    assert make_model("qcloud", "from-flag").name == "from-flag"


def test_make_model_refuses_an_unknown_provider():
    with pytest.raises(ConfigurationError, match="Unknown provider"):
        make_model("nowhere")


def test_main_without_credentials_prints_one_line_and_fails(no_credentials, capsys):
    if importlib.util.find_spec("qcloud_ai") is None:
        pytest.skip("qcloud_ai is not installed")

    async def never(ctx):
        raise AssertionError("the example must not start")

    with pytest.raises(SystemExit) as exited:
        harness.main(
            never, name="x", description="d", inspired_by="i", argv=["--provider", "qcloud"]
        )
    output = capsys.readouterr().out
    assert exited.value.code == 1
    assert "error: No QCloud key: set QCLOUD_API_KEY" in output
    assert "Traceback" not in output


# -- the usage meter and the cost cap ---------------------------------------------------------


async def test_meter_counts_requests_tokens_cost_and_time():
    meter = UsageMeter(priced(lookup_policy, cost=0.01))
    result = await Agent(meter, tools=[lookup]).run("What is the answer?")
    assert result.text == "The answer is 42."
    assert (meter.usage.requests, meter.usage.input_tokens, meter.usage.output_tokens) == (
        2,
        200,
        20,
    )
    assert meter.usage.cost_usd == pytest.approx(0.02)
    assert meter.seconds >= 0 and len(meter.requests) == 2
    assert meter.name == "function" and meter.provider == "test"


async def test_meter_counts_a_request_whose_usage_reports_none():
    meter = UsageMeter(FunctionModel(lambda request: "hi", usage=Usage()))
    await Agent(meter).run("hello")
    assert meter.usage.requests == 1 and meter.usage.cost_usd is None


async def test_cost_cap_stops_before_the_request_that_would_exceed_it():
    inner = priced(lambda request: call("lookup", key="answer"), cost=0.1)  # loops forever
    meter = UsageMeter(inner, max_cost_usd=0.25)
    with pytest.raises(BudgetExceeded, match=r"\$0\.3000 of its \$0\.25 budget"):
        await Agent(meter, tools=[lookup], limits=Limits(max_turns=10)).run("Loop")
    assert len(inner.requests) == 3  # 0.0, 0.1 and 0.2 were below the cap; 0.3 was not


async def test_run_example_reports_a_spent_budget(tmp_path, capsys):
    loop = factory(lambda request: call("lookup", key="answer"), cost=0.2)
    options = RunOptions(provider="qcloud", max_cost_usd=0.3, workdir=tmp_path / "w")
    report = await run_example(lookup_example, name="loop", options=options, model_factory=loop)
    output = capsys.readouterr().out
    assert not report.ok and report.error.startswith("BudgetExceeded")
    assert report.requests == 2 and report.cost_usd == pytest.approx(0.4)
    assert "usage: 2 requests, 200 input / 20 output tokens, $0.4000" in output


# -- cassettes ----------------------------------------------------------------------------------


async def test_record_then_replay_without_the_provider(tmp_path, capsys):
    cassettes = tmp_path / "cassettes"
    recording = RunOptions(provider="anthropic", record=True, cassette_dir=cassettes)
    recorded = await run_example(
        lookup_example, name="lookup", options=recording, model_factory=factory()
    )
    assert recorded.ok, recorded.error
    path = cassettes / "anthropic" / "lookup.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["format"] == "qagents-cassette" and len(data["interactions"]) == 2

    replaying = RunOptions(provider="anthropic", replay=True, cassette_dir=cassettes)
    replayed = await run_example(
        lookup_example, name="lookup", options=replaying, model_factory=no_factory
    )
    output = capsys.readouterr().out
    assert replayed.ok, replayed.error
    assert (replayed.model, replayed.requests) == ("function", 2)
    assert replayed.cost_usd == pytest.approx(recorded.cost_usd)
    assert "(replay)" in output and "as recorded" in output


async def test_replay_reports_a_changed_request(tmp_path):
    cassettes = tmp_path / "cassettes"
    options = RunOptions(provider="qcloud", record=True, cassette_dir=cassettes)
    await run_example(lookup_example, name="lookup", options=options, model_factory=factory())

    async def changed(ctx: ExampleContext) -> ExampleOutcome:
        result = await Agent(ctx.model, tools=[lookup], limits=Limits(max_turns=4)).run("Other?")
        return ExampleOutcome(ok=True, summary=result.text)

    options = RunOptions(provider="qcloud", replay=True, cassette_dir=cassettes)
    report = await run_example(changed, name="lookup", options=options, model_factory=no_factory)
    assert not report.ok and report.error.startswith("CassetteMissError")


async def test_replay_without_a_cassette_says_so(tmp_path):
    options = RunOptions(provider="qcloud", replay=True, cassette_dir=tmp_path)
    report = await run_example(
        lookup_example, name="none", options=options, model_factory=no_factory
    )
    assert not report.ok and "no cassette" in report.error.lower()


# -- printing a run -----------------------------------------------------------------------------


async def test_print_stream_shows_tools_handoffs_and_pauses(tmp_path, capsys):
    @tool(requires_approval=True)
    def refund(order: str) -> str:
        """Refund an order.

        Args:
            order: Order id.
        """
        return "refunded"

    model = ScriptedModel(
        [call("lookup", key="answer"), call("transfer_to_billing"), call("refund", order="A-1")]
    )
    ctx = scripted_context(model, tmp_path)
    billing = Agent(ctx.model, name="billing", description="Refunds.", tools=[refund])
    triage = Agent(ctx.model, name="triage", tools=[lookup], handoffs=[billing])
    result = await ctx.print_stream(triage.stream("Refund A-1"))
    output = capsys.readouterr().out
    assert result.status == "paused"
    assert '-> lookup(key="answer")' in output
    assert "== handoff triage -> billing" in output
    assert 'refund(order="A-1") needs approval' in output
    assert "(run paused:" in output


async def test_print_stream_prefixes_sub_agent_tool_calls(tmp_path, capsys):
    def policy(request):
        if "researcher" in system_text(request):
            return lookup_policy(request)
        if request.messages[-1].role == "tool":
            return "Done: " + last_text(request)
        return call("researcher", task="Find the answer")

    ctx = scripted_context(policy, tmp_path, verbose=True)
    researcher = Agent(
        ctx.model, name="researcher", instructions="You are the researcher.", tools=[lookup]
    )
    lead = Agent(ctx.model, name="lead", instructions="You lead.", tools=[researcher.as_tool()])
    result = await ctx.print_stream(lead.stream("Go"))
    output = capsys.readouterr().out
    assert result.text == "Done: The answer is 42."
    assert '[researcher] -> lookup(key="answer")' in output
    assert "[researcher] <- lookup: 42" in output  # verbose shows results


# -- helpers for tests and examples ---------------------------------------------------------------


async def test_scripted_context_and_policy_helpers(tmp_path):
    from pydantic import BaseModel

    class Answer(BaseModel):
        value: int

    seen = {}

    def policy(request):
        seen.update(system=system_text(request), last=last_text(request), tools=tool_names(request))
        return final_answer(value=42)

    ctx = scripted_context(policy, tmp_path)
    assert ctx.output_mode == "tool" and ctx.workdir == tmp_path / "work" and ctx.workdir.is_dir()
    agent = Agent(
        ctx.model, instructions="Answer.", output_type=Answer, output_mode=ctx.output_mode
    )
    result = await agent.run("What is it?")
    assert result.output == Answer(value=42)
    assert seen["system"].startswith("Answer.") and seen["last"] == "What is it?"
    assert seen["tools"] == {"final_answer"}
    assert ctx.meter is ctx.model and ctx.meter.usage.requests == 1


async def test_scripted_context_accepts_a_list_of_replies(tmp_path):
    ctx = scripted_context(["Hello."], tmp_path)
    assert (await Agent(ctx.model).run("Hi")).text == "Hello."


async def test_ask_answers_the_default_unless_interactive(tmp_path, monkeypatch, capsys):
    quiet = scripted_context([], tmp_path)
    assert await quiet.ask("Approve?", default="yes") == "yes"
    assert "Approve? -> yes" in capsys.readouterr().out
    monkeypatch.setattr("builtins.input", lambda prompt: "no")
    asking = scripted_context([], tmp_path, interactive=True)
    assert await asking.ask("Approve?", default="yes") == "no"


# -- main() with a fake pattern module ------------------------------------------------------------

FAKE_PATTERN = '''
"""A fake pattern for the harness tests."""

from examples.harness import ExampleOutcome, main
from qagents import Agent, Limits

NAME = "fake"
DESCRIPTION = "Says hello."
INSPIRED_BY = "nothing in particular"


async def run(ctx):
    agent = Agent(ctx.model, instructions="Say hello.", limits=Limits(max_turns=2), hooks=ctx.hooks)
    result = await ctx.print_stream(agent.stream("Hi"))
    return ExampleOutcome(ok="hello" in result.text.lower(), summary=result.text)


if __name__ == "__main__":
    main(run, name=NAME, description=DESCRIPTION, inspired_by=INSPIRED_BY)
'''


@pytest.fixture
def fake_pattern(tmp_path):
    path = tmp_path / "fake_pattern.py"
    path.write_text(FAKE_PATTERN, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"fake_pattern_{uuid.uuid4().hex}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call_main(module, argv, model_factory=no_factory):
    with pytest.raises(SystemExit) as exited:
        harness.main(
            module.run,
            name=module.NAME,
            description=module.DESCRIPTION,
            inspired_by=module.INSPIRED_BY,
            argv=argv,
            model_factory=model_factory,
        )
    return exited.value.code


def test_main_runs_a_pattern_and_prints_outcome_and_usage(fake_pattern, tmp_path, capsys):
    hello = factory(lambda request: "Hello there!", cost=0.001)
    code = call_main(
        fake_pattern, ["--provider", "qcloud", "--workdir", str(tmp_path / "w")], hello
    )
    output = capsys.readouterr().out
    assert code == 0
    assert "== fake: Says hello." in output and "inspired by nothing in particular" in output
    assert "Hello there!" in output and "ok: Hello there!" in output
    assert "usage: 1 requests, 100 input / 10 output tokens, $0.0010" in output


def test_main_exits_non_zero_for_a_failed_outcome(fake_pattern, capsys):
    grumpy = factory(lambda request: "Go away.")
    assert call_main(fake_pattern, ["--provider", "qcloud"], grumpy) == 1
    assert "FAILED: Go away." in capsys.readouterr().out


def test_main_prints_the_error_and_the_request_id(fake_pattern, capsys):
    def broken(request):
        raise ModelServerError("upstream exploded", provider="test", request_id="req_123")

    assert call_main(fake_pattern, ["--provider", "qcloud", "--json"], factory(broken)) == 1
    output = capsys.readouterr().out
    assert "error: ModelServerError: upstream exploded (request id req_123)" in output
    report = json.loads(output.strip().splitlines()[-1])
    assert report["ok"] is False and "req_123" in report["error"]


def test_a_pattern_module_records_and_replays_as_a_script(
    fake_pattern, tmp_path, capsys, monkeypatch
):
    cassettes = tmp_path / "cassettes"
    hello = factory(lambda request: "Hello!", cost=0.002)
    assert (
        call_main(
            fake_pattern,
            ["--provider", "anthropic", "--record", "--cassette-dir", str(cassettes)],
            hello,
        )
        == 0
    )
    capsys.readouterr()

    script = tmp_path / "fake_pattern.py"
    argv = [
        str(script),
        "--provider",
        "anthropic",
        "--replay",
        "--cassette-dir",
        str(cassettes),
        "--json",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(str(script), run_name="__main__")
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert exited.value.code == 0
    assert report["ok"] and report["name"] == "fake" and report["provider"] == "anthropic"
    assert report["requests"] == 1 and report["cost_usd"] == pytest.approx(0.002)


# -- run_all over a fake package ------------------------------------------------------------------


@pytest.fixture
def fake_package(tmp_path, monkeypatch):
    name = f"fakepatterns_{uuid.uuid4().hex[:8]}"
    package = tmp_path / "packages" / name
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('"""Fake patterns."""\n', encoding="utf-8")
    (package / "_shared.py").write_text("raise AssertionError('private modules are not run')\n")
    for module in ("alpha", "beta"):
        source = FAKE_PATTERN.replace('NAME = "fake"', f'NAME = "{module}"')
        (package / f"{module}.py").write_text(source, encoding="utf-8")
    (package / "broken.py").write_text('NAME = "broken"\n', encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path / "packages"))
    return name


def run_all_main(package, argv, model_factory):
    return run_all.main(argv, package=package, model_factory=model_factory)


def test_run_all_runs_every_pattern_and_writes_results(fake_package, tmp_path, capsys):
    results = tmp_path / "results"
    hello = factory(lambda request: "Hello!", cost=0.01)
    code = run_all_main(
        fake_package, ["--provider", "qcloud", "--results-dir", str(results)], hello
    )
    output = capsys.readouterr().out
    assert code == 1  # broken has no run()
    data = json.loads((results / "qcloud.json").read_text(encoding="utf-8"))
    reports = {report["name"]: report for report in data["reports"]}
    assert set(reports) == {"alpha", "beta", "broken"} and data["mode"] == "live"
    assert reports["alpha"]["ok"] and reports["beta"]["ok"]
    assert "does not define" in reports["broken"]["error"]
    table = (results / "qcloud.md").read_text(encoding="utf-8")
    assert "| alpha | ok | 1 |" in table and "| broken | FAILED |" in table
    assert "2 of 3 passed, 0.0200 USD" in output


def test_run_all_stops_starting_examples_once_the_budget_is_spent(fake_package, tmp_path, capsys):
    results = tmp_path / "results"
    expensive = factory(lambda request: "Hello!", cost=0.3)
    argv = [
        "--provider",
        "qcloud",
        "--only",
        "alpha,beta",
        "--max-cost",
        "0.25",
        "--results-dir",
        str(results),
    ]
    assert run_all_main(fake_package, argv, expensive) == 1
    reports = json.loads((results / "qcloud.json").read_text(encoding="utf-8"))["reports"]
    assert [(r["name"], r["ok"], r["skipped"]) for r in reports] == [
        ("alpha", True, False),
        ("beta", False, True),
    ]
    assert "skipped" in capsys.readouterr().out


def test_run_all_replays_into_a_separate_results_file(fake_package, tmp_path):
    results, cassettes = tmp_path / "results", tmp_path / "cassettes"
    hello = factory(lambda request: "Hello!", cost=0.01)
    common = [
        "--provider",
        "anthropic",
        "--only",
        "alpha",
        "--results-dir",
        str(results),
        "--cassette-dir",
        str(cassettes),
    ]
    assert run_all_main(fake_package, [*common, "--record"], hello) == 0
    assert run_all_main(fake_package, [*common, "--replay"], no_factory) == 0
    assert (results / "anthropic.md").exists() and (results / "anthropic-replay.md").exists()
    replay = json.loads((results / "anthropic-replay.json").read_text(encoding="utf-8"))
    assert replay["mode"] == "replay" and replay["reports"][0]["ok"]


def test_run_all_refuses_an_unknown_name(fake_package):
    with pytest.raises(SystemExit, match="Unknown example"):
        run_all_main(fake_package, ["--only", "gamma"], no_factory)


def test_the_real_patterns_package_is_discoverable():
    modules = run_all.discover()
    assert all(name.startswith("examples.patterns.") for name in modules)
    assert not any(name.rsplit(".", 1)[-1].startswith("_") for name in modules)


# -- MLflow -------------------------------------------------------------------------------------


async def test_mlflow_option_traces_into_a_local_store(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("sqlalchemy")
    import mlflow.tracking._tracking_service.utils as tracking_utils

    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    previous = tracking_utils._tracking_uri
    try:
        options = RunOptions(provider="qcloud", mlflow=True, mlflow_uri=uri)
        report = await run_example(
            lookup_example, name="traced", options=options, model_factory=factory()
        )
        assert report.ok, report.error
        experiment = mlflow.get_experiment_by_name(harness.MLFLOW_EXPERIMENT)
        traces = mlflow.search_traces(locations=[experiment.experiment_id], return_type="list")
        assert traces and traces[0].info.tags["qagents.example"] == "traced"
    finally:
        mlflow.set_tracking_uri(previous)
