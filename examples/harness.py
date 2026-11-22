"""The shared harness of the qagents examples: provider choice, cost cap, cassettes, tracing.

Every example in ``examples/patterns`` is a module run from the project root:

    python -m examples.patterns.<name>                       # Anthropic, claude-opus-5-5
    python -m examples.patterns.<name> --provider qcloud     # QCloud, GLM by default
    python -m examples.patterns.<name> --provider openai     # any OpenAI-compatible server
    python -m examples.patterns.<name> --record              # also write a cassette
    python -m examples.patterns.<name> --replay              # offline, from the cassette
    python -m examples.patterns.<name> --mlflow              # trace every run in MLflow

Options: ``--model`` (model id), ``--max-cost`` (USD for the whole example, default 0.50),
``--interactive`` (answer the example's questions yourself), ``--workdir`` (instead of a fresh
temporary directory), ``--verbose`` (also print tool results and sub-agent text), ``--json``
(print an `ExampleReport` as the last line), ``--mlflow-uri``.

Credentials come from the environment and are never printed:

- ``anthropic``: ``ANTHROPIC_API_KEY`` (or ``ANTHROPIC_AUTH_TOKEN``, or an ``ant auth login``
  profile). Model: ``--model``, else ``claude-opus-5-5``.
- ``qcloud``: ``QCLOUD_API_KEY`` and the ``qcloud-ai`` SDK. Model: ``--model``, else
  ``QCLOUD_MODEL``, else `DEFAULT_QCLOUD_MODEL` (GLM).
- ``openai``: ``OPENAI_COMPATIBLE_BASE_URL``, plus ``OPENAI_COMPATIBLE_API_KEY`` when the
  server wants one. Model: ``--model`` or ``OPENAI_COMPATIBLE_MODEL``.

``QAGENTS_EXAMPLE_PROVIDER`` sets the default provider. ``--replay`` needs no credentials.

Cassettes are written to ``examples/cassettes/<provider>/<example>.json``. A replay matches each
request exactly (messages, tools, settings, output schema), so an example replays only while its
prompts and tool results stay the same from run to run.

MLflow traces go to ``--mlflow-uri``, else ``MLFLOW_TRACKING_URI``, else the SQLite database
``examples/mlruns/mlflow.db`` (a local file that needs no server; SQLite is the store MLflow 3
recommends; ``*.db`` files are ignored by git). The experiment is ``qagents-examples``, and each
trace is tagged with the example's name. To browse them:

    mlflow ui --backend-store-uri sqlite:///examples/mlruns/mlflow.db

How to write an example
-----------------------

One module per pattern, ``examples/patterns/<name>.py``, with this shape:

    \"\"\"<What it shows, in two or three sentences.>

    Mirrors: <the LangGraph or CrewAI pattern>. In qagents: <how it differs>.

    Run it:

        python -m examples.patterns.<name> --provider anthropic
    \"\"\"

    from examples.harness import ExampleContext, ExampleOutcome, main
    from qagents import Agent, Limits, tool

    NAME = "<name>"                     # also the cassette's file name
    DESCRIPTION = "<one line>"
    INSPIRED_BY = "<LangGraph: ... | CrewAI: ...>"


    @tool(read_only=True)
    def lookup(order_id: str) -> str:
        \"\"\"Look up an order. ...\"\"\"
        return ORDERS[order_id]          # deterministic local data only


    async def run(ctx: ExampleContext) -> ExampleOutcome:
        agent = Agent(
            ctx.model,                   # the one model of the example, always
            instructions="You are a support agent. Call lookup once, then answer briefly.",
            tools=[lookup],
            limits=Limits(max_turns=6, max_cost_usd=0.10),
            hooks=ctx.hooks,             # on every top-level agent
        )
        ctx.log("Asking about order A-1")
        result = await ctx.print_stream(agent.stream("Where is order A-1?"))
        ok = result.status == "completed" and "Lisbon" in result.text
        return ExampleOutcome(ok=ok, summary="...", output=result.output)


    if __name__ == "__main__":
        main(run, name=NAME, description=DESCRIPTION, inspired_by=INSPIRED_BY)

Rules that keep an example working on both providers and replayable:

- Build every agent on ``ctx.model``. Never set temperature, top_p or seed, never force
  ``tool_choice``, and pass no ``ModelSettings.extra``.
- Structured output: ``output_mode=ctx.output_mode`` (native on Anthropic, tool elsewhere).
- Give every agent ``Limits(max_turns=...)``, plus ``max_cost_usd`` where it could loop; pass
  ``hooks=ctx.hooks`` to every top-level agent (child agents inherit them).
- Tools use local, deterministic data: fixtures in the module, a SQLite file or files under
  ``ctx.workdir``. Keep ``ctx.workdir``, times and random ids out of everything the model
  sees, or the cassette will not replay; use paths relative to the work directory.
- Check outcomes robustly (numbers, key facts, structure), never exact wording.
- Human input: ``answer = await ctx.ask("Approve the refund?", default="yes")`` asks the user
  with ``--interactive`` and returns the default otherwise.
- The offline test, ``tests/examples/test_<name>.py``, drives ``run`` end to end:

      from examples.harness import scripted_context, system_text
      from examples.patterns.<name> import run
      from qagents.models.testing import call

      def policy(request):  # one function answers every agent of the example
          if "support agent" in system_text(request) and request.messages[-1].role == "user":
              return call("lookup", order_id="A-1")
          return "Order A-1 is in Lisbon."

      async def test_<name>(tmp_path):
          outcome = await run(scripted_context(policy, tmp_path))
          assert outcome.ok, outcome.summary

  A policy is a function of the `ModelRequest`; dispatch on `system_text` (which agent is
  asking), `tool_names` (which tools it has) or `last_text`. `scripted_context` uses
  ``output_mode="tool"``, so a structured answer is ``final_answer(**fields)``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from qagents import Hooks, Model, OpenAIChatModel, QCloudModel, RunResult, RunStream, Usage
from qagents.agent import OutputMode
from qagents.errors import ConfigurationError, ModelError, QAgentsError
from qagents.messages import ToolCallPart
from qagents.models import (
    ModelCapabilities,
    ModelRequest,
    ModelSettings,
    ModelStreamEvent,
    ResponseDone,
    cassette,
)
from qagents.models.testing import FunctionModel, ScriptedModel, call

EXAMPLES_DIR = Path(__file__).resolve().parent
CASSETTE_DIR = EXAMPLES_DIR / "cassettes"
RESULTS_DIR = EXAMPLES_DIR / "results"
MLFLOW_DIR = EXAMPLES_DIR / "mlruns"
MLFLOW_EXPERIMENT = "qagents-examples"

PROVIDERS = ("anthropic", "qcloud", "openai")
PROVIDER_ENV = "QAGENTS_EXAMPLE_PROVIDER"
DEFAULT_QCLOUD_MODEL = "openrouter-z-ai-glm-5.3-48ba91a385"
DEFAULT_MAX_COST_USD = 0.50


class SetupError(ConfigurationError):
    """The example cannot start (no key, no cassette, no MLflow). The message is one line."""


class MissingCredentials(SetupError):
    """A provider was chosen without the environment it needs."""


class BudgetExceeded(QAgentsError):
    """The example spent its ``--max-cost``; the next model request was not sent."""


# -- models --------------------------------------------------------------------------------


def make_model(provider: str, model: str | None = None) -> Model:
    """The model every agent of an example uses.

    Args:
        provider: ``"anthropic"``, ``"qcloud"`` or ``"openai"`` (any OpenAI-compatible server).
        model: The model id; each provider has a default (see the module docstring).

    Raises:
        MissingCredentials: The provider's key, SDK or base URL is missing.
        ConfigurationError: The provider is unknown.
    """
    if provider == "anthropic":
        if importlib.util.find_spec("anthropic") is None:
            raise MissingCredentials(
                'The anthropic provider needs the anthropic SDK: pip install -e ".[anthropic]".'
            )
        if not _anthropic_credentials():
            raise MissingCredentials(
                "No Anthropic credentials: set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN), "
                "or run --replay to use the recorded cassette."
            )
        from qagents.models.anthropic import DEFAULT_MODEL, AnthropicModel

        return AnthropicModel(model or DEFAULT_MODEL)
    if provider == "qcloud":
        if importlib.util.find_spec("qcloud_ai") is None:
            raise MissingCredentials(
                "The qcloud provider needs the qcloud-ai SDK: pip install "
                '"git+https://github.com/qcloud-host/qcloud-python.git".'
            )
        if not os.environ.get("QCLOUD_API_KEY"):
            raise MissingCredentials(
                "No QCloud key: set QCLOUD_API_KEY, or run --replay to use the recorded cassette."
            )
        return QCloudModel(model or os.environ.get("QCLOUD_MODEL") or DEFAULT_QCLOUD_MODEL)
    if provider == "openai":
        base_url = os.environ.get("OPENAI_COMPATIBLE_BASE_URL")
        if not base_url:
            raise MissingCredentials(
                "No OpenAI-compatible server: set OPENAI_COMPATIBLE_BASE_URL "
                "(and OPENAI_COMPATIBLE_API_KEY if the server needs one)."
            )
        name = model or os.environ.get("OPENAI_COMPATIBLE_MODEL")
        if not name:
            raise MissingCredentials(
                "No model for the OpenAI-compatible server: pass --model or set "
                "OPENAI_COMPATIBLE_MODEL."
            )
        return OpenAIChatModel(
            name, base_url=base_url, api_key=os.environ.get("OPENAI_COMPATIBLE_API_KEY") or None
        )
    raise ConfigurationError(
        f"Unknown provider {provider!r}; choose one of {', '.join(PROVIDERS)}."
    )


def _anthropic_credentials() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    try:  # an `ant auth login` profile; the SDK has no public check for one
        from anthropic.lib.credentials._constants import _has_auto_discoverable_credentials
    except ImportError:
        return False
    return bool(_has_auto_discoverable_credentials())


class UsageMeter(Model):
    """Counts every request of an example and stops it at an overall cost.

    Wraps the example's model, so every agent, sub-agent and parallel branch is counted
    together. ``max_cost_usd`` is checked before each request: once the reported cost has
    reached it, `BudgetExceeded` is raised instead of sending. Requests already in flight
    finish, so parallel branches can overshoot by their own cost. A provider that reports
    no cost (an OpenAI-compatible server without prices) cannot be capped this way.
    """

    def __init__(self, inner: Model, *, max_cost_usd: float | None = None) -> None:
        self.inner = inner
        self.name = inner.name
        self.provider = inner.provider
        self.max_cost_usd = max_cost_usd
        self.usage = Usage()
        """Requests, tokens and cost of every completed request."""
        self.seconds = 0.0
        """Wall time spent inside model requests (overlapping requests count twice)."""
        self.requests: list[ModelRequest] = []
        """Every request received, in order, like ``ScriptedModel.requests``."""

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        spent = self.usage.cost_usd or 0.0
        if self.max_cost_usd is not None and spent >= self.max_cost_usd:
            raise BudgetExceeded(
                f"The example has spent ${spent:.4f} of its ${self.max_cost_usd:.2f} budget, "
                "so the next model request was not sent. Raise --max-cost to allow more."
            )
        self.requests.append(request)
        started = time.perf_counter()
        try:
            async with contextlib.aclosing(self.inner.stream(request)) as events:
                async for event in events:
                    if isinstance(event, ResponseDone):
                        usage = event.response.usage
                        if not usage.requests:
                            usage = usage.model_copy(update={"requests": 1})
                        self.usage = self.usage.add(usage)
                    yield event
        finally:
            self.seconds += time.perf_counter() - started

    async def capabilities(self) -> ModelCapabilities:
        return await self.inner.capabilities()

    def validate_settings(self, settings: ModelSettings) -> None:
        self.inner.validate_settings(settings)

    async def aclose(self) -> None:
        await self.inner.aclose()

    def __repr__(self) -> str:
        return f"UsageMeter({self.inner!r})"


# -- the context an example runs in ---------------------------------------------------------


class ExampleOutcome(BaseModel):
    """What ``run(ctx)`` returns: whether the example did what it shows."""

    ok: bool
    summary: str
    """One line for the results table."""
    output: Any = None
    details: dict[str, Any] = Field(default_factory=dict)


class ExampleReport(BaseModel):
    """One example's run, as ``--json`` prints it and ``run_all`` tabulates it."""

    name: str
    provider: str
    model: str | None = None
    ok: bool = False
    summary: str = ""
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    seconds: float = 0.0
    error: str | None = None
    skipped: bool = False
    """``run_all`` did not start it (the overall budget was spent)."""


@dataclass
class ExampleContext:
    """Everything an example needs from the harness."""

    model: Model
    """The model every agent of the example is built on (a `UsageMeter`)."""
    provider: str
    output_mode: OutputMode
    """``"native"`` on Anthropic, ``"tool"`` elsewhere: pass it as ``Agent(output_mode=...)``."""
    workdir: Path
    """A fresh directory for this run's files (SQLite databases, notes, outputs)."""
    hooks: list[Hooks] = field(default_factory=list)
    """Pass as ``Agent(hooks=ctx.hooks)`` on every top-level agent."""
    interactive: bool = False
    verbose: bool = False
    name: str = "example"

    @property
    def meter(self) -> UsageMeter | None:
        return self.model if isinstance(self.model, UsageMeter) else None

    def log(self, message: str) -> None:
        """Narrate what the example does next."""
        print(f"-- {message}", flush=True)

    async def ask(self, question: str, *, default: str) -> str:
        """Ask the user with ``--interactive``; otherwise answer ``default``."""
        if not self.interactive:
            self.log(f"{question} -> {default} (pass --interactive to answer yourself)")
            return default
        answer = await asyncio.to_thread(input, f"?? {question} [{default}] ")
        return answer.strip() or default

    async def print_stream(self, stream: RunStream) -> RunResult:
        """Print a run as it happens and return its result.

        Text is printed as it streams; tool calls as ``-> name(arguments)``, failed tool
        calls as ``<- name error: ...``, handoffs as ``== handoff a -> b``, pauses as
        ``|| ...`` and guardrail blocks as ``!! ...``. Sub-agent tool calls are prefixed with
        ``[agent]``. With ``--verbose``, tool results and sub-agent text are printed too.
        """
        printer = _Printer(self.verbose)
        async with stream:
            async for event in stream:
                printer.show(event)
        printer.end_line()
        assert stream.result is not None
        return stream.result


class _Printer:
    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose
        self.mid_line = False

    def write(self, text: str) -> None:
        if text:
            sys.stdout.write(text)
            sys.stdout.flush()
            self.mid_line = not text.endswith("\n")

    def end_line(self) -> None:
        if self.mid_line:
            self.write("\n")

    def line(self, text: str) -> None:
        self.end_line()
        self.write(f"{text}\n")

    def show(self, event: Any, prefix: str = "") -> None:
        kind = getattr(event, "type", None)
        if kind == "text_delta":
            if not prefix or self.verbose:
                self.write(event.text)
        elif kind == "tool_call_started":
            self.line(f"  {prefix}-> {event.call.name}({_arguments(event.call.arguments)})")
        elif kind == "tool_call_finished":
            if event.result.is_error:
                self.line(f"  {prefix}<- {event.call.name} error: {_short(event.result.content)}")
            elif self.verbose:
                self.line(f"  {prefix}<- {event.call.name}: {_short(event.result.content)}")
        elif kind == "agent_switched":
            self.line(f"  {prefix}== handoff {event.from_agent} -> {event.to_agent}")
        elif kind == "approval_required":
            pending = event.approval
            self.line(
                f"  {prefix}|| {pending.name}({_arguments(pending.arguments)}) needs approval"
            )
        elif kind == "input_required":
            pending = event.input
            self.line(f"  {prefix}|| {pending.name}({_arguments(pending.arguments)}) needs input")
        elif kind == "guardrail_tripped":
            self.line(f"  {prefix}!! {event.name} blocked the {event.stage}: {event.reason}")
        elif kind == "subagent_event":
            self.show(event.event, prefix=f"{prefix}[{event.agent}] ")
        elif kind == "run_finished" and not prefix:
            result = event.result
            if result.status != "completed":
                self.line(f"  (run {result.status}: {result.stop_reason or '-'})")


def _short(text: str, limit: int = 100) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _arguments(arguments: dict[str, Any]) -> str:
    shown = ", ".join(
        f"{key}={json.dumps(value, ensure_ascii=False, default=str)}"
        for key, value in arguments.items()
    )
    return _short(shown, 80)


# -- policies for offline tests ---------------------------------------------------------------


def system_text(request: ModelRequest) -> str:
    """The system instructions of the agent that sent ``request``."""
    return "\n".join(m.text for m in request.messages if m.role == "system")


def last_text(request: ModelRequest) -> str:
    """Text of the last message: the prompt, or the tool results the model just received."""
    if not request.messages:
        return ""
    last = request.messages[-1]
    if last.role == "tool":
        return "\n".join(result.content for result in last.tool_results)
    return last.text


def tool_names(request: ModelRequest) -> set[str]:
    """Names of the tools offered in ``request`` (handoffs and ``final_answer`` included)."""
    return {spec.name for spec in request.tools}


def final_answer(**arguments: Any) -> ToolCallPart:
    """A structured answer in ``output_mode="tool"``: the fields of the output type, or
    ``value=...`` when the output type is not an object."""
    return call("final_answer", **arguments)


def structured_answer(request: ModelRequest, **fields: Any) -> ToolCallPart | str:
    """A structured answer in either output mode: a ``final_answer`` call when the request
    offers that tool (``output_mode="tool"``), else the fields as JSON text (``"native"``)."""
    if "final_answer" in tool_names(request):
        return final_answer(**fields)
    return json.dumps(fields)


def scripted_context(
    policy: Callable[[ModelRequest], Any] | Model | Sequence[Any],
    tmp_path: Path,
    *,
    provider: str = "scripted",
    output_mode: OutputMode = "tool",
    interactive: bool = False,
    verbose: bool = False,
    max_cost_usd: float | None = None,
) -> ExampleContext:
    """An `ExampleContext` for offline tests.

    Args:
        policy: A function of the request (wrapped in a ``FunctionModel``), a model such
            as ``ScriptedModel``, or a list of replies for a ``ScriptedModel``.
        tmp_path: pytest's ``tmp_path``; the work directory is ``tmp_path / "work"``.
    """
    if isinstance(policy, Model):
        inner = policy
    elif callable(policy):
        inner = FunctionModel(policy)
    else:
        inner = ScriptedModel(list(policy))
    workdir = Path(tmp_path) / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    return ExampleContext(
        model=UsageMeter(inner, max_cost_usd=max_cost_usd),
        provider=provider,
        output_mode=output_mode,
        workdir=workdir,
        interactive=interactive,
        verbose=verbose,
    )


# -- running one example ------------------------------------------------------------------------

Run = Callable[[ExampleContext], Awaitable[ExampleOutcome]]
ModelFactory = Callable[[str, str | None], Model]


@dataclass
class RunOptions:
    """The command-line options of one example run."""

    provider: str = "anthropic"
    model: str | None = None
    record: bool = False
    replay: bool = False
    mlflow: bool = False
    mlflow_uri: str | None = None
    interactive: bool = False
    verbose: bool = False
    workdir: Path | None = None
    max_cost_usd: float | None = DEFAULT_MAX_COST_USD
    cassette_dir: Path = CASSETTE_DIR


def default_provider() -> str:
    return os.environ.get(PROVIDER_ENV, "").strip().lower() or "anthropic"


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """The options shared by ``main`` and ``run_all``."""
    parser.add_argument("--provider", choices=PROVIDERS, default=default_provider())
    parser.add_argument("--model", help="model id (default depends on the provider)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--record", action="store_true", help="also write the cassette")
    mode.add_argument("--replay", action="store_true", help="answer from the cassette, offline")
    parser.add_argument("--mlflow", action="store_true", help="trace every run in MLflow")
    parser.add_argument("--mlflow-uri", help="MLflow tracking URI (default: a local SQLite file)")
    parser.add_argument("--interactive", action="store_true", help="answer questions yourself")
    parser.add_argument("--workdir", type=Path, help="work directory (default: a fresh temp dir)")
    parser.add_argument("--verbose", action="store_true", help="also print tool results")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--cassette-dir", type=Path, default=CASSETTE_DIR, help=argparse.SUPPRESS)


def options_from(args: argparse.Namespace, *, max_cost_usd: float | None) -> RunOptions:
    return RunOptions(
        provider=args.provider,
        model=args.model,
        record=args.record,
        replay=args.replay,
        mlflow=args.mlflow,
        mlflow_uri=args.mlflow_uri,
        interactive=args.interactive,
        verbose=args.verbose,
        workdir=args.workdir,
        max_cost_usd=max_cost_usd,
        cassette_dir=args.cassette_dir,
    )


def cassette_path(options: RunOptions, name: str) -> Path:
    return options.cassette_dir / options.provider / f"{name}.json"


def build_model(options: RunOptions, name: str, model_factory: ModelFactory = make_model) -> Model:
    """The provider's model, behind a cassette when recording or replaying."""
    path = cassette_path(options, name)
    if options.replay:
        if not path.is_file():
            raise SetupError(
                f"No cassette at {path.as_posix()}: record one first with "
                f"--record --provider {options.provider}."
            )
        return cassette(None, path, mode="replay")
    real = model_factory(options.provider, options.model)
    return cassette(real, path, mode="record") if options.record else real


def mlflow_hooks(uri: str | None, *, name: str, provider: str) -> Hooks:
    """`MLflowHooks` with content, writing to ``uri`` (see the module docstring)."""
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")  # no calls home
    os.environ.setdefault("DO_NOT_TRACK", "true")
    try:
        import mlflow

        from qagents.integrations.mlflow import MLflowHooks
    except ImportError as error:
        raise SetupError(f'--mlflow needs MLflow: pip install -e ".[mlflow]" ({error}).') from error
    local = uri is None and not os.environ.get("MLFLOW_TRACKING_URI")
    if local:
        MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
        uri = f"sqlite:///{(MLFLOW_DIR / 'mlflow.db').as_posix()}"
    if uri is not None:
        mlflow.set_tracking_uri(uri)
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is not None:
        experiment_id = experiment.experiment_id
    else:
        artifacts = (MLFLOW_DIR / "artifacts").as_uri() if local else None
        experiment_id = mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=artifacts)
    return MLflowHooks(
        record_content=True,
        experiment_id=experiment_id,
        tags={"qagents.example": name, "qagents.example.provider": provider},
    )


def _workdir(options: RunOptions, name: str) -> Path:
    if options.workdir is not None:
        options.workdir.mkdir(parents=True, exist_ok=True)
        return options.workdir.resolve()
    return Path(tempfile.mkdtemp(prefix=f"qagents-{name}-"))


async def run_example(
    run: Run,
    *,
    name: str,
    options: RunOptions,
    model_factory: ModelFactory = make_model,
) -> ExampleReport:
    """Build the context, run the example, print its outcome and usage, and report.

    Never raises for a failing example: the exception's type and message (and the
    provider's request id, when there is one) go into ``ExampleReport.error``.
    """
    report = ExampleReport(name=name, provider=options.provider)
    meter: UsageMeter | None = None
    hooks: list[Hooks] = []
    started = time.perf_counter()
    try:
        budget = None if options.replay else options.max_cost_usd
        meter = UsageMeter(build_model(options, name, model_factory), max_cost_usd=budget)
        report.model = meter.name
        if options.mlflow:
            hooks.append(mlflow_hooks(options.mlflow_uri, name=name, provider=options.provider))
        ctx = ExampleContext(
            model=meter,
            provider=options.provider,
            output_mode="native" if options.provider == "anthropic" else "tool",
            workdir=_workdir(options, name),
            hooks=hooks,
            interactive=options.interactive,
            verbose=options.verbose,
            name=name,
        )
        mode = " (replay)" if options.replay else " (recording)" if options.record else ""
        ctx.log(f"{options.provider}:{meter.name}{mode}, work directory {ctx.workdir.as_posix()}")
        outcome = await run(ctx)
        if not isinstance(outcome, ExampleOutcome):
            raise TypeError(f"run() returned {type(outcome).__name__}, not ExampleOutcome.")
        report.ok, report.summary = outcome.ok, outcome.summary
        print(f"{'ok' if outcome.ok else 'FAILED'}: {outcome.summary}")
        if options.verbose and outcome.details:
            print(json.dumps(outcome.details, indent=2, ensure_ascii=False, default=str))
    except SetupError as error:
        report.error = str(error)
        print(f"error: {error}")
    except Exception as error:  # an example must never take run_all down with it
        report.error = describe_error(error)
        print(f"error: {report.error}")
    finally:
        report.seconds = round(time.perf_counter() - started, 2)
        if meter is not None:
            report.requests = meter.usage.requests
            report.input_tokens = meter.usage.input_tokens
            report.output_tokens = meter.usage.output_tokens
            report.cost_usd = meter.usage.cost_usd
            with contextlib.suppress(Exception):
                await meter.aclose()
            print(usage_line(report, replayed=options.replay))
        for hook in hooks:
            flush = getattr(hook, "flush", None)
            if flush is not None:
                with contextlib.suppress(Exception):
                    flush()
    return report


def describe_error(error: BaseException) -> str:
    text = f"{type(error).__name__}: {error}"
    request_id = getattr(error, "request_id", None)
    if isinstance(error, ModelError) and request_id:
        text += f" (request id {request_id})"
    return text


def usage_line(report: ExampleReport, *, replayed: bool = False) -> str:
    cost = "cost unknown" if report.cost_usd is None else f"${report.cost_usd:.4f}"
    if replayed:
        cost += " as recorded"
    return (
        f"usage: {report.requests} requests, {report.input_tokens:,} input / "
        f"{report.output_tokens:,} output tokens, {cost}, {report.seconds:.1f} s"
    )


def prepare_stdout() -> None:
    """Never fail on a character the console cannot show (a model may write any)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(Exception):
                reconfigure(errors="replace")


def main(
    run: Run,
    *,
    name: str,
    description: str,
    inspired_by: str,
    argv: Iterable[str] | None = None,
    model_factory: ModelFactory = make_model,
) -> None:
    """The command line of one example. Exits 0 when the outcome is ok, 1 otherwise."""
    parser = argparse.ArgumentParser(
        prog=f"python -m examples.patterns.{name}",
        description=f"{description} (inspired by {inspired_by})",
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--max-cost",
        type=float,
        default=DEFAULT_MAX_COST_USD,
        help=f"USD for the whole example (default {DEFAULT_MAX_COST_USD:.2f})",
    )
    args = parser.parse_args(None if argv is None else list(argv))
    prepare_stdout()
    print(f"== {name}: {description}")
    print(f"   inspired by {inspired_by}")
    options = options_from(args, max_cost_usd=args.max_cost)
    try:
        report = asyncio.run(
            run_example(run, name=name, options=options, model_factory=model_factory)
        )
    except KeyboardInterrupt:
        print("interrupted")
        sys.exit(130)
    if args.json:
        print(report.model_dump_json())
    sys.exit(0 if report.ok else 1)
