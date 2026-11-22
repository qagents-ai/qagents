"""Evaluations: run an agent over a dataset and score what it did.

Use them to compare models and providers, to catch regressions before a
release, and to back a claim about an agent with numbers:

    from qagents.evals import Case, Contains, Dataset, ToolCalled, evaluate

    dataset = Dataset(
        [Case(name="athens", prompt="Weather in Athens?", expected="sunny")],
        name="weather",
    )
    report = await evaluate(
        agent, dataset, [Contains(case_sensitive=False), ToolCalled("get_weather")]
    )
    print(report.to_markdown())

Every case runs in a fresh session. An exception raised while running a case
is recorded as a failed case, never raised, so one broken case cannot hide the
others. A case passes when it ran without an exception, its run completed, and
every evaluator passed.

An evaluator is any callable that takes an `EvalContext` and returns an
`EvaluatorResult` (sync or async). The built-in ones cover exact and partial
matches, schemas and types, tool use, turn, cost and latency budgets, and an
LLM judge. A plain function is wrapped in `Custom`, so it may also return a
bool, a score between 0 and 1, or a ``(value, reason)`` tuple.

Two reports of the same dataset (for example one per provider) are compared
with `compare`; `regressions` lists the cases that got worse, for a CI check.
`log_to_mlflow` records a report in the active MLflow experiment.

The only text this module sends to a model is the judge's, in the
``qagents.prompts.JUDGE_*`` constants; replace them to change what the judge reads.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import re
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    Field,
    TypeAdapter,
    ValidationError,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_core import to_jsonable_python

from . import prompts
from .agent import Agent, Limits, RunStream, extract_json
from .errors import ConfigurationError
from .eval_statistics import PairedComparison as PairedComparison
from .eval_statistics import ReliabilityReport as ReliabilityReport
from .eval_statistics import compare_reliability as compare_reliability
from .eval_statistics import reliability as reliability
from .hooks import Hooks
from .messages import Message, Usage
from .models.base import Model, ModelSettings
from .results import RunResult
from .session import Session, SessionStore
from .tools import RunContext

logger = logging.getLogger("qagents.evals")

# What the judge reads is in qagents.prompts (JUDGE_*).


# --- datasets -------------------------------------------------------------------


class Case(BaseModel):
    """One input to evaluate, with what a good answer looks like.

    ``prompt`` is the user's text, or a list of messages: the last one, when
    it is a user message, is sent as the prompt and the rest seed the
    conversation. In a JSONL file a message may also be written as
    ``{"role": "user", "content": "..."}``.
    """

    name: str
    """Unique within its dataset; reports and comparisons match cases by name."""
    prompt: str | list[Message]
    expected: Any = None
    """What evaluators such as `ExactMatch` and `Contains` compare against, when
    they are not given a value of their own. Keep it JSON-serialisable."""
    metadata: dict[str, Any] = Field(default_factory=dict)
    """Anything you want to slice results by (difficulty, topic, source). Copied
    into every `CaseResult` of the case."""
    deps: Any = Field(default=None, exclude=True)
    """Passed to the agent as ``deps``. Not written to JSONL: build per-case deps
    from ``metadata`` in a target function instead."""

    @field_validator("name")
    @classmethod
    def _name_is_set(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("a case needs a non-empty name")
        return value

    @field_validator("prompt", mode="before")
    @classmethod
    def _accept_content_messages(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [_message_from_content(item) for item in value]
        return value

    @field_validator("prompt")
    @classmethod
    def _prompt_is_not_empty(cls, value: str | list[Message]) -> str | list[Message]:
        if isinstance(value, list) and not value:
            raise ValueError("the prompt is an empty list of messages; give at least one")
        return value


class Dataset(BaseModel):
    """Named, ordered cases.

    dataset = Dataset([Case(name="greet", prompt="Hi")], name="smoke")
    dataset.to_jsonl("smoke.jsonl")
    same = Dataset.from_jsonl("smoke.jsonl")
    """

    cases: list[Case] = Field(default_factory=list)
    name: str = "dataset"

    def __init__(
        self,
        cases: Iterable[Case | Mapping[str, Any]] = (),
        name: str = "dataset",
        **data: Any,
    ) -> None:
        super().__init__(cases=list(cases), name=name, **data)

    @model_validator(mode="after")
    def _names_are_unique(self) -> Dataset:
        names = [case.name for case in self.cases]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"case names must be unique within a dataset; repeated: {', '.join(duplicates)}"
            )
        return self

    def __len__(self) -> int:
        return len(self.cases)

    @classmethod
    def from_jsonl(cls, path: str | Path, *, name: str | None = None) -> Dataset:
        """Read one case per line. Blank lines are skipped.

        Args:
            path: A UTF-8 file of JSON objects with ``name``, ``prompt`` and
                optionally ``expected`` and ``metadata``.
            name: The dataset's name. Defaults to the file name without its suffix.

        Raises:
            ConfigurationError: A line is not valid JSON or not a valid case; the
                message names the file and the line.
        """
        source = Path(path)
        cases: list[Case] = []
        for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                cases.append(Case.model_validate(json.loads(line)))
            except (ValueError, ValidationError) as error:
                raise ConfigurationError(
                    f"{source.name} line {number} is not a valid case: {error}. Each line must "
                    'be a JSON object such as {"name": "greet", "prompt": "Hi"}.'
                ) from error
        try:
            return cls(cases, name=name or source.stem)
        except ValidationError as error:
            raise ConfigurationError(f"{source.name}: {error}") from error

    def to_jsonl(self, path: str | Path) -> None:
        """Write one case per line, as UTF-8 JSON. ``deps`` are not written."""
        lines = [json.dumps(_case_record(case), ensure_ascii=False) for case in self.cases]
        Path(path).write_text(
            "".join(f"{line}\n" for line in lines), encoding="utf-8", newline="\n"
        )


# --- evaluators -----------------------------------------------------------------


class EvaluatorResult(BaseModel):
    """What one evaluator concluded about one run."""

    name: str = ""
    """The evaluator's name; filled in for you when a custom evaluator leaves it empty."""
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    reason: str = ""
    usage: Usage | None = None
    """Model usage the evaluator itself caused (set by `LLMJudge`). It is not part
    of the case's usage or cost."""


class ToolCallRecord(BaseModel):
    """A tool call the model made during a run."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    is_error: bool | None = None
    """Whether the call's result was an error (the tool failed, or the call was
    refused). ``None`` when the call has no result, for example while it waits
    for approval."""


@dataclass
class EvalContext:
    """What an evaluator can look at: the case and the run it produced."""

    case: Case
    result: RunResult
    repeat: int = 0
    """Which repetition of the case this is, from 0."""
    latency_s: float = 0.0
    """Wall-clock seconds the run took, evaluators excluded."""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    """Tool calls the model made in this run, in order."""
    structured: bool | None = None
    """Whether the agent has an ``output_type``, so that ``output`` is a validated value even
    when it is text (a ``Literal`` or ``Enum`` of strings). ``output_type=str`` is the same as
    no ``output_type``: a plain-text agent. ``None`` when that is not known: a target function
    returned the `RunResult` itself, and its text output is treated as a plain-text answer."""

    @property
    def output(self) -> Any:
        """The run's output: the answer text, or the validated ``output_type`` value.
        ``None`` when the run did not complete."""
        return self.result.output

    @property
    def text(self) -> str:
        """Text of the last assistant message. With ``output_mode="tool"`` the answer is not
        in it; use `answer_text`."""
        return self.result.text

    @property
    def answer_text(self) -> str:
        """The answer as text: ``output`` when it is text, otherwise `text`.

        A text answer given with the final-answer tool (``output_mode="tool"``), or replaced
        by an output validator, is in ``output`` only. A structured output is the JSON text the
        model wrote, its last assistant message."""
        return self.output if isinstance(self.output, str) else self.text

    @property
    def expected(self) -> Any:
        return self.case.expected

    @property
    def usage(self) -> Usage:
        return self.result.usage

    @property
    def turns(self) -> int:
        return self.result.turns


EvaluatorFunction = Callable[[EvalContext], Any]
"""A function of the `EvalContext`, sync or async, as accepted by `Custom`."""


class Evaluator(ABC):
    """Base class of the built-in evaluators.

    Subclass it, set ``name`` and implement ``__call__`` (sync or async) to
    write your own; or pass a plain function, which is wrapped in `Custom`.
    """

    name: str = "evaluator"
    """Key of this evaluator's scores in reports. Unique within one evaluation."""

    @abstractmethod
    def __call__(self, ctx: EvalContext) -> EvaluatorResult | Awaitable[EvaluatorResult]:
        """Score one run."""

    def _verdict(
        self, passed: bool, reason: str = "", score: float | None = None
    ) -> EvaluatorResult:
        return EvaluatorResult(
            name=self.name,
            score=float(passed) if score is None else score,
            passed=passed,
            reason=reason,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class ExactMatch(Evaluator):
    """The output equals the expected value.

    Text is compared after stripping surrounding whitespace (``strip``) and,
    with ``case_sensitive=False``, ignoring case. Other values are compared as
    JSON data, so a pydantic model equals the dict it serialises to; a text
    answer is read as JSON when the expected value is not text.
    """

    def __init__(
        self,
        expected: Any = None,
        *,
        strip: bool = True,
        case_sensitive: bool = True,
        name: str = "exact_match",
    ) -> None:
        """
        Args:
            expected: The value to compare with. ``None`` uses the case's ``expected``.
            strip: Ignore whitespace around text.
            case_sensitive: Compare text with its case.
            name: Key of the scores in reports.
        """
        self.expected = expected
        self.strip = strip
        self.case_sensitive = case_sensitive
        self.name = name

    def __call__(self, ctx: EvalContext) -> EvaluatorResult:
        expected = self.expected if self.expected is not None else ctx.expected
        if expected is None:
            return self._verdict(False, _NO_EXPECTED.format(case=ctx.case.name, cls="ExactMatch"))
        if ctx.result.status != "completed":
            return self._verdict(False, f"no output: {_ended(ctx)}")
        actual = ctx.output
        if isinstance(expected, str) and isinstance(actual, str):
            same = self._normal(actual) == self._normal(expected)
        else:
            if isinstance(actual, str):
                try:
                    actual = extract_json(actual)
                except ValueError:
                    return self._verdict(False, f"expected {_preview(expected)}, got text")
            same = to_jsonable_python(actual, fallback=str) == to_jsonable_python(
                expected, fallback=str
            )
        if same:
            return self._verdict(True)
        return self._verdict(False, f"expected {_preview(expected)}, got {_preview(actual)}")

    def _normal(self, text: str) -> str:
        text = text.strip() if self.strip else text
        return text if self.case_sensitive else text.casefold()


class Contains(Evaluator):
    """The answer text (`EvalContext.answer_text`) contains a value, or every one of several.

    The score is the fraction of values found; the evaluator passes when all are.
    """

    def __init__(
        self,
        value: str | Sequence[str] | None = None,
        *,
        case_sensitive: bool = True,
        name: str = "contains",
    ) -> None:
        """
        Args:
            value: Text that must appear, or a list of them. ``None`` uses the
                case's ``expected``, which must then be text or a list of text.
            case_sensitive: Match with case. ``False`` ignores it.
            name: Key of the scores in reports.
        """
        self.value = value
        self.case_sensitive = case_sensitive
        self.name = name

    def __call__(self, ctx: EvalContext) -> EvaluatorResult:
        value = self.value if self.value is not None else ctx.expected
        if value is None:
            return self._verdict(False, _NO_EXPECTED.format(case=ctx.case.name, cls="Contains"))
        needles = [value] if isinstance(value, str) else list(value)
        if not needles or not all(isinstance(item, str) for item in needles):
            return self._verdict(False, "Contains needs text, or a non-empty list of text")
        answer = ctx.answer_text
        haystack = answer if self.case_sensitive else answer.casefold()
        missing = [
            needle
            for needle in needles
            if (needle if self.case_sensitive else needle.casefold()) not in haystack
        ]
        score = (len(needles) - len(missing)) / len(needles)
        if not missing:
            return self._verdict(True, score=score)
        return self._verdict(False, f"missing: {', '.join(map(repr, missing))}", score=score)


class Regex(Evaluator):
    """The answer text (`EvalContext.answer_text`) matches a regular expression."""

    def __init__(
        self, pattern: str, *, flags: int = 0, fullmatch: bool = False, name: str = "regex"
    ) -> None:
        """
        Args:
            pattern: A Python regular expression, searched anywhere in the text.
            flags: ``re`` flags, for example ``re.IGNORECASE``.
            fullmatch: The whole text (stripped) must match, not just a part of it.
            name: Key of the scores in reports.

        Raises:
            ConfigurationError: The pattern does not compile.
        """
        try:
            self.pattern = re.compile(pattern, flags)
        except re.error as error:
            raise ConfigurationError(
                f"Regex pattern {pattern!r} does not compile: {error}"
            ) from error
        self.fullmatch = fullmatch
        self.name = name

    def __call__(self, ctx: EvalContext) -> EvaluatorResult:
        text = ctx.answer_text
        found = (
            self.pattern.fullmatch(text.strip()) if self.fullmatch else self.pattern.search(text)
        )
        if found:
            return self._verdict(True)
        how = "match" if self.fullmatch else "contain a match for"
        return self._verdict(False, f"the answer does not {how} {self.pattern.pattern!r}")


class JSONSchemaValid(Evaluator):
    """The output is valid against a JSON Schema.

    A structured output (any ``output_type`` value, text included) is checked
    as it serialises. A plain-text answer is read as JSON first (bare, fenced,
    or surrounded by a few words), and text that holds no JSON fails. Needs the
    ``jsonschema`` package.
    """

    def __init__(
        self, schema: Mapping[str, Any] | type[BaseModel], *, name: str = "json_schema"
    ) -> None:
        """
        Args:
            schema: A JSON Schema, or a pydantic model class whose schema is used.
            name: Key of the scores in reports.

        Raises:
            ConfigurationError: ``jsonschema`` is not installed, or the schema is invalid.
        """
        try:
            from jsonschema.exceptions import SchemaError
            from jsonschema.validators import validator_for
        except ImportError as error:
            raise ConfigurationError(
                'JSONSchemaValid needs the jsonschema package: pip install "qagents[evals]"'
            ) from error
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            schema = schema.model_json_schema()
        self.schema = dict(schema)
        validator_class = validator_for(self.schema)
        try:
            validator_class.check_schema(self.schema)
        except SchemaError as error:
            raise ConfigurationError(
                f"JSONSchemaValid got an invalid schema: {error.message}"
            ) from error
        self._validator = validator_class(self.schema)
        self.name = name

    def __call__(self, ctx: EvalContext) -> EvaluatorResult:
        if ctx.result.status != "completed":
            return self._verdict(False, f"no output: {_ended(ctx)}")
        try:
            value = _answer_value(ctx)
        except ValueError as error:
            return self._verdict(False, f"the answer is not JSON: {error}")
        errors = sorted(
            self._validator.iter_errors(to_jsonable_python(value, fallback=str)),
            key=lambda item: list(item.path),
        )
        if not errors:
            return self._verdict(True)
        first = errors[0]
        where = "/".join(str(piece) for piece in first.path) or "(root)"
        more = f" (and {len(errors) - 1} more)" if len(errors) > 1 else ""
        return self._verdict(False, f"{where}: {first.message}{more}")


class IsInstance(Evaluator):
    """The output is, or validates as, a type: a pydantic model, a dataclass,
    ``list[int]``, a ``Literal``...

    A structured output (any ``output_type`` value, text included) is validated
    as it is. A plain-text answer passes ``IsInstance(str)``; for other types it
    is read as JSON before it is validated, and text that holds no JSON fails.
    """

    def __init__(
        self, output_type: Any, *, strict: bool = False, name: str = "is_instance"
    ) -> None:
        """
        Args:
            output_type: Anything pydantic can validate.
            strict: Use pydantic's strict mode (no coercion, so ``"5"`` is not an ``int``).
            name: Key of the scores in reports.
        """
        self.output_type = output_type
        self.strict = strict
        self.name = name
        self._adapter: TypeAdapter[Any] = TypeAdapter(output_type)

    def __call__(self, ctx: EvalContext) -> EvaluatorResult:
        if ctx.result.status != "completed":
            return self._verdict(False, f"no output: {_ended(ctx)}")
        if isinstance(self.output_type, type) and isinstance(ctx.output, self.output_type):
            return self._verdict(True)
        try:
            value = _answer_value(ctx)
        except ValueError as error:
            return self._verdict(False, f"the answer is not JSON: {error}")
        try:
            self._adapter.validate_python(value, strict=self.strict)
        except ValidationError as error:
            return self._verdict(False, _brief(str(error)))
        return self._verdict(True)


class ToolCalled(Evaluator):
    """The model called a tool, optionally with certain arguments and a number of times.

    ToolCalled("get_weather")                                # at least once
    ToolCalled("get_weather", {"city": "Athens"}, times=1)   # exactly once, for Athens
    ToolCalled("delete_account", times=0)                    # never
    """

    def __init__(
        self,
        tool: str,
        arguments_subset: Mapping[str, Any] | None = None,
        times: int | None = None,
        *,
        name: str | None = None,
    ) -> None:
        """
        Args:
            tool: The tool's name.
            arguments_subset: Count only calls whose arguments contain these
                keys with these values (nested objects are matched the same way).
            times: The exact number of matching calls. ``None`` means at least one.
            name: Key of the scores in reports. Defaults to ``tool_called:<tool>``.
        """
        if times is not None and times < 0:
            raise ConfigurationError("ToolCalled(times=...) must be 0 or more.")
        self.tool = tool
        self.arguments_subset = dict(arguments_subset) if arguments_subset is not None else None
        self.times = times
        self.name = name or f"tool_called:{tool}"

    def __call__(self, ctx: EvalContext) -> EvaluatorResult:
        named = [call for call in ctx.tool_calls if call.name == self.tool]
        matching = [
            call
            for call in named
            if self.arguments_subset is None or _is_subset(self.arguments_subset, call.arguments)
        ]
        count = len(matching)
        passed = count >= 1 if self.times is None else count == self.times
        if passed:
            return self._verdict(True)
        wanted = "at least once" if self.times is None else f"{self.times} time(s)"
        detail = f" with {_preview(self.arguments_subset)}" if self.arguments_subset else ""
        return self._verdict(
            False,
            f"{self.tool!r}{detail} was called {count} time(s), expected {wanted} "
            f"({len(named)} call(s) to {self.tool!r} in total)",
        )


class MaxTurns(Evaluator):
    """The run used at most ``n`` model requests."""

    def __init__(self, n: int, *, name: str = "max_turns") -> None:
        if n < 0:
            raise ConfigurationError("MaxTurns(n) must be 0 or more.")
        self.n = n
        self.name = name

    def __call__(self, ctx: EvalContext) -> EvaluatorResult:
        if ctx.turns <= self.n:
            return self._verdict(True, f"{ctx.turns} turn(s)")
        return self._verdict(False, f"{ctx.turns} turns, limit {self.n}")


class MaxCostUSD(Evaluator):
    """The run cost at most ``usd`` dollars.

    A run whose model reported no cost fails, because nothing shows it stayed
    within budget; pass ``allow_unknown=True`` to let it pass.
    """

    def __init__(
        self, usd: float, *, allow_unknown: bool = False, name: str = "max_cost_usd"
    ) -> None:
        if usd < 0:
            raise ConfigurationError("MaxCostUSD(usd) must be 0 or more.")
        self.usd = usd
        self.allow_unknown = allow_unknown
        self.name = name

    def __call__(self, ctx: EvalContext) -> EvaluatorResult:
        cost = ctx.usage.cost_usd
        if cost is None:
            return self._verdict(self.allow_unknown, "the model reported no cost")
        if cost <= self.usd:
            return self._verdict(True, f"${cost:.6f}")
        return self._verdict(False, f"${cost:.6f}, limit ${self.usd:.6f}")


class MaxLatency(Evaluator):
    """The run took at most ``seconds`` of wall-clock time (evaluators excluded)."""

    def __init__(self, seconds: float, *, name: str = "max_latency") -> None:
        if seconds <= 0:
            raise ConfigurationError("MaxLatency(seconds) must be more than 0.")
        self.seconds = seconds
        self.name = name

    def __call__(self, ctx: EvalContext) -> EvaluatorResult:
        if ctx.latency_s <= self.seconds:
            return self._verdict(True, f"{ctx.latency_s:.2f} s")
        return self._verdict(False, f"{ctx.latency_s:.2f} s, limit {self.seconds:.2f} s")


class Custom(Evaluator):
    """Your own check, as a function of the `EvalContext` (sync or async).

    The function may return an `EvaluatorResult`, a bool, a score between 0
    and 1 (it passes at ``threshold`` or above), or a ``(value, reason)`` tuple
    of either.

        Custom(lambda ctx: len(ctx.answer_text) < 200, name="short")
    """

    def __init__(
        self, function: EvaluatorFunction, *, name: str | None = None, threshold: float = 1.0
    ) -> None:
        """
        Args:
            function: Called with the `EvalContext` of every run.
            name: Key of the scores in reports. Defaults to the function's name.
            threshold: The lowest numeric score that passes.
        """
        if not callable(function):
            raise ConfigurationError(f"Custom needs a function, got {function!r}.")
        if not 0.0 <= threshold <= 1.0:
            raise ConfigurationError("Custom(threshold=...) must be between 0 and 1.")
        self.function = function
        self.threshold = threshold
        given = name or getattr(function, "name", None) or getattr(function, "__name__", "")
        self.name = given if given and given != "<lambda>" else "custom"

    async def __call__(self, ctx: EvalContext) -> EvaluatorResult:
        value = self.function(ctx)
        if inspect.isawaitable(value):
            value = await value
        return _as_result(self.name, value, self.threshold)


class JudgeVerdict(BaseModel):
    """The structured answer of the judge. The reason comes first so that the
    model writes its justification before it commits to a score."""

    reason: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool


class LLMJudge(Evaluator):
    """A model grades the answer against your rubric.

        LLMJudge("qcloud:<model-id>", "The answer names the city and gives a temperature.")

    The judge is a qagents `Agent` with ``output_type=JudgeVerdict``. Give it a
    model (or a ``"provider:model-id"`` string) and it is built for you with
    `prompts.JUDGE_INSTRUCTIONS`; or give your own agent, whose ``output_type`` must
    have the fields ``score``, ``passed`` and ``reason``. Each grading is a
    fresh run, and a run that did not complete is not sent to the judge. The run
    passes when the judge says it passed and its score is at least
    ``threshold``; the stricter of the two wins. The judge's usage is recorded
    on its `EvaluatorResult`, apart from the case's own usage, also when the
    judge fails (it raises, refuses or gives no valid verdict): that grading
    then fails, with the usage of the requests it made.
    """

    def __init__(
        self,
        judge: Agent | Model | str,
        rubric: str,
        threshold: float = 0.7,
        *,
        name: str = "llm_judge",
        include_input: bool = True,
        include_expected: bool = True,
        output_mode: Literal["native", "prompted", "tool"] = "native",
        settings: ModelSettings | None = None,
        limits: Limits | None = None,
        hooks: Iterable[Hooks] = (),
    ) -> None:
        """
        Args:
            judge: The judge's model, or a complete judge agent.
            rubric: What a good answer does, in plain words.
            threshold: The lowest score that passes, between 0 and 1.
            name: Key of the scores in reports.
            include_input: Show the judge the case's prompt.
            include_expected: Show the judge the case's expected value, when it has one.
            output_mode: How the built judge asks for structured output:
                ``"prompted"`` for models without JSON Schema support, ``"tool"`` for
                a final-answer tool call.
            settings: Model settings of the built judge (``temperature=0`` is a good choice).
            limits: Limits of the built judge.
            hooks: Hooks of the built judge, for example a tracer.

        Raises:
            ConfigurationError: The rubric is empty, the threshold is out of range,
                or a given agent has no ``output_type``.
        """
        if not rubric.strip():
            raise ConfigurationError("LLMJudge needs a rubric: describe what a good answer does.")
        if not 0.0 <= threshold <= 1.0:
            raise ConfigurationError("LLMJudge(threshold=...) must be between 0 and 1.")
        if isinstance(judge, Agent):
            if judge.output_type is None:
                raise ConfigurationError(
                    f"The judge agent {judge.name!r} has no output_type. Give it "
                    "output_type=JudgeVerdict, or pass a model and let LLMJudge build the agent."
                )
            self.agent = judge
        else:
            self.agent = Agent(
                judge,
                name="judge",
                instructions=_judge_instructions,
                output_type=JudgeVerdict,
                output_mode=output_mode,
                settings=settings,
                limits=limits,
                hooks=list(hooks),
            )
        self.rubric = rubric
        self.threshold = threshold
        self.include_input = include_input
        self.include_expected = include_expected
        self.name = name

    def prompt(self, ctx: EvalContext) -> str:
        """The exact message the judge receives for this run."""
        task = (
            prompts.JUDGE_TASK.format(task=_render_prompt(ctx.case.prompt))
            if self.include_input
            else ""
        )
        reference = ""
        if self.include_expected and ctx.expected is not None:
            reference = prompts.JUDGE_REFERENCE.format(reference=_render_value(ctx.expected))
        return prompts.JUDGE_PROMPT.format(
            rubric=self.rubric, task=task, reference=reference, answer=_render_answer(ctx)
        )

    async def __call__(self, ctx: EvalContext) -> EvaluatorResult:
        if ctx.result.status != "completed":
            # Grading an answer that was never finished costs a request and proves nothing.
            return self._verdict(False, f"not graded: {_ended(ctx)}")
        stream: RunStream | None = None
        try:
            stream = self.agent.stream(self.prompt(ctx))
            result = await stream.wait()
        except Exception as error:  # the requests it made before failing were billed
            logger.debug("The judge raised on case %r", ctx.case.name, exc_info=True)
            reason = f"the evaluator raised {type(error).__name__}: {_brief(str(error))}"
            return self._failed(reason, _usage_so_far(stream))
        if result.status != "completed":
            reason = result.stop_reason or result.status
            return self._failed(f"the judge did not finish ({reason})", result.usage)
        if result.output is None:  # a refusal (content_filter) gives no structured output
            reason = result.stop_reason or "no output"
            return self._failed(f"the judge gave no verdict ({reason})", result.usage)
        try:
            verdict = JudgeVerdict.model_validate(to_jsonable_python(result.output, fallback=str))
        except ValidationError as error:
            reason = f"the judge's answer is not a verdict: {_brief(str(error))}"
            return self._failed(reason, result.usage)
        return EvaluatorResult(
            name=self.name,
            score=verdict.score,
            passed=verdict.passed and verdict.score >= self.threshold,
            reason=verdict.reason,
            usage=result.usage,
        )

    def _failed(self, reason: str, usage: Usage) -> EvaluatorResult:
        return EvaluatorResult(name=self.name, score=0.0, passed=False, reason=reason, usage=usage)


# --- results and reports ----------------------------------------------------------


class CaseResult(BaseModel):
    """One run of one case: what came out, how it scored, and what it cost."""

    case: str
    """The case's name."""
    repeat: int = 0
    passed: bool = False
    status: str | None = None
    """The run's status (``completed``, ``paused``, ``limit_reached``...), or ``None``
    when running the case raised."""
    stop_reason: str | None = None
    output_text: str = ""
    """The answer as text (`EvalContext.answer_text`): the output when it is text (also a
    final-answer tool's text), otherwise the text of the last assistant message."""
    output: Any = None
    """The output as JSON data when the agent has an ``output_type`` (a ``Literal`` or ``Enum``
    of strings included; ``output_type=str`` is a plain-text agent, whose answer is only in
    ``output_text``), or when a target function's `RunResult` holds an output that is not
    text."""
    evaluations: list[EvaluatorResult] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    """The run's own usage, also when it raised: the requests made before the failure count.
    Evaluator (judge) usage is in ``judge_usage``."""
    latency_s: float = 0.0
    turns: int = 0
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    error: str | None = None
    """``"ExceptionType: message"`` when running the case raised."""
    error_type: str | None = None
    session_id: str | None = None
    run_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    """The case's metadata."""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def scores(self) -> dict[str, float]:
        """Score per evaluator name."""
        return {item.name: item.score for item in self.evaluations}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cost_usd(self) -> float | None:
        return self.usage.cost_usd

    @computed_field  # type: ignore[prop-decorator]
    @property
    def judge_usage(self) -> Usage:
        """Usage of the evaluators that called a model, summed."""
        total = Usage()
        for item in self.evaluations:
            if item.usage is not None:
                total = total.add(item.usage)
        return total

    @property
    def outcome(self) -> Literal["pass", "fail", "error"]:
        if self.error is not None:
            return "error"
        return "pass" if self.passed else "fail"


class EvalSummary(BaseModel):
    """Aggregates over every run of a report."""

    cases: int
    runs: int
    """Cases times repeats."""
    passed: int
    failed: int
    """Runs that did not pass, errors included."""
    errors: int
    """Runs that raised."""
    pass_rate: float
    mean_scores: dict[str, float] = Field(default_factory=dict)
    """Mean score per evaluator. A run that raised counts as 0 for every evaluator."""
    pass_rates: dict[str, float] = Field(default_factory=dict)
    """Fraction of runs each evaluator passed."""
    total_cost_usd: float | None = None
    """``None`` when no run reported a cost. Costs and tokens include the requests a run made
    before it raised."""
    mean_cost_usd: float | None = None
    """Over the runs that reported a cost."""
    latency_mean_s: float | None = None
    latency_p50_s: float | None = None
    latency_p95_s: float | None = None
    """Latency figures cover the runs that did not raise."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    mean_turns: float = 0.0
    judge_cost_usd: float | None = None
    judge_tokens: int = 0
    pass_all_repeats: float | None = None
    """With ``repeat > 1``: fraction of cases whose every repetition passed."""
    pass_any_repeat: float | None = None
    """With ``repeat > 1``: fraction of cases with at least one passing repetition."""


class EvalReport(BaseModel):
    """The results of `evaluate`, with their aggregates in ``summary``."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str = "evaluation"
    """What was evaluated, for example ``"agent (qcloud:<model-id>)"``."""
    dataset: str = "dataset"
    started_at: str = ""
    """ISO 8601, UTC."""
    duration_s: float = 0.0
    repeat: int = 1
    concurrency: int = 1
    require_completed: bool = True
    evaluators: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    results: list[CaseResult] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def summary(self) -> EvalSummary:
        return _summarize(self)

    def failures(self) -> list[CaseResult]:
        """The runs that did not pass, errors included."""
        return [item for item in self.results if not item.passed]

    def to_json(self, *, indent: int | None = 2) -> str:
        """The whole report, summary included, as JSON."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, text: str) -> EvalReport:
        return cls.model_validate_json(text)

    def save(self, path: str | Path) -> None:
        """Write the report: ``.json`` for the full report, ``.md`` for `to_markdown`."""
        target = Path(path)
        suffix = target.suffix.lower()
        if suffix == ".json":
            target.write_text(self.to_json(), encoding="utf-8", newline="\n")
        elif suffix in (".md", ".markdown"):
            target.write_text(self.to_markdown(), encoding="utf-8", newline="\n")
        else:
            raise ConfigurationError(
                f"Cannot tell how to save {target.name!r}: use a .json or a .md file name."
            )

    @classmethod
    def load(cls, path: str | Path) -> EvalReport:
        """Read a report written by ``save`` to a ``.json`` file."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def to_markdown(self, *, include_content: bool = True, max_chars: int = 80) -> str:
        """A summary table and a table of every run.

        Args:
            include_content: Show output text, error messages and evaluator
                reasons. ``False`` shows only error types and the names of the
                evaluators that failed, for reports that leave your machine.
            max_chars: Longest text shown in one cell.
        """
        summary = self.summary
        runs = f"{summary.cases} case(s) x {self.repeat} = {summary.runs} run(s)"
        lines = [
            f"## Evaluation: {_cell(self.name)}",
            "",
            f"Dataset `{_cell(self.dataset)}`, {runs}, started {self.started_at or 'n/a'}, "
            f"took {self.duration_s:.1f} s.",
            "",
            "| Metric | Value |",
            "|---|---|",
        ]
        lines += [f"| {label} | {value} |" for label, value in _summary_rows(summary)]
        names = list(self.evaluators)
        header = ["Case", *(["Run"] if self.repeat > 1 else []), "Result", *names]
        header += ["Turns", "Latency", "Cost", "Notes"]
        if include_content:
            header.append("Output")
        lines += ["", "| " + " | ".join(_cell(item) for item in header) + " |"]
        lines.append("|" + "---|" * len(header))
        for item in self.results:
            scores = item.scores
            row = [_cell(item.case, max_chars), *([str(item.repeat)] if self.repeat > 1 else [])]
            row.append(item.outcome)
            row += [f"{scores[name]:.2f}" if name in scores else "" for name in names]
            row += [str(item.turns), f"{item.latency_s:.2f} s", _money(item.cost_usd)]
            row.append(_cell(_notes(item, include_content), max_chars))
            if include_content:
                row.append(_cell(_output_preview(item), max_chars))
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines) + "\n"


# --- running ----------------------------------------------------------------------


@dataclass(frozen=True)
class TrialTarget:
    """An evaluation factory receiving (case, zero-based repetition).

    Use the repetition to choose a reproducible seed or create fresh application
    state. The function may be sync or async and returns an Agent or RunResult.
    Existing one-argument case targets keep their behavior.
    """

    function: Callable[[Case, int], Any]

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise ConfigurationError("TrialTarget requires a callable")


Target = Agent | Callable[[Case], Any] | TrialTarget
"""An agent, or a function of the case (sync or async) returning an `Agent` to run
or a `RunResult` it produced itself."""


async def evaluate(
    target: Target,
    dataset: Dataset | Iterable[Case],
    evaluators: Iterable[Evaluator | EvaluatorFunction] = (),
    *,
    concurrency: int = 4,
    repeat: int = 1,
    limits: Limits | None = None,
    name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    store: SessionStore | None = None,
    require_completed: bool = True,
    on_result: Callable[[CaseResult], Any] | None = None,
) -> EvalReport:
    """Run every case through ``target`` and score each run.

    Args:
        target: An `Agent`, a function of the `Case`, or a `TrialTarget` whose
            function also receives the repetition index. The function returns
            an `Agent` to run or a `RunResult` it produced itself. A
            sync function runs in a worker thread, so it may block.
        dataset: The cases to run.
        evaluators: Evaluators, or plain functions (wrapped in `Custom`).
            Their names must be unique.
        concurrency: Most runs in flight at once, evaluators included.
        repeat: Times to run every case. Each repetition is its own `CaseResult`.
        limits: Limits for the agent runs, instead of the agent's own. They do
            not apply to a `RunResult` a target function returns.
        name: What the report calls the target. Defaults to the agent's name,
            provider and model.
        metadata: Anything to record with the report (logged as MLflow params).
        store: Where the case sessions are kept, so failed runs can be inspected
            afterwards. By default every case gets a fresh in-memory session.
        require_completed: A run that did not complete (paused, limit reached,
            cancelled) fails its case, whatever the evaluators say.
        on_result: Called with every `CaseResult` as it is ready (sync or async),
            for progress output. If it raises, the evaluation stops with that error.

    Returns:
        The report, with results in dataset order (then repetition).

    Raises:
        ConfigurationError: The target, the dataset or an evaluator is unusable.
            Exceptions raised while running a case are recorded, not raised.
    """
    cases = dataset if isinstance(dataset, Dataset) else Dataset(dataset, name="cases")
    if not cases.cases:
        raise ConfigurationError("The dataset has no cases. Add at least one Case.")
    if concurrency < 1:
        raise ConfigurationError("evaluate(concurrency=...) must be 1 or more.")
    if repeat < 1:
        raise ConfigurationError("evaluate(repeat=...) must be 1 or more.")
    if not isinstance(target, (Agent, TrialTarget)) and not callable(target):
        raise ConfigurationError(
            f"Cannot evaluate {target!r}: pass an Agent, or a function of the case that "
            "returns an Agent or a RunResult."
        )
    checks = _normalize_evaluators(evaluators)
    report_id = uuid.uuid4().hex
    gate = asyncio.Semaphore(concurrency)
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    began = time.perf_counter()

    def new_session(case: Case, index: int) -> Session:
        if store is None:
            return Session()
        return Session(store, id=f"eval:{report_id[:12]}:{case.name}:{index}")

    async def one(case: Case, index: int) -> CaseResult:
        async with gate:
            outcome = await _run_case(
                target, case, index, checks, limits, new_session, require_completed
            )
        if on_result is not None:
            reported = on_result(outcome)
            if inspect.isawaitable(reported):
                await reported
        return outcome

    tasks = [
        asyncio.ensure_future(one(case, index)) for case in cases.cases for index in range(repeat)
    ]
    try:
        results = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    recorded: dict[str, Any] = {}
    if isinstance(target, Agent):
        recorded = {
            "agent": target.name,
            "provider": target.model.provider,
            "model": target.model.name,
        }
    recorded.update(metadata or {})
    return EvalReport(
        id=report_id,
        name=name or _target_label(target),
        dataset=cases.name,
        started_at=started_at,
        duration_s=time.perf_counter() - began,
        repeat=repeat,
        concurrency=concurrency,
        require_completed=require_completed,
        evaluators=[check.name for check in checks],
        metadata=recorded,
        results=list(results),
    )


def evaluate_sync(
    target: Target,
    dataset: Dataset | Iterable[Case],
    evaluators: Iterable[Evaluator | EvaluatorFunction] = (),
    **options: Any,
) -> EvalReport:
    """`evaluate` for code that is not async. Do not call it from a running event loop."""
    return asyncio.run(evaluate(target, dataset, evaluators, **options))


async def _run_case(
    target: Target,
    case: Case,
    index: int,
    checks: list[Evaluator],
    limits: Limits | None,
    new_session: Callable[[Case, int], Session],
    require_completed: bool,
) -> CaseResult:
    session_id: str | None = None
    result: RunResult | None = None
    error: BaseException | None = None
    stream: RunStream | None = None
    structured: bool | None = None
    began = time.perf_counter()
    try:
        produced = await _produce(target, case, index)
        if isinstance(produced, Agent):
            structured = produced.output_type is not None
            session = new_session(case, index)
            session_id = session.id
            history, prompt = _split_prompt(case.prompt)
            stream = produced.stream(
                prompt, history=history, deps=case.deps, session=session, limits=limits
            )
            result = await stream.wait()
        elif isinstance(produced, RunResult):
            result = produced
        else:
            raise ConfigurationError(
                f"The evaluation target returned {type(produced).__name__}; "
                "return an Agent or a RunResult."
            )
    except Exception as caught:
        error = caught
        logger.debug("Case %r (repeat %d) raised", case.name, index, exc_info=True)
    latency = time.perf_counter() - began

    if result is None:
        assert error is not None
        message = f"{type(error).__name__}: {error}"
        return CaseResult(
            case=case.name,
            repeat=index,
            passed=False,
            evaluations=[
                EvaluatorResult(
                    name=check.name,
                    score=0.0,
                    passed=False,
                    reason="not evaluated: running the case raised",
                )
                for check in checks
            ],
            usage=_usage_so_far(stream),
            latency_s=latency,
            error=message,
            error_type=type(error).__name__,
            session_id=session_id,
            run_id=stream.run_id if stream is not None else None,
            metadata=dict(case.metadata),
        )

    calls = _tool_calls(result.new_messages)
    ctx = EvalContext(
        case=case,
        result=result,
        repeat=index,
        latency_s=latency,
        tool_calls=calls,
        structured=structured,
    )
    evaluations = [await _apply(check, ctx) for check in checks]
    completed = result.status == "completed"
    passed = (completed or not require_completed) and all(item.passed for item in evaluations)
    if structured is None:  # a RunResult from a target function: text is in output_text
        structured = not isinstance(result.output, str)
    return CaseResult(
        case=case.name,
        repeat=index,
        passed=passed,
        status=result.status,
        stop_reason=result.stop_reason,
        output_text=ctx.answer_text,
        output=to_jsonable_python(result.output, fallback=str) if structured else None,
        evaluations=evaluations,
        usage=result.usage,
        latency_s=latency,
        turns=result.turns,
        tool_calls=calls,
        session_id=result.session_id or session_id,
        run_id=result.run_id,
        metadata=dict(case.metadata),
    )


async def _produce(target: Target, case: Case, index: int) -> Any:
    if isinstance(target, Agent):
        return target
    function = target.function if isinstance(target, TrialTarget) else target
    args = (case, index) if isinstance(target, TrialTarget) else (case,)
    if _is_async_callable(function):
        produced = function(*args)
    else:
        produced = await asyncio.to_thread(function, *args)
    if inspect.isawaitable(produced):
        produced = await produced
    return produced


async def _apply(check: Evaluator, ctx: EvalContext) -> EvaluatorResult:
    try:
        value = check(ctx)
        if inspect.isawaitable(value):
            value = await value
        if not isinstance(value, EvaluatorResult):
            raise TypeError(
                f"returned {type(value).__name__}; an evaluator must return an EvaluatorResult"
            )
    except Exception as error:
        logger.debug("Evaluator %r raised on case %r", check.name, ctx.case.name, exc_info=True)
        return EvaluatorResult(
            name=check.name,
            score=0.0,
            passed=False,
            reason=f"the evaluator raised {type(error).__name__}: {_brief(str(error))}",
        )
    if value.name != check.name:
        value = value.model_copy(update={"name": check.name})
    return value


def _normalize_evaluators(items: Iterable[Evaluator | EvaluatorFunction]) -> list[Evaluator]:
    checks: list[Evaluator] = []
    for item in items:
        if isinstance(item, Evaluator):
            checks.append(item)
        elif callable(item):
            checks.append(Custom(item))
        else:
            raise ConfigurationError(
                f"{item!r} is not an evaluator. Pass an Evaluator or a function of the EvalContext."
            )
    names = [check.name for check in checks]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ConfigurationError(
            f"Two or more evaluators are named {', '.join(map(repr, repeated))}. "
            "Give each one a distinct name=..."
        )
    return checks


# --- comparing reports ---------------------------------------------------------------


def compare(
    a: EvalReport,
    b: EvalReport,
    *,
    labels: tuple[str, str] | None = None,
    changed_only: bool = False,
) -> str:
    """A markdown table of how ``b`` differs from ``a``, metric by metric and case by case.

    Typical use: the same dataset on two providers or models, or a saved
    baseline against today's run.

    Args:
        a: The baseline.
        b: The report to compare with it.
        labels: Column titles. Default to the report names.
        changed_only: List only the cases whose pass rate changed.
    """
    first, second = labels or _labels(a, b)
    sa, sb = a.summary, b.summary
    rows: list[tuple[str, float | None, float | None, str]] = [
        ("Pass rate", sa.pass_rate, sb.pass_rate, "rate"),
        ("Errors", sa.errors, sb.errors, "count"),
    ]
    for name in _union(a.evaluators, b.evaluators):
        rows.append(
            (f"Mean score: {name}", sa.mean_scores.get(name), sb.mean_scores.get(name), "score")
        )
    rows += [
        ("Total cost", sa.total_cost_usd, sb.total_cost_usd, "money"),
        ("Mean cost per run", sa.mean_cost_usd, sb.mean_cost_usd, "money"),
        ("Latency p50", sa.latency_p50_s, sb.latency_p50_s, "seconds"),
        ("Latency p95", sa.latency_p95_s, sb.latency_p95_s, "seconds"),
        ("Total tokens", sa.total_tokens, sb.total_tokens, "count"),
        ("Mean turns", sa.mean_turns, sb.mean_turns, "float"),
    ]
    if sa.judge_tokens or sb.judge_tokens:
        rows.append(("Judge cost", sa.judge_cost_usd, sb.judge_cost_usd, "money"))
    lines = [
        f"| Metric | {_cell(first)} | {_cell(second)} | Change |",
        "|---|---|---|---|",
    ]
    for label, left, right, kind in rows:
        lines.append(
            f"| {_cell(label)} | {_format(left, kind)} | {_format(right, kind)} "
            f"| {_change(left, right, kind)} |"
        )

    left_cases, right_cases = _per_case(a), _per_case(b)
    case_lines: list[str] = []
    for case in _union(list(left_cases), list(right_cases)):
        left, right = left_cases.get(case), right_cases.get(case)
        if left is None or right is None:
            change = "only in " + (first if right is None else second)
        else:
            change = _case_change(left, right)
        if changed_only and not change:
            continue
        case_lines.append(
            f"| {_cell(case)} | {_case_cell(left)} | {_case_cell(right)} | {_cell(change)} |"
        )
    if case_lines:
        lines += ["", f"| Case | {_cell(first)} | {_cell(second)} | Change |", "|---|---|---|---|"]
        lines += case_lines
    return "\n".join(lines) + "\n"


def regressions(baseline: EvalReport, current: EvalReport) -> list[str]:
    """Names of the cases that pass less often in ``current`` than in ``baseline``.

    Cases present in only one of the reports are not counted.

        assert not regressions(EvalReport.load("baseline.json"), report)
    """
    before, after = _per_case(baseline), _per_case(current)
    return [
        case
        for case, outcomes in before.items()
        if case in after and _rate(after[case]) < _rate(outcomes)
    ]


# --- MLflow ------------------------------------------------------------------------------


def log_to_mlflow(
    report: EvalReport,
    run_name: str | None = None,
    *,
    record_content: bool = False,
    parent_run_id: str | None = None,
) -> str:
    """Record a report as a run in the active MLflow experiment.

    Logs the summary as metrics (``pass_rate``, ``mean_score/<evaluator>``,
    costs, latency percentiles, tokens), the report's settings and
    ``metadata`` as params, and two artifacts: ``qagents_eval/cases.json``
    (one row per run) and ``qagents_eval/report.md``. When an MLflow run is
    active in the calling thread, or ``parent_run_id`` is given, the report is
    logged as a nested run inside it, so several reports (one per provider,
    say) sit side by side.

    Output text, structured outputs, tool arguments, error messages and
    evaluator reasons stay out of MLflow unless ``record_content=True``.

    MLflow's client blocks; from async code, call this through
    ``asyncio.to_thread``. MLflow keeps the active run per thread, so the worker
    thread does not see yours: pass its id.

        parent = mlflow.active_run()
        await asyncio.to_thread(
            log_to_mlflow, report, parent_run_id=parent.info.run_id if parent else None
        )

    Args:
        report: What `evaluate` returned.
        run_name: The MLflow run's name. Defaults to the report's name.
        record_content: Also log content, and the full report as
            ``qagents_eval/report.json``.
        parent_run_id: Log the report as a nested run of this MLflow run.
            ``None`` nests it in the calling thread's active run, if any.

    Returns:
        The MLflow run id.

    Raises:
        ConfigurationError: MLflow is not installed.
    """
    try:
        import mlflow
    except ImportError as error:
        raise ConfigurationError(
            'log_to_mlflow needs MLflow: pip install "qagents[mlflow]"'
        ) from error

    summary = report.summary
    metrics: dict[str, float] = {
        "pass_rate": summary.pass_rate,
        "runs": summary.runs,
        "passed": summary.passed,
        "errors": summary.errors,
        "input_tokens": summary.input_tokens,
        "output_tokens": summary.output_tokens,
        "total_tokens": summary.total_tokens,
        "mean_turns": summary.mean_turns,
        "duration_s": report.duration_s,
        "judge_tokens": summary.judge_tokens,
    }
    optional = {
        "total_cost_usd": summary.total_cost_usd,
        "mean_cost_usd": summary.mean_cost_usd,
        "latency_mean_s": summary.latency_mean_s,
        "latency_p50_s": summary.latency_p50_s,
        "latency_p95_s": summary.latency_p95_s,
        "judge_cost_usd": summary.judge_cost_usd,
        "pass_all_repeats": summary.pass_all_repeats,
        "pass_any_repeat": summary.pass_any_repeat,
    }
    metrics.update({key: value for key, value in optional.items() if value is not None})
    for name, value in summary.mean_scores.items():
        metrics[_mlflow_key(f"mean_score/{name}")] = value
    for name, value in summary.pass_rates.items():
        # Not "pass_rate/<name>": stores that keep metrics as files cannot have both.
        metrics[_mlflow_key(f"evaluator_pass_rate/{name}")] = value

    params: dict[str, str] = {
        "target": report.name,
        "dataset": report.dataset,
        "cases": str(summary.cases),
        "repeat": str(report.repeat),
        "concurrency": str(report.concurrency),
        "require_completed": str(report.require_completed),
        "evaluators": ", ".join(report.evaluators),
    }
    for key, value in report.metadata.items():
        rendered = (
            value if isinstance(value, str) else json.dumps(to_jsonable_python(value, fallback=str))
        )
        params.setdefault(_mlflow_key(str(key)), rendered)
    params = {key: value[:_MLFLOW_PARAM_CHARS] for key, value in params.items()}

    nesting: dict[str, Any] = {"nested": mlflow.active_run() is not None}
    if parent_run_id is not None:
        nesting = {"nested": True, "parent_run_id": parent_run_id}
    with mlflow.start_run(run_name=run_name or report.name, **nesting) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"qagents.evaluation_id": report.id, "qagents.dataset": report.dataset})
        rows = [_case_row(item, record_content) for item in report.results]
        mlflow.log_dict({"cases": rows}, "qagents_eval/cases.json")
        mlflow.log_text(
            report.to_markdown(include_content=record_content), "qagents_eval/report.md"
        )
        if record_content:
            mlflow.log_text(report.to_json(), "qagents_eval/report.json")
    return run.info.run_id


# --- helpers ---------------------------------------------------------------------------

_NO_EXPECTED = (
    "case {case!r} has no expected value: set Case.expected, or give {cls} a value of its own"
)
_MLFLOW_PARAM_CHARS = 6000
_MLFLOW_UNSAFE = re.compile(r"[^/\w.\- ]")


def _judge_instructions(ctx: RunContext[Any]) -> str:
    # Read at run time, so replacing prompts.JUDGE_INSTRUCTIONS takes effect for existing judges.
    return prompts.JUDGE_INSTRUCTIONS


def _message_from_content(item: Any) -> Any:
    if isinstance(item, Mapping) and "parts" not in item and isinstance(item.get("content"), str):
        rest = {key: value for key, value in item.items() if key != "content"}
        return {**rest, "parts": [{"type": "text", "text": item["content"]}]}
    return item


def _case_record(case: Case) -> dict[str, Any]:
    record: dict[str, Any] = {"name": case.name}
    if isinstance(case.prompt, str):
        record["prompt"] = case.prompt
    else:
        record["prompt"] = [message.model_dump(mode="json") for message in case.prompt]
    if case.expected is not None:
        record["expected"] = to_jsonable_python(case.expected, fallback=str)
    if case.metadata:
        record["metadata"] = to_jsonable_python(case.metadata, fallback=str)
    return record


def _split_prompt(prompt: str | list[Message]) -> tuple[list[Message], str | Message | None]:
    if isinstance(prompt, str):
        return [], prompt
    if prompt[-1].role == "user":
        return list(prompt[:-1]), prompt[-1]
    return list(prompt), None


def _tool_calls(messages: Sequence[Message]) -> list[ToolCallRecord]:
    outcomes = {
        part.tool_call_id: part.is_error
        for message in messages
        if message.role == "tool"
        for part in message.tool_results
    }
    return [
        ToolCallRecord(
            id=call.id, name=call.name, arguments=call.arguments, is_error=outcomes.get(call.id)
        )
        for message in messages
        if message.role == "assistant"
        for call in message.tool_calls
    ]


def _is_subset(expected: Any, actual: Any) -> bool:
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and all(
            key in actual and _is_subset(value, actual[key]) for key, value in expected.items()
        )
    return to_jsonable_python(expected, fallback=str) == to_jsonable_python(actual, fallback=str)


def _as_result(name: str, value: Any, threshold: float) -> EvaluatorResult:
    if isinstance(value, EvaluatorResult):
        return value if value.name else value.model_copy(update={"name": name})
    reason = ""
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], str):
        value, reason = value
    if isinstance(value, bool):
        return EvaluatorResult(name=name, score=float(value), passed=value, reason=reason)
    if isinstance(value, int | float):
        score = float(value)
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"the score {value!r} is not between 0 and 1")
        return EvaluatorResult(name=name, score=score, passed=score >= threshold, reason=reason)
    raise TypeError(
        f"returned {type(value).__name__}; return an EvaluatorResult, a bool, a score between "
        "0 and 1, or a (value, reason) tuple"
    )


def _is_async_callable(function: Any) -> bool:
    return inspect.iscoroutinefunction(function) or inspect.iscoroutinefunction(
        getattr(function, "__call__", None)  # noqa: B004 - callable objects with async __call__
    )


def _target_label(target: Target) -> str:
    if isinstance(target, Agent):
        return f"{target.name} ({target.model.provider}:{target.model.name})"
    function = target.function if isinstance(target, TrialTarget) else target
    return getattr(function, "__name__", None) or type(function).__name__


def _ended(ctx: EvalContext) -> str:
    result = ctx.result
    detail = f" ({result.stop_reason})" if result.stop_reason else ""
    return f"the run ended {result.status}{detail}"


def _render_prompt(prompt: str | list[Message]) -> str:
    if isinstance(prompt, str):
        return prompt
    return "\n".join(
        prompts.JUDGE_MESSAGE.format(role=item.role, text=item.text) for item in prompt
    )


def _render_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(to_jsonable_python(value, fallback=str), ensure_ascii=False)


def _render_answer(ctx: EvalContext) -> str:
    # The output, text included: a final-answer tool's text is not in the last message.
    return _render_value(ctx.output) if ctx.output is not None else ctx.text


def _answer_value(ctx: EvalContext) -> Any:
    """The value a schema or type check looks at.

    A structured output is used as it is, a validated text one (a ``Literal`` or ``Enum`` of
    strings) included. A plain-text answer, or a text output whose kind is unknown, is read
    as JSON.

    Raises:
        ValueError: The text answer holds no JSON. It is never checked as it is: a schema
            without ``type`` would accept any text.
    """
    value = ctx.output
    if isinstance(value, str) and not ctx.structured:
        return extract_json(value)
    return value


def _usage_so_far(stream: RunStream | None) -> Usage:
    """The usage of a run that raised: the requests made before the failure were billed."""
    return stream._run.usage if stream is not None else Usage()


def _preview(value: Any, limit: int = 200) -> str:
    text = repr(value) if isinstance(value, str) else _render_value(value)
    return text if len(text) <= limit else f"{text[:limit]}..."


def _brief(text: str, limit: int = 500) -> str:
    return text if len(text) <= limit else f"{text[:limit]}..."


def _percentile(values: Sequence[float], q: float) -> float | None:
    """Linear interpolation between the closest ranks (NumPy's default method)."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _summarize(report: EvalReport) -> EvalSummary:
    results = report.results
    runs = len(results)
    passed = sum(item.passed for item in results)
    names = _union(report.evaluators, [e.name for item in results for e in item.evaluations])
    mean_scores: dict[str, float] = {}
    pass_rates: dict[str, float] = {}
    for name in names:
        found = [e for item in results for e in item.evaluations if e.name == name]
        if found:
            mean_scores[name] = sum(e.score for e in found) / len(found)
            pass_rates[name] = sum(e.passed for e in found) / len(found)
    costs = [item.usage.cost_usd for item in results if item.usage.cost_usd is not None]
    latencies = [item.latency_s for item in results if item.error is None]
    judge = Usage()
    for item in results:
        judge = judge.add(item.judge_usage)
    by_case = _per_case(report)
    all_repeats = any_repeat = None
    if report.repeat > 1 and by_case:
        all_repeats = sum(all(o == "pass" for o in v) for v in by_case.values()) / len(by_case)
        any_repeat = sum(any(o == "pass" for o in v) for v in by_case.values()) / len(by_case)
    input_tokens = sum(item.usage.input_tokens for item in results)
    output_tokens = sum(item.usage.output_tokens for item in results)
    return EvalSummary(
        cases=len(by_case),
        runs=runs,
        passed=passed,
        failed=runs - passed,
        errors=sum(item.error is not None for item in results),
        pass_rate=passed / runs if runs else 0.0,
        mean_scores=mean_scores,
        pass_rates=pass_rates,
        total_cost_usd=sum(costs) if costs else None,
        mean_cost_usd=sum(costs) / len(costs) if costs else None,
        latency_mean_s=sum(latencies) / len(latencies) if latencies else None,
        latency_p50_s=_percentile(latencies, 0.5),
        latency_p95_s=_percentile(latencies, 0.95),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        mean_turns=sum(item.turns for item in results) / runs if runs else 0.0,
        judge_cost_usd=judge.cost_usd,
        judge_tokens=judge.total_tokens,
        pass_all_repeats=all_repeats,
        pass_any_repeat=any_repeat,
    )


def _summary_rows(summary: EvalSummary) -> list[tuple[str, str]]:
    rows = [("Pass rate", f"{summary.pass_rate:.1%} ({summary.passed}/{summary.runs})")]
    if summary.pass_all_repeats is not None and summary.pass_any_repeat is not None:
        rows.append(("Cases passing every repeat", f"{summary.pass_all_repeats:.1%}"))
        rows.append(("Cases passing any repeat", f"{summary.pass_any_repeat:.1%}"))
    rows.append(("Errors", str(summary.errors)))
    for name, score in summary.mean_scores.items():
        rate = summary.pass_rates.get(name, 0.0)
        rows.append((f"Mean score: {_cell(name)}", f"{score:.2f} (passed {rate:.1%})"))
    rows += [
        ("Total cost", _money(summary.total_cost_usd)),
        ("Mean cost per run", _money(summary.mean_cost_usd)),
        (
            "Latency p50 / p95",
            f"{_seconds(summary.latency_p50_s)} / {_seconds(summary.latency_p95_s)}",
        ),
        (
            "Tokens (input / output)",
            f"{summary.total_tokens:,} ({summary.input_tokens:,} / {summary.output_tokens:,})",
        ),
        ("Mean turns", f"{summary.mean_turns:.2f}"),
    ]
    if summary.judge_tokens or summary.judge_cost_usd is not None:
        rows.append(
            ("Judge usage", f"{summary.judge_tokens:,} tokens, {_money(summary.judge_cost_usd)}")
        )
    return rows


def _notes(item: CaseResult, include_content: bool) -> str:
    if item.error is not None:
        return item.error if include_content else (item.error_type or "error")
    if item.status != "completed":
        return f"{item.status} ({item.stop_reason})" if item.stop_reason else str(item.status)
    failed = [e for e in item.evaluations if not e.passed]
    if not failed:
        return ""
    if include_content:
        return "; ".join(f"{e.name}: {e.reason}" if e.reason else e.name for e in failed)
    return "failed: " + ", ".join(e.name for e in failed)


def _output_preview(item: CaseResult) -> str:
    if item.output is not None:
        return _render_value(item.output)
    return item.output_text


def _case_row(item: CaseResult, record_content: bool) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case": item.case,
        "repeat": item.repeat,
        "passed": item.passed,
        "status": item.status,
        "stop_reason": item.stop_reason,
        "error_type": item.error_type,
        "turns": item.turns,
        "latency_s": item.latency_s,
        "cost_usd": item.cost_usd,
        "input_tokens": item.usage.input_tokens,
        "output_tokens": item.usage.output_tokens,
        "judge_tokens": item.judge_usage.total_tokens,
        "scores": item.scores,
        "evaluators_passed": {e.name: e.passed for e in item.evaluations},
        "tool_calls": [
            {"name": call.name, "is_error": call.is_error}
            | ({"arguments": call.arguments} if record_content else {})
            for call in item.tool_calls
        ],
        "metadata": to_jsonable_python(item.metadata, fallback=str),
    }
    if record_content:
        row["output_text"] = item.output_text
        row["output"] = item.output
        row["error"] = item.error
        row["reasons"] = {e.name: e.reason for e in item.evaluations}
    return row


def _mlflow_key(name: str) -> str:
    # MLflow accepts letters, digits, '_', '-', '.', ' ' and '/' in names on every platform.
    return _MLFLOW_UNSAFE.sub("_", name)[:250]


def _per_case(report: EvalReport) -> dict[str, list[str]]:
    outcomes: dict[str, list[str]] = {}
    for item in report.results:
        outcomes.setdefault(item.case, []).append(item.outcome)
    return outcomes


def _rate(outcomes: list[str]) -> float:
    return sum(outcome == "pass" for outcome in outcomes) / len(outcomes) if outcomes else 0.0


def _case_cell(outcomes: list[str] | None) -> str:
    if outcomes is None:
        return ""
    if len(outcomes) == 1:
        return outcomes[0]
    return f"{sum(outcome == 'pass' for outcome in outcomes)}/{len(outcomes)} passed"


def _case_change(left: list[str], right: list[str]) -> str:
    before, after = _rate(left), _rate(right)
    if after < before:
        return "regressed"
    if after > before:
        return "improved"
    return ""


def _labels(a: EvalReport, b: EvalReport) -> tuple[str, str]:
    if a.name == b.name:
        return f"{a.name} (A)", f"{b.name} (B)"
    return a.name, b.name


def _union(first: Iterable[str], second: Iterable[str]) -> list[str]:
    return list(dict.fromkeys([*first, *second]))


def _format(value: float | None, kind: str) -> str:
    if value is None:
        return "n/a"
    if kind == "rate":
        return f"{value:.1%}"
    if kind == "score":
        return f"{value:.2f}"
    if kind == "money":
        return _money(value)
    if kind == "seconds":
        return _seconds(value)
    if kind == "count":
        return f"{int(value):,}"
    return f"{value:.2f}"


def _change(left: float | None, right: float | None, kind: str) -> str:
    if left is None or right is None:
        return ""
    delta = right - left
    if kind == "rate":
        return f"{delta * 100:+.1f} pp"
    if kind == "money":
        return f"{'-' if delta < 0 else '+'}{_money(abs(delta))}"
    if kind == "seconds":
        return f"{delta:+.2f} s"
    if kind == "count":
        return f"{int(delta):+,}"
    return f"{delta:+.2f}"


def _money(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:.4f}" if value == 0 or abs(value) >= 0.001 else f"${value:.6f}"


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f} s"


def _cell(value: Any, limit: int | None = None) -> str:
    text = " ".join(str(value).split()).replace("|", "\\|")
    if limit is not None and len(text) > limit:
        text = f"{text[: max(limit - 3, 1)]}..."
    return text


__all__ = [
    "Case",
    "CaseResult",
    "Contains",
    "Custom",
    "Dataset",
    "EvalContext",
    "EvalReport",
    "EvalSummary",
    "Evaluator",
    "EvaluatorFunction",
    "EvaluatorResult",
    "ExactMatch",
    "IsInstance",
    "JSONSchemaValid",
    "JudgeVerdict",
    "LLMJudge",
    "MaxCostUSD",
    "MaxLatency",
    "MaxTurns",
    "Regex",
    "Target",
    "ToolCallRecord",
    "ToolCalled",
    "compare",
    "evaluate",
    "evaluate_sync",
    "log_to_mlflow",
    "regressions",
]
