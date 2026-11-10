"""Tools: plain Python functions the model can call.

Decorate a function with `tool`. Its signature becomes the JSON Schema
the model sees, its docstring becomes the description, and every call is
validated against the signature before your code runs:

    @tool(read_only=True)
    async def get_weather(city: str, unit: Literal["c", "f"] = "c") -> str:
        '''Look up the current weather.

        Args:
            city: City name, for example "Athens".
            unit: Temperature unit.
        '''
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model
from pydantic_core import to_jsonable_python

from .errors import ConfigurationError, ToolArgumentError
from .messages import Message, Usage
from .models.base import ToolSpec

DepsT = TypeVar("DepsT")

_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass
class RunContext(Generic[DepsT]):
    """What a tool, hook or instructions function can know about the current run.

    ``deps`` is whatever you passed to ``agent.run(..., deps=...)``: a database
    handle, the current user, an HTTP client. ``usage`` and ``messages`` are
    snapshots taken when the context was created.
    """

    deps: DepsT = None  # type: ignore[assignment]
    run_id: str = ""
    session_id: str | None = None
    agent_name: str = ""
    turn: int = 0
    tool_call_id: str | None = None
    operation_id: str | None = None
    """Identifies this tool call for good: it is the same when the call is
    repeated after a crash and a resume. Pass it as the idempotency key to
    the system the tool talks to (a payment API, a queue, a database)."""
    usage: Usage = field(default_factory=Usage)
    messages: Sequence[Message] = ()
    _emit: Callable[[Any], None] | None = field(default=None, repr=False)
    _add_usage: Callable[[Usage], None] | None = field(default=None, repr=False)

    def emit(self, event: Any) -> None:
        """Put a custom event on the run's event stream (from inside a tool)."""
        if self._emit is not None:
            self._emit(event)

    def add_usage(self, usage: Usage) -> None:
        """Count model usage a tool caused (a sub-agent, an embedding call) in the run total."""
        if self._add_usage is not None:
            self._add_usage(usage)


class Tool:
    """A callable the model can invoke. Build one with `tool`."""

    def __init__(
        self,
        function: Callable[..., Any],
        *,
        name: str,
        description: str,
        arguments_model: type[BaseModel],
        takes_context: bool,
        requires_approval: bool = False,
        read_only: bool = False,
        idempotent: bool = False,
        timeout: float | None = None,
        strict: bool = False,
    ) -> None:
        if not _NAME_PATTERN.match(name):
            raise ConfigurationError(
                f"Tool name {name!r} is not valid: use 1-64 letters, digits, '_' or '-'."
            )
        self.function = function
        self.name = name
        self.description = description
        self.arguments_model = arguments_model
        self.takes_context = takes_context
        self.requires_approval = requires_approval
        """Ask a human before running (unless a permission rule allows it)."""
        self.read_only = read_only
        """The tool changes nothing. Read-only tools may run in parallel."""
        self.idempotent = idempotent
        """Running it twice has the same effect as once, so it is safe to
        repeat after a crash."""
        self.timeout = timeout
        """Seconds before the call is abandoned; ``None`` uses the run's limit."""
        self.strict = strict
        self.parameters = _clean_schema(arguments_model.model_json_schema())

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            strict=self.strict,
        )

    def validate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Check ``arguments`` against the signature; return keyword arguments."""
        try:
            validated = self.arguments_model.model_validate(arguments)
        except ValidationError as error:
            raise ToolArgumentError(_describe_validation_error(error)) from error
        return {name: getattr(validated, name) for name in self.arguments_model.model_fields}

    async def run(
        self, ctx: RunContext[Any], arguments: dict[str, Any], *, timeout: float | None = None
    ) -> Any:
        """Validate ``arguments`` and call the function. Raises what the function raises."""
        kwargs = self.validate(arguments)
        args = (ctx,) if self.takes_context else ()
        if _is_async_callable(self.function):
            pending = self.function(*args, **kwargs)
        else:
            # A thread cannot be stopped: on timeout the call is abandoned, not killed.
            pending = asyncio.to_thread(self.function, *args, **kwargs)
        limit = self.timeout if self.timeout is not None else timeout
        result = await (asyncio.wait_for(pending, limit) if limit else pending)
        if inspect.isawaitable(result):
            result = await result
        return result

    def __repr__(self) -> str:
        return f"Tool({self.name!r})"


class Toolkit:
    """A source of tools that is resolved when a run starts.

    Subclass it when the tool list is not known up front or needs a
    connection (a remote tool server, a per-tenant catalog).
    """

    async def tools(self) -> Sequence[Tool]:
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release what the toolkit holds."""


def tool(
    function: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    requires_approval: bool = False,
    read_only: bool = False,
    idempotent: bool = False,
    timeout: float | None = None,
    strict: bool = False,
) -> Any:
    """Turn a function into a `Tool`. Use bare or with options.

    A first parameter annotated ``RunContext`` (or named ``ctx``) receives the
    run context and is hidden from the model.
    """

    def build(target: Callable[..., Any]) -> Tool:
        return _tool_from_function(
            target,
            name=name,
            description=description,
            requires_approval=requires_approval,
            read_only=read_only,
            idempotent=idempotent,
            timeout=timeout,
            strict=strict,
        )

    return build(function) if function is not None else build


def render_tool_output(value: Any) -> str:
    """The text a model reads for a tool's return value."""
    if value is None:
        return "OK"
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return json.dumps(to_jsonable_python(value, fallback=str), ensure_ascii=False)


def _tool_from_function(function: Callable[..., Any], **options: Any) -> Tool:
    signature = inspect.signature(function)
    try:
        hints = typing.get_type_hints(function, include_extras=True)
    except Exception:  # unresolvable forward references: fall back to raw annotations
        hints = {}
    parameters = list(signature.parameters.values())

    takes_context = False
    if parameters:
        first = parameters[0]
        annotation = hints.get(first.name, first.annotation)
        if _is_run_context(annotation) or (
            first.name == "ctx" and annotation is inspect.Parameter.empty
        ):
            takes_context = True
            parameters = parameters[1:]

    tool_name = options.pop("name") or getattr(function, "__name__", "tool")
    summary, documented = _parse_docstring(inspect.getdoc(function) or "")
    fields: dict[str, Any] = {}
    for parameter in parameters:
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise ConfigurationError(
                f"Tool {tool_name!r}: *args and **kwargs cannot be described to a model. "
                "Give every parameter a name and a type."
            )
        annotation = hints.get(parameter.name, parameter.annotation)
        if annotation is inspect.Parameter.empty:
            annotation = Any
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        note = documented.get(parameter.name)
        fields[parameter.name] = (
            annotation,
            Field(default, description=note) if note else default,
        )
    arguments_model = create_model(
        f"{tool_name}_arguments", __config__=ConfigDict(extra="forbid"), **fields
    )
    return Tool(
        function,
        name=tool_name,
        description=options.pop("description") or summary,
        arguments_model=arguments_model,
        takes_context=takes_context,
        **options,
    )


def _is_run_context(annotation: Any) -> bool:
    return annotation is RunContext or typing.get_origin(annotation) is RunContext


def _is_async_callable(function: Callable[..., Any]) -> bool:
    return inspect.iscoroutinefunction(function) or inspect.iscoroutinefunction(
        getattr(function, "__call__", None)  # noqa: B004 - callable objects with async __call__
    )


_SECTION = re.compile(r"^(Args|Arguments|Parameters|Returns|Raises|Yields|Examples?|Notes?):\s*$")
_PARAM = re.compile(r"^(\*{0,2}\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into its description and per-argument notes."""
    summary: list[str] = []
    params: dict[str, str] = {}
    section: str | None = None
    current: str | None = None
    params_indent = 0
    for line in doc.splitlines():
        stripped = line.strip()
        header = _SECTION.match(stripped)
        if header and not line.startswith((" ", "\t")):
            section = header.group(1)
            current = None
            continue
        if section is None:
            summary.append(line)
        elif section in ("Args", "Arguments", "Parameters"):
            match = _PARAM.match(stripped)
            indent = len(line) - len(line.lstrip())
            if match and (current is None or indent <= params_indent):
                current = match.group(1).lstrip("*")
                params[current] = match.group(2).strip()
                params_indent = indent
            elif current and stripped:
                params[current] = f"{params[current]} {stripped}".strip()
    return "\n".join(summary).strip(), params


def _clean_schema(schema: Any) -> Any:
    """Drop pydantic's auto-generated titles: they cost tokens and tell the model nothing."""
    if isinstance(schema, dict):
        return {
            key: _clean_schema(value)
            for key, value in schema.items()
            if not (key == "title" and isinstance(value, str))
        }
    if isinstance(schema, list):
        return [_clean_schema(item) for item in schema]
    return schema


def _describe_validation_error(error: ValidationError) -> str:
    lines = []
    for issue in error.errors():
        location = ".".join(str(piece) for piece in issue["loc"]) or "(arguments)"
        lines.append(f"- {location}: {issue['msg']}")
    return "\n".join(lines)
