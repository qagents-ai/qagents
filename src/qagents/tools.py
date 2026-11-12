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
import concurrent.futures
import contextlib
import contextvars
import copy
import functools
import inspect
import json
import logging
import os
import re
import threading
import typing
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model
from pydantic_core import to_jsonable_python

from . import prompts
from .errors import ConfigurationError, ToolArgumentError, ToolError
from .messages import Message, ToolResultPart, Usage
from .models.base import ToolSpec

if TYPE_CHECKING:
    from .agent import Limits
    from .permissions import Permissions

DepsT = TypeVar("DepsT")

logger = logging.getLogger("qagents")

TOOL_THREADS = min(32, (os.cpu_count() or 1) + 4)
"""Worker threads that run sync tool functions. They belong to qagents, apart from
the event loop's default executor, so tools that hang cannot starve DNS lookups or
other libraries. Set it before the first sync tool runs.

A thread cannot be killed: a sync call given up after its timeout (or when its run
is cancelled) keeps its thread until the function returns. ``asyncio.run`` does not
wait for it, but the interpreter does at exit, as for every ``concurrent.futures``
thread: a sync tool that hangs for good keeps the process from exiting. Give sync
clients their own socket timeouts, or write the tool as an ``async`` function."""

_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass
class RunContext(Generic[DepsT]):
    """What a tool, hook or instructions function can know about the current run.

    ``deps`` is whatever you passed to ``agent.run(..., deps=...)``: a database
    handle, the current user, an HTTP client. ``usage`` and ``messages`` are
    snapshots taken when the context was created. ``state`` is live: one dict
    shared by everything in the run, and saved with the session.
    """

    deps: DepsT = None  # type: ignore[assignment]
    run_id: str = ""
    session_id: str | None = None
    agent_name: str = ""
    turn: int = 0
    tool_call_id: str | None = None
    """The tool call this context belongs to; set for every tool call, ``None``
    for hooks and instructions that run outside one."""
    operation_id: str | None = None
    """Identifies this tool call for good: it is the same when the call is
    repeated after a crash and a resume. Pass it as the idempotency key to
    the system the tool talks to (a payment API, a queue, a database).
    Always set inside a tool call; ``None`` only outside one (hooks such as
    ``before_model``, instructions functions)."""
    usage: Usage = field(default_factory=Usage)
    messages: Sequence[Message] = ()
    state: dict[str, Any] = field(default_factory=dict)
    """Structured memory of the session, kept outside the transcript.

    The same dict object is passed to every tool, hook and instructions
    function of a run. It is restored from the journal when a run starts, so
    it persists across runs and resumes of the session. After each tool call
    and after the ``before_model``, ``after_model``, ``before_tool`` and
    ``after_tool`` hooks, a change is journaled as a ``state_updated`` entry
    and announced with a `StateUpdated` event.

    Values must survive a JSON round trip unchanged: strings, numbers, booleans,
    ``None``, lists and dicts with string keys. Tuples, sets, dates and models
    are refused with an error that names the key; store
    ``model.model_dump(mode="json")`` or an ISO date string instead.

    Tools that run in parallel (read-only tools, by default) must not change
    it: their writes would race. Give a tool that writes state
    ``read_only=False``, which runs it on its own."""
    parent_run_id: str | None = None
    """Run id of the calling run, for a child run started with ``parent=ctx``
    (an agent used as a tool is one); ``None`` for a top-level run."""
    parent_tool_call_id: str | None = None
    """The parent's tool call that started this run, for a child run."""
    depth: int = 0
    """How many runs this one is nested in: 0 for a top-level run, 1 for a
    sub-agent it calls, and so on, up to ``Limits.max_depth``."""
    _emit: Callable[[Any], None] | None = field(default=None, repr=False)
    _add_usage: Callable[[Usage], None] | None = field(default=None, repr=False)
    _run: Any = field(default=None, repr=False, compare=False)
    """The run this context belongs to; a child run reads its remaining limits."""

    def emit(self, event: Any) -> None:
        """Put a custom event on the run's event stream (from inside a tool).

        Safe to call from the worker thread of a sync tool: the event is
        delivered at once, before the tool's ``tool_call_finished``.
        """
        if self._emit is not None:
            self._emit(event)

    def add_usage(self, usage: Usage) -> None:
        """Count model usage a tool caused (an embedding call, a model request) in the run total.

        The usage of a child run (``parent=ctx``: an agent used as a tool, an
        external agent's turn) is added for you, when the child ends or as it
        is reported; passing its ``result.usage`` here again is recognised and
        not counted twice (see `counted`). Safe to call from the worker thread
        of a sync tool.
        """
        if self._add_usage is not None:
            self._add_usage(usage)

    def counted(self, usage: Usage) -> None:
        """Say that ``usage`` (a child's result total) was already added to this run piece by
        piece, so a later ``add_usage`` of this same object adds nothing.

        Integrations that report a child's usage as it arrives (external
        agents) call it when the child ends, as the agent loop does for an
        agent used as a tool. The object is matched by identity, once. Does
        nothing outside a run.
        """
        run = self._run
        lock = getattr(run, "_usage_lock", None)
        absorbed = getattr(run, "_absorbed", None)
        if lock is None or not isinstance(absorbed, list):
            return
        with lock:
            absorbed.append(usage)

    async def checkpoint(self, key: str, value: Any) -> None:
        """Journal a fact about this tool call now, before the call ends.

        Written at once as a ``tool_checkpoint`` entry ``{tool_call_id,
        operation_id, key, value}``, so the fact is recorded before the tool
        acts on it (record first, act second). When the same call is repeated
        after a crash and a resume (only read-only and idempotent tools are
        repeated), `checkpoints` holds the values of the earlier attempts. Use
        it for ids of work started elsewhere: a job, an upload, a session. A
        call that has a recorded result is never repeated, whatever it
        checkpointed. A later value for the same key replaces the earlier one.

        Args:
            key: Names the fact.
            value: A JSON value (dicts with string keys, lists, strings,
                numbers, booleans, ``None``); tuples come back as lists.

        Raises:
            RuntimeError: Called outside a running tool call of a run (also
                once the call has its result, the run's journal is closed, or
                the run stopped: a context kept after its run ended writes nothing).
            SessionError: ``value`` is not JSON. Nothing was journaled.
        """
        run = self._run
        write = getattr(run, "_checkpoint", None)
        if write is None or self.tool_call_id is None or self.operation_id is None:
            raise RuntimeError("RunContext.checkpoint() works only inside a tool call of a run.")
        await write(self.tool_call_id, self.operation_id, key, value)

    @property
    def checkpoints(self) -> Mapping[str, Any]:
        """What earlier attempts of this tool call journaled with `checkpoint`, and what this
        attempt journaled so far, by key: empty on a first attempt, and outside a tool call.

        A deep copy: changing it, nested values included, changes nothing in the journal
        or in what the next read returns."""
        if self.operation_id is None:
            return {}
        stored = getattr(self._run, "checkpoints", None)
        if not isinstance(stored, Mapping):
            return {}
        return copy.deepcopy(dict(stored.get(self.operation_id, {})))

    @property
    def journaled_usage(self) -> Usage:
        """What this tool call added with `add_usage` that the journal already holds: in
        earlier attempts, when the call is repeated after a crash and a resume, and in
        this attempt so far. ``Usage()`` on a first attempt and outside a tool call.

        Usage that rode on a journal entry before a crash is restored with the run;
        usage that no entry carried yet is lost. A call that reports again, on a
        repeat, usage an earlier attempt already reported (work it re-reads from
        another system's log instead of doing it again) adds only what goes beyond
        this, or the run counts it twice.
        """
        stored = getattr(self._run, "journaled_usage", None)
        if self.operation_id is None or not isinstance(stored, Mapping):
            return Usage()
        held = stored.get(self.operation_id)
        return held.model_copy() if isinstance(held, Usage) else Usage()

    async def save_state(self) -> None:
        """Journal ``state`` now, instead of when the run next saves it.

        For a tool that must record a change before it acts on it (record
        first, act second): a one-shot grant it is about to use, for example.
        Emits ``state_updated`` when the values changed. Does nothing outside a
        run, and once the run's journal is closed or the run stopped (however it
        ended: a context kept after its run ended writes nothing).

        Raises:
            SessionError: A value in ``state`` is not JSON. Nothing was journaled.
        """
        save = getattr(self._run, "_save_state_now", None)
        if save is None:
            return
        updated = await save()
        if updated is not None:
            self.emit(updated)

    async def call_tool(
        self,
        tool: Tool,
        arguments: Mapping[str, Any],
        *,
        source: str,
        approval_timeout_s: float | None = None,
        tags: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ToolResultPart:
        """Run a tool call that did not come from this run's model, as an ordinary tool
        call of the run, while the tool call this context belongs to is running.

        For tools that serve another program: `qagents.integrations.mcp_server`
        serves qagents tools to an external agent this way. The call gets a new
        id (``"<source>-<hex>"``) and is decided, run and recorded as the loop
        does it:

        * its arguments are validated before anyone is asked about it;
        * it counts toward ``Limits.max_tool_calls``: its slot is taken before
          it is decided (calls that arrive together never pass the limit), and
          given back when it is refused;
        * the run's `Permissions` decide it, with the run's ``on_ask`` (bounded
          by ``approval_timeout_s``); "ask" without an ``on_ask`` is refused,
          because nothing can pause the run for it;
        * ``approval_decided``, ``tool_started`` and ``tool_result`` are
          journaled with ``source``, ``caller_tool_call_id`` and ``tags``, the
          start before the tool runs and the result after; the result is not
          part of any assistant message;
        * the ``before_tool`` and ``after_tool`` hooks run, and a hook (or the
          journal) that raises fails the run, as for the model's calls;
        * ``tool_call_started`` and ``tool_call_finished`` reach the run's
          stream, and ``ctx.state`` changes are journaled;
        * ``Limits.parallel_tools`` keeps the calls served through this tool
          call apart from each other (calls served through different calling
          calls, and the model's own calls, are not kept apart from them), and
          a call that runs alone waits for the thread of an earlier sync call
          (of the run or another served one) that timed out or was cancelled
          and still runs.

        The call runs at ``self.depth + 1``: it comes from the agent this tool
        call runs, so an agent or external agent among the served tools starts
        at ``self.depth + 2`` and ``Limits.max_depth`` counts the hop. Once the
        run is cancelled (``cancel()``, or from outside: a deadline around
        ``agent.run``, a task cancel, the cancel of a parent run) or stopped (its
        stream closed early, a failure, its end), its journal closed, or once this
        tool call has ended or is being cancelled (its timeout), a call is refused
        and nothing is journaled for it; a call already started then records
        `TOOL_SERVER_RUN_ENDED` and does not run.
        A call cancelled while it runs records ``TOOL_CANCELLED`` (read-only
        tools) or ``TOOL_INTERRUPTED`` before the cancellation goes on.

        Args:
            tool: The tool to run (not ``deferred``: nothing can wait for its result).
            arguments: Its arguments, as a model would give them.
            source: Where the call came from, a short name (letters, digits,
                ``_`` or ``-``) such as ``"mcp"``; it starts the call id.
            approval_timeout_s: Seconds ``on_ask`` may take before the call is
                refused; ``None``: no limit.
            tags: JSON values added to each journal entry of the call (the
                server it came through, for example).
            metadata: JSON values added to the result's ``metadata``.

        Returns:
            The call's result. A failure of the tool, a refusal or a timeout is
            a result with ``is_error`` set: it never raises for those.

        Raises:
            RuntimeError: Called outside a running tool call of a run.
            ValueError: ``source`` is not a short name, a ``tags`` key is one
                the entries already have (or ``usage``, which a resume would
                count as model usage), or ``tags`` or ``metadata`` is not JSON.
        """
        run = self._run
        outside = getattr(run, "_outside_call", None)
        if outside is None or self.tool_call_id is None:
            raise RuntimeError("RunContext.call_tool() works only inside a tool call of a run.")
        return await outside(
            self,
            tool,
            dict(arguments),
            source=source,
            approval_timeout_s=approval_timeout_s,
            tags=dict(tags or {}),
            metadata=dict(metadata or {}),
        )

    @property
    def limits(self) -> Limits | None:
        """The limits this run stops at; ``None`` for a context that belongs to no run.

        A child run's are already capped by what remained of its parents'
        budgets when it started, and its ``max_depth`` is the lowest along the
        chain of parent runs.
        """
        return getattr(self._run, "limits", None)

    @property
    def permissions(self) -> Permissions | None:
        """The permission rules of this run, with the parent run's ``on_ask`` when its own
        have none; ``None`` for a context that belongs to no run.

        Tools that relay another system's permission requests (an external
        agent) decide them with these rules.
        """
        return getattr(self._run, "permissions", None)

    def remaining_budget(self) -> dict[str, Any]:
        """What is left right now of the run's ``max_total_tokens``, ``max_cost_usd``
        and ``timeout_s``, keyed by those names (``None``: no limit).

        It counts the live usage of every run that shares the budget: the
        parent runs, and child runs that are still working (siblings started in
        parallel included). A cost total that includes requests without a known
        cost counts the whole cost budget as spent.
        """
        remaining = getattr(self._run, "_remaining", None)
        if remaining is None:
            return {"max_total_tokens": None, "max_cost_usd": None, "timeout_s": None}
        return dict(remaining())


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
        deferred: bool = False,
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
        """Seconds before the call is given up; ``None`` uses the run's limit.
        An async function is cancelled at its current ``await``; a sync function
        cannot be stopped: its thread is abandoned and keeps running (and may
        still complete its work), and the model is told the outcome is unknown.
        The interpreter still waits for that thread at exit (`TOOL_THREADS`)."""
        self.strict = strict
        self.deferred = deferred
        """The loop never calls the function: each call pauses the run as a
        `PendingInput` until ``agent.resume(..., results=...)`` supplies its
        result. The signature still defines the arguments the model sees."""
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
        """Validate ``arguments`` and call the function. Raises what the function raises.

        A sync function runs on a worker thread of qagents (`TOOL_THREADS`).
        The timeout (the tool's own, else ``timeout``) covers the whole call,
        including an awaitable that a sync function returns. When it expires,
        `ToolTimeout` is raised; a ``TimeoutError`` raised by the function
        itself passes through unchanged.

        A thread cannot be stopped: when the call times out or is cancelled
        while a sync function runs, its thread is abandoned and goes on. The
        agent loop (and the MCP tool server) then keeps later calls that run
        one at a time (``Limits.parallel_tools``) apart from it.
        """
        kwargs = self.validate(arguments)
        args = (ctx,) if self.takes_context else ()
        limit = self.timeout if self.timeout is not None else timeout
        thread: concurrent.futures.Future[Any] | None = None
        try:
            async with asyncio.timeout(limit or None) as scope:  # None: no deadline
                if _is_async_callable(self.function):
                    result = await self.function(*args, **kwargs)
                else:
                    # A thread cannot be stopped: on timeout the call is abandoned, not killed.
                    thread = _run_in_thread(self.name, self.function, args, kwargs)
                    result = await asyncio.wrap_future(thread)
                if inspect.isawaitable(result):  # a sync wrapper around a coroutine function
                    result = await result
        except TimeoutError:
            if not scope.expired():
                raise  # the function's own timeout, not this one
            started = thread is None or not thread.cancelled()
            running = thread is not None and started and not thread.done()
            if running:
                assert thread is not None
                _abandon(self.name, thread, "timed out")
            raise ToolTimeout(
                limit or 0, started=started, running=running, thread=thread if running else None
            ) from None
        except asyncio.CancelledError:
            # The run was cancelled or timed out: a thread that already runs goes on alone.
            if thread is not None and not thread.done():  # a queued call was cancelled: done
                _abandon(self.name, thread, "was cancelled")
            raise
        return result

    def __repr__(self) -> str:
        return f"Tool({self.name!r})"


class ToolTimeout(TimeoutError):
    """A tool call took longer than its timeout (raised by `Tool.run`).

    Args:
        seconds: The timeout that expired.
        started: The function had started (``False``: a sync tool that waited
            for a free worker thread the whole time, and never ran).
        running: A sync function that could not be stopped and still runs in
            its abandoned thread.
        thread: The future of that abandoned thread, while ``running``; done
            when the function returns at last.
    """

    def __init__(
        self,
        seconds: float,
        *,
        started: bool = True,
        running: bool = False,
        thread: concurrent.futures.Future[Any] | None = None,
    ) -> None:
        super().__init__(f"the tool call did not finish within {seconds:g} seconds")
        self.seconds = seconds
        self.started = started
        self.running = running
        self.thread = thread


_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None
_EXECUTOR_SIZE = 0
_ABANDONED: set[concurrent.futures.Future[Any]] = set()
"""Calls of sync tools that were given up (timed out, or their run was cancelled or
timed out) and still hold a worker thread."""
_LOCK = threading.Lock()


def _run_in_thread(
    name: str, function: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> concurrent.futures.Future[Any]:
    """Start ``function`` on a worker thread, with the caller's context variables."""
    global _EXECUTOR, _EXECUTOR_SIZE
    with _LOCK:
        if _EXECUTOR is None:
            _EXECUTOR_SIZE = max(1, TOOL_THREADS)
            _EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                max_workers=_EXECUTOR_SIZE, thread_name_prefix="qagents-tool"
            )
        stuck = len(_ABANDONED)
        if stuck >= _EXECUTOR_SIZE:
            # A new call would only queue behind threads that may never return: refuse it.
            raise ToolError(prompts.TOOL_THREADS_EXHAUSTED.format(name=name, count=stuck))
        context = contextvars.copy_context()
        return _EXECUTOR.submit(context.run, functools.partial(function, *args, **kwargs))


_ON_ABANDONED: contextvars.ContextVar[Callable[[concurrent.futures.Future[Any]], None] | None] = (
    contextvars.ContextVar("qagents_on_abandoned", default=None)
)
"""Who keeps later calls apart from the thread of a sync call that was given up (timed out
or cancelled) while it still runs: set around `Tool.run` by the agent loop and the MCP tool
server (`_abandoned_threads_to`), called before `Tool.run` raises. A context variable, so
that `Tool` subclasses that override ``run`` keep its signature."""


@contextlib.contextmanager
def _abandoned_threads_to(
    hook: Callable[[concurrent.futures.Future[Any]], None],
) -> Iterator[None]:
    """Hand the threads that `Tool.run` abandons inside the block to ``hook``."""
    token = _ON_ABANDONED.set(hook)
    try:
        yield
    finally:
        _ON_ABANDONED.reset(token)


def _abandon(name: str, thread: concurrent.futures.Future[Any], why: str) -> None:
    with _LOCK:
        _ABANDONED.add(thread)
        stuck = len(_ABANDONED)
    thread.add_done_callback(_release)
    logger.warning(
        "Sync tool %s %s; its thread keeps running (%d of %d worker threads abandoned)",
        name,
        why,
        stuck,
        _EXECUTOR_SIZE,
    )
    hook = _ON_ABANDONED.get()
    if hook is not None:
        try:
            hook(thread)
        except Exception:  # the call's own outcome must still be raised
            logger.exception("Keeping the abandoned thread of sync tool %s failed", name)


def _release(thread: concurrent.futures.Future[Any]) -> None:
    with _LOCK:
        _ABANDONED.discard(thread)


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
    deferred: bool = False,
) -> Any:
    """Turn a function into a `Tool`. Use bare or with options.

    A first parameter annotated ``RunContext`` (or named ``ctx``) receives the
    run context and is hidden from the model.

    The model reads the return value as text (`render_tool_output`). Return a
    `ToolResultPart` to set ``is_error`` and ``metadata`` yourself (the loop
    fills in the call id and name); raise `ToolError` for an error result, and
    give it a ``metadata`` dict to have that copied into the result's metadata.

    Args:
        function: The function to wrap. Leave it out to use ``@tool(...)`` with options.
        name: Tool name the model sees; defaults to the function name.
        description: Description the model sees; defaults to the docstring's summary.
        requires_approval: Ask a human before running (unless a permission rule allows it).
        read_only: The tool changes nothing, so it may run in parallel with
            other read-only calls. It must not change ``ctx.state`` either.
        idempotent: Running it twice has the same effect as once, so it is
            safe to repeat after a crash.
        timeout: Seconds before a call is given up; ``None`` uses the run's
            limit. An async function is cancelled; a sync function cannot be
            stopped, so its thread is abandoned and may still finish its work
            (the model is told the outcome is unknown), and the interpreter
            waits for it at exit (`TOOL_THREADS`). Give sync clients their own
            socket timeouts.
        strict: Ask the provider to enforce the schema exactly, where supported.
        deferred: Never run the function. The arguments are validated against
            the signature, then after the permission check each call pauses
            the run as a `PendingInput`; its result comes from
            ``agent.resume(session, results={tool_call_id: value})``. Use it for
            work done outside the process: a human's answer, an external job.
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
            deferred=deferred,
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
