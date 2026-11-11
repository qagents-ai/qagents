"""The agent and its loop.

The loop is a small state machine over the conversation history. Before every
step it looks at the last message:

* an assistant message with tool calls that have no results -> run the tools;
* an assistant message without tool calls -> that is the final answer;
* anything else (a user message, tool results) -> ask the model.

Because the next step depends only on the history, and the history is written
to the session journal as it grows, a run can stop anywhere (a pause for
approval or for a deferred tool's result, a crash, a deploy) and continue
later from exactly that point.

Around that loop sit the checks of the final answer: the output type, the
output validators, and the input and output guardrails. A run can also be the
child of another run (``parent=ctx``), which links the two in traces and
shares the parent's remaining budget.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy as copying
import inspect
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, create_model
from pydantic_core import PydanticSerializationError

from . import prompts
from .context import Compactor
from .errors import (
    ApprovalTokenError,
    ConfigurationError,
    DeferToolCall,
    GuardrailTripped,
    ModelError,
    ModelResponseError,
    ModelRetry,
    OutputValidationError,
    SessionError,
    ToolArgumentError,
    ToolError,
)
from .events import (
    AgentSwitched,
    ApprovalRequired,
    Event,
    GuardrailTriggered,
    InputRequired,
    ModelResponded,
    RunFinished,
    RunStarted,
    StateUpdated,
    SubagentEvent,
    ToolCallFinished,
    ToolCallStarted,
    TurnStarted,
)
from .hooks import HookChain, Hooks
from .messages import Message, Part, TextPart, ToolCallPart, ToolResultPart, Usage
from .models.base import (
    Model,
    ModelRequest,
    ModelResponse,
    ModelSettings,
    OutputSpec,
    ResponseDone,
    TextDelta,
    ToolSpec,
    resolve_model,
)
from .permissions import Approval, InputResult, Permissions
from .results import GuardrailStage, RunResult, RunStatus, TrippedGuardrail
from .session import PendingApproval, PendingInput, Session, SessionState
from .tools import (
    RunContext,
    Tool,
    Toolkit,
    _clean_schema,
    _describe_validation_error,
    render_tool_output,
)
from .tools import tool as make_tool

logger = logging.getLogger("qagents")

T = TypeVar("T")
Instructions = str | Callable[[RunContext[Any]], Any] | None
"""Fixed text, or a function of the run context (sync or async) that returns it."""
Prompt = str | Message | list[Part] | None
OutputMode = Literal["native", "prompted", "tool"]
OutputValidator = Callable[[RunContext[Any], Any], Any]
"""``(ctx, output)`` -> a replacement output, or ``None`` to keep it. May be async.
Raise `ModelRetry` to send the model a correction request."""
Guardrail = Callable[[RunContext[Any], Any], Any]
"""``(ctx, value)`` -> ``None`` to let the value through. May be async. Raise
`GuardrailTripped` to block it; any other exception fails the run."""

_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class Limits(BaseModel):
    """Hard stops for a run. ``None`` switches a limit off.

    A child run (``parent=ctx``, as an agent used as a tool starts) also stops
    at what remains of its parent's ``max_total_tokens``, ``max_cost_usd`` and
    ``timeout_s`` when it starts, whichever is lower.
    """

    max_turns: int | None = 30
    """Model requests per run."""
    max_total_tokens: int | None = None
    """Input plus output tokens over the whole run."""
    max_cost_usd: float | None = None
    """Needs a model that reports cost (QCloud does, from its catalog prices).
    A budget of zero or less stops the run before its first request."""
    timeout_s: float | None = None
    """Wall-clock seconds for the whole run."""
    max_tool_calls: int | None = None
    """Tool calls per run. A call beyond it is not started: it gets an
    explanatory result, and the run ends with ``stop_reason="max_tool_calls"``."""
    max_depth: int | None = 5
    """How deeply runs may nest: a child run at a greater depth refuses to start.
    The lower of the child's and the parent's value applies."""
    tool_timeout_s: float | None = 120.0
    """Seconds per tool call, unless the tool sets its own ``timeout``."""
    parallel_tools: Literal["read_only", "all", "none"] = "read_only"
    """Which tool calls of one turn run concurrently: only tools declared
    ``read_only`` (the rest run one at a time, in order), all, or none."""
    max_parallel_tools: int = 8
    max_tool_result_chars: int | None = 60_000
    """Longer tool results are shortened in the middle before the model sees them."""
    output_retries: int = 2
    """Extra attempts for a final answer that does not match ``output_type``, that
    an output validator rejects, or (``output_mode="tool"``) that is written as
    plain text instead of a call of the final-answer tool."""
    model_retries: int = 2
    """Extra attempts for model errors that are safe to repeat."""
    retry_ambiguous: bool = False
    """Also repeat requests the provider may already have processed (and billed)."""


class Agent:
    """A model, instructions and tools, run in a loop until the model answers.


        agent = Agent("qcloud:<model-id>", instructions="Be brief.", tools=[get_weather])
        result = await agent.run("What is the weather in Athens?")
        print(result.output)

    The agent you call ``run`` on owns the run's ``limits``, ``permissions``,
    ``hooks`` and input guardrails. After a handoff, the target agent's model,
    instructions, tools, output settings and output validators take over; the
    output guardrails of both agents apply.
    """

    def __init__(
        self,
        model: Model | str,
        *,
        name: str = "agent",
        description: str = "",
        instructions: Instructions = None,
        tools: Iterable[Any] = (),
        output_type: Any = None,
        output_mode: OutputMode = "native",
        output_tool_name: str = "final_answer",
        output_strict: bool = False,
        output_validators: Iterable[OutputValidator] = (),
        input_guardrails: Iterable[Guardrail] = (),
        output_guardrails: Iterable[Guardrail] = (),
        settings: ModelSettings | None = None,
        limits: Limits | None = None,
        permissions: Permissions | None = None,
        hooks: Iterable[Hooks] = (),
        handoffs: Iterable[Agent] = (),
        compactor: Compactor | None = None,
    ) -> None:
        """Build the agent. Settings the provider would reject fail here.

        Args:
            model: A `Model`, or a ``"provider:model-id"`` string.
            name: Name used in events, the journal and handoff tool names.
            description: What the agent does, for agents that delegate to it.
            instructions: The system prompt: text, or a function of the run context.
            tools: Functions, `Tool` and `Toolkit` objects, agents, or lists of them.
            output_type: The type of the final answer; ``None`` or ``str`` for text.
            output_mode: How the model is asked for ``output_type``: ``"native"``
                sends the schema as ``response_format``; ``"prompted"`` writes it
                into the instructions; ``"tool"`` gives the model a final-answer
                tool whose arguments are the answer, for models that call tools
                more reliably than they follow a response format.
            output_tool_name: Name of the final-answer tool (``output_mode="tool"``).
            output_strict: Ask the provider to enforce the schema exactly, where supported.
            output_validators: Functions ``(ctx, output)`` run in order on the
                validated answer, in every output mode. Return a value to replace
                the answer, ``None`` to keep it, or raise `ModelRetry` to ask
                the model for a corrected one. Sync or async.
            input_guardrails: Functions ``(ctx, message)`` run on the new prompt
                (a `Message`) before it joins the conversation. Raise
                `GuardrailTripped` to block the run.
            output_guardrails: Functions ``(ctx, output)`` run on the accepted
                final answer. Raise `GuardrailTripped` to block it.
            settings: Model settings for every request.
            limits: Hard stops; see `Limits`.
            permissions: Who decides whether a tool call runs; see `Permissions`.
            hooks: Your code at fixed points of the loop; see `Hooks`.
            handoffs: Agents this one may hand the conversation to.
            compactor: Shortens long conversations before a request.

        Raises:
            ConfigurationError: An argument is invalid, for example an unknown
                ``output_mode`` or an invalid ``output_tool_name``.
        """
        if output_mode not in ("native", "prompted", "tool"):
            raise ConfigurationError(
                f"output_mode must be 'native', 'prompted' or 'tool', not {output_mode!r}."
            )
        if not _TOOL_NAME.match(output_tool_name):
            raise ConfigurationError(
                f"output_tool_name {output_tool_name!r} is not a valid tool name: "
                "use 1-64 letters, digits, '_' or '-'."
            )
        self.model = resolve_model(model)
        self.name = name
        self.description = description
        self.instructions = instructions
        self.tools: list[Tool | Toolkit] = _normalize_tools(tools)
        self.output_type = None if output_type is str else output_type
        self.output_mode: OutputMode = output_mode
        """``"native"`` sends the schema as ``response_format``; ``"prompted"``
        writes it into the instructions, for models without schema support;
        ``"tool"`` asks for a call of the final-answer tool."""
        self.output_tool_name = output_tool_name
        """Name of the final-answer tool when ``output_mode="tool"``."""
        self.output_strict = output_strict
        self.output_validators = list(output_validators)
        self.input_guardrails = list(input_guardrails)
        self.output_guardrails = list(output_guardrails)
        self.settings = settings or ModelSettings()
        # A setting the provider would reject fails here, not in the middle of a run.
        self.model.validate_settings(self.settings)
        self.limits = limits or Limits()
        self.permissions = permissions or Permissions()
        self.hooks = list(hooks)
        self.handoffs = list(handoffs)
        self.compactor = compactor
        self._output_adapter: TypeAdapter[Any] | None = None
        self._output_schema: dict[str, Any] | None = None
        if self.output_type is not None:
            self._output_adapter = TypeAdapter(self.output_type)
            self._output_schema = self._output_adapter.json_schema()
        self._answer_adapter: TypeAdapter[Any] | None = None
        """Validates the arguments of a final-answer call (``output_mode="tool"``)."""
        self._answer_wrapped = False
        """The answer is the ``value`` argument, because its schema is not an object."""
        self._answer_parameters: dict[str, Any] = {}
        if output_mode == "tool":
            self._setup_answer_tool()

    # -- running ---------------------------------------------------------------

    def stream(
        self,
        prompt: Prompt = None,
        *,
        deps: Any = None,
        session: Session | None = None,
        history: Iterable[Message] | None = None,
        limits: Limits | None = None,
        settings: ModelSettings | None = None,
        state: Mapping[str, Any] | None = None,
        parent: RunContext[Any] | None = None,
    ) -> RunStream:
        """Start a run and return its event stream.

        Args:
            prompt: The new user message. May be left out when ``session`` or
                ``history`` already ends with one.
            deps: Your dependencies, passed to tools and hooks as ``ctx.deps``.
            session: Continues an existing conversation. A new in-memory
                session is created when it is left out.
            history: Seeds a new conversation with messages you kept yourself.
            limits: Replaces the agent's limits for this run.
            settings: Merged over the agent's model settings for this run.
            state: Values merged into the session state (``ctx.state``) when
                the run starts, key by key. The merge is journaled.
            parent: The context of the run (usually a tool call) that starts
                this one. The new run is its child: ``ctx.parent_run_id``,
                ``ctx.parent_tool_call_id`` and ``ctx.depth`` link the two, the
                parent's remaining token, cost and time budget caps this run,
                the parent's hooks with ``propagate=True`` apply here too, and
                the parent's ``on_ask`` answers approvals when this agent's
                permissions have none. Usage is not added to the parent for
                you; call ``parent.add_usage(result.usage)`` (``as_tool`` does).

        Raises:
            ConfigurationError: The child run would nest deeper than ``Limits.max_depth``.
        """
        return RunStream(
            _Run(
                self, prompt=prompt, deps=deps, session=session, history=history,
                limits=limits, settings=settings, approvals=None, results=None,
                state=state, parent=parent, resume=False,
            )
        )  # fmt: skip

    async def run(
        self,
        prompt: Prompt = None,
        *,
        deps: Any = None,
        session: Session | None = None,
        history: Iterable[Message] | None = None,
        limits: Limits | None = None,
        settings: ModelSettings | None = None,
        state: Mapping[str, Any] | None = None,
        parent: RunContext[Any] | None = None,
    ) -> RunResult:
        """Run until the agent answers, pauses, hits a limit or is blocked.

        Takes the same arguments as `stream`. A run started on a session that
        is paused closes the waiting tool calls with an explanatory result, so
        the conversation stays valid; use `resume` to continue it instead.
        """
        stream = self.stream(
            prompt, deps=deps, session=session, history=history, limits=limits,
            settings=settings, state=state, parent=parent,
        )  # fmt: skip
        return await stream.wait()

    def run_sync(self, prompt: Prompt = None, **options: Any) -> RunResult:
        """``run`` for code that is not async. Do not call it from a running event loop."""
        return asyncio.run(self.run(prompt, **options))

    def resume_stream(
        self,
        session: Session,
        *,
        approvals: Mapping[str, Approval | bool] | None = None,
        results: Mapping[str, Any] | None = None,
        deps: Any = None,
        limits: Limits | None = None,
        settings: ModelSettings | None = None,
        state: Mapping[str, Any] | None = None,
    ) -> RunStream:
        """Continue the session's unfinished run and return its event stream.

        Takes the same arguments as `resume`.
        """
        return RunStream(
            _Run(
                self, prompt=None, deps=deps, session=session, history=None,
                limits=limits, settings=settings, approvals=approvals, results=results,
                state=state, parent=None, resume=True,
            )
        )  # fmt: skip

    async def resume(
        self,
        session: Session,
        *,
        approvals: Mapping[str, Approval | bool] | None = None,
        results: Mapping[str, Any] | None = None,
        deps: Any = None,
        limits: Limits | None = None,
        settings: ModelSettings | None = None,
        state: Mapping[str, Any] | None = None,
    ) -> RunResult:
        """Continue the session's unfinished run: after a pause, a crash or an error.

        Tool calls that already have a recorded result are never run again, and
        a call that waits for input is never run at all. Answering only some of
        the waiting calls keeps the run paused with the rest.

            paused = await agent.run("Plan my trip", session=session)
            question = paused.pending_inputs[0]
            done = await agent.resume(session, results={question.tool_call_id: "Lisbon"})

        When the permissions have an ``approval_secret``, every answer must
        carry the token of its pause: ``Approval.for_call(pending)`` for an
        approval, ``InputResult.for_call(pending, value)`` or
        ``{"token": pending.token, "value": value}`` for a result.

        Args:
            session: The session whose run is unfinished.
            approvals: Maps tool call ids from ``result.pending_approvals`` to
                ``True``/``False`` or an `Approval`.
            results: Maps tool call ids from ``result.pending_inputs`` to their
                results. A value is rendered like a tool's return value; an
                exception instance becomes an error result with its message;
                a `ToolResultPart` is used as it is; an `InputResult` carries
                one of these with its token. An id that is not waiting for a
                result raises `SessionError` before anything runs.
            deps: Your dependencies, passed to tools and hooks as ``ctx.deps``.
            limits: Replaces the agent's limits for the resumed run.
            settings: Merged over the agent's model settings.
            state: Values merged into the session state when the run resumes.

        Raises:
            SessionError: The session has no unfinished run, or ``results``
                names a call that waits for no result.
            ApprovalTokenError: Pauses are signed and an answer has no token or
                the wrong one, or the paused call in the journal was changed.
        """
        stream = self.resume_stream(
            session, approvals=approvals, results=results, deps=deps, limits=limits,
            settings=settings, state=state,
        )  # fmt: skip
        return await stream.wait()

    # -- composition -------------------------------------------------------------

    def as_tool(
        self, *, name: str | None = None, description: str | None = None, read_only: bool = False
    ) -> Tool:
        """Expose this agent as a tool another agent can call.

        The sub-agent works in its own fresh conversation: it sees only the
        task text, and the caller sees only its final answer. Each call is a
        child run of the caller's (``parent=ctx``): it is linked to the calling
        tool call, stops at the caller's remaining budget, inherits the
        caller's propagating hooks and ``on_ask``, and refuses to start beyond
        ``Limits.max_depth``. Its usage is added to the caller's run and its
        events are forwarded as `SubagentEvent`. A sub-agent run that pauses,
        is blocked or stops at a limit is reported to the caller as an error
        result.

        Args:
            name: Tool name; defaults to the agent's name.
            description: Tool description; defaults to the agent's description,
                or ``prompts.SUBAGENT_DESCRIPTION``.
            read_only: Declare the sub-agent free of side effects, so calls may
                run in parallel.
        """
        agent = self

        async def delegate(ctx: RunContext[Any], task: str) -> str:
            stream = agent.stream(task, deps=ctx.deps, parent=ctx)
            async with stream:
                async for event in stream:
                    if not isinstance(event, RunStarted | RunFinished):
                        ctx.emit(
                            SubagentEvent(
                                agent=agent.name, tool_call_id=ctx.tool_call_id, event=event
                            )
                        )
            result = stream.result
            assert result is not None
            ctx.add_usage(result.usage)
            if result.status != "completed":
                raise ToolError(
                    prompts.SUBAGENT_STOPPED.format(
                        name=agent.name, reason=result.stop_reason or result.status
                    )
                )
            return (
                result.text if isinstance(result.output, str) else render_tool_output(result.output)
            )

        built = make_tool(
            delegate,
            name=name or _tool_safe(self.name),
            description=description
            or self.description
            or prompts.SUBAGENT_DESCRIPTION.format(name=self.name),
            read_only=read_only,
        )
        built.parameters["properties"]["task"]["description"] = prompts.SUBAGENT_TASK_DESCRIPTION
        return built

    def copy(self, **changes: Any) -> Agent:
        """A new agent with the same configuration, except for ``changes``.

            cheap = agent.copy(model="qcloud:<smaller-model-id>", name="drafter")

        Every keyword argument of the constructor can be changed. Lists (tools,
        hooks, handoffs, validators, guardrails) are copied; the objects in
        them, and the permissions, limits and settings, are shared.

        Raises:
            ConfigurationError: A name in ``changes`` is not a constructor argument.
        """
        current: dict[str, Any] = {
            "model": self.model,
            "name": self.name,
            "description": self.description,
            "instructions": self.instructions,
            "tools": list(self.tools),
            "output_type": self.output_type,
            "output_mode": self.output_mode,
            "output_tool_name": self.output_tool_name,
            "output_strict": self.output_strict,
            "output_validators": list(self.output_validators),
            "input_guardrails": list(self.input_guardrails),
            "output_guardrails": list(self.output_guardrails),
            "settings": self.settings,
            "limits": self.limits,
            "permissions": self.permissions,
            "hooks": list(self.hooks),
            "handoffs": list(self.handoffs),
            "compactor": self.compactor,
        }
        unknown = sorted(set(changes) - set(current))
        if unknown:
            raise ConfigurationError(
                f"Agent.copy() got {unknown}, which are not Agent arguments. "
                f"Arguments that can change: {', '.join(current)}."
            )
        return type(self)(**{**current, **changes})

    def _setup_answer_tool(self) -> None:
        """Prepare the final-answer tool: its parameters and how to validate a call."""
        answer_type = self.output_type if self.output_type is not None else str
        adapter: TypeAdapter[Any] = self._output_adapter or TypeAdapter(answer_type)
        schema = adapter.json_schema()
        if schema.get("type") == "object" and "$ref" not in schema:
            self._answer_adapter = adapter
        else:  # tools take an object, so the answer becomes its one argument
            wrapper = create_model(
                self.output_tool_name,
                __config__=ConfigDict(extra="forbid"),
                value=(answer_type, ...),
            )
            self._answer_adapter = TypeAdapter(wrapper)
            self._answer_wrapped = True
            schema = self._answer_adapter.json_schema()
        self._answer_parameters = _clean_schema(schema)

    def _answer_tool_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.output_tool_name,
            description=prompts.OUTPUT_TOOL_DESCRIPTION,
            parameters=self._answer_parameters,
            strict=self.output_strict,
        )

    def _is_answer_call(self, call: ToolCallPart) -> bool:
        return self.output_mode == "tool" and call.name == self.output_tool_name

    def _validate_answer_call(self, call: ToolCallPart) -> Any:
        """The answer a final-answer call carries. Raises ``ValueError`` when it is invalid."""
        assert self._answer_adapter is not None
        if call.parse_error is not None:
            raise ValueError(call.parse_error)
        validated = self._answer_adapter.validate_python(call.arguments)
        return validated.value if self._answer_wrapped else validated

    @property
    def _handoff_tool_name(self) -> str:
        return f"transfer_to_{_tool_safe(self.name)}"

    def _handoff_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self._handoff_tool_name,
            description=prompts.HANDOFF_DESCRIPTION.format(
                name=self.name, description=self.description
            ).strip(),
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        )

    async def _collect_tools(self) -> dict[str, Tool]:
        collected: dict[str, Tool] = {}
        for item in self.tools:
            for candidate in [item] if isinstance(item, Tool) else await item.tools():
                if collected.get(candidate.name, candidate) is not candidate:
                    raise ConfigurationError(
                        f"Agent {self.name!r} has two tools named {candidate.name!r}."
                    )
                collected[candidate.name] = candidate
        return collected

    async def _render_instructions(self, ctx: RunContext[Any]) -> str:
        instructions = self.instructions
        if instructions is None:
            return ""
        if isinstance(instructions, str):
            return instructions
        rendered = instructions(ctx)
        if inspect.isawaitable(rendered):
            rendered = await rendered
        return rendered or ""

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, model={self.model.provider}:{self.model.name})"


class RunStream:
    """The events of one run, plus control over it.


        stream = agent.stream("Tell me about Athens")
        async for event in stream:
            if event.type == "text_delta":
                print(event.text, end="")
        print(stream.result.usage)

    Iterate it once. ``cancel()`` stops the run at the next opportunity; the
    text produced so far is kept in the history and marked as interrupted.
    """

    def __init__(self, run: _Run) -> None:
        self._run = run
        self._events = run.execute()
        self.result: RunResult | None = None
        """Set once the run has finished."""

    @property
    def run_id(self) -> str:
        return self._run.run_id

    @property
    def session(self) -> Session:
        return self._run.session

    def __aiter__(self) -> RunStream:
        return self

    async def __anext__(self) -> Event:
        event = await self._events.__anext__()
        if isinstance(event, RunFinished):
            self.result = event.result
        return event

    def cancel(self, reason: str = "cancelled") -> None:
        """Ask the run to stop. Keep iterating to receive the final ``run_finished`` event."""
        self._run.cancel(reason)

    async def wait(self) -> RunResult:
        """Consume the remaining events and return the result."""
        async with self:
            async for _ in self:
                pass
        assert self.result is not None
        return self.result

    async def text(self) -> AsyncIterator[str]:
        """Only the text of the answer, as it is written."""
        async with self:
            async for event in self:
                if isinstance(event, TextDelta):
                    yield event.text

    async def aclose(self) -> None:
        await self._events.aclose()

    async def __aenter__(self) -> RunStream:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()


class _Failure:
    """Carries an exception from a tool task to the loop."""

    def __init__(self, error: BaseException) -> None:
        self.error = error


class _Deferred:
    """Tells the loop that a tool task ended by deferring its call."""


class _Rejected(Exception):
    """A final answer the model must correct.

    Args:
        message: What is wrong, for the model.
        problem: How the answer failed, for `OutputValidationError`.
    """

    def __init__(self, message: str, problem: str) -> None:
        super().__init__(message)
        self.message = message
        self.problem = problem


class _Run:
    """The state and the steps of one run."""

    def __init__(
        self,
        agent: Agent,
        *,
        prompt: Prompt,
        deps: Any,
        session: Session | None,
        history: Iterable[Message] | None,
        limits: Limits | None,
        settings: ModelSettings | None,
        approvals: Mapping[str, Approval | bool] | None,
        results: Mapping[str, Any] | None,
        state: Mapping[str, Any] | None,
        parent: RunContext[Any] | None,
        resume: bool,
    ) -> None:
        self.root = agent
        self.agent = agent
        self.run_id = uuid.uuid4().hex
        self.session = session or Session()
        self.deps = deps
        self.prompt = prompt
        self.seed = list(history) if history is not None else []
        self.limits = limits or agent.limits
        if settings is not None:
            agent.model.validate_settings(agent.settings.merged(settings))
        self.settings = settings
        self.parent = parent
        """The context of the tool call that started this run, for a child run."""
        self.parent_run: _Run | None = (
            parent._run if parent is not None and isinstance(parent._run, _Run) else None
        )
        self.depth = parent.depth + 1 if parent is not None else 0
        self._check_depth()
        self.permissions = agent.permissions
        hooks: list[Hooks] = list(agent.hooks)
        if self.parent_run is not None:
            hooks.extend(hook for hook in self.parent_run.hook_list if hook.propagate)
            on_ask = self.parent_run.permissions.on_ask
            if self.permissions.on_ask is None and on_ask is not None:
                self.permissions = copying.copy(self.permissions)
                self.permissions.on_ask = on_ask
        self.hook_list: list[Hooks] = _unique(hooks)
        """This run's hooks: the agent's, then the parent run's propagating ones."""
        self.hooks = HookChain(self.hook_list)
        self.resume = resume
        self.approvals: dict[str, Approval] = {
            call_id: answer if isinstance(answer, Approval) else Approval(approved=bool(answer))
            for call_id, answer in (approvals or {}).items()
        }

        self.history: list[Message] = []
        self.new_messages: list[Message] = []
        self.usage = Usage()
        self.turn = 0
        self.output: Any = None
        self.tools: dict[str, Tool] = {}
        self.handoffs: dict[str, Agent] = {}
        self.results: dict[str, ToolResultPart] = {}
        """Results recorded for calls of the last assistant message."""
        self.started: set[str] = set()
        """Calls that began executing in an earlier process and never reported."""
        self.pending: list[PendingApproval] = []
        """Calls waiting for a decision when the run pauses."""
        self.supplied: dict[str, Any] = dict(results or {})
        """Results given to ``resume`` for calls that wait for input; used once."""
        self.waiting: dict[str, PendingInput] = {}
        """Calls of the last assistant message that wait for a result from outside."""
        self.inputs: list[PendingInput] = []
        """Calls waiting for a result when the run pauses."""
        self.state: dict[str, Any] = {}
        """``ctx.state``: one dict for the whole run, restored from the journal."""
        self._state_seed = dict(state or {})
        self._saved_state = "{}"
        """JSON of the state as last journaled; a different snapshot means it changed."""
        self._state_lock = asyncio.Lock()
        self._last_response: ModelResponse | None = None
        self._refused = False
        """The last model response ended with ``finish_reason="content_filter"``."""
        self._answered = False
        """A final-answer call was accepted (``output_mode="tool"``)."""
        self._ordinal = 0
        """Messages this run has journaled; names tool calls in operation ids."""
        self._output_attempts = 0
        self._tool_calls = 0
        """Tool calls this run has authorised or started, for ``Limits.max_tool_calls``."""
        self._tool_limit_hit = False
        self._deadline: float | None = None
        """``time.monotonic()`` at which ``timeout_s`` stops the run."""
        self.status: RunStatus = "completed"
        self.stop_reason: str | None = None
        self.guardrail: TrippedGuardrail | None = None
        self._journal_open = False
        self._cancel = asyncio.Event()
        self._cancel_reason: str | None = None

    def cancel(self, reason: str = "cancelled") -> None:
        if not self._cancel.is_set():
            self._cancel_reason = reason
            self._cancel.set()

    # -- the loop -----------------------------------------------------------------

    async def execute(self) -> AsyncIterator[Event]:
        async with contextlib.aclosing(self._execute()) as events:
            async for event in events:
                await self.hooks.on_event(self._ctx(), event)
                yield event

    async def _execute(self) -> AsyncIterator[Event]:
        timer: asyncio.TimerHandle | None = None
        try:
            self._inherit_limits()
            await self._prepare()
            yield RunStarted(
                run_id=self.run_id,
                session_id=self.session.id,
                agent=self.agent.name,
                resumed=self.resume,
                parent_run_id=self.parent.run_id if self.parent is not None else None,
                depth=self.depth,
            )
            if (updated := await self._save_state()) is not None:
                yield updated  # the values passed as state=
            await self.hooks.on_run_start(self._ctx())
            timer = self._start_timer()

            if self.guardrail is not None:  # an input guardrail blocked the prompt
                yield self._guardrail_event(self.guardrail)
                self.status, self.stop_reason = "blocked", self.guardrail.name
            else:
                async with contextlib.aclosing(self._steps()) as steps:
                    async for event in steps:
                        yield event
                if self.status == "completed" and not (self._refused and self.output is None):
                    tripped = await self._check_guardrails("output", self.output)
                    if tripped is not None:
                        self.guardrail = tripped
                        self.status, self.stop_reason = "blocked", tripped.name
                        yield self._guardrail_event(tripped)

            if (updated := await self._save_state()) is not None:
                yield updated  # changed by on_event, on_run_start or a guardrail
            if self.status == "paused":
                await self._journal(
                    "run_paused",
                    {
                        "pending": [item.model_dump(mode="json") for item in self.pending],
                        "inputs": [item.model_dump(mode="json") for item in self.inputs],
                    },
                )
            else:
                await self._journal(
                    "run_finished", {"status": self.status, "stop_reason": self.stop_reason}
                )
            result = self._result()
            await self.hooks.on_run_end(self._ctx(), result, None)
            yield RunFinished(result=result)
        except Exception as error:
            # The run stays open in the journal, so agent.resume() can pick it up.
            if self._journal_open:
                with contextlib.suppress(Exception):
                    await self._journal("run_failed", {"error": f"{type(error).__name__}: {error}"})
            try:
                await self.hooks.on_run_end(self._ctx(), None, error)
            except Exception:
                logger.exception("on_run_end raised while reporting a failed run")
            raise
        except BaseException as error:
            # Cancelled, or the stream was closed early: the run stays open in the journal
            # (nothing is written, as before). on_run_end still runs, so tracers close spans.
            try:
                await self.hooks.on_run_end(self._ctx(), None, error)
            except Exception:
                logger.exception("on_run_end raised while reporting a stopped run")
            raise
        finally:
            if timer is not None:
                timer.cancel()

    async def _steps(self) -> AsyncIterator[Event]:
        """Advance the conversation until the run answers, pauses or stops.

        Sets ``status`` and ``stop_reason``; ``status`` stays ``"completed"``
        when the run answered.
        """
        while True:
            if self._answered:  # a final-answer call was accepted, now or before a resume
                return
            if self._cancel.is_set():
                await self._settle_open_calls(prompts.TOOL_CANCELLED)
                if self._cancel_reason == "timeout":
                    self.status, self.stop_reason = "limit_reached", "timeout"
                else:
                    self.status, self.stop_reason = "cancelled", self._cancel_reason
                return
            last = self.history[-1]
            if self._refused:  # the provider refused: end the run, keep its words
                self.output = last.text if self.agent._output_adapter is None else None
                await self._settle_open_calls(prompts.TOOL_NOT_EXECUTED)
                self.stop_reason = "content_filter"
                return
            if last.role == "assistant" and last.tool_calls:
                async with contextlib.aclosing(self._tool_phase(last)) as phase:
                    async for event in phase:
                        yield event
                if self.pending or self.inputs:
                    self.status = "paused"
                    return
                if self._tool_limit_hit:
                    self.status, self.stop_reason = "limit_reached", "max_tool_calls"
                    return
                continue
            if last.role == "assistant":
                if await self._finalize(last):
                    truncated = self._last_response is not None and (
                        self._last_response.finish_reason == "length"
                    )
                    self.stop_reason = "max_output_tokens" if truncated else None
                    return
                continue
            stop_reason = self._limit_reached()
            if stop_reason is not None:
                self.status, self.stop_reason = "limit_reached", stop_reason
                return
            self.turn += 1
            yield TurnStarted(turn=self.turn, agent=self.agent.name)
            async with contextlib.aclosing(self._model_phase()) as phase:
                async for event in phase:
                    yield event

    def _check_depth(self) -> None:
        """Refuse to start a child run nested deeper than ``Limits.max_depth``."""
        if self.parent is None:
            return
        bounds = [self.limits.max_depth]
        if self.parent_run is not None:
            bounds.append(self.parent_run.limits.max_depth)
        known = [bound for bound in bounds if bound is not None]
        if known and self.depth > min(known):
            raise ConfigurationError(
                f"Agent {self.root.name!r} cannot start a child run at depth {self.depth}: "
                f"Limits.max_depth is {min(known)}. Agents that delegate to each other need a "
                "point where they stop; raise Limits(max_depth=...) if this nesting is intended."
            )

    def _inherit_limits(self) -> None:
        """Cap a child run's budget by what remains of its parent's, at this moment."""
        if self.parent_run is None:
            return
        changes: dict[str, Any] = {}
        for key, remaining in self.parent_run._remaining().items():
            own = getattr(self.limits, key)
            if remaining is not None and (own is None or remaining < own):
                changes[key] = max(remaining, 0)
        if changes:
            self.limits = self.limits.model_copy(update=changes)

    def _remaining(self) -> dict[str, Any]:
        """What is left of this run's token, cost and time budget (``None``: no limit)."""
        limits, usage = self.limits, self.usage
        remaining: dict[str, Any] = {"max_total_tokens": None, "max_cost_usd": None}
        if limits.max_total_tokens is not None:
            remaining["max_total_tokens"] = limits.max_total_tokens - usage.total_tokens
        if limits.max_cost_usd is not None:
            remaining["max_cost_usd"] = limits.max_cost_usd - (usage.cost_usd or 0.0)
        remaining["timeout_s"] = (
            None if self._deadline is None else self._deadline - time.monotonic()
        )
        return remaining

    def _start_timer(self) -> asyncio.TimerHandle | None:
        timeout = self.limits.timeout_s
        if timeout is None:
            return None
        self._deadline = time.monotonic() + timeout
        if timeout <= 0:  # a child whose parent has no time left
            self.cancel("timeout")
            return None
        return asyncio.get_running_loop().call_later(timeout, self.cancel, "timeout")

    async def _prepare(self) -> None:
        state = await self.session.state()
        self.history = list(state.messages)
        self.results = dict(state.tool_results)
        self.started = set(state.started_tools)
        self.waiting = {item.tool_call_id: item for item in state.pending_inputs}
        self._saved_state = _state_json(state.values)
        self.state.update(state.values)
        self.state.update(self._state_seed)  # journaled once the run has started
        if self.resume:
            if state.open_run_id is None or not self.history:
                raise SessionError(
                    f"Session {self.session.id!r} has no unfinished run to resume. "
                    "Start a new one with agent.run()."
                )
            if state.agent and state.agent != self.root.name:
                self.agent = self._find_agent(state.agent)
            await self._load_agent()
            self._check_results(state)
            self.run_id = state.open_run_id
            self._check_tokens(state)
            self._journal_open = True
            self._ordinal = state.run_messages
            await self._restore_answer()
            return

        if self.seed and self.history:
            raise ConfigurationError(
                "history= seeds a new conversation, but this session already has messages. "
                "Pass one or the other."
            )
        if not (self.history or self.seed or self.prompt is not None):
            raise ConfigurationError("Nothing to run: give a prompt, a history or a session.")
        await self._load_agent()
        if state.open_run_id is not None:
            await self.session.append(
                "run_finished", {"status": "abandoned"}, run_id=state.open_run_id
            )
        started: dict[str, Any] = {"agent": self.root.name}
        if self.parent is not None:
            started.update(
                parent_run_id=self.parent.run_id,
                parent_tool_call_id=self.parent.tool_call_id,
                depth=self.depth,
            )
        await self._journal("run_started", started)
        self._journal_open = True
        for message in self.seed:
            await self._append(message, new=False)
        # A previous run may have stopped with tool calls still open (crashed,
        # or paused for approvals and inputs). Close them so the conversation
        # stays valid for the provider.
        await self._settle_open_calls(prompts.TOOL_NOT_EXECUTED)
        if self.prompt is not None:
            message = _as_message(self.prompt)
            # A blocked prompt never joins the conversation; the run ends as "blocked".
            self.guardrail = await self._check_guardrails("input", message)
            if self.guardrail is None:
                await self._append(message)

    async def _load_agent(self) -> None:
        self.tools = await self.agent._collect_tools()
        self.handoffs = {target._handoff_tool_name: target for target in self.agent.handoffs}
        clash = set(self.tools) & set(self.handoffs)
        if clash:
            raise ConfigurationError(
                f"Agent {self.agent.name!r}: tool name(s) {sorted(clash)} collide with handoffs."
            )
        answer = self.agent.output_tool_name
        if self.agent.output_mode == "tool" and (answer in self.tools or answer in self.handoffs):
            raise ConfigurationError(
                f"Agent {self.agent.name!r}: the final-answer tool {answer!r} has the name of "
                "another tool. Choose a different output_tool_name."
            )

    async def _check_guardrails(self, stage: GuardrailStage, value: Any) -> TrippedGuardrail | None:
        """Run the guardrails of ``stage`` in order; journal and return the first trip.

        Input guardrails are the root agent's. Output guardrails are the root
        agent's and then the active agent's, so a handoff cannot skip them. An
        exception other than `GuardrailTripped` propagates and fails the run.
        """
        if stage == "input":
            guardrails = self.root.input_guardrails
        else:
            guardrails = _unique([*self.root.output_guardrails, *self.agent.output_guardrails])
        ctx = self._ctx()
        for guardrail in guardrails:
            name = getattr(guardrail, "__name__", type(guardrail).__name__)
            try:
                verdict = guardrail(ctx, value)
                if inspect.isawaitable(verdict):
                    verdict = await verdict
            except GuardrailTripped as trip:
                tripped = TrippedGuardrail(name=trip.name or name, reason=trip.reason, stage=stage)
                await self._journal("guardrail_tripped", tripped.model_dump(mode="json"))
                return tripped
            if verdict is not None:
                raise ConfigurationError(
                    f"Guardrail {name!r} returned {verdict!r}. A guardrail returns None to let "
                    "the value through and raises GuardrailTripped(reason) to block it."
                )
        return None

    async def _restore_answer(self) -> None:
        """On resume, take back a final-answer call that was accepted before the run stopped.

        The run then finishes (output guardrails included) without asking the model again.
        """
        agent = self.agent
        if agent.output_mode != "tool" or len(self.history) < 2 or self.history[-1].role != "tool":
            return
        accepted = {
            result.tool_call_id
            for result in self.history[-1].tool_results
            if result.name == agent.output_tool_name and not result.is_error
        }
        previous = self.history[-2]
        for call in previous.tool_calls if previous.role == "assistant" else []:
            if call.id in accepted:
                try:
                    self.output = await self._validate_output(agent._validate_answer_call(call))
                except (_Rejected, ValueError):
                    return  # no longer accepted (the agent changed): the model answers again
                self._answered = True
                return

    def _guardrail_event(self, tripped: TrippedGuardrail) -> GuardrailTriggered:
        return GuardrailTriggered(
            name=tripped.name, reason=tripped.reason, stage=tripped.stage, agent=self.agent.name
        )

    def _check_results(self, state: SessionState) -> None:
        """Fail before anything runs when ``results`` names a call that waits for no result."""
        last = self.history[-1]
        approvals = {item.tool_call_id for item in state.pending_approvals}
        accepted = set(self.waiting)
        for call in last.tool_calls if last.role == "assistant" else []:
            tool = self.tools.get(call.name)
            if call.id in self.results or call.id in approvals:
                continue
            if tool is not None and tool.deferred:
                accepted.add(
                    call.id
                )  # not journaled as waiting yet (a crash), but it can only wait
        unknown = [call_id for call_id in self.supplied if call_id not in accepted]
        if not unknown:
            return
        hint = (
            " Calls that wait for approval are answered with approvals=, not results=."
            if approvals & set(unknown)
            else ""
        )
        raise SessionError(
            f"results= names tool call(s) {unknown} that do not wait for a result in session "
            f"{self.session.id!r}. Calls waiting for one: {sorted(accepted) or 'none'}.{hint}"
        )

    def _check_tokens(self, state: SessionState) -> None:
        """With signed pauses, refuse answers that do not carry their pause's token.

        An approval must match the call as the journal records it in the
        assistant message, which is what will run, and the recorded pause must
        still match its own token. A result must match the waiting call as
        the journal records it. Results are unwrapped from `InputResult` (and,
        when pauses are signed, from ``{"token": ..., "value": ...}``) here.
        """
        signed = self.permissions.signs_pauses
        unwrapped: dict[str, Any] = {}
        tokens: dict[str, str | None] = {}
        for call_id, value in self.supplied.items():
            if isinstance(value, InputResult):
                unwrapped[call_id], tokens[call_id] = value.value, value.token
            elif signed and isinstance(value, Mapping) and set(value) == {"token", "value"}:
                unwrapped[call_id], tokens[call_id] = value["value"], value["token"]
            else:
                unwrapped[call_id], tokens[call_id] = value, None
        self.supplied = unwrapped
        if not signed:
            return

        last = self.history[-1]
        calls = {call.id: call for call in last.tool_calls} if last.role == "assistant" else {}
        recorded = {item.tool_call_id: item for item in state.pending_approvals}
        for call_id, approval in self.approvals.items():
            call = calls.get(call_id)
            if call is None or call_id in self.results:
                continue  # not an open call: the approval is ignored, as without signing
            self._verify_token(approval.token, call.id, call.name, call.arguments, "approval")
            pause = recorded.get(call_id)
            if pause is not None and not self._signs(pause.token, pause):
                raise self._tampered(call_id, call.name)
        for call_id, token in tokens.items():
            target: PendingInput | ToolCallPart | None = self.waiting.get(call_id)
            if target is None:  # a deferred call whose tool_deferred entry was lost
                target = calls.get(call_id)
            assert target is not None  # _check_results accepted only open calls
            self._verify_token(token, call_id, target.name, target.arguments, "result")

    def _signs(self, token: str | None, pause: PendingApproval | PendingInput) -> bool:
        return self.permissions.verify(
            token,
            run_id=self.run_id,
            tool_call_id=pause.tool_call_id,
            name=pause.name,
            arguments=pause.arguments,
        )

    def _verify_token(
        self,
        token: str | None,
        call_id: str,
        name: str,
        arguments: Mapping[str, Any],
        kind: Literal["approval", "result"],
    ) -> None:
        if token is None:
            how = (
                "Approval.for_call(pending) or Approval(approved=..., token=pending.token), with "
                "pending from result.pending_approvals"
                if kind == "approval"
                else "InputResult.for_call(pending, value) or "
                '{"token": pending.token, "value": value}, with pending from result.pending_inputs'
            )
            raise ApprovalTokenError(
                f"Tool call {call_id!r} ({name}) belongs to a signed pause, so its {kind} needs "
                f"the pause's token. Pass {how}. Nothing was run."
            )
        if not self.permissions.verify(
            token, run_id=self.run_id, tool_call_id=call_id, name=name, arguments=arguments
        ):
            raise self._tampered(call_id, name)

    def _tampered(self, call_id: str, name: str) -> ApprovalTokenError:
        return ApprovalTokenError(
            f"The token for tool call {call_id!r} ({name}) does not match the call recorded in "
            f"session {self.session.id!r}: the token belongs to another call or another secret, "
            "or the journal was changed after the pause. Nothing was run. Check where the "
            "answer came from; to act on the call anyway, start a new run."
        )

    def _find_agent(self, name: str) -> Agent:
        seen: set[int] = set()
        queue = [self.root]
        while queue:
            candidate = queue.pop(0)
            if id(candidate) in seen:
                continue
            seen.add(id(candidate))
            if candidate.name == name:
                return candidate
            queue.extend(candidate.handoffs)
        raise SessionError(
            f"The run was last handled by agent {name!r}, which is not reachable from "
            f"{self.root.name!r} through handoffs."
        )

    def _limit_reached(self) -> str | None:
        limits = self.limits
        if limits.max_turns is not None and self.turn >= limits.max_turns:
            return "max_turns"
        if (
            limits.max_total_tokens is not None
            and self.usage.total_tokens >= limits.max_total_tokens
        ):
            return "max_total_tokens"
        if limits.max_cost_usd is not None and (
            limits.max_cost_usd <= 0
            or (self.usage.cost_usd is not None and self.usage.cost_usd >= limits.max_cost_usd)
        ):
            return "max_cost_usd"
        return None

    # -- model phase ----------------------------------------------------------------

    async def _model_phase(self) -> AsyncIterator[Event]:
        ctx = self._ctx()
        await self._maybe_compact(ctx)
        request = await self.hooks.before_model(ctx, await self._build_request(ctx))
        if (updated := await self._save_state()) is not None:
            yield updated
        model = self.agent.model
        attempt = 0
        while True:
            response: ModelResponse | None = None
            text: list[str] = []
            emitted = False
            try:
                async with contextlib.aclosing(
                    self._until_cancelled(model.stream(request))
                ) as events:
                    async for event in events:
                        if isinstance(event, ResponseDone):
                            response = event.response
                            continue
                        if isinstance(event, TextDelta):
                            text.append(event.text)
                        emitted = True
                        yield event
            except ModelError as error:
                # Never repeat a request whose partial output was already shown.
                may_retry = (
                    error.retryable
                    and (not error.ambiguous or self.limits.retry_ambiguous)
                    and not emitted
                    and attempt < self.limits.model_retries
                    and not self._cancel.is_set()
                )
                if not may_retry:
                    raise
                attempt += 1
                delay = error.retry_after if error.retry_after is not None else 0.5 * 2**attempt
                await self._sleep(min(delay, 60.0))
                continue
            break

        if response is None:
            if not self._cancel.is_set():
                raise ModelResponseError(
                    "The model stream ended without a response.",
                    provider=model.provider,
                    model=model.name,
                    ambiguous=True,
                )
            partial = "".join(text)
            if partial:
                await self._append(
                    Message(
                        role="assistant",
                        parts=[TextPart(text=partial)],
                        name=self.agent.name,
                        metadata={"interrupted": True},
                    )
                )
            return

        response = await self.hooks.after_model(ctx, response)
        if (updated := await self._save_state()) is not None:
            yield updated
        self.usage = self.usage.add(response.usage)
        self._last_response = response
        self._refused = response.finish_reason == "content_filter"
        message = response.message.model_copy(update={"name": self.agent.name})
        await self._append(message)
        yield ModelResponded(response=response, agent=self.agent.name)

    async def _build_request(self, ctx: RunContext[Any]) -> ModelRequest:
        agent = self.agent
        system = await agent._render_instructions(ctx)
        output: OutputSpec | None = None
        addition: str | None = None
        if agent.output_mode == "tool":
            # The tool is offered, never forced: some providers reject a forced tool choice.
            addition = prompts.OUTPUT_TOOL_INSTRUCTIONS.format(name=agent.output_tool_name)
        elif agent._output_schema is not None:
            if agent.output_mode == "prompted":
                schema = json.dumps(agent._output_schema, ensure_ascii=False)
                addition = prompts.OUTPUT_INSTRUCTIONS.format(schema=schema)
            else:
                output = OutputSpec(json_schema=agent._output_schema, strict=agent.output_strict)
        if addition is not None:
            system = f"{system}\n\n{addition}" if system else addition
        messages = ([Message.system(system)] if system else []) + list(self.history)
        specs = [tool.spec for tool in self.tools.values()]
        specs.extend(target._handoff_spec() for target in self.handoffs.values())
        if agent.output_mode == "tool":
            specs.append(agent._answer_tool_spec())
        return ModelRequest(
            messages=messages,
            tools=specs,
            settings=agent.settings.merged(self.settings),
            output=output,
        )

    async def _maybe_compact(self, ctx: RunContext[Any]) -> None:
        compactor = self.root.compactor
        if compactor is None:
            return
        # The first request of a run or a resume has no response yet: an empty one (no usage)
        # makes the compactor estimate the whole history, so long sessions are compacted too.
        last = self._last_response or ModelResponse(message=Message(role="assistant"))
        replacement = await compactor.compact(ctx, list(self.history), last, self.agent.model)
        if replacement:
            self.history = list(replacement)
            await self._journal(
                "compaction",
                {"messages": [message.model_dump(mode="json") for message in replacement]},
            )

    async def _finalize(self, last: Message) -> bool:
        """Accept the text of ``last`` as the final answer, or ask the model to correct it."""
        agent = self.agent
        if agent.output_mode == "tool":
            required = prompts.OUTPUT_TOOL_REQUIRED.format(name=agent.output_tool_name)
            self._count_rejection(
                _Rejected(required, f"was not given with {agent.output_tool_name}")
            )
            await self._append(Message.user(required))
            return False
        try:
            if agent._output_adapter is None:
                answer: Any = last.text
            else:
                try:
                    answer = agent._output_adapter.validate_python(extract_json(last.text))
                except ValueError as error:  # pydantic's ValidationError included
                    raise _Rejected(_brief(error), "did not match the output type") from error
            self.output = await self._validate_output(answer)
            return True
        except _Rejected as rejection:
            self._count_rejection(rejection)
            template = prompts.OUTPUT_RETRY if agent._output_adapter else prompts.OUTPUT_TEXT_RETRY
            await self._append(Message.user(template.format(error=rejection.message)))
            return False

    async def _validate_output(self, output: Any) -> Any:
        """Run the active agent's output validators in order. Raises `_Rejected` on `ModelRetry`."""
        ctx = self._ctx()
        for validator in self.agent.output_validators:
            try:
                replaced = validator(ctx, output)
                if inspect.isawaitable(replaced):
                    replaced = await replaced
            except ModelRetry as retry:
                raise _Rejected(_brief(retry), "was rejected by an output validator") from retry
            if replaced is not None:
                output = replaced
        return output

    def _count_rejection(self, rejection: _Rejected) -> None:
        """Count one more attempt at the final answer, or raise once they are used up."""
        if self._output_attempts >= self.limits.output_retries:
            raise OutputValidationError(
                f"The final answer {rejection.problem} after {self._output_attempts + 1} "
                f"attempts: {rejection.message}"
            ) from rejection
        self._output_attempts += 1

    async def _answer_phase(
        self, answer: ToolCallPart, calls: list[ToolCallPart]
    ) -> AsyncIterator[Event]:
        """Check a final-answer call (``output_mode="tool"``).

        Accepted: the call gets a short result, every other open call of the
        turn gets a result saying it was not run, the tool turn is closed and
        the run completes. Rejected: the call gets an error result naming the
        problem, and the rest of the turn proceeds as usual.
        """
        agent = self.agent
        try:
            try:
                value = agent._validate_answer_call(answer)
            except ValidationError as error:
                problem = _describe_validation_error(error)
                raise _Rejected(problem, "did not match the output type") from error
            except ValueError as error:
                raise _Rejected(_brief(error), "did not match the output type") from error
            output = await self._validate_output(value)
        except _Rejected as rejection:
            self._count_rejection(rejection)
            content = prompts.OUTPUT_TOOL_RETRY.format(error=rejection.message, name=answer.name)
            result = _error_result(answer, _shorten(content, self.limits.max_tool_result_chars))
            await self._record(result)
            yield ToolCallFinished(call=answer, result=result, agent=agent.name)
            return
        self.output = output
        self._answered = True
        accepted = ToolResultPart(
            tool_call_id=answer.id, name=answer.name, content=prompts.OUTPUT_TOOL_ACCEPTED
        )
        await self._record(accepted)
        yield ToolCallFinished(call=answer, result=accepted, agent=agent.name)
        for call in calls:
            if call.id not in self.results:
                skipped = _error_result(call, prompts.OUTPUT_TOOL_SKIPPED.format(name=call.name))
                await self._record(skipped)
                yield ToolCallFinished(call=call, result=skipped, agent=agent.name)
        await self._close_tool_turn(calls)

    # -- tool phase -------------------------------------------------------------------

    async def _tool_phase(self, assistant: Message) -> AsyncIterator[Event]:
        """Give every open call of ``assistant`` a result, or pause.

        A final-answer call (``output_mode="tool"``) is checked first; when it
        is accepted, nothing else of the turn runs. Then each call without a
        result is answered from ``results=``, keeps waiting for input, is
        refused, waits for approval, is deferred (``deferred=True`` tools) or
        runs. The run pauses when any call waits.
        """
        calls = assistant.tool_calls
        answer = next(
            (c for c in calls if self.agent._is_answer_call(c) and c.id not in self.results), None
        )
        if answer is not None:
            async with contextlib.aclosing(self._answer_phase(answer, calls)) as phase:
                async for event in phase:
                    yield event
            if self._answered:
                return

        runnable: list[tuple[ToolCallPart, Tool]] = []
        pending: list[PendingApproval] = []
        handoff: ToolCallPart | None = None
        budget = self.limits.max_tool_calls

        for call in calls:
            if call.id in self.results:
                continue
            if call.id in self.supplied:  # given to resume(results=...)
                result = self._supplied_result(call)
                await self._record(result)
                yield ToolCallFinished(call=call, result=result, agent=self.agent.name)
                continue
            if call.id in self.waiting:
                continue  # authorised earlier; only results= can answer it
            refusal: str | None = None
            if call.parse_error is not None:
                refusal = prompts.INVALID_ARGUMENTS_JSON.format(
                    name=call.name, error=call.parse_error
                )
            elif self.agent._is_answer_call(call):
                refusal = prompts.OUTPUT_TOOL_DUPLICATE
            elif call.name in self.handoffs:
                if handoff is None:
                    handoff = call
                    continue
                refusal = prompts.HANDOFF_DUPLICATE
            elif call.name not in self.tools:
                available = [*self.tools, *self.handoffs]
                if self.agent.output_mode == "tool":
                    available.append(self.agent.output_tool_name)
                refusal = prompts.UNKNOWN_TOOL.format(
                    name=call.name, available=", ".join(available) or "none"
                )
            elif budget is not None and self._tool_calls >= budget:
                refusal = prompts.TOOL_CALL_LIMIT.format(name=call.name, limit=budget)
                self._tool_limit_hit = True
            else:
                tool = self.tools[call.name]
                self._tool_calls += 1
                if call.id in self.started and not (tool.read_only or tool.idempotent):
                    refusal = prompts.TOOL_INTERRUPTED.format(name=call.name)
                else:
                    verdict, payload = await self._authorize(tool, call)
                    if verdict == "run" and tool.deferred:
                        await self._defer(payload)  # never executed by the loop
                        continue
                    if verdict == "run":
                        runnable.append((payload, tool))
                        continue
                    if verdict == "pending":
                        pending.append(payload)
                        continue
                    refusal = prompts.PERMISSION_DENIED.format(name=call.name, reason=payload)
            result = _error_result(call, refusal)
            await self._record(result)
            yield ToolCallFinished(call=call, result=result, agent=self.agent.name)

        for group in self._groups(runnable):
            async with contextlib.aclosing(self._run_group(group)) as events:
                async for event in events:
                    yield event
            if self._cancel.is_set():
                return  # the loop settles the calls that never finished

        self.pending = pending
        self.inputs = [self.waiting[call.id] for call in calls if call.id in self.waiting]
        if self.pending or self.inputs:
            for item in self.pending:
                yield ApprovalRequired(approval=item, agent=self.agent.name)
            for waiting in self.inputs:
                yield InputRequired(input=waiting, agent=self.agent.name)
            return

        target: Agent | None = None
        if handoff is not None:
            target = self.handoffs[handoff.name]
            await self._record(
                ToolResultPart(
                    tool_call_id=handoff.id,
                    name=handoff.name,
                    content=prompts.HANDOFF_RESULT.format(name=target.name),
                )
            )
        await self._close_tool_turn(calls)
        if target is not None:
            previous, self.agent = self.agent, target
            await self._journal("agent_switched", {"agent": target.name})
            await self._load_agent()
            yield AgentSwitched(from_agent=previous.name, to_agent=target.name)

    async def _authorize(self, tool: Tool, call: ToolCallPart) -> tuple[str, Any]:
        """Decide one call.

        Returns ``("run", call)``, ``("pending", PendingApproval)`` or ``("deny", reason)``.
        """
        approval = self.approvals.pop(call.id, None)
        if approval is None:
            ctx = self._ctx(tool_call_id=call.id)
            decision = await self.permissions.decide(ctx, tool, call)
            if decision.behavior == "allow":
                return "run", call
            if decision.behavior == "deny":
                return "deny", decision.reason or "it is not permitted."
            if self.permissions.on_ask is None:
                return "pending", PendingApproval(
                    tool_call_id=call.id,
                    name=call.name,
                    arguments=call.arguments,
                    reason=decision.reason,
                    token=self.permissions.sign(
                        run_id=self.run_id,
                        tool_call_id=call.id,
                        name=call.name,
                        arguments=call.arguments,
                    ),
                )
            approval = await self.permissions.request_approval(ctx, tool, call, decision.reason)
        if not approval.approved:
            return "deny", approval.reason or prompts.APPROVAL_DECLINED
        if approval.arguments is not None:
            call = call.model_copy(update={"arguments": approval.arguments, "raw_arguments": None})
        return "run", call

    def _groups(
        self, runnable: list[tuple[ToolCallPart, Tool]]
    ) -> list[list[tuple[ToolCallPart, Tool]]]:
        """Split calls into groups; a group runs concurrently, groups run in order."""
        mode = self.limits.parallel_tools
        if mode == "all":
            return [runnable] if runnable else []
        groups: list[list[tuple[ToolCallPart, Tool]]] = []
        batch: list[tuple[ToolCallPart, Tool]] = []
        for item in runnable:
            if mode == "read_only" and item[1].read_only:
                batch.append(item)
                continue
            if batch:
                groups.append(batch)
                batch = []
            groups.append([item])
        if batch:
            groups.append(batch)
        return groups

    async def _run_group(self, group: list[tuple[ToolCallPart, Tool]]) -> AsyncIterator[Event]:
        queue: asyncio.Queue[Any] = asyncio.Queue()
        gate = asyncio.Semaphore(max(1, self.limits.max_parallel_tools))
        tasks = [
            asyncio.create_task(self._run_tool(call, tool, queue, gate)) for call, tool in group
        ]
        cancelled = asyncio.ensure_future(self._cancel.wait())
        finished = 0
        try:
            while finished < len(tasks):
                item_ready = asyncio.ensure_future(queue.get())
                await asyncio.wait({item_ready, cancelled}, return_when=asyncio.FIRST_COMPLETED)
                if not item_ready.done():
                    item_ready.cancel()
                    return
                item = item_ready.result()
                if isinstance(item, _Failure):
                    raise item.error
                if isinstance(item, _Deferred):
                    finished += 1
                    continue
                if isinstance(item, ToolCallFinished):
                    finished += 1
                yield item
        finally:
            cancelled.cancel()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_tool(
        self, call: ToolCallPart, tool: Tool, queue: asyncio.Queue[Any], gate: asyncio.Semaphore
    ) -> None:
        agent = self.agent.name
        try:
            async with gate:
                ctx = self._ctx(tool_call_id=call.id, emit=queue.put_nowait)
                await self._journal("tool_started", {"tool_call_id": call.id, "name": call.name})
                queue.put_nowait(ToolCallStarted(call=call, agent=agent))
                began = time.perf_counter()
                hooked = await self.hooks.before_tool(ctx, call)
                if isinstance(hooked, ToolResultPart):
                    result = hooked
                else:
                    try:
                        result = await self._invoke(ctx, tool, hooked)
                    except DeferToolCall as deferral:
                        if (updated := await self._save_state()) is not None:
                            queue.put_nowait(updated)
                        await self._defer(
                            call.model_copy(update={"arguments": hooked.arguments}),
                            deferral.reason,
                            deferral.metadata,
                        )
                        queue.put_nowait(_Deferred())
                        return
                result = await self.hooks.after_tool(ctx, call, result)
                result = result.model_copy(update={"tool_call_id": call.id, "name": call.name})
                # State before result: after a crash, a recorded result implies its state.
                if (updated := await self._save_state()) is not None:
                    queue.put_nowait(updated)
                await self._record(result)
                queue.put_nowait(
                    ToolCallFinished(
                        call=call,
                        result=result,
                        duration_ms=(time.perf_counter() - began) * 1000,
                        agent=agent,
                    )
                )
        except asyncio.CancelledError:
            raise
        except BaseException as error:  # a hook or the journal failed: stop the run
            queue.put_nowait(_Failure(error))

    async def _invoke(self, ctx: RunContext[Any], tool: Tool, call: ToolCallPart) -> ToolResultPart:
        """Run a tool and turn whatever happens into a result the model can read.

        `DeferToolCall` is the one exception that passes through: the call then has no result.
        """
        metadata: dict[str, Any] = {}
        is_error = True
        try:
            value = await tool.run(ctx, call.arguments, timeout=self.limits.tool_timeout_s)
            content = render_tool_output(value)
            is_error = False
        except DeferToolCall:
            raise
        except ToolArgumentError as error:
            content = prompts.INVALID_ARGUMENTS.format(name=call.name, error=error)
        except ToolError as error:
            content = str(error)
        except TimeoutError:
            seconds = tool.timeout if tool.timeout is not None else self.limits.tool_timeout_s
            content = prompts.TOOL_TIMEOUT.format(name=call.name, seconds=seconds or 0)
        except Exception as error:
            logger.exception("Tool %s raised", call.name)
            content = prompts.TOOL_FAILED.format(
                name=call.name, error=f"{type(error).__name__}: {error}"
            )
            metadata["exception"] = type(error).__name__
        return ToolResultPart(
            tool_call_id=call.id,
            name=call.name,
            content=_shorten(content, self.limits.max_tool_result_chars),
            is_error=is_error,
            metadata=metadata,
        )

    async def _record(self, result: ToolResultPart) -> None:
        self.results[result.tool_call_id] = result
        self.waiting.pop(result.tool_call_id, None)
        await self._journal("tool_result", {"result": result.model_dump(mode="json")})

    async def _defer(
        self,
        call: ToolCallPart,
        reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Record that ``call`` waits for a result from outside the run (signed, if set up)."""
        waiting = PendingInput(
            tool_call_id=call.id,
            name=call.name,
            arguments=call.arguments,
            reason=reason,
            metadata=dict(metadata or {}),
            token=self.permissions.sign(
                run_id=self.run_id, tool_call_id=call.id, name=call.name, arguments=call.arguments
            ),
        )
        try:
            data = waiting.model_dump(mode="json")
        except PydanticSerializationError as error:
            raise SessionError(
                f"The deferred call to {call.name!r} cannot be journaled: {error}. "
                "Give DeferToolCall(metadata=...) plain JSON values."
            ) from error
        self.waiting[call.id] = waiting
        await self._journal("tool_deferred", {"input": data})

    def _supplied_result(self, call: ToolCallPart) -> ToolResultPart:
        """Turn the value given to ``resume(results=...)`` for ``call`` into its result."""
        value = self.supplied.pop(call.id)
        limit = self.limits.max_tool_result_chars
        if isinstance(value, ToolResultPart):
            return value.model_copy(update={"tool_call_id": call.id, "name": call.name})
        if isinstance(value, BaseException):
            return _error_result(call, _shorten(str(value) or type(value).__name__, limit))
        return ToolResultPart(
            tool_call_id=call.id, name=call.name, content=_shorten(render_tool_output(value), limit)
        )

    async def _settle_open_calls(self, template: str) -> None:
        """Give every open tool call of the last assistant message a result, then close the turn."""
        if not self.history:
            return
        last = self.history[-1]
        if last.role != "assistant" or not last.tool_calls:
            return
        for call in last.tool_calls:
            if call.id in self.results:
                continue
            if call.id in self.started:
                wording = prompts.TOOL_INTERRUPTED
            elif call.id in self.waiting:
                wording = prompts.TOOL_INPUT_ABANDONED
            else:
                wording = template
            await self._record(_error_result(call, wording.format(name=call.name)))
        await self._close_tool_turn(last.tool_calls)

    async def _close_tool_turn(self, calls: list[ToolCallPart]) -> None:
        await self._append(Message(role="tool", parts=[self.results[call.id] for call in calls]))
        self.results = {}
        self.started = set()
        self.waiting = {}

    # -- plumbing ---------------------------------------------------------------------

    def _ctx(
        self, *, tool_call_id: str | None = None, emit: Callable[[Any], None] | None = None
    ) -> RunContext[Any]:
        return RunContext(
            deps=self.deps,
            run_id=self.run_id,
            session_id=self.session.id,
            agent_name=self.agent.name,
            turn=self.turn,
            tool_call_id=tool_call_id,
            # run + position of the assistant message in the run + call: the
            # same after a resume, and unique even if a provider reuses call ids.
            operation_id=(
                f"{self.run_id}:{self._ordinal}:{tool_call_id}" if tool_call_id else None
            ),
            usage=self.usage,
            messages=tuple(self.history),
            state=self.state,
            parent_run_id=self.parent.run_id if self.parent is not None else None,
            parent_tool_call_id=self.parent.tool_call_id if self.parent is not None else None,
            depth=self.depth,
            _emit=emit,
            _add_usage=self._add_usage,
            _run=self,
        )

    def _add_usage(self, usage: Usage) -> None:
        self.usage = self.usage.add(usage)

    async def _save_state(self) -> StateUpdated | None:
        """Journal ``ctx.state`` if it changed since it was last journaled.

        Every entry holds the full values, and the lock keeps entries in the
        order of the changes, so concurrent tools cannot leave a stale state last.
        """
        async with self._state_lock:
            snapshot = _state_json(self.state)
            if snapshot == self._saved_state:
                return None
            await self._journal("state_updated", {"values": json.loads(snapshot)})
            self._saved_state = snapshot
        return StateUpdated(values=json.loads(snapshot), agent=self.agent.name)

    async def _journal(self, kind: str, data: dict[str, Any]) -> None:
        await self.session.append(kind, data, run_id=self.run_id)

    async def _append(self, message: Message, *, new: bool = True) -> None:
        self.history.append(message)
        self._ordinal += 1
        if new:
            self.new_messages.append(message)
        await self._journal("message", {"message": message.model_dump(mode="json")})

    async def _sleep(self, seconds: float) -> None:
        """Wait, but wake up at once if the run is cancelled."""
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._cancel.wait(), timeout=seconds)

    async def _until_cancelled(self, source: AsyncIterator[T]) -> AsyncIterator[T]:
        """Yield from ``source`` until it ends or the run is cancelled, then close it."""
        cancelled = asyncio.ensure_future(self._cancel.wait())
        step: asyncio.Future[T] | None = None
        try:
            while True:
                step = asyncio.ensure_future(source.__anext__())
                await asyncio.wait({step, cancelled}, return_when=asyncio.FIRST_COMPLETED)
                if not step.done():
                    return
                try:
                    item = step.result()
                except StopAsyncIteration:
                    return
                yield item
        finally:
            cancelled.cancel()
            if step is not None and not step.done():
                step.cancel()
                with contextlib.suppress(BaseException):
                    await step
            close = getattr(source, "aclose", None)
            if close is not None:
                with contextlib.suppress(Exception):
                    await close()

    def _result(self) -> RunResult:
        status = self.status
        answer = next((m for m in reversed(self.history) if m.role == "assistant"), None)
        result = RunResult(
            run_id=self.run_id,
            session_id=self.session.id,
            agent=self.agent.name,
            status=status,
            stop_reason=self.stop_reason,
            output=self.output if status == "completed" else None,
            # A blocked answer stays in messages for audit, but is not handed out as the text.
            text=answer.text if answer is not None and status != "blocked" else "",
            messages=list(self.history),
            new_messages=list(self.new_messages),
            usage=self.usage,
            turns=self.turn,
            pending_approvals=list(self.pending),
            pending_inputs=list(self.inputs),
            guardrail=self.guardrail,
        )
        result._session = self.session
        return result


# -- helpers ---------------------------------------------------------------------------


def _normalize_tools(items: Iterable[Any]) -> list[Tool | Toolkit]:
    normalized: list[Tool | Toolkit] = []
    for item in items:
        if isinstance(item, Tool | Toolkit):
            normalized.append(item)
        elif isinstance(item, Agent):
            normalized.append(item.as_tool())
        elif callable(item):
            normalized.append(make_tool(item))
        elif isinstance(item, Iterable) and not isinstance(item, str | bytes):
            normalized.extend(_normalize_tools(item))
        else:
            raise ConfigurationError(
                f"{item!r} is not a tool. "
                "Pass a function, a Tool, a Toolkit, an Agent or a list of them."
            )
    return normalized


def _unique(items: Iterable[T]) -> list[T]:
    """``items`` without repeats, by identity, in their first order."""
    seen: set[int] = set()
    unique: list[T] = []
    for item in items:
        if id(item) not in seen:
            seen.add(id(item))
            unique.append(item)
    return unique


def _as_message(prompt: Prompt) -> Message:
    if isinstance(prompt, Message):
        return prompt
    if isinstance(prompt, str):
        return Message.user(prompt)
    return Message.user(list(prompt or []))


def _tool_safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)[:48] or "agent"


def _error_result(call: ToolCallPart, content: str) -> ToolResultPart:
    return ToolResultPart(tool_call_id=call.id, name=call.name, content=content, is_error=True)


def _shorten(text: str, limit: int | None) -> str:
    """Keep the start and the end of an over-long text, with a marker in the middle."""
    if limit is None or len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    marker = prompts.TRUNCATED.format(omitted=len(text) - limit)
    return f"{text[:head]}{marker}{text[len(text) - tail :]}"


def _state_json(values: Mapping[str, Any]) -> str:
    """Canonical JSON of the session state.

    Raises `SessionError`, naming the key, when a value would not come back
    unchanged from the journal (a tuple, a set, a date, a model, NaN).
    """
    try:
        text = json.dumps(values, ensure_ascii=False, sort_keys=True, allow_nan=False)
        if json.loads(text) == values:
            return text
    except (TypeError, ValueError):
        pass
    for key, value in values.items():
        if not isinstance(key, str):
            raise SessionError(
                f"Session state keys must be strings; {key!r} is a {type(key).__name__}."
            )
        try:
            kept = json.loads(json.dumps(value, sort_keys=True, allow_nan=False)) == value
        except (TypeError, ValueError):
            kept = False
        if not kept:
            raise SessionError(
                f"ctx.state[{key!r}] holds a {type(value).__name__} that would not come back "
                "unchanged from the session journal, which stores JSON. Store strings, numbers, "
                "booleans, None, lists and dicts with string keys: for a pydantic model use "
                "model.model_dump(mode='json'), for a date its ISO string."
            )
    raise SessionError("The session state is not JSON-serialisable.")


def _brief(error: Exception, limit: int = 1500) -> str:
    text = str(error)
    return text if len(text) <= limit else f"{text[:limit]}..."


_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Read the JSON value out of a model's answer.

    Accepts bare JSON, JSON inside a code fence, and JSON surrounded by a few
    words, which is how models without schema enforcement tend to answer.
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("the reply was empty")
    with contextlib.suppress(ValueError):
        return json.loads(stripped)
    fenced = _FENCE.search(stripped)
    if fenced:
        with contextlib.suppress(ValueError):
            return json.loads(fenced.group(1).strip())
    decoder = json.JSONDecoder()
    for position, character in enumerate(stripped):
        if character in "{[":
            with contextlib.suppress(ValueError):
                value, _ = decoder.raw_decode(stripped, position)
                return value
    raise ValueError("the reply did not contain a JSON value")
