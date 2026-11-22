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
import concurrent.futures
import contextlib
import copy as copying
import dataclasses
import functools
import inspect
import json
import logging
import re
import sys
import threading
import time
import uuid
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from contextvars import ContextVar
from dataclasses import replace
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, create_model
from pydantic_core import PydanticSerializationError, to_jsonable_python

from . import prompts
from . import session as _session
from .context import Compactor
from .effects import _canonical, _engine, _kinds, _operator, _ports
from .effects._declare import Effect
from .effects._types import HOLD_DETAIL, MISSING, EffectRecord, HeldEffect, HoldReason, Settle
from .errors import (
    ApprovalConflictError,
    ApprovalTokenError,
    ConfigurationError,
    DeferToolCall,
    EffectHeld,
    GuardrailTripped,
    HeldEffectsError,
    JournalOutcomeUnknown,
    LeaseError,
    MaxDepthExceeded,
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
    OutputRejected,
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
from .permissions import Approval, InputResult, Permissions, arguments_digest, tool_spec_digest
from .results import GuardrailStage, RunResult, RunStatus, TrippedGuardrail
from .session import (
    Entry,
    PendingApproval,
    PendingInput,
    Session,
    SessionState,
    SessionStore,
    _usage_loads,
)
from .tools import (
    RunContext,
    Tool,
    Toolkit,
    ToolTimeout,
    _abandoned_threads_to,
    _clean_schema,
    _describe_validation_error,
    render_tool_output,
)
from .tools import tool as make_tool

logger = logging.getLogger("qagents")


def _effects_child_refusal(agent: Agent, *, durable_session: bool) -> str | None:
    """Effects children need their own durable session and writer."""
    if not durable_session and agent.effects_by_name():
        return f"Effects agent {agent.name!r} needs a separate durable session."
    return None


def _resends_automatically(tool: Tool) -> bool:
    """Whether rule D3 or D4 can ever re-send one of the tool's effects (SPEC 7.1): D3 needs a
    ``Dedupe`` or ``Unique`` contract, D4 an authoritative lookup. Opaque and reconcilable
    effects are never re-sent automatically, with or without a verifiable decision."""
    return any(
        effect.klass in ("deduplicated", "unique", "authoritative") for effect in tool.effects
    )


def _store_root(store: SessionStore) -> SessionStore:
    """Follow only library-owned writer wrappers, preserving the root store identity."""
    from .runtime import _LeasedStore
    from .session import LeasedWriter

    seen = set()
    while isinstance(store, (_LeasedStore, LeasedWriter)) and id(store) not in seen:
        seen.add(id(store))
        store = store.wrapped
    return store


def _forget_store(store: SessionStore) -> None:
    """Remove the process-local run entries for the harness's disposed store."""
    root = _store_root(store)
    for key, (candidate, _) in list(_LIVE_RUNS.items()):
        if candidate is root:
            _LIVE_RUNS.pop(key, None)


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

RUN_LEASE_TTL_S = 30.0
"""Seconds a run's lease on its session lasts without renewal (it is renewed
every third of that while the run is live). On a store with leases, a run
whose process died keeps its session for at most this long."""

_LIVE_RUNS: dict[tuple[int, str], tuple[Any, _Run]] = {}
"""Runs executing in this process, by (store, session id): one run per session at a time."""

_USAGE_KINDS = frozenset(
    {"message", "tool_result", "tool_checkpoint", "run_paused", "run_failed", "run_finished"}
)
"""Entry kinds that carry ``added_usage``: usage from ``ctx.add_usage`` since the last one."""

_SOURCE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
"""What ``RunContext.call_tool(source=...)`` accepts: it starts the call ids."""

_OUTSIDE_FIELDS = frozenset(
    {
        "source",
        "caller_tool_call_id",
        "tool_call_id",
        "name",
        "result",
        "counted",
        "usage",
        "added_usage",
        "added_usage_by_operation",
        "approved",
        "reason",
        "arguments_changed",
        "arguments",
    }
)
"""Fields the journal entries of a call from outside the model already have, or that
`derive_state` reads from them (``usage`` counts as model usage on resume), so
``RunContext.call_tool(tags=...)`` cannot set them."""


class Limits(BaseModel):
    """Hard stops for a run. ``None`` switches a limit off.

    A child run (``parent=ctx``, as an agent used as a tool starts) shares its
    parents' budgets: it stops at what remains of their ``max_total_tokens``
    and ``max_cost_usd``, counted live, so child runs working in parallel
    cannot together spend more than their parent has left. Its ``timeout_s``
    is capped by what remains of its parent's when it starts, and its
    ``max_depth`` is the lowest along the chain of parent runs.

    ``max_turns``, ``max_total_tokens``, ``max_cost_usd`` and
    ``max_tool_calls`` count the whole run: a resume (after a pause, a crash
    or an error, by you or by `AgentRuntime.recover`) continues from what the
    journal says the run has already used. ``RunContext.limits`` and
    ``RunContext.remaining_budget()`` tell tools and hooks what applies.
    """

    max_turns: int | None = 30
    """Model requests per run, across resumes."""
    max_total_tokens: int | None = None
    """Input plus output tokens over the whole run, across resumes."""
    max_cost_usd: float | None = None
    """Needs a model that reports cost (QCloud does, from its catalog prices).
    A budget of zero or less stops the run before its first request. It fails
    closed: once a response arrives without a cost (a model without prices, an
    unreachable catalog), the run stops as if the budget were spent, because
    the spend can no longer be known (``usage.unpriced_requests`` counts them).
    A response that reports neither tokens nor a cost (what test models such as
    ``ScriptedModel`` report) counts as free, unless its model marks it unpriced:
    a model whose server sent no usage at all sets ``unpriced_requests=1``."""
    timeout_s: float | None = None
    """Wall-clock seconds for each ``run`` or ``resume`` call, counted from the
    run's start (after input guardrails and toolkit loading): the time a run
    spends paused does not count. When it expires, the request, compaction or
    tools in progress are cancelled, and no new ones start."""
    max_tool_calls: int | None = None
    """Tool calls per run that were allowed to run (executed or deferred);
    refused calls do not count. A call beyond it is not started: it gets an
    explanatory result, and the run ends with ``stop_reason="max_tool_calls"``."""
    max_depth: int | None = 5
    """How deeply runs may nest: a child run at a greater depth refuses to start
    (`MaxDepthExceeded`; an agent used as a tool reports it to the calling model
    as an error result). The lowest value along the chain of parent runs applies."""
    tool_timeout_s: float | None = 120.0
    """Seconds per tool call, unless the tool sets its own ``timeout``. An async
    tool is cancelled; a sync tool cannot be stopped, so its thread is abandoned
    and may still complete its work (and the interpreter waits for it at exit,
    see `qagents.tools.TOOL_THREADS`). Either way the model is told the outcome
    is unknown. Agents used as tools often need more: ``as_tool(timeout=...)``."""
    parallel_tools: Literal["read_only", "all", "none"] = "read_only"
    """Which tool calls of one turn run concurrently: only tools declared
    ``read_only`` (the rest run one at a time, in order), all, or none. A call
    that runs one at a time also waits, within its own timeout, for an earlier
    sync call of the run that timed out and still runs in its thread; if that
    thread is still busy then, the call is not run and the model is told so."""
    max_parallel_tools: int = 8
    max_tool_result_chars: int | None = 60_000
    """Longer tool results are shortened in the middle before the model sees them."""
    output_retries: int = 2
    """Extra attempts for a final answer that does not match ``output_type``, that
    an output validator rejects, or (``output_mode="tool"``) that is written as
    plain text instead of a call of the final-answer tool. When they are used
    up, the run raises `OutputValidationError` (it returns no result)."""
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
        lease_ttl_s: float = 30.0,
        build_id: str | None = None,
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
                sends the schema as ``response_format`` (a schema whose root is
                not an object, such as a ``Literal`` or a list, is sent wrapped
                as ``{"value": ...}``, as strict providers require; a bare
                value is accepted too); ``"prompted"`` writes it into the
                instructions; ``"tool"`` gives the model a final-answer tool
                whose arguments are the answer, for models that call tools
                more reliably than they follow a response format. Without an
                ``output_type`` the answer is text, and in ``"tool"`` mode a
                plain-text reply is accepted as well as a final-answer call.
            output_tool_name: Name of the final-answer tool (``output_mode="tool"``).
            output_strict: Ask the provider to enforce the schema exactly, where supported.
            output_validators: Functions ``(ctx, output)`` run in order on the
                validated answer, in every output mode. Return a value to replace
                the answer, ``None`` to keep it, or raise `ModelRetry` to ask
                the model for a corrected one. Sync or async.
            input_guardrails: Functions ``(ctx, message)`` run on the new prompt
                (a `Message`) before it joins the conversation. Raise
                `GuardrailTripped` to block the run. Only the ``prompt``
                argument is checked: messages passed in ``history=`` and
                messages already in the session are not, so pass the new
                user message as ``prompt``.
            output_guardrails: Functions ``(ctx, output)`` run on the accepted
                final answer. Raise `GuardrailTripped` to block it. They check
                the completed answer: its text was already streamed as
                ``text_delta`` events, which a block cannot recall, so a front
                end that must never show a blocked answer holds the text back
                until ``run_finished``.
            settings: Model settings for every request.
            limits: Hard stops; see `Limits`.
            permissions: Who decides whether a tool call runs; see `Permissions`.
            hooks: Your code at fixed points of the loop; see `Hooks`.
            handoffs: Agents this one may hand the conversation to. They are
                kept in the list ``agent.handoffs``, which you may change
                after building; agents that hand off to each other (a mesh)
                are connected that way:
                ``triage.handoffs.append(billing); billing.handoffs.append(triage)``.
                The list is read when a run starts and after each handoff.
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
        self.lease_ttl_s = lease_ttl_s
        self.build_id = build_id
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
        """Agents this one may hand the conversation to; change it to connect agents
        both ways. Names must stay unique among the agents reachable through handoffs."""
        self.compactor = compactor
        self._unsigned_approvals_warned: set[str] = set()
        """Effects tools already warned about: approval decisions that nothing signs."""
        self._check_effect_tools(item for item in self.tools if isinstance(item, Tool))
        for target in self.handoffs:
            if refusal := _effects_child_refusal(target, durable_session=False):
                raise ConfigurationError(refusal)
        self._output_adapter: TypeAdapter[Any] | None = None
        self._output_schema: dict[str, Any] | None = None
        self._native_schema: dict[str, Any] | None = None
        """The schema sent as ``response_format`` (``output_mode="native"``)."""
        self._native_wrapped = False
        """The native schema wraps a non-object answer as ``{"value": ...}``."""
        if self.output_type is not None:
            self._output_adapter = TypeAdapter(self.output_type)
            self._output_schema = self._output_adapter.json_schema()
            self._native_schema = self._output_schema
            if not _is_object_schema(self._output_schema):
                wrapper = create_model(
                    "output", __config__=ConfigDict(extra="forbid"), value=(self.output_type, ...)
                )
                self._native_schema = wrapper.model_json_schema()
                self._native_wrapped = True
        self._answer_adapter: TypeAdapter[Any] | None = None
        """Validates the arguments of a final-answer call (``output_mode="tool"``)."""
        self._answer_wrapped = False
        """The answer is the ``value`` argument, because its schema is not an object."""
        self._answer_parameters: dict[str, Any] = {}
        if output_mode == "tool":
            self._setup_answer_tool()

    # -- running ---------------------------------------------------------------

    async def settle(self, session: Session, answer: Settle, *, deps: Any = None) -> EffectRecord:
        """Settle a held operation using an attributed decision under the session lease."""
        return await _operator.settle(
            session,
            answer,
            effects=self._effect_map((await self._collect_tools()).values()),
            permissions=self.permissions,
            deps=MISSING if deps is None else deps,
            lease_ttl_s=self.lease_ttl_s,
            worker="qagents-settle",
        )

    async def recheck(self, session: Session, *, deps: Any = None) -> list[HeldEffect]:
        """Look up held effects under the session lease without sending them."""
        return await _operator.recheck(
            session,
            effects=self._effect_map((await self._collect_tools()).values()),
            permissions=self.permissions,
            deps=MISSING if deps is None else deps,
            lease_ttl_s=self.lease_ttl_s,
            worker="qagents-recheck",
        )

    def effects_by_name(self) -> dict[str, Effect]:
        """Return the declarations on statically known tools."""
        return self._effect_map(item for item in self.tools if isinstance(item, Tool))

    @staticmethod
    def _effect_map(tools: Iterable[Tool]) -> dict[str, Effect]:
        declarations = {}
        for tool in tools:
            for effect in tool.effects:
                if declarations.setdefault(effect.name, effect) is not effect:
                    raise ConfigurationError(
                        f"Two declarations use the effect name {effect.name!r}."
                    )
        return declarations

    def _check_effect_tools(
        self,
        tools: Iterable[Tool],
        *,
        limits: Limits | None = None,
        timing: _ports.Timing | None = None,
    ) -> None:
        limits = limits or self.limits
        timing = timing or _ports.Timing.for_lease(self.lease_ttl_s, 1.0, effects=True)
        effects_tools = [tool for tool in tools if tool.effects]
        for tool in effects_tools:
            timeout = tool.timeout if tool.timeout is not None else limits.tool_timeout_s
            for warning in _engine.check_declarations(
                tool.effects,
                timing=timing,
                tool_timeout_s=timeout,
            ):
                logger.warning("%s: %s", tool.name, warning)
        self._warn_unsigned_approvals(effects_tools)

    def _warn_unsigned_approvals(self, effects_tools: list[Tool]) -> None:
        """Warn, once per tool, about effects tools that ask for approval while no
        ``approval_secret`` signs the decisions (SPEC 5.1). D3 and D4 re-send only under a
        verifiable decision (5.4), so these effects are never re-sent automatically: where
        one of them would, an effect whose outcome is in doubt holds as
        APPROVAL_UNVERIFIABLE. Only tools with an effect that D3 or D4 can re-send are named:
        for the others the secret changes no hold (7.1 H1). With ``can_use_tool`` the build
        cannot tell which calls ask, so every such tool that no deny pattern refuses may ask
        and is named. The hold is right; the warning tells the developer before the first
        crash does."""
        permissions = self.permissions
        if permissions.signs_pauses:
            return
        per_call = permissions.can_use_tool is not None
        may_ask = permissions._callback_may_ask if per_call else permissions._asks_by_rule
        names = [
            tool.name
            for tool in effects_tools
            if tool.name not in self._unsigned_approvals_warned
            and _resends_automatically(tool)
            and may_ask(tool)
        ]
        if not names:
            return
        self._unsigned_approvals_warned.update(names)
        subject = "this effects tool" if len(names) == 1 else "these effects tools"
        if per_call:
            subject += (
                " may ask for approval (can_use_tool decides per call, so the build cannot tell)"
            )
        else:
            subject += " asks for approval" if len(names) == 1 else " ask for approval"
        logger.warning(
            "%s: %s, but Permissions has no approval_secret, so no decision is signed. Without "
            "a verifiable decision, rules D3 and D4 never re-send an effect automatically, in "
            "the run or after a crash: where one of them would, an effect whose outcome is in "
            "doubt, and that no lookup finds, is held as APPROVAL_UNVERIFIABLE until an "
            "operator settles it (agent.settle). Set Permissions(approval_secret=...) to sign "
            "approval decisions.",
            ", ".join(names),
            subject,
        )

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
        if_held_effects: Literal["raise", "continue"] = "raise",
    ) -> RunStream:
        """Start a run and return its event stream.

        Args:
            prompt: The new user message. May be left out when ``session`` or
                ``history`` already ends with one; input guardrails then do
                not run, because they check only this argument.
            deps: Your dependencies, passed to tools and hooks as ``ctx.deps``.
            session: Continues an existing conversation. A new in-memory
                session is created when it is left out. One run at a time per
                session: a run started while another run of the session is
                executing raises `SessionError` (on a store with leases, also
                when that run executes in another process).
            history: Seeds a new conversation with messages you kept yourself.
            limits: Replaces the agent's limits for this run.
            settings: Merged over the agent's model settings for this run,
                and checked here against the model of every agent reachable
                through handoffs, so a setting a handoff target's provider
                would reject fails before anything runs (with that model's error).
            state: Values merged into the session state (``ctx.state``) when
                the run starts, key by key. The merge is journaled.
            parent: The context of the run (usually a tool call) that starts
                this one. The new run is its child: ``ctx.parent_run_id``,
                ``ctx.parent_tool_call_id`` and ``ctx.depth`` link the two, the
                parent's remaining token, cost and time budget caps this run
                (tokens and cost live, shared with the parent's other child
                runs), the parent's hooks with ``propagate=True`` apply here
                too, and the parent's ``on_ask`` answers approvals when this
                agent's permissions have none. This run's usage is added to
                the parent run when it ends, however it ends (completed,
                failed, cancelled); ``parent.add_usage(result.usage)``, which
                code written for 0.2 calls, is recognised and not counted twice.

        Raises:
            MaxDepthExceeded: The child run would nest deeper than
                ``Limits.max_depth`` (a `ConfigurationError`).
            ConfigurationError: Two different agents reachable through
                handoffs share a name.
            SessionError: ``state`` holds a value the journal cannot store
                (raised here, before anything is journaled), or the session
                already has a run in progress (raised by the stream).
        """
        return RunStream(
            _Run(
                self, prompt=prompt, deps=deps, session=session, history=history,
                limits=limits, settings=settings, approvals=None, results=None,
                state=state, parent=parent, resume=False, if_held_effects=if_held_effects,
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
        if_held_effects: Literal["raise", "continue"] = "raise",
    ) -> RunResult:
        """Run until the agent answers, pauses, hits a limit or is blocked.

        Takes the same arguments as `stream`. A run started on a session that
        is paused closes the waiting tool calls with an explanatory result, so
        the conversation stays valid; use `resume` to continue it instead.
        """
        stream = self.stream(
            prompt, deps=deps, session=session, history=history, limits=limits,
            settings=settings, state=state, parent=parent, if_held_effects=if_held_effects,
        )  # fmt: skip
        return await stream.wait()

    def run_sync(
        self,
        prompt: Prompt = None,
        *,
        if_held_effects: Literal["raise", "continue"] = "raise",
        **options: Any,
    ) -> RunResult:
        """``run`` for code that is not async. Do not call it from a running event loop."""
        return asyncio.run(self.run(prompt, if_held_effects=if_held_effects, **options))

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

        Call ``resume`` on the agent you called ``run`` on (the one that owns
        the run's limits, permissions and hooks), or on any agent from which
        the agent active in the run can be reached through handoffs: the
        journal records agents by ``name``, so names must be unique among the
        agents reachable through handoffs. The run continues with the agent
        that was active when it stopped. A new ``agent.run`` on the session
        always starts with the agent you call.

        The run keeps counting from what it used before it stopped: turns,
        tokens, cost and tool calls (see `Limits`), and ``RunResult.usage``
        and ``turns`` cover the whole run.

        Args:
            session: The session whose run is unfinished.
            approvals: Maps tool call ids from ``result.pending_approvals`` to
                ``True``/``False`` or an `Approval`; any other value raises
                `ConfigurationError`. An approval answers only an open call
                of the paused turn (others are ignored), and the permission
                rules are checked again first: a deny rule refuses the call
                whatever the approval says. Applied approvals are journaled
                (``approval_decided``) before the call runs, for the record:
                a journaled approval is never taken as permission later, so
                when a run stops while an approved call runs, the next resume
                asks again for an idempotent or read-only call and reports
                any other call's outcome as unknown.
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
            SessionError: The session has no unfinished run, ``results``
                names a call that waits for no result, the run is already
                executing (another resume, or a run in another process that
                holds the session's lease), or the agent active in the run
                is not reachable from this one.
            ApprovalTokenError: Pauses are signed and an answer has no token or
                the wrong one, or the paused call in the journal was changed.
            ConfigurationError: An ``approvals`` value is not ``True``,
                ``False`` or an `Approval`.
        """
        stream = self.resume_stream(
            session, approvals=approvals, results=results, deps=deps, limits=limits,
            settings=settings, state=state,
        )  # fmt: skip
        return await stream.wait()

    # -- composition -------------------------------------------------------------

    def as_tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        read_only: bool = False,
        timeout: float | None = None,
    ) -> Tool:
        """Expose this agent as a tool another agent can call.

        The sub-agent works in its own fresh conversation: it sees only the
        task text, and the caller sees only its accepted final answer
        (``result.output``: after output validators, and the final-answer
        call's value in ``output_mode="tool"``). Each call is a child run of
        the caller's (``parent=ctx``): it is linked to the calling tool call,
        shares the caller's remaining budget, inherits the caller's
        propagating hooks and ``on_ask``, and refuses to start beyond
        ``Limits.max_depth``. Its usage is added to the caller's run (also
        when it fails or is cancelled) and its events are forwarded as
        `SubagentEvent`, ``run_started`` included (``run_finished`` is not:
        its result is the tool result). A sub-agent run that pauses, is
        blocked, stops at a limit, is refused by the provider or would nest
        too deep is reported to the caller as an error result.

        Args:
            name: Tool name; defaults to the agent's name.
            description: Tool description; defaults to the agent's description,
                or ``prompts.SUBAGENT_DESCRIPTION``.
            read_only: Declare the sub-agent free of side effects, so calls may
                run in parallel.
            timeout: Seconds for one call of the sub-agent; ``None`` uses the
                caller's ``Limits.tool_timeout_s`` (120 s by default), which
                is often too short for an agent.
        """
        agent = self
        if refusal := _effects_child_refusal(agent, durable_session=False):
            raise ConfigurationError(refusal)

        async def delegate(ctx: RunContext[Any], task: str) -> str:
            try:
                stream = agent.stream(task, deps=ctx.deps, parent=ctx)
            except MaxDepthExceeded as error:
                raise ToolError(
                    prompts.SUBAGENT_TOO_DEEP.format(
                        name=agent.name, depth=error.depth, limit=error.limit
                    )
                ) from error
            async with stream:
                async for event in stream:
                    if not isinstance(event, RunFinished):
                        ctx.emit(
                            SubagentEvent(
                                agent=agent.name, tool_call_id=ctx.tool_call_id, event=event
                            )
                        )
            result = stream.result
            assert result is not None  # its usage reached the caller's run as it ended
            if result.status != "completed" or result.stop_reason == "content_filter":
                raise ToolError(
                    prompts.SUBAGENT_STOPPED.format(
                        name=agent.name, reason=result.stop_reason or result.status
                    )
                )
            if result.output is None and agent.output_type is not None:
                return "null"  # an answer of None, never the "OK" of a tool that returns nothing
            return render_tool_output(result.output)

        built = make_tool(
            delegate,
            name=name or _tool_safe(self.name),
            description=description
            or self.description
            or prompts.SUBAGENT_DESCRIPTION.format(name=self.name),
            read_only=read_only,
            timeout=timeout,
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
            "lease_ttl_s": self.lease_ttl_s,
            "build_id": self.build_id,
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
        if _is_object_schema(schema):
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
    Close a stream you stop reading early (``async with stream:`` or
    ``await stream.aclose()``): until then its run counts as in progress, and
    the session refuses another run.
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
        """Only the text of the answer, as it is written.

        The text is streamed before the answer is checked: when an output
        guardrail blocks it, the text has already been yielded. Check
        ``stream.result.status`` afterwards, or use ``agent.run`` when a
        blocked answer must never be shown.
        """
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


_DONE = object()
"""The last item every tool task puts on its group's queue."""


class _Rejected(Exception):
    """A final answer the model must correct.

    Args:
        message: What is wrong, for the model.
        problem: How the answer failed, for `OutputValidationError`.
    """

    def __init__(self, message: str, problem: str, *, by_validator: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.problem = problem
        self.by_validator = by_validator
        """An output validator raised `ModelRetry` (the answer itself was valid)."""


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
        if_held_effects: Literal["raise", "continue"] = "raise",
    ) -> None:
        self.root = agent
        self.agent = agent
        self.run_id = uuid.uuid4().hex
        self.session = session or Session()
        self._session_given = session is not None
        if if_held_effects not in {"raise", "continue"}:
            raise ConfigurationError("if_held_effects must be 'raise' or 'continue'.")
        self.if_held_effects = if_held_effects
        self.deps = deps
        self.prompt = prompt
        self.seed = list(history) if history is not None else []
        self.limits = limits or agent.limits
        self._limits_given = limits is not None
        if settings is not None:
            # Every agent the run may hand off to gets these settings: check them all now,
            # before anything runs, not in the middle of the run after a handoff.
            for target in _reachable_agents(agent):
                target.model.validate_settings(target.settings.merged(settings))
        self.settings = settings
        self.parent = parent
        """The context of the tool call that started this run, for a child run."""
        self.parent_run: _Run | None = (
            parent._run if parent is not None and isinstance(parent._run, _Run) else None
        )
        self.depth = parent.depth + 1 if parent is not None else 0
        if self.parent_run is not None:  # the lowest bound along the chain of parent runs
            inherited, own = self.parent_run.limits.max_depth, self.limits.max_depth
            if inherited is not None and (own is None or inherited < own):
                self.limits = self.limits.model_copy(update={"max_depth": inherited})
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
        self.approvals: dict[str, Approval] = {}
        """Answers given to ``resume``; only those for open calls of the paused turn are kept."""
        for call_id, answer in (approvals or {}).items():
            if isinstance(answer, Approval):
                self.approvals[call_id] = answer
            elif answer is True or answer is False:  # never bool(): "false" must not approve
                self.approvals[call_id] = Approval(approved=answer)
            else:
                raise ConfigurationError(
                    f"approvals[{call_id!r}] is a {type(answer).__name__}. Give True, False or "
                    "an Approval; nothing was run."
                )

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
        _state_json(self._state_seed)  # a value the journal cannot store fails before anything runs
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
        self._counted: dict[str, None] = {}
        """Tool calls this run allowed to run (executed or deferred), for
        ``Limits.max_tool_calls``; restored from the journal on resume, so a call is
        counted once."""
        self._usage_to_journal = Usage()
        """Usage added through ``ctx.add_usage`` that no journal entry carries yet."""
        self._usage_by_operation: dict[str, Usage] = {}
        """The part of `_usage_to_journal` that tool calls added, by operation id: the
        entry that carries it records it as ``added_usage_by_operation``."""
        self.journaled_usage: dict[str, Usage] = {}
        """What each tool call of the run added through ``ctx.add_usage`` that the journal
        holds, by operation id: restored on resume, and kept as entries are written
        (``RunContext.journaled_usage``)."""
        self._edited: set[str] = set()
        """Calls that run with arguments an approver changed."""
        self._tool_limit_hit = False
        self._deadline: float | None = None
        """``time.monotonic()`` at which ``timeout_s`` stops the run."""
        self.status: RunStatus = "completed"
        self.stop_reason: str | None = None
        self.guardrail: TrippedGuardrail | None = None
        self._journal_open = False
        self._closed = False
        """``run_finished`` or ``run_paused`` is journaled, so an error writes no ``run_failed``."""
        self._finishing = False
        """The run is writing its ``run_finished`` or ``run_paused``: calls from outside the
        model (``RunContext.call_tool``) journal nothing more from here on."""
        self._stopped = False
        """The run has stopped executing, however it ended: its stream was closed, it was
        cancelled from outside, it failed or it finished. From then on nothing more may be
        journaled for it by a context that outlived it (calls from outside the model,
        ``ctx.checkpoint``, ``ctx.save_state``): its session claim is released, and a
        resume elsewhere may own the journal."""
        self._driver: tuple[asyncio.Task[Any], int] | None = None
        """The task that drives the run now (it iterates the run's events), with its
        ``cancelling()`` count when it took the step: a higher count means the run is being
        cancelled from outside (``asyncio.timeout``, a task cancel) and is unwinding."""
        self._call_tasks: dict[str, tuple[asyncio.Task[Any], int]] = {}
        """The task each running call runs in (one of the model's calls, or one from outside
        the model), marked as `_driver` is: a call whose task is being cancelled (its timeout,
        its group's cancel) is ending, and serves no more calls from outside the model."""
        self._hooks_started = False
        """``on_run_start`` was called, so ``on_run_end`` is owed exactly once."""
        self._ended = False
        """``on_run_end`` was called with the result."""
        self._hook_error: Exception | None = None
        """What an ``on_event`` hook raised: the run failed with it."""
        self._batch: ContextVar[tuple[asyncio.Task[Any] | None, list[Entry]] | None] = ContextVar(
            "qagents_journal_batch", default=None
        )
        """Entries collected by `_batched`, owned by the task that opened the block."""
        self._start_events: list[Event] = []
        """Events of journal entries written while the run was prepared."""
        self._claimed: tuple[int, str] | None = None
        self._writer: _session.LeasedWriter | None = None
        self._owns_writer = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._effects: dict[str, Effect] = {}
        self._replay_state = SessionState()
        self._held_calls: dict[str, HeldEffect] = {}
        self._scopes: dict[str, _ports.CallScope] = {}
        self._authorizations: dict[str, _ports.Authorization] = {}
        self._unknown_results: set[str] = set()
        self._committed_state: dict[str, Any] = {}
        self._legacy_pause_tokens = False
        self._lease_task: asyncio.Task[None] | None = None
        self._cancel = asyncio.Event()
        self._cancel_reason: str | None = None
        self._interruptible: asyncio.Task[Any] | None = None
        """The task awaiting a step that a cancel interrupts (see `_unless_cancelled`)."""
        self._interrupting = False
        """``cancel()`` cancelled ``_interruptible``; the CancelledError it gets is ours."""
        self._running: set[str] = set()
        """Calls whose ``tool_started`` this run journaled and that have no result yet."""
        self._call_emit: dict[str, Callable[[Any], None]] = {}
        """Where the events of each call in ``_running`` go (its group's queue): calls from
        outside the model that come through it report there too (`_outside_call`)."""
        self._outside_running: dict[str, Callable[[Any], None]] = {}
        """Calls from outside the model (``RunContext.call_tool``) that started and have no
        result yet, with where their events go."""
        self._outside_gates: weakref.WeakValueDictionary[str, _OutsideGate] = (
            weakref.WeakValueDictionary()
        )
        """``Limits.parallel_tools`` for the calls from outside the model, per calling call."""
        self._outside_failure: Exception | None = None
        """What a hook or the journal raised during a call from outside the model: the run
        fails with it (see `_fail_from_outside`)."""
        self.checkpoints: dict[str, dict[str, Any]] = {}
        """Values tool calls journaled with ``ctx.checkpoint``, by operation id, then key;
        restored from the journal for the open turn, so a repeated call finds them."""
        self._children: list[_Run] = []
        """Child runs that are working: their usage counts against this run's budget."""
        self._absorbed: list[Usage] = []
        """``RunResult.usage`` of child runs already added, so adding it again is a no-op."""
        self._result_usage: Usage | None = None
        self._usage_lock = threading.Lock()
        """``ctx.add_usage`` may be called from a sync tool's worker thread."""
        self._stuck_threads: list[concurrent.futures.Future[Any]] = []
        """Threads of sync calls that run one at a time, timed out or were cancelled, and
        still run: the next such call waits for them (`_invoke`)."""

    def cancel(self, reason: str = "cancelled") -> None:
        if not self._cancel.is_set():
            self._cancel_reason = reason
            self._cancel.set()
            task = self._interruptible
            try:
                current = asyncio.current_task()
            except RuntimeError:  # no running loop here
                current = None
            if task is not None and task is not current and not task.done():
                self._interrupting = True
                task.cancel()  # interrupt the model request or compaction in progress

    # -- the loop -----------------------------------------------------------------

    async def execute(self) -> AsyncIterator[Event]:
        async with contextlib.aclosing(self._execute()) as events:
            while True:
                self._driver = _cancel_mark(asyncio.current_task())  # who drives this step
                try:
                    event = await anext(events)
                except StopAsyncIteration:
                    break
                try:
                    await self.hooks.on_event(self._ctx(), event)
                except Exception as error:
                    # Closing _execute below reports this error as the run's failure.
                    self._hook_error = error
                    self._stopped = True  # the run fails: it takes nothing more from outside
                    raise
                try:
                    yield event
                except GeneratorExit:
                    # The stream is closed early (aclose, or the generator is collected):
                    # the run stops here. Set before _execute unwinds, so a call from
                    # outside the model that arrives while its tools wind down is refused.
                    self._stopped = True
                    raise

    async def _execute(self) -> AsyncIterator[Event]:
        timer: asyncio.TimerHandle | None = None
        result: RunResult | None = None
        try:
            self._loop = asyncio.get_running_loop()
            self._inherit_limits()
            if self.parent_run is not None:  # its spend counts against the parent's budget, live
                self.parent_run._children.append(self)
            await self._claim()
            await self._prepare()
            yield RunStarted(
                run_id=self.run_id,
                session_id=self.session.id,
                agent=self.agent.name,
                resumed=self.resume,
                parent_run_id=self.parent.run_id if self.parent is not None else None,
                depth=self.depth,
            )
            for event in self._start_events:  # journaled with the start of the run
                yield event
            if (updated := await self._save_state()) is not None:
                yield updated  # the values passed as state= to a resume
            self._hooks_started = True
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

            for _ in range(2):  # the second pass saves what on_event did with the first
                if (updated := await self._save_state()) is None:
                    break
                yield updated  # changed by on_event, on_run_start or a guardrail
            self._finishing = True  # from here on the journal belongs to the run's last entry
            result = self._result()
            if self.status == "paused":
                await self._journal(
                    "run_paused",
                    {
                        "pending": [item.model_dump(mode="json") for item in self.pending],
                        "inputs": [item.model_dump(mode="json") for item in self.inputs],
                    },
                )
            else:
                await self._journal("run_finished", _finished_data(result))
            self._closed = True
            self._ended = True  # set first: an on_run_end that raises is not reported twice
            await self.hooks.on_run_end(self._ctx(), result, None)
        except Exception as error:
            self._stopped = True  # before any await: nothing outlives the run's last entry
            if self._ended:
                raise
            # The run stays open in the journal, so agent.resume() can pick it up.
            await self._fail(error)
            raise
        except BaseException as error:
            self._stopped = True  # before any await: a call from outside is refused from here
            if self._ended:
                raise
            if isinstance(error, GeneratorExit) and self._hook_error is not None:
                await self._fail(self._hook_error)  # an on_event hook raised: the run failed
            else:
                # Cancelled, or the stream was closed early: the run stays open in the journal
                # (nothing is written). on_run_end still runs, so tracers close spans.
                await self._report_end(error)
            raise
        finally:
            self._stopped = True  # the claim goes next: from here the journal is not the run's
            if timer is not None:
                timer.cancel()
            try:
                await self._hand_usage_to_parent()
            finally:
                await self._unclaim()
        # Outside the try: a consumer that stops at this event does not stop a finished run.
        yield RunFinished(result=result)

    async def _hand_usage_to_parent(self) -> None:
        """Add and journal child usage in its own parent entry, however the child ended."""
        parent = self.parent_run
        if parent is None or self not in parent._children:
            return
        parent._children.remove(self)  # in the same step as the add: never counted twice or 0
        if self._result_usage is not None:
            parent._absorbed.append(self._result_usage)  # parent.add_usage(result.usage): no-op
        if self.usage == Usage():
            return
        with parent._usage_lock:
            parent.usage = parent.usage.add(self.usage)
        if parent._journal_closed():
            return  # a detached child cannot write after its parent's run closed
        data: dict[str, Any] = {
            "child_run_id": self.run_id,
            "tool_call_id": self.parent.tool_call_id if self.parent is not None else None,
            "added_usage": self.usage.model_dump(mode="json"),
        }
        if self.parent is not None and self.parent.operation_id is not None:
            data["added_usage_by_operation"] = _usage_dumps({self.parent.operation_id: self.usage})
        await parent._journal("child_usage", data)

    async def _fail(self, error: BaseException) -> None:
        """Journal ``run_failed`` for an open run that raised, then report it to the hooks."""
        if self._journal_open and not self._closed:
            with contextlib.suppress(Exception):
                data = _failure_data(error)
                if isinstance(error, HeldEffectsError) and error.retry_after is not None:
                    data["retry_after"] = error.retry_after.timestamp()
                await self._journal("run_failed", data)
        await self._report_end(error)

    async def _report_end(self, error: BaseException) -> None:
        if not self._hooks_started:  # on_run_end pairs with on_run_start
            return
        try:
            await self.hooks.on_run_end(self._ctx(), None, error)
        except Exception:
            logger.exception("on_run_end raised while reporting a run that did not finish")

    # -- one run per session ---------------------------------------------------------

    async def _claim(self) -> None:
        """Make this the session's only executing run, before the journal is read.

        In this process a registry refuses a second run of the session. On a
        store with leases the run also holds the session's lease while it
        executes, and every entry is written only while it holds it, so a run
        in another process is refused too, and a run that lost its lease
        cannot write. Under `AgentRuntime` the runtime's lease does both. A
        store subclass that overrides ``append`` but not ``append_leased`` is
        written through its ``append`` without a lease.
        """
        await self.session._resolve()
        store, session_id = self.session.store, self.session.id
        root = _store_root(store)
        key = (id(root), session_id)
        live = _LIVE_RUNS.get(key)
        if live is not None:
            raise SessionError(
                f"Session {session_id!r} already has a run in progress in this process (run "
                f"{live[1].run_id}). A session runs one run at a time: wait for it to finish or "
                "cancel it. A stream you stopped reading early must be closed first "
                "(`async with stream:` or `await stream.aclose()`)."
            )
        _LIVE_RUNS[key] = (root, self)
        self._claimed = key
        if _fenced_by_runtime(store):
            self._writer = getattr(store, "writer", None)
            return
        if _append_overridden_alone(store):
            return
        ttl = self.root.lease_ttl_s
        if not self.root.effects_by_name() and ttl == 30:
            ttl = RUN_LEASE_TTL_S
        writer = _session.LeasedWriter(
            store,
            session_id,
            worker="qagents-run",
            ttl_s=ttl,
            purpose="recovery" if self.resume else "run",
            journal_acquisition=True if self.root.effects_by_name() else None,
        )
        if not await writer.open():
            raise SessionError(
                f"Session {session_id!r} has a run in progress elsewhere: another process or "
                "worker holds its lease. Wait for that run to stop. If its process died, the "
                f"lease expires within {ttl:g} seconds (AgentRuntime.recover then "
                "takes the run over)."
            )
        self._writer = writer
        self._owns_writer = True
        if writer.leased:
            self._lease_task = asyncio.create_task(
                writer.keep(on_lost=lambda _: self.cancel("lease_lost"))
            )

    async def _unclaim(self) -> None:
        try:
            if self._lease_task is not None:
                self._lease_task.cancel()
                self._lease_task = None
            if self._owns_writer and self._writer is not None:
                try:
                    await self._writer.close()
                except Exception:
                    logger.warning("Could not release the lease of run %s; it expires", self.run_id)
        finally:
            if self._claimed is not None and _LIVE_RUNS.get(self._claimed, (None, None))[1] is self:
                del _LIVE_RUNS[self._claimed]
            self._claimed = None

    async def _steps(self) -> AsyncIterator[Event]:
        """Advance the conversation until the run answers, pauses or stops.

        Sets ``status`` and ``stop_reason``; ``status`` stays ``"completed"``
        when the run answered.
        """
        while True:
            if self._outside_failure is not None:  # a hook or the journal failed in call_tool
                raise self._outside_failure
            if self._answered:  # a final-answer call was accepted, now or before a resume
                return
            if self._cancel.is_set():
                # Calls that never started say so; started ones are cancelled, or, when
                # repeating them is not safe, of unknown outcome.
                for event in await self._settle_open_calls(prompts.TOOL_NOT_EXECUTED):
                    yield event
                if self._held_calls:
                    self.status, self.stop_reason = "paused", "effect_held"
                    self.inputs = list(self.waiting.values())
                    return
                if self._cancel_reason == "timeout":
                    self.status, self.stop_reason = "limit_reached", "timeout"
                else:
                    self.status, self.stop_reason = "cancelled", self._cancel_reason
                return
            last = self.history[-1]
            if self._refused:  # the provider refused: end the run, keep its words
                for event in await self._settle_open_calls(prompts.TOOL_NOT_EXECUTED):
                    yield event
                refusal = next(m for m in reversed(self.history) if m.role == "assistant")
                self.output = refusal.text if self.agent._output_adapter is None else None
                self.stop_reason = "content_filter"
                return
            if last.role == "assistant" and last.tool_calls:
                async with contextlib.aclosing(self._tool_phase(last)) as phase:
                    async for event in phase:
                        yield event
                if self._cancel.is_set():
                    continue  # the top of the loop settles the calls and reports the stop
                if self.pending or self.inputs:
                    self.status = "paused"
                    if any(item.held_record_id for item in self.inputs):
                        self.stop_reason = "effect_held"
                    return
                if self._tool_limit_hit:
                    self.status, self.stop_reason = "limit_reached", "max_tool_calls"
                    return
                continue
            if last.role == "assistant":
                outcome = await self._finalize(last)
                if outcome is True:
                    truncated = self._last_response is not None and (
                        self._last_response.finish_reason == "length"
                    )
                    self.stop_reason = "max_output_tokens" if truncated else None
                    return
                if isinstance(outcome, OutputRejected):
                    yield outcome
                continue
            stop_reason = self._limit_reached()
            if stop_reason is None:
                if not await self._maybe_compact():
                    continue  # cancelled while compacting: the top of the loop stops the run
                stop_reason = self._limit_reached()  # a summary request may use up the budget
            if stop_reason is not None:
                self.status, self.stop_reason = "limit_reached", stop_reason
                return
            self.turn += 1
            yield TurnStarted(turn=self.turn, agent=self.agent.name)
            async with contextlib.aclosing(self._model_phase()) as phase:
                async for event in phase:
                    yield event

    def _check_depth(self) -> None:
        """Refuse to start a child run nested deeper than ``Limits.max_depth``.

        ``self.limits.max_depth`` already holds the lowest bound of the parent runs.
        """
        limit = self.limits.max_depth
        if self.parent is None or limit is None or self.depth <= limit:
            return
        raise MaxDepthExceeded(
            f"Agent {self.root.name!r} cannot start a child run at depth {self.depth}: "
            f"Limits.max_depth is {limit}. Agents that delegate to each other need a "
            "point where they stop; raise Limits(max_depth=...) if this nesting is intended.",
            depth=self.depth,
            limit=limit,
        )

    def _inherit_limits(self) -> None:
        """Cap a child run's limits by what remains of its parents' budgets, at this moment.

        Tokens and cost are also checked live while the child runs (`_limit_reached`).
        """
        if self.parent_run is None:
            return
        changes: dict[str, Any] = {}
        for key, remaining in self.parent_run._remaining().items():
            own = getattr(self.limits, key)
            if remaining is not None and (own is None or remaining < own):
                changes[key] = max(remaining, 0)
        if changes:
            self.limits = self.limits.model_copy(update=changes)

    def _spent(self) -> Usage:
        """This run's usage plus the live usage of its child runs that are still working."""
        total = self.usage
        for child in list(self._children):
            total = total.add(child._spent())
        return total

    def _remaining(self) -> dict[str, Any]:
        """What is left of the token, cost and time budget of this run and its parent runs
        (the lowest of them; ``None``: no limit), counting child runs that are working."""
        limits, spent = self.limits, self._spent()
        remaining: dict[str, Any] = {"max_total_tokens": None, "max_cost_usd": None}
        if limits.max_total_tokens is not None:
            remaining["max_total_tokens"] = limits.max_total_tokens - spent.total_tokens
        if limits.max_cost_usd is not None:
            unknown = not spent.cost_complete  # an unknown cost counts as all of the budget
            cost = limits.max_cost_usd if unknown else spent.cost_usd or 0.0
            remaining["max_cost_usd"] = limits.max_cost_usd - cost
        remaining["timeout_s"] = (
            None if self._deadline is None else self._deadline - self._loop.time()
        )
        if self.parent_run is not None:
            for key, value in self.parent_run._remaining().items():
                if value is not None and (remaining[key] is None or value < remaining[key]):
                    remaining[key] = value
        return remaining

    def _start_timer(self) -> asyncio.TimerHandle | None:
        timeout = self.limits.timeout_s
        if timeout is None:
            return None
        self._deadline = self._loop.time() + timeout
        if timeout <= 0:  # a child whose parent has no time left
            self.cancel("timeout")
            return None
        return self._loop.call_later(timeout, self.cancel, "timeout")

    async def _prepare(self) -> None:
        if self._writer is None:
            state = await self.session.state()
        else:
            state = _session.derive_state(await self._writer.load(), session_id=self.session.id)
        self._replay_state = state
        self._committed_state = copying.deepcopy(state.values)
        self._held_calls = {
            item.tool_call_id: item
            for item in state.effects.held(self.session.id)
            if item.tool_call_id is not None
        }
        self.history = list(state.messages)
        self.results = dict(state.tool_results)
        self.started = set(state.started_tools)
        self.waiting = {item.tool_call_id: item for item in state.pending_inputs}
        self.checkpoints = {key: dict(value) for key, value in state.checkpoints.items()}
        self.journaled_usage = dict(state.operation_usage)
        self._saved_state = _state_json(state.values)
        self.state.update(state.values)
        self.state.update(self._state_seed)  # journaled with the start of the run
        if self.resume:
            await self._prepare_resume(state)
            return
        held = state.effects.held(self.session.id)
        if held and self.if_held_effects == "raise":
            raise HeldEffectsError(held)
        self.checkpoints = {}  # a new run repeats no call of the run it closes
        self.journaled_usage = {}

        if self.seed and self.history:
            raise ConfigurationError(
                "history= seeds a new conversation, but this session already has messages. "
                "Pass one or the other."
            )
        if not (self.history or self.seed or self.prompt is not None):
            raise ConfigurationError("Nothing to run: give a prompt, a history or a session.")
        self._check_agent_names()
        await self._load_agent()
        started: dict[str, Any] = {"agent": self.root.name}
        if self._effects:
            started.update(self._effects_run_fields())
        if held:
            started["kept_held"] = [item.id for item in held]
        if self.parent is not None:
            started.update(
                parent_run_id=self.parent.run_id,
                parent_tool_call_id=self.parent.tool_call_id,
                depth=self.depth,
            )
        if self._limits_given:  # so a recovery elsewhere (AgentRuntime) keeps them
            started["limits"] = self.limits.model_dump(mode="json")
        # The start of a run is one store write: a crash leaves either all of it or nothing,
        # and a guardrail that raises leaves nothing (the previous run stays as it was).
        async with self._batched():
            if state.open_run_id is not None:
                await self._journal("run_finished", dict(_ABANDONED), run_id=state.open_run_id)
            await self._journal("run_started", started)
            if (updated := await self._save_state()) is not None:
                self._start_events.append(updated)  # the values passed as state=
            for message in self.seed:
                await self._append(message, new=False)
            # A previous run may have stopped with tool calls still open (crashed,
            # or paused for approvals and inputs). Close them so the conversation
            # stays valid for the provider.
            await self._settle_open_calls(prompts.TOOL_NOT_EXECUTED, replacing=True)
            self._held_calls.clear()
            if self.prompt is not None:
                message = _as_message(self.prompt)
                # A blocked prompt never joins the conversation; the run ends as "blocked".
                self.guardrail = await self._check_guardrails("input", message)
                if self.guardrail is None:
                    await self._append(message)
        self._journal_open = True

    async def _prepare_resume(self, state: SessionState) -> None:
        if state.open_run_id is None or not self.history:
            raise SessionError(
                f"Session {self.session.id!r} has no unfinished run to resume. "
                "Start a new one with agent.run()."
            )
        self.run_id = state.open_run_id
        requires = set(state.run_requires.get(self.run_id, ()))
        self._legacy_pause_tokens = _kinds.REQUIRES_APPROVALS not in requires and not any(
            "approval_id" in data for data in state.decisions.values()
        )
        if unsupported := requires - _kinds.SUPPORTED_REQUIRES:
            raise ConfigurationError(
                f"This run requires unsupported features: {sorted(unsupported)}."
            )
        if _kinds.REQUIRES_EFFECTS in requires and not (self._writer and self._writer.leased):
            raise ConfigurationError("Resuming Effects requires a fenced journal writer.")
        self._ordinal = state.run_messages
        paused = bool(state.pending_approvals or state.pending_inputs)
        try:
            self._check_agent_names()
            if state.agent:
                self.agent = self._find_agent(state.agent)
            await self._load_agent()
        except Exception:
            # The open run cannot be resumed here (a toolkit that cannot connect, an agent
            # that is gone): count it as a failure, so AgentRuntime's max_failures applies.
            # A paused run stays paused: it waits for a person, not for recovery.
            self._journal_open = not paused
            raise
        self._check_results(state)
        last = self.history[-1]
        open_calls = (
            {call.id for call in last.tool_calls if call.id not in self.results}
            if last.role == "assistant"
            else set()
        )
        # An approval answers an open call of the paused turn and nothing else.
        self.approvals = {key: value for key, value in self.approvals.items() if key in open_calls}
        await self._check_redelivered_approvals(state)
        self._check_tokens(state)
        self._journal_open = True
        if self._effects:
            # A run can be paused with a legacy token. Verify that token before
            # upgrading the journal and use v3 for every subsequent ask.
            await self._journal(
                _kinds.KIND_RUN_RESUMED,
                {**self._effects_run_fields(), "worker": self._writer.worker},
            )
            self._legacy_pause_tokens = False
        self.usage, self.turn = state.run_usage, state.run_turns
        # A call that waited for approval has not been allowed to run yet: it is counted
        # when an approval lets it run (journals list paused calls among the counted ones).
        # One that started after its approval did run (or may have), so it still counts.
        waiting = {item.tool_call_id for item in state.pending_approvals}
        waiting -= set(state.started_tools)
        self._counted = dict.fromkeys(i for i in state.run_tool_call_ids if i not in waiting)
        if state.run_messages == 0 and last.role == "assistant" and not last.tool_calls:
            # The run started, but its prompt never reached the journal (written by an older
            # version, in separate writes). Finishing it would answer with the previous reply.
            await self._journal("run_finished", dict(_ABANDONED))
            self._closed = True
            raise SessionError(
                f"The open run of session {self.session.id!r} has no prompt in the journal, "
                "so it cannot be resumed; it was closed as abandoned. Start a new run."
            )
        self._restore_finish_reason()
        await self._restore_handoff()
        await self._restore_answer()

    def _check_agent_names(self) -> None:
        """Refuse two different agents with one name among those reachable through handoffs.

        The journal records the active agent by name, so a resume must find exactly one.
        """
        names: dict[str, Agent] = {}
        seen: set[int] = set()
        queue = [self.root]
        while queue:
            agent = queue.pop(0)
            if id(agent) in seen:
                continue
            seen.add(id(agent))
            if names.setdefault(agent.name, agent) is not agent:
                raise ConfigurationError(
                    f"Two different agents reachable through handoffs from {self.root.name!r} "
                    f"are named {agent.name!r}. Runs are recorded by agent.name, so give each "
                    "agent a unique name=."
                )
            tools: dict[str, Agent] = {}
            for target in agent.handoffs:
                if tools.setdefault(target._handoff_tool_name, target) is not target:
                    raise ConfigurationError(
                        f"Agent {agent.name!r} has two handoffs named "
                        f"{target._handoff_tool_name!r}. Give the target agents names that "
                        "differ in letters, digits, '_' or '-'."
                    )
            queue.extend(agent.handoffs)

    async def _load_agent(self) -> None:
        self.tools = await self.agent._collect_tools()
        self._effects = self.agent._effect_map(self.tools.values())
        if self._effects:
            if not self._session_given or self.agent is not self.root:
                raise ConfigurationError("Effects tools require an explicit, separate session.")
            if self._writer is None or not self._writer.leased:
                raise ConfigurationError("Effects tools require a fenced session writer.")
            timing = _ports.Timing.for_lease(
                self._writer.lease.ttl_s,
                self._writer.store.takeover_grace_s,
                effects=True,
            )
            self.agent._check_effect_tools(self.tools.values(), limits=self.limits, timing=timing)
            self._writer._journal_acquisition = True
            self._writer.timing = self._writer.lease.timing = timing
            await _engine.check_run_start(
                list(self._effects.values()),
                store=self._writer.store,
                deps=MISSING if self.deps is None else self.deps,
                session_id=self.session.id,
            )
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

    def _effects_run_fields(self) -> dict[str, Any]:
        return {
            "requires": [_kinds.REQUIRES_EFFECTS, _kinds.REQUIRES_APPROVALS],
            "agent_config_digest": _canonical.digest(
                {
                    "agent": self.agent.name,
                    "tools": sorted(tool_spec_digest(tool) for tool in self.tools.values()),
                }
            ),
            "build_id": self.agent.build_id,
            "epoch": self._writer.lease.epoch,
        }

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

    def _restore_finish_reason(self) -> None:
        """On resume, take back how the last model response ended (it is in its metadata).

        A refusal (``content_filter``) then still ends the run without running
        its tool calls or asking again, and a cut-off answer (``length``)
        still reports ``max_output_tokens``.
        """
        last = self.history[-1]
        assistant = self.history[-2] if last.role == "tool" and len(self.history) > 1 else last
        if assistant.role != "assistant":
            return
        reason = assistant.metadata.get("finish_reason")
        self._refused = reason == "content_filter"
        if assistant is last and reason == "length":
            self._last_response = ModelResponse(message=last, finish_reason="length")

    async def _restore_handoff(self) -> None:
        """On resume, finish a handoff whose result was journaled but whose switch was not."""
        if len(self.history) < 2 or self.history[-1].role != "tool":
            return
        previous = self.history[-2]
        if previous.role != "assistant" or previous.name != self.agent.name:
            return
        accepted = {r.tool_call_id for r in self.history[-1].tool_results if not r.is_error}
        for call in previous.tool_calls:
            if call.id in accepted and call.name in self.handoffs:
                self._start_events.append(await self._switch(self.handoffs[call.name], call.id))
                await self._load_agent()
                return

    async def _switch(self, target: Agent, call_id: str) -> AgentSwitched:
        """Journal ``agent_switched`` and make ``target`` the active agent (load it next)."""
        previous = self.agent
        await self._journal("agent_switched", {"agent": target.name})
        self.agent = target
        return AgentSwitched(from_agent=previous.name, to_agent=target.name, tool_call_id=call_id)

    def _guardrail_event(self, tripped: TrippedGuardrail) -> GuardrailTriggered:
        return GuardrailTriggered(
            name=tripped.name, reason=tripped.reason, stage=tripped.stage, agent=self.agent.name
        )

    def _check_results(self, state: SessionState) -> None:
        """Fail before anything runs when ``results`` names a call that waits for no result."""
        last = self.history[-1]
        approvals = {item.tool_call_id for item in state.pending_approvals}
        accepted = {
            call_id for call_id, item in self.waiting.items() if item.held_record_id is None
        }
        for call in last.tool_calls if last.role == "assistant" else []:
            tool = self.tools.get(call.name)
            if call.id in self.results or call.id in approvals:
                continue
            if tool is not None and tool.deferred and _invalid_arguments(tool, call) is None:
                # Not journaled as waiting yet (a crash), but it can only wait. A call with
                # invalid arguments gets an error result instead, never one from results=.
                accepted.add(call.id)
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

    async def _check_redelivered_approvals(self, state: SessionState) -> None:
        """Compare answers for an already decided ask before running any tool."""
        if not self._effects or not self.approvals:
            return
        calls = {call.id: call for call in self.history[-1].tool_calls}
        pending = {item.tool_call_id: item for item in state.pending_approvals}
        for call_id, approval in list(self.approvals.items()):
            prior = self._approval_record(call_id)
            if prior is None:
                continue
            call = calls[call_id]
            fields = self._pause_fields(call, "approval")
            # A new, explicitly journaled ask can supersede an expired or unverifiable
            # answer. Its token differs from the token for the already decided ask.
            if (
                call_id in pending
                and pending[call_id].token is not None
                and approval.token == pending[call_id].token
                and self.permissions.verify(approval.token, **fields)
            ):
                continue
            if prior.arguments_changed_by is not None:
                # A hook may have recorded a second decision after the human answered.
                # Redelivery still refers to the human's signed ask.
                entries = await self.session.entries()
                for entry in reversed(entries):
                    if (
                        entry.kind != "approval_decided"
                        or entry.data.get("operation_id") != prior.operation_id
                        or entry.data.get("arguments_changed_by") is not None
                    ):
                        continue
                    candidate = _kinds.ApprovalRecord.model_validate(entry.data)
                    fields["ask"] = candidate.ask
                    if self.permissions.signs_pauses and not self.permissions.verify(
                        approval.token, **fields
                    ):
                        continue
                    prior = candidate
                    break
            fields["ask"] = prior.ask
            if self.permissions.signs_pauses and not self.permissions.verify(
                approval.token, **fields
            ):
                # _check_tokens supplies the standard mismatch details.
                continue
            self._merge_answer(call, prior, approval)
            del self.approvals[call_id]

    def _merge_answer(
        self, call: ToolCallPart, existing: _kinds.ApprovalRecord, new: Approval
    ) -> _kinds.ApprovalRecord:
        args = new.arguments if new.arguments is not None else call.arguments
        if (
            new.approved != existing.approved
            or new.reason != existing.reason
            or new.by != existing.by
            or arguments_digest(args) != existing.arguments_digest
        ):
            raise ApprovalConflictError(
                f"Tool call {call.id!r} already has a different approval answer.",
                tool_call_id=call.id,
            )
        return existing

    def _check_tokens(self, state: SessionState) -> None:
        """With signed pauses, refuse answers that do not carry their pause's token.

        An approval must match the call as the journal records it in the
        assistant message (the call that runs, unless the approval edits its
        arguments, which the permission rules then check), and the recorded
        pause must still match its own token. A result must match the waiting
        call as the journal records it, its reason and metadata included.
        Results are unwrapped from `InputResult` (and, when pauses are signed,
        from ``{"token": ..., "value": ...}``) here.
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
        for call_id, approval in self.approvals.items():  # only open calls are left
            call = calls[call_id]
            self._verify_token(approval.token, call, "approval")
            pause = recorded.get(call_id)
            if pause is not None and not self.permissions.verify(
                pause.token, **self._pause_fields(pause, "approval")
            ):
                raise self._tampered(call_id, call.name)
        for call_id, token in tokens.items():
            target: PendingInput | ToolCallPart | None = self.waiting.get(call_id)
            if target is None:  # a deferred call whose tool_deferred entry was lost
                target = calls.get(call_id)
            assert target is not None  # _check_results accepted only open calls
            self._verify_token(token, target, "result")

    def _pause_fields(
        self,
        pause: PendingApproval | PendingInput | ToolCallPart,
        kind: Literal["approval", "input"],
    ) -> dict[str, Any]:
        """What the token of a pause of this run signs (see `approval_token`)."""
        is_input = isinstance(pause, PendingInput)
        fields = {
            "session_id": self.session.id,
            "run_id": self.run_id,
            "position": self._ordinal,  # the call's assistant message is the run's last one
            "tool_call_id": pause.id if isinstance(pause, ToolCallPart) else pause.tool_call_id,
            "name": pause.name,
            "arguments": pause.arguments,
            "kind": kind,
            "reason": pause.reason if is_input else None,
            "metadata": pause.metadata if is_input else None,
        }
        if self._effects:
            call_id = fields["tool_call_id"]
            operation_id = self._operation_id(call_id)
            fields["operation_id"] = operation_id
            fields["ask"] = 1 + self._replay_state.decision_counts.get(operation_id, 0)
            if self._legacy_pause_tokens:
                fields["v"] = 2
        return fields

    def _verify_token(
        self,
        token: str | None,
        target: PendingInput | ToolCallPart,
        kind: Literal["approval", "result"],
    ) -> None:
        call_id = target.id if isinstance(target, ToolCallPart) else target.tool_call_id
        name = target.name
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
        pause_kind: Literal["approval", "input"] = "approval" if kind == "approval" else "input"
        if not self.permissions.verify(token, **self._pause_fields(target, pause_kind)):
            raise self._tampered(call_id, name)

    def _tampered(self, call_id: str, name: str) -> ApprovalTokenError:
        return ApprovalTokenError(
            f"The token for tool call {call_id!r} ({name}) does not match the call recorded in "
            f"session {self.session.id!r}: the token belongs to another call or another secret, "
            "or the journal was changed after the pause. Nothing was run. Check where the "
            "answer came from; to act on the call anyway, start a new run."
        )

    def _find_agent(self, name: str) -> Agent:
        for candidate in _reachable_agents(self.root):
            if candidate.name == name:
                return candidate
        raise SessionError(
            f"The run was last handled by agent {name!r}, which is not reachable from "
            f"{self.root.name!r} through handoffs."
        )

    def _limit_reached(self) -> str | None:
        """The limit that stops the run before its next request, if any.

        Tokens and cost are checked for this run and for every parent run, live,
        so child runs working in parallel share what their parents have left.
        """
        limits = self.limits
        if limits.max_turns is not None and self.turn >= limits.max_turns:
            return "max_turns"
        run: _Run | None = self
        while run is not None:
            limits, spent = run.limits, run._spent()
            tokens = limits.max_total_tokens
            if tokens is not None and spent.total_tokens >= tokens:
                return "max_total_tokens"
            if limits.max_cost_usd is not None and (
                limits.max_cost_usd <= 0
                or not spent.cost_complete  # a response without a cost: fail closed
                or (spent.cost_usd is not None and spent.cost_usd >= limits.max_cost_usd)
            ):
                return "max_cost_usd"
            run = run.parent_run
        return None

    # -- model phase ----------------------------------------------------------------

    async def _model_phase(self) -> AsyncIterator[Event]:
        ctx = self._ctx()
        request = await self.hooks.before_model(ctx, await self._build_request(ctx))
        if (updated := await self._save_state()) is not None:
            yield updated
        if self._cancel.is_set():
            return  # stopped while the request was prepared: nothing is sent
        model = self.agent.model
        attempt = 0
        while True:
            response: ModelResponse | None = None
            text: list[str] = []
            emitted = False
            try:
                async with contextlib.aclosing(self._model_events(model, request)) as events:
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
                    ),
                    extra={"turn": self.turn},
                )
            return

        response = await self.hooks.after_model(ctx, response)
        if (updated := await self._save_state()) is not None:
            yield updated
        with self._usage_lock:
            self.usage = self.usage.add(response.usage)
        self._last_response = response
        self._refused = response.finish_reason == "content_filter"
        answered_by = {
            "model": response.model or model.name,
            "provider": response.provider or model.provider,
        }
        # The finish reason and the model that answered travel with the message (metadata
        # is not sent to providers), so a resume still knows that a turn was refused or cut
        # off, and RunResult.messages say which model wrote what.
        metadata = {
            **response.message.metadata,
            "finish_reason": response.finish_reason,
            **answered_by,
        }
        message = response.message.model_copy(
            update={"name": self.agent.name, "metadata": metadata}
        )
        details = {
            "usage": response.usage.model_dump(mode="json"),
            **answered_by,
            "turn": self.turn,
        }
        async with self._batched():
            await self._append(message, extra=details)
            settled: list[ToolCallFinished] = []
            if self._refused:  # no safe point between a refusal and closing its tool calls
                settled = await self._settle_open_calls(prompts.TOOL_NOT_EXECUTED)
        yield ModelResponded(response=response, agent=self.agent.name)
        for event in settled:
            yield event

    async def _model_events(self, model: Model, request: ModelRequest) -> AsyncIterator[Any]:
        # Reset before yielding to the caller; an async iterator can be resumed by
        # another task, and tool/compaction calls must not inherit this logical turn.
        stream = self._until_cancelled(model.stream(request))
        async with contextlib.aclosing(stream):
            while True:
                token = _ports.MODEL_TURN.set(
                    _ports.ModelTurn(self.session.id, self.run_id, self.turn - 1)
                )
                try:
                    event = await anext(stream)
                except StopAsyncIteration:
                    return
                finally:
                    _ports.MODEL_TURN.reset(token)
                yield event

    async def _build_request(self, ctx: RunContext[Any]) -> ModelRequest:
        agent = self.agent
        system = await agent._render_instructions(ctx)
        output: OutputSpec | None = None
        addition: str | None = None
        if agent.output_mode == "tool":
            # The tool is offered, never forced: some providers reject a forced tool choice.
            template = (
                prompts.OUTPUT_TOOL_INSTRUCTIONS
                if agent._output_adapter is not None
                else prompts.OUTPUT_TOOL_TEXT_INSTRUCTIONS
            )
            addition = template.format(name=agent.output_tool_name)
        elif agent._output_schema is not None:
            if agent.output_mode == "prompted":
                schema = json.dumps(agent._output_schema, ensure_ascii=False)
                addition = prompts.OUTPUT_INSTRUCTIONS.format(schema=schema)
            else:
                assert agent._native_schema is not None
                output = OutputSpec(json_schema=agent._native_schema, strict=agent.output_strict)
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

    async def _maybe_compact(self) -> bool:
        """Let the compactor shorten the history before the next request (turn ``turn + 1``).

        Returns ``False`` when the run was cancelled (or timed out) first or
        meanwhile: the compactor's work, a summary request for example, is then
        interrupted and nothing is journaled.
        """
        compactor = self.root.compactor
        if compactor is None:
            return True
        ctx = dataclasses.replace(self._ctx(), turn=self.turn + 1)
        # The first request of a run or a resume has no response yet: an empty one (no usage)
        # makes the compactor estimate the whole history, so long sessions are compacted too.
        last = self._last_response or ModelResponse(message=Message(role="assistant"))
        model, history = self.agent.model, list(self.history)
        compacted, replacement = await self._unless_cancelled(
            lambda: compactor.compact(ctx, history, last, model)
        )
        if not compacted:
            return False
        if replacement:
            self.history = list(replacement)
            await self._journal(
                "compaction",
                {"messages": [message.model_dump(mode="json") for message in replacement]},
            )
        return True

    async def _finalize(self, last: Message) -> Literal[True] | OutputRejected:
        """Accept the text of ``last`` as the final answer, or ask the model to correct it."""
        agent = self.agent
        if agent.output_mode == "tool" and agent._output_adapter is not None:
            required = prompts.OUTPUT_TOOL_REQUIRED.format(name=agent.output_tool_name)
            rejection = _Rejected(required, f"was not given with {agent.output_tool_name}")
            self._count_rejection(rejection)
            await self._append(Message.user(required))
            return self._rejected_event(rejection)
        try:
            if agent._output_adapter is None:  # text, also in output_mode="tool"
                answer: Any = last.text
            else:
                try:
                    data = extract_json(last.text)
                    if agent._native_wrapped and agent.output_mode == "native":
                        data = _unwrap_value(data)
                    answer = agent._output_adapter.validate_python(data)
                except ValueError as error:  # pydantic's ValidationError included
                    raise _Rejected(_brief(error), "did not match the output type") from error
            self.output = await self._validate_output(answer)
            return True
        except _Rejected as rejection:
            self._count_rejection(rejection)
            if agent._output_adapter is None:
                template = prompts.OUTPUT_TEXT_RETRY
            elif rejection.by_validator:  # the correction may need a tool call
                template = prompts.OUTPUT_VALIDATOR_RETRY
            else:
                template = prompts.OUTPUT_RETRY
            await self._append(Message.user(template.format(error=rejection.message)))
            return self._rejected_event(rejection)

    def _rejected_event(self, rejection: _Rejected) -> OutputRejected:
        return OutputRejected(
            reason=rejection.message, attempt=self._output_attempts, agent=self.agent.name
        )

    async def _validate_output(self, output: Any) -> Any:
        """Run the active agent's output validators in order. Raises `_Rejected` on `ModelRetry`."""
        ctx = self._ctx()
        for validator in self.agent.output_validators:
            try:
                replaced = validator(ctx, output)
                if inspect.isawaitable(replaced):
                    replaced = await replaced
            except ModelRetry as retry:
                raise _Rejected(
                    _brief(retry), "was rejected by an output validator", by_validator=True
                ) from retry
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
            yield self._rejected_event(rejection)
            return
        self.output = output
        accepted = ToolResultPart(
            tool_call_id=answer.id, name=answer.name, content=prompts.OUTPUT_TOOL_ACCEPTED
        )
        events = await self._close_answer_turn(calls, accepted, answer)
        self._answered = True
        for event in events:
            yield event

    async def _close_answer_turn(
        self,
        calls: list[ToolCallPart],
        accepted: ToolResultPart | None = None,
        answer: ToolCallPart | None = None,
    ) -> list[Event]:
        """Journal the turn of an accepted answer in one write: the answer's result, a
        "not run" result for every other open call, and the tool message.

        Its events are returned to be yielded afterwards, so nothing can stop
        the run between the accepted answer and the closed turn.
        """
        agent = self.agent.name
        events: list[Event] = []
        async with self._batched():
            if accepted is not None and answer is not None:
                await self._record(accepted)
                events.append(ToolCallFinished(call=answer, result=accepted, agent=agent))
            for call in calls:
                if call.id not in self.results:
                    content = prompts.OUTPUT_TOOL_SKIPPED.format(name=call.name)
                    skipped = _error_result(call, content)
                    await self._record(skipped)
                    events.append(ToolCallFinished(call=call, result=skipped, agent=agent))
            await self._close_tool_turn(calls)
        return events

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
        if len({call.id for call in calls}) != len(calls):
            raise ConfigurationError("The model returned duplicate tool call ids in one turn.")
        prior = next(
            (
                c
                for c in calls
                if self.agent._is_answer_call(c)
                and c.id in self.results
                and not self.results[c.id].is_error
            ),
            None,
        )
        if prior is not None:
            # An answer was accepted before the run stopped, but its turn was not closed.
            # The other calls of the turn were never meant to run: close it without them.
            try:
                self.output = await self._validate_output(self.agent._validate_answer_call(prior))
                answered = True
            except (_Rejected, ValueError):  # the agent changed: the model answers again
                answered = False
            events = await self._close_answer_turn(calls)
            self._answered = answered
            for event in events:
                yield event
            return
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
                recorded = self.results[call.id]
                if call.name in self.handoffs and not recorded.is_error and handoff is None:
                    handoff = call  # accepted before the run stopped; the switch is still due
                continue
            if call.id in self.supplied:  # given to resume(results=...)
                ctx = self._ctx(tool_call_id=call.id)
                result = await self.hooks.after_tool(ctx, call, self._supplied_result(call))
                result = result.model_copy(update={"tool_call_id": call.id, "name": call.name})
                async with self._state_lock, self._batched():
                    updated = await self._save_state_locked()
                    await self._record(result)
                if updated is not None:
                    yield updated
                yield ToolCallFinished(call=call, result=result, agent=self.agent.name)
                continue
            if call.id in self.waiting and self.waiting[call.id].held_record_id is None:
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
                    name=call.name,
                    available=prompts.UNKNOWN_TOOL_SEPARATOR.join(available)
                    or prompts.UNKNOWN_TOOL_NONE,
                )
            elif (
                budget is not None and call.id not in self._counted and len(self._counted) >= budget
            ):
                refusal = prompts.TOOL_CALL_LIMIT.format(name=call.name, limit=budget)
                self._tool_limit_hit = True
            elif self._cancel.is_set():
                continue  # stopped: ask no one, run nothing; the loop settles the call
            else:
                tool = self.tools[call.name]
                if call.id in self.started and not (
                    tool.read_only or tool.idempotent or tool.effects
                ):
                    refusal = prompts.TOOL_INTERRUPTED.format(name=call.name)
                elif (invalid := _invalid_arguments(tool, call)) is not None:
                    refusal = invalid  # before anyone is asked to approve the call
                else:
                    if tool.effects and not self._needs_decision(call):
                        held = self._held_calls.get(call.id)
                        if held is not None and _engine.send_running(
                            self._writer.store, self.session.id, held.id
                        ):
                            if call.id not in self.waiting:
                                await self._defer_hold(call, held)
                            continue
                        prior = self._approval_record(call.id)
                        if prior is not None and prior.arguments_changed:
                            call = call.model_copy(
                                update={
                                    "arguments": prior.arguments,
                                    "raw_arguments": None,
                                }
                            )
                        verdict, payload = "run", call
                    else:
                        verdict, payload = await self._authorize(tool, call)
                    if self._cancel.is_set() and verdict in ("run", "pending"):
                        continue  # stopped while it was decided: the loop settles the call
                    if verdict == "run" and call.id in self._edited:
                        # An approver's edit is checked like the model's arguments.
                        if (invalid := _invalid_arguments(tool, payload)) is not None:
                            verdict, payload = "invalid", invalid
                    if verdict == "run":
                        self._counted[call.id] = None  # once per call, also across resumes
                    if verdict == "run" and tool.deferred:
                        unsignable, updated = await self._defer(payload)
                        if updated is not None:
                            yield updated
                        if unsignable is None:
                            continue
                        verdict, payload = "deny", unsignable
                    if verdict == "run":
                        runnable.append((payload, tool))
                        continue
                    if verdict == "pending":
                        pending.append(payload)
                        continue
                    refusal = (
                        payload
                        if verdict == "invalid"
                        else prompts.PERMISSION_DENIED.format(name=call.name, reason=payload)
                    )
            result = _error_result(call, refusal)
            await self._record(result)
            yield ToolCallFinished(call=call, result=result, agent=self.agent.name)

        for group in self._groups(runnable):
            if self._cancel.is_set():
                return  # the loop settles the calls that never started
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

        target = self.handoffs[handoff.name] if handoff is not None else None
        switched: AgentSwitched | None = None
        # The handoff's result, the tool message and the switch are one write, so a
        # resume never finds the handoff told to the model but not made.
        async with self._batched():
            if handoff is not None and target is not None and handoff.id not in self.results:
                await self._record(
                    ToolResultPart(
                        tool_call_id=handoff.id,
                        name=handoff.name,
                        content=prompts.HANDOFF_RESULT.format(name=target.name),
                    )
                )
            await self._close_tool_turn(calls)
            if target is not None and handoff is not None:
                switched = await self._switch(target, handoff.id)
        if switched is not None:
            await self._load_agent()
            yield switched

    async def _authorize(
        self,
        tool: Tool,
        call: ToolCallPart,
        *,
        ctx: RunContext[Any] | None = None,
        can_pause: bool = True,
        approval_timeout_s: float | None = None,
        tags: Mapping[str, Any] | None = None,
        still_wanted: Callable[[], bool] | None = None,
    ) -> tuple[str, Any]:
        """Decide one call.

        The permission rules are always checked first: a deny refuses the call
        whatever approval was given. An approval given to ``resume`` (or by
        ``on_ask``) then answers the call; one that edits the arguments has
        them checked again, and is journaled (``approval_decided``) before
        the call runs.

        A call from outside the model (`_outside_call`) passes its ``ctx``,
        ``can_pause=False`` (an "ask" without ``on_ask`` is refused with
        `TOOL_SERVER_NO_APPROVER`, because nothing can pause the run for it),
        ``approval_timeout_s`` for ``on_ask`` (`TOOL_SERVER_APPROVAL_TIMEOUT`),
        the ``tags`` its journal entries carry, and ``still_wanted``: when that
        is false once ``on_ask`` answered, nothing is journaled.

        Returns ``("run", call)``, ``("pending", PendingApproval)``, ``("deny", reason)``
        or ``("ended", None)``.
        """
        ctx = ctx or self._ctx(tool_call_id=call.id)
        decision = await self.permissions.decide(ctx, tool, call)
        approval = self.approvals.pop(call.id, None)
        if decision.behavior == "deny":
            return "deny", decision.reason or prompts.PERMISSION_REASON_DEFAULT
        if approval is None:
            if decision.behavior == "allow":
                if tool.effects:
                    self._authorizations[call.id] = _ports.Authorization("allow")
                return "run", call
            prior = await self._reusable_decision(call, tool) if tool.effects else None
            if prior is not None:
                if not prior.approved:
                    return "deny", prior.reason or prompts.APPROVAL_DECLINED
                args = prior.arguments if prior.arguments_changed else call.arguments
                reused = call.model_copy(update={"arguments": args, "raw_arguments": None})
                if (await self.permissions.decide(ctx, tool, reused)).behavior != "deny":
                    self._authorizations[call.id] = self._approval_auth(prior)
                    return "run", reused
            if self.permissions.on_ask is None:
                if not can_pause:
                    return "deny", prompts.TOOL_SERVER_NO_APPROVER
                pending = PendingApproval(
                    tool_call_id=call.id,
                    name=call.name,
                    arguments=call.arguments,
                    reason=decision.reason,
                )
                try:
                    pending.token = self.permissions.sign(**self._pause_fields(pending, "approval"))
                except ValueError:  # NaN, an infinity or a lone surrogate in the arguments
                    return "deny", prompts.UNSIGNABLE_CALL
                return "pending", pending
            try:
                async with asyncio.timeout(approval_timeout_s):
                    approval = await self.permissions.request_approval(
                        ctx, tool, call, decision.reason
                    )
            except TimeoutError:  # request_approval turns the approver's own errors into a no
                seconds = approval_timeout_s or 0
                return "deny", prompts.TOOL_SERVER_APPROVAL_TIMEOUT.format(seconds=seconds)
            if still_wanted is not None and not still_wanted():
                return "ended", None
        edited = approval.approved and approval.arguments not in (None, call.arguments)
        record: dict[str, Any] = {
            "tool_call_id": call.id,
            "name": call.name,
            "approved": approval.approved,
            "reason": approval.reason,
            "arguments_changed": edited,
        }
        if edited:
            record["arguments"] = approval.arguments
        if tool.effects:
            approved_call = (
                call.model_copy(update={"arguments": approval.arguments}) if edited else call
            )
            operation_id = self._operation_id(call.id)
            try:
                signed = self.permissions.decision_record(
                    session_id=self.session.id,
                    run_id=self.run_id,
                    operation_id=operation_id,
                    ask=1 + self._replay_state.decision_counts.get(operation_id, 0),
                    tool=tool,
                    call=approved_call,
                    approved=approval.approved,
                    reason=approval.reason,
                    by=approval.by,
                    decided_at=await self._writer.store_now(),
                    arguments_changed=edited,
                )
            except (ValueError, TypeError):
                return "deny", prompts.UNSIGNABLE_CALL
            record = signed.model_dump(mode="json")
            self._authorizations[call.id] = self._approval_auth(signed)
        await self._journal("approval_decided", {**record, **(tags or {})})  # before the call runs
        if tool.effects:
            self._replay_state.decisions[signed.operation_id] = record
            self._replay_state.decision_counts[signed.operation_id] = signed.ask
        if not approval.approved:
            return "deny", approval.reason or prompts.APPROVAL_DECLINED
        if edited:
            assert approval.arguments is not None
            call = call.model_copy(update={"arguments": approval.arguments, "raw_arguments": None})
            recheck = await self.permissions.decide(ctx, tool, call)
            if recheck.behavior == "deny":  # the approval answers "ask", never a deny
                return "deny", recheck.reason or prompts.PERMISSION_REASON_DEFAULT
            self._edited.add(call.id)
        return "run", call

    def _groups(
        self, runnable: list[tuple[ToolCallPart, Tool]]
    ) -> list[list[tuple[ToolCallPart, Tool]]]:
        """Split calls into groups; a group runs concurrently, groups run in order."""
        groups: list[list[tuple[ToolCallPart, Tool]]] = []
        batch: list[tuple[ToolCallPart, Tool]] = []
        for item in runnable:
            if not self._runs_alone(item[1]):
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
        if self._cancel.is_set():
            return
        queue: asyncio.Queue[Any] = asyncio.Queue()
        gate = asyncio.Semaphore(max(1, self.limits.max_parallel_tools))
        tasks = [
            asyncio.create_task(self._run_tool(call, tool, queue, gate)) for call, tool in group
        ]
        cancelled = asyncio.ensure_future(self._cancel.wait())
        item_ready: asyncio.Future[Any] | None = None
        finished = 0
        try:
            # Each task ends with _DONE, whatever happened in it: events a tool emits
            # (a ToolCallFinished of a sub-agent's included) cannot end the group early.
            while finished < len(tasks):
                item_ready = asyncio.ensure_future(queue.get())
                await asyncio.wait({item_ready, cancelled}, return_when=asyncio.FIRST_COMPLETED)
                if not item_ready.done():
                    return
                item, item_ready = item_ready.result(), None
                if isinstance(item, _Failure):
                    raise item.error
                if item is _DONE:
                    finished += 1
                    continue
                yield item
        finally:
            cancelled.cancel()
            if item_ready is not None:
                item_ready.cancel()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, *[f for f in (item_ready,) if f], return_exceptions=True)

    async def _run_tool(
        self, call: ToolCallPart, tool: Tool, queue: asyncio.Queue[Any], gate: asyncio.Semaphore
    ) -> None:
        """Run one call in its own task; the last thing it puts on ``queue`` is ``_DONE``."""
        try:
            async with gate:
                await self._execute_call(call, tool, queue)
        except asyncio.CancelledError:
            if _cancelled_from_outside():
                raise  # the group, or the run, is being cancelled
            queue.put_nowait(  # a stray CancelledError: never a silent exit that hangs the run
                _Failure(
                    RuntimeError(
                        f"A hook or the journal raised CancelledError while tool call {call.id!r} "
                        f"({call.name}) ran, although the run was not cancelled."
                    )
                )
            )
        except BaseException as error:  # a hook or the journal failed: stop the run
            queue.put_nowait(_Failure(error))
        queue.put_nowait(_DONE)

    async def _execute_call(
        self, call: ToolCallPart, tool: Tool, queue: asyncio.Queue[Any]
    ) -> None:
        if tool.effects:
            await self._execute_effect_call(call, tool, queue)
            return
        agent = self.agent.name
        if self._cancel.is_set():
            return  # stopped while it waited for its turn: it never starts
        if (blocked := await self._wait_for_stuck_threads(tool, call)) is not None:
            self._counted.pop(call.id, None)  # it never ran, so it does not count
            await self._record(blocked)
            queue.put_nowait(ToolCallFinished(call=call, result=blocked, agent=agent))
            return
        if self._cancel.is_set():
            return  # stopped while it waited for an earlier call: it never starts
        ctx = self._ctx(tool_call_id=call.id, emit=queue.put_nowait)
        await self._journal("tool_started", {"tool_call_id": call.id, "name": call.name})
        if self._cancel.is_set():
            return  # stopped while the start was written: nothing ran
        self._running.add(call.id)
        assert ctx._emit is not None
        self._call_emit[call.id] = ctx._emit  # calls from outside the model that it makes
        self._call_tasks[call.id] = _cancel_mark(asyncio.current_task())  # the tool runs here
        queue.put_nowait(ToolCallStarted(call=call, agent=agent))
        began = time.perf_counter()
        hooked = await self.hooks.before_tool(ctx, call)
        if (updated := await self._save_state()) is not None:
            queue.put_nowait(updated)  # a change made in before_tool is journaled before the tool
        if isinstance(hooked, ToolResultPart):
            result = hooked
        else:
            try:
                result = await self._invoke(ctx, tool, hooked)
            except DeferToolCall as deferral:
                unsignable, updated = await self._defer(
                    call.model_copy(update={"arguments": hooked.arguments}),
                    deferral.reason,
                    deferral.metadata,
                )
                if updated is not None:
                    queue.put_nowait(updated)
                if unsignable is None:
                    return
                refusal = prompts.PERMISSION_DENIED.format(name=call.name, reason=unsignable)
                result = _error_result(call, refusal)
        result = await self.hooks.after_tool(ctx, call, result)
        result = result.model_copy(update={"tool_call_id": call.id, "name": call.name})
        if call.id in self._edited:  # tell the model what actually ran
            note = prompts.APPROVAL_ARGUMENTS_CHANGED.format(
                arguments=json.dumps(call.arguments, ensure_ascii=False, default=str)
            )
            content = prompts.APPROVAL_ARGUMENTS_CHANGED_RESULT.format(
                note=note, content=result.content
            )
            result = result.model_copy(update={"content": content})
        # Keep the state lock through the atomic write: another task cannot persist
        # a newer snapshot before this batch writes its older one.
        async with self._state_lock, self._batched():
            updated = await self._save_state_locked()
            await self._record(result)
        if updated is not None:
            queue.put_nowait(updated)
        queue.put_nowait(
            ToolCallFinished(
                call=call,
                result=result,
                duration_ms=(time.perf_counter() - began) * 1000,
                agent=agent,
            )
        )

    def _operation_id(self, call_id: str) -> str:
        starts = self._replay_state.open_turn_starts.get(call_id, ())
        if starts and isinstance(starts[-1].get("operation_id"), str):
            return starts[-1]["operation_id"]
        return f"{self.run_id}:{self._ordinal}:{call_id}"

    def _has_intent(self, call_id: str) -> bool:
        return self._writer is not None and (
            self._operation_id(call_id) in self._writer.view().operations_with_intent
        )

    def _needs_decision(self, call: ToolCallPart) -> bool:
        """An existing intent has already consumed this call's decision."""
        return not self._has_intent(call.id)

    async def _reusable_decision(
        self, call: ToolCallPart, tool: Tool
    ) -> _kinds.ApprovalRecord | None:
        prior = self._approval_record(call.id)
        if prior is None or not self.permissions.verify_decision(prior):
            return None
        if not prior.approved:
            return prior
        args = prior.arguments if prior.arguments_changed else call.arguments
        if (
            args is not None
            and prior.tool_spec_digest == tool_spec_digest(tool)
            and prior.arguments_digest == arguments_digest(args)
            and await self._writer.store_now() < prior.expires_at
        ):
            return prior
        return None

    def _approval_record(self, call_id: str) -> _kinds.ApprovalRecord | None:
        data = self._replay_state.decisions.get(self._operation_id(call_id))
        if data is None:
            return None
        record = _kinds.ApprovalRecord.model_validate(data)
        if (record.session_id, record.run_id, record.operation_id, record.tool_call_id) != (
            self.session.id,
            self.run_id,
            self._operation_id(call_id),
            call_id,
        ):
            raise SessionError("The approval record belongs to a different tool call.")
        return record

    def _approval_auth(self, record: _kinds.ApprovalRecord) -> _ports.Authorization:
        return _ports.Authorization(
            "ask",
            approval_id=record.approval_id,
            expires_at=record.expires_at,
            verifiable=self.permissions.verify_decision(record),
            arguments_digest=record.arguments_digest,
        )

    def _scope_for(
        self,
        call: ToolCallPart,
        tool: Tool,
        ctx: RunContext[Any],
        digest: str,
    ) -> _ports.CallScope:
        operation_id = ctx.operation_id
        starts = self._replay_state.open_turn_starts.get(call.id, ())
        original = None
        if starts and starts[0].get("authorization") is not None:
            data = _kinds.AuthorizationData.model_validate(starts[0]["authorization"])
            original = _ports.Authorization(
                **{
                    key: value
                    for key, value in data.model_dump().items()
                    if key in _ports.Authorization.__dataclass_fields__
                }
            )
            if original.decision == "ask":
                record = self._approval_record(call.id)
                valid = record is not None and (
                    record.approved
                    and record.approval_id == original.approval_id
                    and record.tool_spec_digest == tool_spec_digest(tool)
                    and record.arguments_digest == starts[0].get("arguments_digest")
                )
                original = (
                    self._approval_auth(record)
                    if valid
                    else _ports.Authorization(
                        "ask",
                        verifiable=False,
                    )
                )
        auth = self._authorizations.get(call.id) or original
        if auth is None:
            auth = _ports.Authorization("none", verifiable=False)
        original_digest = starts[0].get("arguments_digest") if starts else digest
        memos = self._replay_state.memos.setdefault(operation_id, {})

        async def record_memo(name: str, value: Any) -> None:
            await self._journal(
                _kinds.KIND_MEMO_RECORDED,
                {
                    "operation_id": operation_id,
                    "name": name,
                    "value": value,
                },
            )

        def apply_patch(patch: _kinds.StatePatchData) -> None:
            for values in (self.state, self._committed_state):
                values.update(copying.deepcopy(patch.set))
                for key in patch.unset:
                    values.pop(key, None)

        return _ports.CallScope(
            self.session.id,
            self.run_id,
            tool.name,
            call.id,
            operation_id,
            1 + self._replay_state.tool_attempts.get(operation_id, 0),
            frozenset(effect.name for effect in tool.effects),
            auth,
            original or auth,
            digest,
            original_digest,
            None,
            MISSING if self.deps is None else self.deps,
            ctx._emit or (lambda _: None),
            self._writer,
            self.permissions,
            memos,
            record_memo,
            lambda: self._committed_state,
            apply_patch,
        )

    def _unresolved_for(self, operation_id: str) -> list[HeldEffect]:
        if self._writer is None:
            return []
        view = self._writer.view()
        owned = {
            op.record_id for op in view.operations.values() if operation_id in op.operation_ids
        }
        return [item for item in view.held(self.session.id) if item.id in owned]

    async def _defer_hold(self, call: ToolCallPart, held: HeldEffect) -> None:
        waiting = PendingInput(
            tool_call_id=call.id,
            name=call.name,
            arguments=call.arguments,
            reason=held.detail,
            held_record_id=held.id,
            held=held,
        )
        try:
            waiting.token = self.permissions.sign(**self._pause_fields(waiting, "input"))
        except ValueError as error:
            raise SessionError("The held tool call cannot be signed.") from error
        await self._journal(
            "tool_deferred",
            {
                "input": waiting.model_dump(mode="json", exclude={"held"}),
                "held_record_id": held.id,
            },
        )
        self._held_calls[call.id] = held
        self.waiting[call.id] = waiting
        self._running.discard(call.id)
        self._call_emit.pop(call.id, None)
        self._call_tasks.pop(call.id, None)
        self.state.clear()
        self.state.update(copying.deepcopy(self._committed_state))
        self._saved_state = _state_json(self.state)

    async def _enforce_holds(
        self,
        call: ToolCallPart,
        tool: Tool,
        held: list[HeldEffect],
    ) -> None:
        policy = next(effect.on_unknown for effect in tool.effects if effect.name == held[0].effect)
        if policy == "fail_run":
            raise HeldEffectsError(held, retry_after=held[0].retry_after)
        await self._defer_hold(call, held[0])

    async def _hook_edit_allowed(
        self,
        ctx: RunContext[Any],
        approved: ToolCallPart,
        tool: Tool,
        changed: ToolCallPart,
        changed_by: str | None,
        auth: _ports.Authorization | None,
    ) -> bool:
        if _invalid_arguments(tool, changed) is not None:
            return False
        if auth is None or auth.decision != "ask":
            return (await self.permissions.decide(ctx, tool, changed)).behavior == "allow"
        editors = [
            hook
            for hook in self.hooks.hooks
            if f"{type(hook).__module__}.{type(hook).__qualname__}" == changed_by
        ]
        if not editors or not all(hook.may_change_approved_arguments for hook in editors):
            return False
        decision = await self.permissions.decide(ctx, tool, changed)
        if decision.behavior == "deny":
            return False
        if decision.behavior == "ask":
            if self.permissions.on_ask is None:
                return False
            approval = await self.permissions.request_approval(ctx, tool, changed, decision.reason)
            if not approval.approved or approval.arguments not in (None, changed.arguments):
                return False
            by = approval.by
        else:
            by = changed_by
        operation = self._operation_id(approved.id)
        signed = self.permissions.decision_record(
            session_id=self.session.id,
            run_id=self.run_id,
            operation_id=operation,
            ask=1 + self._replay_state.decision_counts.get(operation, 0),
            tool=tool,
            call=changed,
            approved=True,
            reason=decision.reason,
            by=by,
            decided_at=await self._writer.store_now(),
            arguments_changed=True,
            arguments_changed_by=changed_by,
        )
        record = signed.model_dump(mode="json")
        await self._journal("approval_decided", record)
        self._replay_state.decisions[operation] = record
        self._replay_state.decision_counts[operation] = signed.ask
        self._authorizations[approved.id] = self._approval_auth(signed)
        self._edited.add(approved.id)
        return True

    async def _execute_effect_call(
        self,
        call: ToolCallPart,
        tool: Tool,
        queue: asyncio.Queue[Any],
    ) -> None:
        if self._cancel.is_set():
            return
        ctx = self._ctx(tool_call_id=call.id, emit=queue.put_nowait)
        operation_id = ctx.operation_id
        try:
            original_digest = arguments_digest(call.arguments)
            hooked, changed_by = await self.hooks.before_tool_traced(ctx, call)
            digest = (
                arguments_digest(hooked.arguments) if isinstance(hooked, ToolCallPart) else None
            )
        except ValueError:
            if self._has_intent(call.id):
                raise SessionError("An open effects call has noncanonical arguments.") from None
            result = _error_result(
                call, prompts.EFFECT_ARGUMENTS_NOT_CANONICAL.format(name=call.name)
            )
            await self._record(result)
            queue.put_nowait(ToolCallFinished(call=call, result=result, agent=self.agent.name))
            return
        auth = self._authorizations.get(call.id)
        if digest != original_digest and isinstance(hooked, ToolCallPart):
            if not self._has_intent(call.id) and not await self._hook_edit_allowed(
                ctx, call, tool, hooked, changed_by, auth
            ):
                result = _error_result(call, prompts.APPROVAL_SCOPE_CHANGED)
                await self._record(result)
                queue.put_nowait(ToolCallFinished(call=call, result=result, agent=self.agent.name))
                return
        if isinstance(hooked, ToolResultPart):
            unresolved = self._unresolved_for(operation_id)
            if unresolved:
                await self._enforce_holds(call, tool, unresolved)
                return
            result = hooked
        else:
            scope = self._scope_for(call, tool, ctx, digest)
            if (
                self._has_intent(call.id)
                and scope.original_arguments_digest != digest
                and (held := self._unresolved_for(operation_id))
            ):
                await self._defer_hold(
                    call,
                    replace(
                        held[0],
                        reason=HoldReason.DIVERGED,
                        detail=HOLD_DETAIL[HoldReason.DIVERGED],
                    ),
                )
                return
            self._scopes[call.id] = scope
            data = {
                "tool_call_id": call.id,
                "name": call.name,
                "operation_id": operation_id,
                "attempt": scope.tool_attempt,
                "arguments_digest": digest,
                "authorization": scope.authorization.data().model_dump(mode="json"),
                "arguments_changed_by": changed_by,
            }
            async with self._state_lock, self._batched():
                await self._save_state_locked(provisional=operation_id)
                await self._journal("tool_started", data)
            self._replay_state.tool_attempts[operation_id] = scope.tool_attempt
            self._replay_state.open_turn_starts.setdefault(call.id, []).append(data)
            self._running.add(call.id)
            self._call_emit[call.id] = ctx._emit
            self._call_tasks[call.id] = _cancel_mark(asyncio.current_task())
            queue.put_nowait(ToolCallStarted(call=hooked, agent=self.agent.name))
            try:
                result = await self._invoke(ctx, tool, hooked, scope=scope)
            except EffectHeld as error:
                await self._defer_hold(hooked, error.held)
                return
            except DeferToolCall:
                raise SessionError(
                    "An effects tool cannot defer a result outside the protocol."
                ) from None
            finally:
                self._scopes.pop(call.id, None)
            unresolved = scope.held or self._unresolved_for(operation_id)
            if unresolved and call.id not in self._unknown_results:
                logger.warning(
                    "%s returned while an effect remained held",
                    tool.name,
                    extra={"qagents_finding": "swallowed_hold"},
                )
                _ports.report_finding("swallowed_hold", f"{tool.name}: returned with a held effect")
                await self._enforce_holds(hooked, tool, unresolved)
                return
        unknown_outcome = call.id in self._unknown_results
        if unknown_outcome:
            self._unknown_results.discard(call.id)
            self.state.clear()
            self.state.update(copying.deepcopy(self._committed_state))
            self._saved_state = _state_json(self.state)
        else:
            result = await self.hooks.after_tool(ctx, call, result)
        result = result.model_copy(update={"tool_call_id": call.id, "name": call.name})
        async with self._state_lock, self._batched():
            await self._save_state_locked()
            await self._record(result, commit_provisional=not unknown_outcome)
        self._held_calls.pop(call.id, None)
        queue.put_nowait(ToolCallFinished(call=call, result=result, agent=self.agent.name))

    def _runs_alone(self, tool: Tool) -> bool:
        """Calls of ``tool`` run one at a time (`Limits.parallel_tools`)."""
        mode = self.limits.parallel_tools
        return bool(tool.effects) or mode == "none" or (mode == "read_only" and not tool.read_only)

    async def _wait_for_stuck_threads(
        self, tool: Tool, call: ToolCallPart
    ) -> ToolResultPart | None:
        """Keep calls that run one at a time apart from a sync call of this run that timed
        out (or was cancelled, as a served call can be on its own) and still runs in its
        thread: wait for it, within this call's timeout.

        Returns ``None`` when the call may start, else its error result (it never started).
        Calls from outside the model wait here too (`_outside_call`), possibly several at
        once, so the list is never swapped out: each waits on a copy, and the threads that
        returned are removed from the shared list in place.
        """
        if not self._stuck_threads or not self._runs_alone(tool):
            return None
        limit = tool.timeout if tool.timeout is not None else self.limits.tool_timeout_s
        finished = await _threads_finished(list(self._stuck_threads), limit or None)
        self._stuck_threads[:] = [thread for thread in self._stuck_threads if not thread.done()]
        if finished:
            return None
        content = prompts.TOOL_WAITED_FOR_EARLIER.format(name=call.name, seconds=limit)
        result = _error_result(call, content)
        result.metadata["timeout"] = limit
        return result

    def _timeout_result(
        self, tool: Tool, call: ToolCallPart, timeout: ToolTimeout, scope: _ports.CallScope | None
    ) -> ToolResultPart | None:
        if scope is None:
            return None
        held = scope.held or self._unresolved_for(scope.operation_id)
        if held:
            if any(effect.on_unknown == "fail_run" for effect in tool.effects):
                raise HeldEffectsError(held, retry_after=held[0].retry_after) from None
            raise EffectHeld(held[0]) from None
        content = (
            prompts.EFFECT_TOOL_TIMEOUT.format(
                seconds=timeout.seconds,
                committed=", ".join(f"{name} {key}" for name, key in scope.committed),
            )
            if scope.committed
            else prompts.EFFECT_TOOL_TIMEOUT_NONE.format(
                seconds=timeout.seconds,
            )
        )
        return _error_result(call, content)

    async def _invoke(
        self,
        ctx: RunContext[Any],
        tool: Tool,
        call: ToolCallPart,
        *,
        scope: _ports.CallScope | None = None,
    ) -> ToolResultPart:
        """Run a tool and turn whatever happens into a result the model can read.

        A tool that returns a `ToolResultPart` keeps its content, ``is_error`` and
        ``metadata`` (the call id and name are set by the loop), and a `ToolError`'s
        ``metadata`` is copied into the result's. Metadata that is not JSON cannot be
        journaled: it is logged and replaced by ``{"metadata_error": <why>}``, and the
        content and ``is_error`` stay as the tool gave them. The tool did run, so the
        model must not read that it failed (it might call it again).

        `DeferToolCall` is the one exception that passes through: the call then has no result.
        A real cancellation of the tool's task passes through too.
        """
        metadata: dict[str, Any] = {}
        is_error = True

        def abandoned(thread: concurrent.futures.Future[Any]) -> None:
            # A sync call that timed out or was cancelled still runs in its thread: the
            # next call that runs alone waits for it (`_wait_for_stuck_threads`). Registered
            # before the error leaves the tool, so before a served call's gate is let go.
            if self._runs_alone(tool):
                self._stuck_threads.append(thread)

        try:
            with _abandoned_threads_to(abandoned):
                ctx._call = scope
                try:
                    value = await tool.run(ctx, call.arguments, timeout=self.limits.tool_timeout_s)
                finally:
                    ctx._call = None
            if isinstance(value, ToolResultPart):
                # A tool that builds its own result keeps its error flag and metadata;
                # the loop still sets the call id and name (as for resume(results=...)).
                content, is_error = value.content, value.is_error
                metadata.update(value.metadata)
            else:
                content = render_tool_output(value)
                is_error = False
        except DeferToolCall:
            raise
        except (HeldEffectsError, JournalOutcomeUnknown, LeaseError):
            raise
        except ToolArgumentError as error:
            content = prompts.INVALID_ARGUMENTS.format(name=call.name, error=error)
        except ToolError as error:
            content = str(error)
            extra = getattr(error, "metadata", None)
            if isinstance(extra, Mapping):
                metadata.update(extra)
            if scope is not None and scope.held and metadata.get("effect_outcome_unknown") is True:
                self._unknown_results.add(call.id)
        except ToolTimeout as timeout:
            if result := self._timeout_result(tool, call, timeout, scope):
                return result
            # Only the framework's own timeout: a TimeoutError of the tool's code is a failure.
            if not timeout.started:
                template = prompts.TOOL_NOT_STARTED
            elif timeout.running:
                template = prompts.TOOL_TIMEOUT_ABANDONED  # its thread is kept (`abandoned`)
            else:
                template = prompts.TOOL_TIMEOUT
            content = template.format(name=call.name, seconds=timeout.seconds)
            metadata["timeout"] = timeout.seconds
        except asyncio.CancelledError as error:
            if _cancelled_from_outside():
                raise
            content = prompts.TOOL_FAILED.format(
                name=call.name,
                error=prompts.TOOL_FAILED_EXCEPTION.format(type="CancelledError", error=error),
            )
            metadata["exception"] = "CancelledError"
        except Exception as error:
            logger.exception("Tool %s raised", call.name)
            content = prompts.TOOL_FAILED.format(
                name=call.name,
                error=prompts.TOOL_FAILED_EXCEPTION.format(type=type(error).__name__, error=error),
            )
            metadata["exception"] = type(error).__name__
        if metadata and (problem := _not_json(metadata)) is not None:
            logger.error(
                "Tool %s gave result metadata that is not JSON; it was dropped: %s",
                call.name,
                problem,
            )
            metadata = {"metadata_error": problem[:500]}
        return ToolResultPart(
            tool_call_id=call.id,
            name=call.name,
            content=_shorten(content, self.limits.max_tool_result_chars),
            is_error=is_error,
            metadata=metadata,
        )

    async def _record(self, result: ToolResultPart, *, commit_provisional: bool = True) -> None:
        self.results[result.tool_call_id] = result
        self.waiting.pop(result.tool_call_id, None)
        self._running.discard(result.tool_call_id)
        self._call_emit.pop(result.tool_call_id, None)
        self._call_tasks.pop(result.tool_call_id, None)
        data: dict[str, Any] = {"result": result.model_dump(mode="json")}
        if result.tool_call_id in self._replay_state.open_turn_starts:
            data["operation_id"] = self._operation_id(result.tool_call_id)
            if not commit_provisional:
                data["commit_provisional"] = False
        if result.tool_call_id in self._counted:
            data["counted"] = True  # toward Limits.max_tool_calls, restored on resume
        await self._journal("tool_result", data)

    async def _defer(
        self,
        call: ToolCallPart,
        reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[str | None, StateUpdated | None]:
        """Atomically record changed state and a call waiting for an outside result.

        Return a signing refusal, or the committed state event when state changed.
        """
        waiting = PendingInput(
            tool_call_id=call.id,
            name=call.name,
            arguments=call.arguments,
            reason=reason,
            metadata=dict(metadata or {}),
        )
        try:
            data = waiting.model_dump(mode="json")
        except PydanticSerializationError as error:
            raise SessionError(
                f"The deferred call to {call.name!r} cannot be journaled: {error}. "
                "Give DeferToolCall(metadata=...) plain JSON values."
            ) from error
        try:
            waiting.token = self.permissions.sign(**self._pause_fields(waiting, "input"))
        except ValueError:  # a number or string outside canonical JSON's domain
            return prompts.UNSIGNABLE_CALL, None
        data["token"] = waiting.token
        self.waiting[call.id] = waiting
        self._running.discard(call.id)  # it waits now; it is not running
        self._call_emit.pop(call.id, None)
        self._call_tasks.pop(call.id, None)
        async with self._state_lock, self._batched():
            updated = await self._save_state_locked()
            await self._journal("tool_deferred", {"input": data})
        return None, updated

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

    async def _settle_open_calls(
        self,
        template: str,
        *,
        replacing: bool = False,
    ) -> list[ToolCallFinished]:
        """Give every open tool call of the last assistant message a result, then close the turn.

        ``template`` is for calls that never started. A call that started in
        an earlier process, or that this run started and stopped while it ran
        and that is not read-only, is reported with an unknown outcome
        (`TOOL_INTERRUPTED`): an idempotent tool may be repeated after a crash,
        but when it is cut short it may already have changed things. Only a
        read-only call is merely cancelled (`TOOL_CANCELLED`).
        Returns the ``tool_call_finished`` events of the calls it settled.
        """
        if not self.history:
            return []
        last = self.history[-1]
        if last.role != "assistant" or not last.tool_calls:
            return []
        settled: list[ToolCallFinished] = []
        for call in last.tool_calls:
            if call.id in self.results:
                continue
            held = self._held_calls.get(call.id)
            if held is None and self._has_intent(call.id):
                unresolved = self._unresolved_for(self._operation_id(call.id))
                held = unresolved[0] if unresolved else None
            if held is not None:
                if not replacing:
                    if not self._writer.lease.lost():
                        await self._defer_hold(call, held)
                    else:
                        self._held_calls[call.id] = held
                    continue
                result = _error_result(
                    call,
                    prompts.EFFECT_HELD_BY_NEW_RUN.format(
                        effect=held.effect,
                        key=held.key,
                    ),
                )
                await self._record(result, commit_provisional=False)
                settled.append(ToolCallFinished(call=call, result=result, agent=self.agent.name))
                continue
            tool = self.tools.get(call.name)
            # Only a read-only call is harmless when it is cut short. An idempotent one may
            # be repeated after a crash, but it may have changed things before it stopped.
            harmless = tool is not None and tool.read_only
            if call.id in self.started or (call.id in self._running and not harmless):
                wording = prompts.TOOL_INTERRUPTED
            elif call.id in self._running:
                wording = prompts.TOOL_CANCELLED
            elif call.id in self.waiting:
                wording = prompts.TOOL_INPUT_ABANDONED
            else:
                wording = template
            result = _error_result(call, wording.format(name=call.name))
            await self._record(result)
            settled.append(ToolCallFinished(call=call, result=result, agent=self.agent.name))
        if all(call.id in self.results for call in last.tool_calls):
            await self._close_tool_turn(last.tool_calls)
        return settled

    async def _close_tool_turn(self, calls: list[ToolCallPart]) -> None:
        await self._append(Message(role="tool", parts=[self.results[call.id] for call in calls]))
        self.results = {}
        self.started = set()
        self._running = set()
        self._call_emit = {}
        self._call_tasks = {}
        self.waiting = {}
        self.approvals = {}  # an answer applies only to the turn it was given for
        self.checkpoints = {}  # the turn's calls all have results: none of them repeats
        self._replay_state.open_turn_starts.clear()
        self._authorizations.clear()

    # -- plumbing ---------------------------------------------------------------------

    def _ctx(
        self, *, tool_call_id: str | None = None, emit: Callable[[Any], None] | None = None
    ) -> RunContext[Any]:
        if emit is not None:
            emit = _thread_safe(emit)  # a sync tool calls ctx.emit from its worker thread
        # run + position of the assistant message in the run + call: the
        # same after a resume, and unique even if a provider reuses call ids.
        operation_id = self._operation_id(tool_call_id) if tool_call_id else None
        return RunContext(
            deps=self.deps,
            run_id=self.run_id,
            session_id=self.session.id,
            agent_name=self.agent.name,
            turn=self.turn,
            tool_call_id=tool_call_id,
            operation_id=operation_id,
            usage=self.usage,
            messages=tuple(self.history),
            state=self.state,
            parent_run_id=self.parent.run_id if self.parent is not None else None,
            parent_tool_call_id=self.parent.tool_call_id if self.parent is not None else None,
            depth=self.depth,
            _emit=emit,
            _add_usage=self._usage_adder(operation_id),
            _run=self,
        )

    def _usage_adder(self, operation_id: str | None) -> Callable[[Usage], None]:
        """``ctx.add_usage`` for a context: a tool call's usage is journaled with its
        operation id, so a repeat of the call can tell what the journal holds of it."""
        if operation_id is None:
            return self._add_usage
        return functools.partial(self._add_usage, operation_id=operation_id)

    def _add_usage(self, usage: Usage, operation_id: str | None = None) -> None:
        with self._usage_lock:  # ctx.add_usage may come from a sync tool's worker thread
            if any(usage is absorbed for absorbed in self._absorbed):
                # A child run's result.usage, already added when the child ended.
                self._absorbed = [item for item in self._absorbed if item is not usage]
                return
            self.usage = self.usage.add(usage)
            self._usage_to_journal = self._usage_to_journal.add(usage)  # with the next entry
            if operation_id is not None:
                pending = self._usage_by_operation.get(operation_id, Usage())
                self._usage_by_operation[operation_id] = pending.add(usage)

    async def _save_state(self) -> StateUpdated | None:
        """Journal ``ctx.state`` if it changed since it was last journaled.

        Every entry holds the full values, and the lock keeps entries in the
        order of the changes, so concurrent tools cannot leave a stale state last.
        """
        async with self._state_lock:
            return await self._save_state_locked()

    async def _save_state_locked(self, *, provisional: str | None = None) -> StateUpdated | None:
        """Save a snapshot while the caller holds `_state_lock` through its write."""
        snapshot = _state_json(self.state)
        if snapshot == self._saved_state:
            return None
        data = {"values": json.loads(snapshot)}
        if provisional is not None:
            data.update(provisional=True, operation_id=provisional)
        await self._journal("state_updated", data)
        if provisional is None:
            self._committed_state = copying.deepcopy(data["values"])
        self._saved_state = snapshot
        return StateUpdated(values=json.loads(snapshot), agent=self.agent.name)

    def _journal_closed(self) -> bool:
        """Tools write nothing more to the journal: it is not open yet, the run is writing
        (or wrote) its ``run_finished`` or ``run_paused``, or it stopped (`_stopped`)."""
        return not self._journal_open or self._closed or self._finishing or self._stopped

    async def _save_state_now(self) -> StateUpdated | None:
        """``RunContext.save_state``: `_save_state`, unless the run's journal is closed."""
        if self._journal_closed():
            return None
        if self._scopes:
            async with self._state_lock:
                scope = next(iter(self._scopes.values()))
                return await self._save_state_locked(provisional=scope.operation_id)
        return await self._save_state()

    async def _checkpoint(self, tool_call_id: str, operation_id: str, key: str, value: Any) -> None:
        """``RunContext.checkpoint``: journal ``key = value`` for a running call, now."""
        if not isinstance(key, str):
            raise TypeError(f"A checkpoint key is a string, not a {type(key).__name__}.")
        if self._journal_closed() or not self._call_running(tool_call_id):
            raise RuntimeError(
                f"RunContext.checkpoint() works only while its tool call runs; call "
                f"{tool_call_id!r} is not running in run {self.run_id}."
            )
        try:
            text = json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise SessionError(
                f"Checkpoint {key!r} is not JSON: {error}. Store strings, numbers, booleans, "
                "None, lists and dicts with string keys."
            ) from error
        stored = json.loads(text)  # what a resumed attempt reads, so both attempts see the same
        data = {
            "tool_call_id": tool_call_id,
            "operation_id": operation_id,
            "key": key,
            "value": json.loads(text),  # its own object: the entry never changes with the run's
        }
        await self._journal("tool_checkpoint", data)  # before the tool acts on it
        self.checkpoints.setdefault(operation_id, {})[key] = stored

    # -- calls from outside the model (RunContext.call_tool) ---------------------------

    def _call_running(self, call_id: str | None) -> bool:
        """``call_id`` started and has no result yet: one of the model's calls, or a call
        from outside the model (served calls can serve calls in turn)."""
        return call_id is not None and (
            call_id in self._running or call_id in self._outside_running
        )

    def _outside_over(self, caller: str | None) -> bool:
        """The run takes no more calls from outside its model through ``caller``: it is
        cancelled, stopping (`_stopping`) or its journal is closed, or the calling tool call
        has ended or is being cancelled (its timeout, its group's cancel)."""
        return (
            self._journal_closed()
            or self._cancel.is_set()
            or not self._call_running(caller)
            or _cancelled_since(self._call_tasks.get(caller or ""))
            or self._stopping()
        )

    def _stopping(self) -> bool:
        """This run, or a run it is a child of, stopped or is stopping: cancelled with
        ``cancel()``, being cancelled from outside while it unwinds (its driving task), or
        done. A parent's cancel reaches a child run only as the cancel of the tool call it
        runs in, a moment later: the child takes no calls from outside its model meanwhile."""
        run: _Run | None = self
        while run is not None:
            if run._stopped or run._cancel.is_set() or _cancelled_since(run._driver):
                return True
            run = run.parent_run
        return False

    def _fail_from_outside(self, caller: str | None, error: Exception) -> None:
        """A hook or the journal raised during a call from outside the model: fail the run
        with it, as when that happens during one of the model's calls (`_run_tool`). The
        calling call's group gets it at once; otherwise the loop raises it at its next step."""
        if self._closed or self._finishing or self._stopped:
            return
        if self._outside_failure is None:
            self._outside_failure = error
        emit = self._call_emit.get(caller or "") or self._outside_running.get(caller or "")
        if emit is not None:
            emit(_Failure(error))

    async def _outside_call(
        self,
        origin: RunContext[Any],
        tool: Tool,
        arguments: dict[str, Any],
        *,
        source: str,
        approval_timeout_s: float | None,
        tags: dict[str, Any],
        metadata: dict[str, Any],
    ) -> ToolResultPart:
        """``RunContext.call_tool``: one call that did not come from the model, decided, run
        and journaled with the loop's own pieces (`_invalid_arguments`, `_authorize`,
        `_wait_for_stuck_threads`, the hooks, `_invoke`), as a call of this run."""
        if not isinstance(source, str) or not _SOURCE.match(source):
            raise ValueError(
                f"call_tool(source={source!r}): give a short name of 1-32 letters, digits, "
                "'_' or '-', such as 'mcp'."
            )
        if clash := sorted(set(tags) & _OUTSIDE_FIELDS):
            raise ValueError(f"call_tool(tags=...) cannot set {clash}: the entries have them.")
        for label, value in (("tags", tags), ("metadata", metadata)):
            if (problem := _not_json(value)) is not None:
                raise ValueError(f"call_tool({label}=...) must be JSON: {problem}")
        caller = origin.tool_call_id
        call = ToolCallPart(id=f"{source}-{uuid.uuid4().hex}", name=tool.name, arguments=arguments)
        emit = self._call_emit.get(caller or "") or self._outside_running.get(caller or "")
        if emit is None or self._outside_over(caller):  # nothing may be journaled for it now
            return _error_result(call, _outside_refusal(call, prompts.TOOL_SERVER_RUN_ENDED))
        operation_id = f"{origin.operation_id}:{call.id}"
        ctx = dataclasses.replace(
            self._ctx(tool_call_id=call.id),
            operation_id=operation_id,
            depth=origin.depth + 1,  # it comes from the agent that the calling call runs
            _emit=emit,
            _add_usage=self._usage_adder(operation_id),
        )
        journal_tags = {"source": source, "caller_tool_call_id": caller, **tags}
        try:
            return await self._outside_steps(
                ctx, tool, call, emit, journal_tags, metadata, approval_timeout_s
            )
        except Exception as error:  # a hook or the journal failed: the run fails
            self._fail_from_outside(caller, error)
            raise

    async def _outside_steps(
        self,
        ctx: RunContext[Any],
        tool: Tool,
        call: ToolCallPart,
        emit: Callable[[Any], None],
        tags: dict[str, Any],
        metadata: dict[str, Any],
        approval_timeout_s: float | None,
    ) -> ToolResultPart:
        caller = tags["caller_tool_call_id"]
        began = time.perf_counter()
        ended = _error_result(call, _outside_refusal(call, prompts.TOOL_SERVER_RUN_ENDED))
        refusal = _invalid_arguments(tool, call)  # before anyone is asked about it
        if refusal is None and tool.deferred:
            refusal = prompts.TOOL_SERVER_CANNOT_DEFER.format(name=call.name)
        budget = self.limits.max_tool_calls
        if refusal is None and budget is not None and len(self._counted) >= budget:
            refusal = prompts.TOOL_CALL_LIMIT.format(name=call.name, limit=budget)
        if refusal is not None:
            return await self._outside_record(call, _error_result(call, refusal), emit, tags, began)
        # The slot is taken before the call is decided, with no await in between: calls that
        # arrive together cannot all pass the check, and nobody is asked to approve a call
        # that could not run. A refused call gives it back.
        self._counted[call.id] = None
        try:
            verdict, payload = await self._authorize(
                tool,
                call,
                ctx=ctx,
                can_pause=False,
                approval_timeout_s=approval_timeout_s,
                tags=tags,
                still_wanted=lambda: not self._outside_over(caller),
            )
            if verdict == "run" and call.id in self._edited:
                # An approver's edit is checked like the caller's arguments.
                if (invalid := _invalid_arguments(tool, payload)) is not None:
                    verdict, payload = "invalid", invalid
        except BaseException:
            self._counted.pop(call.id, None)
            self._edited.discard(call.id)
            raise
        assert verdict != "pending"  # can_pause=False
        if verdict != "run" or self._outside_over(caller):
            self._counted.pop(call.id, None)
            self._edited.discard(call.id)
            if verdict == "ended" or self._outside_over(caller):
                return ended  # it ended while the call was decided: nothing more is written
            text = payload if verdict == "invalid" else _outside_refusal(call, payload)
            return await self._outside_record(call, _error_result(call, text), emit, tags, began)
        try:
            return await self._outside_execute(ctx, tool, payload, emit, tags, metadata, began)
        finally:
            self._edited.discard(call.id)

    async def _outside_execute(
        self,
        ctx: RunContext[Any],
        tool: Tool,
        call: ToolCallPart,
        emit: Callable[[Any], None],
        tags: dict[str, Any],
        metadata: dict[str, Any],
        began: float,
    ) -> ToolResultPart:
        """Run an allowed call from outside the model: kept apart from the other such calls
        (``Limits.parallel_tools``) and from abandoned sync threads, journaled before and
        after, through the hooks; checked again right before the tool runs."""
        caller: str = tags["caller_tool_call_id"]
        ended = _error_result(call, _outside_refusal(call, prompts.TOOL_SERVER_RUN_ENDED))
        gate = self._outside_gates.get(caller)
        if gate is None:  # one per calling call: a served call can serve calls in turn
            gate = _OutsideGate(self.limits)
            self._outside_gates[caller] = gate
        started = False
        try:
            async with gate.hold(self._runs_alone(tool)):
                blocked = await self._wait_for_stuck_threads(tool, call)
                if self._outside_over(caller):  # it waited, and the run or its caller ended
                    self._counted.pop(call.id, None)
                    return ended
                if blocked is not None:
                    self._counted.pop(call.id, None)  # it never ran, so it does not count
                    return await self._outside_record(call, blocked, emit, tags, began)
                await self._journal(
                    "tool_started", {"tool_call_id": call.id, "name": call.name, **tags}
                )
                started = True
                self._outside_running[call.id] = emit
                self._call_tasks[call.id] = _cancel_mark(asyncio.current_task())
                emit(ToolCallStarted(call=call, agent=self.agent.name))
                hooked = await self.hooks.before_tool(ctx, call)
                if (updated := await self._save_state_now()) is not None:
                    emit(updated)  # a change made in before_tool is journaled before the tool
                if isinstance(hooked, ToolResultPart):
                    result = await self.hooks.after_tool(ctx, call, hooked)
                elif self._outside_over(caller):
                    # The start, before_tool and the state were awaited since the call was let
                    # in: the run or its caller ended meanwhile, so the tool does not run.
                    result = ended
                else:
                    try:
                        result = await self._invoke(ctx, tool, hooked)
                    except DeferToolCall:
                        content = prompts.TOOL_SERVER_CANNOT_DEFER.format(name=call.name)
                        result = _error_result(call, content)
                    result = await self.hooks.after_tool(ctx, call, result)
                if result is not ended:
                    update: dict[str, Any] = {"tool_call_id": call.id, "name": call.name}
                    if metadata:
                        update["metadata"] = {**result.metadata, **metadata}
                    result = result.model_copy(update=update)
                    if call.id in self._edited:  # tell the caller what actually ran
                        note = prompts.APPROVAL_ARGUMENTS_CHANGED.format(
                            arguments=json.dumps(call.arguments, ensure_ascii=False, default=str)
                        )
                        content = prompts.APPROVAL_ARGUMENTS_CHANGED_RESULT.format(
                            note=note, content=result.content
                        )
                        result = result.model_copy(update={"content": content})
                finished: list[Any] = []
                async with self._state_lock, self._batched():
                    updated = None if self._journal_closed() else await self._save_state_locked()
                    result = await self._outside_record(call, result, finished.append, tags, began)
                if updated is not None:
                    emit(updated)
                for event in finished:
                    emit(event)
                return result
        except asyncio.CancelledError:
            if started:  # D1: only a read-only call is harmless when it is cut short
                wording = prompts.TOOL_CANCELLED if tool.read_only else prompts.TOOL_INTERRUPTED
            else:
                self._counted.pop(call.id, None)  # it never ran, so it does not count
                wording = prompts.TOOL_CANCELLED
            stopped = _error_result(call, wording.format(name=call.name))
            try:
                await self._outside_record(call, stopped, emit, tags, began)
            except Exception as error:
                self._fail_from_outside(caller, error)
            raise
        finally:
            self._outside_running.pop(call.id, None)
            self._call_tasks.pop(call.id, None)

    async def _outside_record(
        self,
        call: ToolCallPart,
        result: ToolResultPart,
        emit: Callable[[Any], None],
        tags: dict[str, Any],
        began: float,
    ) -> ToolResultPart:
        """Journal the result of a call from outside the model (never into ``self.results``:
        no assistant message has it) and announce it. Nothing is written once the run's
        journal is closed: a later resume would count the entry."""
        if not self._journal_closed():
            data: dict[str, Any] = {"result": result.model_dump(mode="json"), **tags}
            if call.id in self._counted:
                data["counted"] = True  # toward Limits.max_tool_calls, restored on resume
            await self._journal("tool_result", data)
        emit(
            ToolCallFinished(
                call=call,
                result=result,
                duration_ms=(time.perf_counter() - began) * 1000,
                agent=self.agent.name,
            )
        )
        return result

    async def _journal(self, kind: str, data: dict[str, Any], *, run_id: str | None = None) -> None:
        if kind in _USAGE_KINDS and run_id is None:
            with self._usage_lock:
                added, self._usage_to_journal = self._usage_to_journal, Usage()
                by_operation, self._usage_by_operation = self._usage_by_operation, {}
            if added != Usage():
                # Usage from ctx.add_usage (a compactor's summary, a sub-agent) rides on the
                # next entry of the run, so a resume restores it (derive_state sums it).
                data = {**data, "added_usage": added.model_dump(mode="json")}
            if by_operation:
                # Which tool calls added it: a repeat of such a call learns what the journal
                # holds of its usage (RunContext.journaled_usage), whatever entry carried it.
                data = {**data, "added_usage_by_operation": _usage_dumps(by_operation)}
        entry = Entry(kind=kind, data=data, run_id=run_id or self.run_id)
        batch = self._current_batch()
        if batch is not None:
            batch.append(entry)
        else:
            await self._write([entry])

    async def _write(self, entries: list[Entry]) -> None:
        try:
            if self._writer is not None:
                await self._writer.append(entries)
            else:
                await self.session.append_entries(entries)
        except BaseException as error:
            # A lease refusal guarantees no write. Other failures can follow a commit.
            self._unwritten(entries, definite=isinstance(error, LeaseError))
            raise
        for entry in entries:  # written: the journal holds this usage of these calls now
            for operation_id, usage in _usage_loads(entry.data).items():
                held = self.journaled_usage.get(operation_id, Usage())
                self.journaled_usage[operation_id] = held.add(usage)

    def _unwritten(self, entries: list[Entry], definite: bool = True) -> None:
        """Keep added usage only when the entries definitely did not commit."""
        if not definite:
            return
        for entry in entries:
            if "added_usage" in entry.data:
                added = Usage.model_validate(entry.data["added_usage"])
                with self._usage_lock:
                    self._usage_to_journal = self._usage_to_journal.add(added)
            for operation_id, usage in _usage_loads(entry.data).items():
                with self._usage_lock:
                    pending = self._usage_by_operation.get(operation_id, Usage())
                    self._usage_by_operation[operation_id] = pending.add(usage)

    @contextlib.asynccontextmanager
    async def _batched(self) -> AsyncIterator[None]:
        """Collect the entries journaled inside the block and write them in one store call.

        Every store writes one ``append`` atomically, so a crash leaves all of
        them or none. If the block raises, none is written. Nested blocks join
        the outer one in the same task. Other tasks write independently. Do not
        yield events inside the block.
        """
        if self._current_batch() is not None:
            yield
            return
        entries: list[Entry] = []
        token = self._batch.set((asyncio.current_task(), entries))
        try:
            yield
        except BaseException:
            self._unwritten(entries)
            raise
        finally:
            self._batch.reset(token)
        if entries:
            await self._write(entries)

    def _current_batch(self) -> list[Entry] | None:
        batch = self._batch.get()
        # Child tasks inherit ContextVars, but must not inherit a parent's mutable batch.
        if batch is not None and batch[0] is asyncio.current_task():
            return batch[1]
        return None

    async def _append(
        self, message: Message, *, new: bool = True, extra: Mapping[str, Any] | None = None
    ) -> None:
        self.history.append(message)
        self._ordinal += 1
        if new:
            self.new_messages.append(message)
        await self._journal(
            "message", {"message": message.model_dump(mode="json"), **(extra or {})}
        )

    async def _sleep(self, seconds: float) -> None:
        """Wait, but wake up at once if the run is cancelled."""
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._cancel.wait(), timeout=seconds)

    async def _unless_cancelled(self, start: Callable[[], Awaitable[T]]) -> tuple[bool, T | None]:
        """Await ``start()`` in this task, unless the run is cancelled before or meanwhile.

        Returns ``(True, value)``, or ``(False, None)`` when the run's cancel (or
        ``timeout_s``) came first: the awaitable is then interrupted with a
        ``CancelledError`` at its current ``await``, in this task, as a timeout
        would interrupt it. A cancellation of this task from outside passes through.
        """
        if self._cancel.is_set():
            return False, None
        task = asyncio.current_task()
        assert task is not None
        before = task.cancelling()
        self._interruptible = task
        try:
            value = await start()
        except (asyncio.CancelledError, Exception):
            if not self._interrupting:
                raise
            # Interrupted by cancel(): whatever the interrupted code raised for it.
            self._interrupting = False
            if task.uncancel() > before:
                raise  # cancelled from outside as well
            return False, None
        finally:
            self._interruptible = None
        if self._interrupting:  # cancel() came as the step finished: withdraw the request
            self._interrupting = False
            task.uncancel()
        return True, value

    async def _until_cancelled(self, source: AsyncIterator[T]) -> AsyncIterator[T]:
        """Yield from ``source`` until it ends or the run is cancelled, then close it.

        Every step of ``source`` runs in the task that iterates this generator,
        as when ``source`` is iterated directly, so a model stream's own
        ``asyncio.timeout``, context variables and cancel scopes work.
        """
        try:
            while True:
                try:
                    stepped, item = await self._unless_cancelled(source.__anext__)
                except StopAsyncIteration:
                    return
                if not stepped:
                    return
                yield item  # type: ignore[misc]
        finally:
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
            held_effects=list(
                {
                    item.id: item
                    for item in (
                        (self._writer.view().held(self.session.id) if self._writer else [])
                        + list(self._held_calls.values())
                    )
                }.values()
            ),
            guardrail=self.guardrail,
        )
        result._session = self.session
        self._result_usage = result.usage
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


_ABANDONED: dict[str, Any] = {
    "status": "abandoned",
    "stop_reason": None,
    "output": None,
    "text": "",
}
"""``run_finished`` of a run that a later run (or a resume that cannot continue it) closed."""


def _finished_data(result: RunResult) -> dict[str, Any]:
    """``run_finished`` data, enough to rebuild the run's `RunResult` from the journal."""
    try:
        output = to_jsonable_python(result.output)
        json.dumps(output)
    except (PydanticSerializationError, TypeError, ValueError):
        output = None  # not JSON: the journal keeps the messages, which hold the answer
    return {
        "status": result.status,
        "stop_reason": result.stop_reason,
        "output": output,
        "text": result.text,
    }


def _failure_data(error: BaseException) -> dict[str, Any]:
    """``run_failed`` data: the exception's type and its (shortened) message, nothing more."""
    message = _brief(error, 500)
    return {
        "error_type": type(error).__name__,
        "message": message,
        "retryable": bool(getattr(error, "retryable", False)),
        "ambiguous": bool(getattr(error, "ambiguous", False)),
        "error": f"{type(error).__name__}: {message}",
    }


def _fenced_by_runtime(store: Any) -> bool:
    """True for `AgentRuntime`'s per-run store view, whose lease already fences every write."""
    runtime = sys.modules.get("qagents.runtime")
    leased = getattr(runtime, "_LeasedStore", None)
    return isinstance(leased, type) and isinstance(store, leased)


def _append_overridden_alone(store: Any) -> bool:
    """True for a store subclass that overrides ``append`` but not ``append_leased``.

    Such a subclass (an audit mirror, encryption, fault injection) expects every
    entry to pass through its ``append``, which the built-in ``append_leased``
    does not call. A plain run then writes through ``append`` and takes no
    lease, so only runs in this process are kept apart.
    """
    mro = type(store).__mro__
    append = next((cls for cls in mro if "append" in cls.__dict__), None)
    leased = next((cls for cls in mro if "append_leased" in cls.__dict__), None)
    return (
        append is not None
        and leased is not None
        and append is not leased
        and (issubclass(append, leased))
    )


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


def _unwrap_value(data: Any) -> Any:
    """The answer inside ``{"value": ...}``, the wrapper of a non-object native schema.

    A bare value is returned as it is, for models that answer without the wrapper.
    """
    if isinstance(data, dict) and set(data) == {"value"}:
        return data["value"]
    return data


def _is_object_schema(schema: Mapping[str, Any]) -> bool:
    """The schema's root is a plain object, as tool arguments and strict response formats need."""
    return schema.get("type") == "object" and "$ref" not in schema


def _reachable_agents(root: Agent) -> list[Agent]:
    """``root`` and every agent reachable from it through handoffs, each once, nearest first."""
    seen: set[int] = set()
    found: list[Agent] = []
    queue = [root]
    while queue:
        agent = queue.pop(0)
        if id(agent) in seen:
            continue
        seen.add(id(agent))
        found.append(agent)
        queue.extend(agent.handoffs)
    return found


def _cancelled_from_outside() -> bool:
    """The current task is being cancelled (a real cancellation, not a stray ``CancelledError``)."""
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def _usage_dumps(usage: Mapping[str, Usage]) -> dict[str, Any]:
    """``{operation id: usage}`` as an entry's ``added_usage_by_operation``."""
    return {operation_id: item.model_dump(mode="json") for operation_id, item in usage.items()}


def _cancel_mark(task: asyncio.Task[Any] | None) -> tuple[asyncio.Task[Any], int] | None:
    """``task`` with its ``cancelling()`` count now, for `_cancelled_since`."""
    return (task, task.cancelling()) if task is not None else None


def _cancelled_since(mark: tuple[asyncio.Task[Any], int] | None) -> bool:
    """The task of ``mark`` has been asked to cancel since it was marked (and has not taken
    the request back). Relative to the mark: a task that was already being cancelled when
    it started a run (cleanup code that runs an agent) does not count as cancelling it."""
    return mark is not None and mark[0].cancelling() > mark[1]


async def _threads_finished(
    threads: list[concurrent.futures.Future[Any]], timeout: float | None
) -> bool:
    """Wait up to ``timeout`` seconds (``None``: no limit) for ``threads`` to return."""
    waiters = [asyncio.wrap_future(thread) for thread in threads]
    try:
        _, pending = await asyncio.wait(waiters, timeout=timeout)
    finally:
        for waiter in waiters:
            if waiter.done() and not waiter.cancelled():
                waiter.exception()  # retrieved: the earlier call's error was reported already
            else:
                waiter.cancel()  # the thread already runs: cancelling its waiter cannot stop it
    return not pending


def _thread_safe(emit: Callable[[Any], None]) -> Callable[[Any], None]:
    """``emit``, callable from any thread: from another one, it goes through the loop."""
    loop = asyncio.get_running_loop()
    owner = threading.get_ident()

    def safe(event: Any) -> None:
        if threading.get_ident() == owner:
            emit(event)
            return
        with contextlib.suppress(RuntimeError):  # the loop is closed: a thread abandoned late
            loop.call_soon_threadsafe(emit, event)

    return safe


def _invalid_arguments(tool: Tool, call: ToolCallPart) -> str | None:
    """The error result for a call whose arguments do not match the tool, or ``None``."""
    try:
        tool.validate(call.arguments)
    except ToolArgumentError as error:
        return prompts.INVALID_ARGUMENTS.format(name=call.name, error=error)
    return None


def _error_result(call: ToolCallPart, content: str) -> ToolResultPart:
    return ToolResultPart(tool_call_id=call.id, name=call.name, content=content, is_error=True)


def _outside_refusal(call: ToolCallPart, reason: str) -> str:
    """The result text of a call from outside the model that was not run (`TOOL_SERVER_REFUSED`)."""
    return prompts.TOOL_SERVER_REFUSED.format(name=call.name, reason=reason)


def _not_json(value: Any) -> str | None:
    """``"<error type>: <message>"`` when ``value`` cannot be journaled as JSON, else ``None``."""
    try:
        json.dumps(to_jsonable_python(value), allow_nan=False)
    except (PydanticSerializationError, TypeError, ValueError) as error:
        return prompts.TOOL_FAILED_EXCEPTION.format(type=type(error).__name__, error=error)
    return None


class _OutsideGate:
    """``Limits.parallel_tools`` for the calls from outside the model that come through one
    calling call: calls that run alone wait for every other such call, the others run
    together, at most ``max_parallel_tools`` at a time. (They never wait for the calling
    call itself, which waits for them.)"""

    def __init__(self, limits: Limits) -> None:
        self._slots = asyncio.Semaphore(max(1, limits.max_parallel_tools))
        self._readers = 0
        self._writer = False
        self._changed = asyncio.Event()

    @contextlib.asynccontextmanager
    async def hold(self, alone: bool) -> AsyncIterator[None]:
        async with self._slots:
            while True:
                if alone and not self._writer and self._readers == 0:
                    self._writer = True
                    break
                if not alone and not self._writer:
                    self._readers += 1
                    break
                self._changed.clear()
                await self._changed.wait()
            try:
                yield
            finally:  # no await here: a cancelled call always gives its place back
                if alone:
                    self._writer = False
                else:
                    self._readers -= 1
                self._changed.set()


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


def _brief(error: BaseException, limit: int = 1500) -> str:
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
