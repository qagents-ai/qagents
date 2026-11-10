"""The agent and its loop.

The loop is a small state machine over the conversation history. Before every
step it looks at the last message:

* an assistant message with tool calls that have no results -> run the tools;
* an assistant message without tool calls -> that is the final answer;
* anything else (a user message, tool results) -> ask the model.

Because the next step depends only on the history, and the history is written
to the session journal as it grows, a run can stop anywhere (a pause for
approval, a crash, a deploy) and continue later from exactly that point.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, TypeAdapter, ValidationError

from . import prompts
from .context import Compactor
from .errors import (
    ConfigurationError,
    ModelError,
    ModelResponseError,
    OutputValidationError,
    SessionError,
    ToolArgumentError,
    ToolError,
)
from .events import (
    AgentSwitched,
    ApprovalRequired,
    Event,
    ModelResponded,
    RunFinished,
    RunStarted,
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
from .permissions import Approval, Permissions
from .results import RunResult, RunStatus
from .session import PendingApproval, Session
from .tools import RunContext, Tool, Toolkit, render_tool_output
from .tools import tool as make_tool

logger = logging.getLogger("qagents")

T = TypeVar("T")
Instructions = str | Callable[[RunContext[Any]], Any] | None
"""Fixed text, or a function of the run context (sync or async) that returns it."""
Prompt = str | Message | list[Part] | None


class Limits(BaseModel):
    """Hard stops for a run. ``None`` switches a limit off."""

    max_turns: int | None = 30
    """Model requests per run."""
    max_total_tokens: int | None = None
    """Input plus output tokens over the whole run."""
    max_cost_usd: float | None = None
    """Needs a model that reports cost (QCloud does, from its catalog prices)."""
    timeout_s: float | None = None
    """Wall-clock seconds for the whole run."""
    tool_timeout_s: float | None = 120.0
    """Seconds per tool call, unless the tool sets its own ``timeout``."""
    parallel_tools: Literal["read_only", "all", "none"] = "read_only"
    """Which tool calls of one turn run concurrently: only tools declared
    ``read_only`` (the rest run one at a time, in order), all, or none."""
    max_parallel_tools: int = 8
    max_tool_result_chars: int | None = 60_000
    """Longer tool results are shortened in the middle before the model sees them."""
    output_retries: int = 2
    """Extra attempts when the final answer does not match ``output_type``."""
    model_retries: int = 2
    """Extra attempts for model errors that are safe to repeat."""
    retry_ambiguous: bool = False
    """Also repeat requests the provider may already have processed (and billed)."""


class Agent:
    """A model, instructions and tools, run in a loop until the model answers.


        agent = Agent("qcloud:<model-id>", instructions="Be brief.", tools=[get_weather])
        result = await agent.run("What is the weather in Athens?")
        print(result.output)

    The agent you call ``run`` on owns the run's ``limits``, ``permissions``
    and ``hooks``. After a handoff, the target agent's model, instructions,
    tools and ``output_type`` take over.
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
        output_mode: Literal["native", "prompted"] = "native",
        output_strict: bool = False,
        settings: ModelSettings | None = None,
        limits: Limits | None = None,
        permissions: Permissions | None = None,
        hooks: Iterable[Hooks] = (),
        handoffs: Iterable[Agent] = (),
        compactor: Compactor | None = None,
    ) -> None:
        self.model = resolve_model(model)
        self.name = name
        self.description = description
        self.instructions = instructions
        self.tools: list[Tool | Toolkit] = _normalize_tools(tools)
        self.output_type = None if output_type is str else output_type
        self.output_mode = output_mode
        """``"native"`` sends the schema as ``response_format``; ``"prompted"``
        writes it into the instructions, for models without schema support."""
        self.output_strict = output_strict
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
    ) -> RunStream:
        """Start a run and return its event stream.

        ``session`` continues an existing conversation. ``history`` seeds a new
        one with messages you kept yourself.
        """
        return RunStream(
            _Run(
                self, prompt=prompt, deps=deps, session=session, history=history,
                limits=limits, settings=settings, approvals=None, resume=False,
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
    ) -> RunResult:
        """Run until the agent answers, pauses or hits a limit."""
        stream = self.stream(
            prompt, deps=deps, session=session, history=history, limits=limits, settings=settings
        )
        return await stream.wait()

    def run_sync(self, prompt: Prompt = None, **options: Any) -> RunResult:
        """``run`` for code that is not async. Do not call it from a running event loop."""
        return asyncio.run(self.run(prompt, **options))

    def resume_stream(
        self,
        session: Session,
        *,
        approvals: Mapping[str, Approval | bool] | None = None,
        deps: Any = None,
        limits: Limits | None = None,
        settings: ModelSettings | None = None,
    ) -> RunStream:
        """Continue the session's unfinished run and return its event stream."""
        return RunStream(
            _Run(
                self, prompt=None, deps=deps, session=session, history=None,
                limits=limits, settings=settings, approvals=approvals, resume=True,
            )
        )  # fmt: skip

    async def resume(
        self,
        session: Session,
        *,
        approvals: Mapping[str, Approval | bool] | None = None,
        deps: Any = None,
        limits: Limits | None = None,
        settings: ModelSettings | None = None,
    ) -> RunResult:
        """Continue the session's unfinished run: after a pause, a crash or an error.

        ``approvals`` maps tool call ids from ``result.pending_approvals`` to
        ``True``/``False`` or an `Approval`. Tool calls that already
        have a recorded result are never run again.
        """
        stream = self.resume_stream(
            session, approvals=approvals, deps=deps, limits=limits, settings=settings
        )
        return await stream.wait()

    # -- composition -------------------------------------------------------------

    def as_tool(
        self, *, name: str | None = None, description: str | None = None, read_only: bool = False
    ) -> Tool:
        """Expose this agent as a tool another agent can call.

        The sub-agent works in its own fresh conversation: it sees only the
        task text, and the caller sees only its final answer. Its usage is
        added to the caller's run and its events are forwarded as
        `SubagentEvent`.
        """
        agent = self

        async def delegate(ctx: RunContext[Any], task: str) -> str:
            stream = agent.stream(task, deps=ctx.deps)
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
        built.parameters["properties"]["task"]["description"] = "The complete task, self-contained."
        return built

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
        self.permissions = agent.permissions
        self.hooks = HookChain(agent.hooks)
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
        self._last_response: ModelResponse | None = None
        self._ordinal = 0
        """Messages this run has journaled; names tool calls in operation ids."""
        self._output_attempts = 0
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
            await self._prepare()
            yield RunStarted(
                run_id=self.run_id,
                session_id=self.session.id,
                agent=self.agent.name,
                resumed=self.resume,
            )
            await self.hooks.on_run_start(self._ctx())
            if self.limits.timeout_s is not None:
                timer = asyncio.get_running_loop().call_later(
                    self.limits.timeout_s, self.cancel, "timeout"
                )

            status: RunStatus = "completed"
            stop_reason: str | None = None
            while True:
                if self._cancel.is_set():
                    await self._settle_open_calls(prompts.TOOL_CANCELLED)
                    if self._cancel_reason == "timeout":
                        status, stop_reason = "limit_reached", "timeout"
                    else:
                        status, stop_reason = "cancelled", self._cancel_reason
                    break
                last = self.history[-1]
                if last.role == "assistant" and last.tool_calls:
                    async with contextlib.aclosing(self._tool_phase(last)) as phase:
                        async for event in phase:
                            yield event
                    if self.pending:
                        status = "paused"
                        break
                    continue
                if last.role == "assistant":
                    if await self._finalize(last):
                        truncated = self._last_response is not None and (
                            self._last_response.finish_reason == "length"
                        )
                        stop_reason = "max_output_tokens" if truncated else None
                        break
                    continue
                stop_reason = self._limit_reached()
                if stop_reason is not None:
                    status = "limit_reached"
                    break
                self.turn += 1
                yield TurnStarted(turn=self.turn, agent=self.agent.name)
                async with contextlib.aclosing(self._model_phase()) as phase:
                    async for event in phase:
                        yield event

            if status == "paused":
                await self._journal(
                    "run_paused", {"pending": [item.model_dump() for item in self.pending]}
                )
            else:
                await self._journal("run_finished", {"status": status, "stop_reason": stop_reason})
            result = self._result(status, stop_reason)
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
        finally:
            if timer is not None:
                timer.cancel()

    async def _prepare(self) -> None:
        state = await self.session.state()
        self.history = list(state.messages)
        if self.resume:
            if state.open_run_id is None or not self.history:
                raise SessionError(
                    f"Session {self.session.id!r} has no unfinished run to resume. "
                    "Start a new one with agent.run()."
                )
            if state.agent and state.agent != self.root.name:
                self.agent = self._find_agent(state.agent)
            await self._load_agent()
            self.run_id = state.open_run_id
            self._journal_open = True
            self._ordinal = state.run_messages
            self.results = dict(state.tool_results)
            self.started = set(state.started_tools)
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
        await self._journal("run_started", {"agent": self.root.name})
        self._journal_open = True
        for message in self.seed:
            await self._append(message, new=False)
        # A previous run may have stopped with tool calls still open. Close
        # them so the conversation stays valid for the provider.
        self.results = dict(state.tool_results)
        self.started = set(state.started_tools)
        await self._settle_open_calls(prompts.TOOL_NOT_EXECUTED)
        if self.prompt is not None:
            await self._append(_as_message(self.prompt))

    async def _load_agent(self) -> None:
        self.tools = await self.agent._collect_tools()
        self.handoffs = {target._handoff_tool_name: target for target in self.agent.handoffs}
        clash = set(self.tools) & set(self.handoffs)
        if clash:
            raise ConfigurationError(
                f"Agent {self.agent.name!r}: tool name(s) {sorted(clash)} collide with handoffs."
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
        if (
            limits.max_cost_usd is not None
            and self.usage.cost_usd is not None
            and self.usage.cost_usd >= limits.max_cost_usd
        ):
            return "max_cost_usd"
        return None

    # -- model phase ----------------------------------------------------------------

    async def _model_phase(self) -> AsyncIterator[Event]:
        ctx = self._ctx()
        await self._maybe_compact(ctx)
        request = await self.hooks.before_model(ctx, await self._build_request(ctx))
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
        self.usage = self.usage.add(response.usage)
        self._last_response = response
        message = response.message.model_copy(update={"name": self.agent.name})
        await self._append(message)
        yield ModelResponded(response=response, agent=self.agent.name)

    async def _build_request(self, ctx: RunContext[Any]) -> ModelRequest:
        agent = self.agent
        system = await agent._render_instructions(ctx)
        output: OutputSpec | None = None
        if agent._output_schema is not None:
            if agent.output_mode == "prompted":
                schema = json.dumps(agent._output_schema, ensure_ascii=False)
                addition = prompts.OUTPUT_INSTRUCTIONS.format(schema=schema)
                system = f"{system}\n\n{addition}" if system else addition
            else:
                output = OutputSpec(json_schema=agent._output_schema, strict=agent.output_strict)
        messages = ([Message.system(system)] if system else []) + list(self.history)
        specs = [tool.spec for tool in self.tools.values()]
        specs.extend(target._handoff_spec() for target in self.handoffs.values())
        return ModelRequest(
            messages=messages,
            tools=specs,
            settings=agent.settings.merged(self.settings),
            output=output,
        )

    async def _maybe_compact(self, ctx: RunContext[Any]) -> None:
        compactor = self.root.compactor
        if compactor is None or self._last_response is None:
            return
        replacement = await compactor.compact(
            ctx, list(self.history), self._last_response, self.agent.model
        )
        if replacement:
            self.history = list(replacement)
            await self._journal(
                "compaction",
                {"messages": [message.model_dump(mode="json") for message in replacement]},
            )

    async def _finalize(self, last: Message) -> bool:
        """Accept ``last`` as the final answer, or ask the model to correct it."""
        adapter = self.agent._output_adapter
        if adapter is None:
            self.output = last.text
            return True
        try:
            self.output = adapter.validate_python(extract_json(last.text))
            return True
        except (ValueError, ValidationError) as error:
            if self._output_attempts >= self.limits.output_retries:
                raise OutputValidationError(
                    f"The final answer did not match the output type after "
                    f"{self._output_attempts + 1} attempts: {error}"
                ) from error
            self._output_attempts += 1
            await self._append(Message.user(prompts.OUTPUT_RETRY.format(error=_brief(error))))
            return False

    # -- tool phase -------------------------------------------------------------------

    async def _tool_phase(self, assistant: Message) -> AsyncIterator[Event]:
        calls = assistant.tool_calls
        runnable: list[tuple[ToolCallPart, Tool]] = []
        pending: list[PendingApproval] = []
        handoff: ToolCallPart | None = None

        for call in calls:
            if call.id in self.results:
                continue
            refusal: str | None = None
            if call.parse_error is not None:
                refusal = prompts.INVALID_ARGUMENTS_JSON.format(
                    name=call.name, error=call.parse_error
                )
            elif call.name in self.handoffs:
                if handoff is None:
                    handoff = call
                    continue
                refusal = prompts.HANDOFF_DUPLICATE
            elif call.name not in self.tools:
                available = ", ".join([*self.tools, *self.handoffs]) or "none"
                refusal = prompts.UNKNOWN_TOOL.format(name=call.name, available=available)
            else:
                tool = self.tools[call.name]
                if call.id in self.started and not (tool.read_only or tool.idempotent):
                    refusal = prompts.TOOL_INTERRUPTED.format(name=call.name)
                else:
                    verdict, payload = await self._authorize(tool, call)
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

        if pending:
            self.pending = pending
            for item in pending:
                yield ApprovalRequired(approval=item, agent=self.agent.name)
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
                    result = await self._invoke(ctx, tool, hooked)
                result = await self.hooks.after_tool(ctx, call, result)
                result = result.model_copy(update={"tool_call_id": call.id, "name": call.name})
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
        """Run a tool and turn whatever happens into a result the model can read."""
        metadata: dict[str, Any] = {}
        is_error = True
        try:
            value = await tool.run(ctx, call.arguments, timeout=self.limits.tool_timeout_s)
            content = render_tool_output(value)
            is_error = False
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
        await self._journal("tool_result", {"result": result.model_dump(mode="json")})

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
            wording = prompts.TOOL_INTERRUPTED if call.id in self.started else template
            await self._record(_error_result(call, wording.format(name=call.name)))
        await self._close_tool_turn(last.tool_calls)

    async def _close_tool_turn(self, calls: list[ToolCallPart]) -> None:
        await self._append(Message(role="tool", parts=[self.results[call.id] for call in calls]))
        self.results = {}
        self.started = set()

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
            _emit=emit,
            _add_usage=self._add_usage,
        )

    def _add_usage(self, usage: Usage) -> None:
        self.usage = self.usage.add(usage)

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

    def _result(self, status: RunStatus, stop_reason: str | None) -> RunResult:
        answer = next((m for m in reversed(self.history) if m.role == "assistant"), None)
        result = RunResult(
            run_id=self.run_id,
            session_id=self.session.id,
            agent=self.agent.name,
            status=status,
            stop_reason=stop_reason,
            output=self.output if status == "completed" else None,
            text=answer.text if answer is not None else "",
            messages=list(self.history),
            new_messages=list(self.new_messages),
            usage=self.usage,
            turns=self.turn,
            pending_approvals=list(self.pending),
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
