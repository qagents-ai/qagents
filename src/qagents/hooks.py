"""Hooks: your code at fixed points of the agent loop.

Subclass `Hooks` and override what you need. Every method is async.
A hook that raises stops the run with that exception: hooks are part of your
control path, so a broken one fails closed instead of being skipped.

The loop, with its hook points:

    on_run_start
    repeat:
        before_model  -> model request ->  after_model
        for each tool call:
            before_tool  -> tool ->  after_tool
    on_run_end

Hooks may read and change ``ctx.state``. A change made in ``before_model``,
``after_model``, ``before_tool`` or ``after_tool`` is journaled right after the
hook returns; a change made in ``on_run_start`` or ``on_event`` is journaled at
the next of those points or when the run stops. Changes made in
``on_run_end`` are not saved.

Calls of a tool declared ``deferred=True`` pass the permission check but run
no tool hooks. A call whose tool raised `DeferToolCall` ran ``before_tool``
but has no result yet, so ``after_tool`` does not run for it; a result
supplied later with ``agent.resume(results=...)`` is recorded as given.

Child runs (an agent used as a tool, or any run started with ``parent=ctx``)
use their own hooks plus the hooks of the parent run whose class sets
``propagate = True``. A tracer attached once to the top agent then sees the
whole tree of runs; ``ctx.parent_run_id``, ``ctx.parent_tool_call_id`` and
``ctx.depth`` tell it where each run belongs. A hook object is used at most
once per run, even when both agents list it.

The final-answer call of ``output_mode="tool"`` is not a tool and runs no
tool hooks.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar

from .messages import ToolCallPart, ToolResultPart
from .models.base import ModelRequest, ModelResponse
from .tools import RunContext

if TYPE_CHECKING:
    from .results import RunResult


class Hooks:
    """Override the methods you need; the defaults do nothing."""

    propagate: ClassVar[bool] = False
    """Also apply this hook to child runs started from the run it is attached to,
    at any depth. Set it on tracers, loggers and cost meters."""

    async def on_run_start(self, ctx: RunContext[Any]) -> None:
        """The run is about to start."""

    async def before_model(
        self, ctx: RunContext[Any], request: ModelRequest
    ) -> ModelRequest | None:
        """Inspect or replace the exact request about to be sent to the model."""

    async def after_model(
        self, ctx: RunContext[Any], response: ModelResponse
    ) -> ModelResponse | None:
        """Inspect or replace the model's response before the loop acts on it."""

    async def before_tool(
        self, ctx: RunContext[Any], call: ToolCallPart
    ) -> ToolCallPart | ToolResultPart | None:
        """Runs after the call was authorised, right before the tool.

        Return a ``ToolCallPart`` to change the arguments, or a
        ``ToolResultPart`` to answer without running the tool.
        """

    async def after_tool(
        self, ctx: RunContext[Any], call: ToolCallPart, result: ToolResultPart
    ) -> ToolResultPart | None:
        """Inspect or replace a tool result before the model sees it."""

    async def on_event(self, ctx: RunContext[Any], event: Any) -> None:
        """Every event of the run, in order (the same ones ``agent.stream`` yields)."""

    async def on_run_end(
        self, ctx: RunContext[Any], result: RunResult | None, error: BaseException | None
    ) -> None:
        """The run finished (``result``), failed or was stopped (``error``).

        ``error`` is the exception that failed the run, or the
        ``asyncio.CancelledError`` or ``GeneratorExit`` that stopped it (a
        cancelled task, or a stream closed before its end). A stopped run stays
        open in the journal and can be resumed.
        """


class HookChain(Hooks):
    """Runs several hooks in order; each sees what the previous one returned."""

    def __init__(self, hooks: Iterable[Hooks]) -> None:
        self.hooks = list(hooks)

    async def on_run_start(self, ctx: RunContext[Any]) -> None:
        for hook in self.hooks:
            await hook.on_run_start(ctx)

    async def before_model(self, ctx: RunContext[Any], request: ModelRequest) -> ModelRequest:
        for hook in self.hooks:
            request = await hook.before_model(ctx, request) or request
        return request

    async def after_model(self, ctx: RunContext[Any], response: ModelResponse) -> ModelResponse:
        for hook in self.hooks:
            response = await hook.after_model(ctx, response) or response
        return response

    async def before_tool(
        self, ctx: RunContext[Any], call: ToolCallPart
    ) -> ToolCallPart | ToolResultPart:
        for hook in self.hooks:
            changed = await hook.before_tool(ctx, call)
            if isinstance(changed, ToolResultPart):
                return changed
            if changed is not None:
                call = changed
        return call

    async def after_tool(
        self, ctx: RunContext[Any], call: ToolCallPart, result: ToolResultPart
    ) -> ToolResultPart:
        for hook in self.hooks:
            result = await hook.after_tool(ctx, call, result) or result
        return result

    async def on_event(self, ctx: RunContext[Any], event: Any) -> None:
        for hook in self.hooks:
            await hook.on_event(ctx, event)

    async def on_run_end(
        self, ctx: RunContext[Any], result: RunResult | None, error: BaseException | None
    ) -> None:
        for hook in self.hooks:
            await hook.on_run_end(ctx, result, error)
