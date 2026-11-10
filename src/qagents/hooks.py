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
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from .messages import ToolCallPart, ToolResultPart
from .models.base import ModelRequest, ModelResponse
from .tools import RunContext

if TYPE_CHECKING:
    from .results import RunResult


class Hooks:
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
        """The run finished (``result``) or failed (``error``)."""


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
