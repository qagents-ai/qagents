"""Who decides whether a tool call runs.

The decision is made by ordinary code, outside the model, in a fixed order:

1. ``deny`` patterns: a match refuses the call.
2. ``can_use_tool``: your function may allow, ask or deny.
3. ``ask`` patterns: a match asks a human.
4. ``allow`` patterns: a match runs the call.
5. The ``mode`` default.

The more restrictive rule always wins, and anything that goes wrong while
deciding (an exception in your callback) refuses the call.

"Ask" is resolved by ``on_ask`` when you give one (an interactive prompt, a
Slack button). Without it the run pauses: ``agent.run`` returns with
``status == "paused"`` and you continue later with ``agent.resume``.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Iterable
from fnmatch import fnmatchcase
from typing import Any, Literal

from pydantic import BaseModel

from .messages import ToolCallPart
from .tools import RunContext, Tool

logger = logging.getLogger("qagents.permissions")

Behavior = Literal["allow", "ask", "deny"]
Mode = Literal["default", "allow_all", "ask_all", "read_only"]


class Decision(BaseModel):
    behavior: Behavior
    reason: str | None = None


class Approval(BaseModel):
    """A human's answer to an approval request."""

    approved: bool
    reason: str | None = None
    arguments: dict[str, Any] | None = None
    """Run the tool with these arguments instead of the model's."""


CanUseTool = Callable[[RunContext[Any], Tool, ToolCallPart], Any]
"""``(ctx, tool, call)`` -> ``Decision``, ``"allow"``/``"ask"``/``"deny"``, a bool,
or ``None`` for "no opinion". May be async."""
OnAsk = Callable[[RunContext[Any], Tool, ToolCallPart, Any], Any]
"""``(ctx, tool, call, reason)`` -> ``Approval`` or a bool. May be async."""


class Permissions:
    """Rules for tool calls. Patterns are shell-style and match tool names.

    Modes:

    * ``"default"``: tools declared with ``requires_approval=True`` ask, the rest run.
    * ``"allow_all"``: everything runs (``deny`` and ``ask`` rules still apply).
    * ``"ask_all"``: everything asks unless an ``allow`` rule matches.
    * ``"read_only"``: only tools declared ``read_only=True`` run, the rest are refused.
    """

    def __init__(
        self,
        *,
        mode: Mode = "default",
        allow: Iterable[str] = (),
        ask: Iterable[str] = (),
        deny: Iterable[str] = (),
        can_use_tool: CanUseTool | None = None,
        on_ask: OnAsk | None = None,
    ) -> None:
        self.mode: Mode = mode
        self.allow = tuple(allow)
        self.ask = tuple(ask)
        self.deny = tuple(deny)
        self.can_use_tool = can_use_tool
        self.on_ask = on_ask

    async def decide(self, ctx: RunContext[Any], tool: Tool, call: ToolCallPart) -> Decision:
        name = tool.name
        if _matches(name, self.deny):
            return Decision(behavior="deny", reason="a deny rule matches this tool.")
        if self.can_use_tool is not None:
            try:
                answer = self.can_use_tool(ctx, tool, call)
                if inspect.isawaitable(answer):
                    answer = await answer
            except Exception:
                logger.exception("can_use_tool raised for %s; refusing the call", name)
                return Decision(behavior="deny", reason="the permission check failed.")
            decision = _as_decision(answer)
            if decision is not None:
                return decision
        if _matches(name, self.ask):
            return Decision(behavior="ask", reason="an ask rule matches this tool.")
        if _matches(name, self.allow):
            return Decision(behavior="allow")
        if self.mode == "allow_all":
            return Decision(behavior="allow")
        if self.mode == "ask_all":
            return Decision(behavior="ask", reason="every tool call needs approval.")
        if self.mode == "read_only":
            if tool.read_only:
                return Decision(behavior="allow")
            return Decision(behavior="deny", reason="only read-only tools are allowed.")
        if tool.requires_approval:
            return Decision(behavior="ask", reason="this tool requires approval.")
        return Decision(behavior="allow")

    async def request_approval(
        self, ctx: RunContext[Any], tool: Tool, call: ToolCallPart, reason: str | None
    ) -> Approval:
        """Resolve an "ask" through ``on_ask``. Any failure counts as a refusal."""
        assert self.on_ask is not None
        try:
            answer = self.on_ask(ctx, tool, call, reason)
            if inspect.isawaitable(answer):
                answer = await answer
        except Exception:
            logger.exception("on_ask raised for %s; refusing the call", tool.name)
            return Approval(approved=False, reason="the approval request failed.")
        if isinstance(answer, Approval):
            return answer
        return Approval(approved=answer is True)


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatchcase(name, pattern) for pattern in patterns)


def _as_decision(answer: Any) -> Decision | None:
    if answer is None:
        return None
    if isinstance(answer, Decision):
        return answer
    if answer is True:
        return Decision(behavior="allow")
    if answer is False:
        return Decision(behavior="deny", reason="the permission check refused it.")
    if answer in ("allow", "ask", "deny"):
        return Decision(behavior=answer)
    logger.error("can_use_tool returned %r, which is not a decision; refusing the call", answer)
    return Decision(behavior="deny", reason="the permission check failed.")
