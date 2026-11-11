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
``status == "paused"`` and you continue later with ``agent.resume``. A child
run (an agent used as a tool) whose permissions have no ``on_ask`` uses the
parent run's.

Signed pauses: with ``Permissions(approval_secret=...)`` every paused call
(`PendingApproval` and `PendingInput`) carries a ``token``, an HMAC-SHA256 of
the run id, the call id, the tool name and the arguments. ``agent.resume``
accepts an approval or a result for such a call only with that token, and
only while the call recorded in the journal still matches it; otherwise it
raises `ApprovalTokenError` before anything runs. A stored pause can then not
be forged or altered by someone who can write to the session store but does
not know the secret:

    permissions = Permissions(approval_secret=os.environ["APPROVAL_SECRET"])
    paused = await agent.run("Refund order A-1001", session=session)
    pending = paused.pending_approvals[0]  # show it, token included, to a reviewer
    await agent.resume(session, approvals={pending.tool_call_id: Approval.for_call(pending)})

The token protects integrity, not confidentiality: it is stored in the journal
next to the call it signs, and it does not say who approved. Authenticate the
people who answer approvals in your own application.
"""

from __future__ import annotations

import hashlib
import hmac
import inspect
import json
import logging
from collections.abc import Callable, Iterable, Mapping
from fnmatch import fnmatchcase
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict
from pydantic_core import to_jsonable_python

from .errors import ConfigurationError
from .messages import ToolCallPart
from .session import PendingApproval, PendingInput
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
    """Why the call was declined; the model reads it as the call's result."""
    arguments: dict[str, Any] | None = None
    """Run the tool with these arguments instead of the model's."""
    token: str | None = None
    """The ``token`` of the `PendingApproval` this answers. Required when the
    agent's permissions have an ``approval_secret``; ignored otherwise."""

    @classmethod
    def for_call(
        cls,
        pending: PendingApproval,
        approved: bool = True,
        *,
        reason: str | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> Approval:
        """Answer ``pending``, carrying its token.

        Args:
            pending: The call from ``result.pending_approvals``.
            approved: Whether the call may run.
            reason: Why it was declined, for the model.
            arguments: Arguments to run the tool with instead of the model's.
        """
        return cls(approved=approved, reason=reason, arguments=arguments, token=pending.token)


class InputResult(BaseModel):
    """The result of a call that waits for input, with the token of its signed pause.

    Pass it in ``agent.resume(session, results={tool_call_id: InputResult(...)})``.
    It is needed only when the agent's permissions have an ``approval_secret``;
    without one, give the value directly (an `InputResult` is unwrapped all
    the same). When pauses are signed, a mapping of exactly the form
    ``{"token": ..., "value": ...}`` is read the same way, for answers that
    arrive as JSON.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: Any
    """The result: a value rendered like a tool's return value, an exception
    instance for an error result, or a `ToolResultPart` used as it is."""
    token: str | None = None
    """The ``token`` of the `PendingInput` this answers."""

    @classmethod
    def for_call(cls, pending: PendingInput, value: Any) -> InputResult:
        """Answer ``pending`` with ``value``, carrying its token."""
        return cls(value=value, token=pending.token)


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
        approval_secret: str | bytes | None = None,
    ) -> None:
        """Set up the rules.

        Args:
            mode: The default for tools no rule matches; see the class docstring.
            allow: Patterns of tools that run without asking.
            ask: Patterns of tools that always ask.
            deny: Patterns of tools that never run.
            can_use_tool: Your decision function, consulted after ``deny``.
            on_ask: Answers "ask" decisions in-process. Without it the run pauses.
            approval_secret: Signs every pause, so that ``agent.resume`` accepts
                only answers that carry the pause's token, for calls the journal
                still records unchanged. Use at least 16 random bytes, for
                example ``secrets.token_hex(32)``, and keep it out of the code.

        Raises:
            ConfigurationError: ``approval_secret`` is not text or bytes, or is
                shorter than 16 bytes.
        """
        self.mode: Mode = mode
        self.allow = tuple(allow)
        self.ask = tuple(ask)
        self.deny = tuple(deny)
        self.can_use_tool = can_use_tool
        self.on_ask = on_ask
        self._approval_key = _approval_key(approval_secret)

    @property
    def signs_pauses(self) -> bool:
        """True when an ``approval_secret`` was given."""
        return self._approval_key is not None

    def sign(
        self, *, run_id: str, tool_call_id: str, name: str, arguments: Mapping[str, Any]
    ) -> str | None:
        """The token of a paused call, or ``None`` without an ``approval_secret``."""
        if self._approval_key is None:
            return None
        return approval_token(
            self._approval_key,
            run_id=run_id,
            tool_call_id=tool_call_id,
            name=name,
            arguments=arguments,
        )

    def verify(
        self,
        token: str | None,
        *,
        run_id: str,
        tool_call_id: str,
        name: str,
        arguments: Mapping[str, Any],
    ) -> bool:
        """Whether ``token`` signs this call. Always true without an ``approval_secret``."""
        expected = self.sign(
            run_id=run_id, tool_call_id=tool_call_id, name=name, arguments=arguments
        )
        if expected is None:
            return True
        return isinstance(token, str) and hmac.compare_digest(token, expected)

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


def approval_token(
    secret: str | bytes,
    *,
    run_id: str,
    tool_call_id: str,
    name: str,
    arguments: Mapping[str, Any],
) -> str:
    """HMAC-SHA256, as hex, of the canonical JSON of one paused call.

    The JSON object has the keys ``arguments``, ``name``, ``run_id`` and
    ``tool_call_id``, sorted, without spaces, with non-ASCII characters kept
    as UTF-8. Another service that holds the secret can compute the same token.
    """
    key = secret.encode("utf-8") if isinstance(secret, str) else bytes(secret)
    payload = json.dumps(
        {
            "arguments": to_jsonable_python(dict(arguments), fallback=str),
            "name": name,
            "run_id": run_id,
            "tool_call_id": tool_call_id,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _approval_key(secret: str | bytes | None) -> bytes | None:
    if secret is None:
        return None
    if not isinstance(secret, str | bytes):
        raise ConfigurationError(
            f"approval_secret must be str or bytes, not {type(secret).__name__}."
        )
    key = secret.encode("utf-8") if isinstance(secret, str) else bytes(secret)
    if len(key) < 16:
        raise ConfigurationError(
            "approval_secret is too short to sign pauses safely: use at least 16 random bytes, "
            "for example secrets.token_hex(32), and load it from your secret store."
        )
    return key


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
