"""Who decides whether a tool call runs.

The decision is made by ordinary code, outside the model, in a fixed order:

1. ``deny`` patterns: a match refuses the call.
2. ``can_use_tool``: your function may allow, ask or deny.
3. ``ask`` patterns: a match asks a human.
4. ``allow`` patterns: a match runs the call.
5. The ``mode`` default.

The first rule that answers decides. ``deny`` patterns are checked first, so
nothing overrides them, and among the patterns ask beats allow. An allow (or
``True``) from ``can_use_tool`` skips the ``ask`` patterns, ``requires_approval``
and the mode, so return ``None`` from it for calls that must still go through
the ask rules. Anything that goes wrong while deciding (an exception in your
callback) refuses the call.

The rules are checked again when a run is resumed with approvals: an approval
only answers a call the rules do not deny, so a deny rule refuses the call
whatever the approval says. When an approval carries edited ``arguments``,
the rules are checked on the edited call too.

"Ask" is resolved by ``on_ask`` when you give one (an interactive prompt, a
Slack button). Without it the run pauses: ``agent.run`` returns with
``status == "paused"`` and you continue later with ``agent.resume``. A child
run (an agent used as a tool) whose permissions have no ``on_ask`` uses the
parent run's.

Signed pauses: with ``Permissions(approval_secret=...)`` every paused call
(`PendingApproval` and `PendingInput`) carries a ``token``, an HMAC-SHA256 of
the kind of pause, the session id, the run id, the position of the call's
message in the run, the call id, the tool name and the arguments (for a
`PendingInput` also its ``reason`` and ``metadata``); see `approval_token`.
``agent.resume`` accepts an approval or a result for such a call only with
that token, and only while the call recorded in the journal still matches it;
otherwise it raises `ApprovalTokenError` before anything runs. Someone who can
write to the session store but does not know the secret can then not forge a
pause, alter one, copy one into another session, or append a copy of an
answered one again once its run finished. (Inside a run that is still open,
a writer who appends the run's entries again from its ``run_started`` on
resets the position, and an answered call's token then matches again; a
store you do not trust with the journal needs its own append-only checks.)
For example:

    permissions = Permissions(approval_secret=os.environ["APPROVAL_SECRET"])
    paused = await agent.run("Refund order A-1001", session=session)
    pending = paused.pending_approvals[0]  # show it, token included, to a reviewer
    await agent.resume(session, approvals={pending.tool_call_id: Approval.for_call(pending)})

The token protects integrity, not confidentiality: it is stored in the journal
next to the call it signs, and it does not say who approved. Authenticate the
people who answer approvals in your own application. What the token does not
cover: arguments an approver edits (``Approval.arguments``; the rules check
them instead), the ``reason`` of a `PendingApproval` (the text that says why
the call asks: show reviewers the call itself, not only that text), the rest
of the journal (tool results and messages are not signed), and a journal
rolled back by deleting the entries written after a pause, which nothing
inside the journal can detect. ``PendingInput.metadata``
is verified only when ``agent.resume`` runs: if the store is not trusted, do
not act on it (for example by polling a job id from it) before that.
"""

from __future__ import annotations

import hashlib
import hmac
import inspect
import json
import logging
import math
from collections.abc import Callable, Iterable, Mapping
from decimal import Decimal
from fnmatch import fnmatchcase
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict
from pydantic_core import to_jsonable_python

from . import prompts
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
    """Run the tool with these arguments instead of the model's. The permission
    rules are checked again on the edited call (a deny refuses it), the change
    is journaled before the tool runs, and the model is told the arguments
    were changed. The pause's token does not cover the edit."""
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

    The mode applies to calls that no rule and no ``can_use_tool`` answer decided.
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
            ask: Patterns of tools that ask, unless ``can_use_tool`` has
                already decided.
            deny: Patterns of tools that never run.
            can_use_tool: Your decision function, consulted after ``deny``. A
                decision it returns is final: return ``None`` to leave the
                call to the patterns and the mode.
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

    def sign(self, **pause: Any) -> str | None:
        """The token of a paused call, or ``None`` without an ``approval_secret``.

        Takes the keyword arguments of `approval_token` after the secret.

        Raises:
            ValueError: The pause holds a value that has no canonical JSON
                (NaN, an infinity, a string with a lone surrogate).
        """
        if self._approval_key is None:
            return None
        return approval_token(self._approval_key, **pause)

    def verify(self, token: str | None, **pause: Any) -> bool:
        """Whether ``token`` signs this pause. Always true without an ``approval_secret``.

        Takes the keyword arguments of `approval_token` after the secret. A
        token that is not ASCII, or a pause that cannot be signed, is refused.
        """
        try:
            expected = self.sign(**pause)
        except ValueError:
            return False
        if expected is None:
            return True
        return isinstance(token, str) and token.isascii() and hmac.compare_digest(token, expected)

    async def decide(self, ctx: RunContext[Any], tool: Tool, call: ToolCallPart) -> Decision:
        name = tool.name
        if _matches(name, self.deny):
            return Decision(behavior="deny", reason=prompts.PERMISSION_REASON_DENY_RULE)
        if self.can_use_tool is not None:
            try:
                answer = self.can_use_tool(ctx, tool, call)
                if inspect.isawaitable(answer):
                    answer = await answer
            except Exception:
                logger.exception("can_use_tool raised for %s; refusing the call", name)
                return Decision(behavior="deny", reason=prompts.PERMISSION_REASON_CHECK_FAILED)
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
            return Decision(behavior="deny", reason=prompts.PERMISSION_REASON_READ_ONLY)
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
            return Approval(approved=False, reason=prompts.PERMISSION_REASON_APPROVAL_FAILED)
        if isinstance(answer, Approval):
            return answer
        return Approval(approved=answer is True)


def approval_token(
    secret: str | bytes,
    *,
    session_id: str,
    run_id: str,
    position: int,
    tool_call_id: str,
    name: str,
    arguments: Mapping[str, Any],
    kind: Literal["approval", "input"] = "approval",
    reason: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """HMAC-SHA256, as hex, of the canonical JSON of one paused call.

    The payload is the RFC 8785 (JCS) canonical JSON, encoded as UTF-8, of an
    object with the keys ``v`` (``2``), ``kind`` (``"approval"`` or
    ``"input"``), ``session_id``, ``run_id``, ``position`` (how many messages
    the run had journaled when the call's assistant message joined it, that
    message included), ``tool_call_id``, ``name`` and ``arguments``; a pause
    of kind ``"input"`` also has ``reason`` and ``metadata``. JCS sorts the
    keys, leaves out spaces and writes numbers as JavaScript does (``15.0``
    is written ``15``), so another service that holds the secret can compute
    the same token with any JCS implementation. Values that are not JSON are
    first turned into their ``str()``, which only Python reproduces.

    Raises:
        ValueError: A value has no canonical JSON: NaN, an infinity, an
            integer that a double cannot hold exactly, or a string with a lone
            surrogate.
    """
    key = secret.encode("utf-8") if isinstance(secret, str) else bytes(secret)
    payload: dict[str, Any] = {
        "v": 2,
        "kind": kind,
        "session_id": session_id,
        "run_id": run_id,
        "position": position,
        "tool_call_id": tool_call_id,
        "name": name,
        "arguments": dict(arguments),
    }
    if kind == "input":
        payload["reason"] = reason
        payload["metadata"] = dict(metadata or {})
    text = canonical_json(to_jsonable_python(payload, fallback=str))
    return hmac.new(key, text.encode("utf-8"), hashlib.sha256).hexdigest()


def canonical_json(value: Any) -> str:
    """The RFC 8785 (JCS) canonical JSON of a JSON value.

    Raises:
        ValueError: NaN or an infinity, an integer that a double cannot hold
            exactly (JSON numbers are doubles in JCS, so such an integer would
            sign the same as its neighbours: pass large ids as strings), a
            string with a lone surrogate, or a value that is not JSON (dicts,
            lists, strings, numbers, bools, None).
    """
    if value is None or isinstance(value, bool):
        return json.dumps(value)
    if isinstance(value, int):
        if abs(value) <= 2**53:
            return str(value)
        try:
            number = float(value)
        except OverflowError:
            number = math.inf
        if number != value:  # rounded: neighbouring integers would sign the same
            raise ValueError(
                "This integer has no exact JSON number form, so a call with it cannot be signed."
            )
        return _es_number(number)
    if isinstance(value, float):
        return _es_number(value)
    if isinstance(value, str):
        value.encode("utf-8")  # a lone surrogate raises UnicodeEncodeError, a ValueError
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list | tuple):
        return "[" + ",".join(canonical_json(item) for item in value) + "]"
    if isinstance(value, dict):
        members = [
            f"{canonical_json(key)}:{canonical_json(value[key])}"
            for key in sorted(value, key=_utf16_key)
        ]
        return "{" + ",".join(members) + "}"
    raise ValueError(f"{type(value).__name__} is not a JSON value.")


def _utf16_key(key: Any) -> bytes:
    """JCS sorts object keys by their UTF-16 code units."""
    if not isinstance(key, str):
        raise ValueError(f"JSON object keys are strings, not {type(key).__name__}.")
    return key.encode("utf-16-be")


def _es_number(number: float) -> str:
    """A number as JavaScript's ``Number.prototype.toString`` writes it, as JCS requires."""
    if math.isnan(number) or math.isinf(number):
        raise ValueError(f"{number!r} has no JSON form, so a call with it cannot be signed.")
    if number == 0:
        return "0"  # -0.0 included
    _, digit_tuple, exponent = Decimal(repr(abs(number))).as_tuple()
    assert isinstance(exponent, int)
    written = "".join(map(str, digit_tuple))
    digits = written.rstrip("0")
    k = len(digits)
    n = exponent + len(written)  # the value is 0.<digits> times 10**n
    if k <= n <= 21:
        text = digits + "0" * (n - k)
    elif 0 < n <= 21:
        text = f"{digits[:n]}.{digits[n:]}"
    elif -6 < n <= 0:
        text = "0." + "0" * -n + digits
    else:
        mantissa = digits[0] + (f".{digits[1:]}" if k > 1 else "")
        text = f"{mantissa}e{'+' if n > 0 else '-'}{abs(n - 1)}"
    return ("-" if number < 0 else "") + text


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
        return Decision(behavior="deny", reason=prompts.PERMISSION_REASON_CHECK_REFUSED)
    if answer in ("allow", "ask", "deny"):
        return Decision(behavior=answer)
    logger.error("can_use_tool returned %r, which is not a decision; refusing the call", answer)
    return Decision(behavior="deny", reason=prompts.PERMISSION_REASON_CHECK_FAILED)
