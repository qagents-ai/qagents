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
`PendingInput` also its ``reason`` and ``metadata``). Version 3 additionally
binds the durable operation identity and ask number; see `approval_token`.
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
from collections.abc import Awaitable, Callable, Iterable, Mapping
from fnmatch import fnmatchcase
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict
from pydantic_core import to_jsonable_python

from . import prompts
from .effects._canonical import JSONValue, digest, jcs
from .effects._kinds import APPROVAL_MAC_FIELDS, ApprovalRecord
from .effects._types import EffectRecord, Settle
from .errors import ConfigurationError
from .messages import ToolCallPart
from .session import PendingApproval, PendingInput
from .tools import RunContext, Tool

logger = logging.getLogger("qagents.permissions")

Behavior = Literal["allow", "ask", "deny"]
Mode = Literal["default", "allow_all", "ask_all", "read_only"]
APPROVAL_TOKEN_VERSION = 3


class Decision(BaseModel):
    behavior: Behavior
    reason: str | None = None


class Approval(BaseModel):
    """An application-supplied answer to an approval request."""

    approved: bool
    by: str | None = None
    """Reviewer identity asserted and authenticated by the host application."""
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
        by: str | None = None,
        reason: str | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> Approval:
        """Answer ``pending``, carrying its token.

        Args:
            pending: The call from ``result.pending_approvals``.
            approved: Whether the call may run.
            by: Reviewer identity asserted by the application.
            reason: Why it was declined, for the model.
            arguments: Arguments to run the tool with instead of the model's.
        """
        return cls(
            approved=approved, by=by, reason=reason, arguments=arguments, token=pending.token
        )


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
        approval_ttl_s: float = 6 * 3600,
        authorize_settlement: Callable[[Settle, EffectRecord], bool | Awaitable[bool]]
        | None = None,
        resend_min_age_s: float = 900.0,
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
            approval_ttl_s: Decision validity from the recorded store time, in seconds.
            authorize_settlement: Host policy for operator answers; may be asynchronous.
            resend_min_age_s: Minimum operator resend age, in store-clock seconds.

        Raises:
            ConfigurationError: ``approval_secret`` is not text or bytes, or is
                shorter than 16 bytes.
        """
        for setting, value, positive in (
            ("approval_ttl_s", approval_ttl_s, True),
            ("resend_min_age_s", resend_min_age_s, False),
        ):
            if not _finite_number(value) or value < 0 or (positive and value == 0):
                bound = "positive" if positive else "nonnegative"
                raise ConfigurationError(f"{setting} must be a finite {bound} number.")
        if authorize_settlement is not None and not callable(authorize_settlement):
            raise ConfigurationError("authorize_settlement must be callable or None.")
        self.approval_ttl_s = float(approval_ttl_s)
        self.authorize_settlement = authorize_settlement
        self.resend_min_age_s = float(resend_min_age_s)
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

    def decision_record(
        self,
        *,
        session_id: str,
        run_id: str,
        operation_id: str,
        ask: int,
        tool: Tool,
        call: ToolCallPart,
        approved: bool,
        reason: str | None,
        by: str | None,
        decided_at: float,
        arguments_changed: bool = False,
        arguments_changed_by: str | None = None,
    ) -> ApprovalRecord:
        """Record a decision and, when configured, a tamper-evident HMAC.

        The application asserts who approved. Runtime policy separately checks
        the current deny rules, tool specification and store-clock expiry.
        """
        identity = _decision_id(session_id, run_id, operation_id, call.id, ask)
        if type(approved) is not bool or type(arguments_changed) is not bool:
            raise ValueError("Decision and argument-change flags must be bool.")
        if not _finite_number(decided_at):
            raise ValueError("decided_at must be a finite store-clock number.")
        expires_at = decided_at + self.approval_ttl_s
        if not math.isfinite(expires_at):
            raise ValueError("Decision expiry exceeds the finite store-clock range.")
        if call.name != tool.name:
            raise ValueError("A decision must name the tool whose call was reviewed.")
        snapshot = json.loads(jcs(dict(call.arguments)))
        result = ApprovalRecord(
            approval_id=identity,
            session_id=session_id,
            run_id=run_id,
            operation_id=operation_id,
            ask=ask,
            tool_call_id=call.id,
            name=tool.name,
            approved=approved,
            reason=reason,
            by=by,
            arguments_digest=arguments_digest(snapshot),
            tool_spec_digest=tool_spec_digest(tool),
            decided_at=decided_at,
            expires_at=expires_at,
            mac=None,
            arguments_changed=arguments_changed,
            arguments=snapshot if arguments_changed else None,
            arguments_changed_by=arguments_changed_by,
        )
        if self._approval_key is not None:
            result.mac = _decision_mac(self._approval_key, result)
        return result

    def verify_decision(self, record: ApprovalRecord) -> bool:
        """Verify v3 identity, optional argument snapshot and MAC; read no wall clock."""
        return self._approval_key is not None and _mac_valid(self._approval_key, record)

    def sign_settlement(self, fields: Mapping[str, JSONValue]) -> str | None:
        """Sign the supplied canonical settlement fields, or return None without a secret."""
        if self._approval_key is None:
            return None
        return hmac.new(self._approval_key, jcs(dict(fields)), hashlib.sha256).hexdigest()

    def verify_settlement(self, fields: Mapping[str, JSONValue], mac: str | None) -> bool:
        """Whether ``mac`` signs these settlement fields under the ``approval_secret``.

        Takes the fields that `sign_settlement` signs. Always false without an
        ``approval_secret``, for a ``mac`` that is not an ASCII string, and for
        fields that have no canonical JSON. The engine checks an operator
        ``resend`` with it before rule D5 sends.
        """
        try:
            expected = self.sign_settlement(fields)
        except (TypeError, ValueError):
            return False
        return (
            expected is not None
            and isinstance(mac, str)
            and mac.isascii()
            and hmac.compare_digest(mac, expected)
        )

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
    operation_id: str | None = None,
    ask: int = 1,
    v: Literal[2, 3] = 3,
) -> str:
    """HMAC-SHA256, as hex, of the canonical JSON of one paused call.

    The payload is the RFC 8785 (JCS) canonical JSON, encoded as UTF-8, of an
    object with the keys ``v`` (``3``), ``kind`` (``"approval"`` or
    ``"input"``), ``session_id``, ``run_id``, ``position`` (how many messages
    the run had journaled when the call's assistant message joined it, that
    message included), ``tool_call_id``, ``name`` and ``arguments``; a pause
    of kind ``"input"`` also has ``reason`` and ``metadata``. JCS sorts the
    keys, leaves out spaces and writes numbers as JavaScript does (``15.0``
    is written ``15``), so another service that holds the secret can compute
    the same token with any JCS implementation. Version 3 also binds the
    durable ``operation_id`` and ``ask`` number, so an earlier answer cannot
    authorize a re-ask. Non-JSON values are refused.

    Explicit ``v=2`` retains the legacy payload and string fallback for old
    pauses. The caller must select it only for journals without approvals/3;
    there is no automatic downgrade when a v3 token fails verification.

    Raises:
        ValueError: A value has no canonical JSON: NaN, an infinity, an
            integer that a double cannot hold exactly, or a string with a lone
            surrogate.
    """
    if type(v) is not int or v not in (2, 3):
        raise ValueError("Approval token version must be 2 or 3.")
    if type(ask) is not int or ask < 1:
        raise ValueError("Approval ask must be a positive integer.")
    if operation_id is not None and (not isinstance(operation_id, str) or not operation_id):
        raise ValueError("operation_id must be a nonempty string or None.")
    key = secret.encode("utf-8") if isinstance(secret, str) else bytes(secret)
    payload: dict[str, Any] = {
        "v": v,
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
    if v == 3:
        payload.update(operation_id=operation_id, ask=ask)
    else:
        payload = to_jsonable_python(payload, fallback=str)
    text = canonical_json(payload)
    return hmac.new(key, text.encode("utf-8"), hashlib.sha256).hexdigest()


def canonical_json(value: Any) -> str:
    """The RFC 8785 (JCS) canonical JSON of a JSON value.

    Raises:
        ValueError: NaN or an infinity, an integer outside the safe range
            ``-(2**53 - 1)`` through ``2**53 - 1`` (pass large ids as strings), a
            string with a lone surrogate, or a value that is not JSON (dicts,
            lists, strings, numbers, bools, None).
    """
    return jcs(value).decode("utf-8")


def arguments_digest(arguments: Mapping[str, Any]) -> str:
    """Bind a decision to the exact canonical arguments that the tool will receive."""
    return digest(dict(arguments))


def tool_spec_digest(tool: Tool) -> str:
    """Bind name, description, argument schema, flags and declared effect versions."""
    return digest(
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "requires_approval": tool.requires_approval,
            "read_only": tool.read_only,
            "idempotent": tool.idempotent,
            "strict": tool.strict,
            "deferred": tool.deferred,
            "timeout": tool.timeout,
            "takes_context": tool.takes_context,
            "effects": [
                {"name": effect.name, "version": effect.version}
                for effect in getattr(tool, "effects", ())
            ],
        }
    )


def _decision_id(
    session_id: str, run_id: str, operation_id: str, tool_call_id: str, ask: int
) -> str:
    """Bind operation_id and ask through the already-signed approval_id field."""
    if any(
        not isinstance(value, str) or not value
        for value in (session_id, run_id, operation_id, tool_call_id)
    ):
        raise ValueError("Decision identities must be nonempty strings.")
    if type(ask) is not int or ask < 1:
        raise ValueError("Approval ask must be a positive integer.")
    return "ap1:" + digest([session_id, run_id, operation_id, tool_call_id, ask])


def _finite_number(value: Any) -> bool:
    try:
        return (
            not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value)
        )
    except OverflowError:
        return False


def _decision_mac(key: bytes, record: ApprovalRecord) -> str:
    fields = {name: getattr(record, name) for name in APPROVAL_MAC_FIELDS}
    return hmac.new(key, jcs(fields), hashlib.sha256).hexdigest()


def _mac_valid(key: bytes, record: ApprovalRecord) -> bool:
    """M14 seam: verify the decision MAC and bindings before runtime reuse."""
    if not isinstance(record, ApprovalRecord):
        return False
    try:
        if record.v != APPROVAL_TOKEN_VERSION or record.approval_id != _decision_id(
            record.session_id, record.run_id, record.operation_id, record.tool_call_id, record.ask
        ):
            return False
        if record.arguments_changed and record.arguments is None:
            return False
        if (
            record.arguments is not None
            and arguments_digest(record.arguments) != record.arguments_digest
        ):
            return False
        expected = _decision_mac(key, record)
    except (ValueError, TypeError, AttributeError):
        return False
    return (
        isinstance(record.mac, str)
        and record.mac.isascii()
        and hmac.compare_digest(record.mac, expected)
    )


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
