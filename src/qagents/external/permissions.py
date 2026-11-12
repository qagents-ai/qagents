"""Permission requests of an external agent, decided with qagents' rules.

`PermissionBridge` turns each request into a decision of the calling run's `Permissions`
about a synthetic call ``"<tool name>.<external tool>"`` (an `ExternalTool`), answers the
agent with a `PermissionAnswer`, and keeps an `ExternalPermission` record of it.
`ExternalApproval` is a human's answer to a request that paused the calling run
(``on_permission="pause"``); `permission_fingerprint` ties that answer to the one request
it approves.

It lives apart from the turn because the decision is the part that must fail closed: every
path through `PermissionBridge.decide` ends in an answer, and none of them is "allow
always".
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .. import prompts
from ..errors import ConfigurationError, ToolError
from ..messages import ToolCallPart, ToolResultPart
from ..permissions import Approval, Decision, Permissions, canonical_json
from ..session import PendingInput
from ..tools import RunContext, Tool

logger = logging.getLogger("qagents.external")

PermissionPosture = Literal["rules", "pause", "live"]
"""What a permission request that needs a human gets when no ``on_ask`` answers it
(see `PermissionBridge`)."""

APPROVAL_KEY = "qagents.external.approval"
"""``ToolResultPart.metadata`` key of an `ExternalApproval`: ``{"approved", "reason"}``."""


class ExternalPermission(BaseModel):
    """One permission request of the external agent, and how qagents answered it."""

    tool: str
    """The prefixed name the rules matched, for example ``coder.bash``."""
    external_tool: str
    """The agent's own tool name, for example ``bash``."""
    arguments: dict[str, Any] = Field(default_factory=dict)
    tool_call_id: str | None = None
    """The agent's id of the tool call it asked about, when it gave one."""
    decision: Literal["allow", "ask", "deny"] | None = None
    """What the `Permissions` rules said; ``None`` when deciding failed or the turn was
    being cancelled."""
    outcome: Literal["allow", "deny", "cancel"]
    """What the agent was answered: allow once, reject, or cancelled."""
    decided_by: str
    """``rules``, ``on_ask``, ``grant`` (an approval given across a pause), ``declined``
    (a refusal given across a pause), ``no_approver``, ``deferred`` (it paused the
    calling run), ``timeout``, ``error``, ``edited_arguments``, ``no_option`` (allowed,
    but the agent offered no allow-once option), ``unreadable_arguments`` (the tool
    call's input is not a JSON object) or ``cancelling``."""
    reason: str | None = None
    fingerprint: str = ""
    """`permission_fingerprint` of the tool name and the arguments an approval given across a
    pause must match: all of them, except free text the agent rewrites on every attempt
    (`ExternalAgent.grant_arguments`)."""

    @property
    def approved(self) -> bool:
        return self.outcome == "allow"

    def detail(self, limit: int = 80) -> str:
        """The request in a few words: the ``command`` argument, or the shortened arguments."""
        command = self.arguments.get("command")
        if isinstance(command, str):
            text = json.dumps(command, ensure_ascii=False)
        elif self.arguments:
            text = json.dumps(self.arguments, ensure_ascii=False, default=str, sort_keys=True)
        else:
            text = "(no arguments)"
        return text if len(text) <= limit else text[: limit - 3] + "..."


class PermissionAnswer(BaseModel):
    """What the `PermissionBridge` answers an external agent's permission request."""

    outcome: Literal["allow", "deny", "cancel"]
    """``allow``: allow this one call. ``deny``: reject it. ``cancel``: answer "cancelled"
    (the protocol's own word for "no decision"), which the agent treats as a refusal."""
    reason: str | None = None
    decided_by: str = "rules"
    option_id: str | None = None
    """The option to answer with, when the request offered options (ACP): chosen by
    `select_option`. ``None`` with options means: answer "cancelled"."""

    @property
    def approved(self) -> bool:
        return self.outcome == "allow"

    def select_option(self, options: Sequence[Mapping[str, Any]]) -> str | None:
        """The ``optionId`` to answer from ACP-style options (``{optionId, kind}``).

        Allow picks the first ``allow_once`` option and never an ``allow_always``
        one: grants stay one-shot. Deny picks the first ``reject_once`` option,
        else a ``reject_always`` one. ``None`` means: answer "cancelled" (the
        outcome is ``cancel``, or no fitting option exists, which fails closed).
        """
        if self.outcome == "allow":
            wanted: tuple[str, ...] = ("allow_once",)
        elif self.outcome == "deny":
            wanted = ("reject_once", "reject_always")
        else:
            return None
        for kind in wanted:
            for option in options:
                if option.get("kind") == kind and isinstance(option.get("optionId"), str):
                    return str(option["optionId"])
        return None


def _with_option(
    answer: PermissionAnswer, options: Sequence[Mapping[str, Any]]
) -> PermissionAnswer:
    """``answer`` with the option to send; an allow that no option can carry is cancelled."""
    option = answer.select_option(options)
    if option is not None or answer.outcome == "cancel":
        return answer.model_copy(update={"option_id": option})
    if answer.outcome == "allow":
        return PermissionAnswer(
            outcome="cancel", reason=prompts.EXTERNAL_PERMISSION_NO_OPTION, decided_by="no_option"
        )
    return answer.model_copy(update={"outcome": "cancel"})  # still a refusal: cancelled


def permission_fingerprint(tool: str, arguments: Mapping[str, Any] | None) -> str:
    """``sha256`` (hex) of the prefixed tool name and the canonical JSON of the arguments.

    A changed argument gives another fingerprint, so an approval given for one
    request across a pause never covers a different one. The bridge passes the
    arguments a grant must match (`ExternalAgent.grant_arguments`), not always all.
    """
    try:
        body = canonical_json(dict(arguments or {}))
    except (ValueError, TypeError):  # NaN, a lone surrogate: still deterministic, just not JCS
        body = json.dumps(dict(arguments or {}), sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha256(f"{tool}\n{body}".encode()).hexdigest()


class ExternalApproval(ToolResultPart):
    """A human's answer to an external agent's permission request that paused the run.

    With ``on_permission="pause"``, a request that needs a human ends the
    external turn and pauses the calling run with a `PendingInput` (reason
    ``"permission"``; its ``metadata`` has ``agent``, ``external_session``,
    ``tool``, ``arguments`` and ``fingerprint``, and ``continuable: False``
    for a tool built with ``as_tool(continuable=False)``). Answer it as its result:

        pending = result.pending_inputs[0]
        await lead.resume(session, results={
            pending.tool_call_id: ExternalApproval.for_input(pending, approved=True),
        })

    The calling model reads the approval, calls the external agent again with
    the same session (a tool without the ``session`` parameter continues it by
    itself), and the bridge allows the first request with the same
    fingerprint, once: the same tool and arguments, free text the agent
    rewords aside (`ExternalAgent.grant_ignored_arguments`). A changed request
    is decided again. Deny rules still refuse it whatever the approval says.

    ``ExternalApproval(approved=...)`` without the pending input works too; the
    model then reads a shorter text. With signed pauses wrap it in
    ``InputResult.for_call(pending, ...)`` as usual.
    """

    def __init__(self, approved: bool = True, *, reason: str | None = None, **data: Any) -> None:
        data.setdefault("tool_call_id", "")
        data.setdefault("name", "")
        content = data.pop("content", None)
        if not content:
            why = f" ({reason})" if reason else ""
            if approved:
                content = prompts.EXTERNAL_AGENT_APPROVED_GENERIC
            else:
                content = prompts.EXTERNAL_AGENT_DECLINED_GENERIC.format(reason=why)
        metadata = dict(data.pop("metadata", None) or {})
        metadata[APPROVAL_KEY] = {"approved": bool(approved), "reason": reason}
        data.setdefault("is_error", not approved)
        super().__init__(content=content, metadata=metadata, **data)

    @classmethod
    def for_input(
        cls, pending: PendingInput, approved: bool = True, *, reason: str | None = None
    ) -> ExternalApproval:
        """The answer to ``pending``, naming the agent, the request and the session."""
        meta = pending.metadata
        name = str(meta.get("agent") or pending.name)
        tool = str(
            meta.get("external_tool") or meta.get("tool") or prompts.EXTERNAL_AGENT_UNNAMED_TOOL
        )
        record = ExternalPermission(
            tool=str(meta.get("tool") or tool),
            external_tool=tool,
            arguments=dict(meta.get("arguments") or {}),
            outcome="cancel",
            decided_by="deferred",
        )
        session = str(meta.get("external_session") or "")
        if meta.get("continuable") is False:  # a tool without the session parameter
            session = ""  # its next call continues the paused session by itself
        values = {"name": name, "tool": tool, "detail": record.detail(), "session": session}
        if approved:
            template = (
                prompts.EXTERNAL_AGENT_APPROVED
                if session
                else prompts.EXTERNAL_AGENT_APPROVED_NO_SESSION
            )
            content = template.format(**values)
        else:
            template = (
                prompts.EXTERNAL_AGENT_DECLINED
                if session
                else prompts.EXTERNAL_AGENT_DECLINED_NO_SESSION
            )
            content = template.format(**values, reason=f" ({reason})" if reason else "")
        return cls(approved, reason=reason, content=content)

    @property
    def approved(self) -> bool:
        return bool(self.metadata.get(APPROVAL_KEY, {}).get("approved"))


class _AnyArguments(BaseModel):
    model_config = ConfigDict(extra="allow")


def _never_run(*_: Any, **__: Any) -> Any:
    raise ToolError("This stands for a tool of an external agent; qagents never runs it.")


class ExternalTool(Tool):
    """Stands for one tool of an external agent in a permission decision.

    `Permissions.decide`, ``can_use_tool`` and ``on_ask`` receive it. Its
    ``name`` is prefixed (``coder.bash``), which a qagents tool name could not
    be, so it is never sent to a model, and qagents never runs it. It has
    ``requires_approval=True`` (the agent asks only when its own policy says
    so) and ``read_only=False``.
    """

    def __init__(self, name: str, *, agent: str, external_name: str) -> None:
        # Not Tool.__init__: that would refuse the dot in the name.
        self.function = _never_run
        self.name = name
        self.description = f"Tool {external_name!r} of the external agent {agent!r}."
        self.arguments_model = _AnyArguments
        self.takes_context = False
        self.requires_approval = True
        self.read_only = False
        self.idempotent = False
        self.timeout = None
        self.strict = False
        self.deferred = False
        self.parameters = {"type": "object"}
        self.agent = agent
        """The prefix: the tool name the calling model sees (or the agent's name)."""
        self.external_name = external_name
        """The external agent's own name for the tool."""

    def __repr__(self) -> str:
        return f"ExternalTool({self.name!r})"


_DEFAULT_TIMEOUTS: dict[str, float] = {"rules": 60.0, "pause": 60.0, "live": 900.0}


class PermissionBridge:
    """Turns an external agent's permission requests into qagents decisions.

    For each request it builds a synthetic call ``f"{prefix}.{tool}"`` with the
    request's arguments and an `ExternalTool`, asks ``permissions.decide``
    (deny rules, ``can_use_tool``, ask rules, allow rules, the mode default),
    then:

    * deny: answer deny ("reject": this one action is refused, the turn goes
      on). A ``can_use_tool`` that raises is a deny too: `Permissions.decide`
      refuses the call (`PERMISSION_REASON_CHECK_FAILED`), as for the model's
      own calls. So is an ``on_ask`` that raises (a declined approval).
    * allow: answer allow (one call).
    * ask: a grant from an earlier pause answers it once (``"pause"``); else
      ``on_ask`` answers it, within ``approval_timeout_s``; an approval with
      edited arguments is refused (they cannot be passed on); else the posture:
      ``"rules"`` and ``"live"`` refuse (``no_approver``), ``"pause"`` answers
      cancelled, cancels the turn, and the calling run pauses with a
      `PendingInput` (see `ExternalApproval`).
    * the bridge itself failed or the decision timed out, or the turn is being
      cancelled: answer cancelled.

    The postures differ in the default ``approval_timeout_s``: 60 seconds for
    ``"rules"`` and ``"pause"``, 15 minutes for ``"live"``, which is meant for
    an ``on_ask`` that waits for a person (a UI round trip) in a long-lived
    process. The whole decision, rules included, is bounded by that timeout.

    Every decision is appended to ``records`` and reported through
    ``on_record`` (the turn turns it into ``permission_requested`` and
    ``permission_decided`` events).

    Args:
        prefix: Prefix of the synthetic tool names: the tool name the calling
            model sees, or the agent's name.
        permissions: The rules. ``None``: ``Permissions()``, which asks for
            every request, so without ``on_ask`` everything is refused.
        ctx: The calling run's context, passed to ``can_use_tool`` and ``on_ask``.
        posture: See `PermissionPosture`.
        approval_timeout_s: Seconds for one whole decision; ``None``: the
            posture's default.
        grants: Answers given across a pause: ``{fingerprint: approved}``.
            Each is used once.
        grant_arguments: ``(tool, arguments) -> arguments``: the part of a
            request's arguments its fingerprint covers, so that a grant still
            matches when the agent rewords free text such as a ``description``
            (`ExternalAgent.grant_arguments`). ``None``: all of them. If it
            raises, all of them are used (the stricter match).
    """

    def __init__(
        self,
        *,
        prefix: str,
        permissions: Permissions | None = None,
        ctx: RunContext[Any] | None = None,
        posture: PermissionPosture = "rules",
        approval_timeout_s: float | None = None,
        grants: Mapping[str, bool] | None = None,
        on_record: Callable[[ExternalPermission, str], None] | None = None,
        on_defer: Callable[[ExternalPermission], bool] | None = None,
        is_cancelling: Callable[[], bool] | None = None,
        grant_arguments: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> None:
        if posture not in _DEFAULT_TIMEOUTS:
            raise ConfigurationError(
                f"on_permission must be 'rules', 'pause' or 'live', not {posture!r}."
            )
        self.prefix = prefix
        self.permissions = permissions or Permissions()
        self.ctx = ctx
        self.posture: PermissionPosture = posture
        self.approval_timeout_s = (
            approval_timeout_s if approval_timeout_s is not None else _DEFAULT_TIMEOUTS[posture]
        )
        self.grants: dict[str, bool] = dict(grants or {})
        self.used_grants: list[str] = []
        """Fingerprints of grants that answered a request."""
        self.records: list[ExternalPermission] = []
        self._on_record = on_record
        self._on_defer = on_defer
        self._is_cancelling = is_cancelling or (lambda: False)
        self._grant_arguments = grant_arguments
        self._count = 0

    async def decide(
        self,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        tool_call_id: str | None = None,
        reason: str | None = None,
        options: Sequence[Mapping[str, Any]] | None = None,
        parse_error: str | None = None,
    ) -> PermissionAnswer:
        """Decide one request. Never raises: any failure answers ``cancel``.

        Args:
            tool: The external agent's tool name (unprefixed), for example ``bash``.
            arguments: The arguments it wants to run the tool with; empty when unknown.
            tool_call_id: The agent's id of the call, when it gave one.
            reason: The agent's reason for asking, when it gave one.
            options: ACP-style options (``{optionId, kind}``) the agent offered. The
                answer's ``option_id`` is then chosen (never ``allow_always``), and an
                allow with no ``allow_once`` option becomes ``cancel`` (``no_option``)
                before it is recorded.
            parse_error: Set when the tool call's input could not be read as a JSON
                object: the rules cannot see what would run, so the request is answered
                ``cancel`` (``unreadable_arguments``) without asking them (fail closed).
        """
        name = f"{self.prefix}.{tool}"
        args = dict(arguments or {})
        self._count += 1
        record = ExternalPermission(
            tool=name,
            external_tool=tool,
            arguments=args,
            tool_call_id=tool_call_id,
            outcome="cancel",
            decided_by="error",
            reason=reason,
            fingerprint=permission_fingerprint(name, self._matched_arguments(tool, args)),
        )
        self._report(record, "permission_requested")
        if parse_error is not None:
            answer = PermissionAnswer(
                outcome="cancel",
                reason=prompts.EXTERNAL_PERMISSION_UNREADABLE,
                decided_by="unreadable_arguments",
            )
            if options is not None:
                answer = _with_option(answer, options)
            self._finish(record, answer)
            return answer
        try:
            answer = await self._decide(record, tool_call_id)
        except asyncio.CancelledError:
            answer = PermissionAnswer(
                outcome="cancel",
                reason=prompts.EXTERNAL_PERMISSION_CANCELLING,
                decided_by="cancelling",
            )
            self._finish(record, answer)
            raise
        except Exception:
            logger.exception("Deciding the permission request for %s failed", name)
            answer = PermissionAnswer(
                outcome="cancel", reason=prompts.EXTERNAL_PERMISSION_FAILED, decided_by="error"
            )
        if answer.outcome != "cancel" and self._is_cancelling():
            # The turn was cancelled while this was decided: the agent is answered
            # "cancelled", so that is what is recorded (and what the event says).
            answer = PermissionAnswer(
                outcome="cancel",
                reason=prompts.EXTERNAL_PERMISSION_CANCELLING,
                decided_by="cancelling",
            )
        if options is not None:
            answer = _with_option(answer, options)
        self._finish(record, answer)
        return answer

    async def _decide(
        self, record: ExternalPermission, tool_call_id: str | None
    ) -> PermissionAnswer:
        if self._is_cancelling():
            return PermissionAnswer(
                outcome="cancel",
                reason=prompts.EXTERNAL_PERMISSION_CANCELLING,
                decided_by="cancelling",
            )
        ctx = self.ctx if self.ctx is not None else RunContext(agent_name=self.prefix)
        tool = ExternalTool(record.tool, agent=self.prefix, external_name=record.external_tool)
        call = ToolCallPart(
            id=tool_call_id or f"external-permission-{self._count}",
            name=record.tool,
            arguments=record.arguments,
        )
        seconds = self.approval_timeout_s
        try:
            async with asyncio.timeout(seconds if seconds and seconds > 0 else None):
                decision: Decision = await self.permissions.decide(ctx, tool, call)
                record.decision = decision.behavior
                if decision.behavior == "deny":
                    return PermissionAnswer(
                        outcome="deny", reason=decision.reason, decided_by="rules"
                    )
                if decision.behavior == "allow":
                    return PermissionAnswer(outcome="allow", decided_by="rules")
                granted = self.grants.pop(record.fingerprint, None)
                if granted is not None:
                    self.used_grants.append(record.fingerprint)
                    if granted:
                        return PermissionAnswer(outcome="allow", decided_by="grant")
                    return PermissionAnswer(
                        outcome="deny",
                        reason=prompts.EXTERNAL_PERMISSION_DECLINED,
                        decided_by="declined",
                    )
                if self.permissions.on_ask is not None:
                    approval: Approval = await self.permissions.request_approval(
                        ctx, tool, call, decision.reason
                    )
                    if approval.approved and approval.arguments not in (None, call.arguments):
                        return PermissionAnswer(
                            outcome="deny",
                            reason=prompts.EXTERNAL_PERMISSION_EDITED,
                            decided_by="edited_arguments",
                        )
                    if approval.approved:
                        return PermissionAnswer(outcome="allow", decided_by="on_ask")
                    return PermissionAnswer(
                        outcome="deny", reason=approval.reason, decided_by="on_ask"
                    )
        except TimeoutError:
            return PermissionAnswer(
                outcome="cancel",
                reason=prompts.EXTERNAL_PERMISSION_TIMEOUT.format(seconds=seconds or 0),
                decided_by="timeout",
            )
        if self.posture == "pause" and self._on_defer is not None:
            if self._on_defer(record):
                return PermissionAnswer(
                    outcome="cancel",
                    reason=prompts.EXTERNAL_PERMISSION_DEFERRED,
                    decided_by="deferred",
                )
            if self._is_cancelling():  # a second request while the first one pauses the run
                return PermissionAnswer(
                    outcome="cancel",
                    reason=prompts.EXTERNAL_PERMISSION_CANCELLING,
                    decided_by="cancelling",
                )
            # no calling run to pause (a direct run): refuse, as the "rules" posture does
        return PermissionAnswer(
            outcome="deny", reason=prompts.EXTERNAL_PERMISSION_NO_APPROVER, decided_by="no_approver"
        )

    def _matched_arguments(self, tool: str, arguments: dict[str, Any]) -> Mapping[str, Any]:
        """The arguments a grant for this request must match (all, when unsure)."""
        if self._grant_arguments is None:
            return arguments
        try:
            matched = self._grant_arguments(tool, dict(arguments))
        except Exception:
            logger.exception("grant_arguments of %s.%s raised; matching all", self.prefix, tool)
            return arguments
        return matched if isinstance(matched, Mapping) else arguments

    def _finish(self, record: ExternalPermission, answer: PermissionAnswer) -> None:
        record.outcome = answer.outcome
        record.decided_by = answer.decided_by
        record.reason = answer.reason or record.reason
        self.records.append(record)
        logger.info(
            "Permission %s for %s (%s): %s",
            answer.outcome,
            record.tool,
            answer.decided_by,
            answer.reason or "",
        )
        self._report(record, "permission_decided")

    def _report(self, record: ExternalPermission, kind: str) -> None:
        if self._on_record is not None:
            self._on_record(record, kind)
