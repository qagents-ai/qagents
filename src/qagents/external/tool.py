"""`ExternalAgent.as_tool`: a turn of the external agent as one tool call of a calling run.

The tool body runs the turn as a child of the calling run (depth, budgets, events as
`SubagentEvent`, cancellation), turns its `ExternalResult` into the text and metadata the
calling model reads, and keeps what must outlive the call in the calling run's
``ctx.state``: the sessions the tool opened (`STATE_SESSIONS`, the only ones the model may
continue) and the permission requests that paused the run (`STATE_GRANTS`). A resumable
tool journals its result as a checkpoint before it returns, so a repeat after a crash
returns it without running the turn again.

It lives apart from `ExternalAgent` because this is the part that talks to the calling
run: a direct `ExternalAgent.stream` or ``run`` uses none of it.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict, Field, create_model

from .. import prompts
from ..errors import (
    DeferToolCall,
    ExternalAgentConfigurationError,
    ExternalAgentConnectionError,
    ExternalAgentUnavailableError,
    ExternalToolError,
    ToolError,
)
from ..events import SubagentEvent
from ..messages import ToolResultPart
from ..tools import RunContext, Tool
from .contract import ExternalResult, _result_summary, is_permanent_failure
from .permissions import APPROVAL_KEY

if TYPE_CHECKING:
    from .agent import ExternalAgent

logger = logging.getLogger("qagents.external")

STATE_SESSIONS = "qagents.external.sessions"
"""``ctx.state`` key: the sessions each external tool opened in this qagents session,
``{tool name: [{"session_id", "cwd", "backend"}, ...]}`` (latest last, at most 50)."""

STATE_GRANTS = "qagents.external.grants"
"""``ctx.state`` key: permission requests that paused the run (posture ``"pause"``),
``{parent tool call id: {"agent", "session", "tool", "fingerprint", "approved"}}``."""

_RESULT_CHECKPOINT = "external_result"
_SESSION_CHECKPOINT = "external_session"
_MAX_REMEMBERED_SESSIONS = 50


def _build_tool(
    agent: ExternalAgent,
    *,
    name: str | None,
    description: str | None,
    timeout: float | None,
    continuable: bool,
    requires_approval: bool,
    idempotent: bool | None,
) -> Tool:
    """The tool `ExternalAgent.as_tool` returns (see there)."""
    label = name or _tool_safe(agent.name)
    repeatable = agent.resumable_turns if idempotent is None else idempotent
    margin = agent.cancel_grace_s + 5.0
    broken: dict[str, str] = {}

    async def call_external(ctx: RunContext[Any], task: str, session: str | None = None) -> Any:
        return await _call(
            agent,
            ctx,
            task,
            session,
            label=label,
            repeatable=repeatable,
            timeout=timeout,
            margin=margin,
            broken=broken,
            continuable=continuable,
        )

    fields: dict[str, Any] = {
        "task": (str, Field(min_length=1, description=prompts.EXTERNAL_AGENT_TASK_DESCRIPTION))
    }
    if continuable:
        fields["session"] = (
            str | None,
            Field(default=None, description=prompts.EXTERNAL_AGENT_SESSION_DESCRIPTION),
        )
    arguments = create_model(f"{label}_arguments", __config__=ConfigDict(extra="forbid"), **fields)
    text = description or agent.description or prompts.EXTERNAL_AGENT_DESCRIPTION.format(name=label)
    if continuable and description is None:
        text = f"{text} {prompts.EXTERNAL_AGENT_CONTINUE_HINT}"
    return Tool(
        call_external,
        name=label,
        description=text,
        arguments_model=arguments,
        takes_context=True,
        requires_approval=requires_approval,
        read_only=False,
        idempotent=repeatable,
        timeout=timeout + margin if timeout is not None else None,
    )


async def _call(
    agent: ExternalAgent,
    ctx: RunContext[Any],
    task: str,
    session: str | None,
    *,
    label: str,
    repeatable: bool,
    timeout: float | None,
    margin: float,
    broken: dict[str, str],
    continuable: bool = True,
) -> Any:
    """The body of the ``as_tool()`` tool."""
    if (message := broken.get(ctx.run_id)) is not None:
        raise ToolError(message)
    if (refusal := _depth_refusal(ctx, label)) is not None:
        raise ToolError(refusal)
    checkpoints = dict(ctx.checkpoints) if repeatable else {}
    recorded = checkpoints.get(_RESULT_CHECKPOINT)
    if isinstance(recorded, Mapping) and isinstance(recorded.get("content"), str):
        # This call finished in an earlier attempt; only its tool_result was lost. Its
        # usage was journaled with the checkpoint, so the resumed run counts it already.
        summary = (recorded.get("metadata") or {}).get("external") or {}
        if isinstance(summary, Mapping) and summary.get("session_id"):
            _remember(ctx, label, summary["session_id"], agent.workspace, summary.get("backend"))
        if recorded.get("is_error"):
            raise ExternalToolError(recorded["content"], metadata=recorded.get("metadata"))
        return _tool_value(recorded["content"], False, recorded.get("metadata") or {})
    if session is not None and session not in _opened_sessions(ctx, label):
        # Only sessions this tool opened in this qagents session continue: an id the model
        # made up, or one of another tool, workspace or conversation, is refused.
        unknown = prompts.DEVVI_UNKNOWN_SESSION.format(session=session)
        raise ExternalToolError(
            prompts.EXTERNAL_AGENT_FAILED.format(name=label, error=unknown),
            metadata={"unknown_session": session},
        )
    try:
        await agent.ensure_connected()
        refusal = await agent._unmetered_refusal(ctx, label)
    except (ExternalAgentConfigurationError, ExternalAgentUnavailableError) as error:
        raise _trip(broken, ctx, label, error) from error
    except ExternalAgentConnectionError as error:
        raise _trip(broken, ctx, label, error) from error
    if refusal is not None:
        logger.warning(
            "%s reports no token usage, so it cannot run under a token or cost budget. "
            "Build it with allow_unmetered=True to accept that its spend is not counted.",
            label,
        )
        _open_breaker(broken, ctx.run_id, refusal)
        raise ToolError(refusal)

    # One-shot: the answers to earlier pauses are used up now, before the turn runs
    # (record first, act second), so a continuation that is cancelled cannot leave
    # them behind for another call.
    # A tool without the session parameter continues the paused session by itself.
    grants, session = _take_grants(ctx, label, session, any_session=not continuable)
    if grants:
        await ctx.save_state()
    prompt = prompts.EXTERNAL_AGENT_CONTINUE.format(task=task) if grants else task
    turn = agent._open_turn(
        prompt,
        session=session,
        cwd=None,
        parent=ctx,
        permissions=None,
        label=label,
        grants=grants,
        checkpoints=checkpoints,
        resumable=repeatable,
    )
    limits = ctx.limits
    deadline = timeout
    if deadline is None and limits is not None and limits.tool_timeout_s is not None:
        deadline = _graceful_deadline(limits.tool_timeout_s, margin)
    loop = asyncio.get_running_loop()
    call_id = ctx.tool_call_id
    try:
        async with turn:
            expiry = loop.call_later(deadline, turn.cancel, "timeout") if deadline else None
            try:
                async for event in turn:
                    ctx.emit(SubagentEvent(agent=label, tool_call_id=call_id, event=event))
            finally:
                if expiry is not None:
                    expiry.cancel()
    except ExternalAgentConfigurationError as error:
        raise _trip(broken, ctx, label, error) from error
    result = turn.result
    assert result is not None
    _remember_session(ctx, label, result)
    deferred = result.raw.get("permission_deferred")
    if deferred is not None and result.status != "completed":
        _record_pending_grant(ctx, label, deferred)
        if not continuable:  # the answer must not tell the model to pass session=
            deferred = {**deferred, "continuable": False}
        raise DeferToolCall("permission", metadata=deferred)
    content, is_error, metadata = _tool_outcome(label, result, continuable=continuable)
    if result.raw.get("unavailable") or (
        result.status == "error" and is_permanent_failure(result.stop_reason or "")
    ):
        _open_breaker(broken, ctx.run_id, content)
    if repeatable:
        # Journaled before the result is handed back: a repeat after a crash returns it
        # without running the turn again. The turn's usage that no entry carries yet
        # rides on this checkpoint, so the resumed run counts it exactly once.
        value: dict[str, Any] = {"content": content, "is_error": is_error, "metadata": metadata}
        await ctx.checkpoint(_RESULT_CHECKPOINT, value)
    if is_error:
        raise ExternalToolError(content, metadata=metadata)
    return _tool_value(content, False, metadata)


def _tool_value(content: str, is_error: bool, metadata: Mapping[str, Any]) -> ToolResultPart:
    """The tool's return value: a result the loop keeps as it is (content, error flag and
    metadata), filling in the call id and name."""
    return ToolResultPart(
        tool_call_id="",
        name="",
        content=content,
        is_error=is_error,
        metadata=dict(metadata),
    )


def _trip(broken: dict[str, str], ctx: RunContext[Any], label: str, error: Exception) -> ToolError:
    """Open the circuit breaker for this run, and the error result that says why."""
    logger.error("%s is unavailable: %s", label, error)
    reason = error.reason if isinstance(error, ExternalAgentConnectionError) else str(error)
    message = prompts.EXTERNAL_AGENT_UNAVAILABLE.format(name=label, error=_first_line(reason))
    _open_breaker(broken, ctx.run_id, message)
    return ToolError(message)


def _tool_outcome(
    label: str, result: ExternalResult, *, continuable: bool = True
) -> tuple[str, bool, dict[str, Any]]:
    """The content the calling model reads, whether it is an error, and the result's metadata.

    ``continuable=False``: the tool has no ``session`` parameter, so no text offers one."""
    footer = result.footer(label)
    metadata: dict[str, Any] = {"external": _result_summary(result)}
    if result.status == "completed" and result.stop_reason != "content_filter":
        return result.for_model(label), False, metadata
    if result.raw.get("not_started") and result.stop_reason == "busy":
        message = prompts.EXTERNAL_AGENT_BUSY.format(name=label)
    elif result.status == "interrupted":
        metadata["ambiguous"] = True
        reason = result.stop_reason or prompts.EXTERNAL_REASON_LOST
        if not continuable:
            message = prompts.EXTERNAL_AGENT_INTERRUPTED_NOT_CONTINUABLE.format(
                name=label, reason=reason
            )
        elif result.session_id:
            message = prompts.EXTERNAL_AGENT_INTERRUPTED.format(
                name=label, reason=reason, session=result.session_id
            )
        else:  # broke before a session was opened: nothing to continue
            message = prompts.EXTERNAL_AGENT_INTERRUPTED_NO_SESSION.format(
                name=label, reason=reason
            )
    elif result.status == "error":
        error = _first_line(result.stop_reason or prompts.EXTERNAL_AGENT_UNKNOWN_ERROR)
        if result.raw.get("unavailable") or is_permanent_failure(result.stop_reason or ""):
            message = prompts.EXTERNAL_AGENT_UNAVAILABLE.format(name=label, error=error)
        else:
            message = prompts.EXTERNAL_AGENT_FAILED.format(name=label, error=error)
    else:
        message = prompts.EXTERNAL_AGENT_STOPPED.format(
            name=label, reason=result.stop_reason or result.status
        )
        if result.text.strip():
            message = f"{message}\n\n{result.text.strip()}"
    return f"{message}\n\n[{footer}]", True, metadata


def _open_breaker(broken: dict[str, str], run_id: str, message: str) -> None:
    """Further calls in run ``run_id`` fail at once with ``message``."""
    broken[run_id] = message
    while len(broken) > 100:  # keep it small in long-lived processes: oldest runs first
        broken.pop(next(iter(broken)))


def _first_line(text: str, limit: int = 300) -> str:
    """The first line of ``text``, without closing punctuation (the templates end their own
    sentence), shortened to ``limit`` characters."""
    stripped = (text or "").strip()
    line = stripped.splitlines()[0].rstrip(" .!?") if stripped else ""
    return line if len(line) <= limit else line[: limit - 3] + "..."


def _depth_refusal(parent: RunContext[Any] | None, label: str) -> str | None:
    """Why a child turn may not start: it would nest beyond ``Limits.max_depth``."""
    if parent is None:
        return None
    limits = parent.limits
    limit = limits.max_depth if limits is not None else None
    if limit is not None and parent.depth + 1 > limit:
        return prompts.EXTERNAL_AGENT_TOO_DEEP.format(
            name=label, depth=parent.depth + 1, limit=limit
        )
    return None


def _graceful_deadline(tool_timeout_s: float, margin: float) -> float | None:
    """When ``as_tool()`` without a timeout cancels a turn itself: ``margin`` seconds
    (``cancel_grace_s + 5``) before the calling loop's ``tool_timeout_s``, so the cancel
    is graceful and the model is told the session. Only when that leaves the turn more
    than half of ``tool_timeout_s`` (``tool_timeout_s > 2 * margin``); otherwise ``None``:
    the loop's own timeout ends the call, and its text names no session."""
    spare = tool_timeout_s - margin
    return spare if spare > margin else None


# --- grants across a pause -----------------------------------------------------------------


def _record_pending_grant(ctx: RunContext[Any], label: str, deferred: Mapping[str, Any]) -> None:
    grants = ctx.state.setdefault(STATE_GRANTS, {})
    grants[str(ctx.tool_call_id)] = {
        "agent": label,
        "session": deferred.get("external_session"),
        "tool": deferred.get("tool"),
        "fingerprint": deferred.get("fingerprint"),
        "approved": None,
    }


def _take_grants(
    ctx: RunContext[Any], label: str, session: str | None, *, any_session: bool = False
) -> tuple[dict[str, bool], str | None]:
    """Take the answers to earlier pauses of this agent's ``session``: ``{fingerprint:
    approved}``, and the session they belong to.

    The answer is the result the human supplied for the paused call, found in
    the conversation: an `ExternalApproval` approves or declines; any other
    value declines (fail closed). Every answered pause is removed from
    ``ctx.state`` here, before the turn that uses it runs, so a grant is good
    for that one turn whatever happens to it (it may be cancelled or time
    out). Pauses still waiting for an answer stay.

    ``any_session``: the tool has no ``session`` parameter (``continuable=False``).
    Without a ``session``, the latest answered pause of this agent then names the
    session to continue, so the answer reaches the agent that asked.
    """
    stored = ctx.state.get(STATE_GRANTS)
    if not isinstance(stored, dict):
        return {}, session
    answers: dict[str, ToolResultPart] = {}
    for message in ctx.messages:
        for part in message.tool_results:
            if part.tool_call_id in stored:
                answers[part.tool_call_id] = part
    if session is None and any_session:
        for call_id, entry in stored.items():  # in the order of the pauses: the latest wins
            if (
                isinstance(entry, dict)
                and entry.get("agent") == label
                and call_id in answers
                and isinstance(entry.get("session"), str)
                and entry["session"]
            ):
                session = entry["session"]
    if session is None:
        return {}, None
    grants: dict[str, bool] = {}
    for call_id, entry in list(stored.items()):
        if not isinstance(entry, dict) or entry.get("agent") != label:
            continue
        if entry.get("session") != session or not isinstance(entry.get("fingerprint"), str):
            continue
        answer = answers.get(call_id)
        if answer is None:
            continue  # still waiting
        del stored[call_id]  # answered: used up by this call
        approval = answer.metadata.get(APPROVAL_KEY)
        if isinstance(approval, Mapping):
            grants[entry["fingerprint"]] = approval.get("approved") is True
        # Any other answer (a plain value, an abandoned call) grants nothing: the
        # request is decided afresh, and pauses again if it still needs a human.
    if not stored:
        del ctx.state[STATE_GRANTS]
    return grants, session


def _remember_session(ctx: RunContext[Any], label: str, result: ExternalResult) -> None:
    if result.session_id:
        _remember(ctx, label, result.session_id, result.workspace, result.backend)


def _remember(
    ctx: RunContext[Any], label: str, session_id: str, cwd: str | None, backend: str | None
) -> None:
    sessions = ctx.state.setdefault(STATE_SESSIONS, {})
    entries = [
        entry
        for entry in sessions.get(label, [])
        if isinstance(entry, dict) and entry.get("session_id") != session_id
    ]
    entries.append({"session_id": session_id, "cwd": cwd, "backend": backend})
    sessions[label] = entries[-_MAX_REMEMBERED_SESSIONS:]


def _opened_sessions(ctx: RunContext[Any], label: str) -> set[str]:
    """The sessions the tool ``label`` opened (or continued) in this qagents session, as
    `_remember` keeps them in ``ctx.state``: the only ones the calling model may continue."""
    sessions = ctx.state.get(STATE_SESSIONS)
    entries = sessions.get(label) if isinstance(sessions, Mapping) else None
    if not isinstance(entries, list):
        return set()
    return {
        entry["session_id"]
        for entry in entries
        if isinstance(entry, Mapping) and isinstance(entry.get("session_id"), str)
    }


def _tool_safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)[:48] or "agent"
