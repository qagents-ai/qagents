"""What an adapter and qagents exchange about a turn, as plain data.

The status and stop-reason words every adapter uses (`ExternalStatus`, `STOP_REASONS`),
what an agent supports (`ExternalCapabilities`), the outcome an adapter's ``run_turn``
returns (`ExternalOutcome`), and the `ExternalResult` qagents builds from it, with the
`ExternalToolCall` records and the footer the calling model reads.

    result = await coder.run("Make tests/test_parser.py pass.")
    if result.completed:
        print(result.text, result.changed_files)
    else:
        print(result.status, result.stop_reason)  # for example limit_reached, timeout
    print(result.for_model("coder"))  # the text and footer a calling model reads

Everything here is JSON-serialisable and holds no connection, so a result can be journaled,
recorded by a `Workflow` step, and compared in tests.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from .. import prompts
from ..messages import Usage
from .changes import _git_diff
from .permissions import ExternalPermission

ExternalStatus = Literal["completed", "cancelled", "limit_reached", "error", "interrupted"]
"""How an external turn ended.

* ``completed``: the agent finished its turn (``stop_reason`` may still say
  ``max_tokens`` for agents that report it as a normal end).
* ``cancelled``: qagents (or the agent) cancelled the turn. ``stop_reason`` ``busy``: it
  never started, because another turn held its workspace (or session) until the call's
  timeout ran out; ``raw["not_started"]`` is set for every turn cancelled before it started.
* ``limit_reached``: a limit stopped it; ``stop_reason`` names it (``timeout``,
  ``max_total_tokens``, ``max_cost_usd``, ``max_external_turns``, ``max_tokens``).
* ``error``: the agent reported a failed turn; ``stop_reason`` is its message.
* ``interrupted``: the connection broke during the turn (the process exited,
  EOF, a broken pipe): what the agent did is unknown.
"""

LIMIT_REASONS = frozenset({"timeout", "max_total_tokens", "max_cost_usd", "max_external_turns"})
"""Cancel reasons that end a turn as ``limit_reached`` rather than ``cancelled``."""

STOP_REASONS = frozenset(
    {"end_turn", "max_tokens", "cancelled", "content_filter", "busy", *LIMIT_REASONS}
)
"""The ``stop_reason`` words every external agent uses for the same outcome:
``end_turn`` (the agent finished its turn), ``max_tokens`` (an output-token limit cut it
off), ``cancelled`` (qagents or the agent cancelled it), ``content_filter`` (the model
refused), ``busy`` (the call's time ran out before the turn could start: another turn held
the workspace), and the limit names of `LIMIT_REASONS`. Adapters map their agent's own words onto
these and keep the original in ``ExternalResult.raw`` (``acp_stop_reason``,
``agent_stop_reason``). An ``error`` turn's ``stop_reason`` is the agent's message; a word an
adapter cannot map passes through unchanged."""

CHECKPOINT_KIND = "tool_checkpoint"
"""Journal entry kind for facts a tool call records before it ends
(``{tool_call_id, operation_id, key, value}``); readers that do not know it ignore it."""

WRITE_TOOLS = ("write", "edit", "str_replace_editor")
"""Tools whose ``path`` (or ``file_path``) argument is added to ``changed_files``."""

_SUMMARY_FILES = 100
"""Changed files named in the ``turn_finished`` event, the ``external_result`` checkpoint
and the tool result's metadata; the rest are counted (``changed_files_count``)."""


_PERMANENT = re.compile(
    r"\b40[123]\b|unauthori[sz]ed|authenticat|forbidden|invalid[ _-]?api[ _-]?key|"
    r"\bapi[ _-]?key\b|billing|payment required|insufficient[ _-](?:balance|credit|funds|quota)|"
    r"quota exceeded|missing[ _-]?credential",
    re.IGNORECASE,
)


def is_permanent_failure(text: str) -> bool:
    """Whether an agent's error text describes a failure that will repeat.

    Recognises authentication (401, 403, "unauthorized", "invalid API key"),
    billing (402, "insufficient balance", "quota exceeded") and missing
    credentials (``MISSING_CREDENTIAL``). ACP carries no structured error
    codes for these, so the text is all there is.
    """
    return bool(_PERMANENT.search(text or ""))


# --- results ---------------------------------------------------------------------------


class ExternalCapabilities(BaseModel):
    """What an external agent supports. Adapters report it from the handshake."""

    token_usage: bool = False
    """It reports token usage per model request (Devvi SDK: yes; ACP: no). Without it a
    calling run with a token or cost budget refuses to start it, unless the agent was
    built with ``allow_unmetered=True``."""
    live_tokens: bool = False
    """Text arrives token by token. Devvi sends whole committed blocks instead."""
    cancel_turn: bool = False
    """A turn can be cancelled and the process kept (ACP: yes; Devvi SDK: process stop only)."""
    resume_sessions: bool = False
    """A session can be continued in a later turn, also from another process."""
    client_session_ids: bool = False
    """qagents may choose the session id (Devvi SDK), so a repeated call finds its session."""
    permission_requests: bool = False
    """The agent asks qagents before running tools its own policy marks as needing approval."""
    mcp_servers: bool = False
    """The agent can mount MCP servers (qagents tools) per session or per process."""


class ExternalToolCall(BaseModel):
    """One tool call the external agent made during a turn."""

    id: str
    name: str
    """The external agent's own tool name, without the prefix (``bash``, not ``coder.bash``)."""
    arguments: dict[str, Any] = Field(default_factory=dict)
    is_error: bool = False
    finished: bool = False
    """It reported a result. Calls still open when the turn ended have ``False``."""
    duration_ms: float | None = None
    """Measured by qagents from the start event to the result event."""


class ExternalResult(BaseModel):
    """The outcome of one external turn. JSON-serialisable, so `Workflow.step` can record it."""

    status: ExternalStatus
    stop_reason: str | None = None
    """One of `STOP_REASONS` (``end_turn``, ``max_tokens``, ``cancelled``,
    ``content_filter``, a limit name), the agent's error text, or an agent-specific word the
    adapter could not map. ``raw["agent_stop_reason"]`` (or ``raw["acp_stop_reason"]``) keeps
    the agent's own word."""
    text: str = ""
    """The last assistant message of the turn (text after the last tool call, when there is
    some; otherwise the last text before it). Blocks the agent committed separately are joined
    with a blank line (see `ExternalTurnContext.text_delta`)."""
    session_id: str = ""
    """The external agent's session id: pass it as ``session=`` to continue. Empty when the
    turn never began in a session (refused, busy or broken before the agent confirmed one),
    also when the call asked to continue a session."""
    backend: str = ""
    """What served the turn, for example ``"acp:deepseek-harness-acp@0.0.1"``."""
    usage: Usage = Field(default_factory=Usage)
    """Token usage the agent reported; zero when it reports none (ACP)."""
    tool_calls: list[ExternalToolCall] = Field(default_factory=list)
    permissions: list[ExternalPermission] = Field(default_factory=list)
    changed_files: list[str] | None = None
    """Paths (relative to the repository root, ``/``-separated) the turn changed, when
    changes were tracked with git; ``None`` when they were not."""
    workspace: str | None = None
    """The working directory of the turn."""
    raw: dict[str, Any] = Field(default_factory=dict)
    """Backend-specific extras. Keys the base class sets: ``permission_deferred``
    (posture ``"pause"``), ``error`` and ``ambiguous`` (interrupted), ``stderr_tail``,
    ``unavailable``, ``forced`` (the turn did not stop in time and was abandoned),
    ``git_root``."""

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    def footer(self, name: str) -> str:
        """The bracketed summary the calling model reads after the text, without brackets."""
        if self.status == "completed" and self.stop_reason in (None, "end_turn", "completed"):
            status: str = self.status
        else:
            status = prompts.EXTERNAL_AGENT_STATUS.format(
                status=self.status.replace("_", " "), reason=self.stop_reason or self.status
            )
        if self.session_id:
            start = prompts.EXTERNAL_AGENT_FOOTER.format(
                name=name, session=self.session_id, status=status
            )
        else:  # no session was opened: there is nothing to pass back as session=
            start = prompts.EXTERNAL_AGENT_FOOTER_NO_SESSION.format(name=name, status=status)
        parts = [start]
        counts = Counter(call.name for call in self.tool_calls)
        if counts:
            summary = ", ".join(f"{tool} {count}" for tool, count in counts.most_common())
            total = sum(counts.values())
            template = (
                prompts.EXTERNAL_AGENT_FOOTER_ONE_TOOL
                if total == 1
                else prompts.EXTERNAL_AGENT_FOOTER_TOOLS
            )
            parts.append(template.format(count=total, summary=summary))
        else:
            parts.append(prompts.EXTERNAL_AGENT_FOOTER_NO_TOOLS)
        refused = [item for item in self.permissions if not item.approved]
        if refused:
            items = ", ".join(
                prompts.EXTERNAL_AGENT_FOOTER_REFUSED_ITEM.format(
                    tool=item.external_tool, detail=item.detail()
                )
                for item in refused[:5]
            )
            if len(refused) > 5:
                items += ", " + prompts.EXTERNAL_AGENT_FOOTER_MORE.format(count=len(refused) - 5)
            template = (
                prompts.EXTERNAL_AGENT_FOOTER_ONE_REFUSED
                if len(refused) == 1
                else prompts.EXTERNAL_AGENT_FOOTER_REFUSED
            )
            parts.append(template.format(count=len(refused), items=items))
        if self.changed_files:
            files = ", ".join(self.changed_files[:20])
            if len(self.changed_files) > 20:
                more = len(self.changed_files) - 20
                files += ", " + prompts.EXTERNAL_AGENT_FOOTER_MORE.format(count=more)
            parts.append(prompts.EXTERNAL_AGENT_FOOTER_CHANGED.format(files=files))
        return prompts.EXTERNAL_AGENT_FOOTER_SEPARATOR.join(parts)

    def for_model(self, name: str) -> str:
        """The final text and the footer, as the calling model reads them."""
        text = self.text.strip() or prompts.EXTERNAL_AGENT_NO_TEXT.format(name=name)
        return prompts.EXTERNAL_AGENT_RESULT.format(text=text, footer=self.footer(name))

    async def diff(self) -> str:
        """``git diff`` of ``changed_files`` (tracked files only), computed now, on demand.

        It is never sent to a model by qagents. Empty when changes were not tracked.
        Git runs as for change tracking: with a small environment and with the
        programs a repository's config could name switched off (external diff,
        textconv, filters, fsmonitor, pager), so a workspace the agent wrote to
        cannot make qagents run anything. Many paths are split over several git
        calls, so the command line stays within the Windows limit; when one of
        them fails the diff is empty.
        """
        root = self.raw.get("git_root") or self.workspace
        if not self.changed_files or not root:
            return ""
        return await _git_diff(root, self.changed_files)


class ExternalOutcome(BaseModel):
    """What an adapter's `ExternalAgent.run_turn` returns: how the turn ended.

    Everything else (text, usage, tool calls, permissions) was reported through
    the `ExternalTurnContext` while the turn ran.
    """

    status: ExternalStatus = "completed"
    stop_reason: str | None = None
    text: str | None = None
    """Replaces the text collected from ``text_delta`` when set."""
    changed_files: list[str] | None = None
    """Replaces the files found by git tracking when set."""
    raw: dict[str, Any] = Field(default_factory=dict)


def _result_summary(result: ExternalResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "stop_reason": result.stop_reason,
        "session_id": result.session_id,
        "backend": result.backend,
        "usage": result.usage.model_dump(mode="json"),
        "tool_calls": dict(Counter(call.name for call in result.tool_calls)),
        "permissions": [
            {
                "tool": item.tool,
                "outcome": item.outcome,
                "decided_by": item.decided_by,
                "reason": item.reason,
            }
            for item in result.permissions
        ],
        # The first paths only: this goes into events, the journal and the tool's metadata,
        # and a turn can change tens of thousands of files (an install, a build directory).
        "changed_files": (
            result.changed_files[:_SUMMARY_FILES] if result.changed_files is not None else None
        ),
        "changed_files_count": (
            len(result.changed_files) if result.changed_files is not None else None
        ),
    }
