"""External agents: another agent runtime that qagents drives and keeps in check.

A `Model` answers one request and qagents runs the loop. An `Agent` is a qagents
loop. An `ExternalAgent` is a different agent program (Devvi, any ACP agent)
that runs its own loop with its own tools and model, reached over a control
protocol. qagents gives it a task in a workspace, sees what it does as events,
decides every permission request it makes, and gets back text plus artifacts.

    coder = SomeExternalAgent(name="coder", workspace="C:/work/repo")
    lead = Agent(
        model,
        tools=[coder.as_tool(timeout=1800)],
        # The agent asks only about what its own policy marks; these rules decide those asks.
        permissions=Permissions(ask=["coder.execute", "coder.edit"], deny=["coder.delete"]),
    )
    async with coder:
        result = await lead.run("Make tests/test_parser.py pass.")

Used as a tool, a turn of the external agent is a child run of the calling run,
like an agent used as a tool (`Agent.as_tool`): it starts at depth + 1 and
refuses to start beyond ``Limits.max_depth``, its usage is added to the
caller's run as it is reported (so the caller's token and cost budgets stop
it), its events reach the caller's stream as `SubagentEvent`, cancelling the
caller (or its timeout) cancels the turn, and the caller's ``on_ask`` answers
its permission requests. The calling model reads the external agent's final
text followed by a footer (session, status, tool calls, refusals, changed
files), and passes ``session`` back to continue the same conversation.

Permissions: every permission request goes through a `PermissionBridge`, which
asks the caller's `Permissions` about a synthetic call named
``"<tool name>.<external tool>"`` (``coder.bash``), with ``requires_approval``
set, so "ask" is the default when no rule matches. Without an ``on_ask`` the
posture decides (``on_permission``): ``"rules"`` refuses, ``"pause"`` ends the
turn and pauses the calling run until a human answers with `ExternalApproval`,
``"live"`` is like ``"rules"`` with a long ``on_ask`` timeout. A ``can_use_tool``
or ``on_ask`` that raises refuses the request ("reject": that one action, the
turn goes on); a failure of the bridge itself, or a decision that times out,
answers "cancelled". An "allow always" option is never chosen.

Writing an adapter: subclass `ExternalAgent` and implement `capabilities` and
`run_turn`; optionally `connect`, `disconnect` and `cancel_turn`.
`run_turn` receives an `ExternalTurnContext` and reports everything through it
(`ExternalTurnContext.begin`, ``text_delta``, ``tool_started``,
``tool_finished``, ``add_usage``, ``ask_permission``, ...), then returns an
`ExternalOutcome`. The base class turns that into events, the `ExternalResult`,
the journal entries and the tool result.

Tests: `ScriptedExternalAgent` plays scripted turns in memory, the way
`qagents.models.testing.ScriptedModel` plays model replies.

Every string a model reads lives in `qagents.prompts`.

The package, by responsibility (import everything from ``qagents.external`` itself):
``contract`` (status words, capabilities, outcomes and results), ``permissions`` (the
`PermissionBridge`, `ExternalApproval`), ``turn`` (`ExternalTurnContext`, `ExternalTurn`),
``agent`` (`ExternalAgent`), ``tool`` (``as_tool``: the child run, and the sessions and
pauses kept in ``ctx.state``), ``changes`` (git change tracking) and ``scripted``
(`ScriptedExternalAgent`).
"""

from __future__ import annotations

import logging

from .. import prompts as prompts
from ..errors import (
    ExternalAgentConfigurationError,
    ExternalAgentConnectionError,
    ExternalAgentError,
    ExternalAgentProtocolError,
    ExternalAgentTimeoutError,
    ExternalAgentUnavailableError,
    ExternalToolError,
)
from ..events import ExternalAgentEvent
from .agent import ExternalAgent
from .contract import (
    CHECKPOINT_KIND,
    STOP_REASONS,
    ExternalCapabilities,
    ExternalOutcome,
    ExternalResult,
    ExternalStatus,
    ExternalToolCall,
    is_permanent_failure,
)
from .contract import LIMIT_REASONS as LIMIT_REASONS
from .contract import WRITE_TOOLS as WRITE_TOOLS
from .permissions import (
    APPROVAL_KEY,
    ExternalApproval,
    ExternalPermission,
    ExternalTool,
    PermissionAnswer,
    PermissionBridge,
    PermissionPosture,
    permission_fingerprint,
)
from .scripted import (
    EndStep,
    EventStep,
    PermissionStep,
    ReasoningStep,
    ScriptedCall,
    ScriptedExternalAgent,
    ToolStep,
    WaitStep,
)
from .tool import STATE_GRANTS, STATE_SESSIONS
from .tool import _tool_safe as _tool_safe  # qagents.integrations.devvi uses it
from .turn import ExternalTurn, ExternalTurnContext

logger = logging.getLogger("qagents.external")

__all__ = [
    "APPROVAL_KEY",
    "CHECKPOINT_KIND",
    "EndStep",
    "EventStep",
    "ExternalAgent",
    "ExternalAgentConfigurationError",
    "ExternalAgentConnectionError",
    "ExternalAgentError",
    "ExternalAgentEvent",
    "ExternalAgentProtocolError",
    "ExternalAgentTimeoutError",
    "ExternalAgentUnavailableError",
    "ExternalApproval",
    "ExternalCapabilities",
    "ExternalOutcome",
    "ExternalPermission",
    "ExternalResult",
    "ExternalStatus",
    "ExternalTool",
    "ExternalToolCall",
    "ExternalToolError",
    "ExternalTurn",
    "ExternalTurnContext",
    "PermissionAnswer",
    "PermissionBridge",
    "PermissionPosture",
    "PermissionStep",
    "ReasoningStep",
    "STATE_GRANTS",
    "STATE_SESSIONS",
    "STOP_REASONS",
    "ScriptedCall",
    "ScriptedExternalAgent",
    "ToolStep",
    "WaitStep",
    "is_permanent_failure",
    "permission_fingerprint",
]
