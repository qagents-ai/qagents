"""qagents: a small, explicit agent framework.

```python
from qagents import Agent, tool

@tool(read_only=True)
def get_weather(city: str) -> str:
    '''Look up the current weather for a city.'''
    return f"Sunny in {city}"

agent = Agent("qcloud:<model-id>", instructions="Be brief.", tools=[get_weather])
result = agent.run_sync("What is the weather in Athens?")
print(result.output)
```

Four names come from optional extras and are resolved on first access, so
``import qagents`` never imports the anthropic, mcp, mlflow or psycopg
packages: ``AnthropicModel`` (``qagents[anthropic]``), ``MCPToolkit``
(``qagents[mcp]``), ``MLflowHooks`` (``qagents[mlflow]``) and
``PostgresStore`` (``qagents[postgres]``). They are not in ``__all__``, so
``from qagents import *`` does not need the extras either. ``MCPToolkit``
needs its SDK to be imported at all: without it, ``qagents.MCPToolkit`` raises
an ``AttributeError`` that says which extra to install (so ``hasattr``,
``help(qagents)`` and ``inspect`` keep working), and ``dir(qagents)`` leaves
it out. The submodules
``qagents.evals``, ``qagents.tracing``, ``qagents.integrations`` and
``qagents.cli`` are imported only when you import them.

The external-agent names (``ExternalAgent``, ``ExternalApproval``,
``ExternalCapabilities``, ``ExternalResult``, ``ScriptedExternalAgent``) need no
extra; they are resolved on first access too, so ``import qagents`` does not
import ``qagents.external``.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from typing import TYPE_CHECKING, Any

from . import prompts
from .agent import Agent, Limits, RunStream
from .context import Compactor, CompactorChain, SummarizingCompactor, ToolResultClearing
from .errors import (
    ApprovalConflictError,
    ApprovalTokenError,
    ConfigurationError,
    DeferToolCall,
    GuardrailTripped,
    HeldEffectsError,
    LeaseError,
    MaxDepthExceeded,
    ModelError,
    ModelRetry,
    OutputValidationError,
    QAgentsError,
    RunInterrupted,
    SessionError,
    SettlementConflict,
    ToolError,
    WorkflowSuspended,
)
from .hooks import Hooks
from .messages import (
    ImagePart,
    Message,
    ReasoningPart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    Usage,
)
from .models import (
    FallbackModel,
    Model,
    ModelSettings,
    OpenAIChatModel,
    QCloudModel,
    RateLimitedModel,
    cassette,
    register_provider,
)
from .multi import RouteResult, fan_out, route
from .permissions import Approval, Decision, InputResult, Permissions
from .results import RunResult, TrippedGuardrail
from .runtime import AgentRuntime, RunHandle
from .session import (
    InMemoryStore,
    PendingApproval,
    PendingInput,
    Session,
    SessionStatus,
    SessionStore,
    SQLiteStore,
)
from .team import TeamEndpoint, TeamMailbox, TeamMessage
from .tools import RunContext, Tool, Toolkit, ToolTimeout, tool
from .workflow import Workflow, WorkflowStatus

if TYPE_CHECKING:
    # Re-exports for type checkers; at run time __getattr__ resolves them.
    from .eval_statistics import CaseReliability as CaseReliability
    from .eval_statistics import PairedComparison as PairedComparison
    from .eval_statistics import ReliabilityReport as ReliabilityReport
    from .eval_statistics import SuccessEstimate as SuccessEstimate
    from .eval_statistics import compare_reliability as compare_reliability
    from .eval_statistics import reliability as reliability
    from .evals import TrialTarget as TrialTarget
    from .external import ExternalAgent as ExternalAgent
    from .external import ExternalApproval as ExternalApproval
    from .external import ExternalCapabilities as ExternalCapabilities
    from .external import ExternalResult as ExternalResult
    from .external import ScriptedExternalAgent as ScriptedExternalAgent
    from .integrations.mlflow import MLflowHooks as MLflowHooks
    from .models.anthropic import AnthropicModel as AnthropicModel
    from .stores.postgres import PostgresStore as PostgresStore
    from .toolkits.mcp import MCPToolkit as MCPToolkit

__version__ = "0.2.0"

_LAZY = {
    "TrialTarget": "qagents.evals",
    "SuccessEstimate": "qagents.eval_statistics",
    "CaseReliability": "qagents.eval_statistics",
    "ReliabilityReport": "qagents.eval_statistics",
    "PairedComparison": "qagents.eval_statistics",
    "reliability": "qagents.eval_statistics",
    "compare_reliability": "qagents.eval_statistics",
    "AnthropicModel": "qagents.models.anthropic",
    "ExternalAgent": "qagents.external",
    "ExternalApproval": "qagents.external",
    "ExternalCapabilities": "qagents.external",
    "ExternalResult": "qagents.external",
    "ScriptedExternalAgent": "qagents.external",
    "MCPToolkit": "qagents.toolkits.mcp",
    "MLflowHooks": "qagents.integrations.mlflow",
    "PostgresStore": "qagents.stores.postgres",
}
"""Optional extras, external agents and evaluation utilities, resolved on first access."""

_NEEDS = {"MCPToolkit": ("mcp", "mcp_types")}
"""Lazy names whose module cannot be imported without these packages."""


def _installed(name: str) -> bool:
    return all(importlib.util.find_spec(package) is not None for package in _NEEDS.get(name, ()))


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        imported = importlib.import_module(module)
    except ImportError as error:
        # AttributeError, so hasattr(), help() and inspect treat the name as absent.
        raise AttributeError(str(error), name=name, obj=sys.modules[__name__]) from error
    value = getattr(imported, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *(name for name in _LAZY if _installed(name))})


__all__ = [
    "TrialTarget",
    "SuccessEstimate",
    "CaseReliability",
    "ReliabilityReport",
    "PairedComparison",
    "reliability",
    "compare_reliability",
    "Agent",
    "AgentRuntime",
    "Approval",
    "ApprovalConflictError",
    "ApprovalTokenError",
    "Compactor",
    "CompactorChain",
    "ConfigurationError",
    "Decision",
    "DeferToolCall",
    "FallbackModel",
    "GuardrailTripped",
    "HeldEffectsError",
    "Hooks",
    "ImagePart",
    "InMemoryStore",
    "InputResult",
    "LeaseError",
    "Limits",
    "MaxDepthExceeded",
    "Message",
    "Model",
    "ModelError",
    "ModelRetry",
    "ModelSettings",
    "OpenAIChatModel",
    "OutputValidationError",
    "PendingApproval",
    "PendingInput",
    "Permissions",
    "QAgentsError",
    "QCloudModel",
    "RateLimitedModel",
    "ReasoningPart",
    "RouteResult",
    "RunContext",
    "RunHandle",
    "RunInterrupted",
    "RunResult",
    "RunStream",
    "SQLiteStore",
    "Session",
    "SessionError",
    "SessionStatus",
    "SessionStore",
    "SettlementConflict",
    "SummarizingCompactor",
    "TextPart",
    "Tool",
    "ToolCallPart",
    "ToolError",
    "ToolResultClearing",
    "ToolResultPart",
    "ToolTimeout",
    "TeamMailbox",
    "TeamEndpoint",
    "TeamMessage",
    "Toolkit",
    "TrippedGuardrail",
    "Usage",
    "Workflow",
    "WorkflowStatus",
    "WorkflowSuspended",
    "__version__",
    "cassette",
    "fan_out",
    "prompts",
    "register_provider",
    "route",
    "tool",
]
