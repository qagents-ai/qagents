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
``from qagents import *`` does not need the extras either. The submodules
``qagents.evals``, ``qagents.tracing``, ``qagents.integrations`` and
``qagents.cli`` are imported only when you import them.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from . import prompts
from .agent import Agent, Limits, RunStream
from .context import Compactor, CompactorChain, SummarizingCompactor, ToolResultClearing
from .errors import (
    ApprovalTokenError,
    ConfigurationError,
    DeferToolCall,
    GuardrailTripped,
    LeaseError,
    ModelError,
    ModelRetry,
    OutputValidationError,
    QAgentsError,
    RunInterrupted,
    SessionError,
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
from .tools import RunContext, Tool, Toolkit, tool
from .workflow import Workflow, WorkflowStatus

if TYPE_CHECKING:
    # Re-exports for type checkers; at run time __getattr__ resolves them.
    from .integrations.mlflow import MLflowHooks as MLflowHooks
    from .models.anthropic import AnthropicModel as AnthropicModel
    from .stores.postgres import PostgresStore as PostgresStore
    from .toolkits.mcp import MCPToolkit as MCPToolkit

__version__ = "0.2.0"

_LAZY = {
    "AnthropicModel": "qagents.models.anthropic",
    "MCPToolkit": "qagents.toolkits.mcp",
    "MLflowHooks": "qagents.integrations.mlflow",
    "PostgresStore": "qagents.stores.postgres",
}
"""Names from optional extras, resolved on first access."""


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})


__all__ = [
    "Agent",
    "AgentRuntime",
    "Approval",
    "ApprovalTokenError",
    "Compactor",
    "CompactorChain",
    "ConfigurationError",
    "Decision",
    "DeferToolCall",
    "FallbackModel",
    "GuardrailTripped",
    "Hooks",
    "ImagePart",
    "InMemoryStore",
    "InputResult",
    "LeaseError",
    "Limits",
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
    "SummarizingCompactor",
    "TextPart",
    "Tool",
    "ToolCallPart",
    "ToolError",
    "ToolResultClearing",
    "ToolResultPart",
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
