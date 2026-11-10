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
"""

from __future__ import annotations

from . import prompts
from .agent import Agent, Limits, RunStream
from .context import Compactor, SummarizingCompactor
from .errors import (
    ConfigurationError,
    ModelError,
    OutputValidationError,
    QAgentsError,
    SessionError,
    ToolError,
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
    Model,
    ModelSettings,
    OpenAIChatModel,
    QCloudModel,
    register_provider,
)
from .permissions import Approval, Decision, Permissions
from .results import RunResult
from .session import InMemoryStore, PendingApproval, Session, SessionStore, SQLiteStore
from .tools import RunContext, Tool, Toolkit, tool
from .workflow import Workflow

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "Approval",
    "Compactor",
    "ConfigurationError",
    "Decision",
    "Hooks",
    "ImagePart",
    "InMemoryStore",
    "Limits",
    "Message",
    "Model",
    "ModelError",
    "ModelSettings",
    "OpenAIChatModel",
    "OutputValidationError",
    "PendingApproval",
    "Permissions",
    "QAgentsError",
    "QCloudModel",
    "ReasoningPart",
    "RunContext",
    "RunResult",
    "RunStream",
    "SQLiteStore",
    "Session",
    "SessionError",
    "SessionStore",
    "SummarizingCompactor",
    "TextPart",
    "Tool",
    "ToolCallPart",
    "ToolError",
    "ToolResultPart",
    "Toolkit",
    "Usage",
    "Workflow",
    "__version__",
    "prompts",
    "register_provider",
    "tool",
]
