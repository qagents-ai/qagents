"""Adapters between qagents and other protocols and tools.

| Module | For |
|---|---|
| `qagents.integrations.agui` | AG-UI event streams for CopilotKit and other AG-UI clients |
| `qagents.integrations.mlflow` | MLflow tracing of runs, model requests and tool calls |
| `qagents.integrations.acp` | Drive any ACP v1 coding agent as an `ExternalAgent`: `ACPAgent` |
| `qagents.integrations.devvi` | Devvi (devvi.sh) as a coding agent: `DevviAgent`, `TokenPricing` |
| `qagents.integrations.mcp_server` | qagents tools served over MCP to other agents: `tool_server` |

The names below are resolved on first access, so importing this package
imports none of these modules (MLflow only when ``MLflowHooks`` is created, the
MCP SDK only when ``tool_server``, ``ToolServer`` or ``ToolServerSpec`` is first
used): ``from qagents.integrations import MLflowHooks, to_agui_events``.

Without the MCP SDK (``qagents[mcp]``), ``qagents.integrations.tool_server`` (and
``ToolServer``, ``ToolServerSpec``) raises an ``AttributeError`` that says which
extra to install, so ``hasattr``, ``help()`` and ``inspect`` keep working, and
``dir()`` leaves those names out.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Re-exports for type checkers; at run time __getattr__ resolves them.
    from .acp import ACPAgent as ACPAgent
    from .acp import MCPServerHTTP as MCPServerHTTP
    from .acp import MCPServerStdio as MCPServerStdio
    from .agui import AGUIProjector as AGUIProjector
    from .agui import agui_resume as agui_resume
    from .agui import agui_sse as agui_sse
    from .agui import to_agui_events as to_agui_events
    from .devvi import DevviAgent as DevviAgent
    from .devvi import TokenPricing as TokenPricing
    from .mcp_server import ToolServer as ToolServer
    from .mcp_server import ToolServerSpec as ToolServerSpec
    from .mcp_server import tool_server as tool_server
    from .mlflow import MLflowHooks as MLflowHooks

_LAZY = {
    "AGUIProjector": "qagents.integrations.agui",
    "agui_resume": "qagents.integrations.agui",
    "agui_sse": "qagents.integrations.agui",
    "to_agui_events": "qagents.integrations.agui",
    "MLflowHooks": "qagents.integrations.mlflow",
    "ACPAgent": "qagents.integrations.acp",
    "MCPServerHTTP": "qagents.integrations.acp",
    "MCPServerStdio": "qagents.integrations.acp",
    "DevviAgent": "qagents.integrations.devvi",
    "TokenPricing": "qagents.integrations.devvi",
    "ToolServer": "qagents.integrations.mcp_server",
    "ToolServerSpec": "qagents.integrations.mcp_server",
    "tool_server": "qagents.integrations.mcp_server",
}
"""Names resolved on first access."""

_NEEDS = {
    "ToolServer": ("mcp", "mcp_types"),
    "ToolServerSpec": ("mcp", "mcp_types"),
    "tool_server": ("mcp", "mcp_types"),
}
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
