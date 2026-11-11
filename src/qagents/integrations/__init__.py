"""Adapters between qagents and other protocols and tools.

| Module | For |
|---|---|
| `qagents.integrations.agui` | AG-UI event streams for CopilotKit and other AG-UI clients |
| `qagents.integrations.mlflow` | MLflow tracing of runs, model requests and tool calls |

The names below are resolved on first access, so importing this package
imports neither module (and MLflow only when ``MLflowHooks`` is created):
``from qagents.integrations import MLflowHooks, to_agui_events``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Re-exports for type checkers; at run time __getattr__ resolves them.
    from .agui import AGUIProjector as AGUIProjector
    from .agui import agui_resume as agui_resume
    from .agui import agui_sse as agui_sse
    from .agui import to_agui_events as to_agui_events
    from .mlflow import MLflowHooks as MLflowHooks

_LAZY = {
    "AGUIProjector": "qagents.integrations.agui",
    "agui_resume": "qagents.integrations.agui",
    "agui_sse": "qagents.integrations.agui",
    "to_agui_events": "qagents.integrations.agui",
    "MLflowHooks": "qagents.integrations.mlflow",
}
"""Names resolved on first access."""


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})
