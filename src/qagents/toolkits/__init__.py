"""Ready-made tools. Each factory returns ordinary `Tool` objects.

```python
from qagents.toolkits import ask_user, file_tools, notes_tools, todo_tools, web_tools

agent = Agent(
    "qcloud:<model-id>",
    tools=[
        file_tools("./workspace", writable=True),
        web_tools(),
        notes_tools("./memory"),
        todo_tools(),
        ask_user,
    ],
)
```

``MCPToolkit`` (tools of an MCP server, ``pip install "qagents[mcp]"``) is
resolved on first access, so importing this package does not import the MCP
SDK. Without the SDK, ``qagents.toolkits.MCPToolkit`` raises an
``AttributeError`` that names the extra to install, and ``dir()`` leaves it out.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from datetime import UTC, datetime, tzinfo
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..errors import ToolError
from ..tools import tool
from .files import file_tools
from .human import ask_user
from .notes import notes_tools
from .planning import TodoItem, render_todos, todo_tools
from .shell import shell_tool
from .web import web_tools

if TYPE_CHECKING:
    # A re-export for type checkers; at run time __getattr__ resolves it.
    from .mcp import MCPToolkit as MCPToolkit

_LAZY = {"MCPToolkit": "qagents.toolkits.mcp"}
"""Names resolved on first access, so importing this package needs no optional SDK."""


@tool(read_only=True)
def current_time(timezone: str = "UTC") -> str:
    """Tell the current date and time.

    Args:
        timezone: IANA time zone name, for example "Europe/Athens".
    """
    if timezone.upper() == "UTC":
        zone: tzinfo = UTC
    else:
        try:
            zone = ZoneInfo(timezone)
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise ToolError(
                f"Unknown time zone {timezone!r}. "
                "Use an IANA name such as 'Europe/Athens', or 'UTC'."
            ) from error
    return datetime.now(zone).strftime("%A %Y-%m-%d %H:%M:%S %Z")


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
    sdk = all(importlib.util.find_spec(package) is not None for package in ("mcp", "mcp_types"))
    return sorted({*globals(), *(_LAZY if sdk else ())})


__all__ = [
    "TodoItem",
    "ask_user",
    "current_time",
    "file_tools",
    "notes_tools",
    "render_todos",
    "shell_tool",
    "todo_tools",
    "web_tools",
]
