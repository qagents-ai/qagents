"""Ready-made tools. Each factory returns ordinary `Tool` objects.

```python
from qagents.toolkits import file_tools, notes_tools, shell_tool, web_tools

agent = Agent(
    "qcloud:<model-id>",
    tools=[file_tools("./workspace", writable=True), web_tools(), notes_tools("./memory")],
)
```
"""

from __future__ import annotations

from datetime import UTC, datetime, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..errors import ToolError
from ..tools import tool
from .files import file_tools
from .notes import notes_tools
from .shell import shell_tool
from .web import web_tools


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


__all__ = ["current_time", "file_tools", "notes_tools", "shell_tool", "web_tools"]
