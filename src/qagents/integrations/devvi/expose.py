"""qagents tools inside a Devvi session: ``DevviAgent(expose_tools=[...])``.

Devvi has no MCP server, but it is an MCP *client*: it mounts the MCP servers a session
names and shows their tools to its model as ``mcp__<server>__<tool>``. qagents uses that
to hand Devvi its own tools. Two kinds of entries go into ``expose_tools``:

* **In-process tools** (`Tool` objects, plain functions, toolkits, agents: what
  ``Agent(tools=...)`` takes). For every turn qagents starts an MCP Streamable HTTP server
  in its own process (`qagents.integrations.mcp_server.tool_server`) on ``127.0.0.1``,
  with a random port and a random bearer token, and passes it in the session's
  ``mcpServers``. Through ``as_tool()`` the server is bound to the calling run: every call
  Devvi makes is decided by that run's `Permissions` (with its ``on_ask``) and journaled as
  one of its tool calls, with its ``deps`` and ``ctx.state``. Without a calling run (a
  direct ``coder.run(...)``) the turn's own rules decide and nothing is journaled. The
  server is named ``expose_name`` (default ``qagents``), so Devvi sees
  ``mcp__qagents__<tool>``. ACP only: the SDK transport mounts MCP servers through a patch
  file, and the bearer token must not be written to disk.
* **Importable tools**, as ``"package.module:attr"`` strings, and whole
  `ToolServerSpec` objects. Devvi starts these itself, over stdio, as
  ``python -I -m qagents.integrations.mcp_server --spec FILE`` (the spec file is written
  under ``<home>/qagents/tool-servers``). There is no calling run in that process: the
  spec's rules decide (``"ask"`` is refused) and nothing is journaled. The strings share
  one spec named ``<expose_name>_stdio`` with the default rules (tools declared
  ``requires_approval`` are refused); pass a `ToolServerSpec` for other rules, import
  ``paths`` or ``deps``. Both transports: ACP passes the server in ``mcpServers``, the SDK
  transport adds an ``@deepseek-ai/dsh-mcp-client`` row to the patch.

Devvi gives an MCP tool call 60 seconds. An approval asked through ``on_ask`` for an exposed
tool is therefore bounded by `APPROVAL_TIMEOUT_S`, and the calling run's
``Limits.tool_timeout_s`` should stay under a minute for exposed tools.

Everything here needs the MCP SDK (``pip install "qagents[mcp]"``); it is imported only
when ``expose_tools`` is not empty.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...errors import ConfigurationError

if TYPE_CHECKING:
    from ..mcp_server import ToolServer, ToolServerSpec

APPROVAL_TIMEOUT_S = 50.0
"""Seconds an ``on_ask`` gets to answer for an exposed tool: Devvi waits 60 s for an MCP call."""

SPEC_FOLDER = ("qagents", "tool-servers")
"""Where spec files go, under the Devvi home."""

_SERVER_NAME = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
"""Devvi's MCP server names (``patches.SERVER_NAME``): they become part of tool names."""


def _mcp_server() -> Any:
    try:
        from .. import mcp_server
    except ImportError as error:
        raise ConfigurationError(
            "expose_tools needs the MCP Python SDK: pip install 'qagents[mcp]'. "
            f"Import failed with: {error}"
        ) from error
    return mcp_server


@dataclass(frozen=True)
class ExposedTools:
    """What ``expose_tools`` asked for, split by how it is served."""

    name: str
    """The name of the in-process (HTTP) server."""
    in_process: tuple[Any, ...]
    """Tools served in process over Streamable HTTP, one server per turn."""
    specs: tuple[Any, ...]
    """`ToolServerSpec` objects that Devvi starts over stdio."""

    @property
    def server_names(self) -> list[str]:
        names = [self.name] if self.in_process else []
        return [*names, *(spec.name for spec in self.specs)]

    @classmethod
    def build(cls, items: Sequence[Any], *, name: str = "qagents") -> ExposedTools | None:
        """Sort ``expose_tools``; ``None`` when it is empty.

        Raises:
            ConfigurationError: A malformed reference, a bad or repeated server name, or the
                MCP SDK is missing.
        """
        if isinstance(items, (str, bytes, Mapping)) or not isinstance(items, Sequence):
            items = [items]
        if not items:
            return None
        module = _mcp_server()
        spec_type = module.ToolServerSpec
        references: list[str] = []
        specs: list[Any] = []
        in_process: list[Any] = []
        for item in items:
            if isinstance(item, str):
                references.append(item)
            elif isinstance(item, spec_type):
                specs.append(item)
            elif item is None:
                raise ConfigurationError("expose_tools holds None; pass tools or references.")
            else:
                in_process.append(item)
        _check_name(name)
        if references:
            try:
                specs.insert(0, spec_type(name=f"{name}_stdio", tools=references))
            except ValueError as error:  # pydantic: a reference that is not "module:attr"
                raise ConfigurationError(
                    f"expose_tools: {_first_error(error)}. Name importable tools as "
                    "'package.module:attr'."
                ) from error
        exposed = cls(name=name, in_process=tuple(in_process), specs=tuple(specs))
        seen: set[str] = set()
        for server in exposed.server_names:
            _check_name(server)
            if server in seen:
                raise ConfigurationError(
                    f"expose_tools: two MCP servers are named {server!r}; Devvi needs unique "
                    "names (they become part of the tool names, mcp__<server>__<tool>)."
                )
            seen.add(server)
        return exposed

    # -- the stdio servers --------------------------------------------------------------

    def write_specs(self, home: str | Path) -> list[Path]:
        """Write the spec files under ``<home>/qagents/tool-servers``; their paths.

        Files are named after their content, so a spec that a running Devvi reads is never
        rewritten. They hold no secrets: references, rules and paths only.
        """
        folder = Path(home).joinpath(*SPEC_FOLDER)
        return [_write_spec(folder, spec) for spec in self.specs]

    def acp_entries(self, home: str | Path) -> list[dict[str, Any]]:
        """The stdio servers as ACP ``mcpServers`` entries (``{name, command, args, env}``)."""
        module = _mcp_server()
        return [
            module.acp_stdio_config(path, name=spec.name)
            for spec, path in zip(self.specs, self.write_specs(home), strict=True)
        ]

    def patch_rows(self, home: str | Path) -> list[dict[str, Any]]:
        """The stdio servers as Devvi patch rows, for the SDK transport."""
        from .patches import mcp_stdio_row

        module = _mcp_server()
        rows: list[dict[str, Any]] = []
        for spec, path in zip(self.specs, self.write_specs(home), strict=True):
            command = module.stdio_command(path)
            rows.append(mcp_stdio_row(spec.name, command[0], command[1:]))
        return rows

    # -- the in-process server ------------------------------------------------------------

    def http_server(self, turn: Any) -> ToolServer:
        """The in-process server of one turn, not started yet.

        Bound to ``turn.parent`` (the calling tool call) when there is one: the calling
        run's rules decide and its journal records the calls. Otherwise the turn's own
        rules decide and nothing is journaled.
        """
        module = _mcp_server()
        if turn.parent is not None:
            return module.tool_server(
                self.in_process,
                ctx=turn.parent,
                name=self.name,
                approval_timeout_s=APPROVAL_TIMEOUT_S,
            )
        return module.tool_server(
            self.in_process,
            permissions=turn.bridge.permissions,
            name=self.name,
            approval_timeout_s=APPROVAL_TIMEOUT_S,
        )


def _check_name(name: str) -> None:
    if not isinstance(name, str) or not _SERVER_NAME.match(name):
        raise ConfigurationError(
            f"expose_tools: the MCP server name {name!r} must be 1 to 32 letters, digits, "
            "'_' or '-' (Devvi puts it in tool names)."
        )


def _write_spec(folder: Path, spec: ToolServerSpec) -> Path:
    text = spec.model_dump_json(indent=2) + "\n"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    path = folder / f"{spec.name}-{digest}.json"
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return path
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / f".{path.name}.{os.getpid()}.tmp"
    temporary.write_text(text, encoding="utf-8", newline="\n")
    try:
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        if not (path.is_file() and path.read_text(encoding="utf-8") == text):
            raise
    return path


def _first_error(error: ValueError) -> str:
    errors = getattr(error, "errors", None)
    if callable(errors):
        try:
            found = errors()
        except Exception:  # pragma: no cover - pydantic always answers
            found = []
        if found:
            return str(found[0].get("msg", error)).removeprefix("Value error, ")
    return str(error).splitlines()[0]


__all__ = ["APPROVAL_TIMEOUT_S", "ExposedTools"]
