"""Tools from an MCP server.

`MCPToolkit` connects to a server that speaks the Model Context Protocol and
exposes each of its tools as a qagents `Tool`:

```python
from qagents import Agent
from qagents.toolkits.mcp import MCPToolkit

github = MCPToolkit.stdio("npx", ["-y", "@modelcontextprotocol/server-github"], prefix="gh")
async with github:
    agent = Agent("qcloud:<model-id>", tools=[github])
    result = await agent.run("List my open pull requests.")
```

It uses the official MCP Python SDK (``mcp`` 2.x), which negotiates protocol
revision 2026-07-28 and falls back to the 2025-11-25 handshake for older
servers.

What the toolkit does for safety:

* **Approval for side effects.** A tool the server does not declare read-only
  asks for approval unless the server declares it non-destructive
  (``approval="destructive"``, the default). The MCP defaults treat a tool
  without annotations as possibly destructive, and so does this toolkit.
* **Pinned definitions.** Every tool has a `definition_hash`. Pass ``pins`` to
  accept only the definitions you reviewed; a tool whose definition changed is
  not exposed. Within one toolkit the first definition seen is pinned too, so a
  server cannot change a tool's description or schema after the agent started
  using it (a "rug pull").
* **No binary for the model.** Images, audio and binary resources in a result
  become short text placeholders that name the MIME type.
* **Failures are results.** A tool error, a timeout or a lost connection becomes
  an error result the model can read; it never ends the run.

Annotations (``readOnlyHint``, ``destructiveHint``, ``idempotentHint``) are
claims made by the server. Trust them only as far as you trust the server; use
``approval="all"`` or permission rules for servers you do not control.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import fnmatch
import hashlib
import json
import logging
import re
import sys
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TextIO
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, ValidationError

try:
    import mcp_types
    from mcp import Client, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client
    from mcp.server import Server
    from mcp.server.mcpserver import MCPServer
    from mcp.shared.exceptions import MCPError
    from mcp_types.version import MODERN_PROTOCOL_VERSIONS
except ImportError as error:  # pragma: no cover - depends on the environment
    raise ImportError(
        "qagents.toolkits.mcp needs the MCP Python SDK 2.x: pip install 'qagents[mcp]' "
        f"(or pip install 'mcp>=2.3,<3'). Import failed with: {error}"
    ) from error

from .. import prompts
from ..errors import (
    ConfigurationError,
    MCPConnectionError,
    MCPToolChangedError,
    ToolArgumentError,
    ToolError,
)
from ..tools import RunContext, Tool, Toolkit

# The text this module adds to tool results lives in qagents.prompts (MCP_*).

logger = logging.getLogger("qagents.mcp")

ApprovalPolicy = Literal["destructive", "all", "none"]
OnChange = Literal["error", "warn"]
ServerTarget = Any
"""What `MCPToolkit` connects to: a zero-argument function that returns an MCP transport
(an async context manager yielding the SDK's read and write streams), or an in-process
``mcp.server.mcpserver.MCPServer`` / ``mcp.server.Server``."""

_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_INVALID_NAME_CHARACTERS = re.compile(r"[^A-Za-z0-9_-]")
_MAX_NAME = 64
_MAX_LIST_PAGES = 100
_JSON_TYPES: dict[str, Callable[[Any], bool]] = {
    "string": lambda value: isinstance(value, str),
    "integer": lambda value: (
        (isinstance(value, int) and not isinstance(value, bool))
        or (isinstance(value, float) and value.is_integer())
    ),
    "number": lambda value: isinstance(value, int | float) and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
    "array": lambda value: isinstance(value, list | tuple),
    "object": lambda value: isinstance(value, dict),
    "null": lambda value: value is None,
}


class MCPProgress(BaseModel):
    """A progress report from an MCP server for a running tool call.

    Put on the run's event stream with `RunContext.emit`, so it reaches
    ``agent.stream(...)`` consumers and ``Hooks.on_event``.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["mcp_progress"] = "mcp_progress"
    server: str
    """The toolkit's name for the server."""
    tool: str
    """The tool's name as the model sees it."""
    tool_call_id: str | None = None
    progress: float
    total: float | None = None
    message: str | None = None


def definition_hash(definition: mcp_types.Tool) -> str:
    """Fingerprint of what a tool tells the model and how it is treated.

    The SHA-256 of the tool's name, description, input schema and annotations, in
    canonical JSON. Annotations are included because they decide whether a call
    needs approval: a server that starts calling a destructive tool read-only has
    changed the tool.

    Args:
        definition: A tool as listed by the server.

    Returns:
        A lowercase hex digest.
    """
    annotations = (
        definition.annotations.model_dump(mode="json", by_alias=True, exclude_none=True)
        if definition.annotations is not None
        else None
    )
    payload = {
        "name": definition.name,
        "description": definition.description,
        "inputSchema": definition.input_schema,
        "annotations": annotations or None,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class _OpaqueArguments(BaseModel):
    """Placeholder for `Tool.arguments_model`; `MCPTool` validates against the server schema."""

    model_config = ConfigDict(extra="allow")


class MCPTool(Tool):
    """A tool that lives on an MCP server. `MCPToolkit.tools` builds them."""

    def __init__(
        self,
        toolkit: MCPToolkit,
        definition: mcp_types.Tool,
        *,
        name: str,
        requires_approval: bool,
        read_only: bool,
        idempotent: bool,
        timeout: float | None,
    ) -> None:
        super().__init__(
            self._unused,
            name=name,
            description=definition.description or definition.title or "",
            arguments_model=_OpaqueArguments,
            takes_context=True,
            requires_approval=requires_approval,
            read_only=read_only,
            idempotent=idempotent,
            timeout=timeout,
        )
        self.toolkit = toolkit
        """The toolkit that calls the server."""
        self.mcp_name = definition.name
        """The tool's name on the server (``name`` is what the model sees)."""
        self.definition = definition
        """The tool exactly as the server listed it."""
        self.definition_hash = definition_hash(definition)
        """See `definition_hash`; compare it with the ``pins`` you keep."""
        self.parameters = _input_schema(definition.input_schema)

    async def _unused(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError("MCPTool.run calls the server directly.")

    def validate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Check ``arguments`` against the top level of the server's input schema.

        Checks that the arguments are an object, that required keys are present,
        that no unknown key is sent when ``additionalProperties`` is false, and the
        JSON type and ``enum`` of each top-level value. Everything else (formats,
        ranges, nested objects) is left to the server.

        Raises:
            ToolArgumentError: with one line per problem, for the model to read.
        """
        if not isinstance(arguments, dict):
            raise ToolArgumentError(prompts.MCP_ARGUMENTS_NOT_OBJECT)
        schema = self.parameters
        properties: dict[str, Any] = schema.get("properties") or {}
        problems: list[str] = []
        for key in schema.get("required") or ():
            if key not in arguments:
                problems.append(prompts.MCP_ARGUMENT_MISSING.format(name=key))
        if schema.get("additionalProperties") is False:
            for key in arguments:
                if key not in properties:
                    problems.append(prompts.MCP_ARGUMENT_UNKNOWN.format(name=key))
        for key, value in arguments.items():
            rule = properties.get(key)
            if not isinstance(rule, dict):
                continue
            expected = rule.get("type")
            names = [expected] if isinstance(expected, str) else list(expected or [])
            checks = [_JSON_TYPES[n] for n in names if n in _JSON_TYPES]
            if checks and not any(check(value) for check in checks):
                problems.append(
                    prompts.MCP_ARGUMENT_TYPE.format(name=key, expected=" or ".join(names))
                )
                continue
            allowed = rule.get("enum")
            if isinstance(allowed, list) and value not in allowed:
                shown = ", ".join(json.dumps(item, ensure_ascii=False) for item in allowed)
                problems.append(prompts.MCP_ARGUMENT_ENUM.format(name=key, allowed=shown))
        if problems:
            raise ToolArgumentError("\n".join(problems))
        return dict(arguments)

    async def run(
        self, ctx: RunContext[Any], arguments: dict[str, Any], *, timeout: float | None = None
    ) -> Any:
        """Validate ``arguments`` and call the tool on the server.

        Args:
            ctx: The run context; progress reports are emitted on it.
            arguments: The arguments the model sent.
            timeout: The run's per-call limit, used when the tool has no ``timeout``.

        Returns:
            The text the model reads, or the structured content when the result
            has no text (or when the toolkit prefers structured content).

        Raises:
            ToolArgumentError: The arguments do not match the input schema.
            ToolError: The server reported an error, the call timed out, or the
                connection failed. The message is written for the model.
        """
        validated = self.validate(arguments)
        limit = self.timeout if self.timeout is not None else timeout
        return await self.toolkit._call(self, ctx, validated, limit)

    def __repr__(self) -> str:
        return f"MCPTool({self.name!r}, server={self.toolkit.name!r})"


@dataclass
class _Connection:
    loop: asyncio.AbstractEventLoop
    task: asyncio.Task[None]
    stop: asyncio.Event
    client: Client | None = None
    alive: bool = True
    verified: bool = False
    """The tools were listed (and checked against the pins) on this connection."""


class MCPToolkit(Toolkit):
    """The tools of one MCP server, connected lazily and listed once.

    Build one with `MCPToolkit.stdio` or `MCPToolkit.http`, or pass a transport
    factory or an in-process server to the constructor. Use it as ``async with``
    or call `aclose` when done; the server process (or HTTP session) stays open
    between runs until then.

    The connection belongs to one event loop. If the toolkit is used from a new
    loop (``agent.run_sync`` twice, for example), it reconnects there.
    """

    def __init__(
        self,
        server: ServerTarget,
        *,
        name: str = "mcp",
        prefix: str | None = None,
        separator: Literal["__", "_"] = "__",
        include: Iterable[str] | None = None,
        exclude: Iterable[str] = (),
        approval: ApprovalPolicy = "destructive",
        pins: Mapping[str, str] | None = None,
        on_change: OnChange = "error",
        pin_on_first_use: bool = True,
        timeout: float | None = None,
        connect_timeout: float = 30.0,
        prefer_structured: bool = False,
        include_resource_text: bool = True,
        protocol: str = "auto",
    ) -> None:
        """Configure the toolkit; nothing connects until `tools` is awaited.

        Args:
            server: A zero-argument function returning an MCP transport (an async
                context manager that yields the SDK's read and write streams), or
                an in-process ``MCPServer`` or low-level ``Server``.
            name: How logs and error results refer to the server.
            prefix: Put in front of every tool name, joined with ``separator``, to
                keep tools of several servers apart. Default: no prefix.
            separator: ``"__"`` or ``"_"``.
            include: Glob patterns of server tool names to expose. Default: all.
            exclude: Glob patterns of server tool names to hide.
            approval: ``"destructive"`` asks before tools the server does not
                declare read-only or non-destructive; ``"all"`` asks before every
                tool; ``"none"`` never asks (permission rules still apply).
            pins: Server tool name -> expected `definition_hash`. A listed tool
                whose hash differs is not exposed.
            on_change: When a pinned tool changed: ``"error"`` makes `tools`
                raise `MCPToolChangedError`; ``"warn"`` logs a warning and leaves
                the tool out.
            pin_on_first_use: Also pin each tool to the first definition this
                toolkit saw, so it cannot change while in use.
            timeout: Seconds per tool call; ``None`` uses the run's
                ``Limits.tool_timeout_s``.
            connect_timeout: Seconds to start the server and finish the handshake.
            prefer_structured: Give the model a result's structured content (as
                JSON) instead of its text content when it has both.
            include_resource_text: Show the text of embedded text resources; when
                false they become placeholders like binary ones.
            protocol: ``"auto"`` (2026-07-28, falling back to the 2025-11-25
                handshake), ``"legacy"`` (handshake only) or a modern revision
                string to use without probing.

        Raises:
            ConfigurationError: An option is invalid.
        """
        if prefix is not None and not _NAME_PATTERN.match(prefix):
            raise ConfigurationError(
                f"MCP prefix {prefix!r} is not valid: use 1-64 letters, digits, '_' or '-'."
            )
        if separator not in ("__", "_"):
            raise ConfigurationError("MCP separator must be '__' or '_'.")
        if approval not in ("destructive", "all", "none"):
            raise ConfigurationError(
                f"approval={approval!r} is not valid: use 'destructive', 'all' or 'none'."
            )
        if on_change not in ("error", "warn"):
            raise ConfigurationError(
                f"on_change={on_change!r} is not valid: use 'error' or 'warn'."
            )
        if timeout is not None and timeout <= 0:
            raise ConfigurationError("MCP timeout must be a positive number of seconds or None.")
        if connect_timeout <= 0:
            raise ConfigurationError("MCP connect_timeout must be a positive number of seconds.")
        if protocol not in ("auto", "legacy", *MODERN_PROTOCOL_VERSIONS):
            raise ConfigurationError(
                f"protocol={protocol!r} is not valid: use 'auto', 'legacy' or one of "
                f"{list(MODERN_PROTOCOL_VERSIONS)}. For a 2025 revision, use 'legacy'."
            )
        if pins is not None:
            for key, value in pins.items():
                if not re.fullmatch(r"[0-9a-f]{64}", str(value)):
                    raise ConfigurationError(
                        f"The pin for MCP tool {key!r} is not a SHA-256 hex digest. "
                        "Take pins from MCPToolkit.definition_hashes()."
                    )
        self._server = server
        self.name = name
        """How logs and error results refer to the server."""
        self.prefix = prefix
        self.separator = separator
        self.include = tuple(include) if include is not None else None
        self.exclude = tuple(exclude)
        self.approval: ApprovalPolicy = approval
        self.pins: dict[str, str] | None = dict(pins) if pins is not None else None
        self.on_change: OnChange = on_change
        self.pin_on_first_use = pin_on_first_use
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.prefer_structured = prefer_structured
        self.include_resource_text = include_resource_text
        self.protocol = protocol
        self._connection: _Connection | None = None
        self._locks: dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}
        self._tools: list[MCPTool] | None = None
        self._first_seen: dict[str, str] = {}
        self._changed: dict[str, str] = {}
        self._change_error: MCPToolChangedError | None = None

    # -- constructors -----------------------------------------------------------------

    @classmethod
    def stdio(
        cls,
        command: str,
        args: Sequence[str] = (),
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | Path | None = None,
        name: str | None = None,
        errlog: TextIO | None = None,
        **options: Any,
    ) -> MCPToolkit:
        """A server started as a subprocess that speaks MCP on stdin and stdout.

        The process inherits only a minimal environment (``PATH``, the home and
        temp directories and similar), not the agent's secrets; pass what it
        needs in ``env``. It is stopped, with the processes it started, by
        `aclose`.

        Args:
            command: The executable, for example ``"npx"`` or ``"uvx"``.
            args: Its arguments.
            env: Extra environment variables for the process.
            cwd: Working directory of the process.
            name: How logs and errors refer to the server. Default: the command's name.
            errlog: Where the server's stderr goes. Default: ``sys.stderr``.
            **options: Any keyword argument of `MCPToolkit`.
        """
        if not command:
            raise ConfigurationError("MCPToolkit.stdio needs a command to run.")
        parameters = StdioServerParameters(
            command=command,
            args=list(args),
            env=dict(env) if env is not None else None,
            cwd=cwd,
        )

        def connect() -> AbstractAsyncContextManager[Any]:
            return stdio_client(parameters, errlog=errlog if errlog is not None else sys.stderr)

        return cls(connect, name=name or Path(command).stem or "mcp", **options)

    @classmethod
    def http(
        cls,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        name: str | None = None,
        auth: Any = None,
        http_timeout: float = 30.0,
        sse_read_timeout: float = 300.0,
        **options: Any,
    ) -> MCPToolkit:
        """A server reached over MCP's streamable HTTP transport.

        Args:
            url: The MCP endpoint, for example ``"https://example.com/mcp"``.
            headers: Sent with every request, for example an ``Authorization``
                header. They are never logged.
            name: How logs and errors refer to the server. Default: the host name.
            auth: An ``httpx2.Auth``, such as the SDK's
                ``mcp.client.auth.OAuthClientProvider``.
            http_timeout: Seconds to connect, write and wait for a pool slot.
            sse_read_timeout: Seconds to wait for data on a response stream.
            **options: Any keyword argument of `MCPToolkit`.
        """
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ConfigurationError(
                f"MCPToolkit.http needs an http:// or https:// URL, got {url[:80]!r}."
            )
        fixed_headers = dict(headers) if headers is not None else None

        @asynccontextmanager
        async def connect() -> AsyncIterator[Any]:
            import httpx2  # installed with mcp

            client = httpx2.AsyncClient(
                headers=fixed_headers,
                auth=auth,
                timeout=httpx2.Timeout(http_timeout, read=sse_read_timeout),
            )
            async with client, streamable_http_client(url, http_client=client) as streams:
                yield streams

        return cls(connect, name=name or parts.hostname, **options)

    # -- Toolkit -------------------------------------------------------------------------

    async def tools(self, *, refresh: bool = False) -> list[MCPTool]:
        """Connect if needed and return the server's tools as qagents tools.

        The list is cached; ``refresh=True`` lists the tools again. Every listing
        applies the filters and checks the pins.

        Args:
            refresh: Ask the server again instead of using the cached list.

        Raises:
            MCPConnectionError: The server could not be started or reached.
            MCPToolChangedError: A pinned tool changed and ``on_change="error"``. It is
                raised again on every call until a listing shows the pinned
                definitions again, or you change ``pins``.
            ConfigurationError: Two tools end up with the same name.
        """
        async with self._lock():
            if self._tools is None or refresh or not self._usable():
                connection = await self._ensure_connection()
                definitions = await self._list_definitions(connection, refresh=refresh)
                self._tools = self._build(definitions)
            if self._change_error is not None:
                raise self._change_error
            return list(self._tools)

    async def definition_hashes(self) -> dict[str, str]:
        """`definition_hash` of every tool the server lists, before filters and pins.

        Review the tools, then keep this mapping (or part of it) and pass it as
        ``pins`` so later connections only expose those exact definitions.
        """
        async with self._lock():
            connection = await self._ensure_connection()
            definitions = await self._list_definitions(connection, refresh=True)
        return {definition.name: definition_hash(definition) for definition in definitions}

    async def aclose(self) -> None:
        """Close the connection and stop the server process, if there is one.

        The toolkit can be used again afterwards; it reconnects on demand.
        """
        connection, self._connection = self._connection, None
        if connection is None:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - called outside a loop
            running = None
        if connection.loop is not running:
            return  # its loop is gone; closing that loop already stopped the server
        await self._shutdown(connection)

    async def __aenter__(self) -> MCPToolkit:
        await self.tools()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # -- other server features -------------------------------------------------------------

    @property
    def instructions(self) -> str | None:
        """The server's own instructions, once connected.

        They are not added to any prompt; put them in the agent's instructions
        yourself if you want the model to read them.
        """
        connection = self._connection
        if connection is None or connection.client is None:
            return None
        return connection.client.instructions

    @property
    def protocol_version(self) -> str | None:
        """The negotiated protocol revision, once connected."""
        connection = self._connection
        if connection is None or connection.client is None or not connection.alive:
            return None
        return connection.client.protocol_version

    async def list_resources(self) -> list[mcp_types.Resource]:
        """Every resource the server lists (all pages)."""
        client = await self._client()
        found: list[mcp_types.Resource] = []
        cursor: str | None = None
        for _ in range(_MAX_LIST_PAGES):
            page = await client.list_resources(cursor=cursor)
            found.extend(page.resources)
            cursor = page.next_cursor
            if cursor is None:
                break
        return found

    async def read_resource(self, uri: str) -> mcp_types.ReadResourceResult:
        """Read one resource; the result holds text or base64 ``blob`` contents."""
        client = await self._client()
        return await client.read_resource(uri)

    async def list_prompts(self) -> list[mcp_types.Prompt]:
        """Every prompt template the server lists (all pages)."""
        client = await self._client()
        found: list[mcp_types.Prompt] = []
        cursor: str | None = None
        for _ in range(_MAX_LIST_PAGES):
            page = await client.list_prompts(cursor=cursor)
            found.extend(page.prompts)
            cursor = page.next_cursor
            if cursor is None:
                break
        return found

    async def get_prompt(
        self, name: str, arguments: Mapping[str, str] | None = None
    ) -> mcp_types.GetPromptResult:
        """Render one of the server's prompt templates."""
        client = await self._client()
        return await client.get_prompt(name, dict(arguments) if arguments is not None else None)

    # -- calling -------------------------------------------------------------------------

    async def _call(
        self,
        tool: MCPTool,
        ctx: RunContext[Any],
        arguments: dict[str, Any],
        limit: float | None,
    ) -> Any:
        if tool.mcp_name in self._changed:
            raise ToolError(prompts.MCP_TOOL_CHANGED.format(name=tool.name))
        try:
            client = await self._client()
        except (MCPConnectionError, MCPToolChangedError) as error:
            # Only the kind of failure: the full message is for developers and may name
            # paths or addresses the model has no business seeing.
            kind = type(_leaf(error.__cause__ or error)).__name__
            raise ToolError(
                prompts.MCP_SERVER_UNAVAILABLE.format(name=tool.name, server=self.name, error=kind)
            ) from error
        if tool.mcp_name in self._changed:  # the reconnect listed a different definition
            raise ToolError(prompts.MCP_TOOL_CHANGED.format(name=tool.name))

        async def progress(value: float, total: float | None, message: str | None) -> None:
            ctx.emit(
                MCPProgress(
                    server=self.name,
                    tool=tool.name,
                    tool_call_id=ctx.tool_call_id,
                    progress=value,
                    total=total,
                    message=message,
                )
            )

        try:
            async with asyncio.timeout(limit):
                result = await client.call_tool(
                    tool.mcp_name, arguments, progress_callback=progress
                )
        except TimeoutError as error:
            raise ToolError(
                prompts.MCP_TOOL_TIMEOUT.format(name=tool.name, seconds=limit or 0)
            ) from error
        except MCPError as error:
            if error.code == mcp_types.REQUEST_TIMEOUT:
                raise ToolError(
                    prompts.MCP_TOOL_TIMEOUT.format(name=tool.name, seconds=limit or 0)
                ) from error
            if error.code == mcp_types.CONNECTION_CLOSED:
                self._mark_dead()
                raise ToolError(
                    prompts.MCP_CONNECTION_LOST.format(name=tool.name, server=self.name)
                ) from error
            raise ToolError(
                prompts.MCP_REQUEST_FAILED.format(
                    name=tool.name, code=error.code, message=error.message
                )
            ) from error
        except ValidationError as error:
            raise ToolError(prompts.MCP_INVALID_RESPONSE.format(name=tool.name)) from error
        except (OSError, EOFError) as error:
            self._mark_dead()
            raise ToolError(
                prompts.MCP_CONNECTION_LOST.format(name=tool.name, server=self.name)
            ) from error
        except Exception as error:
            if _is_transport_failure(error):
                self._mark_dead()
                raise ToolError(
                    prompts.MCP_CONNECTION_LOST.format(name=tool.name, server=self.name)
                ) from error
            raise
        return self._render(tool, result)

    def _render(self, tool: MCPTool, result: mcp_types.CallToolResult) -> Any:
        texts = [self._render_block(block) for block in result.content]
        text = "\n".join(piece for piece in texts if piece)
        if result.is_error:
            raise ToolError(text or prompts.MCP_TOOL_ERROR_EMPTY.format(name=tool.name))
        structured = result.structured_content
        if structured is not None and (self.prefer_structured or not text):
            return structured
        return text or None

    def _render_block(self, block: Any) -> str:
        kind = getattr(block, "type", None)
        if isinstance(block, mcp_types.TextContent):
            return block.text
        if isinstance(block, mcp_types.ImageContent):
            return prompts.MCP_IMAGE_PLACEHOLDER.format(
                mime_type=block.mime_type, size=_decoded_size(block.data)
            )
        if isinstance(block, mcp_types.AudioContent):
            return prompts.MCP_AUDIO_PLACEHOLDER.format(
                mime_type=block.mime_type, size=_decoded_size(block.data)
            )
        if isinstance(block, mcp_types.ResourceLink):
            return prompts.MCP_RESOURCE_LINK.format(
                uri=block.uri, name=block.name, mime_type=block.mime_type or "unknown type"
            )
        if isinstance(block, mcp_types.EmbeddedResource):
            contents = block.resource
            mime_type = contents.mime_type or "unknown type"
            if isinstance(contents, mcp_types.TextResourceContents):
                if self.include_resource_text:
                    return prompts.MCP_RESOURCE_TEXT.format(
                        uri=contents.uri, mime_type=mime_type, text=contents.text
                    )
                size = len(contents.text.encode("utf-8"))
            else:
                size = _decoded_size(contents.blob)
            return prompts.MCP_RESOURCE_BLOB_PLACEHOLDER.format(
                uri=contents.uri, mime_type=mime_type, size=size
            )
        return prompts.MCP_UNKNOWN_CONTENT.format(type=kind or type(block).__name__)

    # -- listing and pinning -----------------------------------------------------------------

    async def _list_definitions(
        self, connection: _Connection, *, refresh: bool
    ) -> list[mcp_types.Tool]:
        client = connection.client
        assert client is not None
        definitions: list[mcp_types.Tool] = []
        cursor: str | None = None
        try:
            async with asyncio.timeout(self.connect_timeout):
                for _ in range(_MAX_LIST_PAGES):
                    page = await client.list_tools(
                        cursor=cursor, cache_mode="refresh" if refresh else "use"
                    )
                    definitions.extend(page.tools)
                    cursor = page.next_cursor
                    if cursor is None:
                        break
        except Exception as error:
            if isinstance(error, MCPError) and error.code != mcp_types.CONNECTION_CLOSED:
                raise MCPConnectionError(
                    f"MCP server {self.name!r} refused to list its tools: {error.message}"
                ) from error
            self._mark_dead()
            raise MCPConnectionError(
                f"Could not list the tools of MCP server {self.name!r}: {_describe(error)}. "
                "Check that the server is running and speaks MCP."
            ) from error
        connection.verified = True
        return definitions

    def _build(self, definitions: list[mcp_types.Tool]) -> list[MCPTool]:
        """Filter, pin-check and wrap the listed tools; record (not raise) pin mismatches."""
        built: list[MCPTool] = []
        changed: dict[str, tuple[str, str]] = {}
        seen_names: dict[str, str] = {}
        for definition in definitions:
            if not self._selected(definition.name):
                continue
            digest = definition_hash(definition)
            expected = (self.pins or {}).get(definition.name)
            if expected is None and self.pin_on_first_use:
                expected = self._first_seen.get(definition.name)
            if expected is not None and expected != digest:
                changed[definition.name] = (expected, digest)
                self._changed[definition.name] = digest
                continue
            self._changed.pop(definition.name, None)
            if self.pin_on_first_use:
                self._first_seen.setdefault(definition.name, digest)
            if not _is_object_schema(definition.input_schema):
                logger.warning(
                    "MCP server %r: tool %r is left out because its input schema is not an "
                    "object schema.",
                    self.name,
                    definition.name,
                )
                continue
            exposed = self._exposed_name(definition.name)
            if exposed in seen_names:
                raise ConfigurationError(
                    f"MCP server {self.name!r}: tools {seen_names[exposed]!r} and "
                    f"{definition.name!r} both become {exposed!r}. Hide one with exclude=."
                )
            seen_names[exposed] = definition.name
            built.append(self._make_tool(definition, exposed))
        if self.pins:
            missing = sorted(set(self.pins) - {definition.name for definition in definitions})
            if missing:
                logger.warning(
                    "MCP server %r does not list the pinned tools %s.", self.name, missing
                )
        self._change_error = None
        if changed:
            summary = "; ".join(
                f"{tool_name!r} expected {expected[:12]}..., got {actual[:12]}..."
                for tool_name, (expected, actual) in sorted(changed.items())
            )
            message = (
                f"MCP server {self.name!r} changed the definition of {len(changed)} pinned "
                f"tool(s): {summary}. They are not exposed. Review the new definitions "
                "(MCPToolkit.definition_hashes() lists them) and update pins= if you trust them."
            )
            if self.on_change == "error":
                self._change_error = MCPToolChangedError(message, changed=changed)
            else:
                logger.warning("%s", message)
        return built

    def _make_tool(self, definition: mcp_types.Tool, exposed: str) -> MCPTool:
        hints = definition.annotations
        read_only = bool(hints is not None and hints.read_only_hint is True)
        destructive = not read_only and not (hints is not None and hints.destructive_hint is False)
        if self.approval == "all":
            requires_approval = True
        elif self.approval == "none":
            requires_approval = False
        else:
            requires_approval = destructive
        return MCPTool(
            self,
            definition,
            name=exposed,
            requires_approval=requires_approval,
            read_only=read_only,
            idempotent=bool(hints is not None and hints.idempotent_hint is True),
            timeout=self.timeout,
        )

    def _selected(self, tool_name: str) -> bool:
        if self.include is not None and not any(
            fnmatch.fnmatchcase(tool_name, pattern) for pattern in self.include
        ):
            return False
        return not any(fnmatch.fnmatchcase(tool_name, pattern) for pattern in self.exclude)

    def _exposed_name(self, tool_name: str) -> str:
        joined = f"{self.prefix}{self.separator}{tool_name}" if self.prefix else tool_name
        clean = _INVALID_NAME_CHARACTERS.sub("_", joined) or "tool"
        if len(clean) <= _MAX_NAME:
            return clean
        # Too long: keep the start and add a short hash of the full name, so it stays unique.
        digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:8]
        return f"{clean[: _MAX_NAME - 9]}_{digest}"

    # -- connection ----------------------------------------------------------------------

    def _lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        for known in [k for k in self._locks if k.is_closed()]:
            del self._locks[known]
        lock = self._locks.get(loop)
        if lock is None:
            lock = self._locks[loop] = asyncio.Lock()
        return lock

    def _usable(self) -> bool:
        connection = self._connection
        return (
            connection is not None
            and connection.alive
            and connection.loop is asyncio.get_running_loop()
        )

    async def _client(self) -> Client:
        if self._usable():
            assert self._connection is not None and self._connection.client is not None
            return self._connection.client
        async with self._lock():
            connection = await self._ensure_connection()
            if not connection.verified and self._tools is not None:
                # A reconnect: the server may have changed its tools while we were away.
                definitions = await self._list_definitions(connection, refresh=True)
                self._tools = self._build(definitions)
                if self._change_error is not None:
                    logger.warning("%s", self._change_error)
            assert connection.client is not None
            return connection.client

    async def _ensure_connection(self) -> _Connection:
        """Return a live connection on the running loop, opening one if needed. Hold the lock."""
        loop = asyncio.get_running_loop()
        connection = self._connection
        if connection is not None and connection.alive and connection.loop is loop:
            return connection
        if connection is not None:
            self._connection = None
            if connection.loop is loop:
                await self._shutdown(connection)
        stop = asyncio.Event()
        ready: asyncio.Future[Client] = loop.create_future()
        holder: list[_Connection] = []
        task = loop.create_task(self._own(ready, stop, holder), name=f"qagents-mcp:{self.name}")
        connection = _Connection(loop=loop, task=task, stop=stop)
        holder.append(connection)
        try:
            async with asyncio.timeout(self.connect_timeout):
                connection.client = await asyncio.shield(ready)
        except BaseException as error:
            connection.alive = False
            await self._shutdown(connection)
            if isinstance(error, asyncio.CancelledError):
                raise
            if isinstance(error, TimeoutError):
                raise MCPConnectionError(
                    f"MCP server {self.name!r} did not finish the MCP handshake within "
                    f"{self.connect_timeout:g} seconds. Check that the command starts an MCP "
                    "server that speaks on stdout (and logs only to stderr), or raise "
                    "connect_timeout."
                ) from error
            raise MCPConnectionError(
                f"Could not connect to MCP server {self.name!r}: {_describe(error)}. "
                "Check the command or URL, and that the server speaks MCP."
            ) from error
        self._connection = connection
        return connection

    async def _own(
        self, ready: asyncio.Future[Client], stop: asyncio.Event, holder: list[_Connection]
    ) -> None:
        """Hold the SDK client open in a task of its own.

        The SDK's transports are anyio task groups, which must be entered and
        exited by the same task. Runs and `aclose` happen in other tasks, so one
        task owns the connection and the others talk to it through the client.
        """
        try:
            async with self._make_client() as client:
                if not ready.done():
                    ready.set_result(client)
                await stop.wait()
        except asyncio.CancelledError:
            if not ready.done():
                ready.cancel()
            raise
        except BaseException as error:
            if not ready.done():
                ready.set_exception(_leaf(error))
            elif not stop.is_set():
                logger.warning(
                    "The connection to MCP server %r ended: %s", self.name, _describe(error)
                )
        finally:
            if holder:
                holder[0].alive = False

    def _make_client(self) -> Client:
        server = self._server
        target = server if isinstance(server, MCPServer | Server) else server()
        return Client(target, mode=self.protocol, client_info=_client_info())

    def _mark_dead(self) -> None:
        connection = self._connection
        if connection is not None:
            connection.alive = False
            connection.stop.set()

    async def _shutdown(self, connection: _Connection) -> None:
        connection.alive = False
        connection.stop.set()
        task = connection.task
        if task.done():
            with contextlib.suppress(BaseException):
                task.result()
            return
        if connection.client is None:
            # Still in the handshake, so it never reaches ``stop``: cancel it. The transport
            # still stops the server process (close stdin, wait briefly, then kill).
            task.cancel()
        # The stdio transport bounds its own shutdown (about 5 s at worst).
        done, _ = await asyncio.wait({task}, timeout=15)
        if not done:
            task.cancel()
            await asyncio.wait({task}, timeout=15)
        if task.done() and not task.cancelled():
            task.exception()  # retrieved: the owner task already logged what went wrong

    def __repr__(self) -> str:
        return f"MCPToolkit(name={self.name!r}, prefix={self.prefix!r})"


# -- helpers ---------------------------------------------------------------------------


def _client_info() -> mcp_types.Implementation:
    try:
        from importlib.metadata import version

        current = version("qagents")
    except Exception:  # pragma: no cover - running from a source tree
        current = "0"
    return mcp_types.Implementation(name="qagents", version=current)


def _input_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """The schema the model sees: the server's, without generated titles, always an object."""
    cleaned = _drop_titles(dict(schema))
    cleaned.setdefault("type", "object")
    cleaned.setdefault("properties", {})
    return cleaned


def _drop_titles(schema: Any) -> Any:
    """Drop auto-generated ``title`` strings: they cost tokens and tell the model nothing.

    A property that is itself called ``title`` is kept: its value is a schema, not a string.
    """
    if isinstance(schema, dict):
        return {
            key: _drop_titles(value)
            for key, value in schema.items()
            if not (key == "title" and isinstance(value, str))
        }
    if isinstance(schema, list):
        return [_drop_titles(item) for item in schema]
    return schema


def _is_object_schema(schema: Mapping[str, Any]) -> bool:
    kind = schema.get("type", "object")
    return kind == "object" or (isinstance(kind, list) and "object" in kind)


def _decoded_size(data: str) -> int:
    """Bytes of a base64 string, without keeping the decoded data."""
    try:
        return len(base64.b64decode(data, validate=False))
    except (ValueError, TypeError):
        return len(data) * 3 // 4


def _leaf(error: BaseException) -> BaseException:
    """The first real error inside nested exception groups."""
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    return error


def _describe(error: BaseException) -> str:
    leaf = _leaf(error)
    if isinstance(leaf, MCPError):
        return f"{leaf.message} (MCP error {leaf.code})"
    text = str(leaf).strip()
    return f"{type(leaf).__name__}: {text}" if text else type(leaf).__name__


def _is_transport_failure(error: BaseException) -> bool:
    import anyio

    leaf = _leaf(error)
    return isinstance(
        leaf, anyio.ClosedResourceError | anyio.BrokenResourceError | anyio.EndOfStream
    ) or (isinstance(leaf, MCPError) and leaf.code == mcp_types.CONNECTION_CLOSED)


__all__ = [
    "MCPConnectionError",
    "MCPProgress",
    "MCPTool",
    "MCPToolChangedError",
    "MCPToolkit",
    "definition_hash",
]
