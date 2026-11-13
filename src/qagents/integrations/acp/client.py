"""The ACP v1 client: `ACPAgent` drives any ACP agent as a qagents `ExternalAgent`.

The Agent Client Protocol (https://agentclientprotocol.com) is JSON-RPC 2.0 over the
agent's stdin and stdout, one message per line. One process serves many sessions,
one prompt at a time per session. What the client does, in order:

1. ``initialize`` with ``protocolVersion: 1`` and client capabilities that offer
   nothing (no ``fs``, no ``terminal``: the agent uses its own tools). An agent that
   answers another protocol version is refused.
2. ``authenticate``, only when the agent lists ``authMethods`` and you chose one
   (``auth_method=``). Without a choice qagents never starts a login; if the agent
   then refuses ``session/new`` with "authentication required", the error names
   the methods it offers.
3. ``session/new {cwd, mcpServers}`` for a new conversation (``cwd`` is the absolute
   workspace). A session that this connection opened is continued directly while
   it is still open, with the MCP servers it has; an older or closed one with
   ``session/resume`` (or ``session/load``, whose replayed history is ignored) when
   the agent supports it.
   MCP servers are sent again on resume, because agents do not keep them. A
   function ``mcp_servers=`` runs only when servers are sent (see `MCPServers`).
4. ``session/set_config_option`` for ``model=`` and ``config_options=``, when the
   session offers that option and its value differs.
5. ``session/prompt``. ``session/update`` notifications become qagents events
   (see `qagents.integrations.acp.mapping`); ``session/request_permission`` is
   answered by the `PermissionBridge` (an ``allow_always`` option is never
   chosen; any failure answers ``cancelled``); the ``stopReason`` becomes the
   result's status. A connection that breaks before the prompt is sent ends the
   turn as ``error``, not ``interrupted``: nothing of the task reached the agent.
6. Cancel: ``session/cancel``, then ``cancel_grace_s`` seconds for the agent to
   answer ``stopReason: "cancelled"``; after that the process tree is killed and
   the next turn starts a new process. A cancel that comes before the prompt is
   sent ends the turn at once, without sending it (and without
   ``session/cancel``): a setup request still waiting for its answer
   (``session/new``, ``session/resume``, ``session/load``,
   ``session/set_config_option``) is given up, which the agent is told with
   ``$/cancel_request``, and the process is kept. Once a turn is being cancelled,
   every permission request is answered ``cancelled``.
7. ``session/close`` for the idle sessions beyond ``keep_sessions`` once a turn
   ends, when the agent can close and resume sessions: the agent releases what
   each one holds (Devvi: an agent and the session's MCP server processes). A
   later turn in such a session resumes it, once the agent has answered the close
   (an agent refuses to resume a session it still holds). After a failed close,
   that turn tries ``session/resume`` and, if the agent refuses it, prompts the
   session directly: the agent may still have it open.
8. Servers that stop with their turn (``mcp_servers=`` gives an async context
   manager): the turn's session is closed right after the turn, before they stop,
   and the next turn in it resumes it with servers of its own.

Requests the agent sends that qagents does not implement (``fs/*``,
``terminal/*``) are answered -32601, so richer agents work too.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import os
import sys
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ... import prompts
from ... import prompts as _prompts
from ...errors import ConfigurationError
from ...external import (
    ExternalAgent,
    ExternalAgentConfigurationError,
    ExternalAgentConnectionError,
    ExternalAgentProtocolError,
    ExternalAgentTimeoutError,
    ExternalAgentUnavailableError,
    ExternalCapabilities,
    ExternalOutcome,
    ExternalTurnContext,
)
from .._jsonrpc import (
    INVALID_PARAMS,
    INVALID_REQUEST,
    MAX_LINE_BYTES,
    METHOD_NOT_FOUND,
    JsonRpcError,
    JsonRpcPeer,
)
from .._process import ManagedProcess, build_env
from .mapping import (
    AUTH_REQUIRED,
    PROTOCOL_VERSION,
    TITLES_ARE_TOOL_NAMES,
    ToolNamer,
    TurnUpdates,
    default_tool_name,
    error_text,
    outcome_for_prompt_error,
    outcome_for_stop_reason,
    prompt_blocks,
    title_tool_name,
)

logger = logging.getLogger("qagents.integrations.acp")

CLIENT_CAPABILITIES: dict[str, Any] = {
    "fs": {"readTextFile": False, "writeTextFile": False},
    "terminal": False,
}
"""What qagents offers the agent: nothing. The agent reads files and runs commands with its
own tools, under its own sandbox, and asks qagents for permission."""

CANCELLED: dict[str, Any] = {"outcome": {"outcome": "cancelled"}}
"""The answer to a permission request that qagents does not grant: the agent does not run it."""

CLOSE_TIMEOUT_S = 10.0
"""Seconds the agent gets to answer the ``session/close`` of an idle session before qagents
logs it and stops waiting for it elsewhere (the next turn, `disconnect`). The close itself
goes on: a later turn in that session waits for its answer, up to ``request_timeout_s``."""

_PROMPT_CHECKPOINT = "acp_prompt_sent"
_SESSION_CHECKPOINT = "external_session"
_WINDOWS = sys.platform == "win32"


class _Sessions:
    """Ids of the sessions a connection has open, least recently used first.

    A set (``add``, ``discard``, ``in``, ``clear``) that keeps the order of use: ``add``
    of an id that is already there makes it the most recently used.
    """

    def __init__(self) -> None:
        self._ids: dict[str, None] = {}

    def __contains__(self, session_id: object) -> bool:
        return session_id in self._ids

    def __iter__(self) -> Iterator[str]:
        return iter(list(self._ids))

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, session_id: str) -> None:
        self._ids.pop(session_id, None)
        self._ids[session_id] = None

    def discard(self, session_id: str) -> None:
        self._ids.pop(session_id, None)

    def clear(self) -> None:
        self._ids.clear()


@dataclass
class _Setup:
    """What a turn's setup has done so far; `ACPAgent.run_turn` reads it however it ended."""

    session_id: str | None = None
    """The session the turn prompts, once it is open."""
    resumed: bool = False
    options: list[Any] | None = None
    """``configOptions`` to apply with ``session/set_config_option``; ``None``: nothing to set."""
    release: bool = False
    """The session mounts MCP servers that stop with the turn: close it after the turn."""


# --- MCP servers handed to the agent -------------------------------------------------------


def _check_entries(entries: Mapping[str, str], what: str) -> dict[str, str]:
    checked: dict[str, str] = {}
    for key, value in dict(entries).items():
        if not isinstance(key, str) or not key or "=" in key or "\0" in key:
            raise ConfigurationError(f"{what} name {key!r} is not valid.")
        if not isinstance(value, str) or "\0" in value:
            raise ConfigurationError(f"{what} {key} must be a string.")
        checked[key] = value
    return checked


@dataclass(frozen=True)
class MCPServerStdio:
    """An MCP server the agent starts for a session and talks to over stdio.

        MCPServerStdio("tickets", command=sys.executable, args=["-m", "my_tools"])

    The agent starts ``command`` itself, so it must be an absolute path. The
    server gets the agent's environment plus ``env`` and the ``passthrough``
    variables copied from qagents' environment when the session starts (Devvi
    removes variables whose names look like secrets from what it passes on, so
    name them here). Values are never shown in ``repr``.
    """

    name: str
    command: str
    args: Sequence[str] = ()
    env: Mapping[str, str] = field(default_factory=dict, repr=False)
    passthrough: Sequence[str] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigurationError("An MCP server needs a non-empty name.")
        if not isinstance(self.command, str) or not os.path.isabs(self.command):
            raise ConfigurationError(
                f"MCP server {self.name!r}: command must be an absolute path, not "
                f"{self.command!r} (the agent starts it, from its own working directory)."
            )
        if isinstance(self.args, str) or not all(isinstance(arg, str) for arg in self.args):
            raise ConfigurationError(f"MCP server {self.name!r}: args must be a list of strings.")
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "env", _check_entries(self.env, "Environment variable"))
        object.__setattr__(self, "passthrough", tuple(self.passthrough))

    def to_acp(self) -> dict[str, Any]:
        """The ``mcpServers`` entry of ``session/new`` (ACP ``McpServerStdio``)."""
        env = dict(self.env)
        folded = {key.upper(): key for key in os.environ} if _WINDOWS else None
        for name in self.passthrough:
            key = folded.get(name.upper()) if folded is not None else name
            if key is not None and key in os.environ and name not in env:
                env[name] = os.environ[key]
        return {
            "name": self.name,
            "command": self.command,
            "args": list(self.args),
            "env": [{"name": key, "value": value} for key, value in env.items()],
        }


@dataclass(frozen=True)
class MCPServerHTTP:
    """An MCP server the agent reaches over Streamable HTTP.

        MCPServerHTTP("qagents", url="http://127.0.0.1:8765/mcp",
                      headers={"Authorization": f"Bearer {token}"})

    Needs an agent that advertises ``mcpCapabilities.http`` (Devvi does).
    Header values are never shown in ``repr``.
    """

    name: str
    url: str
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigurationError("An MCP server needs a non-empty name.")
        if not isinstance(self.url, str) or not self.url.startswith(("http://", "https://")):
            raise ConfigurationError(
                f"MCP server {self.name!r}: url must be an absolute http(s) URL, not {self.url!r}."
            )
        object.__setattr__(self, "headers", _check_entries(self.headers, "Header"))

    def to_acp(self) -> dict[str, Any]:
        """The ``mcpServers`` entry of ``session/new`` (ACP ``McpServerHttp``)."""
        return {
            "type": "http",
            "name": self.name,
            "url": self.url,
            "headers": [{"name": key, "value": value} for key, value in self.headers.items()],
        }


MCPServer = MCPServerStdio | MCPServerHTTP
"""An MCP server to mount in every session the agent starts."""

MCPServers = (
    Sequence[MCPServer | Mapping[str, Any]]
    | Callable[
        [ExternalTurnContext],
        Sequence[MCPServer | Mapping[str, Any]]
        | Awaitable[Sequence[MCPServer | Mapping[str, Any]]]
        | contextlib.AbstractAsyncContextManager[Sequence[MCPServer | Mapping[str, Any]]],
    ]
)
"""The servers for a session: a list, or a function of the turn that gives one.

A mapping is sent as it is (an ACP ``McpServer`` object). The function runs only when
servers are sent: when the turn opens a session (``session/new``) or reopens one
(``session/resume``, ``session/load``). A session this connection already has open is
continued with the servers it has, and the function does not run. It may return:

* the list, or an awaitable of it: servers that live as long as the session (started
  by the agent, as `MCPServerStdio` ones are, or running anyway);
* an async context manager that yields the list: servers that live for one turn, such
  as a qagents tool server bound to the calling run. qagents enters it before the
  session opens, closes the session right after the turn (``session/close``), then
  exits it, so the next turn in the session resumes it with servers of its own. An
  agent that cannot close sessions keeps the session open, but it is not continued
  directly again: a later turn resumes it, or says why it cannot.

    @contextlib.asynccontextmanager
    async def turn_tools(turn):
        async with tool_server([lookup_customer], ctx=turn.parent) as server:
            yield [server.acp_config()]

    coder = ACPAgent([...], name="coder", mcp_servers=turn_tools)
"""


def _server_wire(server: Any) -> dict[str, Any]:
    if isinstance(server, MCPServerStdio | MCPServerHTTP):
        return server.to_acp()
    if isinstance(server, Mapping):
        return dict(server)
    raise ConfigurationError(prompts.ACP_SERVER_UNKNOWN_TYPE.format(type=type(server).__name__))


# --- the agent -----------------------------------------------------------------------------


class ACPAgent(ExternalAgent):
    """Any agent that speaks ACP v1 over stdio, as a qagents `ExternalAgent`.

        agent = ACPAgent(
            command=["node", "C:/tools/some-agent/bin.js", "--acp"],
            name="other-coder",
            env={"SOME_API_KEY": os.environ["SOME_API_KEY"]},
            workspace="C:/work/repo",
        )
        rules = Permissions(allow=["other-coder.read"],
                            ask=["other-coder.edit", "other-coder.execute"])
        lead = Agent(model, tools=[agent.as_tool(timeout=1800)], permissions=rules)
        async with agent:
            result = await lead.run("Make the failing test pass.")

    An ask rule wins over an allow rule, so ``ask=["other-coder.*"]`` would ask about
    reads too.

    What ACP gives qagents, and what it does not:

    * **No token counts.** ``usage_update {used, size}`` is context pressure and
      becomes an ``external_event`` of kind ``context``. Under a token or cost
      budget the calling run refuses to start the agent unless it was built with
      ``allow_unmetered=True``.
    * **Text arrives as the agent sends it**: per committed block for Devvi, per
      token chunk for agents that stream. Chunks with a new ``messageId``, and
      every block of an agent whose chunks are blocks (`chunks_are_blocks`), are
      separated by a blank line in the turn's text.
    * **Tool names** for permission rules (``"<tool name>.<tool>"``) come from the
      tool call's ``name``, else its ``kind`` (``execute``, ``edit``, ...), from
      everything the agent said about the call before it asked. The ``title`` is
      display text the model can steer (``"rm"``, ``"setup.py"``), so it is used
      only for agents whose code puts the tool's name there: Devvi (``bash``; see
      `TITLES_ARE_TOOL_NAMES`), or any agent with ``tool_name=title_tool_name``.
      Pass ``tool_name=`` to choose differently.
    * **Sessions are closed when idle.** After a turn, sessions beyond
      ``keep_sessions`` that no turn uses are closed with ``session/close``
      (least recently used first), when the agent can close and resume sessions;
      a later turn in one resumes it.
    * **Crash safety.** The agent chooses session ids. With ``resumable=True``
      (only for agents that can resume sessions), a call repeated after a crash
      of the calling run resumes the session the first attempt journaled and
      tells the agent to check the workspace and finish; the tool is then
      idempotent. Otherwise a repeated call is reported as having an unknown
      outcome, the safe default.

    Args:
        command: Program and arguments that start the agent, never a shell
            string. On Windows run ``.js`` agents as ``["node", ".../bin.js"]``.
        name: The agent's name (and the default tool name and permission prefix).
        env: Variables set for the agent process, besides the base allowlist
            (``PATH``, ``SYSTEMROOT``, ``TEMP``, ...). Nothing else of qagents'
            environment reaches it.
        passthrough: Names of variables copied from qagents' environment, such
            as ``["DEEPSEEK_API_KEY"]``.
        inherit_env: Give the agent qagents' whole environment (every secret in
            it too). Off by default.
        cwd: Working directory of the agent process, and the default workspace.
        workspace: The default ``cwd`` of sessions; defaults to ``cwd``. ACP
            needs one: a turn without a workspace is refused.
        mcp_servers: MCP servers to mount in each session (`MCPServerStdio`,
            `MCPServerHTTP`), or a function of the turn that gives them, called
            only when a session is opened or reopened; an async context manager
            for servers that stop with the turn (see `MCPServers`).
        model: The model to select with ``session/set_config_option``: an option
            value, an option name (``"deepseek-v4-pro"``) or, for agents whose
            values are JSON lists (Devvi), the list (``("deepseek-official",
            "deepseek-v4-pro")``). A model the session does not offer is refused.
        config_options: Other options to set, ``{config id: value}``, for
            example ``{"reasoning_effort": "high"}``.
        auth_method: The ``authMethods`` id to authenticate with, when the agent
            lists any. ``None``: never authenticate.
        resumable: Re-attach to the earlier session when a call is repeated
            after a crash (see above). Only for agents that support
            ``session/resume`` or ``session/load``.
        raw_events: Forward ``session/update`` kinds that have no mapping as
            ``external_event`` of kind ``unknown``; otherwise they are counted
            in ``ExternalResult.raw["ignored_events"]``.
        tool_name: ``(tool call fields) -> name``, instead of the rule above
            (`default_tool_name`, or `title_tool_name` for Devvi).
        keep_sessions: Sessions kept open after their turn ended (default 1:
            the last one used). The agent holds each open session in memory, with
            the MCP servers mounted in it (a process per stdio server), so the
            older ones are closed with ``session/close``; continuing one later
            costs a ``session/resume``. ``0`` closes every session after its turn.
            Only for agents that advertise ``sessionCapabilities.close`` and can
            resume sessions; others keep every session until the agent is closed.
        handshake_timeout_s: Seconds for ``initialize`` (and ``authenticate``).
        request_timeout_s: Seconds for ``session/new``, ``session/resume``,
            ``session/load`` and ``session/set_config_option`` (MCP servers
            start during ``session/new``), and for the answer to the
            ``session/close`` of a session a turn continues. A request without
            an answer ends the turn as ``error``. The prompt itself has no
            timeout: use ``as_tool(timeout=...)`` or the calling run's limits.
        close_grace_s: Seconds the process gets to exit after its stdin closes,
            before its tree is killed.
        max_line_bytes: Largest message accepted from the agent.
        **options: The `ExternalAgent` options: ``description``,
            ``permissions``, ``on_permission``, ``approval_timeout_s``,
            ``cancel_grace_s``, ``max_external_turns``, ``allow_unmetered``,
            ``track_changes``.
    """

    backend = "acp"

    chunks_are_blocks: bool = False
    """Each text ``agent_message_chunk`` of this agent is a whole committed block, not a
    token chunk (Devvi: `DevviACPAgent` sets it). Blocks are then separated by a blank line
    in the turn's text. Otherwise chunks are joined as they come, and a new ``messageId``
    starts a new block."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        name: str = "acp",
        env: Mapping[str, str] | None = None,
        passthrough: Iterable[str] = (),
        inherit_env: bool = False,
        cwd: str | Path | None = None,
        workspace: str | Path | None = None,
        mcp_servers: MCPServers = (),
        model: str | Sequence[str] | None = None,
        config_options: Mapping[str, str | bool] | None = None,
        auth_method: str | None = None,
        resumable: bool = False,
        raw_events: bool = False,
        tool_name: ToolNamer | None = None,
        keep_sessions: int = 1,
        handshake_timeout_s: float = 60.0,
        request_timeout_s: float = 120.0,
        close_grace_s: float = 5.0,
        max_line_bytes: int = MAX_LINE_BYTES,
        **options: Any,
    ) -> None:
        super().__init__(
            name=name, workspace=workspace if workspace is not None else cwd, **options
        )
        if isinstance(command, str | bytes) or not command:
            raise ConfigurationError(
                "ACPAgent needs the command as a list of strings, for example "
                "['node', 'C:/tools/agent/bin.js', '--acp'], never a shell string."
            )
        if isinstance(keep_sessions, bool) or not isinstance(keep_sessions, int):
            raise ConfigurationError(
                f"keep_sessions must be a whole number, not {keep_sessions!r}."
            )
        if keep_sessions < 0:
            raise ConfigurationError(f"keep_sessions must be 0 or more, not {keep_sessions}.")
        self.command = [str(part) for part in command]
        build_env(extra=env, source={})  # refuses names and values that are not strings now
        self.env = dict(env or {})
        self.passthrough = list(passthrough)
        self.inherit_env = inherit_env
        self.process_cwd = os.path.abspath(cwd) if cwd is not None else self.workspace
        """Working directory of the agent process."""
        self.mcp_servers = mcp_servers
        if not callable(mcp_servers):
            for server in mcp_servers:
                _server_wire(server)  # refuse a wrong entry when the agent is built
        if model is not None and not isinstance(model, str):
            model = tuple(str(part) for part in model)
        self.model = model
        self.config_options: dict[str, str | bool] = dict(config_options or {})
        self.auth_method = auth_method
        self.resumable_turns = resumable
        self.raw_events = raw_events
        self.tool_name = tool_name
        self.keep_sessions = keep_sessions
        self.handshake_timeout_s = handshake_timeout_s
        self.request_timeout_s = request_timeout_s
        self.close_grace_s = close_grace_s
        self.max_line_bytes = max_line_bytes
        self._process: ManagedProcess | None = None
        self._peer: JsonRpcPeer | None = None
        self._info: dict[str, Any] = {}
        self._live = _Sessions()
        """Sessions this connection opened or resumed and has not closed: they are prompted
        directly. Least recently used first."""
        self._active: dict[str, TurnUpdates] = {}
        """The running turn of each session."""
        self._busy: dict[str, int] = {}
        """Sessions a running turn uses or is about to continue (how many turns): never
        closed as idle."""
        self._prompting: set[ExternalTurnContext] = set()
        """Turns whose ``session/prompt`` is sent (or queued to be written): only these
        get a ``session/cancel``."""
        self._closing: set[asyncio.Task[None]] = set()
        """Session closes still running (also a subclass's, such as DevviACPAgent's): the
        next turn waits for them before it opens a session, and `disconnect` before stdin
        closes."""
        self._close_deadlines: dict[asyncio.Task[None], float] = {}
        """The loop time until which others wait for each of this class's closes: an
        answer that takes longer is still waited for by a turn in that session only."""
        self._closing_ids: dict[str, asyncio.Task[None]] = {}
        """The same closes by session id: a turn that continues one of these waits for it."""
        self._unclosed: set[str] = set()
        """Sessions whose ``session/close`` failed: the agent may still hold them open."""
        self._session_options: dict[str, list[Any]] = {}
        self._disconnecting: set[asyncio.Task[None]] = set()
        """`disconnect` of a cancelled `connect`, running in the background."""

    # -- what the handshake said ------------------------------------------------------------

    @property
    def agent_info(self) -> dict[str, Any]:
        """``agentInfo`` from ``initialize`` (``{name, version, title?}``); empty before."""
        info = self._info.get("agentInfo")
        return dict(info) if isinstance(info, Mapping) else {}

    @property
    def agent_capabilities(self) -> dict[str, Any]:
        """``agentCapabilities`` from ``initialize``; empty before the handshake."""
        caps = self._info.get("agentCapabilities")
        return dict(caps) if isinstance(caps, Mapping) else {}

    @property
    def auth_methods(self) -> list[dict[str, Any]]:
        """``authMethods`` from ``initialize``."""
        methods = self._info.get("authMethods")
        if not isinstance(methods, list):
            return []
        return [dict(method) for method in methods if isinstance(method, Mapping)]

    def session_config(self, session_id: str) -> list[Any]:
        """The latest ``configOptions`` the agent reported for ``session_id``."""
        return list(self._session_options.get(session_id, []))

    @property
    def backend_name(self) -> str:
        """``"acp:<agentInfo.name>@<agentInfo.version>"``, or ``"acp"`` before the handshake."""
        info = self.agent_info
        agent = info.get("name")
        if not isinstance(agent, str) or not agent:
            return "acp"
        version = info.get("version")
        return f"acp:{agent}@{version}" if isinstance(version, str) and version else f"acp:{agent}"

    def _can_resume(self) -> bool:
        sessions = self.agent_capabilities.get("sessionCapabilities")
        return isinstance(sessions, Mapping) and sessions.get("resume") is not None

    def _can_load(self) -> bool:
        return self.agent_capabilities.get("loadSession") is True

    def _can_close(self) -> bool:
        """The agent closes sessions and can reopen them, so an idle one may be closed."""
        sessions = self.agent_capabilities.get("sessionCapabilities")
        closes = isinstance(sessions, Mapping) and sessions.get("close") is not None
        return closes and (self._can_resume() or self._can_load())

    def _default_namer(self) -> ToolNamer:
        """`title_tool_name` for agents known to put tool names in titles, else the default."""
        trusted = self.agent_info.get("name") in TITLES_ARE_TOOL_NAMES
        return title_tool_name if trusted else default_tool_name

    # -- lifecycle ---------------------------------------------------------------------------

    async def capabilities(self) -> ExternalCapabilities:
        await self.ensure_connected()
        return ExternalCapabilities(
            token_usage=False,
            live_tokens=False,
            cancel_turn=True,
            resume_sessions=self._can_resume() or self._can_load(),
            client_session_ids=False,
            permission_requests=True,
            mcp_servers=True,
        )

    async def connect(self) -> None:
        environment = build_env(
            passthrough=self.passthrough, extra=self.env, inherit_env=self.inherit_env
        )
        process = await ManagedProcess.start(
            self.command, env=environment, cwd=self.process_cwd, name=self.name
        )
        peer = JsonRpcPeer.for_process(
            process, cancel_method="$/cancel_request", max_line_bytes=self.max_line_bytes
        )
        peer.on_notification("session/update", self._on_update)
        peer.on_request("session/request_permission", self._on_permission)
        peer.on_other_notification(self._on_other_notification)
        peer.on_close(lambda error: self._on_peer_closed(peer, error))
        self._process, self._peer = process, peer
        peer.start()
        try:
            await self._handshake(peer, process)
            if peer.closed:
                raise self._setup_error(
                    prompts.ACP_STOPPED_AFTER_HANDSHAKE, process, peer.close_error
                )
        except asyncio.CancelledError:
            process.kill()
            self._release_later()
            raise
        except BaseException:
            await self.disconnect()
            raise

    async def disconnect(self) -> None:
        # The references are kept until the process is closed: a disconnect cancelled
        # half way (the loop shutting down) leaves the process reachable by a later
        # disconnect or `_kill`, and `ManagedProcess.close` kills the tree when cancelled.
        peer, process = self._peer, self._process
        try:
            # let the agent finish closing sessions before its stdin closes
            await self._wait_for_closes(limit=CLOSE_TIMEOUT_S)
            if peer is not None:
                await peer.aclose()
        finally:
            if self._peer is peer:  # not when a newer connection took over meanwhile
                self._live.clear()
                self._active.clear()
                self._session_options.clear()
                self._closing_ids.clear()
                self._unclosed.clear()
            try:
                if process is not None:
                    await process.close(grace=self.close_grace_s)
            finally:
                if self._peer is peer:
                    self._peer = None
                if self._process is process:
                    self._process = None

    async def cancel_turn(self, turn: ExternalTurnContext) -> None:
        peer = self._peer
        if peer is None or peer.closed or turn not in self._prompting or not turn.session_id:
            # No prompt sent yet: run_turn checks for the cancel right before it sends one.
            # A session/cancel now would reach the agent before the prompt and do nothing.
            return
        with contextlib.suppress(ExternalAgentConnectionError):
            await peer.notify("session/cancel", {"sessionId": turn.session_id})

    # -- one turn ------------------------------------------------------------------------------

    async def run_turn(self, turn: ExternalTurnContext) -> ExternalOutcome:
        peer = self._peer
        # Until session/prompt is written nothing of the task reaches the agent, so a broken
        # connection ends the turn with a known outcome (not "interrupted"); a repeated call
        # is the exception, since its earlier attempt may have prompted the session.
        reattach = self._reattaching(turn)
        if peer is None or peer.closed:
            raise ExternalAgentConnectionError(f"{self.name} is not connected.", ambiguous=reattach)
        cwd = turn.cwd
        if cwd is None:
            raise ExternalAgentConfigurationError(
                prompts.ACP_NO_WORKING_DIRECTORY.format(name=self.name)
            )
        prompt = self._prompt(turn)
        held: list[str] = []  # sessions this turn keeps from being closed as idle
        target = self._target(turn)
        if target is not None:
            self._hold(target, held)  # before any await: an idle close never takes it now
        setup = _Setup()
        servers = contextlib.AsyncExitStack()  # MCP servers that stop with this turn
        updates: TurnUpdates | None = None
        prompted = stopped = False
        try:
            try:
                ended = await self._set_up(turn, peer, cwd, servers, setup, held)
            except ExternalAgentConnectionError as error:
                if error.ambiguous == reattach:
                    raise
                raise _with_ambiguity(error, reattach) from error
            if ended is not None:
                return ended
            session_id = setup.session_id
            assert session_id is not None
            updates = TurnUpdates(
                turn,
                raw_events=self.raw_events,
                tool_name=self.tool_name or self._default_namer(),
                on_config=lambda value: self._session_options.__setitem__(session_id, value),
                chunks_are_blocks=self.chunks_are_blocks,
            )
            self._active[session_id] = updates
            await turn.begin(session_id, resumed=setup.resumed, backend=self.backend_name)
            if turn.cancel_requested:
                return _cancelled(turn)
            if turn.resumable:
                await turn.checkpoint(_PROMPT_CHECKPOINT, True)  # record first, act second
                if turn.cancel_requested:  # it came in during the journal write: do not prompt
                    return _cancelled(turn)
            if peer.closed:  # it ended while the turn was journaled: the prompt is not sent
                lost = peer.close_error or ExternalAgentConnectionError(
                    f"{self.name} is not connected."
                )
                raise _with_ambiguity(lost, reattach)
            # Nothing awaits from the checks above until session/prompt is written (or queued
            # first for the writer), so a cancel from now on is sent after the prompt.
            self._prompting.add(turn)
            prompted = True
            params = {"sessionId": session_id, "prompt": prompt}
            try:
                result = await peer.request("session/prompt", params)
            except JsonRpcError as error:
                if error.code == AUTH_REQUIRED:
                    raise ExternalAgentUnavailableError(
                        prompts.ACP_PROMPT_NEEDS_AUTH.format(
                            name=self.name, error=error_text(error)
                        )
                    ) from error
                return outcome_for_prompt_error(error, cancelling=turn.cancel_requested)
            finally:
                self._prompting.discard(turn)
            return outcome_for_stop_reason(result)
        except asyncio.CancelledError:
            stopped = True
            if prompted:  # it did not stop within cancel_grace_s: stop the whole process tree
                self._kill()
            raise
        finally:
            active = setup.session_id
            if updates is not None and active is not None and self._active.get(active) is updates:
                del self._active[active]
            try:
                if setup.release and setup.session_id is not None:
                    if stopped:  # its servers stop now: it is never continued directly again
                        self._live.discard(setup.session_id)
                    else:
                        await self._close_after_turn(turn, peer, setup.session_id)
            finally:
                self._turn_ended(peer, held)
                await self._stop_servers(servers)

    async def _set_up(
        self,
        turn: ExternalTurnContext,
        peer: JsonRpcPeer,
        cwd: str,
        servers: contextlib.AsyncExitStack,
        setup: _Setup,
        held: list[str],
    ) -> ExternalOutcome | None:
        """Open (or continue) the turn's session and configure it, filling in ``setup``.

        Returns ``None`` when the session is ready for the prompt, else the outcome that
        ends the turn first. A cancel ends the setup at once, wherever it waits: a request
        still waiting for its answer is given up (the peer tells the agent with
        ``$/cancel_request``) and the process is kept, since nothing of the task was sent.
        """
        if turn.cancel_requested:
            return _cancelled(turn)
        loop = asyncio.get_running_loop()
        try:
            async with asyncio.timeout(None) as scope:
                watcher = loop.create_task(_expire_when_cancelled(turn, scope))
                try:
                    await self._wait_for_closes()  # an earlier turn closes idle sessions
                    ended = await self._open_session(turn, peer, cwd, servers, setup)
                    if ended is not None:
                        return ended
                    assert setup.session_id is not None
                    self._hold(setup.session_id, held)
                    if setup.options is not None:
                        await self._configure(peer, setup.session_id, setup.options)
                finally:
                    watcher.cancel()
        except TimeoutError:
            if not scope.expired():
                raise
            return _cancelled(turn)
        return None

    # -- idle sessions ---------------------------------------------------------------------

    def _hold(self, session_id: str, held: list[str]) -> None:
        self._busy[session_id] = self._busy.get(session_id, 0) + 1
        held.append(session_id)

    def _turn_ended(self, peer: JsonRpcPeer, held: list[str]) -> None:
        """Release the turn's sessions, then close the idle ones beyond ``keep_sessions``.

        The close runs in the background, so the turn's result does not wait for it. The
        next turn waits for it (up to ``CLOSE_TIMEOUT_S``) before it opens a session; a
        turn that continues one of those sessions waits for its answer (up to
        ``request_timeout_s``); `disconnect` waits before stdin closes.
        """
        for session_id in held:
            count = self._busy.get(session_id, 0) - 1
            if count > 0:
                self._busy[session_id] = count
            else:
                self._busy.pop(session_id, None)
            if session_id in self._live:
                self._live.add(session_id)  # now the most recently used
        if peer is not self._peer or peer.closed or not self.connected or not self._can_close():
            return
        idle = [session_id for session_id in self._live if session_id not in self._busy]
        for session_id in idle[: max(0, len(idle) - self.keep_sessions)]:
            self._live.discard(session_id)  # a later turn in it resumes it
            self._session_options.pop(session_id, None)
            self._start_close(peer, session_id)

    async def _close_after_turn(
        self, turn: ExternalTurnContext, peer: JsonRpcPeer, session_id: str
    ) -> None:
        """Close a session whose MCP servers stop with its turn, before they stop.

        The next turn in it then resumes it with servers of its own. The wait ends after
        ``CLOSE_TIMEOUT_S``, or at once when the turn is being cancelled; the close goes
        on, and a later turn in the session waits for it. An agent that cannot close
        sessions keeps the session open, but it is not continued directly again (its
        servers are gone): a later turn resumes it, or says why it cannot.
        """
        if session_id not in self._live:
            return
        self._live.discard(session_id)
        self._session_options.pop(session_id, None)
        if peer is not self._peer or peer.closed or not self.connected or not self._can_close():
            return
        closing = self._start_close(peer, session_id)
        stopping = asyncio.ensure_future(turn.wait_cancelled())
        try:
            await asyncio.wait(
                {closing, stopping}, timeout=CLOSE_TIMEOUT_S, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            stopping.cancel()

    def _start_close(self, peer: JsonRpcPeer, session_id: str) -> asyncio.Task[None]:
        loop = asyncio.get_running_loop()
        task = loop.create_task(self._close_session(peer, session_id))
        self._closing.add(task)
        self._close_deadlines[task] = loop.time() + CLOSE_TIMEOUT_S
        self._closing_ids[session_id] = task

        def finished(_: asyncio.Task[None]) -> None:
            self._closing.discard(task)
            self._close_deadlines.pop(task, None)
            if self._closing_ids.get(session_id) is task:
                del self._closing_ids[session_id]

        task.add_done_callback(finished)
        return task

    async def _wait_for_closes(self, limit: float | None = None) -> None:
        """Wait for the running session closes: this class's each up to ``CLOSE_TIMEOUT_S``
        after it was sent, a subclass's until it ends; all of them at most ``limit``
        seconds."""
        loop = asyncio.get_running_loop()
        start = loop.time()
        others = {task for task in self._closing if task not in self._close_deadlines}
        if others:
            await asyncio.wait(others, timeout=limit)
        mine = {task: end for task, end in self._close_deadlines.items() if not task.done()}
        if not mine:
            return
        until = max(mine.values())
        if limit is not None:
            until = min(until, start + limit)
        if until > loop.time():
            await asyncio.wait(set(mine), timeout=until - loop.time())

    async def _close_session(self, peer: JsonRpcPeer, session_id: str) -> None:
        """``session/close`` of a session no turn uses, waiting for the answer however long
        it takes (an agent keeps the session until its close is done, and refuses to resume
        it before). A slow answer and a failure are logged; after a failure the session is
        remembered as possibly still open (`_unclosed`)."""
        params = {"sessionId": session_id}
        request = asyncio.get_running_loop().create_task(peer.request("session/close", params))
        try:
            await asyncio.wait({request}, timeout=CLOSE_TIMEOUT_S)
            if not request.done():
                logger.warning(
                    "%s did not answer session/close for %s within %g seconds; a turn that "
                    "continues the session waits for the answer",
                    self.name,
                    session_id,
                    CLOSE_TIMEOUT_S,
                )
            await request
        except JsonRpcError as error:
            logger.warning(
                "%s: closing the idle session %s failed: %s",
                self.name,
                session_id,
                error_text(error),
            )
            if peer is self._peer:
                self._unclosed.add(session_id)
        except ExternalAgentConnectionError as error:
            logger.debug("%s: session/close for %s not sent: %s", self.name, session_id, error)
        finally:
            if not request.done():  # this task was cancelled
                request.cancel()

    async def _stop_servers(self, servers: contextlib.AsyncExitStack) -> None:
        """Exit the async context managers of the turn's MCP servers (`MCPServers`)."""
        try:
            await servers.aclose()
        except Exception:
            logger.exception("%s: stopping the MCP servers of the turn failed", self.name)

    def _prompt(self, turn: ExternalTurnContext) -> list[dict[str, Any]]:
        caps = self.agent_capabilities.get("promptCapabilities")
        blocks = prompt_blocks(turn.task, caps if isinstance(caps, Mapping) else {})
        if self._reattaching(turn) and turn.checkpoints.get(_PROMPT_CHECKPOINT):
            # The first attempt sent the task and then the calling run crashed: the agent
            # may be halfway. It checks the workspace and finishes, with the task in view.
            text = _prompts.ACP_RESUME_TASK.format(
                resume=prompts.EXTERNAL_AGENT_RESUME, task=turn.prompt_text
            )
            blocks = [{"type": "text", "text": text}] + [b for b in blocks if b["type"] != "text"]
        return blocks

    @staticmethod
    def _reattaching(turn: ExternalTurnContext) -> bool:
        saved = turn.checkpoints.get(_SESSION_CHECKPOINT)
        return (
            turn.resumable
            and isinstance(saved, Mapping)
            and isinstance(saved.get("session_id"), str)
            and bool(saved["session_id"])
        )

    def _target(self, turn: ExternalTurnContext) -> str | None:
        """The session the turn continues: the caller's, or the one a crashed attempt
        journaled; ``None`` for a new session."""
        if turn.session is not None:
            return turn.session
        if self._reattaching(turn):
            return str(turn.checkpoints[_SESSION_CHECKPOINT]["session_id"])
        return None

    async def _servers_for(
        self, turn: ExternalTurnContext, servers: contextlib.AsyncExitStack
    ) -> tuple[list[dict[str, Any]], bool]:
        """The ``mcpServers`` of a session the turn opens or reopens, and whether they stop
        with the turn (a function gave an async context manager, entered on ``servers``)."""
        given: Any = self.mcp_servers
        per_turn = False
        if callable(given):
            given = given(turn)
            if isinstance(given, contextlib.AbstractAsyncContextManager):
                given = await servers.enter_async_context(given)
                per_turn = True
            elif inspect.isawaitable(given):
                given = await given
        try:
            if given is None:
                given = ()
            if isinstance(given, str | bytes | Mapping) or not isinstance(given, Iterable):
                raise ConfigurationError(
                    prompts.ACP_SERVERS_NOT_A_LIST.format(type=type(given).__name__)
                )
            wire = [_server_wire(server) for server in given]
            self._check_servers(wire)
        except ConfigurationError as error:
            raise ExternalAgentConfigurationError(
                prompts.ACP_SERVERS_FAILED.format(name=self.name, error=error)
            ) from error
        return wire, per_turn

    def _check_servers(self, wire: Sequence[Mapping[str, Any]]) -> None:
        mcp = self.agent_capabilities.get("mcpCapabilities")
        offered = mcp if isinstance(mcp, Mapping) else {}
        for server in wire:
            transport = server.get("type")
            if transport in ("http", "sse") and offered.get(transport) is not True:
                raise ConfigurationError(
                    prompts.ACP_SERVER_NOT_SUPPORTED.format(
                        server=server.get("name"), transport=transport
                    )
                )

    async def _open_session(
        self,
        turn: ExternalTurnContext,
        peer: JsonRpcPeer,
        cwd: str,
        servers: contextlib.AsyncExitStack,
        setup: _Setup,
    ) -> ExternalOutcome | None:
        """Open, reopen or continue the turn's session, filling in ``setup``; or return the
        outcome that ends the turn before it starts. The MCP servers are worked out only
        when they are sent."""
        reattach = self._reattaching(turn)
        target = self._target(turn)
        if target is None:
            mounted, setup.release = await self._servers_for(turn, servers)
            params = {"cwd": cwd, "mcpServers": mounted}
            try:
                result = await self._request(peer, "session/new", params)
            except JsonRpcError as error:
                return self._session_new_failed(error)
            session_id = result.get("sessionId")
            if not isinstance(session_id, str) or not session_id:
                raise ExternalAgentProtocolError(
                    f"{self.name} answered session/new without a sessionId: {result!r}",
                    reason=_prompts.ACP_BAD_ANSWER,
                )
            self._live.add(session_id)
            setup.session_id = session_id
            setup.options = self._remember_options(session_id, result)
            return None
        if target in self._live:  # opened by this connection: prompted with the servers it has
            setup.session_id, setup.resumed = target, True
            return None
        closing = self._closing_ids.get(target)
        if closing is not None:
            # qagents closes it after an earlier turn: an agent keeps a session until its
            # close is done, and refuses to resume it before
            await asyncio.wait({closing}, timeout=self.request_timeout_s)
            if not closing.done():
                return self._cannot_continue(target, _prompts.ACP_STILL_CLOSING, reattach)
        if self._can_resume():
            method = "session/resume"
        elif self._can_load():
            method = "session/load"  # replays the history as updates; no turn listens yet
        else:
            return self._cannot_continue(target, _prompts.ACP_NO_SESSION_RESUME, reattach)
        mounted, setup.release = await self._servers_for(turn, servers)
        params = {"sessionId": target, "cwd": cwd, "mcpServers": mounted}
        try:
            result = await self._request(peer, method, params)
        except JsonRpcError as error:
            if error.code == AUTH_REQUIRED:
                raise self._auth_error(error) from error
            if target in self._unclosed and not setup.release:
                # Its session/close failed, so the agent may still hold it open, and an agent
                # refuses to resume such a session (Devvi: "session is already active").
                logger.info(
                    "%s refused %s for %s, whose close failed: prompting it directly (%s)",
                    self.name,
                    method,
                    target,
                    error_text(error),
                )
                self._unclosed.discard(target)
                self._live.add(target)
                setup.session_id, setup.resumed = target, True
                return None
            return self._cannot_continue(target, error_text(error), reattach)
        self._unclosed.discard(target)
        self._live.add(target)
        setup.session_id, setup.resumed = target, True
        setup.options = self._remember_options(target, result)
        return None

    def _cannot_continue(self, session: str, reason: str, reattach: bool) -> ExternalOutcome:
        if reattach:  # the earlier attempt's outcome is unknown: never start it over
            return ExternalOutcome(
                status="interrupted",
                stop_reason=_prompts.ACP_NOT_REATTACHED.format(session=session),
                raw={"ambiguous": True, "error": reason},
            )
        return ExternalOutcome(
            status="error",
            stop_reason=_prompts.ACP_CANNOT_CONTINUE.format(session=session, reason=reason),
        )

    def _session_new_failed(self, error: JsonRpcError) -> ExternalOutcome:
        if error.code == AUTH_REQUIRED:
            raise self._auth_error(error)
        if error.code in (INVALID_PARAMS, INVALID_REQUEST, METHOD_NOT_FOUND):
            # an MCP server that did not start, a refused cwd: setup, not a failed turn
            raise ExternalAgentConfigurationError(
                prompts.ACP_SESSION_NEW_REFUSED.format(name=self.name, message=error.message)
            )
        return ExternalOutcome(status="error", stop_reason=error_text(error))

    def _auth_error(self, error: JsonRpcError) -> ExternalAgentConfigurationError:
        offered = (
            prompts.EXTERNAL_SETUP_LIST_SEPARATOR.join(
                str(method.get("id")) for method in self.auth_methods
            )
            or prompts.EXTERNAL_SETUP_LIST_NONE
        )
        return ExternalAgentConfigurationError(
            prompts.ACP_SESSION_NEEDS_AUTH.format(
                name=self.name, error=error_text(error), offered=offered
            )
        )

    def _remember_options(self, session_id: str, result: Mapping[str, Any]) -> list[Any] | None:
        options = result.get("configOptions")
        if isinstance(options, list):
            self._session_options[session_id] = options
            return options
        return [] if (self.model is not None or self.config_options) else None

    async def _request(self, peer: JsonRpcPeer, method: str, params: Any) -> Mapping[str, Any]:
        """A setup request, bounded by ``request_timeout_s``; its result must be an object."""
        try:
            result = await peer.request(method, params, timeout=self.request_timeout_s)
        except ExternalAgentTimeoutError as error:
            # The agent may still act on it later, but it carries nothing of the task: the
            # turn's outcome is known (run_turn marks a repeated call's turn ambiguous).
            raise ExternalAgentConnectionError(
                str(error),
                reason=_prompts.ACP_NO_ANSWER.format(method=method, seconds=self.request_timeout_s),
                ambiguous=False,
            ) from error
        if not isinstance(result, Mapping):
            raise ExternalAgentProtocolError(
                f"{self.name} answered {method} with {result!r}, which is not an object.",
                reason=_prompts.ACP_BAD_ANSWER,
            )
        return result

    async def _configure(self, peer: JsonRpcPeer, session_id: str, options: list[Any]) -> None:
        """Select ``model`` and ``config_options`` with ``session/set_config_option``."""
        wanted: list[tuple[str, Any, str]] = []
        if self.model is not None:
            option = next(
                (
                    item
                    for item in options
                    if isinstance(item, Mapping)
                    and (item.get("category") == "model" or item.get("id") == "model")
                ),
                None,
            )
            if option is None or not isinstance(option.get("id"), str):
                raise ExternalAgentConfigurationError(
                    prompts.ACP_NO_MODEL_OPTION.format(name=self.name, model=self.model)
                )
            wanted.append((option["id"], self.model, prompts.ACP_CONFIG_MODEL))
        wanted.extend(
            (key, value, prompts.ACP_CONFIG_OPTION.format(option=key))
            for key, value in self.config_options.items()
        )
        for config_id, value, what in wanted:
            option = next(
                (o for o in options if isinstance(o, Mapping) and o.get("id") == config_id),
                None,
            )
            if option is None:
                ids = prompts.EXTERNAL_SETUP_LIST_SEPARATOR.join(
                    str(o.get("id")) for o in options if isinstance(o, Mapping)
                )
                raise ExternalAgentConfigurationError(
                    prompts.ACP_NO_CONFIG_OPTION.format(
                        name=self.name,
                        option=config_id,
                        offered=ids or prompts.EXTERNAL_SETUP_LIST_NONE,
                    )
                )
            chosen = self._select(option, value, what)
            if option.get("currentValue") == chosen:
                continue
            params: dict[str, Any] = {"sessionId": session_id, "configId": config_id}
            if isinstance(chosen, bool):
                params["type"] = "boolean"
            params["value"] = chosen
            try:
                result = await self._request(peer, "session/set_config_option", params)
            except JsonRpcError as error:
                raise ExternalAgentConfigurationError(
                    prompts.ACP_CONFIG_FAILED.format(
                        name=self.name, what=what, value=value, message=error.message
                    )
                ) from error
            updated = result.get("configOptions")
            if isinstance(updated, list):
                options = updated
                self._session_options[session_id] = updated

    def _select(self, option: Mapping[str, Any], wanted: Any, what: str) -> Any:
        """The option value to send for ``wanted``: a value, a name, or a JSON-list value."""
        if isinstance(wanted, bool):
            return wanted
        values = _option_values(option)
        if not values:
            return wanted  # a free value: the agent checks it
        if isinstance(wanted, str):
            for value, _ in values:
                if value == wanted:
                    return value
            named = [value for value, name in values if name == wanted]
            if len(named) == 1:
                return named[0]
        else:
            for value, _ in values:
                with contextlib.suppress(ValueError, TypeError):
                    if isinstance(value, str) and json.loads(value) == list(wanted):
                        return value
        offered = prompts.EXTERNAL_SETUP_LIST_SEPARATOR.join(
            str(name or value) for value, name in values
        )
        raise ExternalAgentConfigurationError(
            prompts.ACP_CONFIG_NOT_OFFERED.format(
                name=self.name, what=what, wanted=wanted, offered=offered
            )
        )

    # -- the agent's messages ------------------------------------------------------------------

    def _on_update(self, params: Any) -> None:
        if not isinstance(params, Mapping) or not isinstance(params.get("sessionId"), str):
            logger.debug("%s sent a session/update without a sessionId", self.name)
            return
        session_id = params["sessionId"]
        updates = self._active.get(session_id)
        if updates is None:
            update = params.get("update")
            if (
                isinstance(update, Mapping)
                and update.get("sessionUpdate") == "config_option_update"
                and isinstance(update.get("configOptions"), list)
            ):
                self._session_options[session_id] = update["configOptions"]
            logger.debug(
                "%s: dropped a session/update for %s (no turn runs)", self.name, session_id
            )
            return
        updates.apply(params.get("update"))

    async def _on_permission(self, params: Any) -> dict[str, Any]:
        """Answer ``session/request_permission``. Any failure answers ``cancelled``."""
        try:
            return await self._decide(params)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "%s: deciding a permission request failed; answered cancelled", self.name
            )
            return CANCELLED

    async def _decide(self, params: Any) -> dict[str, Any]:
        session_id = params.get("sessionId") if isinstance(params, Mapping) else None
        updates = self._active.get(session_id) if isinstance(session_id, str) else None
        if updates is None:
            logger.warning(
                "%s asked for permission in session %r, where no turn runs; answered cancelled",
                self.name,
                session_id,
            )
            return CANCELLED
        turn = updates.turn
        call_id = updates.note_tool_call(params.get("toolCall"))
        if call_id is None:
            logger.warning(
                "%s asked for permission without a toolCallId; answered cancelled", self.name
            )
            turn.raw["malformed_permission_requests"] = (
                int(turn.raw.get("malformed_permission_requests", 0)) + 1
            )
            return CANCELLED
        offered = params.get("options")
        options = (
            [item for item in offered if isinstance(item, Mapping)]
            if isinstance(offered, list)
            else []
        )
        decision = asyncio.create_task(
            turn.ask_permission(updates.name_of(call_id), tool_call_id=call_id, options=options)
        )
        stopping = asyncio.create_task(turn.wait_cancelled())
        try:
            await asyncio.wait({decision, stopping}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            stopping.cancel()
            if not decision.done():  # ACP: once the turn is cancelled, answer every request
                decision.cancel()  # "cancelled" at once (the bridge records "cancelling")
            await asyncio.gather(decision, stopping, return_exceptions=True)
        if turn.cancel_requested:
            # ACP: once the client cancels, it answers every pending request "cancelled",
            # also one whose decision came in at the same moment (an approval as the
            # timeout fires). Nothing awaits from here until the answer is written, so an
            # allow is never written after session/cancel.
            _record_cancelling(turn, call_id, decision)
            return CANCELLED
        if decision.cancelled() or decision.exception() is not None:
            return CANCELLED
        answer = decision.result()
        if answer.option_id is None:
            return CANCELLED
        return {"outcome": {"outcome": "selected", "optionId": answer.option_id}}

    def _on_other_notification(self, method: str, params: Any) -> None:
        logger.debug("%s sent the notification %r, which qagents does not use", self.name, method)

    def _on_peer_closed(self, peer: JsonRpcPeer, error: ExternalAgentConnectionError) -> None:
        if peer is self._peer:
            self.mark_disconnected()  # the next turn starts a new process

    # -- setup helpers ------------------------------------------------------------------------

    async def _handshake(self, peer: JsonRpcPeer, process: ManagedProcess) -> None:
        from ... import __version__

        params = {
            "protocolVersion": PROTOCOL_VERSION,
            "clientCapabilities": CLIENT_CAPABILITIES,
            "clientInfo": {"name": "qagents", "version": __version__},
        }
        try:
            result = await peer.request("initialize", params, timeout=self.handshake_timeout_s)
        except JsonRpcError as error:
            raise self._setup_error(
                prompts.ACP_HANDSHAKE_REFUSED.format(message=error.message), process
            ) from error
        except ExternalAgentTimeoutError as error:
            raise self._setup_error(
                prompts.ACP_HANDSHAKE_TIMEOUT.format(seconds=self.handshake_timeout_s), process
            ) from error
        except ExternalAgentConnectionError as error:
            raise self._setup_error(
                prompts.ACP_HANDSHAKE_STOPPED.format(reason=error.reason), process, error
            ) from error
        if not isinstance(result, Mapping):
            raise self._setup_error(prompts.ACP_HANDSHAKE_BAD_ANSWER.format(answer=result), process)
        version = result.get("protocolVersion")
        if type(version) is not int or version != PROTOCOL_VERSION:
            raise self._setup_error(
                prompts.ACP_PROTOCOL_VERSION.format(version=version, supported=PROTOCOL_VERSION),
                process,
            )
        self._info = dict(result)
        await self._authenticate(peer, process)
        if not callable(self.mcp_servers):
            try:
                self._check_servers([_server_wire(server) for server in self.mcp_servers])
            except ConfigurationError as error:
                raise self._setup_error(str(error).rstrip("."), process) from error

    async def _authenticate(self, peer: JsonRpcPeer, process: ManagedProcess) -> None:
        methods = self.auth_methods
        if not methods:
            return  # the agent asks for nothing: never authenticate
        if self.auth_method is None:
            logger.info(
                "%s offers authentication (%s); auth_method is not set, so qagents does not log in",
                self.name,
                ", ".join(str(method.get("id")) for method in methods),
            )
            return
        chosen = next((method for method in methods if method.get("id") == self.auth_method), None)
        if chosen is None:
            offered = prompts.EXTERNAL_SETUP_LIST_SEPARATOR.join(
                str(method.get("id")) for method in methods
            )
            raise self._setup_error(
                prompts.ACP_AUTH_METHOD_UNKNOWN.format(method=self.auth_method, offered=offered),
                process,
            )
        if chosen.get("type") == "terminal":
            raise self._setup_error(
                prompts.ACP_AUTH_METHOD_TERMINAL.format(method=self.auth_method), process
            )
        try:
            await peer.request(
                "authenticate", {"methodId": self.auth_method}, timeout=self.handshake_timeout_s
            )
        except JsonRpcError as error:
            raise self._setup_error(
                prompts.ACP_AUTH_FAILED.format(method=self.auth_method, message=error.message),
                process,
            ) from error
        except (ExternalAgentTimeoutError, ExternalAgentConnectionError) as error:
            raise self._setup_error(
                prompts.ACP_AUTH_UNFINISHED.format(method=self.auth_method), process, error
            ) from error

    def _setup_error(
        self, summary: str, process: ManagedProcess, cause: BaseException | None = None
    ) -> ExternalAgentConfigurationError:
        """A setup failure: a short first line (`prompts.ACP_SETUP_FAILED`, which reaches the
        calling model through ``as_tool()``), details for the developer after it."""
        tail = process.stderr_tail() or None
        lines = [prompts.ACP_SETUP_FAILED.format(name=self.name, summary=summary)]
        if cause is not None:
            lines.append(str(cause))
        elif tail:
            lines.append(f"stderr: {tail}")
        return ExternalAgentConfigurationError("\n".join(lines), stderr_tail=tail)

    def _kill(self) -> None:
        process = self._process
        if process is not None:
            process.kill()
        self.mark_disconnected()

    def _release_later(self) -> None:
        """Close the connection in the background (``connect`` was cancelled)."""
        task = asyncio.get_running_loop().create_task(self.disconnect())
        self._disconnecting.add(task)
        task.add_done_callback(self._disconnecting.discard)


def _cancelled(turn: ExternalTurnContext) -> ExternalOutcome:
    """The outcome of a turn that was cancelled before its prompt was sent."""
    return ExternalOutcome(status="cancelled", stop_reason=turn.cancel_reason)


async def _expire_when_cancelled(turn: ExternalTurnContext, scope: asyncio.Timeout) -> None:
    """Expire ``scope`` once ``turn`` is cancelled: what it guards is then cancelled, and
    the ``async with`` raises `TimeoutError` (``scope.expired()`` is true)."""
    await turn.wait_cancelled()
    with contextlib.suppress(RuntimeError):  # the guarded block has just ended
        scope.reschedule(asyncio.get_running_loop().time())


def _with_ambiguity(
    error: ExternalAgentConnectionError, ambiguous: bool
) -> ExternalAgentConnectionError:
    """A copy of ``error`` (same type, message, reason, exit code and stderr tail) whose
    ``ambiguous`` says whether the agent may have received the turn's task."""
    copy = type(error).__new__(type(error))
    ExternalAgentConnectionError.__init__(
        copy,
        str(error),
        reason=error.reason,
        returncode=error.returncode,
        stderr_tail=error.stderr_tail,
        ambiguous=ambiguous,
    )
    return copy


def _record_cancelling(
    turn: ExternalTurnContext, call_id: str, decision: asyncio.Task[Any]
) -> None:
    """Record a decision that was overtaken by the turn's cancel as what was answered.

    The bridge recorded the decision (an allow, a deny) as it was made; the agent was
    answered "cancelled", so the turn's record says ``cancel`` by ``cancelling``. A
    decision that already answers cancelled (deferred to a pause, timed out) keeps its own
    record.
    """
    if decision.cancelled() or decision.exception() is not None:
        return  # the bridge recorded it as cancelling (or as an error) itself
    if decision.result().outcome == "cancel":
        return
    for record in reversed(turn.bridge.records):
        if record.tool_call_id == call_id:
            record.outcome = "cancel"
            record.decided_by = "cancelling"
            record.reason = prompts.EXTERNAL_PERMISSION_CANCELLING
            return


def _option_values(option: Mapping[str, Any]) -> list[tuple[Any, Any]]:
    """``(value, name)`` of a select option's choices, flat or grouped."""
    found: list[tuple[Any, Any]] = []
    entries = option.get("options")
    if not isinstance(entries, list):
        return found
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        if isinstance(entry.get("options"), list):  # a group of choices
            for inner in entry["options"]:
                if isinstance(inner, Mapping) and "value" in inner:
                    found.append((inner["value"], inner.get("name")))
        elif "value" in entry:
            found.append((entry["value"], entry.get("name")))
    return found


__all__ = [
    "CLIENT_CAPABILITIES",
    "ACPAgent",
    "MCPServer",
    "MCPServerHTTP",
    "MCPServerStdio",
    "MCPServers",
]
