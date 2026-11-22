"""Devvi's ACP transport (``devvi --profile acp``), on the generic ACP client.

`DevviAgent(transport="acp")` (the default) is an `ACPAgent` with Devvi's launcher, patch
and environment rules:

* The command is ``[node, bin.js, --profile, acp, --patch, <file>, ...]`` and the
  environment is the launcher's explicit allowlist (`qagents.integrations.devvi.launcher`).
* **One Devvi process per workspace**, started in that workspace. Devvi roots each
  session's file sandbox at the session's ``cwd`` from ``session/new`` (measured, design
  4.2 #3; ``workspaceRoot: process.cwd()`` in the base profile is only the default), so
  this is a choice, not a need: each workspace gets its own process environment and model
  route, and a crash or a forced kill touches one workspace only. A turn with another
  ``cwd`` gets its own process. Each process notices its own end (an exit, a kill after a
  turn that would not stop) and is started again by the next turn in its workspace; the
  sessions it served are continued with ``session/resume``.
* **Tool names** come from the tool call's ``title`` (`title_tool_name`): Devvi puts its
  tool name there (``bash``, ``pwsh``, ``mcp__qagents__lookup``) and sends ``kind:
  "other"`` for every tool. It is passed explicitly, so permission rules such as
  ``coder.bash`` keep matching if a Devvi release renames its ``agentInfo``.
* **The model** is chosen through the session's ``configOptions``: ``model=(provider,
  model id)`` is matched against Devvi's ``["<provider>", "<model>"]`` values and set with
  ``session/set_config_option`` when it is not the current one. A model the session does
  not list is refused with the list it offers (design 4.2 #9). ``reasoning_effort=`` sets
  Devvi's ``reasoning_effort`` option the same way.
* **Crash re-attach** (``resumable=True``, the default): Devvi persists a session as soon
  as ``session/new`` answers and supports ``session/resume``, so a call repeated after a
  crash of the calling run continues the session the first attempt journaled (see
  `ACPAgent`), and the tool is idempotent.
* **qagents tools** (``expose_tools``, `qagents.integrations.devvi.expose`): importable
  tools are mounted as stdio MCP servers in every ``session/new`` and ``session/resume``.
  In-process tools get an MCP Streamable HTTP server per turn, bound to the calling run;
  because Devvi mounts MCP servers only when a session opens, the session is closed
  (``session/close``) after such a turn, and the next turn resumes it with that turn's
  server. A cancel that arrives during that close (the turn already has its outcome) lets
  the close finish in the background: the finished turn stays finished, Devvi is not
  killed, and the next turn waits for the close before it opens a session.
* ``ExternalResult.backend`` is ``devvi-acp@<devvi --version>``: Devvi's ``agentInfo``
  always says ``0.0.1``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ... import prompts as acp_prompts
from ...errors import ConfigurationError
from ...external import (
    ExternalAgentConfigurationError,
    ExternalAgentConnectionError,
    ExternalAgentError,
    ExternalAgentProtocolError,
    ExternalCapabilities,
    ExternalOutcome,
    ExternalTurnContext,
)
from .._jsonrpc import JsonRpcError
from .._process import ManagedProcess
from ..acp import ACPAgent, title_tool_name
from ._aio import settle
from .expose import ExposedTools
from .launcher import DevviLaunch

logger = logging.getLogger("qagents.integrations.devvi")

PROFILE = "acp"
"""The Devvi profile of this transport (``packages/bundle/acp-app``)."""

CLOSE_TIMEOUT_S = 10.0
"""Seconds ``session/close`` may take after a turn that mounted an in-process tool server."""


class DevviACPAgent(ACPAgent):
    """One ``devvi --profile acp`` process, in one workspace: an `ACPAgent` built from a launch.

    `DevviAgent` makes these itself, one per workspace. Use it directly only to talk ACP to
    Devvi without the `DevviAgent` options (the ``doctor`` does).

    Args:
        launch: The prepared launch (command, environment, patch files, version).
        cwd: The workspace: the process's working directory and its sessions' ``cwd``.
        name: The agent's name, for logs and errors.
        model: ``(provider route, model id)`` to select, or ``None`` to keep Devvi's.
        reasoning_effort: Devvi's ``reasoning_effort`` option value, or ``None``.
        exposed: qagents tools to mount in every session (see `ExposedTools`).
        stdio_servers: ACP ``mcpServers`` entries of the stdio tool servers.
        resumable: Re-attach a call repeated after a crash to its journaled session.
        raw_events: Forward ``session/update`` kinds that have no mapping.
        handshake_timeout_s: Seconds for ``initialize`` (Devvi loads its plugins first).
        request_timeout_s: Seconds for ``session/new``, ``session/resume`` and
            ``session/set_config_option`` (MCP servers connect during them).
        close_grace_s: Seconds Devvi gets to exit after its stdin closes.
    """

    backend = "devvi-acp"

    chunks_are_blocks = True
    """Devvi sends one ``agent_message_chunk`` per committed block, with the message's id on
    every block of it (``packages/acp/acp/src/updates.ts``), so each chunk is a block."""

    grant_ignored_arguments = frozenset({"description", "justification"})
    """See `DevviAgent.grant_ignored_arguments`."""

    def __init__(
        self,
        launch: DevviLaunch,
        *,
        cwd: str,
        name: str = "devvi",
        model: tuple[str, str] | None = None,
        reasoning_effort: str | None = None,
        exposed: ExposedTools | None = None,
        stdio_servers: Sequence[Mapping[str, Any]] = (),
        resumable: bool = True,
        raw_events: bool = False,
        handshake_timeout_s: float = 120.0,
        request_timeout_s: float = 120.0,
        close_grace_s: float = 5.0,
    ) -> None:
        options = {"reasoning_effort": reasoning_effort} if reasoning_effort is not None else None
        super().__init__(
            launch.argv(PROFILE),
            name=name,
            env=launch.env,
            cwd=cwd,
            workspace=cwd,
            model=model,
            config_options=options,
            resumable=resumable,
            raw_events=raw_events,
            tool_name=title_tool_name,  # Devvi's titles are its tool names (kind is "other")
            handshake_timeout_s=handshake_timeout_s,
            request_timeout_s=request_timeout_s,
            close_grace_s=close_grace_s,
            track_changes=False,  # DevviAgent tracks changes around the whole turn
        )
        self.launch = launch
        self.exposed = exposed
        self._stdio = [dict(entry) for entry in stdio_servers]
        self._tool_servers: dict[int, Any] = {}
        if self._stdio or (exposed is not None and exposed.in_process):
            self.mcp_servers = self._servers_of

    @property
    def backend_name(self) -> str:
        """``devvi-acp@<devvi --version>``."""
        version = self.launch.version
        return f"{self.backend}@{version}" if version else self.backend

    @property
    def process(self) -> ManagedProcess | None:
        """The running Devvi process, if any."""
        return self._process

    @property
    def serves_tools_in_process(self) -> bool:
        return self.exposed is not None and bool(self.exposed.in_process)

    # -- lifecycle -----------------------------------------------------------------------

    async def connect(self) -> None:
        await super().connect()
        if not self.serves_tools_in_process:
            return
        mcp = self.agent_capabilities.get("mcpCapabilities")
        if isinstance(mcp, Mapping) and mcp.get("http") is True:
            return
        process = self._process
        tail = process.stderr_tail() if process is not None else ""
        await self.disconnect()
        raise ExternalAgentConfigurationError(
            f"{acp_prompts.DEVVI_EXPOSE_NEEDS_HTTP.format(name=self.name)}\n"
            "Expose importable tools ('package.module:attr') instead: they run over stdio.",
            stderr_tail=tail or None,
        )

    async def open_session(self, cwd: str | None = None) -> tuple[str, list[Any]]:
        """Open a session without a prompt: ``(session id, configOptions)``. For ``doctor``.

        Raises:
            JsonRpcError: Devvi refused ``session/new``.
            ExternalAgentConnectionError: The connection broke or Devvi did not answer.
        """
        await self.ensure_connected()
        peer = self._peer
        if peer is None or peer.closed:  # nothing was sent: not an unknown outcome
            raise ExternalAgentConnectionError(f"{self.name} is not connected.", ambiguous=False)
        params = {"cwd": cwd or self.workspace, "mcpServers": []}
        result = await self._request(peer, "session/new", params)
        session_id = result.get("sessionId")
        if not isinstance(session_id, str) or not session_id:
            raise ExternalAgentProtocolError(
                f"{self.name} answered session/new without a sessionId: {result!r}",
                reason=acp_prompts.ACP_BAD_ANSWER,
            )
        self._live.add(session_id)
        options = result.get("configOptions")
        return session_id, list(options) if isinstance(options, list) else []

    # -- a turn ----------------------------------------------------------------------------

    async def run_turn(self, turn: ExternalTurnContext) -> ExternalOutcome:
        if not self.serves_tools_in_process:
            return await super().run_turn(turn)
        assert self.exposed is not None
        server = self.exposed.http_server(turn)
        try:
            await server.start()
        except ConfigurationError as error:
            raise ExternalAgentConfigurationError(
                acp_prompts.DEVVI_EXPOSE_FAILED.format(name=self.name, error=error)
            ) from error
        key = id(turn)
        self._tool_servers[key] = server
        try:
            outcome = await super().run_turn(turn)
        except BaseException:
            self._tool_servers.pop(key, None)
            # The turn is abandoned: refuse Devvi's calls from now on, before the session
            # close is awaited (Devvi may still call tools until it closes the session).
            server.refuse_new_calls()
            try:
                await self._release_session(turn, kill_on_cancel=True)
            finally:
                await asyncio.shield(server.aclose())
            raise
        self._tool_servers.pop(key, None)
        # The turn has its outcome: a cancel from here on must neither discard it nor kill a
        # Devvi that is only closing the session. The next turn waits for the close
        # (ACPAgent waits for `_closing` before it opens a session).
        await settle(self._after_turn(turn, server), self._closing)
        return outcome

    async def _after_turn(self, turn: ExternalTurnContext, server: Any) -> None:
        try:
            await self._release_session(turn)
        finally:
            await asyncio.shield(server.aclose())

    def _servers_of(self, turn: ExternalTurnContext) -> list[dict[str, Any]]:
        servers = [dict(entry) for entry in self._stdio]
        server = self._tool_servers.get(id(turn))
        if server is not None:
            servers.append(server.acp_config())
        return servers

    async def _release_session(
        self, turn: ExternalTurnContext, *, kill_on_cancel: bool = False
    ) -> None:
        """Close the turn's session, so that the next turn mounts its own tool server.

        If the close fails, the process is restarted by the next turn instead (Devvi would
        refuse to resume a session that is still open). A cancel during the close kills
        Devvi only with ``kill_on_cancel`` (a turn that was abandoned); otherwise the
        connection is only marked lost, and the next turn starts Devvi again.
        """
        session_id = turn.session_id
        if not session_id or session_id not in self._live:
            return
        self._live.discard(session_id)
        peer = self._peer
        if peer is None or peer.closed or not self.connected:
            return
        try:
            await peer.request("session/close", {"sessionId": session_id}, timeout=CLOSE_TIMEOUT_S)
        except asyncio.CancelledError:
            if kill_on_cancel:
                self._kill()
            else:
                self.mark_disconnected()
            raise
        except (JsonRpcError, ExternalAgentError) as error:
            logger.warning(
                "%s: closing session %s after the turn failed (%s); Devvi restarts next turn",
                self.name,
                session_id,
                error,
            )
            self.mark_disconnected()


# --- the transport of DevviAgent ---------------------------------------------------------------


@dataclass(frozen=True)
class AcpSettings:
    """What the ACP transport asks of Devvi and how long it waits."""

    model: tuple[str, str] | None = None
    reasoning_effort: str | None = None
    raw_events: bool = False
    resumable: bool = True
    handshake_timeout_s: float = 120.0
    request_timeout_s: float = 120.0
    close_grace_s: float = 5.0


class AcpTransport:
    """`DevviAgent`'s ACP transport: one `DevviACPAgent` per workspace.

    Args:
        settings: Model, timeouts, re-attach.
        launch: Returns the prepared `DevviLaunch` (cached by the agent).
        workspace: The agent's workspace: its process starts on connect.
        name: The agent's name, for logs and errors.
        exposed: qagents tools to mount in every session.
    """

    manages_connections = True
    """Each workspace's connection notices its own loss and starts again on the next turn,
    so a broken one never takes the processes of other workspaces down with it."""

    def __init__(
        self,
        settings: AcpSettings,
        *,
        launch: Callable[[], Awaitable[DevviLaunch]],
        workspace: str | None,
        name: str = "devvi",
        exposed: ExposedTools | None = None,
    ) -> None:
        self.settings = settings
        self.resumable_turns = settings.resumable
        self._launch = launch
        self._workspace = workspace
        self.name = name
        self.exposed = exposed
        self._agents: dict[str, DevviACPAgent] = {}
        self._stdio: list[dict[str, Any]] | None = None

    @property
    def agents(self) -> dict[str, DevviACPAgent]:
        """The connection of each workspace, keyed by its normalized path."""
        return dict(self._agents)

    async def connect(self) -> None:
        """Prepare the launch and start Devvi in the agent's workspace (the handshake)."""
        await self._launch()
        if self._workspace is not None:
            agent = await self.agent_for(self._workspace)
            await agent.ensure_connected()

    async def disconnect(self) -> None:
        """Stop every Devvi process of this agent."""
        agents, self._agents = list(self._agents.values()), {}
        await asyncio.gather(*(agent.aclose() for agent in agents), return_exceptions=True)

    async def capabilities(self) -> ExternalCapabilities:
        if self._workspace is None:  # pragma: no cover - DevviAgent requires a workspace
            raise ExternalAgentConfigurationError(f"{self.name} has no workspace.")
        agent = await self.agent_for(self._workspace)
        return await agent.capabilities()

    async def agent_for(self, cwd: str) -> DevviACPAgent:
        """The connection for ``cwd`` (made, not started, when it is new)."""
        key, directory = _workspace(cwd)
        agent = self._agents.get(key)
        if agent is not None:
            return agent
        launch = await self._launch()
        if self._stdio is None:
            self._stdio = self.exposed.acp_entries(launch.home) if self.exposed else []
        settings = self.settings
        agent = DevviACPAgent(
            launch,
            cwd=directory,
            name=self.name,
            model=settings.model,
            reasoning_effort=settings.reasoning_effort,
            exposed=self.exposed,
            stdio_servers=self._stdio,
            resumable=settings.resumable,
            raw_events=settings.raw_events,
            handshake_timeout_s=settings.handshake_timeout_s,
            request_timeout_s=settings.request_timeout_s,
            close_grace_s=settings.close_grace_s,
        )
        return self._agents.setdefault(key, agent)

    async def run_turn(self, turn: ExternalTurnContext) -> ExternalOutcome:
        cwd = turn.cwd or self._workspace
        if cwd is None:  # pragma: no cover - DevviAgent requires a workspace
            raise ExternalAgentConfigurationError(f"{self.name}: a turn needs a workspace.")
        agent = await self.agent_for(cwd)
        try:
            await agent.ensure_connected()
            outcome = await agent.run_turn(turn)
        except (ExternalAgentConnectionError, asyncio.CancelledError):
            agent.mark_disconnected()  # the next turn here starts a new process
            raise
        reason = outcome.raw.get("acp_stop_reason")
        if isinstance(reason, str):  # the same key as the SDK transport's
            outcome.raw.setdefault("agent_stop_reason", reason)
        return outcome

    async def cancel_turn(self, turn: ExternalTurnContext) -> None:
        cwd = turn.cwd or self._workspace
        if cwd is None:
            return
        agent = self._agents.get(_workspace(cwd)[0])
        if agent is not None:
            with contextlib.suppress(ExternalAgentConnectionError):
                await agent.cancel_turn(turn)


def _workspace(cwd: str) -> tuple[str, str]:
    """``(key, absolute path)`` of a workspace; the key ignores case where the OS does."""
    directory = os.path.abspath(cwd)
    return os.path.normcase(directory), directory


__all__ = ["PROFILE", "AcpSettings", "AcpTransport", "DevviACPAgent"]
