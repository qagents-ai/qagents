"""Devvi (devvi.sh, a fork of DeepSeek Harness) as an external coding agent of qagents.

    from qagents import Agent, Limits, Permissions
    from qagents.integrations.devvi import DevviAgent

    coder = DevviAgent(
        workspace="C:/work/repo",
        model=("deepseek-official", "deepseek-v4-flash"),
        patches=[{"id": "tool-web", "disabled": True}],  # Devvi never asks before web_fetch
        allow_unmetered=True,   # ACP reports no token usage: the lead's cost cap counts the lead
    )
    lead = Agent(
        model,
        tools=[coder.as_tool(name="coder", timeout=1800)],
        permissions=Permissions(ask=["coder.bash", "coder.pwsh"], on_ask=reviewer),
        limits=Limits(max_cost_usd=2.0),
    )
    async with coder:
        result = await lead.run("Make tests/test_parser.py pass.")

Devvi is an agent, not a model, and it has no MCP server: qagents drives it over a
control protocol. Two transports:

* ``transport="acp"`` (the default): Devvi's ACP v1 agent (``devvi --profile acp``),
  through `qagents.integrations.acp.ACPAgent` (`qagents.integrations.devvi.acp`): one
  Devvi process per workspace, ``session/cancel`` for a cancel, the model chosen from the
  session's options, crash re-attach through ``session/resume``, and qagents tools for
  Devvi's model (``expose_tools``). ACP reports no token counts.
* ``transport="sdk"``: Devvi's SDK JSON-RPC (``devvi --profile sdk``,
  `qagents.integrations.devvi.sdk`): token usage per model request (so cost budgets
  work, with ``pricing=``), session ids chosen by qagents, and crash re-attach (the tool
  is idempotent: a call repeated after a crash finds its turn in Devvi's session log).
  There is no per-turn cancel: a cancel stops the Devvi process, one per call by default.

Every permission request of Devvi goes through qagents' `Permissions` on names
``"<tool name>.<Devvi tool>"`` (``coder.bash``; ``coder.pwsh`` on Windows), see
`qagents.external`. Devvi asks only when its own policy says so: in ``workspace-write`` that
is an escalation, a command or file write that asks to leave its sandbox. Reads, edits,
plain commands in the workspace and web fetches run without a request, so rules cannot stop
them; remove a tool with a patch row instead (``{"id": "tool-web", "disabled": True}``
removes ``web_fetch`` and ``web_search``). Devvi itself is
started with an explicit environment, its own home (``~/.qagents/devvi-home``), the
upload of session logs to DeepSeek off, and uncompressed session logs
(`qagents.integrations.devvi.launcher`, `qagents.integrations.devvi.patches`).

qagents tools for Devvi's model (``expose_tools``, `qagents.integrations.devvi.expose`):
in-process tools are served over MCP Streamable HTTP on ``127.0.0.1``, bound to the
calling run (its rules decide every call and its journal records it); importable tools
(``"package.module:attr"``) are served over stdio by a process Devvi starts.

Check a setup without a model call: ``python -m qagents.integrations.devvi doctor``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from ... import prompts
from ... import prompts as base_prompts
from ...errors import ConfigurationError
from ...external import (
    ExternalAgent,
    ExternalCapabilities,
    ExternalOutcome,
    ExternalTurnContext,
    _tool_safe,
)
from ...tools import Tool
from .acp import AcpSettings, AcpTransport, DevviACPAgent
from .events import SessionEventMapper, TokenPricing, turn_end_outcome, usage_from_devvi
from .expose import ExposedTools
from .launcher import (
    DEFAULT_PASSTHROUGH,
    PERMISSION_MODES,
    DevviCommand,
    DevviLaunch,
    PermissionMode,
    prepare_launch,
    resolve_home,
)
from .logs import (
    DevviLogError,
    SessionLog,
    find_session_log,
    list_sessions,
    read_session_log,
    sessions_dir,
)
from .patches import (
    PatchRow,
    check_rows,
    deepseek_models_row,
    devvi_patch,
    mcp_http_row,
    mcp_stdio_row,
    model_route_row,
    model_routes_row,
    openai_route,
    sessions_root,
)
from .sdk import SdkSettings, SdkTransport

__all__ = [
    "DEFAULT_MODEL",
    "DevviACPAgent",
    "DevviAgent",
    "DevviCommand",
    "DevviLaunch",
    "DevviLogError",
    "ExposedTools",
    "PatchRow",
    "PermissionMode",
    "SessionEventMapper",
    "SessionLog",
    "TokenPricing",
    "Transport",
    "deepseek_models_row",
    "devvi_patch",
    "find_session_log",
    "list_sessions",
    "mcp_http_row",
    "mcp_stdio_row",
    "model_route_row",
    "model_routes_row",
    "openai_route",
    "read_session_log",
    "sessions_dir",
    "turn_end_outcome",
    "usage_from_devvi",
]

Transport = Literal["acp", "sdk"]

DEFAULT_MODEL = ("deepseek-official", "deepseek-v4-flash")
"""Devvi's own default route (``python/sdk`` ``DeepSeekHarnessConfig``): what the SDK
transport sends in ``initialize`` when no ``model`` is given."""


class _Transport(Protocol):
    """What `DevviAgent` delegates to: one implementation per transport
    (`AcpTransport`, `SdkTransport`)."""

    resumable_turns: bool
    manages_connections: bool
    """``True``: the transport recovers each process on its own, so a lost connection
    never makes the agent stop all of them (see `DevviAgent.mark_disconnected`)."""

    async def connect(self) -> None: ...

    async def disconnect(self) -> None: ...

    async def capabilities(self) -> ExternalCapabilities: ...

    async def run_turn(self, turn: ExternalTurnContext) -> ExternalOutcome: ...

    async def cancel_turn(self, turn: ExternalTurnContext) -> None: ...


class DevviAgent(ExternalAgent):
    """Devvi as an external agent: use it as a tool, a workflow step, or run it directly.

    Args:
        workspace: The directory Devvi works in (its session cwd and its process cwd,
            which is also the root of Devvi's file sandbox). Required.
        transport: ``"acp"`` (default) or ``"sdk"``.
        name: Agent name; the prefix of permission names for a direct `stream` (``as_tool``
            uses the tool name).
        description: Tool description; by default `DEVVI_DESCRIPTION`.
        model: ``(provider route, model id)``, for example ``("deepseek-official",
            "deepseek-v4-flash")``, or a route declared with `model_route_row`. ACP: chosen
            from the session's model options, and refused when the session does not list
            it; ``None`` keeps Devvi's current model. SDK: sent in ``initialize``; ``None``
            is `DEFAULT_MODEL`.
        reasoning_effort: Devvi's reasoning effort for the route (``"high"``, ...). ACP:
            the session's ``reasoning_effort`` option, refused when not offered.
        max_tokens: Output-token cap per model request (SDK only; on ACP declare it on
            the model route with a patch).
        permission_mode: ``DSH_PERMISSION_MODE``: ``"read-only"``, ``"workspace-write"``
            (default) or ``"danger-full-access"`` (Devvi then asks nothing: only in a
            disposable workspace, with ``as_tool(requires_approval=True)``).
        expose_tools: qagents tools for Devvi's model, which sees them as
            ``mcp__<server>__<tool>`` (`qagents.integrations.devvi.expose`). In-process
            tools (``Tool`` objects, functions, toolkits; ACP only) are served over MCP
            Streamable HTTP on ``127.0.0.1``, bound to the calling run, on the server
            ``expose_name``. ``"package.module:attr"`` strings share a stdio server named
            ``<expose_name>_stdio``; a `ToolServerSpec` is served over stdio as given. Needs
            ``qagents[mcp]``.
        expose_name: Name of the in-process tool server (default ``qagents``).
        resumable: ACP: a call repeated after a crash of the calling run continues the
            Devvi session its first attempt journaled (default; the tool is then
            idempotent). ``False``: the calling model is told the outcome is unknown. The
            SDK transport always re-attaches.
        home: Devvi home; default ``QAGENTS_DEVVI_HOME``, else ``~/.qagents/devvi-home``.
        devvi_bin: The Devvi CLI; default ``QAGENTS_DEVVI_BIN``, else ``devvi`` on PATH. A
            ``.js`` file runs with Node.
        node: Node; default ``QAGENTS_NODE``, else ``node`` on PATH (>= 24.9.0).
        command: An exact command that starts the CLI, instead of ``devvi_bin``/``node``.
        env: Variables set for Devvi explicitly (kept in memory, never in a file).
        env_passthrough: Names copied from this process's environment; default exactly
            ``DEEPSEEK_API_KEY`` and ``DEEPSEEK_BASE_URL``. Nothing else is inherited
            besides the operating-system basics.
        patches: Extra profile patch rows (dicts, written after the defaults) or patch
            files (paths, passed after them). Rows that hold a credential are refused.
        share_session_log: Let Devvi upload its session log with DeepSeek API requests
            (``session-log-deepseek``); off by default.
        pricing: Token prices for ``cost_usd`` (SDK only: ACP reports no token counts).
            Without them a calling run with ``Limits.max_cost_usd`` refuses to start the
            agent.
        process_per_call: SDK: one Devvi process per call (default), or one per workspace
            kept between calls. ACP always keeps one process per workspace.
        wait_for_subagents: SDK: after the turn, wait for Devvi's subagents
            (``session/wait``).
        raw_events: Forward Devvi events with no qagents counterpart (SDK ``raw``, ACP
            ``unknown``).
        initialize_timeout_s: Seconds for Devvi to load and answer ``initialize``.
        shutdown_timeout_s: Seconds a finished call (SDK) or a closing agent (ACP) gives
            Devvi to flush and exit.
        request_timeout_s: ACP: seconds for ``session/new``, ``session/resume`` and
            ``session/set_config_option`` (MCP servers connect during them).
        allow_untested_version: Run a Devvi version other than 0.2.x.
        **options: `ExternalAgent` options: ``permissions``, ``on_permission``,
            ``approval_timeout_s``, ``cancel_grace_s``, ``max_external_turns``,
            ``allow_unmetered`` (ACP reports no token usage, so a calling run with a token
            or cost budget refuses it without this), ``track_changes``.
    """

    backend = "devvi"

    grant_ignored_arguments = frozenset({"description", "justification"})
    """Free text Devvi's model writes anew on every attempt of a shell call or a file
    change (``description``, and the ``justification`` of an escalation). It does not
    change what runs, so an approval given across a pause (``on_permission="pause"``)
    also answers the same request with reworded text. ``command``, ``path``, ``content``,
    ``sandbox_permissions`` and every other argument must match."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        transport: Transport = "acp",
        name: str = "devvi",
        description: str = "",
        model: tuple[str, str] | None = None,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        permission_mode: PermissionMode = "workspace-write",
        home: str | Path | None = None,
        devvi_bin: str | Path | None = None,
        node: str | Path | Sequence[str] | None = None,
        command: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        env_passthrough: Iterable[str] = DEFAULT_PASSTHROUGH,
        patches: Sequence[Mapping[str, Any] | str | Path] = (),
        share_session_log: bool = False,
        pricing: TokenPricing | None = None,
        expose_tools: Sequence[Any] = (),
        expose_name: str = "qagents",
        resumable: bool = True,
        process_per_call: bool = True,
        wait_for_subagents: bool = True,
        raw_events: bool = False,
        initialize_timeout_s: float = 120.0,
        shutdown_timeout_s: float = 5.0,
        request_timeout_s: float = 120.0,
        allow_untested_version: bool = False,
        **options: Any,
    ) -> None:
        if workspace is None or not str(workspace):
            raise ConfigurationError("DevviAgent needs a workspace: the directory Devvi works in.")
        if transport not in ("acp", "sdk"):
            raise ConfigurationError(f"transport must be 'acp' or 'sdk', not {transport!r}.")
        route = _route(model) if model is not None else None
        if permission_mode not in PERMISSION_MODES:
            raise ConfigurationError(
                f"permission_mode must be one of {', '.join(PERMISSION_MODES)}, "
                f"not {permission_mode!r}."
            )
        if max_tokens is not None and (
            isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0
        ):
            raise ConfigurationError("max_tokens must be a positive integer.")
        if pricing is not None and not isinstance(pricing, TokenPricing):
            raise ConfigurationError("pricing must be a TokenPricing.")
        if transport == "acp":
            if pricing is not None:
                raise ConfigurationError(
                    "pricing needs transport='sdk': ACP reports no token counts, so there is "
                    "nothing to price. Use transport='sdk' to count Devvi's cost."
                )
            if max_tokens is not None:
                raise ConfigurationError(
                    "max_tokens needs transport='sdk'. On ACP, declare the limit on the model "
                    "route with a patch (patches=[...])."
                )
        exposed = ExposedTools.build(expose_tools, name=expose_name)
        if exposed is not None and exposed.in_process and transport == "sdk":
            raise ConfigurationError(
                "expose_tools with in-process tools needs transport='acp': the SDK transport "
                "mounts MCP servers through a patch file, and the tool server's bearer token "
                "must not be written to disk. Expose importable tools ('package.module:attr') "
                "instead, or use transport='acp'."
            )
        rows, paths = _split_patches(patches)
        super().__init__(name=name, description=description, workspace=workspace, **options)
        self.transport: Transport = transport
        self.model: tuple[str, str] | None = route if transport == "acp" else route or DEFAULT_MODEL
        """The model route asked for (ACP: ``None`` keeps Devvi's current model)."""
        self.pricing = pricing
        self.permission_mode: PermissionMode = permission_mode
        self.exposed = exposed
        """The qagents tools Devvi's model gets, or ``None``."""
        self._launch_options: dict[str, Any] = {
            "devvi_bin": devvi_bin,
            "node": node,
            "command": list(command) if command is not None else None,
            "home": home,
            "permission_mode": permission_mode,
            "passthrough": tuple(env_passthrough),
            "extra_env": dict(env or {}),
            "patch_rows": rows,
            "patch_paths": paths,
            "share_session_log": share_session_log,
            "allow_untested_version": allow_untested_version,
            # PATH entries inside the workspace are not searched for devvi and node: the
            # coder can write programs there (`launcher.locate_program`).
            "exclude": tuple(path for path in (self.workspace,) if path),
        }
        self._launch: DevviLaunch | None = None
        self._launch_lock: asyncio.Lock | None = None
        self._transport: _Transport
        if transport == "sdk":
            assert self.model is not None
            self._transport = SdkTransport(
                SdkSettings(
                    provider=self.model[0],
                    model=self.model[1],
                    reasoning_effort=reasoning_effort,
                    max_tokens=max_tokens,
                    pricing=pricing,
                    raw_events=raw_events,
                    wait_for_subagents=wait_for_subagents,
                    process_per_call=process_per_call,
                    initialize_timeout_s=initialize_timeout_s,
                    shutdown_timeout_s=shutdown_timeout_s,
                ),
                launch=self.launch,
                sessions_root=lambda launch: sessions_root(launch.rows, launch.home),
                name=self.name,
            )
        else:
            self._transport = AcpTransport(
                AcpSettings(
                    model=self.model,
                    reasoning_effort=reasoning_effort,
                    raw_events=raw_events,
                    resumable=resumable,
                    handshake_timeout_s=initialize_timeout_s,
                    request_timeout_s=request_timeout_s,
                    close_grace_s=shutdown_timeout_s,
                ),
                launch=self.launch,
                workspace=self.workspace,
                name=self.name,
                exposed=exposed,
            )
        self.resumable_turns = self._transport.resumable_turns

    # -- the launch (shared by both transports) --------------------------------------------

    async def launch(self) -> DevviLaunch:
        """Find the CLI and Node, check their versions, set up the home and write the patch.

        Runs once; later calls return the same `DevviLaunch`. On the SDK transport the
        stdio tool servers of ``expose_tools`` become rows of the patch.

        Raises:
            ExternalAgentConfigurationError: Anything that will not fix itself, with the fix.
        """
        if self._launch is not None:
            return self._launch
        if self._launch_lock is None:
            self._launch_lock = asyncio.Lock()
        async with self._launch_lock:
            if self._launch is None:
                options = dict(self._launch_options)
                if self.transport == "sdk" and self.exposed is not None and self.exposed.specs:
                    directory, _ = resolve_home(options["home"])
                    extra = check_rows(self.exposed.patch_rows(directory))
                    options["patch_rows"] = [*options["patch_rows"], *extra]
                self._launch = await prepare_launch(**options)
        return self._launch

    @property
    def connections(self) -> dict[str, DevviACPAgent]:
        """ACP: the Devvi connection of each workspace (keyed by its normalized path)."""
        if isinstance(self._transport, AcpTransport):
            return self._transport.agents
        return {}

    # -- ExternalAgent ---------------------------------------------------------------------

    def mark_disconnected(self) -> None:
        """See `ExternalAgent.mark_disconnected`.

        Both transports recover per process (their ``manages_connections``): on ACP each
        workspace's connection notices its own loss and is started again by the next turn
        in that workspace; on the SDK transport a broken or abandoned turn stops only its
        own process, and a pooled process that died is replaced on its next call. So the
        agent stays connected, and the next turn never stops the processes of other
        workspaces, which may be running a turn.
        """
        if getattr(getattr(self, "_transport", None), "manages_connections", False):
            return
        super().mark_disconnected()

    async def connect(self) -> None:
        await self._transport.connect()

    async def disconnect(self) -> None:
        await self._transport.disconnect()

    async def capabilities(self) -> ExternalCapabilities:
        return await self._transport.capabilities()

    async def run_turn(self, turn: ExternalTurnContext) -> ExternalOutcome:
        return await self._transport.run_turn(turn)

    async def cancel_turn(self, turn: ExternalTurnContext) -> None:
        await self._transport.cancel_turn(turn)

    def as_tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        timeout: float | None = None,
        continuable: bool = True,
        requires_approval: bool = False,
        idempotent: bool | None = None,
    ) -> Tool:
        """`ExternalAgent.as_tool`, described as a coding agent (`DEVVI_DESCRIPTION`) by default.

        Give a ``timeout`` that fits coding work (for example 1800 seconds): without it
        the calling run's ``Limits.tool_timeout_s`` applies (120 s by default).
        """
        if description is None and not self.description:
            label = name or _tool_safe(self.name)
            description = prompts.DEVVI_DESCRIPTION.format(name=label)
            if continuable:
                description = base_prompts.EXTERNAL_AGENT_DESCRIPTION_WITH_HINT.format(
                    description=description, hint=base_prompts.EXTERNAL_AGENT_CONTINUE_HINT
                )
        return super().as_tool(
            name=name,
            description=description,
            timeout=timeout,
            continuable=continuable,
            requires_approval=requires_approval,
            idempotent=idempotent,
        )


def _route(model: Any) -> tuple[str, str]:
    if (
        isinstance(model, (tuple, list))
        and len(model) == 2
        and all(isinstance(part, str) and part for part in model)
    ):
        return str(model[0]), str(model[1])
    raise ConfigurationError(
        "model must be (provider route, model id), for example "
        "('deepseek-official', 'deepseek-v4-flash')."
    )


def _split_patches(
    patches: Sequence[Mapping[str, Any] | str | Path],
) -> tuple[list[PatchRow], list[str]]:
    rows: list[Mapping[str, Any]] = []
    paths: list[str] = []
    if isinstance(patches, (str, Path, Mapping)):
        patches = [patches]
    for patch in patches:
        if isinstance(patch, Mapping):
            rows.append(patch)
        elif isinstance(patch, (str, Path)):
            paths.append(str(patch))
        else:
            raise ConfigurationError(
                f"patches must hold row dicts or patch file paths, not {type(patch).__name__}."
            )
    return check_rows(rows), paths
