"""Find and prepare the Devvi CLI: Node, the CLI script, its home and its environment.

    launch = await prepare_launch(home=None, patch_rows=default_rows(home))
    argv = launch.argv("sdk")       # [node, .../apps/cli/lib/bin.js, --profile, sdk, --patch, ...]
    process = await ManagedProcess.start(argv, env=launch.env, cwd=workspace)

Where things come from, in this order:

* **The CLI.** ``devvi_bin=``, else ``QAGENTS_DEVVI_BIN``, else ``devvi`` on ``PATH``. A
  ``.js`` file (Devvi's ``apps/cli/lib/bin.js``, or ``lib/bin.js`` of the installed
  ``@deepseek-ai/dsh`` package) runs as ``[node, path]``: Windows cannot execute a ``.js``
  file, and on POSIX this way the Node version is checked too. On Windows the ``devvi.cmd``
  shim that ``npm install -g`` (or pnpm) writes is never run: the script it wraps is found
  in it and run with Node instead. ``command=`` replaces all of this with an exact command.
* **Node.** ``node=``, else ``QAGENTS_NODE``, else ``node`` on ``PATH``. It must be at least
  24.9.0 (Devvi's ``engines.node``).
* **Looking a name up on PATH** (``devvi``, ``node``, or a bare name in ``QAGENTS_DEVVI_BIN``,
  ``QAGENTS_NODE`` or ``node=``): only the absolute ``PATH`` entries are searched
  (`qagents.integrations._process.find_program`), never the current directory, and never
  an entry inside the agent's workspace (``exclude``). Windows itself would look in the
  current directory first, and qagents often runs in the workspace, where the coder can
  write a ``node.exe`` that would then run outside Devvi's sandbox. An absolute path, or a
  path with a folder part (relative to the current directory), is used as given. The
  command is always stored with absolute paths.
* **The home.** ``home=``, else ``QAGENTS_DEVVI_HOME``, else ``~/.qagents/devvi-home``: not
  ``~/.devvi``, so sessions qagents starts stay apart from the user's own Devvi sessions.
  Both ``DEVVI_HOME`` and ``DSH_HOME`` are set to it, because Devvi lets ``DEVVI_HOME`` win
  over ``DSH_HOME`` (``packages/util/home-paths``): an inherited ``DEVVI_HOME`` would
  otherwise move the home silently. A different inherited value is overridden and logged.
* **The environment.** An allowlist (`qagents.integrations._process.build_env`): the
  operating-system basics, the names in ``passthrough`` (by default exactly
  ``DEEPSEEK_API_KEY`` and ``DEEPSEEK_BASE_URL``), the home variables,
  ``DSH_PERMISSION_MODE`` and ``DSH_TELEMETRY_DISABLED=1``. No other ``*_KEY`` or token of
  the parent process reaches Devvi unless its name is listed.

Before the first start, ``node --version`` and ``devvi --version`` are run once
(`probe`). The version is recorded in ``ExternalResult.backend`` because Devvi's wire
reports a fixed ``0.0.1``. Versions other than 0.2.x are refused (the SDK protocol has no
version negotiation) unless ``allow_untested_version=True``.

Note: Devvi itself also reads a ``.env`` file in its working directory and in its home,
without overriding variables that are already set (``loadLayeredEnv`` in
``packages/boot/app-boot``). Keep credentials out of a workspace's ``.env``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ... import prompts
from ...external import ExternalAgentConfigurationError
from .._process import build_env, find_program
from .._process import resolve_command as resolve_program
from .patches import PatchRow, devvi_patch, write_patch

logger = logging.getLogger("qagents.integrations.devvi")

ENV_BIN = "QAGENTS_DEVVI_BIN"
"""Path of the Devvi CLI (a ``.js`` file runs with Node)."""

ENV_NODE = "QAGENTS_NODE"
"""Path of the Node executable."""

ENV_HOME = "QAGENTS_DEVVI_HOME"
"""Devvi home for the processes qagents starts."""

HOME_VARIABLES = ("DEVVI_HOME", "DSH_HOME")
"""Both name Devvi's home; the launcher sets both to the same value."""

NODE_MIN_VERSION = (24, 9, 0)
"""Devvi's ``engines.node`` (root ``package.json``)."""

TESTED_VERSIONS = ((0, 2),)
"""Devvi ``major.minor`` versions qagents was written against (0.2.1-rc.3)."""

DEFAULT_PASSTHROUGH = ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL")
"""Variables copied to Devvi by default: the ``deepseek-official`` route's key and URL."""

PermissionMode = Literal["read-only", "workspace-write", "danger-full-access"]
"""``DSH_PERMISSION_MODE``. With ``danger-full-access`` Devvi's approval policy is
``never``: it asks nothing, so use it only in a disposable workspace."""

PERMISSION_MODES: tuple[str, ...] = ("read-only", "workspace-write", "danger-full-access")

SCRIPT_SUFFIXES = (".js", ".mjs", ".cjs")

PROBE_TIMEOUT_S = 60.0

_WINDOWS = sys.platform == "win32"
_RESERVED = {"DEVVI_HOME", "DSH_HOME", "DSH_PERMISSION_MODE"}
_SHIM_SCRIPT = re.compile(r'"%~?dp0%?[\\/]?([^"%*]+?\.(?:c|m)?js)"', re.IGNORECASE)
_DEVVI_VERSION = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:[-+][0-9A-Za-z.+-]+)?$")


# --- the command --------------------------------------------------------------------------


@dataclass(frozen=True)
class DevviCommand:
    """How to start the Devvi CLI, without its arguments."""

    program: tuple[str, ...]
    """The program and the arguments before Devvi's own: ``(node, bin.js)``,
    ``(devvi.exe,)``, or an explicit ``command=``."""
    node: tuple[str, ...] | None = None
    """The Node command when the CLI is a script run by Node (its version is checked)."""
    script: str | None = None
    """The CLI script, when run by Node."""
    source: str = "argument"
    """Where the CLI was found: ``"argument"``, ``"QAGENTS_DEVVI_BIN"``, ``"PATH"`` or
    ``"command"``."""

    def argv(self, *arguments: str) -> list[str]:
        return [*self.program, *arguments]


def resolve_node(
    node: str | Path | Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    exclude: Sequence[str | Path] = (),
) -> tuple[str, ...]:
    """The command that runs Node: ``node=``, else ``QAGENTS_NODE``, else ``node`` on PATH.

    A sequence is used as given (a wrapper such as ``["volta", "run", "node"]``); its first
    item is resolved like ``node=``. A bare name is looked up on ``PATH`` as the module
    docstring says (never in the current directory, never inside ``exclude``); the result
    is an absolute path.

    Raises:
        ExternalAgentConfigurationError: Node was not found, or it is a ``.cmd``/``.bat``
            shim on Windows.
    """
    environ = os.environ if environ is None else environ
    if node is None:
        configured = _get(environ, ENV_NODE)
        node = configured if configured else None
        where = ENV_NODE if configured else "PATH"
    else:
        where = "node="
    if node is None:
        found = locate_program("node", environ=environ, exclude=exclude)
        if found is None:
            raise ExternalAgentConfigurationError(
                f"{prompts.DEVVI_NODE_NOT_FOUND}\n"
                f"Install Node >= {_dotted(NODE_MIN_VERSION)} and put it on PATH, or set "
                f"{ENV_NODE} to the node executable."
                + _hidden_note("node", environ, exclude, ENV_NODE)
            )
        prefix: list[str] = [found]
    elif isinstance(node, (str, Path)):
        prefix = [str(node)]
    else:
        prefix = [str(part) for part in node]
        if not prefix:
            raise ExternalAgentConfigurationError(prompts.DEVVI_NODE_EMPTY)
    program = prefix[0]
    found = locate_program(program, environ=environ, exclude=exclude)
    if found is None:
        raise ExternalAgentConfigurationError(
            f"{prompts.DEVVI_NODE_NOT_FOUND_AT.format(program=program, source=where)}\n"
            f"Set {ENV_NODE} to the node executable, or put node on PATH."
            + _hidden_note(program, environ, exclude, ENV_NODE)
        )
    if _WINDOWS and Path(found).suffix.lower() in (".cmd", ".bat"):
        raise ExternalAgentConfigurationError(
            f"{prompts.DEVVI_NODE_SHIM.format(path=found, suffix=Path(found).suffix)}\n"
            f"Set {ENV_NODE} to the real node.exe."
        )
    return (found, *prefix[1:])


def resolve_command(
    devvi_bin: str | Path | None = None,
    *,
    node: str | Path | Sequence[str] | None = None,
    command: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
    exclude: Sequence[str | Path] = (),
) -> DevviCommand:
    """Find the Devvi CLI (see the module docstring for the order).

    Args:
        exclude: Folders whose ``PATH`` entries are never searched: the agent's workspace,
            where the coder can write programs.

    Raises:
        ExternalAgentConfigurationError: The CLI or Node was not found, or a Windows shim
            does not say which script it runs.
    """
    environ = os.environ if environ is None else environ
    if command is not None:
        program = tuple(str(part) for part in command)
        if not program:
            raise ExternalAgentConfigurationError(prompts.DEVVI_COMMAND_EMPTY)
        return DevviCommand(program=program, source="command")
    configured = str(devvi_bin) if devvi_bin is not None else _get(environ, ENV_BIN)
    if configured:
        source = "argument" if devvi_bin is not None else ENV_BIN
        path = _existing_path(configured, environ, source, exclude)
    else:
        source = "PATH"
        found = locate_program("devvi", environ=environ, exclude=exclude)
        if found is None:
            raise ExternalAgentConfigurationError(
                f"{prompts.DEVVI_CLI_NOT_FOUND}\n"
                f"Set {ENV_BIN} to Devvi's apps/cli/lib/bin.js (or lib/bin.js of the installed "
                "@deepseek-ai/dsh package), or put devvi on PATH."
                + _hidden_note("devvi", environ, exclude, ENV_BIN)
            )
        path = Path(found)
    script = _script_of(path)
    if script is None:
        return DevviCommand(program=(str(path),), source=source)
    runner = resolve_node(node, environ=environ, exclude=exclude)
    return DevviCommand(
        program=(*runner, str(script)), node=runner, script=str(script), source=source
    )


def locate_program(
    program: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
    exclude: Sequence[str | Path] = (),
) -> str | None:
    """The absolute path of ``program``, or ``None`` when there is no such file.

    An absolute path is used as given, and a path with a folder part is taken relative to
    the current directory: both are the caller's own choice. A bare name is looked up with
    `qagents.integrations._process.find_program`: only the absolute entries of ``PATH``
    (from ``environ``), with ``PATHEXT`` on Windows, never the current directory, and never
    an entry inside a folder of ``exclude``.
    """
    environ = os.environ if environ is None else environ
    name = _expand_user(str(program), environ)
    if os.path.isabs(name) or os.path.dirname(name):
        candidate = os.path.abspath(name)
        return candidate if os.path.isfile(candidate) else None
    return find_program(name, _path_of(environ), pathext=_get(environ, "PATHEXT"), exclude=exclude)


def _hidden_note(
    program: str | Path, environ: Mapping[str, str], exclude: Sequence[str | Path], variable: str
) -> str:
    """Why a program on PATH was not used: its folder is inside the workspace."""
    name = str(program)
    if not exclude or os.path.isabs(name) or os.path.dirname(name):
        return ""
    found = find_program(name, _path_of(environ), pathext=_get(environ, "PATHEXT"))
    if found is None:
        return ""
    return (
        f"\n{found} is on PATH, but inside the agent's workspace, where the agent can write "
        f"files, so it is not used. Set {variable} to it explicitly if you trust it."
    )


def _existing_path(
    configured: str, environ: Mapping[str, str], source: str, exclude: Sequence[str | Path]
) -> Path:
    expanded = _expand_user(configured, environ)
    found = locate_program(expanded, environ=environ, exclude=exclude)
    if found is None:
        bare = not (os.path.isabs(expanded) or os.path.dirname(expanded))
        where = prompts.DEVVI_CLI_ON_PATH if bare else ""
        first = prompts.DEVVI_CLI_NOT_FOUND_AT.format(path=configured, where=where, source=source)
        raise ExternalAgentConfigurationError(
            f"{first}\n"
            f"Set {ENV_BIN} to an existing file (an absolute path): Devvi's apps/cli/lib/bin.js, "
            "or lib/bin.js of the installed @deepseek-ai/dsh package."
            + _hidden_note(expanded, environ, exclude, ENV_BIN)
        )
    return Path(found).resolve()


def _script_of(path: Path) -> Path | None:
    """The ``.js`` file to run with Node for ``path``, or ``None`` for a real executable."""
    suffix = path.suffix.lower()
    if suffix in SCRIPT_SUFFIXES:
        return path.resolve()
    if _WINDOWS:
        if suffix in (".cmd", ".bat"):
            script = parse_cmd_shim(path)
            if script is None:
                raise ExternalAgentConfigurationError(
                    f"{prompts.DEVVI_CLI_SHIM.format(path=path, suffix=suffix)}\n"
                    f"Set {ENV_BIN} to the package's lib/bin.js instead."
                )
            return script
        if suffix == ".exe":
            return None
        sibling = path.with_suffix(".cmd")
        if sibling.is_file() and (script := parse_cmd_shim(sibling)) is not None:
            return script
        raise ExternalAgentConfigurationError(
            f"{prompts.DEVVI_CLI_NOT_RUNNABLE.format(path=path)}\n"
            f"Set {ENV_BIN} to the package's lib/bin.js."
        )
    real = path.resolve()
    if real.suffix.lower() in SCRIPT_SUFFIXES:
        return real
    with contextlib.suppress(OSError):
        with real.open("rb") as handle:
            first = handle.readline(200)
        if first.startswith(b"#!") and b"node" in first:
            return real
    return None


def parse_cmd_shim(path: str | Path) -> Path | None:
    """The script a Windows ``.cmd`` shim of npm or pnpm runs (``"%dp0%\\...\\bin.js"``)."""
    shim = Path(path)
    try:
        text = shim.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for match in _SHIM_SCRIPT.finditer(text):
        relative = match.group(1).replace("\\", "/").lstrip("/")
        candidate = Path(os.path.normpath(shim.parent / relative))
        if candidate.is_file():
            return candidate
    return None


# --- versions ------------------------------------------------------------------------------


def parse_node_version(text: str) -> tuple[int, int, int]:
    """``"v24.9.0"`` -> ``(24, 9, 0)``.

    Raises:
        ExternalAgentConfigurationError: The text holds no version.
    """
    match = re.search(r"v?(\d+)\.(\d+)\.(\d+)", text or "")
    if match is None:
        raise ExternalAgentConfigurationError(
            prompts.DEVVI_NODE_VERSION_UNREADABLE.format(text=text.strip()[:80])
        )
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def check_node_version(version: tuple[int, int, int]) -> None:
    """Refuse Node older than `NODE_MIN_VERSION`."""
    if version < NODE_MIN_VERSION:
        raise ExternalAgentConfigurationError(
            prompts.DEVVI_NODE_TOO_OLD.format(
                version=_dotted(version), minimum=_dotted(NODE_MIN_VERSION)
            )
            + "\n"
            f"Install a newer Node, or set {ENV_NODE} to one."
        )


def parse_devvi_version(text: str) -> str:
    """The version line ``devvi --version`` printed, for example ``"0.2.1-rc.3"``.

    Raises:
        ExternalAgentConfigurationError: The output is not a version.
    """
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            if _DEVVI_VERSION.match(line):
                return line.removeprefix("v")
            break
    hint = (
        " An empty output usually means Node is older than 24.2: Devvi's CLI starts only "
        "under import.meta.main."
        if not (text or "").strip()
        else ""
    )
    raise ExternalAgentConfigurationError(
        f"{prompts.DEVVI_VERSION_NOT_PRINTED.format(text=(text or '').strip()[:80])}\n"
        f"Check that {ENV_BIN} points at the Devvi CLI, and that Node is >= 24.9.0.{hint}"
    )


def check_devvi_version(version: str, *, allow_untested: bool = False) -> None:
    """Refuse a Devvi version qagents was not written against (see `TESTED_VERSIONS`)."""
    match = _DEVVI_VERSION.match(version)
    if match is None:
        raise ExternalAgentConfigurationError(
            prompts.DEVVI_VERSION_UNREADABLE.format(version=version)
        )
    if (int(match.group(1)), int(match.group(2))) in TESTED_VERSIONS or allow_untested:
        return
    tested = prompts.EXTERNAL_SETUP_LIST_SEPARATOR.join(
        prompts.DEVVI_TESTED_VERSION.format(major=major, minor=minor)
        for major, minor in TESTED_VERSIONS
    )
    raise ExternalAgentConfigurationError(
        f"{prompts.DEVVI_VERSION_UNTESTED.format(version=version, tested=tested)}\n"
        "Devvi's SDK protocol has no version negotiation, so a newer Devvi may speak it "
        "differently. Pass allow_untested_version=True to try it anyway."
    )


# --- home and environment --------------------------------------------------------------------


def default_home(environ: Mapping[str, str] | None = None) -> Path:
    """``~/.qagents/devvi-home`` (``~`` from ``USERPROFILE`` on Windows, else ``HOME``)."""
    environ = os.environ if environ is None else environ
    base = _get(environ, "USERPROFILE" if _WINDOWS else "HOME") or str(Path.home())
    return Path(base) / ".qagents" / "devvi-home"


def resolve_home(
    home: str | Path | None = None, *, environ: Mapping[str, str] | None = None
) -> tuple[Path, str]:
    """The Devvi home and where it came from: ``home=``, ``QAGENTS_DEVVI_HOME`` or the default.

    Inherited ``DEVVI_HOME`` and ``DSH_HOME`` are never used (see `home_conflicts`).
    """
    environ = os.environ if environ is None else environ
    if home is not None:
        chosen, source = str(home), "home="
    elif configured := _get(environ, ENV_HOME):
        chosen, source = configured, ENV_HOME
    else:
        return default_home(environ), "default"
    path = Path(_expand_user(chosen, environ))
    return (path if path.is_absolute() else Path.cwd() / path), source


def home_conflicts(home: str | Path, environ: Mapping[str, str] | None = None) -> list[str]:
    """Warnings about inherited ``DEVVI_HOME``/``DSH_HOME`` that the launcher overrides."""
    environ = os.environ if environ is None else environ
    target = _same_key(home)
    values = {name: _get(environ, name) for name in HOME_VARIABLES}
    found = {name: value for name, value in values.items() if value}
    warnings: list[str] = []
    if len(found) == 2 and _same_key(found["DEVVI_HOME"]) != _same_key(found["DSH_HOME"]):
        warnings.append(
            "DEVVI_HOME and DSH_HOME are both set and differ; Devvi would use DEVVI_HOME. "
            f"qagents sets both to {home}."
        )
    for name, value in found.items():
        if _same_key(value) != target:
            warnings.append(f"The inherited {name} is overridden: qagents runs Devvi in {home}.")
    return warnings


def build_environment(
    home: str | Path,
    *,
    permission_mode: str = "workspace-write",
    passthrough: Iterable[str] = DEFAULT_PASSTHROUGH,
    extra: Mapping[str, str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The explicit environment of a Devvi process.

    Args:
        home: The Devvi home (``DEVVI_HOME`` and ``DSH_HOME``).
        permission_mode: ``DSH_PERMISSION_MODE``.
        passthrough: Names copied from ``environ`` when set (credentials by name only).
        extra: Variables set explicitly, besides those the launcher sets.
        environ: The parent environment; ``os.environ`` by default.

    Raises:
        ExternalAgentConfigurationError: An unknown permission mode, or ``extra`` or
            ``passthrough`` names a variable the launcher owns.
    """
    if permission_mode not in PERMISSION_MODES:
        modes = prompts.EXTERNAL_SETUP_LIST_SEPARATOR.join(PERMISSION_MODES)
        raise ExternalAgentConfigurationError(
            prompts.DEVVI_PERMISSION_MODE_UNKNOWN.format(modes=modes, mode=permission_mode)
        )
    names = [str(name) for name in passthrough]
    explicit = dict(extra or {})
    for name in [*names, *explicit]:
        if str(name).upper() in _RESERVED:
            raise ExternalAgentConfigurationError(prompts.DEVVI_VARIABLE_RESERVED.format(name=name))
    owned = {
        "DEVVI_HOME": str(home),
        "DSH_HOME": str(home),
        "DSH_PERMISSION_MODE": permission_mode,
        "DSH_TELEMETRY_DISABLED": "1",
    }
    return build_env(passthrough=names, extra={**explicit, **owned}, source=environ)


# --- probing -----------------------------------------------------------------------------------


async def run_probe(
    argv: Sequence[str],
    *,
    env: Mapping[str, str],
    cwd: str | Path | None = None,
    timeout: float = PROBE_TIMEOUT_S,
) -> tuple[int | None, str, str]:
    """Run a short command (``--version``); its exit code, stdout and stderr.

    The exit code is ``None`` when it did not finish within ``timeout`` (it is killed).
    When the caller is cancelled (a turn's forced cancel during its first connect, a
    cancelled ``doctor``), the process is killed too before the cancel goes on.

    Raises:
        ExternalAgentConfigurationError: The program cannot be started.
    """
    command = resolve_program(list(argv), env)
    options: dict[str, Any] = {}
    if _WINDOWS:
        options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=dict(env),
            cwd=str(cwd) if cwd is not None else None,
            **options,
        )
    except NotImplementedError as error:
        raise ExternalAgentConfigurationError(prompts.DEVVI_NO_SUBPROCESSES) from error
    except OSError as error:
        raise ExternalAgentConfigurationError(
            prompts.DEVVI_CANNOT_START.format(program=command[0], error=error)
        ) from error
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except BaseException as error:  # a timeout, or the caller was cancelled: stop it too
        with contextlib.suppress(ProcessLookupError, OSError):
            process.kill()
        with contextlib.suppress(Exception):  # bounded; a second cancel still goes through
            await asyncio.wait_for(asyncio.shield(process.wait()), 5.0)
        transport = getattr(process, "_transport", None)
        if transport is not None:  # release the pipes now, not when it is collected
            with contextlib.suppress(Exception):
                transport.close()
        if isinstance(error, TimeoutError):
            return None, "", ""
        raise
    return (
        process.returncode,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


@dataclass(frozen=True)
class ProbeResult:
    version: str
    """``devvi --version``."""
    node_version: tuple[int, int, int] | None = None
    """``node --version``; ``None`` when the CLI does not run with Node."""


async def probe(
    command: DevviCommand,
    *,
    env: Mapping[str, str],
    cwd: str | Path | None = None,
    timeout: float = PROBE_TIMEOUT_S,
    allow_untested_version: bool = False,
) -> ProbeResult:
    """Check Node's version (when the CLI runs with Node) and read ``devvi --version``.

    Raises:
        ExternalAgentConfigurationError: Node is too old, or ``--version`` fails or prints
            no version, or the version is not one qagents was tested with.
    """
    node_version: tuple[int, int, int] | None = None
    if command.node is not None:
        code, out, err = await run_probe([*command.node, "--version"], env=env, cwd=cwd)
        if code != 0:
            raise ExternalAgentConfigurationError(
                f"{prompts.DEVVI_NODE_DID_NOT_START}\n"
                f"Command: {' '.join(command.node)}; exit code {code}; stderr: {err.strip()[-500:]}"
            )
        node_version = parse_node_version(out)
        check_node_version(node_version)
    code, out, err = await run_probe(command.argv("--version"), env=env, cwd=cwd, timeout=timeout)
    if code != 0:
        how = (
            prompts.DEVVI_PROBE_TIMEOUT
            if code is None
            else prompts.DEVVI_PROBE_EXIT.format(code=code)
        )
        raise ExternalAgentConfigurationError(
            f"{prompts.DEVVI_CLI_DID_NOT_START.format(how=how)}\n"
            f"Command: {' '.join(command.argv('--version'))}; stderr: {err.strip()[-1000:]}\n"
            f"Check {ENV_BIN} (and run `pnpm run build` in a Devvi checkout)."
        )
    version = parse_devvi_version(out)
    check_devvi_version(version, allow_untested=allow_untested_version)
    return ProbeResult(version=version, node_version=node_version)


# --- the whole launch --------------------------------------------------------------------------


@dataclass(frozen=True)
class DevviLaunch:
    """Everything needed to start Devvi processes; prepared once per agent."""

    command: DevviCommand
    home: Path
    home_source: str
    env: dict[str, str] = field(repr=False)
    """The child environment. It may hold credentials: never log or print it."""
    version: str = ""
    node_version: tuple[int, int, int] | None = None
    patch_files: tuple[str, ...] = ()
    rows: tuple[PatchRow, ...] = field(default=(), repr=False)
    """The rows written to the first patch file."""
    warnings: tuple[str, ...] = ()

    def argv(self, profile: str, *arguments: str) -> list[str]:
        """``[*command, --profile, profile, --patch, file, ..., *arguments]``.

        Launcher flags come first: Devvi hands every argument after them to the profile.
        """
        argv = self.command.argv("--profile", profile)
        for path in self.patch_files:
            argv += ["--patch", path]
        return [*argv, *arguments]

    @property
    def passed_names(self) -> list[str]:
        """Names of the variables Devvi gets (never their values)."""
        return sorted(self.env)


async def prepare_launch(
    *,
    devvi_bin: str | Path | None = None,
    node: str | Path | Sequence[str] | None = None,
    command: Sequence[str] | None = None,
    home: str | Path | None = None,
    permission_mode: str = "workspace-write",
    passthrough: Iterable[str] = DEFAULT_PASSTHROUGH,
    extra_env: Mapping[str, str] | None = None,
    patch_rows: Sequence[PatchRow] = (),
    patch_paths: Sequence[str | Path] = (),
    share_session_log: bool = False,
    allow_untested_version: bool = False,
    environ: Mapping[str, str] | None = None,
    probe_timeout: float = PROBE_TIMEOUT_S,
    exclude: Sequence[str | Path] = (),
) -> DevviLaunch:
    """Resolve the CLI and Node, check their versions, set up the home and write the patch.

    `patches.default_rows` for the home, then ``patch_rows``, are written to one
    ``.patch.yml`` file under ``<home>/qagents/patches``; ``patch_paths`` are passed after
    it, in order (later layers win). ``exclude`` (the agent's workspace) names folders
    whose ``PATH`` entries are never searched for ``devvi`` and ``node``.

    Raises:
        ExternalAgentConfigurationError: Anything that will not fix itself.
    """
    environ = os.environ if environ is None else environ
    directory, source = resolve_home(home, environ=environ)
    directory.mkdir(parents=True, exist_ok=True)
    warnings = home_conflicts(directory, environ)
    for warning in warnings:
        logger.warning("%s", warning)
    env = build_environment(
        directory,
        permission_mode=permission_mode,
        passthrough=passthrough,
        extra=extra_env,
        environ=environ,
    )
    cli = resolve_command(devvi_bin, node=node, command=command, environ=environ, exclude=exclude)
    result = await probe(
        cli,
        env=env,
        cwd=directory,
        timeout=probe_timeout,
        allow_untested_version=allow_untested_version,
    )
    rows = devvi_patch(directory, share_session_log=share_session_log, extra=patch_rows)
    files = [str(write_patch(directory / "qagents" / "patches", rows)), *_patch_files(patch_paths)]
    return DevviLaunch(
        command=cli,
        home=directory,
        home_source=source,
        env=env,
        version=result.version,
        node_version=result.node_version,
        patch_files=tuple(files),
        rows=tuple(rows),
        warnings=tuple(warnings),
    )


def _patch_files(paths: Sequence[str | Path]) -> list[str]:
    files: list[str] = []
    for extra in paths:
        path = Path(extra)
        if not path.is_file():
            raise ExternalAgentConfigurationError(
                prompts.DEVVI_PATCH_FILE_MISSING.format(path=str(extra))
            )
        files.append(str(path.resolve()))
    return files


# --- helpers -----------------------------------------------------------------------------------


def _get(environ: Mapping[str, str], name: str) -> str | None:
    """A variable's value, matching the name without regard to case on Windows."""
    if name in environ:
        return environ[name]
    if _WINDOWS:
        for key, value in environ.items():
            if key.upper() == name.upper():
                return value
    return None


def _path_of(environ: Mapping[str, str]) -> str | None:
    return _get(environ, "PATH")


def _expand_user(path: str, environ: Mapping[str, str]) -> str:
    if path == "~" or path.startswith(("~/", "~\\")):
        base = _get(environ, "USERPROFILE" if _WINDOWS else "HOME") or str(Path.home())
        return str(Path(base) / path[2:]) if len(path) > 1 else base
    return path


def _same_key(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _dotted(version: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in version)
