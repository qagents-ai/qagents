"""``python -m qagents.integrations.devvi doctor``: check a Devvi setup, with no model call.

    $env:QAGENTS_DEVVI_BIN = "C:\\...\\devvi-harness\\apps\\cli\\lib\\bin.js"
    python -m qagents.integrations.devvi doctor
    python -m qagents.integrations.devvi doctor --json

It prints, one check per line (``ok``, ``warn`` or ``fail``): the Devvi home and where it
came from, inherited ``DEVVI_HOME``/``DSH_HOME`` that would be overridden, which
pass-through variables are set (names only, never values), the resolved CLI command, the
Node and Devvi versions, the patch file, whether ``--dump-config`` shows the patch rows in
the ``acp`` and ``sdk`` profiles, then the two handshakes, each followed by a clean
shutdown:

* **ACP** (``devvi --profile acp``, the default transport): ``initialize`` (``agentInfo``,
  protocol version, MCP over HTTP, image prompts, session resume, the time to ready), then
  one ``session/new`` in ``<home>/qagents/doctor`` (or ``--workspace``) for the model
  options: ``ACP models`` lists the current model and the offered ones, and fails when
  ``--provider``/``--model`` name a model the session does not offer. Devvi keeps that
  empty session in its home; it costs nothing.
* **SDK** (``devvi --profile sdk``): ``initialize`` (``serverInfo`` and the time to ready).

No prompt is sent, so no model is called. The exit code is 1 when a check failed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ...external import (
    ExternalAgentConfigurationError,
    ExternalAgentConnectionError,
    ExternalAgentError,
)
from .._jsonrpc import JsonRpcError
from ..acp import PROTOCOL_VERSION
from . import DEFAULT_MODEL
from .acp import PROFILE as ACP_PROFILE
from .acp import DevviACPAgent
from .launcher import (
    DEFAULT_PASSTHROUGH,
    ENV_BIN,
    ENV_HOME,
    ENV_NODE,
    DevviLaunch,
    _get,
    build_environment,
    home_conflicts,
    probe,
    resolve_command,
    resolve_home,
    run_probe,
)
from .patches import devvi_patch, write_patch
from .sdk import PROFILE, SdkConnection, SdkSettings


@dataclass
class Check:
    """One line of the doctor's report."""

    name: str
    status: str
    """``ok``, ``warn`` or ``fail``."""
    detail: str

    def line(self) -> str:
        return f"[{self.status:>4}] {self.name}: {self.detail}"


async def doctor(
    *,
    devvi_bin: str | Path | None = None,
    node: str | Path | Sequence[str] | None = None,
    command: Sequence[str] | None = None,
    home: str | Path | None = None,
    workspace: str | Path | None = None,
    model: tuple[str, str] | None = None,
    passthrough: Iterable[str] = DEFAULT_PASSTHROUGH,
    sdk: bool = True,
    acp: bool = True,
    allow_untested_version: bool = False,
    environ: Mapping[str, str] | None = None,
    timeout: float = 120.0,
) -> list[Check]:
    """Run the checks and return them; stops at the first failure that blocks the rest.

    Args:
        model: ``(provider route, model id)``: the SDK handshake's route (default
            `DEFAULT_MODEL`), and on ACP a model that must be offered (default: only
            report what is offered).
        sdk: Run the SDK handshake.
        acp: Run the ACP handshake and open one session for its model options.
    """
    environ = os.environ if environ is None else environ
    checks: list[Check] = []
    names = list(passthrough)

    directory, source = resolve_home(home, environ=environ)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        checks.append(Check("home", "fail", f"{directory} cannot be created: {error}"))
        return checks
    origin = {"home=": "argument", ENV_HOME: ENV_HOME, "default": "default"}[source]
    checks.append(Check("home", "ok", f"{directory} (from {origin})"))
    conflicts = home_conflicts(directory, environ)
    checks.append(
        Check("DEVVI_HOME/DSH_HOME", "warn", " ".join(conflicts))
        if conflicts
        else Check("DEVVI_HOME/DSH_HOME", "ok", "both set to the home above; nothing inherited")
    )
    present = [name for name in names if _get(environ, name)]
    missing = [name for name in names if not _get(environ, name)]
    detail = f"passed through when set: {', '.join(names) or 'none'}"
    if missing:
        detail += f"; not set: {', '.join(missing)}"
    checks.append(Check("environment", "ok" if present or not names else "warn", detail))

    try:
        env = build_environment(directory, passthrough=names, environ=environ)
        exclude = [workspace] if workspace is not None else []
        cli = resolve_command(
            devvi_bin, node=node, command=command, environ=environ, exclude=exclude
        )
    except ExternalAgentConfigurationError as error:
        checks.append(Check("Devvi CLI", "fail", _oneline(error)))
        return checks
    where = {"argument": "--bin", "command": "--command"}.get(cli.source, cli.source)
    checks.append(Check("Devvi CLI", "ok", f"{' '.join(cli.program)} (from {where})"))
    try:
        result = await probe(
            cli, env=env, cwd=directory, allow_untested_version=allow_untested_version
        )
    except ExternalAgentConfigurationError as error:
        checks.append(Check("versions", "fail", _oneline(error)))
        return checks
    if result.node_version is not None:
        checks.append(Check("Node", "ok", ".".join(map(str, result.node_version))))
    checks.append(Check("Devvi", "ok", result.version))

    rows = devvi_patch(directory)
    patch = write_patch(directory / "qagents" / "patches", rows)
    checks.append(Check("patch", "ok", f"{patch} (rows: {', '.join(_row_ids(rows))})"))
    launch = DevviLaunch(
        command=cli,
        home=directory,
        home_source=source,
        env=env,
        version=result.version,
        node_version=result.node_version,
        patch_files=(str(patch),),
        rows=tuple(rows),
    )
    profiles = [name for name, wanted in ((ACP_PROFILE, acp), (PROFILE, sdk)) if wanted]
    problems: list[str] = []
    for profile in profiles:
        code, out, _ = await run_probe(
            launch.argv(profile, "--dump-config"), env=env, cwd=directory, timeout=timeout
        )
        unseen = [row for row in _row_ids(rows) if row not in out]
        if code != 0:
            problems.append(f"{profile}: exit code {code}")
        elif unseen:
            problems.append(f"{profile}: rows not shown: {', '.join(unseen)}")
    if problems:
        detail = f"could not confirm the patch rows ({'; '.join(problems)})"
        checks.append(Check("dump-config", "warn", detail))
    elif profiles:
        shown = " and ".join(profiles)
        checks.append(Check("dump-config", "ok", f"the composed {shown} profiles show every row"))

    folder = Path(workspace) if workspace is not None else directory / "qagents" / "doctor"
    if acp or sdk:
        folder.mkdir(parents=True, exist_ok=True)
    if acp:
        checks.extend(await _check_acp(launch, str(folder.resolve()), model, timeout))
    if sdk:
        route = model or DEFAULT_MODEL
        settings = SdkSettings(provider=route[0], model=route[1])
        try:
            connection = await SdkConnection.open(
                launch,
                str(folder.resolve()),
                settings.initialize_params(str(folder.resolve())),
                name="devvi-doctor",
                timeout=timeout,
            )
        except ExternalAgentConfigurationError as error:
            checks.append(Check("SDK", "fail", _oneline(error)))
        else:
            info = connection.server_info
            ready = connection.ready_s or 0.0
            await connection.stop(shutdown_timeout=5.0, grace=5.0)
            checks.append(
                Check(
                    "SDK",
                    "ok",
                    f"{info.get('name')} {info.get('version')} on route {route[0]}/{route[1]}, "
                    f"ready in {ready:.1f} s; shut down with exit code "
                    f"{connection.process.returncode}",
                )
            )
    return checks


async def _check_acp(
    launch: DevviLaunch, folder: str, model: tuple[str, str] | None, timeout: float
) -> list[Check]:
    """The ``ACP`` and ``ACP models`` checks: handshake, one session, clean shutdown."""
    agent = DevviACPAgent(
        launch,
        cwd=folder,
        name="devvi-doctor",
        handshake_timeout_s=timeout,
        request_timeout_s=timeout,
    )
    began = time.monotonic()
    options: list[Any] | None = None
    failure: str | None = None
    try:
        await agent.ensure_connected()
        ready = time.monotonic() - began
        try:
            _, options = await agent.open_session(folder)
        except JsonRpcError as error:
            failure = f"session/new failed: {error.message}"
        except ExternalAgentConnectionError as error:
            failure = f"session/new did not finish ({error.reason})"
    except ExternalAgentError as error:
        await agent.aclose()
        return [Check("ACP", "fail", _oneline(error))]
    process = agent.process
    await agent.aclose()
    code = process.returncode if process is not None else None
    info, caps = agent.agent_info, agent.agent_capabilities
    mcp = caps.get("mcpCapabilities") if isinstance(caps.get("mcpCapabilities"), Mapping) else {}
    prompt = caps.get("promptCapabilities")
    sessions = caps.get("sessionCapabilities")
    features = ", ".join(
        f"{label} {'yes' if present else 'no'}"
        for label, present in (
            ("MCP over HTTP", mcp.get("http") is True),
            ("image prompts", isinstance(prompt, Mapping) and prompt.get("image") is True),
            ("session resume", isinstance(sessions, Mapping) and "resume" in sessions),
        )
    )
    detail = (
        f"{info.get('name')} {info.get('version')} (ACP v{PROTOCOL_VERSION}): "
        f"{features}; ready in {ready:.1f} s; shut down with exit code {code}"
    )
    checks = [Check("ACP", "ok" if failure is None else "fail", detail)]
    if failure is not None:
        checks.append(Check("ACP models", "fail", failure))
        return checks
    checks.append(_model_check(options or [], model))
    return checks


def _model_check(options: Sequence[Any], model: tuple[str, str] | None) -> Check:
    """What ``configOptions`` offers as models, and whether ``model`` is among them."""
    option = next(
        (
            item
            for item in options
            if isinstance(item, Mapping)
            and (item.get("category") == "model" or item.get("id") == "model")
        ),
        None,
    )
    if option is None:
        status = "fail" if model is not None else "warn"
        return Check("ACP models", status, "the session offers no model option")
    values = _choices(option)
    current = _label(option.get("currentValue"))
    offered = ", ".join(_label(value) for value in values) or "none"
    detail = f"current {current}; offered: {offered}"
    efforts = next(
        (
            item
            for item in options
            if isinstance(item, Mapping) and item.get("id") == "reasoning_effort"
        ),
        None,
    )
    if efforts is not None:
        names = [_label(value) or "provider default" for value in _choices(efforts)]
        detail += f"; reasoning efforts: {', '.join(names)}"
    if model is None:
        return Check("ACP models", "ok", detail)
    wanted = json.dumps(list(model))
    found = any(_same_route(value, model) for value in values) or _same_route(
        option.get("currentValue"), model
    )
    if found:
        return Check("ACP models", "ok", f"{'/'.join(model)} is offered; {detail}")
    return Check(
        "ACP models",
        "fail",
        f"{'/'.join(model)} is not offered, so DevviAgent(model=...) would refuse it "
        f"(value {wanted}); {detail}",
    )


def _choices(option: Mapping[str, Any]) -> list[Any]:
    found: list[Any] = []
    for entry in option.get("options") or []:
        if not isinstance(entry, Mapping):
            continue
        if isinstance(entry.get("options"), list):
            found.extend(
                inner.get("value") for inner in entry["options"] if isinstance(inner, Mapping)
            )
        elif "value" in entry:
            found.append(entry["value"])
    return found


def _label(value: Any) -> str:
    """``'["deepseek-official","deepseek-v4-pro"]'`` -> ``deepseek-official/deepseek-v4-pro``."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return value
        if isinstance(parsed, list) and all(isinstance(part, str) for part in parsed):
            return "/".join(parsed)
        return value
    return "" if value is None else str(value)


def _same_route(value: Any, model: tuple[str, str]) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return json.loads(value) == list(model)
    except ValueError:
        return False


def _row_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    found: list[str] = []
    for row in rows:
        if isinstance(row.get("id"), str):
            found.append(row["id"])
        for inserted in row.get("insert") or []:
            if isinstance(inserted, Mapping) and isinstance(inserted.get("id"), str):
                found.append(inserted["id"])
    return found


def _oneline(error: BaseException) -> str:
    return " ".join(str(error).split("\n")[:2])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m qagents.integrations.devvi",
        description="Tools for running Devvi from qagents.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("doctor", help="check a Devvi setup without a model call")
    check.add_argument("--bin", help=f"the Devvi CLI (default: {ENV_BIN}, else devvi on PATH)")
    check.add_argument("--node", help=f"the node executable (default: {ENV_NODE}, else PATH)")
    check.add_argument(
        "--home", help=f"the Devvi home (default: {ENV_HOME}, else ~/.qagents/devvi-home)"
    )
    check.add_argument(
        "--workspace", help="the directory of the handshakes (default: <home>/qagents/doctor)"
    )
    check.add_argument(
        "--provider",
        help=f"model route: the SDK's route, and one ACP must offer (default {DEFAULT_MODEL[0]})",
    )
    check.add_argument(
        "--model",
        help=f"model id for --provider (SDK default {DEFAULT_MODEL[1]}; ACP: only report)",
    )
    check.add_argument("--no-acp", action="store_true", help="skip the ACP handshake")
    check.add_argument("--no-sdk", action="store_true", help="skip the SDK handshake")
    check.add_argument("--allow-untested-version", action="store_true")
    check.add_argument("--json", action="store_true", help="print the checks as JSON")
    options = parser.parse_args(argv)
    if options.provider and not options.model:
        parser.error("--provider needs --model")
    model = (options.provider or DEFAULT_MODEL[0], options.model) if options.model else None
    checks = asyncio.run(
        doctor(
            devvi_bin=options.bin,
            node=options.node,
            home=options.home,
            workspace=options.workspace,
            model=model,
            sdk=not options.no_sdk,
            acp=not options.no_acp,
            allow_untested_version=options.allow_untested_version,
        )
    )
    if options.json:
        print(json.dumps([asdict(item) for item in checks], indent=2))
    else:
        for item in checks:
            print(item.line())
    return 1 if any(item.status == "fail" for item in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
