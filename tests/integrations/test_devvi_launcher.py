"""The Devvi launcher (test plan 9), profile patches, the DevviAgent options and ``doctor``.

No Devvi is needed: a fake ``node`` (``fakes/fake_node.py``) runs a fake ``bin.js`` that
starts the fake SDK peer, so the ``.js`` -> ``[node, path]`` route runs for real.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from qagents import Agent, Limits
from qagents import prompts as devvi_texts
from qagents import prompts as texts
from qagents.errors import ConfigurationError
from qagents.external import ExternalAgentConfigurationError
from qagents.integrations.devvi import (
    DevviAgent,
    deepseek_models_row,
    devvi_patch,
    launcher,
    mcp_http_row,
    mcp_stdio_row,
    model_route_row,
)
from qagents.integrations.devvi.__main__ import doctor, main
from qagents.integrations.devvi.patches import check_rows, find_secrets, render_patch, write_patch
from qagents.models.testing import ScriptedModel, call

from .fakes.devvi import FAKE_NODE, SDK_FIXTURES, fake_command, starts

WINDOWS = sys.platform == "win32"
PYTHON = os.path.abspath(sys.executable)


def node_command(version: str = "v24.9.0") -> list[str]:
    return [sys.executable, "-I", str(FAKE_NODE), "--node-version", version]


def fake_bin(tmp_path: Path, *args: str, version: str = "0.2.1-rc.3") -> Path:
    """A fake ``bin.js``: JSON that tells ``fake_node.py`` how to start the fake peer."""
    path = tmp_path / "devvi-cli" / "lib" / "bin.js"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": version, "args": list(args)}), encoding="utf-8")
    return path


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def environ(tmp_path: Path, **values: str) -> dict[str, str]:
    base = {
        "PATH": str(tmp_path / "empty-path"),
        "USERPROFILE": str(tmp_path / "user"),
        "HOME": str(tmp_path / "user"),
    }
    for key in ("SYSTEMROOT", "TEMP", "TMP", "PATHEXT", "COMSPEC", "WINDIR", "SYSTEMDRIVE"):
        if key in os.environ:
            base[key] = os.environ[key]
    base.update(values)
    return base


# -- finding the CLI and Node ---------------------------------------------------------------------


def test_a_js_cli_runs_with_node(tmp_path: Path) -> None:
    script = fake_bin(tmp_path)
    command = launcher.resolve_command(script, node=sys.executable, environ=environ(tmp_path))
    assert command.program == (sys.executable, str(script.resolve()))
    assert command.node == (sys.executable,) and command.script == str(script.resolve())
    assert command.source == "argument"
    assert command.argv("--version") == [sys.executable, str(script.resolve()), "--version"]


def test_the_variables_are_read(tmp_path: Path) -> None:
    script = fake_bin(tmp_path)
    env = environ(tmp_path, QAGENTS_DEVVI_BIN=str(script), QAGENTS_NODE=sys.executable)
    command = launcher.resolve_command(environ=env)
    assert command.source == "QAGENTS_DEVVI_BIN" and command.node == (sys.executable,)
    assert command.program[-1] == str(script.resolve())


def test_a_node_command_with_arguments_is_kept(tmp_path: Path) -> None:
    command = launcher.resolve_command(
        fake_bin(tmp_path), node=node_command(), environ=environ(tmp_path)
    )
    assert command.node == tuple(node_command())


def test_an_explicit_command_is_used_as_given(tmp_path: Path) -> None:
    command = launcher.resolve_command(command=["devvi-wrapper", "--x"], environ=environ(tmp_path))
    assert (
        command.program == ("devvi-wrapper", "--x")
        and command.node is None
        and command.source == "command"
    )


NPM_SHIM = r"""@ECHO off
GOTO start
:find_dp0
SET dp0=%~dp0
EXIT /b
:start
SETLOCAL
CALL :find_dp0

IF EXIST "%dp0%\node.exe" (
  SET "_prog=%dp0%\node.exe"
) ELSE (
  SET "_prog=node"
  SET PATHEXT=%PATHEXT:;.JS;=;%
)

endLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & "%_prog%"  "%dp0%\node_modules\@deepseek-ai\dsh\lib\bin.js" %*
"""

PNPM_SHIM = r"""@SETLOCAL
@IF EXIST "%~dp0\node.exe" (
  "%~dp0\node.exe"  "%~dp0\..\@deepseek-ai\dsh\lib\bin.js" %*
) ELSE (
  @SET PATHEXT=%PATHEXT:;.JS;=;%
  node  "%~dp0\..\@deepseek-ai\dsh\lib\bin.js" %*
)
"""


def test_npm_and_pnpm_shims_name_their_script(tmp_path: Path) -> None:
    npm = tmp_path / "npm"
    (npm / "node_modules" / "@deepseek-ai" / "dsh" / "lib").mkdir(parents=True)
    (npm / "node_modules" / "@deepseek-ai" / "dsh" / "lib" / "bin.js").write_text(
        "", encoding="utf-8"
    )
    (npm / "devvi.cmd").write_text(NPM_SHIM, encoding="utf-8")
    assert (
        launcher.parse_cmd_shim(npm / "devvi.cmd")
        == npm / "node_modules" / "@deepseek-ai" / "dsh" / "lib" / "bin.js"
    )
    pnpm = tmp_path / "pnpm" / "bin"
    pnpm.mkdir(parents=True)
    target = tmp_path / "pnpm" / "@deepseek-ai" / "dsh" / "lib" / "bin.js"
    target.parent.mkdir(parents=True)
    target.write_text("", encoding="utf-8")
    (pnpm / "devvi.cmd").write_text(PNPM_SHIM, encoding="utf-8")
    assert launcher.parse_cmd_shim(pnpm / "devvi.cmd") == target
    (pnpm / "other.cmd").write_text("@echo hello\n", encoding="utf-8")
    assert launcher.parse_cmd_shim(pnpm / "other.cmd") is None


@pytest.mark.skipif(not WINDOWS, reason="Windows .cmd shims")
def test_devvi_on_path_on_windows_never_runs_the_cmd_shim(tmp_path: Path) -> None:
    folder = tmp_path / "npm"
    script = folder / "node_modules" / "@deepseek-ai" / "dsh" / "lib" / "bin.js"
    script.parent.mkdir(parents=True)
    script.write_text("", encoding="utf-8")
    (folder / "devvi.cmd").write_text(NPM_SHIM, encoding="utf-8")
    env = environ(tmp_path, PATH=str(folder), QAGENTS_NODE=sys.executable)
    command = launcher.resolve_command(environ=env)
    assert command.source == "PATH" and command.program == (sys.executable, str(script))
    (folder / "broken").mkdir()
    (folder / "broken" / "devvi.cmd").write_text("@echo nothing\n", encoding="utf-8")
    with pytest.raises(ExternalAgentConfigurationError, match="does not say which script"):
        launcher.resolve_command(environ=environ(tmp_path, PATH=str(folder / "broken")))


@pytest.mark.skipif(WINDOWS, reason="POSIX executables")
def test_devvi_on_path_on_posix_runs_with_node(tmp_path: Path) -> None:
    folder = tmp_path / "bin"
    folder.mkdir()
    script = folder / "devvi"
    script.write_text("#!/usr/bin/env node\n", encoding="utf-8")
    script.chmod(0o755)
    command = launcher.resolve_command(
        environ=environ(tmp_path, PATH=str(folder), QAGENTS_NODE=sys.executable)
    )
    assert command.program == (sys.executable, str(script.resolve()))


def test_a_missing_cli_or_node_names_the_variable(tmp_path: Path) -> None:
    with pytest.raises(ExternalAgentConfigurationError, match="QAGENTS_DEVVI_BIN"):
        launcher.resolve_command(environ=environ(tmp_path))
    with pytest.raises(ExternalAgentConfigurationError, match="QAGENTS_DEVVI_BIN"):
        launcher.resolve_command(tmp_path / "nope.js", environ=environ(tmp_path))
    with pytest.raises(ExternalAgentConfigurationError, match="QAGENTS_NODE"):
        launcher.resolve_command(fake_bin(tmp_path), environ=environ(tmp_path))


def test_versions_are_parsed_and_checked() -> None:
    assert launcher.parse_node_version("v24.9.0\n") == (24, 9, 0)
    assert launcher.parse_node_version("v25.1.12") == (25, 1, 12)
    launcher.check_node_version((24, 9, 0))
    with pytest.raises(
        ExternalAgentConfigurationError, match=r"Node 22\.17\.0 is too old.*24\.9\.0"
    ):
        launcher.check_node_version(launcher.parse_node_version("v22.17.0"))
    with pytest.raises(ExternalAgentConfigurationError):
        launcher.parse_node_version("node: not found")
    assert launcher.parse_devvi_version("0.2.1-rc.3\n") == "0.2.1-rc.3"
    with pytest.raises(ExternalAgentConfigurationError, match="not a version") as caught:
        launcher.parse_devvi_version("error: unknown option")
    assert "older than 24.2" not in str(caught.value)
    # Devvi's CLI prints nothing on a Node older than 24.2 (it starts under import.meta.main)
    with pytest.raises(ExternalAgentConfigurationError, match="Node is older than 24.2") as empty:
        launcher.parse_devvi_version("")
    assert (
        str(empty.value).splitlines()[0] == "`devvi --version` printed '', which is not a version."
    )
    launcher.check_devvi_version("0.2.5")
    with pytest.raises(ExternalAgentConfigurationError, match="allow_untested_version"):
        launcher.check_devvi_version("0.3.0")
    launcher.check_devvi_version("0.3.0", allow_untested=True)


# -- home and environment --------------------------------------------------------------------------


def test_the_home_and_both_home_variables(tmp_path: Path) -> None:
    env = environ(tmp_path)
    home, source = launcher.resolve_home(environ=env)
    assert (home, source) == (tmp_path / "user" / ".qagents" / "devvi-home", "default")
    home, source = launcher.resolve_home(environ={**env, "QAGENTS_DEVVI_HOME": str(tmp_path / "h")})
    assert (home, source) == (tmp_path / "h", "QAGENTS_DEVVI_HOME")
    home, source = launcher.resolve_home(tmp_path / "given", environ=env)
    assert source == "home="
    built = launcher.build_environment(tmp_path / "given", environ=env)
    assert built["DEVVI_HOME"] == built["DSH_HOME"] == str(tmp_path / "given")
    assert (
        built["DSH_PERMISSION_MODE"] == "workspace-write" and built["DSH_TELEMETRY_DISABLED"] == "1"
    )
    inherited = {**env, "DEVVI_HOME": str(tmp_path / "a"), "DSH_HOME": str(tmp_path / "b")}
    warnings = launcher.home_conflicts(tmp_path / "given", inherited)
    assert any("differ" in text for text in warnings) and len(warnings) == 3
    assert (
        launcher.home_conflicts(tmp_path / "given", {**env, "DSH_HOME": str(tmp_path / "given")})
        == []
    )
    built = launcher.build_environment(tmp_path / "given", environ=inherited)
    assert built["DEVVI_HOME"] == built["DSH_HOME"] == str(tmp_path / "given")


def test_the_environment_is_an_allowlist(tmp_path: Path) -> None:
    env = environ(
        tmp_path,
        DEEPSEEK_API_KEY="fake-deepseek",
        DEEPSEEK_BASE_URL="http://127.0.0.1:9",
        OPENAI_API_KEY="fake-openai",
        GITHUB_TOKEN="fake-github",
        QCLOUD_API_KEY="fake-qcloud",
        SOME_SECRET="fake",
    )
    built = launcher.build_environment(tmp_path / "h", environ=env)
    assert (
        built["DEEPSEEK_API_KEY"] == "fake-deepseek"
        and built["DEEPSEEK_BASE_URL"] == "http://127.0.0.1:9"
    )
    for name in ("OPENAI_API_KEY", "GITHUB_TOKEN", "QCLOUD_API_KEY", "SOME_SECRET"):
        assert name not in built
    only = launcher.build_environment(tmp_path / "h", passthrough=["QCLOUD_API_KEY"], environ=env)
    assert "QCLOUD_API_KEY" in only and "DEEPSEEK_API_KEY" not in only
    none = launcher.build_environment(
        tmp_path / "h", passthrough=(), extra={"X_FLAG": "1"}, environ=env
    )
    assert not any("KEY" in name or "TOKEN" in name for name in none) and none["X_FLAG"] == "1"
    with pytest.raises(ExternalAgentConfigurationError, match="home="):
        launcher.build_environment(tmp_path / "h", extra={"DSH_HOME": "x"}, environ=env)
    with pytest.raises(ExternalAgentConfigurationError, match="permission_mode="):
        launcher.build_environment(tmp_path / "h", passthrough=["DSH_PERMISSION_MODE"], environ=env)
    with pytest.raises(ExternalAgentConfigurationError, match="permission_mode must be"):
        launcher.build_environment(tmp_path / "h", permission_mode="yolo", environ=env)


# -- patches -------------------------------------------------------------------------------------


def test_patch_files_are_json_and_valid_yaml(tmp_path: Path) -> None:
    rows = devvi_patch(tmp_path / "home")
    assert [row["id"] for row in rows] == [
        "session-persistence-jsonl",
        "session-log-deepseek",
        "session-telemetry-otel",
    ]
    assert rows[0]["config"] == {"root": str(tmp_path / "home" / "sessions"), "compression": "none"}
    assert rows[1]["config"] == {"enabled": False} and rows[2] == {
        "id": "session-telemetry-otel",
        "disabled": True,
    }
    path = write_patch(tmp_path / "patches", rows)
    assert path.name.endswith(".patch.yml")
    text = path.read_text(encoding="utf-8")
    assert json.loads(text) == rows
    yaml = pytest.importorskip("yaml")
    assert yaml.safe_load(text) == rows
    assert write_patch(tmp_path / "patches", rows) == path  # content-addressed: reused
    assert devvi_patch(tmp_path / "home", share_session_log=True)[1]["config"] == {"enabled": True}


def test_credentials_never_go_into_a_patch(tmp_path: Path) -> None:
    secret_rows = [
        [{"id": "llm-x", "config": {"apiKey": "fake-value-123"}}],
        [{"insert": [{"id": "m", "config": {"headers": {"Authorization": "Bearer abcdefghijk"}}}]}],
        [{"id": "x", "config": {"env": {"MY_TOKEN": "abc"}}}],
        [{"id": "x", "config": {"note": "sk-abcdefghijklmnopqrstuvwxyz"}}],
    ]
    for rows in secret_rows:
        with pytest.raises(
            ExternalAgentConfigurationError, match="credential in clear text"
        ) as caught:
            render_patch(rows)
        assert "fake-value" not in str(caught.value) and "abcdefghijk" not in str(caught.value)
    assert find_secrets([{"config": {"apiKey": "v"}}]) == ["rows[0].config.apiKey"]
    assert find_secrets([{"config": {"apiKeyEnv": "QCLOUD_API_KEY", "maxTokens": 5}}]) == []
    with pytest.raises(ExternalAgentConfigurationError, match="written to disk"):
        mcp_stdio_row("tools", sys.executable, env={"SERVICE_TOKEN": "x"})
    with pytest.raises(ExternalAgentConfigurationError, match="written to disk"):
        mcp_http_row("tools", "https://example.com/mcp", headers={"Authorization": "Bearer x"})
    with pytest.raises(ExternalAgentConfigurationError, match="loopback"):
        mcp_http_row("tools", "http://example.com/mcp")
    with pytest.raises(ExternalAgentConfigurationError, match="NAME of an environment variable"):
        model_route_row(
            "qcloud", base_url="https://x", api_key_env="sk-abcdefghijklmnopqrstu", models=["m"]
        )
    with pytest.raises(ConfigurationError, match="credential in clear text"):
        DevviAgent(
            workspace=tmp_path, transport="sdk", patches=[{"id": "x", "config": {"password": "p"}}]
        )
    with pytest.raises(ExternalAgentConfigurationError, match="must be a mapping"):
        check_rows(["not a row"])


def test_mcp_and_model_route_rows() -> None:
    row = mcp_stdio_row(
        "qagents",
        sys.executable,
        ["-m", "tools"],
        env={"LOG_LEVEL": "info"},
        tool_call_timeout_s=30,
    )
    inserted = row["insert"][0]
    assert inserted["id"] == "mcp-qagents" and inserted["name"] == "@deepseek-ai/dsh-mcp-client"
    config = inserted["config"]
    assert (
        config["transport"] == "stdio"
        and config["serverName"] == "qagents"
        and os.path.isabs(config["command"])
    )
    assert (
        config["args"] == ["-m", "tools"]
        and config["toolCallTimeoutMs"] == 30000
        and config["failOnStartupError"]
    )
    http = mcp_http_row("local", "http://127.0.0.1:8123/mcp")["insert"][0]["config"]
    assert http["transport"] == "streamable-http" and http["url"] == "http://127.0.0.1:8123/mcp"
    with pytest.raises(ExternalAgentConfigurationError, match="not valid"):
        mcp_stdio_row("bad name!", sys.executable)
    route = model_route_row(
        "qcloud",
        base_url="https://qcloud.example/v1",
        api_key_env="QCLOUD_API_KEY",
        models=["m1", {"id": "m2", "contextWindow": 131072}],
    )
    assert route == {
        "id": "llm-pi-ai",
        "config": {"providers": {"qcloud": {
            "api": "openai-completions", "baseURL": "https://qcloud.example/v1", "apiKeyEnv": "QCLOUD_API_KEY",
            "models": [{"id": "m1"}, {"id": "m2", "contextWindow": 131072}],
        }}},
    }  # fmt: skip
    assert deepseek_models_row(["smoke-model"]) == {
        "id": "llm-deepseek",
        "config": {"models": [{"id": "smoke-model"}]},
    }
    render_patch([row, route])  # no credential in either


# -- the whole launch, for real, through a fake node -----------------------------------------------


@pytest.mark.devvi_fake
async def test_prepare_launch_probes_node_and_devvi(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    script = fake_bin(tmp_path)
    env = environ(
        tmp_path, DEVVI_HOME=str(tmp_path / "elsewhere"), DEEPSEEK_API_KEY="fake-key-value"
    )
    caplog.set_level(logging.WARNING, logger="qagents.integrations.devvi")
    launch = await launcher.prepare_launch(
        devvi_bin=script, node=node_command("v24.10.1"), home=tmp_path / "home", environ=env
    )
    assert launch.version == "0.2.1-rc.3" and launch.node_version == (24, 10, 1)
    assert launch.home == tmp_path / "home" and launch.home_source == "home="
    assert "overridden" in caplog.text and launch.warnings
    patch = Path(launch.patch_files[0])
    assert patch.parent == tmp_path / "home" / "qagents" / "patches"
    assert launch.rows[0]["config"]["root"] == str(tmp_path / "home" / "sessions")
    assert render_patch(launch.rows) == _read(patch)
    assert launch.argv("sdk") == [
        *node_command("v24.10.1"),
        str(script.resolve()),
        "--profile",
        "sdk",
        "--patch",
        str(patch),
    ]
    assert launch.env["DEEPSEEK_API_KEY"] == "fake-key-value"
    assert "fake-key-value" not in repr(launch)  # the environment may hold credentials
    assert launch.passed_names == sorted(launch.env)


@pytest.mark.devvi_fake
async def test_too_old_node_and_failing_cli_are_refused_with_the_fix(tmp_path: Path) -> None:
    script = fake_bin(tmp_path)
    with pytest.raises(ExternalAgentConfigurationError, match=r"too old for Devvi.*24\.9\.0"):
        await launcher.prepare_launch(
            devvi_bin=script,
            node=node_command("v22.17.0"),
            home=tmp_path / "h",
            environ=environ(tmp_path),
        )
    broken = [*fake_command(None, "--exit-on-start", "2", "--stderr", "Cannot find module")]
    with pytest.raises(ExternalAgentConfigurationError, match="exited with code 2") as caught:
        await launcher.prepare_launch(
            command=broken, home=tmp_path / "h", environ=environ(tmp_path)
        )
    assert "Cannot find module" in str(caught.value)
    newer = fake_bin(tmp_path / "newer", version="0.3.0")
    with pytest.raises(ExternalAgentConfigurationError, match="allow_untested_version"):
        await launcher.prepare_launch(
            devvi_bin=newer, node=node_command(), home=tmp_path / "h", environ=environ(tmp_path)
        )


@pytest.mark.devvi_fake
async def test_a_turn_through_node_and_bin_js_with_an_explicit_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = tmp_path / "record.jsonl"
    script = fake_bin(
        tmp_path,
        "--script",
        str(SDK_FIXTURES / "text-turn.notifications.jsonl"),
        "--record",
        str(record),
    )
    for name, value in {
        "DEEPSEEK_API_KEY": "fake-deepseek",
        "OPENAI_API_KEY": "fake-openai",
        "GITHUB_TOKEN": "fake-gh",
        "DEVVI_HOME": str(tmp_path / "users-own"),
    }.items():
        monkeypatch.setenv(name, value)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    agent = DevviAgent(
        workspace=workspace,
        transport="sdk",
        devvi_bin=script,
        node=node_command(),
        home=tmp_path / "home",
        track_changes=False,
        permission_mode="read-only",
    )
    async with agent:
        result = await agent.run("Reply with exactly: SDK snapshot OK")
    assert result.completed and result.text == "SDK snapshot OK"
    start = starts(record)[0]
    assert start["home"] == {
        "DEVVI_HOME": str(tmp_path / "home"),
        "DSH_HOME": str(tmp_path / "home"),
    }
    assert start["permission_mode"] == "read-only" and start["telemetry_disabled"] == "1"
    assert "DEEPSEEK_API_KEY" in start["env_names"]
    assert "OPENAI_API_KEY" not in start["env_names"] and "GITHUB_TOKEN" not in start["env_names"]
    assert start["cwd"] == str(workspace)
    assert start["argv"][-4:-2] == ["--profile", "sdk"]


@pytest.mark.devvi_fake
async def test_a_setup_failure_makes_the_tool_unavailable(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    agent = DevviAgent(
        workspace=workspace,
        transport="sdk",
        devvi_bin=fake_bin(tmp_path),
        node=node_command("v20.0.0"),
        home=tmp_path / "home",
    )
    lead = Agent(
        ScriptedModel([call("devvi", task="t"), "ok"]),
        tools=[agent.as_tool()],
        limits=Limits(max_turns=4),
    )
    result = await lead.run("go")
    content = [part for message in result.messages for part in message.tool_results][0].content
    assert content.startswith("devvi is unavailable: Node 20.0.0 is too old for Devvi")
    assert "This will not change during this run" in content


# -- the DevviAgent options ------------------------------------------------------------------------


def test_devvi_agent_options_are_checked(tmp_path: Path) -> None:
    assert DevviAgent(workspace=tmp_path).transport == "acp"  # the default
    with pytest.raises(ConfigurationError, match="workspace"):
        DevviAgent(workspace="", transport="sdk")
    with pytest.raises(ConfigurationError, match="transport"):
        DevviAgent(workspace=tmp_path, transport="http")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="provider route, model id"):
        DevviAgent(workspace=tmp_path, transport="sdk", model="deepseek-v4-flash")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="permission_mode"):
        DevviAgent(workspace=tmp_path, transport="sdk", permission_mode="all")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="max_tokens"):
        DevviAgent(workspace=tmp_path, transport="sdk", max_tokens=0)
    with pytest.raises(ConfigurationError, match="on_permission"):
        DevviAgent(workspace=tmp_path, transport="sdk", on_permission="never")  # type: ignore[arg-type]


async def test_devvi_agent_as_a_tool(tmp_path: Path) -> None:
    agent = DevviAgent(workspace=tmp_path, transport="sdk", home=tmp_path / "home")
    tool = agent.as_tool(name="coder", timeout=1800)
    assert tool.name == "coder" and tool.idempotent and not tool.read_only
    assert tool.description.startswith(devvi_texts.DEVVI_DESCRIPTION.format(name="coder"))
    assert tool.description.endswith(texts.EXTERNAL_AGENT_CONTINUE_HINT)
    assert agent.as_tool(description="Mine.").description == "Mine."
    caps = await agent.capabilities()
    assert (
        caps.token_usage
        and caps.client_session_ids
        and caps.resume_sessions
        and not caps.cancel_turn
    )


# -- doctor ---------------------------------------------------------------------------------------


@pytest.mark.devvi_fake
async def test_doctor_reports_every_check(tmp_path: Path) -> None:
    checks = await doctor(
        devvi_bin=fake_bin(tmp_path),
        node=node_command(),
        home=tmp_path / "home",
        workspace=tmp_path / "ws",
        environ=environ(tmp_path, DEEPSEEK_API_KEY="fake"),
    )
    status = {item.name: item.status for item in checks}
    assert status == {
        "home": "ok", "DEVVI_HOME/DSH_HOME": "ok", "environment": "ok", "Devvi CLI": "ok", "Node": "ok",
        "Devvi": "ok", "patch": "ok", "dump-config": "ok", "ACP": "ok", "ACP models": "ok", "SDK": "ok",
    }  # fmt: skip
    detail = {item.name: item.detail for item in checks}
    assert "deepseek-harness-sdk-runtime 0.0.1" in detail["SDK"] and "exit code 0" in detail["SDK"]
    assert detail["Devvi"] == "0.2.1-rc.3" and detail["Node"] == "24.9.0"
    assert "fake" not in detail["environment"]  # names only, never values
    assert "not set: DEEPSEEK_BASE_URL" in detail["environment"]


@pytest.mark.devvi_fake
def test_doctor_command_line_fails_without_a_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("QAGENTS_DEVVI_BIN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    code = main(["doctor", "--home", str(tmp_path / "home"), "--json"])
    printed: list[dict[str, Any]] = json.loads(capsys.readouterr().out)
    assert code == 1
    assert printed[-1]["name"] == "Devvi CLI" and printed[-1]["status"] == "fail"
    assert "QAGENTS_DEVVI_BIN" in printed[-1]["detail"]


# -- review round 2: never a program from the current directory or the workspace -------------------


def plant(folder: Path, name: str) -> Path:
    """An executable file ``name`` (``name.exe`` on Windows) in ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (f"{name}.exe" if WINDOWS else name)
    path.write_bytes(b"MZ planted" if WINDOWS else b"#!/bin/sh\necho planted\n")
    path.chmod(0o755)
    return path


def same(command: tuple[str, ...], path: Path) -> bool:
    """``command`` is just ``path`` (Windows: whatever the case of the extension)."""
    return len(command) == 1 and os.path.normcase(command[0]) == os.path.normcase(str(path))


def test_a_program_in_the_current_directory_is_never_chosen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The coder can write ``node.exe``, ``node.com`` or ``devvi.cmd`` into the workspace,
    and qagents often runs there. Windows (and ``shutil.which``) would search the current
    directory first: the launcher never does, nor for a bare name in the variables."""
    workspace = tmp_path / "workspace"
    for name in ("node", "devvi"):
        plant(workspace, name)
    (workspace / "node.com").write_bytes(b"planted")
    (workspace / "devvi.cmd").write_text(NPM_SHIM, encoding="utf-8")
    (workspace / "bin.js").write_text("", encoding="utf-8")
    monkeypatch.chdir(workspace)
    monkeypatch.delenv("NoDefaultCurrentDirectoryInExePath", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    env = environ(tmp_path)
    assert launcher.locate_program("node", environ=env) is None
    with pytest.raises(ExternalAgentConfigurationError, match="Node.js was not found"):
        launcher.resolve_node(environ=env)
    with pytest.raises(ExternalAgentConfigurationError, match="Node.js was not found"):
        launcher.resolve_node(environ=environ(tmp_path, QAGENTS_NODE="node"))
    with pytest.raises(ExternalAgentConfigurationError, match="Node.js was not found"):
        launcher.resolve_node("node", environ=env)
    with pytest.raises(ExternalAgentConfigurationError, match="The Devvi CLI was not found"):
        launcher.resolve_command(environ=env)
    for configured in ("devvi", "bin.js"):  # bare names: on PATH, never in the cwd
        with pytest.raises(ExternalAgentConfigurationError, match="on PATH"):
            launcher.resolve_command(
                environ=environ(tmp_path, QAGENTS_DEVVI_BIN=configured, QAGENTS_NODE=PYTHON)
            )
    with pytest.raises(ExternalAgentConfigurationError, match="was not found"):
        mcp_stdio_row("tools", "node")
    # A path with a folder part is the caller's explicit choice, relative to the cwd.
    command = launcher.resolve_command(
        environ=environ(tmp_path, QAGENTS_DEVVI_BIN=f".{os.sep}bin.js", QAGENTS_NODE=PYTHON)
    )
    assert command.program == (PYTHON, str((workspace / "bin.js").resolve()))


def test_the_program_found_on_path_is_absolute_and_never_inside_the_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    planted = plant(workspace / "node_modules" / ".bin", "node")
    real = plant(tmp_path / "nodejs", "node")
    both = os.pathsep.join([str(planted.parent), str(real.parent)])
    env = environ(tmp_path, PATH=both)
    assert same(launcher.resolve_node(environ=env, exclude=[workspace]), real)
    assert same(launcher.resolve_node(environ=env), planted)  # without exclude: PATH order
    only_planted = environ(tmp_path, PATH=str(planted.parent))
    with pytest.raises(ExternalAgentConfigurationError, match="inside the agent's workspace"):
        launcher.resolve_node(environ=only_planted, exclude=[workspace])
    script = fake_bin(tmp_path)
    command = launcher.resolve_command(script, environ=env, exclude=[workspace])
    assert command.node is not None and same(command.node, real)
    assert os.path.isabs(command.program[1])


def test_devvi_agent_keeps_its_workspace_out_of_the_program_search(tmp_path: Path) -> None:
    agent = DevviAgent(workspace=tmp_path / "repo", transport="sdk")
    assert agent._launch_options["exclude"] == (str(tmp_path / "repo"),)


async def test_a_cancelled_probe_kills_its_process(tmp_path: Path) -> None:
    from qagents.integrations._process import build_env

    from .fakes.devvi import wait_until_gone

    pid_file = tmp_path / "probe.pid"
    code = "import os, sys, time; open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(60)"
    python = getattr(sys, "_base_executable", None) or sys.executable  # not a venv redirector
    argv = [python, "-I", "-c", code, str(pid_file)]
    probe = asyncio.create_task(launcher.run_probe(argv, env=build_env(), timeout=60))
    for _ in range(1000):
        if pid_file.exists() and pid_file.read_text(encoding="utf-8"):
            break
        await asyncio.sleep(0.02)
    pid = int(pid_file.read_text(encoding="utf-8"))
    probe.cancel()
    with pytest.raises(asyncio.CancelledError):
        await probe
    await wait_until_gone(pid)
    # A timeout still answers (None, "", "") and kills it too.
    pid_file.unlink()
    assert await launcher.run_probe(argv, env=build_env(), timeout=3.0) == (None, "", "")
    await wait_until_gone(int(pid_file.read_text(encoding="utf-8")))
