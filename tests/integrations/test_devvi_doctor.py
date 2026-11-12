"""``python -m qagents.integrations.devvi doctor``: the ACP side (design 3.4), with the fakes.

A fake ``node`` (``fakes/fake_node.py``) runs a fake ``bin.js``: ``--profile acp`` plays an
ACP fixture through ``fakes/fake_devvi_acp.py`` (Devvi's recorded handshake by default),
``--profile sdk`` the SDK fake. No model is called and no Devvi is needed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from qagents.integrations.devvi.__main__ import Check, doctor, main

from .fakes.devvi import ACP_FIXTURES, FAKE_NODE, records

pytestmark = pytest.mark.devvi_fake


def node_command(version: str = "v24.9.0") -> list[str]:
    return [sys.executable, "-I", str(FAKE_NODE), "--node-version", version]


def fake_bin(tmp_path: Path, *, acp_args: list[str] | None = None) -> Path:
    """A fake ``bin.js``: JSON that tells ``fake_node.py`` which fake to run for each profile."""
    path = tmp_path / "devvi-cli" / "lib" / "bin.js"
    path.parent.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {"version": "0.2.1-rc.3", "args": []}
    if acp_args is not None:
        config["acp_args"] = acp_args
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


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


def by_name(checks: list[Check]) -> dict[str, Check]:
    return {item.name: item for item in checks}


def handshake_with(tmp_path: Path, init: dict[str, Any] | None = None, **new: Any) -> Path:
    """Devvi's recorded handshake, with changes to the initialize or session/new answer."""
    lines = (ACP_FIXTURES / "handshake.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    if init is not None:
        rows[2]["result"].update(init)
    rows[4]["result"].update(new)
    path = tmp_path / "handshake-changed.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


async def test_the_acp_handshake_and_the_model_options_are_reported(tmp_path: Path) -> None:
    start = tmp_path / "start.jsonl"
    checks = await doctor(
        devvi_bin=fake_bin(tmp_path, acp_args=["--start-record", str(start)]),
        node=node_command(),
        home=tmp_path / "home",
        environ=environ(tmp_path, DEEPSEEK_API_KEY="fake-secret-value"),
    )
    names = [item.name for item in checks]
    assert names == [
        "home", "DEVVI_HOME/DSH_HOME", "environment", "Devvi CLI", "Node", "Devvi", "patch",
        "dump-config", "ACP", "ACP models", "SDK",
    ]  # fmt: skip
    assert all(item.status == "ok" for item in checks), [item.line() for item in checks]
    found = by_name(checks)
    assert found["dump-config"].detail == "the composed acp and sdk profiles show every row"
    acp = found["ACP"].detail
    assert acp.startswith("deepseek-harness-acp 0.0.1 (ACP v1): MCP over HTTP yes, ")
    assert "image prompts no, session resume yes; ready in " in acp
    assert acp.endswith("shut down with exit code 0")
    assert found["ACP models"].detail == (
        "current deepseek-official/deepseek-v4-flash; offered: "
        "deepseek-official/deepseek-v4-flash, deepseek-official/deepseek-v4-pro"
    )
    assert "fake-secret-value" not in json.dumps([item.detail for item in checks])
    [begun] = [item["start"] for item in records(start)]
    folder = tmp_path / "home" / "qagents" / "doctor"
    assert begun["profile"] == "acp" and begun["cwd"] == str(folder)
    assert begun["home"]["DEVVI_HOME"] == str(tmp_path / "home")


async def test_a_model_that_must_be_offered(tmp_path: Path) -> None:
    common: dict[str, Any] = {
        "devvi_bin": fake_bin(tmp_path),
        "node": node_command(),
        "home": tmp_path / "home",
        "environ": environ(tmp_path),
        "sdk": False,
    }
    offered = by_name(await doctor(model=("deepseek-official", "deepseek-v4-pro"), **common))
    assert offered["ACP models"].status == "ok"
    assert offered["ACP models"].detail.startswith("deepseek-official/deepseek-v4-pro is offered;")
    assert "SDK" not in offered  # --no-sdk
    missing = by_name(await doctor(model=("deepseek-official", "deepseek-v9"), **common))
    assert missing["ACP models"].status == "fail"
    assert "deepseek-official/deepseek-v9 is not offered" in missing["ACP models"].detail


async def test_reasoning_efforts_are_listed(tmp_path: Path) -> None:
    lines = (ACP_FIXTURES / "handshake.jsonl").read_text(encoding="utf-8").splitlines()
    options = json.loads(lines[4])["result"]["configOptions"]
    effort = {
        "id": "reasoning_effort", "category": "thought_level", "type": "select", "currentValue": "",
        "options": [{"value": "", "name": "Provider default"}, {"value": "high", "name": "High"}],
    }  # fmt: skip
    script = handshake_with(tmp_path, configOptions=[*options, effort])
    checks = by_name(
        await doctor(
            devvi_bin=fake_bin(tmp_path, acp_args=["--script", str(script)]),
            node=node_command(),
            home=tmp_path / "home",
            environ=environ(tmp_path),
            sdk=False,
        )
    )
    assert checks["ACP models"].detail.endswith("; reasoning efforts: provider default, high")


async def test_an_acp_handshake_that_fails_is_reported(tmp_path: Path) -> None:
    script = handshake_with(tmp_path, init={"protocolVersion": 2})
    checks = by_name(
        await doctor(
            devvi_bin=fake_bin(tmp_path, acp_args=["--script", str(script)]),
            node=node_command(),
            home=tmp_path / "home",
            environ=environ(tmp_path),
        )
    )
    assert checks["ACP"].status == "fail"
    assert "speaks ACP protocol version 2" in checks["ACP"].detail
    assert "ACP models" not in checks
    assert checks["SDK"].status == "ok"  # the other transport is still checked


async def test_a_session_that_cannot_open_fails_the_model_check(tmp_path: Path) -> None:
    lines = (ACP_FIXTURES / "handshake.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [
        *lines[:4],
        json.dumps(
            {"jsonrpc": "2.0", "id": 2, "error": {"code": -32602, "message": "bad MCP server"}}
        ),
    ]
    script = tmp_path / "refused.jsonl"
    script.write_text("\n".join(rows) + "\n", encoding="utf-8")
    checks = by_name(
        await doctor(
            devvi_bin=fake_bin(tmp_path, acp_args=["--script", str(script)]),
            node=node_command(),
            home=tmp_path / "home",
            environ=environ(tmp_path),
            sdk=False,
        )
    )
    assert checks["ACP"].status == "fail"
    assert checks["ACP models"].status == "fail"
    assert checks["ACP models"].detail == "session/new failed: bad MCP server"


def test_the_command_line_stops_at_a_failing_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("QAGENTS_DEVVI_BIN", str(fake_bin(tmp_path)))
    code = main(
        [
            "doctor", "--home", str(tmp_path / "home"), "--node", sys.executable,
            "--no-sdk", "--json", "--workspace", str(tmp_path / "ws"),
        ]
    )  # fmt: skip
    printed = json.loads(capsys.readouterr().out)
    assert code == 1  # python is not node: "Python 3.13" is too old for Devvi
    assert printed[-1]["name"] == "versions" and printed[-1]["status"] == "fail"
    assert "ACP" not in [item["name"] for item in printed]
    with pytest.raises(SystemExit):
        main(["doctor", "--provider", "deepseek-official"])  # --provider needs --model


def test_the_command_line_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import qagents.integrations.devvi.__main__ as entry

    seen: dict[str, Any] = {}

    async def fake_doctor(**options: Any) -> list[Check]:
        seen.update(options)
        return [Check("ACP", "ok", "fine"), Check("ACP models", "warn", "none")]

    monkeypatch.setattr(entry, "doctor", fake_doctor)
    assert main(["doctor", "--no-sdk", "--model", "deepseek-v4-pro"]) == 0
    assert capsys.readouterr().out.splitlines() == ["[  ok] ACP: fine", "[warn] ACP models: none"]
    assert seen["model"] == ("deepseek-official", "deepseek-v4-pro")
    assert seen["sdk"] is False and seen["acp"] is True
    main(["doctor", "--no-acp"])
    assert seen["model"] is None and seen["acp"] is False
