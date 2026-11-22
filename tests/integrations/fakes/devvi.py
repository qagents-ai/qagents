"""Helpers for the Devvi tests: fixture paths, the fake CLI's command, and scripts.

`sdk_agent` builds a `DevviAgent` on the SDK transport that starts `fake_devvi_sdk.py`
with a script and a record file. Scripts are lists of turns; each turn is a list of
notifications (`notification`, `status`) and directives (``{"$hang": True}``, ...), made
with the builders below. `turn` wraps a body in what every Devvi turn has: the receipt of
the prompt, ``running``, ``turn/start``, ``turn/end`` and ``idle``.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parents[1] / "fixtures" / "devvi"
SDK_FIXTURES = FIXTURES / "sdk"
FAKE_HOME = FIXTURES / "home"
FAKE_DEVVI = HERE / "fake_devvi_sdk.py"
FAKE_NODE = HERE / "fake_node.py"
ROOT = "{{session:1}}"
CHILD = "{{session:2}}"


def fake_command(
    script: str | Path | None = None, *options: str, record: Path | None = None
) -> list[str]:
    """``[python, -I, fake_devvi_sdk.py, --script ..., --record ..., *options]``."""
    command = [sys.executable, "-I", str(FAKE_DEVVI)]
    if script is not None:
        command += ["--script", str(script)]
    if record is not None:
        command += ["--record", str(record)]
    return [*command, *options]


def records(path: Path) -> list[dict[str, Any]]:
    """What the fake received (and its start lines), in order."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def methods(path: Path) -> list[str]:
    return [item["method"] for item in records(path) if "method" in item]


def requests(path: Path, method: str) -> list[dict[str, Any]]:
    return [item for item in records(path) if item.get("method") == method]


def starts(path: Path) -> list[dict[str, Any]]:
    return [item["start"] for item in records(path) if "start" in item]


def answers(path: Path) -> list[Any]:
    """Results the client sent back for the fake's own requests (approvals)."""
    return [
        item.get("result") for item in records(path) if "method" not in item and "start" not in item
    ]


# --- script builders -------------------------------------------------------------------------


def notification(
    type_: str, data: dict[str, Any] | None = None, *, session: str = ROOT, seq: int = 0
) -> dict[str, Any]:
    event = {"type": type_, "seq": seq, "time": 0, "data": data or {}}
    return {"method": "session.event", "params": {"sessionId": session, "event": event}}


def status(value: str, *, session: str = ROOT) -> dict[str, Any]:
    return {"method": "session.status", "params": {"sessionId": session, "status": value}}


def receipt(
    text: str = "Do the task.", *, token: str = "{{message:1}}", session: str = ROOT
) -> dict[str, Any]:
    inserted = [
        {
            "content": [{"type": "text", "text": text}],
            "source": {"kind": "user"},
            "role": "user",
            "id": token,
        }
    ]
    return notification(
        "agent/inbox/spliced",
        {"target": "next-turn", "start": 0, "inserted": inserted},
        session=session,
    )


def assistant(
    text: str = "",
    *,
    usage: dict[str, int] | None = None,
    reasoning: str | None = None,
    calls: Iterable[tuple[str, str, dict[str, Any]]] = (),
    step: int = 1,
    session: str = ROOT,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if reasoning:
        content.append({"type": "reasoning", "text": reasoning})
    if text:
        content.append({"type": "text", "text": text})
    for call_id, name, arguments in calls:
        content.append(
            {"type": "tool-call", "id": call_id, "name": name, "arguments": json.dumps(arguments)}
        )
    data: dict[str, Any] = {
        "turn": 1,
        "step": step,
        "message": {
            "role": "assistant",
            "content": content,
            "source": {
                "kind": "model",
                "provider": "deepseek-official",
                "model": "deepseek-v4-flash",
            },
            "id": f"m-{step}",
        },
        "stream": [],
    }
    if usage is not None:
        data["usage"] = usage
    return notification("assistant/message", data, session=session)


def tool_call(
    call_id: str, name: str, arguments: dict[str, Any] | str, *, session: str = ROOT
) -> dict[str, Any]:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return notification(
        "tool/call",
        {"turn": 1, "step": 1, "callId": call_id, "name": name, "arguments": raw},
        session=session,
    )


def tool_result(
    call_id: str, text: str, *, is_error: bool = False, error: dict[str, str] | None = None
) -> dict[str, Any]:
    message = {
        "role": "tool",
        "toolCallId": call_id,
        "content": [{"type": "text", "text": text}],
        "isError": is_error,
        "id": f"r-{call_id}",
    }
    data: dict[str, Any] = {"turn": 1, "step": 1, "message": message}
    if error is not None:
        data["error"] = error
    return notification("tool/result", data)


def step(number: int = 1) -> dict[str, Any]:
    return notification("step/start", {"turn": 1, "step": number})


def turn(
    *body: dict[str, Any],
    prompt: str = "Do the task.",
    number: int = 1,
    end: dict[str, Any] | None = None,
    token: str = "{{message:1}}",
    close: bool = True,
) -> list[dict[str, Any]]:
    """A whole turn: receipt, running, turn/start, the body, turn/end, idle (``close``)."""
    items = [
        receipt(prompt, token=token),
        status("running"),
        notification("turn/start", {"turn": number}),
        *body,
    ]
    if close:
        items += [
            notification("turn/end", {"turn": number, "reason": end or {"kind": "completed"}}),
            status("idle"),
        ]
    return items


def write_script(path: Path, *turns: list[dict[str, Any]]) -> Path:
    """Write turns as a fake script (turns separated by ``{"$turn": true}``)."""
    lines: list[str] = []
    for index, items in enumerate(turns):
        if index:
            lines.append(json.dumps({"$turn": True}))
        lines.extend(json.dumps(item) for item in items)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def fixture_lines(name: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (SDK_FIXTURES / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --- agents ---------------------------------------------------------------------------------------


def sdk_agent(
    tmp_path: Path,
    script: str | Path | None,
    *fake_options: str,
    record: Path | None = None,
    **options: Any,
) -> tuple[Any, Path]:
    """A `DevviAgent(transport="sdk")` on the fake; returns it and its record file."""
    from qagents.integrations.devvi import DevviAgent

    record = record or tmp_path / "record.jsonl"
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    if script is not None and not Path(script).is_absolute():
        script = SDK_FIXTURES / script
    options.setdefault("home", tmp_path / "devvi-home")
    options.setdefault("track_changes", False)
    agent = DevviAgent(
        workspace=workspace,
        transport="sdk",
        command=fake_command(script, *fake_options, record=record),
        **options,
    )
    return agent, record


async def wait_for_log_text(root: Path, text: str, timeout: float = 15.0) -> Path:
    """Wait until a session log under ``root`` contains ``text``; return its path."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        found = _log_with(root, text)
        if found is not None:
            return found
        await asyncio.sleep(0.05)
    raise TimeoutError(f"no session log under {root} contains {text!r}")


def _log_with(root: Path, text: str) -> Path | None:
    for path in root.rglob("session.v4.jsonl*"):
        try:
            if text in path.read_text(encoding="utf-8"):
                return path
        except OSError:
            pass
    return None


def process_alive(pid: int) -> bool:
    """Whether a process with this pid still runs."""
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        try:
            kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        finally:
            kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


async def wait_until_gone(pid: int, timeout: float = 10.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while process_alive(pid):
        if loop.time() > deadline:
            raise AssertionError(f"process {pid} is still running")
        await asyncio.sleep(0.05)


# --- the ACP transport ---------------------------------------------------------------------------

FAKE_DEVVI_ACP = HERE / "fake_devvi_acp.py"
ACP_FIXTURES = FIXTURES / "acp"


def acp_command(
    script: str | Path | None = None,
    *options: str,
    log: Path | None = None,
    start: Path | None = None,
) -> list[str]:
    """``[python, -I, fake_devvi_acp.py, --script ..., --log ..., --start-record ..., *options]``.

    A relative ``script`` names a fixture of ``tests/fixtures/devvi/acp`` (``"handshake"``).
    """
    command = [sys.executable, "-I", str(FAKE_DEVVI_ACP)]
    if script is not None:
        path = Path(script)
        if not path.is_absolute():
            path = ACP_FIXTURES / f"{script}.jsonl"
        command += ["--script", str(path)]
    if log is not None:
        command += ["--log", str(log)]
    if start is not None:
        command += ["--start-record", str(start)]
    return [*command, *options]


def acp_agent(
    tmp_path: Path,
    script: str | Path | None,
    *fake_options: str,
    **options: Any,
) -> Any:
    """A `DevviAgent` (ACP, the default transport) on the fake ACP peer.

    The fake writes what the client sent to ``tmp_path / "client.jsonl"`` and its start
    line to ``tmp_path / "start.jsonl"``; the workspace is ``tmp_path / "workspace"``.
    """
    from qagents.integrations.devvi import DevviAgent

    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    options.setdefault("home", tmp_path / "devvi-home")
    options.setdefault("track_changes", False)
    options.setdefault("shutdown_timeout_s", 2.0)
    options.setdefault("name", "coder")
    command = acp_command(
        script, *fake_options, log=tmp_path / "client.jsonl", start=tmp_path / "start.jsonl"
    )
    return DevviAgent(workspace=options.pop("workspace", workspace), command=command, **options)


def sent(tmp_path: Path) -> list[dict[str, Any]]:
    """What the client sent to the fake ACP peer, in order."""
    return records(tmp_path / "client.jsonl")


def sent_methods(tmp_path: Path) -> list[str]:
    return [message.get("method", "<answer>") for message in sent(tmp_path)]
