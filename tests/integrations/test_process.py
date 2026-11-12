"""Child processes: explicit environment, stderr ring buffer, graceful shutdown, tree kill."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pytest

from qagents.external import ExternalAgentConfigurationError
from qagents.integrations import _process
from qagents.integrations._process import (
    BASE_ENV,
    ManagedProcess,
    build_env,
    resolve_command,
)

from .fakes import script

WINDOWS = sys.platform == "win32"


# -- the environment -------------------------------------------------------------------


def test_build_env_is_an_allowlist() -> None:
    source = {
        "PATH": "/bin",
        "HOME": "/home/me",
        "LANG": "C.UTF-8",
        "OPENAI_API_KEY": "sk-not-real",
        "AWS_SECRET_ACCESS_KEY": "not-real",
        "DEEPSEEK_API_KEY": "not-real-either",
    }
    env = build_env(source=source)
    assert env["PATH"] == "/bin" and env["HOME"] == "/home/me" and env["LANG"] == "C.UTF-8"
    assert not any("KEY" in name for name in env)


def test_build_env_passthrough_extra_and_inherit() -> None:
    source = {"PATH": "/bin", "DEEPSEEK_API_KEY": "not-real", "OTHER": "x"}
    env = build_env(
        passthrough=["DEEPSEEK_API_KEY", "NOT_SET"],
        extra={"DSH_TELEMETRY_DISABLED": "1", "PATH": "/custom"},
        source=source,
    )
    assert env == {"PATH": "/custom", "DEEPSEEK_API_KEY": "not-real", "DSH_TELEMETRY_DISABLED": "1"}
    assert build_env(inherit_env=True, source=source) == source


@pytest.mark.skipif(not WINDOWS, reason="Windows environment names ignore case")
def test_build_env_matches_names_without_case_on_windows() -> None:
    env = build_env(
        passthrough=["deepseek_api_key"],
        extra={"Path": "C:/override"},
        source={"PATH": "C:/Windows", "DEEPSEEK_API_KEY": "not-real", "SystemRoot": "C:/Windows"},
    )
    assert env["DEEPSEEK_API_KEY"] == "not-real"
    assert env["SystemRoot"] == "C:/Windows"
    assert [name for name in env if name.upper() == "PATH"] == ["Path"]
    assert env["Path"] == "C:/override"


@pytest.mark.parametrize("bad", [{"": "x"}, {"A=B": "x"}, {"A": 1}])
def test_build_env_refuses_bad_extra(bad: dict) -> None:
    with pytest.raises(ExternalAgentConfigurationError):
        build_env(extra=bad, source={})


def test_resolve_command_errors(tmp_path: Path) -> None:
    with pytest.raises(ExternalAgentConfigurationError, match="non-empty list"):
        resolve_command([], {})
    with pytest.raises(ExternalAgentConfigurationError, match="not found"):
        resolve_command(["definitely-not-a-program-qagents"], {"PATH": str(tmp_path)})
    assert resolve_command([sys.executable, "-V"], {})[0] == sys.executable


def test_resolve_command_never_takes_the_program_from_the_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools, workspace = tmp_path / "tools", tmp_path / "workspace"
    tools.mkdir()
    workspace.mkdir()
    filename = "qagents-fake-agent" + (".exe" if WINDOWS else "")
    for folder in (tools, workspace):  # the agent could put one in its workspace
        (folder / filename).write_text("x")
        (folder / filename).chmod(0o755)
    monkeypatch.chdir(workspace)  # qagents often runs in the workspace
    env = {"PATH": os.pathsep.join([".", "relative", str(tools)]), "PATHEXT": ".COM;.EXE"}
    program, flag = resolve_command(["qagents-fake-agent", "--flag"], env)
    assert os.path.normcase(program) == os.path.normcase(str(tools / filename))
    assert flag == "--flag"
    with pytest.raises(ExternalAgentConfigurationError, match="not found"):
        resolve_command(["qagents-fake-agent"], {"PATH": ".", "PATHEXT": ".EXE"})


@pytest.mark.skipif(not WINDOWS, reason="cmd shims are a Windows matter")
def test_resolve_command_refuses_cmd_shims(tmp_path: Path) -> None:
    shim = tmp_path / "devvi.cmd"
    shim.write_text("@echo off\r\n", encoding="ascii")
    with pytest.raises(ExternalAgentConfigurationError, match="node"):
        resolve_command(["devvi"], {"PATH": str(tmp_path), "PATHEXT": ".COM;.EXE;.CMD"})


def test_a_path_that_does_not_exist_says_no_such_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "nope.exe"
    with pytest.raises(ExternalAgentConfigurationError) as raised:
        resolve_command([str(missing)], {"PATH": str(tmp_path)})
    assert str(raised.value) == f"Cannot start {str(missing)!r}: no such file."
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ExternalAgentConfigurationError, match="no such file"):
        resolve_command(["bin/agent.exe"], {"PATH": str(tmp_path)})
    # A relative path is taken from the current directory and made absolute.
    (tmp_path / "bin").mkdir()
    program = tmp_path / "bin" / ("agent.exe" if WINDOWS else "agent")
    program.write_text("x")
    program.chmod(0o755)
    found = resolve_command([os.path.join("bin", program.name)], {})[0]
    assert os.path.normcase(found) == os.path.normcase(str(program))


@pytest.mark.skipif(not WINDOWS, reason="Windows starts only .exe and .com programs")
def test_a_script_is_refused_on_windows_with_the_interpreter_to_use(tmp_path: Path) -> None:
    script = tmp_path / "agent.js"
    script.write_text("console.log(1)\n", encoding="utf-8")
    for command in ([str(script)], [str(tmp_path / "agent")]):  # Windows would try PATHEXT
        with pytest.raises(ExternalAgentConfigurationError) as raised:
            resolve_command(command, {"PATHEXT": ".COM;.EXE;.JS"})
        message = str(raised.value)
        assert "it is a .js file" in message.lower() and "['node', " in message, message
        assert "WinError" not in message


@pytest.mark.skipif(WINDOWS, reason="the executable bit is a POSIX matter")
def test_a_file_that_is_not_executable_is_refused(tmp_path: Path) -> None:
    script = tmp_path / "agent"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    script.chmod(0o644)
    with pytest.raises(ExternalAgentConfigurationError, match="not executable"):
        resolve_command([str(script)], {})


# -- starting -----------------------------------------------------------------------------


async def test_start_refuses_a_missing_program_and_a_missing_cwd(tmp_path: Path) -> None:
    with pytest.raises(ExternalAgentConfigurationError, match="not found"):
        await ManagedProcess.start(
            ["definitely-not-a-program-qagents"], env={"PATH": str(tmp_path)}
        )
    with pytest.raises(ExternalAgentConfigurationError, match="does not exist"):
        await ManagedProcess.start([sys.executable, "-V"], cwd=tmp_path / "missing")


@pytest.mark.devvi_fake
async def test_the_child_gets_only_the_allowlist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("QAGENTS_TEST_FAKE_API_KEY", "not-a-real-key")
    monkeypatch.setenv("QAGENTS_TEST_PASSED", "yes")
    process = await ManagedProcess.start(
        script("process_tree", "env"),
        env=build_env(passthrough=["QAGENTS_TEST_PASSED"], extra={"QAGENTS_EXTRA": "1"}),
        cwd=tmp_path,
    )
    line = await asyncio.wait_for(process.stdout.readline(), 20)
    await process.close()
    seen = {name.upper(): value for name, value in json.loads(line).items()}
    assert "QAGENTS_TEST_FAKE_API_KEY" not in seen
    assert seen["QAGENTS_TEST_PASSED"] == "yes"
    assert seen["QAGENTS_EXTRA"] == "1"
    allowed = {*BASE_ENV, "PATHEXT", "COMSPEC", "WINDIR", "SYSTEMDRIVE"}
    allowed |= {"QAGENTS_TEST_PASSED", "QAGENTS_EXTRA"}
    # Windows adds a few of its own to every process (and the venv launcher one).
    extras = {name for name in seen if name not in allowed}
    assert extras <= {"__PYVENV_LAUNCHER__", "PYTHONHOME", "PYTHONPATH", "PYTHONUTF8"} | {
        name for name in extras if name.startswith("=")
    }


@pytest.mark.devvi_fake
async def test_stderr_ring_buffer_and_exit_explanation() -> None:
    process = await ManagedProcess.start(
        script("process_tree", "stderr", str(200_000)), stderr_bytes=4096, name="noisy"
    )
    assert await process.wait(20) == 3
    await asyncio.sleep(0.2)  # the reader drains the pipe
    tail = process.stderr_tail(100)
    assert tail.endswith("THE END") and len(tail) <= 100
    assert len(process._stderr) <= 4096
    explained = await process.explain_exit()
    assert explained.startswith("noisy exited with code 3; stderr: ")
    assert process.exit_summary() == "the process exited with code 3"
    assert await process.close() == 3


# -- shutting down --------------------------------------------------------------------------


@pytest.mark.devvi_fake
async def test_graceful_close_on_end_of_stdin() -> None:
    process = await ManagedProcess.start(script("process_tree", "polite"))
    assert (await asyncio.wait_for(process.stdout.readline(), 20)).strip() == b"ready"
    began = time.monotonic()
    assert await process.close(grace=10) == 0
    assert time.monotonic() - began < 10
    assert not process.killed
    assert await process.close() == 0  # safe to repeat


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return -1


async def _beating(path: Path) -> bool:
    before = _size(path)
    await asyncio.sleep(0.4)
    return _size(path) > before


def _alive(pid: int) -> bool:
    if WINDOWS:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel32.OpenProcess(0x00100000 | 0x1000, False, pid)  # SYNCHRONIZE | QUERY
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == 0x102  # WAIT_TIMEOUT: running
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def _start_tree(mode: str, heartbeat: Path) -> tuple[ManagedProcess, int]:
    process = await ManagedProcess.start(script("process_tree", mode, str(heartbeat)))
    process.stdin.write(b"go\n")
    await process.stdin.drain()
    line = await asyncio.wait_for(process.stdout.readline(), 30)
    assert line.startswith(b"ready "), line
    child = int(line.split()[1])
    assert await _beating(heartbeat), "the child should be writing its heartbeat"
    return process, child


@pytest.mark.devvi_fake
async def test_a_stubborn_tree_is_killed_after_the_grace_periods(tmp_path: Path) -> None:
    heartbeat = tmp_path / "heartbeat"
    process, child = await _start_tree("stubborn", heartbeat)
    began = time.monotonic()
    await process.close(grace=0.5, terminate_grace=0.5)
    assert time.monotonic() - began < 15
    assert process.killed
    assert process.returncode is not None
    assert not await _beating(heartbeat), "the child outlived the kill"
    assert not _alive(child)


@pytest.mark.devvi_fake
async def test_children_left_behind_by_a_process_that_exited_are_killed(tmp_path: Path) -> None:
    heartbeat = tmp_path / "heartbeat"
    process, child = await _start_tree("orphan", heartbeat)
    process.stdin.close()  # the parent exits now; the child holds its stdout and stderr open
    assert await process.wait(20) == 0
    assert await _beating(heartbeat), "the orphan keeps running until close()"
    await process.close(grace=0.5, terminate_grace=0.5)
    assert not await _beating(heartbeat), "the orphan outlived close()"
    assert not _alive(child)


@pytest.mark.devvi_fake
async def test_kill_is_immediate_and_takes_the_tree(tmp_path: Path) -> None:
    heartbeat = tmp_path / "heartbeat"
    process, child = await _start_tree("stubborn", heartbeat)
    process.kill()
    assert await process.wait(10) is not None
    assert not await _beating(heartbeat)
    assert not _alive(child)
    await process.close()


@pytest.mark.devvi_fake
async def test_async_with_closes_the_process() -> None:
    async with await ManagedProcess.start(script("process_tree", "polite")) as process:
        assert (await asyncio.wait_for(process.stdout.readline(), 20)).strip() == b"ready"
    assert process.returncode == 0


@pytest.mark.devvi_fake
async def test_a_cancelled_close_kills_the_tree_and_later_closes_finish(tmp_path: Path) -> None:
    heartbeat = tmp_path / "heartbeat"
    process, child = await _start_tree("stubborn", heartbeat)
    closing = asyncio.create_task(process.close(grace=5, terminate_grace=2))
    await asyncio.sleep(0.3)  # close() now waits for the exit
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert process.killed, "a cancelled close() must not leave the tree running"
    assert await process.wait(10) is not None
    assert not await _beating(heartbeat)
    assert not _alive(child)
    began = time.monotonic()
    assert await process.close() is not None  # a later close finishes the same shutdown
    assert time.monotonic() - began < 5


@pytest.mark.devvi_fake
async def test_kill_starts_no_program_and_returns_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    heartbeat = tmp_path / "heartbeat"
    process, child = await _start_tree("stubborn", heartbeat)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"kill() started a program: {args}")

    monkeypatch.setattr(_process.subprocess, "run", forbidden)
    monkeypatch.setattr(_process.subprocess, "Popen", forbidden)
    began = time.perf_counter()
    process.kill()
    assert time.perf_counter() - began < 1.0
    monkeypatch.undo()
    assert await process.wait(10) is not None
    assert not await _beating(heartbeat)
    assert not _alive(child)
    await process.close()


@pytest.mark.devvi_fake
@pytest.mark.skipif(not WINDOWS, reason="Job Objects are a Windows matter")
async def test_without_a_job_object_the_tree_is_still_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_process._WindowsJob, "create_for", classmethod(lambda cls, pid: None))
    heartbeat = tmp_path / "orphan"
    process, child = await _start_tree("orphan", heartbeat)
    assert process._job is None
    await process.close(grace=10, terminate_grace=0.5)  # the parent exits on EOF by itself
    assert process.returncode == 0 and not process.killed
    assert not await _beating(heartbeat), "the child it left behind outlived close()"
    assert not _alive(child)

    heartbeat = tmp_path / "stubborn"
    process, child = await _start_tree("stubborn", heartbeat)
    process.kill()
    assert await process.wait(10) is not None
    assert not await _beating(heartbeat)
    assert not _alive(child)
    await process.close()


def _beating_now(path: Path) -> bool:
    before = _size(path)
    time.sleep(0.4)
    return _size(path) > before


@pytest.mark.devvi_fake
def test_a_process_of_an_ended_event_loop_is_killed_without_waiting(tmp_path: Path) -> None:
    heartbeat = tmp_path / "heartbeat"
    process, child = asyncio.run(_start_tree("stubborn", heartbeat))  # this loop ends now
    assert _beating_now(heartbeat)

    async def close_later() -> float:
        began = time.monotonic()
        await process.close(grace=5, terminate_grace=2)
        return time.monotonic() - began

    took = asyncio.run(close_later())
    assert took < 3, f"close() waited {took:.1f} s for a loop that has ended"
    assert process.killed
    assert not _beating_now(heartbeat)
    assert not _alive(child)


@pytest.mark.devvi_fake
def test_processes_never_closed_are_killed_when_python_exits(tmp_path: Path) -> None:
    heartbeat = tmp_path / "heartbeat"
    process, child = asyncio.run(_start_tree("stubborn", heartbeat))
    assert process in _process._LIVE
    _process._kill_live_processes()  # what the atexit hook runs
    assert not _beating_now(heartbeat)
    assert not _alive(child)
    asyncio.run(process.close())
    assert process not in _process._LIVE


def test_exit_summary_reads_the_prompts_module() -> None:
    from qagents import prompts as texts

    assert texts.EXTERNAL_REASON_PROCESS_EXITED.format(code=3) == "the process exited with code 3"
