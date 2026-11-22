"""Transport setup and reply text stays replaceable through qagents.prompts."""

from __future__ import annotations

import asyncio
import sys

import pytest

from qagents import Agent, prompts
from qagents.external import ExternalAgentConfigurationError, ExternalAgentTimeoutError
from qagents.integrations import _process
from qagents.integrations.acp import ACPAgent
from qagents.models.testing import ScriptedModel, call

from .fakes.streams import RawSide


@pytest.mark.parametrize(
    "case",
    [
        "env_name",
        "env_value",
        "empty",
        "missing_file",
        "missing_program",
        "shim",
        "script",
        "extensionless",
        "not_executable",
    ],
)
def test_process_configuration_uses_prompt_constants(monkeypatch, case):
    names = (
        "ENV_NAME",
        "ENV_VALUE",
        "COMMAND_INVALID",
        "FILE_MISSING",
        "PROGRAM_MISSING",
        "SHIM",
        "WINDOWS_FILE",
        "NOT_EXECUTABLE",
    )
    for name in names:
        monkeypatch.setattr(prompts, "EXTERNAL_PROCESS_" + name, name, raising=False)
    monkeypatch.setattr(_process, "_existing_file", lambda *_: "fake.js")
    monkeypatch.setattr(_process, "find_program", lambda *_, **__: None)
    expected = {
        "env_name": "ENV_NAME",
        "env_value": "ENV_VALUE",
        "empty": "COMMAND_INVALID",
        "missing_file": "FILE_MISSING",
        "missing_program": "PROGRAM_MISSING",
        "shim": "SHIM",
        "script": "WINDOWS_FILE",
        "extensionless": "WINDOWS_FILE",
        "not_executable": "NOT_EXECUTABLE",
    }[case]
    with pytest.raises(ExternalAgentConfigurationError) as caught:
        if case == "env_name":
            _process.build_env(extra={"A=B": "value"}, source={})
        elif case == "env_value":
            _process.build_env(extra={"NAME": 1}, source={})
        elif case == "empty":
            _process.resolve_command([], {})
        elif case == "missing_program":
            _process.resolve_command(["missing"], {"PATH": ""})
        else:
            found = {
                "missing_file": None,
                "shim": "fake.cmd",
                "script": "fake.js",
                "extensionless": "fake",
                "not_executable": "fake",
            }[case]
            monkeypatch.setattr(_process, "_existing_file", lambda *_: found)
            monkeypatch.setattr(_process, "_WINDOWS", case != "not_executable")
            monkeypatch.setattr(_process.os, "access", lambda *_: False)
            _process.resolve_command(["./fake"], {})
    assert str(caught.value) == expected


@pytest.mark.parametrize("case", ["cwd", "event_loop", "os_error"])
async def test_process_start_errors_use_prompt_constants(monkeypatch, tmp_path, case):
    for name in ("CWD", "NO_SUBPROCESSES", "START_FAILED"):
        monkeypatch.setattr(prompts, "EXTERNAL_PROCESS_" + name, name, raising=False)

    async def fail(*args, **kwargs):
        raise NotImplementedError if case == "event_loop" else OSError("cannot spawn")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail)
    with pytest.raises(ExternalAgentConfigurationError) as caught:
        await _process.ManagedProcess.start(
            [sys.executable], cwd=tmp_path / "missing" if case == "cwd" else tmp_path
        )
    assert (
        str(caught.value)
        == {
            "cwd": "CWD",
            "event_loop": "NO_SUBPROCESSES",
            "os_error": "START_FAILED",
        }[case]
    )


async def test_process_setup_text_reaches_the_calling_model(monkeypatch, tmp_path):
    monkeypatch.setattr(
        prompts, "EXTERNAL_PROCESS_PROGRAM_MISSING", "No executable {program}", raising=False
    )
    coder = ACPAgent(["qagents-missing-executable-00"], workspace=tmp_path, name="coder")
    model = ScriptedModel([call("coder", task="go"), "done"])
    try:
        await Agent(model, tools=[coder.as_tool()]).run("go")
    finally:
        await coder.aclose()
    content = model.requests[1].messages[-1].tool_results[0].content
    assert content.startswith("coder is unavailable: No executable qagents-missing-executable-00.")


@pytest.mark.parametrize("case", ["missing", "invalid", "broken", "bad_result"])
async def test_jsonrpc_error_replies_use_prompt_constants(monkeypatch, case):
    for name in ("METHOD_NOT_FOUND", "INVALID_REQUEST", "INTERNAL_ERROR", "BAD_RESULT"):
        monkeypatch.setattr(prompts, "EXTERNAL_RPC_" + name, name, raising=False)
    side = RawSide()

    async def broken(params):
        raise ValueError("private detail")

    async def bad_result(params):
        return object()

    side.peer.on_request("broken", broken)
    side.peer.on_request("bad_result", bad_result)
    side.peer.start()
    try:
        side.send({"jsonrpc": "2.0", "id": 1, "method": 42 if case == "invalid" else case})
        answer = await side.receive()
        assert (
            answer["error"]["message"]
            == {
                "missing": "METHOD_NOT_FOUND",
                "invalid": "INVALID_REQUEST",
                "broken": "INTERNAL_ERROR",
                "bad_result": "BAD_RESULT",
            }[case]
        )
    finally:
        await side.peer.aclose()


async def test_jsonrpc_timeout_uses_prompt_constant(monkeypatch):
    monkeypatch.setattr(
        prompts, "EXTERNAL_RPC_TIMEOUT", "{name} waited for {method}", raising=False
    )
    side = RawSide()
    side.peer.start()
    try:
        with pytest.raises(ExternalAgentTimeoutError) as caught:
            await side.peer.request("slow", timeout=0.01)
        assert str(caught.value) == f"{side.peer.name} waited for slow"
    finally:
        await side.peer.aclose()


@pytest.mark.parametrize("explicit", [False, True])
def test_process_search_location_uses_prompt_constant(monkeypatch, explicit):
    monkeypatch.setattr(prompts, "EXTERNAL_PROCESS_PATH_GIVEN", "child path")
    monkeypatch.setattr(prompts, "EXTERNAL_PROCESS_PATH_DEFAULT", "default path")
    monkeypatch.setattr(_process, "find_program", lambda *_, **__: None)
    with pytest.raises(ExternalAgentConfigurationError) as caught:
        _process.resolve_command(["missing"], {"PATH": ""} if explicit else {})
    assert f"on {'child path' if explicit else 'default path'}." in str(caught.value)


@pytest.mark.parametrize("suffix", ["", ".js"])
def test_process_file_description_uses_prompt_constant(monkeypatch, suffix):
    monkeypatch.setattr(prompts, "EXTERNAL_PROCESS_FILE_KIND", "script {suffix}")
    monkeypatch.setattr(prompts, "EXTERNAL_PROCESS_EXTENSIONLESS", "extensionless")
    monkeypatch.setattr(_process, "_WINDOWS", True)
    monkeypatch.setattr(_process, "_existing_file", lambda *_: "fake" + suffix)
    with pytest.raises(ExternalAgentConfigurationError) as caught:
        _process.resolve_command(["./fake"], {})
    assert f"it is {'script .js' if suffix else 'extensionless'}," in str(caught.value)


async def test_jsonrpc_cancel_reply_uses_prompt_constant(monkeypatch):
    monkeypatch.setattr(prompts, "EXTERNAL_RPC_REQUEST_CANCELLED", "cancelled here")
    side = RawSide(cancel_method="$/cancel_request")
    started = asyncio.Event()

    async def wait(params):
        started.set()
        await asyncio.Event().wait()

    side.peer.on_request("wait", wait)
    side.peer.start()
    try:
        side.send({"jsonrpc": "2.0", "id": 3, "method": "wait"})
        await asyncio.wait_for(started.wait(), 2)
        side.send({"jsonrpc": "2.0", "method": "$/cancel_request", "params": {"requestId": 3}})
        assert (await side.receive())["error"]["message"] == "cancelled here"
    finally:
        await side.peer.aclose()
