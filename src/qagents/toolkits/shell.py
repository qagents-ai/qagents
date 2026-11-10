"""A shell tool.

It runs commands on the machine the agent runs on, with the agent process's
own rights. That is not a sandbox. The tool always asks for approval unless a
permission rule allows it; for untrusted input, run the agent inside a
container or VM and let that be the boundary.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from ..errors import ToolError
from ..tools import Tool, tool


def shell_tool(
    cwd: str | Path,
    *,
    timeout: float = 60.0,
    max_output_chars: int = 30_000,
    env: Mapping[str, str] | None = None,
) -> Tool:
    """A ``run_command`` tool that runs in ``cwd``.

    ``env`` replaces the inherited environment when given; use it to keep
    secrets of the agent process away from commands. A command that exceeds
    ``timeout`` is stopped together with the processes it started.
    """
    directory = Path(cwd).resolve()
    if not directory.is_dir():
        raise ValueError(f"{directory} is not a directory")

    @tool(requires_approval=True, timeout=timeout + 15)
    async def run_command(command: str) -> str:
        """Run a shell command in the workspace directory and return its output.

        Args:
            command: The command line to run.
        """
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(directory),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                stdin=asyncio.subprocess.DEVNULL,
                env=dict(env) if env is not None else os.environ.copy(),
                # Its own process group, so a timeout can stop the whole tree.
                start_new_session=sys.platform != "win32",
            )
        except OSError as error:
            raise ToolError(f"The command could not be started: {error}") from error
        try:
            raw, _ = await asyncio.wait_for(process.communicate(), timeout)
        except TimeoutError:
            _kill_tree(process.pid)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(process.communicate(), 5)  # reap it and close the pipes
            raise ToolError(f"The command was stopped after {timeout:g} seconds.") from None
        except asyncio.CancelledError:
            _kill_tree(process.pid)
            raise
        output = raw.decode("utf-8", errors="replace")
        if len(output) > max_output_chars:
            half = max_output_chars // 2
            omitted = len(output) - max_output_chars
            output = f"{output[:half]}\n[... {omitted} characters omitted ...]\n{output[-half:]}"
        return f"exit code {process.returncode}\n{output}".rstrip()

    return run_command


def _kill_tree(pid: int) -> None:
    """Stop a process and everything it started."""
    if sys.platform == "win32":
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                capture_output=True,
                timeout=10,
                check=False,
            )
    else:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pid, signal.SIGKILL)
