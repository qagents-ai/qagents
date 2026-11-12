"""Child processes for external agents: explicit environment, stderr tail, tree kill.

An external agent (Devvi, any ACP agent) is a separate program that qagents
starts and talks to over stdin and stdout. This module starts it and makes
sure nothing of it is left behind:

    env = build_env(passthrough=["DEEPSEEK_API_KEY"], extra={"DSH_TELEMETRY_DISABLED": "1"})
    process = await ManagedProcess.start(["node", "C:/tools/devvi/bin.js"], env=env, cwd=repo)
    peer = JsonRpcPeer.for_process(process)     # qagents.integrations._jsonrpc
    ...
    await process.close()                       # close stdin, wait, terminate, kill the tree

* **Environment.** The child gets an allowlist (`BASE_ENV` plus the names you
  pass), never the whole parent environment unless you ask for it with
  ``inherit_env=True``. Secrets of the parent process stay with it.
* **stderr.** It is read all the time (a full pipe would block the child) into
  a ring buffer of the last ``stderr_bytes`` bytes. `ManagedProcess.stderr_tail`
  and `ManagedProcess.explain_exit` put it into error messages for developers.
  It is never sent to a model.
* **Shutdown.** `ManagedProcess.close` closes stdin and waits ``grace`` seconds
  (most agents exit on EOF), then terminates, waits ``terminate_grace`` seconds,
  then kills the whole process tree. Processes the child started are killed
  too, also when the child itself already exited and left them behind. A
  ``close()`` that is cancelled while it waits kills the tree at once, and a
  process that belongs to an event loop that has ended (``asyncio.run`` called
  again) is killed without waiting. Processes still running when Python exits
  are killed then.
* **Process tree.** On POSIX the child leads a new session
  (``start_new_session=True``), so its process group is signalled. On Windows
  the child runs in a Job Object created with kill-on-close, so the job takes
  every descendant down, at once. When Windows refuses the job (rare), the
  tree is found from a process snapshot instead: by parent process id, and
  checked by start time so that a reused process id is never mistaken for a
  child. It is recorded before stdin is closed, so children that a parent
  leaves behind when it exits are still found. Windows has no reliable polite
  signal for a console process without a window, so there "terminate" is
  skipped and the tree is killed after the stdin grace period. The child gets
  ``CREATE_NO_WINDOW`` (no console window pops up) and
  ``CREATE_NEW_PROCESS_GROUP`` (Ctrl+C in your terminal does not reach it).
  Killing never blocks the event loop for long: it starts no other program.

Windows runs asyncio subprocesses only on the Proactor event loop, the
default since Python 3.8; a Selector loop is refused with a clear error.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import logging
import os
import signal
import subprocess
import sys
import weakref
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Self

from .. import prompts
from ..external import ExternalAgentConfigurationError

logger = logging.getLogger("qagents.integrations.process")

BASE_ENV: tuple[str, ...] = ("PATH", "SYSTEMROOT", "TEMP", "TMP", "USERPROFILE", "HOME", "LANG")
"""Variables every child gets from the parent environment, when they are set.

They hold no secrets and are what Node, git and the shells need to work."""

WINDOWS_BASE_ENV: tuple[str, ...] = ("PATHEXT", "COMSPEC", "WINDIR", "SYSTEMDRIVE")
"""Also passed through on Windows: needed to find and run programs and shells."""

STDERR_BYTES = 64 * 1024
"""Default size of the stderr ring buffer."""

STDOUT_LIMIT = 1024 * 1024
"""Read buffer size of the child's stdout stream (not a line limit: `JsonRpcPeer` has its own)."""

_WINDOWS = sys.platform == "win32"


def build_env(
    *,
    passthrough: Iterable[str] = (),
    extra: Mapping[str, str] | None = None,
    inherit_env: bool = False,
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The environment for a child process: an allowlist, not the parent's whole environment.

        env = build_env(passthrough=["DEEPSEEK_API_KEY"], extra={"DSH_HOME": home})

    Args:
        passthrough: Names of variables to copy from ``source`` when they are
            set, besides `BASE_ENV` (and `WINDOWS_BASE_ENV` on Windows). On
            Windows names match without regard to case, as Windows does.
        extra: Variables set explicitly. They win over copied ones. A value
            of ``None`` is not allowed: leave the name out instead.
        inherit_env: Start from the whole of ``source``. Only when you need it:
            every secret of the parent process then reaches the child.
        source: Where values are copied from; ``os.environ`` by default.

    Raises:
        ExternalAgentConfigurationError: A name or a value in ``extra`` is not
            a string, or a name is empty or contains ``=``.
    """
    environ = os.environ if source is None else source
    names = [*BASE_ENV, *(WINDOWS_BASE_ENV if _WINDOWS else ()), *passthrough]
    env: dict[str, str] = dict(environ) if inherit_env else {}
    if not inherit_env:
        folded = {key.upper(): key for key in environ} if _WINDOWS else None
        for name in names:
            key = folded.get(name.upper()) if folded is not None else name
            if key is not None and key in environ:
                env[key] = environ[key]
    for name, value in (extra or {}).items():
        if not isinstance(name, str) or not name or "=" in name:
            raise ExternalAgentConfigurationError(
                f"Environment variable name {name!r} is not valid: use a non-empty string "
                "without '='."
            )
        if not isinstance(value, str):
            raise ExternalAgentConfigurationError(
                f"Environment variable {name} must be a string, not {type(value).__name__}."
            )
        if _WINDOWS:  # one entry per name: an explicit value replaces a copied one
            for key in [key for key in env if key.upper() == name.upper()]:
                del env[key]
        env[name] = value
    return env


def resolve_command(command: Sequence[str], env: Mapping[str, str]) -> list[str]:
    """``command`` with its program resolved on the child's ``PATH``.

    Windows starts only real executables (``.exe``, ``.com``) reliably: a
    ``.cmd`` or ``.bat`` shim (what ``npm install -g`` creates) would run
    through ``cmd.exe`` with its own quoting rules, and a script (``.js``,
    ``.py``) cannot be started at all. Both are refused with the fix: run the
    script with its interpreter (``["node", ".../bin.js"]``).

    A bare program name is looked up in the absolute ``PATH`` entries only,
    never in the current directory (Windows would look there first, and
    qagents often runs in the agent's workspace, where the agent can put a
    ``node.exe`` of its own) and never in relative entries. A path (absolute,
    or relative with a directory part) is used as given.

    Raises:
        ExternalAgentConfigurationError: The command is empty, the program is
            not found (a path: no such file), it is not executable, or on
            Windows it is not an ``.exe`` or ``.com`` file.
    """
    if not command or not all(isinstance(part, str) for part in command):
        raise ExternalAgentConfigurationError(
            f"The command must be a non-empty list of strings, not {command!r}."
        )
    program = command[0]
    path = next((value for key, value in env.items() if key.upper() == "PATH"), None)
    pathext = next((value for key, value in env.items() if key.upper() == "PATHEXT"), None)
    if os.path.isabs(program) or os.path.dirname(program):  # a path the caller chose
        found = _existing_file(program, pathext)
        if found is None:
            raise ExternalAgentConfigurationError(f"Cannot start {program!r}: no such file.")
    else:
        searched = find_program(program, path, pathext=pathext)
        if searched is None:
            where = "the PATH given to the process" if path is not None else "PATH"
            raise ExternalAgentConfigurationError(
                f"Cannot start {program!r}: it was not found on {where}. Give its absolute "
                "path, or add its directory to PATH."
            )
        found = searched
    suffix = Path(found).suffix
    if _WINDOWS and suffix.lower() in (".cmd", ".bat"):
        raise ExternalAgentConfigurationError(
            f"Cannot start {found!r}: it is a {suffix} shim, which Windows runs "
            "through cmd.exe. Run the script it wraps instead, for example "
            "['node', '<path to the package>/bin.js']."
        )
    if _WINDOWS and suffix.lower() not in _WINDOWS_PROGRAMS:
        kind = f"a {suffix} file" if suffix else "a file without an extension"
        raise ExternalAgentConfigurationError(
            f"Cannot start {found!r}: it is {kind}, and Windows starts only .exe and .com "
            "programs. Run it with its interpreter, for example ['node', "
            f"{found!r}] for a .js file."
        )
    if not _WINDOWS and not os.access(found, os.X_OK):
        raise ExternalAgentConfigurationError(
            f"Cannot start {found!r}: it is not executable. Make it executable (chmod +x), or "
            "run it with its interpreter, for example ['node', '<path>/bin.js']."
        )
    return [found, *command[1:]]


_WINDOWS_PROGRAMS = (".exe", ".com")
"""Suffixes of files Windows can start as a program (CreateProcess)."""


def _existing_file(program: str, pathext: str | None) -> str | None:
    """The absolute path of ``program`` (a path, relative to the current directory) when it
    is a file; on Windows a path without an extension is tried with the ``PATHEXT``
    extensions first, as Windows does."""
    full = os.path.abspath(program)
    if _WINDOWS and not Path(full).suffix:
        source = pathext or os.environ.get("PATHEXT") or ".COM;.EXE;.BAT;.CMD"
        for extension in (ext for ext in source.split(os.pathsep) if ext):
            if os.path.isfile(full + extension):
                return full + extension
    return full if os.path.isfile(full) else None


def find_program(
    name: str,
    path: str | None = None,
    *,
    pathext: str | None = None,
    exclude: Sequence[str | Path] = (),
) -> str | None:
    """The absolute path of program ``name`` on ``path`` (``os.environ["PATH"]`` by default).

    Only absolute entries are searched, in order, and none inside a folder of
    ``exclude``. Unlike ``shutil.which`` (and Windows itself), the current
    directory is never searched. On Windows the extensions of ``pathext``
    (``PATHEXT``) are tried, as Windows does.
    """
    entries = (path if path is not None else os.environ.get("PATH", "")).split(os.pathsep)
    names = [name]
    if _WINDOWS:
        source = pathext or os.environ.get("PATHEXT") or ".COM;.EXE;.BAT;.CMD"
        extensions = [ext for ext in source.split(os.pathsep) if ext]
        if not any(name.lower().endswith(ext.lower()) for ext in extensions):
            names = [name + ext for ext in extensions]
    excluded = [os.path.abspath(folder) for folder in exclude]
    for entry in entries:
        folder = entry.strip().strip('"')
        if not folder or not os.path.isabs(folder):
            continue
        if any(_within(folder, root) for root in excluded):
            continue
        for candidate in (os.path.join(folder, item) for item in names):
            if os.path.isfile(candidate) and (_WINDOWS or os.access(candidate, os.X_OK)):
                return candidate
    return None


def _within(path: str, root: str) -> bool:
    try:
        relative = os.path.relpath(os.path.abspath(path), root)
    except ValueError:  # another drive on Windows
        return False
    return not (relative == os.pardir or relative.startswith(os.pardir + os.sep))


class ManagedProcess:
    """A child process with piped stdio, a stderr ring buffer and a tree kill.

    Build it with `start`. ``stdin`` and ``stdout`` are asyncio streams (give
    them to `JsonRpcPeer`). Always end it with `close` (or ``async with``):
    that closes stdin, waits, terminates and kills whatever of the tree is
    left.
    """

    def __init__(
        self, process: asyncio.subprocess.Process, *, name: str, stderr_bytes: int
    ) -> None:
        self._process = process
        self.name = name
        """Name used in log lines and error messages."""
        self._stderr = bytearray()
        self._stderr_bytes = max(1024, stderr_bytes)
        self._stderr_task: asyncio.Task[None] | None = None
        self._job: Any = None
        """Windows: the Job Object that holds the process tree."""
        self._tree: list[tuple[int, int]] = []
        """Windows without a job: ``(pid, start time)`` of the tree, recorded before stdin
        is closed, to kill what the process leaves behind."""
        self.killed = False
        """The tree was killed (it did not exit by itself within the grace periods)."""
        self.terminated = False
        """The process was asked to terminate (POSIX: SIGTERM to its group)."""
        self._close_task: asyncio.Task[None] | None = None
        self._abandoned = False
        try:
            self._loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    @classmethod
    async def start(
        cls,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        inherit_env: bool = False,
        cwd: str | Path | None = None,
        name: str | None = None,
        stderr_bytes: int = STDERR_BYTES,
    ) -> Self:
        """Start ``command`` with piped stdin, stdout and stderr.

        Args:
            command: Program and arguments, never a shell string. The program is
                resolved on the child's ``PATH`` (`resolve_command`).
            env: The child's whole environment. ``None`` uses `build_env()`:
                the base allowlist only.
            inherit_env: With ``env=None``, give the child the parent's whole
                environment instead. Off by default, so secrets stay in the parent.
            cwd: Working directory of the child; it must exist.
            name: Name for logs and errors; defaults to the program's file name.
            stderr_bytes: Size of the stderr ring buffer.

        Raises:
            ExternalAgentConfigurationError: The program was not found or could
                not be started, ``cwd`` is not a directory, or the event loop
                cannot run subprocesses.
        """
        environment = dict(env) if env is not None else build_env(inherit_env=inherit_env)
        argv = resolve_command(command, environment)
        label = name or Path(argv[0]).name
        if cwd is not None and not _is_dir(cwd):
            raise ExternalAgentConfigurationError(
                f"Cannot start {label}: the working directory {str(cwd)!r} does not exist."
            )
        options: dict[str, Any] = {}
        if _WINDOWS:
            options["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            )
        else:
            options["start_new_session"] = True  # its own process group: the tree can be killed
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=environment,
                cwd=str(cwd) if cwd is not None else None,
                limit=STDOUT_LIMIT,
                **options,
            )
        except NotImplementedError as error:
            raise ExternalAgentConfigurationError(
                f"Cannot start {label}: this event loop cannot run subprocesses. On Windows use "
                "the default Proactor event loop (do not set WindowsSelectorEventLoopPolicy)."
            ) from error
        except OSError as error:
            raise ExternalAgentConfigurationError(f"Cannot start {label}: {error}") from error
        managed = cls(process, name=label, stderr_bytes=stderr_bytes)
        if _WINDOWS:
            managed._job = _WindowsJob.create_for(process.pid)
        _LIVE.add(managed)
        managed._stderr_task = asyncio.create_task(managed._read_stderr())
        logger.debug("Started %s (pid %s): %s", label, process.pid, argv)
        return managed

    # -- what callers use -----------------------------------------------------------------

    @property
    def pid(self) -> int:
        return self._process.pid

    @property
    def returncode(self) -> int | None:
        """The exit code, once the process has exited; ``None`` while it runs."""
        return self._process.returncode

    @property
    def stdin(self) -> asyncio.StreamWriter:
        assert self._process.stdin is not None
        return self._process.stdin

    @property
    def stdout(self) -> asyncio.StreamReader:
        assert self._process.stdout is not None
        return self._process.stdout

    def stderr_tail(self, max_chars: int = 2000) -> str:
        """The last ``max_chars`` characters the process wrote to stderr (decoded as UTF-8)."""
        text = bytes(self._stderr).decode("utf-8", errors="replace")
        return text[-max_chars:].strip()

    async def wait(self, timeout: float | None = None) -> int | None:
        """Wait for the process to exit; its exit code, or ``None`` if it still runs.

        It watches the exit code rather than awaiting ``Process.wait()``,
        which on some Python versions also waits for the pipes to close, and
        a descendant that inherited them can hold them open.
        """
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        while self._process.returncode is None:
            if deadline is not None and loop.time() >= deadline:
                return None
            await asyncio.sleep(0.02)
        return self._process.returncode

    async def explain_exit(self, wait: float = 1.0) -> str:
        """A sentence for error messages: how the process ended, and its stderr tail.

        Waits up to ``wait`` seconds for the exit code (stdout often closes a
        moment before the process is gone). For developers and logs only: the
        stderr tail may hold paths or other details that a model must not see.
        """
        code = self.returncode if self.returncode is not None else await self.wait(wait)
        if code is not None:
            how = f"{self.name} exited with code {code}"
        else:
            how = f"{self.name} closed its output but is still running"
        tail = self.stderr_tail()
        return f"{how}; stderr: {tail}" if tail else how

    def exit_summary(self) -> str:
        """A short, model-safe phrase: ``"the process exited with code 1"``, no stderr."""
        if self.returncode is None:
            return prompts.EXTERNAL_REASON_PROCESS_LOST
        return prompts.EXTERNAL_REASON_PROCESS_EXITED.format(code=self.returncode)

    async def close(self, *, grace: float = 5.0, terminate_grace: float = 2.0) -> int | None:
        """Shut the process down and leave nothing of its tree behind.

        Closes stdin (most agents exit on EOF), waits ``grace`` seconds, then
        terminates (POSIX: SIGTERM to the process group) and waits
        ``terminate_grace`` seconds, then kills the tree. Whatever the
        process started and left running is killed in every case.

        Safe to call more than once (a later call waits for the same
        shutdown) and safe to cancel: when the caller is cancelled while it
        waits, the tree is killed at once and the cancellation goes on. A
        process started on an event loop that has ended (``asyncio.run``
        called again) is killed without waiting.

        Returns:
            The exit code, or ``None`` if it could not be collected.
        """
        loop = asyncio.get_running_loop()
        if self._abandoned:
            return self._poll()
        if self._loop is not None and self._loop is not loop:
            return await self._abandon()
        task = self._close_task
        if task is None:
            task = self._close_task = loop.create_task(
                self._shutdown(grace, terminate_grace), name=f"close:{self.name}"
            )
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.done():  # cancelled while waiting for the exit: leave nothing running
                self.kill()
                self._kill_leftovers()
            raise
        return self.returncode

    def kill(self) -> None:
        """Kill the process and everything it started, now. Synchronous, quick, safe to repeat.

        It starts no other program: on Windows it ends the Job Object (or, without
        one, the processes of the tree it finds); on POSIX it signals the process group.
        """
        self.killed = True
        if _WINDOWS:
            if self._job is not None:
                self._job.terminate()  # every process of the tree, at once
            else:
                _terminate_tree([*self._tree, *_windows_tree(self.pid)])
        else:
            _killpg(self.pid, signal.SIGKILL)
        if self.returncode is None:
            with contextlib.suppress(ProcessLookupError, OSError, RuntimeError):
                self._process.kill()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # -- internals -------------------------------------------------------------------------

    async def _shutdown(self, grace: float, terminate_grace: float) -> None:
        """The body of `close`. It runs as its own task, so a cancelled caller cannot stop
        it halfway; if the task itself is cancelled (the event loop shuts down), it kills."""
        finished = False
        try:
            if _WINDOWS and self._job is None and self.returncode is None:
                # Without a job the tree is found through the parent: record it while the
                # parent lives, so what it leaves behind when it exits is still found.
                self._tree = _windows_tree(self.pid)
            with contextlib.suppress(Exception):
                if not self.stdin.is_closing():
                    self.stdin.close()
            if self.returncode is None and await self.wait(grace) is None:
                self._terminate()
                if terminate_grace > 0 and self.terminated:
                    await self.wait(terminate_grace)
                if self.returncode is None:
                    self.kill()
                    if await self.wait(5.0) is None:
                        logger.error("%s (pid %s) did not exit after a kill", self.name, self.pid)
            finished = True
        finally:
            if not finished:
                self.kill()
            self._kill_leftovers()
            _LIVE.discard(self)
        await self._finish_stderr()
        self._close_transport()

    async def _abandon(self) -> int | None:
        """Kill a process whose event loop has ended, without awaiting anything of that loop."""
        logger.info(
            "%s (pid %s) was started on an event loop that has ended; killing it",
            self.name,
            self.pid,
        )
        self._abandoned = True
        if self._poll() is None:
            self.kill()
        self._kill_leftovers()
        _LIVE.discard(self)
        task = self._stderr_task
        if task is not None and not task.done():
            with contextlib.suppress(RuntimeError):  # its loop is closed
                task.cancel()
        self._release_pipes()
        for _ in range(50):  # the kill is quick; the exit code follows
            if (code := self._poll()) is not None:
                return code
            await asyncio.sleep(0.02)
        return None

    def _poll(self) -> int | None:
        """The exit code, also when the event loop that watched the process has ended."""
        if self._process.returncode is not None:
            return self._process.returncode
        popen = getattr(getattr(self._process, "_transport", None), "_proc", None)
        if popen is None:
            return None
        with contextlib.suppress(Exception):
            return popen.poll()
        return None

    def _close_transport(self) -> None:
        transport = getattr(self._process, "_transport", None)
        if transport is not None:  # release the pipes now, not when it is collected
            with contextlib.suppress(Exception):
                transport.close()

    def _release_pipes(self) -> None:
        """Close the pipe handles of a transport whose event loop has ended.

        The transport's own ``close()`` needs its loop, so the handles are closed
        directly (Proactor pipes keep them in ``_sock``, Unix pipes in ``_pipe``).
        """
        transport = getattr(self._process, "_transport", None)
        if transport is None:
            return
        for protocol in list(getattr(transport, "_pipes", {}).values()):
            pipe = getattr(protocol, "pipe", None)
            for attribute in ("_sock", "_pipe"):
                handle = getattr(pipe, attribute, None)
                if handle is not None:
                    with contextlib.suppress(Exception):
                        handle.close()
                    with contextlib.suppress(Exception):
                        setattr(pipe, attribute, None)
        with contextlib.suppress(Exception):
            transport._closed = True  # nothing is left for its close() to do

    def _terminate(self) -> None:
        if _WINDOWS:
            return  # no reliable polite signal for a windowless console process: kill next
        self.terminated = True
        _killpg(self.pid, signal.SIGTERM)

    def _kill_leftovers(self) -> None:
        """Kill descendants that outlived the process (it exited and left them running)."""
        if _WINDOWS:
            if self._job is not None:
                self._job.terminate()
                self._job.close()
                self._job = None
            elif self._tree:
                tree, self._tree = self._tree, []
                _terminate_tree(tree)
        else:
            _killpg(self.pid, signal.SIGKILL)

    async def _read_stderr(self) -> None:
        stream = self._process.stderr
        assert stream is not None
        try:
            while chunk := await stream.read(65536):
                self._stderr += chunk
                if len(self._stderr) > self._stderr_bytes:
                    del self._stderr[: len(self._stderr) - self._stderr_bytes]
                if logger.isEnabledFor(logging.DEBUG):
                    text = chunk.decode("utf-8", errors="replace").rstrip()
                    logger.debug("%s stderr: %s", self.name, text)
        except (OSError, ValueError):  # the pipe broke: the process is gone or going
            return

    async def _finish_stderr(self) -> None:
        task = self._stderr_task
        if task is None:
            return
        with contextlib.suppress(TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(task), 1.0)
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


_LIVE: weakref.WeakSet[ManagedProcess] = weakref.WeakSet()
"""Processes started and not yet closed. Whatever of them still runs when Python exits is
killed then."""


@atexit.register
def _kill_live_processes() -> None:
    for process in list(_LIVE):
        with contextlib.suppress(Exception):
            if process._poll() is None:
                logger.warning(
                    "%s (pid %s) was never closed; killing it", process.name, process.pid
                )
                process.kill()
            process._kill_leftovers()


def _is_dir(path: str | Path) -> bool:
    return Path(path).is_dir()


def _killpg(pid: int, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(pid, sig)  # pid leads its own process group (start_new_session=True)


# --- Windows: the process tree without a Job Object ----------------------------------------

_PROCESS_TERMINATE = 0x0001
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259


def _windows_tree(root: int) -> list[tuple[int, int]]:
    """``(pid, start time)`` of ``root`` and every live descendant, root first.

    Found by parent process id in a process snapshot. A process counts as a
    child only if it started after its parent: Windows keeps the parent id of
    a process whose parent has exited, so a reused id would otherwise adopt
    unrelated processes. Empty when ``root`` is gone, or off Windows.
    """
    if not _WINDOWS:
        return []
    root_started = _started(root)
    if root_started is None:
        return []
    children: dict[int, list[int]] = {}
    for pid, parent in _snapshot():
        if pid not in (0, parent):
            children.setdefault(parent, []).append(pid)
    found = [(root, root_started)]
    seen = {root}
    queue = [(root, root_started)]
    while queue:
        parent, parent_started = queue.pop()
        for pid in children.get(parent, ()):
            if pid in seen:
                continue
            started = _started(pid)
            if started is None or started < parent_started:
                continue  # gone, or an older process whose parent id was reused
            seen.add(pid)
            found.append((pid, started))
            queue.append((pid, started))
    return found


def _terminate_tree(tree: Sequence[tuple[int, int]]) -> None:
    """Kill each recorded process that still runs (same id, same start time) and whatever
    each of them started since: the deepest first, then a second pass for late children."""
    if not _WINDOWS or not tree:
        return
    for _ in range(2):
        found: dict[int, int] = {}
        for pid, started in tree:
            if _started(pid) == started:
                found.update(_windows_tree(pid))
        if not found:
            return
        for pid, started in reversed(list(found.items())):
            _terminate(pid, started)
        tree = list(found.items())


def _snapshot() -> list[tuple[int, int]]:
    """``(pid, parent pid)`` of every process (a Toolhelp32 snapshot)."""
    import ctypes
    from ctypes import wintypes

    class _Entry(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", ctypes.c_wchar * 260),
        ]

    kernel32 = _kernel32()
    handle = kernel32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    if not handle or handle == ctypes.c_void_p(-1).value:  # INVALID_HANDLE_VALUE
        return []
    entries: list[tuple[int, int]] = []
    try:
        entry = _Entry()
        entry.dwSize = ctypes.sizeof(_Entry)
        ok = kernel32.Process32FirstW(handle, ctypes.byref(entry))
        while ok:
            entries.append((int(entry.th32ProcessID), int(entry.th32ParentProcessID)))
            ok = kernel32.Process32NextW(handle, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(handle)
    return entries


def _process_times(handle: Any) -> int | None:
    """The start time (FILETIME) of an open process handle."""
    import ctypes

    times = [ctypes.c_ulonglong() for _ in range(4)]  # created, exited, kernel, user
    ok = _kernel32().GetProcessTimes(handle, *(ctypes.byref(value) for value in times))
    return int(times[0].value) if ok else None


def _started(pid: int) -> int | None:
    """The start time of a running process, or ``None`` when it is gone (or not ours to see)."""
    import ctypes

    kernel32 = _kernel32()
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return None
        if code.value != _STILL_ACTIVE:
            return None
        return _process_times(handle)
    finally:
        kernel32.CloseHandle(handle)


def _terminate(pid: int, started: int) -> None:
    """Kill ``pid`` if it is still the process that started at ``started``."""
    kernel32 = _kernel32()
    access = _PROCESS_TERMINATE | _PROCESS_QUERY_LIMITED_INFORMATION
    handle = kernel32.OpenProcess(access, False, pid)
    if not handle:
        return
    try:
        if _process_times(handle) == started:
            kernel32.TerminateProcess(handle, 1)
    finally:
        kernel32.CloseHandle(handle)


_KERNEL32: Any = None


def _kernel32() -> Any:
    """kernel32 with the signatures this module uses, set once (Windows only)."""
    global _KERNEL32
    if _KERNEL32 is None:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        kernel32.Process32FirstW.restype = wintypes.BOOL
        kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
        kernel32.Process32NextW.restype = wintypes.BOOL
        kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.c_void_p] * 4)]
        kernel32.TerminateProcess.restype = wintypes.BOOL
        kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        _KERNEL32 = kernel32
    return _KERNEL32


class _WindowsJob:
    """A Job Object with kill-on-close that holds one process and everything it starts.

    Processes the child starts after it joined the job belong to the job too,
    so terminating or closing the job ends the whole tree, even when the child
    itself already exited.
    """

    def __init__(self, handle: int) -> None:
        self._handle: int | None = handle

    @classmethod
    def create_for(cls, pid: int) -> _WindowsJob | None:
        """A job for ``pid``, or ``None`` when Windows refuses (the tree is then found from a
        process snapshot, see `_windows_tree`)."""
        import ctypes

        kernel32 = _kernel32()

        class _Basic(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", ctypes.c_uint32),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", ctypes.c_uint32),
                ("SchedulingClass", ctypes.c_uint32),
            ]

        class _Counters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ("r", "w", "o", "rb", "wb", "ob")]

        class _Extended(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _Basic),
                ("IoInfo", _Counters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            logger.warning(
                "CreateJobObject failed (%s); the process tree is found by parent ids",
                ctypes.get_last_error(),
            )
            return None
        info = _Extended()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        # 9: JobObjectExtendedLimitInformation
        if not kernel32.SetInformationJobObject(
            job, 9, ctypes.addressof(info), ctypes.sizeof(info)
        ):
            logger.warning("SetInformationJobObject failed (%s)", ctypes.get_last_error())
            kernel32.CloseHandle(job)
            return None
        process = kernel32.OpenProcess(0x0100 | 0x0001, False, pid)  # SET_QUOTA | TERMINATE
        assigned = bool(process) and bool(kernel32.AssignProcessToJobObject(job, process))
        error = ctypes.get_last_error()
        if process:
            kernel32.CloseHandle(process)
        if not assigned:
            logger.warning(
                "Could not put pid %s in a Job Object (%s); its tree is found by parent ids",
                pid,
                error,
            )
            kernel32.CloseHandle(job)
            return None
        return cls(job)

    def terminate(self) -> None:
        """Kill every process in the job."""
        if self._handle is not None:
            _kernel32().TerminateJobObject(self._handle, 1)

    def close(self) -> None:
        """Release the job; kill-on-close ends whatever is still in it."""
        if self._handle is not None:
            _kernel32().CloseHandle(self._handle)
            self._handle = None
