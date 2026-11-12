"""Git change tracking: which files an external turn changed, and its diff on demand.

Before a turn starts in a workspace that is a git repository, `_git_snapshot` records the
dirty paths with their ``(status, mtime_ns, size)``; after the turn, `_changed_since`
compares a second snapshot with it, which gives ``ExternalResult.changed_files``.
`_git_diff` computes ``ExternalResult.diff`` later, on demand.

It lives apart because git runs in a directory the external agent can write to, so every
git command here is hardened in the same way (see the comment below `_Snapshot`), and that
is easiest to check when all of them are in one place.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger("qagents.external")

_DIFF_ARGS_CHARS = 24_000
"""Characters of paths per ``git diff`` command line: Windows allows 32,767 in all."""

_Snapshot = tuple[str, dict[str, tuple[str, int, int]]]

# Git runs in the turn's workspace, which the external agent can write to, ``.git/config``
# included. Several settings there name a program for git to run (core.fsmonitor, a
# filter's clean or process command, diff.external, a textconv, a pager, a fetch for a
# missing object), and git would run it outside the agent's sandbox with qagents'
# environment. So every git command gets: a small environment (`_git_env`), no network
# transport, no pager, and ``-c`` options that switch those programs off
# (`_GIT_SAFE_CONFIG`, plus each filter driver the config defines, `_git_safe_config`).
# ``-c`` wins over every config file. Submodules are not entered (their own config is not
# checked): their commits are compared, their work trees are not. And git itself is found
# on PATH by qagents (`_git_program`), never in the current directory or the workspace,
# where the agent could put a program called git.

_GIT_SAFE_CONFIG: tuple[str, ...] = ("core.fsmonitor=", "core.pager=cat", "diff.external=")
"""``-c`` settings for every git command qagents runs."""

_GIT_PASSTHROUGH: tuple[str, ...] = (
    "HOMEDRIVE",
    "HOMEPATH",
    "XDG_CONFIG_HOME",
    "PROGRAMDATA",
    "LC_ALL",
    "LC_CTYPE",
    "GIT_EXEC_PATH",
    "GIT_CONFIG_NOSYSTEM",
    "GIT_CONFIG_SYSTEM",
    "GIT_CONFIG_GLOBAL",
    "GIT_CEILING_DIRECTORIES",
)
"""Copied from qagents' environment for git, besides the base allowlist of
`qagents.integrations._process.build_env`: where the user's own git config is."""

_GIT_ENV: dict[str, str] = {
    "GIT_OPTIONAL_LOCKS": "0",  # never take the index lock from the agent
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_ALLOW_PROTOCOL": "",  # no transport at all, whatever the config allows
    "GIT_NO_LAZY_FETCH": "1",  # a partial clone does not fetch a missing object
    "GIT_PAGER": "cat",
}


def _git_env() -> dict[str, str]:
    """The environment of git: an allowlist, so no secret of qagents reaches it."""
    from ..integrations._process import build_env  # it imports this package

    return build_env(passthrough=_GIT_PASSTHROUGH, extra=_GIT_ENV)


def _git_program(workspace: str) -> str | None:
    """The absolute path of git on ``PATH``, or ``None``.

    Found by hand, not by the operating system: Windows looks in the current
    directory first, and qagents often runs in the workspace, where the agent
    can put a ``git.exe`` of its own. Relative ``PATH`` entries and entries
    inside ``workspace`` are skipped for the same reason, and on Windows only a
    ``git.exe`` counts (a ``.cmd`` or ``.bat`` would run through ``cmd.exe``).
    """
    from ..integrations._process import find_program

    pathext = ".EXE" if sys.platform == "win32" else None
    return find_program("git", pathext=pathext, exclude=[workspace])


async def _git(
    cwd: str,
    *args: str,
    config: Sequence[str] = _GIT_SAFE_CONFIG,
    ok: tuple[int, ...] = (0,),
    timeout: float = 30.0,
) -> bytes | None:
    """Run git in ``cwd`` with ``-c`` ``config``; its stdout, or ``None`` when git is
    missing or exits with a code not in ``ok``. Git is killed when the call times out
    or is cancelled."""
    program = _git_program(cwd)
    if program is None:
        return None
    options: dict[str, Any] = {}
    if sys.platform == "win32":
        options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    command = [program, "--no-pager"]
    for setting in config:
        command += ["-c", setting]
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            *args,
            cwd=cwd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=_git_env(),
            **options,
        )
    except (OSError, NotImplementedError, ValueError):
        return None
    try:
        output, _ = await asyncio.wait_for(process.communicate(), timeout)
    except BaseException as error:  # a timeout, or the turn was cancelled: stop git too
        with contextlib.suppress(ProcessLookupError, OSError):
            process.kill()
        with contextlib.suppress(Exception):  # bounded; a second cancel still goes through
            await asyncio.wait_for(asyncio.shield(process.wait()), 5.0)
        transport = getattr(process, "_transport", None)
        if transport is not None:
            with contextlib.suppress(Exception):
                transport.close()
        if isinstance(error, TimeoutError):
            return None
        raise
    return output if process.returncode in ok else None


async def _git_safe_config(cwd: str) -> list[str] | None:
    """`_GIT_SAFE_CONFIG` plus settings that switch off every filter driver the config
    defines (its ``clean``, ``smudge`` and ``process`` commands, which ``git status`` and
    ``git diff`` run on changed files). ``None`` when that cannot be done: git cannot
    list the config, or a driver's name cannot be written as a ``-c`` setting."""
    names = await _git(
        cwd, "config", "--null", "--name-only", "--get-regexp", r"^filter\.", ok=(0, 1)
    )
    if names is None:
        return None
    drivers: set[str] = set()
    for key in names.decode("utf-8", errors="surrogateescape").split("\0"):
        if not key.startswith("filter."):
            continue
        driver = key[len("filter.") :].rpartition(".")[0]
        if not driver:
            continue
        if "=" in driver or "\n" in driver:
            logger.warning(
                "Not tracking changes in %s: its git config defines a filter (%r) that "
                "cannot be switched off for git status",
                cwd,
                driver,
            )
            return None
        drivers.add(driver)
    config = list(_GIT_SAFE_CONFIG)
    for driver in sorted(drivers):
        config += [
            f"filter.{driver}.clean=",
            f"filter.{driver}.smudge=",
            f"filter.{driver}.process=",
            f"filter.{driver}.required=false",
        ]
    return config


async def _git_snapshot(cwd: str | None) -> _Snapshot | None:
    """The repository root and its dirty paths with ``(status, mtime_ns, size)``."""
    if cwd is None or not _is_dir(cwd):
        return None
    top = await _git(cwd, "rev-parse", "--show-toplevel")
    if top is None:
        return None
    root = top.decode("utf-8", errors="replace").strip()
    config = await _git_safe_config(root)
    if config is None:
        return None
    status = await _git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=dirty",
        config=config,
    )
    if status is None:
        return None
    # In a worker thread: a turn can leave tens of thousands of dirty paths (an install, a
    # build directory that is not ignored), and a stat each would stall the event loop, and
    # with it every connection to an agent.
    return root, await asyncio.to_thread(_status_entries, root, status)


def _status_entries(root: str, status: bytes) -> dict[str, tuple[str, int, int]]:
    """The paths of ``git status --porcelain=v1 -z`` output with ``(status, mtime_ns, size)``."""
    entries: dict[str, tuple[str, int, int]] = {}
    items = status.decode("utf-8", errors="surrogateescape").split("\0")
    index = 0
    while index < len(items):
        item = items[index]
        index += 1
        if len(item) < 4:
            continue
        code, path = item[:2], item[3:]
        if code[0] in "RC":
            index += 1  # the original path of a rename or copy follows
        entries[path] = (code, *_stat(os.path.join(root, path)))
    return entries


def _path_batches(paths: Sequence[str], limit: int) -> list[list[str]]:
    """``paths`` in groups whose command-line length (with quotes and spaces) stays within
    ``limit`` characters; a single longer path gets a group of its own."""
    batches: list[list[str]] = []
    batch: list[str] = []
    size = 0
    for path in paths:
        length = len(path) + 3
        if batch and size + length > limit:
            batches.append(batch)
            batch, size = [], 0
        batch.append(path)
        size += length
    if batch:
        batches.append(batch)
    return batches


async def _changed_since(snapshot: _Snapshot) -> list[str] | None:
    root, before = snapshot
    after_snapshot = await _git_snapshot(root)
    if after_snapshot is None:
        return None
    after = after_snapshot[1]
    changed = {path for path, state in after.items() if before.get(path) != state}
    changed.update(path for path in before if path not in after)  # reverted or committed
    return sorted(changed)


def _is_dir(path: str) -> bool:
    return os.path.isdir(path)


def _stat(path: str) -> tuple[int, int]:
    try:
        info = os.stat(path)
    except OSError:
        return -1, -1
    return info.st_mtime_ns, info.st_size


def _relative_to(path: str, root: str | None) -> str | None:
    if root is None:
        return None
    full = os.path.abspath(os.path.join(root, path))
    try:
        relative = os.path.relpath(full, root)
    except ValueError:  # another drive on Windows
        return None
    if relative.startswith(".."):
        return None
    return Path(relative).as_posix()


async def _git_diff(root: str, changed_files: Sequence[str]) -> str:
    """``git diff`` of ``changed_files`` (tracked files only) in ``root``, as
    `ExternalResult.diff` returns it; empty when one of the git calls fails."""
    config = await _git_safe_config(root)
    if config is None:
        return ""
    pieces: list[bytes] = []
    for batch in _path_batches(changed_files, _DIFF_ARGS_CHARS):
        output = await _git(
            root,
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-color",
            "--ignore-submodules=dirty",
            "--",
            *batch,
            config=config,
        )
        if output is None:
            logger.warning("git diff failed in %s; returning no diff", root)
            return ""
        pieces.append(output)
    return b"".join(pieces).decode("utf-8", errors="replace")
