"""Terminate an owned worker after destination commit and recover in a new process."""

import asyncio
import json
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pytest

from qagents import SQLiteStore

pytestmark = [
    pytest.mark.process,
    pytest.mark.skipif(
        os.environ.get("QAGENTS_RUN_PROCESS_TESTS") != "1",
        reason="run the subprocess recovery test explicitly or in the process CI lane",
    ),
]


async def launch(root, phase):
    entry = Path(__file__).with_name("effect_worker.py")
    source = entry.parents[2] / "src"
    env = {**os.environ, "PYTHONPATH": str(source)}
    return await asyncio.create_subprocess_exec(
        # The Windows venv shim may launch another process. Start the interpreter itself.
        getattr(sys, "_base_executable", sys.executable),
        "-c",
        "import runpy,sys; sys.path[:0]=sys.argv[1].split(chr(31)); "
        "sys.argv=sys.argv[2:]; runpy.run_path(sys.argv[0], run_name='__main__')",
        chr(31).join(sys.path),
        str(entry),
        str(root),
        phase,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )


async def test_recovery_after_real_kill_preserves_one_credit(tmp_path):
    worker = await launch(tmp_path, "first")
    try:
        marker = json.loads(await asyncio.wait_for(worker.stdout.readline(), 12))
        assert marker["pid"] == worker.pid and worker.returncode is None
        worker.kill()
        assert await asyncio.wait_for(worker.wait(), 10) != 0
    finally:
        if worker.returncode is None:
            worker.kill()
        await worker.communicate()

    async with SQLiteStore(tmp_path / "journal.db") as store:
        before = await store.load("case")
    (intent,) = [entry for entry in before if entry.kind == "effect_intent"]
    assert intent.data["idempotency_key"] == marker["effect_key"]
    assert not any(entry.kind in {"effect_committed", "tool_result"} for entry in before)
    with closing(sqlite3.connect(tmp_path / "destination.db")) as db:
        assert db.execute("SELECT SUM(amount) FROM credits").fetchone() == (10,)

    recovery = await launch(tmp_path, "recovery")
    try:
        output, error = await asyncio.wait_for(recovery.communicate(), 18)
        assert recovery.returncode == 0, error.decode(errors="replace")
        assert json.loads(output) == {"pid": recovery.pid, "result": "Credit done."}
    finally:
        if recovery.returncode is None:
            recovery.kill()
            await recovery.communicate()

    async with SQLiteStore(tmp_path / "journal.db") as store:
        after = await store.load("case")
    assert after[: len(before)] == before
    assert sum(entry.kind == "effect_committed" for entry in after) == 1
    assert sum(entry.kind == "tool_result" for entry in after) == 1
    with closing(sqlite3.connect(tmp_path / "destination.db")) as db:
        assert db.execute("SELECT amount FROM credits").fetchall() == [(10,)]
        assert db.execute("SELECT token FROM requests").fetchall() == [
            (marker["effect_key"],),
            (marker["effect_key"],),
        ]
