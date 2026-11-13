"""The public kit checks memory and a SQLite file including a killed writer."""

import subprocess
import sys
from io import BytesIO
from types import SimpleNamespace

import pytest

from qagents.session import InMemoryStore, SQLiteStore
from qagents.testing import stores as kit
from qagents.testing.stores import check_store


async def test_memory_store_contract():
    db = InMemoryStore()
    report = await check_store(lambda: db)
    assert report.ok, report.checks
    assert len(report.checks) == 6
    assert "durability after killed writer" in report.not_checked


async def test_sqlite_contract_and_actual_kill(tmp_path):
    report = await check_store(lambda: SQLiteStore(tmp_path / "kit.db"))
    assert report.ok, report.checks
    assert len(report.checks) == 7
    assert "durability after killed writer" not in report.not_checked
    assert "replication durability" in report.not_checked


async def test_kit_reports_broken_batch_idempotence():
    class Broken(InMemoryStore):
        async def append(self, session_id, entries):
            return await super().append(
                session_id,
                [
                    entry.model_copy(
                        update={"id": f"broken-{len(await self.load(session_id))}-{n}"}
                    )
                    for n, entry in enumerate(entries)
                ],
            )

    db = Broken()
    report = await check_store(lambda: db)
    assert not report.ok
    assert any(
        name == "atomic append, ids and decided failure" and not ok for name, ok, _ in report.checks
    )


async def test_kill_targets_the_acknowledged_writer_pid_and_waits_for_exit(tmp_path, monkeypatch):
    launched = []
    original = subprocess.Popen

    def start(args, **kwargs):
        process = original(args, **kwargs)
        launched.append((args[0], process))
        return process

    monkeypatch.setattr(kit.subprocess, "Popen", start)
    path = tmp_path / "actual-writer.db"
    writer_pid = await kit._sqlite_kill(str(path), "probe")
    executable, process = launched[0]
    assert executable == getattr(sys, "_base_executable", sys.executable)
    assert writer_pid == process.pid
    assert process.poll() is not None
    async with SQLiteStore(path) as reopened:
        assert [entry.id for entry in await reopened.load("probe")] == ["kill-check"]


async def test_a_launcher_pid_mismatch_cannot_report_a_writer_kill(monkeypatch):
    stdin = BytesIO()
    process = SimpleNamespace(
        pid=111,
        stdin=stdin,
        stdout=BytesIO(b'{"ack":"written","pid":222}\n'),
        returncode=None,
    )
    process.kill = lambda: setattr(process, "returncode", -1)
    process.communicate = lambda **kwargs: (b"", b"")
    monkeypatch.setattr(kit.subprocess, "Popen", lambda *args, **kwargs: process)
    with pytest.raises(AssertionError, match="not the SQLite writer"):
        await kit._sqlite_kill("unused.db", "probe")
    assert stdin.closed
    assert process.returncode is not None
