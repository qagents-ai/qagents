"""Exchange messages and resume mailbox tools after terminating real agent processes."""

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from qagents import SQLiteStore, TeamMailbox

from .team_worker import MEMBERS

pytestmark = [
    pytest.mark.process,
    pytest.mark.skipif(
        os.environ.get("QAGENTS_RUN_PROCESS_TESTS") != "1",
        reason="run the separate-process agent tests in the process CI lane",
    ),
]


@asynccontextmanager
async def launch_worker(root, role, action, *, now=1000):
    entry = Path(__file__).with_name("team_worker.py")
    # The virtualenv launcher can have a different PID from the Python worker.
    worker = await asyncio.create_subprocess_exec(
        getattr(sys, "_base_executable", sys.executable),
        "-c",
        "import runpy,sys; sys.path[:0]=sys.argv[1].split(chr(31)); "
        "sys.argv=sys.argv[2:]; runpy.run_path(sys.argv[0], run_name='__main__')",
        chr(31).join([str(entry.parents[2] / "src"), *sys.path]),
        str(entry),
        str(root),
        role,
        action,
        str(now),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        yield worker
    finally:
        if worker.returncode is None:
            worker.kill()
        await asyncio.wait_for(worker.communicate(), 15)


async def collect(worker):
    output, error = await asyncio.wait_for(worker.communicate(), 30)
    assert worker.returncode == 0, error.decode(errors="replace")
    result = json.loads(output)
    assert result["pid"] == worker.pid
    return result


async def test_coordinator_and_two_workers_exchange_messages_across_processes(tmp_path):
    async with (
        launch_worker(tmp_path, "researcher", "exchange") as research_worker,
        launch_worker(tmp_path, "reviewer", "exchange") as review_worker,
    ):
        listeners = (research_worker, review_worker)
        ready = await asyncio.wait_for(
            asyncio.gather(*(worker.stdout.readline() for worker in listeners)), 15
        )
        for worker, line in zip(listeners, ready, strict=True):
            assert json.loads(line) == {"pid": worker.pid, "ready": True}
        async with launch_worker(tmp_path, "lead", "exchange") as lead_worker:
            lead, researcher, reviewer = await asyncio.gather(
                collect(lead_worker), collect(research_worker), collect(review_worker)
            )
    assert len({lead["pid"], researcher["pid"], reviewer["pid"], os.getpid()}) == 4
    assert researcher["received"] == lead["sent"]
    assert reviewer["received"] == researcher["sent"]
    assert lead["received"] == reviewer["sent"]
    assert [
        (row["sender"], row["recipient"], row["sequence"])
        for row in [*researcher["received"], *reviewer["received"], *lead["received"]]
    ] == [("lead", "researcher", 1), ("researcher", "reviewer", 2), ("reviewer", "lead", 3)]


@pytest.mark.parametrize(
    "action,role,tool_name",
    [("cut-send", "lead", "send_message"), ("cut-receive", "researcher", "receive_messages")],
)
async def test_resume_after_terminating_worker_before_it_journals_message_tool_result(
    tmp_path, action, role, tool_name
):
    box = TeamMailbox(MEMBERS, db_path=tmp_path / "mailbox.db")
    lead, inbox = box.endpoint("lead"), box.endpoint("researcher")
    if action == "cut-receive":
        await lead.send("researcher", "Original")
    async with launch_worker(tmp_path, role, action) as worker:
        line = await asyncio.wait_for(worker.stdout.readline(), 15)
        assert line, (await worker.stderr.read()).decode(errors="replace")
        boundary = json.loads(line)
        assert boundary["pid"] == worker.pid and boundary["tool"] == tool_name
        worker.kill()
        assert await asyncio.wait_for(worker.wait(), 10) != 0

    async with SQLiteStore(tmp_path / f"{role}.db") as store:
        entries = await store.load(role)
        assert any(entry.kind == "tool_started" for entry in entries)
        assert not any(entry.kind == "tool_result" for entry in entries)

    if action == "cut-receive":
        later = await lead.send("researcher", "Next")
    # Advance the shared journal's fixture clock past the dead worker's lease.
    async with launch_worker(tmp_path, role, "resume", now=1100) as replacement:
        result = await collect(replacement)

    saved = json.loads(boundary["content"])
    if action == "cut-send":
        assert result["sent"] == [saved]
        (outstanding,) = await inbox.receive()
        assert outstanding.sequence == saved["sequence"] and outstanding.text == saved["text"]
    else:
        assert result["received"] == saved
        assert [item["text"] for item in result["received"]] == ["Original"]
        assert await inbox.receive() == [later]
    assert await inbox.receive() == []
