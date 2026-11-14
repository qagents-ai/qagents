"""Verify actual worker termination at SQLite and destination boundaries."""

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

import pytest
import pytest_asyncio

from qagents import SQLiteStore
from qagents.messages import ToolCallPart
from qagents.session import Entry
from qagents.testing.crash import Faults, crash_test

from .effect_worker import build_agent

pytestmark = [
    pytest.mark.process,
    pytest.mark.skipif(
        os.environ.get("QAGENTS_RUN_PROCESS_TESTS") != "1",
        reason="run the subprocess recovery tests explicitly or in the process CI lane",
    ),
]


@dataclass
class Destination:
    applied: dict = field(default_factory=dict)
    sends: list = field(default_factory=list)
    observations: list = field(default_factory=list)

    def reply(self, event):
        kind = event["kind"]
        self.observations.append(event)
        if kind == "model":
            if event["role"] == "tool":
                return "Done."
            return {
                "type": "tool_call",
                "id": "C1",
                "name": "execute",
                "arguments": {"order_id": "R1"},
            }
        if kind == "send":
            self.sends.append(event)
            payload = {"order_id": event["order_id"], "amount": event["amount"]}
            previous = self.applied.setdefault(event["token"], payload)
            assert previous == payload
            return previous
        return None


async def launch(root, effect_name, now):
    entry = Path(__file__).with_name("effect_worker.py")
    source = entry.parents[2] / "src"
    return await asyncio.create_subprocess_exec(
        # Start the interpreter directly so the handle identifies the process we own.
        getattr(sys, "_base_executable", sys.executable),
        "-c",
        "import runpy,sys; sys.path[:0]=sys.argv[1].split(chr(31)); "
        "sys.argv=sys.argv[2:]; runpy.run_path(sys.argv[0], run_name='__main__')",
        chr(31).join([str(source), *sys.path]),
        str(entry),
        str(root),
        str(now),
        effect_name,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


async def drive(root, destination, effect_name, now, *, cut=None):
    worker = await launch(root, effect_name, now)
    try:
        greeting = json.loads(await asyncio.wait_for(worker.stdout.readline(), 12))
        assert greeting == {"kind": "hello", "pid": worker.pid} and worker.returncode is None
        worker.stdin.write(b"null\n")
        await worker.stdin.drain()
        async with asyncio.timeout(18):
            while line := await worker.stdout.readline():
                event = json.loads(line)
                reply = destination.reply(event)
                if cut is not None and cut(event):
                    worker.kill()
                    assert await worker.wait() != 0
                    return event
                worker.stdin.write((json.dumps(reply) + "\n").encode())
                await worker.stdin.drain()
        error = await worker.stderr.read()
        assert await asyncio.wait_for(worker.wait(), 4) == 0, error.decode(errors="replace")
        assert cut is None, "Worker completed without reaching the selected crash point"
        return event
    finally:
        if worker.returncode is None:
            worker.kill()
        await worker.communicate()


async def entries(root):
    async with SQLiteStore(root / "journal.db") as store:
        return await store.load("case")


def write_boundary(event, phase, kind):
    return event["kind"] == "write" and event["phase"] == phase and kind in event["entries"]


@pytest_asyncio.fixture(
    scope="module", loop_scope="module", params=["payments.refund", "provision.instance"]
)
async def reference(request, tmp_path_factory):
    effect_name = request.param
    root = tmp_path_factory.mktemp("process-reference")
    destination = Destination()
    assert await drive(root, destination, effect_name, 1000) == {
        "kind": "done",
        "text": "Done.",
        "status": "completed",
    }
    assert list(destination.applied.values()) == [{"order_id": "R1", "amount": 10}]
    journal = await entries(root)
    committed = next(item for item in journal if item.kind == "effect_committed")
    return effect_name, committed


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def simulated(reference, tmp_path_factory):
    effect_name, _ = reference
    return await crash_test(
        lambda env: build_agent(env.model, effect_name),
        "Apply R1",
        directory=tmp_path_factory.mktemp("simulated"),
        session_id="case",
        store="sqlite",
        model=[
            ToolCallPart(id="C1", name="execute", arguments={"order_id": "R1"}),
            "Done.",
        ],
        faults=Faults.parse("kill_before,kill_after,commit_then_crash"),
        mode="freeze",
        transport="fake",
        points="all-raw",
    )


def equivalent_point(report, phase, kind):
    if phase == "send":
        matches = [
            point for point in report.points if "commit_then_crash" in point.execution.point.kinds
        ]
    else:
        matches = [
            point
            for point in report.points
            if point.id.startswith(f"{phase}(")
            and any(
                value.split("/", 1)[0] == kind
                for value in unquote(point.id.split("/", 1)[1]).split(",")
            )
        ]
    assert len(matches) == 1, report.explain()
    return matches[0]


@pytest.mark.parametrize(
    ("phase", "kind", "sent_before", "journaled_before", "total_requests"),
    [
        ("before", "effect_intent", 0, False, 1),
        ("after", "effect_intent", 0, False, 1),
        ("send", "", 1, False, 2),
        ("before", "effect_committed", 1, False, 2),
        ("after", "effect_committed", 1, True, 1),
        ("before", "tool_result", 1, True, 1),
        ("after", "tool_result", 1, True, 1),
    ],
)
async def test_real_kill_preserves_one_application(
    tmp_path, reference, simulated, phase, kind, sent_before, journaled_before, total_requests
):
    effect_name, expected_commit = reference
    destination = Destination()

    def cut(event):
        return event["kind"] == "send" if phase == "send" else write_boundary(event, phase, kind)

    event = await drive(tmp_path, destination, effect_name, 1000, cut=cut)
    assert cut(event)
    before = await entries(tmp_path)
    assert bool([item for item in before if item.kind == "effect_committed"]) is journaled_before
    assert len(destination.sends) == len(destination.applied) == sent_before
    assert any(item.kind == "effect_intent" for item in before) is (
        (phase, kind) != ("before", "effect_intent")
    )

    assert await drive(tmp_path, destination, effect_name, 1062) == {
        "kind": "done",
        "text": "Done.",
        "status": "completed",
    }
    after = await entries(tmp_path)
    assert after[: len(before)] == before
    (committed,) = [item for item in after if item.kind == "effect_committed"]
    assert committed.data["receipt"] == expected_commit.data["receipt"]
    assert sum(item.kind == "tool_result" for item in after) == 1
    assert list(destination.applied.values()) == [expected_commit.data["receipt"]]
    assert len(destination.sends) == total_requests
    assert len({request["token"] for request in destination.sends}) == 1
    reached = {event["point"] for event in destination.observations if event["kind"] == "reach"}
    expected_first = (
        "dedupe_resend" if (phase, kind) == ("after", "effect_intent") else "first_send"
    )
    assert expected_first in reached

    simulated_point = equivalent_point(simulated, phase, kind)
    assert simulated_point.verdict == "safe", simulated.explain(simulated_point.id)
    assert simulated_point.outcome == "completed_as_reference"
    assert simulated_point.execution.final_snapshot is not None
    assert len(simulated_point.execution.applications) == len(destination.sends)
    assert sum(not app.replayed for app in simulated_point.execution.applications) == len(
        destination.applied
    )
    observed_reaches = [
        event["point"] for event in destination.observations if event["kind"] == "reach"
    ]
    assert [event.point for event in simulated_point.execution.events] == observed_reaches
    semantic_entries = {"effect_intent", "effect_committed", "tool_result", "run_finished"}
    expected_entries = [
        Entry.model_validate_json(raw)
        for raw in simulated_point.execution.final_snapshot.entries_json
    ]
    assert [e.kind for e in after if e.kind in semantic_entries] == [
        e.kind for e in expected_entries if e.kind in semantic_entries
    ]

    assert [item.fingerprint for item in simulated_point.execution.model_evidence] == [
        event["fingerprint"] for event in destination.observations if event["kind"] == "model"
    ]
    for application, observed in zip(
        simulated_point.execution.applications, destination.sends, strict=True
    ):
        assert (application.fake, application.attempt, application.epoch) == (
            effect_name,
            observed["attempt"],
            observed["epoch"],
        )
