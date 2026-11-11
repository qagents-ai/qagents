"""Offline tests of examples/patterns/map_reduce.py."""

from __future__ import annotations

import json
import re
import runpy
import sys

import pytest

from examples.harness import last_text, main, scripted_context, system_text
from examples.patterns import map_reduce
from examples.patterns.map_reduce import (
    DOCUMENTS,
    INBOX,
    fill_inbox,
    missing_titles,
    run,
    write_brief,
)
from qagents import SQLiteStore
from qagents.models.testing import FunctionModel

STEPS = sorted(["collect", *(f"summarise[{i}]" for i in range(len(DOCUMENTS))), "reduce"])


def policy(request):
    """Plays the summariser (one per document) and the reducer."""
    if system_text(request).startswith("You summarise one document"):
        title = re.match(r"Title: (.+)", last_text(request)).group(1)
        return f"{title}: the figures went the right way."
    assert system_text(request).startswith("You write a combined brief")
    titles = re.findall(r"^## (.+)$", last_text(request), re.MULTILINE)
    return "\n".join(f"- **{title}**: see the summary." for title in titles)


class Tracking(FunctionModel):
    """A FunctionModel that streams slowly and records how many requests overlapped."""

    def __init__(self, function) -> None:
        super().__init__(function, delay=0.002)
        self.active = self.peak = 0

    async def stream(self, request):
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            async for event in super().stream(request):
                yield event
        finally:
            self.active -= 1


async def test_the_second_run_replays_every_step_without_a_request(tmp_path, capsys):
    model = Tracking(policy)
    ctx = scripted_context(model, tmp_path)
    outcome = await run(ctx)
    assert outcome.ok, outcome.summary
    details = outcome.details
    assert (details["first_requests"], details["second_requests"]) == (7, 0)
    assert sorted(details["steps"]) == STEPS and details["missing_titles"] == []
    assert model.peak == 3  # six summaries, never more than three at a time
    printed = capsys.readouterr().out
    assert printed.count("-- summarising ") == 6  # all in run 1: run 2 called no step function
    # The summaries finished in any order, but the reducer got them in document order.
    reduce_request = ctx.meter.requests[-1]
    assert re.findall(r"^## (.+)$", last_text(reduce_request), re.MULTILINE) == list(DOCUMENTS)


async def test_a_failed_item_is_the_only_one_that_runs_again(tmp_path, capsys):
    def flaky(request):
        if "Title: Bike Lane Counts" in last_text(request):
            raise RuntimeError("the provider is down")
        return policy(request)

    store = SQLiteStore(tmp_path / "jobs.db")
    fill_inbox(tmp_path / "work" / INBOX)  # scripted_context works in tmp_path / "work"
    try:
        with pytest.raises(RuntimeError, match="the provider is down"):
            await write_brief(scripted_context(flaky, tmp_path), store)
        capsys.readouterr()
        # The five other summaries were journaled; the next run makes up the missing one.
        ctx = scripted_context(policy, tmp_path)
        brief = await write_brief(ctx, store)
    finally:
        await store.aclose()
    assert missing_titles(brief) == []
    assert ctx.meter.usage.requests == 2  # one summary and the reduce
    assert capsys.readouterr().out.count("-- summarising ") == 1


async def test_the_item_list_is_journaled_so_a_replay_maps_over_the_same_items(tmp_path):
    store = SQLiteStore(tmp_path / "jobs.db")
    fill_inbox(tmp_path / "work" / INBOX)
    try:
        first = await write_brief(scripted_context(policy, tmp_path), store)
        late = tmp_path / "work" / INBOX / "07.txt"
        late.write_text("Late Report\nArrived after run 1.\n", encoding="utf-8")
        ctx = scripted_context(policy, tmp_path)
        second = await write_brief(ctx, store)
    finally:
        await store.aclose()
    # The collect step was replayed, so the late document is not part of this job.
    assert ctx.meter.usage.requests == 0 and second == first and "Late Report" not in second


def test_missing_titles_ignores_case_and_markup():
    brief = "\n".join(f"- **{title.upper()}**: fine." for title in DOCUMENTS)
    assert missing_titles(brief) == []
    assert missing_titles("Ferry pilot results, nothing else") == [
        title for title in DOCUMENTS if title != "Ferry Pilot Results"
    ]


def test_runs_as_a_module_and_replays_from_its_cassette(tmp_path, monkeypatch, capsys):
    common = ["--provider", "anthropic", "--cassette-dir", str(tmp_path / "cassettes"), "--json"]
    with pytest.raises(SystemExit) as recorded:
        main(
            run,
            name=map_reduce.NAME,
            description=map_reduce.DESCRIPTION,
            inspired_by=map_reduce.INSPIRED_BY,
            argv=[*common, "--record", "--workdir", str(tmp_path / "recorded")],
            model_factory=lambda provider, model: FunctionModel(policy),
        )
    assert recorded.value.code == 0
    capsys.readouterr()

    argv = ["map_reduce", *common, "--replay", "--workdir", str(tmp_path / "replayed")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as replayed:
        runpy.run_path(map_reduce.__file__, run_name="__main__")
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert replayed.value.code == 0, report
    assert report["ok"] and report["name"] == "map_reduce" and report["requests"] == 7
