"""Offline tests of examples/patterns/deferred_job.py."""

from __future__ import annotations

import json
import runpy
import sys

import pytest

from examples.harness import last_text, main, scripted_context
from examples.patterns import deferred_job
from examples.patterns.deferred_job import (
    INTRO,
    SESSION_ID,
    WORKFLOW_ID,
    RenderFarm,
    build_renderer,
    run,
    wait_for_job,
)
from qagents import Session, SQLiteStore, Workflow
from qagents.models.testing import FunctionModel, call


def render_call(prompt: str):
    if "'intro'" in prompt:
        return call("render_video", scene="intro", seconds=10, fps=24)
    return call("render_video", scene="outro", seconds=6, fps=30)


def render_policy(request):
    """Call render_video for the scene in the prompt; answer from the job result."""
    last = request.messages[-1]
    if last.role == "user":
        return render_call(last.text)
    job = json.loads(last_text(request))
    return f"Rendered {job['file']} with {job['frames']} frames."


async def test_both_variants_resume_with_the_job_result_and_submit_once(tmp_path, capsys):
    ctx = scripted_context(render_policy, tmp_path)

    outcome = await run(ctx)

    assert outcome.ok, outcome.summary
    assert outcome.output == [
        "Rendered renders/intro.mp4 with 240 frames.",
        "Rendered renders/outro.mp4 with 180 frames.",
    ]
    assert outcome.details["submissions"] == ["intro", "outro"]  # the tool body ran once each
    # two requests per variant: the resume without results in variant 2 asks no model
    assert ctx.meter.usage.requests == 4
    # the model saw the finished job as the tool's result, not as a new message
    results = [r.content for m in ctx.meter.requests[1].messages for r in m.tool_results]
    assert json.loads(results[0])["status"] == "done"
    printed = capsys.readouterr().out
    assert 'render_video(scene="intro", seconds=10, fps=24) needs input' in printed
    assert "(run paused" in printed
    assert "poll 3: job-001 done, 100%" in printed
    assert "step 'render' is paused; waiting for 'job:job-002'" in printed

    store = SQLiteStore(ctx.workdir / "renders.db")
    try:
        entries = await Session(store, id=SESSION_ID).entries()
        deferred = [e.data["input"] for e in entries if e.kind == "tool_deferred"]
        assert [d["metadata"] for d in deferred] == [{"job_id": "job-001"}]
        assert (await Session(store, id=SESSION_ID).state()).open_run_id is None
        status = await Workflow(store, WORKFLOW_ID).status()
        assert status.completed_steps == ["render"] and status.signals == ["job:job-002"]
        assert status.open_agent_steps == [] and status.pending_signals == []
    finally:
        await store.aclose()


async def test_a_second_render_call_fails_the_once_check(tmp_path):
    def twice(request):
        last = request.messages[-1]
        if last.role == "user":
            return [render_call(last.text), render_call(last.text)]
        jobs = [json.loads(result.content) for result in last.tool_results]
        return f"Rendered {jobs[0]['file']} with {jobs[0]['frames']} frames."

    outcome = await run(scripted_context(twice, tmp_path))

    assert not outcome.ok
    assert "intro: the tool body ran 2 times and submitted 2 jobs" in outcome.summary
    assert "outro: the tool body ran 2 times and submitted 2 jobs" in outcome.summary


class Crash(BaseException):
    """Stands in for the process dying: no qagents code catches it."""


class CrashingFarm(RenderFarm):
    """Accepts the first job, then "dies" before the tool can raise DeferToolCall."""

    def submit(self, scene: str, seconds: int, fps: int, *, key: str) -> str:
        job_id = super().submit(scene, seconds, fps, key=key)
        if len(self.submissions) == 1:
            raise Crash
        return job_id


async def test_a_crash_after_submitting_reuses_the_job_on_resume(tmp_path):
    farm, ctx = CrashingFarm(), scripted_context(render_policy, tmp_path)
    store = SQLiteStore(tmp_path / "crash.db")
    try:
        session = Session(store, id="render:crash")
        with pytest.raises(Crash):
            await build_renderer(ctx).run(INTRO, session=session, deps=farm)
        state = await session.state()
        assert state.open_run_id is not None and len(state.started_tools) == 1

        # idempotent=True: the interrupted call runs again; the same operation_id finds the job
        paused = await build_renderer(ctx).resume(session, deps=farm)
        assert paused.status == "paused"
        assert farm.submissions == ["intro", "intro"] and list(farm.jobs) == ["job-001"]
        [waiting] = paused.pending_inputs
        assert waiting.metadata == {"job_id": "job-001"}

        job = await wait_for_job(ctx, farm, "job-001")
        done = await build_renderer(ctx).resume(
            session, results={waiting.tool_call_id: job}, deps=farm
        )
    finally:
        await store.aclose()
    assert done.completed and done.text == "Rendered renders/intro.mp4 with 240 frames."
    assert ctx.meter.usage.requests == 2  # the resumed tool call needed no model request


def test_the_farm_finishes_a_job_after_its_ticks_and_dedupes_by_key():
    farm = RenderFarm(ticks=2)
    job = farm.submit("Big Intro", 2, 10, key="op-1")
    assert farm.submit("Big Intro", 2, 10, key="op-1") == job and len(farm.jobs) == 1
    assert farm.submit("Big Intro", 2, 10, key="op-2") != job
    assert farm.poll(job)["status"] == "running"
    assert farm.poll(job) == {
        "job_id": job,
        "status": "done",
        "file": "renders/big-intro.mp4",
        "frames": 20,
    }


def test_runs_as_a_module_and_replays_from_its_cassette(tmp_path, monkeypatch, capsys):
    common = ["--provider", "qcloud", "--cassette-dir", str(tmp_path / "cassettes"), "--json"]
    with pytest.raises(SystemExit) as recorded:
        main(
            run,
            name=deferred_job.NAME,
            description=deferred_job.DESCRIPTION,
            inspired_by=deferred_job.INSPIRED_BY,
            argv=[*common, "--record", "--workdir", str(tmp_path / "recorded")],
            model_factory=lambda provider, model: FunctionModel(render_policy),
        )
    assert recorded.value.code == 0
    capsys.readouterr()

    replay = [*common, "--replay", "--workdir", str(tmp_path / "replayed")]
    monkeypatch.setattr(sys, "argv", ["deferred_job", *replay])
    with pytest.raises(SystemExit) as replayed:
        runpy.run_path(deferred_job.__file__, run_name="__main__")
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert replayed.value.code == 0, report
    assert report["ok"] and report["name"] == "deferred_job" and report["requests"] == 4
