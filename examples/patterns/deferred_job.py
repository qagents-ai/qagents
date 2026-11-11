"""A long-running external job: the tool submits it, the run pauses, a later run gets the result.

``render_video`` submits a render job to a fake in-process render farm and raises
`DeferToolCall` with the job id in its metadata. The run pauses with a `PendingInput`, kept in
a SQLite journal; nothing waits in memory. The example continues it in two ways:

1. Poll: a later "process" (a new store and a new `Agent`) reads the pending inputs from the
   journal, polls the farm until the job is done, and resumes the run with
   ``agent.resume_stream(session, results={tool_call_id: job_result})``.
2. Signal: the agent is a `Workflow` step that waits for a ``job:<id>`` signal. Whoever learns
   that the job is done (a webhook handler; here the example) sends it with
   `Workflow.send_signal`, and the next run of the workflow resumes the step with it.

Either way the tool body runs once per call, and the job's result is the tool's result.

Mirrors: LangGraph's ``interrupt()`` inside a node, resumed with ``Command(resume=...)`` on the
same thread, and CrewAI's ``Task(async_execution=True)``. In qagents:

- LangGraph runs the interrupted node again from its start on resume, so work done before
  ``interrupt()`` repeats unless guarded. Here the tool is never called again for that call;
  the value given to ``results=`` becomes its result.
- CrewAI runs an async task in the background of a crew that waits for it in memory. Here the
  pause is in the journal, so no process waits, and any process can resume it.
- The tool is ``idempotent`` and passes ``ctx.operation_id`` to the farm as the idempotency
  key: if the process dies after submitting but before the pause is journaled, the call runs
  again on resume and the farm returns the same job.

Run it:

    python -m examples.patterns.deferred_job --provider anthropic
    python -m examples.patterns.deferred_job --provider qcloud
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from examples.harness import ExampleContext, ExampleOutcome, main
from qagents import (
    Agent,
    DeferToolCall,
    Limits,
    RunContext,
    RunResult,
    Session,
    SQLiteStore,
    Workflow,
    WorkflowSuspended,
    tool,
)

NAME = "deferred_job"
DESCRIPTION = "A tool defers to an external render job; the run resumes later with its result"
INSPIRED_BY = "LangGraph interrupt() for async work / CrewAI async tasks (external jobs)"

INTRO = "Render the scene 'intro': 10 seconds at 24 frames per second."
OUTRO = "Render the scene 'outro': 6 seconds at 30 frames per second."
EXPECTED_FRAMES = {"intro": 240, "outro": 180}
SESSION_ID = "render:intro"
WORKFLOW_ID = "render:outro"
MAX_POLLS = 20


class RenderFarm:
    """A fake external render service. A job is done after ``ticks`` polls.

    ``submit`` is idempotent by ``key``: the same key returns the same job.
    """

    def __init__(self, ticks: int = 3) -> None:
        self.ticks = ticks
        self.jobs: dict[str, dict[str, Any]] = {}
        self.submissions: list[str] = []
        """Every call of ``submit`` (that is, every run of the tool body), by scene."""
        self._by_key: dict[str, str] = {}
        self._left: dict[str, int] = {}

    def submit(self, scene: str, seconds: int, fps: int, *, key: str) -> str:
        self.submissions.append(scene)
        if key in self._by_key:
            return self._by_key[key]
        job_id = f"job-{len(self.jobs) + 1:03d}"
        slug = re.sub(r"[^a-z0-9]+", "-", scene.lower()).strip("-") or "scene"
        self.jobs[job_id] = {"file": f"renders/{slug}.mp4", "frames": seconds * fps}
        self._left[job_id] = self.ticks
        self._by_key[key] = job_id
        return job_id

    def poll(self, job_id: str) -> dict[str, Any]:
        self._left[job_id] = left = max(self._left[job_id] - 1, 0)
        if left:
            done = round(100 * (self.ticks - left) / self.ticks)
            return {"job_id": job_id, "status": "running", "progress_pct": done}
        return {"job_id": job_id, "status": "done", **self.jobs[job_id]}


@tool(idempotent=True)  # safe to repeat after a crash: the farm dedupes by operation_id
async def render_video(
    ctx: RunContext[RenderFarm],
    scene: str,
    seconds: Annotated[int, Field(ge=1, le=120)],
    fps: Annotated[int, Field(ge=1, le=120)] = 24,
) -> dict[str, Any]:
    """Render a scene to a video file on the render farm. Rendering takes a while; the
    result (the file and its frame count) arrives when the job is done.

    Args:
        scene: Name of the scene, for example "intro".
        seconds: Length of the video in seconds.
        fps: Frames per second.
    """
    job_id = ctx.deps.submit(scene, seconds, fps, key=ctx.operation_id or "")
    raise DeferToolCall("rendering", metadata={"job_id": job_id})


def build_renderer(ctx: ExampleContext) -> Agent:
    return Agent(
        ctx.model,
        name="renderer",
        instructions="You prepare videos on a render farm. Call render_video exactly once, "
        "with the scene name, the length in seconds and the frames per second from the "
        "request. When it returns, reply in one sentence with the output file and the number "
        "of frames.",
        tools=[render_video],
        limits=Limits(max_turns=4, max_cost_usd=0.10),
        hooks=ctx.hooks,
    )


async def wait_for_job(ctx: ExampleContext, farm: RenderFarm, job_id: str) -> dict[str, Any]:
    """Poll the farm until the job is done (a real poller would sleep between ticks)."""
    for tick in range(1, MAX_POLLS + 1):
        status = farm.poll(job_id)
        ctx.log(f"poll {tick}: {job_id} {status['status']}, {status.get('progress_pct', 100)}%")
        if status["status"] == "done":
            return status
    raise TimeoutError(f"{job_id} was not done after {MAX_POLLS} polls")


# -- variant 1: pause, poll, resume with results= --------------------------------------------


async def resume_with_results(ctx: ExampleContext, farm: RenderFarm, db: Path) -> RunResult:
    ctx.log("Variant 1: the run pauses on the render job, then resumes with results=")
    store = SQLiteStore(db)
    try:
        session = Session(store, id=SESSION_ID)
        paused = await ctx.print_stream(
            build_renderer(ctx).stream(INTRO, session=session, deps=farm)
        )
    finally:
        await store.aclose()
    if paused.status != "paused":
        return paused

    ctx.log("Later, maybe in another process: the journal says which calls wait for what")
    store = SQLiteStore(db)
    try:
        session = Session(store, id=SESSION_ID)
        waiting = (await session.state()).pending_inputs
        results = {
            item.tool_call_id: await wait_for_job(ctx, farm, item.metadata["job_id"])
            for item in waiting
        }
        agent = build_renderer(ctx)  # a new Agent object resumes the run
        return await ctx.print_stream(agent.resume_stream(session, results=results, deps=farm))
    finally:
        await store.aclose()


# -- variant 2: a workflow step that waits for a signal -----------------------------------------


async def render_flow(wf: Workflow, agent: Agent, farm: RenderFarm) -> RunResult:
    """Run the agent step; while it waits for jobs, wait for their ``job:<id>`` signals.

    On a later run the first ``run_agent`` finds the step's run paused and resumes it with
    nothing to give, which only pauses it again (no model request); then the signals are
    there and the second ``run_agent`` delivers them.
    """
    result = await wf.run_agent("render", agent, OUTRO, deps=farm)
    while result.pending_inputs:
        results = {
            item.tool_call_id: await wf.wait_for_signal(f"job:{item.metadata['job_id']}")
            for item in result.pending_inputs
        }
        result = await wf.run_agent("render", agent, OUTRO, results=results, deps=farm)
    return result


async def resume_with_signal(ctx: ExampleContext, farm: RenderFarm, db: Path) -> RunResult:
    ctx.log("Variant 2: a workflow step waits for the job's signal")
    for _ in range(3):  # each pass is a fresh "process": new store, new agent
        store = SQLiteStore(db)
        try:
            wf = Workflow(store, WORKFLOW_ID)
            try:
                result = await render_flow(wf, build_renderer(ctx), farm)
                ctx.log(f"Step 'render' {result.status}: {result.text.strip()}")
                return result
            except WorkflowSuspended as suspended:
                signal = suspended.signal or ""
            status = await wf.status()
            step = status.open_agent_steps[0]
            ctx.log(f"Suspended: step {step.name!r} is {step.status}; waiting for {signal!r}")
            job = await wait_for_job(ctx, farm, signal.removeprefix("job:"))
            await Workflow.send_signal(store, WORKFLOW_ID, signal, job)  # what a webhook does
        finally:
            await store.aclose()
    raise RuntimeError("The workflow kept waiting for more render jobs.")


# -- the example -------------------------------------------------------------------------------


def check(scene: str, result: RunResult, jobs: list[dict[str, Any]], runs: int) -> list[str]:
    """What went wrong in one variant (nothing when it worked)."""
    if result.status != "completed":
        return [f"{scene}: the run ended {result.status} ({result.stop_reason})"]
    if runs != 1 or len(jobs) != 1:
        return [f"{scene}: the tool body ran {runs} times and submitted {len(jobs)} jobs"]
    job, problems = jobs[0], []
    if job["frames"] != EXPECTED_FRAMES[scene]:
        problems.append(f"{scene}: rendered {job['frames']} frames, not {EXPECTED_FRAMES[scene]}")
    if str(job["frames"]) not in result.text or Path(job["file"]).name not in result.text:
        problems.append(f"{scene}: the answer does not report {job['file']} and its frames")
    return problems


async def run(ctx: ExampleContext) -> ExampleOutcome:
    farm, db = RenderFarm(), ctx.workdir / "renders.db"
    answers, problems = [], []
    for scene, variant in [("intro", resume_with_results), ("outro", resume_with_signal)]:
        runs, jobs = len(farm.submissions), len(farm.jobs)
        result = await variant(ctx, farm, db)
        new_jobs = list(farm.jobs.values())[jobs:]
        problems += check(scene, result, new_jobs, len(farm.submissions) - runs)
        answers.append(result.text.strip())

    return ExampleOutcome(
        ok=not problems,
        summary="; ".join(problems) or " | ".join(answers),
        output=answers,
        details={"jobs": farm.jobs, "submissions": farm.submissions, "problems": problems},
    )


if __name__ == "__main__":
    main(run, name=NAME, description=DESCRIPTION, inspired_by=INSPIRED_BY)
