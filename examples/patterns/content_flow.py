"""A content flow: pick a topic, route it to a blog or a social writer, check, wait, publish.

The flow is one async function over durable `Workflow` steps. An idea agent suggests a topic;
`route()` asks a classifier for the format and runs the blog or the social writer; an output
guardrail blocks banned words and plain Python checks the length; a rejected draft gets one
revision. Then the flow waits for an editor's "publish" signal. The example runs the flow twice,
as two processes would: the first pass stops with `WorkflowSuspended`, the example sends the
signal (or asks you, with ``--interactive``), and the second pass, with a new store and new
agents, replays every finished step without a model request, and saves a local artifact.
The artifact is reproducible from the journaled draft and is checked on every pass. This
example does not publish to an external service: that belongs in an agent tool with an
idempotency key enforced by the service, or explicit reconciliation of unknown outcomes.

Mirrors: CrewAI Flows. The pieces map like this:

- ``@start()`` method -> ``wf.run_agent("topic", ...)``
- ``@router()`` that returns "blog" or "social" -> the classifier agent given to `route`
- ``@listen("blog")`` and ``@listen("social")`` -> the ``routes`` mapping of `route`
- ``@listen(write)`` for the checks and the revision -> plain code and ``wf.step("revise")``
- a typed ``Flow[State]`` kept by ``@persist`` -> the `Draft` each step returns, journaled
- waiting for the editor -> ``wf.wait_for_signal("publish")``

In qagents: no decorators and no event graph; the order is the code and ``if`` decides. There
is no mutable state object saved after each method: each step's result is recorded once and
replayed, so a flow that waits for days needs no process to stay alive, and the next run with
the same flow id continues where it stopped.

Run it:

    python -m examples.patterns.content_flow --provider anthropic
    python -m examples.patterns.content_flow --provider qcloud --interactive
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from examples.harness import ExampleContext, ExampleOutcome, main
from qagents import (
    Agent,
    GuardrailTripped,
    Limits,
    RunContext,
    RunResult,
    SQLiteStore,
    Workflow,
    WorkflowSuspended,
    route,
)

NAME = "content_flow"
DESCRIPTION = "A durable content flow: topic, routed writer, guardrail, publish signal, resume"
INSPIRED_BY = "CrewAI Flows (@start, @listen, @router, state)"

FLOW_ID = "content:water-wise-garden"
THEME = "saving water in a small home garden"
BANNED = ("revolutionary", "game-changer", "game changer", "synergy", "disrupt")
SOCIAL_MAX_CHARS = 280
BLOG_WORDS = (100, 400)


class Draft(BaseModel):
    """A draft as the flow records it: the flow's state after the "draft" and "revise" steps."""

    route: Literal["blog", "social"]
    text: str
    problems: list[str] = Field(default_factory=list)
    """Why the draft cannot be published; empty when it can."""
    picked: str = ""
    """What the classifier answered (first draft only). When it names no route, the route
    is route()'s default."""


class PublishDecision(BaseModel):
    """The value of the "publish" signal."""

    approve: bool
    by: str = "editor"


# -- checks -----------------------------------------------------------------------------------


def banned_in(text: str) -> list[str]:
    lowered = text.lower()
    return [word for word in BANNED if word in lowered]


def no_banned_words(ctx: RunContext, output: Any) -> None:
    """Output guardrail of both writers: a draft with a banned word never leaves the run."""
    if found := banned_in(str(output)):
        raise GuardrailTripped(f"uses banned words: {', '.join(found)}")


def check_draft(route: str, text: str) -> list[str]:
    """The plain-Python check: what is wrong with a draft (nothing when it passes)."""
    if route == "social" and len(text) > SOCIAL_MAX_CHARS:
        return [f"the post has {len(text)} characters; the limit is {SOCIAL_MAX_CHARS}"]
    words = len(text.split())
    if route == "blog" and not BLOG_WORDS[0] <= words <= BLOG_WORDS[1]:
        return [f"the post has {words} words; write {BLOG_WORDS[0]} to {BLOG_WORDS[1]}"]
    return []


# -- the agents ---------------------------------------------------------------------------------


@dataclass
class Crew:
    ideas: Agent
    classifier: Agent
    writers: dict[str, Agent]


def build_crew(ctx: ExampleContext) -> Crew:
    """The four agents. Each runs as a top-level run, so each gets ``hooks=ctx.hooks``."""
    limits = Limits(max_turns=3, max_cost_usd=0.10)
    avoid = f"Never use these words: {', '.join(BANNED)}."
    ideas = Agent(
        ctx.model,
        name="ideas",
        instructions="You suggest topics for a gardening magazine. Reply with one topic title "
        "only, at most 12 words, on one line, without quotes.",
        limits=limits,
        hooks=ctx.hooks,
    )
    # Plain text: route() matches the answer to a route name, ignoring case, quotes and a
    # final full stop. (An output_type=Literal[...] classifier would be checked and corrected.)
    classifier = Agent(
        ctx.model,
        name="format_picker",
        instructions="Pick the format for the topic you are given: blog for a topic that needs "
        "a few paragraphs of explanation, social for one quick tip. Answer with the single "
        "word blog or social, nothing else.",
        limits=limits,
        hooks=ctx.hooks,
    )
    blog = Agent(
        ctx.model,
        name="blog_writer",
        instructions="You write blog posts for a gardening magazine. Write one post of 150 to "
        "250 words about the topic you are given: a title line, then short paragraphs. "
        f"{avoid} Reply with the post only, with no preamble.",
        output_guardrails=[no_banned_words],
        limits=limits,
        hooks=ctx.hooks,
    )
    social = blog.copy(  # same model, limits, hooks and guardrail
        name="social_writer",
        instructions="You write social media posts for a gardening magazine. Write one post of "
        f"at most 200 characters about the topic you are given, with one hashtag. {avoid} "
        "Reply with the post only, with no preamble.",
    )
    return Crew(ideas, classifier, {"blog": blog, "social": social})


# -- the flow -------------------------------------------------------------------------------


def first_line(text: str) -> str:
    """The first non-empty line, without the quotes or Markdown marks a model may add."""
    return next((line.strip(" \"'*#") for line in text.splitlines() if line.strip()), "")


def review(route: str, result: RunResult, *, picked: str = "") -> Draft:
    """Record a writer's run as a `Draft`: blocked by the guardrail, stopped, or length-checked."""
    text = result.text.strip()  # empty when the guardrail blocked the answer
    if result.status == "blocked" and result.guardrail is not None:
        problems = [f"blocked by {result.guardrail.name}: {result.guardrail.reason}"]
    elif result.status != "completed":
        problems = [f"{result.agent} stopped early ({result.stop_reason or result.status})"]
    else:
        problems = check_draft(route, text)
    return Draft(route=route, text=text, problems=problems, picked=picked)


async def first_draft(crew: Crew, topic: str) -> Draft:
    """The @router and its @listen methods: route() picks the writer, then runs it."""
    routed = await route(crew.classifier, crew.writers, f"Topic: {topic}", default="blog")
    # routed.route is None when the classifier itself stopped; routed.result then says why.
    return review(routed.route or "blog", routed.result, picked=routed.classification.text)


async def revision(crew: Crew, topic: str, draft: Draft) -> Draft:
    """One more try for a rejected draft, told what was wrong (the guardrail's reason
    is not sent to the model by qagents, so the flow passes it on)."""
    feedback = "; ".join(draft.problems)
    prompt = f"Topic: {topic}\n\nA first draft was rejected: {feedback}. Write a new post."
    return review(draft.route, await crew.writers[draft.route].run(prompt))


def publish(folder: Path, draft: Draft) -> str:
    """Materialize the recorded draft locally; replay repairs a missing artifact.

    An existing different file (including a partial write) requires inspection.
    Repeating this operation never changes an already complete artifact.
    """
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{draft.route}.md"
    content = draft.text + "\n"
    try:
        with path.open("x", encoding="utf-8") as output:
            output.write(content)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(
                f"Artifact {path} differs from the recorded draft; inspect it first."
            ) from None
    return f"{folder.name}/{draft.route}.md"


async def content_flow(wf: Workflow, crew: Crew, out: Path) -> dict[str, Any]:
    """Replay recorded computations, wait for approval, and materialize the local artifact."""
    idea = await wf.run_agent("topic", crew.ideas, f"Suggest one topic about {THEME}.")  # @start
    topic = first_line(idea.text) if idea.completed else ""
    if not topic:
        return {"status": "no topic", "reason": idea.stop_reason or idea.status}
    draft = await wf.step("draft", lambda: first_draft(crew, topic), type=Draft)  # @router
    if draft.problems:
        draft = await wf.step("revise", lambda: revision(crew, topic, draft), type=Draft)
    if draft.problems:
        return {"status": "rejected", "topic": topic, "draft": draft}
    decision = await wf.wait_for_signal("publish", type=PublishDecision)  # may suspend
    if not decision.approve:
        return {"status": "declined", "topic": topic, "draft": draft}
    # Recreate the same local artifact on replay; no external send is hidden in a step.
    file = publish(out, draft)
    return {"status": "published", "topic": topic, "draft": draft, "file": file, "by": decision.by}


# -- the example -------------------------------------------------------------------------------


async def run(ctx: ExampleContext) -> ExampleOutcome:
    db, out = ctx.workdir / "flows.db", ctx.workdir / "published"

    ctx.log("First pass: topic, route, write, check, then wait for 'publish'")
    store = SQLiteStore(db)
    try:
        try:
            early = await content_flow(Workflow(store, FLOW_ID), build_crew(ctx), out)
        except WorkflowSuspended as waiting:
            ctx.log(f"Suspended: the flow waits for the signal {waiting.signal!r}")
        else:
            problems = early["draft"].problems if "draft" in early else [early["reason"]]
            return ExampleOutcome(
                ok=False,
                summary=f"The flow ended before the signal: {early['status']}",
                details={"status": early["status"], "problems": problems},
            )
        # What an editor's screen would read: the journal, without running anything.
        flow = Workflow(store, FLOW_ID)
        status, steps = await flow.status(), await flow.completed_steps()
        first = Draft.model_validate(steps["draft"])
        draft = Draft.model_validate(steps["revise"]) if "revise" in steps else first
        if first.problems:
            ctx.log(f"The first draft was rejected ({'; '.join(first.problems)}) and revised")
        ctx.log(
            f"Finished steps: {status.completed_steps}; waiting for {status.pending_signals}. "
            f"The classifier answered {first.picked.strip()!r}. The {draft.route} draft:"
        )
        print(draft.text)
        answer = await ctx.ask("Publish this piece?", default="yes")
        decision = PublishDecision(approve=answer.strip().lower().startswith("y"))
        await Workflow.send_signal(store, FLOW_ID, "publish", decision)  # from any process
    finally:
        await store.aclose()

    ctx.log("Second pass, as a new process would run it: new store, new agents, same flow id")
    before = ctx.meter.usage.requests if ctx.meter else 0
    store = SQLiteStore(db)
    try:
        final = await content_flow(Workflow(store, FLOW_ID), build_crew(ctx), out)
    finally:
        await store.aclose()
    replayed_requests = (ctx.meter.usage.requests if ctx.meter else 0) - before

    published = final["status"] == "published"
    text = (ctx.workdir / final["file"]).read_text(encoding="utf-8").strip() if published else ""
    clean = text == draft.text and not check_draft(draft.route, text) and not banned_in(text)
    revised = "revise" in steps
    summary = (
        f"{final['status']}: a {draft.route} piece on {final['topic']!r}"
        f"{', revised once' if revised else ''}; the second pass made {replayed_requests} "
        "model requests"
    )
    return ExampleOutcome(
        ok=published and clean and replayed_requests == 0,
        summary=summary,
        output=draft.text,
        details={
            "status": final["status"],
            "topic": final["topic"],
            "route": draft.route,
            "picked": first.picked,
            "revised": revised,
            "first_problems": first.problems,
            "file": final.get("file"),
            "replayed_requests": replayed_requests,
        },
    )


if __name__ == "__main__":
    main(run, name=NAME, description=DESCRIPTION, inspired_by=INSPIRED_BY)
