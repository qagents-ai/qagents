"""Offline tests of examples/patterns/content_flow.py."""

from __future__ import annotations

import json
import runpy
import sys

import pytest

from examples.harness import last_text, main, scripted_context, system_text
from examples.patterns import content_flow
from examples.patterns.content_flow import FLOW_ID, run
from qagents import SQLiteStore, Workflow
from qagents.models.testing import FunctionModel

TOPIC = "Rain barrels for balcony gardens"
BLOG_POST = "# Rain barrels on a balcony\n\n" + " ".join(["Collect rain and water less."] * 30)
SOCIAL_POST = "Catch the rain: one small barrel waters a balcony garden all summer. #watersmart"
HYPED_POST = "A revolutionary barrel for your balcony! #watersmart"


def flow_policy(route: str, first: str, revised: str | None = None, prompts: list | None = None):
    """Answer every agent of the flow; the writer's reply depends on whether it revises."""

    def policy(request):
        system = system_text(request)
        if "suggest topics" in system:
            return f'"{TOPIC}"'  # quoted, as models sometimes do
        if "Pick the format" in system:
            return route
        if "blog posts" in system or "social media posts" in system:
            prompt = last_text(request)
            if prompts is not None:
                prompts.append((system.split(".")[0], prompt))
            return revised if "was rejected" in prompt else first
        raise AssertionError(f"unexpected request: {system[:60]!r}")

    return policy


async def flow_status(ctx):
    store = SQLiteStore(ctx.workdir / "flows.db")
    try:
        return await Workflow(store, FLOW_ID).status()
    finally:
        await store.aclose()


async def test_a_blocked_social_post_is_revised_then_published_on_the_second_pass(tmp_path, capsys):
    prompts: list[tuple[str, str]] = []
    ctx = scripted_context(flow_policy("social", HYPED_POST, SOCIAL_POST, prompts), tmp_path)

    outcome = await run(ctx)

    assert outcome.ok, outcome.summary
    details = outcome.details
    assert details["status"] == "published" and details["route"] == "social"
    assert details["topic"] == TOPIC and details["picked"] == "social"
    assert details["revised"] and outcome.output == SOCIAL_POST
    assert details["first_problems"] == [
        "blocked by no_banned_words: uses banned words: revolutionary"
    ]
    # Both drafts went to the social writer; the revision prompt carries the guardrail's reason.
    writer = "You write social media posts for a gardening magazine"
    assert prompts[0] == (writer, f"Topic: {TOPIC}")
    assert prompts[1][0] == writer and "uses banned words: revolutionary" in prompts[1][1]
    published = (ctx.workdir / "published" / "social.md").read_text(encoding="utf-8")
    assert published.strip() == SOCIAL_POST
    # topic, classifier, first draft, revision; the second pass replays them all
    assert ctx.meter.usage.requests == 4 and details["replayed_requests"] == 0
    printed = capsys.readouterr().out
    assert "Suspended: the flow waits for the signal 'publish'" in printed
    assert "waiting for ['publish']" in printed
    status = await flow_status(ctx)
    assert status.completed_steps == ["topic", "draft", "revise", "publish"]
    assert status.signals == ["publish"] and status.pending_signals == []


async def test_a_clean_blog_post_needs_no_revision(tmp_path):
    ctx = scripted_context(flow_policy("blog", BLOG_POST), tmp_path)

    outcome = await run(ctx)

    assert outcome.ok, outcome.summary
    assert outcome.details["route"] == "blog" and not outcome.details["revised"]
    assert outcome.details["file"] == "published/blog.md"
    assert ctx.meter.usage.requests == 3
    status = await flow_status(ctx)
    assert status.completed_steps == ["topic", "draft", "publish"]


async def test_a_classifier_answer_that_names_no_route_falls_back_to_the_blog(tmp_path):
    ctx = scripted_context(flow_policy("A blog post, I think.", BLOG_POST), tmp_path)

    outcome = await run(ctx)

    assert outcome.ok, outcome.summary
    assert outcome.details["route"] == "blog"
    assert outcome.details["picked"] == "A blog post, I think."


async def test_a_declined_piece_is_not_published(tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "no")
    ctx = scripted_context(flow_policy("social", SOCIAL_POST), tmp_path, interactive=True)

    outcome = await run(ctx)

    assert not outcome.ok and outcome.details["status"] == "declined"
    assert not (ctx.workdir / "published").exists()
    assert (await flow_status(ctx)).signals == ["publish"]


async def test_a_draft_that_fails_twice_ends_the_flow_before_the_signal(tmp_path):
    too_long = "Water early. " * 40  # over the social limit, twice
    ctx = scripted_context(flow_policy("social", too_long, too_long), tmp_path)

    outcome = await run(ctx)

    assert not outcome.ok
    assert outcome.summary == "The flow ended before the signal: rejected"
    assert "characters; the limit is 280" in outcome.details["problems"][0]
    status = await flow_status(ctx)
    assert status.completed_steps == ["topic", "draft", "revise"]
    assert status.pending_signals == [] and status.signals == []


def test_runs_as_a_module_and_replays_from_its_cassette(tmp_path, monkeypatch, capsys):
    common = ["--provider", "qcloud", "--cassette-dir", str(tmp_path / "cassettes"), "--json"]
    with pytest.raises(SystemExit) as recorded:
        main(
            run,
            name=content_flow.NAME,
            description=content_flow.DESCRIPTION,
            inspired_by=content_flow.INSPIRED_BY,
            argv=[*common, "--record", "--workdir", str(tmp_path / "recorded")],
            model_factory=lambda provider, model: FunctionModel(
                flow_policy("social", HYPED_POST, SOCIAL_POST)
            ),
        )
    assert recorded.value.code == 0
    capsys.readouterr()

    replay = [*common, "--replay", "--workdir", str(tmp_path / "replayed")]
    monkeypatch.setattr(sys, "argv", ["content_flow", *replay])
    with pytest.raises(SystemExit) as replayed:
        runpy.run_path(content_flow.__file__, run_name="__main__")
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert replayed.value.code == 0, report
    assert report["ok"] and report["name"] == "content_flow" and report["requests"] == 4
