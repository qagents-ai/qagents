"""Offline tests of examples/patterns/research_crew.py: three tasks, explicit context, replay."""

from __future__ import annotations

import json
import runpy
import sys

import pytest

from examples.harness import final_answer, main, scripted_context, system_text, tool_names
from examples.patterns import research_crew
from examples.patterns.research_crew import ResearchNotes, run, search
from qagents.models.testing import FunctionModel, call

FACTS = [
    {"statement": "17 battery-electric ferries enter service 2027-2031.", "source_id": "src-1"},
    {"statement": "The budget is EUR 436 million, 60% from the EU.", "source_id": "src-2"},
    {"statement": "Each ferry carries 388 passengers.", "source_id": "src-3"},
    {"statement": "Piraeus to Aegina opens in May 2027.", "source_id": "src-4"},
]
BODY = (
    "Greece will put 17 battery-electric ferries into service from 2027. The EUR 436 million "
    "programme starts with Piraeus to Aegina, and each ship carries 388 passengers."
)


def answer(request, **fields):
    """A structured answer in either output mode: a final_answer call, or JSON text (native)."""
    if "final_answer" in tool_names(request):
        return final_answer(**fields)
    return json.dumps(fields)


def policy(request, facts=FACTS):
    """Plays the three crew members; each is recognised by its role in the instructions."""
    system = system_text(request)
    if "Senior research analyst" in system:
        searches = sum(message.role == "tool" for message in request.messages)
        if searches == 0:
            return call("search", query="Aegean Green Ferry programme")
        if searches == 1:
            return call("search", query="ferry route ships")
        return answer(request, facts=facts)
    if "Science and technology writer" in system:
        return answer(request, title="Greece goes electric at sea", body=BODY.replace("388", "380"))
    if "Managing editor" in system:
        return answer(
            request, title="Greece's ferries go electric", body=BODY, changes=["380 -> 388"]
        )
    raise AssertionError(f"unexpected request: {system[:80]!r}")


def test_search_ranks_the_archive_and_needs_more_than_one_query():
    hits = search.function("Aegean Green Ferry programme")
    assert [hit["source_id"] for hit in hits] == ["src-1", "src-2", "src-3"]  # three at most
    assert "src-4" in [hit["source_id"] for hit in search.function("ferry route")]
    assert [hit["source_id"] for hit in search.function("olive harvest")] == ["src-5"]
    assert isinstance(search.function("weather tomorrow"), str)


async def test_research_crew_passes_typed_context_from_task_to_task(tmp_path):
    ctx = scripted_context(policy, tmp_path)
    outcome = await run(ctx)
    assert outcome.ok, outcome.summary
    assert outcome.summary == "4 sourced facts; the article keeps 4 of the 4 researched figures"
    assert outcome.output["body"] == BODY

    prompts = {}
    for request in ctx.meter.requests:
        role = system_text(request).splitlines()[0]
        prompts.setdefault(role, request.messages[1].text)  # the task prompt of each agent
    researcher = prompts["Role: Senior research analyst"]
    writer = prompts["Role: Science and technology writer"]
    editor = prompts["Role: Managing editor"]
    assert "Expected output: Three to six facts" in researcher
    assert "research_notes:" not in researcher
    # Each task's context is exactly the typed output of the tasks before it, as JSON.
    notes = ResearchNotes.model_validate({"facts": FACTS}).model_dump_json(indent=2)
    assert f"research_notes:\n{notes}" in writer and "draft:" not in writer
    assert f"research_notes:\n{notes}" in editor and "draft:" in editor and "380" in editor
    assert len(ctx.meter.requests) == 5  # two searches and the notes, the draft, the edit


async def test_the_crew_runs_in_native_mode_as_on_anthropic(tmp_path):
    ctx = scripted_context(policy, tmp_path, output_mode="native")
    outcome = await run(ctx)
    assert outcome.ok, outcome.summary
    # The schema goes to the provider as the response format; no final_answer tool is offered.
    for request in ctx.meter.requests:
        assert request.output is not None and "final_answer" not in tool_names(request)
    assert {"facts"} <= set(ctx.meter.requests[0].output.json_schema["properties"])


async def test_a_rerun_with_the_same_workdir_replays_the_finished_tasks(tmp_path):
    assert (await run(scripted_context(policy, tmp_path))).ok

    def no_model(request):
        raise AssertionError("finished tasks must not ask the model again")

    again = scripted_context(no_model, tmp_path)
    outcome = await run(again)
    assert outcome.ok, outcome.summary
    assert again.meter.usage.requests == 0


async def test_facts_without_an_on_topic_source_do_not_count(tmp_path):
    unsourced = [{**fact, "source_id": "src-5"} for fact in FACTS[:2]]
    unsourced += [{**FACTS[2], "source_id": "news wire"}, FACTS[3]]
    outcome = await run(scripted_context(lambda r: policy(r, facts=unsourced), tmp_path))
    assert not outcome.ok and outcome.summary.startswith("1 sourced facts")


async def test_an_article_that_drops_a_researched_figure_fails(tmp_path):
    def sloppy(request):
        if "Managing editor" in system_text(request):
            return answer(request, title="Ferries", body=BODY.replace(" 436", ""), changes=[])
        return policy(request)

    outcome = await run(scripted_context(sloppy, tmp_path))
    assert not outcome.ok and "missing: 436" in outcome.summary


async def test_a_task_that_stops_ends_the_crew(tmp_path):
    def stubborn(request):  # the researcher never stops searching
        if "Senior research analyst" in system_text(request):
            return call("search", query="ferry")
        raise AssertionError("the writer and the editor must not run")

    ctx = scripted_context(stubborn, tmp_path)
    outcome = await run(ctx)
    assert not outcome.ok and "research task stopped: limit_reached (max_turns)" in outcome.summary
    assert len(ctx.meter.requests) == 6  # Limits(max_turns=6)


def test_runs_as_a_module_and_replays_from_its_cassette(tmp_path, capsys, monkeypatch):
    cassettes = tmp_path / "cassettes"
    common = ["--provider", "qcloud", "--cassette-dir", str(cassettes)]
    with pytest.raises(SystemExit) as recorded:
        main(
            run,
            name=research_crew.NAME,
            description=research_crew.DESCRIPTION,
            inspired_by=research_crew.INSPIRED_BY,
            argv=[*common, "--record", "--workdir", str(tmp_path / "recorded")],
            model_factory=lambda provider, model: FunctionModel(policy),
        )
    assert recorded.value.code == 0
    assert (cassettes / "qcloud" / "research_crew.json").is_file()
    capsys.readouterr()

    argv = ["research_crew", *common, "--replay", "--workdir", str(tmp_path / "replayed")]
    monkeypatch.setattr(sys, "argv", [*argv, "--json"])
    with pytest.raises(SystemExit) as replayed:
        runpy.run_path(research_crew.__file__, run_name="__main__")
    output = capsys.readouterr().out
    report = json.loads(output.strip().splitlines()[-1])
    assert replayed.value.code == 0, output
    assert report["ok"] and report["name"] == "research_crew" and report["requests"] == 5
    assert "# Greece's ferries go electric" in output
