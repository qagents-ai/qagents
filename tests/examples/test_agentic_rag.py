"""Offline tests of examples/patterns/agentic_rag.py."""

from __future__ import annotations

import json
import runpy
import sys

import pytest

from examples.harness import final_answer, main, scripted_context, tool_names
from examples.patterns import agentic_rag
from examples.patterns.agentic_rag import (
    INDEX,
    CitedAnswer,
    cite_only_retrieved,
    read_document,
    run,
    search_docs,
)
from qagents import (
    Message,
    ModelRetry,
    OutputValidationError,
    RunContext,
    ToolError,
    ToolResultPart,
)
from qagents.models.testing import FunctionModel, call

ANSWER = (
    "The motion sensor's CR2477 battery lasts about 18 months at default sensitivity. To pair "
    "a new one, tap Add device in the app and hold the sensor's button for 7 seconds."
)


def answer(request, **fields):
    """A structured answer: a final_answer call in tool mode, JSON text in native mode."""
    if "final_answer" in tool_names(request):
        return final_answer(**fields)
    return json.dumps(fields)


def policy(request):
    """Plays the model: search, read the top hit, search again, read, then answer.

    The first answer also cites KB-111, which no tool returned, so the validator rejects it.
    """
    results = [r for m in request.messages for r in m.tool_results]
    searches = [json.loads(r.content) for r in results if r.name == "search_docs"]
    reads = [r.content.split("]")[0].lstrip("[") for r in results if r.name == "read_document"]
    if not searches:
        return call("search_docs", query="motion sensor battery life")
    if not reads:
        return call("read_document", doc_id=searches[0][0]["id"])
    if len(searches) == 1:
        return call("search_docs", query="pair new motion sensor")
    if len(reads) == 1:
        return call("read_document", doc_id=searches[1][0]["id"])
    answered = [
        m
        for m in request.messages
        if m.role == "assistant"
        and (any(c.name == "final_answer" for c in m.tool_calls) or (m.text and not m.tool_calls))
    ]
    if not answered:
        return answer(request, answer=ANSWER, sources=[*reads, "KB-111"])
    return answer(request, answer=ANSWER, sources=[source.lower() for source in reads])


@pytest.mark.parametrize("output_mode", ["tool", "native"])
async def test_cites_the_two_documents_after_a_rejected_citation(tmp_path, output_mode):
    ctx = scripted_context(policy, tmp_path, output_mode=output_mode)
    outcome = await run(ctx)
    assert outcome.ok, outcome.summary
    # The validator normalised the ids of the second answer (lower case in, upper case out).
    assert outcome.output == {"answer": ANSWER, "sources": ["KB-104", "KB-105"]}
    assert outcome.details["retrieved"] == ["KB-104", "KB-105", "KB-106"]
    assert outcome.details["searches"] == ["motion sensor battery life", "pair new motion sensor"]
    # The first answer was sent back with the validator's message.
    sent = [m.text for r in ctx.meter.requests for m in r.messages] + [
        res.content for r in ctx.meter.requests for m in r.messages for res in m.tool_results
    ]
    assert any("You cited KB-111" in text for text in sent)


async def test_a_wrong_citation_is_not_ok(tmp_path):
    def cites_a_neighbour(request):  # cites the door sensor, which it did retrieve
        reply = policy(request)
        if isinstance(reply, str) or reply.name == "final_answer":
            return answer(request, answer=ANSWER, sources=["KB-104", "KB-106"])
        return reply

    outcome = await run(scripted_context(cites_a_neighbour, tmp_path))
    assert not outcome.ok and "KB-106" in outcome.summary


def test_bm25_search_ranks_the_right_documents():
    assert INDEX.search("motion sensor battery life")[0]["id"] == "KB-104"
    assert INDEX.search("how do I pair a new motion sensor")[0]["id"] == "KB-105"
    assert INDEX.search("factory reset hub")[0]["id"] == "KB-107"
    assert INDEX.search("hub light amber")[0]["id"] == "KB-103"
    battery = INDEX.search("motion sensor battery")[0]
    assert battery["title"] == "Motion sensor battery" and "18 months" in battery["snippet"]
    assert search_docs.function("xyzzy") == "No document matches 'xyzzy'. Try other keywords."
    assert read_document.function("kb-104").startswith("[KB-104] # Motion sensor battery")
    with pytest.raises(ToolError, match="no document"):
        read_document.function("KB-999")


def test_the_validator_accepts_only_retrieved_ids():
    shown = Message(
        role="tool",
        parts=[ToolResultPart(tool_call_id="c1", name="search_docs", content='[{"id": "KB-104"}]')],
    )
    ctx = RunContext(messages=(shown,))
    sources = ["kb-104", "KB-104: Motion sensor battery", " "]
    accepted = cite_only_retrieved(ctx, CitedAnswer(answer="...", sources=sources))
    assert accepted.sources == ["KB-104"]
    with pytest.raises(ModelRetry, match=r"You cited KB-105, .*retrieved \(KB-104\)"):
        cite_only_retrieved(ctx, CitedAnswer(answer="...", sources=["KB-104", "KB-105"]))
    with pytest.raises(ModelRetry, match="You cited Motion sensor battery"):
        cite_only_retrieved(ctx, CitedAnswer(answer="...", sources=["Motion sensor battery"]))
    nothing = cite_only_retrieved(RunContext(), CitedAnswer(answer="Not covered.", sources=[]))
    assert nothing.sources == []


async def test_a_model_that_keeps_citing_unretrieved_documents_runs_out_of_retries(tmp_path):
    def stubborn(request):  # never searches, always cites a document it has not seen
        return answer(request, answer=ANSWER, sources=["KB-104"])

    with pytest.raises(OutputValidationError, match="rejected by an output validator"):
        await run(scripted_context(stubborn, tmp_path))


def test_runs_as_a_module_and_replays_from_its_cassette(tmp_path, monkeypatch, capsys):
    common = ["--provider", "qcloud", "--cassette-dir", str(tmp_path / "cassettes"), "--json"]
    with pytest.raises(SystemExit) as recorded:
        main(
            run,
            name=agentic_rag.NAME,
            description=agentic_rag.DESCRIPTION,
            inspired_by=agentic_rag.INSPIRED_BY,
            argv=[*common, "--record", "--workdir", str(tmp_path / "recorded")],
            model_factory=lambda provider, model: FunctionModel(policy),
        )
    assert recorded.value.code == 0
    capsys.readouterr()

    argv = ["agentic_rag", *common, "--replay", "--workdir", str(tmp_path / "replayed")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as replayed:
        runpy.run_path(agentic_rag.__file__, run_name="__main__")
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert replayed.value.code == 0, report
    assert report["ok"] and report["requests"] == 6
