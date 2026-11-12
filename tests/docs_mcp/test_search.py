"""The index and the search: known questions find the page that answers them."""

from __future__ import annotations

import pytest

from qagents.docs_mcp.search import parse_query, stem, tokenize

BUILD_SECONDS = 2.0
"""The opt-in bound on building the whole index; about 0.35 s on a developer machine."""


def test_tokens_keep_identifiers_whole_and_split():
    tokens = tokenize("requires_approval ScriptedModel sub-agent approvals")
    for token in ["requires_approval", "requir", "approval", "scriptedmodel", "script"]:
        assert token in tokens
    assert "subagent" in tokens and "agent" in tokens


@pytest.mark.parametrize(
    ("words", "same"),
    [(["approvals", "approval"], True), (["paused", "pause", "pauses"], True),
     (["streaming", "streams", "stream"], True), (["tools", "tool"], True)],
)  # fmt: skip
def test_stemmer_folds_endings(words, same):
    assert (len({stem(w) for w in words}) == 1) is same


def test_phrases_and_names_in_queries():
    query = parse_query("human in the loop for Agent.resume")
    assert "approval" in query.typed and "loop" not in query.typed
    assert "agent.resume" in query.full_names and query.names_code
    assert not parse_query("pause a tool for approval").names_code


def test_the_index_has_pages_examples_and_the_api(service):
    corpus = service.corpus
    assert len(corpus.pages) >= 20 and len(corpus.examples) >= 20 and len(corpus.api) > 300


@pytest.mark.docs_benchmark
def test_the_index_is_built_quickly(service):
    """A wall-clock bound: opt-in (``-m docs_benchmark``), so slow CI runners do not fail."""
    corpus = service.corpus
    assert corpus.load_seconds < BUILD_SECONDS, corpus.load_seconds


def _cites(service, query, limit=5, kind=None):
    hits, _ = service.corpus.index.search(query, limit=limit, kinds=[kind] if kind else None)
    return [hit.doc.cite for hit in hits]


@pytest.mark.parametrize(
    ("query", "expected", "within"),
    [
        ("pause a tool call until a human approves it", "concepts/permissions.md#pause-and-resume", 3),
        ("tool that needs approval, resume after the decision", "concepts/permissions.md#pause-and-resume", 3),
        ("test an agent without a model", "testing.md#scripted-replies", 3),
        ("supervisor with two sub-agents", "concepts/multi-agent.md#sub-agents", 3),
        ("workflow with two steps that resumes after a crash", "concepts/multi-agent.md#workflows", 5),
        ("structured output with a pydantic model", "concepts/structured-output.md", 3),
        ("stream events", "concepts/streaming.md", 3),
        ("system prompt", "concepts/loop.md#instructions", 3),
        ("how do I give a tool a timeout", "concepts/tools.md#flags", 3),
        ("idempotency key for side effects", "concepts/sessions.md#idempotency", 5),
    ],
)  # fmt: skip
def test_known_questions_find_their_page(service, query, expected, within):
    assert expected in _cites(service, query, limit=within), _cites(service, query, limit=8)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("ScriptedModel", "src/qagents/models/testing.py"),
        ("Agent.resume", "src/qagents/agent.py"),
        ("max_turns", "src/qagents/agent.py"),
    ],
)
def test_names_find_their_api_entry_first(service, query, expected):
    first = _cites(service, query, limit=1)[0]
    assert first.startswith(expected + ":")


def test_examples_are_searchable(service):
    assert "examples/patterns/supervisor.py:1" in _cites(service, "supervisor delegates", 3)
    assert all(c.startswith("examples/") for c in _cites(service, "approval", 5, kind="example"))


def test_results_cite_sources_and_say_how_to_read_them(service):
    text = service.search("pause and resume", limit=3)
    assert text.startswith("# Search: 'pause and resume'")
    assert "`concepts/permissions.md#pause-and-resume`" in text
    assert 'Read: read_page("concepts/permissions.md", section="pause-and-resume")' in text
    assert "qagents " + service.version in text


def test_no_results_and_bad_kind(service):
    assert service.search("zzzqqqxxx").startswith("No results for 'zzzqqqxxx'")
    assert "Unknown kind" in service.search("agent", kind="videos")
    assert "[api]" in service.search("Agent.run", kind="api")
