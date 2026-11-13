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


# --- ranking: words, headings, priors ------------------------------------------------


@pytest.mark.parametrize(
    "words", [["retried", "retries", "retry"], ["denied", "deny"], ["died", "dies", "die"]]
)
def test_stemmer_folds_ied_endings(words):
    assert len({stem(w) for w in words}) == 1, [stem(w) for w in words]


def test_an_acronym_also_gives_its_word():
    tokens = tokenize("SQLiteStore")
    assert stem("sqlite") in tokens and "store" not in tokens and stem("store") in tokens
    assert tokenize("MCPToolkit").count("mcptoolkit") == 1  # the joined part is the word


def test_the_subject_word_counts_little_next_to_others():
    from qagents.docs_mcp.search import SUBJECT_WEIGHT

    assert parse_query("agent tools").terms == {"agent": SUBJECT_WEIGHT, "tool": 1.0}
    assert parse_query("agent").terms["agent"] == 1.0  # alone, it is the question
    assert {"agent", "tool"} <= parse_query("agent tools").typed


def test_one_hyphenated_word_is_one_name():
    query = parse_query("AG-UI")
    assert query.full_names == {"agui"} and query.plain_word
    assert not parse_query("Agent").plain_word and not parse_query("max_turns").plain_word


def _toy_index():
    """A core page titled Tools, an integration section that uses the core words often, and
    filler so the IDF of a word says how rare it is."""
    from qagents.docs_mcp.search import BM25Index, SearchDoc

    def doc(cite, text, heading, **options):
        return SearchDoc("docs", heading, text, cite, "", cite.split("#")[0], heading=heading,
                         **options)  # fmt: skip

    docs = [
        doc("tools.md", "A tool is a Python function the model can call.", "Tools"),
        doc(
            "integrations/x.md#planes",
            "An agent drives the agent; the agent has tools. Agent tools, agent tools.",
            "Two planes: who drives the agent, and what tools it has",
            titles=("External agents",), topic="External agents xylo", off_topic=0.5,
        ),
        *(doc(f"filler{i}.md", f"The agent writes notes {i}.", f"Notes {i}") for i in range(8)),
    ]  # fmt: skip
    return BM25Index(docs)


def test_a_core_page_titled_by_the_query_beats_an_integration_that_uses_the_words():
    hits, _ = _toy_index().search("agent tools", limit=3)
    assert hits[0].doc.cite == "tools.md", [h.doc.cite for h in hits]


def test_naming_the_integration_lifts_the_off_topic_prior():
    index = _toy_index()

    def planes(query):
        hits, _ = index.search(query, limit=10)
        return next(h.score for h in hits if h.doc.cite == "integrations/x.md#planes")

    discounted, named = planes("agent tools"), planes("xylo agent tools")
    index.docs[1].off_topic = 1.0
    assert planes("agent tools") == pytest.approx(discounted * 2)  # a core question: halved
    assert planes("xylo agent tools") == pytest.approx(named)  # it names the topic: in full


def test_a_plain_word_prefers_the_page_titled_by_it_over_the_api_member(service):
    """``tools`` is also `Toolkit.tools`; one plain word gets the last-name boost only."""
    hits, _ = service.corpus.index.search("tools", limit=3)
    assert hits[0].doc.cite == "concepts/tools.md", [h.doc.cite for h in hits]
    hits, _ = service.corpus.index.search("Toolkit.tools", limit=1)
    assert hits[0].doc.kind == "api"


def test_code_and_coding_are_one_word():
    assert stem("code") == stem("coding") == stem("coded") == stem("codes")


def test_idioms_expand_and_ambiguous_words_do_not():
    assert "text_delta" in parse_query("stream the answer token by token").terms
    assert "scriptedmodel" in parse_query("run it with no API key").terms
    assert not {"handoff", "human"} & set(parse_query("a person writes a tool by hand").terms)


def test_topic_words_name_a_page_that_does_not_use_them():
    """A page on one integration is about what its `topic_words` name: a query that says
    one weighs it in full, and the page covers that word."""
    from qagents.docs_mcp.search import BM25Index, SearchDoc

    def index(topic_words):
        docs = [
            SearchDoc("docs", "Drive an agent", "An agent drives another agent in a workspace.",
                      "x.md", "", "x.md", heading="Drive an agent", topic="Drive",
                      topic_words=topic_words, off_topic=0.5),
            SearchDoc("docs", "Setup", "Add the docs server to zork.", "setup.md", "",
                      "setup.md", heading="Setup"),
            *(SearchDoc("docs", f"Notes {i}", f"Notes {i} about a workspace.", f"n{i}.md", "",
                        f"n{i}.md") for i in range(6)),
        ]  # fmt: skip
        return BM25Index(docs)

    def drive(topic_words):
        hits, _ = index(topic_words).search("zork in a workspace", limit=10)
        return next(h.score for h in hits if h.doc.cite == "x.md")

    assert drive("zork") > 2 * drive("")  # weighed in full (x2) and covers "zork"


def test_a_plain_word_that_names_no_guide_section_names_the_object(service):
    """``tools`` names the page ``Tools`` and `Toolkit.tools`: the page wins. ``evals``
    names no guide section: the module is the answer, not its members."""
    for word, cite in [
        ("evals", "src/qagents/evals.py:1"),
        ("postgres", "src/qagents/stores/postgres.py:1"),
        ("tools", "concepts/tools.md"),
    ]:
        hits, _ = service.corpus.index.search(word, limit=3)
        assert hits[0].doc.cite == cite, (word, [h.doc.cite for h in hits])


def test_a_reference_stub_is_not_matched_by_its_title(service):
    """``reference/agent.md`` holds only ``::: name`` lines: no snippet, and the objects it
    lists are indexed themselves, so its title ``Agent`` does not make it the answer."""
    stub = next(d for d in service.corpus.index.docs if d.cite == "reference/agent.md")
    assert stub.heading == "" and stub.titles == ()
    prompts = next(d for d in service.corpus.index.docs if d.cite == "reference/prompts.md")
    assert prompts.heading == "Prompts"  # a reference page with prose of its own keeps it
    for word in ["agent", "messages"]:
        hits, _ = service.corpus.index.search(word, limit=3)
        assert not hits[0].doc.cite.startswith("reference/"), [h.doc.cite for h in hits]
