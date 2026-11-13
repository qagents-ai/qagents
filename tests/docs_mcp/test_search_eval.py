"""Search quality over questions people type: `search_eval.toml`, and `search_heldout.toml`.

The benchmark (opt-in: ``pytest tests/docs_mcp -m docs_benchmark``) ranks every question of
both sets and reports the top-1 and top-3 hit rates and the rank of each miss. The ranking
is primarily tuned on the eval set. The second set is a regression/generalization check,
not a pristine holdout: one query informed the MCP-server topic words, and synonym
ablation inspected its scores. The default suite pins the questions in `PINNED`, which
each rank their expected page first.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import pytest

EVAL_SET = Path(__file__).with_name("search_eval.toml")
HELDOUT_SET = Path(__file__).with_name("search_heldout.toml")
DEPTH = 25
"""Results ranked per question; an answer below this counts as not found."""
TOP1_FLOOR = 0.75
TOP3_FLOOR = 0.9
"""The benchmark fails below these rates (measured: see `test_search_quality_over_the_set`)."""
HELDOUT_TOP1_FLOOR = 0.65
HELDOUT_TOP3_FLOOR = 0.8


@dataclass(frozen=True)
class Question:
    topic: str
    style: str
    query: str
    expected: tuple[str, ...]


def load_questions(path: Path = EVAL_SET) -> list[Question]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return [
        Question(q.get("topic", ""), q.get("style", ""), q["query"], tuple(q["expected"]))
        for q in data["question"]
    ]


def matches(cite: str, expected: str) -> bool:
    """Whether a result's cite is the expected page, section, example or source file."""
    return cite == expected or cite.startswith((f"{expected}#", f"{expected}:"))


def rank(service, question: Question) -> int | None:
    """The 1-based rank of the first expected answer, None when it is not in the results."""
    hits, _ = service.corpus.index.search(question.query, limit=DEPTH)
    for number, hit in enumerate(hits, start=1):
        if any(matches(hit.doc.cite, wanted) for wanted in question.expected):
            return number
    return None


def report(service, path: Path = EVAL_SET) -> tuple[float, float, list[str]]:
    """Top-1 and top-3 hit rates, and one line per question that is not first."""
    questions = load_questions(path)
    ranks = [rank(service, question) for question in questions]
    top1 = sum(1 for r in ranks if r == 1) / len(questions)
    top3 = sum(1 for r in ranks if r is not None and r <= 3) / len(questions)
    misses = [
        f"{'-' if r is None else r:>3}  {q.query!r} -> {', '.join(q.expected)}"
        for q, r in zip(questions, ranks, strict=True)
        if r != 1
    ]
    return top1, top3, misses


def test_the_set_covers_every_section_and_is_well_formed(service):
    questions = load_questions()
    assert len(questions) >= 40
    assert len({q.query for q in questions}) == len(questions)
    assert {q.style for q in questions} == {"bare", "natural", "synonym"}
    topics = {q.topic for q in questions}
    for topic in ["getting started", "tools", "sessions and resume", "permissions and approvals",
                  "pauses", "state", "guardrails", "structured output", "streaming",
                  "multi-agent", "workflows", "runtime and recovery", "stores", "tracing",
                  "AG-UI", "model providers", "MCP toolkits", "external agents", "Devvi",
                  "MCP tool server", "CLI", "testing", "evals"]:  # fmt: skip
        assert topic in topics, topic
    heldout = load_questions(HELDOUT_SET)
    assert not {q.query for q in heldout} & {q.query for q in questions}  # held out
    cites = [doc.cite for doc in service.corpus.index.docs]
    for question in questions + heldout:
        for wanted in question.expected:  # every expected answer exists in the corpus
            assert any(matches(cite, wanted) for cite in cites), (question.query, wanted)


@pytest.mark.docs_benchmark
def test_search_quality_over_the_set(service):
    """Measure the current corpus over 86 eval and 73 secondary questions.

    Keep observed rates in the changelog; docs and API additions can change ranking
    even when the search implementation has not changed.
    """
    top1, top3, misses = report(service)
    held1, held3, held_misses = report(service, HELDOUT_SET)
    summary = (
        f"eval: top-1 {top1:.2f}, top-3 {top3:.2f}\n" + "\n".join(misses)
        + f"\nheld out: top-1 {held1:.2f}, top-3 {held3:.2f}\n" + "\n".join(held_misses)
    )  # fmt: skip
    print(summary)
    assert top1 >= TOP1_FLOOR and top3 >= TOP3_FLOOR, summary
    assert held1 >= HELDOUT_TOP1_FLOOR and held3 >= HELDOUT_TOP3_FLOOR, summary


PINNED = [
    "agent tools",
    "how do I make an agent with tools",
    "tools",
    "define a tool",
    "permissions",
    "hooks",
    "tracing",
    "devvi",
    "mcp server",
    "AG-UI",
    "connect a copilotkit frontend",
    "sqlite store",
    "resume a run after the process crashed",
    "show tokens as they arrive",
    "use one agent as a tool of another",
    "is a failed model request retried",
    "test my agent without calling a real model",
    "delegate coding work to codex",
    "let claude code edit files for me",
    "an agent that writes code",
    "gemini cli",
    "evals",
    "agui",
    "postgres",
]
"""Questions that ranked a wrong page first before the ranking change (rank before it:
21, 20, 6, 3, 6, 4, 3, 2, 5, 3, 4, 5, 8, not in the top 25, not in the top 25, 5; then
3, 5, 4, 2), and three module names the first round of the change lost (``evals``,
``agui``, ``postgres``: first before, then not in the top 25, 3 and 9)."""


@pytest.mark.parametrize("query", PINNED)
def test_plain_questions_rank_their_page_first(service, query):
    question = next(q for q in load_questions() if q.query == query)
    hits, _ = service.corpus.index.search(query, limit=5)
    assert rank(service, question) == 1, [hit.doc.cite for hit in hits]


def test_driving_a_coding_agent_finds_the_external_agent_pages(service):
    """People name the agents they drive (Claude Code, Codex, Gemini CLI); the pages do not.
    The first round of the ranking change moved this question from 4 to 15."""
    question = next(
        q for q in load_questions() if q.query == "drive another coding agent such as claude code"
    )
    assert (rank(service, question) or 99) <= 3
