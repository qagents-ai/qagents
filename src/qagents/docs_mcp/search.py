"""BM25 search over the docs corpus, in plain Python.

Tokens are lowercase words. Identifiers are kept whole and also split into their parts, so
``requires_approval`` matches ``requires_approval``, ``approval`` and ``requires``, and
``ScriptedModel`` matches ``scriptedmodel``, ``scripted`` and ``model``. Hyphenated words
also give their joined form (``sub-agent`` gives ``subagent``), and an acronym gives its
joined form with the next part (``SQLiteStore`` gives ``sqlite``). A light suffix stemmer
folds plurals and verb endings (``approvals``, ``approved`` and ``approve`` all match;
``retries`` and ``retried`` too; ``code`` and ``coding``).

A few query words used by other frameworks, or by people new to agents, are expanded to the
words the qagents docs use (`SYNONYMS`, `PHRASES`); expansions count half as much as the
words typed. The word the whole corpus is about (``agent``, `SUBJECT_WORDS`) counts little
next to other words, so ``agent tools`` is a question about tools. `BM25Index` says how the
rest of the ranking works: titles and headings, priors, and names.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache

_WORD = re.compile(r"[^\W\d]\w*|\d+")
_HYPHENATED = re.compile(r"[^\W\d_][^\W_]*(?:-[^\W\d_][^\W_]*)+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

STOPWORDS = frozenset(
    """
    a about after again also an and any are as at be been before both but by can could do
    does each every few for from give has have here how i if in into is it its just let like
    make many me more most much my need not now of on once one only or other our out over
    same should so some such than that the their them then there these this those three to
    two until up us use using very was want we what when where which while who why will
    with without would you your
    """.split()
)
"""Words too common to help ranking. Domain words such as ``run`` and ``tool`` stay."""

SYNONYMS: dict[str, tuple[str, ...]] = {
    "hitl": ("approval", "human"),
    "human": ("approval",),
    "interrupt": ("pause", "approval", "deferred"),
    "confirm": ("approval",),
    "approve": ("approval",),
    "pause": ("resume",),
    "subagent": ("sub", "agent", "as_tool", "child"),
    "delegate": ("as_tool", "subagent"),
    "supervisor": ("as_tool", "orchestrator"),
    "orchestrator": ("supervisor",),
    "crew": ("workflow", "subagent"),
    "graph": ("workflow",),
    "node": ("step", "workflow"),
    "pipeline": ("workflow", "step"),
    "checkpoint": ("session", "journal"),
    "checkpointer": ("session", "store"),
    "memory": ("session", "notes"),
    "persist": ("session", "store", "sqlitestore"),
    "thread": ("session",),
    "mock": ("scriptedmodel", "functionmodel", "testing"),
    "fake": ("scriptedmodel", "functionmodel"),
    "test": ("scriptedmodel", "testing"),
    "unittest": ("scriptedmodel", "testing"),
    "system": ("instructions",),
    "prompt": ("instructions",),
    "iterations": ("max_turns", "limits"),
    "recursion": ("max_turns", "limits"),
    "budget": ("max_cost_usd", "limits"),
    "cost": ("max_cost_usd", "usage"),
    "schema": ("output_type", "structured"),
    "pydantic": ("output_type",),
    "json": ("output_type", "structured"),
    "callback": ("hooks",),
    "middleware": ("hooks",),
    "guardrail": ("output_guardrails", "permissions"),
    "trace": ("tracing", "otelhooks", "mlflowhooks"),
    "telemetry": ("tracing",),
    "stream": ("runstream", "events"),
    "event": ("stream",),
    "handoff": ("handoffs", "transfer"),
    "swarm": ("handoffs",),
    "router": ("route",),
    "parallel": ("fan_out", "gather", "read_only"),
    "mcp": ("mcptoolkit",),
    "provider": ("model",),
    "llm": ("model",),
    "openai": ("openaichatmodel",),
    "claude": ("anthropicmodel",),
    "anthropic": ("anthropicmodel",),
    "retry": ("modelretry", "retry_ambiguous"),
    "idempotency": ("operation_id", "idempotent"),
    "crash": ("resume", "recover", "journal"),
    "replay": ("cassette",),
    "vcr": ("cassette",),
    "die": ("crash", "resume", "recover"),
    "define": ("decorate",),
    "decorator": ("decorate",),
    "exception": ("error",),
    "history": ("conversation", "session"),
    "wait": ("pause",),
    "block": ("output_guardrails", "guardrail"),  # a guardrail ends a run "blocked"
    "frontend": ("front", "browser", "agui"),
    "terminal": ("cli", "command"),
    "cli": ("command", "line"),
}
"""Query expansions, keyed and valued by stemmed tokens. Each expansion counts half."""

_EXPANSION_WEIGHT = 0.5

SUBJECT_WORDS = frozenset({"agent", "qagent"})
"""Stemmed words that name what the whole corpus is about. In ``agent tools`` or ``test my
agent`` the word ``agent`` says what the question is about, not which page answers it,
so next to other words it counts `SUBJECT_WEIGHT` (alone, it counts in full)."""
SUBJECT_WEIGHT = 0.25


def stem(word: str) -> str:
    """A small suffix stemmer: plurals, ``-ing`` and ``-ed`` are folded together."""
    if len(word) <= 3 or not word.isalpha():
        return word
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("ied"):  # retried, denied; died
        return word[:-3] + "y" if len(word) > 4 else word[:-1]
    elif word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    if word.endswith("ing") and len(word) > 5:
        word = word[:-3]
        if len(word) > 2 and word[-1] == word[-2] and word[-1] not in "lsz":
            word = word[:-1]
    elif word.endswith("ed") and len(word) > 4:
        word = word[:-2]
        if len(word) > 2 and word[-1] == word[-2] and word[-1] not in "lsz":
            word = word[:-1]
    if word.endswith("e") and len(word) > 3:  # code, coding and coded: "cod"
        word = word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """The search tokens of a text, in order, with repeats."""
    tokens: list[str] = []
    if "-" in text:
        for match in _HYPHENATED.finditer(text):
            tokens.append(stem(match.group().replace("-", "").lower()))
    for word in _WORD.findall(text):
        tokens.extend(_word_tokens(word))
    return tokens


@lru_cache(maxsize=65536)
def _word_tokens(word: str) -> tuple[str, ...]:
    """The tokens of one word: the whole word stemmed, then its identifier parts."""
    lower = word.lower()
    if lower in STOPWORDS:
        return ()
    tokens: list[str] = []
    if len(lower) > 1 or lower.isdigit():
        tokens.append(stem(lower))
    parts = [part for piece in word.split("_") for part in _CAMEL.findall(piece)]
    if len(parts) > 1:
        for part in parts:
            part = part.lower()
            if len(part) > 1 and part not in STOPWORDS:
                tokens.append(stem(part))
        for first, second in zip(parts, parts[1:], strict=False):
            acronym = len(first) > 1 and first.isupper() and second.istitle()
            if acronym and (first + second).lower() != lower:  # SQLiteStore: also "sqlite"
                tokens.append(stem((first + second).lower()))
    return tuple(tokens)


PHRASES: dict[str, str] = {
    "human in the loop": "approval approve pause resume",
    "human-in-the-loop": "approval approve pause resume",
    "system prompt": "instructions",
    "system message": "instructions",
    "sub agent": "subagent as_tool",
    "multi agent": "multi-agent subagent as_tool handoffs workflow",
    "multi-agent": "multi-agent subagent as_tool handoffs workflow",
    "structured output": "output_type structured",
    "unit test": "testing scriptedmodel",
    "without a model": "scriptedmodel functionmodel testing",
    "no model": "scriptedmodel functionmodel testing",
    "no network": "scriptedmodel functionmodel testing offline",
    "tool approval": "requires_approval approval",
    "max iterations": "max_turns limits",
    "final answer": "output output_type",
    "hello world": "quickstart first agent",
    "as a tool": "as_tool",
    "as they arrive": "stream text_delta",
    "as it arrives": "stream text_delta",
    "as it is generated": "stream text_delta",
    "token by token": "stream text_delta",
    "word by word": "stream text_delta",
    "without an api key": "key scriptedmodel offline",
    "without a key": "key scriptedmodel offline",
    "no api key": "key scriptedmodel offline",
}
"""Phrases that mean one qagents idea. A phrase in a query is replaced by these words.

Each entry is an idiom, not a question: ``as a tool`` is how people say `Agent.as_tool`,
and ``token by token`` how they ask for streaming. Words that are ambiguous on their own
(``hand`` in ``by hand``) are not expanded."""

_STEMMED_SYNONYMS = {stem(key): value for key, value in SYNONYMS.items()}
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+|\w*_\w*|[A-Z][a-z0-9]+[A-Z]\w*")


@dataclass(frozen=True)
class Query:
    """A parsed query: weighted tokens and the names it mentions."""

    terms: dict[str, float]
    typed: frozenset[str]
    """Tokens the user typed (or a phrase stood for), as opposed to expansions. Their weight
    in `terms` is 1, or `SUBJECT_WEIGHT` for a subject word next to other words."""
    full_names: frozenset[str]
    """Identifier-like words as typed, lowercased, without a ``qagents.`` prefix."""
    last_names: frozenset[str]
    """The last dotted part of each identifier-like word."""
    names_code: bool
    """Whether the query names code (``Agent.run``, ``max_turns``, ``ScriptedModel``, or a
    single word) rather than asking a question in prose."""
    plain_word: bool
    """Whether the query is one plain word (``tools``, ``hooks``, ``ACP``): it may name a
    module or a function, but it is as likely to name the idea a guide page explains."""
    phrase: str
    expansions: dict[str, frozenset[str]] = field(default_factory=dict)
    """For each typed token, the tokens it was expanded to."""


def _phrase_pattern(key: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![\w-]){re.escape(key)}(?![\w-])", re.IGNORECASE)


_PHRASE_PATTERNS = {key: _phrase_pattern(key) for key in PHRASES}


def parse_query(text: str) -> Query:
    """Tokens, expansions and names of a search query."""
    phrase = " ".join(text.lower().split())
    kept = " ".join(text.split())
    meanings: list[str] = []
    for key, pattern in _PHRASE_PATTERNS.items():
        if pattern.search(kept):
            kept = pattern.sub(" ", kept)
            meanings.append(PHRASES[key])
    weights: dict[str, float] = {}
    for token in tokenize(kept) + tokenize(" ".join(meanings)):
        weights[token] = 1.0
    expansions: dict[str, frozenset[str]] = {}
    for token in list(weights):
        extra_tokens = {
            expanded
            for extra in _STEMMED_SYNONYMS.get(token, ())
            for expanded in tokenize(extra)
            if expanded != token
        }
        expansions[token] = frozenset(extra_tokens)
        for expanded in extra_tokens:
            weights.setdefault(expanded, _EXPANSION_WEIGHT)
    full: set[str] = set()
    last: set[str] = set()
    words = [word.strip(".") for word in re.findall(r"[A-Za-z_][\w.]*", text)]
    words = [word for word in words if word]
    if len(words) > 1 and _HYPHENATED.fullmatch(text.strip()):
        words = [text.strip().replace("-", "")]  # one hyphenated name: AG-UI, sub-agent
    # A single word is a name; in a longer query only identifier-like words are.
    named = words if len(words) == 1 else [w for w in words if _IDENTIFIER.fullmatch(w)]
    for word in named:
        word = word.lower().removeprefix("qagents.")
        full.add(word)
        last.add(word.rsplit(".", 1)[-1])
    names_code = bool(named)
    plain_word = (
        len(words) == 1 and words[0].isalpha() and (words[0].islower() or words[0].isupper())
    )
    typed = frozenset(token for token, weight in weights.items() if weight == 1.0)
    if typed - SUBJECT_WORDS:
        for token in typed & SUBJECT_WORDS:
            weights[token] = SUBJECT_WEIGHT
    return Query(
        weights, typed, frozenset(full), frozenset(last), names_code, plain_word, phrase,
        expansions,
    )  # fmt: skip


def query_terms(query: str) -> dict[str, float]:
    """The weighted tokens of a query: typed words weigh 1 (a subject word next to others
    `SUBJECT_WEIGHT`), expansions 0.5."""
    return parse_query(query).terms


@dataclass
class SearchDoc:
    """One searchable piece: a page section, an example or an API object."""

    kind: str
    """``"docs"``, ``"example"`` or ``"api"``."""
    title: str
    text: str
    cite: str
    """Where it comes from: ``page.md#anchor`` or ``file:line``."""
    read: str
    """How to read the whole thing, for example ``read_page("concepts/tools.md", "flags")``."""
    group: str
    """Results from one group (a page, an example) are capped, so one page cannot fill the
    list."""
    boost: float = 1.0
    names: tuple[str, ...] = ()
    """Exact names that make this the best answer, for example ``("scriptedmodel",)``."""
    extra: str = ""
    """Text that is searched but not shown in snippets."""
    heading: str = ""
    """The section's own heading. A query that says what it says ranks the section higher
    (`BM25Index.HEADING_WEIGHT`)."""
    titles: tuple[str, ...] = ()
    """Titles above it: the page's title and nav title for a section, the description for
    an example. They count half as much as a heading of its own."""
    topic: str = ""
    """Words that say what the page is about (``Devvi as a coding worker``), for a page on
    one integration. With `off_topic`, a query that says none of them weighs it less. Words
    common to the whole corpus are left out (`BM25Index.UBIQUITOUS`)."""
    topic_words: str = ""
    """More words that name the page's topic, though the page may not use them (``codex``),
    all kept."""
    off_topic: float = 1.0
    """The weight of this document for a query that does not mention its `topic`."""


@dataclass
class Hit:
    doc: SearchDoc
    score: float
    matched: set[str] = field(default_factory=set)


class BM25Index:
    """An Okapi BM25 index with a title boost, a heading match, an exact-name boost and
    per-kind priors.

    A document's score is its BM25 score times:

    - its prior (`SearchDoc.boost`), and `SearchDoc.off_topic` for a page on one
      integration when the query does not name that integration;
    - how much of the query it covers, each typed word weighted by its IDF, so a document
      with every rare word wins but missing a word the whole corpus uses (``agent``) costs
      little. A page on one integration covers the words that name its topic
      (`SearchDoc.topic_words`) though it may not use them;
    - how well the query and one of its headings say the same thing (`HEADING_WEIGHT`);
    - the exact-name boost, when the query names an object or an example.
    """

    TITLE_WEIGHT = 3
    """How many times the title's tokens count."""
    MIN_LENGTH = 0.5
    """Documents shorter than this fraction of the average length are scored as if they
    had it, so a one-line docstring does not outrank the page that explains the idea."""
    PROSE_API_FACTOR = 0.6
    """API entries count this much for a question written in prose, unless they are named."""
    NAME_FACTOR = 4.0
    """An object whose full name the query is (``Agent.resume``, ``ScriptedModel``)."""
    LAST_NAME_FACTOR = 1.5
    """An object whose last name the query mentions, or whose name is the one plain word the
    query is when a guide section is named by that word too (``tools`` names
    `Toolkit.tools`, but it is also the idea the page ``Tools`` explains). A plain word that
    names no section (``evals``, ``postgres``) gets `NAME_FACTOR`."""
    UBIQUITOUS = 0.5
    """A word in more than this share of the guide sections (``agent``, ``tool``) says
    nothing about a page's topic: a query with only such words is about the core."""
    HEADING_WEIGHT = 2.0
    """A section whose heading is exactly the query scores ``1 + HEADING_WEIGHT`` times as
    much; a partial match scores in proportion to the share of the query's words the
    heading has times the share of the heading's words the query has (`_heading_match`).
    The titles above it (`SearchDoc.titles`) count half. A page titled ``Tools`` answers
    ``tools`` better than a long section that uses the word often."""

    def __init__(self, docs: Sequence[SearchDoc], *, k1: float = 1.2, b: float = 0.75) -> None:
        self.docs = list(docs)
        self.k1 = k1
        self.b = b
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._lengths: list[int] = []
        self._headings = [
            tuple(_Heading.of(text) for text in (doc.heading, *doc.titles)) for doc in self.docs
        ]
        for index, doc in enumerate(self.docs):
            counts = Counter(tokenize(doc.text))
            counts.update(tokenize(doc.extra))
            for token in tokenize(doc.title):
                counts[token] += self.TITLE_WEIGHT
            self._lengths.append(sum(counts.values()))
            for token, count in counts.items():
                self._postings[token].append((index, count))
        total = sum(self._lengths)
        self._average = total / len(self._lengths) if self._lengths else 0.0
        sections = sum(1 for doc in self.docs if doc.kind == "docs")
        common = sections * self.UBIQUITOUS

        def in_sections(token: str) -> int:
            return sum(1 for i, _ in self._postings.get(token, ()) if self.docs[i].kind == "docs")

        self._topics = [
            frozenset(t for t in tokenize(doc.topic) if in_sections(t) <= common)
            | frozenset(tokenize(doc.topic_words))
            for doc in self.docs
        ]
        # the tokens of the one-word headings and titles of the guide pages (``Tools``,
        # ``Hooks``): the ideas a guide section is named by
        self._guide_words = frozenset().union(
            *(h.tokens for doc, hs in zip(self.docs, self._headings, strict=True)
              if doc.kind == "docs" for h in hs if len(h.words) == 1)
        )  # fmt: skip

    def __len__(self) -> int:
        return len(self.docs)

    def vocabulary(self) -> Iterable[str]:
        return self._postings.keys()

    def search(
        self, query: str, *, limit: int = 8, kinds: Iterable[str] | None = None, per_group: int = 2
    ) -> tuple[list[Hit], int]:
        """The best matches, and how many documents matched at all.

        Args:
            query: Words, names or a question.
            limit: How many results to return.
            kinds: Only these kinds (``"docs"``, ``"example"``, ``"api"``).
            per_group: At most this many results from one page or example, unless there are
                not enough others.
        """
        parsed = parse_query(query)
        terms = parsed.terms
        if not terms or not self.docs:
            return [], 0
        allowed = set(kinds) if kinds is not None else None
        count = len(self.docs)
        floor = self._average * self.MIN_LENGTH
        scores: dict[int, float] = defaultdict(float)
        matched: dict[int, set[str]] = defaultdict(set)
        idfs: dict[str, float] = {}
        for term, weight in terms.items():
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = math.log(1 + (count - len(postings) + 0.5) / (len(postings) + 0.5))
            idfs[term] = idf
            for index, tf in postings:
                length = max(self._lengths[index], floor)
                norm = (
                    tf
                    * (self.k1 + 1)
                    / (tf + self.k1 * (1 - self.b + self.b * length / (self._average or 1)))
                )
                scores[index] += weight * idf * norm
                matched[index].add(term)

        typed = parsed.typed
        typed_weight = sum(idfs.get(term, 0.0) * terms[term] for term in typed)
        # one plain word that names a guide section too (``tools``) may mean the idea the
        # section explains, not the object; one that names no section (``evals``) is the
        # object's name
        name_factor = (
            self.LAST_NAME_FACTOR
            if parsed.plain_word and typed <= self._guide_words
            else self.NAME_FACTOR
        )
        hits: list[Hit] = []
        for index, score in scores.items():
            doc = self.docs[index]
            if allowed is not None and doc.kind not in allowed:
                continue
            score *= doc.boost
            if doc.off_topic != 1.0 and not self._topics[index] & typed:
                score *= doc.off_topic
            if typed_weight:
                found = matched[index] | self._topics[index]  # a page covers its topic
                covered = sum(
                    idfs.get(term, 0.0) * terms[term]
                    * (1.0 if term in found else 0.5 if parsed.expansions.get(term, set()) & found
                       else 0.0)
                    for term in typed
                )  # fmt: skip
                score *= 0.5 + covered / typed_weight  # documents with every rare word first
            if typed:
                score *= 1 + self.HEADING_WEIGHT * self._heading_match(index, parsed)
            names = set(doc.names)
            if names & parsed.full_names:
                score *= name_factor
            elif names & parsed.last_names:
                score *= self.LAST_NAME_FACTOR
            elif doc.kind == "api" and not parsed.names_code:
                score *= self.PROSE_API_FACTOR
            if len(parsed.phrase) > 6 and " " in parsed.phrase:
                if parsed.phrase in doc.text.lower():
                    score *= 1.5
            hits.append(Hit(doc, score, matched[index] & set(typed)))
        hits.sort(key=lambda hit: (-hit.score, hit.doc.cite))
        return self._diverse(hits, limit, per_group, single_kind=allowed is not None), len(hits)

    def _heading_match(self, index: int, query: Query) -> float:
        """How well the query and the best of a document's headings say the same thing, from
        0 (no shared word) to 1 (the same words): the share of the query's words the heading
        has times the share of the heading's words the query has. Titles count half.

        A query word whose expansion is in the heading (``hand`` and ``Handoffs``) counts
        half, and a subject word (`SUBJECT_WORDS`) counts as little in the heading as in the
        query."""
        typed, terms = query.typed, query.terms
        total = sum(terms[term] for term in typed)
        said = frozenset(terms)  # the typed words and their expansions
        best = 0.0
        for position, heading in enumerate(self._headings[index]):
            if not heading.tokens or not heading.tokens & said:
                continue
            in_query = 0.0
            for term in typed:
                if term in heading.tokens:
                    in_query += terms[term]
                elif query.expansions.get(term, frozenset()) & heading.tokens:
                    in_query += terms[term] * _EXPANSION_WEIGHT
            shared = sum(
                SUBJECT_WEIGHT if word & SUBJECT_WORDS else 1.0
                for word in heading.words
                if word & said
            )
            match = in_query / total * shared / heading.weight
            best = max(best, match if position == 0 else match / 2)
        return best

    def _diverse(
        self, hits: list[Hit], limit: int, per_group: int, *, single_kind: bool
    ) -> list[Hit]:
        """The best ``limit`` hits, with at most ``per_group`` from one page or example and,
        unless one kind was asked for, at most half the list from examples or from the API,
        so the guide pages that explain an idea are never crowded out."""
        per_kind_cap = max(2, -(-limit // 2))
        chosen: list[Hit] = []
        overflow: list[Hit] = []
        per: Counter[str] = Counter()
        kinds: Counter[str] = Counter()
        for hit in hits:
            kind_full = (
                not single_kind and hit.doc.kind != "docs" and kinds[hit.doc.kind] >= per_kind_cap
            )
            if per[hit.doc.group] < per_group and not kind_full:
                per[hit.doc.group] += 1
                kinds[hit.doc.kind] += 1
                chosen.append(hit)
            else:
                overflow.append(hit)
            if len(chosen) >= limit:
                break
        if len(chosen) < limit:
            chosen.extend(overflow[: limit - len(chosen)])
        chosen.sort(key=lambda hit: -hit.score)
        return chosen


@dataclass(frozen=True)
class _Heading:
    """A heading's words that carry meaning, each as the set of its search tokens."""

    words: tuple[frozenset[str], ...]
    tokens: frozenset[str]
    """Every token of every word."""
    weight: float
    """How many words it has, a subject word (`SUBJECT_WORDS`) counting `SUBJECT_WEIGHT`."""

    @classmethod
    def of(cls, text: str) -> _Heading:
        words = tuple(frozenset(t) for t in map(_word_tokens, _WORD.findall(text)) if t)
        weight = sum(SUBJECT_WEIGHT if word & SUBJECT_WORDS else 1.0 for word in words)
        return cls(words, frozenset().union(*words), weight)


def snippet(text: str, terms: Iterable[str], *, width: int = 280) -> str:
    """The lines of ``text`` that match ``terms`` best, joined and shortened to ``width``."""
    wanted = set(terms)
    scored: list[tuple[int, int, str]] = []
    for index, raw in enumerate(text.splitlines()):
        line = raw.strip()
        if not line or line.startswith(("```", "~~~", "::: ", "--8<--", "|---", "| ---")):
            continue
        scored.append((len(set(tokenize(line)) & wanted), index, line))
    if not scored:
        return ""
    matching = [item for item in scored if item[0] > 0]
    if matching:
        picked = sorted(matching, key=lambda item: (-item[0], item[1]))[:3]
        picked.sort(key=lambda item: item[1])  # back in document order
    else:
        picked = scored[:1]
    joined = " ".join(" ... ".join(line for _, _, line in picked).split())
    if len(joined) > width:
        joined = joined[: width - 3].rsplit(" ", 1)[0] + "..."
    return joined
