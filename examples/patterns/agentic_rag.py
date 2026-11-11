"""Retrieval as tools: the agent searches, judges, searches again, reads, and cites its sources.

A knowledge base of twelve short markdown documents about a fictional smart home product is
searched with a small BM25 keyword scorer (qagents has no built-in retriever or vector
store). The agent decides what to search for, which results are worth reading in full, and
when to search again with other words. Its answer is structured (`CitedAnswer`: the answer
and the ids of its sources), and an output validator sends the model back with `ModelRetry`
when it cites a document that no search or read returned in the conversation.

Mirrors: LangGraph's agentic RAG tutorial: ``generate_query_or_respond`` (the model with the
retriever tool), ``retrieve`` (a ``ToolNode``), ``grade_documents`` (a grader model whose
yes/no routes to ``generate_answer`` or to ``rewrite_question``, which loops back). In
qagents: one agent with two tools and no graph. The model's turn is
``generate_query_or_respond``, its ``search_docs`` and ``read_document`` calls are
``retrieve``, and grading and rewriting are decisions it makes inside the same loop, as the
instructions ask: judge the snippets, read what is relevant, search again with other words
when something is missing. There is no separate grader call; the rule that must hold (cite
only what was retrieved) is checked in code by the output validator.

Run it:

    python -m examples.patterns.agentic_rag --provider anthropic
    python -m examples.patterns.agentic_rag --provider qcloud
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

from pydantic import BaseModel, Field

from examples.harness import ExampleContext, ExampleOutcome, main
from qagents import Agent, Limits, Message, ModelRetry, RunContext, ToolError, tool

NAME = "agentic_rag"
DESCRIPTION = "Search and read tools, a structured cited answer, and a citation validator"
INSPIRED_BY = "LangGraph agentic RAG (retrieve, grade documents, rewrite question)"

QUESTION = (
    "How long does the battery of the Larkspur motion sensor last, and how do I pair a new "
    "motion sensor with the hub?"
)
EXPECTED_SOURCES = {"KB-104", "KB-105"}  # the answer is in these two documents, and only these

DOCUMENTS = {
    "KB-101": """# Getting started with the Larkspur Hub
Plug the hub into power and wait until its light turns steady white. Install the Larkspur app,
create an account and tap Add hub. The app finds the hub over Bluetooth and asks for your
Wi-Fi network.""",
    "KB-102": """# Connecting the hub to Wi-Fi
The hub connects to 2.4 GHz Wi-Fi networks with WPA2 or WPA3; 5 GHz networks are not
supported. If the light stays amber, the hub has no internet connection: move it closer to
the router or restart the router.""",
    "KB-103": """# What the hub light means
Steady white: everything works. Amber: no internet connection. Pulsing purple: a firmware
update is installing, do not unplug the hub. Red: the hub has an error; restart it, and
contact support if the light stays red.""",
    "KB-104": """# Motion sensor battery
The motion sensor runs on one CR2477 coin cell. At the default sensitivity the battery lasts
about 18 months; high sensitivity shortens that to about 12 months. The app warns you when
the charge drops below 10 percent. To replace the cell, twist the back cover anticlockwise.""",
    "KB-105": """# Pairing a motion sensor
Open the Larkspur app, go to Devices and tap Add device, then choose Motion sensor. Press and
hold the button on the sensor for 7 seconds, until its LED blinks green. Keep the sensor
within 5 metres of the hub while pairing; it takes up to a minute.""",
    "KB-106": """# Door and window sensor
The door and window sensor uses two AAA batteries, which last about 2 years. To pair it, pull
out the plastic battery tab while the app shows Add device. Mount the magnet less than 2 cm
from the sensor.""",
    "KB-107": """# Factory reset of the hub
Press and hold the reset pin on the back of the hub for 15 seconds, until the light flashes
red. A factory reset removes every paired device, automation and setting from the hub.""",
    "KB-108": """# Firmware updates
The hub installs firmware updates automatically at 03:00 local time. To update now, open
Settings, then Hub, then Check for updates. Sensors are updated through the hub.""",
    "KB-109": """# Automations
An automation links a trigger to actions, for example: when motion is detected after sunset,
turn on the hallway lights. A hub runs up to 50 automations, even without internet.""",
    "KB-110": """# Notifications and quiet hours
Choose which events send a push notification under Settings, then Notifications. During
quiet hours only alarms are delivered; other notifications wait in the app's inbox.""",
    "KB-111": """# Privacy and data
Event history is stored in the EU for 30 days and then deleted. You can export your history
as CSV from Settings, then Privacy, or delete it at any time.""",
    "KB-112": """# Warranty and returns
Larkspur devices have a 2-year warranty. Unused devices can be returned within 30 days of
delivery for a full refund.""",
}

DOC_ID = re.compile(r"\bKB-\d{3}\b")


# -- a small BM25 index -----------------------------------------------------------------------

STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it its my of on or "
    "the then this to what when where which with you your".split()
)


def terms(text: str) -> list[str]:
    """Lower-case words without stopwords, with plural and -ing/-ed endings stripped."""
    found = []
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if word in STOPWORDS:
            continue
        for ending, replacement in (("ies", "y"), ("ing", ""), ("ed", ""), ("s", "")):
            if word.endswith(ending) and len(word) - len(ending) >= 3:
                word = word[: -len(ending)] + replacement
                break
        found.append(word)
    return found


class Index:
    """Okapi BM25 over whole documents; the best-matching sentences are the snippet."""

    def __init__(self, documents: dict[str, str], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = documents
        self.k1, self.b = k1, b
        self.counts = {doc_id: Counter(terms(text)) for doc_id, text in documents.items()}
        self.lengths = {doc_id: sum(counts.values()) for doc_id, counts in self.counts.items()}
        self.average = sum(self.lengths.values()) / len(self.lengths)
        self.containing = Counter(term for counts in self.counts.values() for term in counts)

    def score(self, doc_id: str, query: Iterable[str]) -> float:
        counts, total = self.counts[doc_id], len(self.counts)
        norm = self.k1 * (1 - self.b + self.b * self.lengths[doc_id] / self.average)
        score = 0.0
        for term in query:
            frequency, documents = counts[term], self.containing[term]
            if frequency:
                idf = math.log(1 + (total - documents + 0.5) / (documents + 0.5))
                score += idf * frequency * (self.k1 + 1) / (frequency + norm)
        return score

    def search(self, query: str, limit: int = 3) -> list[dict[str, str]]:
        wanted = set(terms(query))
        scored = [(self.score(doc_id, wanted), doc_id) for doc_id in self.documents]
        best = sorted((item for item in scored if item[0] > 0), key=lambda item: -item[0])
        return [
            {
                "id": doc_id,
                "title": title(self.documents[doc_id]),
                "snippet": self.snippet(doc_id, wanted),
            }
            for _, doc_id in best[:limit]
        ]

    def snippet(self, doc_id: str, wanted: set[str], size: int = 2) -> str:
        """The ``size`` sentences with the most query terms, in document order."""
        body = " ".join(self.documents[doc_id].splitlines()[1:])
        sentences = re.split(r"(?<=[.!?])\s+", body)
        ranked = sorted(
            range(len(sentences)), key=lambda i: -len(wanted & set(terms(sentences[i])))
        )
        return " ".join(sentences[i] for i in sorted(ranked[:size]))


def title(markdown: str) -> str:
    return markdown.splitlines()[0].lstrip("# ").strip()


INDEX = Index(DOCUMENTS)


# -- the tools, the answer and its validator --------------------------------------------------


@tool(read_only=True)
def search_docs(query: str) -> list[dict[str, str]] | str:
    """Search the Larkspur knowledge base. Returns up to three documents: id, title, snippet.

    Args:
        query: A few keywords, for example "hub light amber".
    """
    return INDEX.search(query) or f"No document matches {query!r}. Try other keywords."


@tool(read_only=True)
def read_document(doc_id: str) -> str:
    """Read one document of the knowledge base in full.

    Args:
        doc_id: An id that search_docs returned, for example "KB-101".
    """
    text = DOCUMENTS.get(doc_id.strip().upper())
    if text is None:
        raise ToolError(f"There is no document {doc_id!r}. Use an id that search_docs returned.")
    return f"[{doc_id.strip().upper()}] {text}"


class CitedAnswer(BaseModel):
    answer: str = Field(description="The answer to the question, in two to four sentences.")
    sources: list[str] = Field(
        description='Ids of the documents the answer is based on, for example ["KB-101"].'
    )


def retrieved(messages: Iterable[Message]) -> set[str]:
    """Ids of the documents that search_docs or read_document showed the model."""
    return {
        doc_id
        for message in messages
        for result in message.tool_results
        if result.name in ("search_docs", "read_document") and not result.is_error
        for doc_id in DOC_ID.findall(result.content)
    }


def cite_only_retrieved(ctx: RunContext, answer: CitedAnswer) -> CitedAnswer:
    """Output validator: reject citations of documents the model never retrieved."""
    # "kb-104", "KB-104: Motion sensor battery" -> "KB-104"; anything else is kept as given.
    found = [DOC_ID.findall(s.upper()) or [s.strip()] for s in answer.sources if s.strip()]
    sources = list(dict.fromkeys(doc_id for ids in found for doc_id in ids))
    seen = retrieved(ctx.messages)
    unknown = [source for source in sources if source not in seen]
    if unknown:
        raise ModelRetry(
            f"You cited {', '.join(unknown)}, but no search_docs or read_document result in "
            f"this conversation contains it. Cite only documents you retrieved "
            f"({', '.join(sorted(seen)) or 'none yet'}): remove the others, or search for the "
            "one you need first."
        )
    return answer.model_copy(update={"sources": sources})  # bare ids, no repeats


INSTRUCTIONS = (
    "You answer questions about Larkspur smart home products from the knowledge base only. "
    "1. Search with search_docs, using a few keywords. "
    "2. Judge each result by its title and snippet: read the relevant ones in full with "
    "read_document and ignore the rest. "
    "3. If what you read does not answer every part of the question, search again with "
    "different keywords; make at most three searches in all. "
    "4. Answer from what you read, and give as sources the ids of the documents the answer "
    "uses, and no others. If the knowledge base does not answer the question, say so and give "
    "no sources."
)


async def run(ctx: ExampleContext) -> ExampleOutcome:
    agent = Agent(
        ctx.model,
        name="kb_assistant",
        instructions=INSTRUCTIONS,
        tools=[search_docs, read_document],
        output_type=CitedAnswer,
        output_mode=ctx.output_mode,
        output_validators=[cite_only_retrieved],
        # Each rejected citation uses up one output retry (the default is 2).
        limits=Limits(max_turns=12, max_cost_usd=0.25, output_retries=3),
        hooks=ctx.hooks,
    )
    ctx.log(QUESTION)
    result = await ctx.print_stream(agent.stream(QUESTION))
    if result.status != "completed" or not isinstance(result.output, CitedAnswer):
        return ExampleOutcome(ok=False, summary=f"run {result.status}: {result.stop_reason}")

    answer = result.output
    ctx.log(f"Answer: {answer.answer} Sources: {', '.join(answer.sources) or 'none'}")
    cited = set(answer.sources)
    facts = all(  # 18 months of battery, 7 seconds on the button
        re.search(pattern, answer.answer, re.IGNORECASE)
        for pattern in (r"\b18\b|eighteen", r"\b7\b|seven")
    )
    ok = cited == EXPECTED_SOURCES and facts
    if ok:
        summary = f"cited {', '.join(sorted(cited))} with both facts"
    else:
        summary = f"cited {sorted(cited)} (want {sorted(EXPECTED_SOURCES)}), facts found: {facts}"
    return ExampleOutcome(
        ok=ok,
        summary=summary,
        output=answer.model_dump(),
        details={
            "searches": [
                c.arguments.get("query")
                for m in result.messages
                for c in m.tool_calls
                if c.name == "search_docs"
            ],
            "retrieved": sorted(retrieved(result.messages)),
            "turns": result.turns,
        },
    )


if __name__ == "__main__":
    main(run, name=NAME, description=DESCRIPTION, inspired_by=INSPIRED_BY)
