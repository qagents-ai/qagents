"""Route each question to a quick or an expert agent, on a model chain that survives a 429.

A classifier agent labels each of three questions "simple" or "complex", and `route` sends it to
a quick agent (one-line answers, a small budget) or an expert agent (works step by step, a
larger budget). The three questions run concurrently. Every agent runs on one model chain:

    FallbackModel([AlwaysBusy(), RateLimitedModel(ctx.model, max_concurrent=2, ...)])

`AlwaysBusy` stands in for a primary deployment that is over its quota: it refuses every request
with a rate-limit error before streaming anything, so `FallbackModel` moves each request on to
the next model and logs a warning on the ``qagents.fallback`` logger. `RateLimitedModel` keeps
all agents together under one concurrency and requests-per-minute budget instead of letting
them run into the provider's own 429s. (Wrap a real primary in
``RateLimitedModel(..., max_wait_s=0)`` and its overflow goes to the next model.) A run's result
does not say which model answered; each ``ModelResponse`` does (``Hooks.after_model``, or the
chat spans with ``--mlflow``).

Both routes use ``ctx.model``, because the harness gives each example one metered, recordable
model. With two models you would give the quick route the cheaper one, for example
``quick.copy(model="anthropic:claude-haiku-5-5")`` or a smaller QCloud catalog model, and keep
the stronger model for the expert. Here the routes differ in instructions and limits.

Mirrors: LangGraph's routing workflow with LangChain's model fallbacks. There a router node
calls ``llm.with_structured_output(Route)``, ``add_conditional_edges`` maps the decision to the
next node, each node can be bound to its own chat model, ``model.with_fallbacks([backup])``
fails over to another provider, and an ``InMemoryRateLimiter`` paces requests. In qagents
``route`` is a plain function call, so the choice stays visible in your code; the classifier is
an agent with ``output_type=Literal[...]``, so a label outside the routes is corrected by the
model, not your code; and failover and rate limiting are `Model` wrappers that every agent
shares.

Run it:

    python -m examples.patterns.fallback_router --provider anthropic
    python -m examples.patterns.fallback_router --provider qcloud
"""

from __future__ import annotations

import asyncio
import re
import textwrap
from collections.abc import AsyncIterator
from typing import Literal

from examples.harness import ExampleContext, ExampleOutcome, main
from qagents import Agent, FallbackModel, Limits, Model, RateLimitedModel, route
from qagents.errors import ModelRateLimitError
from qagents.models import ModelRequest, ModelStreamEvent

NAME = "fallback_router"
DESCRIPTION = "route() picks a quick or an expert agent; a FallbackModel survives a busy primary"
INSPIRED_BY = "LangGraph routing with per-node models, LangChain with_fallbacks"

# Each question, the route it should take, and the words or numbers a correct answer contains.
QUESTIONS = (
    ("What is the capital of Portugal?", "simple", ("Lisbon",)),
    (
        "A bakery bakes 3 trays of 48 rolls and 12 rolls burn. It packs the rest in bags of 7. "
        "How many full bags does it fill, and how many rolls are left over?",
        "complex",
        ("18", "6"),
    ),
    ("How many days does a leap year have?", "simple", ("366",)),
)


class AlwaysBusy(Model):
    """A primary model that is always over its quota: every request fails with HTTP 429.

    It fails before streaming anything, which is the one case where `FallbackModel` may try the
    next model without risking a duplicated, billed answer.
    """

    provider = "local"
    name = "always-busy"

    def __init__(self) -> None:
        self.refused = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.refused += 1
        raise ModelRateLimitError(
            "429 Too Many Requests: the always-busy deployment is over its quota.",
            provider=self.provider,
            model=self.name,
            status_code=429,
            retryable=True,
        )
        yield  # never reached; it makes this method an async generator, like every stream


def answered(text: str, expected: tuple[str, ...]) -> bool:
    """Whether the answer holds every expected word: after the last "Answer:", if there is one,
    so numbers in the working do not count."""
    answer = re.split(r"answer:", text, flags=re.IGNORECASE)[-1]
    return all(re.search(rf"\b{re.escape(word)}\b", answer, re.IGNORECASE) for word in expected)


async def run(ctx: ExampleContext) -> ExampleOutcome:
    busy = AlwaysBusy()
    steady = RateLimitedModel(ctx.model, max_concurrent=2, requests_per_minute=60)
    model = FallbackModel([busy, steady])

    classifier = Agent(
        model,
        name="classifier",
        instructions=(
            "Classify the user's question; do not answer it. Answer 'simple' when it asks for "
            "one well-known fact. Answer 'complex' when it needs several steps of reasoning or "
            "arithmetic."
        ),
        output_type=Literal["simple", "complex"],
        output_mode=ctx.output_mode,
        limits=Limits(max_turns=3, max_cost_usd=0.03),
        hooks=ctx.hooks,
    )
    quick = Agent(
        model,
        name="quick",
        instructions=(
            "Answer the question in one short sentence, with no explanation. Write numbers "
            "in digits."
        ),
        limits=Limits(max_turns=2, max_cost_usd=0.03),
        hooks=ctx.hooks,
    )
    expert = Agent(
        model,
        name="expert",
        instructions=(
            "Solve the problem step by step, check each calculation, and finish with one line "
            "that starts with 'Answer:' and states every number asked for, in digits."
        ),
        limits=Limits(max_turns=3, max_cost_usd=0.10),
        hooks=ctx.hooks,
    )
    routes = {"simple": quick, "complex": expert}

    ctx.log(f"Routing {len(QUESTIONS)} questions at once through {model!r}")
    routed = await asyncio.gather(
        *(route(classifier, routes, question) for question, _, _ in QUESTIONS)
    )

    correct = []
    for (question, _, expected), choice in zip(QUESTIONS, routed, strict=True):
        result = choice.result
        correct.append(result.status == "completed" and answered(result.text, expected))
        handler = routes[choice.route].name if choice.route else "nobody"
        ctx.log(f"{question} -> {choice.route} -> {handler}")
        print(textwrap.indent(result.text.strip() or f"(run {result.status})", "   "))

    ctx.log(f"The busy primary refused {busy.refused} requests; the fallback answered them")
    labels = [choice.route for choice in routed]
    expected_labels = [label for _, label, _ in QUESTIONS]
    ok = all(correct) and labels == expected_labels and busy.refused > 0
    if ok:
        summary = f"3 questions answered via the fallback, routed {labels}"
    else:
        summary = (
            f"answers correct: {correct}, routes {labels} (expected {expected_labels}), "
            f"primary refusals: {busy.refused}"
        )
    return ExampleOutcome(
        ok=ok,
        summary=summary,
        output=[choice.result.text for choice in routed],
        details={"routes": labels, "primary_refusals": busy.refused, "correct": correct},
    )


if __name__ == "__main__":
    main(run, name=NAME, description=DESCRIPTION, inspired_by=INSPIRED_BY)
