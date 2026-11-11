"""Evaluate a small tool-using assistant on six cases, save the report, compare providers.

The assistant answers questions about a shop's products and orders with one ``lookup`` tool,
and has a ``place_order`` tool it must never use for a question. ``qagents.evals.evaluate``
runs the six cases, each in a fresh session, and scores every run with ``ExactMatch`` (the reply
is exactly the expected value, as the instructions ask), ``Contains`` (the value appears at
all), ``ToolCalled`` (it looked the answer up, and it never ordered), ``MaxTurns`` and
``MaxCostUSD`` (budgets), and an ``LLMJudge`` on the same model. A case passes when every
evaluator passes. The report is saved as JSON and Markdown in
``examples/results/evals-<provider>.*``; when the other provider's report is there too,
``compare()`` prints the two side by side and ``regressions()`` lists the cases that got worse.

Mirrors: LangSmith's ``evaluate(target, data=dataset, evaluators=[...])`` with reference
outputs (code evaluators and an LLM-as-judge, experiments compared in the LangSmith UI), and
CrewAI's ``crewai test`` (an LLM scores every task of the crew from 1 to 10). In qagents the
evaluators see the whole run (tool calls, turns and cost, not only the final text), nothing
runs on a service (the report is a file to diff; ``log_to_mlflow`` is optional), and
``compare`` and ``regressions`` turn two reports into a table or a CI check.

Run it on both providers, then the second run prints the comparison:

    python -m examples.patterns.evals_compare --provider anthropic
    python -m examples.patterns.evals_compare --provider qcloud
"""

from __future__ import annotations

from examples.harness import RESULTS_DIR, ExampleContext, ExampleOutcome, main
from qagents import Agent, Limits, ToolError, prompts, tool
from qagents.evals import (
    Case,
    CaseResult,
    Contains,
    Dataset,
    EvalReport,
    ExactMatch,
    JudgeVerdict,
    LLMJudge,
    MaxCostUSD,
    MaxTurns,
    ToolCalled,
    compare,
    evaluate,
    regressions,
)
from qagents.models.recording import ReplayModel

NAME = "evals_compare"
DESCRIPTION = "Score a tool-using assistant on six cases and compare the reports of two providers"
INSPIRED_BY = "LangSmith evaluate() with code and LLM-judge evaluators / crewai test"

PASS_RATE = 0.8
PROVIDERS = ("anthropic", "qcloud")

PRODUCTS = {  # fixture data
    "KB-200": {"name": "Mechanical keyboard", "price_eur": 89.00, "stock": 14},
    "MS-310": {"name": "Wireless mouse", "price_eur": 34.50, "stock": 0, "back": "2026-12-01"},
    "HD-550": {"name": "Headset", "price_eur": 129.90, "stock": 6},
}
ORDERS = {
    "A-1001": {"sku": "MS-310", "quantity": 2, "status": "delivered"},
    "A-1002": {"sku": "KB-200", "quantity": 1, "status": "shipped", "tracking": "TRK-55821"},
    "A-1003": {"sku": "HD-550", "quantity": 1, "status": "processing"},
}

DATASET = Dataset(
    [
        Case(name="stock", prompt="How many KB-200 keyboards are in stock?", expected="14"),
        Case(name="price", prompt="How much does the HD-550 headset cost?", expected="129.90"),
        Case(name="status", prompt="What is the status of order A-1002?", expected="shipped"),
        Case(
            name="tracking",
            prompt="What is the tracking number of order A-1002?",
            expected="TRK-55821",
        ),
        Case(
            name="restock",
            prompt="On what date will the MS-310 mouse be back in stock?",
            expected="2026-12-01",
        ),
        Case(
            name="order-in-stock",
            prompt="Is the product of order A-1001 in stock right now?",
            expected="no",
            metadata={"lookups": 2},
        ),
    ],
    name="shop-assistant",
)

INSTRUCTIONS = (
    "You answer questions about the shop's products and orders. Always look the answer up "
    "with lookup, giving a SKU such as KB-200 or an order id such as A-1002; never guess. For "
    "the product of an order, look up the order, then its SKU.\n"
    "Reply with the value only: no sentence, no unit or currency, no full stop. Write prices "
    "with two decimals (89.00) and dates as YYYY-MM-DD, and answer a yes/no question with yes "
    "or no.\n"
    "Call place_order only when the customer explicitly asks to buy something."
)
RUBRIC = (
    "The answer gives the same value as the reference. Ignore formatting such as letter case, "
    "a currency sign or a final full stop. An answer that gives another value fails."
)


@tool(read_only=True)
def lookup(ref: str) -> str:
    """Look up a product by its SKU (like KB-200) or an order by its id (like A-1002).

    Args:
        ref: A SKU or an order id.
    """
    found = [key for key in (*PRODUCTS, *ORDERS) if key in ref.upper()]  # "order A-1002" too
    if not found:
        raise ToolError(f"No product or order {ref!r}. SKUs: {', '.join(PRODUCTS)}.")
    key = found[0]
    if key in PRODUCTS:
        item = PRODUCTS[key]
        text = f"{key}: {item['name']}, price {item['price_eur']:.2f} EUR, {item['stock']} in stock"
        return text + (f", back in stock on {item['back']}" if "back" in item else "")
    order = ORDERS[key]
    text = f"Order {key}: {order['quantity']} x {order['sku']}, status {order['status']}"
    return text + (f", tracking number {order['tracking']}" if "tracking" in order else "")


@tool(requires_approval=True)
def place_order(sku: str, quantity: int) -> str:
    """Order a product for the customer.

    Args:
        sku: The product's SKU.
        quantity: How many to order.
    """
    return f"Ordered {quantity} x {sku}."


def show(result: CaseResult) -> None:
    """Print each case as it is scored."""
    failed = [item.name for item in result.evaluations if not item.passed]
    verdict = result.outcome + (f" ({', '.join(failed)})" if failed else "")
    print(f"  {result.case}: {verdict}; answered {result.output_text.strip()[:40]!r}", flush=True)


async def run(ctx: ExampleContext) -> ExampleOutcome:
    assistant = Agent(
        ctx.model,
        name="shop_assistant",
        instructions=INSTRUCTIONS,
        tools=[lookup, place_order],
        limits=Limits(max_turns=5, max_cost_usd=0.05),
        hooks=ctx.hooks,
    )
    # LLMJudge would build its judge itself, but only in native or prompted output mode; this
    # one answers in the harness's mode (a final-answer tool on QCloud).
    judge = Agent(
        ctx.model,
        name="judge",
        instructions=prompts.JUDGE_INSTRUCTIONS,
        output_type=JudgeVerdict,
        output_mode=ctx.output_mode,
        limits=Limits(max_turns=3, max_cost_usd=0.05),
        hooks=ctx.hooks,
    )
    evaluators = [
        ExactMatch(case_sensitive=False),
        Contains(case_sensitive=False),
        ToolCalled("lookup"),
        ToolCalled("place_order", times=0, name="never_orders"),
        MaxTurns(4),
        MaxCostUSD(0.05, allow_unknown=True),  # an OpenAI-compatible server may report no cost
        LLMJudge(judge, RUBRIC, threshold=0.7),
    ]

    ctx.log(f"Evaluating {len(DATASET)} cases with {len(evaluators)} evaluators")
    report = await evaluate(
        assistant,
        DATASET,
        evaluators,
        concurrency=3,
        name=f"shop assistant ({ctx.provider}:{ctx.model.name})",
        metadata={"provider": ctx.provider},
        on_result=show,
    )
    print(report.to_markdown(max_chars=40))

    replayed = isinstance(getattr(ctx.model, "inner", None), ReplayModel)
    label = f"{ctx.provider}-replay" if replayed else ctx.provider  # never overwrite a live run
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    saved = RESULTS_DIR / f"evals-{label}.json"
    report.save(saved)
    report.save(saved.with_suffix(".md"))
    ctx.log(f"Saved {saved.as_posix()} and its .md")

    compared: dict[str, list[str]] = {}  # other provider -> cases that got worse here
    for other in PROVIDERS:
        path = RESULTS_DIR / f"evals-{other}.json"
        if other != ctx.provider and path.is_file():
            baseline = EvalReport.load(path)
            ctx.log(f"Compared with {path.as_posix()}:")
            print(compare(baseline, report, labels=(other, label)))
            compared[other] = regressions(baseline, report)  # what a CI check would fail on
            ctx.log(f"Regressions against {other}: {', '.join(compared[other]) or 'none'}")

    summary = report.summary
    ok = summary.pass_rate >= PASS_RATE
    failing = [result.case for result in report.failures()]
    return ExampleOutcome(
        ok=ok,
        summary=f"pass rate {summary.pass_rate:.2f} (needs {PASS_RATE:.2f})"
        + (f"; failed: {', '.join(failing)}" if failing else ""),
        output=summary.model_dump(),
        details={"report": saved.as_posix(), "regressions": compared, "failed": failing},
    )


if __name__ == "__main__":
    main(run, name=NAME, description=DESCRIPTION, inspired_by=INSPIRED_BY)
