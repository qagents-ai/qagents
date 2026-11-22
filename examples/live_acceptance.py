"""Live acceptance test against the real QCloud API.

It makes a handful of small, paid requests and checks what the offline tests
cannot: that a real catalog model streams, calls a tool, returns schema-valid
JSON, reports usage, and that errors arrive as the expected types.

    python examples/live_acceptance.py --dry-run   # no key: pick a model, show the spend cap
    python examples/live_acceptance.py             # needs a key; spends a fraction of a cent

The key is read from QCLOUD_API_KEY, then from a ``.env`` file next to
pyproject.toml, then (on Windows) from your user environment variables. It is
never printed. Without ``--model`` or QCLOUD_MODEL, the cheapest catalog model
that lists the "tools" capability and has a non-zero price is used.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import httpx
from pydantic import BaseModel

from qagents import Agent, Limits, ModelSettings, QCloudModel, tool
from qagents.errors import ModelError

DEFAULT_BASE_URL = "https://platform.qcloud.host/api/v1"
REQUESTS = 5  # upper bound on paid requests this script makes
INPUT_TOKENS_PER_REQUEST = 1500  # generous estimate, for the worst-case figure


def find_key() -> str | None:
    if os.environ.get("QCLOUD_API_KEY"):
        return os.environ["QCLOUD_API_KEY"]
    dotenv = Path(__file__).resolve().parents[1] / ".env"
    if dotenv.is_file():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "QCLOUD_API_KEY" and value.strip():
                return value.strip().strip("\"'")
    if sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                return winreg.QueryValueEx(key, "QCLOUD_API_KEY")[0] or None
        except OSError:
            return None
    return None


def pick_model(base_url: str, wanted: str | None) -> dict:
    """Read the public catalog (no key needed) and choose the model to test."""
    catalog_url = base_url.removesuffix("/v1") + "/public/models"
    models = httpx.get(catalog_url, timeout=30).raise_for_status().json()["data"]
    if wanted:
        match = next((m for m in models if m["id"] == wanted), None)
        if match is None:
            sys.exit(f"Model {wanted!r} is not in the catalog at {catalog_url}.")
        return match

    def price(model: dict) -> float:
        pricing = model.get("pricing") or {}
        return pricing.get("input_per_million", 0) + pricing.get("output_per_million", 0)

    usable = [m for m in models if "tools" in (m.get("capabilities") or []) and price(m) > 0]
    if not usable:
        sys.exit("The catalog has no priced model with the 'tools' capability.")
    return min(usable, key=price)


class Place(BaseModel):
    city: str
    country: str


@tool(read_only=True)
def lookup_tracking_code(order_id: str) -> str:
    """Return the tracking code of an order.

    Args:
        order_id: The order id, for example "A-1001".
    """
    return "ZX-4417" if order_id == "A-1001" else "unknown order"


async def run_checks(client, model_id: str, max_tokens: int, cap: float) -> bool:
    model = QCloudModel(model_id, client=client)
    settings = ModelSettings(max_tokens=max_tokens, temperature=0)
    limits = Limits(max_turns=3, max_cost_usd=cap)
    spent = 0.0
    passed = True

    def report(name: str, ok: bool, detail: str) -> None:
        nonlocal passed
        passed = passed and ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    # 1. A streamed answer, with time to first token and usage.
    began = time.perf_counter()
    first_token: float | None = None
    stream = Agent(model, settings=settings, limits=limits).stream("Reply with exactly: ready")
    async for event in stream:
        if event.type == "text_delta" and first_token is None:
            first_token = time.perf_counter() - began
    result = stream.result
    usage = result.usage
    spent += usage.cost_usd or 0
    report(
        "streaming",
        bool(result.text.strip()) and usage.input_tokens > 0 and usage.output_tokens > 0,
        f"text={result.text.strip()[:40]!r} first_token={first_token and round(first_token, 2)}s "
        f"tokens={usage.input_tokens}+{usage.output_tokens} cost_usd={usage.cost_usd}",
    )

    # 2. A tool loop: the answer needs a value only the tool knows.
    agent = Agent(model, tools=[lookup_tracking_code], settings=settings, limits=limits)
    result = await agent.run("What is the tracking code of order A-1001? Use the tool.")
    spent += result.usage.cost_usd or 0
    called = [call.name for message in result.messages for call in message.tool_calls]
    report(
        "tool loop",
        result.completed and "ZX-4417" in result.text and called == ["lookup_tracking_code"],
        f"status={result.status} calls={called} turns={result.turns} "
        f"cost_usd={result.usage.cost_usd}",
    )

    # 3. Structured output through response_format json_schema.
    agent = Agent(model, output_type=Place, settings=settings, limits=limits)
    try:
        result = await agent.run("Which city is the capital of Greece, and in which country?")
        spent += result.usage.cost_usd or 0
        ok = isinstance(result.output, Place) and "athens" in result.output.city.lower()
        report("structured output", ok, f"output={result.output!r} turns={result.turns}")
    except Exception as error:  # report it as a finding, keep going
        report("structured output", False, f"{type(error).__name__}: {error}")

    # 4. An unknown model is refused with a typed error and no charge.
    try:
        await Agent(QCloudModel("qagents-no-such-model", client=client)).run("hi")
        report("unknown model", False, "the request unexpectedly succeeded")
    except ModelError as error:
        report(
            "unknown model",
            error.status_code in (400, 404),
            f"{type(error).__name__} status={error.status_code} request_id={error.request_id}",
        )

    print(f"  total cost reported by qagents: {spent:.6f} USD (cap {cap} USD)")
    return passed


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=os.environ.get("QCLOUD_MODEL"))
    parser.add_argument("--max-usd", type=float, default=0.01, help="refuse to run above this")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true", help="make no paid request")
    args = parser.parse_args()

    base_url = os.environ.get("QCLOUD_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    chosen = pick_model(base_url, args.model)
    pricing = chosen.get("pricing") or {}
    worst_case = (
        REQUESTS
        * (
            INPUT_TOKENS_PER_REQUEST * pricing.get("input_per_million", 0)
            + args.max_tokens * pricing.get("output_per_million", 0)
        )
        / 1_000_000
    )
    print(f"model: {chosen['id']}  capabilities={chosen.get('capabilities')}")
    price_in, price_out = pricing.get("input_per_million"), pricing.get("output_per_million")
    print(f"price per million tokens: in={price_in} out={price_out}")
    print(f"worst case for {REQUESTS} requests: {worst_case:.6f} USD (cap {args.max_usd} USD)")
    if worst_case > args.max_usd:
        print("Refusing to run: the worst case is above the cap. Raise --max-usd or pick a model.")
        return 2
    if args.dry_run:
        print("Dry run: no paid request was made.")
        return 0

    key = find_key()
    if not key:
        print("No QCLOUD_API_KEY found (environment, .env, or Windows user variables).")
        return 2
    from qcloud_ai import AsyncQCloud

    async with AsyncQCloud(api_key=key, base_url=base_url, timeout=120) as client:
        passed = await run_checks(client, chosen["id"], args.max_tokens, args.max_usd)
    print("RESULT:", "all checks passed" if passed else "some checks failed")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
