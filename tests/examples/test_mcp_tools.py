"""Offline tests of examples/patterns/mcp_tools.py: a real stdio MCP server, a scripted model."""

from __future__ import annotations

import json
import runpy
import sys

import pytest

pytest.importorskip("mcp")

from examples import run_all  # noqa: E402
from examples.harness import main, scripted_context  # noqa: E402
from examples.patterns import mcp_tools as example  # noqa: E402
from examples.patterns.mcp_tools import StockPolicy, run  # noqa: E402
from qagents import Tool, tool  # noqa: E402
from qagents.models.testing import FunctionModel, call  # noqa: E402


def calls_so_far(request) -> list[str]:
    return [c.name for m in request.messages if m.role == "assistant" for c in m.tool_calls]


def results_so_far(request) -> dict[str, list]:
    found: dict[str, list] = {}
    for message in request.messages:
        for result in message.tool_results:
            found.setdefault(result.name, []).append(result)
    return found


def policy(request):
    """Lists the stock, books both items in one turn, reads them back, answers."""
    names = calls_so_far(request)
    if not names:
        return call("list_items")
    if "update_stock" not in names:
        return [
            call("update_stock", sku="CB-01", change=5),
            call("update_stock", sku="MN-27", change=40),
        ]
    if "get_item" not in names:
        return [call("get_item", sku="CB-01"), call("get_item", sku="MN-27")]
    return "CB-01: 17 in stock.\nMN-27: 4 in stock; the 40 monitors wait for a manager."


async def test_the_small_change_is_booked_and_the_large_one_waits(tmp_path, capsys):
    ctx = scripted_context(policy, tmp_path)
    outcome = await run(ctx)
    assert outcome.ok, outcome.summary
    assert outcome.details["stock"] == {"CB-01": 17, "MN-27": 4, "KB-10": 9}
    assert outcome.details["decisions"] == [("CB-01", 5, True), ("MN-27", 40, False)]

    printed = capsys.readouterr().out
    assert "list_items: read_only=True, needs approval=False" in printed
    assert "update_stock: read_only=False, needs approval=True" in printed
    assert "approval for update_stock(MN-27, +40): no" in printed

    # What the model read: the server's result for the booking, the policy's refusal for the other.
    last = ctx.meter.requests[-1]
    booked, declined = results_so_far(last)["update_stock"]
    assert booked.content == "CB-01: 12 -> 17 in stock" and not booked.is_error
    assert declined.is_error and declined.content.startswith("'update_stock' was not run: ")
    assert "40 units of MN-27 are over the 10-unit limit and need a manager's" in declined.content
    assert [r.content for r in results_so_far(last)["get_item"]] == [
        "CB-01: USB-C cable, 1 m, 17 in stock",
        "MN-27: 27-inch monitor, 4 in stock",
    ]


async def test_splitting_a_declined_change_is_declined_too(tmp_path):
    def splitter(request):
        steps = [
            call("update_stock", sku="MN-27", change=40),
            call("update_stock", sku="MN-27", change=10),  # within the limit, but MN-27 is held
            call("update_stock", sku="CB-01", change=5),
            call("get_item", sku="CB-01"),
        ]
        done = len(calls_so_far(request))
        return steps[done] if done < len(steps) else "CB-01: 17 in stock. MN-27 waits."

    ctx = scripted_context(splitter, tmp_path)
    outcome = await run(ctx)
    assert outcome.ok, outcome.summary
    assert outcome.details["decisions"] == [
        ("MN-27", 40, False),
        ("MN-27", 10, False),
        ("CB-01", 5, True),
    ]
    over, held, booked = results_so_far(ctx.meter.requests[-1])["update_stock"]
    assert "over the 10-unit limit" in over.content
    assert "MN-27 waits for a manager's approval of an earlier change" in held.content
    assert booked.content == "CB-01: 12 -> 17 in stock"


async def test_an_answer_without_the_new_quantity_is_not_ok(tmp_path):
    def vague(request):
        if "update_stock" not in calls_so_far(request):
            return [
                call("update_stock", sku="CB-01", change=5),
                call("update_stock", sku="MN-27", change=40),
            ]
        return "Done."

    outcome = await run(scripted_context(vague, tmp_path))
    assert not outcome.ok and outcome.details["stock"]["CB-01"] == 17
    assert "the answer does not give CB-01's new quantity" in outcome.summary


def test_the_policy_limits_units_per_item_and_holds_declined_items():
    @tool
    def update_stock(sku: str, change: int) -> str:
        """Change the stock."""
        return "ok"

    assert isinstance(update_stock, Tool)
    stock_policy = StockPolicy(10, log=lambda line: None)

    def ask(**arguments):
        return stock_policy(None, update_stock, call("update_stock", **arguments), None).approved

    assert ask(sku="CB-01", change=6) and ask(sku="CB-01", change=-4)  # 10 units in total
    assert not ask(sku="CB-01", change=1)  # 11 would be over the limit
    assert not ask(sku="KB-10", change=12) and not ask(sku="KB-10", change=1)  # held
    assert not ask(sku="KB-11", change="5")  # not a whole number: declined, but not held
    assert ask(sku="KB-11", change=5)

    @tool(read_only=True)
    def get_item(sku: str) -> str:
        """Show one item."""
        return "ok"

    other = stock_policy(None, get_item, call("get_item", sku="CB-01"), None)
    assert not other.approved and "only stock changes" in other.reason


def test_run_all_skips_the_server_module():
    modules = run_all.discover()
    assert "examples.patterns.mcp_tools" in modules
    assert not any(name.endswith("mcp_server") for name in modules)


def test_runs_as_a_module_and_replays_from_its_cassette(tmp_path, monkeypatch, capsys):
    common = ["--provider", "anthropic", "--cassette-dir", str(tmp_path / "cassettes"), "--json"]
    with pytest.raises(SystemExit) as recorded:
        main(
            run,
            name=example.NAME,
            description=example.DESCRIPTION,
            inspired_by=example.INSPIRED_BY,
            argv=[*common, "--record", "--workdir", str(tmp_path / "recorded")],
            model_factory=lambda provider, model: FunctionModel(policy),
        )
    assert recorded.value.code == 0
    capsys.readouterr()

    argv = [example.__file__, *common, "--replay", "--workdir", str(tmp_path / "replayed")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as replayed:
        runpy.run_path(example.__file__, run_name="__main__")
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert replayed.value.code == 0, report
    assert report["ok"] and report["name"] == example.NAME and report["requests"] == 4
