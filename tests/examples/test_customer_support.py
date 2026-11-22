"""Offline tests of examples/patterns/customer_support.py."""

from __future__ import annotations

import json
import runpy
import sys

import pytest

from examples.harness import last_text, main, scripted_context, system_text, tool_names
from examples.patterns import customer_support
from examples.patterns.customer_support import MESSAGES, run
from qagents.models.testing import FunctionModel, call


def question(request) -> str:
    return next(m.text for m in reversed(request.messages) if m.role == "user")


def called_since_question(request) -> set[str]:
    """Tools the model has called since the last customer message."""
    names: set[str] = set()
    for message in reversed(request.messages):
        if message.role == "user":
            break
        names.update(call.name for call in message.tool_calls)
    return names


def policy(request):
    """Plays the model for triage, then billing."""
    if "triage agent" in system_text(request):
        assert {"transfer_to_billing", "transfer_to_technical_support", "transfer_to_sales"} <= (
            tool_names(request)
        )
        return call("transfer_to_billing")
    assert "billing specialist" in system_text(request), system_text(request)
    done = called_since_question(request)
    if question(request) == MESSAGES[1]:
        if "refund_status" not in done:
            return call("refund_status", order_id="A-1001")
        return f"It is on its way. {last_text(request)}"
    if "lookup_order" not in done:
        return call("lookup_order", order_id="A-1001")
    if "issue_refund" not in done:
        return call("issue_refund", order_id="A-1001", amount_eur=129.0, reason="cracked case")
    return f"Done. {last_text(request)}"


async def test_triage_hands_off_and_billing_keeps_the_customer(tmp_path, capsys):
    ctx = scripted_context(policy, tmp_path)
    outcome = await run(ctx)
    assert outcome.ok, outcome.summary
    assert outcome.details["agents"] == ["billing", "billing"]
    assert outcome.details["refunds"] == {"A-1001": "RF-A-1001"}
    assert outcome.details["unsigned_approval_refused"] is True
    assert outcome.output == [
        "Done. Refund RF-A-1001 of 129.00 EUR issued for order A-1001.",
        "It is on its way. Refund RF-A-1001 of 129.00 EUR is being processed by the card "
        "network; it reaches the card within 3 to 5 business days.",
    ]
    printed = capsys.readouterr().out
    assert "== handoff triage -> billing" in printed
    assert (
        'issue_refund(order_id="A-1001", amount_eur=129.0' in printed
        and "needs approval" in printed
    )
    assert "without the pause's token was refused" in printed

    # The resume (called on triage) continued with billing, which read the refund's result.
    resumed = next(r for r in ctx.meter.requests if "issued for order" in last_text(r))
    assert "billing specialist" in system_text(resumed)
    assert [c.name for c in resumed.messages[-2].tool_calls] == ["issue_refund"]

    # Turn 2 went to billing directly: no triage request after the first one, and the billing
    # request of turn 2 carried the whole conversation of turn 1.
    triage_requests = [r for r in ctx.meter.requests if "triage agent" in system_text(r)]
    assert len(triage_requests) == 1
    turn_two = next(r for r in ctx.meter.requests if r.messages[-1].text == MESSAGES[1])
    assert "billing specialist" in system_text(turn_two)
    assert any(m.text == MESSAGES[0] for m in turn_two.messages)
    assert outcome.details["turn_two_tools"] == ["refund_status"]


async def test_a_declined_refund_is_not_issued(tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "no")
    outcome = await run(scripted_context(policy, tmp_path, interactive=True))
    assert outcome.ok, outcome.summary
    assert outcome.details["approved"] is False and outcome.details["refunds"] == {}
    assert "declined" in outcome.output[0]


async def test_a_triage_agent_that_answers_itself_is_not_ok(tmp_path):
    def chatty(request):
        if "triage agent" in system_text(request):
            return "I am sorry to hear that. A refund will follow."
        return policy(request)

    outcome = await run(scripted_context(chatty, tmp_path))
    assert not outcome.ok and "turn 1: completed by triage" in outcome.summary


def test_runs_as_a_module_and_replays_from_its_cassette(tmp_path, monkeypatch, capsys):
    common = ["--provider", "anthropic", "--cassette-dir", str(tmp_path / "cassettes"), "--json"]
    with pytest.raises(SystemExit) as recorded:
        main(
            run,
            name=customer_support.NAME,
            description=customer_support.DESCRIPTION,
            inspired_by=customer_support.INSPIRED_BY,
            argv=[*common, "--record", "--workdir", str(tmp_path / "recorded")],
            model_factory=lambda provider, model: FunctionModel(policy),
        )
    assert recorded.value.code == 0
    capsys.readouterr()

    argv = ["customer_support", *common, "--replay", "--workdir", str(tmp_path / "replayed")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as replayed:
        runpy.run_path(customer_support.__file__, run_name="__main__")
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert replayed.value.code == 0, report
    # triage, billing x3 in turn 1 (lookup, refund, answer), billing x2 in turn 2
    assert report["ok"] and report["name"] == "customer_support" and report["requests"] == 6
