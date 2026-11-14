"""Run refund contract and Effects recovery probes through real loopback HTTP."""

import asyncio
import json
import os
from pathlib import Path

import httpx
from pydantic import BaseModel

from qagents import Agent, RunContext, Session, SQLiteStore, tool
from qagents.effects import Attempt, EffectContext, Found, Key, Unknown, effect
from qagents.effects.contracts import STRIPE_IDEMPOTENCY_V1
from qagents.models.testing import FunctionModel, call

from ._stripe_http_sandbox import WINDOW_S, stripe_http_sandbox
from .test_stripe_idempotency import _observe_replay, _post, _write_evidence

REFUND = {"charge": "ch_fixture", "amount": "10"}


async def test_local_stripe_http_contract(tmp_path):
    with stripe_http_sandbox() as (base_url, sandbox):
        # No credentials, environment proxies or redirects are needed on loopback.
        async with httpx.AsyncClient(
            base_url=base_url, trust_env=False, follow_redirects=False, timeout=5
        ) as client:

            async def refund(key, *, amount="10"):
                return await _post(client, "/refunds", {**REFUND, "amount": amount}, key)

            first = await refund("retention")
            with sandbox.lock:
                sandbox.clock = WINDOW_S - 0.001
            retained = await refund("retention")
            assert _observe_replay(first, retained, status=200) == "passed"
            mismatch = await refund("retention", amount="11")
            assert mismatch[0] == 400 and mismatch[1]["error"]["type"] == "idempotency_error"
            assert len(sandbox.refunds) == 1
            with sandbox.lock:
                sandbox.clock = WINDOW_S
            fresh = await refund("retention")
            assert fresh[0] == 200 and not fresh[2] and fresh[1]["id"] != retained[1]["id"]

            sandbox.delay_key = "concurrent"
            async with asyncio.TaskGroup() as group:
                pending = group.create_task(refund("concurrent"))
                try:
                    assert await asyncio.to_thread(sandbox.entered.wait, 3)
                    conflict = await refund("concurrent")
                    assert conflict[0] == 409 and not conflict[2]
                    assert conflict[1]["error"]["code"] == "idempotency_key_in_use"
                finally:
                    sandbox.release.set()
            assert (
                _observe_replay(pending.result(), await refund("concurrent"), status=200)
                == "passed"
            )

            sandbox.error_key = "server-error"
            sandbox.refusal_key = "refusal"
            for key, status, code in [
                ("server-error", 500, "execution_error"),
                ("refusal", 400, "charge_already_refunded"),
            ]:
                response = await refund(key)
                assert not response[2]
                assert (
                    _observe_replay(response, await refund(key), status=status, code=code)
                    == "passed"
                )

            invalid = await refund("validation", amount="-1")
            assert invalid[0] == 400 and not invalid[2]
            corrected = await refund("validation")
            assert corrected[0] == 200 and not corrected[2]
            assert (await refund("k" * 255))[0] == 200
            assert (await refund("k" * 256))[0] == 400
            assert len(sandbox.refunds) == 5

    sidecar, mapping = _write_evidence(
        Path(os.environ.get("QAGENTS_LOCAL_STRIPE_PROBE_DIR", str(tmp_path))),
        {
            "schema": 1,
            "scope": "local_http_simulation",
            "contract": STRIPE_IDEMPOTENCY_V1.contract,
            "claims": dict.fromkeys(
                (
                    "retention_window",
                    "binds_params",
                    "concurrent_rejection",
                    "replay_errors",
                    "replay_rejections",
                    "validation_not_cached",
                    "key_length_255",
                ),
                "passed",
            ),
            "simulated_retention_seconds": WINDOW_S,
        },
    )
    assert json.loads(sidecar.read_text())["all_required_claims_observed"]
    assert mapping is None


class RefundRequest(BaseModel):
    id: str
    amount: int


class RefundReceipt(BaseModel):
    id: str
    amount: int


async def test_effect_recovers_a_lost_refund_response_over_http(tmp_path):
    with stripe_http_sandbox() as (base_url, sandbox):
        sandbox.disconnect_once = True
        sandbox.hide_next_lookup = True
        async with httpx.AsyncClient(
            base_url=base_url, trust_env=False, follow_redirects=False, timeout=2
        ) as client:

            @effect(
                "payments.refund",
                key=Key("request", "id"),
                dedupe=STRIPE_IDEMPOTENCY_V1,
                timeout_s=5,
            )
            async def send(ctx: EffectContext, attempt: Attempt[RefundRequest]) -> RefundReceipt:
                response = await ctx.deps.post(
                    "/refunds",
                    data={
                        **REFUND,
                        "amount": str(attempt.request.amount),
                        "metadata[operation_tag]": attempt.operation_tag,
                    },
                    headers={"Idempotency-Key": attempt.idempotency_key},
                )
                response.raise_for_status()
                return RefundReceipt.model_validate(response.json())

            @send.lookup
            async def lookup(ctx, inquiry):
                response = await ctx.deps.get("/refunds", params={"charge": REFUND["charge"]})
                response.raise_for_status()
                for row in response.json()["data"]:
                    if row.get("metadata", {}).get("operation_tag") == inquiry.operation_tag:
                        return Found(RefundReceipt.model_validate(row), tag=inquiry.operation_tag)
                return Unknown()

            @tool(effects=[send])
            async def refund(ctx: RunContext, id: str, amount: int) -> str:
                """Refund the synthetic payment."""
                return (await ctx.perform(send, RefundRequest(id=id, amount=amount))).id

            agent = Agent(
                FunctionModel(
                    lambda request: (
                        "Refund recorded."
                        if request.messages[-1].role == "tool"
                        else call("refund", id="order-A", amount=10)
                    )
                ),
                tools=[refund],
            )
            path = tmp_path / "refunds.sqlite"
            async with SQLiteStore(path) as store:
                session = Session(store, id="refund")
                paused = await agent.run("Refund order A", session=session, deps=client)
                assert paused.status == "paused" and len(paused.held_effects) == 1
                assert not any(item.kind == "effect_committed" for item in await session.entries())

            async with SQLiteStore(path) as store:
                session = Session(store, id="refund")
                assert not await agent.recheck(session, deps=client)
                result = await agent.resume(session, deps=client)
                entries = await session.entries()

    assert result.text == "Refund recorded." and not result.held_effects
    (intent,) = [item for item in entries if item.kind == "effect_intent"]
    assert intent.data["rule"] == "D1"
    assert sandbox.attempts == {intent.data["idempotency_key"]: 1}
    assert len(sandbox.refunds) == 1
    (committed,) = [item for item in entries if item.kind == "effect_committed"]
    assert committed.data["via"] == "lookup"
    assert committed.data["receipt"] == {"id": "re_1", "amount": 10}
