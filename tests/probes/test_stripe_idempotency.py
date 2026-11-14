"""Five API-v1 claim probes: deterministic offline checks plus opt-in observations.

No test implies that offline evidence validates Stripe. Live requests are disabled
unless QAGENTS_RUN_STRIPE_PROBES=1 and STRIPE_TEST_SECRET_KEY starts with sk_test_.
All remote operations create sandbox objects. Execution-time 500 replay and full
24-hour retention cannot be asserted from an ordinary short successful refund test;
the live artifact preserves those as unobserved, never passed.
Sources: https://docs.stripe.com/api/idempotent_requests and /api/refunds/create.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path

import httpx
import pytest

from qagents.testing.crash._clock import ManualClock
from qagents.testing.crash._fakes import FakeStripeRefunds

_REQUIRED = frozenset(
    {
        "retention_window",
        "concurrent_rejection",
        "replay_errors",
        "replay_rejections",
        "binds_params",
    }
)


def _write_evidence(directory: Path, evidence: dict, *, effect_name: str = "payments.refund"):
    """Keep observations separate from the frozen effect-name probe assurance mapping."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"stripe-probes-{uuid.uuid4().hex}"
    sidecar = directory / f"{stem}.evidence.json"
    claims = evidence["claims"]
    complete = all(claims.get(name) == "passed" for name in _REQUIRED)
    evidence = {**evidence, "all_required_claims_observed": complete}
    with sidecar.open("x", encoding="utf-8") as output:
        json.dump(evidence, output, indent=2, sort_keys=True)
    mapping_path = None
    if evidence["scope"] == "stripe_sandbox":
        mapping = {}
        if "failed" in claims.values():
            mapping[effect_name] = "failed"
        elif complete:
            mapping[effect_name] = "passed"
        mapping_path = directory / f"{stem}.probes.json"
        with mapping_path.open("x", encoding="utf-8") as output:
            json.dump(mapping, output, indent=2, sort_keys=True)
    return sidecar, mapping_path


@pytest.mark.parametrize("scope", ["offline", "stripe_sandbox"])
def test_probe_export_never_promotes_offline_or_partial_observations(tmp_path, scope):
    evidence = {
        "scope": scope,
        "api_version": "2026-10-28.endive",
        "contract": "Stripe API v1 idempotency",
        "claims": {"binds_params": "passed"},
    }
    sidecar, mapping = _write_evidence(tmp_path, evidence)
    assert not json.loads(sidecar.read_text())["all_required_claims_observed"]
    if scope == "offline":
        assert mapping is None
    else:
        assert json.loads(mapping.read_text()) == {}


def test_probe_export_requires_all_five_observed_live_claims(tmp_path):
    evidence = {
        "scope": "stripe_sandbox",
        "api_version": "2026-10-28.endive",
        "contract": "Stripe API v1 idempotency",
        "claims": dict.fromkeys(_REQUIRED, "passed"),
    }
    _, mapping = _write_evidence(tmp_path, evidence)
    assert json.loads(mapping.read_text()) == {"payments.refund": "passed"}
    evidence["claims"]["replay_errors"] = "failed"
    _, mapping = _write_evidence(tmp_path, evidence)
    assert json.loads(mapping.read_text()) == {"payments.refund": "failed"}


def test_offline_key_retained_until_exact_window_and_replays_body():
    async def run():
        clock = ManualClock()
        client = FakeStripeRefunds(clock=clock)
        first = await client.create_refund(charge="ch_A", amount=10, idempotency_key="retention")
        clock.advance(86400 - 0.001)
        replay = await client.create_refund(charge="ch_A", amount=10, idempotency_key="retention")
        assert replay == first and replay is not first
        clock.advance(0.001)
        fresh = await client.create_refund(charge="ch_A", amount=10, idempotency_key="retention")
        assert (first.id, fresh.id) == ("re_1", "re_2")
        assert len([row async for row in client.iter_refunds(charge="ch_A")]) == 2

    asyncio.run(run())


@pytest.mark.parametrize(
    "error_type", [FakeStripeRefunds.ServerError, FakeStripeRefunds.InvalidRequest]
)
def test_offline_execution_errors_and_refusals_are_replayed(error_type):
    async def run():
        client = FakeStripeRefunds()
        client.inject_execution_error(error_type("controlled execution result"))
        for _ in range(2):
            with pytest.raises(error_type, match="controlled execution result"):
                await client.create_refund(charge="ch_A", amount=10, idempotency_key="execution")
        assert not [row async for row in client.iter_refunds(charge="ch_A")]
        fresh = await client.create_refund(
            charge="ch_A", amount=10, idempotency_key="new-generation"
        )
        assert fresh.id == "re_1"

    asyncio.run(run())


def test_offline_changed_parameters_refused_without_application():
    async def run():
        client = FakeStripeRefunds()
        await client.create_refund(charge="ch_A", amount=10, idempotency_key="parameters")
        with pytest.raises(client.IdempotencyConflict):
            await client.create_refund(charge="ch_A", amount=11, idempotency_key="parameters")
        assert len([row async for row in client.iter_refunds(charge="ch_A")]) == 1

    asyncio.run(run())


def test_offline_validation_and_concurrent_conflicts_are_not_cached():
    async def run():
        client = FakeStripeRefunds()
        with pytest.raises(client.InvalidRequest):
            await client.create_refund(charge="ch_A", amount=-1, idempotency_key="validation")
        valid = await client.create_refund(charge="ch_A", amount=10, idempotency_key="validation")
        assert valid.id == "re_1"
        first = asyncio.create_task(
            client.create_refund(charge="ch_A", amount=10, idempotency_key="concurrent")
        )
        await asyncio.sleep(0)
        with pytest.raises(client.ConcurrentRequest):
            await client.create_refund(charge="ch_A", amount=10, idempotency_key="concurrent")
        actual = await first
        replay = await client.create_refund(charge="ch_A", amount=10, idempotency_key="concurrent")
        assert actual == replay and actual.id == "re_2"

    asyncio.run(run())


def test_offline_key_length_bound_is_255_characters():
    async def run():
        client = FakeStripeRefunds()
        receipt = await client.create_refund(charge="ch_A", amount=10, idempotency_key="k" * 255)
        assert receipt.id == "re_1"
        with pytest.raises(client.InvalidRequest, match="255"):
            await client.create_refund(charge="ch_A", amount=10, idempotency_key="k" * 256)
        assert len([row async for row in client.iter_refunds(charge="ch_A")]) == 1

    asyncio.run(run())


def _live_key() -> str:
    if os.environ.get("QAGENTS_RUN_STRIPE_PROBES") != "1":
        pytest.skip("Stripe sandbox probes need explicit QAGENTS_RUN_STRIPE_PROBES=1")
    key = os.environ.get("STRIPE_TEST_SECRET_KEY", "")
    if not key:
        pytest.skip("STRIPE_TEST_SECRET_KEY is not configured")
    if not key.startswith("sk_test_"):
        pytest.fail("Stripe probes refuse a key that is not a sandbox sk_test_ key", pytrace=False)
    return key


def test_probe_gate_refuses_non_test_credentials(monkeypatch):
    monkeypatch.setenv("QAGENTS_RUN_STRIPE_PROBES", "1")
    monkeypatch.setenv("STRIPE_TEST_SECRET_KEY", "sk_live_not_a_real_secret")
    with pytest.raises(pytest.fail.Exception, match="sandbox"):
        _live_key()


@pytest.mark.network
def test_live_stripe_api_v1_observations(tmp_path):
    """Write observed evidence; missing error/retention evidence remains explicit."""
    key = _live_key()
    directory = Path(os.environ.get("QAGENTS_STRIPE_PROBE_DIR", str(tmp_path)))

    async def run():
        nonce = uuid.uuid4().hex
        evidence = {
            "schema": 1,
            "scope": "stripe_sandbox",
            "api_version": "2026-10-28.endive",
            "contract": "Stripe API v1 idempotency, docs checked 2026-11-13",
            "claims": {},
            "unobserved": ["replay_errors", "replay_rejections", "retention_window"],
            "all_required_claims_observed": False,
        }
        # No credential/headers/request repr is ever written into an artifact.
        async with httpx.AsyncClient(
            base_url="https://api.stripe.com/v1",
            auth=(key, ""),
            timeout=30,
            headers={"Stripe-Version": evidence["api_version"]},
            follow_redirects=False,
            trust_env=False,
        ) as client:

            async def post(endpoint, data, identity):
                response = await client.post(
                    endpoint, data=data, headers={"Idempotency-Key": identity}
                )
                try:
                    body = response.json()
                except ValueError:
                    pytest.fail("Stripe returned a non-JSON response", pytrace=False)
                return response.status_code, body

            status, charge = await post(
                "/charges",
                {"amount": "10000", "currency": "eur", "source": "tok_visa"},
                nonce + "charge",
            )
            if status != 200 or charge.get("livemode") is not False:
                pytest.fail(f"Sandbox charge setup failed with HTTP {status}", pytrace=False)
            request = {"charge": charge["id"], "amount": "10"}
            started = time.monotonic()
            status, first = await post("/refunds", request, nonce + "refund")
            second_status, second = await post("/refunds", request, nonce + "refund")
            evidence["claims"]["same_key_response_replay"] = (
                "passed" if status == second_status == 200 and first == second else "failed"
            )
            evidence["retention_observed_minutes"] = (time.monotonic() - started) / 60
            status, _ = await post("/refunds", {**request, "amount": "11"}, nonce + "refund")
            evidence["claims"]["binds_params"] = "passed" if status == 400 else "failed"
            invalid_status, _ = await post(
                "/refunds", {**request, "amount": "-1"}, nonce + "validation"
            )
            valid_status, _ = await post("/refunds", request, nonce + "validation")
            evidence["claims"]["validation_not_cached"] = (
                "passed" if invalid_status == 400 and valid_status == 200 else "failed"
            )
            long_status, _ = await post("/refunds", request, nonce + "k" * (255 - len(nonce)))
            too_long_status, _ = await post("/refunds", request, nonce + "k" * (256 - len(nonce)))
            evidence["claims"]["key_length_255"] = (
                "passed" if long_status == 200 and too_long_status == 400 else "failed"
            )
            replies = await asyncio.gather(
                *(post("/refunds", request, nonce + "concurrent") for _ in range(8))
            )
            conflicts = [body for status, body in replies if status == 409]
            successes = [body for status, body in replies if status == 200]
            if conflicts and successes:
                final_status, final = await post("/refunds", request, nonce + "concurrent")
                evidence["claims"]["concurrent_rejection"] = (
                    "passed"
                    if final_status == 200 and final["id"] == successes[0]["id"]
                    else "failed"
                )
            else:
                evidence["unobserved"].append("concurrent_rejection")
        path, _ = _write_evidence(directory, evidence)
        assert "failed" not in evidence["claims"].values(), f"Probe failed; evidence: {path}"
        pytest.skip(
            f"Partial sandbox observations saved at {path}; unobserved claims cannot pass the release gate"
        )

    asyncio.run(run())
