"""Five API-v1 claim probes: deterministic offline checks plus opt-in observations.

No test implies that offline evidence validates Stripe. Live requests are disabled
unless QAGENTS_RUN_STRIPE_PROBES=1 and STRIPE_TEST_SECRET_KEY starts with sk_test_.
All remote operations create sandbox objects. Retention measures a configurable
interval strictly inside the documented 24-hour window. Execution-time 500 replay
remains unobserved unless the sandbox returns and replays a real HTTP 500.
Sources: https://docs.stripe.com/api/idempotent_requests,
https://docs.stripe.com/api/refunds/create, and https://docs.stripe.com/error-low-level.
"""

from __future__ import annotations

import asyncio
import json
import math
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


async def _post(client, endpoint, data, key):
    response = await client.post(endpoint, data=data, headers={"Idempotency-Key": key})
    try:
        body = response.json()
    except ValueError:
        pytest.fail("Stripe probe returned a non-JSON response", pytrace=False)
    replayed = response.headers.get("Idempotent-Replayed", "").lower() == "true"
    return response.status_code, body, replayed


def _observe_replay(first, second, *, status, code=None):
    if first[0] != status:
        return None
    if code is not None and first[1].get("error", {}).get("code") != code:
        return None
    return "passed" if first[:2] == second[:2] and second[2] else "failed"


@pytest.mark.parametrize("status,code", [(400, "charge_already_refunded"), (500, None)])
def test_identical_errors_require_stripe_replay_confirmation(status, code):
    body = {"error": {"code": code}}
    assert (
        _observe_replay((status, body, False), (status, body, True), status=status, code=code)
        == "passed"
    )
    assert (
        _observe_replay((status, body, False), (status, body, False), status=status, code=code)
        == "failed"
    )
    assert _observe_replay((429, body, False), (429, body, True), status=status, code=code) is None


@pytest.mark.network
def test_live_stripe_api_v1_observations(tmp_path):
    """Write replay evidence, including the length of the observed retention window."""
    key = _live_key()
    directory = Path(os.environ.get("QAGENTS_STRIPE_PROBE_DIR", str(tmp_path)))
    try:
        retention_minutes = float(os.environ.get("QAGENTS_STRIPE_RETENTION_MINUTES", "1"))
    except ValueError:
        retention_minutes = math.nan
    if not 1 <= retention_minutes <= 1439:
        pytest.fail("Stripe retention delay must be 1 to 1439 minutes", pytrace=False)

    async def run():
        nonce = uuid.uuid4().hex
        evidence = {
            "schema": 1,
            "scope": "stripe_sandbox",
            "api_version": "2026-10-28.endive",
            "contract": "Stripe API v1 idempotency, docs checked 2026-11-14",
            "claims": {},
        }
        claims = evidence["claims"]
        # No credential, header dictionary, or response body is saved to an artifact.
        async with httpx.AsyncClient(
            base_url="https://api.stripe.com/v1",
            auth=(key, ""),
            timeout=30,
            headers={"Stripe-Version": evidence["api_version"]},
            follow_redirects=False,
            trust_env=False,
        ) as client:

            async def post(endpoint, data, identity):
                return await _post(client, endpoint, data, nonce + identity)

            async def new_charge(identity):
                status, charge, _ = await post(
                    "/charges",
                    {"amount": "10000", "currency": "eur", "source": "tok_visa"},
                    identity,
                )
                if status != 200 or charge.get("livemode") is not False:
                    pytest.fail(f"Sandbox charge setup returned HTTP {status}", pytrace=False)
                return charge

            charge = await new_charge("charge")
            request = {"charge": charge["id"], "amount": "10"}
            refund_key = "refund"
            first = await post("/refunds", request, refund_key)
            started = time.monotonic()
            second = await post("/refunds", request, refund_key)
            error_replay = _observe_replay(first, second, status=500)
            if error_replay is not None:
                claims["replay_errors"] = error_replay
                charge = await new_charge("independent-charge")
                request = {"charge": charge["id"], "amount": "10"}
                refund_key = "independent-refund"
                first = await post("/refunds", request, refund_key)
                started = time.monotonic()
                second = await post("/refunds", request, refund_key)
            claims["same_key_response_replay"] = (
                _observe_replay(first, second, status=200) or "failed"
            )

            status, body, _ = await post("/refunds", {**request, "amount": "11"}, refund_key)
            claims["binds_params"] = (
                "passed"
                if status == 400 and body.get("error", {}).get("type") == "idempotency_error"
                else "failed"
            )
            invalid_status, _, _ = await post("/refunds", {**request, "amount": "-1"}, "validation")
            valid_status, _, _ = await post("/refunds", request, "validation")
            claims["validation_not_cached"] = (
                "passed" if invalid_status == 400 and valid_status == 200 else "failed"
            )
            long_status, _, _ = await post("/refunds", request, "k" * (255 - len(nonce)))
            too_long_status, _, _ = await post("/refunds", request, "k" * (256 - len(nonce)))
            claims["key_length_255"] = (
                "passed" if long_status == 200 and too_long_status == 400 else "failed"
            )
            replies = await asyncio.gather(
                *(post("/refunds", request, "concurrent") for _ in range(8))
            )
            conflicts = [
                body
                for status, body, _ in replies
                if status == 409 and body.get("error", {}).get("code") == "idempotency_key_in_use"
            ]
            successes = [body for status, body, _ in replies if status == 200]
            if conflicts and successes:
                final = await post("/refunds", request, "concurrent")
                claims["concurrent_rejection"] = (
                    "passed"
                    if final[0] == 200
                    and final[2]
                    and all(body["id"] == final[1]["id"] for body in successes)
                    else "failed"
                )

            if claims["same_key_response_replay"] == "passed":
                await asyncio.sleep(max(0, retention_minutes * 60 - (time.monotonic() - started)))
                elapsed = time.monotonic() - started
                evidence["retention_observed_minutes"] = elapsed / 60
                if elapsed < 23 * 3600 + 59 * 60 + 30:
                    delayed = await post("/refunds", request, refund_key)
                    claims["retention_window"] = (
                        _observe_replay(first, delayed, status=200) or "failed"
                    )

            # Exhaust the charge to exercise a refusal inside the actual refund operation.
            status, _, _ = await post("/refunds", {"charge": charge["id"]}, "refund-remainder")
            if status == 200:
                rejected = await post("/refunds", request, "rejection")
                replay = await post("/refunds", request, "rejection")
                observed = _observe_replay(
                    rejected, replay, status=400, code="charge_already_refunded"
                )
                if observed is not None:
                    claims["replay_rejections"] = observed

        evidence["unobserved"] = sorted(_REQUIRED - claims.keys())
        path, _ = _write_evidence(directory, evidence)
        assert "failed" not in claims.values(), f"Probe failed; evidence: {path}"
        if evidence["unobserved"]:
            pytest.skip(
                f"Partial sandbox observations at {path}; missing "
                + ", ".join(evidence["unobserved"])
            )

    asyncio.run(run())
