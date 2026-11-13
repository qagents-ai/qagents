"""Public helper values can be passed to actual declared send and lookup functions."""

import time

import pytest
from pydantic import BaseModel

from qagents import ApprovalConflictError, HeldEffectsError, SettlementConflict
from qagents.effects import (
    Attempt,
    EffectContext,
    Found,
    Inquiry,
    Key,
    effect,
)
from qagents.testing.effects import attempt_for, inquiry_for


class Request(BaseModel):
    request_id: str
    values: list[int]


class Receipt(BaseModel):
    id: str


@effect(name="refund", key=Key("refund", "request_id"), applies_within_s=2)
async def refund(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
    assert attempt.idempotency_key.startswith("q1_")
    return Receipt(id=attempt.request.request_id)


@refund.lookup
async def lookup(ctx: EffectContext, inquiry: Inquiry[Request]) -> Found[Receipt]:
    return Found(Receipt(id=inquiry.request.request_id), tag=inquiry.operation_tag)


@pytest.mark.asyncio
async def test_attempt_for_inquiry_for_work_with_declared_functions():
    request = Request(request_id="R1", values=[1])
    before = time.monotonic()
    attempt = attempt_for(refund, request, number=2, generation=3, timeout_s=5)
    assert before + 5 <= attempt.deadline <= time.monotonic() + 5
    assert attempt.not_after is not None
    assert (attempt.debug.number, attempt.debug.generation) == (2, 3)
    assert attempt.key == "refund:R1"
    request.values.append(2)
    assert attempt.request.values == [1]
    assert (await refund.send(EffectContext(), attempt)).id == "R1"
    inquiry = inquiry_for(refund, request, attempts=2, first_use_at=100)
    assert inquiry.first_use_at == 100
    assert [old.number for old in inquiry.attempts] == [1, 2]
    assert inquiry.idempotency_keys == (inquiry.attempts[0].idempotency_key,)
    assert inquiry.attempts[0].idempotency_key != attempt.idempotency_key
    found = await refund.lookup_fn(EffectContext(), inquiry)
    assert found.tag == inquiry.operation_tag
    assert found.tag != attempt.operation_tag
    assert found.receipt.id == "R1"


def test_helper_key_override_changes_derived_key_but_not_request():
    request = Request(request_id="R1", values=[])
    normal = attempt_for(refund, request)
    overridden = attempt_for(refund, request, key="manual")
    assert overridden.key == "manual"
    assert overridden.request == normal.request
    assert overridden.idempotency_key != normal.idempotency_key
    assert overridden.operation_tag != normal.operation_tag


@pytest.mark.parametrize(
    "arguments", [{"number": 0}, {"generation": -1}, {"timeout_s": float("inf")}, {"key": ""}]
)
def test_invalid_helper_parameters_fail(arguments):
    with pytest.raises(ValueError):
        attempt_for(refund, Request(request_id="R1", values=[]), **arguments)


def test_errors_reexported_from_both_public_modules():
    import qagents.effects as effects

    assert effects.HeldEffectsError is HeldEffectsError
    assert effects.SettlementConflict is SettlementConflict
    assert effects.ApprovalConflictError is ApprovalConflictError
