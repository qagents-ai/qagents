"""Atomic claim transitions shared by the built-in stores."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel
from pydantic import JsonValue as JSONValue

from ..errors import ClaimConflict
from ._canonical import LIMIT_BYTES, Descriptor, jcs
from ._kinds import ClaimAttemptData, EffectClaimedData, EffectIntentData

if TYPE_CHECKING:
    from ..session import Entry


class Claim(BaseModel):
    namespace: str
    effect: str
    key: str
    descriptor_digest: str
    session_id: str
    incarnation: str
    record_id: str
    first_use_at: float
    generation: int
    status: Literal["sent", "committed", "rejected", "held", "not_sent"]
    receipt: dict[str, JSONValue] | None = None
    committed_at: float | None = None
    updated_at: float
    # [additive] every attempt of every owner, so a new owner's lookup can ask about them
    attempts: list[ClaimAttemptData] = []


def next_claim(
    existing: Claim | None, entry: Entry, *, session_id: str, incarnation: str, now: float
) -> Claim | None:
    """Derive a claim inside the append transaction, preserving global first use."""
    data = entry.data
    if entry.kind == "effect_intent":
        intent = EffectIntentData.model_validate(data)
        descriptor = Descriptor(
            intent.name, intent.version, intent.destination_digest, intent.request_digest
        ).digest()
        if existing is not None:
            same_owner = (existing.session_id, existing.incarnation, existing.record_id) == (
                session_id,
                incarnation,
                intent.record_id,
            )
            if (
                descriptor != existing.descriptor_digest
                or existing.status == "committed"
                or (not same_owner and existing.status not in ("rejected", "not_sent"))
                or intent.generation < existing.generation
            ):
                _conflict(existing)
        attempt = ClaimAttemptData(
            session_id=session_id,
            number=intent.attempt,
            generation=intent.generation,
            idempotency_key=intent.idempotency_key,
            epoch=intent.epoch,
            store_ts=intent.store_ts,
        )
        return Claim(
            namespace=intent.namespace,
            effect=intent.name,
            key=intent.key,
            descriptor_digest=descriptor,
            session_id=session_id,
            incarnation=incarnation,
            record_id=intent.record_id,
            first_use_at=existing.first_use_at if existing else intent.first_use_at,
            generation=intent.generation,
            status="held" if existing and existing.status in ("sent", "held") else "sent",
            updated_at=now,
            attempts=[*(existing.attempts if existing else []), attempt],
        )
    if entry.kind == "effect_claimed":
        return _transfer(existing, entry, session_id=session_id, incarnation=incarnation, now=now)
    if existing is None:
        return None
    if entry.kind == "effect_committed" and data.get("via") == "claim":
        return existing  # the local replay never acquires the original owner's claim
    if (existing.session_id, existing.incarnation, existing.record_id) != (
        session_id,
        incarnation,
        data["record_id"],
    ):
        _conflict(existing)
    claim = existing.model_copy(deep=True)
    if entry.kind == "effect_committed":
        receipt = data.get("receipt")
        if receipt is not None and len(jcs(receipt)) > LIMIT_BYTES:
            raise ValueError("Claim receipt exceeds the 16 KiB limit.")
        claim.status = "committed"
        claim.receipt = receipt
        claim.committed_at = now
        _record_outcome(claim, session_id, data.get("attempt"), "committed")
    elif entry.kind == "effect_failed":
        why = data.get("why")
        claim.status = why if why in ("rejected", "not_sent") else "held"
        if existing.status in ("held", "committed"):
            claim.status = "held"  # earlier uncertainty or contradictory terminal evidence
        definite = data.get("committed") is False
        _record_outcome(
            claim,
            session_id,
            data.get("attempt"),
            "not_sent" if definite and why == "not_sent" else "rejected" if definite else "unknown",
        )
    elif entry.kind in ("effect_settled", "effect_conflict"):
        claim.status = "held"
    elif entry.kind == "effect_observed" and data.get("reason") is not None:
        if existing.status != "committed":
            claim.status = "held"
    else:
        return existing
    claim.updated_at = now
    return claim


def _record_outcome(claim: Claim, session_id: str, number: object, outcome: str) -> None:
    """Keep each attempt's known outcome, as the fold does; a commit is never overwritten."""
    for attempt in reversed(claim.attempts):
        if attempt.session_id == session_id and attempt.number == number:
            if attempt.outcome != "committed":
                attempt.outcome = outcome
            return


def _transfer(
    existing: Claim | None, entry: Entry, *, session_id: str, incarnation: str, now: float
) -> Claim:
    """Move a rejected claim, unchanged since the new owner read it, to the new owner."""
    data = EffectClaimedData.model_validate(entry.data)
    if existing is None:
        raise ClaimConflict(
            "This effect key has no claim to take over.",
            namespace=data.namespace,
            effect=data.name,
            key=data.key,
            owner_session="",
        )
    descriptor = Descriptor(
        data.name, data.version, data.destination_digest, data.request_digest
    ).digest()
    previous = (existing.session_id, existing.incarnation, existing.record_id)
    named = tuple(
        data.previous_owner.get(name) for name in ("session_id", "incarnation", "record_id")
    )
    if (
        descriptor != existing.descriptor_digest
        or existing.status != "rejected"
        or named != previous
        or previous == (session_id, incarnation, data.record_id)
        or data.generation != existing.generation
        or data.first_use_at != existing.first_use_at
        or [item.model_dump(mode="json") for item in data.attempts]
        != [item.model_dump(mode="json") for item in existing.attempts]
    ):
        _conflict(existing)
    claim = existing.model_copy(deep=True)
    claim.session_id, claim.incarnation, claim.record_id = session_id, incarnation, data.record_id
    claim.updated_at = now
    return claim


def _conflict(claim: Claim) -> None:
    raise ClaimConflict(
        "This effect key is already claimed with incompatible ownership or identity.",
        namespace=claim.namespace,
        effect=claim.effect,
        key=claim.key,
        owner_session=claim.session_id,
    )
