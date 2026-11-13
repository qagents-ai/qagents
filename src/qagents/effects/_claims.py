"""Atomic claim transitions shared by the built-in stores."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel
from pydantic import JsonValue as JSONValue

from ..errors import ClaimConflict
from ._canonical import LIMIT_BYTES, Descriptor, jcs
from ._kinds import EffectIntentData

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
        )
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
    elif entry.kind == "effect_failed":
        why = data.get("why")
        claim.status = why if why in ("rejected", "not_sent") else "held"
        if existing.status in ("held", "committed"):
            claim.status = "held"  # earlier uncertainty or contradictory terminal evidence
    elif entry.kind in ("effect_settled", "effect_conflict"):
        claim.status = "held"
    elif entry.kind == "effect_observed" and data.get("reason") is not None:
        if existing.status != "committed":
            claim.status = "held"
    else:
        return existing
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
