"""Frozen Effects interfaces; behavior is implemented by the owning stage."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from qagents.session import Entry

from ._canonical import JSONValue


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
    """Pure transition used inside every store's append transaction (7.10 "Writing");
    raises ClaimConflict."""
    raise NotImplementedError("Effects implementation is pending its owning stage.")
