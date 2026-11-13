"""Frozen Effects interfaces; behavior is implemented by the owning stage."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from qagents.session import Session

from ._declare import Effect
from ._types import MISSING, EffectRecord, HeldEffect, Settle

if TYPE_CHECKING:
    from qagents.permissions import Permissions


async def settle(
    session: Session,
    answer: Settle,
    *,
    effects: Mapping[str, Effect],
    permissions: Permissions,
    deps: Any = MISSING,
    lease_ttl_s: float,
    worker: str,
) -> EffectRecord:
    """O2, O3, O3r, O4 rows under a new lease epoch (a LeasedWriter, purpose settle) with
    revision compare-and-swap; raises SettlementConflict."""
    raise NotImplementedError("Effects implementation is pending its owning stage.")


async def recheck(
    session: Session,
    *,
    effects: Mapping[str, Effect],
    permissions: Permissions,
    deps: Any = MISSING,
    lease_ttl_s: float,
    worker: str,
) -> list[HeldEffect]:
    """O1 row: lookups only for D, X, NA, CU records under a new epoch (purpose recheck);
    nothing is sent; returns what is still held."""
    raise NotImplementedError("Effects implementation is pending its owning stage.")
