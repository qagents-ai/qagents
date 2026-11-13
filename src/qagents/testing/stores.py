"""Public store-contract report and its staged implementation entry point."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from qagents.session import SessionStore


@dataclass(frozen=True)
class StoreReport:
    ok: bool
    checks: list[tuple[str, bool, str]]
    not_checked: list[str]


async def check_store(
    factory: Callable[[], SessionStore | Awaitable[SessionStore]],
    *,
    kill: Callable[[], Awaitable[None]] | None = None,
) -> StoreReport:
    raise NotImplementedError("The store contract kit is pending its owning stage.")
