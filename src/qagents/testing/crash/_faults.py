"""Frozen crash-harness interfaces; behavior is pending its owner stage."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import ClassVar


class Faults:
    KINDS: ClassVar[frozenset[str]] = frozenset(
        [
            "kill_before",
            "kill_after",
            "recovery_kill",
            "freeze_after_intent",
            "freeze_before_fence2",
            "freeze_in_send",
            "reject",
            "fail_before_commit",
            "commit_then_lose_response",
            "commit_then_error_response",
            "commit_then_reject_shaped_error",
            "commit_then_crash",
            "commit_late",
            "not_yet_visible_then_commit",
            "unstorable_receipt",
            "duplicate_key",
            "rejected_on",
            "slow_arrival",
            "ack_lost_committed",
            "ack_lost_uncommitted",
            "stall",
            "cancel_during",
            "lease_ack_lost",
            "lease_renew_refused",
            "lease_error",
            "advance_boundaries",
            "store_step",
            "store_step_beyond_sigma",
            "approval_redelivered",
            "stale_answer",
            "double_resume",
            "new_run_on_held",
            "operator_race",
            "fresh_ids",
            "retry_effect",
            "retry_changed",
            "permuted",
            "abandon",
        ]
    )
    DEFAULT_KINDS: ClassVar[frozenset[str]] = KINDS
    QUICK_KINDS: ClassVar[frozenset[str]] = frozenset(
        {"kill_before", "kill_after", "commit_then_lose_response", "retry_effect"}
    )
    ALIASES: ClassVar[Mapping[str, frozenset[str]]] = {
        "zombie": frozenset({"freeze_after_intent", "freeze_before_fence2", "freeze_in_send"})
    }

    @classmethod
    def default(cls) -> Faults:
        raise NotImplementedError("The crash harness is pending its owning stage.")

    @classmethod
    def quick(cls) -> Faults:
        raise NotImplementedError("The crash harness is pending its owning stage.")

    @classmethod
    def parse(cls, spec: str) -> Faults:
        raise NotImplementedError("The crash harness is pending its owning stage.")

    def __init__(
        self,
        *,
        kinds: Iterable[str],
        adapters: Sequence[str] = ("openai_chat", "anthropic"),
        stall_s: float = 3600.0,
    ) -> None:
        raise NotImplementedError("The crash harness is pending its owning stage.")

    kinds: frozenset[str]
