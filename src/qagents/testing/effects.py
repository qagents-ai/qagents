"""Build real effect inputs for focused unit tests, without a model or a session store."""

from __future__ import annotations

import math
import time
from typing import Any

from qagents.effects import _canonical as canonical
from qagents.effects._declare import Effect
from qagents.effects._types import Attempt, AttemptDebug, AttemptInfo, Inquiry, ReqT


def attempt_for(
    effect: Effect[ReqT, Any],
    request: ReqT,
    *,
    number: int = 1,
    generation: int = 0,
    key: str | None = None,
    salt: bytes = b"test-salt",
    timeout_s: float | None = None,
) -> Attempt[ReqT]:
    """Freeze and rebuild the request, using the same identities as a test-session perform."""
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        raise ValueError("Attempt number must be a positive integer.")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise ValueError("Attempt generation must be a nonnegative integer.")
    timeout = effect.timeout_s if timeout_s is None else timeout_s
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Attempt timeout_s must be positive and finite.")
    frozen = canonical.freeze(effect, request)
    restored = canonical.rebuild(effect, frozen)
    rendered = effect.key.render(
        restored, operation_id="test:1:call", effect=effect.name, occurrence=None
    )
    if key is not None:
        if not isinstance(key, str) or not key:
            raise ValueError("Attempt key must be a nonempty string.")
        rendered = key
    # No dependency argument exists in this helper. Callable namespaces can be tested by
    # overriding their dependency at declaration or by using the full runtime.
    namespace = canonical.resolve_namespace(effect, None, "test")
    return Attempt(
        restored,
        rendered,
        canonical.idempotency_key(
            salt, namespace, effect.name, rendered, generation, effect.idempotency_key_max_len
        ),
        canonical.operation_tag(namespace, effect.name, rendered, frozen.digest),
        time.monotonic() + timeout,
        None
        if effect.applies_within_s is None
        else time.time() + timeout + effect.applies_within_s,
        AttemptDebug(number, generation, 1, 1),
    )


def inquiry_for(
    effect: Effect[ReqT, Any],
    request: ReqT,
    *,
    attempts: int = 1,
    first_use_at: float | None = None,
    salt: bytes = b"test-salt",
) -> Inquiry[ReqT]:
    """Build a lookup input with recorded unknown attempts under the original key."""
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError("Inquiry attempts must be a positive integer.")
    attempt = attempt_for(effect, request, salt=salt)
    stamp = time.time() if first_use_at is None else first_use_at
    return Inquiry(
        attempt.request,
        attempt.key,
        attempt.operation_tag,
        (attempt.idempotency_key,),
        tuple(
            AttemptInfo(number, 0, attempt.idempotency_key, 1, stamp, "unknown")
            for number in range(1, attempts + 1)
        ),
        first_use_at,
    )
