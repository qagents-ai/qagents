"""Destination contracts declared from documentation; verify them with sandbox probes."""

from ._declare import Dedupe

STRIPE_IDEMPOTENCY_V1 = Dedupe(
    window_s=24 * 3600,
    contract=(
        "Stripe API v1 idempotency, docs checked 2026-11-12 "
        "(API reference version 2026-10-28.endive)"
    ),
    concurrent="rejects",
    replays_errors=True,
    replays_rejections=True,
    binds_params=True,
)
"""Declared from Stripe's docs; probes pending.

Source: https://docs.stripe.com/api/idempotent_requests . Execution results,
including errors, are retained for at least 24 hours. Changed parameters are
refused, as are concurrent requests using the same key. Execution-time
rejections may be retained; validation refusals and concurrent conflicts are
not saved. The effect engine therefore treats declared rejections conservatively.
"""
