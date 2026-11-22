"""Keep a model inside its provider's rate limits, across every agent that shares it.


    from qagents.models.ratelimit import RateLimitedModel

    model = RateLimitedModel(
        QCloudModel("<model-id>"), max_concurrent=4, requests_per_minute=50
    )
    researcher = Agent(model, name="researcher")
    writer = Agent(model, name="writer")  # the same budget for both

Share one `RateLimitedModel` between the agents, sub-agents and workflow
branches that use the same provider account. Requests wait their turn here
instead of failing with HTTP 429 at the provider.

Three limits, each off unless you set it:

* ``max_concurrent``: requests streaming at the same time;
* ``requests_per_minute``: a token bucket that refills continuously;
* ``tokens_per_minute``: a bucket from which the input and output tokens each
  response reports are taken after it arrives. A request starts when the bucket
  is not in debt, so the limit is kept on average, and a few requests started
  together may overshoot it.

When the provider still answers with a rate-limit error that says when to retry
(``retry_after``), every request through this model waits until then, at most
`MAX_PAUSE_S` seconds.

Waiting is cancellation-safe: a request cancelled while it waits takes nothing
from the buckets, and a request that ends in any way (answer, error, cancel)
gives its concurrency slot back.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import time
import weakref
from collections.abc import AsyncIterator

from ..errors import ConfigurationError, ModelRateLimitError
from ..messages import Usage
from .base import (
    Model,
    ModelCapabilities,
    ModelRequest,
    ModelSettings,
    ModelStreamEvent,
    ResponseDone,
    resolve_model,
)

logger = logging.getLogger("qagents.ratelimit")

MAX_PAUSE_S = 60.0
"""The longest pause a provider's ``retry_after`` can impose on every request."""


class _Bucket:
    """A token bucket that refills continuously and may go into debt."""

    def __init__(self, per_minute: float, burst_s: float) -> None:
        self.per_second = per_minute / 60.0
        self.capacity = max(1.0, self.per_second * burst_s)
        self.level = self.capacity
        self.updated = time.monotonic()

    def _refill(self, now: float) -> None:
        if now > self.updated:
            self.level = min(self.capacity, self.level + (now - self.updated) * self.per_second)
            self.updated = now

    def wait_time(self, amount: float, now: float) -> float:
        """Seconds until the bucket holds ``amount``."""
        self._refill(now)
        missing = amount - self.level
        return 0.0 if missing <= 1e-9 else missing / self.per_second

    def take(self, amount: float, now: float) -> None:
        self._refill(now)
        self.level -= amount


class RateLimitedModel(Model):
    """Passes requests to ``inner`` no faster than the limits allow.

    Args:
        inner: The model to protect. ``"provider:model-id"`` strings are resolved.
        max_concurrent: Requests that may stream at the same time.
        requests_per_minute: Requests that may start per minute, on average.
        tokens_per_minute: Input plus output tokens per minute, on average, as
            reported in each response's usage.
        burst_s: How many seconds of budget each bucket holds when full. The
            default of 60 lets a whole minute's budget go out at once, which is
            how providers that use token buckets count; lower it to spread
            requests evenly.
        max_wait_s: Raise `ModelRateLimitError` instead of waiting longer than
            this for a turn; the wait for a concurrency slot and the wait for
            the buckets share this one budget. The request is not sent. Inside
            a `FallbackModel` this moves the request to the next model; the
            agent loop retries it after ``retry_after`` otherwise. ``None``
            waits as long as needed.

    Raises:
        ConfigurationError: A limit is zero or negative.

    The waiting primitives belong to the event loop that is running; use one
    rate-limited model from one event loop (one thread) at a time.
    """

    def __init__(
        self,
        inner: Model | str,
        *,
        max_concurrent: int | None = None,
        requests_per_minute: float | None = None,
        tokens_per_minute: float | None = None,
        burst_s: float = 60.0,
        max_wait_s: float | None = None,
    ) -> None:
        if max_concurrent is not None and (
            isinstance(max_concurrent, bool) or not isinstance(max_concurrent, int)
        ):
            raise ConfigurationError(f"max_concurrent must be an integer, not {max_concurrent!r}.")
        for label, value in (
            ("max_concurrent", max_concurrent),
            ("requests_per_minute", requests_per_minute),
            ("tokens_per_minute", tokens_per_minute),
            ("burst_s", burst_s),
        ):
            if value is not None and not (math.isfinite(value) and value > 0):
                raise ConfigurationError(
                    f"{label} must be a positive number, not {value!r}. "
                    "Pass None to switch a limit off."
                )
        if max_wait_s is not None and not (math.isfinite(max_wait_s) and max_wait_s >= 0):
            raise ConfigurationError(f"max_wait_s must be zero or more, not {max_wait_s!r}.")
        self.inner = resolve_model(inner)
        """The model that answers."""
        self.name = self.inner.name
        self.provider = self.inner.provider
        self.max_concurrent = max_concurrent
        """Requests that may stream at the same time; ``None`` for no limit."""
        self.requests_per_minute = requests_per_minute
        """Average requests per minute; ``None`` for no limit."""
        self.tokens_per_minute = tokens_per_minute
        """Average input plus output tokens per minute; ``None`` for no limit."""
        self.max_wait_s = max_wait_s
        """The longest a request waits for its turn before failing; ``None`` waits."""
        self._requests = _Bucket(requests_per_minute, burst_s) if requests_per_minute else None
        self._tokens = _Bucket(tokens_per_minute, burst_s) if tokens_per_minute else None
        self._paused_until = 0.0
        self._loop: weakref.ref[asyncio.AbstractEventLoop] | None = None
        self._slots: asyncio.Semaphore | None = None
        self._turnstile: asyncio.Lock | None = None

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        slots, turnstile = self._primitives()
        # One budget for the whole wait: for a slot, then for the buckets.
        deadline = None if self.max_wait_s is None else time.monotonic() + self.max_wait_s
        if slots is not None:
            await self._acquire_slot(slots, deadline)
        try:
            await self._admit(turnstile, deadline)
            async with contextlib.aclosing(self.inner.stream(request)) as events:
                async for event in events:
                    if isinstance(event, ResponseDone):
                        self._account(event.response.usage)
                    yield event
        except ModelRateLimitError as error:
            if error.retry_after and error.code != "local_rate_limit":
                self._pause(error.retry_after)
            raise
        finally:
            if slots is not None:
                slots.release()

    async def capabilities(self) -> ModelCapabilities:
        return await self.inner.capabilities()

    def validate_settings(self, settings: ModelSettings) -> None:
        self.inner.validate_settings(settings)

    async def aclose(self) -> None:
        await self.inner.aclose()

    # -- waiting ----------------------------------------------------------------------

    def _primitives(self) -> tuple[asyncio.Semaphore | None, asyncio.Lock]:
        """The semaphore and the queue lock of the running event loop."""
        loop = asyncio.get_running_loop()
        if self._loop is None or self._loop() is not loop or self._turnstile is None:
            self._loop = weakref.ref(loop)
            self._slots = asyncio.Semaphore(self.max_concurrent) if self.max_concurrent else None
            self._turnstile = asyncio.Lock()
        return self._slots, self._turnstile

    async def _acquire_slot(self, slots: asyncio.Semaphore, deadline: float | None) -> None:
        if deadline is None:
            await slots.acquire()
            return
        try:
            async with asyncio.timeout(max(deadline - time.monotonic(), 0.0)):
                await slots.acquire()
        except TimeoutError:
            raise self._too_long(
                f"all {self.max_concurrent} concurrent request slot(s) stayed busy", None
            ) from None

    async def _admit(self, turnstile: asyncio.Lock, deadline: float | None) -> None:
        """Wait until the buckets allow one more request, then take it from them."""
        start = time.monotonic()
        if self._requests is None and self._tokens is None and start >= self._paused_until:
            return
        # One request waits at a time, so requests are admitted in arrival order.
        try:
            if deadline is None:
                await turnstile.acquire()
            else:
                async with asyncio.timeout(max(deadline - time.monotonic(), 0.0)):
                    await turnstile.acquire()
        except TimeoutError:
            wait = self._wait_time(time.monotonic())
            raise self._too_long("earlier requests are still waiting", wait) from None
        try:
            while True:
                now = time.monotonic()
                wait = self._wait_time(now)
                if wait <= 0:
                    if self._requests is not None:
                        self._requests.take(1.0, now)
                    return
                if deadline is not None and now + wait > deadline:
                    raise self._too_long("the rate limit is used up", wait)
                logger.debug("Waiting %.3f s for a rate-limit turn on %s.", wait, self._label)
                await asyncio.sleep(wait)
        finally:
            turnstile.release()

    def _wait_time(self, now: float) -> float:
        wait = max(self._paused_until - now, 0.0)
        if self._requests is not None:
            wait = max(wait, self._requests.wait_time(1.0, now))
        if self._tokens is not None:
            wait = max(wait, self._tokens.wait_time(0.0, now))  # start when not in debt
        return wait

    def _account(self, usage: Usage) -> None:
        if self._tokens is not None:
            self._tokens.take(usage.input_tokens + usage.output_tokens, time.monotonic())

    def _pause(self, seconds: float) -> None:
        until = time.monotonic() + min(seconds, MAX_PAUSE_S)
        if until > self._paused_until:
            self._paused_until = until
            logger.warning(
                "%s reported a rate limit; holding requests for %.1f s.", self._label, seconds
            )

    def _too_long(self, why: str, wait: float | None) -> ModelRateLimitError:
        return ModelRateLimitError(
            f"The request to {self._label} was not sent: {why}, and waiting for a turn would "
            f"take longer than max_wait_s={self.max_wait_s:g}. Raise max_wait_s, lower the "
            "load, or put a FallbackModel around this model.",
            provider=self.provider,
            model=self.name,
            code="local_rate_limit",
            retry_after=wait,
            retryable=True,
        )

    @property
    def _label(self) -> str:
        return f"{self.inner.provider}:{self.inner.name}"

    def __repr__(self) -> str:
        return (
            f"RateLimitedModel({self._label}, max_concurrent={self.max_concurrent}, "
            f"requests_per_minute={self.requests_per_minute}, "
            f"tokens_per_minute={self.tokens_per_minute})"
        )
