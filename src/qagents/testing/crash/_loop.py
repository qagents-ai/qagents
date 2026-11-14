"""A verified asyncio selector boundary that advances only idle virtual time."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import contextvars
import functools
import inspect
import sys
import threading
import time
from collections import deque
from typing import Any

from qagents import tools
from qagents.errors import ConfigurationError

from ._clock import ManualClock, _finite

WORLD_THREAD: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "qagents_crash_world_thread", default=None
)


class NoTerminalState(RuntimeError):
    """The point has no bounded path to a terminal state under its current schedule."""


def _check_shape(loop: asyncio.BaseEventLoop) -> None:
    """Verify the private CPython hooks used by the selector proxy before installing it."""
    reasons = []
    if sys.implementation.name != "cpython":
        reasons.append("requires CPython")
    selector = getattr(loop, "_selector", None)
    if selector is None or not callable(getattr(selector, "select", None)):
        reasons.append("missing selector.select")
    if not isinstance(getattr(loop, "_ready", None), deque):
        reasons.append("ready callbacks are not a deque")
    if not isinstance(getattr(loop, "_scheduled", None), list):
        reasons.append("scheduled callbacks are not a list")
    try:
        implementation = inspect.getsource(asyncio.BaseEventLoop._run_once)
        for fragment in ("self._selector.select(timeout)", "self._ready", "self._scheduled"):
            if fragment not in implementation:
                reasons.append(f"unverified _run_once boundary: {fragment}")
    except (OSError, TypeError):
        reasons.append("cannot inspect _run_once")
    if reasons:
        raise ConfigurationError(
            "crash_test needs the virtual-time loop; this Python's asyncio internals differ: "
            + "; ".join(reasons)
        )


class _SelectorProxy:
    def __init__(self, loop: _VirtualTime, inner: Any) -> None:
        self.loop = loop
        self.inner = inner
        self.empty_polls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def select(self, timeout: float | None = None) -> Any:
        loop = self.loop
        loop._sync_real_time()
        loop._check_bounds()
        events = self.inner.select(0)
        if events or loop._ready:
            self.empty_polls = 0
            return events
        if timeout == 0:
            return []
        if loop.real_io_active:
            self.empty_polls = 0
            return self.inner.select(0.05 if timeout is None else min(timeout, 0.05))
        if loop.busy_threads:
            self.empty_polls = 0
            if loop._thread_bound_exceeded():
                if timeout is None:
                    raise NoTerminalState("thread exceeded real-time bound without a timer")
                loop.clock.advance(timeout)
                return []
            return self.inner.select(0.05 if timeout is None else min(timeout, 0.05))
        if timeout is None:
            events = self.inner.select(loop.deadlock_timeout_s)
            if events or loop._ready:
                self.empty_polls = 0
                return events
            raise NoTerminalState("no runnable work or timer")
        self.empty_polls += 1
        if self.empty_polls >= 2:
            loop.clock.advance(timeout)
            self.empty_polls = 0
            loop._check_bounds()
        return []


class _VirtualTime:
    """Shared virtual-time policy for native Proactor and Selector loop implementations."""

    def __init__(
        self,
        clock: ManualClock | None = None,
        *,
        real_timeout_s: float = 120.0,
        deadlock_timeout_s: float = 0.2,
        thread_timeout_s: float = 10.0,
    ) -> None:
        self.clock = clock if clock is not None else ManualClock()
        self.real_timeout_s = _finite(real_timeout_s, "real_timeout_s")
        self.deadlock_timeout_s = _finite(deadlock_timeout_s, "deadlock_timeout_s")
        self.thread_timeout_s = _finite(thread_timeout_s, "thread_timeout_s")
        if min(self.real_timeout_s, self.deadlock_timeout_s, self.thread_timeout_s) <= 0:
            raise ValueError("loop real-time bounds must be positive")
        self._started_at: float | None = None
        self._tearing_down = False
        self._leg_progress: dict[int, tuple[float, float]] = {}
        self._thread_lock = threading.Lock()
        self._threads: set[concurrent.futures.Future[Any]] = set()
        self._threads_changed = asyncio.Event()
        self._owned_threads: dict[int, int] = {}
        self._resting_threads: dict[int, str] = {}
        self._thread_progress = time.monotonic()
        self.threads_ran_in_real_time = False
        self.real_io_used = False
        self._io_pending = 0
        self._io_handles: list[tuple[str, Any]] = []
        self._io_last_tick: float | None = None
        self.inconclusive_reasons: list[str] = []
        self.thread_started = None
        super().__init__()
        try:
            _check_shape(self)
        except BaseException:
            super().close()
            raise
        self._selector = _SelectorProxy(self, self._selector)

    def time(self) -> float:
        return self.clock.monotonic()

    def run_forever(self) -> None:
        if self._started_at is None:
            self._started_at = time.monotonic()
        super().run_forever()

    @property
    def pending_threads(self) -> tuple[concurrent.futures.Future[Any], ...]:
        with self._thread_lock:
            return tuple(self._threads)

    @property
    def busy_threads(self) -> int:
        with self._thread_lock:
            running = sum(future.running() for future in self._threads)
            delivered_later = sum(future.done() for future in self._threads)
            queued = len(self._threads) - running - delivered_later
            # Queued executor calls cannot run while all workers await the world.
            # A newly submitted call with no running worker still needs a real poll.
            return (
                max(0, running - len(self._resting_threads))
                + delivered_later
                + int(running == 0 and queued > 0)
            )

    def _track_thread(self, future: concurrent.futures.Future[Any]) -> bool:
        with self._thread_lock:
            if future in self._threads:
                return False
            self._threads.add(future)
            self._thread_progress = time.monotonic()
            self.threads_ran_in_real_time = True
        if self.thread_started is not None:
            self.thread_started()
        return True

    def _thread_ended(self, future: concurrent.futures.Future[Any]) -> None:
        with self._thread_lock:
            self._threads.discard(future)
            self._thread_progress = time.monotonic()
        if not self.is_closed():
            self._threads_changed.set()

    async def threads_drained(self) -> None:
        """Wait for completion accounting, including callbacks posted after result delivery."""
        while self.pending_threads:
            self._threads_changed.clear()
            if self.pending_threads:
                await self._threads_changed.wait()

    def observe_thread(self, future: concurrent.futures.Future[Any]) -> None:
        """Observe qagents' concurrent future after its asyncio wrapper exists."""
        if self._track_thread(future):
            future.add_done_callback(
                lambda done: self._deliver_thread_callback(self._thread_ended, done)
            )

    @contextlib.contextmanager
    def observe_tool_threads(self) -> Any:
        """Bind the real qagents thread producer for this sequential point only."""
        previous = tools.set_thread_observer(self.observe_thread)
        try:
            yield
        finally:
            tools.set_thread_observer(previous)

    def _deliver_thread_callback(self, callback: Any, future: Any) -> None:
        try:
            self.call_soon_threadsafe(callback, future)
        except RuntimeError:
            if not self.is_closed():
                raise
            # An inconclusive point may close after its bounded join expired.
            # The real worker still ends independently; never touch a closed loop.
            self._thread_ended(future)

    def thread_state(self, ident: int, state: str) -> None:
        """Move an owned worker between computing, waiting on the world and parking."""
        if state not in {"running", "waiting", "parked"}:
            raise ValueError("unknown thread state")
        with self._thread_lock:
            if not self._owns_thread(ident):
                raise RuntimeError("world call from an unobserved thread")
            if state == "running":
                self._resting_threads.pop(ident, None)
            else:
                self._resting_threads[ident] = state
            self._thread_progress = time.monotonic()
        self.call_soon_threadsafe(lambda: None)

    def _owns_thread(self, ident: int) -> bool:
        # tools.start_thread publishes this private identity before its user body;
        # only Futures observed by this point may confer ownership.
        return ident in self._owned_threads or any(
            future.running() and getattr(future, "_qagents_thread_id", None) == ident
            for future in self._threads
        )

    def _thread_bound_exceeded(self) -> bool:
        if self._tearing_down:
            return False
        with self._thread_lock:
            exceeded = time.monotonic() - self._thread_progress > self.thread_timeout_s
        if exceeded and "thread exceeded real-time bound" not in self.inconclusive_reasons:
            self.inconclusive_reasons.append("thread exceeded real-time bound")
        return exceeded

    @property
    def real_io_active(self) -> bool:
        def active(kind: str, handle: Any) -> bool:
            if kind == "socket":
                return handle.fileno() >= 0
            if kind == "process":
                return handle.get_returncode() is None or any(
                    pipe is not None and not pipe.is_closing()
                    for pipe in (handle.get_pipe_transport(fd) for fd in (0, 1, 2))
                )
            return not handle.is_closing()

        self._io_handles[:] = [
            (kind, handle) for kind, handle in self._io_handles if active(kind, handle)
        ]
        return bool(self._io_pending or self._io_handles)

    def _sync_real_time(self) -> None:
        now = time.monotonic()
        if self.real_io_active:
            if self._io_last_tick is not None:
                self.clock.advance(max(0, now - self._io_last_tick))
            self._io_last_tick = now
        else:
            self._io_last_tick = None

    def _io_started(self) -> None:
        self.real_io_used = True
        self._io_pending += 1
        if self._io_last_tick is None:
            self._io_last_tick = time.monotonic()

    def _io_ended(self) -> None:
        self._sync_real_time()
        self._io_pending -= 1

    async def create_connection(self, *args: Any, **kwargs: Any) -> Any:
        self._io_started()
        try:
            transport, protocol = await super().create_connection(*args, **kwargs)
            self._io_handles.append(("transport", transport))
            return transport, protocol
        finally:
            self._io_ended()

    async def sock_connect(self, sock: Any, address: Any) -> None:
        self._io_started()
        try:
            await super().sock_connect(sock, address)
            self._io_handles.append(("socket", sock))
        finally:
            self._io_ended()

    async def getaddrinfo(self, *args: Any, **kwargs: Any) -> Any:
        self._io_started()
        try:
            return await super().getaddrinfo(*args, **kwargs)
        finally:
            self._io_ended()

    async def subprocess_exec(self, *args: Any, **kwargs: Any) -> Any:
        self._io_started()
        try:
            transport, protocol = await super().subprocess_exec(*args, **kwargs)
            self._io_handles.append(("process", transport))
            return transport, protocol
        finally:
            self._io_ended()

    def call_from_thread(self, coroutine: Any) -> Any:
        """Run a world coroutine without counting its blocked caller as computing."""
        ident = threading.get_ident()
        if ident == self._thread_id:
            coroutine.close()
            raise RuntimeError("synchronous world calls must run off the event-loop thread")
        if self.is_closed():
            coroutine.close()
            raise RuntimeError("point loop is closed")
        with self._thread_lock:
            owned = self._owns_thread(ident)
        if not owned:
            coroutine.close()
            reason = "world call from an unobserved thread"
            if reason not in self.inconclusive_reasons:
                self.inconclusive_reasons.append(reason)
            raise RuntimeError(reason)

        async def guarded() -> Any:
            token = WORLD_THREAD.set(ident)
            try:
                return await coroutine
            finally:
                # This runs before concurrent.Future wakes its blocking caller.
                self.thread_state(ident, "running")
                WORLD_THREAD.reset(token)

        with self._thread_lock:
            guarded_call = guarded()
            try:
                future = asyncio.run_coroutine_threadsafe(guarded_call, self)
            except BaseException:
                guarded_call.close()
                coroutine.close()
                raise
            self._resting_threads[ident] = "waiting"
            self._thread_progress = time.monotonic()
        self.call_soon_threadsafe(lambda: None)
        return future.result()

    def run_in_executor(self, executor: Any, func: Any, *args: Any) -> asyncio.Future[Any]:
        self._check_closed()
        if self._debug:
            self._check_callback(func, "run_in_executor")
        if executor is None:
            self._check_default_executor()
            executor = self._default_executor
            if executor is None:
                executor = concurrent.futures.ThreadPoolExecutor(thread_name_prefix="qagents-crash")
                self._default_executor = executor
        context = contextvars.copy_context()

        def owned_call() -> Any:
            ident = threading.get_ident()
            with self._thread_lock:
                self._owned_threads[ident] = self._owned_threads.get(ident, 0) + 1
            try:
                return context.run(functools.partial(func, *args))
            finally:
                with self._thread_lock:
                    self._owned_threads[ident] -= 1
                    if not self._owned_threads[ident]:
                        del self._owned_threads[ident]
                        self._resting_threads.pop(ident, None)

        submitted = executor.submit(owned_call)
        result = self.create_future()
        self._track_thread(submitted)

        def deliver(done: concurrent.futures.Future[Any]) -> None:
            try:
                if not result.cancelled():
                    if done.cancelled():
                        result.cancel()
                    elif (error := done.exception()) is not None:
                        result.set_exception(error)
                    else:
                        result.set_result(done.result())
            finally:
                # set_result schedules the waiter before computing stops counting.
                self._thread_ended(done)

        submitted.add_done_callback(lambda done: self._deliver_thread_callback(deliver, done))
        result.add_done_callback(lambda done: submitted.cancel() if done.cancelled() else None)
        return result

    def register_leg(self, leg: int, *, idle_horizon_s: float) -> None:
        horizon = _finite(idle_horizon_s, "idle_horizon_s")
        if horizon <= 0:
            raise ValueError("idle horizon must be positive")
        self._leg_progress[leg] = self.time(), horizon

    def deactivate_leg(self, leg: int) -> None:
        self._leg_progress.pop(leg, None)

    def note_world_call(self, leg: int, *, renewal: bool = False) -> None:
        if not renewal and leg in self._leg_progress:
            _, horizon = self._leg_progress[leg]
            self._leg_progress[leg] = self.time(), horizon

    def begin_teardown(self) -> None:
        self._tearing_down = True
        self._leg_progress.clear()

    def _check_bounds(self) -> None:
        if self._tearing_down:
            return
        if (
            self._started_at is not None
            and time.monotonic() - self._started_at > self.real_timeout_s
        ):
            raise NoTerminalState("point exceeded real-time bound")
        for leg, (progress, horizon) in self._leg_progress.items():
            if self.time() - progress >= horizon:
                raise NoTerminalState(f"leg-{leg} exceeded idle horizon")


if sys.platform == "win32":

    class VirtualLoop(_VirtualTime, asyncio.ProactorEventLoop):
        """Windows virtual-time loop retaining native subprocess support."""

else:

    class VirtualLoop(_VirtualTime, asyncio.SelectorEventLoop):
        """Selector virtual-time loop for POSIX platforms."""


class SelectorVirtualLoop(_VirtualTime, asyncio.SelectorEventLoop):
    """Selector variant for integrations requiring selector readiness on Windows."""
