"""Small asyncio helpers of the Devvi transports."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger("qagents.integrations.devvi")


async def settle(cleanup: Awaitable[Any], pending: set[asyncio.Task[Any]]) -> bool:
    """Await ``cleanup`` once a turn has its outcome, so that a cancel cannot discard it.

    The cleanup (stopping a process, closing a session) runs as its own task. When the
    caller is cancelled while it waits (the calling run's timeout, the ``as_tool``
    deadline, the forced cancel after ``cancel_grace_s``), the cleanup goes on in the
    background, in ``pending`` (which its owner awaits before it disconnects), and the
    cancel is consumed, so the caller still returns the outcome it has. An error of the
    cleanup itself is raised when it finished here, and logged when it finished in the
    background.

    Returns:
        ``True`` when the cleanup finished here, ``False`` when it goes on in the background.
    """
    task = asyncio.ensure_future(cleanup)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if current is None or current.cancelling() == 0:
            return False  # the cleanup itself was cancelled (the event loop is closing)
        pending.add(task)
        task.add_done_callback(_forget(pending))
        current.uncancel()  # consumed: the caller goes on and returns its outcome
        return False
    return True


def _forget(pending: set[asyncio.Task[Any]]) -> Callable[[asyncio.Task[Any]], None]:
    def done(task: asyncio.Task[Any]) -> None:
        pending.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.warning("Cleaning up after a Devvi turn failed: %s", error, exc_info=error)

    return done


__all__ = ["settle"]
