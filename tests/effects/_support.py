"""Small test hosts for the frozen perform contract; no model calls or background workers."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from qagents.effects._fold import EffectView
from qagents.effects._ports import Timing
from qagents.errors import EffectHeld, EffectRejected, EntryConflict, HeldEffectsError
from qagents.session import Entry, InMemoryStore


class ManualLease:
    def __init__(self, *, remaining: float = 60.0, epoch: int = 1):
        self.seconds = remaining
        self.epoch = epoch
        self.owner = "test#owner"
        self.worker = "test"
        self.acquired_at = 100.0
        self.ttl_s = 60.0
        self.timing = Timing.for_lease(60, 1, effects=True)
        self._tracked = set()

    def remaining(self):
        return self.seconds

    def lost(self):
        return self.seconds <= 0

    def track(self, future):
        self._tracked.add(future)
        future.add_done_callback(self._tracked.discard)

    def running_sends(self):
        return len(self._tracked)

    async def sends_drained(self):
        while self._tracked:
            await asyncio.gather(*self._tracked, return_exceptions=True)


class MemoryJournal:
    """Append-only host with stable ids, an explicit clock, and visible append completion."""

    def __init__(self, entries=(), *, session_id="session", now=100.0):
        self.session_id = session_id
        self.worker = "test"
        self.now = now
        self.store = InMemoryStore()
        self.lease = ManualLease()
        self.timing = self.lease.timing
        self.entries = deepcopy(list(entries))
        self._view = EffectView()
        self._poisoned = False
        self.appending = False
        self.before_append = None
        self.after_append = None
        for item in self.entries:
            self._view.apply(item)

    def view(self):
        return self._view

    async def store_now(self):
        return self.now

    async def salt(self):
        return b"test-salt"

    def poisoned(self):
        return self._poisoned

    def track(self, future):
        self.lease.track(future)

    def assert_send_ready(self):
        assert not self.appending, "send started before the intent append returned"
        assert any(item.kind == "effect_intent" for item in self.entries), "send without intent"

    async def append_effect(self, entries: Sequence[Entry]):
        assert not self.appending, "overlapping append in test host"
        self.appending = True
        try:
            if self.before_append is not None:
                await self.before_append(entries)
            known = {item.id: item.seq for item in self.entries}
            present = [item.id in known for item in entries]
            if any(present):
                if not all(present):
                    raise EntryConflict("Batch contains both existing and new entry ids.")
                return [known[item.id] for item in entries]
            seqs = []
            for item in entries:
                item.seq = len(self.entries) + 1
                recorded = item.model_copy(deep=True)
                self.entries.append(recorded)
                self._view.apply(recorded)
                seqs.append(item.seq)
            if self.after_append is not None:
                await self.after_append(entries)
            return seqs
        finally:
            self.appending = False


class ScriptedEngine:
    """Replace perform with explicit results while maintaining current-attempt accounting."""

    def __init__(self, *outcomes):
        self.outcomes = iter(outcomes)
        self.calls = []
        self._patch = None

    async def perform(self, effect, request, *, scope, occurrence=None):
        key = scope.key_override or effect.key.render(
            request, effect=effect.name, operation_id=scope.operation_id, occurrence=occurrence
        )
        self.calls.append((effect, request, scope, occurrence))
        outcome = next(self.outcomes)
        if isinstance(outcome, SimpleNamespace) and hasattr(outcome, "wait"):
            await asyncio.sleep(outcome.wait)
            outcome = outcome.result
        if isinstance(outcome, EffectHeld):
            scope.held.append(outcome.held)
        elif isinstance(outcome, HeldEffectsError):
            scope.held.extend(outcome.held)
        elif isinstance(outcome, EffectRejected):
            scope.rejected.append((effect.name, key))
        elif not isinstance(outcome, BaseException):
            scope.committed.append((effect.name, key))
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def __enter__(self):
        self._patch = patch("qagents.effects._engine.perform", self.perform)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
