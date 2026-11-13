"""Sessions: a conversation that outlives a process.

A session is an append-only journal of JSON entries. The agent loop writes an
entry for every message, every tool start and every tool result *as it
happens*, so the journal always says exactly how far a run got. Conversation
state is derived by reading the journal; nothing is pickled and nothing
depends on the shape of your code, so a run can be resumed by a different
process, a different deployment, or after the agent's tools changed.

Entry kinds:

| Kind | Meaning (data) |
|---|---|
| `run_started` | a run began (`agent`; `limits` when given to the run) |
| `message` | a message joined the conversation (`message`; see below) |
| `tool_started` | a tool call began executing (`tool_call_id`, `name`) |
| `tool_result` | a tool call finished (`result`; see below) |
| `tool_deferred` | a tool call waits for a result from outside the run (`input`) |
| `child_usage` | a child run ended (`child_run_id`, `tool_call_id`, `added_usage`) |
| `tool_checkpoint` | a running call noted a fact (`tool_call_id`, `operation_id`, `key`, `value`) |
| `approval_decided` | an approval was applied to a call, before it ran (see below) |
| `state_updated` | the session state changed; the full new values (`values`) |
| `agent_switched` | a handoff changed the active agent (`agent`) |
| `run_paused` | the run waits for approvals and inputs (`pending`, `inputs`) |
| `guardrail_tripped` | a guardrail blocked the prompt or the answer (`name`, `reason`, `stage`) |
| `run_finished` | the run ended (`status`, `stop_reason`, `output`, `text`) |
| `run_failed` | the run raised and stays open for a resume (see below) |
| `compaction` | the history was replaced by a shorter one (`messages`) |
| `workflow_step` | a workflow step completed (`name`, `result`) |

Details:

* A `message` from a model response also has the response's `usage`, the
  `model` and `provider` that answered, and the run's `turn`; a resume adds
  them up, so limits and `RunResult.usage` cover the whole run.
* A `tool_result` has `counted` when the call counts toward
  ``Limits.max_tool_calls``. A call waiting for approval counts only once an
  approval lets it run (its `tool_started`).
* A child run's usage is written in its own `child_usage` entry when it ends,
  linked to the delegating tool call. Usage added through ``ctx.add_usage``
  (for example a compactor's summary request) is written as `added_usage` on
  the run's next `message`,
  `tool_result`, `tool_checkpoint`, `run_paused`, `run_failed` or
  `run_finished` entry.
* `approval_decided` has `tool_call_id`, `name`, `approved`, `reason`,
  `arguments_changed`, and `arguments` when the approver changed them.
* A `tool_checkpoint` is written by ``RunContext.checkpoint`` while its call
  runs. A resume gives the values of the open turn's calls back to a
  repeated call (``RunContext.checkpoints``), by `operation_id`.
* `approval_decided`, `tool_started` and `tool_result` entries with a
  `source` (and `caller_tool_call_id`) are tool calls that did not come from
  the model (``RunContext.call_tool``: a tool served over MCP, for example).
  No assistant message has them; they count toward ``Limits.max_tool_calls``,
  and readers may show them apart.
* `run_failed` has `error_type`, `message` (the exception's text, shortened),
  `retryable`, `ambiguous`, and `error` (type and message joined).

Readers must ignore kinds they do not know: new ones may be added. A
`run_paused` entry written before 0.2 has no ``inputs``; it loads as a pause
with no pending inputs. Entries a run writes after its own `run_finished`
(a run that was abandoned while it still ran) are ignored by `derive_state`,
so the conversation stays well-formed.

Run status and leases (0.2) are optional store capabilities that
`qagents.runtime.AgentRuntime` builds on. A store that has them keeps one
`SessionStatus` row per session next to the journal, updated in the same
transaction as every `append`, so finding the sessions whose run is
``"running"`` or ``"failed"`` does not mean reading every journal. A lease
names the one worker allowed to drive a session's run; `append_leased`
writes only while the caller still holds it. `InMemoryStore`, `SQLiteStore`
and `qagents.stores.postgres.PostgresStore` implement both.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import functools
import json
import sqlite3
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar, Literal, Self, get_args

from pydantic import BaseModel, ConfigDict, Field

from .effects._fold import EffectView
from .effects._ports import Timing
from .effects._types import EffectRecord, HeldEffect
from .errors import LeaseError, SessionError
from .messages import Message, ToolResultPart, Usage


def _owner_token(worker: str) -> str:
    """Return a fresh lease owner for one acquisition by this worker (the M8 seam)."""
    return f"{worker}#{uuid.uuid4().hex}"


class Entry(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    kind: str
    data: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None
    ts: float = Field(default_factory=time.time)
    seq: int = 0
    """Position in the session, assigned by the store."""


class PendingApproval(BaseModel):
    """A tool call that is waiting for a human decision."""

    tool_call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None
    token: str | None = None
    """Opaque token that proves the pause is genuine, when pauses are signed."""


class PendingInput(BaseModel):
    """A tool call whose result must come from outside the run.

    The loop creates one for every call of a tool declared ``deferred=True``
    and for every call whose tool raised `DeferToolCall`. Supply the result
    with ``agent.resume(session, results={tool_call_id: value})``.
    """

    tool_call_id: str
    name: str
    """Name of the tool the model called."""
    arguments: dict[str, Any] = Field(default_factory=dict)
    """The arguments of the call, after approvals and hooks changed them."""
    reason: str | None = None
    """``None`` for a deferred tool; the reason given to `DeferToolCall` otherwise."""
    metadata: dict[str, Any] = Field(default_factory=dict)
    """Data the tool attached with `DeferToolCall`, for example an external job id."""
    token: str | None = None
    """Opaque token that proves the pause is genuine, when pauses are signed."""
    held: HeldEffect | None = None
    held_record_id: str | None = None


class SessionState(BaseModel):
    """What the journal says about a session right now."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    effects: EffectView = Field(default_factory=EffectView)
    open_turn_starts: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    tool_attempts: dict[str, int] = Field(default_factory=dict)
    decisions: dict[str, dict[str, Any]] = Field(default_factory=dict)
    decision_counts: dict[str, int] = Field(default_factory=dict)
    memos: dict[str, dict[str, Any]] = Field(default_factory=dict)
    run_requires: dict[str, list[str]] = Field(default_factory=dict)
    assistant_messages: int = 0

    messages: list[Message] = Field(default_factory=list)
    open_run_id: str | None = None
    """Id of a run that started and has not finished (running, paused or crashed)."""
    agent: str | None = None
    """Agent that was active in the open run: the one that started it, or the
    target of its last handoff. ``None`` when no run is open."""
    tool_results: dict[str, ToolResultPart] = Field(default_factory=dict)
    """Results already recorded for tool calls of the last assistant message."""
    started_tools: list[str] = Field(default_factory=list)
    """Tool calls that began executing and have no recorded result."""
    pending_approvals: list[PendingApproval] = Field(default_factory=list)
    """Tool calls of the paused run that wait for a decision."""
    pending_inputs: list[PendingInput] = Field(default_factory=list)
    """Tool calls of the open run that wait for a result from outside the run."""
    values: dict[str, Any] = Field(default_factory=dict)
    """The session state that tools and hooks see as ``ctx.state``, as last journaled."""
    run_messages: int = 0
    """How many messages the latest run has written to the journal."""
    run_usage: Usage = Field(default_factory=Usage)
    """Usage the latest run has journaled: its model responses and the usage
    added through ``ctx.add_usage`` (by tools, sub-agents or a compactor). A
    resume continues from it, so limits cover the whole run."""
    run_turns: int = 0
    """Model requests of the latest run that produced a journaled response."""
    run_tool_call_ids: list[str] = Field(default_factory=list)
    """Tool calls of the latest run that count toward ``Limits.max_tool_calls``."""
    checkpoints: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Values journaled with ``RunContext.checkpoint`` by the tool calls of the open run's
    last turn, by operation id, then key (the latest value of a key wins)."""
    operation_usage: dict[str, Usage] = Field(default_factory=dict)
    """Usage that tool calls of the latest run added with ``ctx.add_usage`` and that the
    journal holds, by operation id (part of `run_usage`). A call repeated after a crash
    reads its own as ``RunContext.journaled_usage``."""

    @property
    def run_tool_calls(self) -> int:
        """How many tool calls of the latest run count toward ``Limits.max_tool_calls``."""
        return len(self.run_tool_call_ids)

    @property
    def unresolved_tool_calls(self) -> bool:
        """True when the last message is an assistant message whose tool calls lack results."""
        return bool(
            self.messages and self.messages[-1].role == "assistant" and self.messages[-1].tool_calls
        )


def derive_state(entries: list[Entry], *, session_id: str | None = None) -> SessionState:
    """Replay a journal into the current `SessionState`.

    Args:
        entries: The session's entries, oldest first. Unknown kinds are ignored.

    Returns:
        The state after the last entry. ``values`` is a copy: changing it does
        not change the journal.
    """
    state = SessionState()
    started: dict[str, None] = {}
    inputs: dict[str, PendingInput] = {}
    values: dict[str, Any] = {}
    counted: dict[str, None] = {}
    checkpoints: dict[str, dict[str, Any]] = {}
    finished: set[str] = set()
    """Runs closed by ``run_finished``. What such a run writes later (it was
    abandoned while it still ran) is ignored, so the conversation stays valid."""
    for entry in entries:
        kind, data = entry.kind, entry.data
        if entry.run_id is not None and entry.run_id in finished and kind != "run_finished":
            continue
        if kind == "message":
            state.messages.append(Message.model_validate(data["message"]))
            state.run_messages += 1
            state.tool_results = {}
            state.pending_approvals = []
            started = {}
            inputs = {}
            checkpoints = {}  # a new message closes the turn whose calls recorded them
            if entry.run_id is not None and entry.run_id == state.open_run_id:
                _count_usage(state, data)
                state.run_turns = max(state.run_turns, int(data.get("turn") or 0))
        elif kind == "tool_started":
            started[data["tool_call_id"]] = None
            counted[data["tool_call_id"]] = None
        elif kind == "tool_result":
            result = ToolResultPart.model_validate(data["result"])
            state.tool_results[result.tool_call_id] = result
            started.pop(result.tool_call_id, None)
            inputs.pop(result.tool_call_id, None)
            state.pending_approvals = [
                item for item in state.pending_approvals if item.tool_call_id != result.tool_call_id
            ]
            if data.get("counted"):
                counted[result.tool_call_id] = None
            if entry.run_id is not None and entry.run_id == state.open_run_id:
                _count_usage(state, data)
        elif kind == "tool_deferred":
            waiting = PendingInput.model_validate(data["input"])
            started.pop(waiting.tool_call_id, None)
            inputs[waiting.tool_call_id] = waiting
            counted[waiting.tool_call_id] = None
        elif kind == "tool_checkpoint":
            if entry.run_id is not None and entry.run_id == state.open_run_id:
                operation, key = data.get("operation_id"), data.get("key")
                if isinstance(operation, str) and isinstance(key, str):
                    checkpoints.setdefault(operation, {})[key] = data.get("value")
                _count_usage(state, data)  # usage from ctx.add_usage may ride on it
        elif kind == "state_updated":
            values = data.get("values", {})
        elif kind == "run_started":
            state.open_run_id = entry.run_id
            state.agent = data.get("agent")
            state.run_messages = 0
            state.run_usage = Usage()
            state.operation_usage = {}
            state.run_turns = 0
            counted = {}
            checkpoints = {}
        elif kind == "agent_switched":
            state.agent = data.get("agent")
        elif kind == "run_paused":
            if entry.run_id is not None and entry.run_id == state.open_run_id:
                _count_usage(state, data)
            state.pending_approvals = [
                PendingApproval.model_validate(item) for item in data.get("pending", [])
            ]
            if "inputs" in data:  # absent in journals written before 0.2
                listed = [PendingInput.model_validate(item) for item in data["inputs"]]
                inputs = {item.tool_call_id: item for item in listed}
        elif kind in {"run_failed", "child_usage"}:
            if entry.run_id is not None and entry.run_id == state.open_run_id:
                _count_usage(state, data)
        elif kind == "run_finished":
            if entry.run_id is not None:
                finished.add(entry.run_id)
            if entry.run_id == state.open_run_id:
                state.open_run_id = None
                state.agent = None
                state.pending_approvals = []
                inputs = {}
        elif kind == "compaction":
            state.messages = [Message.model_validate(item) for item in data["messages"]]
        elif kind == "guardrail_tripped" and data.get("stage") == "output":
            _mark_blocked(state.messages)
    state.started_tools = list(started)
    state.pending_inputs = list(inputs.values())
    state.run_tool_call_ids = list(counted)
    state.values = copy.deepcopy(values)
    if state.open_run_id is not None:  # only the open run's last turn can repeat a call
        state.checkpoints = copy.deepcopy(checkpoints)
    return state


def _mark_blocked(messages: list[Message]) -> None:
    """Mark the last assistant message, an answer an output guardrail blocked."""
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == "assistant":
            blocked = {**messages[index].metadata, "blocked": True}
            messages[index] = messages[index].model_copy(update={"metadata": blocked})
            return


def _count_usage(state: SessionState, data: dict[str, Any]) -> None:
    """Add an entry's usage: a model response's ``usage``, and ``added_usage``
    (from ``ctx.add_usage``: a compactor's summary request, a sub-agent), with the
    tool calls that added it (``added_usage_by_operation``)."""
    for key in ("usage", "added_usage"):
        if usage := data.get(key):
            state.run_usage = state.run_usage.add(Usage.model_validate(usage))
    for operation_id, usage in _usage_loads(data).items():
        held = state.operation_usage.get(operation_id, Usage())
        state.operation_usage[operation_id] = held.add(usage)


def _usage_loads(data: dict[str, Any]) -> dict[str, Usage]:
    """An entry's ``added_usage_by_operation``: ``{operation id: Usage}``, the part of its
    ``added_usage`` that each tool call added. Empty when the entry has none (or a value
    this version cannot read: it is ignored, as unknown entry kinds are)."""
    raw = data.get("added_usage_by_operation")
    if not isinstance(raw, dict):
        return {}
    usage: dict[str, Usage] = {}
    for operation_id, item in raw.items():
        if isinstance(operation_id, str) and isinstance(item, dict):
            try:
                usage[operation_id] = Usage.model_validate(item)
            except ValueError:
                continue
    return usage


SessionRunStatus = Literal["idle", "running", "paused", "failed"]
"""Where a session's latest run stands, as the store tracks it (`SessionStatus.status`)."""

_RUN_STATUSES = frozenset(get_args(SessionRunStatus))


class SessionStatus(BaseModel):
    """What a store tracks about a session next to its journal.

    Returned by `SessionStore.status` and `SessionStore.list_sessions`. The
    run fields follow the entry kinds that `append` writes:

    | After | ``status`` |
    |---|---|
    | `run_started` | ``"running"`` |
    | `run_paused` | ``"paused"`` |
    | `run_failed` | ``"failed"``: the run raised and is still open, so it can be resumed |
    | `run_finished` of the open run | ``"idle"`` |
    | any other entry of a paused or failed open run | ``"running"``: it was resumed |

    A resumed run writes no `run_started`, so a paused or failed run turns
    ``"running"`` again with its first new entry. Until then it keeps its old
    status: a resume that is waiting for the model still shows ``"failed"``.
    """

    session_id: str
    lease_epoch: int = 0
    lease_acquired_at: float | None = None
    has_effects: bool = False
    retry_after: float | None = None
    incarnation: str | None = None
    status: SessionRunStatus = "idle"
    open_run_id: str | None = None
    """Id of the run that started and has not finished."""
    agent: str | None = None
    """Name of the agent that started the latest run, from its `run_started`
    entry. After a handoff it is still the first agent: the one to call
    ``resume`` on."""
    failures: int = 0
    """How many `run_failed` entries the open run has written."""
    updated_at: float | None = None
    """When entries were last appended, in seconds since the epoch on the
    store's clock. ``None`` when the session has no entries."""
    lease_owner: str | None = None
    """The worker that holds the lease, or held it until it expired. ``None``
    when the lease was released or never taken."""
    lease_expires_at: float | None = None
    """When the lease expires, in seconds since the epoch on the store's clock."""

    def lease_active(self, now: float | None = None) -> bool:
        """True when a worker holds a lease that has not expired.

        Args:
            now: The current time on the store's clock. Defaults to
                ``time.time()``, which is the clock of `InMemoryStore` and
                `SQLiteStore`. `PostgresStore` uses the database server's
                clock, so with it this is only as exact as the two clocks agree.
        """
        if self.lease_owner is None or self.lease_expires_at is None:
            return False
        return self.lease_expires_at > (time.time() if now is None else now)


def advance_status(status: SessionStatus, entry: Entry) -> None:
    """Update ``status`` in place for one appended entry. For store implementers.

    Stores call it for every entry inside the transaction of `append`, so the
    status always matches the journal. It reads only ``entry.kind``,
    ``entry.run_id`` and, for `run_started`, ``entry.data["agent"]``. Entries
    that do not belong to the open run (a workflow step, the late close of an
    older run) leave the status as it is.
    """
    kind = entry.kind
    if kind == "run_started":
        status.status = "running"
        status.open_run_id = entry.run_id
        status.agent = entry.data.get("agent")
        status.failures = 0
    elif status.open_run_id is None or entry.run_id != status.open_run_id:
        return
    elif kind == "run_finished":
        status.status = "idle"
        status.open_run_id = None
        status.failures = 0
    elif kind == "run_paused":
        status.status = "paused"
    elif kind == "run_failed":
        status.status = "failed"
        status.failures += 1
    else:
        status.status = "running"


class LeaseGrant(BaseModel):
    owner: str
    epoch: int
    acquired_at: float
    expires_at: float


FEATURES = frozenset({"leases", "epochs", "entry_ids", "now", "salt", "registry", "claims"})
EFFECTS_REQUIRED = FEATURES - {"claims"}


class SessionStore(ABC):
    """Where journals live. Implement three methods to add a backend.

    `append`, `load` and `delete` are all a session needs. The other methods
    are optional capabilities used by `qagents.runtime.AgentRuntime`: run
    status (`status`, `list_sessions`) and leases (`acquire_lease`,
    `renew_lease`, `release_lease`, `append_leased`). Their defaults raise
    `NotImplementedError`. A store that implements them must update the
    status in the same transaction as the entries (see `advance_status`) and
    check the lease in the same transaction as an `append_leased` write.
    """

    features: ClassVar[frozenset[str]] = frozenset()
    takeover_grace_s: float = 1.0

    async def now(self) -> float:
        raise NotImplementedError(_unsupported(self, "store clock"))

    async def acquire_lease_grant(
        self, session_id: str, owner: str, ttl_s: float
    ) -> LeaseGrant | None:
        raise NotImplementedError(_unsupported(self, "lease epochs"))

    async def get_claim(self, namespace: str, effect: str, key: str):
        raise NotImplementedError(_unsupported(self, "effect claims"))

    async def purge_claims(self, *, older_than: float) -> int:
        raise NotImplementedError(_unsupported(self, "effect claims"))

    async def register_effect(
        self, effect: str, version: str, key_spec_digest: str, request_schema_digest: str
    ) -> Literal["registered", "matches", "key_changed", "schema_changed"]:
        raise NotImplementedError(_unsupported(self, "effect registry"))

    async def salt(self) -> bytes:
        raise NotImplementedError(_unsupported(self, "effect salt"))

    @abstractmethod
    async def append(self, session_id: str, entries: list[Entry]) -> None:
        """Durably add ``entries`` to the end of the session, in order.

        One call is all or nothing: it writes every entry or, when it raises,
        none of them. The agent relies on this to journal related entries
        together (the start of a run, an answer with its turn, a handoff with
        its switch), so a crash never leaves half of them. A store that writes
        entries one at a time must wrap a call in one transaction.
        """

    @abstractmethod
    async def load(self, session_id: str) -> list[Entry]:
        """Return every entry of the session, oldest first."""

    @abstractmethod
    async def delete(
        self, session_id: str, *, force: bool | Literal["forget_effects"] = False
    ) -> None:
        """Remove the session."""

    async def aclose(self) -> None:  # noqa: B027 - optional to override
        """Release what the store holds."""

    async def __aenter__(self) -> Self:
        """``async with SQLiteStore("agents.db") as store:`` closes the store at the end."""
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    # -- optional: run status --------------------------------------------------------

    async def status(self, session_id: str) -> SessionStatus:
        """Return the tracked status of one session.

        A session the store has never seen comes back as ``"idle"`` with
        ``updated_at=None``.
        """
        raise NotImplementedError(_unsupported(self, "run status"))

    async def list_sessions(
        self,
        status: str | Iterable[str] | None = None,
        prefix: str | None = None,
        limit: int | None = None,
    ) -> list[SessionStatus]:
        """List sessions that have entries, least recently updated first.

        Args:
            status: Keep only sessions with this status, or with any of these
                statuses (``"idle"``, ``"running"``, ``"paused"``, ``"failed"``).
            prefix: Keep only session ids that start with this text.
            limit: Return at most this many.

        Raises:
            ValueError: ``status`` names an unknown status, or ``limit`` is negative.
        """
        raise NotImplementedError(_unsupported(self, "run status"))

    # -- optional: leases ------------------------------------------------------------

    async def acquire_lease(self, session_id: str, owner: str, ttl_s: float) -> bool:
        """Take the session's lease for ``ttl_s`` seconds.

        Succeeds when the lease is free, has expired, or is already held by
        ``owner`` (which renews it). The session need not have entries yet.

        Returns:
            True when ``owner`` now holds the lease.
        """
        raise NotImplementedError(_unsupported(self, "leases"))

    async def renew_lease(self, session_id: str, owner: str, ttl_s: float) -> bool:
        """Extend ``owner``'s lease to ``ttl_s`` seconds from now.

        Returns:
            True when ``owner`` still held the lease. A lease that expired but
            was not taken by another worker is still held and is renewed.
        """
        raise NotImplementedError(_unsupported(self, "leases"))

    async def release_lease(self, session_id: str, owner: str) -> None:
        """Give up ``owner``'s lease. Does nothing when ``owner`` does not hold it."""
        raise NotImplementedError(_unsupported(self, "leases"))

    async def append_leased(self, session_id: str, entries: list[Entry], owner: str) -> None:
        """Append like `append`, but only while ``owner`` holds the session's lease.

        The lease check and the write happen in one transaction, so a worker
        whose lease was taken over can no longer write to the journal (the
        lease acts as a fencing token). Like `append`, one call writes all of
        its entries or none.

        On a store with leases, a plain `Agent.run` or `Agent.resume` also
        writes every entry through this method, not through `append`. The
        built-in stores do not call `append` from here, so a subclass that
        customizes writes (to mirror, encrypt or audit them) must override
        both. A subclass that overrides `append` alone keeps getting every
        entry of a plain run through `append`; such a run then takes no lease
        (only runs in the same process are kept apart).

        Raises:
            LeaseError: ``owner`` does not hold the lease. Nothing was written.
        """
        raise NotImplementedError(_unsupported(self, "leases"))


def _unsupported(store: SessionStore, capability: str) -> str:
    return (
        f"{type(store).__name__} does not support {capability}. Use InMemoryStore, SQLiteStore "
        "or qagents.stores.postgres.PostgresStore, or implement the optional SessionStore "
        "methods (status, list_sessions, acquire_lease, renew_lease, release_lease, "
        "append_leased)."
    )


def _status_filter(status: str | Iterable[str] | None) -> set[str] | None:
    """Normalise the ``status`` argument of `SessionStore.list_sessions`."""
    if status is None:
        return None
    wanted = {status} if isinstance(status, str) else set(status)
    unknown = sorted(wanted - _RUN_STATUSES)
    if unknown:
        raise ValueError(
            f"Unknown session status {unknown}; use one or more of {sorted(_RUN_STATUSES)}."
        )
    return wanted


def _check_limit(limit: int | None) -> None:
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be None or at least 0, not {limit}.")


def _check_ttl(ttl_s: float) -> None:
    if not ttl_s > 0 or ttl_s == float("inf"):
        raise ValueError(f"A lease needs a finite ttl_s above 0 seconds, not {ttl_s!r}.")


def _lease_lost(session_id: str, owner: str, holder: str | None) -> str:
    who = f"worker {holder!r} holds it now" if holder else "nobody holds it now"
    return (
        f"Worker {owner!r} no longer holds the lease on session {session_id!r} ({who}), so "
        "its entries were not written. Another worker may be driving this run; stop this one."
    )


class InMemoryStore(SessionStore):
    """Keeps journals in a dict. For tests and for sessions that need not survive a restart.

    Run status and leases are supported, with ``time.time()`` as the clock.
    They only coordinate tasks of one process, since nothing else can see
    the store.
    """

    def __init__(
        self, *, clock: Callable[[], float] | None = None, takeover_grace_s: float = 1.0
    ) -> None:
        self._clock = clock or time.time
        self.takeover_grace_s = takeover_grace_s
        self._sessions: dict[str, list[Entry]] = {}
        self._status: dict[str, SessionStatus] = {}

    def _row(self, session_id: str) -> SessionStatus:
        return self._status.setdefault(session_id, SessionStatus(session_id=session_id))

    def _write(self, session_id: str, entries: list[Entry]) -> None:
        if not entries:
            return
        journal = self._sessions.setdefault(session_id, [])
        row = self._row(session_id)
        for entry in entries:
            stored = entry.model_copy(update={"seq": len(journal) + 1})
            journal.append(stored)
            advance_status(row, stored)
        row.updated_at = time.time()

    async def append(self, session_id: str, entries: list[Entry]) -> None:
        self._write(session_id, entries)

    async def append_leased(self, session_id: str, entries: list[Entry], owner: str) -> None:
        row = self._status.get(session_id)
        holder = row.lease_owner if row is not None else None
        if holder != owner:
            raise LeaseError(_lease_lost(session_id, owner, holder))
        self._write(session_id, entries)

    async def load(self, session_id: str) -> list[Entry]:
        return list(self._sessions.get(session_id, []))

    async def delete(
        self, session_id: str, *, force: bool | Literal["forget_effects"] = False
    ) -> None:
        self._sessions.pop(session_id, None)
        self._status.pop(session_id, None)

    async def status(self, session_id: str) -> SessionStatus:
        row = self._status.get(session_id)
        return row.model_copy() if row is not None else SessionStatus(session_id=session_id)

    async def list_sessions(
        self,
        status: str | Iterable[str] | None = None,
        prefix: str | None = None,
        limit: int | None = None,
    ) -> list[SessionStatus]:
        wanted = _status_filter(status)
        _check_limit(limit)
        rows = [
            row
            for row in self._status.values()
            if row.updated_at is not None
            and (wanted is None or row.status in wanted)
            and (prefix is None or row.session_id.startswith(prefix))
        ]
        rows.sort(key=lambda row: (row.updated_at or 0.0, row.session_id))
        return [row.model_copy() for row in rows[:limit]]

    async def acquire_lease(self, session_id: str, owner: str, ttl_s: float) -> bool:
        _check_ttl(ttl_s)
        now = time.time()
        row = self._row(session_id)
        if row.lease_owner not in (None, owner) and (row.lease_expires_at or 0.0) > now:
            return False
        row.lease_owner, row.lease_expires_at = owner, now + ttl_s
        return True

    async def renew_lease(self, session_id: str, owner: str, ttl_s: float) -> bool:
        _check_ttl(ttl_s)
        row = self._status.get(session_id)
        if row is None or row.lease_owner != owner:
            return False
        row.lease_expires_at = time.time() + ttl_s
        return True

    async def release_lease(self, session_id: str, owner: str) -> None:
        row = self._status.get(session_id)
        if row is not None and row.lease_owner == owner:
            row.lease_owner, row.lease_expires_at = None, None


_SQLITE_SCHEMA_VERSION = 1
"""Stored in ``PRAGMA user_version``. 0 is a 0.1 database: the entries table only."""

_SQLITE_STATUS_COLUMNS = (
    "session_id, status, open_run_id, agent, failures, updated_at, lease_owner, lease_expires_at"
)


class SQLiteStore(SessionStore):
    """Journals in one SQLite file (WAL mode). Good for a single host.

    Calls run on the store's own worker thread, so the event loop is never
    blocked on disk, and tools that hang in the loop's default thread pool
    cannot hold up journal writes or lease renewals. Every append is synced
    to disk before it returns (``PRAGMA synchronous=FULL``), so an entry
    survives an OS crash or a power loss, as "record first, act second" needs.
    Close the store with ``await store.aclose()`` or ``async with``.

    Run status and leases are kept in a ``sessions`` table next to the
    ``entries`` table. A database written by qagents 0.1 is migrated in place
    the first time it is opened: the table is added and filled from the
    journals, and no entry is changed. Leases use ``time.time()``, which every
    process on the host shares; do not share the file across hosts.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], float] | None = None,
        takeover_grace_s: float = 1.0,
    ) -> None:
        self._clock = clock or time.time
        self.takeover_grace_s = takeover_grace_s
        self.path = str(path)
        self._lock = threading.Lock()
        self._connection: sqlite3.Connection | None = None
        self._executor: ThreadPoolExecutor | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            connection = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
            try:
                _enable_wal(connection)
                # FULL syncs the WAL at every commit: a committed tool_started survives a
                # power loss. NORMAL would only sync at checkpoints.
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute("PRAGMA fullfsync=ON")  # macOS: flush the drive cache too
                connection.execute("PRAGMA checkpoint_fullfsync=ON")
                self._migrate(connection)
            except BaseException:
                connection.close()
                raise
            self._connection = connection
        return self._connection

    def _migrate(self, connection: sqlite3.Connection) -> None:
        """Bring the schema to `_SQLITE_SCHEMA_VERSION`, keeping every entry."""
        (version,) = connection.execute("PRAGMA user_version").fetchone()
        if version == _SQLITE_SCHEMA_VERSION:
            return
        connection.execute("BEGIN IMMEDIATE")
        try:
            (version,) = connection.execute("PRAGMA user_version").fetchone()  # another process?
            if version > _SQLITE_SCHEMA_VERSION:
                raise SessionError(
                    f"{self.path} was written by a newer qagents (schema version {version}; "
                    f"this version reads up to {_SQLITE_SCHEMA_VERSION}). Upgrade qagents."
                )
            if version < 1:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS entries ("
                    " session_id TEXT NOT NULL, seq INTEGER NOT NULL, ts REAL NOT NULL,"
                    " run_id TEXT, kind TEXT NOT NULL, data TEXT NOT NULL,"
                    " PRIMARY KEY (session_id, seq))"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS sessions ("
                    " session_id TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'idle',"
                    " open_run_id TEXT, agent TEXT, failures INTEGER NOT NULL DEFAULT 0,"
                    " updated_at REAL, lease_owner TEXT, lease_expires_at REAL)"
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS sessions_by_status ON sessions (status, updated_at)"
                )
                self._backfill(connection)
            connection.execute(f"PRAGMA user_version = {_SQLITE_SCHEMA_VERSION}")
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise

    @staticmethod
    def _backfill(connection: sqlite3.Connection) -> None:
        """Derive a status row for every session of a 0.1 journal."""
        rows: list[SessionStatus] = []
        current: SessionStatus | None = None
        cursor = connection.execute(
            "SELECT session_id, ts, run_id, kind,"
            " CASE WHEN kind = 'run_started' THEN data END"
            " FROM entries ORDER BY session_id, seq"
        )
        for session_id, ts, run_id, kind, data in cursor:
            if current is None or current.session_id != session_id:
                current = SessionStatus(session_id=session_id)
                rows.append(current)
            entry = Entry(kind=kind, run_id=run_id, ts=ts, data=json.loads(data) if data else {})
            advance_status(current, entry)
            current.updated_at = ts
        connection.executemany(
            "INSERT OR REPLACE INTO sessions"
            " (session_id, status, open_run_id, agent, failures, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    row.session_id,
                    row.status,
                    row.open_run_id,
                    row.agent,
                    row.failures,
                    row.updated_at,
                )
                for row in rows
            ],
        )

    def _append(self, session_id: str, entries: list[Entry], owner: str | None = None) -> None:
        with self._lock:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            try:
                found = connection.execute(
                    "SELECT status, open_run_id, agent, failures, lease_owner FROM sessions"
                    " WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if owner is not None and (found is None or found[4] != owner):
                    raise LeaseError(_lease_lost(session_id, owner, found and found[4]))
                status = SessionStatus(session_id=session_id)
                if found is not None:
                    status.status, status.open_run_id, status.agent, status.failures = found[:4]
                (last,) = connection.execute(
                    "SELECT COALESCE(MAX(seq), 0) FROM entries WHERE session_id = ?", (session_id,)
                ).fetchone()
                rows = []
                for offset, entry in enumerate(entries, start=1):
                    rows.append(
                        (
                            session_id,
                            last + offset,
                            entry.ts,
                            entry.run_id,
                            entry.kind,
                            _entry_json(entry),
                        )
                    )
                    advance_status(status, entry)
                connection.executemany(
                    "INSERT INTO entries (session_id, seq, ts, run_id, kind, data)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    rows,
                )
                connection.execute(
                    "INSERT INTO sessions"
                    " (session_id, status, open_run_id, agent, failures, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (session_id) DO UPDATE SET"
                    " status = excluded.status, open_run_id = excluded.open_run_id,"
                    " agent = excluded.agent, failures = excluded.failures,"
                    " updated_at = excluded.updated_at",
                    (
                        session_id,
                        status.status,
                        status.open_run_id,
                        status.agent,
                        status.failures,
                        time.time(),
                    ),
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    def _load(self, session_id: str) -> list[Entry]:
        with self._lock:
            rows = (
                self._connect()
                .execute(
                    "SELECT seq, ts, run_id, kind, data FROM entries"
                    " WHERE session_id = ? ORDER BY seq",
                    (session_id,),
                )
                .fetchall()
            )
        return [
            Entry(seq=seq, ts=ts, run_id=run_id, kind=kind, data=json.loads(data))
            for seq, ts, run_id, kind, data in rows
        ]

    def _delete(self, session_id: str) -> None:
        with self._lock:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute("DELETE FROM entries WHERE session_id = ?", (session_id,))
                connection.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    def _status(self, session_id: str) -> SessionStatus:
        with self._lock:
            found = (
                self._connect()
                .execute(
                    f"SELECT {_SQLITE_STATUS_COLUMNS} FROM sessions WHERE session_id = ?",
                    (session_id,),
                )
                .fetchone()
            )
        return _status_row(found) if found else SessionStatus(session_id=session_id)

    def _list(self, wanted: set[str] | None, prefix: str | None, limit: int | None) -> list[Any]:
        clauses, params = ["updated_at IS NOT NULL"], []
        if wanted is not None:
            clauses.append(f"status IN ({', '.join('?' * len(wanted))})")
            params.extend(sorted(wanted))
        if prefix is not None:
            clauses.append("substr(session_id, 1, ?) = ?")
            params.extend([len(prefix), prefix])
        query = (
            f"SELECT {_SQLITE_STATUS_COLUMNS} FROM sessions WHERE {' AND '.join(clauses)}"
            " ORDER BY updated_at, session_id"
        )
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with self._lock:
            return self._connect().execute(query, params).fetchall()

    def _acquire(self, session_id: str, owner: str, ttl_s: float) -> bool:
        now = time.time()
        with self._lock:
            cursor = self._connect().execute(
                "INSERT INTO sessions (session_id, lease_owner, lease_expires_at) VALUES (?, ?, ?)"
                " ON CONFLICT (session_id) DO UPDATE SET lease_owner = excluded.lease_owner,"
                " lease_expires_at = excluded.lease_expires_at"
                " WHERE sessions.lease_owner IS NULL OR sessions.lease_owner = excluded.lease_owner"
                " OR sessions.lease_expires_at <= ?",
                (session_id, owner, now + ttl_s, now),
            )
            return cursor.rowcount == 1

    def _renew(self, session_id: str, owner: str, ttl_s: float) -> bool:
        with self._lock:
            cursor = self._connect().execute(
                "UPDATE sessions SET lease_expires_at = ? WHERE session_id = ? AND lease_owner = ?",
                (time.time() + ttl_s, session_id, owner),
            )
            return cursor.rowcount == 1

    def _release(self, session_id: str, owner: str) -> None:
        with self._lock:
            self._connect().execute(
                "UPDATE sessions SET lease_owner = NULL, lease_expires_at = NULL"
                " WHERE session_id = ? AND lease_owner = ?",
                (session_id, owner),
            )

    def _close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    async def append(self, session_id: str, entries: list[Entry]) -> None:
        if entries:
            await self._call(self._append, session_id, entries)

    async def append_leased(self, session_id: str, entries: list[Entry], owner: str) -> None:
        if entries:
            await self._call(self._append, session_id, entries, owner)

    async def load(self, session_id: str) -> list[Entry]:
        return await self._call(self._load, session_id)

    async def delete(
        self, session_id: str, *, force: bool | Literal["forget_effects"] = False
    ) -> None:
        await self._call(self._delete, session_id)

    async def status(self, session_id: str) -> SessionStatus:
        return await self._call(self._status, session_id)

    async def list_sessions(
        self,
        status: str | Iterable[str] | None = None,
        prefix: str | None = None,
        limit: int | None = None,
    ) -> list[SessionStatus]:
        wanted = _status_filter(status)
        _check_limit(limit)
        rows = await self._call(self._list, wanted, prefix, limit)
        return [_status_row(row) for row in rows]

    async def acquire_lease(self, session_id: str, owner: str, ttl_s: float) -> bool:
        _check_ttl(ttl_s)
        return await self._call(self._acquire, session_id, owner, ttl_s)

    async def renew_lease(self, session_id: str, owner: str, ttl_s: float) -> bool:
        _check_ttl(ttl_s)
        return await self._call(self._renew, session_id, owner, ttl_s)

    async def release_lease(self, session_id: str, owner: str) -> None:
        await self._call(self._release, session_id, owner)

    async def aclose(self) -> None:
        if self._executor is None:
            return
        await self._call(self._close)
        executor, self._executor = self._executor, None
        executor.shutdown(wait=False)

    async def _call(self, function: Any, *args: Any) -> Any:
        """Run ``function`` on the store's own thread; one is enough, as ``_lock`` serialises."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qagents-sqlite")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, functools.partial(function, *args))


def _entry_json(entry: Entry) -> str:
    """The JSON of an entry's data, with non-ASCII text kept as it is.

    A string with a lone surrogate (half of an emoji cut by another service)
    cannot be encoded as UTF-8, so such an entry is written with ASCII escapes,
    which bring the same string back when it is read.
    """
    text = json.dumps(entry.data, ensure_ascii=False)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        text = json.dumps(entry.data, ensure_ascii=True)
    return text


def _enable_wal(connection: sqlite3.Connection, timeout_s: float = 5.0) -> None:
    """Switch the database to WAL mode, waiting while another connection is switching it.

    Changing the journal mode does not wait for locks the way other statements
    do, so processes that open a new database at the same time retry here.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            (mode,) = connection.execute("PRAGMA journal_mode").fetchone()
            if str(mode).lower() != "wal":
                connection.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError as error:
            if "locked" not in str(error) or time.monotonic() > deadline:
                raise
            time.sleep(0.02)


def _status_row(row: Any) -> SessionStatus:
    session_id, status, open_run_id, agent, failures, updated_at, owner, expires = row
    return SessionStatus(
        session_id=session_id,
        status=status,
        open_run_id=open_run_id,
        agent=agent,
        failures=failures,
        updated_at=updated_at,
        lease_owner=owner,
        lease_expires_at=expires,
    )


class Session:
    """One conversation in a store.

    ``Session()`` alone keeps the conversation in memory. Pass a store and an
    id to make it durable:

        store = SQLiteStore("agents.db")
        session = Session(store, id=f"user-{user_id}")
        await agent.run("Hello", session=session)
    """

    def __init__(self, store: SessionStore | None = None, id: str | None = None) -> None:
        self.store = store or InMemoryStore()
        self.id = id or uuid.uuid4().hex

    async def _resolve(self) -> None:
        """Resolve a specialized session's physical id before I/O (ordinary sessions are ready)."""

    async def entries(self) -> list[Entry]:
        await self._resolve()
        return await self.store.load(self.id)

    async def state(self) -> SessionState:
        return derive_state(await self.entries())

    async def messages(self) -> list[Message]:
        return (await self.state()).messages

    async def append(self, kind: str, data: dict[str, Any], *, run_id: str | None = None) -> None:
        await self._resolve()
        await self.store.append(self.id, [Entry(kind=kind, data=data, run_id=run_id)])

    async def append_entries(self, entries: list[Entry]) -> None:
        """Append several entries in one store call, which every store writes atomically."""
        await self._resolve()
        await self.store.append(self.id, entries)

    async def effects(
        self, status: str | Iterable[str] | None = None, namespace: str | None = None
    ) -> list[EffectRecord]:
        raise NotImplementedError("Effect session views are pending the store integration stage.")

    async def clear(self, *, force: bool | Literal["forget_effects"] = False) -> None:
        await self._resolve()
        if force is False:
            await self.store.delete(self.id)
        else:
            await self.store.delete(self.id, force=force)

    def __repr__(self) -> str:
        return f"Session(id={self.id!r}, store={type(self.store).__name__})"


async def append_fenced(
    store: SessionStore,
    session_id: str,
    entries: list[Entry],
    owner: str | None,
    *,
    fence: Callable[[], bool],
    attempts: int = 3,
) -> list[int] | None:
    """SPEC 7.7.2-3 in one place: first append (append_leased with owner, or append when
    owner is None); a definite refusal (LeaseError, validation SessionError,
    EntryConflict, ClaimConflict) re-raised; any other error, CancelledError included,
    is an unknown outcome: re-append the same ids while fence() holds, up to `attempts`,
    then load(): all ids present -> their seqs, none -> the last error re-raised as
    definite, a read that fails -> JournalOutcomeUnknown. A CancelledError during the
    first append is an unknown outcome like any other: the outcome is decided first,
    then the CancelledError is re-raised. What the caller does after that (record the
    commit in the scope, return the receipt) is protected only by the engine's _shield
    (3.7, M39)."""
    raise NotImplementedError("Effects implementation is pending its owning stage.")


class Lease:
    """One lease's token, epoch, deadline, renewals and running sends; implementation pending."""

    def __init__(
        self,
        store: SessionStore,
        session_id: str,
        *,
        worker: str,
        ttl_s: float,
        purpose: Literal["run", "recovery", "settle", "recheck", "workflow"],
        timing: Timing,
    ) -> None:
        """owner = _owner_token(worker)."""
        raise NotImplementedError("Effects implementation is pending its owning stage.")

    owner: str
    worker: str
    epoch: int | None
    acquired_at: float | None
    ttl_s: float
    timing: Timing

    async def acquire(self) -> bool:
        """With feature "epochs": acquire_lease_grant. Without it: the 0.2 acquire_lease (epoch
        None, nothing journaled). NotImplementedError from a store without leases propagates
        (the caller runs unleased, as 0.2 did). Captures the running loop; t_sent =
        loop.time() before the call; L = t_sent + ttl_s * (1 - rho)."""
        raise NotImplementedError("Effects implementation is pending its owning stage.")

    async def renew(self) -> bool:
        """Success moves L; failure leaves it."""
        raise NotImplementedError("Effects implementation is pending its owning stage.")

    def remaining(self) -> float:
        """L - captured_loop.time(); callable from any thread; -inf before a grant."""
        raise NotImplementedError("Effects implementation is pending its owning stage.")

    def lost(self) -> bool:
        raise NotImplementedError("Effects implementation is pending its owning stage.")

    def first_entries(self) -> list[Entry]:
        """[lease_acquired] with the fixed id of 3.3 until mark_first_written(); [] without an
        epoch."""
        raise NotImplementedError("Effects implementation is pending its owning stage.")

    def mark_first_written(self) -> None:
        raise NotImplementedError("Effects implementation is pending its owning stage.")

    def track(self, fut: asyncio.Future[Any] | concurrent.futures.Future[Any]) -> None:
        """Count a running send (a task or a thread) until it is done; thread-safe; the
        decrement is posted to the captured loop with call_soon_threadsafe from the future's
        done callback."""
        raise NotImplementedError("Effects implementation is pending its owning stage.")

    def running_sends(self) -> int:
        raise NotImplementedError("Effects implementation is pending its owning stage.")

    async def sends_drained(self) -> None:
        """Return when no tracked send runs (M47 seam)."""
        raise NotImplementedError("Effects implementation is pending its owning stage.")

    async def keep(self, *, on_lost: Callable[[str], None]) -> None:
        """Renew every ttl/3; call on_lost(reason) when a renewal returns False or remaining()
        < timing.m_s (F2); keeps renewing while tracked sends run after the owner finished."""
        raise NotImplementedError("Effects implementation is pending its owning stage.")

    async def release(self) -> None:
        """release_lease; callers use LeasedWriter.close, which waits for sends first."""
        raise NotImplementedError("Effects implementation is pending its owning stage.")


class LeasedWriter:
    """Frozen leased-writer contract; implementation is pending the store owner stage."""

    def __init__(
        self,
        store: SessionStore,
        session_id: str,
        *,
        worker: str,
        ttl_s: float,
        purpose: Literal["run", "recovery", "settle", "recheck", "workflow"],
        journal_acquisition: bool | None = None,
    ) -> None:
        """journal_acquisition None: decided after acquisition from status().has_effects; True
        for agents with effects tools and for operators. Timing from Timing.for_lease(ttl_s,
        store.takeover_grace_s, effects=journal_acquisition is True)."""
        raise NotImplementedError("Effects implementation is pending its owning stage.")

    store: SessionStore
    session_id: str
    worker: str
    lease: Lease
    timing: Timing
    leased: bool

    async def open(self) -> bool:
        """Acquire before any read (7.3); False when another owner holds the lease. A store
        without leases: leased=False, True."""
        raise NotImplementedError("Effects implementation is pending its owning stage.")

    async def load(self) -> list[Entry]:
        """Load the journal and rebuild view() from it."""
        raise NotImplementedError("Effects implementation is pending its owning stage.")

    async def append(self, entries: Sequence[Entry]) -> list[int] | None:
        """Every write of this writer, whoever calls it (the run, the runtime's limits,
        run_recovered, a crash-loop run_failed). Prepends lease.first_entries() to the first
        write; append_fenced with the lease owner (fence = not lease.lost()); folds every
        entry it wrote into view(). None only from a 0.2-style store."""
        raise NotImplementedError("Effects implementation is pending its owning stage.")

    async def append_effect(self, entries: Sequence[Entry]) -> list[int]:
        """An effect entry as its own write, never batched: append_fenced with fence =
        lease.remaining() >= timing.m_s; raises JournalOutcomeUnknown and poisons the writer
        when the outcome stays unknown."""
        raise NotImplementedError("Effects implementation is pending its owning stage.")

    def view(self) -> EffectView:
        raise NotImplementedError("Effects implementation is pending its owning stage.")

    async def store_now(self) -> float:
        raise NotImplementedError("Effects implementation is pending its owning stage.")

    async def salt(self) -> bytes:
        raise NotImplementedError("Effects implementation is pending its owning stage.")

    def poisoned(self) -> bool:
        raise NotImplementedError("Effects implementation is pending its owning stage.")

    def track(self, fut: asyncio.Future[Any] | concurrent.futures.Future[Any]) -> None:
        """lease.track."""
        raise NotImplementedError("Effects implementation is pending its owning stage.")

    async def keep(self, *, on_lost: Callable[[str], None]) -> None:
        """lease.keep (the run's or the runtime's renewal task)."""
        raise NotImplementedError("Effects implementation is pending its owning stage.")

    async def close(self) -> None:
        """Wait for sends_drained() while renewing, then release. Never releases early: if the
        wait is cancelled, the rest (renew until drained, release) moves to a detached task
        kept in session._DETACHED, and the CancelledError is re-raised."""
        raise NotImplementedError("Effects implementation is pending its owning stage.")

    wrapped: SessionStore
