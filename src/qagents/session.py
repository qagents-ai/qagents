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
import secrets
import sqlite3
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar, Literal, Self, get_args

from pydantic import BaseModel, ConfigDict, Field

from .effects._claims import Claim, next_claim
from .effects._fold import EffectView, fold_effects
from .effects._kinds import EFFECT_KINDS, LIFECYCLE_KINDS, REQUIRES_EFFECTS, lease_entry_id
from .effects._ports import Timing
from .effects._types import EffectRecord, HeldEffect
from .errors import ClaimConflict, EntryConflict, JournalOutcomeUnknown, LeaseError, SessionError
from .messages import Message, ToolResultPart, Usage


def _owner_token(worker: str) -> str:
    """Return a fresh lease owner for one acquisition by this worker (the M8 seam)."""
    return f"{worker}#{uuid.uuid4().hex}"


class Entry(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    """Stable identity, preserved when an uncertain append is retried."""
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


def _provisional_counts(entry: Entry, later_results: set[str]) -> bool:
    """Only a later result for the same operation commits a provisional snapshot (M21)."""
    return not entry.data.get("provisional", False) or (
        isinstance(entry.data.get("operation_id"), str)
        and entry.data["operation_id"] in later_results
    )


def _restore_message(data: dict[str, Any]) -> Message:
    """Restore one journaled message (M28)."""
    return Message.model_validate(data)


def derive_state(entries: list[Entry], *, session_id: str | None = None) -> SessionState:
    """Replay a journal into the current `SessionState`.

    Args:
        entries: The session's entries, oldest first. Unknown kinds are ignored.

    Returns:
        The state after the last entry. ``values`` is a copy: changing it does
        not change the journal.
    """
    state = SessionState()
    state.effects = fold_effects(entries)
    accepted = []
    closed = set()
    for entry in entries:
        if entry.kind == "run_finished":
            closed.add(entry.run_id)
        if entry.run_id is None or entry.run_id not in closed or entry.kind == "run_finished":
            accepted.append(entry)
    later_results: set[str] = set()
    provisional_counts: dict[str, bool] = {}
    for entry in reversed(accepted):
        if (
            entry.kind == "tool_result"
            and isinstance(entry.data.get("operation_id"), str)
            and entry.data.get("commit_provisional", True)
        ):
            later_results.add(entry.data["operation_id"])
        elif entry.kind == "state_updated":
            provisional_counts[entry.id] = _provisional_counts(entry, later_results)
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
        if kind == "effect_committed" and data.get("state_patch") is not None:
            patch = data["state_patch"]
            values.update(copy.deepcopy(patch.get("set", {})))
            for key in patch.get("unset", []):
                values.pop(key, None)
        if kind == "message" and data.get("message", {}).get("role") == "assistant":
            state.assistant_messages += 1
        if entry.run_id is not None and entry.run_id in finished and kind != "run_finished":
            continue
        if kind == "message":
            state.messages.append(_restore_message(data["message"]))
            state.run_messages += 1
            state.tool_results = {}
            state.pending_approvals = []
            started = {}
            inputs = {}
            checkpoints = {}  # a new message closes the turn whose calls recorded them
            state.open_turn_starts = {}
            if entry.run_id is not None and entry.run_id == state.open_run_id:
                _count_usage(state, data)
                state.run_turns = max(state.run_turns, int(data.get("turn") or 0))
        elif kind == "tool_started":
            state.open_turn_starts.setdefault(data["tool_call_id"], []).append(copy.deepcopy(data))
            operation_id = data.get("operation_id")
            if isinstance(operation_id, str):
                state.tool_attempts[operation_id] = state.tool_attempts.get(operation_id, 0) + 1
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
            waiting.held_record_id = data.get("held_record_id", waiting.held_record_id)
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
            if provisional_counts.get(entry.id, False):
                values = copy.deepcopy(data.get("values", {}))
        elif kind == "run_started":
            if entry.run_id is not None:
                state.run_requires[entry.run_id] = list(data.get("requires", []))
            state.open_run_id = entry.run_id
            state.agent = data.get("agent")
            state.run_messages = 0
            state.run_usage = Usage()
            state.operation_usage = {}
            state.run_turns = 0
            counted = {}
            checkpoints = {}
        elif kind == "run_resumed":
            if entry.run_id is not None:
                state.run_requires[entry.run_id] = list(data.get("requires", []))
        elif kind == "approval_decided":
            operation_id = data.get("operation_id")
            if isinstance(operation_id, str):
                state.decisions[operation_id] = copy.deepcopy(data)
                state.decision_counts[operation_id] = state.decision_counts.get(operation_id, 0) + 1
        elif kind == "memo_recorded":
            state.memos.setdefault(data["operation_id"], {})[data["name"]] = copy.deepcopy(
                data["value"]
            )
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
            state.messages = [_restore_message(item) for item in data["messages"]]
        elif kind == "guardrail_tripped" and data.get("stage") == "output":
            _mark_blocked(state.messages)
    state.started_tools = list(started)
    state.pending_inputs = list(inputs.values())
    if session_id is not None:
        held = {item.id: item for item in state.effects.held(session_id)}
        for waiting in state.pending_inputs:
            waiting.held = held.get(waiting.held_record_id)
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
    lease_epoch: int = 0
    lease_acquired_at: float | None = None
    has_effects: bool = False
    retry_after: float | None = None
    incarnation: str | None = None

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
    if (
        kind in EFFECT_KINDS
        or kind == "lease_acquired"
        or (kind == "run_started" and REQUIRES_EFFECTS in entry.data.get("requires", []))
    ):
        status.has_effects = True
    if entry.run_id is None or kind == "lease_acquired":
        return
    if kind == "run_started":
        status.status = "running"
        status.open_run_id = entry.run_id
        status.agent = entry.data.get("agent")
        status.failures = 0
        status.retry_after = None
    elif status.open_run_id is None or entry.run_id != status.open_run_id:
        return
    elif kind == "run_finished":
        status.status = "idle"
        status.open_run_id = None
        status.failures = 0
        status.retry_after = None
    elif kind == "run_paused":
        status.status = "paused"
    elif kind == "run_failed":
        status.status = "failed"
        status.failures += 1
        status.retry_after = entry.data.get("retry_after")
    else:
        status.status = "running"
        if kind == "run_resumed":
            status.retry_after = None


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
        """Return the store clock, required when the store advertises ``now``."""
        raise NotImplementedError(_unsupported(self, "store time"))

    async def acquire_lease_grant(
        self, session_id: str, owner: str, ttl_s: float
    ) -> LeaseGrant | None:
        """Acquire or renew a lease with a monotonic per-session epoch."""
        raise NotImplementedError(_unsupported(self, "lease epochs"))

    async def get_claim(self, namespace: str, effect: str, key: str) -> Claim | None:
        raise NotImplementedError(_unsupported(self, "claims"))

    async def purge_claims(self, *, older_than: float) -> int:
        """Remove claims last changed before this store-clock timestamp."""
        raise NotImplementedError(_unsupported(self, "claims"))

    async def register_effect(
        self, effect: str, version: str, key_spec_digest: str, request_schema_digest: str
    ) -> Literal["registered", "matches", "key_changed", "schema_changed"]:
        raise NotImplementedError(_unsupported(self, "effect registry"))

    async def salt(self) -> bytes:
        """Return the random salt created once for this store."""
        raise NotImplementedError(_unsupported(self, "salt"))

    @abstractmethod
    async def append(self, session_id: str, entries: list[Entry]) -> list[int] | None:
        """Durably add ``entries`` to the end of the session, in order.

        One batch is atomic. Repeated entry ids return their original positions;
        a partial overlap raises EntryConflict. Validation SessionError means
        nothing was written. Other errors may have lost an acknowledgement and
        require identical retries or read-back through append_fenced. A failed
        call must finish deciding its write before returning, including cancellation.
        Built-in stores return positions; a legacy store may return None.
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

    async def append_leased(
        self, session_id: str, entries: list[Entry], owner: str
    ) -> list[int] | None:
        """Append like `append`, but only while ``owner`` holds the session's lease.

        Entry identities are checked before the owner: a fully repeated batch
        succeeds even after takeover. The lease check and write are atomic, so a worker
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


def _check_grace(grace: float) -> None:
    if not 0 <= grace < float("inf"):
        raise ValueError("takeover_grace_s must be finite and nonnegative.")


def _admit_batch(
    existing_ids: Mapping[str, int], entries: list[Entry], holder: str | None, owner: str | None
) -> Literal["write", "duplicate"]:
    """Check identities before the lease, without changing the journal (M38)."""
    ids = [entry.id for entry in entries]
    if any(not entry_id for entry_id in ids) or len(set(ids)) != len(ids):
        raise EntryConflict("An append needs distinct, nonempty entry ids.")
    present = sum(entry_id in existing_ids for entry_id in ids)
    if present == len(ids):
        return "duplicate"
    if present:
        raise EntryConflict("Only part of this append is already present in the journal.")
    if owner is not None and holder != owner:
        raise LeaseError(_lease_lost("<current>", owner, holder))
    return "write"


def _check_unleased(entry: Entry, status: SessionStatus, owner: str | None) -> None:
    if owner is None and (
        entry.kind in EFFECT_KINDS or (status.has_effects and entry.kind in LIFECYCLE_KINDS)
    ):
        raise SessionError("Effect and effect-session lifecycle entries require a leased writer.")


def _claim_key(entry: Entry) -> tuple[str, str, str]:
    try:
        key = tuple(entry.data[name] for name in ("namespace", "name", "key"))
        if not all(isinstance(value, str) and value for value in key):
            raise ValueError("invalid effect reference")
        return key
    except (KeyError, ValueError) as error:
        raise SessionError("Effect entries need a namespace, name and key.") from error


def _registry_result(old: tuple[str, str] | None, key: str, schema: str) -> str:
    if old is None:
        return "registered"
    if old[0] != key:
        return "key_changed"
    return "matches" if old[1] == schema else "schema_changed"


def _apply_fold(view: EffectView, entry: Entry) -> None:
    try:
        view.apply(entry)
    except (ValueError, TypeError, KeyError) as error:
        raise SessionError("Invalid effect journal entry.") from error


def _reconcile_claim(claim: Claim, view: EffectView, session_id: str, incarnation: str) -> Claim:
    """Only all-attempt fold evidence can release earlier uncertainty."""
    if (claim.session_id, claim.incarnation) != (session_id, incarnation):
        return claim
    operation = view.get(claim.namespace, claim.effect, claim.key)
    if operation is None:
        return claim
    result = claim.model_copy(deep=True)
    if operation.state == "C":
        result.status = "committed"
        result.receipt = copy.deepcopy(operation.receipt)
    elif operation.state == "R":
        result.status = "rejected"
    elif operation.state == "N" and operation.attempts:
        result.status = "not_sent"
    elif operation.state in ("X", "NA", "CU"):
        result.status = "held"
    elif any(attempt.terminal is not None for attempt in operation.attempts):
        result.status = "held"
    return result


def _check_delete(
    entries: list[Entry],
    session_id: str,
    features: frozenset[str],
    force: bool | Literal["forget_effects"],
) -> None:
    if type(force) is not bool and force != "forget_effects":
        raise ValueError("force must be False, True or 'forget_effects'.")
    if "claims" not in features and any(e.kind in EFFECT_KINDS for e in entries):
        if force != "forget_effects":
            raise SessionError("This store has no claims; use force='forget_effects' explicitly.")
    if not force and fold_effects(entries).held(session_id):
        raise SessionError("The session has held effects; use force=True to retain tombstones.")


def _reappend_entries(entries: list[Entry]) -> list[Entry]:
    """Preserve every entry identity and payload during uncertain append retries (M19)."""
    return entries


async def append_fenced(
    store: SessionStore,
    session_id: str,
    entries: list[Entry],
    owner: str | None,
    *,
    fence: Callable[[], bool],
    attempts: int = 3,
) -> list[int] | None:
    """Resolve uncertain writes by identical retries and read-back (SPEC 7.7.2–3)."""
    if type(attempts) is not int or attempts < 0:
        raise ValueError("attempts must be a nonnegative integer.")

    async def write(batch):
        if owner is None:
            return await store.append(session_id, batch)
        return await store.append_leased(session_id, batch, owner)

    cancelled = None
    try:
        return await write(entries)
    except (LeaseError, EntryConflict, ClaimConflict):
        raise
    except SessionError as error:
        if not isinstance(error, JournalOutcomeUnknown):
            raise
        last_error = error
    except BaseException as error:
        if not isinstance(error, (Exception, asyncio.CancelledError)):
            raise
        last_error = error
        if isinstance(error, asyncio.CancelledError):
            cancelled = error

    async def decide():
        nonlocal last_error, cancelled
        for _ in range(attempts):
            if not fence():
                break
            try:
                return await write(_reappend_entries(entries))
            except (Exception, asyncio.CancelledError) as error:
                last_error = error
                if isinstance(error, asyncio.CancelledError):
                    cancelled = error
        try:
            found = {entry.id: entry.seq for entry in await store.load(session_id)}
        except (Exception, asyncio.CancelledError) as error:
            raise JournalOutcomeUnknown("Append outcome could not be read back.") from error
        present = [entry.id in found for entry in entries]
        if all(present):
            return [found[entry.id] for entry in entries]
        if any(present):
            raise JournalOutcomeUnknown("Read-back found only part of an atomic append.")
        raise last_error

    deciding = asyncio.create_task(decide())
    while not deciding.done():
        try:
            await asyncio.shield(deciding)
        except asyncio.CancelledError as error:
            cancelled = error
        except Exception:
            break
    if cancelled is not None:
        if not deciding.cancelled():
            error = deciding.exception()
            if isinstance(error, JournalOutcomeUnknown):
                raise error
        raise cancelled
    return deciding.result()


_DETACHED: set[asyncio.Task[Any]] = set()


def _detach(task: asyncio.Task[Any]) -> None:
    _DETACHED.add(task)

    def finished(done):
        _DETACHED.discard(done)
        if not done.cancelled():
            done.exception()

    task.add_done_callback(finished)


class Lease:
    """One acquisition, its monotonic self-fence, and the sends it must outlive."""

    owner: str
    worker: str
    epoch: int | None
    acquired_at: float | None
    ttl_s: float
    timing: Timing

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
        _check_ttl(ttl_s)
        self.store, self.session_id, self.worker = store, session_id, worker
        self.ttl_s, self.timing, self.purpose = ttl_s, timing, purpose
        self.owner = _owner_token(worker)
        self.epoch: int | None = None
        self.acquired_at: float | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._deadline = float("-inf")
        self._lost = False
        self._first_written = False
        self._counter_lock = threading.Lock()
        self._tracked: set[Any] = set()
        self._drained: asyncio.Event | None = None

    async def acquire(self) -> bool:
        self._loop = asyncio.get_running_loop()
        self._drained = asyncio.Event()
        self._drained.set()
        sent_at = self._loop.time()
        if "epochs" in self.store.features:
            grant = await self.store.acquire_lease_grant(self.session_id, self.owner, self.ttl_s)
            if grant is None:
                return False
            self.epoch, self.acquired_at = grant.epoch, grant.acquired_at
        elif not await self.store.acquire_lease(self.session_id, self.owner, self.ttl_s):
            return False
        self._deadline = sent_at + self.ttl_s * (1 - self.timing.rho)
        return True

    async def renew(self) -> bool:
        if self._loop is None:
            return False
        sent_at = self._loop.time()
        renewed = await self.store.renew_lease(self.session_id, self.owner, self.ttl_s)
        if renewed:
            self._deadline = sent_at + self.ttl_s * (1 - self.timing.rho)
        return renewed

    def remaining(self) -> float:
        return self._deadline - self._loop.time() if self._loop else float("-inf")

    def lost(self) -> bool:
        return self._lost or self.remaining() <= 0

    def first_entries(self) -> list[Entry]:
        if self.epoch is None or self._first_written:
            return []
        return [
            Entry(
                id=lease_entry_id(self.owner),
                kind="lease_acquired",
                data={
                    "epoch": self.epoch,
                    "acquired_at": self.acquired_at,
                    "worker": self.worker,
                    "purpose": self.purpose,
                },
            )
        ]

    def mark_first_written(self) -> None:
        self._first_written = True

    def track(self, fut: asyncio.Future[Any] | concurrent.futures.Future[Any]) -> None:
        if self._loop is None or self._drained is None:
            raise SessionError("Acquire the lease before tracking a send.")
        with self._counter_lock:
            if fut in self._tracked:
                return
            self._tracked.add(fut)
        self._loop.call_soon_threadsafe(self._drained.clear)

        def ended(_):
            self._loop.call_soon_threadsafe(self._untrack, fut)

        fut.add_done_callback(ended)

    def _untrack(self, fut) -> None:
        with self._counter_lock:
            self._tracked.discard(fut)
            empty = not self._tracked
        if empty and self._drained is not None:
            self._drained.set()

    def running_sends(self) -> int:
        with self._counter_lock:
            return len(self._tracked)

    async def sends_drained(self) -> None:
        if self._drained is None:
            return
        while self.running_sends():
            self._drained.clear()
            await self._drained.wait()

    async def keep(self, *, on_lost: Callable[[str], None]) -> None:
        notified = False
        renewing = None

        def lose(reason):
            nonlocal notified
            self._lost = True
            if not notified:
                notified = True
                on_lost(reason)

        try:
            while True:
                remaining = self.remaining() - self.timing.m_s
                if self._lost or remaining <= 0:
                    lose("Lease self-fence reached its safety margin.")
                    if not self.running_sends():
                        return
                # Loss stops new sends permanently, but existing sends and their
                # completion watchers still need renewal until they finish.
                await asyncio.sleep(
                    self.ttl_s / 3 if self._lost else min(self.ttl_s / 3, remaining)
                )
                remaining = self.remaining() - self.timing.m_s
                if remaining <= 0 and not self._lost:
                    continue
                if renewing is None:
                    renewing = asyncio.create_task(self.renew())
                timeout = self.ttl_s / 3 if self._lost else remaining
                done, _ = await asyncio.wait({renewing}, timeout=timeout)
                if not done:
                    lose("Lease renewal did not finish before the self-fence.")
                    continue  # retain one in-flight renewal, not an unbounded queue
                try:
                    renewed = renewing.result()
                except (Exception, asyncio.CancelledError):
                    renewed = False
                renewing = None
                if not renewed or self.remaining() < self.timing.m_s:
                    lose("Lease renewal failed or arrived after the self-fence.")
        finally:
            if renewing is not None:
                _detach(renewing)

    async def release(self) -> None:
        await self.store.release_lease(self.session_id, self.owner)
        self._lost = True


class LeasedWriter:
    """The shared journal writer for runs, operators, runtimes and workflows."""

    store: SessionStore
    session_id: str
    worker: str
    lease: Lease
    timing: Timing
    leased: bool
    wrapped: SessionStore

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
        self.store, self.wrapped, self.session_id, self.worker = store, store, session_id, worker
        self.timing = Timing.for_lease(
            ttl_s, store.takeover_grace_s, effects=journal_acquisition is True
        )
        self.lease = Lease(
            store, session_id, worker=worker, ttl_s=ttl_s, purpose=purpose, timing=self.timing
        )
        self.leased = True
        self._journal_acquisition = journal_acquisition
        self._view = EffectView()
        self._poisoned = False
        self._opened = False
        self._closed = False
        self._append_lock = asyncio.Lock()
        self._closing: asyncio.Task[None] | None = None

    async def open(self) -> bool:
        if self._opened or self._closed:
            raise SessionError("A LeasedWriter can be opened only once.")
        try:
            if not await self.lease.acquire():
                return False
        except NotImplementedError:
            if "leases" in self.store.features or "epochs" in self.store.features:
                raise
            self.leased = False
        try:
            if self._journal_acquisition is None:
                try:
                    self._journal_acquisition = (
                        await self.store.status(self.session_id)
                    ).has_effects
                except NotImplementedError:
                    self._journal_acquisition = False
            self.timing = Timing.for_lease(
                self.lease.ttl_s, self.store.takeover_grace_s, effects=self._journal_acquisition
            )
            self.lease.timing = self.timing
            self._opened = True
            return True
        except BaseException:
            if self.leased:
                await self.lease.release()
            raise

    def _ready_to_write(self, *, completion: bool = False) -> None:
        if not self._opened or self._closed or (self._closing is not None and not completion):
            raise SessionError("Open this writer before use; a closing writer cannot write.")
        if self._poisoned:
            raise JournalOutcomeUnknown("This writer is poisoned after an unresolved append.")

    async def load(self) -> list[Entry]:
        self._ready_to_write()
        async with self._append_lock:
            entries = await self.store.load(self.session_id)
            self._view = fold_effects(entries)
            return entries

    async def _append(self, entries: Sequence[Entry], *, effect: bool) -> list[int] | None:
        completion = effect and bool(entries) and all(self._admitted_completion(e) for e in entries)
        self._ready_to_write(completion=completion)
        async with self._append_lock:
            self._ready_to_write(completion=completion)
            if effect and not self.leased:
                raise SessionError("Effect writes require a lease.")
            first = self.lease.first_entries() if self._journal_acquisition else []
            batch = [*first, *entries]
            try:
                seqs = await append_fenced(
                    self.store,
                    self.session_id,
                    batch,
                    self.lease.owner if self.leased else None,
                    fence=lambda: (
                        self.lease.remaining() >= self.timing.m_s
                        if effect
                        else not self.lease.lost()
                    ),
                )
            except JournalOutcomeUnknown:
                self._poisoned = True
                raise
            except asyncio.CancelledError:
                # append_fenced resolved the outcome before preserving cancellation.
                # Refresh acquisition bookkeeping so a later cleanup write cannot
                # accidentally combine an existing acquisition with fresh entry ids.
                reading = asyncio.create_task(self.store.load(self.session_id))
                while not reading.done():
                    try:
                        await asyncio.shield(reading)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                try:
                    loaded = reading.result()
                    self._view = fold_effects(loaded)
                    if first and any(e.id == first[0].id for e in loaded):
                        self.lease.mark_first_written()
                except (Exception, asyncio.CancelledError) as error:
                    self._poisoned = True
                    raise JournalOutcomeUnknown(
                        "Cancelled append could not refresh the view."
                    ) from error
                raise
            if seqs is not None:
                for entry, seq in zip(batch, seqs, strict=True):
                    self._view.apply(entry.model_copy(update={"seq": seq}))
            else:
                self._view = fold_effects(await self.store.load(self.session_id))
            if first:
                self.lease.mark_first_written()
            return seqs[len(first) :] if seqs is not None else None

    def _admitted_completion(self, entry: Entry) -> bool:
        if entry.kind not in {"effect_send_ended", "effect_committed", "effect_failed"}:
            return False
        data = entry.data
        operation = self._view.get(data.get("namespace"), data.get("name"), data.get("key"))
        return (
            operation is not None
            and operation.record_id == data.get("record_id")
            and any(
                attempt.number == data.get("attempt")
                and attempt.intent_seq == data.get("intent_seq")
                for attempt in operation.attempts
            )
        )

    async def append(self, entries: Sequence[Entry]) -> list[int] | None:
        return await self._append(entries, effect=False)

    async def append_effect(self, entries: Sequence[Entry]) -> list[int]:
        seqs = await self._append(entries, effect=True)
        if seqs is None:
            raise SessionError("Effect writes require entry-id acknowledgements.")
        return seqs

    def view(self) -> EffectView:
        return self._view

    async def store_now(self) -> float:
        return await self.store.now()

    async def salt(self) -> bytes:
        return await self.store.salt()

    def poisoned(self) -> bool:
        return self._poisoned

    def track(self, fut: asyncio.Future[Any] | concurrent.futures.Future[Any]) -> None:
        """Track sends and their already-created completion watchers before closing."""
        self.lease.track(fut)

    async def keep(self, *, on_lost: Callable[[str], None]) -> None:
        if self.leased:
            await self.lease.keep(on_lost=on_lost)

    async def _close(self) -> None:
        keeper = (
            asyncio.create_task(self.lease.keep(on_lost=lambda _: None)) if self.leased else None
        )
        try:
            await self.lease.sends_drained()
            async with self._append_lock:
                if self.leased:
                    await self.lease.release()
                self._closed = True
        finally:
            if keeper is not None:
                keeper.cancel()
                await asyncio.gather(keeper, return_exceptions=True)

    async def close(self) -> None:
        if not self._opened or self._closed:
            return
        if self._closing is None:
            self._closing = asyncio.create_task(self._close())
        try:
            await asyncio.shield(self._closing)
        except asyncio.CancelledError:
            _detach(self._closing)
            raise


def _lease_lost(session_id: str, owner: str, holder: str | None) -> str:
    who = f"worker {holder!r} holds it now" if holder else "nobody holds it now"
    return (
        f"Worker {owner!r} no longer holds the lease on session {session_id!r} ({who}), so "
        "its entries were not written. Another worker may be driving this run; stop this one."
    )


class InMemoryStore(SessionStore):
    """Keeps journals in a dict. For tests and for sessions that need not survive a restart.

    Run status and leases use an injectable store clock (``time.time`` by default)
    and a one-second takeover grace.
    They only coordinate tasks of one process, since nothing else can see
    the store.
    """

    features = FEATURES

    def __init__(
        self, *, clock: Callable[[], float] | None = None, takeover_grace_s: float = 1.0
    ) -> None:
        _check_grace(takeover_grace_s)
        self._clock = clock or time.time
        self.takeover_grace_s = takeover_grace_s
        self._sessions: dict[str, list[Entry]] = {}
        self._status: dict[str, SessionStatus] = {}
        self._epochs: dict[str, int] = {}
        self._claims: dict[tuple[str, str, str], Claim] = {}
        self._registry: dict[tuple[str, str], tuple[str, str]] = {}
        self._salt = secrets.token_bytes(32)

    async def now(self) -> float:
        return self._clock()

    def _row(self, session_id: str) -> SessionStatus:
        return self._status.setdefault(session_id, SessionStatus(session_id=session_id))

    def _write(self, session_id: str, entries: list[Entry], owner: str | None = None) -> list[int]:
        journal = self._sessions.get(session_id, [])
        row = self._status.get(session_id, SessionStatus(session_id=session_id)).model_copy()
        ids = {entry.id: entry.seq for entry in journal}
        if _admit_batch(ids, entries, row.lease_owner, owner) == "duplicate":
            return [ids[entry.id] for entry in entries]
        now = self._clock()
        row.incarnation = row.incarnation or uuid.uuid4().hex
        claims = self._claims.copy()
        view = fold_effects(journal) if any(e.kind in EFFECT_KINDS for e in entries) else None
        added = []
        for offset, entry in enumerate(entries, start=1):
            _check_unleased(entry, row, owner)
            _entry_json(entry)  # validate the entire batch before publishing it
            stored = entry.model_copy(deep=True, update={"seq": len(journal) + offset})
            added.append(stored)
            if view is not None:
                _apply_fold(view, stored)
            if entry.kind in EFFECT_KINDS:
                key = _claim_key(entry)
                try:
                    claim = next_claim(
                        claims.get(key),
                        stored,
                        session_id=session_id,
                        incarnation=row.incarnation,
                        now=now,
                    )
                except ValueError as error:
                    raise SessionError("Invalid effect journal entry.") from error
                if claim is not None:
                    claims[key] = _reconcile_claim(claim, view, session_id, row.incarnation)
            advance_status(row, stored)
        row.updated_at = now
        self._sessions[session_id] = journal + added
        self._status[session_id] = row
        self._claims = claims
        return [entry.seq for entry in added]

    async def append(self, session_id: str, entries: list[Entry]) -> list[int]:
        return self._write(session_id, entries)

    async def append_leased(self, session_id: str, entries: list[Entry], owner: str) -> list[int]:
        return self._write(session_id, entries, owner)

    async def load(self, session_id: str) -> list[Entry]:
        return [entry.model_copy(deep=True) for entry in self._sessions.get(session_id, [])]

    async def delete(
        self, session_id: str, *, force: bool | Literal["forget_effects"] = False
    ) -> None:
        _check_delete(self._sessions.get(session_id, []), session_id, self.features, force)
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
        return await self.acquire_lease_grant(session_id, owner, ttl_s) is not None

    async def acquire_lease_grant(
        self, session_id: str, owner: str, ttl_s: float
    ) -> LeaseGrant | None:
        _check_ttl(ttl_s)
        now = self._clock()
        row = self._row(session_id)
        if (
            row.lease_owner not in (None, owner)
            and (row.lease_expires_at or 0.0) + self.takeover_grace_s > now
        ):
            return None
        if row.lease_owner != owner or row.lease_epoch == 0:
            row.lease_epoch = self._epochs.get(session_id, 0) + 1
            self._epochs[session_id] = row.lease_epoch
            row.lease_acquired_at = now
        row.lease_owner, row.lease_expires_at = owner, now + ttl_s
        return LeaseGrant(
            owner=owner,
            epoch=row.lease_epoch,
            acquired_at=row.lease_acquired_at,
            expires_at=row.lease_expires_at,
        )

    async def renew_lease(self, session_id: str, owner: str, ttl_s: float) -> bool:
        _check_ttl(ttl_s)
        row = self._status.get(session_id)
        if row is None or row.lease_owner != owner:
            return False
        row.lease_expires_at = self._clock() + ttl_s
        return True

    async def release_lease(self, session_id: str, owner: str) -> None:
        row = self._status.get(session_id)
        if row is not None and row.lease_owner == owner:
            row.lease_owner, row.lease_expires_at = None, None

    async def get_claim(self, namespace: str, effect: str, key: str) -> Claim | None:
        claim = self._claims.get((namespace, effect, key))
        return claim.model_copy(deep=True) if claim is not None else None

    async def purge_claims(self, *, older_than: float) -> int:
        keys = [key for key, claim in self._claims.items() if claim.updated_at < older_than]
        for key in keys:
            del self._claims[key]
        return len(keys)

    async def register_effect(
        self, effect: str, version: str, key_spec_digest: str, request_schema_digest: str
    ) -> Literal["registered", "matches", "key_changed", "schema_changed"]:
        key = effect, version
        result = _registry_result(self._registry.get(key), key_spec_digest, request_schema_digest)
        if result == "registered":
            self._registry[key] = key_spec_digest, request_schema_digest
        return result

    async def salt(self) -> bytes:
        return self._salt


_SQLITE_SCHEMA_VERSION = 2
"""Stored in ``PRAGMA user_version``. 0 is a 0.1 database: the entries table only."""

_SQLITE_STATUS_COLUMNS = (
    "session_id, status, open_run_id, agent, failures, updated_at, lease_owner, lease_expires_at,"
    " lease_epoch, lease_acquired_at, has_effects, retry_after, incarnation"
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

    features = FEATURES

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], float] | None = None,
        takeover_grace_s: float = 1.0,
    ) -> None:
        _check_grace(takeover_grace_s)
        self._clock = clock or time.time
        self.takeover_grace_s = takeover_grace_s
        self.path = str(path)
        self._lock = threading.Lock()
        self._connection: sqlite3.Connection | None = None
        self._executor: ThreadPoolExecutor | None = None

    async def now(self) -> float:
        return self._clock()

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
            if version < 2:
                connection.execute("ALTER TABLE entries ADD COLUMN id TEXT")
                connection.execute("UPDATE entries SET id = 'seq:' || seq")
                connection.execute("CREATE UNIQUE INDEX entries_by_id ON entries (session_id, id)")
                for definition in (
                    "lease_epoch INTEGER NOT NULL DEFAULT 0",
                    "lease_acquired_at REAL",
                    "has_effects INTEGER NOT NULL DEFAULT 0",
                    "retry_after REAL",
                    "incarnation TEXT",
                ):
                    connection.execute(f"ALTER TABLE sessions ADD COLUMN {definition}")
                for (session_id,) in connection.execute(
                    "SELECT session_id FROM sessions WHERE updated_at IS NOT NULL"
                ).fetchall():
                    connection.execute(
                        "UPDATE sessions SET incarnation = ? WHERE session_id = ?",
                        (uuid.uuid4().hex, session_id),
                    )
                connection.execute(
                    "CREATE TABLE lease_epochs (session_id TEXT PRIMARY KEY,"
                    " epoch INTEGER NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE claims (namespace TEXT, effect TEXT, key TEXT,"
                    " data TEXT NOT NULL,"
                    " updated_at REAL NOT NULL, PRIMARY KEY(namespace, effect, key))"
                )
                connection.execute(
                    "CREATE TABLE effect_registry (effect TEXT, version TEXT, key_spec_digest TEXT,"
                    " request_schema_digest TEXT, PRIMARY KEY(effect, version))"
                )
                connection.execute("CREATE TABLE store_meta (key TEXT PRIMARY KEY, value BLOB)")
                connection.execute(
                    "INSERT INTO store_meta VALUES ('salt', ?)", (secrets.token_bytes(32),)
                )
            if version < 1:
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
            " CASE WHEN kind IN ('run_started', 'run_failed') THEN data END"
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
            " (session_id, status, open_run_id, agent, failures, updated_at,"
            " retry_after, incarnation)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row.session_id,
                    row.status,
                    row.open_run_id,
                    row.agent,
                    row.failures,
                    row.updated_at,
                    row.retry_after,
                    uuid.uuid4().hex,
                )
                for row in rows
            ],
        )

    def _append(self, session_id: str, entries: list[Entry], owner: str | None = None) -> list[int]:
        with self._lock:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            try:
                found = connection.execute(
                    f"SELECT {_SQLITE_STATUS_COLUMNS} FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                status = _status_row(found) if found else SessionStatus(session_id=session_id)
                ids = dict(
                    connection.execute(
                        "SELECT id, seq FROM entries WHERE session_id = ?", (session_id,)
                    )
                )
                if _admit_batch(ids, entries, status.lease_owner, owner) == "duplicate":
                    connection.execute("COMMIT")
                    return [ids[entry.id] for entry in entries]
                status.incarnation = status.incarnation or uuid.uuid4().hex
                now = self._clock()
                (last,) = connection.execute(
                    "SELECT COALESCE(MAX(seq), 0) FROM entries WHERE session_id = ?", (session_id,)
                ).fetchone()
                rows = []
                view = (
                    fold_effects(self._load_locked(connection, session_id))
                    if any(e.kind in EFFECT_KINDS for e in entries)
                    else None
                )
                for offset, entry in enumerate(entries, start=1):
                    _check_unleased(entry, status, owner)
                    rows.append(
                        (
                            session_id,
                            last + offset,
                            entry.ts,
                            entry.run_id,
                            entry.kind,
                            _entry_json(entry),
                            entry.id,
                        )
                    )
                    if view is not None:
                        _apply_fold(view, entry.model_copy(update={"seq": last + offset}))
                    if entry.kind in EFFECT_KINDS:
                        key = _claim_key(entry)
                        existing = connection.execute(
                            "SELECT data FROM claims WHERE namespace = ?"
                            " AND effect = ? AND key = ?",
                            key,
                        ).fetchone()
                        try:
                            claim = next_claim(
                                Claim.model_validate_json(existing[0]) if existing else None,
                                entry,
                                session_id=session_id,
                                incarnation=status.incarnation,
                                now=now,
                            )
                        except ValueError as error:
                            raise SessionError("Invalid effect journal entry.") from error
                        if claim is not None:
                            claim = _reconcile_claim(claim, view, session_id, status.incarnation)
                            connection.execute(
                                "INSERT INTO claims VALUES (?, ?, ?, ?, ?)"
                                " ON CONFLICT(namespace, effect, key) DO UPDATE SET"
                                " data = excluded.data, updated_at = excluded.updated_at",
                                (*key, claim.model_dump_json(), claim.updated_at),
                            )
                    advance_status(status, entry)
                connection.executemany(
                    "INSERT INTO entries (session_id, seq, ts, run_id, kind, data, id)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
                connection.execute(
                    "INSERT INTO sessions"
                    " (session_id, status, open_run_id, agent, failures, updated_at, incarnation,"
                    " has_effects, retry_after)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (session_id) DO UPDATE SET"
                    " status = excluded.status, open_run_id = excluded.open_run_id,"
                    " agent = excluded.agent, failures = excluded.failures,"
                    " updated_at = excluded.updated_at, incarnation = excluded.incarnation,"
                    " has_effects = excluded.has_effects, retry_after = excluded.retry_after",
                    (
                        session_id,
                        status.status,
                        status.open_run_id,
                        status.agent,
                        status.failures,
                        now,
                        status.incarnation,
                        status.has_effects,
                        status.retry_after,
                    ),
                )
                connection.execute("COMMIT")
                return list(range(last + 1, last + len(entries) + 1))
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    @staticmethod
    def _load_locked(connection: sqlite3.Connection, session_id: str) -> list[Entry]:
        rows = connection.execute(
            "SELECT seq, ts, run_id, kind, data, id FROM entries WHERE session_id = ? ORDER BY seq",
            (session_id,),
        ).fetchall()
        return [
            Entry(seq=seq, ts=ts, run_id=run_id, kind=kind, data=json.loads(data), id=entry_id)
            for seq, ts, run_id, kind, data, entry_id in rows
        ]

    def _load(self, session_id: str) -> list[Entry]:
        with self._lock:
            return self._load_locked(self._connect(), session_id)

    def _delete(self, session_id: str, force: bool | Literal["forget_effects"] = False) -> None:
        with self._lock:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            try:
                _check_delete(
                    self._load_locked(connection, session_id), session_id, self.features, force
                )
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

    def _acquire(self, session_id: str, owner: str, ttl_s: float) -> LeaseGrant | None:
        with self._lock:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            try:
                now = self._clock()  # after both locks, never before waiting for a writer
                found = connection.execute(
                    f"SELECT {_SQLITE_STATUS_COLUMNS} FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                row = _status_row(found) if found else SessionStatus(session_id=session_id)
                if (
                    row.lease_owner not in (None, owner)
                    and (row.lease_expires_at or 0.0) + self.takeover_grace_s > now
                ):
                    connection.execute("COMMIT")
                    return None
                if row.lease_owner != owner or row.lease_epoch == 0:
                    old = connection.execute(
                        "SELECT epoch FROM lease_epochs WHERE session_id = ?", (session_id,)
                    ).fetchone()
                    row.lease_epoch = (old[0] if old else 0) + 1
                    row.lease_acquired_at = now
                    connection.execute(
                        "INSERT INTO lease_epochs VALUES (?, ?) ON CONFLICT(session_id)"
                        " DO UPDATE SET epoch = excluded.epoch",
                        (session_id, row.lease_epoch),
                    )
                connection.execute(
                    "INSERT INTO sessions (session_id, lease_owner, lease_expires_at, lease_epoch,"
                    " lease_acquired_at) VALUES (?, ?, ?, ?, ?)"
                    " ON CONFLICT(session_id) DO UPDATE SET lease_owner = excluded.lease_owner,"
                    " lease_expires_at = excluded.lease_expires_at,"
                    " lease_epoch = excluded.lease_epoch,"
                    " lease_acquired_at = excluded.lease_acquired_at",
                    (session_id, owner, now + ttl_s, row.lease_epoch, row.lease_acquired_at),
                )
                grant = LeaseGrant(
                    owner=owner,
                    epoch=row.lease_epoch,
                    acquired_at=row.lease_acquired_at,
                    expires_at=now + ttl_s,
                )
                connection.execute("COMMIT")
                return grant
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    def _renew(self, session_id: str, owner: str, ttl_s: float) -> bool:
        with self._lock:
            cursor = self._connect().execute(
                "UPDATE sessions SET lease_expires_at = ? WHERE session_id = ? AND lease_owner = ?",
                (self._clock() + ttl_s, session_id, owner),
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

    async def append(self, session_id: str, entries: list[Entry]) -> list[int]:
        return await self._call(self._append, session_id, entries)

    async def append_leased(self, session_id: str, entries: list[Entry], owner: str) -> list[int]:
        return await self._call(self._append, session_id, entries, owner)

    async def load(self, session_id: str) -> list[Entry]:
        return await self._call(self._load, session_id)

    async def delete(
        self, session_id: str, *, force: bool | Literal["forget_effects"] = False
    ) -> None:
        await self._call(self._delete, session_id, force)

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
        return await self.acquire_lease_grant(session_id, owner, ttl_s) is not None

    async def acquire_lease_grant(
        self, session_id: str, owner: str, ttl_s: float
    ) -> LeaseGrant | None:
        _check_ttl(ttl_s)
        return await self._call(self._acquire, session_id, owner, ttl_s)

    async def renew_lease(self, session_id: str, owner: str, ttl_s: float) -> bool:
        _check_ttl(ttl_s)
        return await self._call(self._renew, session_id, owner, ttl_s)

    async def release_lease(self, session_id: str, owner: str) -> None:
        await self._call(self._release, session_id, owner)

    def _get_claim(self, namespace: str, effect: str, key: str) -> Claim | None:
        with self._lock:
            found = (
                self._connect()
                .execute(
                    "SELECT data FROM claims WHERE namespace = ? AND effect = ? AND key = ?",
                    (namespace, effect, key),
                )
                .fetchone()
            )
        return Claim.model_validate_json(found[0]) if found else None

    async def get_claim(self, namespace: str, effect: str, key: str) -> Claim | None:
        return await self._call(self._get_claim, namespace, effect, key)

    def _purge_claims(self, older_than: float) -> int:
        with self._lock:
            return (
                self._connect()
                .execute("DELETE FROM claims WHERE updated_at < ?", (older_than,))
                .rowcount
            )

    async def purge_claims(self, *, older_than: float) -> int:
        return await self._call(self._purge_claims, older_than)

    def _register_effect(self, effect: str, version: str, key: str, schema: str) -> str:
        with self._lock:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            try:
                old = connection.execute(
                    "SELECT key_spec_digest, request_schema_digest FROM effect_registry"
                    " WHERE effect = ? AND version = ?",
                    (effect, version),
                ).fetchone()
                result = _registry_result(old, key, schema)
                if result == "registered":
                    connection.execute(
                        "INSERT INTO effect_registry VALUES (?, ?, ?, ?)",
                        (effect, version, key, schema),
                    )
                connection.execute("COMMIT")
                return result
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    async def register_effect(
        self, effect: str, version: str, key_spec_digest: str, request_schema_digest: str
    ) -> Literal["registered", "matches", "key_changed", "schema_changed"]:
        return await self._call(
            self._register_effect, effect, version, key_spec_digest, request_schema_digest
        )

    def _read_salt(self) -> bytes:
        with self._lock:
            return bytes(
                self._connect()
                .execute("SELECT value FROM store_meta WHERE key = 'salt'")
                .fetchone()[0]
            )

    async def salt(self) -> bytes:
        return await self._call(self._read_salt)

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
        pending = loop.run_in_executor(self._executor, functools.partial(function, *args))
        cancelled = None
        while not pending.done():
            try:
                await asyncio.shield(pending)
            except asyncio.CancelledError as error:
                cancelled = error
        # A cancelled caller may read back as soon as this call returns. The worker
        # must have decided its write before that happens (SPEC 7.7.3).
        if cancelled is not None:
            if not pending.cancelled():
                pending.exception()  # consume a worker failure without replacing cancellation
            raise cancelled
        return pending.result()


def _entry_json(entry: Entry) -> str:
    """The JSON of an entry's data, with non-ASCII text kept as it is.

    A string with a lone surrogate (half of an emoji cut by another service)
    cannot be encoded as UTF-8, so such an entry is written with ASCII escapes,
    which bring the same string back when it is read.
    """
    try:
        text = json.dumps(entry.data, ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise SessionError("Entry data must contain finite JSON values.") from error
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
    session_id, status, open_run_id, agent, failures, updated_at, owner, expires, *extra = row
    return SessionStatus(
        session_id=session_id,
        status=status,
        open_run_id=open_run_id,
        agent=agent,
        failures=failures,
        updated_at=updated_at,
        lease_owner=owner,
        lease_expires_at=expires,
        **dict(
            zip(
                ("lease_epoch", "lease_acquired_at", "has_effects", "retry_after", "incarnation"),
                extra,
                strict=False,
            )
        ),
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
        return derive_state(await self.entries(), session_id=self.id)

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
        records = fold_effects(await self.entries()).records(self.id)
        wanted = (
            {status} if isinstance(status, str) else set(status) if status is not None else None
        )
        return [
            record
            for record in records
            if (wanted is None or record.status in wanted)
            and (namespace is None or record.namespace == namespace)
        ]

    async def clear(self, *, force: bool | Literal["forget_effects"] = False) -> None:
        await self._resolve()
        if force is False:
            await self.store.delete(self.id)
        else:
            await self.store.delete(self.id, force=force)

    def __repr__(self) -> str:
        return f"Session(id={self.id!r}, store={type(self.store).__name__})"
