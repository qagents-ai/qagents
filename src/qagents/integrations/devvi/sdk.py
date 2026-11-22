"""Devvi's SDK JSON-RPC transport (``devvi --profile sdk``), spoken natively with asyncio.

The wire (``packages/sdk/protocol/src/types.ts``, newline-delimited JSON-RPC 2.0 on stdio):

* ``initialize {cwd, provider, model, reasoningEffort?, maxTokens?, approvalRequests}``:
  once per process; every session of the process uses that model route.
* ``session/resume {sessionId}``: before every prompt, to continue a stored session or
  learn that it is new (``session "<id>" not found``).
* ``session/prompt {sessionId, contentBlocks}`` -> ``{messageId}``: an unknown id creates
  the session, so qagents chooses the ids.
* ``session/working-directory/get {sessionId}`` -> ``{cwd}``: where a resumed session
  works, when its log cannot say (see "Workspaces" below).
* ``session/wait {sessionId}``: after the root is idle, wait for subagents.
* ``shutdown``: at the end of a call, and to cancel (there is no per-turn cancel).
* From Devvi, ``approval/request {sessionId, toolName, callId?, reason?}``: decided by
  qagents (`PermissionBridge`) on the arguments of that session's own call ``callId``,
  answered ``allowed-once``, ``rejected`` or ``cancelled``. A request about a call the
  session has not started is answered ``cancelled`` without a decision (``external_event``
  ``approval_refused``): Devvi records every call before it asks about it.
* From Devvi, the notifications ``session.event``, ``session.status``,
  ``subagent.started`` and ``subagent.finished``: mapped by `SessionEventMapper`.

A turn: ``session/resume`` (or learn the session is new), check that the session works in
this workspace, journal the log's length (``BASELINE_CHECKPOINT``), ``begin()`` (journals
the session id), ``session/prompt``, then wait for ``session.status`` ``idle`` of the root
after the ``agent/inbox/spliced`` event that received the prompt's ``messageId`` (as the
Python SDK's ``run`` does), then ``session/wait``. ``user-question/ask`` is not
advertised, so Devvi answers its own question tool with "no provider".

Workspaces: Devvi's ``session/resume`` takes no ``cwd``. It restores the stored session
with its own directory, and its log store finds a session id in any project folder of the
home, which every `DevviAgent` with the same home shares. So a session id passed as
``session=`` (by your code, or by the calling model for a session that an ``as_tool()``
tool of the same name opened) could make this agent work in another agent's workspace,
outside this agent's workspace lock, change tracking and rules. Before resuming, the
stored log's header ``cwd`` is compared with the turn's workspace (``normcase`` and
``realpath``); a session of another workspace is refused without being opened
(``DEVVI_OTHER_WORKSPACE``). When the log is not where qagents expects it, Devvi's own
answer to ``session/working-directory/get`` decides after the resume; when neither can
say, the session is refused too (``DEVVI_WORKSPACE_UNKNOWN``) and the process that opened
it is stopped.

Shape: one Devvi process per call (``process_per_call=True``, the default), because the SDK
has no per-turn cancel (abandoning a turn means stopping the process) and one process
serves one model route. ``process_per_call=False`` keeps one process per workspace between
calls; a cancel still stops it. The pool looks after each process on its own: a dead one
is replaced on the next call in its workspace, and a turn that breaks or is abandoned stops
only its own process (``manages_connections``), never those serving other workspaces.

Crash re-attach (design 2.8): session ids are ``qa-<sha256(operation id)>``, so a call
repeated after a crash uses the same session. The repeat resumes it and reads its JSONL
log (`logs`): a turn the first attempt finished is replayed from the log (events, usage,
final text) and not prompted again, and of its usage the run counts only what its journal
does not hold already (``RunContext.journaled_usage``: the first attempt reported usage as
it came, and some of it may have ridden on a journal entry); an open or interrupted one gets
``EXTERNAL_AGENT_RESUME``; a prompt the log shows no trace of is sent as new. A prompt is
matched by the ``messageId`` the first attempt journaled, else by its text, and then only
when Devvi received it after ``BASELINE_CHECKPOINT`` (the log's length before the first
attempt prompted): a continued session (``session=``) holds earlier calls' prompts, whose
text can be the same. A continued session without a baseline gets
``DEVVI_RESUME_UNKNOWN``. When Devvi
no longer has the session at all (its log was removed, the home or sessions root changed,
Devvi died before its first write), the first attempt may still have sent its prompt, so
the call ends ``interrupted`` with an unknown outcome (``DEVVI_NOT_REATTACHED``) and the
task is not sent again.

Once a turn has its outcome, stopping the process (``shutdown``, then the exit) cannot
discard it: a cancel that arrives meanwhile (a deadline, the calling run's cancel) lets
the stop finish in the background and the finished turn is still reported as finished.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ... import prompts
from ... import prompts as base_prompts
from ...external import (
    ExternalAgentConfigurationError,
    ExternalAgentConnectionError,
    ExternalAgentProtocolError,
    ExternalAgentTimeoutError,
    ExternalCapabilities,
    ExternalOutcome,
    ExternalTurnContext,
)
from ...messages import ImagePart, Part, TextPart, Usage
from .._jsonrpc import INVALID_PARAMS, JsonRpcError, JsonRpcPeer
from .._process import ManagedProcess
from . import logs
from ._aio import settle
from .events import SessionEventMapper, TokenPricing, content_text, user_receipt
from .launcher import DevviLaunch

logger = logging.getLogger("qagents.integrations.devvi")

PROFILE = "sdk"
"""The Devvi profile of this transport (``packages/bundle/sdk-app``)."""

MESSAGE_CHECKPOINT = "devvi_message_id"
"""Checkpoint key: the ``messageId`` of the prompt a call sent, for re-attach."""

SESSION_CHECKPOINT = "external_session"
"""The base class's checkpoint key: ``{"session_id", "cwd"}``, journaled by ``begin()``."""

BASELINE_CHECKPOINT = "devvi_prompt_baseline"
"""Checkpoint key: ``{"events": n}``, how many events the session's log held before the
call's first prompt (``0`` for a new session, ``None`` when the log could not be read).
Journaled before ``begin()``, so a repeat that finds the session id finds it too; re-attach
then matches a prompt by its text only when Devvi received it after these events."""

USAGE_BASE_CHECKPOINT = "devvi_usage_base"
"""Checkpoint key: the usage the journal held for the call (``RunContext.journaled_usage``,
as JSON) when a repeated call prompted a new turn. Re-attach replays a finished turn's
usage against only what was journaled after it (`ExternalTurnContext.replaying`)."""

APPROVAL_OUTCOMES = {"allow": "allowed-once", "deny": "rejected", "cancel": "cancelled"}
"""qagents' answer to ``approval/request`` (never ``allow-always``: grants stay one-shot)."""

PERMANENT_CODES = frozenset(
    {"AUTH", "MISSING_CREDENTIAL", "INVALID_CREDENTIAL", "QUOTA", "ACCOUNT_QUOTA"}
)
"""``LlmFailure.code`` values of a ``turn/end`` error that will repeat (``packages/llm``)."""

IMAGE_TYPES = ("image/png", "image/jpeg", "image/webp", "image/gif")

_NOT_FOUND = re.compile(r"\bnot found\b", re.IGNORECASE)
_ALREADY_OPEN = re.compile(r"already (?:open|owned)|opening", re.IGNORECASE)


class _Cancelled(Exception):
    """The turn was asked to stop while waiting for Devvi."""


class _BadInput(Exception):
    pass


@dataclass(frozen=True)
class SdkSettings:
    """What the SDK transport sends to Devvi and how long it waits."""

    provider: str = "deepseek-official"
    model: str = "deepseek-v4-flash"
    reasoning_effort: str | None = None
    max_tokens: int | None = None
    pricing: TokenPricing | None = None
    raw_events: bool = False
    wait_for_subagents: bool = True
    process_per_call: bool = True
    initialize_timeout_s: float = 120.0
    """``initialize`` waits for Devvi's plugins to load (MCP discovery included)."""
    request_timeout_s: float = 120.0
    """``session/resume`` and ``session/prompt`` (the prompt answer is an enqueue receipt)."""
    shutdown_timeout_s: float = 5.0
    """How long a finished call waits for Devvi to flush and exit after ``shutdown``."""

    def initialize_params(self, cwd: str) -> dict[str, Any]:
        params: dict[str, Any] = {
            "cwd": cwd,
            "provider": self.provider,
            "model": self.model,
            "approvalRequests": True,
        }
        if self.reasoning_effort is not None:
            params["reasoningEffort"] = self.reasoning_effort
        if self.max_tokens is not None:
            params["maxTokens"] = self.max_tokens
        return params


# --- one turn's notifications ----------------------------------------------------------------


class _TurnState:
    """The notifications of one turn: held back until the prompt's receipt, then mapped."""

    def __init__(self, turn: ExternalTurnContext, mapper: SessionEventMapper) -> None:
        self.turn = turn
        self.mapper = mapper
        self.session_id = mapper.root
        self.message_id: str | None = None
        self.collecting = False
        """Set just before ``session/prompt``; earlier notifications (resume repairs) are
        dropped."""
        self.received = False
        """The receipt of the prompt arrived: everything from it on belongs to this turn."""
        self.idle = asyncio.Event()
        self.ready = asyncio.Event()
        """The prompt's message id is known (approval requests wait for it)."""
        self._pending: list[tuple[str, Mapping[str, Any]]] = []

    def notify(self, method: str, params: Mapping[str, Any]) -> None:
        if not self.collecting:
            return
        if not self.received:
            if self.message_id is None:  # the receipt may come before the prompt's answer
                self._pending.append((method, params))
                return
            if not self._is_receipt(method, params):
                return
            self.received = True
        self._deliver(method, params)

    def set_message_id(self, message_id: str) -> None:
        self.message_id = message_id
        self.ready.set()
        pending, self._pending = self._pending, []
        for index, (method, params) in enumerate(pending):
            if self._is_receipt(method, params):
                self.received = True
                for item in pending[index:]:
                    self._deliver(*item)
                return

    def _deliver(self, method: str, params: Mapping[str, Any]) -> None:
        if method == "session.status":
            if params.get("sessionId") == self.session_id and params.get("status") == "idle":
                self.idle.set()
            return
        self.mapper.handle_notification(method, params)

    def _is_receipt(self, method: str, params: Mapping[str, Any]) -> bool:
        if method != "session.event" or params.get("sessionId") != self.session_id:
            return False
        event = params.get("event")
        return isinstance(event, Mapping) and any(
            message_id == self.message_id for message_id, _ in user_receipt(event)
        )


# --- one Devvi process --------------------------------------------------------------------------


class SdkConnection:
    """One ``devvi --profile sdk`` process and its JSON-RPC peer, after ``initialize``."""

    def __init__(self, process: ManagedProcess, peer: JsonRpcPeer, cwd: str) -> None:
        self.process = process
        self.peer = peer
        self.cwd = cwd
        self.server_info: dict[str, Any] = {}
        self.ready_s: float | None = None
        """Seconds from the start of the process to the ``initialize`` answer."""
        self._turns: dict[str, _TurnState] = {}
        self._stopping: asyncio.Future[None] | None = None

    @classmethod
    async def open(
        cls,
        launch: DevviLaunch,
        cwd: str,
        params: Mapping[str, Any],
        *,
        name: str = "devvi",
        timeout: float = 120.0,
    ) -> SdkConnection:
        """Start Devvi in ``cwd`` and run ``initialize``.

        Raises:
            ExternalAgentConfigurationError: Devvi did not start, refused ``initialize``
                (an unknown provider route, a bad patch), or did not answer in time.
        """
        began = time.monotonic()
        process = await ManagedProcess.start(
            launch.argv(PROFILE), env=launch.env, cwd=cwd, name=name
        )
        peer = JsonRpcPeer.for_process(process)
        connection = cls(process, peer, cwd)
        for method in ("session.event", "session.status", "subagent.started", "subagent.finished"):
            peer.on_notification(method, _route(connection, method))
        peer.on_request("approval/request", connection._on_approval)
        peer.start()
        try:
            answer = await peer.request("initialize", dict(params), timeout=timeout)
        except BaseException as error:
            tail = process.stderr_tail()
            await connection.stop(shutdown_timeout=0, grace=1.0)
            if isinstance(error, JsonRpcError):
                why = prompts.DEVVI_INITIALIZE_FAILED.format(message=error.message)
            elif isinstance(error, ExternalAgentTimeoutError):
                why = prompts.DEVVI_INITIALIZE_TIMEOUT.format(seconds=timeout)
            elif isinstance(error, ExternalAgentConnectionError):
                why = prompts.DEVVI_STOPPED_DURING_START.format(reason=error.reason)
            else:
                raise
            raise ExternalAgentConfigurationError(
                f"{prompts.DEVVI_DID_NOT_START.format(why=why)}\n"
                f"Command: {' '.join(launch.argv(PROFILE))}\nstderr: {tail}",
                stderr_tail=tail,
            ) from error
        info = answer.get("serverInfo") if isinstance(answer, Mapping) else None
        if not isinstance(info, Mapping):
            await connection.stop(shutdown_timeout=0, grace=1.0)
            raise ExternalAgentConfigurationError(
                f"{prompts.DEVVI_NO_SERVER_INFO}\nAnswer: {str(answer)[:300]}"
            )
        connection.server_info = dict(info)
        connection.ready_s = time.monotonic() - began
        return connection

    @property
    def alive(self) -> bool:
        return self._stopping is None and not self.peer.closed and self.process.returncode is None

    def open_turn(self, state: _TurnState) -> None:
        self._turns[state.session_id] = state

    def close_turn(self, session_id: str) -> None:
        self._turns.pop(session_id, None)

    def state_for(self, session_id: Any) -> _TurnState | None:
        for state in list(self._turns.values()):
            if state.mapper.owns(session_id):
                return state
        return None

    async def stop(self, *, shutdown_timeout: float = 1.0, grace: float = 2.0) -> None:
        """``shutdown`` (bounded), close stdin, wait, then kill the tree. Safe to repeat.

        The stop runs as its own task: when the caller is cancelled while waiting, it
        still finishes (nothing of the process tree is left behind).
        """
        if self._stopping is None:
            self._stopping = asyncio.ensure_future(self._stop(shutdown_timeout, grace))
        await asyncio.shield(self._stopping)

    async def _stop(self, shutdown_timeout: float, grace: float) -> None:
        try:
            if shutdown_timeout > 0 and not self.peer.closed and self.process.returncode is None:
                with contextlib.suppress(Exception):
                    await self.peer.request("shutdown", timeout=shutdown_timeout)
            await self.process.close(grace=grace, terminate_grace=1.0)
        finally:
            await self.peer.aclose()

    def kill(self) -> None:
        """Kill the process tree now (synchronous); `stop` then only cleans up."""
        self.process.kill()

    async def _on_approval(self, params: Any) -> dict[str, str]:
        if not isinstance(params, Mapping):
            raise JsonRpcError(INVALID_PARAMS, "approval/request needs an object")
        session = params.get("sessionId")
        tool = params.get("toolName")
        call_id = params.get("callId") if isinstance(params.get("callId"), str) else None
        reason = params.get("reason") if isinstance(params.get("reason"), str) else None
        state = self.state_for(session)
        if state is None or not isinstance(tool, str) or not tool:
            return {"outcome": "cancelled"}  # not a turn qagents runs: refuse
        try:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(state.ready.wait(), 10.0)
            # Only the requesting session's own call: call ids repeat across sessions.
            arguments = state.mapper.arguments_for(session, call_id)
            if arguments is None:
                logger.warning(
                    "Devvi asked about call %s of session %s, which that session has not "
                    "started; refusing it",
                    call_id,
                    session,
                )
                state.turn.event(
                    "approval_refused",
                    {"toolName": tool, "callId": call_id, "reason": "unknown_call"},
                    session_id=session,
                )
                return {"outcome": "cancelled"}
            # Never None here, so the base class cannot fall back to the root's calls.
            answer = await state.turn.ask_permission(
                tool, arguments, tool_call_id=call_id, reason=reason
            )
        except Exception:  # fail closed
            logger.exception("Deciding Devvi's approval request for %s failed", tool)
            return {"outcome": "cancelled"}
        return {"outcome": APPROVAL_OUTCOMES.get(answer.outcome, "cancelled")}


def _route(connection: SdkConnection, method: str) -> Callable[[Any], None]:
    def handle(params: Any) -> None:
        if not isinstance(params, Mapping):
            return
        if method == "subagent.started":
            key = params.get("parentSessionId")
        elif method == "subagent.finished":
            key = params.get("childSessionId")
        else:
            key = params.get("sessionId")
        state = connection.state_for(key)
        if state is not None:
            state.notify(method, params)

    return handle


# --- the transport ------------------------------------------------------------------------------


class SdkTransport:
    """`DevviAgent`'s SDK transport: connects, runs turns, cancels them.

    Args:
        settings: Route, prices and timeouts.
        launch: Returns the prepared `DevviLaunch` (cached by the agent).
        sessions_root: Where the launch's patch puts session logs.
        name: Name of the processes in logs and errors.
    """

    resumable_turns = True

    manages_connections = True
    """The pool recovers per process: `_acquire` replaces a pooled process that died, and a
    turn that broke or was abandoned stops only its own (`_release`, `_abandon`). So
    `DevviAgent.mark_disconnected` leaves the agent connected, and the next turn never
    stops the processes that serve other workspaces (``process_per_call=False``)."""

    def __init__(
        self,
        settings: SdkSettings,
        *,
        launch: Callable[[], Awaitable[DevviLaunch]],
        sessions_root: Callable[[DevviLaunch], Path],
        name: str = "devvi",
    ) -> None:
        self.settings = settings
        self._launch = launch
        self._sessions_root = sessions_root
        self.name = name
        self._pool: dict[str, SdkConnection] = {}
        self._closing: set[asyncio.Task[None]] = set()

    # -- lifecycle -------------------------------------------------------------------------

    async def connect(self) -> None:
        """Prepare the launch (find the CLI and Node, check versions, write the patch)."""
        await self._launch()

    async def disconnect(self) -> None:
        """Stop pooled processes (``process_per_call=False``)."""
        pool, self._pool = list(self._pool.values()), {}
        await asyncio.gather(
            *(connection.stop(shutdown_timeout=1.0, grace=1.0) for connection in pool),
            return_exceptions=True,
        )
        if self._closing:
            await asyncio.gather(*list(self._closing), return_exceptions=True)

    async def capabilities(self) -> ExternalCapabilities:
        return ExternalCapabilities(
            token_usage=True,
            live_tokens=False,
            cancel_turn=False,
            resume_sessions=True,
            client_session_ids=True,
            permission_requests=True,
            mcp_servers=True,
        )

    @staticmethod
    def backend(launch: DevviLaunch) -> str:
        return f"devvi-sdk@{launch.version}" if launch.version else "devvi-sdk"

    async def cancel_turn(self, turn: ExternalTurnContext) -> None:
        """Nothing to send: `run_turn` sees the cancel and stops the process itself."""

    # -- a turn ----------------------------------------------------------------------------

    async def run_turn(self, turn: ExternalTurnContext) -> ExternalOutcome:
        settings = self.settings
        self._check_budget(turn)
        try:
            blocks = content_blocks(turn.task)
        except _BadInput:
            return ExternalOutcome(status="error", stop_reason=prompts.DEVVI_IMAGE_URL)
        launch = await self._launch()
        cwd = turn.cwd or os.getcwd()
        root = self._sessions_root(launch)
        repeat = isinstance(turn.checkpoints.get(SESSION_CHECKPOINT), Mapping)
        session_id = self._session_id(turn)
        stored = await asyncio.to_thread(_stored_session, root, session_id, cwd)
        if stored.cwd is not None and not _same_directory(stored.cwd, cwd):
            # Refused before Devvi opens it: a resume would repair and lock that session.
            return _other_workspace(session_id, cwd, stored.cwd)
        connection = await self._acquire(cwd, launch)
        broken = forced = False
        try:
            resumed, failure = await self._resume(connection, session_id, turn)
            if failure is not None:
                return failure
            if repeat and not resumed:
                return self._session_gone(turn, session_id)
            if turn.session is not None and not resumed:
                # A session the caller passed that Devvi does not have. Through as_tool(),
                # a session the tool never opened is refused before this, with
                # EXTERNAL_AGENT_UNKNOWN_SESSION. This one comes from developer code, or is
                # one the tool opened that Devvi no longer has (see DEVVI_UNKNOWN_SESSION).
                return ExternalOutcome(
                    status="error",
                    stop_reason=prompts.DEVVI_UNKNOWN_SESSION.format(session=turn.session),
                )
            if resumed and stored.cwd is None:  # its log did not say where it works
                refusal = await self._confirm_workspace(connection, session_id, cwd, turn)
                if refusal is not None:
                    broken = True  # this process holds that session open now: stop it
                    return refusal
            if turn.resumable and not repeat:  # record first: before the session id
                events = stored.events if resumed else 0
                await turn.checkpoint(BASELINE_CHECKPOINT, {"events": events})
            await turn.begin(session_id, resumed=resumed, backend=self.backend(launch))
            turn.raw["session_log"] = str(logs.session_log_path(root, session_id, cwd))
            mapper = SessionEventMapper(
                turn, root=session_id, pricing=settings.pricing, raw_events=settings.raw_events
            )
            state = _TurnState(turn, mapper)
            connection.open_turn(state)
            if repeat and resumed:
                plan = self._reattach(turn, session_id, cwd, root, mapper, blocks)
                if plan is None:
                    return self._outcome(mapper, recovered=True)
                blocks = plan
                await self._mark_usage_base(turn)  # record first: before the new prompt
            state.collecting = True
            answer = await self._race(
                connection,
                connection.peer.request(
                    "session/prompt",
                    {"sessionId": session_id, "contentBlocks": blocks},
                    timeout=settings.request_timeout_s,
                ),
                turn,
            )
            message_id = answer.get("messageId") if isinstance(answer, Mapping) else None
            if not isinstance(message_id, str) or not message_id:
                raise ExternalAgentProtocolError(
                    f"Devvi answered session/prompt without a messageId: {str(answer)[:200]}",
                    reason=base_prompts.EXTERNAL_REASON_BAD_ANSWER,
                )
            state.set_message_id(message_id)
            turn.raw["message_id"] = message_id
            if turn.resumable:
                await turn.checkpoint(MESSAGE_CHECKPOINT, message_id)
            await self._race(connection, state.idle.wait(), turn)
            failed: str | None = None
            if settings.wait_for_subagents:
                try:
                    await self._race(
                        connection,
                        connection.peer.request("session/wait", {"sessionId": session_id}),
                        turn,
                    )
                except JsonRpcError as error:  # a live failure Devvi did not commit as a turn end
                    failed = _first_line(error.message)
            outcome = self._outcome(mapper)
            if failed is not None and outcome.status == "completed":
                return ExternalOutcome(status="error", stop_reason=failed)
            return outcome
        except _Cancelled:
            broken = True
            try:
                await connection.stop(shutdown_timeout=1.0, grace=1.0)
            except asyncio.CancelledError:
                forced = True
                self._abandon(connection)
                raise
            return ExternalOutcome(status="cancelled", stop_reason=turn.cancel_reason)
        except JsonRpcError as error:
            # session/prompt or session/wait refused: Devvi's own message, nothing ambiguous.
            return ExternalOutcome(status="error", stop_reason=_first_line(error.message))
        except asyncio.CancelledError:
            forced = True  # it did not stop within the grace period
            self._abandon(connection)
            raise
        except BaseException:
            broken = True
            raise
        finally:
            connection.close_turn(session_id)
            if not forced:
                # The turn has its outcome (or its error): a cancel while the process stops
                # lets the stop finish in the background instead of discarding it.
                await settle(self._release(connection, cwd, broken), self._closing)

    # -- steps -----------------------------------------------------------------------------

    def _check_budget(self, turn: ExternalTurnContext) -> None:
        if self.settings.pricing is not None or turn.parent is None:
            return
        if turn.parent.remaining_budget().get("max_cost_usd") is not None:
            raise ExternalAgentConfigurationError(
                f"{prompts.DEVVI_NEEDS_PRICING}\n"
                "The calling run has Limits.max_cost_usd, and qagents has no prices for "
                "Devvi's models: pass DevviAgent(pricing=TokenPricing(...))."
            )

    @staticmethod
    def _session_id(turn: ExternalTurnContext) -> str:
        recorded = turn.checkpoints.get(SESSION_CHECKPOINT)
        if isinstance(recorded, Mapping):
            value = recorded.get("session_id")
            if isinstance(value, str) and value:
                return value
        return turn.session or turn.suggested_session_id

    async def _acquire(self, cwd: str, launch: DevviLaunch) -> SdkConnection:
        key = os.path.normcase(cwd)
        if not self.settings.process_per_call:
            pooled = self._pool.get(key)
            if pooled is not None:
                if pooled.alive:
                    return pooled
                # It died between calls: release what is left of it, then start a new one.
                del self._pool[key]
                await pooled.stop(shutdown_timeout=0, grace=1.0)
        connection = await SdkConnection.open(
            launch,
            cwd,
            self.settings.initialize_params(cwd),
            name=self.name,
            timeout=self.settings.initialize_timeout_s,
        )
        if not self.settings.process_per_call:
            self._pool[key] = connection
        return connection

    async def _release(self, connection: SdkConnection, cwd: str, broken: bool) -> None:
        if not (self.settings.process_per_call or broken or not connection.alive):
            return
        key = os.path.normcase(cwd)
        if self._pool.get(key) is connection:
            del self._pool[key]
        timeout = self.settings.shutdown_timeout_s
        await connection.stop(shutdown_timeout=timeout, grace=timeout)

    def _abandon(self, connection: SdkConnection) -> None:
        """Kill the process now and finish cleaning up in the background."""
        connection.kill()
        for key, pooled in list(self._pool.items()):
            if pooled is connection:
                del self._pool[key]
        task = asyncio.get_running_loop().create_task(connection.stop(shutdown_timeout=0))
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)

    async def _resume(
        self, connection: SdkConnection, session_id: str, turn: ExternalTurnContext
    ) -> tuple[bool, ExternalOutcome | None]:
        """``(resumed, failure)``: the session exists (now open), or is new."""
        try:
            await self._race(
                connection,
                connection.peer.request(
                    "session/resume",
                    {"sessionId": session_id},
                    timeout=self.settings.request_timeout_s,
                ),
                turn,
            )
        except JsonRpcError as error:
            if _NOT_FOUND.search(error.message):
                return False, None
            if _ALREADY_OPEN.search(error.message):
                return True, None  # open in this process already (process_per_call=False)
            reason = _first_line(error.message)
            stop = prompts.DEVVI_RESUME_FAILED.format(session=session_id, reason=reason)
            return False, ExternalOutcome(status="error", stop_reason=stop)
        return True, None

    async def _confirm_workspace(
        self, connection: SdkConnection, session_id: str, cwd: str, turn: ExternalTurnContext
    ) -> ExternalOutcome | None:
        """After resuming a session whose log qagents could not read: ``None`` when Devvi
        says the session works in ``cwd``, else the refusal (fail closed)."""
        where: str | None = None
        try:
            answer = await self._race(
                connection,
                connection.peer.request(
                    "session/working-directory/get",
                    {"sessionId": session_id},
                    timeout=self.settings.request_timeout_s,
                ),
                turn,
            )
        except JsonRpcError as error:
            logger.warning(
                "Devvi did not say where session %s works (%s); not continuing it",
                session_id,
                _first_line(error.message),
            )
        else:
            value = answer.get("cwd") if isinstance(answer, Mapping) else None
            where = value if isinstance(value, str) and value else None
        if where is None:
            stop = prompts.DEVVI_WORKSPACE_UNKNOWN.format(session=session_id)
            return ExternalOutcome(status="error", stop_reason=stop, raw={"session_cwd": None})
        if not _same_directory(where, cwd):
            return _other_workspace(session_id, cwd, where)
        return None

    @staticmethod
    async def _mark_usage_base(turn: ExternalTurnContext) -> None:
        """A repeated call prompts a new turn: journal how much of the call's usage the
        journal holds already (``USAGE_BASE_CHECKPOINT``). It belongs to older turns, so a
        later repeat that replays the new turn does not count it against that turn."""
        held = turn.parent.journaled_usage if turn.parent is not None else Usage()
        if held != Usage() or USAGE_BASE_CHECKPOINT in turn.checkpoints:
            await turn.checkpoint(USAGE_BASE_CHECKPOINT, held.model_dump(mode="json"))

    @staticmethod
    def _session_gone(turn: ExternalTurnContext, session_id: str) -> ExternalOutcome:
        """A repeated call whose journaled session Devvi does not have (``not found``).

        The first attempt journaled the session id before it sent its prompt, so it may have
        sent it; with ``MESSAGE_CHECKPOINT`` Devvi even took it. Either way what the prompt
        did is unknown, and the session that would show it is gone: the call ends
        ``interrupted`` (ambiguous) and the task is not sent again, as on ACP.
        """
        accepted = isinstance(turn.checkpoints.get(MESSAGE_CHECKPOINT), str)
        turn.raw["reattach"] = "session_missing"
        return ExternalOutcome(
            status="interrupted",
            stop_reason=prompts.DEVVI_NOT_REATTACHED.format(session=session_id),
            raw={"ambiguous": True, "earlier_session": session_id, "prompt_accepted": accepted},
        )

    def _reattach(
        self,
        turn: ExternalTurnContext,
        session_id: str,
        cwd: str,
        root: Path,
        mapper: SessionEventMapper,
        blocks: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        """A repeated call: what to send, or ``None`` when the first attempt's turn finished
        (it was replayed into ``mapper`` from the log)."""
        path = logs.find_session_log(root, session_id, cwd=cwd)
        log: logs.SessionLog | None = None
        if path is not None:
            try:
                log = logs.read_session_log(path)
            except logs.DevviLogError as error:
                logger.warning("Cannot re-attach from Devvi's log: %s", error)
        recorded = turn.checkpoints.get(MESSAGE_CHECKPOINT)
        message_id = recorded if isinstance(recorded, str) and recorded else None
        after = _baseline(turn.checkpoints.get(BASELINE_CHECKPOINT))
        unknown = prompts.DEVVI_RESUME_UNKNOWN.format(task=turn.prompt_text)
        if log is None:
            turn.raw["reattach"] = "unknown"
            return [{"type": "text", "text": unknown}, *_images(blocks)]
        if message_id is None and after is None and session_id != turn.suggested_session_id:
            # A continued session holds the prompts of earlier calls, and without the
            # baseline their text cannot be told apart from this call's.
            turn.raw["reattach"] = "unknown"
            return [{"type": "text", "text": unknown}, *_images(blocks)]
        # The first attempt sent the task's text parts as blocks of one message, and Devvi
        # logs that message's text with its blocks joined as they are (`content_text`), not
        # with the blank line of ``prompt_text``: both are matched.
        sent = [
            turn.prompt_text,
            content_text(blocks),
            base_prompts.EXTERNAL_AGENT_RESUME,
            unknown,
        ]
        record = logs.find_prompt_turn(log, message_id=message_id, prompts=sent, after=after)
        if record is None:
            if message_id is not None:  # Devvi took the prompt, yet its log does not show it
                turn.raw["reattach"] = "unknown"
                return [{"type": "text", "text": unknown}, *_images(blocks)]
            turn.raw["reattach"] = "not_sent"
            return blocks
        if record.finished:
            turn.raw["reattach"] = "finished"
            # The first attempt reported this turn's usage as it came, and some of it may
            # have reached the journal before the crash: the run gets only the rest.
            since = _usage(turn.checkpoints.get(USAGE_BASE_CHECKPOINT))
            with turn.replaying(since=since):
                for event in log.turn_events(record):
                    mapper.handle_event(session_id, event)
            return None
        turn.raw["reattach"] = "resumed"
        return [{"type": "text", "text": base_prompts.EXTERNAL_AGENT_RESUME}]

    async def _race(
        self, connection: SdkConnection, awaitable: Awaitable[Any], turn: ExternalTurnContext
    ) -> Any:
        """Await ``awaitable`` unless the turn is cancelled or the connection closes first."""
        work = asyncio.ensure_future(awaitable)
        if turn.cancel_requested:
            work.cancel()
            await asyncio.gather(work, return_exceptions=True)
            raise _Cancelled
        closed = asyncio.ensure_future(connection.peer.wait_closed())
        cancelled = asyncio.ensure_future(turn.wait_cancelled())
        try:
            done, _ = await asyncio.wait(
                {work, closed, cancelled}, return_when=asyncio.FIRST_COMPLETED
            )
        except asyncio.CancelledError:
            for task in (work, closed, cancelled):
                task.cancel()
            raise
        finally:
            for task in (closed, cancelled):
                if not task.done():
                    task.cancel()
        if work in done:
            try:
                return work.result()
            except ExternalAgentTimeoutError as error:
                raise ExternalAgentConnectionError(
                    str(error), reason=base_prompts.EXTERNAL_REASON_NO_ANSWER
                ) from error
        work.cancel()
        await asyncio.gather(work, return_exceptions=True)
        if cancelled in done:
            raise _Cancelled
        error = connection.peer.close_error
        if error is not None:
            raise error
        raise ExternalAgentConnectionError(
            "The connection to Devvi closed.", reason=connection.process.exit_summary()
        )

    def _outcome(self, mapper: SessionEventMapper, *, recovered: bool = False) -> ExternalOutcome:
        outcome = mapper.outcome()
        if outcome is None:
            outcome = ExternalOutcome(status="error", stop_reason=prompts.DEVVI_NO_TURN_END)
        if outcome.status == "error" and outcome.raw.get("error_code") in PERMANENT_CODES:
            outcome.raw["unavailable"] = True
        if recovered:
            outcome.raw["recovered"] = True
        return outcome


# --- prompt content ------------------------------------------------------------------------------


def content_blocks(parts: Sequence[Part]) -> list[dict[str, Any]]:
    """The task as SDK ``contentBlocks``: text blocks, and inline images.

    Raises:
        _BadInput: An image Devvi cannot take (a URL, or an unsupported type).
    """
    blocks: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, TextPart):
            if part.text:
                blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            data, media = part.data, part.media_type
            if data is None and part.url and part.url.startswith("data:"):
                header, _, payload = part.url.partition(",")
                if ";base64" in header:
                    data, media = payload, header[5:].split(";")[0] or media
            if not data or media not in IMAGE_TYPES:
                raise _BadInput
            blocks.append({"type": "image", "data": data, "mimeType": media})
    return blocks or [{"type": "text", "text": ""}]


# --- workspaces and baselines ------------------------------------------------------------------


@dataclass(frozen=True)
class _Stored:
    """What the log of a stored session says, read before Devvi opens the session."""

    cwd: str | None = None
    """The header's ``cwd``: where the session works (``None``: no log, or not readable)."""
    events: int | None = None
    """How many events the log holds (``None``: no log, or not readable)."""


def _stored_session(root: Path, session_id: str, cwd: str) -> _Stored:
    path = logs.find_session_log(root, session_id, cwd=cwd)
    if path is None:
        return _Stored()
    try:
        header, events = logs.peek_session_log(path)
    except logs.DevviLogError as error:
        logger.info("Cannot read the header of Devvi's log %s: %s", path, error)
        return _Stored()
    owner = header.get("cwd")
    return _Stored(cwd=owner if isinstance(owner, str) and owner else None, events=events)


def _same_directory(first: str, second: str) -> bool:
    """The same folder, with case and links resolved as the operating system does."""
    try:
        return os.path.normcase(os.path.realpath(first)) == os.path.normcase(
            os.path.realpath(second)
        )
    except (OSError, ValueError):
        return False


def _other_workspace(session_id: str, cwd: str, owner: str) -> ExternalOutcome:
    logger.warning(
        "Refusing to continue Devvi session %s in %s: it works in %s", session_id, cwd, owner
    )
    stop = prompts.DEVVI_OTHER_WORKSPACE.format(session=session_id)
    return ExternalOutcome(status="error", stop_reason=stop, raw={"session_cwd": owner})


def _baseline(value: Any) -> int | None:
    events = value.get("events") if isinstance(value, Mapping) else None
    if isinstance(events, int) and not isinstance(events, bool) and events >= 0:
        return events
    return None


def _usage(value: Any) -> Usage:
    """A ``USAGE_BASE_CHECKPOINT`` value; ``Usage()`` when there is none (a first prompt)."""
    if not isinstance(value, Mapping):
        return Usage()
    try:
        return Usage.model_validate(value)
    except ValueError:
        return Usage()


def _images(blocks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(block) for block in blocks if block.get("type") == "image"]


def _first_line(text: str, limit: int = 300) -> str:
    """The first line of ``text``, cut to ``limit`` characters with
    `prompts.EXTERNAL_AGENT_LINE_SHORTENED`: it becomes a stop reason the model reads."""
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    if len(line) <= limit:
        return line
    marker = base_prompts.EXTERNAL_AGENT_LINE_SHORTENED
    return line[: max(limit - len(marker), 0)] + marker
