"""Every exception qagents raises.

Model errors carry two flags the agent loop acts on:

* ``retryable``: the request can be sent again and may succeed.
* ``ambiguous``: the provider may already have processed (and billed) the
  request. The loop never replays an ambiguous request by itself.

Four exceptions are signals rather than failures:

* `DeferToolCall`: a tool raises it to pause the run until its result is
  supplied with ``agent.resume(..., results=...)``.
* `ModelRetry`: a tool or an output validator raises it to send the model a
  correction request.
* `GuardrailTripped`: a guardrail raises it to block a prompt or an answer.
* `WorkflowSuspended`: a workflow waits for a signal or a timer; run it again
  later.

Errors of optional integrations (`MCPConnectionError`, `MCPToolChangedError`,
`CassetteMissError`) are defined here too, so you can catch them without
importing the integration.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class QAgentsError(Exception):
    """Base class for every error raised by qagents."""


class ConfigurationError(QAgentsError):
    """The agent, a tool or a model was set up incorrectly."""


class MaxDepthExceeded(ConfigurationError):
    """A child run would nest deeper than ``Limits.max_depth``, so it was not started.

    An agent used as a tool reports it to the calling model as an error result
    (`prompts.SUBAGENT_TOO_DEEP`) instead of failing the tool call.
    """

    def __init__(self, message: str, *, depth: int, limit: int) -> None:
        super().__init__(message)
        self.depth = depth
        """The depth the refused run would have had."""
        self.limit = limit
        """The ``Limits.max_depth`` that applies (the lowest along the chain of parent runs)."""


class SessionError(QAgentsError):
    """A session could not be loaded, resumed or written."""


class OutputValidationError(QAgentsError):
    """The model's final answer did not match ``output_type`` after all retries.

    Also raised when output validators keep rejecting it, or (``output_mode="tool"``)
    when the model keeps answering without the final-answer tool. ``agent.run``
    raises it rather than returning a `RunResult`; the run is journaled as
    failed (``run_failed``) and can be resumed.
    """


class ToolError(QAgentsError):
    """Raise this inside a tool to report a failure to the model.

    The message is sent back as the tool result (marked as an error), so write
    it for the model: say what went wrong and what it can do instead.
    """


class ToolArgumentError(ToolError):
    """The model called a tool with arguments that do not match its schema."""


class ModelRetry(ToolError):
    """Ask the model to try again, with a message that says what to correct.

    Raise it inside a tool to ask for a corrected call: the message becomes
    the call's error result, exactly as with `ToolError`, and the model can
    call again.

        @tool
        def book_room(day: str) -> str:
            '''Book the meeting room.'''
            if day not in free_days():
                raise ModelRetry(f"{day} is taken. Free days: {', '.join(free_days())}.")
            ...

    Raise it inside an output validator (``Agent(output_validators=[...])``)
    to reject a final answer: the message goes back to the model, which
    answers again. Each rejection counts toward ``Limits.output_retries``;
    beyond that the run raises `OutputValidationError`.

    Args:
        message: What is wrong and what to do instead, written for the model.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
        """The text sent to the model."""


class DeferToolCall(QAgentsError):
    """Raise this inside a tool to pause the run until its result is supplied.

    Use it when the answer is produced somewhere else and later: a long job in
    another system, a reviewer, a webhook. The call is not reported to the
    model as an error. It becomes a `PendingInput` in
    ``result.pending_inputs`` and the run ends with ``status == "paused"``
    once the other calls of the turn are done. Continue with
    ``agent.resume(session, results={tool_call_id: value})``.

    The tool is never run again for this call, not even on resume: its result
    comes only from ``results=``. Whatever the tool did before raising is
    therefore done once.

        @tool
        async def render_video(ctx: RunContext, script: str) -> str:
            '''Render a video from a script.'''
            job = await renderer.submit(script, key=ctx.operation_id)
            raise DeferToolCall("rendering", metadata={"job_id": job.id})

    Args:
        reason: Short text for your own code and logs, for example ``"rendering"``.
            It is not sent to the model.
        metadata: JSON-serialisable data kept with the pending input, for
            example an external job id to poll.
    """

    def __init__(
        self, reason: str | None = None, metadata: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__(reason or "the tool call was deferred")
        self.reason = reason
        """Why the call waits, for your code and logs."""
        self.metadata: dict[str, Any] = dict(metadata or {})
        """Data kept with the pending input, such as an external job id."""


class GuardrailTripped(QAgentsError):
    """Raise this inside a guardrail to block the prompt or the answer it checks.

    The run then ends with ``status == "blocked"``, ``stop_reason`` set to
    the guardrail's name and ``result.guardrail`` describing the trip. Any
    other exception inside a guardrail fails the run: a broken guardrail
    never lets a value through.

        def no_account_numbers(ctx: RunContext, message: Message) -> None:
            if IBAN.search(message.text):
                raise GuardrailTripped("the prompt contains an account number")

    By design the reason is not sent to the model and the run does not
    continue: ``RunResult.text`` and ``output`` are empty. To have the model
    revise a blocked answer, start a new run on the same session with a prompt
    of your own that says what to change. To ask for a correction within the
    run instead, use an output validator that raises `ModelRetry`.

    An output guardrail checks the completed answer: text already streamed
    (``text_delta`` events, ``RunStream.text()``, ``model_responded``) cannot
    be recalled. A front end that must not show a blocked answer holds back
    the text until ``run_finished`` and checks ``result.status``.

    Args:
        reason: Why the value was blocked, for your code, logs and the journal.
            It is not sent to the model.
        name: Name recorded for the trip; defaults to the guardrail
            function's ``__name__``.
    """

    def __init__(self, reason: str, *, name: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        """Why the value was blocked."""
        self.name = name
        """Name recorded for the trip, or ``None`` to use the guardrail's name."""


class WorkflowSuspended(QAgentsError):
    """The workflow cannot continue yet: stop now and run it again later.

    `Workflow.wait_for_signal` raises it while the signal has not been sent,
    and `Workflow.wait_for_timer` while the timer is not due. Nothing is
    lost: every step that finished is in the journal, and the next run of
    the same workflow id replays them and comes back to the same wait.

        try:
            outcome = await publish_after_review(store, id)
        except WorkflowSuspended as suspended:
            schedule_again(id, at=suspended.resume_at)  # None: when the signal arrives

    Args:
        signal: The signal the workflow waits for, if it waits for one.
        timer: The timer the workflow waits for, if it waits for one.
        resume_at: When the timer is due, in seconds since the epoch.
        workflow_id: The workflow that stopped.
    """

    def __init__(
        self,
        signal: str | None = None,
        *,
        timer: str | None = None,
        resume_at: float | None = None,
        workflow_id: str | None = None,
    ) -> None:
        self.signal = signal
        """The signal the workflow waits for, or ``None``."""
        self.timer = timer
        """The timer the workflow waits for, or ``None``."""
        self.resume_at = resume_at
        """When the timer is due (seconds since the epoch), or ``None`` for a signal."""
        self.workflow_id = workflow_id
        """The workflow that stopped."""
        super().__init__(self._describe())

    def _describe(self) -> str:
        workflow = f"Workflow {self.workflow_id!r}" if self.workflow_id else "The workflow"
        if self.signal is not None:
            return (
                f"{workflow} waits for the signal {self.signal!r}. Send it with "
                f"Workflow.send_signal(store, {self.workflow_id!r}, {self.signal!r}, value), "
                "then run the workflow again."
            )
        if self.timer is not None and self.resume_at is not None:
            due = datetime.fromtimestamp(self.resume_at, tz=UTC).isoformat(timespec="seconds")
            return (
                f"{workflow} waits for the timer {self.timer!r}, which is due at {due}. "
                "Run the workflow again after that."
            )
        return f"{workflow} is suspended. Run it again later."


class ApprovalTokenError(SessionError):
    """An approval or a result did not carry the token of a signed pause.

    Raised by ``agent.resume`` before anything runs, when
    ``Permissions(approval_secret=...)`` is set and an answer has no token,
    a wrong token, or the paused call in the journal no longer matches the
    token it was signed with.
    """


class LeaseError(SessionError):
    """Another worker holds the session's lease, or the caller lost it.

    `SessionStore.append_leased` raises it when the writer no longer holds the
    lease (nothing is written), and `qagents.runtime.AgentRuntime` raises it
    when another worker is driving the session.
    """


class RunInterrupted(SessionError):
    """The run was stopped before it finished and stays open in the journal.

    Raised by `RunHandle.wait` and `RunHandle.events` after `AgentRuntime.shutdown`
    stopped the run, or after the worker lost the session's lease to another
    worker. Continue it with `AgentRuntime.resume`, or let `AgentRuntime.recover`
    pick it up.
    """


class ModelError(QAgentsError):
    """A model request failed."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        code: str | None = None,
        request_id: str | None = None,
        retry_after: float | None = None,
        retryable: bool = False,
        ambiguous: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.code = code
        self.request_id = request_id
        self.retry_after = retry_after
        self.retryable = retryable
        self.ambiguous = ambiguous


class ModelBadRequestError(ModelError):
    """The provider rejected the request before running it: a 4xx other than 401, 402,
    403, 408, 409 and 429 (400, 404, 413 and 422, for example), or a redirect (3xx)."""


class ModelAuthenticationError(ModelError):
    """The API key is missing, invalid or not allowed to use this model (401, 403)."""


class ModelBillingError(ModelError):
    """The account cannot pay for the request (402)."""


class ModelRateLimitError(ModelError):
    """The provider refused the request at admission (429). Safe to retry."""


class ModelServerError(ModelError):
    """The provider failed while handling the request (5xx), or reported a conflict (409)."""


class ModelConnectionError(ModelError):
    """The provider could not be reached, or the connection broke."""


class ModelTimeoutError(ModelConnectionError):
    """The provider did not answer in time."""


class ModelResponseError(ModelError):
    """The provider answered with something that is not a valid response."""


class CassetteMissError(ModelError):
    """A `ReplayModel` has no recorded response for a request.

    It is not retryable: the same request would miss again. Either the agent's
    behaviour changed on purpose, and the cassette must be recorded again, or
    it changed by accident, and the message says where.
    """

    def __init__(
        self,
        message: str,
        *,
        fingerprint: str,
        path: Path,
        closest_fingerprint: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(message, provider=provider, model=model, code="cassette_miss")
        self.fingerprint = fingerprint
        """Fingerprint of the request that found no recording."""
        self.path = path
        """The cassette file."""
        self.closest_fingerprint = closest_fingerprint
        """Fingerprint of the most similar recorded request, if the cassette has any."""


class MCPConnectionError(QAgentsError):
    """The toolkit could not connect to the MCP server or list its tools."""


class MCPToolChangedError(ConfigurationError):
    """The server offers a tool whose definition differs from the pinned one."""

    def __init__(self, message: str, *, changed: Mapping[str, tuple[str, str]]) -> None:
        super().__init__(message)
        self.changed = dict(changed)
        """Server tool name -> (expected hash, actual hash)."""
