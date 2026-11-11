"""Workflows: multi-step and multi-agent work written as ordinary Python.

There is no graph to declare. You write an async function, and wrap each unit
of work in ``workflow.step(name, ...)``. A finished step is written to the
store with its result; when the same workflow id runs again (after a crash, a
deploy, a pause for approval) finished steps return their recorded result
instead of running again, and the function continues where it stopped:

    async def research_report(topic: str, store: SessionStore) -> str:
        wf = Workflow(store, id=f"report:{topic}")
        notes = await wf.run_agent("research", researcher, f"Research {topic}")
        draft = await wf.run_agent("write", writer, f"Write a report from:\\n{notes.text}")
        await wf.step("publish", lambda: publish(draft.text))
        return draft.text

Guarantees, stated plainly: a step that *finished* never runs again. A step
that was in progress when the process died runs again from its start, except
agent steps started with ``run_agent``, which resume inside the agent run at
the exact tool call (see `session`). Make other side-effecting
steps idempotent.

Use ``asyncio.gather`` to run steps in parallel, or ``map`` to run one step
per item with an optional bound on concurrency; branches do not wait for
each other.

Long waits (hours or days) do not keep a process alive. ``wait_for_signal``
returns a value another process sent with `Workflow.send_signal`, and
``wait_for_timer`` returns once a durable timer is due. Until then they raise
`WorkflowSuspended`: stop, and run the workflow again later. Everything that
finished is replayed from the journal, so the function reaches the wait
again almost at once:

    async def publish_after_review(store: SessionStore, id: str) -> str:
        wf = Workflow(store, id=id)
        draft = await wf.run_agent("draft", writer, "Draft the announcement")
        verdict = await wf.wait_for_signal("review")  # sent by a reviewer, maybe days later
        if verdict != "approved":
            return "rejected"
        await wf.step("publish", lambda: publish(draft.text))
        return "published"

    # In the web handler of the review form:
    await Workflow.send_signal(store, id, "review", "approved")

`Workflow.status` reads the journal and tells which steps finished, which
signals and timers the workflow waits for, and which agent steps have a run
that is still open (paused for approvals or inputs, failed, or running).

Journal entry kinds (in the session ``workflow:<id>``; readers ignore kinds
they do not know):

| Kind | Meaning (data) |
|---|---|
| `workflow_step` | a step finished (`name`, `result`) |
| `workflow_agent_started` | an agent step started its first run (`name`, `session_id`) |
| `workflow_signal` | a signal was sent (`name`, `value`); the first value of a name wins |
| `workflow_waiting` | the workflow stopped to wait for a signal (`signal`) |
| `workflow_timer` | a durable timer was started (`name`, `due`, seconds since the epoch) |
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import math
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError, to_jsonable_python

from .agent import Agent, Prompt
from .errors import ConfigurationError, SessionError, WorkflowSuspended
from .permissions import Approval
from .results import RunResult
from .session import (
    PendingApproval,
    PendingInput,
    Session,
    SessionStatus,
    SessionStore,
    advance_status,
    derive_state,
)

T = TypeVar("T")

_SHARED_OPTIONS = ("deps", "limits", "settings")
"""Options of `Workflow.run_agent` that both ``agent.run`` and ``agent.resume`` accept."""


class AgentStepStatus(BaseModel):
    """An agent step of a workflow whose run is still open (`WorkflowStatus`)."""

    name: str
    """The step name given to `Workflow.run_agent`."""
    session_id: str
    """The session of the step, as `Workflow.session` returns it."""
    run_id: str
    """The open run."""
    status: Literal["running", "paused", "failed"]
    """``"paused"``: it waits for the approvals and inputs listed here; call
    ``run_agent`` again with ``approvals=`` or ``results=``. ``"failed"``: the
    run raised; call ``run_agent`` again to resume it. ``"running"``: it is
    running now, or the process that ran it died."""
    pending_approvals: list[PendingApproval] = Field(default_factory=list)
    """Tool calls that wait for a decision (``status == "paused"``)."""
    pending_inputs: list[PendingInput] = Field(default_factory=list)
    """Tool calls that wait for a result from outside the run (``status == "paused"``)."""


class WorkflowStatus(BaseModel):
    """Where a workflow stands, read from its journal (`Workflow.status`)."""

    id: str
    """The workflow id."""
    completed_steps: list[str] = Field(default_factory=list)
    """Names of the steps that finished, in the order they finished."""
    signals: list[str] = Field(default_factory=list)
    """Names of the signals that were sent."""
    pending_signals: list[str] = Field(default_factory=list)
    """Signals the workflow stopped to wait for that have not been sent yet."""
    pending_timers: dict[str, float] = Field(default_factory=dict)
    """Timers that are not due yet, with when they are due (seconds since the epoch)."""
    open_agent_steps: list[AgentStepStatus] = Field(default_factory=list)
    """Agent steps whose run started and has not finished."""


@dataclass
class _Journal:
    """What the workflow journal says, read in one pass."""

    completed: dict[str, Any] = field(default_factory=dict)
    signals: dict[str, Any] = field(default_factory=dict)
    waiting: list[str] = field(default_factory=list)
    timers: dict[str, float] = field(default_factory=dict)
    agent_steps: list[str] = field(default_factory=list)


class Workflow:
    """A durable multi-step job, identified by ``id`` within ``store``.

    Create a new ``Workflow`` object every time the job runs (in any process):
    what the journal says has finished is replayed, and the rest runs.

    Args:
        store: Where the workflow's journal and its agent steps' sessions live.
        id: Names the job. Running the same id again continues the same job.
    """

    def __init__(self, store: SessionStore, id: str) -> None:
        self.store = store
        self.id = id
        self._journal = Session(store, id=_journal_id(id))
        self._completed: dict[str, Any] | None = None
        self._running: set[str] = set()
        self._agent_steps: set[str] | None = None
        self._signals: dict[str, Any] = {}
        self._timers: dict[str, float] = {}

    def session(self, name: str) -> Session:
        """A durable session that belongs to this workflow (for an agent step)."""
        return Session(self.store, id=f"workflow:{self.id}:{name}")

    async def completed_steps(self) -> dict[str, Any]:
        """Names and recorded results of the steps that have finished."""
        if self._completed is None:
            await self._scan()
        assert self._completed is not None
        return self._completed

    # -- steps -------------------------------------------------------------------------

    async def step(self, name: str, function: Callable[[], Any], *, type: Any = None) -> Any:
        """Run ``function`` once per workflow id and remember its result.

        The result must be JSON-serialisable (pydantic models are). It comes
        back as plain JSON data on a later run unless you pass ``type``, in
        which case it is validated into that type.

        Args:
            name: The step's durable identity: unique within the workflow, and
                the same on every run.
            function: Called without arguments; sync, or returning an awaitable.
            type: Validate a recorded result into this type when it is replayed.

        Raises:
            ConfigurationError: Another step with this name is running at the
                same time.
        """
        completed = await self.completed_steps()
        if name in completed:
            return _revive(completed[name], type)
        with self._claim(name):
            result = function()
            if inspect.isawaitable(result):
                result = await result
            recorded = to_jsonable_python(result)
            await self._journal.append("workflow_step", {"name": name, "result": recorded})
            completed[name] = recorded
            return result

    async def map(
        self,
        name: str,
        items: Iterable[Any],
        function: Callable[[Any], Any],
        *,
        concurrency: int | None = None,
        type: Any = None,
    ) -> list[Any]:
        """Run ``function(item)`` as one durable step per item, concurrently.

        Item ``i`` is the step ``f"{name}[{i}]"``, so on a later run of the
        same workflow id only the items that did not finish run again. The
        items must therefore be the same, in the same order, on every run:
        results are matched by position.

            summaries = await wf.map("summarise", documents, summarise, concurrency=4)

        When items fail, the others still run to the end and are recorded.
        Then the first failure in input order is raised; a `WorkflowSuspended`
        is raised only when no item failed otherwise.

        Args:
            name: Prefix of the item steps' names.
            items: The items, in order.
            function: Called with one item; sync, or returning an awaitable.
                Its result must be JSON-serialisable, as for `step`.
            concurrency: How many items may run at once; ``None`` runs all of
                them at once. Finished items do not count.
            type: Validate replayed results into this type, as for `step`.
                Without it, an item that ran now returns its live result and a
                replayed one returns plain JSON data.

        Returns:
            The results, in the order of ``items``.

        Raises:
            ConfigurationError: ``concurrency`` is not ``None`` or at least 1.
        """
        if concurrency is not None and (isinstance(concurrency, bool) or concurrency < 1):
            raise ConfigurationError(
                f"Workflow.map(concurrency=...) must be None or at least 1, not {concurrency!r}."
            )
        values = list(items)
        if not values:
            return []
        completed = await self.completed_steps()
        gate = asyncio.Semaphore(concurrency) if concurrency is not None else None

        async def run_item(index: int, item: Any) -> Any:
            step_name = f"{name}[{index}]"
            if gate is None or step_name in completed:
                return await self.step(step_name, lambda: function(item), type=type)
            async with gate:
                return await self.step(step_name, lambda: function(item), type=type)

        tasks = [asyncio.create_task(run_item(index, item)) for index, item in enumerate(values)]
        try:
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            for task in tasks:  # only unfinished when this call was cancelled
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
        if failures:
            raise next(
                (failure for failure in failures if not isinstance(failure, WorkflowSuspended)),
                failures[0],
            )
        return outcomes

    async def run_agent(
        self,
        name: str,
        agent: Agent,
        prompt: Prompt,
        *,
        approvals: Mapping[str, Approval | bool] | None = None,
        results: Mapping[str, Any] | None = None,
        state: Mapping[str, Any] | None = None,
        **options: Any,
    ) -> RunResult:
        """Run an agent as a step, in its own durable session.

        If the step's run is open (it paused, failed or was interrupted), the
        agent run is resumed rather than restarted. A run that pauses for
        approvals or inputs is returned as it is and not recorded as
        finished: call ``run_agent`` again with ``approvals`` or ``results``
        to continue it.

            result = await wf.run_agent("plan", planner, "Plan the launch")
            if result.status == "paused":
                answers = {item.tool_call_id: ask_a_person(item) for item in result.pending_inputs}
                result = await wf.run_agent("plan", planner, "Plan the launch", results=answers)

        A finished step returns its recorded result without touching the
        agent. Its ``output`` is validated into the agent's ``output_type``
        when the agent that answered is ``agent``.

        Args:
            name: The step's durable identity; also names its session.
            agent: The agent to run or resume.
            prompt: The first message of the step's run; not used on resume.
            approvals: Decisions for the paused run's ``pending_approvals``.
                Approvals for calls that do not wait are ignored, as by
                ``agent.resume``.
            results: Results for the paused run's ``pending_inputs``.
            state: Values merged into the step session's state.
            **options: ``deps``, ``limits`` or ``settings``, passed to
                ``agent.run`` or ``agent.resume``.

        Raises:
            ConfigurationError: An option is not one that both ``agent.run``
                and ``agent.resume`` accept, or another step with this name is
                running at the same time.
            SessionError: ``results`` were given but the step has no open run
                to give them to, or a recorded output no longer matches the
                agent's ``output_type``.
        """
        unknown = sorted(set(options) - set(_SHARED_OPTIONS))
        if unknown:
            raise ConfigurationError(
                f"Workflow.run_agent() got {unknown}. Its options are passed to agent.run() when "
                "the step starts and to agent.resume() when it continues, so it accepts only "
                f"the ones both take: {', '.join(_SHARED_OPTIONS)}."
            )
        completed = await self.completed_steps()
        if name in completed:
            return self._revive_result(name, completed[name], agent)
        with self._claim(name):
            session = self.session(name)
            current = await session.state()
            if current.open_run_id is not None and current.messages:
                result = await agent.resume(
                    session, approvals=approvals, results=results, state=state, **options
                )
            else:
                if results:
                    raise SessionError(
                        f"Workflow {self.id!r}: step {name!r} has no open run, so the results= "
                        f"for {sorted(results)} have nothing to answer. The step was never "
                        "started, or its last run ended. Call run_agent without results= to "
                        "start it."
                    )
                await self._note_agent_step(name, session)
                result = await agent.run(prompt, session=session, state=state, **options)
            if result.status == "completed":
                recorded = to_jsonable_python(result)
                await self._journal.append("workflow_step", {"name": name, "result": recorded})
                completed[name] = recorded
            return result

    # -- signals and timers --------------------------------------------------------------

    @classmethod
    async def send_signal(
        cls, store: SessionStore, workflow_id: str, name: str, value: Any = None
    ) -> None:
        """Record a signal for a workflow, from any process.

        The workflow reads it with `wait_for_signal`, now or on a later run.
        A signal is recorded once per name: when the same name is sent again,
        the first value stays the one the workflow sees, so a replay always
        takes the same path.

        Args:
            store: The store the workflow uses.
            workflow_id: The workflow's id.
            name: The signal's name, as the workflow waits for it.
            value: JSON-serialisable data (pydantic models are).

        Raises:
            ConfigurationError: ``value`` cannot be stored as JSON.
        """
        try:
            recorded = to_jsonable_python(value)
        except PydanticSerializationError as error:
            raise ConfigurationError(
                f"The value of signal {name!r} cannot be stored as JSON ({error}). Send strings, "
                "numbers, booleans, None, lists, dicts or pydantic models."
            ) from error
        journal = Session(store, id=_journal_id(workflow_id))
        await journal.append("workflow_signal", {"name": name, "value": recorded})

    async def wait_for_signal(self, name: str, *, type: Any = None) -> Any:
        """Return the value of signal ``name``, or suspend the workflow until it is sent.

            decision = await wf.wait_for_signal("approval", type=Decision)

        The first time the signal is missing, the wait is recorded, so
        `status` lists it in ``pending_signals``.

        Args:
            name: The signal's name.
            type: Validate the value into this type.

        Raises:
            WorkflowSuspended: The signal has not been sent yet. Stop, and run
                the workflow again once it has been.
        """
        if name not in self._signals:
            journal = await self._scan()
            if name not in journal.signals:
                if name not in journal.waiting:
                    await self._journal.append("workflow_waiting", {"signal": name})
                raise WorkflowSuspended(name, workflow_id=self.id)
        return _revive(self._signals[name], type)

    async def signals(self) -> dict[str, Any]:
        """Every signal sent to this workflow so far, by name, as recorded (plain JSON)."""
        return dict((await self._scan()).signals)

    async def wait_for_timer(self, name: str, seconds: float) -> None:
        """Return once the durable timer ``name`` is due, or suspend the workflow until then.

        The first call starts the timer: it records that it is due ``seconds``
        from now. Later calls, in this run or a later one, use that recorded
        time, whatever ``seconds`` they pass.

            await wf.step("remind", send_reminder)
            await wf.wait_for_timer("grace-period", 3 * 24 * 3600)
            await wf.step("close", close_ticket)

        Args:
            name: The timer's name, unique within the workflow.
            seconds: How long after the first call the timer is due.

        Raises:
            ConfigurationError: ``seconds`` is negative or not finite.
            WorkflowSuspended: The timer is not due yet; ``resume_at`` says when it is.
        """
        if not (math.isfinite(seconds) and seconds >= 0):
            raise ConfigurationError(
                f"wait_for_timer({name!r}, seconds=...) needs a finite number of seconds, "
                f"0 or more, not {seconds!r}."
            )
        due = self._timers.get(name)
        if due is None:
            journal = await self._scan()
            due = journal.timers.get(name)
            if due is None:
                due = time.time() + seconds
                await self._journal.append("workflow_timer", {"name": name, "due": due})
                self._timers[name] = due
        if time.time() < due:
            raise WorkflowSuspended(timer=name, resume_at=due, workflow_id=self.id)

    # -- status ---------------------------------------------------------------------------

    async def status(self) -> WorkflowStatus:
        """Read where the workflow stands from the store, without running anything.

        Agent steps are listed when they were started by qagents 0.2 or later,
        which records them in the workflow journal.
        """
        journal = await self._scan()
        now = time.time()
        open_steps: list[AgentStepStatus] = []
        for name in journal.agent_steps:
            if name in journal.completed:
                continue
            session = self.session(name)
            entries = await session.entries()
            state = derive_state(entries)
            if state.open_run_id is None:
                continue
            tracked = SessionStatus(session_id=session.id)
            for entry in entries:
                advance_status(tracked, entry)
            open_steps.append(
                AgentStepStatus(
                    name=name,
                    session_id=session.id,
                    run_id=state.open_run_id,
                    status="running" if tracked.status == "idle" else tracked.status,
                    pending_approvals=state.pending_approvals,
                    pending_inputs=state.pending_inputs,
                )
            )
        return WorkflowStatus(
            id=self.id,
            completed_steps=list(journal.completed),
            signals=list(journal.signals),
            pending_signals=[name for name in journal.waiting if name not in journal.signals],
            pending_timers={name: due for name, due in journal.timers.items() if due > now},
            open_agent_steps=open_steps,
        )

    # -- plumbing -------------------------------------------------------------------------

    async def _scan(self) -> _Journal:
        """Read the workflow journal and refresh what this object remembers of it."""
        journal = _Journal()
        for entry in await self._journal.entries():
            kind, data = entry.kind, entry.data
            if kind == "workflow_step":
                journal.completed[data["name"]] = data["result"]
            elif kind == "workflow_signal":
                journal.signals.setdefault(data["name"], data.get("value"))
            elif kind == "workflow_waiting":
                if data["signal"] not in journal.waiting:
                    journal.waiting.append(data["signal"])
            elif kind == "workflow_timer":
                journal.timers.setdefault(data["name"], float(data["due"]))
            elif kind == "workflow_agent_started":
                if data["name"] not in journal.agent_steps:
                    journal.agent_steps.append(data["name"])
        if self._completed is None:
            self._completed = dict(journal.completed)
        else:
            for name, result in journal.completed.items():
                self._completed.setdefault(name, result)
        for name, value in journal.signals.items():
            self._signals.setdefault(name, value)
        for name, due in journal.timers.items():
            self._timers.setdefault(name, due)
        self._agent_steps = (self._agent_steps or set()) | set(journal.agent_steps)
        return journal

    @contextlib.contextmanager
    def _claim(self, name: str) -> Iterator[None]:
        """Mark step ``name`` as running in this object, refusing a second one at once."""
        if name in self._running:
            raise ConfigurationError(
                f"Workflow {self.id!r} has two steps named {name!r} running at once. "
                "Step names must be unique within a workflow."
            )
        self._running.add(name)
        try:
            yield
        finally:
            self._running.discard(name)

    async def _note_agent_step(self, name: str, session: Session) -> None:
        """Record an agent step's first run in the journal, so `status` can find it."""
        if self._agent_steps is None:
            await self._scan()
        assert self._agent_steps is not None
        if name not in self._agent_steps:
            self._agent_steps.add(name)
            await self._journal.append(
                "workflow_agent_started", {"name": name, "session_id": session.id}
            )

    def _revive_result(self, name: str, recorded: Any, agent: Agent) -> RunResult:
        """A finished agent step's recorded result, with its output in the agent's type."""
        result = RunResult.model_validate(recorded)
        # After a handoff the answer has another agent's type: leave it as recorded.
        answered_here = result.agent == agent.name
        if agent.output_type is not None and result.output is not None and answered_here:
            try:
                result.output = TypeAdapter(agent.output_type).validate_python(result.output)
            except ValidationError as error:
                raise SessionError(
                    f"Workflow {self.id!r}: the recorded output of step {name!r} does not match "
                    f"the output_type of agent {agent.name!r} any more:\n{error}\n"
                    "Restore the output type the step finished with, or give the step a new "
                    "name so that it runs again."
                ) from error
        result._session = self.session(name)
        return result


def _journal_id(workflow_id: str) -> str:
    return f"workflow:{workflow_id}"


def _revive(value: Any, type: Any) -> Any:
    return value if type is None else TypeAdapter(type).validate_python(value)
