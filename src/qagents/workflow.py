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
the exact tool call (see `session`); an agent step whose run completed just
before the process died, before the step was recorded, is recorded from the
step's journal instead of running again. Make other side-effecting steps
idempotent.

Use ``asyncio.gather`` to run steps in parallel, or ``map`` to run one step
per item with an optional bound on concurrency; branches do not wait for
each other. A sync function passed to ``step`` runs on the event loop and
blocks it; ``map`` runs a sync function in a worker thread. To map an agent
over items so that each item resumes inside its own run after a crash, call
``run_agent`` once per item (``f"{name}[{index}]"``) under ``asyncio.gather``.

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

Journal entry kinds (in ``workflow-journal:<encoded-id>``; readers ignore kinds
they do not know):

| Kind | Meaning (data) |
|---|---|
| `workflow_step` | a step finished (`name`, `result`) |
| `workflow_agent_started` | an agent step bound a run (`name`, `session_id`, `run_id`) |
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
from urllib.parse import quote

from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError, to_jsonable_python

from .agent import Agent, Prompt, RunStream
from .errors import ConfigurationError, SessionError, WorkflowSuspended
from .events import Event
from .messages import Message, Usage
from .permissions import Approval
from .results import RunResult
from .session import (
    Entry,
    PendingApproval,
    PendingInput,
    Session,
    SessionStatus,
    SessionStore,
    advance_status,
    derive_state,
)
from .tools import _is_async_callable

T = TypeVar("T")

_SHARED_OPTIONS = ("deps", "limits", "settings")
"""Options of `Workflow.run_agent` that both ``agent.run`` and ``agent.resume`` accept."""

EventCallback = Callable[[Event], Any]
"""Called with every event of an agent step's run (`Workflow.run_agent`). May be async."""


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
    New journals and step sessions have separate prefixes and percent-encoded
    identities, so punctuation in ids cannot make them collide. Existing
    ``workflow:`` journals keep their original ids; a step session resolves
    its legacy id before its first read, write, or lease acquisition.

    Args:
        store: Where the workflow's journal and its agent steps' sessions live.
        id: Names the job. Running the same id again continues the same job.
    """

    def __init__(self, store: SessionStore, id: str) -> None:
        self.store = store
        self.id = id
        self._journal = Session(store, id=_journal_id(id))
        self._layout_ready = False
        self._legacy_layout = False
        self._completed: dict[str, Any] | None = None
        self._running: set[str] = set()
        self._agent_steps: set[str] | None = None
        self._agent_runs: dict[str, str] = {}
        self._signals: dict[str, Any] = {}
        self._timers: dict[str, float] = {}

    def session(self, name: str) -> Session:
        """A durable agent-step session; existing legacy ids resolve on first I/O."""
        return _StepSession(self, name)

    async def completed_steps(self) -> dict[str, Any]:
        """Names and recorded results of the steps that have finished (a copy)."""
        return dict(await self._completed_map())

    async def _completed_map(self) -> dict[str, Any]:
        """What this object remembers of the finished steps; the steps add to it."""
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
                A sync function runs on the event loop and blocks it while it
                runs; wrap blocking work as ``lambda: asyncio.to_thread(work)``.
            type: Validate a recorded result into this type when it is replayed.

        Raises:
            ConfigurationError: Another step with this name is running at the
                same time.
        """
        completed = await self._completed_map()
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
            function: Called with one item; async, or sync (it then runs in a
                worker thread, so sync items run concurrently too and do not
                block the event loop; a sync function that returns an
                awaitable has it awaited on the loop). Its result must be
                JSON-serialisable, as for `step`.
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
        completed = await self._completed_map()
        gate = asyncio.Semaphore(concurrency) if concurrency is not None else None

        async def call(item: Any) -> Any:
            if _is_async_callable(function):
                return await function(item)
            produced = await asyncio.to_thread(function, item)  # never block the loop
            if inspect.isawaitable(produced):
                produced = await produced
            return produced

        async def run_item(index: int, item: Any) -> Any:
            step_name = f"{name}[{index}]"
            if gate is None or step_name in completed:
                return await self.step(step_name, lambda: call(item), type=type)
            async with gate:
                return await self.step(step_name, lambda: call(item), type=type)

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
        on_event: EventCallback | None = None,
        **options: Any,
    ) -> RunResult:
        """Run an agent as a step, in its own durable session.

        If the step's run is open (it paused, failed or was interrupted), the
        agent run is resumed rather than restarted. A run that pauses for
        approvals or inputs is returned as it is and not recorded as
        finished: call ``run_agent`` again with ``approvals`` or ``results``
        to continue it. Calling it again without them resumes the run, which
        checks the permissions again and pauses again for what still waits
        (with a new ``run_paused`` entry); to read a paused step without
        running anything, use `status`.

        Only a ``completed`` run is recorded as the step's result. When the
        step's run ended otherwise (blocked by a guardrail, at a limit,
        cancelled), the next ``run_agent`` starts a new run in the step's
        session: it sends ``prompt`` again, and the model sees the earlier
        attempt. When the step's run completed but the process died before
        the step was recorded, the result is rebuilt from the step's journal
        and recorded, and the agent is not run again.

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
            on_event: Called with every event of the run as it happens (sync
                or async), to narrate a step live. If it raises, the run
                stops and stays open (like a stream closed early); calling
                ``run_agent`` again resumes it.
            **options: ``deps``, ``limits`` or ``settings``, passed to
                ``agent.run`` or ``agent.resume``.

        Raises:
            ConfigurationError: An option is not one that both ``agent.run``
                and ``agent.resume`` accept, or another step with this name is
                running at the same time.
            SessionError: ``results`` were given but the step has no open run
                to give them to, a recorded output no longer matches the
                agent's ``output_type``, or the step's run completed without
                being recorded and its journal (written before qagents 0.3)
                cannot rebuild the result.
        """
        unknown = sorted(set(options) - set(_SHARED_OPTIONS))
        if unknown:
            raise ConfigurationError(
                f"Workflow.run_agent() got {unknown}. Its options are passed to agent.run() when "
                "the step starts and to agent.resume() when it continues, so it accepts only "
                f"the ones both take: {', '.join(_SHARED_OPTIONS)}."
            )
        completed = await self._completed_map()
        if name in completed:
            return self._revive_result(name, completed[name], agent)
        with self._claim(name):
            session = self.session(name)
            entries = await session.entries()
            current = derive_state(entries)
            unrecorded = self._finished_unrecorded(name, session.id, entries)
            if unrecorded is not None:  # the bound run completed before the step was recorded
                await self._journal.append("workflow_step", {"name": name, "result": unrecorded})
                completed[name] = unrecorded
                return self._revive_result(name, unrecorded, agent)
            if current.open_run_id is not None and current.messages:
                bound = self._agent_runs.get(name)
                if bound is not None and current.open_run_id != bound:
                    raise SessionError(
                        f"Workflow step {name!r} belongs to run {bound!r}, but its session "
                        f"has another open run {current.open_run_id!r}; nothing was resumed."
                    )
                await self._note_agent_step(name, session, current.open_run_id)
                stream = agent.resume_stream(
                    session, approvals=approvals, results=results, state=state, **options
                )
                result = await _drive(stream, on_event)
            else:
                if results:
                    raise SessionError(
                        f"Workflow {self.id!r}: step {name!r} has no open run, so the results= "
                        f"for {sorted(results)} have nothing to answer. The step was never "
                        "started, or its last run ended. Call run_agent without results= to "
                        "start it."
                    )
                stream = agent.stream(prompt, session=session, state=state, **options)
                await self._note_agent_step(name, session, stream.run_id)
                result = await _drive(stream, on_event)
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
        workflow = cls(store, id=workflow_id)
        await workflow._resolve_layout()
        await workflow._journal.append("workflow_signal", {"name": name, "value": recorded})

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

    async def _resolve_layout(self) -> None:
        """Read legacy workflows at their original ids, and refuse ambiguous journals."""
        if self._layout_ready:
            return
        current = await self._journal.entries()
        if not current:
            legacy = Session(self.store, id=f"workflow:{self.id}")
            old = await legacy.entries()
            if old:
                self._check_journal(old, legacy.id)
                self._journal = legacy
                self._legacy_layout = True
        else:
            self._check_journal(current, self._journal.id)
        self._layout_ready = True

    @staticmethod
    def _check_journal(entries: list[Entry], session_id: str) -> None:
        if any(not entry.kind.startswith("workflow_") for entry in entries):
            raise SessionError(
                f"Workflow journal id {session_id!r} is already an agent-step session; "
                "choose another workflow id. Nothing was written."
            )

    async def _scan(self) -> _Journal:
        """Read the workflow journal and refresh what this object remembers of it."""
        await self._resolve_layout()
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
                if isinstance(data.get("run_id"), str):
                    self._agent_runs[data["name"]] = data["run_id"]
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

    async def _note_agent_step(self, name: str, session: Session, run_id: str) -> None:
        """Bind the step to this run before it starts; retries record a new binding."""
        if self._agent_steps is None:
            await self._scan()
        assert self._agent_steps is not None
        if self._agent_runs.get(name) != run_id:
            await self._journal.append(
                "workflow_agent_started",
                {"name": name, "session_id": session.id, "run_id": run_id},
            )
            self._agent_steps.add(name)
            self._agent_runs[name] = run_id

    def _finished_unrecorded(
        self, name: str, session_id: str, entries: list[Entry]
    ) -> dict[str, Any] | None:
        """Rebuild the bound run, falling back to the latest only for legacy journals."""
        bound = self._agent_runs.get(name)
        start = next(
            (
                index
                for index in range(len(entries) - 1, -1, -1)
                if entries[index].kind == "run_started"
                and (bound is None or entries[index].run_id == bound)
            ),
            None,
        )
        if start is None:
            return None
        run_id = entries[start].run_id
        own = [entry for entry in entries[start:] if entry.run_id == run_id]
        finish = next((entry for entry in own if entry.kind == "run_finished"), None)
        if finish is None or finish.data.get("status") != "completed":
            return None  # still open, or ended otherwise: the step runs again
        data = finish.data
        if "output" not in data:
            raise SessionError(
                f"Workflow {self.id!r}: the agent run of step {name!r} completed, but the step "
                "was not recorded (the process stopped in between), and its journal was written "
                "by a qagents that does not keep the output with the run. Rather than run the "
                "agent again, nothing was run: give the step a new name to run it again."
            )
        agent_name = entries[start].data.get("agent")
        for entry in own:
            if entry.kind == "agent_switched":
                agent_name = entry.data.get("agent", agent_name)
        finish_index = entries.index(finish)
        own = [entry for entry in entries[start : finish_index + 1] if entry.run_id == run_id]
        state = derive_state(entries[: finish_index + 1])
        usage = state.run_usage
        if added := data.get("added_usage"):
            usage = usage.add(Usage.model_validate(added))
        result = RunResult(
            run_id=run_id or "",
            session_id=session_id,
            agent=agent_name or "",
            status="completed",
            stop_reason=data.get("stop_reason"),
            output=data["output"],
            text=data.get("text") or "",
            messages=state.messages,
            new_messages=[
                Message.model_validate(entry.data["message"])
                for entry in own
                if entry.kind == "message"
            ],
            usage=usage,
            turns=state.run_turns,
        )
        return to_jsonable_python(result)

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


class _StepSession(Session):
    """Resolve a legacy step before reads, writes, or an agent takes its lease."""

    def __init__(self, workflow: Workflow, name: str) -> None:
        self._workflow, self._name = workflow, name
        super().__init__(workflow.store, id=_step_id(workflow.id, name))

    async def _resolve(self) -> None:
        await self._workflow._resolve_layout()
        if self._workflow._legacy_layout:
            self.id = f"workflow:{self._workflow.id}:{self._name}"


def _journal_id(workflow_id: str) -> str:
    return f"workflow-journal:{quote(workflow_id, safe='')}"


def _step_id(workflow_id: str, name: str) -> str:
    return f"workflow-step:{quote(workflow_id, safe='')}:{quote(name, safe='')}"


async def _drive(stream: RunStream, on_event: EventCallback | None) -> RunResult:
    """Run ``stream`` to its end, handing each event to ``on_event``."""
    async with stream:
        async for event in stream:
            if on_event is not None:
                outcome = on_event(event)
                if inspect.isawaitable(outcome):
                    await outcome
    assert stream.result is not None
    return stream.result


def _revive(value: Any, type: Any) -> Any:
    return value if type is None else TypeAdapter(type).validate_python(value)
