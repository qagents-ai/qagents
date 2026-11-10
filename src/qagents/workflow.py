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

Use ``asyncio.gather`` to run steps in parallel; branches do not wait for
each other.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from pydantic import TypeAdapter
from pydantic_core import to_jsonable_python

from .agent import Agent, Prompt
from .errors import ConfigurationError
from .permissions import Approval
from .results import RunResult
from .session import Session, SessionStore

T = TypeVar("T")


class Workflow:
    def __init__(self, store: SessionStore, id: str) -> None:
        self.store = store
        self.id = id
        self._journal = Session(store, id=f"workflow:{id}")
        self._completed: dict[str, Any] | None = None
        self._running: set[str] = set()

    def session(self, name: str) -> Session:
        """A durable session that belongs to this workflow (for an agent step)."""
        return Session(self.store, id=f"workflow:{self.id}:{name}")

    async def completed_steps(self) -> dict[str, Any]:
        """Names and recorded results of the steps that have finished."""
        if self._completed is None:
            self._completed = {
                entry.data["name"]: entry.data["result"]
                for entry in await self._journal.entries()
                if entry.kind == "workflow_step"
            }
        return self._completed

    async def step(self, name: str, function: Callable[[], Any], *, type: Any = None) -> Any:
        """Run ``function`` once per workflow id and remember its result.

        The result must be JSON-serialisable (pydantic models are). It comes
        back as plain JSON data on a later run unless you pass ``type``, in
        which case it is validated into that type.
        """
        completed = await self.completed_steps()
        if name in completed:
            return _revive(completed[name], type)
        if name in self._running:
            raise ConfigurationError(
                f"Workflow {self.id!r} has two steps named {name!r} running at once. "
                "Step names must be unique within a workflow."
            )
        self._running.add(name)
        try:
            result = function()
            if inspect.isawaitable(result):
                result = await result
            recorded = to_jsonable_python(result)
            await self._journal.append("workflow_step", {"name": name, "result": recorded})
            completed[name] = recorded
            return result
        finally:
            self._running.discard(name)

    async def run_agent(
        self,
        name: str,
        agent: Agent,
        prompt: Prompt,
        *,
        approvals: Mapping[str, Approval | bool] | None = None,
        **options: Any,
    ) -> RunResult:
        """Run an agent as a step, in its own durable session.

        If the step was interrupted, the agent run is resumed rather than
        restarted. A run that pauses for approval is returned as it is and not
        recorded as finished: call ``run_agent`` again with ``approvals`` to
        continue it.
        """
        completed = await self.completed_steps()
        if name in completed:
            return RunResult.model_validate(completed[name])
        session = self.session(name)
        state = await session.state()
        if state.open_run_id is not None and state.messages:
            result = await agent.resume(session, approvals=approvals, **options)
        else:
            result = await agent.run(prompt, session=session, **options)
        if result.status == "completed":
            recorded = to_jsonable_python(result)
            await self._journal.append("workflow_step", {"name": name, "result": recorded})
            completed[name] = recorded
        return result


def _revive(value: Any, type: Any) -> Any:
    return value if type is None else TypeAdapter(type).validate_python(value)
