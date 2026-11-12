"""Two multi-agent patterns built from linked child runs: fan-out and routing.

**Fan-out (orchestrator and workers).** `fan_out` turns an agent into a tool
that takes a list of tasks and runs one child run of the agent per task, in
parallel. The orchestrating model splits the work, the workers each see only
their own task, and the orchestrator reads one short answer per task:

    researcher = Agent(model, name="researcher", description="Finds facts.", tools=[search])
    lead = Agent(
        model,
        name="lead",
        instructions="Split the question into independent research tasks, then write the answer.",
        tools=[fan_out(researcher, max_parallel=4)],
    )
    result = await lead.run("Compare the solar markets of Spain, Greece and Italy.")

Every worker is a child run of the calling tool call (``parent=ctx``), like an
agent used as a tool: it shares the caller's remaining token, cost and time
budget, uses the caller's propagating hooks (tracers) and ``on_ask``, and
counts toward ``Limits.max_depth``. Its usage is added to the caller's run
however it ends (answered, failed, cancelled).

**Routing.** `route` asks a classifier agent which agent should handle a
prompt, then runs that agent on the same prompt. It is a plain function, so
the choice stays visible in your code:

    triage = Agent(model, name="triage", instructions="Classify the request.",
                   output_type=Literal["billing", "technical"])
    routed = await route(triage, {"billing": billing, "technical": support}, prompt)
    print(routed.route, routed.used_default, routed.result.output)

The strings this module sends to a model are the module-level constants below.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import re
import typing
from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from . import prompts
from .agent import Agent, Prompt
from .errors import ConfigurationError, MaxDepthExceeded, OutputValidationError
from .events import RunFinished, RunStarted, SubagentEvent
from .messages import Usage
from .results import RunResult
from .session import Session
from .tools import RunContext, Tool

logger = logging.getLogger("qagents")


FAN_OUT_TIMEOUT_S = 600.0
"""Default ``timeout`` of a `fan_out` call: seconds for all of its tasks together."""


def fan_out(
    agent: Agent,
    *,
    name: str | None = None,
    description: str | None = None,
    max_parallel: int = 4,
    max_tasks: int | None = 8,
    read_only: bool = False,
    timeout: float | None = FAN_OUT_TIMEOUT_S,
) -> Tool:
    """Make a tool that runs ``agent`` on several tasks in parallel.

    The tool takes one argument, ``tasks``: a list of task texts. Each task
    is one child run of ``agent`` (``parent=ctx``) in its own fresh
    conversation, at most ``max_parallel`` at a time. The tool returns a JSON
    list in task order with ``{"task", "answer"}`` for a task the agent
    finished, or ``{"task", "error"}`` for one that stopped early (a limit, a
    pause, a guardrail, a refusal by the provider), raised, or would nest
    deeper than ``Limits.max_depth``. One failed task does not stop the others.

    The child runs inherit from the calling run: the remaining token, cost
    and time budget (shared live, so children running together cannot spend
    more than the caller has left), the hooks with ``propagate = True``, the
    ``on_ask`` handler, and the nesting depth. Each child's usage is added to
    the caller's run when the child ends, however it ends (also when it
    raised or was cancelled), and their events reach the caller's stream as
    `SubagentEvent`, with ``agent`` set to ``"<name>[<index>]"`` so the tasks
    can be told apart.

    If the calling run is cancelled or times out, the tool call is cancelled
    and so are the child runs that are still working.

    Args:
        agent: The worker agent.
        name: Tool name; defaults to ``"<agent name>_parallel"``.
        description: Tool description; defaults to `prompts.FAN_OUT_DESCRIPTION`
            filled with the agent's name and description.
        max_parallel: How many child runs may work at once.
        max_tasks: How many tasks one call may give; a call with more gets an
            error result asking for fewer. ``None`` removes the limit.
        read_only: Declare the whole fan-out free of side effects. Only for
            workers whose tools change nothing. It has three effects: the call
            may run in parallel with other read-only calls; it is allowed
            under ``Permissions(mode="read_only")``; and when the calling run
            is interrupted while the call works (a crash, a cancel, a
            deploy), a resume runs the whole call again: every task and every
            worker model request, nothing reused. With the default
            ``False``, a resume tells the model the call's outcome is unknown.
        timeout: Seconds for the whole call, all tasks included. Defaults to
            `FAN_OUT_TIMEOUT_S` (10 minutes), since agents need far longer
            than one tool call; ``None`` uses ``Limits.tool_timeout_s``.

    Raises:
        ConfigurationError: ``max_parallel`` or ``max_tasks`` is below 1.
    """
    if max_parallel < 1:
        raise ConfigurationError(
            f"fan_out(max_parallel=...) must be at least 1, not {max_parallel}."
        )
    if max_tasks is not None and max_tasks < 1:
        raise ConfigurationError(
            f"fan_out(max_tasks=...) must be None or at least 1, not {max_tasks}."
        )
    tool_name = name or f"{_tool_safe(agent.name)}_parallel"

    async def run_tasks(ctx: RunContext[Any], tasks: list[str]) -> list[dict[str, Any]]:
        gate = asyncio.Semaphore(max_parallel)

        async def work(index: int, task: str) -> dict[str, Any]:
            async with gate:
                return await _run_task(agent, ctx, f"{agent.name}[{index}]", task)

        workers = [asyncio.create_task(work(index, task)) for index, task in enumerate(tasks)]
        try:
            return list(await asyncio.gather(*workers))
        finally:
            for worker in workers:  # still running only when this call was cancelled
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

    arguments = create_model(
        f"{tool_name}_arguments",
        __config__=ConfigDict(extra="forbid"),
        tasks=(
            list[Annotated[str, Field(min_length=1)]],
            Field(
                min_length=1, max_length=max_tasks, description=prompts.FAN_OUT_TASKS_DESCRIPTION
            ),
        ),
    )
    return Tool(
        run_tasks,
        name=tool_name,
        description=description
        or prompts.FAN_OUT_DESCRIPTION.format(
            name=agent.name, description=agent.description
        ).strip(),
        arguments_model=arguments,
        takes_context=True,
        read_only=read_only,
        timeout=timeout,
    )


async def _run_task(agent: Agent, ctx: RunContext[Any], label: str, task: str) -> dict[str, Any]:
    """Run one task as a child run of ``ctx`` and describe how it ended.

    The child's usage reaches the caller's run when the child ends, however it
    ends (the child run hands it over itself), so nothing is added here.
    """
    try:
        stream = agent.stream(task, deps=ctx.deps, parent=ctx)
        async with stream:
            async for event in stream:
                if not isinstance(event, RunStarted | RunFinished):
                    ctx.emit(SubagentEvent(agent=label, tool_call_id=ctx.tool_call_id, event=event))
    except MaxDepthExceeded as error:  # expected in recursive setups: no traceback
        too_deep = prompts.SUBAGENT_TOO_DEEP.format(
            name=agent.name, depth=error.depth, limit=error.limit
        )
        return {"task": task, "error": too_deep}
    except Exception as error:  # one broken task must not take the others down
        logger.exception("Child run %s of tool call %s raised", label, ctx.tool_call_id)
        problem = f"{type(error).__name__}: {error}"
        return {
            "task": task,
            "error": prompts.FAN_OUT_TASK_FAILED.format(name=agent.name, error=problem),
        }
    result = stream.result
    assert result is not None
    if result.status != "completed" or result.stop_reason == "content_filter":
        # A refusal is not an answer, even though the run ended "completed".
        stopped = prompts.SUBAGENT_STOPPED.format(
            name=agent.name, reason=result.stop_reason or result.status
        )
        return {"task": task, "error": stopped}
    if result.output is None and agent.output_type is None:
        return {"task": task, "answer": result.text}
    return {"task": task, "answer": result.output}


class RouteResult(BaseModel):
    """What `route` returns: the route taken and the runs that took it."""

    route: str | None
    """The route that handled the prompt; ``None`` when the classifier did not
    finish (it paused, hit a limit, was blocked or refused) and no route ran."""
    used_default: bool = False
    """The classifier's answer named no route, so ``default=`` was taken."""
    result: RunResult
    """The run of the chosen agent. When no route ran, the classifier's run,
    so ``result.status`` says why."""
    classification: RunResult
    """The classifier's run."""

    @property
    def usage(self) -> Usage:
        """Usage of the classifier and the chosen agent together."""
        if self.route is None:
            return self.classification.usage
        return self.classification.usage.add(self.result.usage)


async def route(
    classifier: Agent,
    routes: Mapping[str, Agent],
    prompt: Prompt,
    *,
    default: str | None = None,
    deps: Any = None,
    session: Session | None = None,
    state: Mapping[str, Any] | None = None,
    parent: RunContext[Any] | None = None,
    classifier_session: Session | None = None,
) -> RouteResult:
    """Ask ``classifier`` which agent should handle ``prompt``, then run that agent on it.

        triage = Agent(model, name="triage", output_type=Literal["billing", "technical"],
                       instructions="Which team should answer this customer message?")
        routed = await route(triage, {"billing": billing, "technical": support}, message)

    The classifier's answer is read from ``result.output``, so it works in
    every output mode. Give the classifier ``output_type=Literal[...]`` (or an
    `enum.Enum`) of the route names: the agent then checks the answer and asks
    the model to correct it. With a plain-text classifier, the answer must be
    a route name, give or take case, spaces, quotes and a final full stop.

    The classifier runs in a fresh in-memory conversation of its own and sees
    only ``prompt``, unless you pass ``classifier_session``: then it runs on
    that session, which journals the classification (for example
    ``wf.session("triage")`` inside a workflow) and which it continues. The
    chosen agent runs on ``session`` (a new in-memory one if you leave it
    out), so it continues that conversation. ``RouteResult.used_default``
    says when ``default`` was taken.

    Args:
        classifier: The agent that picks the route.
        routes: Route names and the agents that handle them.
        prompt: The message to classify and then handle.
        default: The route to take when a plain-text classifier's answer names
            no route.
        deps: Passed to both runs as ``ctx.deps``.
        session: The chosen agent's session.
        state: Merged into the chosen agent's session state.
        parent: The calling run's context, when called from a tool: both
            runs become its child runs, and their usage is added to it
            (also when one of them raises).
        classifier_session: The classifier's session; a new in-memory one
            when left out.

    Returns:
        The route taken, the chosen agent's run, and the classifier's run.

    Raises:
        ConfigurationError: ``routes`` is empty or holds something other than
            agents, ``default`` is not a route, or the classifier's
            ``output_type`` allows answers that are not routes.
        OutputValidationError: The classifier's answer names no route and
            there is no ``default``.
    """
    _check_routes(classifier, routes, default)
    classification = await classifier.run(
        prompt, deps=deps, session=classifier_session, parent=parent
    )
    if classification.status != "completed" or classification.stop_reason == "content_filter":
        return RouteResult(route=None, result=classification, classification=classification)
    chosen = _match_route(classification.output, routes)
    used_default = chosen is None
    if chosen is None:
        if default is None:
            raise OutputValidationError(
                f"Classifier {classifier.name!r} answered {classification.output!r}, which names "
                f"none of the routes ({', '.join(routes)}). Give it output_type=Literal[...] of "
                "the route names so that its answer is checked and corrected, or pass "
                "route(default=...)."
            )
        chosen = default
    result = await routes[chosen].run(
        prompt, deps=deps, session=session, state=state, parent=parent
    )
    return RouteResult(
        route=chosen, result=result, classification=classification, used_default=used_default
    )


def _check_routes(classifier: Agent, routes: Mapping[str, Agent], default: str | None) -> None:
    if not routes:
        raise ConfigurationError("route() needs at least one route.")
    for name, target in routes.items():
        if not isinstance(target, Agent):
            raise ConfigurationError(f"Route {name!r} is {target!r}; every route must be an Agent.")
    if default is not None and default not in routes:
        raise ConfigurationError(
            f"default={default!r} is not one of the routes: {', '.join(routes)}."
        )
    output_type = classifier.output_type
    if output_type is None:
        return
    if typing.get_origin(output_type) is typing.Literal:
        answers = list(typing.get_args(output_type))
    elif isinstance(output_type, type) and issubclass(output_type, enum.Enum):
        answers = [member.value for member in output_type]
    else:
        raise ConfigurationError(
            f"route() reads the answer of classifier {classifier.name!r} as a route name, but its "
            f"output_type is {output_type!r}. Use output_type=Literal[...] of the route names, an "
            "Enum of them, or no output_type."
        )
    strays = [answer for answer in answers if answer not in routes]
    if strays:
        raise ConfigurationError(
            f"Classifier {classifier.name!r} can answer {strays}, which are not routes "
            f"({', '.join(routes)}). Make its output_type list route names only."
        )


def _match_route(answer: Any, routes: Mapping[str, Agent]) -> str | None:
    """The route ``answer`` names, if any."""
    if isinstance(answer, enum.Enum):
        answer = answer.value
    if not isinstance(answer, str):
        return None
    if answer in routes:
        return answer
    cleaned, previous = answer, None
    while cleaned != previous:  # in any order: '"billing".', '**Billing**.', "'billing'."
        previous = cleaned
        cleaned = cleaned.strip().rstrip(".").strip().strip("\"'`*")
    folded = {key.casefold(): key for key in routes}
    return folded.get(cleaned.casefold())


def _tool_safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)[:48] or "agent"
