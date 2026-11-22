"""A long research task that survives the crash of its worker.

A research agent plans with a todo list (``todo_tools``, shown to it on every turn by
``render_todos`` in its instructions), keeps what it learns in notes (``notes_tools``, Markdown
files under the work directory), reads a fixture survey library through ``search``, and has
old search results cleared from its context (``ToolResultClearing``). It runs in the background
under an ``AgentRuntime`` over a SQLite journal. Worker A starts the run and dies in the middle
of its second search: its database connection is gone, so it can neither renew nor release its
lease, and its task is cancelled where it stands. Once the short lease expires, worker B's
``recover()`` takes the run over and finishes it. The finished tool calls are not run again,
the interrupted search is (it is read-only), and the plan and the notes carry on.

Mirrors: Deep Agents (``deepagents``) on LangGraph's durable execution. Deep Agents gives the
model ``write_todos``, a file system (``write_file``, ``read_file``, ...) and context
management; here they are ``todo_tools``, ``notes_tools`` and ``ToolResultClearing`` (like
LangChain's ``ClearToolUsesEdit``). LangGraph saves a checkpoint per super-step of a thread,
and after a crash your code invokes the thread again (``graph.invoke(None, config)``), which
re-runs the interrupted node in full. In qagents the journal records every tool start and
result, a lease names the one worker driving a run, and any worker's ``recover()`` (or
``run_forever()``) resumes a run whose worker stopped renewing its lease: only the tool call
that was interrupted runs again, and only because it is read-only.

Run it:

    python -m examples.patterns.long_running_research --provider anthropic
    python -m examples.patterns.long_running_research --provider qcloud
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Iterator
from typing import Any

from examples.harness import ExampleContext, ExampleOutcome, main
from qagents import (
    Agent,
    AgentRuntime,
    Hooks,
    Limits,
    RunContext,
    RunHandle,
    Session,
    SQLiteStore,
    ToolCallPart,
    ToolResultClearing,
    tool,
)
from qagents.session import Entry
from qagents.toolkits import notes_tools, render_todos, todo_tools
from qagents.toolkits.planning import read_todo_list

NAME = "long_running_research"
DESCRIPTION = "A planning research agent whose run is recovered by a second worker after a crash"
INSPIRED_BY = "Deep Agents on LangGraph (todos, files, context editing, recovery)"

LEASE_TTL_S = 2.0
"""Seconds a lease lasts without a heartbeat: worker B takes over about this long after A dies."""
SESSION = "research:data-centre-sites"
CRASH_AT_SEARCH = 2
"""Worker A dies while it runs this search call."""

SURVEYS = {  # fixture data: the "library" the agent researches
    "Larissa": (
        "Site survey: Larissa (Thessaly), plot L-4 north of the ring road.\n"
        "Grid: 40 MW of spare capacity at the Larissa II substation; connection in 14 months.\n"
        "Power price: 92 EUR/MWh on a ten-year contract.\n"
        "Fibre: 4 ms to Athens over two independent routes.\n"
        "Cooling: mean summer high 33 C; dry coolers need adiabatic assist in July and August.\n"
        "Risks: the eastern edge of the plot lies in the Pineios flood zone (1-in-100 years).\n"
        "Land: 6 ha at 38 EUR/m2. Permits: zoning already industrial.\n"
        "Notes: the surveyor recommends raising the ground floor by 1.2 m on the eastern side."
    ),
    "Patras": (
        "Site survey: Patras (Western Greece), plot P-2 in the industrial area of Patras port.\n"
        "Grid: 18 MW of spare capacity at the port substation; more needs a new line (4 years).\n"
        "Power price: 78 EUR/MWh on a ten-year contract.\n"
        "Fibre: 3 ms to the Italy landing station, 6 ms to Athens.\n"
        "Cooling: sea-water cooling possible; mean summer high 31 C.\n"
        "Risks: seismic zone II; foundations must follow the stricter EC8 class.\n"
        "Land: 4 ha at 55 EUR/m2. Permits: port authority approval needed (about 9 months).\n"
        "Notes: the port authority prefers tenants that reuse waste heat."
    ),
    "Kavala": (
        "Site survey: Kavala (Eastern Macedonia), plot K-7 near the Nea Karvali gas plant.\n"
        "Grid: 55 MW of spare capacity at the Nea Karvali substation; connection in 10 months.\n"
        "Power price: 85 EUR/MWh on a ten-year contract.\n"
        "Fibre: 7 ms to Athens, 5 ms to Sofia; a second route is planned for 2027.\n"
        "Cooling: mean summer high 30 C; free cooling about 4,900 hours a year.\n"
        "Risks: low seismic and flood risk; the gas plant needs a 300 m safety distance.\n"
        "Land: 8 ha at 21 EUR/m2. Permits: zoning already industrial.\n"
        "Notes: the regional authority offers a 20 percent grant for the grid connection."
    ),
}
RECOMMENDED = "Kavala"  # lowest power price among the sites with at least 30 MW spare

INSTRUCTIONS = """\
You are a research agent working through a long task one step at a time. Use only the survey \
library (the search tool), never your own knowledge of the sites.

Task: compare three candidate sites for a data centre (Larissa, Patras and Kavala) and \
recommend one.

How to work:
1. First call write_todos with four pending items: "Research Larissa", "Research Patras", \
"Research Kavala" and "Write the recommendation".
2. Then, for each site in that order: call search with that one site name; call save_note, \
named after the site in lower case, with its power price, spare grid capacity and fibre latency; \
then call write_todos with that site's item completed. Finish one site before you start the next.
3. Old search results are removed from the conversation to save space. Your notes keep the \
facts, and read_note gives them back.
4. When the three sites are done, call write_todos with every item completed. Then reply with \
one line per site and the site you recommend: the lowest power price among the sites with at \
least 30 MW of spare grid capacity. Then stop."""

TASK = "Research the three candidate sites and recommend one for our data centre."


@tool(read_only=True)
def search(query: str) -> str:
    """Search the site-survey library. Returns the full survey of every site the query names.

    Args:
        query: A site name, for example "Larissa".
    """
    hits = [survey for site, survey in SURVEYS.items() if site.lower() in query.lower()]
    return "\n\n".join(hits) or f"No survey found. The library has: {', '.join(SURVEYS)}."


def instructions(ctx: RunContext[Any]) -> str:
    """The fixed instructions plus the current plan, so the agent sees it on every turn."""
    plan = render_todos(ctx.state)
    return f"{INSTRUCTIONS}\n\n{plan}" if plan else INSTRUCTIONS


# -- simulating the crash of worker A ----------------------------------------------------------


class WorkerConnection(SQLiteStore):
    """Worker A's own connection to the journal file. ``crash()`` cuts it, as if the process
    had died: from then on it can neither write, renew its lease nor release it."""

    alive = True

    def crash(self) -> None:
        self.alive = False

    def _check(self) -> None:
        if not self.alive:
            raise ConnectionError("worker-a is gone")

    async def append_leased(self, session_id: str, entries: list[Entry], owner: str) -> None:
        self._check()
        await super().append_leased(session_id, entries, owner)

    async def renew_lease(self, session_id: str, owner: str, ttl_s: float) -> bool:
        self._check()
        return await super().renew_lease(session_id, owner, ttl_s)

    async def release_lease(self, session_id: str, owner: str) -> None:
        self._check()
        await super().release_lease(session_id, owner)


class Freeze(Hooks):
    """Fault injection on worker A: the given call of ``tool`` never returns."""

    def __init__(self, tool: str, call: int) -> None:
        self.tool, self.call = tool, call
        self.seen = 0
        self.reached = asyncio.Event()

    async def before_tool(self, ctx: RunContext[Any], call: ToolCallPart) -> None:
        if call.name == self.tool:
            self.seen += 1
            if self.seen == self.call:
                self.reached.set()
                await asyncio.Event().wait()  # hangs until the task is cancelled


class LogLines(logging.Handler):
    """Prints the runtime's log records as one line each, without tracebacks."""

    def emit(self, record: logging.LogRecord) -> None:
        print(f"  [{record.name}] {record.getMessage()}", flush=True)


def brief(call: ToolCallPart) -> str:
    """A tool call's arguments in a few words; a plan as the status of each item."""
    todos = call.arguments.get("todos") if call.name == "write_todos" else None
    if isinstance(todos, list):
        return ", ".join(str(item.get("status")) for item in todos if isinstance(item, dict))
    shown = json.dumps(call.arguments, ensure_ascii=False)
    return shown if len(shown) <= 70 else shown[:67] + "..."


async def follow(handle: RunHandle, worker: str) -> None:
    """Print the tool calls of a background run as they happen."""
    with contextlib.suppress(Exception):  # the run's own error is reported by handle.wait()
        async for event in handle.events():
            if event.type == "tool_call_started":
                print(f"  {worker} -> {event.call.name}({brief(event.call)})", flush=True)


@contextlib.contextmanager
def runtime_logs() -> Iterator[None]:
    """Print what the runtime logs (a lease it could not release, a recovery) while inside."""
    logger = logging.getLogger("qagents.runtime")
    handler, level = LogLines(), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)


def tool_calls(entries: list[Entry]) -> tuple[dict[str, str], set[str]]:
    """The tool calls a stretch of journal started (id -> tool name) and those it finished."""
    started = {e.data["tool_call_id"]: e.data["name"] for e in entries if e.kind == "tool_started"}
    finished = {e.data["result"]["tool_call_id"] for e in entries if e.kind == "tool_result"}
    return started, finished


# -- the example -----------------------------------------------------------------------------


async def run(ctx: ExampleContext) -> ExampleOutcome:
    journal, notes = ctx.workdir / "journal.db", ctx.workdir / "notes"
    researcher = Agent(
        ctx.model,
        name="researcher",
        instructions=instructions,
        tools=[todo_tools(), notes_tools(notes), search],
        # Surveys are long: once a request reaches 1,500 tokens, keep the three newest tool
        # results and replace older ones with a placeholder (the notes keep the facts).
        compactor=ToolResultClearing(trigger_tokens=1_500, keep_last=3),
        limits=Limits(max_turns=24, max_cost_usd=0.40, timeout_s=600),
        hooks=ctx.hooks,
    )
    freeze = Freeze("search", CRASH_AT_SEARCH)
    connection_a = WorkerConnection(journal)  # worker A's own connection: it will die
    store = SQLiteStore(journal)  # worker B's connection, also used to read the journal
    worker_a = AgentRuntime(
        connection_a,
        [researcher.copy(hooks=[*researcher.hooks, freeze])],
        worker_id="worker-a",
        lease_ttl_s=LEASE_TTL_S,
    )
    worker_b = AgentRuntime(store, [researcher], worker_id="worker-b", lease_ttl_s=LEASE_TTL_S)

    async with contextlib.AsyncExitStack() as stack:
        stack.push_async_callback(store.aclose)
        stack.push_async_callback(connection_a.aclose)
        await stack.enter_async_context(worker_a)  # shut down on the way out
        await stack.enter_async_context(worker_b)
        stack.enter_context(runtime_logs())

        ctx.log("worker-a starts the run in the background")
        handle_a = await worker_a.start("researcher", TASK, session_id=SESSION)
        printing = asyncio.create_task(follow(handle_a, "worker-a"))  # ends with the run
        frozen = asyncio.create_task(freeze.reached.wait())
        await asyncio.wait({frozen, printing}, return_when=asyncio.FIRST_COMPLETED)
        if not frozen.done():
            frozen.cancel()
            ended = handle_a.result.status if handle_a.result else f"failed: {handle_a.error!r}"
            return ExampleOutcome(
                ok=False,
                summary=f"the run ended on worker-a ({ended}) before the simulated crash: the "
                f"model searched {freeze.seen} time(s), fewer than {CRASH_AT_SEARCH}",
            )

        await asyncio.sleep(0.05)  # let the printer show the call worker A is frozen in
        ctx.log(f"worker-a dies during search #{CRASH_AT_SEARCH}: no release, no heartbeat")
        connection_a.crash()
        await worker_a.shutdown(timeout_s=0)  # cancels the run task at once, where it stands
        await printing
        written_by_a = len(await store.load(SESSION))
        status = await store.status(SESSION)
        left = max((status.lease_expires_at or 0.0) - time.time(), 0.0)
        ctx.log(
            f"the journal says {status.status!r}; {status.lease_owner} holds the lease "
            f"for {left:.1f} s more"
        )

        # A real worker calls run_forever(), which calls recover() every few seconds.
        ctx.log("worker-b calls recover() until the lease has expired")
        crashed_at = time.monotonic()
        while not (recovered := await worker_b.recover()):
            if time.monotonic() - crashed_at > LEASE_TTL_S * 5:
                return ExampleOutcome(ok=False, summary="worker-b never recovered the run")
            await asyncio.sleep(0.2)
        (handle_b,) = recovered
        ctx.log(f"worker-b took the run over after {time.monotonic() - crashed_at:.1f} s")
        printing = asyncio.create_task(follow(handle_b, "worker-b"))
        result = await handle_b.wait()
        await printing

        entries = await store.load(SESSION)
        values = (await Session(store, id=SESSION).state()).values

    started_a, finished_a = tool_calls(entries[:written_by_a])
    started_b, _ = tool_calls(entries[written_by_a:])
    interrupted = [name for call_id, name in started_a.items() if call_id not in finished_a]
    rerun = [name for call_id, name in started_b.items() if call_id in started_a]
    print(
        f"journal: worker-a wrote {written_by_a} entries; of its {len(started_a)} tool calls, "
        f"{', '.join(interrupted) or 'none'} never finished. worker-b ran "
        f"{', '.join(rerun) or 'none of them'} again and wrote {len(entries) - written_by_a} more."
    )
    print(render_todos(values) or "(no todo list)")
    print(f"answer:\n{result.text}")

    todos = read_todo_list(values)
    completed = bool(todos) and all(item.status == "completed" for item in todos)
    ok = result.status == "completed" and handle_b.resumed and completed
    summary = (
        f"worker-b recovered the run and completed it with all {len(todos)} todos done; it "
        f"recommends {RECOMMENDED if RECOMMENDED in result.text else 'another site'}"
        if ok
        else f"{result.status} ({result.stop_reason}); todos: {[t.status for t in todos]}"
    )
    return ExampleOutcome(
        ok=ok,
        summary=summary,
        output=result.text,
        details={
            "tools_on_worker_a": list(started_a.values()),
            "tools_on_worker_b": list(started_b.values()),
            "interrupted_on_worker_a": interrupted,
            "run_again_on_worker_b": rerun,
            "worker_a_handle": handle_a.status,
            "compactions": sum(entry.kind == "compaction" for entry in entries),
            "notes": sorted(path.stem for path in notes.glob("*.md")),
            "todos": [item.model_dump() for item in todos],
            "recommends_expected_site": RECOMMENDED in result.text,
        },
    )


if __name__ == "__main__":
    main(run, name=NAME, description=DESCRIPTION, inspired_by=INSPIRED_BY)
