# AGENTS.md

Instructions for coding agents (Codex, Claude Code, Cursor, Copilot and others). Part 1 is for
agents that **write code that uses qagents**. Part 2 is for agents that **work on this
repository**.

## Part 1: writing code with qagents

### Look it up, never guess

qagents is small and young. Names from LangGraph, CrewAI, the OpenAI Agents SDK or
pydantic-ai usually do not exist here (`system_prompt`, `Runner`, `result.final_output`,
`@agent.tool`, `max_iterations`). Check every name against the **installed** version before
you use it:

| Need | MCP tool (server `qagents-docs`) | Command line, same answer |
|---|---|---|
| Find how to do something | `search_docs("pause a tool for approval")` | `python -m qagents.docs_mcp search "pause a tool for approval"` |
| Read a guide page or one section | `read_page("concepts/permissions.md", "pause-and-resume")` | `python -m qagents.docs_mcp read concepts/permissions.md pause-and-resume` |
| The table of contents | `list_pages()` | `python -m qagents.docs_mcp pages` |
| A signature and docstring | `get_api("Agent.resume")` | `python -m qagents.docs_mcp api Agent.resume` |
| A runnable example | `list_examples()`, `read_example("supervisor")` | `python -m qagents.docs_mcp examples`, `... example supervisor` |
| Check code you wrote | `check_code(source)` | `python -m qagents.docs_mcp check my_agent.py` |

Rules:

1. Before you call a qagents class, function or method, read it with `get_api`. Pass only the
   parameters it shows.
2. If `get_api` says a name does not exist, it does not exist in this version. Do not write it.
3. Run `check_code` (or `... check file.py`) on every file you write. It reports invented
   imports, attributes and keyword arguments with the closest real names, and coroutines
   that are not awaited.
4. When a guide page and `get_api` disagree, `get_api` wins: it reads the installed code.
5. Test with `ScriptedModel` or `FunctionModel` before you use a real model.

Connect the server: see `docs/coding-agents.md`. In short, `claude mcp add qagents-docs --
python -m qagents.docs_mcp` (Claude Code) or `codex mcp add qagents-docs -- python -m
qagents.docs_mcp` (Codex CLI); it needs `pip install "qagents[mcp]"`.

### Golden rules

- **Tools are typed functions with a docstring.** `@tool` turns the signature into the JSON
  Schema and the docstring into the description. Arguments are validated before your code runs.
- **Mark what a tool does.** `read_only=True` if it changes nothing (such calls may run in
  parallel). `requires_approval=True` if it changes something a person should confirm.
  `idempotent=True` only if running it twice is harmless.
- **Side effects use `ctx.operation_id`** as the idempotency key: it is the same when a call is
  repeated after a crash and a resume. Delivery is at-least-once.
- **Check `result.status` before `result.output`.** It is one of `completed`, `paused`,
  `limit_reached`, `cancelled`, `blocked`.
- **Bound every run** with `Limits(max_turns=..., max_cost_usd=..., timeout_s=...)`.
- **State lives in a `Session`** (`SQLiteStore` to survive restarts) or a `Workflow`, not in
  globals. Close a `SQLiteStore` with `async with` (Windows keeps the file locked otherwise).
- **`run`, `resume` and workflow steps are coroutines:** `await` them. `run_sync` is only for
  code with no running event loop.
- **Nothing hidden.** The model sees your instructions, messages and tool descriptions. The few
  strings qagents adds are in `qagents.prompts`.

### Core patterns

An agent with a tool. The model is `"provider:model-id"` (`qcloud:`, `anthropic:`) or a `Model`.

```python
from qagents import Agent, Limits, tool


@tool(read_only=True)
def get_weather(city: str) -> str:
    """Look up the current weather for a city."""
    return f"Sunny in {city}"


agent = Agent(
    "qcloud:<model-id>",
    instructions="Be brief.",
    tools=[get_weather],
    limits=Limits(max_turns=10, max_cost_usd=0.50),
)
result = await agent.run("What is the weather in Athens?")
if result.status == "completed":
    print(result.output)
```

A tool that needs approval: the run pauses, and `resume` continues it from the journal, in
any process. Calls that already ran are not run again.

```python
from qagents import Agent, RunContext, Session, SQLiteStore, tool


@tool(requires_approval=True)
def refund_order(ctx: RunContext, order_id: str, amount: float) -> str:
    """Refund an order."""
    payments.refund(order_id, amount, idempotency_key=ctx.operation_id)
    return f"Refunded {amount} on {order_id}"


agent = Agent("qcloud:<model-id>", tools=[refund_order])
async with SQLiteStore("agents.db") as store:
    session = Session(store, id="user-42")
    result = await agent.run("Refund order A-1001", session=session)
    if result.status == "paused":
        decisions = {p.tool_call_id: True for p in result.pending_approvals}
        result = await agent.resume(session, approvals=decisions)
```

Rules decide without a person, in a fixed order (deny, `can_use_tool`, ask, allow, mode):
`Agent(..., permissions=Permissions(allow=["read_*"], ask=["send_*"], deny=["delete_*"]))`.

A typed answer: `result.output` is a validated instance.

```python
from pydantic import BaseModel

from qagents import Agent


class Invoice(BaseModel):
    number: str
    total: float


agent = Agent("qcloud:<model-id>", output_type=Invoice)
invoice = (await agent.run("Extract the invoice from: ...")).output
```

A supervisor with sub-agents. An agent in `tools` becomes a tool with one argument, `task`;
it sees only the task and returns only its answer. `handoffs=[...]` instead gives the whole
conversation to another agent.

```python
from qagents import Agent

researcher = Agent("qcloud:<model-id>", name="researcher", description="Finds facts.")
analyst = Agent("qcloud:<model-id>", name="analyst", description="Does the arithmetic.")
supervisor = Agent(
    "qcloud:<model-id>",
    name="supervisor",
    instructions="Delegate, then answer.",
    tools=[researcher, analyst.as_tool(read_only=True)],
)
result = await supervisor.run("How many people do Acme and Borealis employ?")
```

A durable multi-step job. A finished step never runs again for the same workflow id; an
interrupted agent step resumes at the exact tool call.

```python
from qagents import SQLiteStore, Workflow

async with SQLiteStore("jobs.db") as store:
    wf = Workflow(store, id="report:solar")
    notes = await wf.run_agent("research", researcher, "Research solar growth in 2025")
    draft = await wf.run_agent("write", writer, f"Write a report from:\n{notes.text}")
    await wf.run_agent("publish", publisher, draft.text)
```

External changes belong in durable agent tools. Pass `ctx.operation_id` to a service that
enforces an idempotency key; set `idempotent=True` only with that guarantee. Otherwise keep
the tool non-idempotent and reconcile an unknown outcome before trying again. An unfinished
plain `Workflow.step` may execute again after a crash.


Streaming: events are pydantic models with a `type` field.

```python
stream = agent.stream("Tell me about Athens")
async for event in stream:
    if event.type == "text_delta":
        print(event.text, end="")
print(stream.result.usage)
```

### Testing

`ScriptedModel` answers with the replies you give, in order: text is an answer, `call(...)` is
a tool call. `FunctionModel(function)` computes each reply from the request. Every request is
kept in `model.requests`.

```python
from qagents import Agent, Session, tool
from qagents.models.testing import FunctionModel, ScriptedModel, call


@tool(requires_approval=True)
def refund_order(order_id: str) -> str:
    """Refund an order."""
    return f"Refunded {order_id}"


async def test_refund_waits_for_approval():
    model = ScriptedModel([call("refund_order", order_id="A-1"), "Refunded."])
    agent = Agent(model, tools=[refund_order])
    session = Session()
    paused = await agent.run("Refund A-1", session=session)
    assert paused.status == "paused"
    done = await agent.resume(session, approvals={paused.pending_approvals[0].tool_call_id: True})
    assert done.output == "Refunded."
    assert model.requests[1].messages[-1].tool_results[0].content == "Refunded A-1"


def reply(request):
    """FunctionModel calls this for every request: answer once the tool has run."""
    last = request.messages[-1]
    if last.role == "tool":
        return f"Got {last.tool_results[0].content}"
    return call("refund_order", order_id="B-2")


model = FunctionModel(reply)
```

Async tests need `pytest-asyncio` (or `anyio`); this repository sets `asyncio_mode = "auto"`, so
its async test functions need no marker. A sub-agent gets its own scripted model.

## Part 2: working on the qagents repository

### Layout

```text
src/qagents/          the package: agent.py (Agent, the loop, RunStream), tools.py, permissions.py,
                      hooks.py, session.py (journal, stores), workflow.py, multi.py, team.py, events.py,
                      messages.py, results.py, errors.py, context.py, tracing.py, runtime.py,
                      evals.py, cli.py (the qagents command: chat, run, models, crashtest, docs)
src/qagents/external/ ExternalAgent: the base class for agents that run their own loop and
                      are driven over a control protocol; ScriptedExternalAgent for tests.
                      Modules: contract, permissions, turn, agent, tool, changes, scripted
src/qagents/prompts.py the strings qagents itself sends to a model (see "Nothing hidden")
src/qagents/eval_statistics.py repeated-trial reliability estimates and paired case bootstrap
src/qagents/models/   QCloudModel, OpenAIChatModel, AnthropicModel, FallbackModel, testing models
src/qagents/toolkits/ files, shell, web, notes, planning, human, mcp (MCPToolkit)
src/qagents/stores/   PostgresStore (the postgres extra)
src/qagents/effects/  Effect declarations, receipt types, journal folding and runtime contracts
src/qagents/testing/  Effect/store test helpers and the crash-harness interfaces
src/qagents/integrations/  agui, mlflow; names resolved lazily through _LAZY
  acp/                ACPAgent: drives any Agent Client Protocol agent over stdio
  devvi/              DevviAgent: drives Devvi over ACP or the Devvi SDK
  mcp_server.py       serves qagents tools over MCP to an external agent
  _jsonrpc.py, _process.py  the JSON-RPC connection and the child process they share
src/qagents/docs_mcp/ this documentation server and its command line
docs/                 MkDocs pages; mkdocs.yml has the nav
examples/             runnable examples; examples/patterns has one module per pattern
tests/                pytest suite; tests/examples tests every pattern offline
scripts/              build_llms_txt.py writes docs/llms.txt and docs/llms-full.txt
```

### Commands

Use the project's virtual environment (`.venv`). On Windows:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_tools.py -q
.venv\Scripts\python.exe -m ruff check src tests
.venv\Scripts\python.exe -m ruff format src tests
.venv\Scripts\python.exe scripts\build_llms_txt.py
```

On Linux or macOS use `.venv/bin/python`. The suite makes no network calls. `ruff` uses line
length 100 and the rules E, F, I, B, UP and ASYNC; run it on every file you touch.

### Principles

- **Record first, act second.** A model response is journaled before any tool it asked for
  starts; every tool start and result is journaled as it happens.
- **A finished tool call never runs twice.** An interrupted call is repeated on resume only if
  the tool is `read_only` or `idempotent`; otherwise the model is told its outcome is unknown.
- **Ambiguous requests are not replayed.** A model request that may have been billed is not
  sent again unless `Limits(retry_ambiguous=True)`.
- **Nothing hidden.** Every model-facing string lives in `src/qagents/prompts.py` as a plain
  module attribute with a docstring. Do not put prompt text anywhere else. (The docs server's
  strings are in `src/qagents/docs_mcp/prompts.py`.)
- **Fail closed, safe defaults.** A permission check that raises refuses the call; a hook that
  raises stops the run. Tools with side effects ask first; runs have a turn limit.
- **Thin adapters, no new runtime dependencies.** The core needs `pydantic` and `httpx`.
  Optional SDKs (`mcp`, `anthropic`, `mlflow`, `psycopg`) sit behind extras and are imported
  lazily, so `import qagents` never needs them.
- **Plain JSON journals.** No pickle; readers ignore entry kinds they do not know.

### Conventions

- Python 3.11+, typed. Google-style docstrings; plain English, short sentences.
- Public API is exported from `src/qagents/__init__.py`; optional-extra names and the
  external-agent API go in its `_LAZY`.
- Every Python block in `README.md` and `docs/**/*.md` is compiled by `tests/test_examples.py`.
  Keep them valid Python; use `text`, `bash`, `powershell`, `toml` or `json` fences for the rest.
- After changing docs or the public API, run `scripts/build_llms_txt.py` and check the result
  with `python -m qagents.docs_mcp check` on any new snippets.
- Windows is the main development machine: asyncio subprocesses use the Proactor loop, kill
  whole process trees, and expect long paths.
