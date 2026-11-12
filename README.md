<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo/qagents-logo-dark.svg">
    <img alt="qagents" src="assets/logo/qagents-logo.svg" width="360">
  </picture>
</p>

# qagents

A small, explicit agent framework for Python. One loop, typed tools, durable sessions, and
QCloud as the first-class model provider, without being tied to it.

```python
from qagents import Agent, tool


@tool(read_only=True)
def get_weather(city: str) -> str:
    """Look up the current weather for a city."""
    return f"Sunny in {city}"


agent = Agent("qcloud:<model-id>", instructions="Be brief.", tools=[get_weather])
result = agent.run_sync("What is the weather in Athens?")
print(result.output)
```

**Status: 0.2.0, alpha.** The test suite runs without network access, including the QCloud
provider driven through the real `qcloud_ai` SDK against a local transport. No request has been
made against the live QCloud API yet; see [What is verified](#what-is-verified).

## Documentation

The full documentation is in [`docs/`](docs/index.md): concepts, guides, design notes and an API
reference generated from the docstrings. Build and browse it locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

It is built with MkDocs and Material for MkDocs. Material reaches end of life on 5 May 2027; its
successor, Zensical, is made by the same team and reads the same `mkdocs.yml`.

Coding agents can query the docs of the installed version through an MCP server
(`python -m qagents.docs_mcp`, with the `mcp` extra) or its command line, and follow the
rules in [`AGENTS.md`](AGENTS.md). The site also serves `llms.txt` and `llms-full.txt`. See
[Docs for coding agents](docs/coding-agents.md).

## Install

Python 3.11 or newer.

```bash
pip install -e ".[dev]"
```

The core needs only `pydantic` and `httpx`. Optional SDKs sit behind extras: `anthropic`, `mcp`,
`postgres`, `mlflow`, `otel`, `evals`, and `all` for every one of them. The `acp` and `devvi`
extras install nothing: the ACP client and Devvi need no extra dependency.

The QCloud provider uses the `qcloud-ai` SDK, which is not on PyPI yet. Install it from source:

```bash
pip install "git+https://github.com/qcloud-host/qcloud-python.git"
```

Set `QCLOUD_API_KEY`, and pick a model id from `client.models.list()` that lists the `tools`
capability.

## What it gives you

| You want | You write |
|---|---|
| An answer | `await agent.run("...")` or `agent.run_sync("...")` |
| Tokens and tool activity as they happen | `async for event in agent.stream("...")` |
| A tool | a Python function with type hints and a docstring, decorated with `@tool` |
| A typed answer | `Agent(..., output_type=Invoice)` |
| A conversation that survives restarts | `session=Session(SQLiteStore("agents.db"), id="user-42")` |
| A human in the loop | `@tool(requires_approval=True)`, then `agent.resume(session, approvals=...)` |
| Hard stops | `Limits(max_turns=..., max_cost_usd=..., timeout_s=...)` |
| Your code inside the loop | subclass `Hooks` |
| Several agents | `tools=[other_agent]` (sub-agent) or `handoffs=[other_agent]` |
| Another coding agent as a worker | `tools=[DevviAgent(workspace=repo).as_tool(name="coder", timeout=1800)]`: Devvi runs its own loop, driven over ACP; every permission request it sends is decided by your `Permissions` on names like `coder.pwsh` |
| Your tools inside that coding agent | `DevviAgent(..., expose_tools=[...])`, or `tool_server(tools, ctx=ctx)` over MCP |
| A multi-step job that resumes | `Workflow(store, id=...)` with `step` and `run_agent` |
| Traces | `hooks=[OTelHooks()]`, with `from qagents.tracing import OTelHooks` |
| Tests without a model | `ScriptedModel([...])` |
| Docs your coding agent can query | `python -m qagents.docs_mcp` (MCP), or `python -m qagents.docs_mcp search "..."`; see [Docs for coding agents](docs/coding-agents.md) |

## How it works

The loop is a state machine over the conversation. Before each step it looks at the last message:

- an assistant message with tool calls that have no results: run the tools;
- an assistant message without tool calls: that is the answer;
- anything else: ask the model.

Every message, tool start and tool result is appended to the session journal as it happens, as
plain JSON. Because the next step depends only on the journal, a run can stop anywhere (an
approval, a crash, a deploy) and a different process can continue it with `agent.resume(session)`.

```text
agent.py        the Agent, the loop, RunStream
messages.py     provider-neutral messages: text, image, tool call, tool result, reasoning
tools.py        @tool, Tool, Toolkit, RunContext
permissions.py  allow / ask / deny rules and approvals
hooks.py        seven hook points: the run, the model request, each tool call, each event
session.py      the journal: Session, InMemoryStore, SQLiteStore
workflow.py     durable steps for multi-step and multi-agent jobs
multi.py        fan_out (parallel children) and route (pick a specialist)
runtime.py      AgentRuntime: background runs with leases, heartbeats and recovery
context.py      compaction of long conversations
tracing.py      OpenTelemetry spans
evals.py        datasets, evaluators and reports
prompts.py      every string the framework itself sends to a model
cli.py          the qagents command: chat, run, models, docs
models/         QCloudModel, OpenAIChatModel, AnthropicModel, FallbackModel, test models
stores/         PostgresStore
toolkits/       file, shell, web, notes, clock, planning, ask_user and MCP tools
external/       external agents: other agent programs driven over a control protocol
integrations/   the ACP client, Devvi, the MCP tool server, AG-UI events, MLflow tracing
docs_mcp/       the docs MCP server and command line for coding agents
```

### Nothing hidden

The request contains your instructions, your messages and your tool descriptions. The only text
the framework adds (error results, the structured-output instruction, handoff descriptions) lives
in `qagents/prompts.py` as plain module attributes you can read and replace. `Hooks.before_model`
receives the exact request before it is sent.

## Tools

```python
from typing import Literal

from qagents import RunContext, tool


@tool(read_only=True)
async def search_orders(
    ctx: RunContext[Deps], customer: str, status: Literal["open", "closed"] = "open"
) -> list[Order]:
    """Find a customer's orders.

    Args:
        customer: Customer email.
        status: Which orders to return.
    """
    return await ctx.deps.db.orders(customer, status)
```

- The signature becomes the JSON Schema; the docstring becomes the description. Arguments are
  validated before your code runs, and a mistake by the model (unknown tool, invalid JSON, wrong
  types) goes back to it as an error result so it can correct itself.
- A first parameter annotated `RunContext` receives `deps`, ids and usage, and is hidden from the model.
- Raise `ToolError("...")` to tell the model what went wrong. Any other exception is also reported
  to the model and logged. Return a `ToolResultPart` to set `is_error` and `metadata` yourself
  (the loop fills in the call id and name).
- Flags: `read_only` (may run in parallel with other read-only calls; everything else runs one at
  a time, in order), `requires_approval`, `idempotent` (safe to repeat after a crash), `timeout`.
- `ctx.operation_id` is the same when a call is repeated after a resume. Pass it to the system
  your tool calls as the idempotency key.

Ready-made tools, in `qagents.toolkits`: `file_tools(root, writable=False)`, `shell_tool(cwd)`,
`web_tools()`, `notes_tools(directory)`, `current_time`, `todo_tools()` (a plan kept in the
session state) and `ask_user` (a question that pauses the run). The tools of an MCP server come
from `qagents.toolkits.mcp.MCPToolkit` (extra `mcp`).

## Permissions and approvals

Decisions are made by code, outside the model, in a fixed order: `deny` rules, your
`can_use_tool` function, `ask` rules, `allow` rules, then the mode default. The first rule that
answers decides (a deny always wins), and an exception while deciding refuses the call.

```python
from qagents import Permissions

Permissions(mode="default", allow=["read_*"], ask=["send_*"], deny=["delete_*"])
```

"Ask" is answered by `on_ask` when you provide one. Without it the run pauses:

```python
result = await agent.run("Refund order A-1001", session=session)
if result.status == "paused":
    call = result.pending_approvals[0]
    result = await agent.resume(session, approvals={call.tool_call_id: True})
```

Tool calls of the same turn that needed no approval have already run, and are not run again.

## Sessions and recovery

```python
from qagents import Session, SQLiteStore

store = SQLiteStore("agents.db")
await agent.run("My name is Maria.", session=Session(store, id="user-42"))
await agent.run("What is my name?", session=Session(store, id="user-42"))
```

What the journal guarantees, stated plainly:

- A tool call with a recorded result never runs again.
- A tool call that had started when the process died is re-run only if the tool is `read_only`
  or `idempotent`. Otherwise the model is told its outcome is unknown.
- A model request that was in flight is sent again on resume. Its first response was lost, so
  this costs tokens; no tool had acted on it.
- If a new message arrives while tool calls are still open, they are closed with an explanatory
  result, so the conversation stays valid for the provider.
- Delivery is at-least-once, like every agent framework. Use `ctx.operation_id` for side effects.

`agent.resume` continues a run; nothing calls it for you, unless you run `AgentRuntime`: it
holds a lease on each session it drives (renewed by a heartbeat, on `SQLiteStore` or
`PostgresStore`), and its `recover()` resumes runs whose worker died.

## Streaming, cancel and voice

`agent.stream()` returns a `RunStream`. Events are pydantic models with a `type` field
(`text_delta`, `tool_call_started`, `tool_call_finished`, `approval_required`, `run_finished`, ...),
so they serialise straight to a websocket or an SSE response.

`stream.cancel()` stops the run at the next chunk. The text produced so far stays in the history,
marked `interrupted`, and the next `agent.run` on that session continues from it. That is the
barge-in path for a voice agent. Speech-to-text, text-to-speech and transports are outside this
package.

## Structured output

```python
agent = Agent("qcloud:<model-id>", output_type=Invoice)
invoice = (await agent.run("Extract the invoice from ...")).output
```

`output_mode="native"` sends the schema as `response_format`; `"prompted"` writes it into the
instructions for models without schema support. Either way the answer is validated, and the model
gets the validation error and `Limits.output_retries` more attempts.

## Several agents

```python
from qagents import Agent

researcher = Agent(model, name="researcher", description="Finds facts.", tools=[search])
lead = Agent(model, tools=[researcher])  # sub-agent: own context, returns its answer
triage = Agent(model, name="triage", handoffs=[billing])  # billing takes over the conversation
```

A sub-agent sees only the task text and the caller sees only its answer; its usage is counted in
the caller's run. There are no role-play crews or group chats: the evidence favours one
orchestrator with isolated workers and a single writer.

For jobs with several steps, write a normal async function and wrap the units in a `Workflow`:

```python
from qagents import Workflow

wf = Workflow(store, id=f"report:{topic}")
notes = await wf.run_agent("research", researcher, f"Research {topic}")
draft = await wf.run_agent("write", writer, f"Write a report from:\n{notes.text}")
await wf.step("publish", lambda: publish(draft.text))
```

A finished step never runs again for the same workflow id. An agent step that was interrupted or
paused is resumed at the exact tool call. Use `asyncio.gather` for parallel branches.

## Coding agents as workers

A coding agent such as Devvi runs its own loop, model, tools and sandbox. qagents drives it as an
external agent over ACP (or Devvi's SDK protocol): it sends the task, sees each step as an event,
decides every permission request with your `Permissions`, and cancels it with your limits. MCP
is not how you drive an agent: MCP gives an agent tools, and qagents uses it only to hand its
own tools to the coding agent.

```python
from qagents import Agent, Permissions
from qagents.integrations.devvi import DevviAgent

coder = DevviAgent(
    workspace="C:/work/repo",
    patches=[{"id": "tool-web", "disabled": True}],  # Devvi never asks before a web fetch
    allow_unmetered=True,  # ACP reports no token counts
)
lead = Agent(
    "qcloud:<model-id>",
    tools=[coder.as_tool(name="coder", timeout=1800)],
    permissions=Permissions(ask=["coder.bash", "coder.pwsh"]),  # Devvi asks only to leave its sandbox
)
```

`pip install "qagents[devvi]"` needs no extra dependency; giving Devvi qagents tools
(`expose_tools`) needs `qagents[mcp]`. See [External agents](docs/concepts/external-agents.md)
and [Devvi as a coding worker](docs/integrations/devvi.md).

## Providers

`QCloudModel` uses the `qcloud_ai.AsyncQCloud` client, so authentication, pooling, the SDK's retry
policy and stream validation stay with the SDK. On top of it the adapter:

- checks request fields against QCloud's published allow-list when the agent is built, so a
  setting the API would reject fails immediately with the list of accepted fields;
- reads the model's catalog entry once, fails fast when a model lacks `tools` or `vision`, and
  prices every response from the catalog, which makes `Limits(max_cost_usd=...)` work;
- maps QCloud errors to typed errors with the request id, and marks the ones that may have been
  billed as `ambiguous`. Ambiguous requests are never repeated unless you set
  `Limits(retry_ambiguous=True)`.

Any other server that speaks OpenAI Chat Completions (vLLM, SGLang, llama.cpp, Ollama, OpenAI):

```python
from qagents import OpenAIChatModel

model = OpenAIChatModel("my-model", base_url="http://localhost:8000/v1", api_key="...")
```

Claude through the Messages API: `Agent("anthropic:<model-id>")` or `AnthropicModel(...)`, with
the `anthropic` extra. `FallbackModel([...])` tries the next model when one refuses at admission,
and `RateLimitedModel(model, ...)` stays under a provider's limits.

To add a provider, subclass `Model` and implement `stream`. The message model keeps reasoning
parts with opaque provider data, so adapters for other APIs (the Responses API, for example) can
be added without changing stored conversations.

## Testing your agents

```python
from qagents import Agent
from qagents.models.testing import ScriptedModel, call

model = ScriptedModel([call("get_weather", city="Athens"), "It is sunny."])
result = await Agent(model, tools=[get_weather]).run("Weather in Athens?")
assert model.requests[1].messages[-1].tool_results[0].content == "Sunny in Athens"
```

## What is verified

| Area | How it was checked |
|---|---|
| Loop, tools, permissions, hooks, limits, cancel, structured output | Unit tests with scripted models |
| Sessions, crash recovery, approvals, compaction, workflows | Unit tests, including SQLite across store instances and tasks killed mid-tool |
| OpenAI-compatible provider | Tests against a local HTTP transport: streaming, tool-call deltas, retries, error mapping |
| QCloud provider | Tests through the real `qcloud_ai` 0.4.0 SDK with its HTTP transport replaced locally |
| Toolkits | Unit tests, including path-escape and non-public-address refusals |
| Anthropic provider | Tests through the real `anthropic` SDK with its HTTP transport replaced locally |
| `PostgresStore`, `AgentRuntime` | The store and recovery tests run on `InMemoryStore`, `SQLiteStore` and, with `QAGENTS_TEST_POSTGRES_DSN` set, a real PostgreSQL server |
| MCP client toolkit | Tests against a real MCP server, as a subprocess over stdio and over streamable HTTP on 127.0.0.1 |
| Tracing | Tests with the OpenTelemetry SDK's in-memory exporter, and MLflow against a real tracking store (SQLite) |
| AG-UI events, command line | Tests with scripted models and no network |
| External agents (ACP client, Devvi, MCP tool server) | Offline tests against Devvi's recorded ACP and SDK transcripts and fake agent processes, process-tree kill on Windows, the official MCP client; opt-in tests against the real Devvi CLI with a local mock model |
| Docs server for coding agents | Tests of search, pages, API introspection, the code checker, the command line, and the MCP server over stdio with the official client |

Examples are executed and documentation code blocks are compiled by `tests/test_examples.py`.

Not verified yet: a request against the live QCloud API (`examples/live_acceptance.py` exists for
this and has only been run with `--dry-run`; `examples/quickstart.py` has not been run),
tool-calling reliability of specific catalog models, Python 3.11 and 3.12, Linux and macOS (the
suite ran on Python 3.13 on Windows, where the POSIX file-symlink test of `file_tools` is
skipped, so that test has never run; the junction tests do run), behaviour under load, and
external agents against a paid model route or another ACP agent than Devvi.

## Roadmap

Shipped in 0.2 (see the [changelog](CHANGELOG.md)): `PostgresStore` and `AgentRuntime` with
leases and recovery, the MCP client toolkit, signed approval tokens, the final-answer tool as a
third structured-output mode, budgets and depth inherited by sub-agents, and the Anthropic and
AG-UI adapters. Since 0.2 (unreleased): external agents (the ACP client, Devvi, the MCP tool
server) and the docs server for coding agents.

Next, in this order:

1. Live acceptance on QCloud: one paid tool loop with usage and cost reconciled, and a small
   tool-calling eval per model you plan to use.
2. Deferred tool loading for large tool sets.
3. External agents: live Devvi acceptance (one paid task, usage reconciled), recorded fixtures
   for more ACP agents, the devvi.sh platform as a `Model`, and qagents as an ACP agent server.

Deliberately not planned: a graph DSL, role-play crews, built-in RAG or vector stores, a universal
provider gateway, in-process code sandboxes, pickled state, and driving agents over MCP (MCP
gives an agent tools; agents are driven over ACP).
