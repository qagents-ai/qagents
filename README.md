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

**Status: 0.1.0, alpha.** The test suite runs without network access, including the QCloud
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

## Install

Python 3.11 or newer.

```bash
pip install -e ".[dev]"
```

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
| A multi-step job that resumes | `Workflow(store, id=...)` with `step` and `run_agent` |
| Traces | `hooks=[OTelHooks()]` |
| Tests without a model | `ScriptedModel([...])` |

## How it works

The loop is a state machine over the conversation. Before each step it looks at the last message:

- an assistant message with tool calls that have no results: run the tools;
- an assistant message without tool calls: that is the answer;
- anything else: ask the model.

Every message, tool start and tool result is appended to the session journal as it happens, as
plain JSON. Because the next step depends only on the journal, a run can stop anywhere (an
approval, a crash, a deploy) and a different process can continue it with `agent.resume(session)`.

```
agent.py        the Agent, the loop, RunStream
messages.py     provider-neutral messages: text, image, tool call, tool result, reasoning
tools.py        @tool, Tool, Toolkit, RunContext
permissions.py  allow / ask / deny rules and approvals
hooks.py        six hook points around the run, the model request and each tool call
session.py      the journal: Session, InMemoryStore, SQLiteStore
workflow.py     durable steps for multi-step and multi-agent jobs
context.py      compaction of long conversations
tracing.py      OpenTelemetry spans
prompts.py      every string the framework itself sends to a model
models/         QCloudModel, OpenAIChatModel, ScriptedModel
toolkits/       file, shell, web, notes and clock tools
```

### Nothing hidden

The request contains your instructions, your messages and your tool descriptions. The only text
the framework adds (error results, the structured-output instruction, handoff descriptions) lives
in `qagents/prompts.py` as plain module attributes you can read and replace. `Hooks.before_model`
receives the exact request before it is sent.

## Tools

```python
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
  to the model and logged.
- Flags: `read_only` (may run in parallel with other read-only calls; everything else runs one at
  a time, in order), `requires_approval`, `idempotent` (safe to repeat after a crash), `timeout`.
- `ctx.operation_id` is the same when a call is repeated after a resume. Pass it to the system
  your tool calls as the idempotency key.

Ready-made tools: `file_tools(root, writable=False)`, `shell_tool(cwd)`, `web_tools()`,
`notes_tools(directory)`, `current_time`.

## Permissions and approvals

Decisions are made by code, outside the model, in a fixed order: `deny` rules, your
`can_use_tool` function, `ask` rules, `allow` rules, then the mode default. The more restrictive
rule wins, and an exception while deciding refuses the call.

```python
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

`agent.resume` continues a run; nothing calls it for you. Detecting dead runs across replicas
(leases, heartbeats) is on the roadmap, not in 0.1.

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
researcher = Agent(model, name="researcher", description="Finds facts.", tools=[search])
lead = Agent(model, tools=[researcher])  # sub-agent: own context, returns its answer
triage = Agent(
    model, name="triage", handoffs=[billing]
)  # handoff: billing takes over the conversation
```

A sub-agent sees only the task text and the caller sees only its answer; its usage is counted in
the caller's run. There are no role-play crews or group chats: the evidence favours one
orchestrator with isolated workers and a single writer.

For jobs with several steps, write a normal async function and wrap the units in a `Workflow`:

```python
wf = Workflow(store, id=f"report:{topic}")
notes = await wf.run_agent("research", researcher, f"Research {topic}")
draft = await wf.run_agent("write", writer, f"Write a report from:\n{notes.text}")
await wf.step("publish", lambda: publish(draft.text))
```

A finished step never runs again for the same workflow id. An agent step that was interrupted or
paused is resumed at the exact tool call. Use `asyncio.gather` for parallel branches.

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

To add a provider, subclass `Model` and implement `stream`. The message model keeps reasoning
parts with opaque provider data, so adapters for Anthropic Messages or the Responses API can be
added without changing stored conversations.

## Testing your agents

```python
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
| Tracing | Tests with the OpenTelemetry SDK's in-memory exporter |

Examples are executed and documentation code blocks are compiled by `tests/test_examples.py`.

Not verified yet: a request against the live QCloud API (`examples/live_acceptance.py` exists for
this and has only been run with `--dry-run`; `examples/quickstart.py` has not been run),
tool-calling reliability of specific catalog models, Python 3.11 and 3.12, Linux and macOS (the
suite ran on Python 3.13 on Windows, where the `file_tools` symlink test is skipped, so that test
has never run), and behaviour under load.

## Roadmap

Next, in this order:

1. Live acceptance on QCloud: one paid tool loop with usage and cost reconciled, and a small
   tool-calling eval per model you plan to use.
2. `PostgresStore` with leases and a recovery worker that finds and resumes dead runs.
3. MCP client toolkit (spec revision 2026-07-28, with fallback to 2025-11-25) and deferred tool
   loading for large tool sets.
4. Signed approval tokens, so a stored pause cannot be forged.
5. A final-answer tool as a third structured-output mode, and budgets inherited by sub-agents.
6. Adapters: Anthropic Messages, AG-UI event projection.

Deliberately not planned: a graph DSL, role-play crews, built-in RAG or vector stores, a universal
provider gateway, in-process code sandboxes, and pickled state.
