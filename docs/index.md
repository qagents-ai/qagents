# qagents

A small, explicit agent framework for Python. One loop, typed tools, durable sessions, and QCloud
as the first-class model provider, without being tied to it.

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

!!! warning "Status: 0.2.0, alpha"
    The offline test suite passes, including the QCloud provider driven through the real
    `qcloud_ai` SDK. No request has been made against the live QCloud API yet. See
    [what is verified](design.md#what-is-verified).

## What you get

| You want | You write | Read more |
|---|---|---|
| An answer | `await agent.run("...")` or `agent.run_sync("...")` | [The loop](concepts/loop.md) |
| Tokens and tool activity as they happen | `async for event in agent.stream("...")` | [Streaming](concepts/streaming.md) |
| A tool | a typed Python function with `@tool` | [Tools](concepts/tools.md) |
| A typed answer | `Agent(..., output_type=Invoice)` | [Structured output](concepts/structured-output.md) |
| A conversation that survives restarts | `session=Session(SQLiteStore("agents.db"), id="user-42")` | [Sessions](concepts/sessions.md) |
| A human in the loop | `@tool(requires_approval=True)` and `agent.resume(...)` | [Permissions](concepts/permissions.md) |
| Hard stops | `Limits(max_turns=..., max_cost_usd=..., timeout_s=...)` | [The loop](concepts/loop.md#limits) |
| Your code inside the loop | subclass `Hooks` | [Hooks](concepts/hooks.md) |
| Several agents | `tools=[other_agent]` or `handoffs=[other_agent]` | [Several agents](concepts/multi-agent.md) |
| Another coding agent as a worker | `tools=[DevviAgent(workspace=repo).as_tool(timeout=1800)]` | [External agents](concepts/external-agents.md), [Devvi](integrations/devvi.md) |
| Any ACP agent | `ACPAgent(["node", ".../bin.js", "--acp"], workspace=...)` | [Any ACP agent](integrations/acp.md) |
| Your tools inside a coding agent | `tool_server(tools, ctx=ctx)`, `expose_tools=[...]` | [Serving tools over MCP](integrations/mcp-server.md) |
| A multi-step job that resumes | `Workflow(store, id=...)` | [Workflows](concepts/multi-agent.md#workflows) |
| Traces | `hooks=[OTelHooks()]` | [Tracing](concepts/hooks.md#tracing) |
| Tests without a model | `ScriptedModel([...])` | [Testing](testing.md) |
| Docs your coding agent can query | `python -m qagents.docs_mcp` (MCP) or `qagents docs search "..."` | [Docs for coding agents](coding-agents.md) |

## Why another framework

The 2026 research behind this project ([report, in Greek](research.md)) found that agent
frameworks converged on the same small kernel, and that their documented gaps repeat. qagents is
built around the gaps that matter for a private, OpenAI-compatible model platform:

- **Resume at the tool call, not at the node.** Graph frameworks checkpoint at step boundaries
  and re-run the whole step after a crash or an approval. qagents journals every message, tool
  start and tool result as it happens, so a finished tool call never runs twice.
- **Nothing hidden.** Every string the framework sends to a model is in one file,
  [`qagents.prompts`](reference/prompts.md), and `Hooks.before_model` sees the exact request.
- **Provider rules enforced early.** QCloud rejects unknown request fields; qagents rejects them
  when the agent is built, with the list of accepted fields.
- **No replay of billable requests.** A request the provider may already have processed is never
  sent again unless you opt in.
- **Control by code, outside the model.** Permissions are deterministic rules; a failing check
  refuses the call.
- **Coding agents under your rules.** A coding agent such as Devvi is driven over ACP as an
  external agent: its permission requests go through your `Permissions`, your limits stop it,
  and your journal records the call. MCP is used only to give it tools.
  See [External agents](concepts/external-agents.md).

## Layout

```text
agent.py        the Agent, the loop, RunStream
messages.py     provider-neutral messages
tools.py        @tool, Tool, Toolkit, RunContext
permissions.py  allow / ask / deny rules and approvals
hooks.py        hook points around the run, the model request and each tool call
session.py      the journal: Session, InMemoryStore, SQLiteStore
workflow.py     durable steps for multi-step and multi-agent jobs
multi.py        fan_out and route
runtime.py      AgentRuntime: background runs with leases and recovery
context.py      compaction of long conversations
tracing.py      OpenTelemetry spans
evals.py        datasets, evaluators and reports
prompts.py      every string the framework itself sends to a model
models/         QCloudModel, OpenAIChatModel, AnthropicModel, FallbackModel, test models
stores/         PostgresStore
toolkits/       file, shell, web, notes, clock, planning, ask_user and MCP tools
external/       external agents: other agent programs driven over a control protocol
integrations/   the ACP client, Devvi, the MCP tool server, AG-UI events, MLflow tracing
docs_mcp/       the docs MCP server and command line for coding agents
cli.py          the qagents command: chat, run, models, docs
```
