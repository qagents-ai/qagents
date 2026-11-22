# Serving tools over MCP

A coding agent runs its own loop with its own tools. To let it use qagents tools as well, a
customer lookup or a ticket system, serve them to it over MCP.
`qagents.integrations.mcp_server` does that.

MCP gives an agent tools; it does not drive the agent. qagents drives agents over ACP or the
Devvi SDK ([External agents](../concepts/external-agents.md)), and serves them tools over MCP.
In this direction the coding agent is the MCP client and qagents is the server.

It needs the MCP Python SDK: `pip install "qagents[mcp]"`.

| Mode | Where the tools run | Who decides each call | Journal | Use it for |
|---|---|---|---|---|
| [Bound to a run](#bound-to-a-run) (HTTP) | in your process | the run's `Permissions`, with its `on_ask` | the run's session | tools of a qagents run that drives the agent; the default |
| [On its own](#on-its-own) (HTTP) | in your process | the server's `permissions=` | none | a server outside any run |
| [stdio](#over-stdio-in-a-separate-process) | in a process the agent starts | the spec's rules; "ask" is refused | none | importable tools, any MCP client |

With Devvi you rarely call this module yourself: `DevviAgent(expose_tools=[...])` starts the
right server for every turn ([qagents tools inside Devvi](devvi.md#qagents-tools-inside-devvi)).

## Bound to a run

`tool_server(tools, ctx=ctx)` inside a running tool call binds the server to that call's run.
Every call that arrives over MCP is then an ordinary tool call of that run, made with
`RunContext.call_tool`:

- the run's `Permissions` decide it, with the run's `on_ask`. "Ask" without an `on_ask` is
  refused, because a call over MCP cannot pause the run. An approval with edited arguments is
  checked again and runs with the new arguments;
- it is journaled in the run's session before and after it runs (`approval_decided`,
  `tool_started`, `tool_result`), and counts toward `Limits.max_tool_calls` from before it is
  decided: calls that arrive together never pass the limit, and a refused call does not count;
- `Limits.parallel_tools` applies among the calls served through the same calling tool call: one
  that must run alone waits for the others from that caller. Calls served through different
  calling calls, and the model's own calls, are not kept apart from it. A call that runs alone
  also waits, within its own timeout, for the thread of an earlier sync call (served or the
  run's own, from anywhere in the run) that timed out and still runs. If that thread is still
  busy then, the call is not run (`TOOL_WAITED_FOR_EARLIER`);
- the run's `before_tool` and `after_tool` hooks see it (a hook that raises fails the run, as
  for the model's calls), and its `tool_call_started` and `tool_call_finished` events reach
  the run's stream;
- the tool gets the run's `deps` and `ctx.state`, and its usage is added to the run;
- its `ctx.operation_id` is `<calling call's operation id>:<call id>`, stable for the call;
- it runs one level below the calling tool call (`ctx.depth + 1`), because it comes from the
  agent that the tool call runs. An agent or external agent among the served tools therefore
  starts at `ctx.depth + 2`, and `Limits.max_depth` bounds chains through served tools as it
  bounds sub-agents.

So a qagents tool never runs without a qagents decision.

```python
from qagents import RunContext, tool
from qagents.integrations.acp import ACPAgent
from qagents.integrations.mcp_server import tool_server


@tool
async def coder(ctx: RunContext, task: str) -> str:
    """Delegate a coding task. The coding agent can look up customers and create tickets."""
    async with tool_server([lookup_customer, create_ticket], ctx=ctx) as server:
        agent = ACPAgent(
            ["node", "C:/tools/some-agent/bin.js", "--acp"],
            name="coder",
            workspace="C:/work/repo",
            mcp_servers=[server.acp_config()],  # an ACP session/new mcpServers entry
            allow_unmetered=True,
        )
        async with agent:
            result = await agent.run(task, parent=ctx)
    return result.for_model("coder")
```

This starts the agent for each call, with the server of that call. For Devvi,
`expose_tools` does the same with one long-lived process per workspace.

The journal entries of a served call carry `source: "mcp"`, `server` (the server name) and
`caller_tool_call_id` (the id of the tool call the server is bound to). Served call ids start
with `mcp-`.

| Entry | Written |
|---|---|
| `approval_decided` | when `on_ask` answered, before the call runs |
| `tool_started` | before the tool runs |
| `tool_result` | after it, with the result text and its usage |

The server refuses calls once the tool call it is bound to has ended, and journals nothing for
them. A call that was already journaled as started (it waited in `before_tool`, for example) is
checked again right before the tool runs: it then records the same refusal and does not run.
Start the server inside the tool, and close it before the tool returns.

## On its own

Without `ctx` the server decides calls with its own rules and journals nothing:

```python
from qagents import Limits, Permissions
from qagents.integrations.mcp_server import tool_server

rules = Permissions(allow=["lookup_customer"], deny=["delete_*"])
limits = Limits(tool_timeout_s=30)
async with tool_server([lookup_customer], permissions=rules, limits=limits) as server:
    url, headers = server.url, server.headers  # give both to the agent; never log the headers
    await run_agent_that_speaks_mcp(url, headers)
```

`permissions=` defaults to `Permissions()`: tools declared `requires_approval` are refused unless
an allow rule or an `on_ask` lets them run. `deps=` sets `ctx.deps` for the tools. Of `limits=`,
`tool_timeout_s`, `max_tool_result_chars`, `max_tool_calls`, `parallel_tools` and
`max_parallel_tools` apply.

`tool_server(tools, *, ctx=None, permissions=None, deps=None, limits=None, name="qagents",
approval_timeout_s=60.0, shutdown_timeout_s=10.0)` returns a `ToolServer`, not started yet; use it
with `async with`, or call `start()` and `aclose()`. `tools` takes what `Agent(tools=...)` takes:
tools, functions, agents, toolkits and lists of them. Two tools with the same name, and deferred
tools (their result would come from `agent.resume`, which an MCP caller cannot wait for), are
refused when the server starts. `ctx` together with `permissions`, `deps` or `limits` is refused:
a bound server uses the run's.

While it runs:

| Attribute | Value |
|---|---|
| `server.url` | `http://127.0.0.1:<port>/mcp` |
| `server.headers` | `{"Authorization": "Bearer <token>"}`: pass it only to the agent that should call the tools; never log it |
| `server.acp_config(name=None)` | `{"type": "http", "name", "url", "headers": [{"name", "value"}]}`, an ACP `session/new` `mcpServers` entry |
| `server.port`, `server.tools`, `server.bound`, `server.running` | the port, the served tools, whether it is bound, whether it serves |

## Over stdio, in a separate process

`python -I -m qagents.integrations.mcp_server --spec FILE` serves the tools a JSON spec lists, on
its standard input and output, until its input closes. The agent starts this process itself, so
the tools must be importable: there is no parent run here, the spec's rules decide, "ask" is
refused, and nothing is journaled.

```json
{
  "name": "qagents",
  "tools": ["my_app.tools:lookup_customer", "my_app.tools:TICKET_TOOLS"],
  "permissions": {"allow": ["create_ticket"], "deny": ["delete_*"]},
  "deps": "my_app.tools:make_deps"
}
```

| Field | Default | Meaning |
|---|---|---|
| `name` | `"qagents"` | the MCP server name the agent sees |
| `tools` | required | `"package.module:attr"` references: a tool, a function, an agent, a toolkit, or a list of them |
| `permissions` | `{}` | `mode`, `allow`, `ask`, `deny`, as in `Permissions`, without callbacks |
| `paths` | `[]` | absolute directories put first on `sys.path` before the tools are imported |
| `deps` | `null` | `"package.module:attr"` of `ctx.deps`: a value, or a function (sync or async) called once, with no arguments, when the server starts |
| `tool_timeout_s` | 120 | seconds per call |
| `max_tool_result_chars` | 60000 | longer results are cut |
| `max_tool_calls` | `null` | calls the server runs before it refuses more, for the process's lifetime |

The process runs with `-I`: no current directory, no `PYTHONPATH` and no user site-packages on
`sys.path`. Install the tools' package in the environment, or list its directory in `paths`.

Build the spec and the command in code:

```python
from qagents.integrations.mcp_server import (
    ToolRules,
    ToolServerSpec,
    acp_stdio_config,
    stdio_command,
    tool_reference,
)

spec = ToolServerSpec(
    name="qagents",
    tools=[tool_reference(lookup_customer), "my_app.tools:TICKET_TOOLS"],
    permissions=ToolRules(allow=["create_ticket"], deny=["delete_*"]),
)
path = spec.write("C:/work/qagents-tools.json")  # the absolute path
command = stdio_command(path)  # [python, "-I", "-m", "qagents.integrations.mcp_server", ...]
entry = acp_stdio_config(path, name="qagents")  # an ACP session/new mcpServers entry
```

`tool_reference(tool)` gives `"module:attr"` for a tool defined at module level, and refuses one
defined in a function or in `__main__`, which another process cannot import. `stdio_command`
uses this interpreter's path made absolute but not resolved, so a virtual environment keeps its
packages. Put no secret in `acp_stdio_config(env=...)`: the agent may log its configuration.
Do not count on the server's environment either. The agent starts the server, and an agent may
drop every variable whose name looks like a credential before it does: Devvi removes names
matching `KEY|PASSWORD|SECRET|TOKEN` and names starting with `DSH_`
(`packages/subprocess/subprocess/src/index.ts` in the Devvi repository). Tools that need a credential
load it in a `deps` factory (the spec's `deps`) that reads a file or a credential store, for
example:

```python
import json
from pathlib import Path


def make_deps() -> dict:
    """The spec's "deps": "my_app.tools:make_deps". Called once, when the server starts."""
    path = Path.home() / ".my_app" / "tickets.json"  # not in the agent's workspace
    return {"tickets_token": json.loads(path.read_text(encoding="utf-8"))["token"]}
```

A tool then reads `ctx.deps["tickets_token"]`. `Path.home()` works in the server's scrubbed
environment, since `USERPROFILE` and `HOME` are kept. A file is not hidden from the agent:
Devvi's sandbox does not confine reads on Windows, so the agent's own tools could read it too.
Where that matters, read the credential from the operating system's credential store instead.

Any MCP client can start it. For example, in a client configuration file:

```json
{
  "mcpServers": {
    "qagents-tools": {
      "command": "C:\\work\\.venv\\Scripts\\python.exe",
      "args": ["-I", "-m", "qagents.integrations.mcp_server", "--spec", "C:\\work\\qagents-tools.json"]
    }
  }
}
```

To check a spec by hand:

```powershell
python -I -m qagents.integrations.mcp_server --spec C:\work\qagents-tools.json --log-level INFO
```

A bad spec, or a tool that cannot be served, exits at once with code 2 and the reason on
stderr. A good one waits for a client on its standard input (on Windows, `Ctrl+Z` then Enter
ends it). Exit codes: 0 when its input closed, 2 for a bad spec, 1 for any other failure. While it serves, stray output of the tools
(`print`) goes to stderr, not into the protocol stream.

## What the calling agent sees

Each tool's name, its description and the JSON Schema of its arguments, exactly as a qagents
model would see them, plus MCP annotations from the tool's flags: `readOnlyHint` for
`read_only`, `idempotentHint` for `idempotent`. A result is one text block, the text a qagents
model would read. Errors (invalid arguments, a `ToolError`, a timeout, a refusal) are MCP results
with `isError` set, so the calling agent's model can read them and adapt.

Refusals read `'create_ticket' was not run: <reason>`. The reasons are `TOOL_SERVER_*` strings,
in `qagents.prompts`:

| Reason | When |
|---|---|
| the rule's own reason | a deny rule, `can_use_tool` (also when it raises), a declined approval |
| `it needs approval, and nobody can approve it while the calling agent waits.` | "ask" with no `on_ask` |
| `the approval did not arrive within 60 seconds.` | `on_ask` took longer than `approval_timeout_s` |
| `the qagents run that serves this tool is no longer running.` | a bound server whose tool call ended |
| `the qagents tool server is shutting down.` | a call during shutdown |

Every other text is the agent loop's own wording (`qagents.prompts`).

## Security

- The server listens on `127.0.0.1` only, on a random port. On Windows the port is bound with
  `SO_EXCLUSIVEADDRUSE`, so no other process can share it.
- Every request must carry the server's bearer token, a new random value per server, compared
  in constant time; anything else gets `401`. The token keeps out other local users and
  browser pages, not code that runs as your own user.
- DNS rebinding protection: the `Host` must be `127.0.0.1:<port>`, and a browser `Origin` must
  be `http://127.0.0.1:<port>`.
- At most 16 MCP sessions at a time; no websockets.
- Only `127.0.0.1` (IPv4). A client that resolves `localhost` to `::1` must use `server.url` as
  given.

## With Devvi

`DevviAgent(expose_tools=[...])` starts a bound server for every turn and passes it in the
session's `mcpServers`; Devvi shows the tools to its model as `mcp__qagents__<tool>`. Two gates
apply:

1. Devvi's own pipeline could ask permission for `mcp__qagents__<tool>`; with its default
   profile it asked nothing in tier 1. The bridge would decide such a request as
   `coder.mcp__qagents__<tool>`; with no rule it is refused.
2. The server decides the call itself with the lead's rules, on the tool's own name. This is
   the gate that always applies.

So `allow=["coder.mcp__qagents__*"]` at the bridge is safe; put the rules for each tool on its
qagents name. Devvi gives an MCP tool call 60 seconds
(`packages/mcp/mcp-client/src/index.ts` in the Devvi repository): keep the lead's
`Limits.tool_timeout_s` under a minute (an `on_ask` for an exposed tool is bounded by 50
seconds). If Devvi gives up first, it cancels the request, and the server records the call as
interrupted.

## With any MCP-capable agent

Give the agent `server.url` and `server.headers` (HTTP), or the stdio command. For an ACP agent,
pass `server.acp_config()` or `acp_stdio_config(path)` in `ACPAgent(mcp_servers=[...])`; MCP
servers are sent again on every session resume, because ACP agents do not keep them.

## Shutting down

Leaving `async with` (or `await server.aclose()`) refuses new calls, cancels running ones, ends
the MCP sessions and closes the port, within `shutdown_timeout_s`. `aclose()` is safe to cancel,
for example by a deadline around `async with`: the shutdown goes on in a task of its own and
still closes the port, and a second `aclose()` waits for it. A cancelled bound call records
`TOOL_INTERRUPTED` ("outcome unknown"), or `TOOL_CANCELLED` for a read-only tool: only a
read-only call is harmless when it is cut short.

!!! note "Not in this version"
    - Results are text only: structured content and images are not sent as MCP
      `structuredContent` or image blocks.
    - "Ask" cannot pause the parent run over MCP; it is refused unless an `on_ask` answers.

