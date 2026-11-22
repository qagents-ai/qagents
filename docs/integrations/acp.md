# Any ACP agent

The [Agent Client Protocol](https://agentclientprotocol.com) (ACP) is how a program drives a
coding agent: open a session in a workspace, send a prompt, follow what the agent does, answer
its permission requests, cancel it. It is JSON-RPC 2.0 over the agent's standard input and
output, one message per line. `qagents.integrations.acp.ACPAgent` is a generic ACP v1 client:
any agent that speaks ACP v1 becomes a qagents [external agent](../concepts/external-agents.md).

ACP drives an agent; MCP gives an agent tools. `ACPAgent` does the first. Its `mcp_servers=`
option does the second: it hands the agent MCP servers to mount in each session, for example
[qagents tools served over MCP](mcp-server.md).

`ACPAgent` needs no extra dependency: `pip install "qagents[acp]"` is an empty extra. Serving
qagents tools to the agent with `tool_server` needs `pip install "qagents[mcp]"`.

!!! note "Verified with Devvi only"
    `ACPAgent` is tested against Devvi's recorded ACP transcripts (`devvi --profile acp`).
    Other ACP agents are expected to work, but none has a recorded fixture yet. Record one
    before you rely on an agent ([below](#adding-a-recorded-fixture-for-a-new-agent)). For
    Devvi itself, use [`DevviAgent`](devvi.md), which adds Devvi's launcher, home, patch and
    environment rules on top of this client.

## Quick start

```python
from qagents import Agent, Permissions
from qagents.integrations.acp import ACPAgent

coder = ACPAgent(
    ["node", "C:/tools/some-agent/bin.js", "--acp"],  # a list, never a shell string
    name="coder",
    workspace="C:/work/repo",
    env={"SOME_SETTING": "1"},  # set for the agent process
    passthrough=["SOME_API_KEY"],  # copied from this process's environment, by name
    allow_unmetered=True,  # ACP reports no token counts
)
lead = Agent(
    "qcloud:<model-id>",
    tools=[coder.as_tool(timeout=1800)],
    permissions=Permissions(
        allow=["coder.read"],
        ask=["coder.edit", "coder.execute"],  # without an on_ask, "ask" is refused
    ),
)

async with coder:
    result = await lead.run("Make the failing test pass.")
```

Rules are checked in a fixed order: deny, `can_use_tool`, ask, allow, then the mode. An ask rule
therefore wins over an allow rule: `allow=["coder.read"], ask=["coder.*"]` asks about
`coder.read` too, and with no `on_ask` refuses it. Keep allow and ask patterns apart, as above,
or put the default in the mode. The lead's rules also decide the lead's own tool calls, the
call to `coder` included, so `mode="ask_all"` needs that tool in the allow list:
`Permissions(allow=["coder", "coder.read"], mode="ask_all")` runs `coder`, allows
`coder.read`, and asks about every other request of the agent. The names come from the agent's
tool calls ([Tool names for permission rules](#tool-names-for-permission-rules)), and the agent
sends a request only for the calls it chooses to ask about.

On Windows a `.js` agent runs as `["node", ".../bin.js"]`: Windows cannot execute a `.js` file,
and qagents never starts a `.cmd` or `.bat` shim (it would run through `cmd.exe`). A command
that names a `.js` file is refused with `... Windows starts only .exe and .com programs. Run it
with its interpreter, for example ['node', ...]`, and a path that does not exist with
`Cannot start '<path>': no such file.` A bare program name is looked up only in the absolute
`PATH` entries, never in the current directory.

## What the client does

1. **Starts the process** with `command`, in `cwd` (default: the workspace), with an explicit
   environment: the operating-system basics (`PATH`, `SYSTEMROOT`, `TEMP`, `TMP`,
   `USERPROFILE`, `HOME`, `LANG`, and on Windows `PATHEXT`, `COMSPEC`, `WINDIR`,
   `SYSTEMDRIVE`), plus `env=`, plus the `passthrough=` names. Nothing else of qagents'
   environment reaches the agent unless you pass `inherit_env=True`.
2. **`initialize`** with `protocolVersion: 1`, `clientInfo: {name: "qagents", version}` and
   client capabilities that offer nothing:

    ```json
    {"fs": {"readTextFile": false, "writeTextFile": false}, "terminal": false}
    ```

    The agent reads files and runs commands with its own tools, under its own sandbox, and asks
    qagents for permission. An agent that answers another protocol version is refused. Requests
    the agent sends that qagents does not implement (`fs/*`, `terminal/*`) are answered
    JSON-RPC -32601, so richer agents work too.
3. **`authenticate`**, only when the agent lists `authMethods` and you chose one with
   `auth_method=`. Without a choice qagents never starts a login. A method of type `terminal`
   (interactive) is refused: log in to the agent yourself first. If the agent then refuses
   `session/new` with "authentication required", the error lists the methods it offers.
4. **`session/new {cwd, mcpServers}`** for a new conversation. `cwd` is the absolute workspace;
   a turn without one is refused. A session this connection opened is continued directly while
   it is still open; an older or closed one with `session/resume`, or `session/load` (its
   replayed history is ignored) when the agent offers only that. MCP servers are sent again on
   every resume, because agents do not keep them.
5. **`session/set_config_option`** for `model=` and `config_options=`, when the session offers
   that option and its current value differs.
6. **`session/prompt`**. `session/update` notifications become qagents events;
   `session/request_permission` is answered by the permission bridge; the `stopReason` becomes
   the result's status.
7. **Cancel**: `session/cancel`, then `cancel_grace_s` seconds (default 5) for the agent to answer
   `stopReason: "cancelled"`; after that the process tree is killed and the next turn starts a
   new process. A cancel that comes before `session/prompt` is sent ends the turn at once,
   without sending it and without `session/cancel`: a setup request still waiting for its
   answer (`session/new`, `session/resume`, `session/load`, `session/set_config_option`) is
   given up, which the agent is told with `$/cancel_request`, and the process is kept.
8. **`session/close`** after a turn, for the idle sessions beyond `keep_sessions` (default 1),
   least recently used first, when the agent advertises `sessionCapabilities.close` and can
   resume or load sessions. The agent then releases what it holds for each one (Devvi: an agent
   in memory and a process per stdio MCP server). A later turn in such a session resumes it
   (step 4). The close runs in the background: the turn's result does not wait for it, and the
   next turn waits for it (up to 10 seconds) before it opens a session. A later turn in that
   session waits for the agent's answer (up to `request_timeout_s`), since an agent refuses to
   resume a session it is still closing; after that it ends `error` ("the agent is still
   closing it"). A failed close is logged; a later turn in that session tries
   `session/resume`, and if the agent refuses it, prompts the session directly.

One process serves many sessions, one prompt at a time per session. qagents also runs one turn
at a time per workspace. Only the `keep_sessions` most recently used idle sessions stay open,
so with the default, continuing an older session costs a `session/resume`. Raise
`keep_sessions` for an agent that is slow to resume; `0` closes every session after its turn.
An agent that cannot close and resume sessions keeps them all open until the agent is closed.

## Updates and events

| ACP `sessionUpdate` | qagents |
|---|---|
| `agent_message_chunk`, text | `text_delta` (a whole block for Devvi, a token chunk for agents that stream); a chunk with a new `messageId`, and each of Devvi's blocks, starts after a blank line |
| `agent_message_chunk`, other content | `external_event` `image`, `audio`, `resource_link`, ...; bytes are never forwarded |
| `agent_thought_chunk` | `reasoning_delta` |
| `tool_call` | `tool_call_started`, named `<prefix>.<tool>`; counts toward `max_external_turns` |
| `tool_call_update` | new input: `external_event` `tool_arguments`; a later `name` or `kind` (for Devvi, `title`) renames the call for the permission requests that follow and in `ExternalResult.tool_calls`; `completed` or `failed`: `tool_call_finished` |
| `usage_update {used, size}` | `external_event` `context`: context pressure, not token usage |
| `config_option_update` | `external_event` `config` |
| `plan` | `external_event` `plan` |
| anything else | `external_event` `unknown` with `raw_events=True`; otherwise counted in `ExternalResult.raw["ignored_events"]` |

| `stopReason` | status | `stop_reason` |
|---|---|---|
| `end_turn` | `completed` | `end_turn` |
| `max_tokens` | `limit_reached` | `max_tokens` |
| `max_turn_requests` | `limit_reached` | `max_turn_requests` |
| `refusal` | `completed` | `content_filter`; through `as_tool()` an error result |
| `cancelled` | `cancelled` | `cancelled` |
| another value | `completed` | the value (logged as a warning) |
| JSON-RPC -32603 `"Internal error: turn failed: X"` | `error` | `turn failed: X` |

`ExternalResult.raw["acp_stop_reason"]` keeps the raw `stopReason`. `ExternalResult.backend` is
`acp:<agentInfo.name>@<agentInfo.version>`.

## Tool names for permission rules

Every permission request is decided by your `Permissions` on `"<prefix>.<tool>"`, where the
prefix is the tool name the calling model sees. ACP gives a tool call a `title` for display, an
optional `name`, and a `kind` (`read`, `edit`, `execute`, ...).

The tool name is the tool call's `name`, else its `kind` (`execute`, `edit`, ...), else
`"unknown"`, from everything the agent said about the call before it asked (a
`tool_call_update` that adds a `kind` counts). This is `default_tool_name`. The `title` is
not used: most agents build it from the tool's input (`"make"`, `"setup.py"`, `"Run ls -la"`),
so the model would choose the name your rules match, and a title such as `"rm"` would get past
`deny=["coder.execute"]`. So write rules on kinds, as in the [quick start](#quick-start):
`coder.read`, `coder.edit`, `coder.execute`. A rule on a title, such as `deny=["coder.rm"]`,
never matches.

Devvi is the exception: its code sets the title to the tool's name (`bash`,
`mcp__db__query`), and its `kind` is always `other`. For Devvi (`agentInfo.name`
`deepseek-harness-acp`, listed in `TITLES_ARE_TOOL_NAMES`) qagents uses `title_tool_name`
instead: the `name`, else the title when it looks like a tool name, else the kind. Pass
`tool_name=title_tool_name` for another agent whose code does the same, or your own function
of the tool call's fields that returns a name:

```python
from qagents.integrations.acp import ACPAgent, title_tool_name

coder = ACPAgent(
    ["node", "C:/tools/some-agent/bin.js", "--acp"],
    name="coder",
    workspace="C:/work/repo",
    tool_name=title_tool_name,  # only for an agent whose code puts tool names in titles
)
```

## Permission answers

A request names its tool call (`toolCall.toolCallId`) and may describe it (`title`, `kind`,
`rawInput`). Devvi sends only the id, after the matching `tool_call` update with the title and
the arguments. qagents decides on the call's latest known fields and arguments, and starts the
call from the request when the request is the first to describe it. It answers:

| qagents decision | Answer |
|---|---|
| allow | `{"outcome": "selected", "optionId": <the first allow_once option>}` |
| deny | `{"outcome": "selected", "optionId": <the first reject_once option, else reject_always>}` |
| cancel (an error, a timeout, arguments that are not a JSON object, a turn being cancelled, even when the decision came in at that moment, a posture that defers) | `{"outcome": "cancelled"}` |

An `allow_always` option is never chosen: when the agent offers no `allow_once`, an allow is
answered "cancelled". A request without a tool call id, or for a session where no turn runs, is
answered "cancelled". As soon as the turn is cancelled, every pending request is answered
"cancelled". Who decides, in what order, and the postures, are in
[External agents](../concepts/external-agents.md#permission-requests).

## No token counts

ACP carries no token usage, only context pressure. (ACP also has an unstable `usage` field
with token counts; qagents does not read it yet.) `capabilities().token_usage` is `False`, so a
calling run with `max_total_tokens` or `max_cost_usd` refuses the agent at its first call,
unless you build it with `allow_unmetered=True` and accept that its spend is not counted. The
refusal comes after the handshake: the agent process is started, answers `initialize` and stays
up, but no session is opened and no prompt is sent.

## Crash safety

The agent chooses session ids, so a crash of the calling run between `session/new` and the
journaled result would lose the link. With `resumable=True` (only for agents that support
`session/resume` or `session/load`), the turn journals the session id as soon as it is known,
and journals that the prompt was sent before sending it. A call repeated after the crash then
resumes that session instead of opening a new one, and the tool is declared idempotent:

- the prompt had not been sent: the task is sent as usual;
- the prompt had been sent: the agent receives `EXTERNAL_AGENT_RESUME` ("The previous attempt was
  interrupted. Check the workspace state and finish the task.") followed by the task;
- the agent cannot reopen the session: the turn ends `interrupted` and is not started over.

Without `resumable=True` (the default) a repeated call is not made: the calling model is told
the outcome is unknown.

## MCP servers for the agent

```python
import sys

from qagents.integrations.acp import ACPAgent, MCPServerHTTP, MCPServerStdio

tickets = MCPServerStdio(
    "tickets",
    command=sys.executable,  # absolute: the agent starts it from its own directory
    args=["-m", "my_app.ticket_server"],
    passthrough=["TICKETS_TOKEN"],  # copied from this process's environment when a session opens
)
docs = MCPServerHTTP(
    "docs", url="http://127.0.0.1:8765/mcp", headers={"Authorization": "Bearer ..."}
)

coder = ACPAgent(
    ["node", "C:/tools/some-agent/bin.js", "--acp"],
    name="coder",
    workspace="C:/work/repo",
    mcp_servers=[tickets, docs],
)
```

- `MCPServerStdio(name, command, args=(), env={}, passthrough=())`: a server the agent starts
  for each session. `command` must be an absolute path. Values of `env` are never shown in
  `repr`. Name credentials explicitly: an agent may strip variables that look like secrets from
  what it passes on (Devvi does).
- `MCPServerHTTP(name, url, headers={})`: a server the agent reaches over Streamable HTTP.
  Needs an agent that advertises `mcpCapabilities.http`; otherwise the handshake fails with a
  configuration error. Header values are never shown in `repr`.
- A plain mapping is sent as it is (an ACP `McpServer` object).
- `mcp_servers=` can be a function of the turn. It runs only when servers are sent: when
  the turn opens a session (`session/new`) or reopens one (`session/resume`,
  `session/load`). A session this connection already has open is continued with the
  servers it has, and the function does not run.
    - Return the list (or an awaitable of it) for servers that live as long as the session.
    - Return an async context manager that yields the list for servers that live for one
      turn, such as a qagents tool server bound to the calling run. qagents enters it before
      the session opens, closes the session right after the turn (`session/close`), then
      exits it, so the next turn resumes the session with servers of its own. An agent that
      cannot close sessions keeps the session open, but it is not continued directly again: a
      later turn resumes it, or says why it cannot.

For example, a qagents tool server that lives for one turn and is bound to the calling run
([Serving tools over MCP](mcp-server.md#bound-to-a-run)):

```python
import contextlib

from qagents.integrations.acp import ACPAgent
from qagents.integrations.mcp_server import tool_server


def lookup_customer(email: str) -> str:
    """Find a customer by email."""
    return "Maria"


@contextlib.asynccontextmanager
async def turn_tools(turn):
    async with tool_server([lookup_customer], ctx=turn.parent) as server:
        yield [server.acp_config()]


coder = ACPAgent(
    ["node", "C:/tools/some-agent/bin.js", "--acp"],
    name="coder",
    workspace="C:/work/repo",
    mcp_servers=turn_tools,
)
```

`DevviAgent(expose_tools=...)` does the same for Devvi.

## Options

`ACPAgent(command, *, ...)`:

| Option | Default | Meaning |
|---|---|---|
| `command` | required | program and arguments, as a list |
| `name` | `"acp"` | the agent's name; the default tool name and permission prefix |
| `env` | `{}` | variables set for the agent process |
| `passthrough` | `()` | names copied from qagents' environment, such as `["SOME_API_KEY"]` |
| `inherit_env` | `False` | give the agent qagents' whole environment, every secret in it too |
| `cwd` | the workspace | working directory of the agent process |
| `workspace` | `cwd` | the default `cwd` of sessions |
| `mcp_servers` | `()` | MCP servers to mount in each session, or a function of the turn |
| `model` | `None` | the model to select: an option value, an option name (`"deepseek-v4-pro"`), or, for agents whose values are JSON lists (Devvi), the list as a tuple; a model the session does not offer is refused |
| `config_options` | `{}` | other options, `{config id: value}`, for example `{"reasoning_effort": "high"}` |
| `auth_method` | `None` | the `authMethods` id to authenticate with; `None`: never authenticate |
| `resumable` | `False` | re-attach a call repeated after a crash (see above) |
| `raw_events` | `False` | forward unmapped `session/update` kinds as `external_event` `unknown` |
| `tool_name` | `default_tool_name` (`title_tool_name` for Devvi) | `(tool call fields) -> name`; see [Tool names for permission rules](#tool-names-for-permission-rules) |
| `keep_sessions` | 1 | sessions kept open after their turn; older idle ones are closed with `session/close` (when the agent can close and resume sessions), and a later turn in one resumes it; `0` closes each session after its turn |
| `handshake_timeout_s` | 60 | seconds for `initialize` and `authenticate` |
| `request_timeout_s` | 120 | seconds for `session/new`, `session/resume`, `session/load`, `session/set_config_option`; the prompt itself has no timeout (use `as_tool(timeout=...)` or the calling run's limits); also how long a turn waits for the `session/close` of the session it continues; a request without an answer ends the turn as `error` |
| `close_grace_s` | 5 | seconds the process gets to exit after its stdin closes, before its tree is killed |
| `max_line_bytes` | 32 MiB | largest message accepted from the agent |

Plus the `ExternalAgent` options: `description`, `permissions`, `on_permission`,
`approval_timeout_s`, `cancel_grace_s`, `max_external_turns`, `allow_unmetered`,
`track_changes`.

After the handshake, `agent.agent_info`, `agent.agent_capabilities`, `agent.auth_methods` and
`agent.session_config(session_id)` show what the agent reported.

## Errors

| Situation | Result |
|---|---|
| the program is missing, a `.cmd` shim on Windows, the handshake is refused or times out, another protocol version, a requested capability is missing, authentication fails | `ExternalAgentConfigurationError` when the agent connects; through `as_tool()` the model reads `... is unavailable` and later calls in the run fail at once |
| `session/new` answered -32602, -32600 or -32601 (an MCP server did not start, a refused `cwd`) | `ExternalAgentConfigurationError` |
| a model or option the session does not offer | `ExternalAgentConfigurationError`, listing what it offers |
| the turn failed (-32603) | status `error` with the agent's message; through `as_tool()` an error result the model can read |
| the error text says authentication, billing or a missing credential | status `error`, and the calling run stops calling the agent (`EXTERNAL_AGENT_UNAVAILABLE`) |
| the process exited, the stream ended or a pipe broke before `session/prompt` was sent, or a setup request got no answer | status `error` with the reason (nothing of the task reached the agent); `interrupted` and `raw["ambiguous"] = True` for a call repeated after a crash; the next turn starts a new process |
| the process exited, the stream ended or a pipe broke after `session/prompt` was sent | status `interrupted`, `raw["ambiguous"] = True`, the stderr tail in `raw["stderr_tail"]`; the next turn starts a new process |
| a cancel while the session is set up | status `cancelled`; the pending request is given up with `$/cancel_request`; the process is kept |
| a line over `max_line_bytes`, or three malformed lines in a row | the connection is closed: `interrupted` |
| the session to continue cannot be reopened | status `error`: `cannot continue session ...` |

## Adding a recorded fixture for a new agent

The offline tests drive `ACPAgent` against `tests/integrations/fakes/fake_acp_agent.py`, a
standard-library script that replays a recorded transcript as a real process. To add an agent,
do four things.

**Record its side of the wire.** Run the agent with a small task and save every line it writes
on stdout: the `initialize` result, the `session/new` result, the `session/update`
notifications, any `session/request_permission`, and the `session/prompt` result. Replace
volatile values with the placeholders the fake understands: `{{sessionId}}`, `{{messageId}}`,
`{{cwd}}`. Devvi's fixtures in `tests/fixtures/devvi/acp/` were made this way from Devvi's own
snapshots.

**Write the client side as directives** between the recorded lines. The fixture is JSON Lines,
and each line is one of:

| Line | Does |
|---|---|
| a JSON-RPC message (has `"jsonrpc"`) | written as recorded, after the placeholders are replaced; a response gets the id of the client request it answers |
| `{"expect": {"id", "method", "params"}}` | reads the next client message and checks the method and the given params (a subset); patterns `"$any"`, `"$absent"`, `"$absolute"`, `"$list"`, `"$string"` |
| `{"answer": {"id", "accept": [...]}, "otherwise": FILE}` | waits for the client's answer to the agent's request `id`; a different answer continues in `FILE` (a branch, such as the rejected recording) |
| `{"raw": TEXT}`, `{"big_chunk": N}`, `{"sleep": S}`, `{"exit": CODE, "stderr": TEXT}`, `{"hang": true}`, `{"write_file": {"path", "text"}}` | a malformed line, a large message, a pause, a crash, an agent that stops answering, an edit in the workspace |
| `{"fixture": ...}`, `{"comment": ...}` | nothing (a header or a note) |

A mismatch is written to stderr and ends the fake with exit code 3, so the test sees the
connection break and why.

**Test it** like the existing ones:

```python
import sys
from pathlib import Path

from qagents.integrations.acp import ACPAgent

FAKE = Path("tests/integrations/fakes/fake_acp_agent.py").resolve()
FIXTURE = Path("tests/fixtures/other-agent/acp/text-turn.jsonl").resolve()


async def test_other_agent_text_turn(tmp_path):
    agent = ACPAgent(
        [sys.executable, "-I", str(FAKE), "--script", str(FIXTURE)],
        name="other",
        workspace=tmp_path,
    )
    async with agent:
        result = await agent.run("Say hello")
    assert result.status == "completed"
    assert result.backend.startswith("acp:")
```

**Keep the agent's license notice** next to its fixtures, as `tests/fixtures/devvi/LICENSE` does
for Devvi's, and say in the docs which version of the agent the recording comes from.

Options of the fake that help: `--log FILE` (every client message, one JSON line each),
`--hang-after-cancel` (an agent that does not stop), `--child HEARTBEAT` (a child process, to
check that the whole tree is killed).

## Not in this version

- ACP's unstable `usage` field (token counts) is not read, so an ACP agent is unmetered.
- The `fs/*` and `terminal/*` client capabilities: qagents offers neither, and answers such
  requests JSON-RPC -32601.
- qagents as an ACP agent server, so that ACP clients could drive a qagents agent.
- Authentication is never automatic: qagents authenticates only with the `auth_method` you
  choose, and never with an interactive (`terminal`) method.
