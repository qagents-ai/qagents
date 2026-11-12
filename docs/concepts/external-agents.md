# External agents

A coding agent such as Devvi runs its own loop, with its own model, tools and sandbox. qagents
can put one to work: give it a task in a workspace, follow what it does, decide every permission
request it sends, stop it, and read its answer. Such an agent is an **external agent**.

| Thing | What it is | Who runs the loop | Who runs the tools |
|---|---|---|---|
| `Model` | one request in, one response out | qagents | qagents |
| `Agent` | a qagents loop over a `Model` | qagents | qagents |
| `ExternalAgent` | another agent program, reached over a control protocol | the external agent | the external agent; qagents decides every permission request it sends |

An external agent is not a model. Wrapping a coding agent as a `Model` would hide its tool calls
from the qagents journal and its permission requests from your rules. So qagents talks to it as
an agent, through `qagents.external.ExternalAgent`. Two adapters ship:

- [`ACPAgent`](../integrations/acp.md) drives any agent that speaks ACP v1, the Agent Client
  Protocol.
- [`DevviAgent`](../integrations/devvi.md) drives Devvi, over ACP (the default) or over Devvi's
  own SDK protocol.

## Two planes: who drives the agent, and what tools it has

Two different protocols are in play, and they point in opposite directions.

```text
 CONTROL PLANE: who drives the agent. ACP v1 or the Devvi SDK, JSON-RPC over the agent's stdio.

 +---------------------------+   task, session, cancel, answers   +------------------------+
 | qagents                   | ---------------------------------> | coding agent           |
 |   your lead Agent         |                                    |   Devvi, any ACP agent |
 |   coder = DevviAgent(...) | <--------------------------------- |   own loop and model,  |
 |   tools=[coder.as_tool()] |   text per step, tool calls,       |   own tools, sandbox   |
 +---------------------------+   permission requests, turn end    +------------------------+
                                                                              |
 TOOL PLANE: what tools the agent can call. MCP.                              | MCP tool calls
 The agent is the MCP client; qagents serves the tools.                       v
                                                       +-------------------------------------+
                                                       | qagents MCP tool server             |
                                                       |   your qagents tools, decided by    |
                                                       |   your Permissions, journaled in    |
                                                       |   your run                          |
                                                       +-------------------------------------+
```

**The control plane drives the agent.** qagents is the client and the agent is the server. The
client opens a session in a workspace, sends a task, receives the agent's steps as they are
committed, answers its permission requests, and can cancel the turn while keeping the session.
ACP is the standard protocol for this, and Devvi also has its own SDK protocol. Both are
JSON-RPC 2.0 over the agent's standard input and output.

**The tool plane gives the agent tools.** Here the agent is the client. MCP, the Model Context
Protocol, lets an agent list a server's tools and call them. qagents can serve its own tools
over MCP to an agent it drives ([Serving tools over MCP](../integrations/mcp-server.md)).

!!! warning "There is no MCP for driving a coding agent"
    MCP is a tool protocol. One MCP tool call is one request and one result. A coding agent's
    turn is something else: a session in a workspace, many steps, permission requests that
    come back to the caller in the middle of the turn, and a cancel that stops the turn but
    keeps the session. MCP is not built for that; ACP is.

    - **Devvi has no MCP server.** It ships an MCP client only (`packages/mcp/mcp-client`;
      paths like this one on this page name files in the Devvi repository); the MCP server
      SDK appears only in that package's development dependencies
      (`packages/mcp/mcp-client/package.json`). There is nothing for an MCP client to connect
      to.
    - **`MCPToolkit` cannot drive Devvi or any coding agent.** `qagents.toolkits.mcp.MCPToolkit`
      gives a qagents agent the tools of an MCP server. An agent that serves itself as an MCP
      tool (Codex: `codex mcp-server`) can be sent a task that way, but MCPToolkit cannot
      follow its steps, answer its permission requests or cancel it.
    - Even where an agent can be wrapped as a single MCP tool, the whole turn is then one
      opaque call: qagents could not see its steps, decide its permission requests one by one,
      or cancel the turn and keep the session.

    To make a coding agent work for you, use `ACPAgent` or `DevviAgent`. To give that agent
    your tools, serve them over MCP. The two combine: qagents drives Devvi over ACP and hands
    it qagents tools over MCP in the same session (`DevviAgent(expose_tools=[...])`).

Which piece to use for which job:

| You want | Use | Protocol | Who is the client |
|---|---|---|---|
| A coding agent does work for your qagents agent | `DevviAgent`, `ACPAgent`, then `.as_tool()` | ACP v1 or the Devvi SDK | qagents |
| Your qagents agent calls the tools of an MCP server | `qagents.toolkits.mcp.MCPToolkit` | MCP | qagents |
| A coding agent calls your qagents tools | `tool_server(...)`, or `DevviAgent(expose_tools=[...])` | MCP | the coding agent |
| A coding agent reads the qagents docs | `python -m qagents.docs_mcp` ([Docs for coding agents](../coding-agents.md)) | MCP | the coding agent |

## Using an external agent

The usual shape is a lead agent that plans and reviews, with a coding agent as one of its tools.

```python
from qagents import Agent, Limits, Permissions
from qagents.integrations.devvi import DevviAgent

coder = DevviAgent(
    workspace="C:/work/repo",
    patches=[{"id": "tool-web", "disabled": True}],  # Devvi without its web tools
    allow_unmetered=True,  # ACP reports no token counts; the cost cap then counts the lead only
)
lead = Agent(
    "qcloud:<model-id>",
    instructions="Plan the change, delegate the code edits to `coder`, then review them.",
    tools=[coder.as_tool(name="coder", timeout=1800)],
    permissions=Permissions(
        ask=["coder.bash", "coder.pwsh"],  # Devvi asks only to leave its sandbox
        deny=["coder.web_fetch"],  # defense in depth: Devvi does not ask before a web fetch
    ),
    limits=Limits(max_cost_usd=2.0, timeout_s=3600),
)

async with coder:
    result = await lead.run("Make tests/test_parser.py pass.")
```

The lead's model sees one tool, `coder(task, session=None)`. Each call is one turn of the
external agent, run as a child of the lead's tool call:

- it starts at depth + 1 and is refused beyond `Limits.max_depth`;
- its events reach the lead's stream as `subagent_event`;
- the permission requests it sends are decided by the lead's `Permissions`, on names such as
  `coder.bash`, with the lead's `on_ask`; Devvi sends one only for a step that asks to leave its
  sandbox (see [below](#what-qagents-decides-and-what-the-agent-decides));
- cancelling the lead, or the lead's timeout, cancels the turn;
- its token usage, when the agent reports any, is added to the lead's run as it arrives. In a
  tool of your own that runs `result = await coder.run(task, parent=ctx)`, the usage is already
  counted, so `ctx.add_usage(result.usage)` adds nothing, as for an `Agent` child.

The lead's model reads the agent's final message and a footer that qagents adds. Here the coder
read a file, ran the tests without asking, then asked to run a command outside its sandbox;
an `ask` rule matched, and with no `on_ask` the request was refused:

```text
Fixed parse_line; the test passes.

[coder: session coder-1; completed; 3 tool calls (pwsh 2, read 1); 1 permission refused (pwsh: "Remove-Item -Recurse C:/build")]
```

To continue the same conversation, the model passes `session` from the footer in its next call.
It can continue only a session that this tool opened in this qagents session (the latest 50
are kept in `ctx.state["qagents.external.sessions"]`). Any other session id is a tool error
that says so (`coder failed: session ... does not exist; leave session out to start a new
one`), and the agent is not reached. Your own code may continue any session:
`coder.run(task, session=...)` and `coder.stream(task, session=...)`.
Give `as_tool()` a `timeout` that fits coding work: without one the lead's
`Limits.tool_timeout_s` applies, which is 120 seconds by default, and the turn is cancelled at
about 110 seconds ([Budgets, limits and cancel](#budgets-limits-and-cancel)).

You can also run a turn directly and follow its events:

```python
async with coder:
    async with coder.stream("Add type hints to utils.py") as turn:
        async for event in turn:
            print(event.type)
    print(turn.result.status, turn.result.changed_files)
```

`coder.run(task)` does the same and returns the `ExternalResult`. The result is a pydantic model,
so a `Workflow` step can record it; see [Devvi as a worker](../integrations/devvi.md#as-a-workflow-step).

## What qagents decides, and what the agent decides

| qagents decides | The external agent decides |
|---|---|
| Whether the agent runs at all: `as_tool(requires_approval=True)`, the depth limit, a budget it cannot report against | Which tools it has and what they do (`read`, `edit`, `bash`, ...) |
| The answer to every permission request the agent sends, through your `Permissions` | When to ask: qagents sees only the requests the agent chooses to send |
| Time: `as_tool(timeout=...)`, `Limits.tool_timeout_s`, `Limits.timeout_s` of the calling run | Its sandbox: what a command may touch without asking |
| Tokens and cost, when the agent reports usage | Its model, its system prompt, its context management |
| Steps per turn: `max_external_turns` (50 by default) | How it plans and when it is done |
| Cancelling a turn, and killing the process tree when it does not stop | Its own session log, the full inner transcript |
| What the calling run journals, and whether a call is repeated after a crash | |
| The process: its command, working directory and environment variables | |
| Which qagents tools it gets, and the rules for each call it makes to them | |

Two consequences matter.

**A tool call the agent does not ask about runs without a qagents decision.** You see it as a
`tool_call_started` event while it runs, not before, and no rule of yours is consulted. Devvi
allows a tool call unless one of its gates says "ask"
(`packages/core/tools/src/pipeline.ts`). With its default profile only an
escalation asks: a shell command or file change sent with `sandbox_permissions` (for example
`"danger-full-access"` in Devvi's recorded transcript
`snapshots/acp/escalation-approved/stdout.expected.jsonl`). Measured in tier 1
under the default `workspace-write` mode: file writes and ordinary commands inside the
workspace, web fetches and MCP tool calls ran with no request, even with a deny rule on
`coder.web_fetch` (`test_a_web_fetch_runs_without_a_permission_request` in
`tests/integrations/test_integration_docs.py`). So a rule on `coder.web_fetch` or `coder.read`
is never consulted: to take a tool away, remove it from the agent itself, for Devvi with a patch
row ([Devvi permissions](../integrations/devvi.md#permissions)).

**The agent's sandbox is the agent's, not qagents'.** Devvi's sandbox modes govern file effects
only; network and process policy are outside them
(`packages/sandbox/sandbox-policy/README.md`). On Windows its enforcement
is reported as `partial`: reads stay unconfined and NTFS hard links can alias files outside the
workspace (`packages/sandbox/sandbox-local/README.md`). Do not treat it as
a security boundary. For real isolation, run the agent in a disposable clone or a container, and
gate the delegation instead of each command (see
[the recipe](../integrations/devvi.md#gate-the-delegation-not-each-command)).

## Permission requests

Every permission request of the agent goes through a `PermissionBridge`. It asks the turn's
`Permissions` about a synthetic call named `"<prefix>.<agent tool>"`: the prefix is the tool
name the calling model sees (`coder`), or the agent's `name` for a direct run. The arguments
are those of the matching tool call. The synthetic tool is declared `requires_approval=True`,
so when no rule matches, the decision is "ask".

The rules are, in this order: `stream(permissions=...)`, the agent's own `permissions=`, the
calling run's. When the chosen rules have no `on_ask`, the calling run's `on_ask` is used.

| Decision | What the agent is answered |
|---|---|
| a deny rule, or `can_use_tool` refuses or raises | reject (`decided_by="rules"`): this one action is refused and the turn goes on |
| an allow rule, or the mode allows | allow this one call (`rules`) |
| ask, and `on_ask` approves | allow this one call (`on_ask`) |
| ask, and `on_ask` declines or raises | reject (`on_ask`) |
| ask, and `on_ask` approves with edited arguments | reject (`edited_arguments`): qagents cannot change what the agent runs |
| ask, no `on_ask` | depends on the posture, below |
| the decision took longer than `approval_timeout_s`, or failed outside your rules and `on_ask` | "cancelled" (`timeout`, `error`): the agent does not run the tool |
| the tool call's latest input is not a JSON object | "cancelled" (`unreadable_arguments`), without asking the rules: they cannot see what would run |
| a request that arrives, or a decision that ends, while the turn is being cancelled | "cancelled" (`cancelling`), also for an allow that came in at that moment |

An "allow always" option is never chosen, even when the agent offers one: every grant covers
one call. When the agent offers no "allow once" option, an allow is sent as "cancelled"
(`no_option`), and the record says so.

**Postures.** `on_permission=` on the agent chooses what "ask" without an `on_ask` does:

| Posture | Ask without `on_ask` | Default `approval_timeout_s` | Use it |
|---|---|---|---|
| `"rules"` (default) | refused (`no_approver`); the refusal is in the result and in the footer | 60 s | rules decide everything; the lead's model reads the refusal and adapts |
| `"pause"` | the turn ends, and the calling run pauses until a human answers | 60 s | a person approves commands, possibly days later |
| `"live"` | refused, as `"rules"` | 900 s | an `on_ask` that waits for a person in a long-lived process |

### Pause for a human

With `on_permission="pause"`, a request that needs a human answers the agent "cancelled", ends
the external turn, and pauses the calling run with a `PendingInput` (reason `"permission"`). Its
`metadata` names the agent, the external session, the tool, the arguments and a fingerprint.
Answer it with `ExternalApproval`:

```python
from qagents import Agent, Session, SQLiteStore
from qagents.external import ExternalApproval
from qagents.integrations.devvi import DevviAgent

coder = DevviAgent(workspace="C:/work/repo", on_permission="pause", allow_unmetered=True)
lead = Agent("qcloud:<model-id>", tools=[coder.as_tool(name="coder", timeout=1800)])

async with coder, SQLiteStore("agents.db") as store:
    session = Session(store, id="fix-42")
    result = await lead.run("Clean the build folder, then run the tests.", session=session)
    if result.status == "paused":
        pending = result.pending_inputs[0]
        print(pending.metadata["tool"], pending.metadata["arguments"])  # coder.bash {...}
        answer = ExternalApproval.for_input(pending, approved=True)
        result = await lead.resume(session, results={pending.tool_call_id: answer})
```

The lead's model reads the answer as the result of its paused call, for example
`Approved: coder may run bash "rm -rf build" once. Call coder again with session=... to
continue.` When it calls `coder` again with that session, the agent is told the action was
decided (`EXTERNAL_AGENT_CONTINUE`), and the first request with the same fingerprint (the
prefixed tool name and the canonical JSON of the arguments) is allowed, once. Free text that
the agent writes anew on each attempt is left out of the fingerprint
(`ExternalAgent.grant_ignored_arguments`; for Devvi `description` and `justification`), so the
same command with a reworded description is still the approved request. If the agent asks
for something else, that request is decided afresh. Deny rules still refuse it whatever the
approval says. A direct run (no calling run to pause) refuses as `"rules"` does.

A tool built with `as_tool(continuable=False)` has no `session` parameter: its next call
continues the paused session by itself, and the approval text does not mention `session=`.

## Budgets, limits and cancel

| Trigger | What happens |
|---|---|
| The calling run is cancelled, or its `timeout_s` expires | the turn is cancelled; the call's result says its outcome is unknown (`TOOL_INTERRUPTED`), because the agent may have changed files before it stopped, and the session stays in `ctx.state["qagents.external.sessions"]` |
| `as_tool(timeout=T)` expires | the turn is cancelled with reason `timeout`; the model is told, with the session to continue |
| No `timeout`, and the calling run's `Limits.tool_timeout_s` is more than `2 × (cancel_grace_s + 5)` (20 s by default) | the turn is cancelled at `tool_timeout_s - (cancel_grace_s + 5)` seconds, so the cancel is graceful and the model is told the session |
| No `timeout`, and a shorter `Limits.tool_timeout_s` | the loop's own tool timeout ends the call at `tool_timeout_s`: the model reads `TOOL_TIMEOUT` (`'coder' did not finish within 3 seconds and was cancelled. ...`), without the session or the footer |
| The calling run's `max_total_tokens` or `max_cost_usd` is used up by the agent's reported usage | the turn is cancelled as `limit_reached` |
| The calling run has `max_total_tokens` or `max_cost_usd`, and the agent reports no token usage | the call is refused before a session or a turn starts, unless the agent was built with `allow_unmetered=True`. The agent process may already be running: the handshake comes first (for an ACP agent, the process starts and answers `initialize`) |
| More than `max_external_turns` steps (default 50) | the turn is stopped as `limit_reached` (`max_external_turns`) |
| The call would nest deeper than `Limits.max_depth` | refused before the agent starts |

A step is one model request where the protocol reports it (the Devvi SDK's `step/start`), and
one tool call where it does not (ACP).

**One turn at a time.** One turn runs at a time per workspace (across all the external agents
of the process) and per session; a call waits for that, and the `as_tool(timeout=...)` time
includes the wait. A call whose time runs out while it waits was never started: the model
reads `EXTERNAL_AGENT_BUSY`, and the footer says `no session; cancelled (busy)`. A turn
cancelled before it starts, or while it waits, ends at once (`raw["not_started"]`), without
starting the agent.

**Cancel.** qagents asks the agent to stop (ACP: `session/cancel`; the Devvi SDK has no
per-turn cancel, so the process is shut down). The agent then has `cancel_grace_s` seconds
(default 5) to end the turn. If it does not, the turn is abandoned, the whole process tree is
killed, and the result has `raw["forced"] = True`; the next turn starts a new process.

**Unmetered agents.** ACP carries no token counts, only context pressure. A calling run with a
token or cost budget therefore refuses an ACP agent, because the budget could not be enforced.
Build the agent with `allow_unmetered=True` to accept that its spend is not counted (the lead's
own spend still is). The refusal happens at the first call, not when the agent is built, and
after the handshake: the agent's process is started and stays up, but no session is opened and
no prompt is sent. To count Devvi's tokens and cost, use its SDK transport with prices
([Token usage and cost](../integrations/devvi.md#token-usage-and-cost)).

## The journal and crash recovery

The calling run journals an external call like any tool call: `tool_started` before it runs,
`tool_result` after, with the final text plus the footer as the content and the usage the
agent reported. The inner transcript is not copied into the calling conversation: the lead's
context stays small, and the journal does not depend on the agent's format. The agent's own
session log stays the record of what happened inside.

Besides that, qagents keeps:

| Where | What |
|---|---|
| `ctx.state["qagents.external.sessions"]` | per tool name, the sessions it used in this qagents session (`session_id`, `cwd`, `backend`; the last 50) |
| `ctx.state["qagents.external.grants"]` | pauses waiting for an answer (posture `"pause"`) |
| journal entries of kind `tool_checkpoint` | facts a resumable call records before it ends: the external session id (`external_session`), adapter facts such as "the prompt was sent" or Devvi's message id, and the finished result (`external_result`); readers that do not know the kind ignore it |

**After a crash.** When the process running the lead dies and the run is resumed:

- A call whose `tool_result` was journaled is not run again.
- A call that was running is repeated only when the agent's turns are resumable
  (`resumable_turns`). `DevviAgent` is resumable on both transports by default; `ACPAgent` only
  with `resumable=True`. The repeat re-attaches to the session the first attempt journaled:
  - if the first attempt had finished and only its `tool_result` was lost, the recorded result
    is returned and no turn runs;
  - otherwise the agent continues the same session and is told the previous attempt was
    interrupted (`EXTERNAL_AGENT_RESUME`: check the workspace state and finish the task);
  - on the Devvi SDK transport, a turn that Devvi's session log shows as finished is replayed
    from the log and not prompted again;
  - if the agent no longer has the journaled session, or cannot reopen it, the call ends
    `interrupted` with an unknown outcome and the task is not sent again
    (`ACP_NOT_REATTACHED`, `DEVVI_NOT_REATTACHED`): the earlier attempt may have sent it. The
    model is told to check the workspace and to call again without a session
    (`EXTERNAL_AGENT_INTERRUPTED_NO_SESSION`).
- A call to an agent that is not resumable is not repeated: the lead's model is told that its
  outcome is unknown, as for any interrupted tool with side effects.

### Guarantees for external calls

| Guarantee | Scope |
|---|---|
| A call whose result was journaled never runs again | always (the agent loop) |
| A call that was running when the process died is repeated only for a resumable agent, and the repeat continues the journaled session instead of starting a new one | `DevviAgent` (both transports, by default), `ACPAgent(resumable=True)` |
| A repeated call whose first attempt finished returns the recorded result without a new turn | resumable agents, once the result checkpoint was written |
| A repeated call does not prompt Devvi again when its session log shows the turn finished | Devvi SDK transport, with the default uncompressed log |
| Every permission request is decided by your rules, and never answered "allow always" | always |
| A permission check that raises (`can_use_tool`, `on_ask`) answers reject; a timeout or any other failure while deciding answers "cancelled"; either way the tool does not run | always |
| A turn whose connection broke is reported `interrupted`, with `raw["ambiguous"] = True`, and never counted as completed; a plain `error` only when the adapter knows nothing of the turn was sent | always |
| A repeated call whose session the agent no longer has (or cannot reopen) is not sent again | resumable agents |
| One turn at a time per workspace, and per continued session | within one process |
| The agent's reported usage counts against the caller's budgets as it arrives | agents that report usage (Devvi SDK) |

Not guaranteed:

- **Exactly-once work inside the agent.** A repeated call asks the agent to check the workspace
  and finish; it may redo part of the work. Use a git workspace and review the diff.
- **A decision before every tool call.** Only the calls the agent asks about are decided
  before they run.
- **Cost of agents that report no usage.** With `allow_unmetered=True` it is not counted.
- **Live tokens.** See below.
- **The agent's sandbox as a security boundary.**

## Text arrives per step, not per token

Devvi sends whole committed blocks: each `text_delta` holds a finished paragraph or message of
the agent, not a few tokens. Its ACP bridge derives updates from committed session events
(`packages/acp/acp/src/updates.ts`), and its ACP README says raw provider
deltas stay off the wire (`packages/acp/acp/README.md`). The SDK protocol
sends committed session events too (`session.event`,
`packages/sdk/protocol/src/types.ts`): a whole `assistant/message` per model
request. `ExternalCapabilities.live_tokens` is `False` for Devvi on both transports.

So a UI shows the coder's work step by step: a message, a tool call, its result, the next
message. Do not build a "typing" effect on it. An ACP agent that streams token chunks gets them
forwarded as it sends them; qagents does not merge or split text.

Committed blocks are separated by a blank line when the adapter tells blocks apart
(`text_delta(text, message_id=...)`): Devvi's blocks on both transports, and ACP chunks with a
new `messageId`. Two blocks with no tool call between them then read
`"Fixed it.\n\nTests pass."`, not `"Fixed it.Tests pass."`; the blank line is also at the
start of the second block's `text_delta`. Token chunks of one message are joined as they
arrive. A tool call starts a new message: the turn's final text is the text after the last
tool call.

## Events

A turn emits ordinary qagents events. A turn used as a tool reaches the caller's stream as
`SubagentEvent` (its `agent` is the tool name), which `RunStream` consumers, `Hooks.on_event`
and the AG-UI projection handle as for an agent used as a tool.

A direct turn (`agent.stream(...)`) is like a `RunStream`: it has `run_id`, `text()` (only the
text), `cancel()`, `wait()` and `aclose()`. But it ends with the `external_event`
`turn_finished`, not `run_finished`. `to_agui_events(turn)` ends the AG-UI run at that event:
`RUN_FINISHED` for `completed`, `cancelled` and `limit_reached`, `RUN_ERROR` for `error` and
`interrupted`. The agent works in its own process
whether or not you read the events, so use a direct turn in `async with`, or read it to the
end: breaking out of `async for` cancels the turn.

| Event | When |
|---|---|
| `run_started` | the session is known (`run_id` is `ext:<session>:<12 hex>`, unique for every turn; an `as_tool()` call keeps its id when the calling run repeats it after a crash; `resumed` says whether the session was continued) |
| `turn_started` | a model request of the agent starts (Devvi SDK only) |
| `text_delta`, `reasoning_delta` | the agent committed a message or a thought |
| `model_responded` | a model response with usage (Devvi SDK only) |
| `tool_call_started`, `tool_call_finished` | a tool call of the agent, named `<prefix>.<tool>`; qagents measures the duration |
| `external_event` | anything without a qagents counterpart; `kind` says what |

`ExternalAgentEvent` (`type="external_event"`) kinds:

| `kind` | Meaning |
|---|---|
| `permission_requested`, `permission_decided` | every permission request and its answer (`tool`, `arguments`, `outcome`, `rule`, `decision`, `reason`) |
| `tool_arguments` | a started tool call got new input: `{tool_call_id, name, arguments}`. On ACP every `tool_call_update` with new `rawInput` emits one, so expect it often. The `tool_call_started` event keeps the first arguments; `tool_call_finished` carries the latest |
| `approval_refused` | Devvi SDK: a request about a call its session never started, answered "cancelled" without consulting your rules (`{toolName, callId, reason: "unknown_call"}`) |
| `context` | ACP context pressure, `{used, size}`; not token usage |
| `config` | the agent changed a session option |
| `plan` | an ACP plan update |
| `image`, `audio` | the agent sent an image or audio (`mimeType` only; the bytes are not forwarded) |
| `resource_link`, `resource` | ACP: the agent sent a link to a resource (`name`, `uri`, `mimeType`) or an embedded resource (`uri`, `mimeType`; not its content) |
| `retry`, `compaction`, `job`, `attempt` | Devvi SDK: a model retry, a context compaction (its usage is counted), a background job, an abandoned attempt |
| `web_search` | Devvi SDK: the DeepSeek web search made a model request of its own, which Devvi records without usage; qagents counts it as one request whose tokens and cost are unknown (`usage.unpriced_requests`) |
| `approval_asked`, `approval_decided` | Devvi SDK: Devvi's own audit records of a request that qagents also reports as `permission_*` |
| `subagent_started`, `subagent_finished` | Devvi SDK: one of Devvi's own subagents started (`parent_session_id`, `depth`) or finished (`status`, `stopReason`, a short `text`); `session_id` is the subagent's |
| `subagent_event` | Devvi SDK: a step of one of Devvi's own subagents |
| `unknown`, `raw` | unmapped updates, only with `raw_events=True`; otherwise they are counted in `ExternalResult.raw["ignored_events"]` |
| `turn_finished` | always the last event: status, stop reason, session, backend, usage, tool call counts, permissions, changed files (`changed_files`: the first 100 paths; `changed_files_count`: all of them; `ExternalResult.changed_files` has every path) |

## What the calling model reads

The tool is `name(task, session=None)` (`continuable=False` drops `session`). Its description is
the agent's `description`, or a default that tells the model the agent sees only the task text.

A completed turn returns the final text and the footer. Anything else is an error result whose
footer names the session when the turn had one. A turn that never began in a session (it was
refused, it waited too long for its workspace, or it broke before the agent confirmed one)
names no session, also when the call asked to continue one: `[coder: no session; cancelled
(busy)]`.

| Status | The calling model reads |
|---|---|
| `completed` | the final text, then `[name: session ...; completed; N tool calls (...); refused requests; changed: files]` |
| `cancelled`, `limit_reached` | `coder stopped before finishing (timeout).`, the text so far, the footer |
| `error` | `coder failed: <the agent's message>`, the footer |
| `error` that will repeat (authentication, billing, a missing credential), or a setup failure | `coder is unavailable: ... do not call coder again`; later calls in the same run fail at once |
| `interrupted` (the process exited, a pipe broke) | `coder stopped unexpectedly (...) ... Its outcome is unknown: check what it changed before calling it again.`, with the session to continue |
| `interrupted` with no session (it broke before one was opened, or a repeated call's session is gone) | `coder stopped unexpectedly (...) before it had a session it can continue. ... To try again, call it without a session.` |
| refused before the start | `coder was not started: ...` (too deep, or an unmetered agent under a budget) |

Every one of these strings, and the few the ACP and Devvi adapters add, is a plain module
attribute in [`qagents.prompts`](../reference/prompts.md). Nothing else is added to what the
models read.

`ExternalResult` holds the same facts as data: `status`, `stop_reason`, `text`, `session_id`,
`backend` (for example `devvi-acp@0.2.1-rc.3`), `usage`, `tool_calls`, `permissions`,
`changed_files` (with git tracking, by default when the workspace is a git repository), and
`raw`. `await result.diff()` gives the `git diff` of the changed files on demand; qagents never
sends it to a model.

Through `as_tool()` the same facts reach the tool result: `ToolResultPart.metadata["external"]`
has `status`, `stop_reason`, `session_id`, `backend`, `usage`, `tool_calls` (counts per name),
`permissions`, `changed_files` (the first 100) and `changed_files_count`, and
`metadata["ambiguous"]` is `True` when the turn was interrupted and its outcome is unknown.

The agent can write the repository's `.git/config`, so git runs with a small environment (no
secrets, no network transport) and with every program that config could name switched off:
fsmonitor, filters, an external diff, textconv and the pager. Submodules' work trees are not
looked into. git itself is found on `PATH`, never in the workspace or the current directory.

`stop_reason` uses one vocabulary for every agent: `end_turn`, `max_tokens`, `cancelled`,
`content_filter` and the limit names (`STOP_REASONS` in `qagents.external`). An `error` turn's
`stop_reason` is the agent's message, and a word an adapter cannot map passes through. The
agent's own word stays in `raw`: `raw["acp_stop_reason"]` for an ACP agent,
`raw["agent_stop_reason"]` for Devvi on both transports.

## Testing offline with `ScriptedExternalAgent`

`ScriptedExternalAgent` plays scripted turns in memory, the way `ScriptedModel` plays model
replies. Everything else is the real code: the permission bridge, the footer, the events,
budgets and the journal.

```python
from qagents import Agent, Permissions
from qagents.external import ScriptedExternalAgent, ToolStep
from qagents.messages import Usage
from qagents.models.testing import ScriptedModel, call


ESCALATION = {
    "command": "Remove-Item -Recurse C:/build",
    "sandbox_permissions": "danger-full-access",  # Devvi asks only for steps like this one
}


async def test_the_lead_delegates_and_an_escalation_is_refused():
    coder = ScriptedExternalAgent(
        [
            [  # one turn of the coder
                "Reading the parser.",
                ToolStep("read", {"path": "src/parser.py"}, result="def parse_line..."),
                ToolStep("pwsh", {"command": "pytest -q"}, result="1 passed"),  # not asked
                ToolStep("pwsh", ESCALATION, ask=True),  # asked: it wants to leave the sandbox
                Usage(requests=1, input_tokens=120, output_tokens=40),
                "Fixed parse_line; the test passes.",
            ]
        ],
        name="coder",
    )
    model = ScriptedModel([call("coder", task="Fix parse_line and run the test."), "Done."])
    lead = Agent(model, tools=[coder.as_tool()], permissions=Permissions(ask=["coder.pwsh"]))

    result = await lead.run("Fix the parser.")

    assert result.output == "Done."
    assert [(a.outcome, a.decided_by) for a in coder.answers] == [("deny", "no_approver")]
    footer = model.requests[1].messages[-1].tool_results[0].content
    assert "1 permission refused (pwsh" in footer
```

Only the step with `ask=True` reaches the rules, as with a real agent: the first `pwsh` call
runs without a decision although an `ask` rule matches its name.

A turn is a list of steps:

| Step | Plays |
|---|---|
| `"text"` | a committed message (`text_delta`); two in a row are separated by a blank line |
| `ReasoningStep(text)` | a thought |
| `ToolStep(name, arguments, result="OK", is_error=False, ask=False)` | a tool call; with `ask=True` the agent asks first and runs the tool only when allowed |
| `PermissionStep(name, arguments, reason=None)` | a permission request without a tool call |
| `EventStep(kind, data)` | an `external_event` |
| `Usage(...)` | reported token usage (added to the calling run at once) |
| `WaitStep(seconds=None, ignore_cancel=False)` | wait, or until cancelled; `ignore_cancel=True` plays an agent that does not stop |
| `EndStep(status, stop_reason, text)` | end the turn with that status |
| an exception instance | raised; an `ExternalAgentConnectionError` plays a crashed process |
| a function of the `ExternalTurnContext` | anything else, sync or async |

The agent records what it received: `calls` (task, session, cwd, `resumed`, checkpoints,
answers), `prompts`, `answers` (every `PermissionAnswer`), `connects`, `disconnects` and
`cancel_requests`. New sessions are named `"<name>-<n>"`. By default it reports token usage
and is not resumable, unlike `DevviAgent` on ACP: pass `capabilities=ExternalCapabilities()`
to test an unmetered agent, and `resumable_turns=True` to test crash re-attach. To stand in for
`DevviAgent` in a test of your lead, set both
([Test your lead offline](../integrations/devvi.md#test-your-lead-offline)).

To test an adapter against a real process without the real agent, the test suite has fake
peers (`tests/integrations/fakes/fake_acp_agent.py` replays recorded ACP transcripts); see
[Adding a recorded fixture](../integrations/acp.md#adding-a-recorded-fixture-for-a-new-agent).

## Writing an adapter

Subclass `qagents.external.ExternalAgent` and implement `capabilities()` and
`run_turn(turn)`; optionally `connect()`, `disconnect()` and `cancel_turn(turn)`. `run_turn`
reports everything through the `ExternalTurnContext` it receives (`begin`, `text_delta`,
`tool_started`, `tool_finished`, `add_usage`, `ask_permission`, ...) and returns an
`ExternalOutcome`. Raise `ExternalAgentConnectionError` when the connection broke: the turn
ends `interrupted` with an unknown outcome. Pass `ambiguous=False` when nothing of the turn
reached the agent (the connection was gone before you sent anything): the turn then ends as a
plain `error`, without "outcome unknown". Raise `ExternalAgentUnavailableError` for a failure
that will repeat. Map the agent's stop words onto `STOP_REASONS` and keep its own word in
`raw`. Report new input for a started call with `tool_started` again (same id, new
`arguments`): it emits `tool_arguments`. The base class turns the turn into events, the
`ExternalResult`, the journal entries and the tool result, and provides the locks, change
tracking, budgets and cancel. See the [API reference](../reference/external.md).

**Event loops.** Use an external agent inside `async with` (or close it with `aclose()`) on the
event loop that runs it. `Agent.run_sync` and each `asyncio.run` start a new loop per call: an
agent used across such calls stops its process and starts a new one on the next loop, so each
call pays the start-up again. A process left running is stopped when Python exits. An
`aclose()` (or the end of `async with`) that is cancelled while the agent stops still lets the
process stop, and a later `aclose()` waits for it.

!!! note "Not in this version"
    - An unmetered agent under a token or cost budget is refused at its first call, after the
      handshake, not when the agent is built.
    - No tracing spans or span events are emitted for the steps inside an external turn.
    - ACP's unstable `usage` field is not read yet, so ACP agents stay unmetered.
    - Only Devvi has recorded ACP transcripts. Other ACP agents are expected to work and need
      a fixture of their own before they are called supported.

