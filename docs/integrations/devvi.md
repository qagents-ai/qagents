# Devvi as a coding worker

[Devvi](https://devvi.sh) is a coding agent: a fork of DeepSeek Harness, MIT-licensed. It reads
and edits files, runs commands, and keeps its own sessions.
`qagents.integrations.devvi.DevviAgent` lets a qagents agent hand it coding work and keep
control of permissions, time, budgets and the journal. Paths on this page such as `LICENSE` or
`packages/mcp/mcp-client` name files in the Devvi repository.

!!! warning "Devvi is an agent, not a model, and it has no MCP server"
    qagents drives Devvi over a control protocol: ACP v1 (`devvi --profile acp`, the default)
    or Devvi's SDK protocol (`devvi --profile sdk`). Devvi ships an MCP *client* only
    (`packages/mcp/mcp-client`), so `MCPToolkit` cannot drive it. MCP is used
    in the other direction, to give Devvi qagents tools
    ([qagents tools inside Devvi](#qagents-tools-inside-devvi)). The picture is in
    [External agents](../concepts/external-agents.md#two-planes-who-drives-the-agent-and-what-tools-it-has).

## What you need

| Item | Version | Source |
|---|---|---|
| Devvi CLI (`@deepseek-ai/dsh`) | 0.2.x; written against 0.2.1-rc.3. Other versions are refused unless `allow_untested_version=True` | `apps/cli/package.json` |
| Node.js | 24.9.0 or newer (Devvi's `engines.node`); checked before the first start | `package.json` |
| qagents | `pip install "qagents[devvi]"`: an empty extra, Devvi needs no Python dependency. `expose_tools` needs `pip install "qagents[mcp]"` too | |
| A model route for Devvi | for example `DEEPSEEK_API_KEY` for Devvi's `deepseek-official` route, or a route you add with a patch | |

## Set up and check

On Windows PowerShell. Build Devvi once from a checkout, and again after you pull Devvi changes:

```powershell
cd C:\src\devvi-harness                # your Devvi checkout
pnpm install
pnpm run build
node apps\cli\lib\bin.js --version     # expect 0.2.1-rc.3
```

Point qagents at it and check the setup. The doctor makes no model call:

```powershell
cd C:\src\qagents                      # your qagents checkout
pip install -e ".[dev]"
$env:QAGENTS_DEVVI_BIN = "C:\src\devvi-harness\apps\cli\lib\bin.js"
$env:QAGENTS_DEVVI_HOME = "$env:LOCALAPPDATA\qagents\devvi-home"
python -m qagents.integrations.devvi doctor
```

The doctor prints one check per line, `ok`, `warn` or `fail`: the Devvi home and where it came
from, inherited `DEVVI_HOME` or `DSH_HOME` values that the launcher overrides, which
pass-through variables are set (names only, never values), the resolved CLI command, the Node
and Devvi versions, the patch file, whether `--dump-config` shows the patch rows in the `acp`
and `sdk` profiles, then the two handshakes, each followed by a clean shutdown. The ACP lines
of a real run (Devvi 0.2.1-rc.3 on Windows, `DEEPSEEK_API_KEY` set):

```text
[  ok] ACP: deepseek-harness-acp 0.0.1 (ACP v1): MCP over HTTP yes, image prompts no, session resume yes; ready in 2.5 s; shut down with exit code 0
[  ok] ACP models: current deepseek-official/deepseek-v4-flash; offered: deepseek-official/deepseek-v4-flash, deepseek-official/deepseek-flash, deepseek-official/deepseek-v4-pro; reasoning efforts: off, low, high, max
```

The time to ready varies: about 2.5 s once Node's caches are warm, about 11 s on a cold first
start. Without `DEEPSEEK_API_KEY` (or your route's key) the session offers only Devvi's default
model, so `offered:` lists `deepseek-official/deepseek-v4-flash` alone.

The ACP check opens one empty session in `<home>\qagents\doctor` (or `--workspace DIR`) to read
the model options; Devvi keeps that empty session in its home. Options: `--bin`, `--node`,
`--home`, `--workspace`, `--provider P --model M` (the SDK handshake's route, and a model the
ACP session must offer), `--no-acp`, `--no-sdk`, `--allow-untested-version`, `--json`. The exit
code is 1 when a check failed.

On Linux or macOS, `export` replaces `$env:`, and `QAGENTS_DEVVI_BIN` may simply be `devvi`
when the CLI is on `PATH`.

## A lead with a coder

```python
from qagents import Agent, Limits, Permissions
from qagents.integrations.devvi import DevviAgent

coder = DevviAgent(
    workspace="C:/work/repo",
    name="coder",
    permission_mode="workspace-write",  # Devvi's sandbox: writes inside the workspace only
    patches=[{"id": "tool-web", "disabled": True}],  # no web_fetch or web_search for Devvi
    max_external_turns=40,  # tool calls per delegation before qagents stops the turn
    allow_unmetered=True,  # ACP reports no tokens: the lead's cost cap counts the lead only
)
lead = Agent(
    "qcloud:<model-id>",
    instructions="Plan the fix, delegate the edits to `coder`, then review the changed files.",
    tools=[coder.as_tool(name="coder", timeout=1800)],
    permissions=Permissions(
        ask=["coder.bash", "coder.pwsh"],  # Devvi asks only to leave its sandbox
        deny=["coder.web_fetch", "coder.web_search"],  # defense in depth: Devvi does not ask
        on_ask=my_reviewer,  # optional: without it, "ask" is refused
    ),
    limits=Limits(max_turns=8, max_cost_usd=0.50, timeout_s=3600),
)

async with coder:
    result = await lead.run("Make tests/test_parser.py pass.")
```

With Devvi's default profile, reads, edits and ordinary commands inside the workspace run
without a permission request, and so do web fetches. Only a step that asks to leave the sandbox
reaches your rules ([Permissions](#permissions)). The patch row removes Devvi's web tools; the
deny rule is there only in case a future Devvi or another profile asks.

`async with coder:` starts Devvi in the workspace and runs the handshake; leaving it stops the
process tree. Without it, Devvi starts on the first call; close it with `await coder.aclose()`.

`examples/devvi_coder.py` is a setup like this one, end to end: the lead plans, delegates,
reads the changed file and answers. `--dry-run` runs it offline with a scripted lead and a fake
Devvi (Devvi's recorded handshake, then a turn shaped like Devvi's recorded escalations: a read
and an edit that run without asking, an escalated test run that the reviewer allows, and an
escalated `pip install` that the reviewer refuses). No deny rule decides anything in it; on
Windows it prints `coder.pwsh: allow by on_ask` and `coder.pwsh: deny by on_ask` (`coder.bash`
elsewhere):

```powershell
python examples/devvi_coder.py --workspace C:\tmp\devvi-sandbox --dry-run
$env:DEEPSEEK_API_KEY = "<your key>"           # Devvi's model route; never commit it
python examples/devvi_coder.py --workspace C:\tmp\devvi-sandbox --lead qcloud:<model-id>
```

Use a disposable workspace: Devvi edits it.

## Recipes

### Run it directly and stream

```python
from qagents.integrations.devvi import DevviAgent

async with DevviAgent(workspace="C:/work/repo") as coder:
    async with coder.stream("Add type hints to utils.py") as turn:
        async for event in turn:
            if event.type == "text_delta":
                print(event.text)  # a whole committed message, not a token
            elif event.type == "tool_call_started":
                print("tool:", event.call.name, event.call.arguments)
    print(turn.result.status, turn.result.changed_files)
```

A direct run has no calling run: its permission requests are decided by `stream(permissions=...)`
or the agent's `permissions=`, and with neither, every request is refused.

### As a workflow step

```python
from qagents import Workflow
from qagents.external import ExternalResult
from qagents.integrations.devvi import DevviAgent

coder = DevviAgent(workspace="C:/work/repo", allow_unmetered=True)

wf = Workflow(store, id=f"fix:{issue}")
plan = await wf.run_agent("plan", planner, f"Plan a fix for {issue}")
code = await wf.step("code", lambda: coder.run(plan.text), type=ExternalResult)
print(code.status, code.session_id, code.changed_files)
```

`wf.step` records the `ExternalResult`, so a finished step never runs again for this workflow
id. A `wf.step` that was running when the process died runs again from its start: Devvi gets the
task again, in a new session. (Crash re-attach happens only for calls through `as_tool()`.)

### Gate the delegation, not each command

Ask a person once per task, and let Devvi work without asking inside a throwaway copy:

```python
from qagents import Agent
from qagents.integrations.devvi import DevviAgent

coder = DevviAgent(
    workspace="C:/tmp/clone-of-repo",  # a disposable clone or a container mount
    permission_mode="danger-full-access",  # Devvi asks nothing in this mode
    allow_unmetered=True,
)
lead = Agent(
    "qcloud:<model-id>",
    tools=[coder.as_tool(name="coder", requires_approval=True, timeout=1800)],
)
```

The lead's run pauses before each delegation (`result.pending_approvals`), as for any tool that
requires approval. This is honest about where the boundary is: the clone, not Devvi's sandbox.
Keep `timeout=`: without it the lead's `Limits.tool_timeout_s` (120 s by default) cuts each
delegation at about 110 s.

### Pause for a human on each request

`DevviAgent(..., on_permission="pause")` ends the turn on a request that needs a person and
pauses the lead's run; answer with `ExternalApproval`. See
[Pause for a human](../concepts/external-agents.md#pause-for-a-human).

The approval answers the same request once, when the lead calls the coder again. Devvi's
model writes the `description` of a shell call, and the `justification` of an escalation, anew
on every attempt, so `DevviAgent` leaves those two out of the match
(`DevviAgent.grant_ignored_arguments`). The `command`, the `path`, the `content`,
`sandbox_permissions` and every other argument must be the same; another command is asked
again (`test_a_pause_grant_survives_reworded_free_text_but_not_another_command`).

### Choose Devvi's model

```python
from qagents.integrations.devvi import DevviAgent

coder = DevviAgent(
    workspace="C:/work/repo",
    model=("deepseek-official", "deepseek-v4-pro"),  # (provider route, model id)
    reasoning_effort="high",
    allow_unmetered=True,
)
```

On ACP the model is matched against the session's model options (the doctor's `ACP models`
line lists them) and set with `session/set_config_option`; a model the session does not offer
is refused with the list it offers. `model=None` keeps Devvi's current model.
`reasoning_effort` sets Devvi's `reasoning_effort` option the same way.

### Devvi on QCloud

Devvi can run on any OpenAI-compatible endpoint through its `llm-pi-ai` routes, which stay
dormant until a profile supplies providers (`packages/bundle/base/cordis.patch.yml`).
`model_route_row` writes that row; the key is referenced by its variable name, never written:

```python
from qagents.integrations.devvi import DevviAgent, model_route_row

route = model_route_row(
    "qcloud",
    base_url="<QCloud OpenAI-compatible base URL>",
    api_key_env="QCLOUD_API_KEY",  # the NAME of the variable, not the key
    models=[{"id": "<model-id>", "contextWindow": 131072}],
)
coder = DevviAgent(
    workspace="C:/work/repo",
    model=("qcloud", "<model-id>"),
    patches=[route],
    env_passthrough=["QCLOUD_API_KEY"],  # replaces the default list: add DEEPSEEK_* if needed
    allow_unmetered=True,
)
```

Not verified against QCloud: the base URL, tool calling and reasoning compatibility, and
whether the route shows in the ACP session's model options. Check with the doctor
(`--provider qcloud --model <model-id>`).

### Token usage and cost

ACP reports no token counts. The SDK transport does, per model request, so cost budgets work
when you give prices:

```python
from qagents import Agent, Limits
from qagents.integrations.devvi import DevviAgent, TokenPricing

coder = DevviAgent(
    workspace="C:/work/repo",
    transport="sdk",
    model=("deepseek-official", "deepseek-v4-flash"),
    pricing=TokenPricing(input=0.27, output=1.10, cache_read=0.07),  # USD per million tokens
)
lead = Agent(
    "qcloud:<model-id>",
    tools=[coder.as_tool(name="coder", timeout=1800)],
    limits=Limits(max_cost_usd=2.0),  # stops the coder too, as its usage arrives
)
```

The prices are yours to supply: qagents has no price table for Devvi's models, and the devvi.sh
platform may add its own margin. Devvi reports disjoint counts: `inputTokens` is uncached input
only, and cache reads and writes are counted apart (`packages/llm/llm/src/types.ts`).
qagents counts all three as `input_tokens`, the cache reads also as `cached_input_tokens`, and
prices each part. Without `pricing=`, a lead with `max_cost_usd` refuses the coder at the first
call (the cost budget could not be enforced).

Devvi's context compactions are model requests too: their usage counts against the budgets as
soon as Devvi reports it (a compaction without usage counts as one request whose cost is
unknown). Devvi's DeepSeek web search makes a model request whose usage Devvi does not record:
qagents counts it as one request of unknown cost, so under `Limits.max_cost_usd` the first web
search stops the turn (`limit_reached`, `max_cost_usd`). Disable Devvi's web tools for a coder
with a cost budget: `patches=[{"id": "tool-web", "disabled": True}]`.

### qagents tools inside Devvi

Devvi is an MCP client: it mounts the MCP servers a session names and shows their tools to its
model as `mcp__<server>__<tool>` (`packages/acp/acp/src/mcp.ts`). `expose_tools`
uses that to give Devvi qagents tools. It needs `pip install "qagents[mcp]"`.

```python
from qagents import Agent, Limits, Permissions
from qagents.integrations.devvi import DevviAgent

coder = DevviAgent(
    workspace="C:/work/repo",
    expose_tools=[lookup_customer, create_ticket],  # in-process qagents tools
    allow_unmetered=True,
)
lead = Agent(
    "qcloud:<model-id>",
    tools=[coder.as_tool(name="coder", timeout=1800)],
    permissions=Permissions(
        allow=["coder.mcp__qagents__*", "lookup_customer"],  # the server decides each call again
        ask=["create_ticket"],
        on_ask=my_reviewer,
    ),
    limits=Limits(tool_timeout_s=50),  # served calls: Devvi gives an MCP call 60 seconds
)  # the coder tool keeps its own timeout=1800: a tool's own timeout wins
```

| Entry in `expose_tools` | Served | Devvi sees | Who decides each call | Journal |
|---|---|---|---|---|
| `Tool` objects, functions, toolkits, agents (in process) | MCP Streamable HTTP in qagents' own process, on `127.0.0.1`, random port and bearer token, one server per turn | `mcp__qagents__<tool>` (`expose_name=`) | through `as_tool()`: the lead's `Permissions` and `on_ask` (bounded by 50 s); in a direct run: the turn's rules | the lead's session, as ordinary tool calls (`source: "mcp"`) |
| `"package.module:attr"` strings | stdio: Devvi starts `python -I -m qagents.integrations.mcp_server --spec FILE` | `mcp__qagents_stdio__<tool>` | the spec's rules; tools declared `requires_approval` are refused | none |
| a `ToolServerSpec` | stdio, as given | `mcp__<spec name>__<tool>` | its rules | none |

The qagents server decides every call itself, with the lead's rules, on the tool's own name
(`lookup_customer`), and journals it in the lead's session. Devvi asked no permission for MCP
calls in tier 1 (`test_acp_exposed_qagents_tools_are_called_by_devvis_model_over_mcp`). Should a
Devvi profile ask, the request reaches the bridge as `coder.mcp__qagents__<tool>` and, with no
rule, is refused. So `allow=["coder.mcp__qagents__*"]` at the bridge is safe; put the per-tool
rules on the qagents names.

Facts that shape this, from Devvi's code:

- Devvi mounts MCP servers only when a session opens (`packages/acp/acp/src/mcp.ts`).
  After a turn that mounted an in-process server, qagents closes the session (`session/close`);
  the next turn resumes it with a fresh server. Tier 1 runs two such turns of one session.
  Other sessions are closed too once more than one is idle (`keep_sessions=1` of
  [`ACPAgent`](acp.md#what-the-client-does)), so a long-lived Devvi process does not keep an
  agent and its stdio MCP servers for every delegation; continuing one resumes it.
- A stdio server's command must be an absolute path, and its child gets Devvi's environment
  without names matching `KEY|PASSWORD|SECRET|TOKEN` or starting with `DSH_`
  (`packages/subprocess/subprocess/src/index.ts`). Tools that need a credential
  must load it another way: a `deps` factory that reads a file or a credential store
  ([an example](mcp-server.md#over-stdio-in-a-separate-process)).
- Devvi gives an MCP tool call 60 seconds (`packages/mcp/mcp-client/src/index.ts`).
  qagents bounds `on_ask` for exposed tools to 50 seconds; keep `Limits.tool_timeout_s` under a
  minute.
- In-process tools need Devvi's `mcpCapabilities.http`, which Devvi 0.2 advertises
  (`snapshots/acp/handshake/stdout.expected.jsonl`).
- The SDK transport mounts MCP servers through a patch file, so it refuses in-process tools when
  the agent is built (the bearer token would be written to disk); importable tools work there.

The server itself is described in [Serving tools over MCP](mcp-server.md).

## Two transports

| | `transport="acp"` (default) | `transport="sdk"` |
|---|---|---|
| Devvi profile | `devvi --profile acp` | `devvi --profile sdk` |
| Protocol | ACP v1, the generic `ACPAgent` client | Devvi's SDK JSON-RPC (`packages/sdk/protocol/src/types.ts`) |
| Processes | one per workspace, kept between calls | one per call by default (`process_per_call=True`), or one per workspace |
| Token usage and cost | none; `allow_unmetered=True` under a token or cost budget | per model request; cost with `pricing=` |
| Cancel | `session/cancel`, then the process tree is killed after `cancel_grace_s` | no per-turn cancel in the protocol: the process is shut down |
| Session ids | chosen by Devvi | chosen by qagents (`qa-<sha256 of the call's operation id>`) |
| Crash re-attach | `session/resume` of the journaled session (`resumable=True`, default) | `session/resume`, then Devvi's session log shows whether the turn finished |
| Model | chosen from the session's options; `None` keeps Devvi's | sent in `initialize`; one route per process; default `("deepseek-official", "deepseek-v4-flash")` |
| `pricing=`, `max_tokens=` | refused when the agent is built | supported |
| `expose_tools` | in-process (HTTP) and importable (stdio) | importable (stdio) only |
| Steps counted by `max_external_turns` | tool calls | model requests (`step/start`) |
| `ExternalResult.backend` | `devvi-acp@<devvi --version>` | `devvi-sdk@<devvi --version>` |
| Devvi's own stop word | `raw["agent_stop_reason"]` (ACP `stopReason`) | `raw["agent_stop_reason"]` (`turn/end` `reason.kind`) |

Both transports report the same `stop_reason` for the same outcome: `end_turn` (status
`completed`), `max_tokens` (status `limit_reached`) and `cancelled`. So
`result.stop_reason == "end_turn"` works on both. Devvi's SDK words `completed`, `max-tokens`
and `aborted` stay in `ExternalResult.raw["agent_stop_reason"]`; the full mapping is
`TURN_END_KINDS` in `qagents.integrations.devvi.events`.

On the SDK transport with `process_per_call=False`, each pooled process is looked after on its
own: one that died is replaced on the next call in its workspace, and a turn that breaks or is
abandoned stops only its own process. Turns in other workspaces keep running.

Devvi's wire reports a fixed version `0.0.1` (`agentInfo` in
`snapshots/acp/handshake/stdout.expected.jsonl`), so qagents runs
`devvi --version` before the first start and records that.

Devvi roots each session's file sandbox at that session's `cwd` from `session/new`: its write
tool, its sandboxed shell and the working directory it tells its model all follow it. The base
profile's `workspaceRoot: process.cwd()`
(`packages/bundle/base/cordis.patch.yml`) is only the default that a session's
`cwd` replaces. Measured in tier 1
(`test_acp_the_file_sandbox_follows_the_session_cwd_not_the_process_cwd`): with the process
started in `A` and the session in `B`, Devvi wrote in `B` and refused a write in `A`.

So one Devvi process could serve several workspaces. qagents still starts one ACP process per
workspace, in that workspace: each workspace gets its own process environment and model route,
and a crash or a forced kill after a cancel touches one workspace only. The next turn there
starts the process again and resumes its sessions; other workspaces are not touched.

The SDK transport, turn by turn: `session/resume` (a new id is learned as "not found"), the
session id is journaled, `session/prompt` returns a message id (journaled too), the turn ends
when Devvi reports the root session idle after receiving that message, then `session/wait` waits
for Devvi's subagents (`wait_for_subagents=True`). A call repeated after a crash reads Devvi's
session log: a finished turn is replayed from the log and not prompted again; an open or
interrupted one gets `EXTERNAL_AGENT_RESUME`; a prompt the log shows no trace of is sent as
new. Without the prompt's message id, a prompt is matched by its text only when Devvi received
it after the call started (`devvi_prompt_baseline`, journaled before the session id), so an
earlier call's prompt with the same text (for example "Continue") is never taken for this
call's. A continued session without that record gets `DEVVI_RESUME_UNKNOWN`. When Devvi no
longer has the session at all (its log was removed, Devvi now uses another
home or sessions root, or it died before its first write), the earlier attempt may still have
sent its prompt: the call ends `interrupted` with an unknown outcome (`DEVVI_NOT_REATTACHED`),
and the task is not sent again, as on ACP (`ACP_NOT_REATTACHED`). The lead's model is told to
check the workspace and to call again without a session.
Devvi's `user-question/ask` is not advertised, so Devvi answers its own question tool itself.

On the SDK transport a session continues only in its own workspace. Devvi's `session/resume`
restores the session's stored directory, and every `DevviAgent` with the same home sees the
same sessions, so qagents checks the session's directory first. A session of another
workspace is refused before Devvi opens it (`DEVVI_OTHER_WORKSPACE`); continue it with the
agent of that workspace. When qagents cannot read the session's log, it asks Devvi after the
resume (`session/working-directory/get`), and refuses the turn when the answer is another
workspace or there is none (`DEVVI_WORKSPACE_UNKNOWN`). Measured in tier 1
(`test_sdk_a_session_of_another_workspace_is_refused`).

## Permissions

Devvi's permission requests reach your `Permissions` as `<tool name>.<Devvi tool>`. But Devvi
sends one only for an **escalation**: a step that asks to leave its sandbox. Everything else
runs without a request, so no qagents rule is consulted for it. With Devvi's default profile:

| Name | Devvi asks |
|---|---|
| `coder.pwsh` (the shell on Windows), `coder.bash` (Linux and macOS) | only for a command sent with `sandbox_permissions` and a `justification`, for example to write outside the workspace (`packages/shell/tool-pwsh/src/index.ts`). Ordinary commands, the test run included, run without asking |
| `coder.write`, `coder.edit` | only for a file change sent with `sandbox_permissions`, for example after the sandbox refused a write outside the workspace (`packages/fs/tool-fs/src`) |
| `coder.read`, `coder.grep`, `coder.glob` | never (`packages/fs/tool-fs-search/src`) |
| `coder.web_fetch`, `coder.web_search` | never: these tools have no approval step (`packages/web/tool-web/src`) |
| `coder.todo_write` | never |
| `coder.mcp__<server>__<tool>` | never seen: Devvi asked nothing for MCP calls in tier 1 |

Tier 1 measured this: under `workspace-write`, an ordinary command and a web fetch ran with no
request, even with a deny rule on `coder.web_fetch`, and an escalated command was decided by
qagents' rules (`test_a_web_fetch_runs_without_a_permission_request`,
`test_acp_escalation_is_decided_by_qagents_rules`). Devvi's tool pipeline allows a call unless a
gate says "ask" (`packages/core/tools/src/pipeline.ts`), and in the base
profile only the escalation path asks.

So rules on `coder.read`, `coder.grep`, `coder.glob`, `coder.todo_write`, `coder.web_fetch` and
`coder.web_search` are never consulted. Keep a deny rule on them only as defense in depth, in
case a future Devvi or another profile asks. To take a tool away, remove it from Devvi's
profile with a patch row:

```python
from qagents.integrations.devvi import DevviAgent

NO_WEB = {"id": "tool-web", "disabled": True}  # Devvi's model is offered no web_fetch or web_search

coder = DevviAgent(workspace="C:/work/repo", patches=[NO_WEB], allow_unmetered=True)
```

Tier 1 checks that this row removes both tools
(`test_a_disabled_tool_web_row_removes_devvis_web_tools`). It does not take Devvi off the
network: a shell command can still reach it (`curl`, `Invoke-WebRequest`). For no network at
all, run Devvi in a container or a VM without one.

Write rules for both `coder.bash` and `coder.pwsh` if your code runs on both systems: Devvi's
base profile mounts `tool-pwsh` instead of `tool-bash` on Windows
(`packages/bundle/base/cordis.patch.yml`).

`permission_mode=` sets `DSH_PERMISSION_MODE`, which picks Devvi's preset
(`packages/bundle/base/cordis.patch.yml`):

| `permission_mode` | Devvi's sandbox | Devvi asks |
|---|---|---|
| `"read-only"` | read-only | for an escalation |
| `"workspace-write"` (default) | writes inside the workspace | for an escalation: a command or file change that asks to leave the sandbox |
| `"danger-full-access"` | none | never: the approval policy is `never`. Use it only in a disposable workspace |

On the SDK transport a request is decided on the arguments of the requesting session's own
call: call ids repeat across Devvi's subagent tree. A request about a call that session never
started is answered `cancelled` without consulting your rules, and the turn emits the
`external_event` `approval_refused` (`{toolName, callId, reason: "unknown_call"}`, with
`session_id` the requesting session).

What qagents answers, the three postures (`on_permission="rules"`, `"pause"`, `"live"`) and the
fail-closed rules are in [External agents](../concepts/external-agents.md#permission-requests).

**Devvi's sandbox is not a security boundary on Windows.** It governs file effects only; network
and process policy are outside it (`packages/sandbox/sandbox-policy/README.md`).
On Windows its enforcement is `partial`: reads stay unconfined and NTFS hard links can alias files
outside the workspace (`packages/sandbox/sandbox-local/README.md`).

## Configuration

`DevviAgent(*, workspace, ...)`: every option is a keyword.

| Option | Default | Meaning |
|---|---|---|
| `workspace` | required | the directory Devvi works in: its session `cwd` (the root of its file sandbox) and its process `cwd` |
| `transport` | `"acp"` | `"acp"` or `"sdk"` |
| `name` | `"devvi"` | the agent's name; the permission prefix of a direct run (`as_tool(name=...)` sets the tool's) |
| `description` | Devvi's default | the tool description the lead's model reads |
| `model` | `None` (ACP), `("deepseek-official", "deepseek-v4-flash")` (SDK) | `(provider route, model id)` |
| `reasoning_effort` | `None` | Devvi's reasoning effort for the route |
| `max_tokens` | `None` | output-token cap per model request (SDK only) |
| `permission_mode` | `"workspace-write"` | `DSH_PERMISSION_MODE` |
| `expose_tools`, `expose_name` | `()`, `"qagents"` | qagents tools for Devvi's model |
| `resumable` | `True` | ACP: a call repeated after a crash continues its journaled session; the SDK transport always does |
| `home` | `QAGENTS_DEVVI_HOME`, else `~/.qagents/devvi-home` | Devvi's home for the processes qagents starts |
| `devvi_bin` | `QAGENTS_DEVVI_BIN`, else `devvi` on `PATH` | the CLI; a `.js` file runs with Node |
| `node` | `QAGENTS_NODE`, else `node` on `PATH` | Node, or a wrapper command such as `["volta", "run", "node"]` |
| `command` | `None` | an exact command that starts the CLI, instead of `devvi_bin` and `node` (no Node check) |
| `env` | `{}` | variables set for Devvi explicitly, kept in memory |
| `env_passthrough` | `("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL")` | names copied from this process's environment; the list replaces the default |
| `patches` | `()` | extra profile patch rows (dicts, written after the defaults) or patch files (paths, passed after them) |
| `share_session_log` | `False` | let Devvi upload its session log with DeepSeek API requests |
| `pricing` | `None` | `TokenPricing` for `cost_usd` (SDK only) |
| `process_per_call` | `True` | SDK: one process per call, or one per workspace kept between calls |
| `wait_for_subagents` | `True` | SDK: after the turn, wait for Devvi's subagents |
| `raw_events` | `False` | forward Devvi events that have no qagents counterpart |
| `initialize_timeout_s` | 120 | seconds for Devvi to load and answer `initialize` |
| `shutdown_timeout_s` | 5 | seconds a finished call (SDK) or a closing agent (ACP) gives Devvi to exit |
| `request_timeout_s` | 120 | ACP: seconds for `session/new`, `session/resume`, `session/set_config_option` |
| `allow_untested_version` | `False` | run a Devvi version other than 0.2.x |

The `ExternalAgent` options work too: `permissions`, `on_permission`, `approval_timeout_s`,
`cancel_grace_s` (5), `max_external_turns` (50), `allow_unmetered`, `track_changes` (git, by
default when the workspace is a repository).

Environment variables:

| Variable | Read by | Meaning |
|---|---|---|
| `QAGENTS_DEVVI_BIN` | the qagents launcher | path of the Devvi CLI. A `.js` file runs as `[node, path]` (required on Windows). On Windows a `devvi.cmd` shim written by npm or pnpm is never run: the script it wraps is found in it and run with Node |
| `QAGENTS_NODE` | the qagents launcher | the Node executable; must be at least 24.9.0 |
| `QAGENTS_DEVVI_HOME` | the qagents launcher | Devvi's home for qagents' processes; not `~/.devvi`, so they stay apart from your own Devvi sessions |
| `DEVVI_HOME`, `DSH_HOME` | Devvi | both set to the home above. Devvi lets `DEVVI_HOME` win over `DSH_HOME` (`packages/util/home-paths/src/index.ts`), so an inherited `DEVVI_HOME` would otherwise move the home silently; a different inherited value is overridden and logged |
| `DSH_PERMISSION_MODE` | Devvi | from `permission_mode=` |
| `DSH_TELEMETRY_DISABLED=1` | Devvi | always set |
| `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL` | Devvi's `deepseek-official` route | passed through by default when set |
| `PATH`, `SYSTEMROOT`, `TEMP`, `TMP`, `USERPROFILE`, `HOME`, `LANG`; on Windows also `PATHEXT`, `COMSPEC`, `WINDIR`, `SYSTEMDRIVE` | the operating system | always passed, so Node, git and the shells work |

Nothing else of qagents' environment reaches Devvi: no other `*_KEY` or token, unless you list
its name. `DEVVI_HOME`, `DSH_HOME` and `DSH_PERMISSION_MODE` are refused in `env=` and
`env_passthrough=`; use `home=` and `permission_mode=`.

`devvi` and `node` are looked up on `PATH` only in its absolute entries, never in the current
directory (which Windows would search first, and qagents often runs in the workspace, where the
coder can write a `node.exe`), and never in an entry inside the agent's workspace. A bare name
in `QAGENTS_DEVVI_BIN` or `QAGENTS_NODE` is looked up the same way; give an absolute path
otherwise.

## What qagents writes, and where

What qagents writes goes under the Devvi home (`QAGENTS_DEVVI_HOME`, else
`~/.qagents/devvi-home`):

| Path | What |
|---|---|
| `qagents\patches\qagents-<hash>.patch.yml` | the profile patch of a launch: JSON, which YAML 1.2 accepts; named by its content and written atomically, so a file a running Devvi reads is never rewritten |
| `sessions\...\session.v4.jsonl` | Devvi's own session logs, uncompressed (qagents reads them, never writes) |
| `qagents\tool-servers\<name>-<hash>.json` | specs of the stdio tool servers of `expose_tools`: references, rules and paths, no secrets |
| `qagents\doctor` | the doctor's workspace and its empty session |

Devvi itself also writes outside the home: temporary files in the `TEMP` (or `TMP`) folder it
inherits from qagents' process, such as `dsh-spill-*`, `dsh-acl-locks\`, `dsh-acl-skill-*` and
`dsh-subprocess-*` (seen in tier 1). To keep them apart, set `TEMP` and `TMP` of the process that
runs qagents to a folder of your choice; the launcher passes both on.

The default patch has three rows:

```json
[
  {"id": "session-persistence-jsonl", "config": {"root": "<home>/sessions", "compression": "none"}},
  {"id": "session-log-deepseek", "config": {"enabled": false}},
  {"id": "session-telemetry-otel", "disabled": true}
]
```

`patches=[...]` rows are appended after them; a row replaces the whole config of the row it
names. Rows that hold something that looks like a credential are refused, with the place named
and the value never shown: patch files are written to disk. Reference a key by its variable
name (`apiKeyEnv`) and list that name in `env_passthrough=`.

`qagents.integrations.devvi.logs` reads Devvi's session logs, read-only. `find_session_log`
takes the sessions directory (`<home>/sessions`, which `sessions_dir(home)` gives), not the home,
and returns `None` when there is no log for that session:

```python
from qagents.integrations.devvi.logs import find_session_log, read_session_log, sessions_dir

path = find_session_log(sessions_dir(home), session_id)  # None: no log for this session
if path is not None:
    log = read_session_log(path)
    for turn in log.turns():
        print(turn.number, turn.end_kind, turn.final_text)
```

The module also has `list_sessions` and a Markdown export (`to_markdown`). A compressed
(`.zstd`) log cannot be read.

## Privacy defaults

- **No session-log upload.** Devvi's `session-log-deepseek` row is on by default in Devvi
  (`packages/session/session-log-deepseek/src/index.ts`) and sends the
  session log with official DeepSeek API requests. qagents turns it off;
  `share_session_log=True` turns it back on.
- **No telemetry.** `DSH_TELEMETRY_DISABLED=1`, and the `session-telemetry-otel` row is
  disabled.
- **An explicit environment.** Only the variables listed above reach Devvi.
- **No credential in a file.** Patch rows and tool-server specs are checked.
- **Keep credentials out of the workspace's `.env`.** Devvi itself loads a `.env` file from its
  working directory and from its home, without overriding variables that are already set
  (`loadLayeredEnv` in `packages/boot/app-boot/src/index.ts`). qagents cannot stop
  that.

## Windows notes

- Run qagents on the default Proactor event loop. A Selector loop cannot start subprocesses and
  gives `this event loop cannot run subprocesses`.
- `.js` CLIs run as `[node, bin.js]`. A `.cmd` or `.bat` shim is never started (Windows would run
  it through `cmd.exe`); `QAGENTS_NODE` must name the real `node.exe`.
- Devvi starts shells, MCP servers and subagents. qagents runs the Devvi process in a Job Object
  that kills the whole tree when the job closes, so a cancel or `aclose()` leaves no orphans.
  If Windows refuses the job, the tree is found from a process snapshot by parent id, checked
  by start time so that a reused process id is never mistaken for a child. Killing starts no
  other program. The job handle also closes when qagents' own process ends.
- The shell tool is `pwsh`, not `bash`: write rules for `coder.pwsh`. Without PowerShell 7 it
  falls back to Windows PowerShell 5.1.
- Devvi's sandbox enforcement is partial on Windows (see [Permissions](#permissions)).
- Under `workspace-write`, Devvi runs shell commands with a restricted token. On the development
  machine that process could not read any workspace under `%LOCALAPPDATA%\Temp`, whose parent
  folders carry another sandboxing tool's ACLs (the likely cause: Devvi's `sandbox-windows-acl`
  README names such trees). Commands failed with access denied, and PowerShell ran in
  ConstrainedLanguage mode. Devvi's own `write` tool was not affected, and `danger-full-access`
  worked. Put the workspace outside such trees (for example under `C:\work`), or check its ACLs
  and its parents' with `icacls`.
- Devvi writes temporary files to the `TEMP` folder it inherits
  ([What qagents writes, and where](#what-qagents-writes-and-where)).
- Paths in `ExternalResult.changed_files` are relative to the repository root and use `/`.

## Troubleshooting

Setup errors are `ExternalAgentConfigurationError`. Through `as_tool()` the lead's model reads
`coder is unavailable: <first line>` and later calls in that run fail at once; the full message
(with the end of Devvi's stderr) is in the log and in `str(error)` of a direct run.

| Message, or what you see | Cause | Fix |
|---|---|---|
| `The Devvi CLI was not found.` | no `QAGENTS_DEVVI_BIN`, no `devvi` on `PATH` | set `QAGENTS_DEVVI_BIN` to `apps\cli\lib\bin.js` of a built checkout, or `lib\bin.js` of the installed `@deepseek-ai/dsh` package |
| `The Devvi CLI was not found at '...' (from QAGENTS_DEVVI_BIN).` | the path does not exist | check the path; run `pnpm run build` in the checkout |
| `Node.js was not found, so the Devvi CLI cannot start.` | no `node` on `PATH` | install Node 24.9 or newer, or set `QAGENTS_NODE` |
| `Node 22.17.0 is too old for Devvi, which needs >= 24.9.0.` | an old Node | install a newer Node, or set `QAGENTS_NODE` to one |
| `Node at '...' is a .cmd shim` / `... is a .cmd shim that Windows would run through cmd.exe` | a shim instead of a program (Windows) | set `QAGENTS_NODE` to `node.exe`; set `QAGENTS_DEVVI_BIN` to the package's `lib\bin.js` |
| `The Devvi CLI did not start: devvi --version exited with code 1` | a broken or unbuilt CLI | run `node <bin.js> --version` yourself; rebuild Devvi |
| `devvi --version printed '...', which is not a version.` | `QAGENTS_DEVVI_BIN` names another program; with `printed ''`, a Node older than 24.2 (Devvi's CLI then prints nothing; `command=` skips qagents' own Node check) | point it at the Devvi CLI; use Node 24.9 or newer |
| `Devvi 0.3.0 is not a version qagents was tested with (0.2.x).` | a newer Devvi; the SDK protocol has no version negotiation | `allow_untested_version=True` to try it |
| `DEVVI_HOME is set by qagents: use home= ...` | a launcher-owned name in `env=` or `env_passthrough=` | use `home=` or `permission_mode=` |
| warning `The inherited DEVVI_HOME is overridden: qagents runs Devvi in ...` | your shell sets `DEVVI_HOME` or `DSH_HOME` | nothing to do; set `QAGENTS_DEVVI_HOME` to choose the home |
| `A Devvi patch would hold a credential in clear text (at rows[3]...)` | a key or token in a patch row | use `apiKeyEnv` and `env_passthrough=` |
| `refused the ACP handshake`, `did not answer the ACP handshake (initialize) within 120 seconds`, `stopped during the ACP handshake (the process exited with code 1)` | Devvi failed to load: a bad patch, an unknown route, a plugin error | run the doctor; read the stderr in the message; raise `initialize_timeout_s` on a slow first start |
| `speaks ACP protocol version 2; qagents speaks version 1` | an agent release without ACP v1 | use a Devvi 0.2.x build |
| `model ('deepseek-official', 'deepseek-v9') is not offered by the session (it offers: ...)` | a model id Devvi does not list, or `DEEPSEEK_API_KEY` (your route's key) is not set: on ACP Devvi then offers only its default model, `deepseek-v4-flash` | set the key and keep its name in `env_passthrough`; pick a model from the doctor's `ACP models` line, or add the route with a patch |
| `the session offers no config option 'reasoning_effort'` | the route has no effort option | drop `reasoning_effort=` |
| `refused session/new: ...` | an MCP server failed to start (Devvi fails the session then), a relative `cwd` | check the server's command; the workspace must be an absolute, existing directory |
| `the agent did not answer session/new within 120 seconds` (the turn ends `error`: nothing of the task was sent) | MCP servers slow to connect | raise `request_timeout_s` |
| the model reads `coder failed: session qa-... works in another workspace, so it was not continued here` | SDK: the model passed the session of another workspace (`DEVVI_OTHER_WORKSPACE`) | continue it with the agent of that workspace, or leave `session` out |
| `expose_tools needs the MCP Python SDK` | `qagents[mcp]` is not installed | `pip install "qagents[mcp]"` |
| `expose_tools with in-process tools needs transport='acp'` | in-process tools on the SDK transport | use ACP, or expose importable tools (`"package.module:attr"`) |
| `pricing needs transport='sdk'`, `max_tokens needs transport='sdk'` | ACP has no token counts and no such option | use `transport="sdk"`, or declare the limit on the route with a patch |
| the model reads `coder was not started: this run has a token or cost budget, and coder does not report its token usage` | ACP under `max_cost_usd` or `max_total_tokens` | `allow_unmetered=True`, or `transport="sdk"` with `pricing=` |
| the model reads `coder is unavailable: it has no token prices, so the cost budget of this run cannot be enforced` | SDK under `max_cost_usd` without prices | `pricing=TokenPricing(...)` |
| the model reads `coder is unavailable: ... 401 ...`, `... MISSING_CREDENTIAL ...` | Devvi's model route has no valid key; this repeats, so the run stops calling the coder | set `DEEPSEEK_API_KEY` (or your route's variable) and keep it in `env_passthrough` |
| the model reads `coder stopped before finishing (timeout).` | the turn ran longer than `as_tool(timeout=...)`, or than the lead's `tool_timeout_s` (120 s by default) less `cancel_grace_s + 5` | give `as_tool()` a timeout that fits coding work, such as 1800 |
| the model reads `'coder' did not finish within 20 seconds and was cancelled. ...`, with no session | no `as_tool(timeout=...)`, and a lead `tool_timeout_s` of at most `2 × (cancel_grace_s + 5)` (20 s by default): the loop's own timeout ended the call, so the session was not reported | give `as_tool()` a timeout |
| the model reads `coder stopped before finishing (max_external_turns).` | more steps than `max_external_turns` | raise it, or split the task |
| the model reads `coder stopped unexpectedly (the process exited with code 1) ... Its outcome is unknown` | Devvi died during the turn; `result.raw` has `stderr_tail` and `returncode` | check the workspace; the next call starts a new process and can continue the session |
| the model reads `coder stopped unexpectedly (the earlier attempt of this call used session qa-..., which Devvi no longer has, so its outcome is unknown) before it had a session it can continue.` | SDK, after a crash of the lead: the repeated call's session log is gone (removed, another home), so the task is not sent again | check the workspace; the model calls again without a session |
| the model reads `coder failed: session qa-... does not exist; leave session out to start a new one` | SDK: the model passed a session Devvi does not have | the model calls again without `session` |
| the model reads `coder failed: cannot continue session ...` | ACP: Devvi refused to resume that session | start a new session |
| Devvi's shell commands fail with access denied (`PermissionDenied`, `UnauthorizedAccess`), PowerShell reports ConstrainedLanguage mode (Windows, `workspace-write`) | Devvi's restricted-token sandbox cannot read the workspace, often because a parent folder carries another sandboxing tool's ACLs (seen under `%LOCALAPPDATA%\Temp`) | move the workspace out of that tree, or check the ACLs with `icacls`; see [Windows notes](#windows-notes) |
| a deny rule on `coder.web_fetch` (or `coder.read`, ...) never refuses anything | Devvi asks only for escalations; these tools run without a request | remove the tool with a patch row, such as `{"id": "tool-web", "disabled": True}`; see [Permissions](#permissions) |
| a permission is refused with `no_approver` | "ask" with no `on_ask` under posture `"rules"` | add an allow rule, an `on_ask`, or `on_permission="pause"` |
| a permission is answered `cancelled` with `timeout` | `on_ask` took longer than `approval_timeout_s` (60 s) | raise `approval_timeout_s`, or use `on_permission="live"` (15 min) |

## Test your lead offline

To test your own lead without Devvi, a key or a network, put a `ScriptedExternalAgent` where
the `DevviAgent` goes. It plays scripted turns in memory; the permission bridge, the footer,
the events, the budgets and the journal are the real code
([Testing offline](../concepts/external-agents.md#testing-offline-with-scriptedexternalagent)).
Its defaults are not `DevviAgent`'s, so set the two that change what the lead's run does:

```python
from qagents import Agent, Approval, Limits, Permissions
from qagents.external import ExternalCapabilities, ScriptedExternalAgent, ToolStep
from qagents.models.testing import ScriptedModel, call

ESCALATED_TEST = {  # the shape of Devvi's escalation requests
    "command": "python -m pytest -q",
    "sandbox_permissions": "danger-full-access",
    "justification": "The tests write a cache outside the workspace.",
}


def scripted_devvi() -> ScriptedExternalAgent:
    """One Devvi turn, with what DevviAgent reports on ACP."""
    return ScriptedExternalAgent(
        [
            [
                ToolStep("read", {"path": "src/parser.py"}, result="def parse_line(..."),
                ToolStep("edit", {"path": "src/parser.py"}, result="edited"),  # not asked
                ToolStep("pwsh", ESCALATED_TEST, ask=True, result="1 passed"),  # asked
                "Fixed parse_line; the test passes.",
            ]
        ],
        name="coder",
        capabilities=ExternalCapabilities(),  # no token usage, as on ACP
        resumable_turns=True,  # as DevviAgent(resumable=True), the default
        allow_unmetered=True,  # the real coder needs it under a cost budget too
    )


def allow_test_runs(ctx, tool, request, reason) -> Approval:
    return Approval(approved=request.arguments.get("command", "").startswith("python -m pytest"))


def make_lead(model, coder) -> Agent:
    return Agent(
        model,
        tools=[coder.as_tool(name="coder", timeout=1800)],
        permissions=Permissions(ask=["coder.bash", "coder.pwsh"], on_ask=allow_test_runs),
        limits=Limits(max_cost_usd=0.50),
    )


async def test_the_lead_delegates_and_the_test_run_is_allowed():
    coder = scripted_devvi()
    model = ScriptedModel([call("coder", task="Fix parse_line, then run the tests."), "Done."])

    result = await make_lead(model, coder).run("Make tests/test_parser.py pass.")

    assert result.output == "Done."
    assert coder.prompts == ["Fix parse_line, then run the tests."]
    assert [(a.outcome, a.decided_by) for a in coder.answers] == [("allow", "on_ask")]
```

In your application, `make_lead` gets the real `DevviAgent(..., allow_unmetered=True)`; the
test gets `scripted_devvi()`. What the two settings mirror:

- **`capabilities=ExternalCapabilities()`**: no token usage, as `DevviAgent` on ACP. The
  default of `ScriptedExternalAgent` reports usage, and then a lead with
  `Limits(max_cost_usd=...)` passes the test while the real coder is refused at its first call
  (the model reads `coder was not started: this run has a token or cost budget, and coder does
  not report its token usage`). With this setting a coder built without `allow_unmetered=True`
  fails the test the same way. For `transport="sdk"` with `pricing=`, keep the default and add
  `Usage(...)` steps.
- **`resumable_turns=True`**: as `DevviAgent`, whose `resumable` is on by default. The tool is
  then idempotent, so a call that was running when the lead's process died is repeated on
  resume and re-attaches to its session. With the default (`False`) the lead's model is told
  the outcome is unknown instead.

Only steps with `ask=True` reach the lead's rules, as with Devvi, which asks only for
escalations ([Permissions](#permissions)). On Linux and macOS Devvi's shell tool is `bash`.

## Running the tests

On Windows PowerShell, from the qagents checkout:

```powershell
pytest tests/integrations -q                      # tier 0: offline, no Devvi, always
$env:QAGENTS_DEVVI_BIN = "C:\src\devvi-harness\apps\cli\lib\bin.js"
pytest -m devvi_mock tests/integrations -q        # tier 1: real Devvi, local mock model, opt-in
python examples/devvi_coder.py --workspace C:\tmp\devvi-sandbox --dry-run
python examples/devvi_coder.py --workspace C:\tmp\devvi-sandbox --lead qcloud:<model-id>
```

Tier 0 replays Devvi's own recorded transcripts (copied into `tests/fixtures/devvi/`, with
Devvi's MIT notice) through fake peers that the tests start as real processes. Tests marked
`devvi_mock` are skipped unless `-m` names the marker (`tests/integrations/conftest.py`).

Tier 1 (`tests/integrations/test_devvi_mock.py`, plus the `devvi_mock` tests in
`tests/integrations/test_integration_docs.py` that check this page's claims) runs the real
Devvi CLI against a local mock of the model API (`tests/integrations/mock_model_server.py`,
ported from Devvi's own `scripts/smoke-python-runtime.py`), so it needs no key and
costs nothing. It needs `QAGENTS_DEVVI_BIN` (the tests skip without it); `QAGENTS_NODE` and
`QAGENTS_DEVVI_TMP` (where each test's own Devvi home and workspace go) are optional. Every test
starts one or two Devvi processes, 3 to 10 seconds each on Windows. With a Node older than
24.9.0 the tests start Devvi through `tests/integrations/devvi_node_compat.mjs`, a module hook
for `import.meta.main`, and warn that they do.

A tier 2 against a paid model route is planned: the conftest registers a `devvi_live` marker,
but no test uses it yet.

## What is verified

| Area | How |
|---|---|
| ACP transport: handshake, sessions, model options, permissions, cancel, re-attach after a crash, one process per workspace, pause posture | offline tests against Devvi's recorded ACP transcripts and a fake `devvi` process |
| SDK transport: event mapping, usage and cost, approvals, cancel by shutdown, re-attach from the session log | offline tests against Devvi's recorded SDK snapshots and a fake Devvi home |
| Launcher: CLI and Node discovery, `.cmd` shims, versions, home, environment allowlist, patch checks | offline tests |
| `expose_tools` | offline tests with the official MCP client |
| The example | `--dry-run`, offline |
| The real Devvi CLI, both transports: the doctor's handshakes, a text turn, a shell tool that writes in the workspace, escalations decided by qagents rules, cancel during a tool, resuming a session from a second agent, SDK usage and cost, SDK crash recovery from the session log | tier 1 (`-m devvi_mock`) |
| What Devvi asks about: ordinary commands and web fetches run without a request under `workspace-write`; only escalations are asked; the `tool-web` patch row removes the web tools | tier 1: `test_acp_escalation_is_decided_by_qagents_rules`, `test_a_web_fetch_runs_without_a_permission_request`, `test_a_disabled_tool_web_row_removes_devvis_web_tools` |
| The file sandbox follows the session's `cwd`, not the process's | tier 1: `test_acp_the_file_sandbox_follows_the_session_cwd_not_the_process_cwd` |
| `expose_tools` with the real Devvi: its MCP client takes the bearer header and the `127.0.0.1` URL; Devvi's model calls the tool; Devvi asks no permission for it; two turns of one session, closed after each and resumed with a new server; the calls are journaled in the lead's session | tier 1: `test_acp_exposed_qagents_tools_are_called_by_devvis_model_over_mcp` |
| SDK: `session/prompt` for a stored session that was not resumed is refused (-32603 `session "<id>" already exists`): no new session, no fork | tier 1: `test_sdk_a_prompt_for_a_stored_session_that_was_not_resumed_is_refused` |
| SDK: a session of another workspace is refused before Devvi opens it | tier 1: `test_sdk_a_session_of_another_workspace_is_refused` |
| A missing `DEEPSEEK_API_KEY` fails closed on both transports, without a model call | tier 1: `test_a_missing_api_key_fails_closed` |

The tier 1 results above come from Devvi 0.2.1-rc.3 on Windows 11 with Node 22.17, started
through `tests/integrations/devvi_node_compat.mjs`. Re-run tier 1 on Node 24.9 or newer, where
the tests start Devvi without the hook, before you rely on them for your setup. On that machine
the two `workspace-write` shell tests were expected failures (xfail) because of the sandbox ACLs
in [Windows notes](#windows-notes).

Not covered by any test against a running Devvi: QCloud as Devvi's model route; a paid model
route (tier 2); Linux and macOS; Node 24.9 or newer without the compatibility hook.

## Not in this version

- **Authentication is never automatic.** qagents starts no login for Devvi. Give Devvi's model
  route its key through the environment (`env_passthrough`, by variable name). Headless model
  authentication for Devvi is planned for later.
- The devvi.sh platform's model endpoint as a qagents `Model` (a `DevviPlatformModel`).
- qagents as an ACP agent server, so that ACP clients could drive a qagents agent.
- Tests against a paid model route (tier 2): the `devvi_live` marker exists, no test uses it
  yet.
