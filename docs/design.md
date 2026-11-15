# Principles and guarantees

qagents is designed for agents whose work spans model requests, tool calls, human decisions,
and process restarts. This page describes the design and the guarantees available to applications.

## Principles

**The journal is the product; the loop is small.** Frameworks that froze something else first
(the conversation between agents, a graph runtime, one vendor's wire format) paid for it with
rewrites. The durable artefact here is the append-only session journal. Sessions, approvals,
resume, compaction and workflows are all views of it.

**Record first, act second.** A model response is written to the journal before any tool it
requested starts, and every tool start and result is written as it happens. After a failure the
journal says exactly how far the run got.

**Nothing hidden.** The framework sends your instructions, your messages and your tool
descriptions. The few strings it adds are in `qagents.prompts`. There are no default personas,
no hidden system prompts and no prompt templating language.

**Control lives in code, outside the model.** Permissions are deterministic rules evaluated in a
fixed order: the first rule that answers decides, `deny` is checked first and always wins, and
any failure while deciding refuses the call.
Hooks that raise stop the run. Published prompt-injection defences that rely on detection have
all been bypassed; rules and least privilege are what holds.

**Safe by default.** Runs have a turn limit. Tools that are not declared read-only run one at a
time. Tools with side effects ask first. Requests that may have been billed are not repeated.
Message content stays out of traces unless you ask for it.

**Fail early and say why.** A setting the provider would reject fails when the agent is built.
A model without the needed capability fails before the request is sent. Errors name what to do.

**Provider- and agent-neutral.** Models and agents sit behind thin adapters. Conversations are
stored as typed parts, not as one vendor's JSON, with opaque provider data kept for replay;
QCloud is the first-class provider, not the only one. Another agent program (Devvi, any ACP
agent) is an external agent behind one contract, `ExternalAgent`: qagents gives it a task,
decides its permission requests and keeps the journal, whichever agent it is. There is no
universal gateway dependency.

## Compared with what exists

For a dated, sourced comparison with LangGraph, CrewAI, Microsoft Agent Framework, the OpenAI
Agents SDK, Google ADK and Pydantic AI, including where they are stronger, see
[Comparison with other frameworks](comparison.md).

| Documented gap elsewhere | What qagents does |
|---|---|
| Graph frameworks re-run the whole node after a crash or an approval, and match interrupts by position | Resume is per tool call, keyed by call id; a finished call never runs again |
| Checkpoints serialised with pickle | The journal is plain JSON |
| Prompts injected by the framework that the developer cannot see | One file of strings, all replaceable |
| No default limits on turns or spend | `max_turns=30` by default; token, cost and time limits available |
| Hooks or permission checks that fail open | A failing check refuses; a failing hook stops the run |
| Unknown request fields dropped silently, or rejected mid-run with a 400 | Checked against the provider's allow-list when the agent is built |
| Vendor SDKs tied to one model family or one API | Any provider behind one `stream` method; Chat Completions is first-class |
| Every pause and resume re-runs side effects | Recorded results are reused; `ctx.operation_id` gives tools a stable idempotency key |
| Coding agents wrapped as one opaque tool call, or "driven" over a tool protocol | External agents are driven over ACP or the Devvi SDK: every permission request goes through your rules, each step is an event, a cancel stops the turn and keeps the session |

## Guarantees

What you can rely on, and what you cannot.

| Guarantee | Scope |
|---|---|
| A tool call with a recorded result is never run again | per session journal |
| A model response is journaled before its tool calls start | always |
| An interrupted, non-idempotent tool call is not repeated; the model is told the outcome is unknown | on `resume` and on the next `run` |
| `ctx.operation_id` is identical when a call is repeated after a resume | per run |
| An ambiguous model request is not repeated | unless `Limits(retry_ambiguous=True)` |
| A denied or failed permission check never runs the tool | always |
| The conversation sent to the provider is always well-formed (every tool call has a result) | always |
| An external agent's permission request is decided by your rules, never answered "allow always", and refused when deciding fails: a `can_use_tool` or `on_ask` that raises answers reject, a decision that times out answers cancelled | always |

The guarantees for calls to external agents, including crash recovery, are listed in
[External agents](concepts/external-agents.md#guarantees-for-external-calls).

Not guaranteed:

- **Exactly-once side effects.** Delivery is at-least-once. Use `ctx.operation_id` with the
  system your tool calls.
- **Liveness.** Nothing detects a dead run and resumes it for you unless you run
  `AgentRuntime`, whose `recover()` resumes runs whose worker's lease expired.
- **Cost accuracy.** `cost_usd` is computed from catalog prices and reported tokens.
- **Model behaviour.** Whether a model calls tools reliably or honours a JSON schema depends on
  the model and how it is served. Measure it on the models you use.

## What is verified

| Area | How it was checked |
|---|---|
| Loop, tools, permissions, hooks, limits, cancel, structured output | unit tests with scripted models |
| Sessions, crash recovery, approvals, compaction, workflows | unit tests, including SQLite across store instances and tasks killed mid-tool |
| OpenAI-compatible provider | tests against a local HTTP transport: streaming, tool-call deltas, retries, error mapping |
| QCloud provider | tests through the real `qcloud_ai` 0.4.0 SDK with its HTTP transport replaced locally |
| Toolkits | unit tests, including path-escape and non-public-address refusals |
| Tracing | tests with the OpenTelemetry SDK's in-memory exporter |
| Examples and documentation | examples are executed, documentation code blocks are compiled |
| External agents: ACP client, Devvi (ACP and SDK transports), MCP tool server | offline tests against Devvi's recorded ACP and SDK transcripts, fake agent processes started as real subprocesses, process-tree kill on Windows, the official MCP client over stdio and Streamable HTTP; opt-in tests (`-m devvi_mock`) against the real Devvi 0.2.1-rc.3 CLI and a local mock model |
| Docs server for coding agents | tests of search ranking on known questions, sections, live API introspection, the code checker on correct and wrong code, the command line, and the MCP server over stdio with the official client |

Not verified yet:

- A request against the live QCloud API. `examples/live_acceptance.py` exists for this and has
  only been run with `--dry-run`.
- Tool-calling and structured-output reliability of specific catalog models.
- Python 3.11 and 3.12, Linux and macOS. The suite ran on Python 3.13 on Windows, where the
  POSIX file-symlink test of `file_tools` is skipped (the junction tests run).
- Behaviour under load and with long journals.
- External agents against a paid model route (Devvi tier 2), on Linux or macOS, or on Node 24.9
  or newer without the compatibility hook the tier-1 tests used on Node 22.17. Other ACP agents
  than Devvi have no recorded fixture yet.

## Deliberately left out

| Not included | Why |
|---|---|
| A graph DSL or Pregel-style runtime | Most flows are sequential; a superstep barrier ties parallel branches together. Workflows are plain Python. |
| Role-play crews, group chat, an LLM manager | Studies of multi-agent failures point at coordination, not model capability. One orchestrator with isolated workers is what holds up. |
| Built-in RAG, vector stores, a memory product | Retrieval is a tool you write; file-based notes are competitive with memory layers in published benchmarks. |
| A universal provider gateway | A compromised release of a popular gateway shipped a credential stealer in 2026, and gateways lose reasoning state. Thin adapters are less risk. |
| An in-process code sandbox | An in-process sandbox is not a security boundary. |
| Pickled state | Stored state is a trust boundary. |
| Driving agents over MCP | MCP is a tool protocol: one call, one result. A coding agent's turn has a session, many steps, permission requests back to the caller, and a cancel that keeps the session. Agents are driven over ACP (or the Devvi SDK); MCP only gives them tools. |

## Roadmap

Shipped in 0.2: `PostgresStore` and `AgentRuntime` with leases and a recovery worker that
resumes dead runs, the MCP client toolkit (spec revision 2026-07-28, with fallback to
2025-11-25), signed approvals, the final-answer tool as a third structured-output mode, limits
inherited by sub-agents, and the Anthropic Messages and AG-UI adapters. Since 0.2: external
agents (the ACP client, Devvi, the MCP tool server) and the docs server for coding agents.

Next, in this order:

1. **Live acceptance on QCloud**: the paid checks in `examples/live_acceptance.py`, with usage
   and cost reconciled, and a small tool-calling evaluation per model in use.
2. **Deferred tool loading** for large tool sets, and journal snapshots.
3. **External agents**: live Devvi acceptance (one paid task, usage reconciled), recorded
   fixtures for more ACP agents, the devvi.sh platform as a `Model`, and qagents as an ACP
   agent server.

## Stability

The project is 0.x: names can still change between minor versions. The session journal is the
part meant to freeze first. New entry kinds and new event types may be added; readers should
ignore kinds they do not know. The strings in `qagents.prompts` are part of the behaviour, so
changes to them will be listed in release notes.
