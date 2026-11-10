# Principles and guarantees

qagents was designed after a survey of the agent frameworks of late 2026
([research report, in Greek](research.md)). This page records the decisions that follow from it,
what the library guarantees, and what it deliberately leaves out.

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
fixed order, the more restrictive rule wins, and any failure while deciding refuses the call.
Hooks that raise stop the run. Published prompt-injection defences that rely on detection have
all been bypassed; rules and least privilege are what holds.

**Safe by default.** Runs have a turn limit. Tools that are not declared read-only run one at a
time. Tools with side effects ask first. Requests that may have been billed are not repeated.
Message content stays out of traces unless you ask for it.

**Fail early and say why.** A setting the provider would reject fails when the agent is built.
A model without the needed capability fails before the request is sent. Errors name what to do.

**Provider-neutral inside, QCloud first outside.** Conversations are stored as typed parts, not
as one vendor's JSON, with opaque provider data kept for replay. Adapters translate at the edge.
There is no universal gateway dependency.

## Compared with what exists

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

Not guaranteed:

- **Exactly-once side effects.** Delivery is at-least-once. Use `ctx.operation_id` with the
  system your tool calls.
- **Liveness.** Nothing detects a dead run and resumes it for you in 0.1.
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

Not verified yet:

- A request against the live QCloud API. `examples/live_acceptance.py` exists for this and has
  only been run with `--dry-run`.
- Tool-calling and structured-output reliability of specific catalog models.
- Python 3.11 and 3.12, Linux and macOS. The suite ran on Python 3.13 on Windows, where the
  `file_tools` symlink test is skipped.
- Behaviour under load and with long journals.

## Deliberately left out

| Not included | Why |
|---|---|
| A graph DSL or Pregel-style runtime | Most flows are sequential; a superstep barrier ties parallel branches together. Workflows are plain Python. |
| Role-play crews, group chat, an LLM manager | Studies of multi-agent failures point at coordination, not model capability. One orchestrator with isolated workers is what holds up. |
| Built-in RAG, vector stores, a memory product | Retrieval is a tool you write; file-based notes are competitive with memory layers in published benchmarks. |
| A universal provider gateway | A compromised release of a popular gateway shipped a credential stealer in 2026, and gateways lose reasoning state. Thin adapters are less risk. |
| An in-process code sandbox | An in-process sandbox is not a security boundary. |
| Pickled state | Stored state is a trust boundary. |

## Roadmap

In this order:

1. **Live acceptance on QCloud**: the paid checks in `examples/live_acceptance.py`, with usage
   and cost reconciled, and a small tool-calling evaluation per model in use.
2. **`PostgresStore`** with leases and a recovery worker that finds and resumes dead runs, plus
   journal snapshots and a schema version.
3. **MCP client toolkit** (spec revision 2026-07-28, with fallback to 2025-11-25) and deferred
   tool loading for large tool sets.
4. **Signed approvals**, so a stored pause cannot be forged.
5. **A final-answer tool** as a third structured-output mode, and limits inherited by sub-agents.
6. **Adapters**: Anthropic Messages, and an AG-UI projection of the event stream.

## Stability

The project is 0.x: names can still change between minor versions. The session journal is the
part meant to freeze first. New entry kinds and new event types may be added; readers should
ignore kinds they do not know. The strings in `qagents.prompts` are part of the behaviour, so
changes to them will be listed in release notes.
