# Comparison with other frameworks

This page compares qagents with six other Python agent frameworks on the points that decide a
choice for long-running agents: durable runs, human approvals, what happens to external
writes after a crash, and how much of the surrounding product each one provides. It was
checked on 2026-11-14. The recovery section also covers Microsoft Foundry Agent Service,
with vendor sources checked on 2026-11-16.

**Repository access update, 2026-11-22:** QAgents source is now public on GitHub.
References to private access below describe the original review snapshot. The
package remains unpublished on PyPI and has no open-source license.

## Scope and sources

The comparison uses each framework's documentation, package entry and repository
at the recorded snapshot. [Notes and sources](#notes-and-sources) links the evidence
for each framework. This is a historical comparison, so check the current product
documentation before choosing a deployment.

The ratings distinguish built-in support, partial support and integrations.
An absence rating means the review found the feature absent with high or medium
confidence. A preview remains partial even when its local tests pass.

The QAgents column covers release 0.2.0 and development snapshot `ecd114d`,
including the Effects review fixes and persistent team messaging. AutoGen and
Semantic Kernel are omitted because their READMEs identify archival or succession
by Microsoft Agent Framework. Other frameworks were outside this review.

## Versions compared

| Framework | Package | Version | Released |
|---|---|---|---|
| qagents | `qagents` (not on PyPI) | 0.2.0 and development snapshot `ecd114d` | 2026-11-11 (0.2.0) |
| LangGraph | `langgraph`, with `langchain` and `langchain-core` | 1.2.22 (langchain 1.5.0, langchain-core 1.8.0, langgraph-api 0.18.0) | 2026-11-14 |
| CrewAI | `crewai` | 1.15.41 | 2026-11-12 |
| Microsoft Agent Framework | `agent-framework` (Python), `Microsoft.Agents.AI` (.NET) | Python 1.26.0, .NET 1.28.0, Go v0.3.1 prerelease | 2026-11-12 (Python), 2026-11-06 (.NET) |
| OpenAI Agents SDK | `openai-agents` | 0.24.0 | 2026-10-15 |
| Google ADK | `google-adk` | 2.14.1 | 2026-11-12 |
| Pydantic AI | `pydantic-ai` | 2.68.0 | 2026-11-12 |

## Capabilities at the recorded snapshot

| | qagents | LangGraph | CrewAI | Microsoft Agent Framework | OpenAI Agents SDK | Google ADK | Pydantic AI |
|---|---|---|---|---|---|---|---|
| License, languages | No license file. Python | Built in: MIT. Python, JS | Built in: MIT. Python | Built in: MIT. Python, .NET, Go preview | Built in: MIT. Python, TS | Built in: Apache-2.0. 5 languages | Built in: MIT. Python |
| Stable release | 0.2.0 alpha. No tag, not on PyPI | Built in: 1.x | Built in: 1.x | Built in: 1.x | Partial: 0.x | Built in: 2.x | Built in: 2.x |
| Simple and lower-level layers | Built in: Agent, Workflow, runtime | Built in: create_agent, StateGraph | Built in: Crews, Flows | Built in: Agent, Workflows | Partial: no workflow layer | Built in: LlmAgent, graph workflows | Built in: Agent, graph |
| Model providers | Partial: 3 adapters, OpenAI-compatible | Built in: many packages | Built in: native SDKs | Built in: many, some beta | Partial: OpenAI first | Partial: Gemini first | Built in: many native |
| Typed tools, structured output | Built in: | Built in: | Built in: | Built in: | Built in: | Built in: | Built in: |
| MCP, A2A | Partial: MCP client. Server unreleased. No A2A | Partial: MCP client beta. Server, A2A in Agent Server | Partial: MCP client, A2A. No MCP server | Partial: MCP client, server. A2A beta | Partial: MCP client. No A2A | Built in: MCP client, server, A2A | Partial: MCP client. A2A separate |
| Multi-agent | Partial: sub-agents, handoffs, fan-out, SQLite team mailbox (preview) | Built in: subagents, handoffs, router | Built in: crews, LLM manager | Built in: incl. group chat | Built in: handoffs, agents as tools | Built in: sub-agents, graphs | Built in: delegation, graphs |
| Approval pause, resume elsewhere | Built in: signed tokens | Built in: interrupt() | Built in: Flows | Built in: in checkpoints | Built in: RunState JSON | Partial: not with DB sessions | Built in: deferred tools |
| Durable runs, crash recovery | Built in: SQLite, Postgres, leases | Built in: checkpointers. Leases in Agent Server | Partial: JSON, SQLite. Manual resume | Partial: manual. Automatic via Durable Task | Integration: Dapr, Temporal, Restate, DBOS | Partial: manual. Automatic via Temporal plugin (experimental) | Integration: Temporal, DBOS, others |
| Side effects after a crash | Built in: finished calls never re-run. Interrupted writes reported as unknown, not repeated. Effects preview | Not provided: at-least-once, documented | Not provided: not documented | Not provided: left to the app | Not provided: left to the app | Not provided: at-least-once | Integration: Harness ledger |
| Crash and fault-injection tests | Partial: preview | Not provided: | Not provided: | Not provided: | Partial: scripted errors | Partial: simulated tool errors | Not provided: |
| Streaming to a UI | Built in: typed events, AG-UI | Built in: AG-UI via adapter | Built in: AG-UI via package | Built in: AG-UI adapter | Partial: no AG-UI adapter | Partial: AG-UI via middleware | Built in: AG-UI, Vercel AI |
| Realtime voice | Not provided: | Partial: tutorial | Not provided: | Integration: Azure Voice Live | Built in: | Partial: Gemini Live, experimental | Built in: |
| Tracing | Built in: OpenTelemetry, MLflow | Integration: LangSmith, OTel export | Built in: event bus, OTel | Built in: OpenTelemetry | Built in: own dashboard, processors | Built in: OpenTelemetry | Built in: OpenTelemetry |
| Test models, evals | Built in: scripted, cassettes, evals | Built in: fake models, evals | Partial: LLM judge only | Partial: local evaluators | Partial: fakes. No evals | Built in: eval suite | Built in: test models, evals |
| Memory | Partial: journal, state, notes | Built in: stores with search | Built in: memory with vector recall | Built in: context providers | Partial: sessions | Built in: memory service | Integration: long-term in Harness |
| Coding agents | Partial: unreleased: ACP client, docs server | Partial: docs for agents | Partial: docs for agents | Built in: drives coding SDKs | Partial: experimental tool | Partial: docs for agents | Partial: docs, ACP experimental |
| Managed hosting | Not provided: self-hosted | Integration: LangSmith | Integration: CrewAI AMP | Integration: Foundry Agent Service / Azure Functions | Not provided: | Integration: Google Cloud | Not provided: |

## Choosing a framework

- **qagents.** Choose it when a run must survive restarts and human decisions with a
  plain-JSON journal, when approvals cross process boundaries and must be signed, and when you
  want to test recovery offline, including, as a preview, the outcome of external writes. Do
  not choose it yet if you need a license (without one, nobody else is granted the right to
  use or change the code), public access (the repository is private), a PyPI package, a stable
  API, A2A, many native providers, voice or managed hosting.
- **LangGraph.** Choose it for an explicit graph runtime in which checkpoints, interrupts and
  time travel are one mechanism, for its many checkpointers and integrations, and when a
  managed platform (LangSmith Deployment) is welcome.
- **CrewAI.** Choose it when role-based crews with an LLM manager fit the work, or for
  event-driven Flows with built-in memory, A2A and a commercial platform (AMP).
- **Microsoft Agent Framework.** Choose it for .NET or Azure work, for prebuilt
  orchestrations including group chat and Magentic, for OpenTelemetry by default, and for a
  migration path from Semantic Kernel or AutoGen.
- **OpenAI Agents SDK.** Choose it when you build on OpenAI models and want a small set of
  primitives, realtime voice agents and OpenAI's tracing dashboard. Add Dapr, Temporal,
  Restate or DBOS for durable runs.
- **Google ADK.** Choose it on Google Cloud, when you need SDKs in several languages, built-in
  A2A and MCP serving, and an evaluation suite with user and environment simulation.
- **Pydantic AI.** Choose it for typed agents end to end, many native providers, realtime
  sessions, AG-UI and Vercel AI adapters, and durable execution through an engine you already
  run.

## Recovery for long-running agent teams

The main reason to evaluate qagents is a workflow that spans workers and handoffs, pauses
for review, and makes changes in external systems. Keep the approval and tool results in
the run journal, keep inter-agent messages in a persistent mailbox, and rehearse the
points where a worker can stop. Effects 0.3 adds a preview for recording external
operations before they are sent, checking an ambiguous result and holding it for an
operator when the available evidence cannot support an automatic decision.
[Team messaging](concepts/multi-agent.md) · [Recover an external action](guides/recover-effect.md)

| Platform | What it already provides for this workload |
|---|---|
| LangGraph | Explicit graph checkpoints, interrupts and resume. Its fault-tolerance guide describes stable idempotency keys and how to handle a missing acknowledgement. [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence) · [Fault tolerance](https://docs.langchain.com/oss/python/langgraph/fault-tolerance) |
| CrewAI | Role-based Crews and event-driven Flows. Agents, Crews and Flows support JSON or SQLite checkpoints and explicit restore. Automatic checkpoint writes are best-effort. [Crews](https://docs.crewai.com/en/concepts/crews) · [Checkpointing](https://docs.crewai.com/en/concepts/checkpointing) |
| Microsoft Agent Framework | Workflow graph checkpoints and built-in orchestrations. The Durable Extension adds persistent sessions and distributed recovery for Python and .NET on Azure Functions or self-hosted workers backed by Durable Task. [Workflows](https://learn.microsoft.com/en-us/agent-framework/concepts/workflows/) · [Durable Extension](https://learn.microsoft.com/en-us/agent-framework/hosting/durable-extension) |
| Microsoft Foundry Agent Service | Managed endpoints, scaling, Entra identity, session persistence and observability. Hosted Agents can run a container built with another framework. [Foundry Agent Service](https://learn.microsoft.com/en-us/azure/foundry/agents/overview?view=foundry) |

The matched Gym experiment gives qagents and LangGraph the same credit workflow and a
separate destination ledger. A rerun at `ecd114d` passed all 40 positive trials and detected
the deliberately duplicated credit in all 10 negative controls with destination deduplication
disabled. Across the three enabled-deduplication crash cases, median kill-to-result time
ranged from 7.07 to 7.13 seconds for Effects and 1.25 to 1.34 seconds for LangGraph on one
Windows host. The Effects setup used a 6.1-second worker lease. The single-worker LangGraph
setup used no equivalent lease. [Gym benchmark and reproduction steps](https://github.com/qagents-ai/qagent-gym/blob/main/docs/recovery-benchmark.md#effects-development-snapshot-november-14)

An operator path is covered by a separate runnable example. After the simulated
destination commits a credit and loses its response, a fresh worker records the
operator's verified receipt and resumes the agent. The example verifies one credit in the
destination ledger. [Operator recovery example](guides/recover-effect.md)

The fictional KA220 rehearsal tests a different boundary. Three agents run in separate
processes, and the writer is terminated after SQLite commits a draft message and before its
agent journal records the tool result. Both restart and restoration from backup delivered
the same draft sequence, completed all 75 consistency checks and produced matching final
proposal hashes. Six evidence warnings remained for a human reviewer. The rehearsal uses
scripted models and compares two qagents recovery paths.
[KA220 pilot](https://github.com/qagents-ai/qagent-gym/blob/main/docs/erasmus-example.md#rehearse-worker-restart-and-backup-restore)

### What to verify before an organizational rollout

1. Fix false positives in the default crash harness, then require it to pass on known-good
   fixtures and detect seeded faults. A violated invariant should identify the run,
   operation and exact fault point.
2. Test the Effects journal and team mailbox with workers on separate hosts. Rehearse
   failover after send and receive commits, stale-worker fencing and consistent restoration.
   The Effects preview and persistent mailbox currently use SQLite for durable storage.
3. Document how operators inspect a held effect, verify a provider receipt and record who
   settled it. Run the same commit-before-ack workload in each framework with equivalent
   retries, provider contracts and worker coordination. Compare duplicate writes, lost
   messages, recovery time and manual interventions.

## Notes and sources

### qagents

| Dimension | Rating | Note | Evidence |
|---|---|---|---|
| License, languages | Not provided: | Python 3.11+. There is no LICENSE file and `pyproject.toml` declares no license, so no open-source license is granted. The repository is private. | `pyproject.toml` |
| Release | Not provided: | 0.2.0, alpha, dated 2026-11-11 in the changelog. No release tag, not on PyPI. Effects 0.3 is unreleased work on `main`. | `CHANGELOG.md` |
| Layers | Built in: | `Agent(...).run()`. Below it, `Workflow`, `AgentRuntime`, `Hooks` and the single `Model.stream` contract. No graph DSL, by design. | [Principles](design.md#deliberately-left-out) |
| Model providers | Partial: | QCloud, OpenAI Chat Completions and Anthropic adapters, any OpenAI-compatible endpoint, and custom providers. No Gemini, Bedrock, Azure, Mistral or OpenAI Responses adapter. | [Model providers](providers.md) |
| Typed tools, structured output | Built in: | `@tool` schemas from type hints, arguments validated before anything runs. `output_type` in three modes, validators and guardrails. | [Tools](concepts/tools.md), [Structured output](concepts/structured-output.md) |
| MCP, A2A | Partial: | MCP client toolkit (released). MCP tool server bound to a run or over stdio (unreleased). No A2A. | [Ready-made tools](toolkits.md), [Serving tools over MCP](integrations/mcp-server.md) |
| Multi-agent | Partial: | Sub-agents with shared live budgets, handoffs, `fan_out`, `route` and durable workflows. An experimental SQLite `TeamMailbox` persists messages across restarts and serves processes on the same host. | [Several agents](concepts/multi-agent.md) |
| Approval pause, resume elsewhere | Built in: | An unanswered approval pauses the run. Any process can resume it from the journal. With `Permissions(approval_secret=...)`, pause tokens are HMAC-SHA256 signed. Without a secret, pauses are not signed. A writer with store access can replay an answered signed pause inside an open run. | [Permissions](concepts/permissions.md) |
| Durable runs, crash recovery | Built in: | Memory, SQLite or Postgres journals. Resume at the exact tool call. Leases, heartbeats and `recover()` in `AgentRuntime`. Effects run on memory and SQLite only. | [Sessions](concepts/sessions.md), [Serving](serving.md) |
| Side effects after a crash | Built in: | Released: a call with a recorded result never runs again, an interrupted non-idempotent call is not repeated, and `operation_id` is a stable key. Exactly-once is not claimed. Effects (preview) adds durable intents, contracts, lookups and holds. | [What is guaranteed](concepts/sessions.md#what-is-guaranteed), [Recover an external action](guides/recover-effect.md) |
| Crash and fault-injection tests | Partial: | Unreleased: `crash_test` with 38 faults, subprocess kills, reports and a `crashtest` command. A process lane of 26 tests (14 real-kill recovery cases) and 68 mutations in the repository. Default faults still flag correct fixtures. | [Recover an external action](guides/recover-effect.md) |
| Streaming to a UI | Built in: | Typed events with token deltas, cancel that reaches the provider request, AG-UI projection and a FastAPI example. Effect events are not projected. | [Streaming](concepts/streaming.md) |
| Realtime voice | Not provided: | No audio API. Barge-in means cancelling the stream and starting a new run. | [Real-time and voice](concepts/streaming.md#real-time-and-voice) |
| Tracing | Built in: | `OTelHooks` with content off by default, `MLflowHooks`. No tracing platform of its own. Effect events are not traced. | [Tracing](concepts/hooks.md#tracing) |
| Test models, evals | Built in: | `ScriptedModel`, `FunctionModel`, record and replay cassettes, `qagents.evals` with an LLM judge, compare and regressions. No hosted evaluation service. | [Testing agents](testing.md) |
| Memory | Partial: | Journal, session state, compactors, todos and file-based notes. No vector store or memory product, by design. | [Principles](design.md#deliberately-left-out) |
| Coding agents | Partial: | Unreleased: `ExternalAgent`, an ACP client and Devvi. The docs server, `llms.txt`, `AGENTS.md` and a skill. No ACP server, and ACP usage is not metered. | [External agents](concepts/external-agents.md), [Docs for coding agents](coding-agents.md) |
| Managed hosting | Not provided: | Self-hosted only: `AgentRuntime` workers, SQLite or Postgres, an example FastAPI server. | [Serving over HTTP](serving.md) |

### LangGraph

| Dimension | Rating | Note | Source |
|---|---|---|---|
| License, languages | Built in: | Python and TypeScript. The library and checkpointers are MIT. The Agent Server packages are Elastic-2.0. | [repository](https://github.com/langchain-ai/langgraph) |
| Release | Built in: | langgraph 1.2.22 on 2026-11-14. Langchain 1.5.0 needs Python 3.11+. | [PyPI](https://pypi.org/project/langgraph/) |
| Layers | Built in: | `create_agent` runs a model and tool loop with middleware. Below it, `StateGraph` or the Functional API. | [agents](https://docs.langchain.com/oss/python/langchain/agents) |
| Model providers | Built in: | First-party provider packages through `init_chat_model`. OpenAI-compatible base URLs. Not tied to one vendor. | [providers-and-models](https://docs.langchain.com/oss/python/concepts/providers-and-models) |
| Typed tools, structured output | Built in: | `@tool` from type hints or a Pydantic schema. `response_format` returns a validated structured response. | [structured-output](https://docs.langchain.com/oss/python/langchain/structured-output) |
| MCP, A2A | Partial: | MCP client adapter is beta. Serving agents over MCP or A2A works only through Agent Server. | [server-mcp](https://docs.langchain.com/langsmith/server-mcp) |
| Multi-agent | Built in: | Subagents as tools, handoffs, router and custom graphs. No group-chat primitive. | [multi-agent](https://docs.langchain.com/oss/python/langchain/multi-agent) |
| Approval pause, resume elsewhere | Built in: | `interrupt()` waits. Any process can resume through a persistent checkpointer. No signed approval tokens documented. | [human-in-the-loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop) |
| Durable runs, crash recovery | Built in: | SQLite, Postgres, MongoDB and other checkpointers. In the library you resume yourself. Worker leases and requeue are Agent Server features. | [checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers) |
| Side effects after a crash | Not provided: | A retry or resume re-runs the whole node (at-least-once). Side effects inside a node are outside the checkpoint, and the application must make them idempotent. The docs show idempotency-key patterns and treat a lost acknowledgement as unknown. | [fault-tolerance](https://docs.langchain.com/oss/python/langgraph/fault-tolerance) |
| Crash and fault-injection tests | Not provided: | No fault-injection or process-kill tooling documented (medium confidence). | [test](https://docs.langchain.com/oss/python/langgraph/test) |
| Streaming to a UI | Built in: | Stream modes and typed event streams. AG-UI needs a separate adapter. | [event-streaming](https://docs.langchain.com/oss/python/langgraph/event-streaming) |
| Realtime voice | Partial: | A tutorial chains third-party speech-to-text and text-to-speech. No realtime session API. | [voice-agent](https://docs.langchain.com/oss/python/langchain/voice-agent) |
| Tracing | Integration: | Traces go to LangSmith. OpenTelemetry export is available. | [trace-with-opentelemetry](https://docs.langchain.com/langsmith/trace-with-opentelemetry) |
| Test models, evals | Built in: | Fake chat models, in-memory checkpointer, trajectory and LLM-judge evaluators, LangSmith datasets. | [unit-testing](https://docs.langchain.com/oss/python/langchain/test/unit-testing) |
| Memory | Built in: | Per-thread checkpoints and a namespaced long-term store with optional embedding search. | [stores](https://docs.langchain.com/oss/python/langgraph/stores) |
| Coding agents | Partial: | `llms.txt` and two docs MCP servers. Deep Agents can be exposed to ACP editors. | [use-these-docs](https://docs.langchain.com/use-these-docs) |
| Managed hosting | Integration: | Agent Server on LangSmith Deployment. Self-hosting it needs an Enterprise license. | [deploy-to-self-hosted-overview](https://docs.langchain.com/langsmith/deploy-to-self-hosted-overview) |

### CrewAI

| Dimension | Rating | Note | Source |
|---|---|---|---|
| License, languages | Built in: | Python only, MIT. The AMP platform is a separate commercial product. | [PyPI](https://pypi.org/project/crewai/1.15.41/) |
| Release | Built in: | 1.15.41 on 2026-11-12. Stable 1.x since 2025-10-20, with patches almost daily. | [release](https://github.com/crewAIInc/crewAI/releases/tag/1.15.41) |
| Layers | Built in: | Role-based Crews and `Agent.kickoff()`. Below them, Flows with typed state. | [production-architecture](https://docs.crewai.com/en/concepts/production-architecture) |
| Model providers | Built in: | Native SDK adapters for several vendors, OpenAI-compatible endpoints, LiteLLM as an optional fallback. | [llms](https://docs.crewai.com/en/concepts/llms) |
| Typed tools, structured output | Built in: | Tools with Pydantic schemas. While tools are offered, the output schema is only in the prompt and checked after the answer. | [tools](https://docs.crewai.com/en/concepts/tools) |
| MCP, A2A | Partial: | Built-in MCP client and A2A client and server (the `crewai[a2a]` extra). No MCP server in the library. | [a2a-agent-delegation](https://docs.crewai.com/en/learn/a2a-agent-delegation) |
| Multi-agent | Built in: | Sequential or hierarchical crews with an LLM manager, delegation tools, remote delegation over A2A. | [collaboration](https://docs.crewai.com/en/concepts/collaboration) |
| Approval pause, resume elsewhere | Built in: | `@human_feedback` pauses a Flow and resumes later, also in another process. Per-tool approval only through a blocking hook. | [human-feedback-in-flows](https://docs.crewai.com/en/learn/human-feedback-in-flows) |
| Durable runs, crash recovery | Partial: | Agents, Crews and Flows support JSON or SQLite snapshots and explicit restoration. Automatic checkpoint writes are best-effort. | [checkpointing](https://docs.crewai.com/en/concepts/checkpointing) |
| Side effects after a crash | Not provided: | Restore skips completed tasks. Nothing documented about idempotency or unfinished work (medium confidence). | [checkpointing](https://docs.crewai.com/en/concepts/checkpointing) |
| Crash and fault-injection tests | Not provided: | None documented (medium confidence). | [checkpointing](https://docs.crewai.com/en/concepts/checkpointing) |
| Streaming to a UI | Built in: | Streaming for Flows, model calls and Crews. AG-UI through a separate package. | [streaming](https://docs.crewai.com/en/concepts/streaming) |
| Realtime voice | Not provided: | No realtime or voice agents. Audio files can be passed as multimodal input (`AudioFile`). Speech input and output are left to the application or a transport adapter. | [conversational-flows](https://docs.crewai.com/en/guides/flows/conversational-flows), [files](https://docs.crewai.com/en/concepts/files) |
| Tracing | Built in: | Event bus, tracing to AMP after consent, host OpenTelemetry providers. | [tracing](https://docs.crewai.com/en/observability/tracing) |
| Test models, evals | Partial: | `crewai test` scores repeated runs with an LLM judge. Experimental evaluation module. No fake model documented. | [testing](https://docs.crewai.com/en/concepts/testing) |
| Memory | Built in: | One memory class with vector recall (LanceDB by default) and pluggable storage. | [memory](https://docs.crewai.com/en/concepts/memory) |
| Coding agents | Partial: | `llms.txt`, a docs MCP server, a skills pack and an AGENTS.md guide. Nothing for driving external agents. | [build-with-ai](https://docs.crewai.com/en/guides/coding-tools/build-with-ai) |
| Managed hosting | Integration: | CrewAI AMP: managed cloud with a free tier, and Enterprise including self-hosted Kubernetes. | [platform introduction](https://docs-platform.crewai.com/platform/en/introduction) |

### Microsoft Agent Framework

| Dimension | Rating | Note | Source |
|---|---|---|---|
| License, languages | Built in: | MIT. Python, .NET, and Go in public preview. | [repository](https://github.com/microsoft/agent-framework) |
| Release | Built in: | Python 1.26.0, .NET 1.28.0. GA 1.0 on 2026-04-03. Many integration packages are still beta or alpha. | [PyPI](https://pypi.org/project/agent-framework/) |
| Layers | Built in: | Agents with sessions and middleware, a Harness agent, and typed graph Workflows. | [overview](https://learn.microsoft.com/en-us/agent-framework/overview/) |
| Model providers | Built in: | OpenAI, Azure OpenAI and Foundry stable. Anthropic, Bedrock, Gemini, Ollama and Mistral packages beta in Python. | [model-providers](https://learn.microsoft.com/en-us/agent-framework/integrations/by-component/model-providers/) |
| Typed tools, structured output | Built in: | `@tool` from type hints with Pydantic validation. `response_format` returns a parsed value. | [function-tools](https://learn.microsoft.com/en-us/agent-framework/agents/tools/function-tools) |
| MCP, A2A | Partial: | MCP client and `as_mcp_server()` built in. A2A client and server in a beta package. | [local-mcp-tools](https://learn.microsoft.com/en-us/agent-framework/agents/tools/local-mcp-tools) |
| Multi-agent | Built in: | Sequential, concurrent, handoff, group chat and Magentic orchestrations. Agents as tools. Sub-workflows. | [orchestrations](https://learn.microsoft.com/en-us/agent-framework/workflows/orchestrations/) |
| Approval pause, resume elsewhere | Built in: | Pending approvals are held in the `AgentSession` and, in workflows, in checkpoints that a new workflow instance can restore. A response authorizes a call only when it matches a pending request in the same session. No signed tokens. | [tool-approval](https://learn.microsoft.com/en-us/agent-framework/agents/tools/tool-approval), [checkpoints](https://learn.microsoft.com/en-us/agent-framework/workflows/checkpoints) |
| Durable runs, crash recovery | Partial: | Checkpoints to memory, file, Cosmos DB (beta) or Foundry. Manual resume. Leases and automatic recovery through the Durable Task extension. | [checkpoints](https://learn.microsoft.com/en-us/agent-framework/workflows/checkpoints) |
| Side effects after a crash | Not provided: | The Durable Extension docs assign tool idempotency to the application because tools can execute again after retries or recovery. | [durable-extension](https://learn.microsoft.com/en-us/agent-framework/hosting/durable-extension) |
| Crash and fault-injection tests | Not provided: | None documented. Mock clients are internal test fixtures (medium confidence). | [repository](https://github.com/microsoft/agent-framework) |
| Streaming to a UI | Built in: | Streaming updates and workflow events. A stable AG-UI adapter in Python. | [ag-ui](https://learn.microsoft.com/en-us/agent-framework/integrations/by-component/ui/ag-ui/) |
| Realtime voice | Integration: | No voice API in the framework. Hosted-agent docs point to Azure Speech Voice Live (medium confidence). | [foundry-hosted-agent](https://learn.microsoft.com/en-us/agent-framework/hosting/foundry-hosted-agent) |
| Tracing | Built in: | OpenTelemetry traces, logs and metrics per the GenAI conventions. | [observability](https://learn.microsoft.com/en-us/agent-framework/agents/observability) |
| Test models, evals | Partial: | Local evaluators in core. LLM-judge evaluators through Foundry. No public fake model (medium confidence). | [evaluation](https://learn.microsoft.com/en-us/agent-framework/agents/evaluation) |
| Memory | Built in: | Sessions with pluggable history and context providers. A built-in `FileMemoryProvider` keeps memory across sessions (on by default in the Harness agent). Long-term stores through Mem0 (beta), Redis, Foundry and others. | [context-providers](https://learn.microsoft.com/en-us/agent-framework/concepts/agents/conversations/context-providers) |
| Coding agents | Built in: | Agent wrappers for the GitHub Copilot SDK and the Claude Agent SDK (beta), and builder skills. No ACP (medium confidence). | [build-with-coding-agents](https://learn.microsoft.com/en-us/agent-framework/get-started/build-with-coding-agents) |
| Managed hosting | Integration: | Foundry Hosted Agents provide managed endpoints, scaling and identity. The Durable Extension supports Azure Functions and self-hosted workers. | [Foundry Agent Service](https://learn.microsoft.com/en-us/azure/foundry/agents/overview?view=foundry), [Durable Extension](https://learn.microsoft.com/en-us/agent-framework/hosting/durable-extension) |

### OpenAI Agents SDK

| Dimension | Rating | Note | Source |
|---|---|---|---|
| License, languages | Built in: | MIT, Python. A separate TypeScript SDK. | [PyPI](https://pypi.org/pypi/openai-agents/json) |
| Release | Partial: | 0.24.0 on 2026-10-15. Pre-1.0: breaking changes land in minor releases. A new runtime is in alpha. | [releases](https://github.com/openai/openai-agents-python/releases) |
| Layers | Partial: | An Agent and Runner loop with hooks, handoffs and guardrails. No workflow engine. | [documentation](https://openai.github.io/openai-agents-python/) |
| Model providers | Partial: | Native models from its vendor, any compatible `base_url` or custom provider. Adapters for other vendors are best-effort beta. | [models](https://openai.github.io/openai-agents-python/models/) |
| Typed tools, structured output | Built in: | `@function_tool` with Pydantic validation. `output_type` accepts any Pydantic type. | [tools](https://openai.github.io/openai-agents-python/tools/) |
| MCP, A2A | Partial: | MCP client over stdio, SSE, Streamable HTTP and hosted MCP. No MCP server and no A2A in the stable release (medium confidence). | [mcp](https://openai.github.io/openai-agents-python/mcp/) |
| Multi-agent | Built in: | Handoffs and agents as tools. No group chat or messaging primitive. | [multi_agent](https://openai.github.io/openai-agents-python/multi_agent/) |
| Approval pause, resume elsewhere | Built in: | `needs_approval` pauses. The run state is stored as JSON and reloaded later. Snapshots are not authenticated. | [human_in_the_loop](https://openai.github.io/openai-agents-python/human_in_the_loop/) |
| Durable runs, crash recovery | Integration: | Sessions persist conversation items. Crash recovery needs Dapr, Temporal, Restate or DBOS. | [sessions](https://openai.github.io/openai-agents-python/sessions/), [running_agents](https://openai.github.io/openai-agents-python/running_agents/) |
| Side effects after a crash | Not provided: | Approval does not guarantee exactly-once. A crash after a tool acts can rerun it. The app adds idempotency keys. | [human_in_the_loop](https://openai.github.io/openai-agents-python/human_in_the_loop/) |
| Crash and fault-injection tests | Partial: | A scripted model can raise errors. No process-kill or resume tests (medium confidence). | [testing](https://openai.github.io/openai-agents-python/testing/) |
| Streaming to a UI | Partial: | Streamed events and cancel. No first-party AG-UI adapter. | [streaming](https://openai.github.io/openai-agents-python/streaming/) |
| Realtime voice | Built in: | Realtime agents over WebSocket or SIP, and a speech-to-text, agent, text-to-speech pipeline. | [realtime guide](https://openai.github.io/openai-agents-python/realtime/guide/) |
| Tracing | Built in: | Tracing on by default to the vendor's dashboard, with custom and third-party processors. No native OpenTelemetry. | [tracing](https://openai.github.io/openai-agents-python/tracing/) |
| Test models, evals | Partial: | Scripted fakes for models, sandbox, realtime and voice. No evaluation tooling in the SDK. | [testing](https://openai.github.io/openai-agents-python/testing/) |
| Memory | Partial: | Sessions hold history. A sandbox capability writes lessons to files. No long-term store (medium confidence). | [sandbox guide](https://openai.github.io/openai-agents-python/sandbox/guide/) |
| Coding agents | Partial: | An experimental tool wraps the vendor's coding agent for workspace tasks. `llms.txt` (medium confidence). | [tools](https://openai.github.io/openai-agents-python/tools/) |
| Managed hosting | Not provided: | The SDK runs inside your application. OpenAI's separate Agents API (beta) runs OpenAI's own managed Codex harness with optional hosted sandboxes. It does not host SDK code. | [agents guide](https://developers.openai.com/api/docs/guides/agents), [Agents API](https://developers.openai.com/api/docs/guides/agents-api/overview) |

### Google ADK

| Dimension | Rating | Note | Source |
|---|---|---|---|
| License, languages | Built in: | Apache-2.0. Python, TypeScript, Go, Java and Kotlin. | [repository](https://github.com/google/adk-python) |
| Release | Built in: | 2.14.1 on 2026-11-12. 2.x stable since 2026-05-19. | [PyPI](https://pypi.org/pypi/google-adk/json) |
| Layers | Built in: | `LlmAgent`, and a graph `Workflow` runtime with routes and retries. | [ADK 2.0](https://adk.dev/2.0/index.md) |
| Model providers | Partial: | Google's models first. Claude classes. Other providers documented through LiteLLM (medium confidence). | [models](https://adk.dev/agents/models/openai/index.md) |
| Typed tools, structured output | Built in: | `FunctionTool` from the signature. `output_schema` for the final response. | [function-tools](https://adk.dev/tools-custom/function-tools/index.md) |
| MCP, A2A | Built in: | MCP client, an agent exposed as an MCP server, and A2A expose and consume. | [agent-as-server](https://adk.dev/tools-custom/mcp-tools/agent-as-server/index.md) |
| Multi-agent | Built in: | Coordinator sub-agents, agent transfer, `AgentTool`, graph workflows and remote A2A agents. | [collaboration](https://adk.dev/workflows/collaboration/index.md) |
| Approval pause, resume elsewhere | Partial: | Tool confirmation and input nodes. Confirmation is not supported with the database or Vertex session services. | [confirmation](https://adk.dev/tools-custom/confirmation/index.md) |
| Durable runs, crash recovery | Partial: | Many session services. A resumable app restarts an invocation by id. Manual resume, no worker leases (medium confidence). Temporal's ADK plugin (`temporalio[google-adk]`, experimental) recovers automatically from the last successful step. | [resume](https://adk.dev/runtime/resume/index.md), [Temporal plugin](https://adk.dev/integrations/temporal/) |
| Side effects after a crash | Not provided: | Tools run at least once and may run more than once. Failed nodes rerun on resume. | [resume](https://adk.dev/runtime/resume/index.md) |
| Crash and fault-injection tests | Partial: | The eval environment simulator injects tool errors with seeded probabilities. No process-kill tests (medium confidence). | [environment simulation](https://adk.dev/evaluate/environment_simulation/index.md) |
| Streaming to a UI | Partial: | Event streams with SSE and bidirectional modes. AG-UI through separate middleware (medium confidence). | [ag-ui](https://adk.dev/integrations/ag-ui/index.md) |
| Realtime voice | Partial: | Live audio and video agents on Google's Live API, marked experimental (medium confidence). | [live](https://adk.dev/live/index.md) |
| Tracing | Built in: | Native OpenTelemetry GenAI spans with OTLP export. | [traces](https://adk.dev/observability/traces/index.md) |
| Test models, evals | Built in: | Trajectory and rubric metrics, user and environment simulation, multi-run evals. No fake model. | [evaluate](https://adk.dev/evaluate/index.md) |
| Memory | Built in: | Session state and a memory service with Vertex and MongoDB backends (medium confidence). | [memory](https://adk.dev/sessions/memory/index.md) |
| Coding agents | Partial: | `llms.txt`, a docs MCP server and skills. No ACP (medium confidence). | [coding with AI](https://adk.dev/tutorials/coding-with-ai/index.md) |
| Managed hosting | Integration: | Agent Runtime on Google Cloud (paid), Cloud Run or GKE. Self-hosting with its API server. | [agent-runtime](https://adk.dev/deploy/agent-runtime/index.md) |

### Pydantic AI

| Dimension | Rating | Note | Source |
|---|---|---|---|
| License, languages | Built in: | MIT, Python only. | [PyPI](https://pypi.org/pypi/pydantic-ai/json) |
| Release | Built in: | 2.68.0 on 2026-11-12. V2 stable since 2026-06-23. | [releases](https://github.com/pydantic/pydantic-ai/releases) |
| Layers | Built in: | `Agent.run` for the simple path. `agent.iter()` steps the graph nodes. A standalone graph library. | [agent](https://pydantic.dev/docs/ai/core-concepts/agent/) |
| Model providers | Built in: | Many native providers and OpenAI-compatible endpoints. | [models overview](https://pydantic.dev/docs/ai/models/overview/) |
| Typed tools, structured output | Built in: | Validated tool arguments. Tool, native and prompted output modes with retries. | [output](https://pydantic.dev/docs/ai/core-concepts/output/) |
| MCP, A2A | Partial: | MCP client with sampling and elicitation. A2A lives in the separate fasta2a package. | [mcp overview](https://pydantic.dev/docs/ai/mcp/overview/) |
| Multi-agent | Built in: | Delegation through tools, programmatic hand-off and graph control flow. | [multi-agent applications](https://pydantic.dev/docs/ai/guides/multi-agent-applications/) |
| Approval pause, resume elsewhere | Built in: | Deferred tools and a stored conversation resume later in any process. Approval is not an authorization boundary. No signing. | [deferred-tools](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/) |
| Durable runs, crash recovery | Integration: | Crash recovery through an engine (Temporal, DBOS, Prefect, Restate and others). Step persistence in the separate Harness. | [durable execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/) |
| Side effects after a crash | Integration: | An in-flight step runs again. Tools need idempotency keys. The Harness marks started, unfinished calls as unknown. | [production](https://pydantic.dev/docs/ai/production/) |
| Crash and fault-injection tests | Not provided: | Fake models only. No process-kill or resume tests (medium confidence). | [testing](https://pydantic.dev/docs/ai/guides/testing/) |
| Streaming to a UI | Built in: | Event streams with partial validation. AG-UI and Vercel AI adapters. Cancellation tokens. | [ui overview](https://pydantic.dev/docs/ai/integrations/ui/overview/) |
| Realtime voice | Built in: | Realtime speech-to-speech sessions with agent tools over several providers. | [realtime overview](https://pydantic.dev/docs/ai/realtime/overview/) |
| Tracing | Built in: | OpenTelemetry GenAI spans to Logfire or any backend. Prompts included by default. | [logfire](https://pydantic.dev/docs/ai/integrations/logfire/) |
| Test models, evals | Built in: | `TestModel`, `FunctionModel`, model overrides, and the pydantic-evals package. | [testing](https://pydantic.dev/docs/ai/guides/testing/) |
| Memory | Integration: | Message history in core. Long-term memory in the separate Harness. | [persistence](https://pydantic.dev/docs/ai/core-concepts/persistence/) |
| Coding agents | Partial: | `llms.txt`, skills, and an experimental ACP capability that runs an agent inside editors (medium confidence). | [harness acp](https://pydantic.dev/docs/ai/harness/acp/) |
| Managed hosting | Not provided: | No hosted runtime. Agents run in your process. | [enterprise](https://pydantic.dev/docs/ai/production/enterprise/) |

### Open questions

The research left these points unverified. None of them changes a rating above.

- Whether LangGraph Agent Server re-drives a run after a hard worker crash through lease
  expiry. The docs describe requeue after retriable failures.
- What CrewAI AMP does when a worker crashes mid-execution.
- Whether a Microsoft Agent Framework durable agent turn re-runs a tool call that was in
  flight. The docs do not say.
- Whether ADK workflow input pauses survive with the database session service. Only tool
  confirmation is documented as unsupported there.

## Where qagents falls short

- **Alpha, unlicensed, unpublished.** Version 0.2.0 is pre-1.0 and names can change between
  minor versions. The release has no git tag and no PyPI package. There is no LICENSE file
  and the repository is private, so third parties cannot use it or contribute. The `qcloud`
  extra needs `qcloud-ai`, which is not on PyPI.
- **Python only, narrow CI.** CI runs Ubuntu with Python 3.11 and 3.13 and Windows with 3.13.
  macOS and Python 3.12 are not tested, and the QCloud adapter tests do not run in CI.
- **Few providers.** Three native adapters. No OpenAI Responses, Gemini, Bedrock or Azure
  adapter, and no registered `openai:` model string, so the command line cannot reach OpenAI
  without code.
- **Not shipped.** A2A, realtime voice, a hosted platform, a vector store or memory product,
  group chat or crews, qagents as an ACP server, and ACP usage metering.
- **Permissive defaults for writes.** With the default `Permissions`, a tool without
  `requires_approval` runs without asking, although `AGENTS.md` says tools with side effects
  ask first. The notes tools need no approval outside the `qagents` command, and the shell
  tool passes the whole process environment to commands.
- **Effects is an unreleased preview.** It does not run on Postgres. Effect-level approvals,
  `Workflow.effect` and the hermetic-replay invariant are missing, and there is no operator
  command line. The Stripe contract has not been checked against Stripe. The five safety defects
  found by the 2026-11-14 review are fixed, one in part (a string record id in `Settle` still
  skips the revision check). The
  [README](https://github.com/qagents-ai/qagents#effects-03-preview) lists what is still open. Effect events are not traced or projected to AG-UI.
- **The crash harness is not a reliable gate yet.** The default faults flag correct
  fixtures, the `crashtest` example in the guide fails on correct agents, budget sampling
  drops depth-2 points without saying so, Postgres is refused, and most of the mutation
  catalogue does not go through `crash_test`.
- **Team messaging is experimental.** `TeamMailbox` and its SQLite mailbox are missing
  from the changelog, and the SQLite mailbox works only between processes on one host.
- **Known 0.2 limitations.** An answered signed pause can be replayed inside an open run by a
  writer with store access. Output guardrails see the answer after it has streamed. A
  worker's in-flight model request is re-sent on recovery. OpenAI strict tools with defaulted
  parameters are rejected by OpenAI. `fetch_url` resolves a host twice, which leaves a DNS
  rebinding window. The exception text of a `fan_out` worker reaches the calling model.
- **Plain workflow steps are at-least-once.** An unfinished `Workflow.step` runs again after a
  crash.
- **Engineering debt.** `agent.py` has 4,416 lines around one large run class. An SQLite
  append re-folds the whole journal under its write lock (37.6 ms at 20,000 entries). The
  public API is one flat list of about 70 names. CI jobs have no timeouts, actions are not
  pinned to commits, and there is no lock file.
- **Records lag the code.** The changelog stops before the stage-0 fixes, Effects, team
  messaging and the newest examples.
- **Live acceptance is missing.** Live QCloud has had only a dry run. There are no load or
  long-journal runs, and the live Devvi and Stripe lanes have not run.
