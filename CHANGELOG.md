# Changelog

All notable changes to qagents. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (before 1.0, a minor version may change behaviour;
every such change is listed under "Changed").

## [0.2.0] - 2026-11-11

0.2 is about agents that run for a long time, agents that work together, and seeing what they
did. Everything from 0.1 keeps working; the behaviour changes are listed under "Changed".

### Added

#### Pausing for outside input (human in the loop)

- `tool(..., deferred=True)` declares a tool the loop never runs: each call pauses the run as a
  `PendingInput` (`result.pending_inputs`, status `"paused"`). A running tool can pause the
  same way by raising `qagents.errors.DeferToolCall(reason, metadata)`; it is never run again
  for that call.
- `Agent.resume(session, results={tool_call_id: value})` and `resume_stream(..., results=...)`
  supply the results. A value is rendered like a tool return value, an `Exception` becomes an
  error result, a `ToolResultPart` is used as given. Partial answers keep the run paused with
  the rest; approvals and results can be given together. An id that waits for approval is
  refused with a hint to use `approvals=`.
- `qagents.toolkits.ask_user`: a deferred tool that asks the user a question (optional
  `choices`).
- New event `input_required` (`InputRequired`); new journal kind `tool_deferred {input}`;
  `run_paused` now also carries `inputs`.

#### Session state

- `RunContext.state`: one dict per session, shared by tools, hooks and instructions functions,
  restored on every run and resume. `state=` on `run`, `stream`, `resume` and `resume_stream`
  merges values in key by key. Values must survive a JSON round trip (otherwise
  `SessionError` names the key).
- Changes are journaled as `state_updated {values}` and emitted as `state_updated`
  (`StateUpdated`) events. `SessionState.values` exposes the current state.

#### Planning

- `qagents.toolkits.todo_tools(key="todos")` gives the agent `write_todos` and `read_todos`, a
  plan kept in the session state. `render_todos(state)` shows the plan inside an instructions
  function, so a long-running agent sees it on every turn, also after compaction. `TodoItem`
  and `read_todo_list` are in `qagents.toolkits.planning`.

#### Structured output, validators and guardrails

- `Agent(output_mode="tool")`: the final answer is given by calling a final-answer tool
  (`output_tool_name`, default `"final_answer"`). Rejected answers, validator rejections and
  plain-text replies share `Limits.output_retries`.
- `Agent(output_validators=[fn(ctx, output)])`: return a replacement, return `None` to keep
  the answer, or raise `qagents.errors.ModelRetry(message)` to send the model a correction.
  `ModelRetry` raised inside a tool works like `ToolError`.
- `Agent(input_guardrails=..., output_guardrails=...)`: each `fn(ctx, value)` returns `None`
  to pass or raises `GuardrailTripped(reason)` to block. A trip ends the run with the new
  status `"blocked"`, sets `RunResult.guardrail` (`TrippedGuardrail`), journals
  `guardrail_tripped {name, reason, stage}` and emits a `guardrail_tripped` event
  (`GuardrailTriggered`). Guardrails fail closed: any other exception or return value fails
  the run.

#### Several agents

- Linked child runs: `Agent.run` / `Agent.stream` accept `parent=ctx`, and `Agent.as_tool()`
  uses it. A child's `RunContext` has `parent_run_id`, `parent_tool_call_id` and
  `depth`; its token, cost and time budgets are capped at what remains of the parent's; it
  inherits the parent's `on_ask` and the parent's hooks whose class sets
  `Hooks.propagate = True`. `RunStarted` carries `parent_run_id` and `depth`.
- `Limits.max_depth` (default 5) and `Limits.max_tool_calls` (default `None`).
- `Agent.copy(**changes)`: the same agent with some constructor arguments replaced.
- `qagents.multi.fan_out(agent, ...)`: a tool that runs one child run per task, in parallel
  (`max_parallel`, `max_tasks`, `timeout`), and returns a JSON list of answers and errors.
- `qagents.multi.route(classifier, routes, prompt, ...)`: pick a specialist with a classifier
  agent (`Literal`, `Enum` or plain-text answer) and run it; returns a `RouteResult`.
- Workflows: `Workflow.map(name, items, function, concurrency=...)` (one durable step per
  item), signals (`Workflow.send_signal`, `wait_for_signal`, `signals`), durable timers
  (`wait_for_timer`), `Workflow.status()` (`WorkflowStatus`, `AgentStepStatus`), and
  `WorkflowSuspended`, raised while a workflow waits. `run_agent` gains `results=` and
  `state=`. New journal kinds in workflow sessions: `workflow_agent_started`,
  `workflow_signal`, `workflow_waiting`, `workflow_timer`.

#### Signed approvals

- `Permissions(approval_secret=...)` signs every pause: `PendingApproval.token` and
  `PendingInput.token` hold an HMAC-SHA256 of the paused call. On resume, approvals
  (`Approval.for_call(pending)`, `Approval.token`) and results (`InputResult.for_call`,
  or a `{"token", "value"}` mapping) must carry the matching token, otherwise
  `ApprovalTokenError` is raised before anything runs. `permissions.approval_token` computes a
  token for other services.

#### Long-running agents

- `AgentRuntime(store, agents, ...)` runs agents in the background with leases and heartbeats:
  `start`, `resume`, `recover` (takes over dead or failed runs), `run_forever`, and a graceful
  `shutdown` that stops runs at safe points and leaves them resumable. `RunHandle` offers
  `wait()`, `events()` (several subscribers, with replay) and `cancel()`. `RunInterrupted` is
  raised for runs that were stopped and stay open. `max_failures` bounds automatic resumes of
  failed runs.
- Store extensions: `SessionStore.status`, `list_sessions`, `acquire_lease`, `renew_lease`,
  `release_lease` and `append_leased` (a write fenced by the lease; raises `LeaseError`).
  `SessionStatus` and `advance_status` describe where a session's latest run stands.
  `InMemoryStore` and `SQLiteStore` implement all of them.
- `qagents.stores.PostgresStore` (extra `postgres`): the same journal in PostgreSQL (jsonb),
  with database-clock leases, for workers on several hosts. The DSN comes from `dsn=` or
  `QAGENTS_POSTGRES_DSN`.

#### Context management

- `ToolResultClearing`: replaces old tool results with a placeholder, without a model request.
- `CompactorChain`: tries compactors in order; the first result wins.
- `trigger_fraction` (a share of the model's context window) next to `trigger_tokens`.
- `estimate_tokens`, `estimate_request_tokens`, `next_request_tokens` and
  `unpaired_tool_calls` in `qagents.context`.

#### Models

- `AnthropicModel` (extra `anthropic`) and the `"anthropic:"` prefix: Claude through the
  Messages API with streaming, thinking replay, prompt caching, server-side fallbacks,
  structured output, a price table, and error mapping that never replays a request that may
  have been billed.
- `FallbackModel([...])`: tries the next model when one refuses at admission (rate limit,
  server refusal, connection never made, billing or authentication); ambiguous errors do not
  fall back unless `allow_ambiguous=True`.
- `RateLimitedModel(model, max_concurrent=..., requests_per_minute=..., tokens_per_minute=...,
  max_wait_s=...)`: stays under provider limits and honours `retry_after`.
- `cassette(model, path)`, `RecordingModel` and `ReplayModel`: record real responses once and
  replay them offline; `CassetteMissError` says what changed. `QAGENTS_CASSETTE_MODE=replay`
  makes CI fail instead of making paid calls.

#### Tools of MCP servers

- `qagents.toolkits.mcp.MCPToolkit` (extra `mcp`): tools of an MCP server over stdio or
  streamable HTTP (protocol 2026-07-28, falling back to 2025-11-25). Annotations map to
  `read_only` and `idempotent`; tools that may be destructive ask for approval by default;
  definitions can be pinned by hash (`MCPToolChangedError`); binary content never reaches the
  model; progress notifications become `mcp_progress` events.

#### Tracing and evaluation

- `qagents.integrations.mlflow.MLflowHooks` (extra `mlflow`): one MLflow trace per top-level
  run, with AGENT, CHAT_MODEL and TOOL spans, sub-agents nested under the tool call that
  started them, token usage and cost, session grouping and status tags. Content is recorded
  only with `record_content=True`. `enable(...)` is the one-line setup.
- `OTelHooks` nests child runs under the parent's `execute_tool` span and gains attributes for
  run ids, depth, cached input tokens, deferred calls and pending calls.
- `qagents.evals`: datasets (`Case`, `Dataset`, JSONL), eleven evaluators including
  `ToolCalled`, `MaxCostUSD` and `LLMJudge`, `evaluate` / `evaluate_sync` with concurrency and
  repeats, `EvalReport` (markdown and JSON), `compare`, `regressions`, and `log_to_mlflow`.
  `JSONSchemaValid` needs the extra `evals`.

#### Interfaces

- `qagents.integrations.agui`: AG-UI event streams for CopilotKit and other AG-UI clients
  (`to_agui_events`, `agui_sse`, `AGUIProjector`, `agui_resume`), with interrupts for
  approvals and deferred calls and three modes for sub-agent events.
- Command line: `qagents chat`, `qagents run "prompt"` and `qagents models qcloud` (also
  `python -m qagents ...`), with streaming, inline approvals, tool sets, sessions and cost
  limits.

#### Package

- New top-level exports: `AgentRuntime`, `RunHandle`, `fan_out`, `route`, `RouteResult`,
  `FallbackModel`, `RateLimitedModel`, `cassette`, `CompactorChain`, `ToolResultClearing`,
  `InputResult`, `PendingInput`, `SessionStatus`, `TrippedGuardrail`, `WorkflowStatus`,
  `DeferToolCall`, `ModelRetry`, `GuardrailTripped`, `ApprovalTokenError`, `LeaseError`,
  `RunInterrupted`, `WorkflowSuspended`. `AnthropicModel`, `MCPToolkit`, `MLflowHooks` and
  `PostgresStore` are also importable from `qagents` but are resolved on first access, so
  `import qagents` never imports an optional dependency.
- Error classes of the new modules are defined in `qagents.errors` (`LeaseError`,
  `RunInterrupted`, `WorkflowSuspended`, `CassetteMissError`, `MCPConnectionError`,
  `MCPToolChangedError`) and stay importable from the modules that raise them.
- Every new model-facing text is a constant in `qagents.prompts`, including those of the
  toolkits and integrations (`FAN_OUT_*`, `TOOL_RESULT_CLEARED`, `MCP_*`, `INPUT_CANCELLED`,
  `JUDGE_*`, `SUBAGENT_TASK_DESCRIPTION`, `OUTPUT_TOOL_*`, `TOOL_CALL_LIMIT`,
  `TOOL_INPUT_ABANDONED`, the todo and `ask_user` texts).
- A `qagents` console script.

### Changed

- `Hooks.on_run_end` is now also called when a run is stopped by cancellation or by closing
  its stream early, with the `asyncio.CancelledError` or `GeneratorExit` as `error`. The run
  stays open in the journal, as before. Tracers therefore close the spans of such runs.
- The compactor is now also asked before the first request of a run and of a resume; it
  receives an empty response with no usage, so the whole history is estimated. Long sessions
  whose runs each make one request are now compacted.
- `SummarizingCompactor` counts the messages added since the last response (usually tool
  results), so it can trigger earlier; with providers that report no usage it estimates the
  history instead of never triggering. Summaries are marked with metadata
  `{"compaction": "summary"}`, system messages at the start of the history are kept, and a
  lone earlier summary is not summarised again. `trigger_tokens` is optional, but one of
  `trigger_tokens` and `trigger_fraction` is required.
- A model response with finish reason `"content_filter"` (a provider refusal) ends the run
  with status `"completed"` and stop reason `"content_filter"`, without a retry; its tool calls
  are not run. The output is the refusal text for a plain-text agent and `None` when
  `output_type` is set.
- `Limits.max_cost_usd` of zero or less stops a run before its first request.
- `Agent.as_tool()` starts linked child runs: nesting deeper than `Limits.max_depth` (5) is
  refused, and a child's budgets are capped at what remains of the parent's.
- A hook object listed more than once for one run now runs once.
- `OTelHooks.propagate` is `True`: attached to a parent agent, it also traces sub-agents, as
  one trace. A deferred tool call is no longer closed as an error.
- `Workflow.run_agent` accepts only `deps`, `limits` and `settings` as extra options
  (others used to fail on resume), raises `SessionError` for `results=` on a step with no open
  run, refuses two concurrent calls with the same step name, and returns a replayed step's
  output in the agent's `output_type` (0.1 returned a dict).
- `SQLiteStore` adds a `sessions` table and records schema version 1 in `PRAGMA user_version`.
  A 0.1 database is migrated in place on first open; the migration is not reversible, and every
  process that writes the file should run 0.2. Several processes opening a new database at once
  no longer fail with "database is locked".
- A new run, or a cancel, closes deferred calls that are still waiting with
  `prompts.TOOL_INPUT_ABANDONED`, the same way it closes pending approvals.
- `PendingApproval` gains `token` (`None` unless pauses are signed), so its JSON dump has one
  more key.

### New optional dependencies

| Extra | Installs | For |
|---|---|---|
| `anthropic` | `anthropic>=2.8,<3` | `AnthropicModel` |
| `mlflow` | `mlflow>=3.5,<4` | `MLflowHooks`, `qagents.evals.log_to_mlflow` |
| `mcp` | `mcp>=2.3,<3` | `MCPToolkit` |
| `postgres` | `psycopg[binary]>=3.2,<4`, `psycopg-pool>=3.2,<4` | `PostgresStore` |
| `evals` | `jsonschema>=4.18` | `qagents.evals.JSONSchemaValid` |
| `all` | all of the above plus `otel` | everything except `qcloud` |

`otel`, `qcloud`, `docs` and `dev` exist as before; `dev` now includes `all`. The core
dependencies are unchanged (`httpx`, `pydantic`).

## [0.1.0] - 2026-11-11

Initial release.
