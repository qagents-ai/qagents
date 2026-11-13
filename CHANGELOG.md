# Changelog

All notable changes to qagents. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (before 1.0, a minor version may change behaviour;
every such change is listed under "Changed").

## [Unreleased]

### Added

#### External agents

- `qagents.external`: `ExternalAgent`, the contract for another agent program that runs its
  own loop and is driven over a control protocol. Use one as a tool (`as_tool()`, a child of
  the calling run: depth, budgets, cancel, `SubagentEvent`s, a footer the model reads), as a
  workflow step, or directly (`stream()`, `run()`). Every permission request it sends is
  decided by your `Permissions` on `"<tool>.<agent tool>"` and never answered "allow always";
  a `can_use_tool` or `on_ask` that raises answers "reject" (that one action is refused, the
  turn goes on), and a decision that times out answers "cancelled". Postures `rules`, `pause`
  (pauses the calling run, answered with `ExternalApproval`) and `live`. One turn at a time
  per workspace and per session, git change tracking (`changed_files`, `diff()`), and crash
  re-attach for resumable agents. Through `as_tool()` the model can continue only a session
  that this tool opened in this qagents session (any other id is a tool error that says so);
  your own code may pass any session. The footer of a turn that never began names no session.
  `ExternalResult`, `ExternalCapabilities`, `ScriptedExternalAgent` for offline tests, and
  the `external_event` (`ExternalAgentEvent`) kinds. The external-agent names are also
  importable from `qagents` (resolved on first access).
- `qagents.integrations.acp.ACPAgent`: a generic Agent Client Protocol v1 client (stdio,
  `session/new`/`resume`/`load`/`close`, `set_config_option`, permission requests, cancel with
  a process-tree kill), with `MCPServerStdio` and `MCPServerHTTP` for the agent's sessions. It
  authenticates only with the `auth_method` you choose, never by itself.
- `qagents.integrations.devvi.DevviAgent`: Devvi (devvi.sh) as a coding worker over ACP (the
  default) or Devvi's SDK protocol (token usage and cost with `TokenPricing`, client-chosen
  session ids, re-attach from Devvi's session log). Its own home, an explicit environment,
  the session-log upload and telemetry off, `expose_tools` (qagents tools for Devvi's model
  over MCP; needs the `mcp` extra), and `python -m qagents.integrations.devvi doctor`.
- `qagents.integrations.mcp_server`: serve qagents tools over MCP, bound to a run (Streamable
  HTTP on 127.0.0.1 with a bearer token; the run's rules, journal, hooks and limits apply) or
  over stdio from a JSON spec.
- `RunContext.checkpoint()` and `checkpoints`: a tool call journals a fact (`checkpoint(key,
  value)`, written as a `tool_checkpoint` entry before it returns, so before the tool acts on
  it) and reads, on a later attempt of the same call, what earlier attempts recorded
  (`checkpoints`, a deep copy, kept per operation id across a resume). A call with a recorded
  result still never runs again, whatever it checkpointed. The entry can carry usage.
- `RunContext.call_tool(tool, arguments, *, source, approval_timeout_s=None, tags=None,
  metadata=None)`: a call from outside the model (the MCP tool server uses it) made as an
  ordinary tool call of the run, with the loop's own validation, `Permissions` and `on_ask`
  ("ask" without an `on_ask` is refused), journal entries, hooks, events, `Limits` and
  usage. It writes nothing and is refused once the run is cancelled or stopped, its journal is
  closed, or the calling tool call has ended; `tags` cannot set `usage`.
- `RunContext.permissions`, `save_state()` and `counted()`.
- The external-agent errors (`ExternalAgentError` and its subclasses, `ExternalToolError`)
  live in `qagents.errors`; `qagents.external` re-exports them.
- `ExternalAgentEvent` is part of the `qagents.events.Event` union.
- The AG-UI projection opens tool calls that arrive without a model response (an external
  agent's, a served MCP call's) and ends a direct external turn at its `turn_finished` event.

#### Tracing of external agents

- `OTelHooks` records two span events for an external agent run through `as_tool()`, on the
  `execute_tool` span of that call (or on the run span when the event names no call):
  `qagents.external.permission_decided` and `qagents.external.turn_finished` (names in
  `qagents.tracing.EVENTS`, attributes in `ATTR`).
  - `permission_decided` carries `gen_ai.agent.name` and
    `qagents.external.permission.approved`; valid fields supplied by the event add
    `qagents.external.permission.tool`, `.tool_call_id`, `.outcome`,
    `.decision` and `.decided_by`.
  - `turn_finished` carries `gen_ai.agent.name`, `gen_ai.usage.input_tokens`,
    `gen_ai.usage.output_tokens` and `qagents.external.tool_calls`. Valid optional fields add
    `qagents.external.status`, `qagents.external.session_id`, `qagents.external.backend`,
    `qagents.usage.cached_input_tokens`, `qagents.cost_usd` and
    `qagents.external.changed_files_count`, and `qagents.external.stop_reason` when the
    stop reason is a known word (`qagents.external.STOP_REASONS`, `permission`, `closed`, and
    the adapters' `max_turn_requests`, `forked`, `blocked` and `interrupted`).
  - Only with `record_content=True`: `qagents.external.permission.arguments` (JSON),
    `qagents.external.permission.reason`, `qagents.external.changed_files`, and any other stop
    reason (an error turn's stop reason is the agent's message).
  - A field of the wrong type (a custom adapter's event data) is left out, so tracing never
    fails a run. The tool span stays open until the call's `tool_call_finished` event, then
    ends at the time its tool returned; events that arrive later carry that time. A calling
    run that is cancelled mid-turn records no `turn_finished`. `MLflowHooks` records neither
    event: export the `OTelHooks` spans to MLflow's `/v1/traces` endpoint instead.

#### Docs for coding agents

- `python -m qagents.docs_mcp`: an MCP server (stdio, `qagents[mcp]`) with the docs of the
  installed qagents for coding agents: `search_docs`, `read_page`, `list_pages`, `get_api`,
  `list_examples`, `read_example`, `check_code`, resources and two prompts. The same answers
  on the command line (`search`, `read`, `pages`, `api`, `examples`, `example`, `check`,
  `llms`, `skill`), also as `qagents docs ...`. Wheels bundle the docs, so the server answers
  from the docs of the installed version.
- `AGENTS.md`, a Claude Code skill (`skills/qagents/SKILL.md`), `docs/coding-agents.md`, and
  `scripts/build_llms_txt.py` for `docs/llms.txt` and `docs/llms-full.txt`.
- `search_docs` ranks plain questions better. A section whose heading says what the query
  says scores higher; a missing word costs less the more common it is; `agent` counts a
  quarter next to other words; integration pages weigh half unless the query names their
  topic (each page's topic words, such as `claude`, `codex`, `gemini` or `worker` for the
  external-agent pages, are data in `docs_mcp/corpus.py`); API reference stubs and the
  research page weigh 0.6; one plain word that names a guide section (`tools`, `hooks`) finds
  the guide first, any other plain word (`evals`, `agui`) the module. The tokenizer folds
  `-ied` endings and `code`/`coding`, and splits `SQLiteStore` into `sqlite`. Measured over
  `tests/docs_mcp/search_eval.toml` (86 questions, the set the ranking was tuned on): top-1
  68/86 (0.79); top-3 80/86 (0.93). Over `search_heldout.toml` (73 questions):
  top-1 53/73 (0.73); top-3 60/73 (0.82). This second set is a regression check rather
  than a pristine holdout: synonym ablation inspected its scores and one question
  informed MCP-server topic words. "agent tools" and "how do I make an agent with tools"
  put `concepts/tools.md` first; "connect a copilotkit frontend" finds the AG-UI API
  first. The benchmark is opt-in: `pytest tests/docs_mcp -m docs_benchmark`.
- `get_api` on a module constant, and a module's member list (which `llms-full.txt` is built
  from), show the constant's docs: the string right after the assignment, or a `#:` comment
  above it or at the end of its line. A constant that a package re-exports shows the docs of
  the module that defines it (for example `qagents.external.APPROVAL_KEY`).

#### Docs and package

- Docs: `concepts/external-agents.md`, `integrations/devvi.md`, `integrations/acp.md`,
  `integrations/mcp-server.md`, `reference/external.md`; `examples/devvi_coder.py`. The logo
  (`assets/logo/`) in the README and on the docs site.
- Extras `acp` and `devvi` (no dependencies).

### Changed

- A tool may return a `ToolResultPart` to set `is_error` and `metadata`; `ToolError.metadata`
  is copied into the result.
- **Behaviour change:** an idempotent tool that is running when the run is cancelled or times
  out is now reported as interrupted (outcome unknown, `TOOL_INTERRUPTED`), not cancelled
  (`TOOL_CANCELLED`), so the model is told to check before calling it again. Only a read-only
  call is harmless when it is cut short. This applies to the model's calls and to served
  calls. After a crash nothing changes: an idempotent call that had started is still run
  again on resume.
- Result metadata that is not JSON is logged and replaced by `{"metadata_error": ...}`; the
  tool's content and `is_error` stay as the tool gave them.
- A sync tool's thread that is abandoned when its call is cancelled (not only on a timeout) is
  remembered by the run, or by a standalone tool server, so a later call there that must run
  alone waits for it.
- Every model-facing string of the new modules is in `qagents.prompts` (`EXTERNAL_*`,
  `TOOL_SERVER_*`, `ACP_*`, `DEVVI_*`).
- **Model-facing:** through `as_tool()`, a session that the tool did not open is refused with
  the new `prompts.EXTERNAL_AGENT_UNKNOWN_SESSION`, inside `EXTERNAL_AGENT_FAILED`: "{name}
  failed: session {session} was not opened by {name} in this conversation; leave session out
  to start a new one". It said "session {session} does not exist; ..." (`DEVVI_UNKNOWN_SESSION`)
  before. Once the tool keeps its 50 latest sessions, the refusal is the new
  `EXTERNAL_AGENT_FORGOTTEN_SESSION` instead ("session {session} is not among the last
  {limit} sessions that {name} opened or continued in this conversation, which are the only
  ones it can continue; ..."), because the session may be an older one that was dropped.
  `DEVVI_UNKNOWN_SESSION` remains the Devvi SDK stop reason for a session Devvi does not have:
  one your code passed, or one the tool opened that Devvi has lost.
- 116 more `qagents.prompts` constants, each with a docstring, for texts that were built
  inline. With their default values every text reads as before, apart from the refusals above.
  - The `as_tool()` result layout: `EXTERNAL_AGENT_ERROR_RESULT` (`"{message}\n\n[{footer}]"`,
    every turn that did not complete) and `EXTERNAL_AGENT_STOPPED_WITH_TEXT`
    (`"{message}\n\n{text}"`, a stopped turn that has text); `EXTERNAL_AGENT_RESULT` is not
    reused for them.
  - Joiners and markers: `EXTERNAL_AGENT_DESCRIPTION_WITH_HINT`,
    `EXTERNAL_AGENT_TEXT_BLOCK_SEPARATOR`, `EXTERNAL_AGENT_TASK_SEPARATOR`,
    `EXTERNAL_AGENT_LINE_SHORTENED`, `EXTERNAL_AGENT_FOOTER_TOOL_COUNT`,
    `EXTERNAL_AGENT_FOOTER_LIST_SEPARATOR`, `EXTERNAL_AGENT_NO_ARGUMENTS`,
    `EXTERNAL_AGENT_DECLINED_REASON`, `EXTERNAL_TOOL_DESCRIPTION`, `EXTERNAL_TOOL_NEVER_RUN`,
    `EXTERNAL_SETUP_LIST_SEPARATOR` and `EXTERNAL_SETUP_LIST_NONE`.
  - ACP: the tool-call content placeholders (`ACP_DIFF_PLACEHOLDER`,
    `ACP_TERMINAL_PLACEHOLDER`, `ACP_MEDIA_PLACEHOLDER`, `ACP_RESOURCE_LINK_PLACEHOLDER`,
    `ACP_RESOURCE_PLACEHOLDER`, `ACP_TOOL_CONTENT_SEPARATOR`), `ACP_JSONRPC_ERROR`, and the
    first lines of the ACP client's setup errors, which the model can read inside
    `EXTERNAL_AGENT_UNAVAILABLE` (`ACP_SETUP_FAILED`, `ACP_STOPPED_AFTER_HANDSHAKE`,
    `ACP_HANDSHAKE_*`, `ACP_AUTH_*`, `ACP_SERVER*`,
    `ACP_PROTOCOL_VERSION`, `ACP_NO_WORKING_DIRECTORY`, `ACP_PROMPT_NEEDS_AUTH`,
    `ACP_SESSION_*`, `ACP_CONFIG_*`, `ACP_NO_*_OPTION`).
  - Devvi: `DEVVI_UNKNOWN_TURN_END`, `DEVVI_ERROR_WITHOUT_MESSAGE`, and the first lines of the
    launcher, patch, start and `expose_tools` errors (`DEVVI_NODE_*`, `DEVVI_CLI_*`,
    `DEVVI_VERSION_*`, `DEVVI_PATCH_*`, `DEVVI_MCP_*`, `DEVVI_DID_NOT_START`,
    `DEVVI_EXPOSE_*` and others). The developer detail after the first line stays in the code.
  - The MCP tool server: `TOOL_SERVER_DUPLICATE_NAME`, `TOOL_SERVER_DEFERRED_TOOL`,
    `TOOL_SERVER_NOT_A_TOOL`, and `UNKNOWN_TOOL_SEPARATOR`, `UNKNOWN_TOOL_NONE`,
    `APPROVAL_ARGUMENTS_CHANGED_RESULT` and `TOOL_FAILED_EXCEPTION`; the agent loop and
    fan-out task failures now use the shared constants too.
  - Shared transport setup and replies: `EXTERNAL_PROCESS_*` and `EXTERNAL_RPC_*`.
- The Devvi SDK re-attach also finds a task of several text parts in Devvi's session log (it
  matches the text Devvi logged), so after a crash such a prompt is not sent a second time.
- A text shortened with `EXTERNAL_AGENT_LINE_SHORTENED` leaves only the marker when the
  limit is shorter than the marker; otherwise the marker counts toward the limit.

### Not in this version

- ACP's unstable `usage` field is not read, so ACP agents are unmetered.
- Headless model authentication for Devvi, the devvi.sh platform as a `Model`, and qagents as
  an ACP agent server.

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
- `CompactorChain`: asks each compactor in order, each on the previous one's replacement.
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
  `read_only` and `idempotent`; tools not declared read-only ask for approval by default;
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
  `MaxDepthExceeded`, `ToolTimeout`,
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

#### From the pre-release review

An adversarial review of 0.2 before its release found defects in every area. Their fixes change
these behaviours; the fixes themselves are listed under "Fixed".

Behaviour you will notice:

- `qagents.multi.fan_out` is no longer read-only by default (`read_only=False`): a run resumed
  after an interruption does not repeat a fan-out that may have acted, and several fan-out
  calls of one turn run one at a time under `parallel_tools="read_only"`. Pass
  `read_only=True` when every tool of the workers is read-only. Its `timeout` defaults to
  `multi.FAN_OUT_TIMEOUT_S` (600 s) instead of `Limits.tool_timeout_s`.
- `MCPToolkit` asks for approval before every tool the server does not declare read-only (new
  default `approval="side_effects"`; `"destructive"` keeps the old rule). `pins` is an
  allow-list (`allow_unpinned=True` to opt out), `approval="all"` ignores the server's
  annotations, and `definition_hash` covers the tool's title, so pins of titled tools must be
  taken again from `definition_hashes()`.
- `web_tools()` without `allow_hosts`: `fetch_url` asks for approval and is not read-only
  (`requires_approval=False` restores the old behaviour).
- Signed pauses use a new token payload (version 2, RFC 8785 canonical JSON, bound to the
  session, the call's position in the run and the kind of pause). Pauses signed before this
  change no longer verify; restart those runs. A call whose arguments cannot be signed (NaN, an
  infinity, broken Unicode, an integer a double cannot hold exactly) is refused with
  `prompts.UNSIGNABLE_CALL` instead of crashing the run.
- One run per session at a time: a run or resume while another run of the session executes
  raises `SessionError`. On a store with leases a plain `Agent.run` holds a 30 s lease
  (`agent.RUN_LEASE_TTL_S`, owner `qagents-run:<uuid>`), so `AgentRuntime.recover()` leaves it
  alone. Close a stream you stop early before the next run of its session.
- `Limits` count the whole run across resumes (`max_turns`, `max_total_tokens`,
  `max_cost_usd`, `max_tool_calls`; `RunResult.usage` and `turns` too); `timeout_s` applies to
  each `run()` or `resume()` call. Limits passed as `limits=` are journaled in `run_started`
  (and by `AgentRuntime` as `runtime_limits`), so a recovered run keeps them.
- `Limits.max_cost_usd` fails closed: a response with an unknown cost (`Usage.unpriced_requests`,
  set by `OpenAIChatModel` and `QCloudModel` when the server reports no usage) stops the run.
  `max_tool_calls` counts only calls allowed to run; refused, invalid, waiting and declined
  calls do not count. `max_depth` is inherited down the chain of runs (`MaxDepthExceeded`), and
  children that run at once share the parent's token and cost budget live.
- `Agent.as_tool()` returns the accepted `result.output` (validators and final-answer tool
  included), reports a refused or too-deep child as an error result
  (`prompts.SUBAGENT_TOO_DEEP`), and takes `timeout=`.
- Tool arguments are validated before anyone is asked, so `can_use_tool`, `on_ask`,
  `before_tool` and pauses only see valid arguments; a `before_tool` hook can no longer repair
  invalid ones. Approvals: the permission rules are checked first and a deny always wins,
  approvals for calls that are not open are ignored, edited arguments are checked again,
  journaled (`approval_decided`) and disclosed to the model, and `approvals=` values must be
  `True`, `False` or an `Approval`.
- Cancel and timeouts tell the model what happened: a call that never started gets
  `TOOL_NOT_EXECUTED`, a running read-only or idempotent call `TOOL_CANCELLED`, any other
  running call `TOOL_INTERRUPTED` (outcome unknown). `TOOL_TIMEOUT` is reworded. Sync tools run
  on a pool qagents owns (`tools.TOOL_THREADS`); a timed-out call raises `tools.ToolTimeout`
  and its thread is abandoned, not killed.
- Model streams and compaction run in the task that iterates the stream: a provider's own
  `asyncio.timeout` works, and an outer cancellation is no longer swallowed.
- `AgentRuntime.start()` journals the run before it returns, so errors of its preparation
  (nothing to run, an input guardrail that raises, a session in use) are raised by `start()`.
  `recover()` does not resume a run whose last failure was ambiguous (unless the run's limits
  set `retry_ambiguous`), a `run_failed` without flags, or one that needs a person; after
  `max_recoveries` (new, default 5) recoveries without progress it marks the run failed
  (`RecoveryLimitReached`). `shutdown()` waits for calls in flight and no longer treats
  `RunStarted` as a safe point.
- `Workflow.map` runs sync functions in threads (functions that need the event loop must be
  async); `Workflow.completed_steps()` returns a copy.
- AG-UI: error text, guardrail reasons, the text of tool exceptions (`TOOL_ERROR_MESSAGE`) and
  sub-agents' usage and reasoning data are not sent unless asked for (`expose_errors=True`,
  `guardrail_reasons=True`); `agui_resume` refuses edited arguments unless
  `allow_argument_edits=True`; a run blocked by an output guardrail always ends with a
  `MESSAGES_SNAPSHOT` without the blocked answer.
- CLI: `save_note`, `delete_note` and `fetch_url` ask first, approval prompts show the whole
  call, model and tool text is escaped before it reaches the terminal, and `--max-cost` and
  `/usage` count runs that fail part-way and paused runs once.
- `OpenAIChatModel` and `QCloudModel`: 4xx errors other than 401/402/403/408/409/429 are
  `ModelBadRequestError` (not retried, not ambiguous), 408 is a retryable `ModelTimeoutError`,
  redirects are not followed, and a missing usage gives `cost_usd=None`. Error messages never
  quote credentials in `base_url`; an `api_key` or header with control or non-ASCII characters
  is refused at construction.
- Cassettes: secret detection is broader (keys are matched in snake_case form), so more settings
  are redacted and request fingerprints can change: record affected cassettes again.
  `RecordingModel(overwrite=True)` replaces the file only on its first recorded response, and
  `QAGENTS_CASSETTE_MODE=replay` also refuses `cassette(mode="record")`.
- `AnthropicModel`: `thread` and `mcp_servers` are no longer allowed in `extra`; unsupported
  combinations (effort on Haiku 4.5, `xhigh` on 4.6 models, fallbacks on Haiku 5.5,
  `thinking_display="updates"` on a model without it, a `max_tokens` above the model's limit)
  fail when the model is built; open maps and bare objects in strict or structured schemas are
  refused before sending; a turn still paused after `max_continuations` ends with finish reason
  `"length"`; a tool call cut off by `max_tokens` is never run; usage includes every billed
  iteration.
- `SummarizingCompactor`: `max_summary_tokens` defaults to `None` (the adapter's limit); a
  summary that is cut off, refused or empty raises `ModelResponseError` instead of replacing the
  history; its request goes through the run's hooks. `CompactorChain` asks every compactor in
  turn, each on the previous one's replacement. Compactors count the system prompt, tools and
  output schema before the first request, and reasoning that is not sent back is not counted.
- With `record_content=False`, `OTelHooks` and `MLflowHooks` keep error text (exception
  messages, stack traces, failed tool results) out of traces: spans get the exception type only.
- `qagents.evals`: `Contains`, `Regex`, `JSONSchemaValid` and `IsInstance` check the accepted
  answer (`EvalContext.answer_text`, `output`), and runs or judges that raise keep their usage.
- `file_tools`: `search_files` runs its regex in a child `python -I` process stopped at
  `search_timeout` (new `max_file_bytes`, `search_timeout`, `search_python`); listings skip
  Windows junctions. `notes_tools` refuses linked notes and writes notes atomically (mode 0600
  on POSIX).
- `qagents.MCPToolkit` and `qagents.toolkits.MCPToolkit` raise `AttributeError` (naming the
  `qagents[mcp]` extra) when the MCP SDK is missing, and `dir()` leaves them out, so `hasattr`,
  `help(qagents)` and `inspect` work; `from qagents import MCPToolkit` still raises
  `ImportError`.

Other API changes:

- New: `errors.MaxDepthExceeded` and `tools.ToolTimeout` (both exported from `qagents`),
  `models.is_secret_setting_key`, `permissions.canonical_json`, `Session.append_entries`,
  `RunContext.limits` and `remaining_budget()`, `Usage.unpriced_requests` and `cost_complete`,
  `ModelResponse.context_tokens`, `AgentSwitched.tool_call_id`, the `OutputRejected` event,
  `AgentRuntime(max_recoveries=)`, `route(classifier_session=)` and `RouteResult.used_default`,
  `Workflow.run_agent(on_event=)`, `FallbackModel(model_settings=, cooldown_s=)` and
  `metadata["fallback"]` on its responses, `QCloudModel(catalog_timeout=)`,
  `RecordingModel(redact_keys=)`, `AGUIProjector(hold_final_answer=, guardrail_reasons=)`,
  `to_agui_messages(expose_errors=)`, `web_tools(requires_approval=)`, and the prompt constants
  `APPROVAL_ARGUMENTS_CHANGED`, `UNSIGNABLE_CALL`, `OUTPUT_VALIDATOR_RETRY`,
  `OUTPUT_TOOL_TEXT_INSTRUCTIONS`, `TOOL_TIMEOUT_ABANDONED`, `TOOL_NOT_STARTED`,
  `TOOL_THREADS_EXHAUSTED`, `TOOL_WAITED_FOR_EARLIER`, `TOOL_CALL_TRUNCATED`,
  `SUBAGENT_TOO_DEEP` and the `PERMISSION_REASON_*` texts (now used by `Permissions`).
- `output_mode="tool"` without an `output_type` accepts a plain-text answer; native mode sends
  a schema whose root is not an object wrapped as `{"value": ...}`.
- Journal: `run_finished` holds `{status, stop_reason, output, text}`; `run_failed` adds
  `error_type`, `message`, `retryable` and `ambiguous`; assistant messages carry `usage`,
  `model`, `provider`, `turn` and `metadata["finish_reason"]`; usage added through
  `ctx.add_usage` rides on the next entry as `added_usage`; new kinds `approval_decided`,
  `runtime_limits` and `run_recovered`. `derive_state` ignores entries a run writes after its
  own `run_finished`, and marks an answer an output guardrail blocked with
  `metadata["blocked"]`.
- `SQLiteStore` uses `PRAGMA synchronous=FULL` and its own executor thread; `PostgresStore`
  refuses NaN, infinities and lone surrogates with `SessionError` and runs no DDL when its
  schema already exists. Every store supports `async with`.

### Fixed

Defects found by the pre-release review of 0.2, by area. Each fix has a regression test.

- **Agent loop and journal.** A crash can no longer leave a half-written start of a run, turn,
  refusal or handoff: related entries are written in one append, and resume repairs journals
  written in pieces. A finished tool call is never run twice: one run per session, fenced
  writes, and an approved call that started before a crash still counts. Usage, turns and tool
  calls are restored on resume, so pausing cannot reset a limit; a declined or waiting call, or
  one that never started, no longer uses up `max_tool_calls`. `on_run_end` runs once, after
  `on_run_start`, also when an `on_event` hook raises. A stray `CancelledError` from a tool, or
  a tool that emits `ToolCallFinished`, no longer hangs or ends a tool group early; nothing
  starts after a cancel or timeout, including during retry back-off and compaction; state
  changed in `before_tool` is journaled before the tool runs; an invalid `state=` fails before
  anything is written; handoff names must be unique.
- **Permissions and signed pauses.** Tokens cover the session, the call's position and the
  pause kind, so a pause cannot be copied into another session or replayed after its run
  finished; an integer that a double cannot hold exactly is refused rather than signed
  ambiguously; refusal reasons come from `qagents.prompts`.
- **Tools.** Tool timeouts cover an awaitable a sync function returns, a tool's own
  `TimeoutError` is reported as a failure, abandoned sync threads are bounded, and later calls
  that run one at a time never overlap them.
- **Child runs.** A child's usage reaches the parent once, however the child ends; parallel
  children cannot overshoot the parent's budget; depth limits apply down the whole chain.
- **Runtime, workflows and Postgres.** Shutdown races no longer launch runs after `shutdown()`
  or cut a tool call that already started; a slow preparation keeps its lease (the heartbeat
  starts with the lease); limits survive recovery; crash loops are bounded; a cancelled queued
  run is closed; a finished run keeps its result when its lease release is cancelled; a
  recovery that backs out writes nothing; interrupted runs say whether their `state=` and
  `limits=` were kept and whether `recover()` will resume them. `Workflow.run_agent` never runs
  a completed but unrecorded step again. `PostgresStore`: appends and deletes no longer race, a
  role with only table privileges works, and polls no longer write a lease row per candidate.
- **Multi-agent helpers.** `fan_out` and `route` report a refusing worker as an error; `route`
  reads quoted or Markdown route names.
- **Model adapters.** OpenAI-compatible and QCloud: error classes match the HTTP status,
  tool-call fragments without an index are assembled correctly, a stream labelled as JSON is
  read, a completion without a finish reason is refused instead of taken as complete, refusals
  map to `content_filter`, multi-part text is not run together, and strict schemas get
  `additionalProperties: false`. Anthropic: a billed continuation is never sent again, server
  tool input is kept and replayed, costs are per iteration and model, compactors read the size
  of the last request (`ModelResponse.context_tokens`) instead of the billed sum, thinking
  blocks are replayed where the API allows and a rejected signature is retried without them
  outside an open tool round, and foreign or repeated tool-call ids are made valid.
  `RecordingModel` redacts more secrets and no longer deletes a cassette it never wrote;
  `RateLimitedModel` bounds the whole wait; `FallbackModel` says which model answered.
- **AG-UI and CLI.** Signed pauses can be declined from AG-UI; dict events project like event
  models; interleaved tool-call deltas and handoff results are sent correctly, also on resume;
  a blocked answer is never sent again in a later run's snapshot, and with
  `hold_final_answer=True` a native sub-agent's answer that passed is sent. The CLI no longer
  clips approval prompts or double-counts the usage of resumed runs.
- **Tracing, compaction and evals.** Error text stays out of traces without
  `record_content=True`; summary requests are traced; a cut-off summary no longer replaces the
  history; a system message already in the history is counted once; MLflow no longer uploads
  `data:` images; evaluators read the accepted answer.
- **Toolkits.** File and notes tools never follow junctions or links out of their folder, and
  work on Linux and macOS (where `stat` has no junction constant); `search_files` cannot freeze
  the process; `fetch_url` checks the host name it connects to, bounds decompression and
  refuses NAT64 and other internal addresses; `MCPToolkit.aclose()` waits for a connect in
  progress.
- **Package.** `help(qagents)`, `hasattr` and `inspect` work without the MCP extra.

### Known limitations

Not fixed in 0.2; each is documented where it applies.

- A writer who can append to the session store can replay an answered signed pause inside a run
  that is still open, by appending the run's entries again from its `run_started`. Signed
  pauses protect against forgery, alteration, and copies across sessions or finished runs.
- Output guardrails check the completed answer, whose text has been streamed by then. Hold it
  until `run_finished` (`AGUIProjector(hold_final_answer=True)` for AG-UI) or use `agent.run`.
- An OpenAI strict tool with defaulted parameters is sent with a partial `required` list, which
  OpenAI rejects (vLLM and SGLang accept it).
- The in-flight model request of a worker that died is sent again on recovery: the journal
  cannot tell it from one never sent.
- `Permissions(mode="read_only")` allows, without asking, a read-only tool that also sets
  `requires_approval=True`.
- `fetch_url` resolves a host twice (DNS rebinding); a note that is a hard link is read through.
- A timed-out `search_files` keeps its child process until `search_timeout`.
- The text of a `fan_out` worker's exception reaches the calling model and AG-UI clients
  (credentials in `base_url` are no longer part of it).
- Hooks cannot tell a compactor's summary request from a loop request.
- A store subclass that overrides only `append` runs plain runs without a lease, so only runs in
  the same process are kept apart.
- `AnthropicModel` replaces an `anthropic-beta` header set in the client's `default_headers`; an
  exhausted `pause_turn` is reported as finish reason `"length"`; the usage of a billed pause
  segment is lost when its continuation fails; open-map schemas fail when the request is built,
  not when the agent is.
- By design, and documented: input guardrails check only `prompt=` (not `history=`); a run whose
  output retries are used up raises `OutputValidationError`; a guardrail's reason is not sent to
  the model; skipped calls keep an `OUTPUT_TOOL_SKIPPED` result; `Workflow.run_agent` runs a
  step whose run did not complete again, and checks permissions again on a paused step;
  `MLflowHooks` uses MLflow's process-wide tracking URI; there is no streaming `route()`, no
  public crash simulation for tests, and no per-case evaluator scoping.

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
