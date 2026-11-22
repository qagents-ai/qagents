# qagents examples

Twenty-two runnable agent patterns, each one module in [`patterns/`](patterns), each a qagents
version of a well-known LangGraph or CrewAI example. Every pattern runs against a real model on
Anthropic (Claude Opus 5.5 by default) or QCloud (an OpenAI-compatible catalog model, GLM by
default), can be recorded once and replayed offline, and has an offline test in
[`tests/examples/`](../tests/examples).

Each module's docstring says what it shows, which LangGraph or CrewAI pattern it mirrors, how the
qagents version differs, and how to run it. Read the code: it is meant as documentation.

- [Run one example](#run-one-example)
- [Options](#options)
- [Run them all](#run-them-all)
- [Gallery](#gallery)
- [If you know LangGraph](#if-you-know-langgraph)
- [If you know CrewAI](#if-you-know-crewai)
- [Offline tests](#offline-tests)
- [Other examples in this folder](#other-examples-in-this-folder)

## Run one example

Run every command from the project root, in an environment with qagents installed
(`pip install -e ".[dev]"` installs every optional extra the examples use).

**Anthropic** (the default provider). Needs the `anthropic` extra and a key:

```bash
export ANTHROPIC_API_KEY=...            # PowerShell: $env:ANTHROPIC_API_KEY = "..."
python -m examples.patterns.react_assistant
python -m examples.patterns.react_assistant --model claude-opus-5-5
```

`ANTHROPIC_AUTH_TOKEN`, or a profile from `ant auth login`, works as well.

**QCloud.** Needs the `qcloud-ai` SDK (not on PyPI yet; see the [main README](../README.md#providers))
and a key. The model is `--model`, else `QCLOUD_MODEL`, else the harness's GLM default:

```bash
export QCLOUD_API_KEY=...
export QCLOUD_MODEL=<catalog model id with the "tools" capability>   # optional
python -m examples.patterns.react_assistant --provider qcloud
```

**Any OpenAI-compatible server** (vLLM, SGLang, llama.cpp, Ollama):
`--provider openai` with `OPENAI_COMPATIBLE_BASE_URL`, `OPENAI_COMPATIBLE_MODEL` (or `--model`)
and, if the server wants one, `OPENAI_COMPATIBLE_API_KEY`.

`QAGENTS_EXAMPLE_PROVIDER=qcloud` changes the default provider. Credentials are read from the
environment only and are never printed.

**Replayed cassettes, offline.** Record a run once, then replay it with no key and no network:

```bash
python -m examples.patterns.react_assistant --provider qcloud --record   # live, also writes the cassette
python -m examples.patterns.react_assistant --provider qcloud --replay   # offline, no credentials
```

Cassettes are written to `examples/cassettes/<provider>/<example>.json`. A replay must ask
exactly the recorded requests (messages, tools, settings, output schema), so after you change
an example's prompts or fixtures, record it again. No cassettes are committed yet: `--replay`
works for an example once someone has recorded it with `--record`.

Every run prints what it does (`-- ...` lines), the streamed text and tool calls
(`-> tool(args)`, `<- tool error: ...`, `[sub-agent] ...`, `|| ... needs approval`), then the
outcome and the usage of every agent together:

```
ok: <the example's one-line summary>        (or FAILED: ..., or error: ...)
usage: <n> requests, <n> input / <n> output tokens, $<cost>, <seconds> s
```

The exit status is 0 when the example did what it shows (its `ExampleOutcome.ok`), 1 otherwise.
An outcome checks results robustly (numbers, key facts, structure, which tools ran), never exact
wording.

## Options

Every example accepts the same options (`python -m examples.patterns.<name> --help`):

| Option | What it does |
|---|---|
| `--provider anthropic\|qcloud\|openai` | The model provider (default `anthropic`, or `QAGENTS_EXAMPLE_PROVIDER`). |
| `--model ID` | The model id; each provider has a default. |
| `--record` | Run live and also write the cassette. |
| `--replay` | Answer from the cassette, offline. Spends nothing, so `--max-cost` does not apply. |
| `--max-cost USD` | The budget of the whole example, every agent and sub-agent together (default 0.50). Checked before each model request: once it is spent, the next request is not sent (`BudgetExceeded`). Requests already in flight finish, so parallel branches can overshoot a little. Every agent also has its own `Limits`. |
| `--mlflow` | Trace every run in MLflow (needs the `mlflow` extra). Traces go to `--mlflow-uri`, else `MLFLOW_TRACKING_URI`, else the local SQLite file `examples/mlruns/mlflow.db`, in the experiment `qagents-examples`, tagged with the example's name. Browse them with `mlflow ui --backend-store-uri sqlite:///examples/mlruns/mlflow.db`. |
| `--mlflow-uri URI` | The MLflow tracking URI for `--mlflow`. |
| `--interactive` | Answer the example's questions yourself on the terminal. Without it a scripted answer is used and printed. Used by `human_in_the_loop` (the passenger's name, the booking approval), `customer_support` (the refund approval) and `content_flow` (the editor's publish decision). |
| `--workdir DIR` | Keep the example's files (SQLite journals, notes, outputs) in `DIR` instead of a fresh temporary directory. Running a workflow example twice with the same `--workdir` shows the replay of finished steps (`plan_and_execute`, `research_crew`, `map_reduce`). |
| `--verbose` | Also print tool results, sub-agent text and the outcome's details. |
| `--json` | Print the `ExampleReport` as JSON on the last line. |

## Run them all

```bash
python -m examples.run_all --provider anthropic
python -m examples.run_all --provider qcloud --only react_assistant,supervisor --max-cost 2
python -m examples.run_all --provider qcloud --record    # also write every cassette
python -m examples.run_all --provider qcloud --replay    # offline, from the cassettes
```

`run_all` runs every module in `examples/patterns` whose name does not start with `_`, one after
another. `--max-cost` (default 5 USD) is the budget of the whole run: each example may spend what
is left, at most `--max-cost-each` (default 0.50), and once the budget is spent the remaining
examples are reported as skipped. A failing or broken example never stops the others. The other
options are passed to every example; with `--workdir DIR`, each example gets `DIR/<name>`.

The table is printed at the end and written to `examples/results/<provider>.md` and
`examples/results/<provider>.json` (`<provider>-replay.*` for a replay, so a replay never
overwrites a live run's results). It lists each example's status, requests, tokens, cost,
seconds and one-line summary. The exit status is 0 when every example passed.

`evals_compare` also writes its own reports to `examples/results/evals-<provider>.json` and
`.md`; run it on both providers and the second run prints the comparison.
Git ignores only `*.db` files here (the MLflow database), not `examples/results/`, so a live
run leaves result files you may want to commit or delete.

## Gallery

| Example | What it shows | Mirrors | qagents features |
|---|---|---|---|
| [`react_assistant`](patterns/react_assistant.py) | A tool-calling assistant over two turns; the second question only makes sense with the first in memory | LangGraph `create_react_agent` with a checkpointer and `thread_id` | `@tool(read_only=True)`, `Literal` arguments, `ToolError`, `Session(SQLiteStore, id=...)`, `agent.stream`, `Session.messages()` |
| [`human_in_the_loop`](patterns/human_in_the_loop.py) | A booking that pauses twice: a question to the user, then an approval; each resume from a new `Agent` over a reopened store | LangGraph `interrupt()` and `Command(resume=...)` | `toolkits.ask_user` (deferred tool), `@tool(requires_approval=True, idempotent=True)`, `resume_stream(results=..., approvals=...)`, `Approval(arguments=...)`, `RunContext[Deps]`, `ctx.operation_id` |
| [`agentic_rag`](patterns/agentic_rag.py) | The model searches, judges the snippets, reads, searches again, and cites; an output validator rejects citations of documents it never retrieved | LangGraph agentic RAG (retrieve, grade documents, rewrite question) | read-only search/read tools over an in-module BM25 index, `output_type` with `output_mode`, `output_validators` raising `ModelRetry`, `Limits(output_retries=...)` |
| [`plan_and_execute`](patterns/plan_and_execute.py) | A planner, an executor with tools and a bounded replanner, each run a durable workflow step | LangGraph plan-and-execute | `Workflow.run_agent`, `Workflow.completed_steps()`, typed `Plan`/`Review`, `output_validators`, parallel read-only calls, `Limits(max_tool_calls=...)` |
| [`reflection`](patterns/reflection.py) | A writer and a typed critic in a revise loop of at most three rounds; hard rules checked in Python | LangGraph reflection / evaluator-optimizer | one `Session` for the writer's drafts, a fresh critic run with `output_type=Critique`, `ctx.print_stream` |
| [`sql_agent`](patterns/sql_agent.py) | List tables, describe schemas, run one SELECT with a join and an aggregate; refused statements go back to the model | LangGraph SQL agent | tools taking `RunContext[Path]` (`deps`), `ModelRetry` inside tools, a read-only SQLite connection with a query time limit |
| [`supervisor`](patterns/supervisor.py) | A supervisor delegates to a researcher and an analyst, then writes the report | LangGraph supervisor (`langgraph-supervisor`), tool-calling variant | `Agent.as_tool()` linked child runs, child usage and cost rolled up, inherited budget, `SubagentEvent`, a `Hooks.on_event` counter |
| [`hierarchical_teams`](patterns/hierarchical_teams.py) | A manager with a research team (parallel searchers) and a writing team (writer, editor), three levels deep | LangGraph hierarchical agent teams | `fan_out(max_parallel=..., timeout=...)`, nested `as_tool()`, `Limits(max_depth=...)`, one cost cap for the tree, a propagating `Hooks` subclass |
| [`map_reduce`](patterns/map_reduce.py) | Documents read at run time are summarised in parallel as durable steps and reduced to one brief; a second run replays everything | LangGraph map-reduce with `Send()` | `Workflow.step(type=...)`, `Workflow.map(concurrency=...)`, `Workflow.run_agent`, `Workflow.status()` |
| [`customer_support`](patterns/customer_support.py) | Triage hands off to billing; a signed approval gates the refund; the second message goes to the specialist | LangGraph swarm / OpenAI Agents SDK handoffs | `Agent(handoffs=[...])`, `result.agent`, `@tool(requires_approval=True)`, `Permissions(approval_secret=...)`, `Approval.for_call`, `ApprovalTokenError`, `Agent.resume_stream` |
| [`debate`](patterns/debate.py) | Two debaters argue for two rounds; a judge returns a typed verdict that a validator checks | CrewAI / LangGraph debate with a judge | `Agent.copy(...)`, the rounds as a `for` loop, `output_type=Verdict`, `output_validators` raising `ModelRetry` |
| [`fallback_router`](patterns/fallback_router.py) | A classifier routes three concurrent questions to a quick or an expert agent, on a model chain whose primary always answers 429 | LangGraph routing with per-node models, LangChain `with_fallbacks` | `route(classifier, routes, prompt)`, `output_type=Literal[...]`, `FallbackModel`, `RateLimitedModel`, `asyncio.gather` |
| [`research_crew`](patterns/research_crew.py) | A researcher, a writer and an editor as typed, durable workflow steps, each task's context written into the next prompt | CrewAI sequential crew (role, goal, backstory, `expected_output`) | `Workflow(SQLiteStore, id=...).run_agent`, pydantic `output_type` per task, `RunResult.completed` / `stop_reason` checked after each step |
| [`hierarchical_crew`](patterns/hierarchical_crew.py) | A manager writes questions for market, legal and technical specialists, answered in parallel, and returns a structured report | CrewAI hierarchical process | one `fan_out(agent, name="ask_<area>")` tool per specialist, a shared cost budget, `output_type=LaunchReport`, checking which specialists really answered |
| [`trip_planner`](patterns/trip_planner.py) | Pick a city, then the local expert and the concierge work in parallel, then the concierge plans within budget | CrewAI trip planner crew | `asyncio.gather` of `Workflow.run_agent` (CrewAI `async_execution=True`), `concierge.copy(output_type=...)`, an output validator reading `deps`, `OutputValidationError` |
| [`content_flow`](patterns/content_flow.py) | A durable flow: topic, routed writer, guardrail, one revision, then a wait for the editor's publish signal; the second pass replays every step | CrewAI Flows (`@start`, `@router`, `@listen`, state) | `Workflow.step(type=Draft)`, `route(..., default=...)`, `output_guardrails` with `GuardrailTripped`, `wait_for_signal` / `send_signal`, `WorkflowSuspended` |
| [`stock_analysis`](patterns/stock_analysis.py) | Two analyst sub-agents run in parallel for an advisor whose typed recommendation is validated and checked against computed metrics | CrewAI stock analysis crew | `as_tool(read_only=True)` in parallel, `output_type=Recommendation`, two `output_validators`, `Limits(tool_timeout_s=...)` |
| [`deferred_job`](patterns/deferred_job.py) | A tool submits an external job and pauses the run; the result arrives later by polling or by a workflow signal | LangGraph `interrupt()` for async work / CrewAI async tasks | `raise DeferToolCall(reason, metadata=...)`, `PendingInput.metadata`, `resume_stream(results=...)`, `Workflow.run_agent(results=...)`, `wait_for_signal`, `@tool(idempotent=True)` |
| [`long_running_research`](patterns/long_running_research.py) | A planning research agent whose worker dies mid-tool; a second worker recovers the run after the lease expires | Deep Agents on LangGraph (todos, files, context editing, recovery) | `todo_tools` + `render_todos`, `notes_tools`, `ToolResultClearing`, `AgentRuntime` (`start`, `recover`, `RunHandle`), leases, `RunContext.state` |
| [`mcp_tools`](patterns/mcp_tools.py) | An agent uses a local stdio MCP server; destructive stock changes are approved or held by a policy in code | langchain-mcp-adapters / CrewAI `MCPServerAdapter` | `MCPToolkit.stdio(...)` (extra `mcp`), annotations mapped to `read_only` and approval, `Permissions(on_ask=...)` returning `Approval(reason=...)` |
| [`mlflow_tracing`](patterns/mlflow_tracing.py) | A supervisor with two workers traced as one MLflow trace, then read back and printed as a span tree | LangGraph / CrewAI runs traced with MLflow autolog | `MLflowHooks(record_content=True)` (extra `mlflow`), `Hooks.propagate`, nested AGENT/TOOL/CHAT_MODEL spans, `mlflow.search_traces` |
| [`evals_compare`](patterns/evals_compare.py) | Score a tool-using assistant on six cases, save the report, compare it with the other provider's | LangSmith `evaluate()` / `crewai test` | `qagents.evals`: `Dataset`, `evaluate`, `ExactMatch`, `Contains`, `ToolCalled`, `MaxTurns`, `MaxCostUSD`, `LLMJudge`, `EvalReport`, `compare`, `regressions` |

`_mcp_server.py` is not an example: it is the MCP server that `mcp_tools` starts.

## If you know LangGraph

qagents has no graph to compile. An agent is one loop (ask the model, run the tools it called,
repeat), and anything bigger is an ordinary async function that calls agents. Durability comes
from the journal: every message, tool start and tool result is appended as it happens.

| LangGraph | qagents | See |
|---|---|---|
| `create_react_agent(model, tools, prompt=...)`; the agent node, `ToolNode` and `tools_condition` | `Agent(model, instructions=..., tools=[...])`: the loop is the ReAct cycle; read-only calls of one turn run in parallel | `react_assistant` |
| `@tool` | `@tool` on a typed function; the docstring becomes the description; flags `read_only`, `requires_approval`, `idempotent`, `deferred`, `timeout` | every example |
| `StateGraph` nodes, edges, conditional edges | Python: agent runs, `if`, `for`, `asyncio.gather` | `plan_and_execute`, `reflection` |
| State with reducers (`Annotated[list, operator.add]`) | Variables and return values in your function; `RunContext.state` for values tools, hooks and instructions share in one session | `map_reduce`, `long_running_research` |
| Checkpointer + `thread_id` | `Session(SQLiteStore(path), id=...)` for a conversation (`InMemoryStore()` instead of `MemorySaver`) | `react_assistant` |
| Functional API `@entrypoint` / `@task`, durable graphs | `Workflow(store, id=...)` with `step`, `run_agent` and `map`: a finished step is replayed, never re-run | `plan_and_execute`, `map_reduce`, `research_crew` |
| `graph.get_state(config)` | `await session.messages()`, `await session.state()`, `await workflow.status()` | `react_assistant`, `deferred_job` |
| `Send()` fan-out | `Workflow.map(name, items, fn, concurrency=...)`, a `fan_out(agent)` tool, or `asyncio.gather` | `map_reduce`, `hierarchical_teams` |
| `interrupt(value)` in a node, `Command(resume=...)` | A deferred tool (`toolkits.ask_user`, `@tool(deferred=True)`, or `raise DeferToolCall` in a running tool), then `agent.resume(session, results={call_id: value})`. The run continues at the waiting call; the node is not re-run | `human_in_the_loop`, `deferred_job` |
| Approve / edit / reject before a tool (`interrupt_before`) | `@tool(requires_approval=True)` or `Permissions(ask=[...])`, then `resume(session, approvals={call_id: True / False / Approval(arguments=...)})`; or decide in code with `Permissions(on_ask=...)` | `human_in_the_loop`, `customer_support`, `mcp_tools` |
| `with_structured_output(Schema)` | `Agent(output_type=Schema, output_mode=...)`; `output_validators` raising `ModelRetry` send corrections | `agentic_rag`, `plan_and_execute` |
| `recursion_limit` | `Limits(max_turns=..., max_tool_calls=..., max_cost_usd=..., timeout_s=...)` on each agent | every example |
| `create_supervisor` (`langgraph-supervisor`) | Workers as tools: `tools=[worker]` or `worker.as_tool()`; each worker sees only its task | `supervisor`, `stock_analysis` |
| `create_swarm`, `create_handoff_tool`, `Command(goto=...)` | `Agent(handoffs=[...])` gives `transfer_to_<name>` tools; `result.agent` names the agent that answered | `customer_support` |
| Hierarchical teams (subgraphs) | Agents with agent tools, nested; usage, budget and propagating hooks roll up; `Limits(max_depth=...)` | `hierarchical_teams` |
| Router node + `add_conditional_edges` | `route(classifier, {"label": agent, ...}, prompt)` | `fallback_router`, `content_flow` |
| `stream_mode="messages"` / `"updates"` | `agent.stream(...)`: typed events (`text_delta`, `tool_call_started`, `subagent_event`, `approval_required`, ...) | every example (`ctx.print_stream`) |
| `model.with_fallbacks([...])`, `InMemoryRateLimiter` | `FallbackModel([...])`, `RateLimitedModel(model, max_concurrent=..., requests_per_minute=...)` | `fallback_router` |
| Deep Agents `write_todos`, its file system, `ClearToolUsesEdit` | `todo_tools()` with `render_todos`, `notes_tools(directory)`, `Agent(compactor=ToolResultClearing(...))` | `long_running_research` |
| Re-invoking a thread after a crash | `AgentRuntime(store, agents).recover()` or `run_forever()`: leases find dead runs; only the interrupted call runs again, and only if it is `read_only` or `idempotent` | `long_running_research` |
| `MultiServerMCPClient(...).get_tools()` | `async with MCPToolkit.stdio(command, args) as mcp:` then `tools=[mcp]` | `mcp_tools` |
| LangSmith `evaluate(target, data, evaluators)` | `qagents.evals.evaluate(agent, dataset, evaluators)`, `compare`, `regressions`; the report is a JSON or Markdown file | `evals_compare` |
| `mlflow.langchain.autolog()` | `hooks=[MLflowHooks(record_content=True)]` on the top agent | `mlflow_tracing`, `--mlflow` |

## If you know CrewAI

qagents has no Crew, Task or Flow objects. An agent's role is its instructions, a task is a
prompt, a process is the async function that calls the agents, and context is what your code
writes into the next prompt, so you can read and test exactly what each agent receives.

| CrewAI | qagents | See |
|---|---|---|
| `Agent(role=, goal=, backstory=)` | `Agent(model, name=..., instructions=...)`: role, goal and backstory become the instructions | `research_crew` |
| `Agent(llm=...)` | `Agent(model)`: a `Model` object or a string such as `"anthropic:claude-opus-5-5"` | every example |
| `Agent(tools=[...])`, `BaseTool`, `@tool` | `@tool` on a typed function, or a `Toolkit` | every example |
| `Task(description=, expected_output=)` | The prompt of one `agent.run(...)` or `Workflow.run_agent(...)` | `research_crew` |
| `Task(output_pydantic=Model)` | `Agent(output_type=Model, output_mode=...)` | `research_crew`, `trip_planner` |
| `Task(context=[earlier_task])` | Put the earlier result (often `result.output` as JSON) into the next prompt | `research_crew`, `trip_planner` |
| `Task(guardrail=...)` | `output_validators=[fn]` raising `ModelRetry(...)` to send the model back; `output_guardrails` to block the answer (status `"blocked"`) | `stock_analysis`, `debate`, `content_flow` |
| `Crew(process=Process.sequential).kickoff()` | An async function that calls each agent in order; `Workflow.run_agent` makes each task a durable step | `research_crew` |
| `Process.hierarchical` with `manager_llm` / `manager_agent` | A manager `Agent` with the specialists as tools (`as_tool()` or `fan_out(agent)`) | `hierarchical_crew` |
| `allow_delegation`, "Delegate work to coworker" / "Ask question to coworker" | A delegation tool per specialist; each child run sees only the question the manager wrote | `hierarchical_crew`, `supervisor` |
| `Task(async_execution=True)` | `asyncio.gather` of two runs or `run_agent` steps | `trip_planner` |
| `Task(human_input=True)` | A deferred `ask_user` tool or `@tool(requires_approval=True)`, answered with `agent.resume(...)` from any process | `human_in_the_loop` |
| Flows: `@start`, `@listen`, `@router` | An async function over `Workflow` steps; `route(classifier, routes, prompt)` for the router | `content_flow` |
| Typed `Flow[State]` with `@persist` | Each step's typed return value, journaled once and replayed (`Workflow.step(..., type=Model)`) | `content_flow` |
| Waiting for an outside event in a flow | `Workflow.wait_for_signal(name)` / `Workflow.send_signal(store, id, name, value)`, or `wait_for_timer` | `content_flow`, `deferred_job` |
| `max_iter`, `max_execution_time` | `Limits(max_turns=..., timeout_s=...)`, plus `max_cost_usd` | every example |
| `max_rpm` | `RateLimitedModel(model, requests_per_minute=...)` | `fallback_router` |
| Memory | `Session(store, id=...)` for the conversation, `RunContext.state` for values, `notes_tools` for notes in files | `react_assistant`, `long_running_research` |
| `MCPServerAdapter(StdioServerParameters(...))` | `MCPToolkit.stdio(command, args)` | `mcp_tools` |
| `crewai test` | `qagents.evals.evaluate(...)` with deterministic evaluators and `LLMJudge` | `evals_compare` |
| `verbose=True` | `agent.stream(...)` events; `Hooks` for your own logging; `MLflowHooks` for traces | every example |

## Offline tests

Every pattern has `tests/examples/test_<name>.py`. A test drives `run(ctx)` end to end with a
`FunctionModel` policy (a function of the request that answers as each agent would, dispatching
on `system_text`, `tool_names` or `last_text`) through `scripted_context`. It checks the outcome
and the details that prove the pattern (which tools ran, what each agent saw, how many requests
were made, what the journal holds), and it records a cassette through the command line and
replays it by running the module as `__main__`. No test touches the network.

```bash
python -m pytest tests/examples tests/test_examples.py -q
```

How to write a new example (module shape, rules for both providers, the test) is in the
docstring of [`harness.py`](harness.py).

## Other examples in this folder

| File | What it is |
|---|---|
| [`quickstart.py`](quickstart.py) | A support assistant on QCloud: tools, streaming, an approval and a durable session. |
| [`offline_tour.py`](offline_tour.py) | The loop, an approval pause, a resume and a handoff with a scripted model; no key needed. |
| [`reliability.py`](reliability.py) | Repeated bounded agent runs, pass@k, pass^k and paired case bootstrap; offline. |
| [`repository_activity.py`](repository_activity.py) | Indexed commits and CI evidence for yesterday/last-week questions, with a fast local query path and a checked read-only agent. |
| [`custom_provider.py`](custom_provider.py) | A complete model provider in thirty lines. |
| [`serve_fastapi.py`](serve_fastapi.py) | An agent over HTTP with server-sent events, approvals and questions to the user. |
| [`live_acceptance.py`](live_acceptance.py) | A short paid acceptance test against the live QCloud API. |
| [`support_review.py`](support_review.py) | Inspect an order, propose a support case, pause for approval and write the approved case to a local SQLite queue. |
| [`scoped_memory.py`](scoped_memory.py) | Tenant- and user-scoped preference memory in session state, across a reopened SQLite store; offline. |
| [`team_messages.py`](team_messages.py) | A coordinator, a researcher and a reviewer exchange messages through a `TeamMailbox` (unreleased). |
| [`devvi_coder.py`](devvi_coder.py) | A lead agent plans a change, Devvi makes the edits over ACP, and the lead reviews them (unreleased). |
| [`recover_credit.py`](recover_credit.py) | Recover a credit whose response was lost, through an operator settlement across two workers (Effects 0.3 preview; not run by the test suite). |
| [`subprocess_refund.py`](subprocess_refund.py) | A subprocess-isolated crash test of a simulated Stripe refund (Effects 0.3 preview; not run by the test suite). |
