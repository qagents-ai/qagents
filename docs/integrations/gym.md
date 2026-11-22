# QAgent Gym and integration recipes

qagents runs the agent loop: tools, permissions, sessions, events and recovery. QAgent Gym is
a separate package for learning from completed episodes: ACE-style playbooks, feedback,
memory adapters, evaluation and training-data exports. Changing a playbook changes the
context of later runs. It does not train model weights.

The [QAgent Gym repository](https://github.com/qagents-ai/qagent-gym) contains the adapter
recipes below. Optional SDKs belong
in the application or Gym environment. Qagents does not need them to import or run.

## Start with session memory

Structured session state is useful for preferences and decisions that belong to one
conversation or application identity. It is separate from the transcript and is saved in
the journal. A retrieval service such as Mem0 is a different store, with its own scoping,
retention and write semantics.

The offline [scoped-memory example](https://github.com/qagents-ai/qagents/blob/main/examples/scoped_memory.py)
shows the existing qagents API without an external SDK:

```bash
python -m examples.scoped_memory --database preferences.db
```

Run this from the qagents checkout. The example asks for approval before storing a scripted
preference, closes and reopens SQLite, and reads it with a new agent. The same tenant/user
sees `el`. Another user or tenant sees `unset`. It builds the session id from an unambiguous
pair supplied by the application. The model chooses a preference key and value, not a tenant
or user identity. The sample's automatic approval is only for its offline demonstration;
an application supplies the authenticated person's decision.

Its `RunAudit` hook records run status and token/request counts without recording prompts,
tool arguments or preference values. The tests also cover a declined write and resuming an
approval after reopening the store with a new agent.

## Adapter recipes

| Recipe | How it connects | What to check in your application |
| --- | --- | --- |
| [Mem0 memory](https://github.com/qagents-ai/qagent-gym/blob/main/examples/mem0_memory.py) | Retrieve scoped memories for later episodes | Derive tenant/user scope from authenticated application context. Inspect what is stored |
| [LangGraph agent](https://github.com/qagents-ai/qagent-gym/blob/main/examples/langgraph_agent.py) | Run an existing graph through the Gym generation contract | Map the graph's actual input/output fields and preserve its checkpoint ownership |
| [DSPy agent](https://github.com/qagents-ai/qagent-gym/blob/main/examples/dspy_agent.py) | Evaluate a callable DSPy program using the same episode records | Use a separate evaluation set when optimizing a program |
| [ElevenLabs voice](https://github.com/qagents-ai/qagent-gym/blob/main/examples/elevenlabs_voice.py) | Place speech synthesis around the agent's text boundary | Measure the audio provider's latency separately from agent latency |
| [Twilio voice](https://github.com/qagents-ai/qagent-gym/blob/main/examples/twilio_voice.py) | Connect telephony transport to a voice application | Validate incoming requests and handle interruptions at the application boundary |
| [OpenTelemetry tracing](https://github.com/qagents-ai/qagent-gym/blob/main/examples/otel_tracing.py) | Link Gym episode spans with qagents run/model/tool spans | Keep content capture off unless explicitly needed. Choose an exporter in the application |

The recipes document their offline fixtures and optional SDK paths. An offline fixture checks
the integration contract. Live service behavior and end-to-end latency need separate runs
with that service. An adapter does not make external memory writes, calls or messages safe
to repeat after an interrupted request.

## Existing qagents connection points

| Need | qagents entry point | Guide |
| --- | --- | --- |
| Local or hosted OpenAI-compatible models | `OpenAIChatModel` | [Model providers](../providers.md#openai-compatible-servers) |
| Tools hosted by another application | `MCPToolkit` | [MCP tools](../toolkits.md#connect-external-tools) |
| qagents tools used by another agent | `tool_server` | [Serving tools over MCP](mcp-server.md) |
| Traces in the application's telemetry backend | `OTelHooks(tracer, record_content=False)` | [Tracing](../concepts/hooks.md#tracing) |
| Conversation and structured state | `Session`, `SQLiteStore`, `RunContext.state` | [Sessions](../concepts/sessions.md) |
| Multi-step orchestration | `Workflow.run_agent` | [Workflows](../concepts/multi-agent.md#workflows) |
| Observe or adjust requests | `Hooks.before_model`, `Hooks.on_run_end` | [Hooks](../concepts/hooks.md#hooks) |
| Reproducible offline runs | `ScriptedModel`, `FunctionModel` | [Testing](../testing.md) |

Use normal typed tools when invoking another framework. Only mark a tool `read_only=True`
when the entire delegated operation makes no external changes. Report extra model or
embedding spend with `ctx.add_usage(...)` so it contributes to the run's budget.

For a write to an external memory service, pass `ctx.operation_id` only if the destination
actually enforces that idempotency key. Otherwise leave the tool non-idempotent and reconcile
an unknown outcome before trying it again. Plain `Workflow.step` may repeat unfinished work;
it is for pure computations and read-only work. See [recovery guarantees](../concepts/sessions.md#what-is-guaranteed).

## Evaluation and training

Use a fixed task set and scoring rule to compare the same agent before and after playbook or
memory changes. Keep training episodes separate from evaluation episodes. Gym can prepare
evaluation and training-data exports. A file export is not evidence that fine-tuning or RL
improved a model. Weight training remains a separate, explicitly configured training job.

The package version remains 0.2.0. The current main checkout also includes the Effects 0.3
preview and crash harness. Check the installed API when using unreleased capabilities.

Gym's [environment rollout guide](https://github.com/qagents-ai/qagent-gym/blob/main/docs/environment-rollouts.md)
adds multi-step observations, actions and environment-supplied rewards, including an optional
Gymnasium adapter. A Gym generator can select each action using a fixed context snapshot.
QAgents' [reliability utilities](../guides/reliability.md) measure repeated outcomes and compare
paired cases. `TrialTarget` supplies the repetition index for seeded environment trials.
