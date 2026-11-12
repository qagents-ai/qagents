# Model providers

A provider is a subclass of `Model`. Three ship with qagents, and two more are for tests.

| Class | Talks to | Transport |
|---|---|---|
| `QCloudModel` | QCloud | the `qcloud_ai` SDK |
| `OpenAIChatModel` | any server that speaks OpenAI Chat Completions | `httpx`, no vendor SDK |
| `AnthropicModel` (`"anthropic:<model-id>"`, extra `anthropic`) | Claude through the Messages API | the `anthropic` SDK |
| `ScriptedModel`, `FunctionModel` | nothing | see [testing](testing.md) |

`FallbackModel` and `RateLimitedModel` wrap other models (the next model when one refuses at
admission; staying under a provider's limits).

!!! note "Coding agents are not providers"
    Devvi and other coding agents run their own loop, tools and sandbox, so they are not a
    `Model`: wrapping one as a model would hide its tool calls from the journal and its
    permission requests from your rules. Use them as [external agents](concepts/external-agents.md)
    (`DevviAgent`, `ACPAgent`). A provider for the devvi.sh platform's model endpoint is not
    part of this release.

## QCloud

```python
from qagents import Agent, QCloudModel

agent = Agent("qcloud:<model-id>")  # client built from QCLOUD_* variables
agent = Agent(QCloudModel("<model-id>", client=shared_client))  # your AsyncQCloud client
```

The SDK keeps ownership of transport: authentication, connection pooling, its retry policy and
its stream validation. Create one client per process and share it. The adapter adds what an agent
loop needs:

**Allow-list checked early.** QCloud's Chat Completions endpoint accepts a fixed set of request
fields and answers HTTP 400 for anything else. qagents checks `ModelSettings.extra` against that
list (`qagents.models.qcloud.ALLOWED_PARAMS`) when the agent is built and again before each
request, and the error lists the accepted fields. Reasoning controls, `store`, `metadata` and
provider routing overrides are not accepted by the platform.

**Catalog-aware.** The first request reads the model's catalog entry once. From it the adapter:

- fails fast, with a clear message, when the agent has tools but the model does not list the
  `tools` capability, or when the conversation contains an image and the model lacks `vision`;
- prices every response from the catalog's per-million-token prices, so `result.usage.cost_usd`
  is filled in and `Limits(max_cost_usd=...)` works.

If the catalog cannot be reached, inference continues without cost. Pass `use_catalog=False` to
skip the lookup.

!!! note "Cost is an estimate"
    `cost_usd` is computed from the catalog price and the reported token counts. It is not the
    amount QCloud settles; reconcile against your usage records.

**Errors keep their meaning.**

| QCloud answer | qagents error | `retryable` | `ambiguous` |
|---|---|---|---|
| 400, 404, 422 | `ModelBadRequestError` | no | no |
| 401, 403 | `ModelAuthenticationError` | no | no |
| 402 | `ModelBillingError` | no | no |
| 429 | `ModelRateLimitError` | yes | no |
| 5xx, error inside a started stream | `ModelServerError` | yes | yes |
| timeout or broken connection after sending | `ModelTimeoutError`, `ModelConnectionError` | no | yes |
| stream ended before `[DONE]` | `ModelResponseError` | no | yes |

Each error carries `status_code`, `code`, `request_id` (quote it to support) and `retry_after`.
Chat requests have no idempotency key on the platform, so an ambiguous request is
[never repeated automatically](concepts/loop.md#retries-and-billing).

## OpenAI-compatible servers

```python
from qagents import Agent, OpenAIChatModel

model = OpenAIChatModel("my-model", base_url="http://localhost:8000/v1", api_key="...")
agent = Agent(model, tools=[...])
```

This reaches vLLM, SGLang, llama.cpp, Ollama, OpenAI and gateways in front of them. The adapter
tolerates the ways such servers differ: tool-call deltas without an index, arguments sent as an
object, a repeated function name, a missing call id, and reasoning under either `reasoning` or
`reasoning_content`.

| Option | Purpose |
|---|---|
| `headers` | extra HTTP headers |
| `timeout`, `max_retries` | per request; only requests that provably never ran are retried |
| `max_tokens_param` | `"max_completion_tokens"` for servers that require the newer name |
| `include_usage` | set `False` if the server rejects `stream_options` |
| `capabilities` | a `ModelCapabilities` with prices, to get `cost_usd` |

Tool calling on self-hosted servers depends on how the server was started (for vLLM, a tool-call
parser per model). If a model answers in text instead of calling tools, check the server
configuration before changing the agent.

## Writing a provider

Implement one method. It receives a `ModelRequest` and yields stream events, ending with exactly
one `ResponseDone` that carries the complete response.

```python title="examples/custom_provider.py"
--8<-- "examples/custom_provider.py"
```

Guidelines for a real adapter:

- Translate `request.messages` (typed parts) to the wire format and back. Keep provider data that
  must be replayed unchanged (signed or encrypted reasoning) in `ReasoningPart.opaque`.
- Raise `ModelError` subclasses, and set `ambiguous=True` whenever the request may have run.
- Parse tool-call arguments into `ToolCallPart.arguments`; on invalid JSON set `parse_error`
  instead of raising, so the model can be told.
- Fill `usage`, and `usage.cost_usd` if you know prices.
