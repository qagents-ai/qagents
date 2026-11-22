# Structured output

Give the agent an `output_type` and `result.output` is a validated value instead of text.

```python
from pydantic import BaseModel
from qagents import Agent


class Invoice(BaseModel):
    number: str
    total: float
    currency: str = "EUR"


agent = Agent("qcloud:<model-id>", instructions="Extract invoices.", output_type=Invoice)
result = await agent.run("Invoice INV-7 for 120 euros")
invoice: Invoice = result.output
```

`output_type` can be a pydantic model, a dataclass, a `TypedDict`, or any type pydantic can
validate, such as `list[Invoice]`.

## How it works

The agent can still call tools first. When the model stops calling tools, its last message is
taken as the answer, parsed as JSON and validated:

1. The JSON is read tolerantly: bare JSON, JSON in a code fence, or JSON with a few words around
   it are all accepted.
2. It is validated against `output_type`.
3. On failure the model receives the validation error and tries again, up to
   `Limits.output_retries` times (default 2). After that, `OutputValidationError` is raised.

## Two modes

| `output_mode` | What is sent | Use it when |
|---|---|---|
| `"native"` (default) | the schema as `response_format: {"type": "json_schema", ...}` | the model or server enforces JSON schemas |
| `"prompted"` | the schema appended to the instructions | the model has no schema support |

```python
agent = Agent("qcloud:<model-id>", output_type=Invoice, output_mode="prompted")
```

The appended text is `qagents.prompts.OUTPUT_INSTRUCTIONS`; you can read and replace it.

A schema constraint guarantees shape, not correctness, so validation and retries run in both
modes. `output_strict=True` sets `strict` on the schema for providers that support strict mode;
they then require every property to be listed as required and `additionalProperties: false`.

!!! note "Model support varies"
    Whether a given catalog model honours `response_format` depends on the model and on how it
    is served. Run `examples/live_acceptance.py` against the model you plan to use; it includes a
    structured-output check. If it fails there, use `output_mode="prompted"`.
