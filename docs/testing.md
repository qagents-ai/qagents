# Testing agents

The loop, your tools, your permissions and your hooks can all run in a unit test, with no network
and no model. Replace the model with a scripted one.

```python
from qagents import Agent, tool
from qagents.models.testing import ScriptedModel, call


@tool(read_only=True)
def get_weather(city: str) -> str:
    """Look up the weather for a city."""
    return f"Sunny in {city}"


async def test_weather_question():
    model = ScriptedModel(
        [
            call("get_weather", city="Athens"),  # first reply: the model calls the tool
            "It is sunny in Athens.",  # second reply: the model answers
        ]
    )
    result = await Agent(model, tools=[get_weather]).run("Weather in Athens?")

    assert result.output == "It is sunny in Athens."
    assert model.requests[1].messages[-1].tool_results[0].content == "Sunny in Athens"
```

## Scripted replies

`ScriptedModel(replies)` answers with the given replies, in order, and fails the test if the
agent asks for more replies than the script has. A reply can be:

| Reply | Meaning |
|---|---|
| `"text"` | an answer |
| `call("tool", arg=1)` | one tool call |
| `[call(...), call(...)]` | several tool calls in one turn |
| a `Message` or `ModelResponse` | full control, for example over usage or finish reason |
| an exception instance | the request fails with it |
| a function of the request | compute any of the above from what was sent |

`FunctionModel(function)` calls your function for every request instead of following a script.

## Assert on what was sent

Every request is kept in `model.requests`, complete: system message, history, tool definitions
and settings.

```python
request = model.requests[0]
assert request.messages[0].role == "system"
assert [spec.name for spec in request.tools] == ["get_weather"]
assert request.settings.temperature is None
```

## Test the paths that are hard to reach live

```python
from qagents import Session
from qagents.errors import ModelRateLimitError, ModelServerError


async def test_refund_waits_for_approval():
    model = ScriptedModel([call("refund_order", order="A-1", amount=30), "Refunded."])
    agent = Agent(model, tools=[refund_order])
    session = Session()

    paused = await agent.run("Refund A-1", session=session)
    assert paused.status == "paused"

    done = await agent.resume(session, approvals={paused.pending_approvals[0].tool_call_id: True})
    assert done.output == "Refunded."


async def test_rate_limit_is_retried():
    model = ScriptedModel([ModelRateLimitError("slow down", retryable=True, retry_after=0), "ok"])
    assert (await Agent(model).run("hi")).output == "ok"
```

Set `usage=Usage(...)` on the scripted model to test token and cost limits, and `delay=0.02` to
slow streaming down enough to test cancellation.

## Test a tool on its own

A tool is a function plus validation. Call it directly:

```python
from qagents import RunContext


async def test_get_weather():
    assert await get_weather.run(RunContext(), {"city": "Athens"}) == "Sunny in Athens"


async def test_my_orders_uses_deps():
    ctx = RunContext(deps=Deps(db=FakeDb(), user_id="u-1"))
    assert await my_orders.run(ctx, {}) == []
```

## Running this project's tests

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

The suite makes no network requests. The QCloud tests use the real `qcloud_ai` SDK with its HTTP
transport replaced by a local one, and are skipped when the SDK is not installed. Documentation
code blocks are compiled, and the examples are executed, by `tests/test_examples.py`.
