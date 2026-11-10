from __future__ import annotations

import asyncio
import threading
from typing import Literal

import pytest
from pydantic import BaseModel

from qagents import RunContext, tool
from qagents.errors import ConfigurationError, ToolArgumentError
from qagents.tools import render_tool_output


class Address(BaseModel):
    city: str
    zip_code: str


def test_schema_comes_from_signature_and_docstring():
    @tool
    def get_weather(city: str, unit: Literal["c", "f"] = "c", days: int = 1) -> str:
        """Look up the weather.

        Longer explanation that belongs to the description.

        Args:
            city: City name.
            unit: Temperature unit,
                continued on a second line.

        Returns:
            Ignored.
        """
        return city

    assert get_weather.name == "get_weather"
    assert get_weather.description == (
        "Look up the weather.\n\nLonger explanation that belongs to the description."
    )
    schema = get_weather.parameters
    assert schema["required"] == ["city"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["city"] == {"type": "string", "description": "City name."}
    assert schema["properties"]["unit"]["enum"] == ["c", "f"]
    assert schema["properties"]["unit"]["description"] == (
        "Temperature unit, continued on a second line."
    )
    assert "title" not in schema and "title" not in schema["properties"]["days"]


async def test_context_is_injected_and_hidden_from_the_model():
    @tool(name="whoami")
    async def identify(ctx: RunContext[dict], greeting: str) -> str:
        """Greet the current user."""
        return f"{greeting}, {ctx.deps['user']}"

    assert list(identify.parameters["properties"]) == ["greeting"]
    assert identify.name == "whoami"
    result = await identify.run(RunContext(deps={"user": "Maria"}), {"greeting": "Hello"})
    assert result == "Hello, Maria"


async def test_sync_functions_run_off_the_event_loop():
    @tool
    def where() -> str:
        """Report the thread."""
        return threading.current_thread().name

    assert await where.run(RunContext(), {}) != threading.current_thread().name


async def test_invalid_arguments_are_described_for_the_model():
    @tool
    def ship(address: Address, quantity: int) -> str:
        """Ship an order."""
        return f"{quantity} to {address.city}"

    assert (
        await ship.run(
            RunContext(), {"address": {"city": "Athens", "zip_code": "10431"}, "quantity": 2}
        )
        == "2 to Athens"
    )
    with pytest.raises(ToolArgumentError) as failure:
        await ship.run(RunContext(), {"address": {"city": "Athens"}, "quantity": "many", "x": 1})
    message = str(failure.value)
    assert "address.zip_code" in message
    assert "quantity" in message
    assert "x" in message  # unknown arguments are refused, not silently dropped


def test_functions_that_cannot_be_described_are_rejected():
    with pytest.raises(ConfigurationError, match="kwargs"):

        @tool
        def anything(**kwargs: str) -> str:
            """Do anything."""
            return ""

    with pytest.raises(ConfigurationError, match="not valid"):
        tool(lambda x: x, name="bad name!")


async def test_a_tool_can_set_its_own_timeout():
    @tool(timeout=0.01)
    async def slow() -> str:
        """Take too long."""
        await asyncio.sleep(1)
        return "never"

    with pytest.raises(TimeoutError):
        await slow.run(RunContext(), {})


def test_tool_output_rendering():
    assert render_tool_output(None) == "OK"
    assert render_tool_output("plain") == "plain"
    assert render_tool_output({"a": 1, "b": [1, 2]}) == '{"a": 1, "b": [1, 2]}'
    assert render_tool_output(Address(city="Αθήνα", zip_code="1")) == (
        '{"city": "Αθήνα", "zip_code": "1"}'
    )
