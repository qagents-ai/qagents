"""Model providers and the wrappers that combine them.

``QCloudModel`` is the primary provider. ``OpenAIChatModel`` reaches any other
server that speaks OpenAI Chat Completions, and ``AnthropicModel`` the Claude
Messages API (``pip install "qagents[anthropic]"``). To add a provider,
subclass `Model` and implement ``stream``.

Wrappers are models too, so they nest:

| Wrapper | For |
|---|---|
| `FallbackModel` | try a second provider when the first refuses at admission |
| `RateLimitedModel` | stay under a provider's concurrency, request and token limits |
| `cassette`, `RecordingModel`, `ReplayModel` | record real responses once, replay them in tests |

The ``"anthropic:"`` prefix is registered without importing the Anthropic
adapter: ``Agent("anthropic:claude-opus-5-5")`` imports it on first use, and
``qagents.models.AnthropicModel`` is resolved the same way.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from ..errors import CassetteMissError
from .base import (
    Model,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelSettings,
    ModelStreamEvent,
    OutputSpec,
    ReasoningDelta,
    ResponseDone,
    TextDelta,
    ToolCallDelta,
    ToolSpec,
    register_provider,
    resolve_model,
)
from .fallback import FallbackModel
from .openai_chat import OpenAIChatModel
from .qcloud import QCloudModel
from .ratelimit import RateLimitedModel
from .recording import RecordingModel, ReplayModel, cassette, request_fingerprint

if TYPE_CHECKING:
    # Re-exports for type checkers; at run time __getattr__ resolves them.
    from .anthropic import AnthropicModel as AnthropicModel
    from .anthropic import AnthropicPricing as AnthropicPricing

_LAZY = {
    "AnthropicModel": "qagents.models.anthropic",
    "AnthropicPricing": "qagents.models.anthropic",
}
"""Names resolved on first access, so importing this package imports no provider SDK."""


def _anthropic(name: str) -> Model:
    from .anthropic import AnthropicModel

    return AnthropicModel(name)


register_provider("qcloud", QCloudModel)
register_provider("anthropic", _anthropic)


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})


__all__ = [
    "CassetteMissError",
    "FallbackModel",
    "Model",
    "ModelCapabilities",
    "ModelRequest",
    "ModelResponse",
    "ModelSettings",
    "ModelStreamEvent",
    "OpenAIChatModel",
    "OutputSpec",
    "QCloudModel",
    "RateLimitedModel",
    "ReasoningDelta",
    "RecordingModel",
    "ReplayModel",
    "ResponseDone",
    "TextDelta",
    "ToolCallDelta",
    "ToolSpec",
    "cassette",
    "register_provider",
    "request_fingerprint",
    "resolve_model",
]
