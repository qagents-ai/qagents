"""Model providers.

``QCloudModel`` is the primary provider. ``OpenAIChatModel`` reaches any other
server that speaks OpenAI Chat Completions. To add a provider, subclass
`Model` and implement ``stream``.
"""

from __future__ import annotations

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
from .openai_chat import OpenAIChatModel
from .qcloud import QCloudModel

register_provider("qcloud", QCloudModel)

__all__ = [
    "Model",
    "ModelCapabilities",
    "ModelRequest",
    "ModelResponse",
    "ModelSettings",
    "ModelStreamEvent",
    "OpenAIChatModel",
    "OutputSpec",
    "QCloudModel",
    "ReasoningDelta",
    "ResponseDone",
    "TextDelta",
    "ToolCallDelta",
    "ToolSpec",
    "register_provider",
    "resolve_model",
]
