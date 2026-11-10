"""Every exception qagents raises.

Model errors carry two flags the agent loop acts on:

* ``retryable``: the request can be sent again and may succeed.
* ``ambiguous``: the provider may already have processed (and billed) the
  request. The loop never replays an ambiguous request by itself.
"""

from __future__ import annotations


class QAgentsError(Exception):
    """Base class for every error raised by qagents."""


class ConfigurationError(QAgentsError):
    """The agent, a tool or a model was set up incorrectly."""


class SessionError(QAgentsError):
    """A session could not be loaded, resumed or written."""


class OutputValidationError(QAgentsError):
    """The model's final answer did not match ``output_type`` after all retries."""


class ToolError(QAgentsError):
    """Raise this inside a tool to report a failure to the model.

    The message is sent back as the tool result (marked as an error), so write
    it for the model: say what went wrong and what it can do instead.
    """


class ToolArgumentError(ToolError):
    """The model called a tool with arguments that do not match its schema."""


class ModelError(QAgentsError):
    """A model request failed."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        code: str | None = None,
        request_id: str | None = None,
        retry_after: float | None = None,
        retryable: bool = False,
        ambiguous: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.code = code
        self.request_id = request_id
        self.retry_after = retry_after
        self.retryable = retryable
        self.ambiguous = ambiguous


class ModelBadRequestError(ModelError):
    """The provider rejected the request before running it (400, 404, 422)."""


class ModelAuthenticationError(ModelError):
    """The API key is missing, invalid or not allowed to use this model (401, 403)."""


class ModelBillingError(ModelError):
    """The account cannot pay for the request (402)."""


class ModelRateLimitError(ModelError):
    """The provider refused the request at admission (429). Safe to retry."""


class ModelServerError(ModelError):
    """The provider failed while handling the request (5xx)."""


class ModelConnectionError(ModelError):
    """The provider could not be reached, or the connection broke."""


class ModelTimeoutError(ModelConnectionError):
    """The provider did not answer in time."""


class ModelResponseError(ModelError):
    """The provider answered with something that is not a valid response."""
