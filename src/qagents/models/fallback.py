"""Several models behind one: the next one answers when the previous one cannot.


    from qagents.models.fallback import FallbackModel

    model = FallbackModel(["qcloud:<model-id>", anthropic_model])
    agent = Agent(model, tools=[...])

Each request goes to the first model. The next model is tried only when the
failure is one that a different provider can fix and that cannot have cost
anything:

* the request failed before any event was streamed: once text has been shown,
  it cannot be taken back, so a later failure is raised;
* the error is one of ``fallback_on``: by default a rate limit, a server
  error, a connection that was never made, a billing problem or a rejected API
  key, all of which belong to one provider account;
* the error is not ``ambiguous``: a provider that may already have processed
  (and billed) the request is not followed by a second, paid request unless you
  set ``allow_ambiguous=True``.

A bad request (HTTP 400, 404, 422, or a missing capability) never falls back:
the next model would get the same request.

The `ModelResponse` names the model and provider that answered, so traces and
logs show when a fallback happened. Every fallback is also logged as a warning
on the ``qagents.fallback`` logger.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator, Iterable, Sequence

from ..errors import (
    ConfigurationError,
    ModelAuthenticationError,
    ModelBadRequestError,
    ModelBillingError,
    ModelConnectionError,
    ModelError,
    ModelRateLimitError,
    ModelServerError,
)
from .base import (
    Model,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelSettings,
    ModelStreamEvent,
    ResponseDone,
    resolve_model,
)

logger = logging.getLogger("qagents.fallback")

DEFAULT_FALLBACK_ON: tuple[type[ModelError], ...] = (
    ModelRateLimitError,
    ModelServerError,
    ModelConnectionError,
    ModelBillingError,
    ModelAuthenticationError,
)
"""The errors that move a request to the next model unless you pass ``fallback_on``.

Connection errors include timeouts, but a timeout after the request was sent is
``ambiguous`` and therefore only falls back with ``allow_ambiguous=True``. The
QCloud and OpenAI-compatible adapters also mark server errors (5xx) as
ambiguous, for the same reason."""


class FallbackModel(Model):
    """Try ``models`` in order until one answers.

    ``name``, ``provider`` and `capabilities` are those of the first model,
    which is the one an agent built on this model is configured for. The
    response of each request names the model that actually answered.

    Args:
        models: The models to try, first choice first. Strings of the form
            ``"provider:model-id"`` are resolved like ``Agent(model=...)``.
        fallback_on: Error types that move the request to the next model.
            Every type must be a subclass of `ModelError`. Bad requests never
            fall back, whatever this contains.
        allow_ambiguous: Also fall back after an error the provider may have
            processed and billed. Leave it off unless a second, possibly
            duplicated, paid request is acceptable.

    Raises:
        ConfigurationError: There are no models, or ``fallback_on`` contains a
            type that is not a model error or that is a bad-request error.
    """

    def __init__(
        self,
        models: Sequence[Model | str],
        *,
        fallback_on: Iterable[type[ModelError]] = DEFAULT_FALLBACK_ON,
        allow_ambiguous: bool = False,
    ) -> None:
        if isinstance(models, str | Model) or not models:
            raise ConfigurationError(
                "FallbackModel needs a list of models, first choice first, for example "
                "FallbackModel(['qcloud:<model-id>', other_model])."
            )
        self.models: list[Model] = [resolve_model(model) for model in models]
        """The models in the order they are tried."""
        self.fallback_on: tuple[type[ModelError], ...] = _checked_error_types(fallback_on)
        """Error types that move a request to the next model."""
        self.allow_ambiguous = allow_ambiguous
        """Whether an ambiguous error (possibly billed) may move a request to the next model."""
        primary = self.models[0]
        self.name = primary.name
        self.provider = primary.provider

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        failures: list[tuple[Model, ModelError]] = []
        for position, model in enumerate(self.models):
            emitted = False
            try:
                async with contextlib.aclosing(model.stream(request)) as events:
                    async for event in events:
                        if isinstance(event, ResponseDone):
                            event = ResponseDone(response=_answered_by(event.response, model))
                        emitted = True
                        yield event
                return
            except ModelError as error:
                following = self.models[position + 1] if position + 1 < len(self.models) else None
                if following is None or emitted or not self.falls_back_on(error):
                    _note_earlier_failures(error, failures)
                    raise
                failures.append((model, error))
                logger.warning(
                    "Model %s failed with %s (%s); falling back to %s.",
                    _label(model),
                    type(error).__name__,
                    _one_line(error.message),
                    _label(following),
                )

    def falls_back_on(self, error: ModelError) -> bool:
        """Whether ``error``, raised before any event was streamed, moves a request on.

        Args:
            error: The error the current model raised.

        Returns:
            ``True`` when the next model should be tried.
        """
        if isinstance(error, ModelBadRequestError):
            return False
        if error.ambiguous and not self.allow_ambiguous:
            return False
        return isinstance(error, self.fallback_on)

    async def capabilities(self) -> ModelCapabilities:
        """The capabilities of the first model."""
        return await self.models[0].capabilities()

    def validate_settings(self, settings: ModelSettings) -> None:
        """Check ``settings`` against every model, since each one may receive them.

        Raises:
            ConfigurationError: One of the models rejects the settings. The
                message names the model.
        """
        for position, model in enumerate(self.models):
            try:
                model.validate_settings(settings)
            except Exception as error:
                raise ConfigurationError(
                    f"Model {position + 1} of the FallbackModel ({_label(model)}) rejects these "
                    f"settings: {error} A FallbackModel sends the same settings to every model, "
                    "so provider-specific settings do not mix: neither fields in "
                    "ModelSettings.extra that only one provider knows, nor standard fields that "
                    "one provider refuses. Keep only settings that every model in the list "
                    "accepts."
                ) from error

    async def aclose(self) -> None:
        """Close every model, even when closing one of them fails."""
        errors: list[Exception] = []
        for model in self.models:
            try:
                await model.aclose()
            except Exception as error:
                errors.append(error)
        if errors:
            raise errors[0]

    def __repr__(self) -> str:
        chain = ", ".join(_label(model) for model in self.models)
        return f"FallbackModel([{chain}])"


def _checked_error_types(types: Iterable[type[ModelError]]) -> tuple[type[ModelError], ...]:
    checked = tuple(types)
    for kind in checked:
        if not (isinstance(kind, type) and issubclass(kind, ModelError)):
            raise ConfigurationError(
                f"fallback_on accepts subclasses of qagents.errors.ModelError, not {kind!r}. "
                "Providers map their failures to these types."
            )
        if issubclass(kind, ModelBadRequestError):
            raise ConfigurationError(
                f"fallback_on cannot contain {kind.__name__}: a bad request is wrong for every "
                "model, so it never falls back. Fix the request, or remove the setting, tool "
                "or input that the provider rejected."
            )
    return checked


def _answered_by(response: ModelResponse, model: Model) -> ModelResponse:
    """Fill in which model answered, when the provider adapter left it out."""
    if response.model is not None and response.provider is not None:
        return response
    return response.model_copy(
        update={
            "model": response.model or model.name,
            "provider": response.provider or model.provider,
        }
    )


def _note_earlier_failures(error: ModelError, failures: list[tuple[Model, ModelError]]) -> None:
    if not failures:
        return
    tried = "; ".join(
        f"{_label(model)}: {type(failure).__name__}: {_one_line(failure.message)}"
        for model, failure in failures
    )
    note = f"FallbackModel tried these models first: {tried}"
    if note not in getattr(error, "__notes__", ()):
        error.add_note(note)


def _label(model: Model) -> str:
    return f"{model.provider}:{model.name}"


def _one_line(text: str, limit: int = 300) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else f"{flat[:limit]}..."
