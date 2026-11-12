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

A bad request (a 4xx other than 401, 402, 403, 408, 409 and 429, a redirect, or
a missing capability) never falls back: the next model would get the same
request.

Models in one chain often need different settings (one provider takes a
``temperature``, another refuses it): ``model_settings`` gives each model its
own changes. With ``cooldown_s``, a model that just failed is tried after the
others for a while, instead of failing again on every request.

The `ModelResponse` names the model and provider that answered, and its
message's ``metadata["fallback"]`` records which models failed first, so the
journal, traces and logs show when a fallback happened. Every fallback is also
logged as a warning on the ``qagents.fallback`` logger.
"""

from __future__ import annotations

import contextlib
import logging
import math
import time
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
    response of each request names the model that actually answered, and its
    message carries ``metadata["fallback"]``: ``answered_by`` (``"provider:model"``),
    ``position`` (its index in ``models``) and ``failed`` (the models that failed
    first, each as ``"provider:model: ErrorType"``).

    Args:
        models: The models to try, first choice first. Strings of the form
            ``"provider:model-id"`` are resolved like ``Agent(model=...)``.
        fallback_on: Error types that move the request to the next model.
            Every type must be a subclass of `ModelError`. Bad requests never
            fall back, whatever this contains.
        allow_ambiguous: Also fall back after an error the provider may have
            processed and billed. Leave it off unless a second, possibly
            duplicated, paid request is acceptable.
        model_settings: One entry per model, merged over the request's
            settings when that model is asked (``None`` changes nothing). A
            field set to ``None`` explicitly removes it, as in
            ``ModelSettings(temperature=None)``; keys in ``extra`` can be added
            or changed, not removed, so keep provider-specific ``extra`` fields
            here rather than in the agent's settings.
        cooldown_s: After a model fails with an error that moves the request on,
            try it after the other models for this many seconds. ``None`` (the
            default) always keeps the order of ``models``.

    Raises:
        ConfigurationError: There are no models, ``fallback_on`` contains a
            type that is not a model error or that is a bad-request error,
            ``model_settings`` does not have one entry per model, or
            ``cooldown_s`` is negative.
    """

    def __init__(
        self,
        models: Sequence[Model | str],
        *,
        fallback_on: Iterable[type[ModelError]] = DEFAULT_FALLBACK_ON,
        allow_ambiguous: bool = False,
        model_settings: Sequence[ModelSettings | None] | None = None,
        cooldown_s: float | None = None,
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
        self.model_settings: list[ModelSettings | None] = _checked_settings(
            model_settings, len(self.models)
        )
        """Per-model settings merged over each request's settings, one entry per model."""
        if cooldown_s is not None and not (math.isfinite(cooldown_s) and cooldown_s >= 0):
            raise ConfigurationError(f"cooldown_s must be zero or more, not {cooldown_s!r}.")
        self.cooldown_s = cooldown_s
        """How long a failed model is tried after the others; ``None`` for never."""
        self._cooling_until: dict[int, float] = {}
        primary = self.models[0]
        self.name = primary.name
        self.provider = primary.provider

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        failures: list[tuple[Model, ModelError]] = []
        order = self._order()
        for step, position in enumerate(order):
            model = self.models[position]
            emitted = False
            try:
                async with contextlib.aclosing(
                    model.stream(self._request_for(position, request))
                ) as events:
                    async for event in events:
                        if isinstance(event, ResponseDone):
                            event = ResponseDone(
                                response=_answered_by(event.response, model, position, failures)
                            )
                        emitted = True
                        yield event
                self._cooling_until.pop(position, None)
                return
            except ModelError as error:
                moves_on = not emitted and self.falls_back_on(error)
                if moves_on and self.cooldown_s:
                    self._cooling_until[position] = time.monotonic() + self.cooldown_s
                following = self.models[order[step + 1]] if step + 1 < len(order) else None
                if following is None or not moves_on:
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
        """Check ``settings``, with each model's ``model_settings``, against every model.

        Raises:
            ConfigurationError: One of the models rejects the settings. The
                message names the model.
        """
        for position, model in enumerate(self.models):
            try:
                model.validate_settings(settings.merged(self.model_settings[position]))
            except Exception as error:
                raise ConfigurationError(
                    f"Model {position + 1} of the FallbackModel ({_label(model)}) rejects these "
                    f"settings: {error} A FallbackModel sends the agent's settings to every "
                    "model, so provider-specific settings do not mix: neither fields in "
                    "ModelSettings.extra that only one provider knows, nor standard fields that "
                    "one provider refuses. Keep only settings that every model accepts, and "
                    "give each model its own with FallbackModel(..., model_settings=[...])."
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

    def _request_for(self, position: int, request: ModelRequest) -> ModelRequest:
        changes = self.model_settings[position]
        if changes is None:
            return request
        return request.model_copy(update={"settings": request.settings.merged(changes)})

    def _order(self) -> list[int]:
        """Positions in the order to try: models cooling down after a failure go last."""
        positions = list(range(len(self.models)))
        if not self._cooling_until:
            return positions
        now = time.monotonic()
        cooling = [p for p in positions if self._cooling_until.get(p, 0.0) > now]
        if cooling:
            logger.debug(
                "Trying %s last: %s failed less than cooldown_s=%g s ago.",
                ", ".join(_label(self.models[p]) for p in cooling),
                "it" if len(cooling) == 1 else "they",
                self.cooldown_s,
            )
        return [p for p in positions if p not in cooling] + cooling

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


def _checked_settings(
    settings: Sequence[ModelSettings | None] | None, count: int
) -> list[ModelSettings | None]:
    if settings is None:
        return [None] * count
    checked = list(settings)
    if len(checked) != count or not all(
        item is None or isinstance(item, ModelSettings) for item in checked
    ):
        raise ConfigurationError(
            f"model_settings needs one ModelSettings or None per model ({count}), in the order "
            "of models, for example model_settings=[None, ModelSettings(temperature=None)]."
        )
    return checked


def _answered_by(
    response: ModelResponse,
    model: Model,
    position: int,
    failures: list[tuple[Model, ModelError]],
) -> ModelResponse:
    """Record which model answered, and which failed before it."""
    record = {
        "answered_by": _label(model),
        "position": position,
        "failed": [f"{_label(failed)}: {type(error).__name__}" for failed, error in failures],
    }
    message = response.message.model_copy(
        update={"metadata": {**response.message.metadata, "fallback": record}}
    )
    return response.model_copy(
        update={
            "message": message,
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
