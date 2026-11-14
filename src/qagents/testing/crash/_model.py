"""Per-leg model clients over one point's durable-turn script and spend ledger."""

from __future__ import annotations

import math
import os
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from qagents.effects import _ports
from qagents.effects._canonical import digest
from qagents.errors import CassetteMissError, ModelError
from qagents.messages import Message, ToolCallPart, Usage
from qagents.models.base import Model, ModelRequest, ModelResponse, ModelStreamEvent, ResponseDone
from qagents.models.recording import ReplayModel, request_fingerprint
from qagents.models.testing import ScriptedModel


class UnscriptedRequest(ModelError):
    """A request has no script answer; the crash point is inconclusive."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False, ambiguous=False)


@dataclass(frozen=True)
class _When:
    predicate: Callable[[ModelRequest], bool]
    reply: Any


def when(predicate: Callable[[ModelRequest], bool], reply: Any) -> Any:
    """Choose a reply by its request, including requests without a logical turn."""
    if not callable(predicate):
        raise TypeError("when requires a callable predicate")
    return _When(predicate, reply)


@dataclass(frozen=True)
class Cassette:
    """A strict ReplayModel cassette opened afresh for every recovery leg."""

    path: Path

    def __init__(self, path: str | os.PathLike[str]) -> None:
        object.__setattr__(self, "path", Path(path))


def _logical_turn(request: ModelRequest, turn: _ports.ModelTurn | None) -> int:
    """Use the host's journal-derived turn, never the compacted message count (M29)."""
    if turn is None or type(turn.turn) is not int or turn.turn < 0:
        raise UnscriptedRequest("No valid logical turn; add a when(...) reply for this request.")
    return turn.turn


def _on_cassette_miss(error: CassetteMissError) -> Literal["inconclusive"]:
    """A cassette miss is missing evidence, not a successful recovery (M30)."""
    return "inconclusive"


@dataclass
class ModelCall:
    """One actual model consultation, including a retry of an unrecorded response."""

    leg: int
    session_id: str | None
    run_id: str | None
    turn: int | None
    fingerprint: str
    request: ModelRequest
    response: ModelResponse | None = None
    usage: Usage = field(default_factory=Usage)
    outcome: str = "pending"
    error_type: str | None = None
    perturbation: str | None = None
    changed_field: str | None = None
    key_sensitive: bool = False


PERTURBATIONS = frozenset({"fresh_ids", "retry_effect", "retry_changed", "permuted", "abandon"})


class ModelLedger:
    """Shared request/reply evidence and usage for the legs of one crash point.

    Positional replies are indexed by MODEL_TURN. Matching replies do not take a
    positional slot. Every consultation is recorded; recovering a lost response
    does not erase its spend or consume the next logical reply.
    """

    def __init__(self, script: Sequence[Any] | Cassette, *, seed: int = 0) -> None:
        if type(seed) is not int:
            raise ValueError("model ledger seed must be an integer")
        if isinstance(script, str | bytes):
            raise TypeError("model script must be a sequence of replies, not text")
        self.script = script if isinstance(script, Cassette) else tuple(script)
        self.seed = seed
        self.calls: list[ModelCall] = []

    @property
    def usage(self) -> Usage:
        result = Usage()
        for call in self.calls:
            result = result.add(call.usage)
        return result

    def for_leg(
        self,
        leg: int,
        *,
        recovering: bool = False,
        perturbation: str | None = None,
        effect_tools: Sequence[str] = (),
        key_sensitive_fields: Mapping[str, Sequence[str]] | None = None,
        change_field: str | None = None,
    ) -> LedgerModel:
        """Create a client with independent cassette consumption for this leg."""
        return LedgerModel(
            self,
            leg=leg,
            recovering=recovering,
            perturbation=perturbation,
            effect_tools=effect_tools,
            key_sensitive_fields=key_sensitive_fields,
            change_field=change_field,
        )


def _changed_value(value: Any) -> Any:
    # Deterministic probes retain the primitive type and mutate one top-level field.
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        return value + "-changed"
    if isinstance(value, int):
        return value + 1 if value < 2**53 - 1 else value - 1
    if isinstance(value, float):
        result = value + 1.0
        return result if math.isfinite(result) and result != value else 0.0
    if value is None:
        return "changed"
    if isinstance(value, list):
        return [] if value else [None]
    if isinstance(value, dict):
        return {} if value else {"changed": True}
    raise TypeError("model argument probes require JSON values")


def change_candidates(
    call: ToolCallPart, *, key_sensitive_fields: Sequence[str] = ()
) -> list[tuple[str, ToolCallPart]]:
    """One changed argument per candidate, measured key-sensitive fields first.

    The harness supplies key-sensitive fields after probing actual operation keys;
    names alone never establish that an argument feeds an effect's key.
    """
    fields = list(dict.fromkeys([*key_sensitive_fields, *call.arguments]))
    result = []
    for name in fields:
        if name not in call.arguments:
            continue
        changed = call.model_copy(deep=True)
        changed.arguments[name] = _changed_value(call.arguments[name])
        changed.raw_arguments = None
        changed.parse_error = None
        result.append((name, changed))
    return result


class LedgerModel(Model):
    """A model client for one leg; perturb only a real recovery consultation."""

    name = "crash-ledger"
    provider = "test"

    def __init__(
        self,
        ledger: ModelLedger,
        *,
        leg: int,
        recovering: bool = False,
        perturbation: str | None = None,
        effect_tools: Sequence[str] = (),
        key_sensitive_fields: Mapping[str, Sequence[str]] | None = None,
        change_field: str | None = None,
    ) -> None:
        if type(leg) is not int or leg < 0:
            raise ValueError("model leg must be a nonnegative integer")
        if perturbation is not None and perturbation not in PERTURBATIONS:
            raise ValueError(f"unknown model perturbation {perturbation!r}")
        self.ledger = ledger
        self.leg = leg
        self.recovering = recovering
        self.perturbation = perturbation
        self.effect_tools = frozenset(effect_tools)
        self.key_sensitive_fields = {
            key: tuple(values) for key, values in (key_sensitive_fields or {}).items()
        }
        self.change_field = change_field
        self.requests: list[ModelRequest] = []
        self._perturbed = False
        self._replay = (
            ReplayModel(ledger.script.path) if isinstance(ledger.script, Cassette) else None
        )

    def _reply(self, request: ModelRequest, turn: _ports.ModelTurn | None) -> Any:
        positional = []
        for reply in self.ledger.script:
            if isinstance(reply, _When):
                matched = reply.predicate(request.model_copy(deep=True))
                if type(matched) is not bool:
                    raise TypeError("when predicates must return bool")
                if matched:
                    return reply.reply
            else:
                positional.append(reply)
        index = _logical_turn(request, turn)
        if index >= len(positional):
            raise UnscriptedRequest(f"No scripted reply for logical turn {index}.")
        return positional[index]

    def _fresh_id(self, call: ModelCall, position: int) -> str:
        return (
            "call_"
            + digest([self.ledger.seed, self.leg, call.session_id, call.turn, position])[:24]
        )

    def _last_effect(self, current: ModelCall) -> ToolCallPart:
        for record in reversed(self.ledger.calls):
            if (
                record is current
                or record.session_id != current.session_id
                or record.response is None
            ):
                continue
            for call in reversed(record.response.message.tool_calls):
                if call.name in self.effect_tools:
                    return call.model_copy(deep=True)
        raise UnscriptedRequest("Recovery perturbation has no preceding effects call to retry.")

    def _perturb(self, response: ModelResponse, call: ModelCall) -> ModelResponse:
        # Auxiliary requests (e.g. compaction) are not logical main-model turns.
        if not self.recovering or not self.perturbation or self._perturbed or call.turn is None:
            return response
        kind = self.perturbation
        changed = response.model_copy(deep=True)
        if kind in ("retry_effect", "retry_changed"):
            retry = self._last_effect(call)
            if kind == "retry_changed":
                sensitive = self.key_sensitive_fields.get(retry.name, ())
                candidates = change_candidates(retry, key_sensitive_fields=sensitive)
                if not candidates:
                    raise UnscriptedRequest(
                        "Effects call has no argument that retry_changed can change."
                    )
                selected = (
                    next(
                        (
                            candidate
                            for candidate in candidates
                            if candidate[0] == self.change_field
                        ),
                        None,
                    )
                    if self.change_field is not None
                    else candidates[0]
                )
                if selected is None:
                    raise UnscriptedRequest(f"Effects call has no argument {self.change_field!r}.")
                call.changed_field, retry = selected
                call.key_sensitive = call.changed_field in sensitive
            retry.id = self._fresh_id(call, 0)
            changed.message = Message.assistant(tool_calls=[retry])
            changed.finish_reason = "tool_calls"
        elif kind == "fresh_ids":
            if not changed.message.tool_calls:
                return response
            for position, tool_call in enumerate(changed.message.tool_calls):
                tool_call.id = self._fresh_id(call, position)
        elif kind == "permuted":
            if len(changed.message.tool_calls) < 2:
                return response
            reversed_calls = iter(reversed(changed.message.tool_calls))
            changed.message.parts = [
                next(reversed_calls) if isinstance(part, ToolCallPart) else part
                for part in changed.message.parts
            ]
        elif kind == "abandon":
            changed.message = Message.assistant("The scripted recovery model stopped this task.")
            changed.finish_reason = "stop"
        self._perturbed = True
        call.perturbation = kind
        return changed

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        turn = _ports.MODEL_TURN.get()
        snapshot = request.model_copy(deep=True)
        self.requests.append(snapshot.model_copy(deep=True))
        record = ModelCall(
            self.leg,
            turn.session_id if turn else None,
            turn.run_id if turn else None,
            turn.turn if turn else None,
            request_fingerprint(snapshot),
            snapshot,
        )
        self.ledger.calls.append(record)
        try:
            model = self._replay or ScriptedModel([self._reply(snapshot, turn)], name=self.name)
            events = [event async for event in model.stream(snapshot.model_copy(deep=True))]
            complete = next(
                (event.response for event in reversed(events) if isinstance(event, ResponseDone)),
                None,
            )
            if complete is None:
                raise UnscriptedRequest("Scripted model stream has no complete response.")
            # The provider consultation happened even if the later perturbation is unavailable.
            record.usage = complete.usage.model_copy(deep=True)
            response = self._perturb(complete.model_copy(deep=True), record)
            record.response = response.model_copy(deep=True)
            record.outcome = "completed"
        except CassetteMissError as error:
            record.outcome = _on_cassette_miss(error)
            record.error_type = type(error).__name__
            raise
        except BaseException as error:
            record.outcome = (
                "inconclusive" if isinstance(error, UnscriptedRequest) else "failed_unexpected"
            )
            record.error_type = type(error).__name__
            raise
        if record.perturbation is None:
            for event in events:
                yield event.model_copy(deep=True)
        else:
            # Original cassette deltas contain the old tool ids/arguments; do not replay them.
            yield ResponseDone(response=response.model_copy(deep=True))

    async def aclose(self) -> None:
        if self._replay is not None:
            await self._replay.aclose()
