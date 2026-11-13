"""Effect declarations, explicit business keys, and destination contracts."""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, Literal, NoReturn, TypeVar, get_args, get_origin, get_type_hints

from pydantic import BaseModel

from qagents.errors import ConfigurationError, ToolError

from ._canonical import JSONValue, digest, jcs
from ._types import Attempt, EffectContext

ReqT = TypeVar("ReqT", bound=BaseModel)
RecT = TypeVar("RecT", bound=BaseModel)
DepsT = TypeVar("DepsT")
Send = Callable[[EffectContext[Any], Attempt[ReqT]], Awaitable[RecT] | RecT]
Verdict = "Found[Any] | NotYetVisible | Absent | Unknown"


def _pct(value: str, separator: str = ":") -> str:
    jcs(value)
    return value.replace("%", "%25").replace(separator, f"%{ord(separator):02X}")


@dataclass(frozen=True, init=False)
class Key:
    """A business identity over scalar request fields; use per_call for call-only protection."""

    prefix: str
    fields: tuple[str, ...]
    is_per_call: bool

    def __init__(self, prefix: str, *fields: str) -> None:
        """Render a business key as prefix:v1:v2, escaping delimiters within values."""
        if not isinstance(prefix, str) or not prefix:
            raise ConfigurationError("Key prefix must be a nonempty string.")
        if not fields or any(not isinstance(field, str) or not field for field in fields):
            raise ConfigurationError("Key requires one or more request field names.")
        if len(set(fields)) != len(fields):
            raise ConfigurationError("Key field names must not repeat.")
        jcs(prefix)
        object.__setattr__(self, "prefix", prefix)
        object.__setattr__(self, "fields", tuple(fields))
        object.__setattr__(self, "is_per_call", False)

    @classmethod
    def per_call(cls) -> Key:
        """A call/<operation_id>/<effect>[/<occurrence>] key protects reruns of one call only."""
        result = object.__new__(cls)
        object.__setattr__(result, "prefix", "call")
        object.__setattr__(result, "fields", ())
        object.__setattr__(result, "is_per_call", True)
        return result

    def render(
        self, request: BaseModel, *, effect: str, operation_id: str, occurrence: int | None
    ) -> str:
        """Return an injectively encoded key within this declared key schema."""
        if occurrence is not None and (type(occurrence) is not int or occurrence < 0):
            raise ConfigurationError("occurrence must be a nonnegative integer or None.")
        if self.is_per_call:
            key = f"call/{_pct(operation_id, '/')}/{_pct(effect, '/')}"
            return key if occurrence is None else f"{key}/{occurrence}"
        values = []
        for field in self.fields:
            value = getattr(request, field)
            if type(value) not in (str, int, bool):
                raise ConfigurationError(f"Key field {field!r} must be str, int, or bool.")
            jcs(value)
            text = ("true" if value else "false") if isinstance(value, bool) else str(value)
            values.append(_pct(text))
        return self.prefix + ":" + ":".join(values)

    def spec_digest(self) -> str:
        """Hash the declared field order and key mode for the effect registry."""
        return digest(
            {"prefix": self.prefix, "fields": list(self.fields), "per_call": self.is_per_call}
        )


def _finite_nonnegative(value: float, name: str, *, positive: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        bound = "positive" if positive else "nonnegative"
        raise ConfigurationError(f"{name} must be a finite {bound} number.")


@dataclass(frozen=True)
class Dedupe:
    """A destination's bounded key-retention contract, with pessimistic defaults.

    Concurrency defaults to unknown, errors are presumed replayed, rejection
    replay and parameter binding are not promised. Prefer a probed preset.
    """

    window_s: float
    contract: str
    concurrent: Literal["rejects", "waits", "unknown"] = "unknown"
    replays_errors: bool = True
    replays_rejections: bool = False
    binds_params: bool = False
    margin_s: float | None = None

    def __post_init__(self) -> None:
        _finite_nonnegative(self.window_s, "window_s", positive=True)
        if not isinstance(self.contract, str) or not self.contract.strip():
            raise ConfigurationError(
                "Dedupe contract must name the destination's documented promise."
            )
        if self.concurrent not in ("rejects", "waits", "unknown"):
            raise ConfigurationError("concurrent must be rejects, waits, or unknown.")
        for name in ("replays_errors", "replays_rejections", "binds_params"):
            if type(getattr(self, name)) is not bool:
                raise ConfigurationError(f"{name} must be bool.")
        if self.margin_s is not None:
            _finite_nonnegative(self.margin_s, "margin_s")

    def margin(self) -> float:
        """Return the explicit margin, or max(60 seconds, 5% of the retention window)."""
        return self.margin_s if self.margin_s is not None else max(60.0, 0.05 * self.window_s)


@dataclass(frozen=True)
class Unique:
    """A destination's permanent unique-key promise and its definite duplicate exceptions."""

    contract: str
    on_duplicate: tuple[type[BaseException], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.contract, str) or not self.contract.strip():
            raise ConfigurationError(
                "Unique contract must name the destination's permanent constraint."
            )
        if not isinstance(self.on_duplicate, tuple) or not self.on_duplicate:
            raise ConfigurationError(
                "Unique on_duplicate needs a nonempty tuple of exception types."
            )
        if any(
            not isinstance(kind, type) or not issubclass(kind, BaseException)
            for kind in self.on_duplicate
        ):
            raise ConfigurationError("Unique on_duplicate entries must be exception types.")


REACH_NAMES = (
    "first_send",
    "retry_after_rejection",
    "dedupe_resend",
    "absence_resend",
    "operator_resend",
    "found_by_lookup",
    "receipt_replayed",
    "claim_replayed",
    "held",
    "held_after_window",
    "operator_commit",
    "committed",
    "unknown_outcome",
)


@dataclass(frozen=True)
class ReachPoint:
    """A named decision boundary that tests may require an effect to reach."""

    effect: str
    name: str


class ReachPoints:
    """Expose exactly the supported coverage points for one declared effect."""

    def __init__(self, effect: str) -> None:
        self._effect = effect

    def __getattr__(self, name: str) -> ReachPoint:
        if name not in REACH_NAMES:
            raise AttributeError(name)
        return ReachPoint(self._effect, name)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(REACH_NAMES))


class Effect(Generic[ReqT, RecT]):
    """A declared action created by @effect; execute it through RunContext.perform."""

    def __init__(
        self,
        name: str,
        send: Send[ReqT, RecT],
        *,
        key: Key,
        destination: Callable[[Any], JSONValue] | None,
        dedupe: Dedupe | Unique | None,
        applies_within_s: float | None,
        timeout_s: float,
        rejected_on: tuple[type[BaseException], ...],
        receipt_identity: tuple[str, ...] | None,
        namespace: str | Callable[[Any], str],
        on_unknown: Literal["pause", "fail_run", "tell_model"],
        approval: Literal["tool", "effect"],
        version: str,
        idempotency_key_max_len: int,
        request_type: type[ReqT],
        receipt_type: type[RecT],
    ) -> None:
        self.name = name
        self.key = key
        self.destination = destination
        self.dedupe = dedupe
        self.applies_within_s = applies_within_s
        self.timeout_s = timeout_s
        self.rejected_on = rejected_on
        self.receipt_identity = receipt_identity
        self.namespace = namespace
        self.on_unknown = on_unknown
        self.approval = approval
        self.version = version
        self.idempotency_key_max_len = idempotency_key_max_len
        self.request_type = request_type
        self.receipt_type = receipt_type
        self.send = send
        self.is_async = inspect.iscoroutinefunction(send)
        self.lookup_fn: Callable[..., Any] | None = None
        self.authoritative = False
        self.on_commit_fn: Callable[[dict[str, Any], ReqT, RecT], None] | None = None
        self.reached = ReachPoints(name)
        self.key_spec_digest = key.spec_digest()
        self.request_schema_digest = digest(request_type.model_json_schema())

    @property
    def klass(self) -> Literal["deduplicated", "unique", "authoritative", "reconcilable", "opaque"]:
        """The strongest declared destination contract; no stronger than its assumptions."""
        if isinstance(self.dedupe, Dedupe):
            return "deduplicated"
        if isinstance(self.dedupe, Unique):
            return "unique"
        if self.authoritative:
            return "authoritative"
        return "reconcilable" if self.lookup_fn is not None else "opaque"

    def __call__(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError(f"call `await ctx.perform({self.name}, request)` instead")

    def lookup(self, fn: Callable | None = None, *, authoritative: bool = False) -> Any:
        """Attach a lookup; authoritative absence needs a bounded, asynchronous send."""
        if type(authoritative) is not bool:
            raise ConfigurationError("authoritative must be bool.")
        if authoritative and (self.applies_within_s is None or not self.is_async):
            raise ConfigurationError("authoritative=True needs applies_within_s and an async send.")

        def attach(function: Callable) -> Callable:
            if not callable(function):
                raise ConfigurationError("An effect lookup must be callable.")
            self.lookup_fn = function
            self.authoritative = authoritative
            return function

        return attach if fn is None else attach(fn)

    def on_commit(self, fn: Callable[[dict[str, Any], ReqT, RecT], None]) -> Callable:
        """Attach a synchronous patch of committed state, journaled with the receipt."""
        if not callable(fn) or inspect.iscoroutinefunction(fn):
            raise ConfigurationError("on_commit must be a synchronous function.")
        self.on_commit_fn = fn
        return fn


def effect(
    name: str,
    *,
    key: Key,
    destination: Callable[[Any], JSONValue] | None = None,
    dedupe: Dedupe | Unique | None = None,
    applies_within_s: float | None = None,
    timeout_s: float = 20.0,
    rejected_on: tuple[type[BaseException], ...] = (),
    receipt_identity: tuple[str, ...] | None = None,
    namespace: str | Callable[[Any], str] = "session",
    on_unknown: Literal["pause", "fail_run", "tell_model"] = "pause",
    approval: Literal["tool", "effect"] = "tool",
    version: str = "1",
    idempotency_key_max_len: int = 64,
) -> Callable[[Send[ReqT, RecT]], Effect[ReqT, RecT]]:
    """Declare a typed effect, validating local contracts before any run starts.

    Unique's required lookup is checked when an agent is built, after the
    lookup decorator has had a chance to attach it.
    """
    if not isinstance(name, str) or not name.strip():
        raise ConfigurationError("effect name must be a nonempty string.")
    if not isinstance(key, Key):
        raise ConfigurationError("effect key must be a Key declaration.")
    if destination is not None and not callable(destination):
        raise ConfigurationError("destination must be a function of trusted dependencies.")
    if dedupe is not None and not isinstance(dedupe, Dedupe | Unique):
        raise ConfigurationError("dedupe must be Dedupe, Unique, or None.")
    _finite_nonnegative(timeout_s, "timeout_s", positive=True)
    if applies_within_s is not None:
        _finite_nonnegative(applies_within_s, "applies_within_s")
    if not isinstance(rejected_on, tuple):
        raise ConfigurationError("rejected_on must be a tuple of definite refusal exception types.")
    for kind in rejected_on:
        if (
            not isinstance(kind, type)
            or not issubclass(kind, BaseException)
            or kind in (BaseException, Exception, ToolError)
            or issubclass(kind, (TimeoutError, asyncio.CancelledError))
        ):
            raise ConfigurationError(
                "rejected_on must not classify broad, timeout, or cancelled errors."
            )
    if on_unknown not in ("pause", "fail_run", "tell_model"):
        raise ConfigurationError("on_unknown must be pause, fail_run, or tell_model.")
    if on_unknown == "tell_model" and key.is_per_call:
        raise ConfigurationError("tell_model needs a business Key, not Key.per_call().")
    if approval != "tool":
        raise ConfigurationError("approval='effect' is unavailable until the effect-approval tier.")
    if not isinstance(version, str) or not version:
        raise ConfigurationError("version must be a nonempty string.")
    if not callable(namespace) and (not isinstance(namespace, str) or not namespace):
        raise ConfigurationError(
            "namespace must be a nonempty string or a function of dependencies."
        )
    if type(idempotency_key_max_len) is not int or idempotency_key_max_len < 24:
        raise ConfigurationError("idempotency_key_max_len must be an integer >= 24.")
    jcs([name, version])

    def decorate(send: Send[ReqT, RecT]) -> Effect[ReqT, RecT]:
        if not callable(send) or inspect.isasyncgenfunction(send):
            raise ConfigurationError("send must be a sync or async function returning a receipt.")
        parameters = list(inspect.signature(send).parameters.values())
        if len(parameters) != 2 or any(
            p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in parameters
        ):
            raise ConfigurationError(
                "send needs exactly (ctx: EffectContext, attempt: Attempt[Request])."
            )
        frame = inspect.currentframe()
        try:
            localns = frame.f_back.f_locals if frame is not None and frame.f_back else None
            hints = get_type_hints(send, localns=localns)
        except (NameError, TypeError) as error:
            raise ConfigurationError(f"Cannot resolve {name} send annotations: {error}") from error
        finally:
            del frame
        context = hints.get(parameters[0].name)
        attempt = hints.get(parameters[1].name)
        request_type = get_args(attempt)[0] if get_origin(attempt) is Attempt else None
        receipt_type = hints.get("return")
        if context is not EffectContext and get_origin(context) is not EffectContext:
            raise ConfigurationError(
                "send ctx must be annotated EffectContext or EffectContext[Deps]."
            )
        if (
            not isinstance(request_type, type)
            or request_type is BaseModel
            or not issubclass(request_type, BaseModel)
        ):
            raise ConfigurationError("send attempt must be annotated Attempt[RequestModel].")
        if (
            not isinstance(receipt_type, type)
            or receipt_type is BaseModel
            or not issubclass(receipt_type, BaseModel)
        ):
            raise ConfigurationError("send return annotation must be a receipt BaseModel type.")
        for field_name in key.fields:
            field_info = request_type.model_fields.get(field_name)
            if field_info is None or field_info.annotation not in (str, int, bool):
                raise ConfigurationError(
                    f"Key field {field_name!r} must exist and be str, int, or bool."
                )
        if receipt_identity is not None and (
            not isinstance(receipt_identity, tuple)
            or not receipt_identity
            or any(
                not isinstance(field, str) or field not in receipt_type.model_fields
                for field in receipt_identity
            )
            or len(set(receipt_identity)) != len(receipt_identity)
        ):
            raise ConfigurationError(
                "receipt_identity must name distinct fields of the receipt model."
            )
        return Effect(
            name,
            send,
            key=key,
            destination=destination,
            dedupe=dedupe,
            applies_within_s=applies_within_s,
            timeout_s=timeout_s,
            rejected_on=rejected_on,
            receipt_identity=receipt_identity,
            namespace=namespace,
            on_unknown=on_unknown,
            approval=approval,
            version=version,
            idempotency_key_max_len=idempotency_key_max_len,
            request_type=request_type,
            receipt_type=receipt_type,
        )

    return decorate
