"""Strict canonical JSON and frozen request values for effect identities."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, TypeAlias

from pydantic import BaseModel

if TYPE_CHECKING:
    from ._declare import Effect

JSONValue: TypeAlias = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
LIMIT_BYTES = 16 * 1024


def _number(value: float) -> str:
    """Format the shortest binary64 round-trip decimal using ECMAScript's cutoffs."""
    if not math.isfinite(value):
        raise ValueError("A non-finite number is not strict JSON; send it as a string.")
    if value == 0:
        return "0"
    _, digit_tuple, exponent = Decimal(repr(abs(value))).as_tuple()
    assert isinstance(exponent, int)
    written = "".join(map(str, digit_tuple))
    digits = written.rstrip("0")
    n = exponent + len(written)
    k = len(digits)
    if k <= n <= 21:
        text = digits + "0" * (n - k)
    elif 0 < n <= 21:
        text = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        text = "0." + "0" * -n + digits
    else:
        mantissa = digits[0] + ("." + digits[1:] if k > 1 else "")
        text = f"{mantissa}e{'+' if n > 0 else '-'}{abs(n - 1)}"
    return ("-" if value < 0 else "") + text


def _string(value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("Strict JSON strings cannot contain lone surrogates.") from error
    return json.dumps(value, ensure_ascii=False)


def _encode(value: JSONValue, ancestors: set[int]) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        if abs(value) > 2**53 - 1:
            raise ValueError("Integer outside the safe JSON range; send it as a string.")
        return str(value)
    if isinstance(value, float):
        return _number(value)
    if isinstance(value, str):
        return _string(value)
    if not isinstance(value, list | dict):
        raise ValueError(f"{type(value).__name__} is not strict JSON; send it as a string.")
    identity = id(value)
    if identity in ancestors:
        raise ValueError("Cyclic objects are not JSON values.")
    ancestors.add(identity)
    try:
        if isinstance(value, list):
            return "[" + ",".join(_encode(item, ancestors) for item in value) + "]"
        for key in value:
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings.")
            _string(key)
        keys = sorted(value, key=lambda key: key.encode("utf-16-be"))
        return (
            "{"
            + ",".join(_string(key) + ":" + _encode(value[key], ancestors) for key in keys)
            + "}"
        )
    finally:
        ancestors.remove(identity)


def jcs(value: JSONValue) -> bytes:
    """RFC 8785 bytes, refusing unsafe integers, non-finite numbers, and non-JSON values.

    Values outside the safe integer range must be sent as strings. Object keys
    follow UTF-16 code-unit order; strings are not Unicode-normalized.
    """
    try:
        return _encode(value, set()).encode("utf-8")
    except RecursionError as error:
        raise ValueError("JSON nesting exceeds the interpreter's safe recursion depth.") from error


def digest(value: JSONValue) -> str:
    """The lowercase SHA-256 digest of strict JCS bytes (jcs-sha256/1)."""
    return hashlib.sha256(jcs(value)).hexdigest()


@dataclass(frozen=True)
class Frozen:
    """An independent JSON snapshot of the validated request, ready for the journal."""

    json: dict[str, JSONValue]
    digest: str
    size: int


def freeze(effect: Effect, request: BaseModel) -> Frozen:
    """Validate and snapshot an effect request; refuse a canonical size over 16 KiB."""
    if not isinstance(request, effect.request_type):
        raise TypeError(f"{effect.name} requires {effect.request_type.__name__}.")
    # Validate from values even if an instance came from model_construct().
    validated = effect.request_type.model_validate(
        request.model_dump(mode="python", warnings=False)
    )
    data = validated.model_dump(mode="json", warnings="error")
    if not isinstance(data, dict):
        raise ValueError("An effect request must serialize as a JSON object.")
    encoded = jcs(data)
    if len(encoded) > LIMIT_BYTES:
        raise ValueError(
            f"{effect.name} request exceeds the {LIMIT_BYTES}-byte canonical JSON limit."
        )
    return Frozen(json.loads(encoded), hashlib.sha256(encoded).hexdigest(), len(encoded))


def rebuild(effect: Effect, frozen: Frozen) -> BaseModel:
    """Build a fresh typed request from the journal snapshot for send or lookup."""
    return effect.request_type.model_validate(json.loads(jcs(frozen.json)))
