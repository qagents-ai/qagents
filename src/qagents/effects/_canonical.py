"""Strict canonical JSON and frozen request values for effect identities."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from pydantic import JsonValue as JSONValue

if TYPE_CHECKING:
    from ._declare import Effect

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


def check_model_numbers(value: Any) -> None:
    """Reject numeric loss before Pydantic's JSON conversion can replace it with null.

    Other Python values, such as dates and UUIDs, retain their normal model JSON
    conversion. This check complements strict JCS validation after serialization.
    """
    ancestors: set[int] = set()

    def check(item: Any) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("A non-finite number is not strict JSON; send it as a string.")
        if isinstance(item, int) and not isinstance(item, bool) and abs(item) > 2**53 - 1:
            raise ValueError("Integer outside the safe JSON range; send it as a string.")
        if not isinstance(item, Mapping | list | tuple | set | frozenset):
            return
        identity = id(item)
        if identity in ancestors:
            raise ValueError("Cyclic objects are not JSON values.")
        ancestors.add(identity)
        try:
            if isinstance(item, Mapping):
                for key, child in item.items():
                    check(key)
                    check(child)
            else:
                for child in item:
                    check(child)
        finally:
            ancestors.remove(identity)

    try:
        check(value)
    except RecursionError as error:
        raise ValueError("JSON nesting exceeds the interpreter's safe recursion depth.") from error


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
    check_model_numbers(validated.model_dump(mode="python", warnings="error"))
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


@dataclass(frozen=True)
class Descriptor:
    """The first intent's effect, version, destination, and request identity."""

    effect: str
    version: str
    destination_digest: str
    request_digest: str

    def same_without_version(self, other: Descriptor) -> bool:
        """Compare meaning without code version; the recovery table owns this check."""
        return (self.effect, self.destination_digest, self.request_digest) == (
            other.effect,
            other.destination_digest,
            other.request_digest,
        )

    def digest(self) -> str:
        """The claim descriptor digest, deliberately excluding the code version."""
        return digest([self.effect, self.destination_digest, self.request_digest])


def idempotency_key(
    salt: bytes, namespace: str, effect: str, key: str, generation: int, max_len: int
) -> str:
    """Salted q1_ key for one generation; RFC 4648 base32, unpadded, at least 24 characters."""
    if type(max_len) is not int or max_len < 24:
        raise ValueError("idempotency_key_max_len must be an integer >= 24.")
    if type(generation) is not int or generation < 0:
        raise ValueError("generation must be a nonnegative integer.")
    hashed = hashlib.sha256(salt + jcs([namespace, effect, key, generation])).digest()
    return ("q1_" + base64.b32encode(hashed).decode().rstrip("="))[:max_len]


def operation_tag(namespace: str, effect: str, key: str, request_digest: str) -> str:
    """An op1_ binding tag that stays stable across every attempt and generation."""
    hashed = hashlib.sha256(jcs([namespace, effect, key, request_digest])).digest()
    return "op1_" + base64.b32encode(hashed).decode()[:26]


def record_id(namespace: str, effect: str, key: str) -> str:
    """An ef_ display identity; ambiguous shortened IDs must be refused by callers."""
    hashed = hashlib.sha256(jcs([namespace, effect, key])).digest()
    return "ef_" + base64.b32encode(hashed).decode().lower()[:8]


def identity_digest(receipt: dict[str, JSONValue], receipt_identity: Sequence[str] | None) -> str:
    """Hash declared identity fields, or the whole receipt when no fields were declared."""
    return digest(
        receipt
        if receipt_identity is None
        else {field: receipt[field] for field in receipt_identity}
    )


def resolve_namespace(effect: Effect, deps: Any, session_id: str) -> str:
    """Resolve session identity or call a trusted namespace function once per perform."""
    namespace = effect.namespace
    if namespace == "session":
        return "session:" + session_id.replace("%", "%25").replace(":", "%3A")
    resolved = namespace(deps) if callable(namespace) else namespace
    if not isinstance(resolved, str) or not resolved:
        raise ValueError("An effect namespace must be a nonempty string.")
    jcs(resolved)  # Refuse invalid Unicode before hashing or persisting the namespace.
    return resolved


def descriptor(effect: Effect, destination_json: JSONValue, request_digest: str) -> Descriptor:
    """Bind the effect's declared version, trusted destination, and frozen request."""
    return Descriptor(effect.name, effect.version, digest(destination_json), request_digest)
