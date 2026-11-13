"""Strict JSON boundaries and RFC 8785 binary64/string ordering examples."""

import hashlib
import json
import struct
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

from qagents.effects._canonical import LIMIT_BYTES, digest, freeze, jcs, rebuild


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 2**53, -(2**53)])
def test_refuse_numbers_that_must_be_strings(value):
    with pytest.raises(ValueError, match="send it as a string"):
        jcs(value)


@pytest.mark.parametrize("value", [(1, 2), {1: "integer key"}, b"bytes", {1, 2}, datetime.now(UTC)])
def test_refuse_non_json_values(value):
    with pytest.raises(ValueError):
        jcs(value)


@pytest.mark.parametrize("value", ["\ud800", "\udfff", {"\udfff": 1}])
def test_refuse_lone_surrogates(value):
    with pytest.raises(ValueError, match="surrogates"):
        jcs(value)


def test_safe_integer_boundary_and_negative_zero():
    assert jcs([2**53 - 1, -(2**53 - 1), -0.0]) == b"[9007199254740991,-9007199254740991,0]"


@pytest.mark.parametrize(
    ("bits", "expected"),
    [
        ("0000000000000001", "5e-324"),
        ("7fefffffffffffff", "1.7976931348623157e+308"),
        ("4340000000000000", "9007199254740992"),
        ("4430000000000000", "295147905179352830000"),
        ("44b52d02c7e14af5", "9.999999999999997e+22"),
        ("44b52d02c7e14af6", "1e+23"),
        ("44b52d02c7e14af7", "1.0000000000000001e+23"),
        ("444b1ae4d6e2ef4e", "999999999999999700000"),
        ("444b1ae4d6e2ef4f", "999999999999999900000"),
        ("444b1ae4d6e2ef50", "1e+21"),
        ("3eb0c6f7a0b5ed8c", "9.999999999999997e-7"),
        ("3eb0c6f7a0b5ed8d", "0.000001"),
        ("41b3de4355555555", "333333333.3333333"),
        ("becbf647612f3696", "-0.0000033333333333333333"),
        ("43143ff3c1cb0959", "1424953923781206.2"),
    ],
)
def test_rfc8785_number_vectors(bits, expected):
    # Appendix B: https://www.rfc-editor.org/rfc/rfc8785.html#appendix-B
    value = struct.unpack(">d", bytes.fromhex(bits))[0]
    assert jcs(value).decode() == expected


def test_utf16_key_order_and_escaping_preserve_string_data():
    data = {"\ufb33": 1, "\U0001f600": 2, "\u20ac": 3, "\r": 4, "1": 5, "\u0080": 6}
    assert list(json.loads(jcs(data))) == ["\r", "1", "\u0080", "\u20ac", "\U0001f600", "\ufb33"]
    assert jcs('\b\t\n\f\r\x0f"\\/') == b'"\\b\\t\\n\\f\\r\\u000f\\"\\\\/"'
    assert jcs("é") != jcs("e\u0301")
    assert jcs({"z": [{"b": 1, "a": 2}], "a": False}) == b'{"a":false,"z":[{"a":2,"b":1}]}'


def test_shared_containers_allowed_but_cycles_refused():
    item = [1]
    assert jcs([item, item]) == b"[[1],[1]]"
    item.append(item)
    with pytest.raises(ValueError, match="Cyclic"):
        jcs(item)


class Request(BaseModel):
    text: str
    values: list[int] = []


def test_freeze_rebuild_is_independent_and_digest_is_stable():
    effect = SimpleNamespace(name="store", request_type=Request)
    request = Request(text="hello", values=[1])
    frozen = freeze(effect, request)
    request.values.append(2)
    rebuilt = rebuild(effect, frozen)
    assert rebuilt == Request(text="hello", values=[1])
    rebuilt.values.append(3)
    assert frozen.json["values"] == [1]
    assert frozen.digest == hashlib.sha256(jcs(frozen.json)).hexdigest()
    assert digest({"b": 1, "a": 2}) == digest({"a": 2, "b": 1})


def test_freeze_size_limit_includes_exact_utf8_json_bytes():
    effect = SimpleNamespace(name="store", request_type=Request)
    overhead = len(jcs({"text": "", "values": []}))
    assert freeze(effect, Request(text="a" * (LIMIT_BYTES - overhead))).size == LIMIT_BYTES
    with pytest.raises(ValueError, match="16384-byte"):
        freeze(effect, Request(text="a" * (LIMIT_BYTES - overhead + 1)))
    with pytest.raises(ValueError, match="16384-byte"):
        freeze(effect, Request(text="é" * (LIMIT_BYTES // 2)))


def test_freeze_validates_type_and_bypassed_model_construction():
    effect = SimpleNamespace(name="store", request_type=Request)

    class Other(BaseModel):
        pass

    with pytest.raises(TypeError, match="requires Request"):
        freeze(effect, Other())
    with pytest.raises(ValidationError):
        freeze(effect, Request.model_construct(text="x", values=["not-an-integer"]))
