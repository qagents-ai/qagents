"""Stable operation identities, injective key rendering, and namespace boundaries."""

import base64
import hashlib
from types import SimpleNamespace

import pytest

from qagents.effects._canonical import (
    Descriptor,
    descriptor,
    digest,
    idempotency_key,
    identity_digest,
    jcs,
    operation_tag,
    record_id,
    resolve_namespace,
)


def test_idempotency_key_binds_store_namespace_effect_key_and_generation():
    args = (b"store-A", "shop", "refund", "r:1", 0, 64)
    expected = "q1_" + base64.b32encode(
        hashlib.sha256(b"store-A" + b'["shop","refund","r:1",0]').digest()
    ).decode().rstrip("=")
    assert idempotency_key(*args) == expected
    assert idempotency_key(*args[:-1], 24) == expected[:24]
    for index, changed in enumerate((b"store-B", "other", "charge", "r:2", 1)):
        variant = list(args)
        variant[index] = changed
        assert idempotency_key(*variant) != expected
    with pytest.raises(ValueError, match=">= 24"):
        idempotency_key(*args[:-1], 23)
    with pytest.raises(ValueError, match="nonnegative"):
        idempotency_key(b"s", "n", "e", "k", -1, 64)


def test_operation_tag_and_record_id_use_structured_identity():
    identity = ["shop", "refund", "r:1"]
    hashed = hashlib.sha256(jcs(identity)).digest()
    assert record_id(*identity) == "ef_" + base64.b32encode(hashed).decode().lower()[:8]
    expected = (
        "op1_" + base64.b32encode(hashlib.sha256(jcs([*identity, "abc"])).digest()).decode()[:26]
    )
    assert operation_tag(*identity, "abc") == expected
    assert operation_tag(*identity, "changed") != expected
    assert record_id("a:b", "c", "d") != record_id("a", "b:c", "d")


def test_descriptor_ignores_only_version():
    original = Descriptor("refund", "1", "account-A", "request-A")
    next_version = Descriptor("refund", "2", "account-A", "request-A")
    assert original.same_without_version(next_version)
    assert original.digest() == next_version.digest()
    for changed in (
        Descriptor("charge", "1", "account-A", "request-A"),
        Descriptor("refund", "1", "account-B", "request-A"),
        Descriptor("refund", "1", "account-A", "request-B"),
    ):
        assert not original.same_without_version(changed)
        assert original.digest() != changed.digest()
    effect = SimpleNamespace(name="refund", version="1")
    assert descriptor(effect, {"account": "A"}, "request-A").destination_digest == digest(
        {"account": "A"}
    )


def test_receipt_identity_allows_status_updates_but_not_another_application():
    pending = {"id": "refund-1", "status": "pending"}
    done = {"status": "succeeded", "id": "refund-1"}
    assert identity_digest(pending, ("id",)) == identity_digest(done, ("id",))
    assert identity_digest(pending, None) != identity_digest(done, None)
    assert identity_digest(pending, ("id",)) != identity_digest({"id": "refund-2"}, ("id",))
    with pytest.raises(KeyError):
        identity_digest(pending, ("missing",))


def test_namespace_resolution_and_percent_encoding():
    session = SimpleNamespace(namespace="session")
    assert resolve_namespace(session, None, "a:b%3Ac") == "session:a%3Ab%253Ac"
    assert resolve_namespace(session, None, "a:b") != resolve_namespace(session, None, "a%3Ab")
    assert resolve_namespace(SimpleNamespace(namespace="account:123"), None, "s") == "account:123"
    calls = []

    def scoped(deps):
        calls.append(deps)
        return "tenant:" + deps

    assert resolve_namespace(SimpleNamespace(namespace=scoped), "trusted", "s") == "tenant:trusted"
    assert calls == ["trusted"]
    with pytest.raises(ValueError, match="nonempty"):
        resolve_namespace(SimpleNamespace(namespace=lambda deps: 42), None, "s")
