"""Explicit journal payloads shared by model and fold tests."""

from qagents.effects._canonical import digest, identity_digest, record_id


def reference(*, key="R1"):
    return {
        "namespace": "shop",
        "name": "refund",
        "key": key,
        "record_id": record_id("shop", "refund", key),
    }


def intent(*, attempt=1, generation=0, operation_id="run:1:call", key="R1", **changes):
    request = {"request_id": key, "amount": 500}
    return {
        **reference(key=key),
        "protocol": 1,
        "key_spec_digest": "key-spec",
        "version": "1",
        "operation_id": operation_id,
        "tool_call_id": "call",
        "occurrence": None,
        "attempt": attempt,
        "generation": generation,
        "idempotency_key": f"q1_{generation}",
        "operation_tag": "op1_refund",
        "digest_alg": "jcs-sha256/1",
        "request_digest": digest(request),
        "destination_digest": digest({"account": "one"}),
        "request": request,
        "destination": {"account": "one"},
        "epoch": 1,
        "store_ts": 100.0 + attempt,
        "first_use_at": 101.0,
        "not_after": 124.0,
        "may_overrun": False,
        "authorization": {"decision": "allow", "rule": "test"},
        "rule": "D1" if attempt == 1 else "D3",
        "timing": {
            "timeout_s": 20.0,
            "applies_within_s": 2.0,
            "m_s": 3.0,
            "sigma_s": 1.0,
            "window_s": 86400.0,
            "margin_s": 4320.0,
            "unique": False,
        },
        "receipt_identity": ["id"],
        **changes,
    }


def committed(*, attempt=1, intent_seq=1, key="R1", receipt=None, **changes):
    receipt = receipt if receipt is not None else {"id": "refund-1", "status": "pending"}
    return {
        **reference(key=key),
        "attempt": attempt,
        "intent_seq": intent_seq,
        "via": "direct",
        "receipt": receipt,
        "identity_digest": identity_digest(receipt, ["id"]),
        "send_ended": True,
        "ended_at": 103.0,
        **changes,
    }


def failed(*, attempt=1, intent_seq=1, key="R1", why="unknown", **changes):
    return {
        **reference(key=key),
        "attempt": attempt,
        "intent_seq": intent_seq,
        "committed": False if why in ("rejected", "not_sent") else None,
        "why": why,
        "error_kind": "timeout" if why == "unknown" else "raised",
        "send_ended": True,
        "ended_at": 103.0,
        **changes,
    }


def claim_attempt(*, session_id="a", number=1, generation=0, outcome="rejected", **changes):
    return {
        "session_id": session_id,
        "number": number,
        "generation": generation,
        "idempotency_key": f"q1_{generation}",
        "epoch": 1,
        "store_ts": 100.0 + number,
        "outcome": outcome,
        **changes,
    }


def claimed(*, generation=0, attempts=None, key="R1", previous_owner=None, **changes):
    """A claim transfer (effect_claimed) of the rejected claim written by intent()+failed()."""
    request = {"request_id": key, "amount": 500}
    return {
        **reference(key=key),
        "version": "1",
        "request_digest": digest(request),
        "destination_digest": digest({"account": "one"}),
        "request": request,
        "operation_tag": "op1_refund",
        "receipt_identity": ["id"],
        "operation_id": "run:2:call",
        "tool_call_id": "call",
        "previous_owner": previous_owner
        or {
            "session_id": "a",
            "incarnation": "inc-a",
            "record_id": record_id("shop", "refund", key),
        },
        "status": "rejected",
        "generation": generation,
        "first_use_at": 101.0,
        "attempts": attempts if attempts is not None else [claim_attempt(generation=generation)],
        "store_ts": 200.0,
        **changes,
    }
