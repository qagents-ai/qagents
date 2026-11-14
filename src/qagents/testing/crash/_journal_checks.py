"""Check terminal evidence without accepting the engine's folded status as proof."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from math import isclose

from qagents.effects._canonical import identity_digest, jcs
from qagents.effects._kinds import STATUS_CHANGING_KINDS
from qagents.effects._types import HeldEffect
from qagents.messages import Message, Usage
from qagents.session import Entry, derive_state

from . import _invariants
from ._invariants import Check, Finding, _check, _read


def check_starts(entries: Sequence[Entry], replayable: set[str]) -> Check:
    """I3: completed calls stay completed and unsafe calls cannot start again."""
    finished, started = set(), set()
    turn = defaultdict(int)
    findings = []
    for entry in entries:
        data = entry.data
        if entry.kind == "message":
            turn[entry.run_id] += 1
        elif entry.kind == "tool_result":
            part = data.get("result", {})
            finished.add(
                data.get("operation_id")
                or (
                    entry.run_id,
                    turn[entry.run_id],
                    part.get("tool_call_id"),
                )
            )
        elif entry.kind == "tool_started":
            operation = data.get("operation_id") or (
                entry.run_id,
                turn[entry.run_id],
                data.get("tool_call_id"),
            )
            if operation in finished or (
                operation in started and data.get("name") not in replayable
            ):
                findings.append(
                    Finding(
                        "I3",
                        "tool_started_again",
                        f"Tool {data.get('name')!r} started again.",
                        entry_id=entry.id,
                    )
                )
            started.add(operation)
    return _check("I3", findings)


def check_decisions(entries: Sequence[Entry], application_evidence, applications, macs=()) -> Check:
    """I5: verify journal attribution, digest binding and expiry at the destination."""
    decisions = {}
    latest = {}
    mac_status = dict(macs)
    spent_calls = set()
    findings = []
    for entry in entries:
        data = entry.data
        operation = data.get("operation_id")
        if entry.kind == "approval_decided" and "approval_id" in data:
            decisions[data["approval_id"]] = (entry.id, data)
            latest[operation] = data
        elif entry.kind == "tool_started" and operation is not None:
            decision = latest.get(operation)
            if decision is not None and (
                not decision.get("approved")
                or decision.get("arguments_digest") != data.get("arguments_digest")
            ):
                findings.append(
                    Finding(
                        "I5",
                        "start_decision_mismatch",
                        "Tool start does not match its approval.",
                        entry_id=entry.id,
                    )
                )
        elif entry.kind == "effect_intent":
            auth = data.get("authorization", {})
            if auth.get("decision") == "ask":
                entry_id, decision = decisions.get(auth.get("approval_id"), (None, None))
                if entry_id is not None and mac_status.get(entry_id) is False:
                    findings.append(
                        Finding(
                            "I5",
                            "invalid_approval_mac",
                            "Intent relied on an invalid approval MAC.",
                            entry_id=entry.id,
                        )
                    )
                if (
                    decision is None
                    or not decision.get("approved")
                    or (
                        decision.get("operation_id") != operation
                        or decision.get("tool_call_id") != data.get("tool_call_id")
                        or decision.get("arguments_digest") != auth.get("arguments_digest")
                    )
                ):
                    findings.append(
                        Finding(
                            "I5",
                            "intent_decision_mismatch",
                            "Intent lacks the approval for its call.",
                            entry_id=entry.id,
                        )
                    )
            spent_calls.add((entry.run_id, data.get("tool_call_id")))
        elif entry.kind == "run_paused":
            for pending in data.get("pending", ()):
                if (entry.run_id, pending.get("tool_call_id")) in spent_calls:
                    findings.append(
                        Finding(
                            "I5",
                            "intent_asked_again",
                            "A call with a durable intent was asked again.",
                            entry_id=entry.id,
                        )
                    )
    for captured in application_evidence:
        if not 0 <= captured.application_index < len(applications):
            continue
        application = applications[captured.application_index]
        candidates = [
            entry
            for entry in entries
            if entry.kind == "effect_intent"
            and entry.data.get("record_id") == captured.record_id
            and entry.data.get("attempt") == application.attempt
            and entry.data.get("epoch") == application.epoch
        ]
        for candidate in candidates:
            auth = candidate.data.get("authorization", {})
            expiry = auth.get("expires_at")
            if expiry is None:
                continue
            issued = captured.issued_store_time
            if not isinstance(expiry, int | float) or issued is None:
                findings.append(
                    Finding(
                        "I5",
                        "send_time_missing",
                        "Approval expiry cannot be checked at issue time.",
                        "inconclusive",
                        entry_id=candidate.id,
                    )
                )
            elif max(candidate.data.get("store_ts", 0), issued) >= expiry:
                findings.append(
                    Finding(
                        "I5",
                        "expired_intent",
                        "The sending attempt used an expired approval.",
                        entry_id=candidate.id,
                    )
                )
    return _check("I5", findings, notes=("Checks recorded attribution and send-time expiry.",))


def check_receipts(entries: Sequence[Entry]) -> Check:
    """I11: receipts bind to the intent and lookup evidence they cite."""
    intents = {}
    first_by_record = {}
    findings = []
    for entry in entries:
        data = entry.data
        record = data.get("record_id")
        if entry.kind == "effect_intent":
            intents[entry.seq] = entry
            first_by_record.setdefault(record, entry)
            continue
        if entry.kind != "effect_committed":
            continue
        via = data.get("via")
        if via == "claim":
            owner = data.get("claim_owner")
            if (
                not isinstance(owner, dict)
                or not owner.get("session_id")
                or not owner.get("record_id")
            ):
                findings.append(
                    Finding(
                        "I11.receipt",
                        "claim_owner_missing",
                        "Claim replay lacks a durable owner.",
                        entry_id=entry.id,
                    )
                )
            continue
        target = (
            first_by_record.get(record)
            if via == "operator"
            else intents.get(data.get("intent_seq"))
        )
        if (
            target is None
            or any(
                target.data.get(key) != data.get(key)
                for key in ("record_id", "namespace", "name", "key")
            )
            or (via != "operator" and target.data.get("attempt") != data.get("attempt"))
        ):
            findings.append(
                Finding(
                    "I11.receipt",
                    "commit_without_intent",
                    "Receipt cites no matching earlier intent.",
                    entry_id=entry.id,
                )
            )
            continue
        receipt = data.get("receipt")
        if receipt is None:
            valid_receipt = bool(data.get("receipt_error")) and data.get("identity_digest") is None
        else:
            try:
                valid_receipt = isinstance(receipt, dict) and identity_digest(
                    receipt,
                    target.data.get("receipt_identity"),
                ) == data.get("identity_digest")
            except (KeyError, TypeError, ValueError):
                valid_receipt = False
        if not valid_receipt:
            findings.append(
                Finding(
                    "I11.receipt",
                    "receipt_identity_mismatch",
                    "Receipt and its declared identity differ.",
                    entry_id=entry.id,
                )
            )
        if via == "lookup":
            seen, tag = data.get("found_seen"), data.get("found_tag")
            request = target.data.get("request", {})
            try:
                bound = bool(seen or tag) and (
                    tag is None or tag == target.data.get("operation_tag")
                )
                bound = bound and (
                    seen is None
                    or isinstance(seen, dict)
                    and all(
                        field in request and jcs(value) == jcs(request[field])
                        for field, value in seen.items()
                    )
                )
            except (TypeError, ValueError):
                bound = False
            if not bound:
                findings.append(
                    Finding(
                        "I11.receipt",
                        "lookup_binding_mismatch",
                        "Recovered receipt does not match its durable request.",
                        entry_id=entry.id,
                    )
                )
    return _check("I11.receipt", findings)


def check_patches(entries: Sequence[Entry], final_state: Mapping | None) -> Check:
    """I10: each durable patch is applied once, in order, and visible in final state."""
    patch_ids, prior_commits, closed = set(), set(), set()
    revisions = {}
    findings = []
    state = {}
    results = {
        entry.data["operation_id"]: entry.seq
        for entry in entries
        if entry.kind == "tool_result"
        and "operation_id" in entry.data
        and entry.data.get("commit_provisional", True)
    }
    for entry in entries:
        data = entry.data
        record = data.get("record_id")
        previous_revision = revisions.get(record)
        if entry.kind in STATUS_CHANGING_KINDS:
            revisions[record] = entry.seq
        if entry.kind == "run_finished" and entry.run_id is not None:
            closed.add(entry.run_id)
        if (
            entry.kind == "state_updated"
            and entry.run_id not in closed
            and (
                not data.get("provisional") or results.get(data.get("operation_id"), -1) > entry.seq
            )
        ):
            state = deepcopy(data.get("values", {}))
        if entry.kind != "effect_committed":
            continue
        patch = data.get("state_patch")
        if patch is not None:
            authorized_resolution = (
                data.get("via") == "operator"
                and data.get("expected_revision") == previous_revision
                and bool(data.get("by"))
                and bool(data.get("evidence"))
            )
            if (
                entry.id in patch_ids
                or data.get("via") == "claim"
                or (record in prior_commits and not authorized_resolution)
            ):
                findings.append(
                    Finding(
                        "I10",
                        "patch_repeated",
                        "A repeated receipt wrote another state patch.",
                        entry_id=entry.id,
                    )
                )
            else:
                patch_ids.add(entry.id)
                state.update(deepcopy(patch.get("set", {})))
                for key in patch.get("unset", ()):
                    state.pop(key, None)
        prior_commits.add(record)
    if patch_ids and final_state is None:
        findings.append(
            Finding(
                "I10",
                "state_missing",
                "No final state was observed after a durable patch.",
                "inconclusive",
            )
        )
    elif final_state is not None and jcs(dict(final_state)) != jcs(state):
        findings.append(
            Finding(
                "I10",
                "final_state_differs",
                "Final state differs from the ordered durable patches.",
            )
        )
    return _check("I10", findings)


def check_terminal(entries: Sequence[Entry], held: Sequence[HeldEffect]) -> Check:
    """I12: each outstanding operation has a terminal entry or a surfaced hold."""
    intents = defaultdict(list)
    terminal = {}
    resolved = {}
    conflicts = {}
    surfaced = {item.id for item in held}
    findings = []
    for entry in entries:
        data = entry.data
        record = data.get("record_id")
        if entry.kind == "effect_intent":
            intents[record].append(entry)
        elif entry.kind == "effect_failed" and data.get("committed") is False:
            terminal[data.get("intent_seq")] = entry.seq
        elif entry.kind == "effect_committed":
            resolved[record] = entry.seq
        elif entry.kind == "effect_settled" and data.get("outcome") in (
            "not_applied",
            "close_unknown",
        ):
            resolved[record] = entry.seq
        elif entry.kind == "effect_conflict":
            conflicts[record] = entry.seq
    for record, attempts in intents.items():
        if record in surfaced:
            continue
        closed_at = resolved.get(record, -1)
        if conflicts.get(record, -1) > closed_at:
            latest = attempts[-1]
        else:
            missing = [
                entry for entry in attempts if entry.seq > closed_at and entry.seq not in terminal
            ]
            if not missing:
                continue
            latest = missing[-1]
        findings.append(
            Finding(
                "I12",
                "intent_not_surfaced",
                "An unresolved intent has no terminal evidence or surfaced hold.",
                entry_id=latest.id,
            )
        )
    return _check("I12", findings)


def check_business_outcomes(raw, reference, entries: Sequence[Entry]) -> Check:
    """I2: work from the reference either applied or has a matching surfaced hold."""
    missing = _invariants._count_identities(reference.applications) - _invariants._count_identities(
        raw.applications
    )
    if not missing:
        return _check("I2", [])
    surfaced = {(item.namespace, item.effect, item.key) for item in raw.held}
    # Use the reference issue snapshot to bind external business identities to
    # operation keys. A hold from unrelated work cannot explain a missing outcome.
    expected = defaultdict(Counter)
    seen = set()
    for proof in reference.application_evidence:
        index = proof.application_index
        if index in seen or not 0 <= index < len(reference.applications):
            continue
        seen.add(index)
        app = reference.applications[index]
        identity = (app.fake, app.identity)
        if app.replayed or identity not in missing:
            continue
        try:
            prefix = _read(proof.snapshot)
        except (ValueError, TypeError):
            continue
        matches = [
            entry.data
            for entry in prefix
            if entry.kind == "effect_intent"
            and entry.data.get("name") == app.fake
            and entry.data.get("record_id") == proof.record_id
            and entry.data.get("key") == proof.operation_key
            and entry.data.get("attempt") == app.attempt
            and entry.data.get("epoch") == app.epoch
        ]
        if len(matches) == 1:
            intent = matches[0]
            expected[identity][(intent["namespace"], intent["name"], intent["key"])] += 1

    persisted = {
        (entry.data.get("namespace"), entry.data.get("name"), entry.data.get("key"))
        for entry in entries
        if entry.kind in {"effect_intent", "effect_observed"}
    }
    findings = []
    for identity, count in missing.items():
        fake, _ = identity
        if sum(expected[identity].values()) < count:
            findings.append(
                Finding(
                    "I2",
                    "reference_binding_missing",
                    f"{fake}: missing reference identity evidence.",
                    "inconclusive",
                )
            )
        elif sum(key in surfaced and key in persisted for key in expected[identity]) < count:
            findings.append(
                Finding(
                    "I2",
                    "missing_application_or_hold",
                    f"{fake}: {count} missing applications lack matching surfaced holds.",
                )
            )
    return _check("I2", findings)


def check_usage(entries: Sequence[Entry], model_calls) -> Check:
    """I7a: reconcile persisted assistant usage against independent consultations."""
    remaining = list(model_calls)
    findings = []
    seen_turns = set()
    for entry in entries:
        data = entry.data
        if (
            entry.kind != "message"
            or data.get("message", {}).get("role") != "assistant"
            or "usage" not in data
        ):
            continue
        tag = (entry.run_id, data.get("turn"))
        if tag[1] is None:
            continue
        if tag in seen_turns:
            findings.append(
                Finding(
                    "I7a",
                    "recorded_turn_twice",
                    "Model usage was recorded twice for one turn.",
                    entry_id=entry.id,
                )
            )
            continue
        seen_turns.add(tag)
        usage = Usage.model_validate(data["usage"])
        parts = Message.model_validate(data["message"]).parts
        matched = next(
            (
                index
                for index, call in enumerate(remaining)
                if (call.run_id, call.turn + 1 if call.turn is not None else None) == tag
                and call.response is not None
                and call.response.message.parts == parts
                and call.usage == usage
            ),
            None,
        )
        if matched is None:
            findings.append(
                Finding(
                    "I7a",
                    "model_usage_does_not_match",
                    "Journaled model usage cannot be attributed to a matching response.",
                    entry_id=entry.id,
                )
            )
        else:
            remaining.pop(matched)
    return _check("I7a", findings)


def check_usage_totals(entries: Sequence[Entry]) -> Check:
    """I7b: sum reported usage independently and compare it with the fold."""
    latest = next((item.run_id for item in reversed(entries) if item.kind == "run_started"), None)
    if latest is None:
        return _check("I7b", [])
    subtotal = Usage()
    countable = {
        "message",
        "tool_result",
        "tool_checkpoint",
        "run_paused",
        "run_failed",
        "child_usage",
    }
    for entry in entries:
        if entry.run_id != latest:
            continue
        if entry.kind == "run_finished":
            break
        if entry.kind in countable:
            for key in ("usage", "added_usage"):
                if data := entry.data.get(key):
                    subtotal = subtotal.add(Usage.model_validate(data))
    reported = derive_state(list(entries)).run_usage
    subtotal_cost, reported_cost = subtotal.cost_usd, reported.cost_usd
    subtotal.cost_usd = reported.cost_usd
    cost_matches = subtotal_cost == reported_cost or (
        subtotal_cost is not None
        and reported_cost is not None
        and isclose(subtotal_cost, reported_cost)
    )
    if subtotal != reported or not cost_matches:
        return _check(
            "I7b",
            [
                Finding(
                    "I7b",
                    "usage_total_mismatch",
                    "Replay usage differs from the journal's entries.",
                )
            ],
        )
    return _check("I7b", [])


def check_spend(raw) -> Check:
    """I7c: reconcile charged work and allow only bounded crash and final-call overshoot."""
    limits = {leg: (tokens, cost) for leg, tokens, cost in raw.usage_limits}
    if not any(tokens is not None or cost is not None for tokens, cost in limits.values()):
        return _check("I7c", [], notes=("No token or cost limit configured.",))
    calls = [
        call for call in raw.model_calls if call.turn is not None and call.response is not None
    ]
    findings = []
    if any(call.leg not in limits for call in calls):
        return _check(
            "I7c",
            [
                Finding(
                    "I7c",
                    "missing_limits",
                    "One or more model consultations have no observed limits.",
                    "inconclusive",
                )
            ],
        )
    groups = defaultdict(list)
    for call in calls:
        groups[(call.session_id, call.run_id)].append(call)
    allowance = len(raw.crash_snapshots) + 1  # one request can cross the limit on a healthy leg
    for consultations in groups.values():
        observed = {limits[call.leg] for call in consultations}
        if len(observed) != 1:
            findings.append(
                Finding(
                    "I7c",
                    "changed_limits",
                    "The run's budget differed across recovery legs.",
                    "inconclusive",
                )
            )
            continue
        token_limit, cost_limit = next(iter(observed))
        usages = [call.usage for call in consultations]
        if cost_limit is not None and any(not usage.cost_complete for usage in usages):
            findings.append(
                Finding(
                    "I7c",
                    "unpriced_spend",
                    "An unpriced model response prevents measuring spend.",
                    "inconclusive",
                )
            )
        fields = (
            ("tokens", token_limit, [usage.total_tokens for usage in usages]),
            ("cost", cost_limit, [usage.cost_usd or 0.0 for usage in usages]),
        )
        for field, limit, values in fields:
            if limit is None:
                continue
            bound = max(0, limit) + allowance * max(values, default=0)
            total = sum(values)
            if (limit <= 0 and values) or (total > bound and not isclose(total, bound)):
                findings.append(
                    Finding(
                        "I7c",
                        "spend_above_bound",
                        f"Observed {field} {total} exceeds the budget and recovery allowance.",
                    )
                )
    return _check("I7c", findings)
