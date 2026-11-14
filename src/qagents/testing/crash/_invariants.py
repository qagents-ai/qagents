"""Independent checks over durable journals and observations of the destination."""

from __future__ import annotations

import hashlib
import inspect
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from qagents.effects._canonical import jcs
from qagents.session import Entry, derive_state

if TYPE_CHECKING:
    from ._world import Application, World

Status = Literal["safe", "violated", "inconclusive", "not_applicable"]


@dataclass(frozen=True)
class Finding:
    invariant: str
    code: str
    message: str
    status: Literal["violated", "inconclusive"] = "violated"
    application_index: int | None = None
    entry_id: str | None = None


@dataclass(frozen=True)
class Check:
    invariant: str
    status: Status
    findings: tuple[Finding, ...] = ()
    independent: bool = True
    notes: tuple[str, ...] = ()


def _check(invariant: str, findings: Sequence[Finding], **kwargs: Any) -> Check:
    status: Status = (
        "violated"
        if any(f.status == "violated" for f in findings)
        else "inconclusive"
        if findings
        else "safe"
    )
    return Check(invariant, status, tuple(findings), **kwargs)


@dataclass(frozen=True)
class JournalSnapshot:
    """A read of the durable medium captured at an observed external boundary.

    JSON strings keep the evidence immutable even if a caller later edits an
    Entry. Memory-store copies are explicitly not independent disk reads.
    """

    session_id: str
    entries_json: tuple[str, ...]
    entry_ids: tuple[str, ...]
    independent_read: bool


@dataclass(frozen=True)
class ApplicationEvidence:
    """The journal at issue time, retained unchanged if application happens later."""

    application_index: int
    operation_key: str | None
    record_id: str | None
    snapshot: JournalSnapshot
    issued_store_time: float | None = None


@dataclass(frozen=True)
class ModelEvidence:
    """The durable prefix immediately before a model consultation advances."""

    leg: int
    consultation: int
    fingerprint: str
    snapshot: JournalSnapshot


def _count_identities(applications: Iterable[Application]) -> Counter:
    """M43: count applications, including repeats under a different generation/key."""
    return Counter((app.fake, app.identity) for app in applications if not app.replayed)


def _read(snapshot: JournalSnapshot) -> list[Entry]:
    entries = [Entry.model_validate_json(value) for value in snapshot.entries_json]
    if tuple(entry.id for entry in entries) != snapshot.entry_ids:
        raise ValueError("snapshot ids do not match its serialized entries")
    return entries


def check_applications(
    applications: Sequence[Application],
    reference: Sequence[Application],
    evidence: Sequence[ApplicationEvidence],
) -> tuple[Check, ...]:
    """I1a-c and I11 application checks over observed applications and issue-time prefixes.

    Idempotency keys and business operation keys are deliberately distinct. A new
    idempotency generation does not create a new business operation or erase a repeat.
    This oracle never reads EffectView or accepts the engine's own status as proof.
    """
    findings: dict[str, list[Finding]] = {key: [] for key in ("I1a", "I1b", "I1c", "I11")}
    seen_keys: Counter = Counter()
    operation_keys: dict[tuple, set[tuple[str, str]]] = defaultdict(set)
    by_index: dict[int, list[ApplicationEvidence]] = defaultdict(list)
    for item in evidence:
        by_index[item.application_index].append(item)
    independent = all(item.snapshot.independent_read for item in evidence)
    for index, app in enumerate(applications):
        if app.replayed:
            continue
        if not app.key_passed or not app.idempotency_key:
            findings["I1a"].append(
                Finding(
                    "I1a",
                    "key_not_passed",
                    f"{app.fake}: key not passed",
                    application_index=index,
                )
            )
        else:
            key = (app.fake, app.idempotency_key)
            seen_keys[key] += 1
            if seen_keys[key] > 1:
                findings["I1a"].append(
                    Finding(
                        "I1a",
                        "key_applied_twice",
                        f"{app.fake}: idempotency key applied twice",
                        application_index=index,
                    )
                )
        captured = by_index.get(index, ())
        if len(captured) != 1:
            for invariant in ("I1c", "I11"):
                findings[invariant].append(
                    Finding(
                        invariant,
                        "application_evidence_missing",
                        f"{app.fake}: expected one immutable issue-time snapshot, "
                        f"got {len(captured)}",
                        "inconclusive",
                        index,
                    )
                )
            continue
        proof = captured[0]
        try:
            entries = _read(proof.snapshot)
        except (ValueError, TypeError) as error:
            for invariant in ("I1c", "I11"):
                findings[invariant].append(
                    Finding(
                        invariant,
                        "application_evidence_invalid",
                        str(error),
                        "inconclusive",
                        index,
                    )
                )
            continue
        matches = [
            entry
            for entry in entries
            if entry.kind == "effect_intent"
            and entry.data.get("name") == app.fake
            and entry.data.get("idempotency_key") == app.idempotency_key
            and type(app.attempt) is int
            and type(app.epoch) is int
            and type(entry.data.get("attempt")) is int
            and type(entry.data.get("epoch")) is int
            and entry.data.get("attempt") == app.attempt
            and entry.data.get("epoch") == app.epoch
        ]
        if len(matches) != 1:
            findings["I11"].append(
                Finding(
                    "I11",
                    "intent_not_before_application",
                    f"{app.fake}: no unique matching intent in the issue-time durable prefix",
                    application_index=index,
                )
            )
            findings["I1c"].append(
                Finding(
                    "I1c",
                    "operation_binding_missing",
                    f"{app.fake}: application has no unique durable operation binding",
                    "inconclusive",
                    index,
                )
            )
            continue
        entry = matches[0]
        intent = entry.data
        binding = (
            isinstance(intent.get("namespace"), str)
            and isinstance(intent.get("key"), str)
            and isinstance(intent.get("record_id"), str)
            and intent["key"] == proof.operation_key
            and intent["record_id"] == proof.record_id
        )
        if not binding:
            findings["I11"].append(
                Finding(
                    "I11",
                    "application_binding_mismatch",
                    f"{app.fake}: captured record/operation does not match its intent",
                    application_index=index,
                    entry_id=entry.id,
                )
            )
            findings["I1c"].append(
                Finding(
                    "I1c",
                    "operation_binding_missing",
                    f"{app.fake}: application operation key cannot be established",
                    "inconclusive",
                    index,
                    entry.id,
                )
            )
        else:
            operation_keys[(app.fake, app.identity)].add((intent["namespace"], intent["key"]))
        try:
            actual_digest = hashlib.sha256(jcs(intent["request"])).hexdigest()
        except (ValueError, TypeError, KeyError):
            actual_digest = None
        if actual_digest is None or not (
            actual_digest == intent.get("request_digest") == app.payload_digest
        ):
            findings["I11"].append(
                Finding(
                    "I11",
                    "payload_digest_mismatch",
                    f"{app.fake}: applied payload does not match the journaled frozen request",
                    application_index=index,
                    entry_id=entry.id,
                )
            )
    actual = _count_identities(applications)
    expected = _count_identities(reference)
    for (fake, _identity), count in actual.items():
        allowed = expected[(fake, _identity)]
        if count > allowed:
            findings["I1b"].append(
                Finding(
                    "I1b",
                    "identity_applied_too_often",
                    f"{fake}: business identity applied {count} times; reference permits {allowed}",
                )
            )
    for (fake, _identity), keys in operation_keys.items():
        if len(keys) > 1:
            findings["I1c"].append(
                Finding(
                    "I1c",
                    "identity_has_multiple_keys",
                    f"{fake}: the key does not capture the identity; "
                    "multiple operation keys applied",
                )
            )
    return tuple(
        _check(key, values, independent=independent if key in ("I1c", "I11") else True)
        for key, values in findings.items()
    )


def check_journal(final: JournalSnapshot | None, crashes: Sequence[JournalSnapshot] = ()) -> Check:
    """I6: independently read durable prefixes survive, with valid every-prefix derivation."""
    if final is None:
        return _check(
            "I6",
            [
                Finding(
                    "I6",
                    "final_snapshot_missing",
                    "No final durable journal read was captured.",
                    "inconclusive",
                )
            ],
            independent=False,
        )
    findings = []
    independent = final.independent_read and all(item.independent_read for item in crashes)
    notes = () if independent else ("InMemoryStore evidence is a copy, not an independent read.",)
    try:
        entries = _read(final)
    except (ValueError, TypeError) as error:
        return _check(
            "I6",
            [
                Finding(
                    "I6",
                    "snapshot_invalid",
                    str(error),
                    "inconclusive",
                )
            ],
            independent=independent,
            notes=notes,
        )
    if len(set(final.entry_ids)) != len(final.entry_ids):
        findings.append(Finding("I6", "duplicate_entry_id", "Journal entry ids are not unique."))
    if [entry.seq for entry in entries] != list(range(1, len(entries) + 1)):
        findings.append(
            Finding("I6", "noncontiguous_sequence", "Journal sequence is not contiguous.")
        )
    try:
        final_data = [jcs(entry.model_dump(mode="json")) for entry in entries]
    except (ValueError, TypeError) as error:
        return _check(
            "I6",
            [
                Finding(
                    "I6",
                    "snapshot_not_canonical",
                    str(error),
                    "inconclusive",
                )
            ],
            independent=independent,
            notes=notes,
        )
    for crash in crashes:
        try:
            prefix = [jcs(entry.model_dump(mode="json")) for entry in _read(crash)]
        except (ValueError, TypeError) as error:
            findings.append(Finding("I6", "crash_snapshot_invalid", str(error), "inconclusive"))
            continue
        if crash.session_id != final.session_id or final_data[: len(prefix)] != prefix:
            findings.append(
                Finding(
                    "I6",
                    "durable_prefix_changed",
                    "The durable journal at a crash is not a prefix of the final journal.",
                )
            )
    for length in range(len(entries) + 1):
        try:
            derive_state(entries[:length])
        except Exception as error:
            findings.append(
                Finding(
                    "I6",
                    "prefix_not_derivable",
                    f"Journal prefix {length} cannot be derived: {type(error).__name__}: {error}",
                    entry_id=entries[length - 1].id if length else None,
                )
            )
            break
    return _check("I6", findings, independent=independent, notes=notes)


OUTCOMES = frozenset(
    {
        "completed_as_reference",
        "completed_after_model_told",
        "completed_alternative",
        "paused_awaiting_operator",
        "paused_awaiting_approval",
        "failed_retry_later",
        "failed_exhausted",
        "failed_unexpected",
        "no_terminal_state",
    }
)


def check_progress(outcome: str, *, legs: int, max_legs: int) -> Check:
    """I8: a known, allowed outcome is reached within the declared execution bound."""
    if type(legs) is not int or type(max_legs) is not int or max_legs < 1 or legs < 1:
        raise ValueError("legs and max_legs must be positive integers")
    findings = []
    if outcome not in OUTCOMES or outcome in {"failed_unexpected", "no_terminal_state"}:
        findings.append(Finding("I8", "no_valid_outcome", f"Point ended as {outcome!r}."))
    if legs > max_legs:
        findings.append(Finding("I8", "max_legs_exceeded", f"Used {legs} legs; limit {max_legs}."))
    return _check("I8", findings)


def check_user_invariant(
    world: World, invariant: Callable[[World], bool | None], *, name: str
) -> Check:
    """An unjudged or broken user predicate never silently becomes a pass."""
    key = f"user:{name}"
    try:
        result = invariant(world)
        if result is True:
            return Check(key, "safe", independent=False)
        if result is False:
            return _check(key, [Finding(key, "user_invariant_failed", name)], independent=False)
        if result is None:
            return Check(key, "not_applicable", independent=False)
        if inspect.iscoroutine(result):
            result.close()
        detail = f"{name} returned {type(result).__name__}; expected True, False or None."
    except Exception as error:
        detail = f"{name} raised {type(error).__name__}: {error}"
    return _check(
        key,
        [
            Finding(
                key,
                "user_invariant_unjudged",
                detail,
                "inconclusive",
            )
        ],
        independent=False,
    )
