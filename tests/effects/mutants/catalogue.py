"""Each mutation changes one named production boundary in one isolated process."""

from collections.abc import Callable
from dataclasses import dataclass

from .._inventory import SEAMS


@dataclass(frozen=True)
class Case:
    id: str
    seam: str
    test: str
    replacement: Callable


def constant(value):
    def replace(original, *args, **kwargs):
        return value

    return replace


async def always(*args, **kwargs):
    return True


async def nothing(*args, **kwargs):
    return None


def all_entries_out_of_scope(original, entry, finished):
    return entry.run_id not in finished


def late_window(original, op):
    return op.attempts[-1].intent.store_ts


def ignore_authorization(original, rule, scope, now):
    if rule in {"D1", "D2", "D3", "D4"}:
        return None
    return original(rule, scope, now)


def new_entry_ids(original, entries):
    from uuid import uuid4

    return [entry.model_copy(update={"id": uuid4().hex}) for entry in entries]


def count_usage_again(original, run, entries, definite=True):
    return original(run, entries, definite=True)


def drop_reasoning(original, data):
    message = original(data)
    message.parts = [part for part in message.parts if part.type != "reasoning"]
    return message


def latest_only(original, op):
    from dataclasses import replace

    return original(replace(op, attempts=op.attempts[-1:]))


def retry_rejection_without_lookup(original, effect, op):
    return False, op.generation


def newest_epoch(original, acquisitions, epoch):
    newer = [item for item in acquisitions if item[0] > epoch] if epoch is not None else []
    return max(newer, key=lambda item: item[2])[1] if newer else None


def ignore_arrival(original, now, start, effect, timing):
    return now < start + effect.dedupe.window_s


def lease_before_entry_ids(original, ids, entries, holder, owner):
    from qagents.errors import LeaseError

    if owner is not None and holder != owner:
        raise LeaseError("mutant checked the owner before the recorded entry IDs")
    return original(ids, entries, holder, owner)


async def await_unshielded(original, awaitable, *, scope):
    return await awaitable


def count_distinct(original, applications):
    from collections import Counter

    return Counter({(app.fake, app.identity): 1 for app in applications if not app.replayed})


def discard_session_claims(original, store, session_id, force=False):
    result = original(store, session_id, force)
    with store._lock, store._connect() as connection:
        connection.execute(
            "DELETE FROM claims WHERE json_extract(data, '$.session_id') = ?", (session_id,)
        )
    return result


def no_acknowledgement_durability(original, store, session_id, entries, owner=None):
    return list(range(1, len(entries) + 1))


async def _send_early(original, effect, op, frozen, scope, rule, generation):
    from qagents.effects import _canonical, _engine
    from qagents.effects._types import Attempt, AttemptDebug, EffectContext

    prepared = _engine._DISPATCH_INPUT.get()
    attempt = Attempt(
        _canonical.rebuild(effect, frozen),
        prepared.key,
        "q1_unjournaled",
        "unjournaled",
        0,
        0,
        AttemptDebug(1, generation, scope.journal.lease.epoch, 0),
    )
    await effect.send(EffectContext(scope.deps, session_id=scope.session_id), attempt)
    return await original(effect, op, frozen, scope, rule, generation)


def _timeout_as_result(original, run, tool, call, timeout, scope):
    from qagents.messages import ToolResultPart

    if scope is None:
        return original(run, tool, call, timeout, scope)
    # Remove the call-local evidence too, as a timeout handler that finalizes the
    # call would do. The later host guard still sees durable evidence.
    scope.held.clear()
    return ToolResultPart(
        tool_call_id=call.id,
        name=call.name,
        content="timed out",
        is_error=True,
        metadata={"effect_outcome_unknown": True},
    )


async def _discard_result(original, run, result, **kwargs):
    # Keep the in-memory outcome but lose the journaled result.
    run.results[result.tool_call_id] = result


def _omit_approval(original, pending):
    from qagents.permissions import Approval

    return Approval.for_call(pending, approved=False)


def _hide_cassette_miss(original, error):
    return "completed"


def _borrow_operator_lease(original, session, **kwargs):
    from contextlib import asynccontextmanager

    from qagents.session import LeasedWriter

    @asynccontextmanager
    async def borrow():
        status = await session.store.status(session.id)
        if status.lease_owner is None:
            async with original(session, **kwargs) as writer:
                yield writer
            return
        writer = LeasedWriter(
            session.store,
            session.id,
            worker="borrowed",
            ttl_s=kwargs["ttl_s"],
            purpose=kwargs["purpose"],
        )
        writer.lease.owner = status.lease_owner
        writer.lease.epoch = status.lease_epoch
        writer.lease._loop = __import__("asyncio").get_running_loop()
        writer.lease._deadline = float("inf")
        writer._opened = True
        await writer.load()
        yield writer

    return borrow()


async def _close_held_as_result(original, run, call, tool, held):
    from qagents.messages import ToolResultPart

    await run._record(
        ToolResultPart(
            tool_call_id=call.id,
            name=call.name,
            content="operation complete",
            is_error=False,
        )
    )


def _lookup_at_end(original, started, quiescent):
    return quiescent is not None


def _local_attempts_only(original, effect, op):
    # fable-3: only this record's own attempts let the lookup run, not a transferred claim's.
    return "none_declared" if not op.attempts else original(effect, op)


E = "tests/effects/test_"
M = "tests/effects/mutants/test_scenarios.py::"

# The seed descriptions and boundary names are frozen in the Effects plan.
_ROWS = [
    (M + "test_plain_absence_keeps_unknown_attempt_held", constant(True)),
    (
        E + "quiescence.py::test_all_attempts_must_be_quiescent_even_if_latest_was_not_sent",
        constant(-1),
    ),
    (E + "effects.py::test_d1_p3_p1_p2_and_p1k", constant(True)),
    (E + "engine_guards.py::test_m4_toolerror_is_unknown_and_warned_once", constant("rejected")),
    (E + "quiescence.py::test_d3_first_use_does_not_move_with_new_intent", late_window),
    (E + "quiescence.py::test_d3_concurrency_contract_or_evidence_required", constant(True)),
    (E + "engine_guards.py::test_m7_failed_or_late_renewal_never_authorizes_send", always),
    (
        "tests/test_session.py::test_owner_token_is_unique_for_each_acquisition_by_the_same_worker",
        lambda _, worker: worker,
    ),
    (
        E + "engine_dispatch.py::test_e1_intent_returns_before_send_and_commit_before_return",
        _send_early,
    ),
    (E + "fold.py::test_late_commit_after_run_finished_is_not_dropped", all_entries_out_of_scope),
    (
        E + "lifecycle.py::test_timed_out_effect_tool_cannot_close_an_unresolved_intent",
        _timeout_as_result,
    ),
    (
        E + "engine_guards.py::test_m12_second_per_call_invocation_requires_explicit_occurrence",
        constant(None),
    ),
    (
        E
        + "engine_dispatch.py::test_request_mutation_while_intent_is_writing_does_not_reach_sender",
        lambda _, effect, frozen, original: original,
    ),
    (
        E + "permission_records.py::test_changed_identity_or_signed_value_cannot_reuse_decision",
        constant(True),
    ),
    (
        M + "test_signed_approval_delivery_recovers_and_refuses_conflicting_answers",
        lambda _, run, call, existing, new: existing,
    ),
    (E + "lifecycle.py::test_hook_edit_requires_a_new_policy_decision[False]", always),
    (
        E
        + "engine_guards.py::test_m17_automatic_resend_requires_original_proof_and_store_clock_expiry",
        ignore_authorization,
    ),
    (
        M + "test_sqlite_acknowledgement_means_an_independent_reader_sees_entries",
        no_acknowledgement_durability,
    ),
    (M + "test_lost_ack_preserves_entry_identity_before_retry", new_entry_ids),
    (
        "tests/test_journal_integrity.py::test_added_usage_is_restored_only_after_a_definite_append_refusal[committed]",
        count_usage_again,
    ),
    (
        E
        + "store_state.py::test_late_effect_patch_survives_finished_run_and_provisional_state_needs_same_operation",
        constant(True),
    ),
    (M + "test_effect_calls_always_run_in_separate_groups", constant(False)),
    (E + "lifecycle.py::test_effect_tools_are_refused_outside_fenced_agent_runs", constant(None)),
    (
        "tests/test_workflow_ext.py::test_an_agent_step_that_completed_but_was_not_recorded_is_not_run_again",
        constant(None),
    ),
    (
        E + "fold.py::test_operator_outcomes_are_distinct",
        lambda orig, outcome: "D" if outcome == "not_applied" else orig(outcome),
    ),
    (M + "test_verified_decision_is_reused_after_restart", nothing),
    (M + "test_durable_tool_result_precedes_the_followup_model_request", _discard_result),
    (M + "test_replay_preserves_opaque_provider_reasoning", drop_reasoning),
    (
        E + "crash_model.py::test_turn_survives_compaction_and_crash",
        lambda _, request, turn: len([m for m in request.messages if m.role == "assistant"]),
    ),
    (
        E + "crash_model.py::test_cassette_clients_are_fresh_per_leg_and_misses_are_inconclusive",
        _hide_cassette_miss,
    ),
    (M + "test_approved_reference_exercises_an_effect", _omit_approval),
    (E + "fold.py::test_all_attempts_count_when_latest_attempt_was_not_sent", latest_only),
    (
        E + "effects.py::test_d2_checks_lookup_and_preserves_or_advances_generation",
        retry_rejection_without_lookup,
    ),
    (
        E
        + "quiescence.py::test_q2_earliest_journaled_takeover_survives_two_recoveries_and_clock_step",
        newest_epoch,
    ),
    (M + "test_sync_attempt_records_possible_overrun", constant(False)),
    (E + "quiescence.py::test_d3_first_use_does_not_move_with_new_intent", ignore_arrival),
    (E + "quiescence.py::test_d4_lookup_start_must_be_at_or_after_quiescence", _lookup_at_end),
    (
        E + "store_foundation.py::test_ids_precede_lease_and_partial_overlap_is_atomic",
        lease_before_entry_ids,
    ),
    (
        E + "engine_concurrency.py::test_cancelled_caller_still_waits_for_failure_to_be_accounted",
        await_unshielded,
    ),
    (E + "effects.py::test_d1_p3_p1_p2_and_p1k", constant(False)),
    (
        E + "settlement.py::test_settlement_requires_own_lease_and_busy_call_writes_nothing",
        _borrow_operator_lease,
    ),
    (M + "test_held_unsigned_call_is_not_asked_again", constant(True)),
    (
        E + "crash_invariants.py::test_same_key_applied_again_is_not_a_deduplicated_resend",
        count_distinct,
    ),
    (
        E + "crash_driver.py::test_swallowed_pause_remains_held_and_appears_in_harness_report",
        _close_held_as_result,
    ),
    (
        E
        + "engine_helpers.py::test_every_custom_namespace_requires_claims_even_with_session_prefix",
        constant(None),
    ),
    (
        E
        + "store_claims.py::test_committed_claim_survives_delete_and_replay_preserves_owner[sqlite]",
        discard_session_claims,
    ),
    (
        E + "store_writer.py::test_cancel_close_twice_never_releases_before_thread_send_drains",
        nothing,
    ),
    # M48: without the no-answer guard, a timeout that also derives from a declared
    # refusal counts as rejected, and D2 re-sends it under a new key (AF-1).
    (
        E
        + "engine_no_answer.py::test_no_answer_after_application_is_held_not_resent_under_a_new_key[memory-timeout_subclass]",
        constant(False),
    ),
    # M49: a claim taken over from another session as rejected skips the declared lookup,
    # and D2 applies the operation again under a new key (AF-2).
    (
        E
        + "engine_claim_transfer.py::test_applied_then_rejection_shaped_error_is_found_not_applied_again[memory-True]",
        _local_attempts_only,
    ),
]

CASES = tuple(
    Case(identifier, seam, case[0], case[1])
    for (identifier, seam, _), case in zip(SEAMS, _ROWS, strict=True)
)
