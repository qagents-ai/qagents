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


def _raised_error_only(original, error):
    # AF-1 review: only the raised error's own class counts, not its cause or context.
    return iter((error,))


def _cancellation_only(original, effect, timed_out, error):
    # AF-1 review: the old rule, where only a CancelledError after the deadline is the timeout.
    import asyncio

    return timed_out and isinstance(error, asyncio.CancelledError)


def _first_set_only(original):
    # AF-1 review: the set before it held responses cut short and client-library types.
    return original()[:5]


def _without_httpx2(original):
    # AF-1 recheck: the set without httpx2 (anthropic>=2.8, mcp>=2.3) and httpcore.
    dropped = ("httpx2", "httpcore", "httpcore2")
    return tuple(kind for kind in original() if kind.__module__.split(".")[0] not in dropped)


def _lease_kept_only(original, journal, effect):
    # AF-2 review: the claim moved whenever the lease was not lost, without the self-fence.
    return not journal.lease.lost()


def _rejected_reads_no_claim(original, op, claim, prepared, session_id):
    # AF-2 review: a rejected record never read the claim again, so a moved claim locked it out.
    return "own" if op.state == "R" else original(op, claim, prepared, session_id)


def _unsent_claim_held_elsewhere(original, op, claim, prepared, session_id):
    # AF-2 recheck: a rejected record held a claim the other owner never sent, so after a
    # taker's D2 ended not_sent the owner's retry was held CLAIMED_ELSEWHERE for good.
    kind = original(op, claim, prepared, session_id)
    return "held_elsewhere" if kind == "not_sent" and op.state == "R" else kind


def _every_replayed_rejection_changes_key(original, effect, op):
    # AF-2 recheck: under replays_rejections D2 always took generation + 1, also when no
    # attempt at the current generation had been sent.
    from qagents.effects import Dedupe

    changes = isinstance(effect.dedupe, Dedupe) and effect.dedupe.replays_rejections
    return effect.lookup_fn is not None, op.generation + int(changes)


def _folded_state(original, permissions, session_id, op, answer):
    # AF-3 review: the operator path checked a resend against the folded state, so a forged
    # resend that reopened a closed record let a real one through.
    return op.state


def _pending_resend_only(original, permissions, session_id, op, answer):
    # AF-3 recheck: only the pending resend was set aside when it did not verify, so an
    # earlier forged resend still made a closed record held for a real one.
    from dataclasses import replace

    from qagents.effects import _engine, _fold

    if answer.outcome != "resend" or _engine._resend_verified(permissions, session_id, op):
        return op.state
    others = [item for item in op.settlements if item is not op.pending_resend]
    return _fold._status_of(replace(op, settlements=others, pending_resend=None))


def _count_from_zero(original, runtime, handle):
    # AF-4 review: the caller's count was compared with 0, not with its count when the
    # preparation began, so an older cancel request turned a lost lease into CancelledError.
    handle._preparing_mark = 0
    return original(runtime, handle)


def _only_a_cancel_is_the_leases(original, owner, mark):
    # AF-4 recheck: the cancel was taken back, but only a CancelledError became
    # session_busy; an error that authorize_settlement raised for the lease's cancel
    # reached the caller as that error.
    import asyncio
    import sys

    took_back = original(owner, mark)
    return took_back and isinstance(sys.exc_info()[1], asyncio.CancelledError)


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
    # M50: an operator resend is trusted without its MAC, so a forged effect_settled entry
    # written straight to the store authorizes a D5 send (AF-3).
    (
        E + "engine_settlement_mac.py::test_forged_resend_is_held_and_never_sent[memory-no_mac]",
        constant(True),
    ),
    # M51: the build check misses an effects tool that asks for approval, so nothing warns
    # that, without approval_secret, its effects hold APPROVAL_UNVERIFIABLE (AF-5).
    (
        E
        + "unsigned_approvals.py::test_build_warns_once_for_an_approval_gated_effects_tool_without_a_secret[requires_approval]",
        constant(False),
    ),
    # M52: without the chain walk, a client's refusal type raised from an httpx timeout
    # counts as rejected, and D2 re-sends it under a new key (AF-1 review).
    (
        E
        + "engine_no_answer.py::test_no_answer_after_application_is_held_not_resent_under_a_new_key[memory-wrapped_from]",
        _raised_error_only,
    ),
    # M53: a declared refusal type that an async send raises after the engine cancelled it
    # at its deadline counts as rejected, and D2 re-sends it under a new key (AF-1 review).
    (
        E
        + "engine_no_answer.py::test_refusal_raised_after_the_engine_cancelled_the_send_is_its_timeout[memory-after]",
        _cancellation_only,
    ),
    # M54: the no-answer set without the standard library's responses cut short and the
    # client libraries' transport types, so an IncompleteReadError counts as rejected.
    (
        E
        + "engine_guards.py::test_a_response_cut_short_or_a_client_library_disconnect_is_unknown[IncompleteReadError]",
        _first_set_only,
    ),
    # M55: a worker whose lease cannot cover a send still takes a rejected claim over, and
    # the owner's own retry then fails with ClaimConflict (AF-2 review).
    (
        E
        + "engine_claim_transfer.py::test_a_takeover_whose_self_fence_fails_journals_nothing_and_keeps_the_claim[memory]",
        _lease_kept_only,
    ),
    # M56: a rejected record never reads the claim again, so after another session took
    # the claim over and died before sending, the owner's retry fails with ClaimConflict.
    (
        E
        + "engine_claim_transfer.py::test_the_owner_takes_its_claim_back_from_a_taker_that_sent_nothing[memory-False]",
        _rejected_reads_no_claim,
    ),
    # M57: D2 sends after a lookup that said NotYetVisible or Unknown, so an application
    # answered rejection-shaped is applied again under a new key (AF-2 review).
    (
        E
        + "engine_claim_transfer.py::test_an_inconclusive_lookup_in_another_session_holds_instead_of_sending[memory-not_yet_visible-True]",
        constant(False),
    ),
    # M58: the operator path accepts a resend on a record that only an unverified resend
    # reopened, so D5 re-sends a not_applied record (AF-3 review).
    (
        E
        + "engine_settlement_mac.py::test_forged_resend_cannot_reopen_a_record_for_an_operator_resend[memory-not_applied-1_forged]",
        _folded_state,
    ),
    # M59: an unverifiable decision names the hold on every held record, also where no row
    # would re-send, so without approval_secret an opaque effect hides RESPONSE_LOST (AF-5).
    (
        E
        + "unsigned_approvals.py::test_an_effect_no_row_would_resend_holds_for_its_own_reason_without_a_secret[opaque]",
        constant(True),
    ),
    # M60: the warning names an effects tool whose effects no row ever re-sends, and tells
    # the developer that approval_secret would let them be re-sent (AF-5 review).
    (
        E
        + "unsigned_approvals.py::test_an_approval_gated_effects_tool_with_no_automatic_resend_does_not_warn[opaque]",
        constant(True),
    ),
    # M61: the build check ignores can_use_tool, so an effects tool that the callback sends
    # to "ask" holds APPROVAL_UNVERIFIABLE with no warning (AF-5 review).
    (
        E
        + "unsigned_approvals.py::test_build_warns_once_when_can_use_tool_may_ask_for_an_effects_tool[no_rule]",
        constant(False),
    ),
    # M62: a caller that caught an older cancel without uncancel() gets the runtime's own
    # CancelledError, not LeaseError, when its lease is lost while the run is prepared (AF-4).
    (
        "tests/test_runtime.py::test_an_older_cancel_request_on_the_caller_does_not_hide_a_lost_lease[start]",
        _count_from_zero,
    ),
    # M63: the operator lease never takes its own cancel back and turns any CancelledError
    # after a lost lease into session_busy, so the caller is left cancelling (AF-4 review).
    (
        E
        + "settlement.py::test_a_lease_lost_while_the_operator_waits_is_session_busy_and_takes_its_cancel_back[memory-settle-False]",
        constant(True),
    ),
    # M64: the no-answer set without httpx2 and httpcore, so a client's refusal type raised
    # from httpx2's error for a server that closed without answering counts as rejected,
    # and D2 re-sends it under a new key (AF-1 recheck).
    (
        E
        + "engine_no_answer.py::test_no_answer_after_application_is_held_not_resent_under_a_new_key[memory-httpx2_wrapped_from]",
        _without_httpx2,
    ),
    # M65: a rejected record holds a claim whose last owner never sent its D2, so the
    # owner's retry holds CLAIMED_ELSEWHERE on every call, also after a delete (AF-2 recheck).
    (
        E
        + "engine_claim_transfer.py::test_the_owner_takes_back_a_claim_whose_taker_never_sent_its_d2[memory-True-lookup-kept]",
        _unsent_claim_held_elsewhere,
    ),
    # M66: D2 after taking back a never-sent claim skips the key the other owner chose and
    # never sent, so the sessions retrying one business key use two keys (AF-2 recheck).
    (
        E
        + "engine_claim_transfer.py::test_the_owner_takes_back_a_claim_whose_taker_never_sent_its_d2[memory-True-no_lookup-kept]",
        _every_replayed_rejection_changes_key,
    ),
    # M67: the operator path sets aside only the pending resend when it does not verify, so
    # an earlier forged resend still makes a not_applied record held, and D5 re-sends it
    # under a genuine resend (AF-3 recheck).
    (
        E
        + "engine_settlement_mac.py::test_forged_resend_cannot_reopen_a_record_for_an_operator_resend[memory-not_applied-2_forged]",
        _pending_resend_only,
    ),
    # M68: an error that authorize_settlement raises for the lost lease's cancel reaches
    # the caller as that error, not as session_busy (AF-4 recheck).
    (
        E
        + "settlement.py::test_an_error_the_authorization_raises_for_the_lease_cancel_is_session_busy[memory-False]",
        _only_a_cancel_is_the_leases,
    ),
]

CASES = tuple(
    Case(identifier, seam, case[0], case[1])
    for (identifier, seam, _), case in zip(SEAMS, _ROWS, strict=True)
)
