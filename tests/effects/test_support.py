"""The test doubles enforce append and perform boundaries used by later owner tests."""

from types import SimpleNamespace

import pytest

from qagents.effects import _engine
from qagents.errors import EffectHeld, EntryConflict
from qagents.session import Entry

from ._samples import intent
from ._support import MemoryJournal, ScriptedEngine
from .test_testing_effects import Receipt, Request, refund


@pytest.mark.asyncio
async def test_memory_journal_reappend_and_partial_batch_are_idempotent():
    journal = MemoryJournal()
    first = Entry(kind="effect_intent", data=intent())
    second = Entry(kind="unknown_kind")
    assert await journal.append_effect([first]) == [1]
    assert await journal.append_effect([first]) == [1]
    with pytest.raises(EntryConflict):
        await journal.append_effect([first, second])
    assert len(journal.entries) == 1
    assert second.seq == 0
    journal.assert_send_ready()


@pytest.mark.asyncio
async def test_memory_journal_refuses_send_before_intent_append_returns():
    journal = MemoryJournal()

    async def before_append(entries):
        journal.assert_send_ready()

    journal.before_append = before_append
    with pytest.raises(AssertionError, match="before the intent append returned"):
        await journal.append_effect([Entry(kind="effect_intent", data=intent())])
    assert not journal.entries
    assert not journal.appending


@pytest.mark.asyncio
async def test_scripted_engine_records_current_attempt_outcome_and_restores_patch():
    journal = MemoryJournal()
    await journal.append_effect([Entry(kind="effect_intent", data=intent())])
    hold = journal.view().held("session")[0]
    original = _engine.perform
    scope = SimpleNamespace(
        key_override=None, operation_id="run:1:call", committed=[], rejected=[], held=[]
    )
    request = Request(request_id="R1", values=[])
    with ScriptedEngine(Receipt(id="one"), EffectHeld(hold)):
        assert (await _engine.perform(refund, request, scope=scope)).id == "one"
        with pytest.raises(EffectHeld):
            await _engine.perform(refund, request, scope=scope)
    assert _engine.perform is original
    assert scope.committed == [("refund", "refund:R1")]
    assert scope.held == [hold]
