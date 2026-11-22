"""Commit/CI evidence survives reopen and answers indexed calendar questions."""

import json
import os
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import UTC, datetime
from threading import Event

import pytest

from examples import repository_activity_index as activity
from examples.repository_activity import (
    ActivityDeps,
    build_agent,
    factual_answer,
    offline_model,
    run_agent,
)
from examples.repository_activity_index import ActivityIndex, calendar_window
from qagents import Session
from qagents.errors import OutputValidationError
from qagents.models.testing import ScriptedModel, call


def git(repo, *args, date=None):
    env = dict(os.environ)
    if date:
        env.update(GIT_AUTHOR_DATE="2020-01-01T12:00:00+00:00", GIT_COMMITTER_DATE=date)
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], env=env, text=True, encoding="utf-8"
    ).strip()


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "repository"
    repo.mkdir()
    git(repo, "init", "--quiet")
    git(repo, "config", "user.name", "Activity fixture")
    git(repo, "config", "user.email", "activity@example.invalid")
    git(repo, "config", "commit.gpgsign", "false")
    git(repo, "remote", "add", "origin", "https://github.com/example/activity.git")
    return repo


def commit(repo, subject, date):
    git(repo, "commit", "--allow-empty", "--quiet", "-m", subject, date=date)
    return git(repo, "rev-parse", "HEAD")


NOW = datetime(2026, 11, 22, 12, tzinfo=UTC)


@pytest.mark.parametrize(
    "question", ["What changed yesterday?", "Τι έγινε χθες;", "τι έγινε χτες;"]
)
def test_local_yesterday_uses_committer_dates_and_excludes_next_midnight(
    repository, tmp_path, question
):
    first = commit(repository, "Fix the calendar", "2026-11-20T22:00:00+00:00")
    second = commit(repository, "Review voice results", "2026-11-21T21:59:59+00:00")
    commit(repository, "Next local day", "2026-11-21T22:00:00+00:00")
    path = tmp_path / "index.db"
    ActivityIndex(path).refresh(repository)
    report = ActivityIndex(path).query(question, now=NOW)
    assert report["commit_count"] == 2
    assert {row["sha"] for row in report["commits"]} == {first, second}
    assert report["start"] == "2026-11-20T22:00:00+00:00"
    assert report["end_exclusive"] == "2026-11-21T22:00:00+00:00"


def test_previous_week_is_a_calendar_week_and_dst_day_can_have_25_hours():
    english = calendar_window("What happened last week?", NOW, "Europe/Athens")
    greek = calendar_window("Τι έγινε την προηγούμενη εβδομάδα;", NOW, "Europe/Athens")
    assert english == greek
    assert datetime.fromtimestamp(english[1], UTC).isoformat() == "2026-11-08T22:00:00+00:00"
    assert datetime.fromtimestamp(english[2], UTC).isoformat() == "2026-11-15T22:00:00+00:00"
    _, start, end = calendar_window(
        "yesterday", datetime(2026, 10, 26, 12, tzinfo=UTC), "Europe/Athens"
    )
    assert end - start == 25 * 3600
    with pytest.raises(ValueError, match="timezone"):
        calendar_window("yesterday", datetime(2026, 11, 22), "Europe/Athens")
    with pytest.raises(ValueError, match="Specify"):
        calendar_window("what happened", NOW, "Europe/Athens")


def test_incremental_refresh_reopen_and_queries_do_not_run_git(repository, tmp_path, monkeypatch):
    first = commit(repository, "Publish documentation", "2026-11-21T12:00:00+00:00")
    path = tmp_path / "index.db"
    assert ActivityIndex(path).refresh(repository)["new_commits"] == 1
    assert ActivityIndex(path).refresh(repository)["new_commits"] == 0
    second = commit(repository, "Improve documentation search", "2026-11-21T14:00:00+00:00")
    update = ActivityIndex(path).refresh(repository)
    assert update["incremental"] is True and update["new_commits"] == 1
    monkeypatch.setattr(
        activity, "_command", lambda *a, **k: pytest.fail("query invoked a process")
    )
    report = ActivityIndex(path).query("yesterday", now=NOW, search="documentation", limit=1)
    assert report["commit_count"] == 2 and report["commits_truncated"] is True
    assert report["commits"][0]["sha"] == second
    with closing(sqlite3.connect(path)) as db:
        assert db.execute("SELECT COUNT(*) FROM commit_search").fetchone()[0] == 2
        plan = db.execute(
            "EXPLAIN QUERY PLAN SELECT sha FROM commits WHERE committed_at >= ? AND committed_at < ?",
            (0, 2000000000),
        ).fetchall()
        assert "commits_by_time" in str(plan)
    assert first != second


def test_rewritten_ref_discards_removed_commits_and_stale_ci(repository, tmp_path):
    first = commit(repository, "Original", "2026-11-21T12:00:00+00:00")
    second = commit(repository, "Removed history", "2026-11-21T13:00:00+00:00")
    index = ActivityIndex(tmp_path / "index.db")
    index.refresh(repository)
    index.record_ci("example/activity", [ci_run(second)], complete=True)
    git(repository, "reset", "--hard", first)
    update = index.refresh(repository)
    assert update["incremental"] is False and update["commits"] == 1
    report = index.query("yesterday", now=NOW)
    assert [row["sha"] for row in report["commits"]] == [first]
    assert report["ci_runs"] == [] and report["ci_index"] is None


def ci_run(sha, **changes):
    return {
        "id": 42,
        "head_sha": sha,
        "name": "Tests",
        "status": "completed",
        "conclusion": "failure",
        "updated_at": "2026-11-21T15:00:00Z",
        "html_url": "https://github.com/example/activity/actions/runs/42",
        **changes,
    }


def test_ci_has_evidence_unknowns_and_separate_update_timestamps(repository, tmp_path):
    sha = commit(repository, "Change from last week", "2026-11-12T12:00:00+00:00")
    index = ActivityIndex(tmp_path / "index.db")
    index.refresh(repository)
    assert index.record_ci("example/activity", [ci_run(sha)], complete=False) == 1
    report = index.query("yesterday", now=NOW)
    assert report["commit_count"] == 0 and report["ci_update_count"] == 1
    assert report["ci_runs"][0]["conclusion"] == "failure"
    assert report["ci_runs"][0]["url"].endswith("/actions/runs/42")
    assert report["ci_index"]["complete"] is False
    index.record_ci(
        "example/activity", [ci_run(sha, status="in_progress", conclusion=None)], complete=True
    )
    assert index.query("yesterday", now=NOW)["ci_runs"][0]["conclusion"] is None
    assert index.record_ci("example/activity", [ci_run("a" * 40)], complete=True) == 0


def test_ci_rejects_forged_urls_and_wrong_repository_without_replacing_evidence(
    repository, tmp_path
):
    sha = commit(repository, "Evidence", "2026-11-21T12:00:00+00:00")
    index = ActivityIndex(tmp_path / "index.db")
    index.refresh(repository)
    index.record_ci("example/activity", [ci_run(sha)], complete=True)
    with pytest.raises(ValueError, match="URL"):
        index.record_ci(
            "example/activity", [ci_run(sha, html_url="https://example.invalid")], complete=True
        )
    with pytest.raises(ValueError, match="origin"):
        index.record_ci("another/repository", [], complete=True)
    assert index.query("yesterday", now=NOW)["ci_update_count"] == 1


def test_missing_index_and_limits_and_hostile_search_are_explicit(repository, tmp_path):
    missing = tmp_path / "missing.db"
    with pytest.raises(FileNotFoundError, match="index command"):
        ActivityIndex(missing).query("yesterday", now=NOW)
    assert not missing.exists()
    commit(repository, "A real commit", "2026-11-21T12:00:00+00:00")
    index = ActivityIndex(tmp_path / "index.db")
    index.refresh(repository)
    assert (
        index.query("yesterday", now=NOW, search='" OR 1=1; DROP TABLE commits --')["commit_count"]
        == 0
    )
    assert index.query("yesterday", now=NOW)["commit_count"] == 1
    for limit in (0, 21):
        with pytest.raises(ValueError, match="limit"):
            index.query("yesterday", now=NOW, limit=limit)
    with pytest.raises(ValueError, match="1 to 8"):
        index.query("yesterday", now=NOW, search="one two three four five six seven eight nine")


async def test_real_agent_loop_reads_indexed_evidence_in_greek(repository, tmp_path):
    commit(repository, "Ship voice fixes", "2026-11-21T12:00:00+00:00")
    index = ActivityIndex(tmp_path / "index.db")
    index.refresh(repository)
    report = index.query("Τι έγινε χθες;", now=NOW)
    model = offline_model("Τι έγινε χθες;")
    result = await run_agent(model, "Τι έγινε χθες;", report)
    assert result["model_requests"] == 2 and result["answer"]["commit_count"] == 1
    assert "Δεν έχουν εισαχθεί" in result["answer"]["summary"]
    assert result["answer"]["commit_shas"] == [report["commits"][0]["sha"]]
    assert model.requests[1].messages[-1].tool_results[0].name == "read_activity"


@pytest.mark.parametrize(
    "change",
    [{"commit_count": 99}, {"commit_shas": ["f" * 40]}, {"ci_update_count": 7}, {"run_ids": [123]}],
)
async def test_agent_rejects_invented_facts_and_citations(repository, tmp_path, change):
    commit(repository, "Evidence", "2026-11-21T12:00:00+00:00")
    index = ActivityIndex(tmp_path / "index.db")
    index.refresh(repository)
    report = index.query("yesterday", now=NOW)
    bad = factual_answer(report, "yesterday").model_dump()
    bad.update(change)
    agent = build_agent(ScriptedModel([call("read_activity"), json.dumps(bad), json.dumps(bad)]))
    with pytest.raises(OutputValidationError):
        await agent.run("yesterday", deps=ActivityDeps(report))


async def test_even_empty_answers_require_retrieval(repository, tmp_path):
    commit(repository, "Old evidence", "2026-11-01T12:00:00+00:00")
    index = ActivityIndex(tmp_path / "index.db")
    index.refresh(repository)
    report = index.query("yesterday", now=NOW)
    answer = factual_answer(report, "yesterday").model_dump_json()
    with pytest.raises(OutputValidationError):
        await build_agent(ScriptedModel([answer, answer])).run(
            "yesterday", deps=ActivityDeps(report)
        )


def test_query_in_a_new_process_needs_no_git_or_network(repository, tmp_path):

    commit(repository, "Persistent evidence", "2026-11-21T12:00:00+00:00")
    path = tmp_path / "index.db"
    ActivityIndex(path).refresh(repository)
    env = {**os.environ, "PATH": "", "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.repository_activity",
            "query",
            "--database",
            str(path),
            "--question",
            "Τι έγινε χθες;",
            "--as-of",
            NOW.isoformat(),
        ],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=30,
    )
    output = json.loads(result.stdout)
    assert output["answer"]["commit_count"] == 1
    assert output["evidence"]["ci_index"] is None


def test_failed_refresh_preserves_the_previous_snapshot(repository, tmp_path, monkeypatch):
    sha = commit(repository, "Preserved", "2026-11-21T12:00:00+00:00")
    index = ActivityIndex(tmp_path / "index.db")
    index.refresh(repository)
    original = activity._command

    def malformed(args, **options):
        if "log" in args:
            return subprocess.CompletedProcess(args, 0, "malformed\0record\0", "")
        return original(args, **options)

    monkeypatch.setattr(activity, "_command", malformed)
    with pytest.raises(ValueError, match="decode"):
        index.refresh(repository)
    report = index.query("yesterday", now=NOW)
    assert report["snapshot"]["head"] == sha and report["commit_count"] == 1


def test_a_concurrent_refresh_cannot_mix_old_metadata_and_new_rows(
    repository, tmp_path, monkeypatch
):
    commit(repository, "First", "2026-11-21T12:00:00+00:00")
    index = ActivityIndex(tmp_path / "index.db")
    index.refresh(repository)
    paused, release = Event(), Event()
    original = activity._metadata

    def pause_reader(db):
        result = original(db)
        if db.row_factory is sqlite3.Row and not paused.is_set():
            paused.set()
            assert release.wait(10)
        return result

    monkeypatch.setattr(activity, "_metadata", pause_reader)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(index.query, "yesterday", now=NOW)
        try:
            assert paused.wait(10)
            commit(repository, "Second", "2026-11-21T13:00:00+00:00")
            index.refresh(repository)
        finally:
            release.set()
        report = pending.result(timeout=10)
    assert report["commit_count"] == report["snapshot"]["commits"] == 1
    assert index.query("yesterday", now=NOW)["commit_count"] == 2


async def test_agent_cannot_turn_a_failed_ci_result_into_success(repository, tmp_path):
    sha = commit(repository, "A failing change", "2026-11-21T12:00:00+00:00")
    index = ActivityIndex(tmp_path / "index.db")
    index.refresh(repository)
    index.record_ci("example/activity", [ci_run(sha)], complete=True)
    report = index.query("yesterday", now=NOW)
    bad = factual_answer(report, "yesterday").model_dump()
    bad["ci_outcomes"][0]["conclusion"] = "success"
    model = ScriptedModel([call("read_activity"), json.dumps(bad), json.dumps(bad)])
    with pytest.raises(OutputValidationError):
        await build_agent(model).run("yesterday", deps=ActivityDeps(report))


async def test_session_evidence_from_an_old_snapshot_does_not_authorize_a_new_answer(
    repository, tmp_path
):
    commit(repository, "First snapshot", "2026-11-21T12:00:00+00:00")
    index = ActivityIndex(tmp_path / "index.db")
    index.refresh(repository)
    old_report = index.query("yesterday", now=NOW)
    session = Session()
    await build_agent(offline_model("yesterday")).run(
        "yesterday", deps=ActivityDeps(old_report), session=session
    )
    commit(repository, "Second snapshot", "2026-11-21T13:00:00+00:00")
    index.refresh(repository)
    new_report = index.query("yesterday", now=NOW)
    unsupported = factual_answer(new_report, "yesterday").model_dump_json()
    with pytest.raises(OutputValidationError):
        await build_agent(ScriptedModel([unsupported, unsupported])).run(
            "yesterday", deps=ActivityDeps(new_report), session=session
        )
