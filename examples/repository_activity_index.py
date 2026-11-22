"""Local commit and CI evidence for the repository-activity agent.

Indexing runs Git separately from questions. Questions reopen SQLite read-only;
they never walk Git history, contact GitHub or create a missing database.
"""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import unicodedata
from contextlib import closing
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

SHA = re.compile(r"[a-f0-9]{40}(?:[a-f0-9]{24})?\Z")
GITHUB_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS commits (
    sha TEXT PRIMARY KEY, committed_at INTEGER NOT NULL, subject TEXT NOT NULL, body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS commits_by_time ON commits(committed_at DESC, sha);
CREATE VIRTUAL TABLE IF NOT EXISTS commit_search USING fts5(sha UNINDEXED, subject, body);
CREATE TABLE IF NOT EXISTS ci_runs (
    id INTEGER PRIMARY KEY, head_sha TEXT NOT NULL, name TEXT NOT NULL,
    status TEXT NOT NULL, conclusion TEXT, updated_at INTEGER NOT NULL, url TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ci_by_time ON ci_runs(updated_at DESC, id);
"""


def _command(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=check,
    )


def _git(repo: Path, *args: str) -> str:
    return _command(["git", "-C", str(repo), *args]).stdout.strip()


def _metadata(db: sqlite3.Connection) -> dict:
    return {key: json.loads(value) for key, value in db.execute("SELECT key, value FROM metadata")}


def _save(db: sqlite3.Connection, key: str, value) -> None:
    db.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)", (key, json.dumps(value)))


def _stamp(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("CI timestamps must include a timezone")
    return int(parsed.timestamp())


def _normalized(value: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFD", value.casefold())
        if not unicodedata.combining(char)
    )


def calendar_window(question: str, now: datetime, timezone: str) -> tuple[str, int, int]:
    """Resolve local calendar days/weeks, with an exclusive end and UTC SQL bounds."""
    if not 1 <= len(question) <= 4000:
        raise ValueError("Use a question of 1 to 4000 characters")
    if now.tzinfo is None:
        raise ValueError("now must include a timezone")
    words = _normalized(question)
    matches = [
        period
        for period, terms in (
            ("yesterday", ("yesterday", "χθες", "χτες", "xthes", "xtes")),
            (
                "last_week",
                ("last week", "previous week", "προηγούμενη εβδομάδα", "proigoumeni evdomada"),
            ),
            ("today", ("today", "σήμερα")),
        )
        if any(_normalized(term) in words for term in terms)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Specify exactly one period: yesterday / χθες, "
            "last week / προηγούμενη εβδομάδα, or today / σήμερα"
        )
    period = matches[0]
    today = now.astimezone(ZoneInfo(timezone)).date()
    if period == "yesterday":
        start, end = today - timedelta(days=1), today
    elif period == "last_week":
        end = today - timedelta(days=today.weekday())
        start = end - timedelta(days=7)
    else:
        start, end = today, today + timedelta(days=1)
    zone = ZoneInfo(timezone)
    return (
        period,
        int(datetime.combine(start, time.min, zone).timestamp()),
        int(datetime.combine(end, time.min, zone).timestamp()),
    )


class ActivityIndex:
    """One host-selected Git repository/ref per database, with explicit refreshes."""

    def __init__(self, path: Path):
        self.path = Path(path).resolve()

    def refresh(self, repo: Path, ref: str = "HEAD") -> dict:
        repo = Path(_git(Path(repo).resolve(), "rev-parse", "--show-toplevel"))
        head = _git(repo, "rev-parse", "--verify", "--end-of-options", ref + "^{commit}")
        if not SHA.fullmatch(head):
            raise ValueError("Git did not return a commit SHA")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=10)) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)
            db.execute("BEGIN IMMEDIATE")
            with db:
                previous = _metadata(db).get("snapshot", {})
                same = previous.get("repository") == str(repo) and previous.get("ref") == ref
                old = previous.get("head", "")
                incremental = (
                    same
                    and SHA.fullmatch(old)
                    and _command(
                        ["git", "-C", str(repo), "merge-base", "--is-ancestor", old, head],
                        check=False,
                    ).returncode
                    == 0
                )
                revision = f"{old}..{head}" if incremental else head
                # NUL separates fields and records, preserving multiline commit bodies.
                raw = _command(
                    [
                        "git",
                        "-C",
                        str(repo),
                        "log",
                        "-z",
                        "--format=%H%x00%ct%x00%s%x00%b",
                        revision,
                        "--",
                    ]
                ).stdout
                fields = raw.split("\0")
                if fields[-1] == "":
                    fields.pop()
                if len(fields) % 4:
                    raise ValueError("Could not decode Git's commit records")
                rows = []
                for offset in range(0, len(fields), 4):
                    sha, stamp, subject, body = fields[offset : offset + 4]
                    if not SHA.fullmatch(sha):
                        raise ValueError("Invalid commit identity")
                    rows.append((sha, int(stamp), subject[:2000], body[:16000]))
                if not incremental:
                    db.execute("DELETE FROM commits")
                    db.execute("DELETE FROM commit_search")
                    db.execute("DELETE FROM ci_runs")
                    db.execute("DELETE FROM metadata")
                db.executemany("INSERT INTO commits VALUES (?, ?, ?, ?)", rows)
                db.executemany(
                    "INSERT INTO commit_search VALUES (?, ?, ?)",
                    [(sha, subject, body) for sha, _, subject, body in rows],
                )
                snapshot = {
                    "repository": str(repo),
                    "ref": ref,
                    "head": head,
                    "indexed_at": datetime.now(UTC).isoformat(),
                    "commits": db.execute("SELECT COUNT(*) FROM commits").fetchone()[0],
                }
                _save(db, "snapshot", snapshot)
        return {**snapshot, "new_commits": len(rows), "incremental": bool(incremental)}

    def record_ci(self, repository: str, runs: list[dict], *, complete: bool) -> int:
        """Store validated Actions metadata, never logs, tokens or environment values."""
        if not GITHUB_REPOSITORY.fullmatch(repository):
            raise ValueError("Use a GitHub owner/repository name")
        rows = []
        for run in runs:
            run_id = run["id"]
            sha = run["head_sha"]
            if type(run_id) is not int or run_id <= 0 or not SHA.fullmatch(sha):
                raise ValueError("Invalid CI identity")
            url = f"https://github.com/{repository}/actions/runs/{run_id}"
            if run["html_url"] != url:
                raise ValueError("CI URL does not belong to the selected repository/run")
            status, conclusion = run["status"], run.get("conclusion")
            if status not in {
                "completed",
                "queued",
                "in_progress",
                "requested",
                "waiting",
                "pending",
            }:
                raise ValueError("Unknown CI status")
            if conclusion is not None and conclusion not in {
                "success",
                "failure",
                "neutral",
                "cancelled",
                "skipped",
                "timed_out",
                "action_required",
                "stale",
                "startup_failure",
            }:
                raise ValueError("Unknown CI conclusion")
            rows.append(
                (
                    run_id,
                    sha,
                    str(run["name"])[:300],
                    status,
                    conclusion,
                    _stamp(run["updated_at"]),
                    url,
                )
            )
        with closing(self._read()) as db:
            snapshot = _metadata(db).get("snapshot")
        if snapshot is None:
            raise ValueError("The index has no completed refresh; run the index command first")
        origin = _git(Path(snapshot["repository"]), "remote", "get-url", "origin")
        allowed_origins = {
            f"https://github.com/{repository}",
            f"https://github.com/{repository}.git",
            f"git@github.com:{repository}.git",
            f"ssh://git@github.com/{repository}.git",
        }
        if origin not in allowed_origins:
            raise ValueError("GitHub repository does not match the indexed Git origin")
        with closing(sqlite3.connect(self.path, timeout=10)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            if _metadata(db).get("snapshot") != snapshot:
                raise ValueError("The index changed while CI metadata was being recorded; retry")
            if complete:
                db.execute("DELETE FROM ci_runs")
            for row in rows:
                db.execute(
                    "INSERT OR REPLACE INTO ci_runs SELECT ?, ?, ?, ?, ?, ?, ? "
                    "WHERE EXISTS (SELECT 1 FROM commits WHERE sha = ?)",
                    (*row, row[1]),
                )
            _save(
                db,
                "ci",
                {
                    "repository": repository,
                    "complete": complete,
                    "fetched_at": datetime.now(UTC).isoformat(),
                    "fetched_runs": len(rows),
                },
            )
            return db.execute("SELECT COUNT(*) FROM ci_runs").fetchone()[0]

    def _read(self) -> sqlite3.Connection:
        if not self.path.is_file():
            raise FileNotFoundError("Create the index with the index command first")
        db = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=2)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        return db

    def query(
        self,
        question: str,
        *,
        now: datetime,
        timezone: str = "Europe/Athens",
        search: str = "",
        limit: int = 8,
    ) -> dict:
        if not 1 <= limit <= 20 or len(search) > 500:
            raise ValueError("Use a result limit of 1 to 20 and search text up to 500 characters")
        period, start, end = calendar_window(question, now, timezone)
        clause = "committed_at >= ? AND committed_at < ?"
        parameters: list = [start, end]
        if search.strip():
            tokens = re.findall(r"\w+", search, flags=re.UNICODE)
            if not 1 <= len(tokens) <= 8:
                raise ValueError("Search must contain 1 to 8 words")
            expression = " AND ".join('"' + token + '"' for token in tokens)
            clause += " AND sha IN (SELECT sha FROM commit_search WHERE commit_search MATCH ?)"
            parameters.append(expression)
        with closing(self._read()) as db:
            metadata = _metadata(db)
            if "snapshot" not in metadata:
                raise ValueError("The index has no completed refresh; run the index command first")
            commits = [
                dict(row)
                for row in db.execute(
                    "SELECT sha, committed_at, subject FROM commits WHERE "
                    + clause
                    + " ORDER BY committed_at DESC, sha LIMIT ?",
                    (*parameters, limit),
                )
            ]
            total = db.execute(
                "SELECT COUNT(*) FROM commits WHERE " + clause, parameters
            ).fetchone()[0]
            runs = [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM ci_runs WHERE updated_at >= ? AND updated_at < ? "
                    "ORDER BY updated_at DESC, id DESC LIMIT ?",
                    (start, end, limit),
                )
            ]
            run_total = db.execute(
                "SELECT COUNT(*) FROM ci_runs WHERE updated_at >= ? AND updated_at < ?",
                (start, end),
            ).fetchone()[0]
            outcomes = [
                dict(row)
                for row in db.execute(
                    "SELECT status, conclusion, COUNT(*) AS count FROM ci_runs "
                    "WHERE updated_at >= ? AND updated_at < ? GROUP BY status, conclusion "
                    "ORDER BY status, conclusion",
                    (start, end),
                )
            ]
        github = metadata.get("ci", {}).get("repository")
        for commit in commits:
            commit["url"] = (
                f"https://github.com/{github}/commit/{commit['sha']}" if github else None
            )
        return {
            "period": period,
            "timezone": timezone,
            "start": datetime.fromtimestamp(start, UTC).isoformat(),
            "end_exclusive": datetime.fromtimestamp(end, UTC).isoformat(),
            "snapshot": {
                **metadata["snapshot"],
                "repository": Path(metadata["snapshot"]["repository"]).name,
            },
            "ci_index": metadata.get("ci"),
            "commit_count": total,
            "commits": commits,
            "commits_truncated": total > len(commits),
            "ci_update_count": run_total,
            "ci_runs": runs,
            "ci_truncated": run_total > len(runs),
            "ci_outcomes": outcomes,
            "search": search,
        }


def fetch_ci(repository: str, pages: int = 3) -> tuple[list[dict], bool]:
    """Fetch a bounded latest-runs snapshot using the operator's authenticated gh CLI."""
    if not GITHUB_REPOSITORY.fullmatch(repository) or not 1 <= pages <= 10:
        raise ValueError("Use owner/repository and 1 to 10 pages")
    runs = []
    total = 0
    for page in range(1, pages + 1):
        response = json.loads(
            _command(
                [
                    "gh",
                    "api",
                    f"repos/{repository}/actions/runs?per_page=100&page={page}",
                ]
            ).stdout
        )
        total = response["total_count"]
        runs.extend(response["workflow_runs"])
        if len(runs) >= total:
            break
    return runs, len(runs) >= total
