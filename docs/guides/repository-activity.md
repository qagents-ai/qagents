# Ask about repository activity

Answer questions from a local index of commits and GitHub Actions outcomes.
Refresh the index separately, then query SQLite. Questions do not walk Git
history, contact GitHub or need a model request.

| English | Greek |
| --- | --- |
| What changed yesterday? | Τι άλλαξε χθες; |
| What happened last week and what were the results? | Τι έγινε την προηγούμενη εβδομάδα και ποια ήταν τα αποτελέσματα; |
| What changed today? | Τι άλλαξε σήμερα; |

The [runnable application](https://github.com/qagents-ai/qagents/blob/main/examples/repository_activity.py)
uses the [SQLite index](https://github.com/qagents-ai/qagents/blob/main/examples/repository_activity_index.py).
Run the commands below from a QAgents checkout with the package installed.

## Index before asking

```console
python -m examples.repository_activity index --repo . --database .work/activity.db
```

This indexes commits reachable from `HEAD`. Later refreshes read only new commits
when the previous head is still an ancestor. A rewritten ref rebuilds the index
atomically and discards stale CI evidence. Queries reopen the database read-only,
so it persists across process restarts.

To include actual CI outcomes, install and authenticate the GitHub CLI, then run:

```console
python -m examples.repository_activity index --repo . --database .work/activity.db --github qagents-ai/qagents
```

Use the owner/repository matching your checkout's GitHub origin. The index stores
workflow names, statuses, conclusions, update timestamps and source URLs. It does
not download logs or store GitHub credentials. Only runs whose commit belongs to
the indexed history are retained. API pagination is bounded and partial coverage
is reported explicitly.

The database belongs in an ignored application directory such as `.work/`.
Choose authorized source history and review what you send to a model. Keep the
database out of published documentation and source control.

## Get a fast answer

```console
python -m examples.repository_activity query --database .work/activity.db --question "Τι έγινε χθες;"
python -m examples.repository_activity query --database .work/activity.db --question "What happened last week and what were the results?"
```

This path produces a deterministic summary, counts and citations. No inference
service is involved. The output includes the indexed head, refresh time, UTC
bounds and application timezone. An empty window returns zero matching records.
A missing CI index reports unknown results. Neither implies that work elsewhere
did not happen or that tests passed.

Use `--search voice` to filter commit messages through SQLite FTS5. Counts cover
all matching indexed commits; `--limit` bounds the displayed source list. CI
outcomes are a separate time-window report and are not filtered by commit-message
search. Aggregate outcomes cover all matching CI rows even when their citation
list is truncated.

## Calendar boundaries and freshness

The default timezone is `Europe/Athens`. Yesterday means the previous local
calendar day. Last week means the previous Monday-to-Monday calendar week.
Daylight-saving transitions can make a local day shorter or longer than 24 hours.
Use `--timezone UTC` for UTC windows and `--as-of` with a timezone-aware ISO
timestamp for reproducible questions. Windows may need `python -m pip install tzdata`.

Commit activity uses **committer timestamps**. CI activity uses the workflow
run's **last update timestamp**, which can fall in a different window from its
commit. The Actions snapshot contains the latest fetched state of each run;
it does not reconstruct earlier rerun attempts or prove business outcomes.

Refresh before questions when you need recent activity. No hidden background
sync runs in this example. After a history squash, commits outside the selected
ref are absent from its index; indexing cannot reconstruct removed ancestry.
Use `--ref` only for another history your application is authorized to read.

## Run it through a QAgents agent

```console
python -m examples.repository_activity agent --database .work/activity.db --question "Τι έγινε την προηγούμενη εβδομάδα;"
```

The application takes one SQLite snapshot and exposes it as a typed, read-only
tool. The model cannot choose a filesystem path, repository or another time
window. The default `FunctionModel` is a deterministic routing fixture; it runs
the real QAgents tool loop and needs no key.

For model-written prose, select an explicitly configured provider:

```console
python -m examples.repository_activity agent --database .work/activity.db --provider openai --question "What changed yesterday?"
```

QCloud and Anthropic use the same provider options and environment setup as the
[support application](support-review.md#configure-the-model). The agent has finite
turn, token and time limits. Its output validator rejects incorrect counts,
fabricated outcome aggregates and source identities absent from the tool result.
The natural-language summary still needs a model-quality evaluation. Provider
and network latency are separate from SQLite retrieval time.

## Measure lookup latency

```console
python -m examples.repository_activity benchmark --database .work/activity.db --question "What changed today?" --repetitions 200
```

The report includes the indexed commit count, warm-up time and p50/p95 lookup
latency over repeated reopened SQLite connections. It excludes indexing, Python
startup, inference and network time. Measure your own repository and hardware;
this example does not promise a production response-time target.

The [tests](https://github.com/qagents-ai/qagents/blob/main/tests/examples/test_repository_activity.py)
cover Greek/English dates, DST, incremental refresh, rewritten history, persistence,
read-only indexed queries, CI provenance, hostile search input and invented answers.
