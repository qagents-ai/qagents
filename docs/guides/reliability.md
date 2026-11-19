# Reliability across repeated attempts

Use repeated evaluation when an agent must work consistently. A single successful
answer gives little evidence about repeated workflows, tool use or provider failures.

QAgents already runs repeated cases with `evaluate(..., repeat=10)`. The
`reliability` report turns those trials into two different questions:

| Metric | Question |
| --- | --- |
| Pass@k | Would at least one of k attempts succeed? |
| Pass^k | Would all k attempts succeed? |

For example, three successful trials out of five give pass@2 = 0.9 and
pass^2 = 0.3. These are without-replacement estimates, rather than powers of the
observed average. Errors count as unsuccessful trials. Incomplete runs count as
unsuccessful when the evaluation requires completion.

## Run the offline example

From a checkout with QAgents installed:

```console
python -m examples.reliability
```

This runs 12 cases ten times for each of two scripted candidates through the real
agent loop. Every run has finite token, turn, cost and time limits. It prints
per-case counts, the aggregate estimates and a paired comparison. The candidates
are deliberately constructed fixtures. Their scores demonstrate the implementation,
and do not measure a model's quality.

## Evaluate an existing agent

Supply the same dataset, evaluator configuration and benchmark revision to both
candidates. Keep answer keys outside agent instructions.

```python
from qagents import compare_reliability, reliability
from qagents.evals import ExactMatch, evaluate


async def compare_agents(baseline_agent, candidate_agent, dataset, revision):
    baseline = await evaluate(
        baseline_agent, dataset, [ExactMatch()], repeat=10,
        metadata={"dataset_revision": revision},
    )
    candidate = await evaluate(
        candidate_agent, dataset, [ExactMatch()], repeat=10,
        metadata={"dataset_revision": revision},
    )
    estimates = reliability(candidate, ks=(1, 3, 5))
    paired = compare_reliability(
        baseline, candidate, k=3, metric="pass_power_k",
        confidence=0.95, resamples=10000, seed=42,
    )
    return estimates, paired
```

Use `model_dump_json(indent=2)` to save either result, or
`estimates.to_markdown()` for a compact metric table. The functions perform no
model or service calls.

## Match trials before comparing

The API rejects missing or duplicated repetitions, an oversized k, different
case sets, evaluator names, completion rules, case metadata or benchmark revisions.
Each case must contain repetitions 0 through `report.repeat - 1`. Paired reports
must use the same dataset name and repetition count.

A name or revision label cannot prove identical task inputs or scoring rules.
Keep the actual inputs, verifier, memory snapshot and trial sampling conditions
fixed. A fresh agent session resets the conversation; it does not reset an
external database or service. Create isolated application state per attempt when
a task changes external state.

## Understand the estimates

Use `TrialTarget(function)` when each repetition needs its own deterministic seed
or fresh application fixture. The function receives `(case, repetition)`, with a
zero-based repetition index, and returns an Agent or RunResult. Sync functions
run in a worker thread; async functions run on the event loop. Setup failures
become failed trials in the report. The wrapper avoids shared counters and changes
no dataset metadata.

Treat the case as read-only. The factory owns deadlines and usage accounting when
it returns a RunResult it ran itself. Evaluation limits apply to an Agent returned
by the factory, as in the existing one-argument target contract.

For n trials and c successes, the estimators are:

```text
pass@k = 1 - choose(n-c, k) / choose(n, k)
pass^k = choose(c, k) / choose(n, k)
```

The report averages cases equally. Independent, representative repeated attempts
are needed to interpret these estimates as future success probabilities.

The paired comparison resamples whole cases, retaining every repetition belonging
to each sampled case. It returns the mean candidate-minus-baseline change and a
percentile bootstrap interval. It requires at least two cases and 100 resamples.
Its seeded random generator is local; result ordering does not affect the output.
The interval describes variation across the supplied cases and does not establish
performance on other workloads.

This design is informed by the repeatability question in
[τ-bench](https://arxiv.org/abs/2406.12045) and by the need to retain independent
task verification in [τ²-bench](https://github.com/sierra-research/tau2-bench).
QAgents does not claim to reproduce their benchmark results.

