# Recover an external action after a lost response

A provider can apply a write and lose the response. The worker needs a
durable intent before it sends the request. During recovery, the recorded
receipt, the destination's lookup and idempotency contract, or an operator
decision determine how execution continues.

This example uses a scripted model and two SQLite databases. One database
holds the run journal. The other simulates the destination's durable ledger.
The first worker commits a credit and then loses its response. A second
worker opens the journal, records an operator's verified receipt, and resumes
the agent. The example checks that the destination contains one credit.

```bash
python -m examples.recover_credit --directory work/credit-recovery-demo
```

Use a fresh directory for each run. The example runs on the Effects
development branch. In an application, verify the destination record against
the business identity and amount, then supply the authenticated operator's
identity to `Settle.committed()`.

```python
--8<-- "examples/recover_credit.py"
```

The [Gym recovery benchmark](https://github.com/mpvasilis/qagent-gym/blob/main/docs/recovery-benchmark.md)
uses worker process termination and a separate HTTP destination. It includes
trials with destination deduplication disabled to confirm that the benchmark
detects duplicate credits.


## Run the offline crash harness

Build an agent with the model supplied by the harness and define its
destination through the crash environment. Record a cassette with
the same model requests and synthetic fixture data.

```bash
qagents crashtest my_fixture:build "Credit DEMO-1" --cassette fixture.json \
  --store sqlite --transport fake --faults kill_after,advance_boundaries \
  --json results.json --junit results.xml
```

The CLI exits with 1 for a violation, 2 for a usage error and 3 for
an inconclusive report. With strict mode it exits with 4 when the
coverage or provider evidence is incomplete. A strict run requires
exercise through the client transport and evidence for destination
assumptions. The in-process harness uses a virtual clock. The Gym
benchmark measures real worker termination on the local machine.
