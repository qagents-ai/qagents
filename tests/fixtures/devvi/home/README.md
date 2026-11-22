# A fake Devvi home

`sessions/--work-repo--/qa-fixture/session.v4.jsonl` is laid out as Devvi's
`session-persistence-jsonl` writes it (`<root>/<project key of the cwd>/<session id>/
session.v4.jsonl`, uncompressed) for a session `qa-fixture` with cwd `/work/repo`.

* Turn 1 is `../sdk/bash-tool.session.v4.jsonl` with its tokens filled in (session
  `qa-fixture`, prompt message `msg-1`, cwd `/work/repo`) and `seq`/`time` added, as a
  real log has them. It completed.
* Turn 2 (prompt `msg-2`, "Now add a test.") is written by hand: it started a `write` tool
  call and was never closed, as after a crash.
* The last line is torn (no newline): readers must ignore it.

The log reader tests (`tests/integrations/test_devvi_sdk.py`) only read this directory.
