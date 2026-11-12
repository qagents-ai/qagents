"""A fake ``devvi`` CLI for the ACP transport. Standard library only, run with ``python -I``.

    python -I fake_devvi_acp.py [--start-record FILE] [fake_acp_agent options] --version
    python -I fake_devvi_acp.py [...] --profile acp --patch FILE [--patch FILE...] [--dump-config]
    python -I fake_devvi_acp.py [--start-record FILE] [--script FIXTURE] [...] --profile acp --patch FILE

It reads what the Devvi launcher puts on the command line (``--profile``, ``--patch``,
``--dump-config``, ``--version``) and plays an ACP fixture with ``fake_acp_agent.py``, which
gets every other argument (``--script``, ``--log``, ``--session-id``, ...). Without
``--script`` it plays ``tests/fixtures/devvi/acp/handshake.jsonl``.

* ``--plan FILE`` picks the fixture by working directory and start count instead: FILE is
  JSON ``{"<absolute cwd>": ["first.jsonl", "second.jsonl", ...]}``; the Nth process started
  in a directory plays the Nth fixture (the last one repeats). The counts are kept in
  ``FILE.count.json``. This is how a test gives one workspace a fixture per process (a
  restart after a crash) and another workspace its own.

* ``--version`` (without ``--profile``) prints ``--version-text`` (default 0.2.1-rc.3).
* ``--dump-config`` prints the patch files, as ``devvi --dump-config`` shows the composed rows.
* ``--start-record FILE`` appends one JSON line before playing: the arguments, the process
  id, the working directory, the names (never the values) of the environment variables,
  ``DEVVI_HOME``/``DSH_HOME``, ``DSH_PERMISSION_MODE`` and ``DSH_TELEMETRY_DISABLED``.
* ``--exit-on-start CODE`` (with ``--stderr TEXT``) exits at once, like a CLI that fails.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HANDSHAKE = HERE.parents[1] / "fixtures" / "devvi" / "acp" / "handshake.jsonl"
VERSION = "0.2.1-rc.3"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--version-text", default=VERSION)
    parser.add_argument("--profile")
    parser.add_argument("--patch", action="append", default=[])
    parser.add_argument("--dump-config", action="store_true")
    parser.add_argument("--start-record")
    parser.add_argument("--exit-on-start", type=int)
    parser.add_argument("--stderr")
    parser.add_argument("--plan")
    options, rest = parser.parse_known_args(argv)
    if options.stderr:
        sys.stderr.write(options.stderr + "\n")
        sys.stderr.flush()
    if options.exit_on_start is not None:
        return options.exit_on_start
    if options.version and not options.profile:
        print(options.version_text)
        return 0
    if options.dump_config:
        for patch in options.patch:
            print(Path(patch).read_text(encoding="utf-8"))
        return 0
    if options.start_record:
        start = {
            "argv": argv,
            "profile": options.profile,
            "patches": options.patch,
            "pid": os.getpid(),
            "cwd": os.getcwd(),
            "env_names": sorted(os.environ),
            "home": {key: os.environ.get(key) for key in ("DEVVI_HOME", "DSH_HOME")},
            "permission_mode": os.environ.get("DSH_PERMISSION_MODE"),
            "telemetry_disabled": os.environ.get("DSH_TELEMETRY_DISABLED"),
        }
        with open(options.start_record, "a", encoding="utf-8") as file:
            file.write(json.dumps({"start": start}) + "\n")
    if options.plan:
        rest = ["--script", planned(Path(options.plan)), *rest]
    elif "--script" not in rest:
        rest = ["--script", str(HANDSHAKE), *rest]
    sys.argv = [str(HERE / "fake_acp_agent.py"), *rest]
    runpy.run_path(str(HERE / "fake_acp_agent.py"), run_name="__main__")
    return 0


def planned(plan: Path) -> str:
    """The fixture of this start: by working directory, then by how many started there."""
    here = os.path.normcase(os.getcwd())
    scripts = {os.path.normcase(key): value for key, value in json.loads(plan.read_text()).items()}
    if here not in scripts:
        raise SystemExit(f"fake_devvi_acp: the plan has no fixture for {os.getcwd()}")
    counter = plan.with_name(plan.name + ".count.json")
    counts = json.loads(counter.read_text()) if counter.exists() else {}
    number = counts.get(here, 0)
    counts[here] = number + 1
    counter.write_text(json.dumps(counts))
    chosen = scripts[here][min(number, len(scripts[here]) - 1)]
    return str(plan.parent / chosen)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
