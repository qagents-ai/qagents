"""A fake ``node`` for the Devvi launcher tests. Standard library only, run with ``python -I``.

    python -I fake_node.py [--node-version v24.9.0] --version
    python -I fake_node.py [--node-version ...] <bin.js> [devvi arguments...]

``--version`` prints the Node version. With a script, the script is a fake ``bin.js``
holding JSON: ``{"version": "0.2.1-rc.3", "args": [...], "acp_args": [...]}``.
``<bin.js> --version`` prints that version. ``<bin.js> --profile acp ...`` runs
``fake_devvi_acp.py`` with ``acp_args`` followed by the arguments given; any other call
runs ``fake_devvi_sdk.py`` with ``args`` followed by them, as Node would run the real
``bin.js``.
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(argv: list[str]) -> int:
    version = "v24.9.0"
    if len(argv) >= 2 and argv[0] == "--node-version":
        version, argv = argv[1], argv[2:]
    if not argv or argv[0] == "--version":
        print(version)
        return 0
    script, rest = Path(argv[0]), argv[1:]
    config = json.loads(script.read_text(encoding="utf-8"))
    if "--version" in rest and "--profile" not in rest:
        print(config.get("version", "0.2.1-rc.3"))
        return 0
    profile = rest[rest.index("--profile") + 1] if "--profile" in rest[:-1] else None
    if profile == "acp":
        fake, extra = HERE / "fake_devvi_acp.py", config.get("acp_args", [])
    else:
        fake, extra = HERE / "fake_devvi_sdk.py", config.get("args", [])
    sys.argv = [str(fake), *extra, *rest]
    runpy.run_path(str(fake), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
