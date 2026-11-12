"""Test helpers for the external-agent integrations.

* `streams`: in-memory stream pairs, to connect two `JsonRpcPeer` objects (or a test and
  one peer) without a process, with a drain that a test can hold back.
* `jsonrpc_peer.py`: a scripted JSON-RPC 2.0 peer, run as a subprocess.
* `process_tree.py`: a process that starts a child writing a heartbeat file, to prove
  that a process tree is gone after `ManagedProcess.close`.

The two scripts use the standard library only and run with ``python -I``, so they
import nothing from the project. `script(name)` gives the command line for one.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def script(name: str, *args: str) -> list[str]:
    """``[python, -I, <fakes>/<name>.py, *args]``."""
    return [sys.executable, "-I", str(HERE / f"{name}.py"), *args]
