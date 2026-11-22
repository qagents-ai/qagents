"""Make the project root importable, so tests can import ``examples.harness`` and the patterns.

See ``__init__.py`` of this package for why ``examples`` resolves to both directories.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
