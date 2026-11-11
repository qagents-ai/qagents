"""Settings shared by the whole test suite.

Tests make no network calls. MLflow 3 sends anonymous usage telemetry unless it is
switched off before ``mlflow`` is first imported, and test modules import it while
pytest collects them, so the switch is set here, before any test module loads.
"""

from __future__ import annotations

import os

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("DO_NOT_TRACK", "true")
