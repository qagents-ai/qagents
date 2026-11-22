"""Run each frozen seed in an owned interpreter and require an observed regression."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from .catalogue import CASES

pytestmark = pytest.mark.mutants


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
def test_mutation(case, tmp_path):
    if os.environ.get("QAGENTS_RUN_MUTATIONS") != "1":
        pytest.skip("set QAGENTS_RUN_MUTATIONS=1 for the isolated mutation matrix")
    root = Path(__file__).resolve().parents[3]

    def run(mode):
        result_path = tmp_path / (mode + ".json")
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root / "tests")]),
            "QAGENTS_MUTATION_ID": case.id,
            "QAGENTS_MUTATION_MODE": mode,
            "QAGENTS_MUTATION_RESULT": str(result_path),
        }
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--tb=short",
                "-p",
                "effects.mutants._inject",
                case.test,
            ],
            env=env,
            cwd=root,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=45,
        )
        output = result.stdout + result.stderr
        assert result_path.is_file(), output
        evidence = json.loads(result_path.read_text(encoding="utf-8"))
        return evidence, output

    baseline, output = run("baseline")
    assert baseline["exitstatus"] == 0 and baseline["passed"], output
    result, output = run("mutation")
    assert result["calls"] > 0, f"{case.id} did not reach {case.seam}\n{output}"
    assert result["exitstatus"] == 1 and any(
        item["phase"] == "call" and item["oracle"] for item in result["failed"]
    ), f"{case.id} did not produce an assertion failure\n{output}"
