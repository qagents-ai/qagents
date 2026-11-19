"""Measure repeatability using bounded real agent runs and scripted responses.

python -m examples.reliability
This checks evaluation arithmetic and runtime integration; it measures no live model.
"""

from __future__ import annotations

import asyncio
import hashlib
import json

from qagents import Agent, Limits, TrialTarget, compare_reliability, reliability
from qagents.evals import Case, Dataset, ExactMatch, evaluate
from qagents.models.testing import ScriptedModel


async def run_example() -> dict:
    dataset = Dataset(
        [
            Case(
                name=f"conversion-{index}",
                prompt=f"Convert {index + 1} metres to centimetres. Return only the number.",
                expected=str((index + 1) * 100),
                metadata={"index": index},
            )
            for index in range(12)
        ],
        name="repeatability-contract",
    )
    canonical = json.dumps(
        [case.model_dump(mode="json", exclude={"deps"}) for case in dataset.cases],
        sort_keys=True,
    )
    revision = hashlib.sha256(canonical.encode()).hexdigest()

    def target(candidate: bool):
        async def make_agent(case: Case, trial: int) -> Agent:
            index = case.metadata["index"]
            successes = 7 + index % 4 if candidate else 6 + index % 3
            answer = case.expected if trial % 10 < successes else "incorrect"
            return Agent(
                ScriptedModel([answer]),
                limits=Limits(max_turns=2, max_total_tokens=256, max_cost_usd=0.05, timeout_s=5),
            )

        return TrialTarget(make_agent)

    baseline = await evaluate(
        target(False),
        dataset,
        [ExactMatch()],
        repeat=10,
        name="baseline fixture",
        metadata={"dataset_revision": revision},
    )
    current = await evaluate(
        target(True),
        dataset,
        [ExactMatch()],
        repeat=10,
        name="candidate fixture",
        metadata={"dataset_revision": revision},
    )
    return {
        "mode": "scripted runtime and statistics contract",
        "cases": len(dataset),
        "repeat": 10,
        "baseline": reliability(baseline, ks=(1, 3, 5)).model_dump(mode="json"),
        "current": reliability(current, ks=(1, 3, 5)).model_dump(mode="json"),
        "paired": compare_reliability(baseline, current, k=3, seed=42).model_dump(mode="json"),
    }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run_example()), indent=2))
