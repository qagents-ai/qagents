"""Run every example pattern, one after another, and write a results table.

    python -m examples.run_all --provider anthropic
    python -m examples.run_all --provider qcloud --only react_assistant,supervisor --max-cost 2
    python -m examples.run_all --provider anthropic --record   # also write the cassettes
    python -m examples.run_all --provider anthropic --replay   # offline, from the cassettes

``--max-cost`` (default 5 USD) is the budget of the whole run: each example may spend what is
left of it, at most ``--max-cost-each`` (default 0.50), and once it is spent no further example
is started. The table is printed at the end and written to ``examples/results/<provider>.md``
and ``.json`` (``<provider>-replay.*`` for a replay, so a replay never overwrites the results of
a live run). The exit status is 0 when every example succeeded.

The other options (``--model``, ``--mlflow``, ``--interactive``, ``--verbose``, ``--workdir``)
are passed to every example; see ``examples/harness.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import pkgutil
import sys
from collections.abc import Iterable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from examples.harness import (
    DEFAULT_MAX_COST_USD,
    RESULTS_DIR,
    ExampleReport,
    ModelFactory,
    RunOptions,
    add_common_arguments,
    make_model,
    options_from,
    prepare_stdout,
    run_example,
)

PACKAGE = "examples.patterns"
REQUIRED = ("NAME", "DESCRIPTION", "INSPIRED_BY", "run")


def discover(package: str = PACKAGE) -> list[str]:
    """Full names of the example modules in ``package``, sorted; ``_private`` ones left out."""
    root = importlib.import_module(package)
    return sorted(
        f"{package}.{info.name}"
        for info in pkgutil.iter_modules(root.__path__)
        if not info.name.startswith("_")
    )


def select(modules: list[str], only: str | None) -> list[str]:
    """The modules named in ``--only`` (short module names, comma separated)."""
    if not only:
        return modules
    wanted = [name.strip() for name in only.split(",") if name.strip()]
    by_name = {module.rsplit(".", 1)[-1]: module for module in modules}
    unknown = [name for name in wanted if name not in by_name]
    if unknown:
        raise SystemExit(
            f"Unknown example(s) {', '.join(unknown)}. Available: {', '.join(sorted(by_name))}."
        )
    return [by_name[name] for name in wanted]


async def run_all(
    modules: Iterable[str],
    options: RunOptions,
    *,
    max_cost_usd: float,
    max_cost_each: float = DEFAULT_MAX_COST_USD,
    model_factory: ModelFactory = make_model,
) -> list[ExampleReport]:
    """Run the modules in order and return one report each.

    A replay spends nothing, so it is never stopped by the budget.
    """
    reports: list[ExampleReport] = []
    spent = 0.0
    for module_name in modules:
        short = module_name.rsplit(".", 1)[-1]
        left = max_cost_usd - spent
        if not options.replay and left <= 0:
            reports.append(
                ExampleReport(
                    name=short,
                    provider=options.provider,
                    skipped=True,
                    error=f"not started: the ${max_cost_usd:.2f} budget was spent",
                )
            )
            continue
        print(f"\n== {short}", flush=True)
        try:
            module = importlib.import_module(module_name)
            missing = [attribute for attribute in REQUIRED if not hasattr(module, attribute)]
            if missing:
                raise AttributeError(f"{module_name} does not define {', '.join(missing)}")
        except Exception as error:  # one broken module must not stop the others
            print(f"error: {type(error).__name__}: {error}")
            reports.append(
                ExampleReport(
                    name=short, provider=options.provider, error=f"{type(error).__name__}: {error}"
                )
            )
            continue
        print(f"   {module.DESCRIPTION} (inspired by {module.INSPIRED_BY})")
        each = replace(
            options,
            max_cost_usd=min(max_cost_each, left),
            workdir=None if options.workdir is None else options.workdir / module.NAME,
        )
        report = await run_example(
            module.run, name=module.NAME, options=each, model_factory=model_factory
        )
        spent += report.cost_usd or 0.0
        reports.append(report)
    return reports


def _cost(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _status(report: ExampleReport) -> str:
    return "skipped" if report.skipped else "ok" if report.ok else "FAILED"


def _cell(text: str | None) -> str:
    return " ".join((text or "").split()).replace("|", "\\|")


def markdown_table(reports: list[ExampleReport], *, provider: str, mode: str) -> str:
    """The results as a Markdown page."""
    models = sorted({report.model for report in reports if report.model})
    passed = sum(report.ok for report in reports)
    total_cost = sum(report.cost_usd or 0.0 for report in reports)
    lines = [
        f"# Example results: {provider}",
        "",
        f"- Model: {', '.join(models) or '-'}",
        f"- Mode: {mode}",
        f"- Run at: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Passed: {passed} of {len(reports)}; cost {total_cost:.4f} USD",
        "",
        "| Example | Status | Requests | Input tokens | Output tokens | Cost (USD) | Seconds "
        "| Summary |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for report in reports:
        lines.append(
            f"| {report.name} | {_status(report)} | {report.requests} | "
            f"{report.input_tokens:,} | {report.output_tokens:,} | {_cost(report.cost_usd)} | "
            f"{report.seconds:.1f} | {_cell(report.error or report.summary)} |"
        )
    return "\n".join(lines) + "\n"


def text_table(reports: list[ExampleReport]) -> str:
    """The results as a plain-text table for the terminal."""
    header = ("example", "status", "requests", "tokens in/out", "cost", "seconds")
    rows = [
        (
            report.name,
            _status(report),
            str(report.requests),
            f"{report.input_tokens:,}/{report.output_tokens:,}",
            _cost(report.cost_usd),
            f"{report.seconds:.1f}",
        )
        for report in reports
    ]
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    lines = [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        for row in [header, *rows]
    ]
    return "\n".join(lines)


def write_results(
    reports: list[ExampleReport], *, provider: str, mode: str, results_dir: Path
) -> tuple[Path, Path]:
    """Write ``<provider>.md`` and ``<provider>.json`` (``<provider>-replay.*`` for a replay)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{provider}-replay" if mode == "replay" else provider
    markdown = results_dir / f"{stem}.md"
    data = results_dir / f"{stem}.json"
    markdown.write_text(markdown_table(reports, provider=provider, mode=mode), encoding="utf-8")
    payload = {
        "provider": provider,
        "mode": mode,
        "reports": [report.model_dump(mode="json") for report in reports],
    }
    data.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return markdown, data


def main(
    argv: Iterable[str] | None = None,
    *,
    package: str = PACKAGE,
    model_factory: ModelFactory = make_model,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m examples.run_all", description=__doc__.split("\n")[0]
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--only", help="comma-separated module names, such as react_assistant,supervisor"
    )
    parser.add_argument("--max-cost", type=float, default=5.0, help="USD for the whole run")
    parser.add_argument(
        "--max-cost-each",
        type=float,
        default=DEFAULT_MAX_COST_USD,
        help=f"USD per example (default {DEFAULT_MAX_COST_USD:.2f})",
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help=argparse.SUPPRESS)
    args = parser.parse_args(None if argv is None else list(argv))
    prepare_stdout()

    modules = select(discover(package), args.only)
    options = options_from(args, max_cost_usd=args.max_cost_each)
    mode = "replay" if args.replay else "record" if args.record else "live"
    try:
        reports = asyncio.run(
            run_all(
                modules,
                options,
                max_cost_usd=args.max_cost,
                max_cost_each=args.max_cost_each,
                model_factory=model_factory,
            )
        )
    except KeyboardInterrupt:
        print("interrupted")
        return 130
    markdown, data = write_results(
        reports, provider=args.provider, mode=mode, results_dir=args.results_dir
    )
    print(f"\n{text_table(reports)}")
    passed = sum(report.ok for report in reports)
    total_cost = sum(report.cost_usd or 0.0 for report in reports)
    print(f"\n{passed} of {len(reports)} passed, {total_cost:.4f} USD")
    print(f"results: {markdown.as_posix()} and {data.as_posix()}")
    if args.json:
        print(json.dumps([report.model_dump(mode="json") for report in reports]))
    return 0 if reports and passed == len(reports) else 1


if __name__ == "__main__":
    sys.exit(main())
