"""Train, select, confirm, and persist the public synthetic benchmark."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark.artifacts import save_benchmark_run
from benchmark.experiment import run_benchmark
from benchmark.synthetic_data import generate_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="Repository root for evidence, artifacts, examples, and assets",
    )
    args = parser.parse_args()

    command = "python scripts/run_public_benchmark.py"
    started = time.perf_counter()
    splits = generate_benchmark()
    run = run_benchmark(splits)
    outputs = save_benchmark_run(run, splits, args.output_root, command)
    metrics = run.report["confirmation"]["metrics"]

    print(f"Bundle: {outputs['bundle_id']}")
    print(f"Selected: {run.report['validation']['selected_candidate']['name']}")
    print(f"Confirmation chips: {metrics['total_chips']:,}")
    print(f"Observed escapees: {metrics['escapees']}")
    print(f"Escape-rate upper 95%: {metrics['escape_rate_upper_95']:.4%}")
    print(f"Overtest rate: {metrics['overtest_rate']:.2%}")
    print(f"Test-time reduction: {metrics['time_reduction_percent']:.2f}%")
    robustness = run.report["ood_confirmation"]["metrics"]
    print(f"Robustness confirmation recall: {robustness['defect_recall']:.2%}")
    print(f"Accepted: {run.report['acceptance']['passed']}")
    print(f"Elapsed: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()