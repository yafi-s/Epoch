#!/usr/bin/env python3
"""Validate benchmark summary acceptance criteria and print key metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt_pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate modal benchmark summary metrics.")
    parser.add_argument(
        "--summary-path",
        default="results/modal_benchmark_t4_w10/ga_benchmark_g12_p120_summary.json",
        help="Path to run summary JSON",
    )
    parser.add_argument(
        "--expected-jobs",
        type=int,
        default=1440,
        help="Expected jobs_submitted and jobs_returned",
    )
    parser.add_argument(
        "--min-active-workers",
        type=int,
        default=10,
        help="Minimum active_workers_min",
    )
    parser.add_argument(
        "--max-failure-rate",
        type=float,
        default=0.0,
        help="Maximum acceptable failure_rate",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary_path)
    if not summary_path.exists():
        raise SystemExit(f"Summary file not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    checks = [
        (
            int(summary.get("jobs_submitted", -1)) == args.expected_jobs,
            f"jobs_submitted == {args.expected_jobs}",
            f"got {summary.get('jobs_submitted')}",
        ),
        (
            int(summary.get("jobs_returned", -1)) == args.expected_jobs,
            f"jobs_returned == {args.expected_jobs}",
            f"got {summary.get('jobs_returned')}",
        ),
        (
            float(summary.get("failure_rate", 1.0)) <= float(args.max_failure_rate),
            f"failure_rate <= {args.max_failure_rate}",
            f"got {summary.get('failure_rate')}",
        ),
        (
            int(summary.get("active_workers_min", 0)) >= args.min_active_workers,
            f"active_workers_min >= {args.min_active_workers}",
            f"got {summary.get('active_workers_min')}",
        ),
        (
            bool(summary.get("stopped_early", True)) is False,
            "stopped_early == false",
            f"got {summary.get('stopped_early')}",
        ),
    ]

    failed = [f"[FAIL] {name} ({detail})" for ok, name, detail in checks if not ok]
    passed = [f"[OK]   {name}" for ok, name, _ in checks if ok]

    print("Acceptance checks")
    for line in passed + failed:
        print(f"  {line}")

    print("\nKey metrics")
    print(f"  jobs_per_sec:                {summary.get('jobs_per_sec', 0.0):.6f}")
    print(f"  raw_jobs_per_sec:            {summary.get('raw_jobs_per_sec', 0.0):.6f}")
    print(
        "  jobs_per_sec_wall_adjusted:  "
        f"{summary.get('jobs_per_sec_wall_adjusted', 0.0):.6f}"
    )
    print(
        "  dispatch_latency_p90_ms:     "
        f"{summary.get('dispatch_latency_p90_ms', 0.0):.6f}"
    )
    print(
        "  worker_idle_gap_p90_ms:      "
        f"{summary.get('worker_idle_gap_p90_ms', 0.0):.6f}"
    )
    print(f"  queue_wait_p90_ms:           {summary.get('queue_wait_p90_ms', 0.0):.6f}")
    print(
        "  queue_overhead_p90_pct:      "
        f"{_fmt_pct(float(summary.get('queue_overhead_p90_pct', 0.0)))}"
    )
    print(f"  best_fitness_peak:           {summary.get('best_fitness_peak', 0.0):.6f}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
