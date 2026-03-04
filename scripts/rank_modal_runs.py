#!/usr/bin/env python3
"""Rank Modal run summaries by throughput-first priorities.

Priority order:
1) jobs_per_sec (desc)
2) queue_overhead_p90_pct (asc)
3) avg_job_ms (asc)
4) best_fitness_peak (desc)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RankedRun:
    run_id: str
    summary_path: Path
    jobs_per_sec: float
    queue_overhead_p90_pct: float
    avg_job_ms: float
    best_fitness_peak: float
    stop_reason: str
    jobs_submitted: int
    jobs_returned: int


def _load_summary(path: Path) -> RankedRun | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    run_id = path.stem.replace("_summary", "")
    best_fitness_peak = data.get("best_fitness_peak")
    if best_fitness_peak is None and "best_fitness_final" in data:
        best_fitness_peak = data.get("best_fitness_final")
    if best_fitness_peak is None and "best_fitness" in data:
        best_fitness_peak = data.get("best_fitness")
    if best_fitness_peak is None:
        run_json_path = path.with_name(f"{run_id}.json")
        if run_json_path.exists():
            try:
                run_data = json.loads(run_json_path.read_text(encoding="utf-8"))
                generations = run_data.get("generations", [])
                if generations:
                    best_fitness_peak = max(
                        float(g.get("best_fitness", 0.0)) for g in generations
                    )
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                best_fitness_peak = None
    if best_fitness_peak is None:
        best_fitness_peak = 0.0

    return RankedRun(
        run_id=str(run_id),
        summary_path=path,
        jobs_per_sec=float(data.get("jobs_per_sec", 0.0)),
        queue_overhead_p90_pct=float(data.get("queue_overhead_p90_pct", 0.0)),
        avg_job_ms=float(data.get("avg_job_ms_raw", data.get("avg_job_ms", 0.0))),
        best_fitness_peak=float(best_fitness_peak),
        stop_reason=str(data.get("stop_reason", "unknown")),
        jobs_submitted=int(data.get("jobs_submitted", data.get("jobs_total", 0))),
        jobs_returned=int(data.get("jobs_returned", 0)),
    )


def _rank(runs: list[RankedRun]) -> list[RankedRun]:
    return sorted(
        runs,
        key=lambda r: (
            -r.jobs_per_sec,
            r.queue_overhead_p90_pct,
            r.avg_job_ms,
            -r.best_fitness_peak,
        ),
    )


def _print_table(runs: list[RankedRun]) -> None:
    if not runs:
        print("No summary files found.")
        return

    header = (
        "rank",
        "run",
        "jobs/s",
        "q_over_p90%",
        "avg_job_ms",
        "best_peak",
        "submitted",
        "returned",
        "stop_reason",
    )
    print(
        f"{header[0]:>4}  {header[1]:<24} {header[2]:>8} {header[3]:>11} "
        f"{header[4]:>10} {header[5]:>9} {header[6]:>9} {header[7]:>8} {header[8]}"
    )
    print("-" * 105)

    for idx, run in enumerate(runs, start=1):
        print(
            f"{idx:>4}  {run.run_id:<24.24} {run.jobs_per_sec:>8.3f} "
            f"{run.queue_overhead_p90_pct * 100:>10.2f}% {run.avg_job_ms:>10.1f} "
            f"{run.best_fitness_peak:>9.4f} {run.jobs_submitted:>9d} {run.jobs_returned:>8d} {run.stop_reason}"
        )


def _write_markdown(path: Path, runs: list[RankedRun]) -> None:
    lines = [
        "# Ranked Modal Runs",
        "",
        "Sorted by: jobs_per_sec desc, queue_overhead_p90_pct asc, avg_job_ms asc, best_fitness_peak desc.",
        "",
        "| Rank | Run | Jobs/s | Queue p90 % | Avg job ms | Best peak | Submitted | Returned | Stop reason |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for idx, run in enumerate(runs, start=1):
        lines.append(
            "| "
            f"{idx} | {run.run_id} | {run.jobs_per_sec:.3f} | "
            f"{run.queue_overhead_p90_pct * 100:.2f}% | {run.avg_job_ms:.1f} | "
            f"{run.best_fitness_peak:.4f} | {run.jobs_submitted} | {run.jobs_returned} | {run.stop_reason} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank Modal run summaries.")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory to recursively search for *_summary.json",
    )
    parser.add_argument(
        "--pattern",
        default="*_summary.json",
        help="Glob pattern used under --results-dir (default: *_summary.json)",
    )
    parser.add_argument(
        "--markdown-output",
        default="",
        help="Optional markdown report output path",
    )
    parser.add_argument(
        "--write-markdown",
        action="store_true",
        help="Write markdown report to <results-dir>/ranked_report.md",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    paths = sorted(results_dir.rglob(args.pattern))
    runs = [loaded for loaded in (_load_summary(path) for path in paths) if loaded is not None]
    ranked = _rank(runs)

    _print_table(ranked)

    markdown_path = None
    if args.markdown_output:
        markdown_path = Path(args.markdown_output)
    elif args.write_markdown:
        markdown_path = results_dir / "ranked_report.md"

    if markdown_path is not None:
        _write_markdown(markdown_path, ranked)
        print(f"\nMarkdown report written to: {markdown_path}")


if __name__ == "__main__":
    main()
