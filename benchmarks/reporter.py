"""Aggregate reporting for local benchmark comparisons."""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from metrics.collector import MetricsStore
from metrics.summary import observed_throughput, legacy_modeled_throughput, total_elapsed_seconds


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _peak_fitness(store: MetricsStore) -> float:
    return max((float(g.best_so_far) for g in store.generations), default=0.0)


def _strategy_summary(name: str, per_seed: dict[int, MetricsStore]) -> dict[str, Any]:
    seed_runs: list[dict[str, Any]] = []
    peaks: list[float] = []
    evals: list[float] = []
    wall_times: list[float] = []

    for seed in sorted(per_seed.keys()):
        store = per_seed[seed]
        peak = _peak_fitness(store)
        eval_count = len(store.generations)
        wall_s = float(store.total_wall_clock_ms) / 1000.0

        seed_runs.append(
            {
                "seed": int(seed),
                "evals_completed": int(eval_count),
                "best_fitness_peak": peak,
                "final_best_fitness": float(store.final_best_fitness),
                "wall_clock_s": wall_s,
            }
        )
        peaks.append(peak)
        evals.append(float(eval_count))
        wall_times.append(wall_s)

    return {
        "name": name,
        "reference_only": False,
        "seed_runs": seed_runs,
        "mean_best_fitness_peak": _mean(peaks),
        "median_best_fitness_peak": _median(peaks),
        "mean_evals_completed": _mean(evals),
        "median_evals_completed": _median(evals),
        "mean_wall_clock_s": _mean(wall_times),
        "median_wall_clock_s": _median(wall_times),
    }


def _load_reference_row(reference_summary_path: str) -> dict[str, Any] | None:
    path = Path(reference_summary_path)
    if not path.exists():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    observed = observed_throughput(data)
    return {
        "name": "Modal GA (g12_p120)",
        "reference_only": True,
        "source_path": str(path),
        "best_fitness_peak": float(data.get("best_fitness_peak", 0.0)),
        "wall_clock_s": float(data.get("wall_clock_s", 0.0)),
        "wall_clock_total_s": total_elapsed_seconds(data),
        "jobs_submitted": int(data.get("jobs_submitted", data.get("jobs_total", 0))),
        "jobs_returned": int(data.get("jobs_returned", 0)),
        "completed": observed.completed if observed else None,
        "jobs_per_sec": observed.jobs_per_sec if observed else None,
        "legacy_modeled_jobs_per_sec": legacy_modeled_throughput(data),
        "throughput_scope": "successful_completions_per_total_observed_second" if observed else "unknown",
        "raw_jobs_per_sec": float(data.get("raw_jobs_per_sec", 0.0)),
    }


def build_aggregate_summary(
    *,
    results: dict[str, dict[int, MetricsStore]],
    config: dict[str, Any],
    reference_summary_path: str,
) -> dict[str, Any]:
    """Build ranked aggregate summary from per-seed strategy results."""
    local_rows = [_strategy_summary(name, seed_runs) for name, seed_runs in results.items()]
    ranked = sorted(
        local_rows,
        key=lambda row: (
            -row["median_best_fitness_peak"],
            -row["median_evals_completed"],
            -row["mean_best_fitness_peak"],
        ),
    )

    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "ranked_algorithms": ranked,
        "reference_row": _load_reference_row(reference_summary_path),
    }


def write_ranked_report(*, summary: dict[str, Any], output_path: str | Path) -> None:
    """Write markdown report for aggregate benchmark results."""
    lines: list[str] = []
    config = summary.get("config", {})
    lines.append("# Local Algorithm Benchmark Report")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Space mode: `{config.get('space_mode', 'unknown')}`")
    lines.append(f"- Dataset: `{config.get('dataset', 'unknown')}`")
    lines.append(f"- Seeds: `{config.get('seeds', [])}`")
    lines.append(f"- Max wall-clock per run (s): `{config.get('max_wall_clock_s', 0.0)}`")
    lines.append(f"- Eval budget per run: `{config.get('budget', 0)}` (0 means disabled)")
    lines.append("")
    lines.append("## Ranked Local Algorithms")
    lines.append("")
    lines.append(
        "| Rank | Algorithm | Median peak | Mean peak | Median evals | Mean evals | "
        "Median time (s) | Mean time (s) |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for row in summary.get("ranked_algorithms", []):
        lines.append(
            f"| {row['rank']} | {row['name']} | "
            f"{row['median_best_fitness_peak']:.6f} | {row['mean_best_fitness_peak']:.6f} | "
            f"{row['median_evals_completed']:.1f} | {row['mean_evals_completed']:.1f} | "
            f"{row['median_wall_clock_s']:.3f} | {row['mean_wall_clock_s']:.3f} |"
        )

    reference = summary.get("reference_row")
    if reference:
        lines.append("")
        lines.append("## Reference Only (Not Ranked)")
        lines.append("")
        lines.append(
            "| Name | Best peak | Wall clock total (s) | Jobs submitted | Jobs returned | "
            "Completed/s (total) | Legacy modeled jobs/s | Source |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        observed_rate = reference.get('jobs_per_sec')
        modeled_rate = reference.get('legacy_modeled_jobs_per_sec')
        observed_text = f'{observed_rate:.6f}' if observed_rate is not None else 'unknown'
        modeled_text = f'{modeled_rate:.6f}' if modeled_rate is not None else 'unknown'
        total_time = reference.get('wall_clock_total_s')
        total_text = f'{total_time:.6f}' if total_time is not None else 'unknown'
        lines.append(
            f"| {reference.get('name', 'reference')} | "
            f"{reference.get('best_fitness_peak', 0.0):.6f} | "
            f"{total_text} | "
            f"{reference.get('jobs_submitted', 0)} | "
            f"{reference.get('jobs_returned', 0)} | {observed_text} | "
            f"{modeled_text} | `{reference.get('source_path', '')}` |"
        )

    lines.append("")
    lines.append("## Per-Seed Results")
    lines.append("")
    lines.append("| Algorithm | Seed | Evals completed | Best peak | Final best | Wall clock (s) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in summary.get("ranked_algorithms", []):
        for seed_row in row.get("seed_runs", []):
            lines.append(
                f"| {row['name']} | {seed_row['seed']} | {seed_row['evals_completed']} | "
                f"{seed_row['best_fitness_peak']:.6f} | {seed_row['final_best_fitness']:.6f} | "
                f"{seed_row['wall_clock_s']:.3f} |"
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
