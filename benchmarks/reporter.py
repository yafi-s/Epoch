"""Report generation for benchmark comparisons."""

from __future__ import annotations

import json
from pathlib import Path

from metrics.collector import MetricsStore


def generate_report(results: dict[str, MetricsStore], output_dir: str = "results") -> str:
    """Generate a markdown comparison report.

    Args:
        results: Mapping of strategy name to MetricsStore.
        output_dir: Directory to save the report.

    Returns:
        The report as a markdown string.
    """
    lines: list[str] = []
    lines.append("# Epoch Benchmark Report\n")
    lines.append("## Summary\n")
    lines.append(f"| {'Strategy':<25} | {'Final Best':>11} | {'Time (s)':>10} | {'Conv. Rate':>11} |")
    lines.append(f"|{'-'*27}|{'-'*13}|{'-'*12}|{'-'*13}|")

    for name, store in results.items():
        s = store.summary()
        lines.append(
            f"| {name:<25} | {s['final_best_fitness']:>11.4f} | "
            f"{s['total_wall_clock_ms'] / 1000.0:>10.1f} | "
            f"{s['convergence_rate']:>11.6f} |"
        )

    lines.append("")
    lines.append("## Per-Strategy Details\n")

    for name, store in results.items():
        lines.append(f"### {name}\n")
        lines.append(f"- Generations: {len(store.generations)}")
        lines.append(f"- Final best fitness: {store.final_best_fitness:.4f}")
        lines.append(f"- Total time: {store.total_wall_clock_ms / 1000.0:.1f}s")
        lines.append(f"- Convergence rate: {store.convergence_rate:.6f}")
        lines.append("")

    report = "\n".join(lines)

    path = Path(output_dir) / "benchmark_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)

    return report
