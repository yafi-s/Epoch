"""Visualization dashboard for Epoch optimization metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metrics.collector import MetricsStore


def plot_convergence(
    store: MetricsStore,
    output_path: str | Path = "results/convergence.png",
) -> None:
    """Plot fitness vs generation (best, avg, worst).

    Args:
        store: MetricsStore with recorded generation data.
        output_path: Path to save the plot.
    """
    gens = [g.generation for g in store.generations]
    best = [g.best_fitness for g in store.generations]
    avg = [g.avg_fitness for g in store.generations]
    worst = [g.worst_fitness for g in store.generations]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gens, best, "g-o", label="Best", linewidth=2)
    ax.plot(gens, avg, "b-s", label="Average", linewidth=1.5)
    ax.plot(gens, worst, "r-^", label="Worst", linewidth=1)
    ax.fill_between(gens, worst, best, alpha=0.1, color="blue")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (Validation Accuracy)")
    ax.set_title(f"Convergence — {store.run_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_wall_clock(
    store: MetricsStore,
    output_path: str | Path = "results/wall_clock.png",
) -> None:
    """Plot wall-clock time per generation.

    Args:
        store: MetricsStore with recorded generation data.
        output_path: Path to save the plot.
    """
    gens = [g.generation for g in store.generations]
    times = [g.wall_clock_ms / 1000.0 for g in store.generations]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(gens, times, color="steelblue", alpha=0.8)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Wall-Clock Time (seconds)")
    ax.set_title(f"Per-Generation Time — {store.run_name}")
    ax.grid(True, alpha=0.3, axis="y")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_speedup(
    stores_by_workers: dict[int, MetricsStore],
    output_path: str | Path = "results/speedup.png",
) -> None:
    """Plot speedup ratio vs number of workers.

    Args:
        stores_by_workers: Mapping of worker count to MetricsStore.
        output_path: Path to save the plot.
    """
    worker_counts = sorted(stores_by_workers.keys())
    total_times = [stores_by_workers[w].total_wall_clock_ms for w in worker_counts]

    if not total_times or total_times[0] == 0:
        return

    baseline = total_times[0]
    speedups = [baseline / t if t > 0 else 0 for t in total_times]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(worker_counts, speedups, "b-o", linewidth=2, label="Actual Speedup")
    ax.plot(worker_counts, worker_counts, "k--", alpha=0.5, label="Ideal (Linear)")

    ax.set_xlabel("Number of Workers")
    ax.set_ylabel("Speedup (×)")
    ax.set_title("Scalability: Speedup vs Workers")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(
    results: dict[str, MetricsStore],
    output_path: str | Path = "results/comparison.png",
) -> None:
    """Plot convergence curves for multiple strategies side by side.

    Args:
        results: Mapping of strategy name to its MetricsStore.
        output_path: Path to save the plot.
    """
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for i, (name, store) in enumerate(results.items()):
        color = colors[i % len(colors)]
        gens = [g.generation for g in store.generations]
        best = [g.best_fitness for g in store.generations]
        ax1.plot(gens, best, color=color, linewidth=2, label=name, marker="o", markersize=4)

    ax1.set_xlabel("Evaluation Step")
    ax1.set_ylabel("Best Fitness (Validation Accuracy)")
    ax1.set_title("Convergence Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bar chart of final best fitness and total time
    names = list(results.keys())
    final_fitness = [r.final_best_fitness for r in results.values()]
    total_time_s = [r.total_wall_clock_ms / 1000.0 for r in results.values()]

    x = range(len(names))
    bars = ax2.bar(x, final_fitness, color=colors[: len(names)], alpha=0.8)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(names, rotation=15)
    ax2.set_ylabel("Final Best Fitness")
    ax2.set_title("Final Performance")
    ax2.grid(True, alpha=0.3, axis="y")

    # Add time annotations on bars
    for bar, t in zip(bars, total_time_s):
        ax2.annotate(
            f"{t:.1f}s",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
