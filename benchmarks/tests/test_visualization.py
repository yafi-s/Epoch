"""Tests for benchmark visualization synthesis."""

from __future__ import annotations

import json
import math
from pathlib import Path

from benchmarks.visualization import (
    SeriesData,
    _apply_display_margin,
    _apply_zoomed_axis_display,
    compute_radar_metrics,
    compute_time_to_peak_fraction,
    generate_visualizations,
    resample_to_progress_bins,
)


def _write_run(path: Path, *, run_name: str, best_values: list[float], wall_ms: list[int]) -> None:
    generations = []
    best_so_far = 0.0
    for idx, (best, step_ms) in enumerate(zip(best_values, wall_ms)):
        best_so_far = max(best_so_far, float(best))
        generations.append(
            {
                "generation": idx,
                "best_fitness": float(best),
                "best_so_far": float(best_so_far),
                "avg_fitness": float(best),
                "worst_fitness": float(best),
                "wall_clock_ms": int(step_ms),
            }
        )
    payload = {"run_name": run_name, "generations": generations}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_resample_to_progress_bins_monotonic() -> None:
    generations = [
        {"best_fitness": 0.10, "best_so_far": 0.10, "wall_clock_ms": 1000},
        {"best_fitness": 0.18, "best_so_far": 0.18, "wall_clock_ms": 2000},
        {"best_fitness": 0.16, "best_so_far": 0.18, "wall_clock_ms": 1000},
        {"best_fitness": 0.31, "best_so_far": 0.31, "wall_clock_ms": 1000},
    ]
    best_bins, runtime_bins, total_s = resample_to_progress_bins(generations, bins=6)

    assert len(best_bins) == 6
    assert len(runtime_bins) == 6
    assert math.isclose(total_s, 5.0, rel_tol=0, abs_tol=1e-9)
    assert all(runtime_bins[idx] >= runtime_bins[idx - 1] for idx in range(1, len(runtime_bins)))
    assert all(best_bins[idx] >= best_bins[idx - 1] for idx in range(1, len(best_bins)))


def test_compute_time_to_peak_fraction_with_fallback() -> None:
    reached = compute_time_to_peak_fraction(
        best_bins=[0.2, 0.5, 0.7, 0.9],
        runtime_bins_s=[10.0, 20.0, 30.0, 40.0],
        total_runtime_s=40.0,
        fraction=0.9,
    )
    assert math.isclose(reached, 40.0, rel_tol=0, abs_tol=1e-9)

    fallback = compute_time_to_peak_fraction(
        best_bins=[],
        runtime_bins_s=[],
        total_runtime_s=55.0,
        fraction=0.9,
    )
    assert math.isclose(fallback, 55.0, rel_tol=0, abs_tol=1e-9)


def test_compute_radar_metrics_direct_runtime_t90_and_evals() -> None:
    a = SeriesData(
        key="epoch",
        label="Epoch",
        color="#123456",
        best_bins=[0.8, 0.9],
        runtime_bins_s=[3.0, 6.0],
        peak_fitness=0.9,
        throughput_jobs_per_s=10.0,
        time_to_90_peak_s=3.0,
        total_runtime_s=6.0,
        total_evals=100.0,
        convergence_auc=0.85,
        source="test",
    )
    b = SeriesData(
        key="random_search",
        label="Random Search",
        color="#654321",
        best_bins=[0.6, 0.7],
        runtime_bins_s=[4.0, 8.0],
        peak_fitness=0.7,
        throughput_jobs_per_s=5.0,
        time_to_90_peak_s=8.0,
        total_runtime_s=8.0,
        total_evals=80.0,
        convergence_auc=0.65,
        source="test",
    )

    axis_names, raw, normalized = compute_radar_metrics([a, b])
    assert "Total Run Time (min)" in axis_names
    assert "Time to 90% Peak (min)" in axis_names
    assert raw["epoch"]["Peak Fitness"] > raw["random_search"]["Peak Fitness"]
    assert (
        normalized["epoch"]["Total Run Time (min)"]
        < normalized["random_search"]["Total Run Time (min)"]
    )
    assert (
        normalized["epoch"]["Time to 90% Peak (min)"]
        < normalized["random_search"]["Time to 90% Peak (min)"]
    )
    assert (
        normalized["epoch"]["Total Evals Completed"]
        > normalized["random_search"]["Total Evals Completed"]
    )


def test_display_margin_maps_unit_interval() -> None:
    values = [0.0, 0.5, 1.0]
    mapped = _apply_display_margin(values, lower=0.08, upper=0.92)
    assert math.isclose(mapped[0], 0.08, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(mapped[1], 0.5, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(mapped[2], 0.92, rel_tol=0, abs_tol=1e-9)


def test_zoomed_axis_display_expands_top_cluster() -> None:
    values = [0.173, 0.615, 0.702, 0.921, 0.933, 0.943]
    baseline = _apply_display_margin(values)
    zoomed = _apply_zoomed_axis_display(values)

    baseline_gap = baseline[-1] - baseline[-2]
    zoomed_gap = zoomed[-1] - zoomed[-2]
    assert zoomed_gap > baseline_gap
    assert zoomed[-1] <= 0.92
    assert zoomed[0] >= 0.08


def test_generate_visualizations_creates_expected_outputs(tmp_path: Path) -> None:
    per_seed_dir = tmp_path / "per_seed"
    out_dir = tmp_path / "figures"

    algorithms = [
        "Bayesian (Optuna TPE)",
        "Random Search",
        "Grid Search",
        "Differential Evolution",
        "Simulated Annealing",
    ]

    for algo_idx, algo in enumerate(algorithms):
        for seed in (42, 43, 44):
            base = 0.20 + (algo_idx * 0.05)
            best_values = [base, base + 0.05, base + 0.07, base + 0.08]
            wall = [1100 + algo_idx * 10, 1000, 1200 + (seed - 42) * 20, 900]
            safe_name = algo.lower().replace(" ", "_").replace("(", "").replace(")", "")
            _write_run(
                per_seed_dir / f"{safe_name}_seed{seed}.json",
                run_name=algo,
                best_values=best_values,
                wall_ms=wall,
            )

    modal_run = tmp_path / "ga_run.json"
    modal_summary = tmp_path / "ga_summary.json"
    _write_run(
        modal_run,
        run_name="stress_test",
        best_values=[0.55, 0.72, 0.84, 0.91, 0.93],
        wall_ms=[1000, 1000, 1000, 1000, 1000],
    )
    modal_summary.write_text(
        json.dumps(
            {
                "jobs_per_sec": 46.10,
                "wall_clock_total_s": 531.9059,
                "jobs_submitted": 1440,
                "best_fitness_peak": 0.93,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    outputs = generate_visualizations(
        local_per_seed_dir=per_seed_dir,
        modal_run_path=modal_run,
        modal_summary_path=modal_summary,
        output_dir=out_dir,
        bins=12,
    )

    assert outputs["fitness_chart"].exists()
    assert outputs["runtime_chart"].exists()
    assert outputs["radar_chart"].exists()
    assert outputs["summary"].exists()

    summary_text = outputs["summary"].read_text(encoding="utf-8")
    assert "Failure/timeout axis is omitted" in summary_text
    assert "Time to 90% Peak (min)" in summary_text
    assert "Total Run Time (min)" in summary_text

    epoch = next(row for row in outputs["series"] if row.key == "epoch")
    random_local = next(row for row in outputs["series"] if row.key == "random_search")
    assert math.isclose(epoch.throughput_jobs_per_s, 46.10, rel_tol=0, abs_tol=1e-9)
    assert random_local.total_runtime_s > 10.0
    assert random_local.total_evals == 12.0
    axis_names, raw_metrics, _ = compute_radar_metrics(outputs["series"])
    assert "Time to 90% Peak (min)" in axis_names
    for row in outputs["series"]:
        assert math.isclose(
            raw_metrics[row.key]["Total Run Time (min)"],
            row.runtime_bins_s[-1] / 60.0,
            rel_tol=0,
            abs_tol=1e-6,
        )
