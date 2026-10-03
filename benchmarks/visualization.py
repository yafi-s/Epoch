"""Visualization synthesis for local-vs-Modal benchmark comparisons."""

from __future__ import annotations

import json
import math
import statistics
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

DEFAULT_BINS = 12
DEFAULT_LOCAL_PER_SEED_DIR = Path("results/local_benchmark_g12_p120/per_seed")
DEFAULT_MODAL_RUN_PATH = Path("results/modal_benchmark_t4_w10/ga_benchmark_g12_p120.json")
DEFAULT_MODAL_SUMMARY_PATH = Path(
    "results/modal_benchmark_t4_w10/ga_benchmark_g12_p120_summary.json"
)
DEFAULT_OUTPUT_DIR = Path("results/local_benchmark_g12_p120/figures")

RUN_NAME_TO_KEY = {
    "Bayesian (Optuna TPE)": "optuna_tpe",
    "Random Search": "random_search",
    "Grid Search": "grid_search",
    "Differential Evolution": "differential_evolution",
    "Simulated Annealing": "simulated_annealing",
}

ALGO_STYLES = {
    "epoch": {"label": "Epoch", "color": "#0B4FBA"},
    "optuna_tpe": {"label": "Optuna TPE (local 3070)", "color": "#148A08"},
    "random_search": {"label": "Random Search (local 3070)", "color": "#E26D1F"},
    "grid_search": {"label": "Grid Search (local 3070)", "color": "#7A3EA1"},
    "differential_evolution": {"label": "Differential Evolution (local 3070)", "color": "#D7263D"},
    "simulated_annealing": {"label": "Simulated Annealing (local 3070)", "color": "#008B8B"},
}

DISPLAY_ORDER = [
    "epoch",
    "optuna_tpe",
    "random_search",
    "grid_search",
    "differential_evolution",
    "simulated_annealing",
]

RADAR_AXES = [
    ("Peak Fitness", "peak_fitness", "clip01", 1.0),
    ("Time to 90% Peak (min)", "time_to_90_peak_s", "max_ratio", 1.0 / 60.0),
    ("Total Run Time (min)", "total_runtime_s", "max_ratio", 1.0 / 60.0),
    ("Total Evals Completed", "total_evals", "max_ratio", 1.0),
    ("Convergence AUC", "convergence_auc", "clip01", 1.0),
]

RADAR_ZOOMED_AXES = {
    "Peak Fitness",
    "Total Run Time (min)",
    "Convergence AUC",
}

RADAR_ZOOM_SETTINGS = {
    "Peak Fitness": {
        "lower": 0.08,
        "lower_end": 0.11,
        "upper_start": 0.12,
        "upper": 0.92,
    },
    "Convergence AUC": {
        "lower": 0.08,
        "lower_end": 0.14,
        "upper_start": 0.16,
        "upper": 0.92,
    },
    "Total Run Time (min)": {
        "lower": 0.08,
        "lower_end": 0.30,
        "upper_start": 0.42,
        "upper": 0.92,
    },
}


@dataclass
class SeriesData:
    """Computed comparison series for one algorithm."""

    key: str
    label: str
    color: str
    best_bins: list[float]
    runtime_bins_s: list[float]
    peak_fitness: float
    throughput_jobs_per_s: float
    time_to_90_peak_s: float
    total_runtime_s: float
    total_evals: float
    convergence_auc: float
    source: str


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_generations(path: Path) -> list[dict[str, Any]]:
    data = _load_json(path)
    return list(data.get("generations", []))


def _validate_bins(bins: int) -> int:
    if bins <= 0:
        raise ValueError("bins must be > 0")
    return int(bins)


def resample_to_progress_bins(
    generations: list[dict[str, Any]],
    *,
    bins: int = DEFAULT_BINS,
) -> tuple[list[float], list[float], float]:
    """Map one run to fixed progress bins over cumulative runtime."""
    bins = _validate_bins(bins)
    if not generations:
        return [0.0] * bins, [0.0] * bins, 0.0

    cumulative_s: list[float] = []
    best_so_far: list[float] = []
    total_s = 0.0
    running_best = 0.0

    for row in generations:
        step_s = max(0.0, float(row.get("wall_clock_ms", 0.0)) / 1000.0)
        total_s += step_s
        cumulative_s.append(total_s)
        running_best = max(
            running_best,
            float(row.get("best_so_far", row.get("best_fitness", 0.0))),
        )
        best_so_far.append(running_best)

    if total_s <= 0.0:
        best = best_so_far[-1] if best_so_far else 0.0
        return [best] * bins, [0.0] * bins, 0.0

    best_bins: list[float] = []
    runtime_bins: list[float] = []
    for idx in range(1, bins + 1):
        target_s = (total_s * idx) / bins
        series_idx = bisect_left(cumulative_s, target_s)
        if series_idx >= len(cumulative_s):
            series_idx = len(cumulative_s) - 1
        best_bins.append(best_so_far[series_idx])
        runtime_bins.append(cumulative_s[series_idx])

    return best_bins, runtime_bins, total_s


def compute_convergence_auc(best_bins: list[float]) -> float:
    """Compute normalized AUC over progress [0,1] using best-so-far bins."""
    if not best_bins:
        return 0.0
    n = len(best_bins)
    x = np.linspace(0.0, 1.0, n + 1)
    y = np.array([best_bins[0], *best_bins], dtype=float)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(y, x=x))


def compute_time_to_peak_fraction(
    best_bins: list[float],
    runtime_bins_s: list[float],
    *,
    total_runtime_s: float,
    fraction: float = 0.9,
) -> float:
    """Return runtime when best-so-far first reaches fraction * peak."""
    if not best_bins or not runtime_bins_s:
        return float(total_runtime_s)
    peak = max(float(v) for v in best_bins)
    target = float(fraction) * peak
    for best, elapsed in zip(best_bins, runtime_bins_s, strict=True):
        if float(best) >= target:
            return float(elapsed)
    return float(total_runtime_s)


def _series_for_run(
    *,
    key: str,
    source: str,
    generations: list[dict[str, Any]],
    throughput_jobs_per_s: float | None = None,
    total_runtime_s: float | None = None,
    total_evals: float | None = None,
    peak_fitness: float | None = None,
    bins: int = DEFAULT_BINS,
) -> SeriesData:
    style = ALGO_STYLES[key]
    best_bins, runtime_bins, observed_runtime_s = resample_to_progress_bins(generations, bins=bins)
    evals = float(total_evals) if total_evals is not None else float(len(generations))
    runtime_s = float(total_runtime_s) if total_runtime_s is not None else observed_runtime_s
    if observed_runtime_s > 0 and runtime_s > 0 and not math.isclose(observed_runtime_s, runtime_s):
        scale = runtime_s / observed_runtime_s
        runtime_bins = [value * scale for value in runtime_bins]
    observed_rate = (evals / runtime_s) if runtime_s > 0 else 0.0
    time_to_90_peak_s = compute_time_to_peak_fraction(
        best_bins,
        runtime_bins,
        total_runtime_s=runtime_s,
        fraction=0.9,
    )

    return SeriesData(
        key=key,
        label=str(style["label"]),
        color=str(style["color"]),
        best_bins=best_bins,
        runtime_bins_s=runtime_bins,
        peak_fitness=float(
            peak_fitness if peak_fitness is not None else (best_bins[-1] if best_bins else 0.0)
        ),
        throughput_jobs_per_s=float(
            throughput_jobs_per_s if throughput_jobs_per_s is not None else observed_rate
        ),
        time_to_90_peak_s=time_to_90_peak_s,
        total_runtime_s=runtime_s,
        total_evals=evals,
        convergence_auc=compute_convergence_auc(best_bins),
        source=source,
    )


def aggregate_local_series(
    per_seed_dir: Path | str,
    *,
    bins: int = DEFAULT_BINS,
) -> dict[str, SeriesData]:
    """Build local series from per-seed JSON files.

    Fitness curves are averaged across seeds, while runtime/eval metrics are
    aggregated as totals across seeds to reflect full run cost.
    """
    per_seed_dir = Path(per_seed_dir)
    by_key: dict[str, list[SeriesData]] = {}
    for path in sorted(per_seed_dir.glob("*.json")):
        data = _load_json(path)
        run_name = str(data.get("run_name", ""))
        key = RUN_NAME_TO_KEY.get(run_name)
        if key is None:
            continue
        generations = list(data.get("generations", []))
        run_series = _series_for_run(
            key=key,
            source=f"local_seed:{path.name}",
            generations=generations,
            bins=bins,
        )
        by_key.setdefault(key, []).append(run_series)

    missing = [key for key in DISPLAY_ORDER if key != "epoch" and key not in by_key]
    if missing:
        raise ValueError(f"Missing local strategy files for keys: {missing}")

    aggregated: dict[str, SeriesData] = {}
    for key, runs in by_key.items():
        style = ALGO_STYLES[key]
        best_bins = [_mean([r.best_bins[i] for r in runs]) for i in range(bins)]
        runtime_bins = [sum(r.runtime_bins_s[i] for r in runs) for i in range(bins)]
        total_runtime_s = sum(r.total_runtime_s for r in runs)
        total_evals = sum(r.total_evals for r in runs)
        throughput_jobs_per_s = (total_evals / total_runtime_s) if total_runtime_s > 0 else 0.0
        time_to_90_peak_s = compute_time_to_peak_fraction(
            best_bins,
            runtime_bins,
            total_runtime_s=total_runtime_s,
            fraction=0.9,
        )
        aggregated[key] = SeriesData(
            key=key,
            label=str(style["label"]),
            color=str(style["color"]),
            best_bins=best_bins,
            runtime_bins_s=runtime_bins,
            peak_fitness=_mean([r.peak_fitness for r in runs]),
            throughput_jobs_per_s=throughput_jobs_per_s,
            time_to_90_peak_s=time_to_90_peak_s,
            total_runtime_s=total_runtime_s,
            total_evals=total_evals,
            convergence_auc=_mean([r.convergence_auc for r in runs]),
            source=f"local_aggregate_{len(runs)}_seeds",
        )
    return aggregated


def build_epoch_series(
    modal_run_path: Path | str,
    modal_summary_path: Path | str,
    *,
    bins: int = DEFAULT_BINS,
) -> SeriesData:
    """Build Epoch reference series from Modal run JSON + summary JSON."""
    modal_run_path = Path(modal_run_path)
    modal_summary_path = Path(modal_summary_path)
    run_generations = _load_generations(modal_run_path)
    summary = _load_json(modal_summary_path)
    total_runtime_s = float(
        summary.get(
            "wall_clock_total_s",
            summary.get("wall_clock_s", 0.0),
        )
    )
    total_evals = float(
        summary.get("jobs_submitted", summary.get("jobs_total", len(run_generations)))
    )
    peak_fitness = float(summary.get("best_fitness_peak", 0.0))
    throughput = float(summary.get("jobs_per_sec", 0.0))
    return _series_for_run(
        key="epoch",
        source="modal_single_run",
        generations=run_generations,
        throughput_jobs_per_s=throughput,
        total_runtime_s=total_runtime_s if total_runtime_s > 0 else None,
        total_evals=total_evals,
        peak_fitness=peak_fitness,
        bins=bins,
    )


def build_comparison_series(
    *,
    local_per_seed_dir: Path | str,
    modal_run_path: Path | str,
    modal_summary_path: Path | str,
    bins: int = DEFAULT_BINS,
) -> list[SeriesData]:
    """Return ordered comparison series (Epoch + local means)."""
    bins = _validate_bins(bins)
    local = aggregate_local_series(local_per_seed_dir, bins=bins)
    epoch = build_epoch_series(modal_run_path, modal_summary_path, bins=bins)

    by_key = {"epoch": epoch, **local}
    return [by_key[key] for key in DISPLAY_ORDER]


def _full_range(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    v_min = min(values)
    v_max = max(values)
    if math.isclose(v_min, v_max):
        return v_min - 0.1, v_max + 0.1
    pad = (v_max - v_min) * 0.08
    return v_min - pad, v_max + pad


def _fitness_zoom_range(series: list[SeriesData]) -> tuple[float, float]:
    finals = sorted((s.best_bins[-1] for s in series), reverse=True)
    focus = finals[:3] if len(finals) >= 3 else finals
    low = max(0.0, min(focus) - 0.03)
    high = min(1.0, max(focus) + 0.01)
    if high - low < 0.05:
        low = max(0.0, high - 0.05)
    return low, high


def _runtime_zoom_range(series: list[SeriesData]) -> tuple[float, float]:
    finals = sorted((s.runtime_bins_s[-1] for s in series))
    focus = finals[:3] if len(finals) >= 3 else finals
    low = max(0.0, min(focus) - 3.0)
    high = max(focus) + 7.0
    if high - low < 12.0:
        high = low + 12.0
    return low, high


def _plot_line_chart(
    *,
    series: list[SeriesData],
    y_selector: str,
    ylabel: str,
    title: str,
    zoom_range: tuple[float, float] | None,
    zoom_note: str | None,
    output_path: Path,
    show_inset: bool = True,
    annotate_epoch_scores: bool = False,
    force_zero_baseline: bool = False,
) -> None:
    x = list(range(1, len(series[0].best_bins) + 1))
    fig, ax = plt.subplots(figsize=(12, 7))

    all_values: list[float] = []
    for row in series:
        y = row.best_bins if y_selector == "best" else row.runtime_bins_s
        all_values.extend(y)
        linewidth = 3.2 if row.key == "epoch" else 2.2
        marker_size = 6.0 if row.key == "epoch" else 4.5
        zorder = 5 if row.key == "epoch" else 3
        ax.plot(
            x,
            y,
            label=row.label,
            color=row.color,
            linewidth=linewidth,
            marker="o",
            markersize=marker_size,
            zorder=zorder,
        )

    ax.set_xlim(1, len(x))
    ax.set_xticks(x)
    ax.set_xlabel("Generation-Equivalent Bin")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.28)
    if zoom_range is not None:
        ax.set_ylim(zoom_range[0], zoom_range[1])
    elif force_zero_baseline:
        high = max(all_values) if all_values else 1.0
        ax.set_ylim(0.0, max(1.0, high * 1.05))
    else:
        low, high = _full_range(all_values)
        ax.set_ylim(low, high)

    if zoom_note:
        ax.text(
            0.01,
            0.01,
            zoom_note,
            transform=ax.transAxes,
            fontsize=9,
            color="#334155",
        )

    if annotate_epoch_scores:
        epoch_row = next((row for row in series if row.key == "epoch"), None)
        if epoch_row is not None:
            y_epoch = epoch_row.best_bins if y_selector == "best" else epoch_row.runtime_bins_s
            for xpos, ypos in zip(x, y_epoch, strict=True):
                ax.annotate(
                    f"{ypos:.3f}",
                    xy=(xpos, ypos),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=epoch_row.color,
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8},
                )
    ax.legend(loc="upper left", fontsize=9)

    if show_inset:
        full_low, full_high = _full_range(all_values)
        inset = ax.inset_axes([0.58, 0.10, 0.39, 0.36])
        for row in series:
            y = row.best_bins if y_selector == "best" else row.runtime_bins_s
            inset.plot(
                x,
                y,
                color=row.color,
                linewidth=2.2 if row.key == "epoch" else 1.3,
                alpha=0.95,
            )
        inset.set_title("Full Scale", fontsize=8)
        inset.set_xlim(1, len(x))
        inset.set_ylim(full_low, full_high)
        inset.set_xticks([1, 4, 8, 12])
        inset.tick_params(labelsize=8)
        inset.grid(alpha=0.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _normalize_clip01(values: list[float]) -> list[float]:
    return [min(1.0, max(0.0, float(v))) for v in values]


def _normalize_max_ratio(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    if max_value <= 0:
        return [0.0 for _ in values]
    return [float(v) / max_value for v in values]


def _normalize_inverse_min_ratio(values: list[float]) -> list[float]:
    if not values:
        return []
    positive_values = [v for v in values if v > 0]
    if not positive_values:
        return [0.0 for _ in values]
    min_value = min(positive_values)
    return [min_value / float(v) if v > 0 else 0.0 for v in values]


def _apply_display_margin(
    values: list[float],
    *,
    lower: float = 0.08,
    upper: float = 0.92,
) -> list[float]:
    span = float(upper) - float(lower)
    return [float(lower) + (span * float(v)) for v in values]


def _apply_zoomed_axis_display(
    values: list[float],
    *,
    lower: float = 0.08,
    lower_end: float = 0.30,
    upper_start: float = 0.42,
    upper: float = 0.92,
) -> list[float]:
    """Expand the top cluster on a spoke while compressing the lower range."""
    if not values:
        return []

    unique_values = sorted(set(float(value) for value in values))
    if len(unique_values) < 3:
        return _apply_display_margin(values, lower=lower, upper=upper)

    gaps = [right - left for left, right in zip(unique_values, unique_values[1:])]
    max_gap = max(gaps, default=0.0)
    if max_gap <= 1e-6:
        return _apply_display_margin(values, lower=lower, upper=upper)

    gap_index = gaps.index(max_gap)
    pivot = unique_values[gap_index + 1]
    low = unique_values[0]
    high = unique_values[-1]

    if math.isclose(low, high):
        return _apply_display_margin(values, lower=lower, upper=upper)

    compressed_span = lower_end - lower
    expanded_span = upper - upper_start
    display_values: list[float] = []
    for value in values:
        numeric = float(value)
        if numeric < pivot:
            if math.isclose(pivot, low):
                mapped = lower
            else:
                mapped = lower + (((numeric - low) / (pivot - low)) * compressed_span)
        else:
            if math.isclose(high, pivot):
                mapped = upper
            else:
                mapped = upper_start + (((numeric - pivot) / (high - pivot)) * expanded_span)
        display_values.append(mapped)
    return display_values


def compute_radar_metrics(
    series: list[SeriesData],
) -> tuple[list[str], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Return radar axis names, raw metrics, and normalized metrics."""
    axis_names = [axis for axis, _, _, _ in RADAR_AXES]
    raw: dict[str, dict[str, float]] = {row.key: {} for row in series}
    normalized: dict[str, dict[str, float]] = {row.key: {} for row in series}

    for axis_name, field_name, norm_mode, unit_scale in RADAR_AXES:
        values = [float(getattr(row, field_name)) * float(unit_scale) for row in series]
        if norm_mode == "clip01":
            normalized_values = _normalize_clip01(values)
        elif norm_mode == "max_ratio":
            normalized_values = _normalize_max_ratio(values)
        elif norm_mode == "inverse_min_ratio":
            normalized_values = _normalize_inverse_min_ratio(values)
        else:
            raise ValueError(f"Unsupported radar normalization mode: {norm_mode}")
        for idx, row in enumerate(series):
            raw[row.key][axis_name] = values[idx]
            normalized[row.key][axis_name] = normalized_values[idx]

    return axis_names, raw, normalized


def _plot_radar_chart(
    series: list[SeriesData], output_path: Path
) -> tuple[list[str], dict[str, dict[str, float]]]:
    axis_names, raw, normalized = compute_radar_metrics(series)

    angles = np.linspace(0, 2 * np.pi, len(axis_names), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, polar=True)
    ax.set_position([0.04, 0.05, 0.80, 0.88])
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_names, fontsize=9)
    ax.tick_params(axis="x", pad=14)
    ax.yaxis.grid(True, color="#334155", alpha=0.55, linewidth=1.0)
    ax.xaxis.grid(True, color="#475569", alpha=0.45, linewidth=0.9)
    ring_ticks = np.arange(0.1, 1.01, 0.1)
    ax.set_yticks(ring_ticks)
    ax.set_yticklabels(
        [f"{tick:.1f}" if int(round(tick * 10)) % 2 == 0 else "" for tick in ring_ticks],
        fontsize=8,
    )

    axis_display_values: dict[str, list[float]] = {}
    for axis_name in axis_names:
        axis_values = [normalized[row.key][axis_name] for row in series]
        if axis_name in RADAR_ZOOMED_AXES:
            axis_display_values[axis_name] = _apply_zoomed_axis_display(
                axis_values,
                **RADAR_ZOOM_SETTINGS.get(axis_name, {}),
            )
        else:
            axis_display_values[axis_name] = _apply_display_margin(axis_values)

    display_values_map: dict[str, list[float]] = {row.key: [] for row in series}
    for axis_name in axis_names:
        for idx, row in enumerate(series):
            display_values_map[row.key].append(axis_display_values[axis_name][idx])

    for row in series:
        values = display_values_map[row.key]
        closed_values = [*values, values[0]]
        linewidth = 3.0 if row.key == "epoch" else 1.8
        line_alpha = 0.95 if row.key == "epoch" else 0.78
        ax.plot(
            angles,
            closed_values,
            color=row.color,
            linewidth=linewidth,
            alpha=line_alpha,
            label=row.label,
        )
        if row.key == "epoch":
            ax.fill(angles, closed_values, color=row.color, alpha=0.05)

    ax.set_title("Benchmark Radar Comparison", pad=24)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.00),
        fontsize=10,
        framealpha=0.95,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return axis_names, normalized


def write_visualization_summary(
    *,
    series: list[SeriesData],
    output_path: Path | str,
    chart_paths: dict[str, Path],
    bins: int,
    local_per_seed_dir: Path,
    modal_run_path: Path,
    modal_summary_path: Path,
) -> None:
    """Write markdown summary for generated chart artifacts and metrics."""
    axis_names, raw_metrics, _ = compute_radar_metrics(series)
    lines = [
        "# Benchmark Visualization Summary",
        "",
        "## Inputs",
        "",
        f"- Local per-seed metrics: `{local_per_seed_dir}`",
        f"- Modal GA run metrics: `{modal_run_path}`",
        f"- Modal GA summary: `{modal_summary_path}`",
        "",
        "## Outputs",
        "",
        f"- Fitness line chart: `{chart_paths['fitness']}`",
        f"- Runtime line chart: `{chart_paths['runtime']}`",
        f"- Radar chart: `{chart_paths['radar']}`",
        "",
        "## Method",
        "",
        f"- Progress alignment uses `{bins}` generation-equivalent runtime bins per run.",
        "- Local methods use mean fitness over seeds 42/43/44.",
        "- Runtime and eval metrics for local methods are totals across all 3 seeds.",
        "- Epoch is a single Modal reference run.",
        "- Fitness chart uses a zoomed primary axis plus full-scale inset.",
        "- Runtime chart uses full y-axis scale (not truncated).",
        "- Radar uses direct raw/max scaling with display margin [0.08, 0.92].",
        "- Radar expands crowded spokes for `Peak Fitness`, `Total Run Time`, and "
        "`Convergence AUC`.",
        "",
        "## Radar Raw Metrics",
        "",
        "| Algorithm | Peak fitness | Time to 90% Peak (min) | Total Run Time (min) | "
        "Total evals | Convergence AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in series:
        raw = raw_metrics[row.key]
        lines.append(
            f"| {row.label} | {raw['Peak Fitness']:.6f} | "
            f"{raw['Time to 90% Peak (min)']:.3f} | "
            f"{raw['Total Run Time (min)']:.3f} | {raw['Total Evals Completed']:.0f} | "
            f"{raw['Convergence AUC']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Failure/timeout axis is omitted because local per-seed JSONs do not include "
            "explicit status counts.",
            "- Local total runtime and total evals are aggregate values across 3 seed runs.",
            "- `Time to 90% Peak` falls back to total runtime if threshold is never reached.",
            "- Radar omits on-chart raw metric callouts to reduce clutter.",
            "- `Peak Fitness`, `Total Run Time`, and `Convergence AUC` use spoke-specific zoom to "
            "separate close scores while preserving rank order.",
            "",
            "## Optional Follow-up",
            "",
            "- Instrument local benchmark outputs with completed/timeout/failed counts, then "
            "add that as radar axis.",
            "",
            "## Radar Axes",
            "",
            f"- {', '.join(axis_names)}",
            "",
        ]
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_visualizations(
    *,
    local_per_seed_dir: Path | str = DEFAULT_LOCAL_PER_SEED_DIR,
    modal_run_path: Path | str = DEFAULT_MODAL_RUN_PATH,
    modal_summary_path: Path | str = DEFAULT_MODAL_SUMMARY_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    bins: int = DEFAULT_BINS,
) -> dict[str, Any]:
    """Generate line/radar charts and markdown synthesis summary."""
    local_per_seed_dir = Path(local_per_seed_dir)
    modal_run_path = Path(modal_run_path)
    modal_summary_path = Path(modal_summary_path)
    output_dir = Path(output_dir)
    bins = _validate_bins(bins)

    series = build_comparison_series(
        local_per_seed_dir=local_per_seed_dir,
        modal_run_path=modal_run_path,
        modal_summary_path=modal_summary_path,
        bins=bins,
    )

    fitness_path = output_dir / "fitness_vs_generation_bins.png"
    runtime_path = output_dir / "runtime_vs_generation_bins.png"
    radar_path = output_dir / "radar_comparison.png"
    summary_path = output_dir / "visualization_summary.md"

    _plot_line_chart(
        series=series,
        y_selector="best",
        ylabel="Best-So-Far Peak Fitness",
        title="Peak Fitness vs Generation-Equivalent Progress",
        zoom_range=_fitness_zoom_range(series),
        zoom_note="Primary axis zoomed to top-performing range; see inset for full scale.",
        output_path=fitness_path,
        show_inset=True,
        annotate_epoch_scores=True,
        force_zero_baseline=False,
    )
    _plot_line_chart(
        series=series,
        y_selector="runtime",
        ylabel="Total Run Time (s)",
        title="Total Run Time vs Generation-Equivalent Bins",
        zoom_range=None,
        zoom_note=None,
        output_path=runtime_path,
        show_inset=False,
        annotate_epoch_scores=False,
        force_zero_baseline=True,
    )
    _plot_radar_chart(series, radar_path)
    write_visualization_summary(
        series=series,
        output_path=summary_path,
        chart_paths={"fitness": fitness_path, "runtime": runtime_path, "radar": radar_path},
        bins=bins,
        local_per_seed_dir=local_per_seed_dir,
        modal_run_path=modal_run_path,
        modal_summary_path=modal_summary_path,
    )

    return {
        "series": series,
        "fitness_chart": fitness_path,
        "runtime_chart": runtime_path,
        "radar_chart": radar_path,
        "summary": summary_path,
    }
