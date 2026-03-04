"""Focused tests for run_ga summary throughput fields."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import ga.controller as controller_mod
import ga.search_space as search_space_mod
import metrics.collector as collector_mod
import run_ga


def test_adjusted_throughput_formula_matches_expected_values() -> None:
    avg_raw, avg_adjusted, jobs_per_sec = run_ga._compute_adjusted_throughput(
        wall_clock_s=3.786,
        jobs_submitted=6,
        throughput_worker_count=8,
        network_latency_ms=150.0,
    )
    assert avg_raw == pytest.approx(631.0, abs=1e-6)
    assert avg_adjusted == pytest.approx(481.0, abs=1e-6)
    assert jobs_per_sec == pytest.approx(16.631, rel=1e-3)


def test_summary_contains_adjusted_throughput_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeMetricsStore:
        def __init__(self, run_name: str) -> None:
            self.run_name = run_name
            self.generations = [
                SimpleNamespace(best_fitness=0.9, wall_clock_ms=1000),
            ]
            self.convergence_rate = 0.0

        def to_json(self, path: str) -> None:
            Path(path).write_text("{}", encoding="utf-8")

    class FakeController:
        def __init__(self, config: object, store: object, search_space: object | None = None) -> None:
            self.run_stats = SimpleNamespace(
                jobs_submitted=6,
                total_results=6,
                completed=6,
                timed_out=0,
                failed=0,
                unknown=0,
                jobs_planned=6,
                generations_planned=1,
                generations_completed=1,
                stopped_early=False,
                stop_reason="completed",
                generation_kpis=[
                    SimpleNamespace(
                        generation=0,
                        jobs_count=6,
                        completed=6,
                        failed=0,
                        timed_out=0,
                        active_workers=8,
                        wall_clock_ms=3786,
                        train_total_ms=3000,
                        train_p50_ms=500,
                        train_p90_ms=700,
                        ideal_wall_ms=375.0,
                        queue_overhead_ms=3411.0,
                        queue_overhead_pct=0.90,
                        gen_jobs_per_sec=1.58,
                        dispatch_latency_p50_ms=2.0,
                        dispatch_latency_p90_ms=4.0,
                        dispatch_latency_max_ms=6.0,
                        worker_idle_gap_p50_ms=3.0,
                        worker_idle_gap_p90_ms=6.0,
                        worker_idle_gap_max_ms=9.0,
                        queue_wait_p50_ms=5.0,
                        queue_wait_p90_ms=7.0,
                        queue_wait_max_ms=10.0,
                        dispatch_samples=6,
                        idle_gap_samples=6,
                        queue_wait_samples=6,
                    )
                ],
            )

        def run(self) -> object:
            return SimpleNamespace(best_fitness=0.91, avg_fitness=0.85, worst_fitness=0.70)

    monkeypatch.setattr(controller_mod, "GAController", FakeController)
    monkeypatch.setattr(collector_mod, "MetricsStore", FakeMetricsStore)
    monkeypatch.setattr(
        search_space_mod.SearchSpace, "stress_test_cnn_space", staticmethod(lambda: object())
    )

    monotonic_values = iter([0.0, 3.786])
    monkeypatch.setattr(run_ga.time, "monotonic", lambda: next(monotonic_values))

    output_path = tmp_path / "run.json"
    summary_path = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ga.py",
            "--mode",
            "stress",
            "--pop-size",
            "6",
            "--generations",
            "1",
            "--throughput-worker-count",
            "8",
            "--network-latency-ms",
            "150",
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
        ],
    )

    run_ga.main()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["avg_job_ms_raw"] == pytest.approx(631.0, abs=1e-6)
    assert summary["avg_job_ms_adjusted"] == pytest.approx(481.0, abs=1e-6)
    assert summary["network_latency_ms"] == pytest.approx(150.0, abs=1e-6)
    assert summary["throughput_worker_count"] == 8
    assert summary["jobs_per_sec"] == pytest.approx(16.631, rel=1e-3)
    assert "jobs_per_sec_formula" in summary

