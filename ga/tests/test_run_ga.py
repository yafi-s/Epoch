"""Focused tests for run_ga summary throughput and modal teardown fields."""

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


def _base_kpi(*, wall_clock_ms: int = 3786, plateau_reheat_applied: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        generation=0,
        jobs_count=6,
        completed=6,
        failed=0,
        timed_out=0,
        active_workers=8,
        wall_clock_ms=wall_clock_ms,
        train_total_ms=3000,
        train_p50_ms=500,
        train_p90_ms=700,
        ideal_wall_ms=375.0,
        queue_overhead_ms=max(float(wall_clock_ms) - 375.0, 0.0),
        queue_overhead_pct=(max(float(wall_clock_ms) - 375.0, 0.0) / float(wall_clock_ms)),
        gen_jobs_per_sec=1.58,
        plateau_reheat_applied=plateau_reheat_applied,
        dispatch_latency_p50_ms=2.0,
        dispatch_latency_p90_ms=4.0,
        dispatch_latency_max_ms=6.0,
        worker_idle_gap_p50_ms=3.0,
        worker_idle_gap_p90_ms=6.0,
        worker_idle_gap_max_ms=9.0,
        queue_wait_p50_ms=5.0,
        queue_wait_p90_ms=7.0,
        queue_wait_max_ms=10.0,
        queue_wait_min_ms=2.0,
        dispatch_samples=6,
        idle_gap_samples=6,
        queue_wait_samples=6,
    )


def _make_run_stats(
    *,
    jobs_submitted: int = 6,
    total_results: int = 6,
    jobs_planned: int = 6,
    first_generation_submit_offset_s: float = 0.0,
    first_dispatch_offset_s: float = 0.0,
    generation_kpis: list[SimpleNamespace] | None = None,
    plateau_events_count: int = 0,
    reheat_generations_applied: int = 0,
    stale_generations_final: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        jobs_submitted=jobs_submitted,
        total_results=total_results,
        completed=total_results,
        timed_out=0,
        failed=0,
        unknown=0,
        jobs_planned=jobs_planned,
        generations_planned=1,
        generations_completed=1,
        stopped_early=False,
        stop_reason="completed",
        first_generation_submit_offset_s=first_generation_submit_offset_s,
        first_dispatch_offset_s=first_dispatch_offset_s,
        plateau_events_count=plateau_events_count,
        reheat_generations_applied=reheat_generations_applied,
        stale_generations_final=stale_generations_final,
        generation_kpis=generation_kpis if generation_kpis is not None else [_base_kpi()],
    )


def _patch_controller_stack(
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_stats: SimpleNamespace,
    final_pop: object | None = None,
    run_exception: Exception | None = None,
    captured: dict[str, int] | None = None,
) -> None:
    class FakeMetricsStore:
        def __init__(self, run_name: str) -> None:
            self.run_name = run_name
            self.generations = [
                SimpleNamespace(best_fitness=0.9, best_so_far=0.9, wall_clock_ms=1000),
            ]
            self.convergence_rate = 0.0

        def to_json(self, path: str) -> None:
            Path(path).write_text("{}", encoding="utf-8")

    class FakeController:
        def __init__(self, config: object, store: object, search_space: object | None = None) -> None:
            if captured is not None:
                captured["population_size"] = config.ga_config.population_size
            self.run_stats = run_stats

        def run(self) -> object:
            if run_exception is not None:
                raise run_exception
            if final_pop is not None:
                return final_pop
            return SimpleNamespace(best_fitness=0.91, avg_fitness=0.85, worst_fitness=0.70)

    monkeypatch.setattr(controller_mod, "GAController", FakeController)
    monkeypatch.setattr(collector_mod, "MetricsStore", FakeMetricsStore)
    monkeypatch.setattr(
        search_space_mod.SearchSpace, "stress_test_cnn_space", staticmethod(lambda: object())
    )


def test_wall_adjusted_throughput_formula_matches_expected_values() -> None:
    avg_raw, avg_adjusted, jobs_per_sec = run_ga._compute_wall_adjusted_throughput(
        wall_clock_s=3.786,
        jobs_submitted=6,
        network_latency_ms=150.0,
    )
    assert avg_raw == pytest.approx(631.0, abs=1e-6)
    assert avg_adjusted == pytest.approx(481.0, abs=1e-6)
    assert jobs_per_sec == pytest.approx(6.0 / (3.786 - 0.9), rel=1e-6)


def test_legacy_throughput_formula_matches_expected_values() -> None:
    avg_adjusted, jobs_per_sec = run_ga._compute_legacy_throughput(
        avg_job_ms_raw=631.0,
        throughput_worker_count=8,
        network_latency_ms=150.0,
    )
    assert avg_adjusted == pytest.approx(481.0, abs=1e-6)
    assert jobs_per_sec == pytest.approx((8.0 * 1000.0) / 481.0, rel=1e-6)


def test_summary_contains_legacy_and_wall_adjusted_throughput_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_controller_stack(
        monkeypatch,
        run_stats=_make_run_stats(generation_kpis=[_base_kpi(wall_clock_ms=3786)]),
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
            "--no-modal-auto-stop",
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
    assert summary["jobs_per_sec"] == pytest.approx((8.0 * 1000.0) / 481.0, rel=1e-6)
    assert summary["jobs_per_sec_total"] == pytest.approx((8.0 * 1000.0) / 481.0, rel=1e-6)
    assert summary["jobs_per_sec_wall_adjusted"] == pytest.approx(6.0 / (3.786 - 0.9), rel=1e-6)
    assert summary["jobs_per_sec_wall_adjusted_total"] == pytest.approx(
        6.0 / (3.786 - 0.9), rel=1e-6
    )
    assert summary["raw_jobs_per_sec"] == pytest.approx(6.0 / 3.786, rel=1e-6)
    assert summary["raw_jobs_per_sec_total"] == pytest.approx(6.0 / 3.786, rel=1e-6)
    assert summary["wall_clock_s"] == pytest.approx(3.786, rel=1e-6)
    assert summary["wall_clock_total_s"] == pytest.approx(3.786, rel=1e-6)
    assert summary["metric_clock_source"] == "first_dispatch"
    assert summary["metric_clock_fallback_to_total"] is True
    assert "jobs_per_sec_formula" in summary
    assert "jobs_per_sec_wall_adjusted_formula" in summary
    assert summary["modal_auto_stop_enabled"] is False
    assert summary["modal_auto_stop_attempted"] is False
    assert summary["modal_auto_stop_success"] is False
    assert summary["plateau_events_count"] == 0
    assert summary["reheat_generations_applied"] == 0
    assert summary["stale_generations_final"] == 0
    assert summary["generation_kpis"][0]["plateau_reheat_applied"] is False


def test_pop_per_worker_derives_population_size_and_default_worker_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, int] = {}
    _patch_controller_stack(
        monkeypatch,
        run_stats=_make_run_stats(jobs_submitted=20, total_results=20, jobs_planned=20, generation_kpis=[]),
        captured=captured,
    )

    monotonic_values = iter([0.0, 1.0])
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
            "--expected-workers",
            "10",
            "--pop-per-worker",
            "2",
            "--no-modal-auto-stop",
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
        ],
    )

    run_ga.main()

    assert captured["population_size"] == 20
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["population_size"] == 20
    assert summary["requested_population_size"] == 6
    assert summary["pop_per_worker"] == 2
    assert summary["throughput_worker_count"] == 10


def test_summary_includes_plateau_observability_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_controller_stack(
        monkeypatch,
        run_stats=_make_run_stats(
            generation_kpis=[_base_kpi(wall_clock_ms=1000, plateau_reheat_applied=True)],
            plateau_events_count=2,
            reheat_generations_applied=3,
            stale_generations_final=1,
        ),
    )

    monotonic_values = iter([0.0, 1.0])
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
            "--no-modal-auto-stop",
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
        ],
    )

    run_ga.main()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["plateau_events_count"] == 2
    assert summary["reheat_generations_applied"] == 3
    assert summary["stale_generations_final"] == 1
    assert summary["generation_kpis"][0]["plateau_reheat_applied"] is True


def test_first_dispatch_clock_source_uses_offset_when_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    kpi = _base_kpi(wall_clock_ms=1000)
    _patch_controller_stack(
        monkeypatch,
        run_stats=_make_run_stats(
            first_generation_submit_offset_s=0.2,
            first_dispatch_offset_s=0.5,
            generation_kpis=[kpi],
        ),
    )

    monotonic_values = iter([0.0, 10.0])
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
            "--metric-clock-source",
            "first_dispatch",
            "--network-latency-ms",
            "150",
            "--no-modal-auto-stop",
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
        ],
    )

    run_ga.main()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["metric_clock_fallback_to_total"] is False
    assert summary["wall_clock_total_s"] == pytest.approx(10.0, rel=1e-6)
    assert summary["wall_clock_s"] == pytest.approx(9.5, rel=1e-6)


def test_modal_auto_stop_calls_stop_helper_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_controller_stack(monkeypatch, run_stats=_make_run_stats())

    calls: list[tuple[str, str]] = []

    def _fake_stop_modal_app(*, modal_app_name: str, modal_env: str) -> tuple[bool, str | None]:
        calls.append((modal_app_name, modal_env))
        return True, None

    monkeypatch.setattr(run_ga, "_stop_modal_app", _fake_stop_modal_app)
    monotonic_values = iter([0.0, 1.0])
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
            "--modal-app-name",
            "epoch-workers-custom",
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
        ],
    )

    run_ga.main()

    assert calls == [("epoch-workers-custom", "")]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["modal_auto_stop_enabled"] is True
    assert summary["modal_app_name"] == "epoch-workers-custom"
    assert summary["modal_auto_stop_attempted"] is True
    assert summary["modal_auto_stop_success"] is True
    assert summary["modal_auto_stop_error"] is None


def test_modal_auto_stop_skipped_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_controller_stack(monkeypatch, run_stats=_make_run_stats())

    def _fail_if_called(*, modal_app_name: str, modal_env: str) -> tuple[bool, str | None]:
        raise AssertionError("modal stop helper should not be called when disabled")

    monkeypatch.setattr(run_ga, "_stop_modal_app", _fail_if_called)
    monotonic_values = iter([0.0, 1.0])
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
            "--no-modal-auto-stop",
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
        ],
    )

    run_ga.main()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["modal_auto_stop_enabled"] is False
    assert summary["modal_auto_stop_attempted"] is False
    assert summary["modal_auto_stop_success"] is False


def test_modal_auto_stop_failure_is_non_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_controller_stack(monkeypatch, run_stats=_make_run_stats())

    def _fake_stop_modal_app(*, modal_app_name: str, modal_env: str) -> tuple[bool, str | None]:
        return False, "failed-to-stop"

    monkeypatch.setattr(run_ga, "_stop_modal_app", _fake_stop_modal_app)
    monotonic_values = iter([0.0, 1.0])
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
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
        ],
    )

    run_ga.main()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["modal_auto_stop_attempted"] is True
    assert summary["modal_auto_stop_success"] is False
    assert summary["modal_auto_stop_error"] == "failed-to-stop"


def test_modal_auto_stop_runs_when_controller_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_controller_stack(
        monkeypatch,
        run_stats=_make_run_stats(),
        run_exception=RuntimeError("controller failed"),
    )

    calls: list[tuple[str, str]] = []

    def _fake_stop_modal_app(*, modal_app_name: str, modal_env: str) -> tuple[bool, str | None]:
        calls.append((modal_app_name, modal_env))
        return True, None

    monkeypatch.setattr(run_ga, "_stop_modal_app", _fake_stop_modal_app)
    monotonic_values = iter([0.0, 1.0])
    monkeypatch.setattr(run_ga.time, "monotonic", lambda: next(monotonic_values))

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
        ],
    )

    with pytest.raises(RuntimeError, match="controller failed"):
        run_ga.main()

    assert calls == [("epoch-workers", "")]
