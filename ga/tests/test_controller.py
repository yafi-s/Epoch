"""Tests for GAController KPI instrumentation and guard behavior."""

from __future__ import annotations

import sys

import pytest

import ga.controller as controller_mod
from ga.controller import ControllerConfig, GAController
from ga.engine import GAConfig
from ga.search_space import SearchSpace

sys.path.insert(0, "generated/python")
import epoch_pb2


class _FakeChannel:
    def close(self) -> None:
        return None


class _FakeStub:
    def __init__(
        self,
        responses: dict[int, epoch_pb2.GetResultsResponse],
        status_responses: list[epoch_pb2.SchedulerStatusResponse] | None = None,
    ) -> None:
        self._responses = responses
        self._status_responses = status_responses or [
            epoch_pb2.SchedulerStatusResponse(
                connected_workers=0,
                idle_workers=0,
                busy_workers=0,
                pending_jobs=0,
            )
        ]
        self.submit_requests: list[epoch_pb2.SubmitGenerationRequest] = []
        self.result_requests: list[epoch_pb2.GetResultsRequest] = []
        self.status_requests: list[epoch_pb2.SchedulerStatusRequest] = []

    def SubmitGeneration(
        self, request: epoch_pb2.SubmitGenerationRequest, timeout: float | None = None
    ) -> epoch_pb2.SubmitGenerationResponse:
        self.submit_requests.append(request)
        return epoch_pb2.SubmitGenerationResponse(accepted=True, num_jobs=len(request.configs))

    def GetGenerationResults(
        self, request: epoch_pb2.GetResultsRequest, timeout: float | None = None
    ) -> epoch_pb2.GetResultsResponse:
        self.result_requests.append(request)
        return self._responses[request.generation_id]

    def GetSchedulerStatus(
        self, request: epoch_pb2.SchedulerStatusRequest, timeout: float | None = None
    ) -> epoch_pb2.SchedulerStatusResponse:
        self.status_requests.append(request)
        if len(self.status_requests) >= len(self._status_responses):
            return self._status_responses[-1]
        return self._status_responses[len(self.status_requests) - 1]


def _result(
    job_id: str,
    worker_id: str,
    train_ms: int,
    status: int = epoch_pb2.JOB_STATUS_COMPLETED,
    acc: float = 0.5,
    generation_id: int = 0,
) -> epoch_pb2.TrainingResult:
    return epoch_pb2.TrainingResult(
        job_id=job_id,
        generation_id=generation_id,
        worker_id=worker_id,
        validation_accuracy=acc,
        training_loss=0.1,
        training_time_ms=train_ms,
        status=status,
        error_message="",
    )


def _runtime_metrics(
    *,
    dispatch_p50: float = 0.0,
    dispatch_p90: float = 0.0,
    dispatch_max: float = 0.0,
    idle_p50: float = 0.0,
    idle_p90: float = 0.0,
    idle_max: float = 0.0,
    queue_p50: float = 0.0,
    queue_p90: float = 0.0,
    queue_max: float = 0.0,
    queue_min: float = 0.0,
    dispatch_samples: int = 0,
    idle_samples: int = 0,
    queue_samples: int = 0,
) -> epoch_pb2.GenerationRuntimeMetrics:
    return epoch_pb2.GenerationRuntimeMetrics(
        dispatch_latency_p50_ms=dispatch_p50,
        dispatch_latency_p90_ms=dispatch_p90,
        dispatch_latency_max_ms=dispatch_max,
        worker_idle_gap_p50_ms=idle_p50,
        worker_idle_gap_p90_ms=idle_p90,
        worker_idle_gap_max_ms=idle_max,
        queue_wait_p50_ms=queue_p50,
        queue_wait_p90_ms=queue_p90,
        queue_wait_max_ms=queue_max,
        queue_wait_min_ms=queue_min,
        dispatch_samples=dispatch_samples,
        idle_gap_samples=idle_samples,
        queue_wait_samples=queue_samples,
    )


def _build_controller(
    monkeypatch: pytest.MonkeyPatch,
    responses: dict[int, epoch_pb2.GetResultsResponse],
    *,
    pop_size: int,
    generations: int,
    max_wall_clock_s: float = 0.0,
    expected_workers: int = 0,
    wait_for_idle_workers: int = 0,
    readiness_timeout_s: float = 5.0,
    status_responses: list[epoch_pb2.SchedulerStatusResponse] | None = None,
) -> tuple[GAController, _FakeStub]:
    stub = _FakeStub(responses, status_responses=status_responses)

    monkeypatch.setattr(controller_mod.grpc, "insecure_channel", lambda _addr: _FakeChannel())
    monkeypatch.setattr(controller_mod.epoch_pb2_grpc, "SchedulerControlStub", lambda _ch: stub)

    cfg = ControllerConfig(
        scheduler_address="fake:50051",
        ga_config=GAConfig(
            population_size=pop_size,
            num_generations=generations,
            dataset="mnist",
            epochs=1,
            seed=42,
        ),
        poll_interval=0.0,
        rpc_timeout_s=1.0,
        generation_timeout_s=5.0,
        progress_log_interval_s=1.0,
        max_wall_clock_s=max_wall_clock_s,
        expected_workers=expected_workers,
        wait_for_idle_workers=wait_for_idle_workers,
        readiness_timeout_s=readiness_timeout_s,
    )
    controller = GAController(cfg, search_space=SearchSpace.stress_test_cnn_space())
    return controller, stub


def test_generation_kpi_computation(monkeypatch: pytest.MonkeyPatch) -> None:
    results = [
        _result("job0", "w0", 100, acc=0.6),
        _result("job1", "w1", 200, acc=0.7),
        _result("job2", "w2", 300, acc=0.8),
        _result("job3", "w3", 400, acc=0.9),
        _result("job4", "w4", 500, acc=0.95),
    ]
    responses = {
        0: epoch_pb2.GetResultsResponse(
            complete=True,
            wall_clock_ms=450,
            results=results,
            runtime_metrics=_runtime_metrics(
                dispatch_p50=1.1,
                dispatch_p90=2.2,
                dispatch_max=4.4,
                idle_p50=6.0,
                idle_p90=9.0,
                idle_max=12.0,
                queue_p50=8.0,
                queue_p90=11.0,
                queue_max=13.0,
                queue_min=2.0,
                dispatch_samples=5,
                idle_samples=3,
                queue_samples=5,
            ),
        ),
    }
    controller, _stub = _build_controller(
        monkeypatch,
        responses,
        pop_size=5,
        generations=1,
    )

    controller.run()
    stats = controller.run_stats

    assert stats.jobs_submitted == 5
    assert stats.total_results == 5
    assert len(stats.generation_kpis) == 1

    kpi = stats.generation_kpis[0]
    assert kpi.jobs_count == 5
    assert kpi.completed == 5
    assert kpi.failed == 0
    assert kpi.timed_out == 0
    assert kpi.active_workers == 5
    assert kpi.train_total_ms == 1500
    assert kpi.train_p50_ms == 300
    assert kpi.train_p90_ms == 500
    assert kpi.ideal_wall_ms == pytest.approx(300.0)
    assert kpi.queue_overhead_ms == pytest.approx(150.0)
    assert kpi.queue_overhead_pct == pytest.approx(150.0 / 450.0)
    assert kpi.gen_jobs_per_sec == pytest.approx(5 / 0.45)
    assert kpi.dispatch_latency_p50_ms == pytest.approx(1.1)
    assert kpi.dispatch_latency_p90_ms == pytest.approx(2.2)
    assert kpi.dispatch_latency_max_ms == pytest.approx(4.4)
    assert kpi.worker_idle_gap_p50_ms == pytest.approx(6.0)
    assert kpi.worker_idle_gap_p90_ms == pytest.approx(9.0)
    assert kpi.worker_idle_gap_max_ms == pytest.approx(12.0)
    assert kpi.queue_wait_p50_ms == pytest.approx(8.0)
    assert kpi.queue_wait_p90_ms == pytest.approx(11.0)
    assert kpi.queue_wait_max_ms == pytest.approx(13.0)
    assert kpi.queue_wait_min_ms == pytest.approx(2.0)
    assert kpi.dispatch_samples == 5
    assert kpi.idle_gap_samples == 3
    assert kpi.queue_wait_samples == 5


def test_queue_overhead_clamped_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    results = [
        _result("job0", "w0", 100),
        _result("job1", "w1", 300),
    ]
    responses = {
        0: epoch_pb2.GetResultsResponse(complete=True, wall_clock_ms=150, results=results),
    }
    controller, _stub = _build_controller(
        monkeypatch,
        responses,
        pop_size=2,
        generations=1,
    )

    controller.run()
    kpi = controller.run_stats.generation_kpis[0]

    # Ideal = (100 + 300) / 2 = 200ms, so overhead should clamp to zero.
    assert kpi.ideal_wall_ms == pytest.approx(200.0)
    assert kpi.queue_overhead_ms == pytest.approx(0.0)
    assert kpi.queue_overhead_pct == pytest.approx(0.0)


def test_budget_guard_stops_before_next_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = {
        0: epoch_pb2.GetResultsResponse(
            complete=True,
            wall_clock_ms=1000,
            results=[
                _result("job0", "w0", 500),
                _result("job1", "w1", 500),
                _result("job2", "w2", 500),
                _result("job3", "w3", 500),
            ],
        ),
        1: epoch_pb2.GetResultsResponse(complete=True, wall_clock_ms=1000, results=[]),
    }
    controller, stub = _build_controller(
        monkeypatch,
        responses,
        pop_size=4,
        generations=3,
        max_wall_clock_s=0.2,
    )

    controller.run()
    stats = controller.run_stats

    assert stats.stopped_early is True
    assert stats.stop_reason == "max_wall_clock_reached"
    assert stats.generations_completed == 1
    assert stats.jobs_submitted == 4
    assert len(stub.submit_requests) == 1


def test_expected_workers_guard_stops_after_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = {
        0: epoch_pb2.GetResultsResponse(
            complete=True,
            wall_clock_ms=300,
            results=[
                _result("job0", "w0", 150),
                _result("job1", "w1", 150),
                _result("job2", "w1", 150),
            ],
        ),
        1: epoch_pb2.GetResultsResponse(complete=True, wall_clock_ms=300, results=[]),
    }
    controller, stub = _build_controller(
        monkeypatch,
        responses,
        pop_size=3,
        generations=4,
        expected_workers=8,
    )

    controller.run()
    stats = controller.run_stats

    assert stats.stopped_early is True
    assert stats.stop_reason == "insufficient_workers"
    assert stats.generations_completed == 1
    assert stats.jobs_submitted == 3
    assert len(stub.submit_requests) == 1


def test_run_stats_preserves_status_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = {
        0: epoch_pb2.GetResultsResponse(
            complete=True,
            wall_clock_ms=350,
            results=[
                _result("job0", "w0", 120, status=epoch_pb2.JOB_STATUS_COMPLETED),
                _result("job1", "w1", 120, status=epoch_pb2.JOB_STATUS_TIMEOUT),
                _result("job2", "w2", 120, status=epoch_pb2.JOB_STATUS_FAILED),
            ],
        )
    }
    controller, _stub = _build_controller(
        monkeypatch,
        responses,
        pop_size=3,
        generations=1,
    )

    controller.run()
    stats = controller.run_stats

    assert stats.total_results == 3
    assert stats.completed == 1
    assert stats.timed_out == 1
    assert stats.failed == 1
    assert stats.unknown == 0


def test_smoke_run_pop6_gen1_populates_kpis(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = {
        0: epoch_pb2.GetResultsResponse(
            complete=True,
            wall_clock_ms=420,
            results=[
                _result("job0", "w0", 140, acc=0.60),
                _result("job1", "w1", 140, acc=0.61),
                _result("job2", "w2", 140, acc=0.62),
                _result("job3", "w0", 140, acc=0.63),
                _result("job4", "w1", 140, acc=0.64),
                _result("job5", "w2", 140, acc=0.65),
            ],
        )
    }
    controller, _stub = _build_controller(
        monkeypatch,
        responses,
        pop_size=6,
        generations=1,
    )

    controller.run()
    stats = controller.run_stats
    assert stats.jobs_submitted == 6
    assert stats.jobs_planned == 6
    assert stats.generations_completed == 1
    assert len(stats.generation_kpis) == 1

    kpi = stats.generation_kpis[0]
    assert kpi.generation == 0
    assert kpi.jobs_count == 6
    assert kpi.active_workers == 3
    assert kpi.train_total_ms == 840
    assert kpi.wall_clock_ms == 420


def test_plateau_reheat_persists_for_configured_generations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = {}
    for gen in range(6):
        responses[gen] = epoch_pb2.GetResultsResponse(
            complete=True,
            wall_clock_ms=300,
            results=[
                _result(f"job{gen}_0", "w0", 100, acc=0.8010, generation_id=gen),
                _result(f"job{gen}_1", "w1", 100, acc=0.8000, generation_id=gen),
                _result(f"job{gen}_2", "w2", 100, acc=0.7900, generation_id=gen),
                _result(f"job{gen}_3", "w3", 100, acc=0.7800, generation_id=gen),
            ],
        )

    controller, _stub = _build_controller(
        monkeypatch,
        responses,
        pop_size=4,
        generations=6,
    )
    controller.config.ga_config.plateau_min_delta = 0.01
    controller.config.ga_config.plateau_patience_gens = 3
    controller.config.ga_config.plateau_immigrant_rate = 0.25
    controller.config.ga_config.plateau_mutation_rate_floor = 0.4
    controller.config.ga_config.plateau_reheat_gens = 2

    overrides: list[tuple[float | None, float | None, bool]] = []
    original_evolve = controller._engine.evolve

    def _wrapped_evolve(
        population,
        immigrant_rate_override=None,
        mutation_rate_floor_override=None,
        reexpand_constraints=False,
    ):
        overrides.append(
            (
                immigrant_rate_override,
                mutation_rate_floor_override,
                reexpand_constraints,
            )
        )
        return original_evolve(
            population,
            immigrant_rate_override=immigrant_rate_override,
            mutation_rate_floor_override=mutation_rate_floor_override,
            reexpand_constraints=reexpand_constraints,
        )

    controller._engine.evolve = _wrapped_evolve
    controller.run()

    assert len(overrides) == 5
    assert overrides[0] == (None, None, False)
    assert overrides[1] == (None, None, False)
    assert overrides[2] == (None, None, False)
    assert overrides[3] == (pytest.approx(0.25), pytest.approx(0.4), True)
    assert overrides[4] == (pytest.approx(0.25), pytest.approx(0.4), True)
    stats = controller.run_stats
    assert stats.plateau_events_count == 1
    assert stats.reheat_generations_applied == 2
    assert stats.stale_generations_final == 2
    assert [kpi.plateau_reheat_applied for kpi in stats.generation_kpis].count(True) == 2


def test_plateau_resets_on_any_new_best_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = {}
    best_per_generation = [0.8000, 0.8005, 0.8008, 0.8010]
    for gen, best in enumerate(best_per_generation):
        responses[gen] = epoch_pb2.GetResultsResponse(
            complete=True,
            wall_clock_ms=300,
            results=[
                _result(f"job{gen}_0", "w0", 100, acc=best, generation_id=gen),
                _result(f"job{gen}_1", "w1", 100, acc=max(best - 0.01, 0.0), generation_id=gen),
                _result(f"job{gen}_2", "w2", 100, acc=max(best - 0.02, 0.0), generation_id=gen),
                _result(f"job{gen}_3", "w3", 100, acc=max(best - 0.03, 0.0), generation_id=gen),
            ],
        )

    controller, _stub = _build_controller(
        monkeypatch,
        responses,
        pop_size=4,
        generations=4,
    )
    controller.config.ga_config.plateau_min_delta = 0.01
    controller.config.ga_config.plateau_patience_gens = 1
    controller.config.ga_config.plateau_reset_on_any_new_best = True

    overrides: list[tuple[float | None, float | None, bool]] = []
    original_evolve = controller._engine.evolve

    def _wrapped_evolve(
        population,
        immigrant_rate_override=None,
        mutation_rate_floor_override=None,
        reexpand_constraints=False,
    ):
        overrides.append(
            (
                immigrant_rate_override,
                mutation_rate_floor_override,
                reexpand_constraints,
            )
        )
        return original_evolve(
            population,
            immigrant_rate_override=immigrant_rate_override,
            mutation_rate_floor_override=mutation_rate_floor_override,
            reexpand_constraints=reexpand_constraints,
        )

    controller._engine.evolve = _wrapped_evolve
    controller.run()

    assert len(overrides) == 3
    assert overrides == [(None, None, False), (None, None, False), (None, None, False)]
    stats = controller.run_stats
    assert stats.plateau_events_count == 0
    assert stats.reheat_generations_applied == 0
    assert stats.stale_generations_final == 0


def test_wait_for_idle_workers_reads_scheduler_status(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = {
        0: epoch_pb2.GetResultsResponse(
            complete=True,
            wall_clock_ms=200,
            results=[_result("job0", "w0", 100, acc=0.8)],
            runtime_metrics=_runtime_metrics(queue_min=5.0, queue_samples=1),
        )
    }
    status_responses = [
        epoch_pb2.SchedulerStatusResponse(connected_workers=1, idle_workers=0, busy_workers=1, pending_jobs=0),
        epoch_pb2.SchedulerStatusResponse(connected_workers=2, idle_workers=1, busy_workers=1, pending_jobs=0),
        epoch_pb2.SchedulerStatusResponse(connected_workers=2, idle_workers=2, busy_workers=0, pending_jobs=0),
    ]
    controller, stub = _build_controller(
        monkeypatch,
        responses,
        pop_size=1,
        generations=1,
        wait_for_idle_workers=2,
        status_responses=status_responses,
    )

    controller.run()
    assert len(stub.status_requests) >= 3
    assert controller.run_stats.generations_completed == 1


def test_first_dispatch_offset_uses_queue_wait_min(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = {
        0: epoch_pb2.GetResultsResponse(
            complete=True,
            wall_clock_ms=300,
            results=[
                _result("job0", "w0", 100, acc=0.8),
                _result("job1", "w1", 100, acc=0.7),
            ],
            runtime_metrics=_runtime_metrics(queue_min=123.0, queue_samples=2),
        )
    }
    controller, _stub = _build_controller(
        monkeypatch,
        responses,
        pop_size=2,
        generations=1,
    )

    controller.run()
    stats = controller.run_stats
    assert stats.first_generation_submit_offset_s >= 0.0
    assert stats.first_dispatch_offset_s >= stats.first_generation_submit_offset_s
    assert stats.first_dispatch_offset_s == pytest.approx(
        stats.first_generation_submit_offset_s + 0.123,
        abs=0.05,
    )
