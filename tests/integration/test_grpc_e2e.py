"""End-to-end gRPC integration tests against the real C++ scheduler binary.

Covers:
- scheduler process lifecycle (built and launched by fixtures in conftest.py)
- the WorkerService.Connect bidirectional stream (registration, heartbeats,
  job assignment, result reporting, auth and duplicate-id rejection)
- job state transitions Pending -> Dispatched -> Completed, observed through
  GetSchedulerStatus, GetGenerationResults, and the scheduler's JSONL metrics
  event log

All timing-sensitive assertions poll via `wait_until` instead of sleeping for
fixed durations, and the Dispatched state is pinned deterministically by
gating the FakeWorker's result delivery.
"""

from __future__ import annotations

import json

import epoch_pb2  # noqa: E402
import grpc  # noqa: E402
import pytest
from e2e_utils import wait_until

pytestmark = pytest.mark.integration

RPC_TIMEOUT_S = 2.0


def _status(stub) -> epoch_pb2.SchedulerStatusResponse:
    return stub.GetSchedulerStatus(epoch_pb2.SchedulerStatusRequest(), timeout=RPC_TIMEOUT_S)


def _results(stub, generation_id: int) -> epoch_pb2.GetResultsResponse:
    return stub.GetGenerationResults(
        epoch_pb2.GetResultsRequest(generation_id=generation_id), timeout=RPC_TIMEOUT_S
    )


def _config() -> epoch_pb2.HyperparamConfig:
    return epoch_pb2.HyperparamConfig(
        learning_rate=0.001,
        batch_size=32,
        optimizer=epoch_pb2.OPTIMIZER_ADAM,
        conv_filters=[8],
        kernel_size=3,
        dense_units=[16],
        dropout_rate=0.1,
        activation=epoch_pb2.ACTIVATION_RELU,
        epochs=1,
        dataset="mnist",
    )


def _submit(stub, generation_id: int, num_jobs: int) -> epoch_pb2.SubmitGenerationResponse:
    return stub.SubmitGeneration(
        epoch_pb2.SubmitGenerationRequest(
            generation_id=generation_id,
            configs=[_config() for _ in range(num_jobs)],
        ),
        timeout=5.0,
    )


def _read_metric_events(scheduler_process) -> list[dict]:
    text = scheduler_process.metrics_log_path.read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ─── Scheduler and stream lifecycle ──────────────────────────────────────────


def test_fresh_scheduler_reports_empty_status(control_stub):
    status = _status(control_stub)
    assert status.connected_workers == 0
    assert status.idle_workers == 0
    assert status.busy_workers == 0
    assert status.pending_jobs == 0


def test_worker_registration_and_disconnect(control_stub, worker_factory):
    worker = worker_factory("worker-lifecycle")

    wait_until(
        lambda: _status(control_stub).connected_workers == 1,
        description="worker to register",
    )
    status = _status(control_stub)
    assert status.idle_workers == 1
    assert status.busy_workers == 0

    # Graceful disconnect (half-close) must remove the worker from the pool.
    worker.stop()
    wait_until(
        lambda: _status(control_stub).connected_workers == 0,
        description="worker to be removed after disconnect",
    )


def test_registration_rejected_with_bad_auth_key(control_stub, worker_factory):
    worker = worker_factory("worker-bad-auth", auth_key="wrong-key")

    assert worker.wait_stream_done(timeout_s=10.0), "stream should terminate on bad auth"
    assert worker.stream_error is not None
    assert worker.stream_error.code() == grpc.StatusCode.UNAUTHENTICATED
    assert _status(control_stub).connected_workers == 0


def test_duplicate_worker_id_rejected(control_stub, worker_factory):
    worker_factory("worker-dup")
    wait_until(
        lambda: _status(control_stub).connected_workers == 1,
        description="first worker to register",
    )

    imposter = worker_factory("worker-dup")
    assert imposter.wait_stream_done(timeout_s=10.0), "duplicate stream should terminate"
    assert imposter.stream_error is not None
    assert imposter.stream_error.code() == grpc.StatusCode.ALREADY_EXISTS
    assert _status(control_stub).connected_workers == 1


# ─── Job state transitions ───────────────────────────────────────────────────


def test_job_lifecycle_pending_dispatched_completed(
    scheduler_process, control_stub, worker_factory
):
    generation_id = 5
    num_jobs = 3

    worker = worker_factory("worker-jobs")
    worker.gate.clear()  # hold dispatched jobs open so we can observe the state
    wait_until(
        lambda: _status(control_stub).idle_workers == 1,
        description="worker to register and go idle",
    )

    # An unknown generation is never complete.
    assert _results(control_stub, generation_id).complete is False

    submit = _submit(control_stub, generation_id, num_jobs)
    assert submit.accepted is True
    assert submit.num_jobs == num_jobs

    # Pending -> Dispatched: the single worker takes exactly one job and the
    # rest stay queued. This state is stable while the worker gate is closed.
    wait_until(
        lambda: _status(control_stub).busy_workers == 1,
        description="first job to be dispatched",
    )
    wait_until(
        lambda: len(worker.assigned_job_ids) == 1,
        description="worker to receive the assignment",
    )
    status = _status(control_stub)
    assert status.pending_jobs == num_jobs - 1
    assert status.busy_workers == 1
    assert status.idle_workers == 0
    assert worker.assigned_job_ids[0].startswith(f"gen{generation_id}_job")
    assert _results(control_stub, generation_id).complete is False

    # Dispatched -> Completed: release the gate and let all jobs drain.
    worker.gate.set()

    def fetch_complete():
        response = _results(control_stub, generation_id)
        return response if response.complete else None

    final = wait_until(fetch_complete, description="generation to complete")

    expected_job_ids = {f"gen{generation_id}_job{i}" for i in range(num_jobs)}
    assert {r.job_id for r in final.results} == expected_job_ids
    for result in final.results:
        assert result.status == epoch_pb2.JOB_STATUS_COMPLETED
        assert result.generation_id == generation_id
        assert result.worker_id == "worker-jobs"
        assert result.validation_accuracy == pytest.approx(0.9)
    assert final.wall_clock_ms > 0

    # With a single worker, jobs are dispatched strictly FIFO.
    assert worker.assigned_job_ids == sorted(expected_job_ids)

    # The pool returns to fully idle once everything is completed.
    wait_until(
        lambda: _status(control_stub).idle_workers == 1 and _status(control_stub).pending_jobs == 0,
        description="worker to return to idle with empty queue",
    )

    # Runtime metrics cover every dispatch; the first dispatch has no idle gap.
    metrics = final.runtime_metrics
    assert metrics.dispatch_samples == num_jobs
    assert metrics.queue_wait_samples == num_jobs
    assert metrics.idle_gap_samples == num_jobs - 1
    assert metrics.queue_wait_min_ms <= metrics.queue_wait_p50_ms <= metrics.queue_wait_max_ms
    assert metrics.dispatch_latency_p50_ms <= metrics.dispatch_latency_max_ms
    # Jobs 2 and 3 sat in the queue while the gate was closed.
    assert metrics.queue_wait_max_ms > 0

    # The JSONL event log proves Pending -> Dispatched -> Completed ordering:
    # every job has exactly one dispatch event strictly before its result event.
    events = _read_metric_events(scheduler_process)
    for job_id in expected_job_ids:
        dispatch_indices = [
            i for i, e in enumerate(events) if e["event"] == "dispatch" and e["job_id"] == job_id
        ]
        result_indices = [
            i for i, e in enumerate(events) if e["event"] == "result" and e["job_id"] == job_id
        ]
        assert len(dispatch_indices) == 1, f"{job_id} should be dispatched exactly once"
        assert len(result_indices) == 1, f"{job_id} should complete exactly once"
        assert (
            dispatch_indices[0] < result_indices[0]
        ), f"{job_id} must be dispatched before it completes"
        assert events[result_indices[0]]["status"] == epoch_pb2.JOB_STATUS_COMPLETED


def test_generation_fans_out_across_two_workers(control_stub, worker_factory):
    generation_id = 7
    num_jobs = 6

    first = worker_factory("worker-a")
    second = worker_factory("worker-b")
    first.gate.clear()
    second.gate.clear()
    wait_until(
        lambda: _status(control_stub).idle_workers == 2,
        description="both workers to register",
    )

    submit = _submit(control_stub, generation_id, num_jobs)
    assert submit.num_jobs == num_jobs

    # With both gates closed, each worker holds exactly one in-flight job.
    wait_until(
        lambda: _status(control_stub).busy_workers == 2,
        description="both workers to be busy",
    )
    wait_until(
        lambda: len(first.assigned_job_ids) == 1 and len(second.assigned_job_ids) == 1,
        description="each worker to hold one assignment",
    )
    assert _status(control_stub).pending_jobs == num_jobs - 2

    first.gate.set()
    second.gate.set()

    def fetch_complete():
        response = _results(control_stub, generation_id)
        return response if response.complete else None

    final = wait_until(fetch_complete, description="generation to complete")

    expected_job_ids = {f"gen{generation_id}_job{i}" for i in range(num_jobs)}
    assert {r.job_id for r in final.results} == expected_job_ids
    assert all(r.status == epoch_pb2.JOB_STATUS_COMPLETED for r in final.results)

    # Both workers participated, every job ran exactly once, and the
    # scheduler's per-result worker attribution matches what each worker did.
    completed_by_first = set(first.completed_job_ids)
    completed_by_second = set(second.completed_job_ids)
    assert completed_by_first and completed_by_second
    assert completed_by_first.isdisjoint(completed_by_second)
    assert completed_by_first | completed_by_second == expected_job_ids
    for result in final.results:
        owner = completed_by_first if result.worker_id == "worker-a" else completed_by_second
        assert result.job_id in owner
