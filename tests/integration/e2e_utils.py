"""Helpers for end-to-end gRPC integration tests.

Provides a polling helper for timing-uncertain assertions and a FakeWorker
that speaks the real WorkerService.Connect bidirectional stream protocol
without pulling in TensorFlow.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATED_PYTHON = REPO_ROOT / "generated" / "python"
if str(GENERATED_PYTHON) not in sys.path:
    sys.path.insert(0, str(GENERATED_PYTHON))

import epoch_pb2  # noqa: E402
import epoch_pb2_grpc  # noqa: E402
import grpc  # noqa: E402

_STOP = object()


def wait_until(
    predicate: Callable[[], object],
    timeout_s: float = 20.0,
    interval_s: float = 0.05,
    description: str = "condition",
):
    """Poll `predicate` until it returns a truthy value or the timeout expires.

    Returns the truthy value, so callers can capture the observed state.
    Raising AssertionError (not returning False) keeps timing failures loud.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(interval_s)
    raise AssertionError(f"Timed out after {timeout_s:.1f}s waiting for {description}")


class FakeWorker:
    """Minimal Python worker for integration tests.

    Implements the WorkerService.Connect protocol: sends a registration as the
    first stream message, then periodic heartbeats, and answers each
    JobAssignment with a COMPLETED TrainingResult.

    Job completion can be held back by clearing `gate`, which lets tests
    observe the Dispatched (worker busy, result outstanding) state without
    sleeping for arbitrary durations.
    """

    def __init__(
        self,
        worker_id: str,
        scheduler_address: str,
        auth_key: str,
        heartbeat_interval_s: float = 1.0,
        validation_accuracy: float = 0.9,
        training_time_ms: int = 5,
    ) -> None:
        self.worker_id = worker_id
        self.scheduler_address = scheduler_address
        self.auth_key = auth_key
        self.heartbeat_interval_s = heartbeat_interval_s
        self.validation_accuracy = validation_accuracy
        self.training_time_ms = training_time_ms

        # Set by default: jobs complete immediately. Clear to hold jobs
        # in the Dispatched state.
        self.gate = threading.Event()
        self.gate.set()

        self._outgoing: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._assigned_job_ids: list[str] = []
        self._completed_job_ids: list[str] = []
        self._stream_error: grpc.RpcError | None = None
        self._stream_done = threading.Event()

        self._channel: grpc.Channel | None = None
        self._call = None
        self._reader_thread: threading.Thread | None = None
        self._heartbeat_thread: threading.Thread | None = None

    # ─── Lifecycle ───────────────────────────────────────────────────────

    def start(self) -> "FakeWorker":
        self._channel = grpc.insecure_channel(self.scheduler_address)
        stub = epoch_pb2_grpc.WorkerServiceStub(self._channel)
        self._call = stub.Connect(self._request_iterator())
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        return self

    def stop(self, timeout_s: float = 10.0) -> None:
        """Disconnect gracefully: half-close the stream and join threads."""
        self.gate.set()
        self._stop_event.set()
        self._outgoing.put(_STOP)
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=timeout_s)
            if self._reader_thread.is_alive() and self._call is not None:
                self._call.cancel()
                self._reader_thread.join(timeout=timeout_s)
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=timeout_s)
        if self._channel is not None:
            self._channel.close()

    # ─── Observed state ──────────────────────────────────────────────────

    @property
    def assigned_job_ids(self) -> list[str]:
        with self._lock:
            return list(self._assigned_job_ids)

    @property
    def completed_job_ids(self) -> list[str]:
        with self._lock:
            return list(self._completed_job_ids)

    @property
    def stream_error(self) -> grpc.RpcError | None:
        return self._stream_error

    def wait_stream_done(self, timeout_s: float = 10.0) -> bool:
        return self._stream_done.wait(timeout=timeout_s)

    # ─── Stream plumbing ─────────────────────────────────────────────────

    def _request_iterator(self):
        """Registration first, then queued heartbeats/results until stopped."""
        yield epoch_pb2.WorkerMessage(
            registration=epoch_pb2.WorkerRegistration(
                worker_id=self.worker_id,
                num_gpus=0,
                memory_mb=1024,
                auth_key=self.auth_key,
            )
        )
        while not self._stop_event.is_set():
            try:
                item = self._outgoing.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is _STOP:
                return
            yield item

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(timeout=self.heartbeat_interval_s):
            self._outgoing.put(
                epoch_pb2.WorkerMessage(
                    heartbeat=epoch_pb2.Heartbeat(
                        worker_id=self.worker_id,
                        timestamp_ms=int(time.time() * 1000),
                    )
                )
            )

    def _read_loop(self) -> None:
        try:
            for msg in self._call:
                if msg.HasField("shutdown"):
                    break
                if msg.HasField("job_assignment"):
                    self._handle_job(msg.job_assignment)
        except grpc.RpcError as err:
            self._stream_error = err
        finally:
            self._stream_done.set()

    def _handle_job(self, job: epoch_pb2.JobAssignment) -> None:
        with self._lock:
            self._assigned_job_ids.append(job.job_id)

        # The scheduler keeps this worker Busy until a result arrives, so
        # blocking here pins the job in the Dispatched state for the test.
        self.gate.wait()

        result = epoch_pb2.TrainingResult(
            job_id=job.job_id,
            generation_id=job.generation_id,
            worker_id=self.worker_id,
            validation_accuracy=self.validation_accuracy,
            training_loss=0.1,
            training_time_ms=self.training_time_ms,
            status=epoch_pb2.JOB_STATUS_COMPLETED,
        )
        self._outgoing.put(epoch_pb2.WorkerMessage(result=result))
        with self._lock:
            self._completed_job_ids.append(job.job_id)
