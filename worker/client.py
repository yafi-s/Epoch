"""gRPC client for worker-to-scheduler communication."""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Any

import grpc

sys.path.insert(0, "generated/python")
import epoch_pb2
import epoch_pb2_grpc

from worker.trainer import Trainer

logger = logging.getLogger(__name__)


class WorkerClient:
    """Manages the bidirectional gRPC stream between a worker and the scheduler.

    Sends registration and periodic heartbeats. Receives job assignments,
    dispatches to the Trainer, and sends results back.

    Attributes:
        worker_id: Unique identifier for this worker.
        scheduler_address: Address of the scheduler (host:port).
        heartbeat_interval: Seconds between heartbeat messages.
        trainer: Trainer instance for executing jobs.
    """

    def __init__(
        self,
        worker_id: str,
        scheduler_address: str = "localhost:50051",
        heartbeat_interval: float = 10.0,
        trainer: Trainer | None = None,
        auth_key: str = "superkey",
        reconnect_delay_s: float = 3.0,
    ) -> None:
        self.worker_id = worker_id
        self.scheduler_address = scheduler_address
        self.heartbeat_interval = heartbeat_interval
        self.trainer = trainer or Trainer()
        self.auth_key = auth_key
        self.reconnect_delay_s = reconnect_delay_s
        self._stop_event = threading.Event()
        self._outgoing: list[epoch_pb2.WorkerMessage] = []
        self._outgoing_lock = threading.Lock()
        self._has_messages = threading.Event()

    def run(self) -> None:
        """Connect to the scheduler and start processing jobs.

        Blocks until the connection is closed or stop() is called.
        """
        # Start heartbeat thread once. It keeps enqueueing messages across reconnects.
        heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        heartbeat_thread.start()

        while not self._stop_event.is_set():
            channel = grpc.insecure_channel(self.scheduler_address)
            stub = epoch_pb2_grpc.WorkerServiceStub(channel)

            logger.info("Worker %s connecting to %s", self.worker_id, self.scheduler_address)

            try:
                responses = stub.Connect(self._message_generator())
                for scheduler_msg in responses:
                    if self._stop_event.is_set():
                        break
                    self._handle_scheduler_message(scheduler_msg)
            except grpc.RpcError as e:
                if self._stop_event.is_set():
                    break
                logger.error("gRPC error: %s", e)
            finally:
                channel.close()

            if self._stop_event.is_set():
                break

            logger.info(
                "Worker %s reconnecting in %.1fs", self.worker_id, self.reconnect_delay_s
            )
            self._stop_event.wait(timeout=self.reconnect_delay_s)

        logger.info("Worker %s disconnected", self.worker_id)

    def stop(self) -> None:
        """Signal the worker to stop."""
        self._stop_event.set()

    def _message_generator(self):
        """Yields WorkerMessages: registration first, then heartbeats and results."""
        # Send registration
        reg_msg = epoch_pb2.WorkerMessage(
            registration=epoch_pb2.WorkerRegistration(
                worker_id=self.worker_id,
                auth_key=self.auth_key,
            )
        )
        yield reg_msg
        logger.info("Worker %s registered", self.worker_id)

        # Yield queued messages (heartbeats, results)
        while not self._stop_event.is_set():
            with self._outgoing_lock:
                msgs = list(self._outgoing)
                self._outgoing.clear()
                self._has_messages.clear()

            for msg in msgs:
                yield msg

            # Wait for new messages or timeout (heartbeat will set the event periodically)
            self._has_messages.wait(timeout=0.5)

    def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat messages."""
        while not self._stop_event.is_set():
            hb = epoch_pb2.WorkerMessage(
                heartbeat=epoch_pb2.Heartbeat(
                    worker_id=self.worker_id,
                    timestamp_ms=int(time.time() * 1000),
                )
            )
            with self._outgoing_lock:
                self._outgoing.append(hb)
                self._has_messages.set()
            self._stop_event.wait(timeout=self.heartbeat_interval)

    def _handle_scheduler_message(self, msg: epoch_pb2.SchedulerMessage) -> None:
        """Process a message from the scheduler."""
        if msg.HasField("shutdown"):
            logger.info("Worker %s received shutdown signal", self.worker_id)
            self.stop()
            return

        if msg.HasField("job_assignment"):
            job = msg.job_assignment
            logger.info(
                "Worker %s received job %s (gen %d)",
                self.worker_id,
                job.job_id,
                job.generation_id,
            )
            self._execute_job(job)

    def _execute_job(self, job: epoch_pb2.JobAssignment) -> None:
        """Train a model and send the result back."""
        config = job.config
        genome = {
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "optimizer": {1: "sgd", 2: "adam", 3: "rmsprop", 4: "adamw"}.get(
                config.optimizer, "adam"
            ),
            "conv_filters_1": config.conv_filters[0] if len(config.conv_filters) > 0 else 32,
            "conv_filters_2": config.conv_filters[1] if len(config.conv_filters) > 1 else 64,
            "conv_filters_3": config.conv_filters[2] if len(config.conv_filters) > 2 else 128,
            "kernel_size": config.kernel_size or 3,
            "dense_units_1": config.dense_units[0] if len(config.dense_units) > 0 else 128,
            "dropout_rate": config.dropout_rate,
            "activation": {1: "relu", 2: "elu", 3: "selu", 4: "tanh", 5: "swish"}.get(
                config.activation, "relu"
            ),
            "epochs": config.epochs or 10,
            "dataset": config.dataset or "mnist",
        }

        result = self.trainer.train(genome)

        # Build protobuf result
        status_map = {
            "completed": epoch_pb2.JOB_STATUS_COMPLETED,
            "failed": epoch_pb2.JOB_STATUS_FAILED,
            "timeout": epoch_pb2.JOB_STATUS_TIMEOUT,
        }
        proto_result = epoch_pb2.TrainingResult(
            job_id=job.job_id,
            generation_id=job.generation_id,
            worker_id=self.worker_id,
            validation_accuracy=result.validation_accuracy,
            training_loss=result.training_loss,
            training_time_ms=result.training_time_ms,
            status=status_map.get(result.status, epoch_pb2.JOB_STATUS_FAILED),
            error_message=result.error_message,
        )

        msg = epoch_pb2.WorkerMessage(result=proto_result)
        with self._outgoing_lock:
            self._outgoing.append(msg)
            self._has_messages.set()

        logger.info(
            "Worker %s completed job %s: acc=%.4f time=%dms",
            self.worker_id,
            job.job_id,
            result.validation_accuracy,
            result.training_time_ms,
        )
