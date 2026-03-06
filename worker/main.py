"""Worker entry point."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys

# Suppress TF/XLA noise and disable XLA JIT (compilation overhead >> runtime for small models)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _configure_gpu(memory_limit_mb: int = 750) -> None:
    """Limit each worker to a fixed slice of GPU memory and disable XLA."""
    import tensorflow as tf

    # Disable XLA JIT — compilation overhead far exceeds runtime for small stress-test models
    tf.config.optimizer.set_jit(False)

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.set_logical_device_configuration(
            gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
        )


def main() -> None:
    """Parse arguments and start the worker client."""
    parser = argparse.ArgumentParser(description="Epoch training worker")
    parser.add_argument(
        "--scheduler-address",
        default=os.environ.get("EPOCH_SCHEDULER_ADDRESS", "localhost:50051"),
        help="Scheduler gRPC address (default: localhost:50051)",
    )
    parser.add_argument(
        "--worker-id",
        required=True,
        help="Unique worker identifier",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=10.0,
        help="Heartbeat interval in seconds (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-job training timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--gpu-memory-limit",
        type=int,
        default=750,
        help="GPU memory limit per worker in MB (default: 750)",
    )
    parser.add_argument(
        "--train-subset-size",
        type=int,
        default=0,
        help="Limit training data to N samples (0 = full dataset, default: 0)",
    )
    parser.add_argument(
        "--val-subset-size",
        type=int,
        default=0,
        help="Limit validation data to N samples (0 = full test set, default: 0)",
    )
    parser.add_argument(
        "--run-eagerly",
        action="store_true",
        help="Run TF in eager mode (avoids graph retracing overhead for varied architectures)",
    )
    parser.add_argument(
        "--gc-every-n-jobs",
        type=int,
        default=int(os.environ.get("EPOCH_GC_EVERY_N_JOBS", "1")),
        help="Run Python gc.collect() every N completed jobs (default: 1)",
    )
    parser.add_argument(
        "--deterministic-eval",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("EPOCH_DETERMINISTIC_EVAL", True),
        help="Use deterministic per-genome RNG seeding for training (default: enabled)",
    )
    parser.add_argument(
        "--deterministic-seed-offset",
        type=int,
        default=int(os.environ.get("EPOCH_DETERMINISTIC_SEED_OFFSET", "0")),
        help="Integer offset added to deterministic per-genome seed (default: 0)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--auth-key",
        default=os.environ.get("EPOCH_WORKER_AUTH_KEY", "superkey"),
        help="Worker registration auth key (default: EPOCH_WORKER_AUTH_KEY or superkey)",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=3.0,
        help="Seconds to wait before reconnecting after gRPC disconnects (default: 3.0)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=f"[%(asctime)s] [{args.worker_id}] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    _configure_gpu(args.gpu_memory_limit)

    from worker.client import WorkerClient
    from worker.trainer import Trainer

    trainer = Trainer(
        timeout_seconds=args.timeout,
        train_subset_size=args.train_subset_size,
        val_subset_size=args.val_subset_size,
        run_eagerly=args.run_eagerly,
        gc_every_n_jobs=args.gc_every_n_jobs,
        deterministic_eval=args.deterministic_eval,
        deterministic_seed_offset=args.deterministic_seed_offset,
    )
    client = WorkerClient(
        worker_id=args.worker_id,
        scheduler_address=args.scheduler_address,
        heartbeat_interval=args.heartbeat_interval,
        trainer=trainer,
        auth_key=args.auth_key,
        reconnect_delay_s=args.reconnect_delay,
    )

    def handle_signal(signum, frame):
        logging.info("Received signal %d, shutting down...", signum)
        client.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    client.run()


if __name__ == "__main__":
    main()
