#!/usr/bin/env python3
"""Launch long-lived Epoch training workers on Modal GPUs.

Examples:
    # Launch 8 remote workers against your scheduler
    modal run scripts/modal_workers.py::launch \
        --scheduler-address your.host.name:50051 \
        --num-workers 8

    # Use a stronger shared auth key directly
    modal run scripts/modal_workers.py::launch \
        --scheduler-address your.host.name:50051 \
        --num-workers 8 \
        --auth-key replace-me

    # Or load the auth key from a Modal Secret named "epoch-worker-auth"
    # where EPOCH_WORKER_AUTH_KEY is set.
    modal run scripts/modal_workers.py::launch \
        --scheduler-address your.host.name:50051 \
        --num-workers 8 \
        --modal-secret-name epoch-worker-auth
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    import modal
except ImportError as exc:  # pragma: no cover - optional runtime dependency
    raise SystemExit(
        "Missing dependency 'modal'. Install it with `pip install modal`."
    ) from exc


def _resolve_project_root() -> Path:
    """Resolve project root for both local runs and Modal container imports."""
    local_root = Path(__file__).resolve().parents[1]
    if (local_root / "worker").exists() and (local_root / "generated").exists():
        return local_root

    modal_root = Path("/root/epoch")
    if (modal_root / "worker").exists() and (modal_root / "generated").exists():
        return modal_root

    return local_root


PROJECT_ROOT = _resolve_project_root()
WORKER_DIR = PROJECT_ROOT / "worker"
GENERATED_DIR = PROJECT_ROOT / "generated"

if not (GENERATED_DIR / "python" / "epoch_pb2.py").exists():
    raise SystemExit(
        "Missing generated/python/epoch_pb2.py. Run `bash scripts/generate_protos.sh` first."
    )


DEFAULT_GPU = os.environ.get("EPOCH_MODAL_GPU", "T4")
DEFAULT_NUM_WORKERS = int(os.environ.get("EPOCH_MODAL_NUM_WORKERS", "8"))
DEFAULT_AUTH_KEY = os.environ.get("EPOCH_WORKER_AUTH_KEY", "superkey")
DEFAULT_DETERMINISTIC_EVAL = os.environ.get("EPOCH_DETERMINISTIC_EVAL", "1").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
DEFAULT_DETERMINISTIC_SEED_OFFSET = int(os.environ.get("EPOCH_DETERMINISTIC_SEED_OFFSET", "0"))
APP_NAME = os.environ.get("EPOCH_MODAL_APP_NAME", "epoch-workers")
DEFAULT_MODAL_SECRET_NAME = "superkey"


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "grpcio==1.62.0",
        "protobuf==4.25.0",
        "numpy==1.26.0",
        "tensorflow==2.16.2",
    )
    .add_local_dir(str(WORKER_DIR), remote_path="/root/epoch/worker")
    .add_local_dir(str(GENERATED_DIR), remote_path="/root/epoch/generated")
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    timeout=24 * 60 * 60,
    gpu=DEFAULT_GPU,
    secrets=[modal.Secret.from_name(DEFAULT_MODAL_SECRET_NAME)],
)
def run_worker(
    worker_id: str,
    scheduler_address: str,
    auth_key: str,
    heartbeat_interval: float,
    timeout_seconds: float,
    reconnect_delay_s: float,
    gpu_memory_limit_mb: int,
    train_subset_size: int,
    val_subset_size: int,
    run_eagerly: bool,
    gc_every_n_jobs: int,
    deterministic_eval: bool,
    deterministic_seed_offset: int,
    log_level: str,
) -> None:
    """Run one worker process inside a Modal container."""
    import logging
    import os
    import signal
    import sys

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

    os.chdir("/root/epoch")
    sys.path.insert(0, "/root/epoch")

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.WARNING),
        format=f"[%(asctime)s] [{worker_id}] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from worker.main import _configure_gpu
    from worker.client import WorkerClient
    from worker.trainer import Trainer

    resolved_auth_key = (
        auth_key
        or os.environ.get("EPOCH_WORKER_AUTH_KEY")
        or os.environ.get("key1")
        or "superkey"
    )

    _configure_gpu(gpu_memory_limit_mb)

    trainer = Trainer(
        timeout_seconds=timeout_seconds,
        train_subset_size=train_subset_size,
        val_subset_size=val_subset_size,
        run_eagerly=run_eagerly,
        gc_every_n_jobs=gc_every_n_jobs,
        deterministic_eval=deterministic_eval,
        deterministic_seed_offset=deterministic_seed_offset,
    )
    client = WorkerClient(
        worker_id=worker_id,
        scheduler_address=scheduler_address,
        heartbeat_interval=heartbeat_interval,
        trainer=trainer,
        auth_key=resolved_auth_key,
        reconnect_delay_s=reconnect_delay_s,
    )

    def _handle_signal(signum, _frame) -> None:
        client.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    client.run()


@app.local_entrypoint()
def launch(
    scheduler_address: str,
    num_workers: int = DEFAULT_NUM_WORKERS,
    worker_prefix: str = "modal-w",
    gpu_type: str = DEFAULT_GPU,
    auth_key: str = DEFAULT_AUTH_KEY,
    modal_secret_name: str = "",
    heartbeat_interval: float = 10.0,
    timeout_seconds: float = 15.0,
    reconnect_delay_s: float = 3.0,
    gpu_memory_limit_mb: int = 896,
    train_subset_size: int = 512,
    val_subset_size: int = 256,
    run_eagerly: bool = False,
    gc_every_n_jobs: int = 1,
    deterministic_eval: bool = DEFAULT_DETERMINISTIC_EVAL,
    deterministic_seed_offset: int = DEFAULT_DETERMINISTIC_SEED_OFFSET,
    log_level: str = "WARNING",
    wait_for_workers: bool = False,
) -> None:
    """Spawn N long-lived remote workers.

    `auth_key` defaults to EPOCH_WORKER_AUTH_KEY or "superkey".
    For stronger key handling, set `modal_secret_name` and store
    EPOCH_WORKER_AUTH_KEY in that Modal Secret.
    """
    # modal==1.3.x does not expose Function.with_options().
    # Keep runtime behavior compatible by using function-level defaults.
    if gpu_type != DEFAULT_GPU:
        print(
            f"WARNING: modal client in this environment does not support per-run GPU override. "
            f"Using configured GPU={DEFAULT_GPU}. "
            f"Set EPOCH_MODAL_GPU={gpu_type} before `modal run` to change it."
        )

    if modal_secret_name and modal_secret_name != DEFAULT_MODAL_SECRET_NAME:
        raise ValueError(
            "This launcher is configured with secret 'superkey' at function definition time. "
            "Use --modal-secret-name superkey, or set --auth-key directly."
        )

    if modal_secret_name:
        effective_auth_key = ""
    else:
        effective_auth_key = auth_key

    calls = []
    for i in range(num_workers):
        worker_id = f"{worker_prefix}-{i}"
        call = run_worker.spawn(
            worker_id=worker_id,
            scheduler_address=scheduler_address,
            auth_key=effective_auth_key,
            heartbeat_interval=heartbeat_interval,
            timeout_seconds=timeout_seconds,
            reconnect_delay_s=reconnect_delay_s,
            gpu_memory_limit_mb=gpu_memory_limit_mb,
            train_subset_size=train_subset_size,
            val_subset_size=val_subset_size,
            run_eagerly=run_eagerly,
            gc_every_n_jobs=gc_every_n_jobs,
            deterministic_eval=deterministic_eval,
            deterministic_seed_offset=deterministic_seed_offset,
            log_level=log_level,
        )
        calls.append((worker_id, call))
        print(f"Spawned {worker_id}")

    print(
        f"Launched {num_workers} Modal workers on GPU={gpu_type} for scheduler {scheduler_address}"
    )
    if effective_auth_key == "superkey" and not modal_secret_name:
        print("WARNING: using default auth key 'superkey'. Set a stronger key before production.")

    if wait_for_workers:
        for worker_id, call in calls:
            print(f"Waiting on {worker_id}...")
            call.get()
