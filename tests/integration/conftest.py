"""Fixtures for end-to-end gRPC integration tests.

Builds the C++ scheduler binary (or reuses an existing build), launches it on
a dynamically allocated port for each test, and tears it down afterwards.

Binary resolution order:
1. `EPOCH_SCHEDULER_BIN` env var (use a prebuilt binary as-is).
2. Configured build tree at `build/` -> cheap incremental rebuild.
3. Existing binary at `build/scheduler/epoch_scheduler`.
4. Cold configure+build, only when `EPOCH_INTEGRATION_BUILD=1` (it fetches and
   compiles gRPC from source, which takes a long time); otherwise skip.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATED_PYTHON = REPO_ROOT / "generated" / "python"
if str(GENERATED_PYTHON) not in sys.path:
    sys.path.insert(0, str(GENERATED_PYTHON))

import epoch_pb2  # noqa: E402
import epoch_pb2_grpc  # noqa: E402
import grpc  # noqa: E402

TEST_AUTH_KEY = "integration-test-key"
BUILD_DIR = REPO_ROOT / "build"
SCHEDULER_BINARY = BUILD_DIR / "scheduler" / "epoch_scheduler"

READY_TIMEOUT_S = 20.0
SHUTDOWN_TIMEOUT_S = 10.0
CONFIGURE_TIMEOUT_S = 1800
BUILD_TIMEOUT_S = 3600


def _tail(text: str, lines: int = 30) -> str:
    return "\n".join(text.splitlines()[-lines:])


def _run_cmake(args: list[str], timeout_s: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


@pytest.fixture(scope="session")
def scheduler_binary() -> Path:
    """Resolve (building if needed) the epoch_scheduler binary."""
    env_bin = os.environ.get("EPOCH_SCHEDULER_BIN")
    if env_bin:
        path = Path(env_bin)
        if not path.is_file():
            pytest.fail(f"EPOCH_SCHEDULER_BIN={env_bin} does not exist")
        return path

    cmake = shutil.which("cmake")
    if cmake is None:
        if SCHEDULER_BINARY.is_file():
            return SCHEDULER_BINARY
        pytest.skip("cmake is not available and no prebuilt scheduler binary found")

    configured = (BUILD_DIR / "CMakeCache.txt").is_file()
    if not configured:
        if os.environ.get("EPOCH_INTEGRATION_BUILD") != "1":
            if SCHEDULER_BINARY.is_file():
                return SCHEDULER_BINARY
            pytest.skip(
                "build/ is not configured; configure it with "
                "`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release` or set "
                "EPOCH_INTEGRATION_BUILD=1 to allow the (slow) cold build"
            )
        configure = _run_cmake(
            [cmake, "-S", str(REPO_ROOT), "-B", str(BUILD_DIR), "-DCMAKE_BUILD_TYPE=Release"],
            CONFIGURE_TIMEOUT_S,
        )
        if configure.returncode != 0:
            # Cold configure fetches gRPC from GitHub; treat failure as an
            # environment limitation rather than a product regression.
            pytest.skip(
                "cmake configure failed (network/toolchain issue?):\n"
                + _tail(configure.stdout + configure.stderr)
            )

    build = _run_cmake(
        [
            cmake,
            "--build",
            str(BUILD_DIR),
            "--target",
            "epoch_scheduler",
            "-j",
            str(os.cpu_count() or 2),
        ],
        BUILD_TIMEOUT_S,
    )
    if build.returncode != 0:
        pytest.fail("scheduler build failed:\n" + _tail(build.stdout + build.stderr))
    if not SCHEDULER_BINARY.is_file():
        pytest.fail(f"build succeeded but {SCHEDULER_BINARY} is missing")
    return SCHEDULER_BINARY


@dataclass
class SchedulerProcess:
    """Handle for a running scheduler instance under test."""

    address: str
    process: subprocess.Popen
    log_path: Path
    metrics_log_path: Path

    def log_tail(self, lines: int = 40) -> str:
        try:
            return _tail(self.log_path.read_text(errors="replace"), lines)
        except OSError:
            return "<no log captured>"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_ready(handle: SchedulerProcess) -> None:
    """Block until GetSchedulerStatus answers, proving the server is up."""
    deadline = time.monotonic() + READY_TIMEOUT_S
    with grpc.insecure_channel(handle.address) as channel:
        stub = epoch_pb2_grpc.SchedulerControlStub(channel)
        last_error: grpc.RpcError | None = None
        while time.monotonic() < deadline:
            if handle.process.poll() is not None:
                pytest.fail(
                    f"scheduler exited early with code {handle.process.returncode}:\n"
                    + handle.log_tail()
                )
            try:
                stub.GetSchedulerStatus(epoch_pb2.SchedulerStatusRequest(), timeout=1.0)
                return
            except grpc.RpcError as err:
                last_error = err
                time.sleep(0.1)
    pytest.fail(
        f"scheduler did not become ready within {READY_TIMEOUT_S}s "
        f"(last error: {last_error}):\n" + handle.log_tail()
    )


def _terminate(handle: SchedulerProcess) -> None:
    if handle.process.poll() is None:
        handle.process.terminate()
        try:
            handle.process.wait(timeout=SHUTDOWN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            handle.process.kill()
            handle.process.wait(timeout=SHUTDOWN_TIMEOUT_S)


@pytest.fixture
def scheduler_process(scheduler_binary: Path, tmp_path: Path):
    """Launch a scheduler on a free port; SIGTERM and reap it on teardown."""
    address = f"127.0.0.1:{_find_free_port()}"
    log_path = tmp_path / "scheduler.log"
    metrics_log_path = tmp_path / "scheduler_metrics.jsonl"

    with log_path.open("wb") as log_file:
        process = subprocess.Popen(
            [
                str(scheduler_binary),
                "--listen-address",
                address,
                "--dispatch-interval",
                "5",
                "--heartbeat-timeout",
                "10000",
                "--worker-auth-key",
                TEST_AUTH_KEY,
                "--metrics-log-path",
                str(metrics_log_path),
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=REPO_ROOT,
        )

    handle = SchedulerProcess(
        address=address,
        process=process,
        log_path=log_path,
        metrics_log_path=metrics_log_path,
    )
    try:
        _wait_for_ready(handle)
        yield handle
    finally:
        _terminate(handle)


@pytest.fixture
def control_channel(scheduler_process: SchedulerProcess):
    with grpc.insecure_channel(scheduler_process.address) as channel:
        yield channel


@pytest.fixture
def control_stub(control_channel) -> epoch_pb2_grpc.SchedulerControlStub:
    return epoch_pb2_grpc.SchedulerControlStub(control_channel)


@pytest.fixture
def worker_factory(scheduler_process: SchedulerProcess):
    """Create FakeWorkers connected to the test scheduler; stop them all on teardown."""
    from e2e_utils import FakeWorker

    workers: list[FakeWorker] = []

    def make(worker_id: str, auth_key: str = TEST_AUTH_KEY, **kwargs) -> FakeWorker:
        worker = FakeWorker(
            worker_id=worker_id,
            scheduler_address=scheduler_process.address,
            auth_key=auth_key,
            **kwargs,
        ).start()
        workers.append(worker)
        return worker

    yield make

    for worker in workers:
        worker.stop()
