#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Configuration (override via environment variables)
NUM_WORKERS="${NUM_WORKERS:-4}"
GPU_MEMORY_MB="${GPU_MEMORY_MB:-1536}"
TRAIN_SUBSET="${TRAIN_SUBSET:-1000}"
VAL_SUBSET="${VAL_SUBSET:-500}"
POP_SIZE="${POP_SIZE:-100}"
NUM_GENS="${NUM_GENS:-50}"
DATASET="${DATASET:-mnist}"
WORKER_TIMEOUT="${WORKER_TIMEOUT:-12}"
RUN_EAGERLY="${RUN_EAGERLY:-0}"
DISPATCH_INTERVAL_MS="${DISPATCH_INTERVAL_MS:-10}"
HEARTBEAT_TIMEOUT_MS="${HEARTBEAT_TIMEOUT_MS:-30000}"
SCHEDULER_LISTEN_ADDRESS="${SCHEDULER_LISTEN_ADDRESS:-0.0.0.0:50051}"
SCHEDULER_PUBLIC_ADDRESS="${SCHEDULER_PUBLIC_ADDRESS:-localhost:50051}"
GA_SCHEDULER_ADDRESS="${GA_SCHEDULER_ADDRESS:-localhost:50051}"
GA_RPC_TIMEOUT_S="${GA_RPC_TIMEOUT_S:-10}"
GA_GENERATION_TIMEOUT_S="${GA_GENERATION_TIMEOUT_S:-300}"
GA_PROGRESS_LOG_INTERVAL_S="${GA_PROGRESS_LOG_INTERVAL_S:-10}"
GA_MAX_WALL_CLOCK_S="${GA_MAX_WALL_CLOCK_S:-0}"
GA_EXPECTED_WORKERS="${GA_EXPECTED_WORKERS:-0}"
THROUGHPUT_WORKER_COUNT="${THROUGHPUT_WORKER_COUNT:-8}"
NETWORK_LATENCY_MS="${NETWORK_LATENCY_MS:-150}"
START_LOCAL_WORKERS="${START_LOCAL_WORKERS:-1}"
WORKER_AUTH_KEY="${WORKER_AUTH_KEY:-${EPOCH_WORKER_AUTH_KEY:-superkey}}"
RESULTS_DIR="${RESULTS_DIR:-results/stress_test}"
OUTPUT_NAME="${OUTPUT_NAME:-stress_test}"

# Resolve python interpreter
if command -v poetry &>/dev/null; then
    PYTHON="poetry run python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi

# Pre-flight checks
SCHEDULER_BIN="./build/scheduler/epoch_scheduler"
if [[ ! -x "$SCHEDULER_BIN" ]]; then
    echo "ERROR: Scheduler binary not found at $SCHEDULER_BIN"
    echo "Build it first:"
    echo "  mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . --target epoch_scheduler -j\$(nproc)"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "     Epoch Scheduler Stress Test"
echo "========================================"
echo "  Workers:          $NUM_WORKERS"
echo "  GPU mem/worker:   ${GPU_MEMORY_MB}MB"
echo "  Train subset:     $TRAIN_SUBSET samples"
echo "  Val subset:       $VAL_SUBSET samples"
echo "  Population:       $POP_SIZE"
echo "  Generations:      $NUM_GENS"
echo "  Dataset:          $DATASET"
echo "  Worker timeout:   ${WORKER_TIMEOUT}s"
echo "  Run eagerly:      $RUN_EAGERLY"
echo "  Dispatch:         ${DISPATCH_INTERVAL_MS}ms"
echo "  Heartbeat TO:     ${HEARTBEAT_TIMEOUT_MS}ms"
echo "  Scheduler bind:   $SCHEDULER_LISTEN_ADDRESS"
echo "  Worker addr:      $SCHEDULER_PUBLIC_ADDRESS"
echo "  GA addr:          $GA_SCHEDULER_ADDRESS"
echo "  GA RPC TO:        ${GA_RPC_TIMEOUT_S}s"
echo "  GA Gen TO:        ${GA_GENERATION_TIMEOUT_S}s"
echo "  GA Max wall:      ${GA_MAX_WALL_CLOCK_S}s"
echo "  GA Exp workers:   ${GA_EXPECTED_WORKERS}"
echo "  TP workers:       ${THROUGHPUT_WORKER_COUNT}"
echo "  Net latency:      ${NETWORK_LATENCY_MS}ms"
echo "  Local workers:    $START_LOCAL_WORKERS"
echo "  Output name:      $OUTPUT_NAME"
echo "  Python:           $PYTHON"
echo ""

# Cleanup handler
cleanup() {
    echo ""
    echo "Shutting down..."
    for pid in "${WORKER_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
    kill "$SCHEDULER_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

# Start scheduler
"$SCHEDULER_BIN" \
    --listen-address "$SCHEDULER_LISTEN_ADDRESS" \
    --dispatch-interval "$DISPATCH_INTERVAL_MS" \
    --heartbeat-timeout "$HEARTBEAT_TIMEOUT_MS" \
    --worker-auth-key "$WORKER_AUTH_KEY" &
SCHEDULER_PID=$!
sleep 2

# Start workers
WORKER_PIDS=()
EAGER_FLAG=()
if [[ "$RUN_EAGERLY" == "1" ]]; then
    EAGER_FLAG+=(--run-eagerly)
fi
if [[ "$START_LOCAL_WORKERS" == "1" ]]; then
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        $PYTHON -m worker.main \
            --scheduler-address "$SCHEDULER_PUBLIC_ADDRESS" \
            --worker-id "stress-w${i}" \
            --gpu-memory-limit "$GPU_MEMORY_MB" \
            --train-subset-size "$TRAIN_SUBSET" \
            --val-subset-size "$VAL_SUBSET" \
            --timeout "$WORKER_TIMEOUT" \
            --auth-key "$WORKER_AUTH_KEY" \
            "${EAGER_FLAG[@]}" \
            --log-level WARNING &
        WORKER_PIDS+=($!)
    done
    sleep 3
else
    echo "Skipping local worker spawn (START_LOCAL_WORKERS=$START_LOCAL_WORKERS)"
    echo "Ensure remote workers are already connected to $SCHEDULER_PUBLIC_ADDRESS"
fi

# Run GA via run_ga.py
$PYTHON run_ga.py \
    --mode stress \
    --pop-size "$POP_SIZE" \
    --generations "$NUM_GENS" \
    --dataset "$DATASET" \
    --scheduler-address "$GA_SCHEDULER_ADDRESS" \
    --rpc-timeout "$GA_RPC_TIMEOUT_S" \
    --generation-timeout "$GA_GENERATION_TIMEOUT_S" \
    --progress-log-interval "$GA_PROGRESS_LOG_INTERVAL_S" \
    --max-wall-clock-s "$GA_MAX_WALL_CLOCK_S" \
    --expected-workers "$GA_EXPECTED_WORKERS" \
    --throughput-worker-count "$THROUGHPUT_WORKER_COUNT" \
    --network-latency-ms "$NETWORK_LATENCY_MS" \
    --output "$RESULTS_DIR/${OUTPUT_NAME}.json" \
    --summary-output "$RESULTS_DIR/${OUTPUT_NAME}_summary.json" \
    --log-level INFO

echo ""
echo "Stress test complete! Results in $RESULTS_DIR"
