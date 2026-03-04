#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NUM_WORKERS=8

echo "Starting Epoch (WSL + CUDA)..."
mkdir -p results

# Skip XLA autotuning — not worth the overhead for GA hyperparameter search
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"

# Start scheduler in background
echo "Starting scheduler..."
"$SCRIPT_DIR/build-wsl/scheduler/epoch_scheduler" --listen-address 0.0.0.0:50051 &
SCHEDULER_PID=$!
sleep 3

# Start workers in background
WORKER_PIDS=()
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    echo "Starting worker w$i..."
    poetry run python -m worker.main \
        --scheduler-address localhost:50051 \
        --worker-id "w$i" \
        --log-level INFO &
    WORKER_PIDS+=($!)
done
sleep 5

# Run GA controller
echo "Running GA controller..."
poetry run python run_ga.py
echo "Done. Results saved to results/my_run.json"

# Cleanup
echo "Shutting down..."
kill "$SCHEDULER_PID" 2>/dev/null || true
for pid in "${WORKER_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
done
wait 2>/dev/null || true
echo "All processes stopped."
