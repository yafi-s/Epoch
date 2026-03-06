#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

SCHEDULER_ADDRESS="${1:-${SCHEDULER_ADDRESS:-}}"
if [[ -z "$SCHEDULER_ADDRESS" ]]; then
    echo "Usage: $0 <scheduler-host:port>"
    echo "Example: $0 0.tcp.ngrok.io:14768"
    exit 1
fi

MODAL_BIN="${MODAL_BIN:-$HOME/.local/bin/modal}"
GPU_TYPE="${GPU_TYPE:-T4}"
NUM_WORKERS="${NUM_WORKERS:-10}"
WORKER_AUTH_KEY="${WORKER_AUTH_KEY:-${EPOCH_WORKER_AUTH_KEY:-superkey}}"
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-5}"
RECONNECT_DELAY_S="${RECONNECT_DELAY_S:-2}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-20}"
GC_EVERY_N_JOBS="${GC_EVERY_N_JOBS:-3}"
TRAIN_SUBSET_SIZE="${TRAIN_SUBSET_SIZE:-1536}"
VAL_SUBSET_SIZE="${VAL_SUBSET_SIZE:-768}"
DETERMINISTIC_SEED_OFFSET="${DETERMINISTIC_SEED_OFFSET:-0}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

"$MODAL_BIN" run --detach scripts/modal_workers.py::launch \
  --scheduler-address "$SCHEDULER_ADDRESS" \
  --gpu-type "$GPU_TYPE" \
  --num-workers "$NUM_WORKERS" \
  --auth-key "$WORKER_AUTH_KEY" \
  --heartbeat-interval "$HEARTBEAT_INTERVAL" \
  --reconnect-delay-s "$RECONNECT_DELAY_S" \
  --timeout-seconds "$TIMEOUT_SECONDS" \
  --gc-every-n-jobs "$GC_EVERY_N_JOBS" \
  --train-subset-size "$TRAIN_SUBSET_SIZE" \
  --val-subset-size "$VAL_SUBSET_SIZE" \
  --deterministic-eval \
  --deterministic-seed-offset "$DETERMINISTIC_SEED_OFFSET" \
  --log-level "$LOG_LEVEL"
