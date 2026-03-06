#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

RESULTS_DIR="${RESULTS_DIR:-results/modal_benchmark_t4_w10}"
mkdir -p "$RESULTS_DIR"

SCHEDULER_BIN="${SCHEDULER_BIN:-./build-wsl/scheduler/epoch_scheduler}"
if [[ ! -x "$SCHEDULER_BIN" ]]; then
    echo "Scheduler binary not executable at $SCHEDULER_BIN"
    echo "Set SCHEDULER_BIN or build it first."
    exit 1
fi

LISTEN_ADDRESS="${LISTEN_ADDRESS:-0.0.0.0:50051}"
DISPATCH_INTERVAL_MS="${DISPATCH_INTERVAL_MS:-1}"
DISPATCH_STRATEGY="${DISPATCH_STRATEGY:-estimated_cost}"
HEARTBEAT_TIMEOUT_MS="${HEARTBEAT_TIMEOUT_MS:-120000}"
WORKER_AUTH_KEY="${WORKER_AUTH_KEY:-${EPOCH_WORKER_AUTH_KEY:-superkey}}"
METRICS_LOG_PATH="${METRICS_LOG_PATH:-$RESULTS_DIR/scheduler_metrics.jsonl}"

exec "$SCHEDULER_BIN" \
  --listen-address "$LISTEN_ADDRESS" \
  --dispatch-interval "$DISPATCH_INTERVAL_MS" \
  --dispatch-strategy "$DISPATCH_STRATEGY" \
  --heartbeat-timeout "$HEARTBEAT_TIMEOUT_MS" \
  --worker-auth-key "$WORKER_AUTH_KEY" \
  --metrics-log-path "$METRICS_LOG_PATH"
