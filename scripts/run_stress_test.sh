#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Configuration (override via environment variables)
NUM_WORKERS="${NUM_WORKERS:-4}"
GPU_MEMORY_MB="${GPU_MEMORY_MB:-1536}"
TRAIN_SUBSET_INPUT="${TRAIN_SUBSET:-}"
VAL_SUBSET_INPUT="${VAL_SUBSET:-}"
TRAIN_SUBSET="${TRAIN_SUBSET_INPUT:-1000}"
VAL_SUBSET="${VAL_SUBSET_INPUT:-500}"
POP_SIZE="${POP_SIZE:-100}"
NUM_GENS="${NUM_GENS:-50}"
DATASET="${DATASET:-mnist}"
WORKER_TIMEOUT="${WORKER_TIMEOUT:-12}"
RUN_EAGERLY="${RUN_EAGERLY:-0}"
GC_EVERY_N_JOBS="${GC_EVERY_N_JOBS:-1}"
DISPATCH_INTERVAL_MS="${DISPATCH_INTERVAL_MS:-10}"
DISPATCH_STRATEGY="${DISPATCH_STRATEGY:-fifo}"
HEARTBEAT_TIMEOUT_MS="${HEARTBEAT_TIMEOUT_MS:-30000}"
SCHEDULER_LISTEN_ADDRESS="${SCHEDULER_LISTEN_ADDRESS:-0.0.0.0:50051}"
SCHEDULER_PUBLIC_ADDRESS="${SCHEDULER_PUBLIC_ADDRESS:-localhost:50051}"
GA_SCHEDULER_ADDRESS="${GA_SCHEDULER_ADDRESS:-localhost:50051}"
GA_RPC_TIMEOUT_S="${GA_RPC_TIMEOUT_S:-10}"
GA_GENERATION_TIMEOUT_S="${GA_GENERATION_TIMEOUT_S:-300}"
GA_PROGRESS_LOG_INTERVAL_S="${GA_PROGRESS_LOG_INTERVAL_S:-10}"
GA_MAX_WALL_CLOCK_S="${GA_MAX_WALL_CLOCK_S:-0}"
GA_EXPECTED_WORKERS="${GA_EXPECTED_WORKERS:-0}"
GA_POP_PER_WORKER="${GA_POP_PER_WORKER:-0}"
GA_IMMIGRANT_RATE="${GA_IMMIGRANT_RATE:-0.15}"
GA_PLATEAU_PATIENCE_GENS="${GA_PLATEAU_PATIENCE_GENS:-2}"
GA_PLATEAU_MIN_DELTA="${GA_PLATEAU_MIN_DELTA:-0.0009765625}"
GA_PLATEAU_IMMIGRANT_RATE="${GA_PLATEAU_IMMIGRANT_RATE:-0.35}"
GA_PLATEAU_MUTATION_RATE_FLOOR="${GA_PLATEAU_MUTATION_RATE_FLOOR:-0.30}"
GA_PLATEAU_REHEAT_GENS="${GA_PLATEAU_REHEAT_GENS:-2}"
GA_PLATEAU_RESET_ON_ANY_NEW_BEST="${GA_PLATEAU_RESET_ON_ANY_NEW_BEST:-1}"
GA_GENOME_DEDUPE_MAX_RETRIES="${GA_GENOME_DEDUPE_MAX_RETRIES:-8}"
GA_ADAPTIVE_NARROWING_ENABLED="${GA_ADAPTIVE_NARROWING_ENABLED:-1}"
GA_ADAPTIVE_NARROWING_START_GEN="${GA_ADAPTIVE_NARROWING_START_GEN:-4}"
GA_ADAPTIVE_ELITE_FRACTION="${GA_ADAPTIVE_ELITE_FRACTION:-0.25}"
GA_ADAPTIVE_QUANTILE="${GA_ADAPTIVE_QUANTILE:-0.20}"
GA_ADAPTIVE_MIN_SPAN_RATIO="${GA_ADAPTIVE_MIN_SPAN_RATIO:-0.35}"
GA_ADAPTIVE_CATEGORICAL_BIAS="${GA_ADAPTIVE_CATEGORICAL_BIAS:-0.60}"
GA_LATE_EPOCH_BIAS_INPUT="${GA_LATE_EPOCH_BIAS:-}"
GA_LATE_EPOCH_BIAS="${GA_LATE_EPOCH_BIAS_INPUT:-0.0}"
GA_WAIT_FOR_IDLE_WORKERS="${GA_WAIT_FOR_IDLE_WORKERS:-0}"
GA_READINESS_TIMEOUT_S="${GA_READINESS_TIMEOUT_S:-180}"
GA_METRIC_CLOCK_SOURCE="${GA_METRIC_CLOCK_SOURCE:-first_dispatch}"
THROUGHPUT_WORKER_COUNT="${THROUGHPUT_WORKER_COUNT:-10}"
NETWORK_LATENCY_MS="${NETWORK_LATENCY_MS:-150}"
START_LOCAL_WORKERS="${START_LOCAL_WORKERS:-1}"
WORKER_AUTH_KEY="${WORKER_AUTH_KEY:-${EPOCH_WORKER_AUTH_KEY:-superkey}}"
WORKER_DETERMINISTIC_EVAL="${WORKER_DETERMINISTIC_EVAL:-1}"
WORKER_DETERMINISTIC_SEED_OFFSET="${WORKER_DETERMINISTIC_SEED_OFFSET:-0}"
RESULTS_DIR="${RESULTS_DIR:-results/stress_test}"
OUTPUT_NAME="${OUTPUT_NAME:-stress_test}"
SIGNAL_PRESET="${SIGNAL_PRESET:-default}"

if [[ "$SIGNAL_PRESET" == "balanced" ]]; then
    if [[ -z "$TRAIN_SUBSET_INPUT" ]]; then
        TRAIN_SUBSET=1536
    fi
    if [[ -z "$VAL_SUBSET_INPUT" ]]; then
        VAL_SUBSET=768
    fi
    if [[ -z "$GA_LATE_EPOCH_BIAS_INPUT" ]]; then
        GA_LATE_EPOCH_BIAS=0.35
    fi
fi

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
echo "  Signal preset:    $SIGNAL_PRESET"
echo "  Population:       $POP_SIZE"
echo "  Generations:      $NUM_GENS"
echo "  Dataset:          $DATASET"
echo "  Worker timeout:   ${WORKER_TIMEOUT}s"
echo "  Run eagerly:      $RUN_EAGERLY"
echo "  GC every N jobs:  $GC_EVERY_N_JOBS"
echo "  Dispatch:         ${DISPATCH_INTERVAL_MS}ms"
echo "  Dispatch strat:   ${DISPATCH_STRATEGY}"
echo "  Heartbeat TO:     ${HEARTBEAT_TIMEOUT_MS}ms"
echo "  Scheduler bind:   $SCHEDULER_LISTEN_ADDRESS"
echo "  Worker addr:      $SCHEDULER_PUBLIC_ADDRESS"
echo "  GA addr:          $GA_SCHEDULER_ADDRESS"
echo "  GA RPC TO:        ${GA_RPC_TIMEOUT_S}s"
echo "  GA Gen TO:        ${GA_GENERATION_TIMEOUT_S}s"
echo "  GA Max wall:      ${GA_MAX_WALL_CLOCK_S}s"
echo "  GA Exp workers:   ${GA_EXPECTED_WORKERS}"
echo "  GA pop/worker:    ${GA_POP_PER_WORKER}"
echo "  GA immigrant:     ${GA_IMMIGRANT_RATE}"
echo "  GA plateau:       patience=${GA_PLATEAU_PATIENCE_GENS} delta=${GA_PLATEAU_MIN_DELTA} imm_rate=${GA_PLATEAU_IMMIGRANT_RATE} mut_floor=${GA_PLATEAU_MUTATION_RATE_FLOOR} reheat=${GA_PLATEAU_REHEAT_GENS} reset_any_new_best=${GA_PLATEAU_RESET_ON_ANY_NEW_BEST}"
echo "  GA dedupe retries:${GA_GENOME_DEDUPE_MAX_RETRIES}"
echo "  GA adaptive:      enabled=${GA_ADAPTIVE_NARROWING_ENABLED} start=${GA_ADAPTIVE_NARROWING_START_GEN} elite_frac=${GA_ADAPTIVE_ELITE_FRACTION} q=${GA_ADAPTIVE_QUANTILE}"
echo "  GA cat bias:      ${GA_ADAPTIVE_CATEGORICAL_BIAS}"
echo "  GA late epoch:    ${GA_LATE_EPOCH_BIAS}"
echo "  Worker determinism: eval=${WORKER_DETERMINISTIC_EVAL} seed_offset=${WORKER_DETERMINISTIC_SEED_OFFSET}"
echo "  GA readiness:     idle>=${GA_WAIT_FOR_IDLE_WORKERS} timeout=${GA_READINESS_TIMEOUT_S}s"
echo "  Metric clock:     ${GA_METRIC_CLOCK_SOURCE}"
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
    --dispatch-strategy "$DISPATCH_STRATEGY" \
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
DETERMINISTIC_FLAG=()
if [[ "$WORKER_DETERMINISTIC_EVAL" == "0" ]]; then
    DETERMINISTIC_FLAG+=(--no-deterministic-eval)
else
    DETERMINISTIC_FLAG+=(--deterministic-eval)
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
            --gc-every-n-jobs "$GC_EVERY_N_JOBS" \
            --deterministic-seed-offset "$WORKER_DETERMINISTIC_SEED_OFFSET" \
            --auth-key "$WORKER_AUTH_KEY" \
            "${DETERMINISTIC_FLAG[@]}" \
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
GA_PLATEAU_RESET_FLAG=()
if [[ "$GA_PLATEAU_RESET_ON_ANY_NEW_BEST" == "0" ]]; then
    GA_PLATEAU_RESET_FLAG+=(--no-plateau-reset-on-any-new-best)
else
    GA_PLATEAU_RESET_FLAG+=(--plateau-reset-on-any-new-best)
fi

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
    --pop-per-worker "$GA_POP_PER_WORKER" \
    --immigrant-rate "$GA_IMMIGRANT_RATE" \
    --plateau-patience-gens "$GA_PLATEAU_PATIENCE_GENS" \
    --plateau-min-delta "$GA_PLATEAU_MIN_DELTA" \
    --plateau-immigrant-rate "$GA_PLATEAU_IMMIGRANT_RATE" \
    --plateau-mutation-rate-floor "$GA_PLATEAU_MUTATION_RATE_FLOOR" \
    --plateau-reheat-gens "$GA_PLATEAU_REHEAT_GENS" \
    "${GA_PLATEAU_RESET_FLAG[@]}" \
    --genome-dedupe-max-retries "$GA_GENOME_DEDUPE_MAX_RETRIES" \
    --adaptive-narrowing-enabled "$GA_ADAPTIVE_NARROWING_ENABLED" \
    --adaptive-narrowing-start-gen "$GA_ADAPTIVE_NARROWING_START_GEN" \
    --adaptive-elite-fraction "$GA_ADAPTIVE_ELITE_FRACTION" \
    --adaptive-quantile "$GA_ADAPTIVE_QUANTILE" \
    --adaptive-min-span-ratio "$GA_ADAPTIVE_MIN_SPAN_RATIO" \
    --adaptive-categorical-bias "$GA_ADAPTIVE_CATEGORICAL_BIAS" \
    --late-epoch-bias "$GA_LATE_EPOCH_BIAS" \
    --deterministic-seed-offset "$WORKER_DETERMINISTIC_SEED_OFFSET" \
    "${DETERMINISTIC_FLAG[@]}" \
    --wait-for-idle-workers "$GA_WAIT_FOR_IDLE_WORKERS" \
    --readiness-timeout-s "$GA_READINESS_TIMEOUT_S" \
    --metric-clock-source "$GA_METRIC_CLOCK_SOURCE" \
    --throughput-worker-count "$THROUGHPUT_WORKER_COUNT" \
    --network-latency-ms "$NETWORK_LATENCY_MS" \
    --output "$RESULTS_DIR/${OUTPUT_NAME}.json" \
    --summary-output "$RESULTS_DIR/${OUTPUT_NAME}_summary.json" \
    --log-level INFO

echo ""
echo "Stress test complete! Results in $RESULTS_DIR"
