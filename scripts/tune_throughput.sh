#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

RESULTS_DIR="${RESULTS_DIR:-results/throughput}"
CSV_PATH="${CSV_PATH:-$RESULTS_DIR/throughput_runs.csv}"
REPORT_PATH="${REPORT_PATH:-$RESULTS_DIR/throughput_report.md}"
PRESET_PATH="${PRESET_PATH:-$RESULTS_DIR/throughput_local_3070.env}"

SHORT_POP="${SHORT_POP:-20}"
SHORT_GENS="${SHORT_GENS:-3}"
MEDIUM_POP="${MEDIUM_POP:-100}"
MEDIUM_GENS="${MEDIUM_GENS:-10}"
LONG_POP="${LONG_POP:-100}"
LONG_GENS="${LONG_GENS:-20}"

FAILURE_RATE_LIMIT="${FAILURE_RATE_LIMIT:-0.10}"
RUN_TIMEOUT_SHORT_S="${RUN_TIMEOUT_SHORT_S:-900}"
RUN_TIMEOUT_MEDIUM_S="${RUN_TIMEOUT_MEDIUM_S:-3600}"
RUN_TIMEOUT_LONG_S="${RUN_TIMEOUT_LONG_S:-7200}"

mkdir -p "$RESULTS_DIR"

if command -v poetry &>/dev/null; then
    PYTHON_CMD=(poetry run python)
elif command -v python3 &>/dev/null; then
    PYTHON_CMD=(python3)
else
    PYTHON_CMD=(python)
fi

py() {
    "${PYTHON_CMD[@]}" "$@"
}

if [[ ! -f "$CSV_PATH" ]]; then
    echo "timestamp,stage,run_id,num_workers,gpu_memory_mb,train_subset,val_subset,worker_timeout,run_eagerly,dispatch_interval_ms,pop_size,num_gens,jobs_total,wall_clock_s,jobs_per_sec,avg_job_ms,timeouts,failures,failure_rate,p50_generation_ms,p90_generation_ms,exit_code,hang_detected,smoke_ok,log_path,summary_path" >"$CSV_PATH"
fi

LAST_JOBS_TOTAL=0
LAST_WALL_CLOCK_S=0
LAST_JPS=0
LAST_AVG_JOB_MS=0
LAST_TIMEOUTS=0
LAST_FAILURES=0
LAST_FAILURE_RATE=1
LAST_P50_GEN_MS=0
LAST_P90_GEN_MS=0
LAST_EXIT_CODE=0
LAST_HANG=0
LAST_SMOKE_OK=0
LAST_LOG_PATH=""
LAST_SUMMARY_PATH=""

python_ge() {
    py - "$1" "$2" <<'PY'
import sys
a = float(sys.argv[1])
b = float(sys.argv[2])
print(1 if a >= b else 0)
PY
}

python_le() {
    py - "$1" "$2" <<'PY'
import sys
a = float(sys.argv[1])
b = float(sys.argv[2])
print(1 if a <= b else 0)
PY
}

is_better_candidate() {
    py - "$1" "$2" "$3" "$4" "$FAILURE_RATE_LIMIT" <<'PY'
import sys
new_jps = float(sys.argv[1])
new_fr = float(sys.argv[2])
best_jps = float(sys.argv[3])
best_fr = float(sys.argv[4])
limit = float(sys.argv[5])

better = False
if new_fr <= limit and best_fr > limit:
    better = True
elif new_fr <= limit and best_fr <= limit and new_jps > best_jps:
    better = True
elif new_fr > limit and best_fr > limit and new_jps > best_jps:
    better = True

print(1 if better else 0)
PY
}

calc_failure_rate() {
    py - "$1" "$2" "$3" <<'PY'
import sys
jobs = int(float(sys.argv[1]))
timeouts = int(float(sys.argv[2]))
failures = int(float(sys.argv[3]))
if jobs <= 0:
    print("1.0")
else:
    print((timeouts + failures) / jobs)
PY
}

run_case() {
    local stage="$1"
    local run_id="$2"
    local num_workers="$3"
    local gpu_memory_mb="$4"
    local train_subset="$5"
    local val_subset="$6"
    local worker_timeout="$7"
    local run_eagerly="$8"
    local dispatch_interval_ms="$9"
    local pop_size="${10}"
    local num_gens="${11}"
    local run_timeout_s="${12}"

    local log_path="$RESULTS_DIR/${run_id}.log"
    local summary_path="$RESULTS_DIR/${run_id}_summary.json"
    local exit_code=0
    local hang_detected=0
    local smoke_ok=0

    set +e
    timeout "${run_timeout_s}s" env \
        NUM_WORKERS="$num_workers" \
        GPU_MEMORY_MB="$gpu_memory_mb" \
        TRAIN_SUBSET="$train_subset" \
        VAL_SUBSET="$val_subset" \
        WORKER_TIMEOUT="$worker_timeout" \
        RUN_EAGERLY="$run_eagerly" \
        DISPATCH_INTERVAL_MS="$dispatch_interval_ms" \
        HEARTBEAT_TIMEOUT_MS="30000" \
        POP_SIZE="$pop_size" \
        NUM_GENS="$num_gens" \
        RESULTS_DIR="$RESULTS_DIR" \
        OUTPUT_NAME="$run_id" \
        bash scripts/run_stress_test.sh >"$log_path" 2>&1
    exit_code=$?
    set -e

    if [[ "$exit_code" -eq 124 ]]; then
        hang_detected=1
    fi

    local jobs_total=0
    local wall_clock_s=0
    local jobs_per_sec=0
    local avg_job_ms=0
    local timeouts=0
    local failures=0
    local p50_generation_ms=0
    local p90_generation_ms=0

    if [[ -f "$summary_path" ]]; then
        read -r jobs_total wall_clock_s jobs_per_sec avg_job_ms timeouts failures p50_generation_ms p90_generation_ms <<<"$(py - "$summary_path" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)
print(
    int(data.get("jobs_total", 0)),
    float(data.get("wall_clock_s", 0.0)),
    float(data.get("jobs_per_sec", 0.0)),
    float(data.get("avg_job_ms", 0.0)),
    int(data.get("timeouts", 0)),
    int(data.get("failures", 0)),
    int(data.get("p50_generation_ms", 0)),
    int(data.get("p90_generation_ms", 0)),
)
PY
)"
    else
        jobs_total=$((pop_size * num_gens))
        wall_clock_s=0
        jobs_per_sec=0
        avg_job_ms=0
        timeouts=$jobs_total
        failures=0
    fi

    local failure_rate
    failure_rate="$(calc_failure_rate "$jobs_total" "$timeouts" "$failures")"

    if grep -q "Generation 0 complete" "$log_path"; then
        smoke_ok=1
    fi

    local timestamp
    timestamp="$(date -Iseconds)"
    echo "$timestamp,$stage,$run_id,$num_workers,$gpu_memory_mb,$train_subset,$val_subset,$worker_timeout,$run_eagerly,$dispatch_interval_ms,$pop_size,$num_gens,$jobs_total,$wall_clock_s,$jobs_per_sec,$avg_job_ms,$timeouts,$failures,$failure_rate,$p50_generation_ms,$p90_generation_ms,$exit_code,$hang_detected,$smoke_ok,$log_path,$summary_path" >>"$CSV_PATH"

    echo "[$stage] $run_id -> jps=$jobs_per_sec failure_rate=$failure_rate exit=$exit_code"

    LAST_JOBS_TOTAL="$jobs_total"
    LAST_WALL_CLOCK_S="$wall_clock_s"
    LAST_JPS="$jobs_per_sec"
    LAST_AVG_JOB_MS="$avg_job_ms"
    LAST_TIMEOUTS="$timeouts"
    LAST_FAILURES="$failures"
    LAST_FAILURE_RATE="$failure_rate"
    LAST_P50_GEN_MS="$p50_generation_ms"
    LAST_P90_GEN_MS="$p90_generation_ms"
    LAST_EXIT_CODE="$exit_code"
    LAST_HANG="$hang_detected"
    LAST_SMOKE_OK="$smoke_ok"
    LAST_LOG_PATH="$log_path"
    LAST_SUMMARY_PATH="$summary_path"
}

mean_and_std() {
    py - "$@" <<'PY'
import statistics
import sys
values = [float(x) for x in sys.argv[1:]]
if not values:
    print("0 0")
else:
    mean = statistics.mean(values)
    stddev = statistics.pstdev(values) if len(values) > 1 else 0.0
    print(mean, stddev)
PY
}

cpp_correctness_guard() {
    echo "Running C++ correctness guard..."
    if [[ ! -x "./build/scheduler/epoch_scheduler" ]]; then
        echo "ERROR: ./build/scheduler/epoch_scheduler not found or not executable"
        exit 1
    fi
    cmake --build build --target test_job_queue test_worker_pool -j"$(nproc)"
    ./build/scheduler/tests/test_job_queue >/dev/null
    ./build/scheduler/tests/test_worker_pool >/dev/null
    echo "C++ correctness guard passed."
}

cpp_correctness_guard

smoke_status="pass"
run_case "smoke" "smoke_baseline" 4 768 1000 500 12 0 10 5 2 "$RUN_TIMEOUT_SHORT_S"
if [[ "$LAST_SMOKE_OK" -ne 1 || "$LAST_HANG" -ne 0 ]]; then
    smoke_status="fail"
fi

baseline_jps=()
baseline_fr=()
for i in 1 2 3; do
    run_case "baseline" "baseline_${i}" 4 768 1000 500 12 0 10 "$SHORT_POP" "$SHORT_GENS" "$RUN_TIMEOUT_SHORT_S"
    baseline_jps+=("$LAST_JPS")
    baseline_fr+=("$LAST_FAILURE_RATE")
done

read -r baseline_jps_mean baseline_jps_std <<<"$(mean_and_std "${baseline_jps[@]}")"
read -r baseline_fr_mean baseline_fr_std <<<"$(mean_and_std "${baseline_fr[@]}")"

cfg_workers=4
cfg_mem=768
cfg_train_subset=1000
cfg_val_subset=500
cfg_timeout=12
cfg_eager=0
cfg_dispatch=10

short_best_jps=-1
short_best_fr=999
short_best_run_id=""

update_short_best() {
    local run_id="$1"
    local candidate_better
    candidate_better="$(is_better_candidate "$LAST_JPS" "$LAST_FAILURE_RATE" "$short_best_jps" "$short_best_fr")"
    if [[ "$candidate_better" -eq 1 ]]; then
        short_best_jps="$LAST_JPS"
        short_best_fr="$LAST_FAILURE_RATE"
        short_best_run_id="$run_id"
    fi
}

for w in 2 3 4 5 6; do
    run_id="workers_w${w}"
    run_case "sweep_workers" "$run_id" "$w" "$cfg_mem" "$cfg_train_subset" "$cfg_val_subset" "$cfg_timeout" "$cfg_eager" "$cfg_dispatch" "$SHORT_POP" "$SHORT_GENS" "$RUN_TIMEOUT_SHORT_S"
    update_short_best "$run_id"
done

best_value="$cfg_workers"
best_jps=-1
best_fr=999
for w in 2 3 4 5 6; do
    row_jps="$(py - "$CSV_PATH" "workers_w${w}" <<'PY'
import csv
import sys
path, run_id = sys.argv[1], sys.argv[2]
with open(path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["run_id"] == run_id:
            print(row["jobs_per_sec"], row["failure_rate"])
            break
PY
)"
    read -r jps fr <<<"$row_jps"
    better="$(is_better_candidate "$jps" "$fr" "$best_jps" "$best_fr")"
    if [[ "$better" -eq 1 ]]; then
        best_value="$w"
        best_jps="$jps"
        best_fr="$fr"
    fi
done
cfg_workers="$best_value"

for mem in 512 640 768 896; do
    run_id="memory_m${mem}"
    run_case "sweep_memory" "$run_id" "$cfg_workers" "$mem" "$cfg_train_subset" "$cfg_val_subset" "$cfg_timeout" "$cfg_eager" "$cfg_dispatch" "$SHORT_POP" "$SHORT_GENS" "$RUN_TIMEOUT_SHORT_S"
    update_short_best "$run_id"
done
best_value="$cfg_mem"
best_jps=-1
best_fr=999
for mem in 512 640 768 896; do
    row_jps="$(py - "$CSV_PATH" "memory_m${mem}" <<'PY'
import csv
import sys
path, run_id = sys.argv[1], sys.argv[2]
with open(path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["run_id"] == run_id:
            print(row["jobs_per_sec"], row["failure_rate"])
            break
PY
)"
    read -r jps fr <<<"$row_jps"
    better="$(is_better_candidate "$jps" "$fr" "$best_jps" "$best_fr")"
    if [[ "$better" -eq 1 ]]; then
        best_value="$mem"
        best_jps="$jps"
        best_fr="$fr"
    fi
done
cfg_mem="$best_value"

for eager in 0 1; do
    run_id="eager_e${eager}"
    run_case "sweep_eager" "$run_id" "$cfg_workers" "$cfg_mem" "$cfg_train_subset" "$cfg_val_subset" "$cfg_timeout" "$eager" "$cfg_dispatch" "$SHORT_POP" "$SHORT_GENS" "$RUN_TIMEOUT_SHORT_S"
    update_short_best "$run_id"
done
best_value="$cfg_eager"
best_jps=-1
best_fr=999
for eager in 0 1; do
    row_jps="$(py - "$CSV_PATH" "eager_e${eager}" <<'PY'
import csv
import sys
path, run_id = sys.argv[1], sys.argv[2]
with open(path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["run_id"] == run_id:
            print(row["jobs_per_sec"], row["failure_rate"])
            break
PY
)"
    read -r jps fr <<<"$row_jps"
    better="$(is_better_candidate "$jps" "$fr" "$best_jps" "$best_fr")"
    if [[ "$better" -eq 1 ]]; then
        best_value="$eager"
        best_jps="$jps"
        best_fr="$fr"
    fi
done
cfg_eager="$best_value"

subset_pairs=("512 256" "768 384" "1000 500")
for pair in "${subset_pairs[@]}"; do
    read -r train_subset val_subset <<<"$pair"
    run_id="subset_t${train_subset}_v${val_subset}"
    run_case "sweep_subset" "$run_id" "$cfg_workers" "$cfg_mem" "$train_subset" "$val_subset" "$cfg_timeout" "$cfg_eager" "$cfg_dispatch" "$SHORT_POP" "$SHORT_GENS" "$RUN_TIMEOUT_SHORT_S"
    update_short_best "$run_id"
done
best_train="$cfg_train_subset"
best_val="$cfg_val_subset"
best_jps=-1
best_fr=999
for pair in "${subset_pairs[@]}"; do
    read -r train_subset val_subset <<<"$pair"
    run_id="subset_t${train_subset}_v${val_subset}"
    row_jps="$(py - "$CSV_PATH" "$run_id" <<'PY'
import csv
import sys
path, run_id = sys.argv[1], sys.argv[2]
with open(path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["run_id"] == run_id:
            print(row["jobs_per_sec"], row["failure_rate"])
            break
PY
)"
    read -r jps fr <<<"$row_jps"
    better="$(is_better_candidate "$jps" "$fr" "$best_jps" "$best_fr")"
    if [[ "$better" -eq 1 ]]; then
        best_train="$train_subset"
        best_val="$val_subset"
        best_jps="$jps"
        best_fr="$fr"
    fi
done
cfg_train_subset="$best_train"
cfg_val_subset="$best_val"

for timeout_s in 8 10 12 15; do
    run_id="timeout_s${timeout_s}"
    run_case "sweep_timeout" "$run_id" "$cfg_workers" "$cfg_mem" "$cfg_train_subset" "$cfg_val_subset" "$timeout_s" "$cfg_eager" "$cfg_dispatch" "$SHORT_POP" "$SHORT_GENS" "$RUN_TIMEOUT_SHORT_S"
    update_short_best "$run_id"
done
best_value="$cfg_timeout"
best_jps=-1
best_fr=999
for timeout_s in 8 10 12 15; do
    run_id="timeout_s${timeout_s}"
    row_jps="$(py - "$CSV_PATH" "$run_id" <<'PY'
import csv
import sys
path, run_id = sys.argv[1], sys.argv[2]
with open(path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["run_id"] == run_id:
            print(row["jobs_per_sec"], row["failure_rate"])
            break
PY
)"
    read -r jps fr <<<"$row_jps"
    better="$(is_better_candidate "$jps" "$fr" "$best_jps" "$best_fr")"
    if [[ "$better" -eq 1 ]]; then
        best_value="$timeout_s"
        best_jps="$jps"
        best_fr="$fr"
    fi
done
cfg_timeout="$best_value"

for dispatch_ms in 10 5 2 1; do
    run_id="dispatch_d${dispatch_ms}"
    run_case "sweep_dispatch" "$run_id" "$cfg_workers" "$cfg_mem" "$cfg_train_subset" "$cfg_val_subset" "$cfg_timeout" "$cfg_eager" "$dispatch_ms" "$SHORT_POP" "$SHORT_GENS" "$RUN_TIMEOUT_SHORT_S"
    update_short_best "$run_id"
done
best_value="$cfg_dispatch"
best_jps=-1
best_fr=999
for dispatch_ms in 10 5 2 1; do
    run_id="dispatch_d${dispatch_ms}"
    row_jps="$(py - "$CSV_PATH" "$run_id" <<'PY'
import csv
import sys
path, run_id = sys.argv[1], sys.argv[2]
with open(path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["run_id"] == run_id:
            print(row["jobs_per_sec"], row["failure_rate"])
            break
PY
)"
    read -r jps fr <<<"$row_jps"
    better="$(is_better_candidate "$jps" "$fr" "$best_jps" "$best_fr")"
    if [[ "$better" -eq 1 ]]; then
        best_value="$dispatch_ms"
        best_jps="$jps"
        best_fr="$fr"
    fi
done
cfg_dispatch="$best_value"

cat >"$PRESET_PATH" <<EOF
# Preset: throughput_local_3070
# Generated by scripts/tune_throughput.sh on $(date -Iseconds)
NUM_WORKERS=$cfg_workers
GPU_MEMORY_MB=$cfg_mem
TRAIN_SUBSET=$cfg_train_subset
VAL_SUBSET=$cfg_val_subset
WORKER_TIMEOUT=$cfg_timeout
RUN_EAGERLY=$cfg_eager
DISPATCH_INTERVAL_MS=$cfg_dispatch
HEARTBEAT_TIMEOUT_MS=30000
EOF

run_case "failure_mode" "failure_mode_aggressive" 8 384 1000 500 8 0 "$cfg_dispatch" "$SHORT_POP" "$SHORT_GENS" "$RUN_TIMEOUT_SHORT_S"
failure_mode_hang="$LAST_HANG"

run_case "medium_validation" "medium_validation" "$cfg_workers" "$cfg_mem" "$cfg_train_subset" "$cfg_val_subset" "$cfg_timeout" "$cfg_eager" "$cfg_dispatch" "$MEDIUM_POP" "$MEDIUM_GENS" "$RUN_TIMEOUT_MEDIUM_S"
medium_jps="$LAST_JPS"
medium_fr="$LAST_FAILURE_RATE"
medium_hang="$LAST_HANG"
medium_smoke="$LAST_SMOKE_OK"

repeat_jps=()
for i in 1 2 3; do
    run_case "repeatability" "repeatability_${i}" "$cfg_workers" "$cfg_mem" "$cfg_train_subset" "$cfg_val_subset" "$cfg_timeout" "$cfg_eager" "$cfg_dispatch" "$SHORT_POP" "$SHORT_GENS" "$RUN_TIMEOUT_SHORT_S"
    repeat_jps+=("$LAST_JPS")
done
read -r repeat_mean repeat_std <<<"$(mean_and_std "${repeat_jps[@]}")"
repeat_cv="$(py - "$repeat_mean" "$repeat_std" <<'PY'
import sys
mean = float(sys.argv[1])
std = float(sys.argv[2])
print(0.0 if mean <= 0 else std / mean)
PY
)"

run_case "long_stress" "long_stress" "$cfg_workers" "$cfg_mem" "$cfg_train_subset" "$cfg_val_subset" "$cfg_timeout" "$cfg_eager" "$cfg_dispatch" "$LONG_POP" "$LONG_GENS" "$RUN_TIMEOUT_LONG_S"
long_jps="$LAST_JPS"
long_fr="$LAST_FAILURE_RATE"
long_p50="$LAST_P50_GEN_MS"
long_p90="$LAST_P90_GEN_MS"
long_hang="$LAST_HANG"

medium_threshold="$(py - "$short_best_jps" <<'PY'
import sys
print(float(sys.argv[1]) * 0.9)
PY
)"
medium_pass=1
if [[ "$(python_le "$medium_fr" "$FAILURE_RATE_LIMIT")" -ne 1 ]]; then
    medium_pass=0
fi
if [[ "$(python_ge "$medium_jps" "$medium_threshold")" -ne 1 ]]; then
    medium_pass=0
fi
if [[ "$medium_hang" -ne 0 || "$medium_smoke" -ne 1 ]]; then
    medium_pass=0
fi

repeat_pass=0
if [[ "$(python_le "$repeat_cv" "0.15")" -eq 1 ]]; then
    repeat_pass=1
fi

hardware="unknown"
if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_line="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -n 1 || true)"
    if [[ -n "$gpu_line" ]]; then
        hardware="$gpu_line"
    fi
fi

cat >"$REPORT_PATH" <<EOF
# Throughput Report

## Hardware
- GPU: $hardware
- Target profile: throughput_local_3070

## Selected Knobs
- NUM_WORKERS: $cfg_workers
- GPU_MEMORY_MB: $cfg_mem
- TRAIN_SUBSET: $cfg_train_subset
- VAL_SUBSET: $cfg_val_subset
- WORKER_TIMEOUT: $cfg_timeout
- RUN_EAGERLY: $cfg_eager
- DISPATCH_INTERVAL_MS: $cfg_dispatch
- HEARTBEAT_TIMEOUT_MS: 30000

## Baseline (3 runs)
- Jobs/sec mean: $baseline_jps_mean
- Jobs/sec stddev: $baseline_jps_std
- Failure-rate mean: $baseline_fr_mean
- Failure-rate stddev: $baseline_fr_std

## Sweep Winner
- Best short-run ID: $short_best_run_id
- Best short-run jobs/sec: $short_best_jps
- Best short-run failure-rate: $short_best_fr

## Acceptance Checks
- Architecture smoke: $smoke_status
- Failure-mode hang check (8 workers, low memory): $( [[ "$failure_mode_hang" -eq 0 ]] && echo "pass" || echo "fail" )
- Medium validation pass: $( [[ "$medium_pass" -eq 1 ]] && echo "pass" || echo "fail" )
- Repeatability pass (std/mean <= 15%): $( [[ "$repeat_pass" -eq 1 ]] && echo "pass" || echo "fail" )

## Medium Validation (POP_SIZE=$MEDIUM_POP, NUM_GENS=$MEDIUM_GENS)
- Jobs/sec: $medium_jps
- Failure rate: $medium_fr
- Min acceptable jobs/sec (90% of short winner): $medium_threshold

## Long Stress (POP_SIZE=$LONG_POP, NUM_GENS=$LONG_GENS)
- Jobs/sec: $long_jps
- Failure rate: $long_fr
- P50 generation time (ms): $long_p50
- P90 generation time (ms): $long_p90
- Hang detected: $long_hang

## Stability Notes
- Throughput and failure metrics captured for every run in CSV.
- All run logs and summaries are saved under \`$RESULTS_DIR\`.

## Artifacts
- CSV: \`$CSV_PATH\`
- Preset: \`$PRESET_PATH\`
- Report: \`$REPORT_PATH\`
EOF

echo "Throughput tuning complete."
echo "CSV:    $CSV_PATH"
echo "Preset: $PRESET_PATH"
echo "Report: $REPORT_PATH"
