# Epoch Startup (WSL, Optimized 10 Workers)

Use this order:
1. ngrok
2. Modal workers
3. Scheduler + GA

## Terminal 1: ngrok tunnel
```bash
cd "/mnt/a/Careers and Jobs/Coding/Epoch"
./tools/ngrok/ngrok.exe tcp 50051
```

Copy the forwarding address from ngrok, for example:
`tcp://4.tcp.ngrok.io:12167`

Use only:
`4.tcp.ngrok.io:12167`
for the worker command below.

## Terminal 2: Modal workers (10x T4)
```bash
cd "/mnt/a/Careers and Jobs/Coding/Epoch"
~/.local/bin/modal run --detach scripts/modal_workers.py::launch \
  --scheduler-address "4.tcp.ngrok.io:12167" \
  --gpu-type T4 \
  --num-workers 10 \
  --auth-key superkey \
  --heartbeat-interval 5 \
  --reconnect-delay-s 2 \
  --timeout-seconds 20 \
  --gc-every-n-jobs 3 \
  --train-subset-size 1536 \
  --val-subset-size 768 \
  --log-level INFO
```

## Terminal 3: Scheduler + GA

### Step 3A: start scheduler (same terminal)
```bash
cd "/mnt/a/Careers and Jobs/Coding/Epoch"
./build-wsl/scheduler/epoch_scheduler \
  --listen-address "0.0.0.0:50051" \
  --dispatch-interval 1 \
  --dispatch-strategy estimated_cost \
  --heartbeat-timeout 120000 \
  --worker-auth-key superkey \
  --metrics-log-path "results/modal_opt_t4_w10/scheduler_metrics.jsonl"
```

Keep this running. Open another tab/pane in Terminal 3 for GA.

### Step 3B: run GA (new tab/pane)
```bash
cd "/mnt/a/Careers and Jobs/Coding/Epoch"
poetry run python run_ga.py \
  --mode stress \
  --pop-size 72 \
  --generations 10 \
  --scheduler-address "localhost:50051" \
  --expected-workers 10 \
  --wait-for-idle-workers 10 \
  --readiness-timeout-s 240 \
  --metric-clock-source first_dispatch \
  --immigrant-rate 0.10 \
  --plateau-patience-gens 2 \
  --plateau-min-delta 0.00390625 \
  --plateau-immigrant-rate 0.20 \
  --plateau-mutation-rate-floor 0.20 \
  --genome-dedupe-max-retries 8 \
  --adaptive-narrowing-enabled 1 \
  --adaptive-narrowing-start-gen 2 \
  --adaptive-elite-fraction 0.25 \
  --adaptive-quantile 0.20 \
  --adaptive-min-span-ratio 0.20 \
  --adaptive-categorical-bias 0.75 \
  --late-epoch-bias 0.35 \
  --network-latency-ms 150 \
  --max-wall-clock-s 540 \
  --output "results/modal_opt_t4_w10/modal_opt_t4_w10_g10_p72.json" \
  --summary-output "results/modal_opt_t4_w10/modal_opt_t4_w10_g10_p72_summary.json"
```

## After run: ranking report
```bash
cd "/mnt/a/Careers and Jobs/Coding/Epoch"
poetry run python scripts/rank_modal_runs.py \
  --results-dir results/modal_opt_t4_w10 \
  --write-markdown
```

## Notes
- If GA errors with `UNIMPLEMENTED` for `GetSchedulerStatus`, rebuild scheduler:
```bash
cd "/mnt/a/Careers and Jobs/Coding/Epoch"
cmake -S . -B build-wsl -DCMAKE_BUILD_TYPE=Release
cmake --build build-wsl --target epoch_scheduler -j4
```
- Do not include `tcp://` in `--scheduler-address`.
- `run_ga.py` now auto-stops Modal app `epoch-workers` on exit to prevent idle GPU spend. Use `--no-modal-auto-stop` if you want workers to stay up for another run.

## Large Benchmark Baseline (1440 jobs, 10x T4)

This is the reproducible distributed baseline for cross-algorithm comparisons
using equal evaluation budget (`120 * 12 = 1440` jobs).

### Terminal 1: ngrok
```bash
cd "/mnt/a/Careers and Jobs/Coding/Epoch"
./tools/ngrok/ngrok.exe tcp 50051
```

### Terminal 2: Modal workers
```bash
cd "/mnt/a/Careers and Jobs/Coding/Epoch"
bash scripts/launch_modal_benchmark_workers.sh 0.tcp.ngrok.io:14768
```

### Terminal 3: scheduler
```bash
cd "/mnt/a/Careers and Jobs/Coding/Epoch"
bash scripts/start_modal_benchmark_scheduler.sh
```

### Terminal 4: GA run
```bash
cd "/mnt/a/Careers and Jobs/Coding/Epoch"
bash scripts/run_modal_benchmark_ga.sh
```

### Post-run: rank + validate acceptance checks
```bash
cd "/mnt/a/Careers and Jobs/Coding/Epoch"
bash scripts/rank_modal_benchmark.sh
python scripts/validate_modal_benchmark_summary.py
```

## Local Algorithm Benchmark (Native PC, Time-Matched)

This runs local algorithms against the stress search space using the
`g12_p120`-aligned settings and a fixed wall-clock budget.

```bash
cd "/mnt/a/Careers and Jobs/Coding/Epoch"
poetry run python -m benchmarks.runner \
  --space-mode stress \
  --dataset mnist \
  --seeds 42,43,44 \
  --max-wall-clock-s 531.9059779640011 \
  --budget 0 \
  --output-dir results/local_benchmark_g12_p120 \
  --train-subset-size 1536 \
  --val-subset-size 768 \
  --timeout-seconds 20 \
  --gc-every-n-jobs 3 \
  --deterministic-eval \
  --deterministic-seed-offset 0 \
  --reference-summary-path results/modal_benchmark_t4_w10/ga_benchmark_g12_p120_summary.json
```

Expected total duration for 5 algorithms x 3 seeds:
- about 133 minutes (plus small startup/teardown overhead)

## Repo Scrutiny + Tests (Pre-Push)

```bash
cd "/mnt/a/Careers and Jobs/Coding/Epoch"
poetry run pytest -q
gitleaks detect --source . --verbose
```

Optional lint check:
```bash
cd "/mnt/a/Careers and Jobs/Coding/Epoch"
poetry run ruff check .
```

## Benchmark Visualization Synthesis

Generate the 2 line charts + radar chart from local benchmark outputs and the
Modal GA reference:

```bash
cd "/mnt/a/Careers and Jobs/Coding/Epoch"
poetry run python scripts/plot_benchmark_comparison.py \
  --local-per-seed-dir results/local_benchmark_g12_p120/per_seed \
  --modal-run-json results/modal_benchmark_t4_w10/ga_benchmark_g12_p120.json \
  --modal-summary-json results/modal_benchmark_t4_w10/ga_benchmark_g12_p120_summary.json \
  --output-dir results/local_benchmark_g12_p120/figures \
  --bins 12
```

## Optional: Scale Modal Workers (e.g., 19 workers)

Use the launcher script with an override:

```bash
cd "/mnt/a/Careers and Jobs/Coding/Epoch"
NUM_WORKERS=19 bash scripts/launch_modal_benchmark_workers.sh 0.tcp.ngrok.io:14768
```

## Publish Graph Assets To README

`results/` is gitignored, so copy generated charts to a tracked folder before push:

```bash
cd "/mnt/a/Careers and Jobs/Coding/Epoch"
mkdir -p docs/figures
cp results/local_benchmark_g12_p120/figures/fitness_vs_generation_bins.png docs/figures/
cp results/local_benchmark_g12_p120/figures/runtime_vs_generation_bins.png docs/figures/
cp results/local_benchmark_g12_p120/figures/radar_comparison.png docs/figures/
```

## Push All Changes To Main

```bash
cd "/mnt/a/Careers and Jobs/Coding/Epoch"
poetry run pytest -q
gitleaks detect --source . --verbose
git add -A
git commit -m "Update benchmark docs, visuals, and latest codebase changes"
git push origin HEAD:main
```
