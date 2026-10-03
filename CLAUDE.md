# CLAUDE.md - Epoch Project Conventions

## Project Overview
Epoch is a distributed hyperparameter optimization system:
- C++ scheduler (`scheduler/`) handles job queueing, dispatch, and worker liveness.
- Python GA controller (`ga/`, `run_ga.py`) evolves hyperparameters generation-by-generation.
- Python workers (`worker/`) train TensorFlow/Keras models and report results over gRPC.
- Modal remote workers (`scripts/modal_workers.py`) can be used without changing scheduler code.

The scheduler/proto surface now includes exact scheduler-side runtime metrics:
- `dispatch_latency_ms`
- `worker_idle_gap_ms`
- `queue_wait_ms`
- `queue_wait_min_ms`

These are exposed via `GetGenerationResults.runtime_metrics` and can also be logged as JSONL from the scheduler.
Scheduler control also exposes readiness/liveness status via `GetSchedulerStatus`.

## Repository Layout
- `protos/` - protobuf/gRPC API definitions (source of truth)
- `scheduler/` - C++ scheduler
- `worker/` - Python training workers
- `ga/` - Python GA engine/controller
- `metrics/` - run metrics store/utilities
- `benchmarks/` - strategy comparisons
- `generated/` - generated protobuf bindings (tracked in repo)
- `scripts/` - run/ops tooling (stress, Modal launcher, ranking, proto generation)
- `results/` - local run outputs (gitignored)

## Build And Test

### Python
```bash
poetry install
bash scripts/generate_protos.sh
poetry run pytest
```

### C++ scheduler
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target epoch_scheduler -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

## Runtime Architecture

### gRPC services
- `WorkerService.Connect` (bidirectional stream):
  - worker -> scheduler: registration, heartbeat, training results
  - scheduler -> worker: job assignments, shutdown
- `SchedulerControl` (unary):
  - `SubmitGeneration`
  - `GetGenerationResults`
  - `GetSchedulerStatus`

### Scheduler threads
- Dispatch thread: assigns pending jobs to idle workers.
- Heartbeat thread: marks timed-out workers dead and fails in-flight jobs.

### Scheduler runtime telemetry
- Per-dispatch samples recorded in scheduler:
  - dispatch write latency
  - queue wait (enqueue -> dispatch)
  - worker idle gap (last processed result -> next dispatch for same worker)
- Per-generation summaries returned in `GetGenerationResults.runtime_metrics`:
  - p50/p90/max + sample counts for all three metrics
  - queue wait min (`queue_wait_min_ms`) for first-dispatch timing
- Optional JSONL event log:
  - CLI: `--metrics-log-path <path>`
  - env: `EPOCH_METRICS_LOG_PATH=<path>`

## Throughput Definition

`run_ga.py` supports two run-level clocks via `--metric-clock-source`:
- `first_dispatch` (default): excludes startup/warmup before first dispatch
- `total`: includes full process wall clock

Metric schema 2 reports successful completions divided by observed elapsed seconds.
`raw_jobs_per_sec` reports submissions per elapsed second. Legacy worker-multiplied
and network-subtracted formulas are labeled modeled compatibility metrics; neither
is observed throughput. See docs/DURABLE.md for the historical correction.

Defaults:
- `network_latency_ms=150`
- `metric_clock_source=first_dispatch`
- `throughput_worker_count=10`
- `modal_auto_stop=true` (stops Modal app `epoch-workers` unless overridden)
- `modal_auto_stop` failures are warning-only (run result remains valid)

Current GA defaults (plateau-resistant stress profile):
- `immigrant_rate=0.15`
- `plateau_patience_gens=2`
- `plateau_min_delta=1/1024`
- `plateau_immigrant_rate=0.35`
- `plateau_mutation_rate_floor=0.30`
- `plateau_reheat_gens=2`
- `plateau_reset_on_any_new_best=true`
- `adaptive_narrowing_start_gen=4`
- `adaptive_min_span_ratio=0.35`
- `adaptive_categorical_bias=0.60`
- `deterministic_eval=true`
- `deterministic_seed_offset=0`

Summary output includes:
- primary metrics (`jobs_per_sec`, `jobs_per_sec_total`, `jobs_per_sec_formula`)
- observed wall-clock metrics (`wall_clock_s`, `raw_jobs_per_sec`, `wall_clock_total_s`, `raw_jobs_per_sec_total`)
- explicit wall-adjusted throughput (`jobs_per_sec_wall_adjusted`, `jobs_per_sec_wall_adjusted_total`)
- offsets (`first_generation_submit_offset_s`, `first_dispatch_offset_s`)
- plateau observability (`plateau_events_count`, `reheat_generations_applied`, `stale_generations_final`)
- per-generation reheat marker (`generation_kpis[*].plateau_reheat_applied`)
- deterministic metadata (`deterministic_eval`, `deterministic_seed_offset`)
- Modal auto-stop fields (`modal_auto_stop_*`)

Legacy `throughput_worker_count` is retained as a compatibility field in summaries.

## Main Entry Points

### GA run
```bash
poetry run python run_ga.py --mode stress --pop-size 20 --generations 5
```

Important flags:
- `--max-wall-clock-s`
- `--expected-workers`
- `--wait-for-idle-workers`
- `--readiness-timeout-s`
- `--metric-clock-source`
- `--pop-per-worker`
- `--immigrant-rate`
- `--plateau-patience-gens`
- `--plateau-min-delta`
- `--plateau-immigrant-rate`
- `--plateau-mutation-rate-floor`
- `--plateau-reheat-gens`
- `--[no-]plateau-reset-on-any-new-best`
- `--adaptive-narrowing-*`
- `--deterministic-eval`
- `--deterministic-seed-offset`
- `--throughput-worker-count`
- `--network-latency-ms`
- `--[no-]modal-auto-stop`
- `--modal-app-name`
- `--modal-env`
- `--summary-output`

### Full stress orchestration
```bash
bash scripts/run_stress_test.sh
```

Key env overrides:
- scheduling: `DISPATCH_INTERVAL_MS`, `DISPATCH_STRATEGY`, `HEARTBEAT_TIMEOUT_MS`
- GA guards/readiness: `GA_MAX_WALL_CLOCK_S`, `GA_EXPECTED_WORKERS`, `GA_WAIT_FOR_IDLE_WORKERS`, `GA_READINESS_TIMEOUT_S`, `GA_POP_PER_WORKER`
- GA diversity/plateau: `GA_IMMIGRANT_RATE`, `GA_PLATEAU_PATIENCE_GENS`, `GA_PLATEAU_MIN_DELTA`, `GA_PLATEAU_IMMIGRANT_RATE`, `GA_PLATEAU_MUTATION_RATE_FLOOR`, `GA_PLATEAU_REHEAT_GENS`, `GA_PLATEAU_RESET_ON_ANY_NEW_BEST`
- GA adaptive search: `GA_ADAPTIVE_NARROWING_ENABLED`, `GA_ADAPTIVE_NARROWING_START_GEN`, `GA_ADAPTIVE_ELITE_FRACTION`, `GA_ADAPTIVE_QUANTILE`, `GA_ADAPTIVE_MIN_SPAN_RATIO`, `GA_ADAPTIVE_CATEGORICAL_BIAS`, `GA_GENOME_DEDUPE_MAX_RETRIES`
- throughput model: `THROUGHPUT_WORKER_COUNT`, `NETWORK_LATENCY_MS`
- metric source: `GA_METRIC_CLOCK_SOURCE`
- signal shaping: `SIGNAL_PRESET` with value `balanced` (optional, off by default), `GA_LATE_EPOCH_BIAS`
- worker determinism/runtime: `WORKER_DETERMINISTIC_EVAL`, `WORKER_DETERMINISTIC_SEED_OFFSET`, `GC_EVERY_N_JOBS`
- topology: `START_LOCAL_WORKERS`, `SCHEDULER_PUBLIC_ADDRESS`, `GA_SCHEDULER_ADDRESS`
- auth: `WORKER_AUTH_KEY` or `EPOCH_WORKER_AUTH_KEY`

### Modal remote workers
```bash
modal run scripts/modal_workers.py \
  --scheduler-address <public-host:port> \
  --gpu-type T4 \
  --num-workers 8 \
  --auth-key <shared-key> \
  --wait-for-workers
```

Notes:
- `scripts/modal_workers.py` currently binds Modal secret name `superkey` at function definition.
- Worker launch supports deterministic controls: `--deterministic-eval`, `--deterministic-seed-offset`.
- For production, avoid `superkey`; use a strong shared key and secret-managed envs.

## Plateau And Determinism Behavior
- New-best handling: with `plateau_reset_on_any_new_best=true`, stale generations reset whenever a new peak appears, even if gain is smaller than `plateau_min_delta`.
- Reheat handling: when plateau triggers, reheat persists for `plateau_reheat_gens` evolve steps (not single-shot), forcing immigrant/mutation overrides and adaptive-constraint re-expansion on each reheat generation.
- Deterministic eval: worker fitness evaluation can be seeded from a canonical genome hash (`deterministic_eval=true`), with optional integer offset (`deterministic_seed_offset`) for controlled stream shifts.

### Run ranking
```bash
poetry run python scripts/rank_modal_runs.py --results-dir results/<run_dir> --write-markdown
```

Ranking priority:
1. `jobs_per_sec` desc
2. `queue_overhead_p90_pct` asc
3. `avg_job_ms` asc
4. `best_fitness_peak` desc

### Local algorithm benchmark (native PC)
```bash
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

Local benchmark defaults mirror the `g12_p120` baseline constraints where applicable.
Algorithms included:
- Random Search
- Grid Search
- Bayesian (Optuna TPE)
- Differential Evolution
- Simulated Annealing

Artifacts:
- per-seed JSONs: `results/local_benchmark_g12_p120/per_seed/`
- aggregate summary: `results/local_benchmark_g12_p120/benchmark_aggregate_summary.json`
- ranked report: `results/local_benchmark_g12_p120/ranked_report.md`

Ranking for local algorithms is wall-clock-budgeted and sorted by median `best_fitness_peak` (desc),
with a reference-only Modal GA row included but excluded from ranking.

### Benchmark visualization synthesis
```bash
poetry run python scripts/plot_benchmark_comparison.py \
  --local-per-seed-dir results/local_benchmark_g12_p120/per_seed \
  --modal-run-json results/modal_benchmark_t4_w10/ga_benchmark_g12_p120.json \
  --modal-summary-json results/modal_benchmark_t4_w10/ga_benchmark_g12_p120_summary.json \
  --output-dir results/local_benchmark_g12_p120/figures \
  --bins 12
```

Generated outputs:
- `results/local_benchmark_g12_p120/figures/fitness_vs_generation_bins.png`
- `results/local_benchmark_g12_p120/figures/runtime_vs_generation_bins.png`
- `results/local_benchmark_g12_p120/figures/radar_comparison.png`
- `results/local_benchmark_g12_p120/figures/visualization_summary.md`

Radar behavior (current):
- axes: `Peak Fitness`, `Time to 90% Peak (min)`, `Total Run Time (min)`, `Total Evals Completed`, `Convergence AUC`
- direct raw/max normalization with display margin
- spoke-specific zoom expansion on `Peak Fitness`, `Total Run Time`, and `Convergence AUC` for readability

## Common Change Workflows

### Add a hyperparameter
1. Update `protos/epoch.proto` (`HyperparamConfig`).
2. Regenerate Python protos (`bash scripts/generate_protos.sh`) and rebuild C++.
3. Update `ga/search_space.py`.
4. Map in `ga/individual.py` (`to_protobuf` / `from_protobuf`).
5. Consume in `worker/model_builder.py` or `worker/trainer.py`.

### Add scheduler/runtime metric
1. Extend `protos/epoch.proto` response schema.
2. Update scheduler collector/service plumbing in `scheduler/`.
3. Regenerate/builder protobuf outputs.
4. Ingest and persist in `ga/controller.py` and `run_ga.py`.
5. Add/adjust Python + C++ tests.

## Security And Publishing
- Never commit keys/tokens/tunnel endpoints.
- Keep `.env*`, local logs, and `results/` out of git.
- `EPOCH_WORKER_AUTH_KEY` defaults to `superkey` only for development; override for any shared/public environment.

Pre-push scrutiny commands:
```bash
poetry run pytest -q
gitleaks detect --source . --verbose
```

Publish-to-README note:
- `results/` is gitignored. Copy chart PNGs to `docs/figures/` before pushing if README embeds them.
- Suggested push command when publishing current branch state directly to main:
```bash
git push origin HEAD:main
```

Notes:
- Prefer `gitleaks detect --source .` (git mode). `--no-git` can report large false-positive volumes from local build artifacts/dependency trees.
- `poetry run ruff check .` currently includes generated files and may flag style issues outside hand-authored code; treat as advisory unless generated-file policy changes.

## Style Conventions

### C++
- C++17, Google-style formatting via `.clang-format`
- PascalCase class/method names, trailing underscore members
- `std::mutex` for shared mutable state

### Python
- Black/Ruff conventions
- Type hints for non-trivial functions
- snake_case functions, PascalCase classes
- Keep controller/scheduler integration code explicit and test-backed
