"""Main entry point for Epoch distributed hyperparameter optimization.

Usage:
    # Default stress-test mode (small models, high throughput)
    python run_ga.py

    # Custom configuration
    python run_ga.py --pop-size 50 --generations 25

    # Full-scale optimization (larger models, full dataset)
    python run_ga.py --mode full --pop-size 20 --generations 10 --epochs 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, int(len(sorted_values) * pct))
    return float(sorted_values[index])


def _compute_adjusted_throughput(
    *,
    wall_clock_s: float,
    jobs_submitted: int,
    throughput_worker_count: int,
    network_latency_ms: float,
) -> tuple[float, float, float]:
    """Return (avg_job_ms_raw, avg_job_ms_adjusted, jobs_per_sec)."""
    if jobs_submitted <= 0:
        return 0.0, 1.0, 0.0
    avg_job_ms_raw = (wall_clock_s / float(jobs_submitted)) * 1000.0
    avg_job_ms_adjusted = max(avg_job_ms_raw - network_latency_ms, 1.0)
    jobs_per_sec = (float(throughput_worker_count) * 1000.0) / avg_job_ms_adjusted
    return avg_job_ms_raw, avg_job_ms_adjusted, jobs_per_sec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Epoch: distributed hyperparameter optimization via genetic algorithm"
    )
    parser.add_argument(
        "--mode",
        choices=["stress", "full"],
        default="stress",
        help="Search space mode: 'stress' for fast throughput testing, 'full' for real optimization "
        "(default: stress)",
    )
    parser.add_argument("--pop-size", type=int, default=100, help="Population size (default: 100)")
    parser.add_argument(
        "--generations", type=int, default=50, help="Number of generations (default: 50)"
    )
    parser.add_argument("--dataset", default="mnist", help="Dataset: mnist or cifar10")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs per individual")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument(
        "--scheduler-address",
        default=os.environ.get("EPOCH_SCHEDULER_ADDRESS", "localhost:50051"),
        help="Scheduler gRPC address",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=0.1, help="Result poll interval in seconds"
    )
    parser.add_argument(
        "--rpc-timeout",
        type=float,
        default=10.0,
        help="Per-RPC timeout in seconds for scheduler calls (default: 10)",
    )
    parser.add_argument(
        "--generation-timeout",
        type=float,
        default=300.0,
        help="Max wait per generation in seconds before failing (0 disables, default: 300)",
    )
    parser.add_argument(
        "--progress-log-interval",
        type=float,
        default=10.0,
        help="Seconds between waiting-progress logs while polling results (default: 10)",
    )
    parser.add_argument(
        "--max-wall-clock-s",
        type=float,
        default=0.0,
        help="Hard run wall-clock cap in seconds; checked at generation boundaries (0 disables)",
    )
    parser.add_argument(
        "--expected-workers",
        type=int,
        default=0,
        help="Minimum expected active workers per generation (0 disables)",
    )
    parser.add_argument(
        "--throughput-worker-count",
        type=int,
        default=8,
        help="Worker count used for adjusted throughput formula (default: 8)",
    )
    parser.add_argument(
        "--network-latency-ms",
        type=float,
        default=150.0,
        help="Fixed per-job network latency subtraction for adjusted throughput (default: 150)",
    )
    parser.add_argument(
        "--output", default=None, help="Output JSON path (default: results/<mode>_run.json)"
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Run summary JSON path (default: results/<mode>_run_summary.json)",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    sys.path.insert(0, "generated/python")

    from ga.controller import ControllerConfig, GAController
    from ga.engine import GAConfig
    from ga.search_space import SearchSpace
    from metrics.collector import MetricsStore

    # Select search space and defaults based on mode
    if args.mode == "stress":
        search_space = SearchSpace.stress_test_cnn_space()
        default_epochs = 1
        run_name = "stress_test"
    else:
        search_space = SearchSpace.default_cnn_space()
        default_epochs = 10
        run_name = "full_optimization"

    epochs = args.epochs if args.epochs is not None else default_epochs
    output_path = args.output or f"results/{run_name}.json"
    summary_output_path = args.summary_output or f"results/{run_name}_summary.json"

    ga_cfg = GAConfig(
        population_size=args.pop_size,
        num_generations=args.generations,
        dataset=args.dataset,
        epochs=epochs,
        seed=args.seed,
    )
    ctrl_cfg = ControllerConfig(
        scheduler_address=args.scheduler_address,
        ga_config=ga_cfg,
        poll_interval=args.poll_interval,
        rpc_timeout_s=args.rpc_timeout,
        generation_timeout_s=args.generation_timeout,
        progress_log_interval_s=args.progress_log_interval,
        max_wall_clock_s=args.max_wall_clock_s,
        expected_workers=args.expected_workers,
    )
    store = MetricsStore(run_name=run_name)
    controller = GAController(ctrl_cfg, store, search_space=search_space)

    jobs_planned = args.pop_size * args.generations

    print("=" * 38)
    print("  Epoch Hyperparameter Optimization")
    print("=" * 38)
    print(f"  Mode:           {args.mode}")
    print(f"  Population:     {args.pop_size}")
    print(f"  Generations:    {args.generations}")
    print(f"  Total jobs:     {jobs_planned}")
    print(f"  Dataset:        {args.dataset}")
    print(f"  Epochs/job:     {epochs}")
    print(f"  Scheduler:      {args.scheduler_address}")
    print(f"  Throughput workers: {args.throughput_worker_count}")
    print(f"  Network latency: {args.network_latency_ms:.0f}ms")
    if args.max_wall_clock_s > 0:
        print(f"  Max wall-clock: {args.max_wall_clock_s:.0f}s")
    if args.expected_workers > 0:
        print(f"  Expected workers: {args.expected_workers}")
    print()

    start = time.monotonic()
    final_pop = controller.run()
    total_s = time.monotonic() - start

    store.to_json(output_path)
    run_stats = controller.run_stats

    jobs_submitted = run_stats.jobs_submitted
    jobs_returned = run_stats.total_results
    avg_job_ms_raw, avg_job_ms_adjusted, jobs_per_sec = _compute_adjusted_throughput(
        wall_clock_s=total_s,
        jobs_submitted=jobs_submitted,
        throughput_worker_count=args.throughput_worker_count,
        network_latency_ms=float(args.network_latency_ms),
    )
    avg_job_ms = avg_job_ms_raw

    generation_kpis = run_stats.generation_kpis
    gen_wall_clock_values = [float(kpi.wall_clock_ms) for kpi in generation_kpis]
    queue_overhead_ms_values = [float(kpi.queue_overhead_ms) for kpi in generation_kpis]
    queue_overhead_pct_values = [float(kpi.queue_overhead_pct) for kpi in generation_kpis]
    active_worker_values = [float(kpi.active_workers) for kpi in generation_kpis]
    dispatch_latency_p50_values = [float(kpi.dispatch_latency_p50_ms) for kpi in generation_kpis]
    dispatch_latency_p90_values = [float(kpi.dispatch_latency_p90_ms) for kpi in generation_kpis]
    dispatch_latency_max_values = [float(kpi.dispatch_latency_max_ms) for kpi in generation_kpis]
    worker_idle_gap_p50_values = [float(kpi.worker_idle_gap_p50_ms) for kpi in generation_kpis]
    worker_idle_gap_p90_values = [float(kpi.worker_idle_gap_p90_ms) for kpi in generation_kpis]
    worker_idle_gap_max_values = [float(kpi.worker_idle_gap_max_ms) for kpi in generation_kpis]
    queue_wait_p50_values = [float(kpi.queue_wait_p50_ms) for kpi in generation_kpis]
    queue_wait_p90_values = [float(kpi.queue_wait_p90_ms) for kpi in generation_kpis]
    queue_wait_max_values = [float(kpi.queue_wait_max_ms) for kpi in generation_kpis]
    dispatch_samples_total = sum(int(kpi.dispatch_samples) for kpi in generation_kpis)
    idle_gap_samples_total = sum(int(kpi.idle_gap_samples) for kpi in generation_kpis)
    queue_wait_samples_total = sum(int(kpi.queue_wait_samples) for kpi in generation_kpis)

    if gen_wall_clock_values:
        p50_generation_ms = int(_percentile(gen_wall_clock_values, 0.50))
        p90_generation_ms = int(_percentile(gen_wall_clock_values, 0.90))
    else:
        # Fallback to existing per-generation timing store if no KPI rows exist.
        generation_times_ms = [float(g.wall_clock_ms) for g in store.generations]
        p50_generation_ms = int(_percentile(generation_times_ms, 0.50))
        p90_generation_ms = int(_percentile(generation_times_ms, 0.90))

    failures = run_stats.failed + run_stats.unknown
    failure_denominator = jobs_submitted if jobs_submitted > 0 else jobs_planned
    failure_rate = (
        (run_stats.timed_out + failures) / failure_denominator if failure_denominator > 0 else 0.0
    )

    best_fitness_peak = (
        max((float(g.best_fitness) for g in store.generations), default=float(final_pop.best_fitness))
        if store.generations
        else float(final_pop.best_fitness)
    )

    summary = {
        # Existing fields (kept for backward compatibility)
        "run_name": run_name,
        "mode": args.mode,
        "dataset": args.dataset,
        "jobs_total": jobs_planned,
        "jobs_returned": jobs_returned,
        "completed": run_stats.completed,
        "timeouts": run_stats.timed_out,
        "failures": failures,
        "failure_rate": failure_rate,
        "wall_clock_s": total_s,
        "jobs_per_sec": jobs_per_sec,
        "avg_job_ms": avg_job_ms,
        "avg_job_ms_raw": avg_job_ms_raw,
        "avg_job_ms_adjusted": avg_job_ms_adjusted,
        "network_latency_ms": float(args.network_latency_ms),
        "throughput_worker_count": int(args.throughput_worker_count),
        "jobs_per_sec_formula": (
            "jobs_per_sec = (throughput_worker_count * 1000) / "
            "max(avg_job_ms_raw - network_latency_ms, 1)"
        ),
        "p50_generation_ms": p50_generation_ms,
        "p90_generation_ms": p90_generation_ms,
        "population_size": args.pop_size,
        "generations": args.generations,
        "epochs": epochs,
        "scheduler_address": args.scheduler_address,
        # New planning fields
        "jobs_planned": run_stats.jobs_planned,
        "jobs_submitted": jobs_submitted,
        "generation_wall_clock_p50_ms": p50_generation_ms,
        "generation_wall_clock_p90_ms": p90_generation_ms,
        "queue_overhead_avg_ms": (
            sum(queue_overhead_ms_values) / len(queue_overhead_ms_values)
            if queue_overhead_ms_values
            else 0.0
        ),
        "queue_overhead_p50_ms": _percentile(queue_overhead_ms_values, 0.50),
        "queue_overhead_p90_ms": _percentile(queue_overhead_ms_values, 0.90),
        "queue_overhead_avg_pct": (
            sum(queue_overhead_pct_values) / len(queue_overhead_pct_values)
            if queue_overhead_pct_values
            else 0.0
        ),
        "queue_overhead_p50_pct": _percentile(queue_overhead_pct_values, 0.50),
        "queue_overhead_p90_pct": _percentile(queue_overhead_pct_values, 0.90),
        "active_workers_min": int(min(active_worker_values)) if active_worker_values else 0,
        "active_workers_p50": int(_percentile(active_worker_values, 0.50)) if active_worker_values else 0,
        "active_workers_p90": int(_percentile(active_worker_values, 0.90)) if active_worker_values else 0,
        "active_workers_max": int(max(active_worker_values)) if active_worker_values else 0,
        "dispatch_latency_p50_ms": _percentile(dispatch_latency_p50_values, 0.50),
        "dispatch_latency_p90_ms": _percentile(dispatch_latency_p90_values, 0.90),
        "dispatch_latency_max_ms": max(dispatch_latency_max_values) if dispatch_latency_max_values else 0.0,
        "worker_idle_gap_p50_ms": _percentile(worker_idle_gap_p50_values, 0.50),
        "worker_idle_gap_p90_ms": _percentile(worker_idle_gap_p90_values, 0.90),
        "worker_idle_gap_max_ms": max(worker_idle_gap_max_values) if worker_idle_gap_max_values else 0.0,
        "queue_wait_p50_ms": _percentile(queue_wait_p50_values, 0.50),
        "queue_wait_p90_ms": _percentile(queue_wait_p90_values, 0.90),
        "queue_wait_max_ms": max(queue_wait_max_values) if queue_wait_max_values else 0.0,
        "dispatch_samples_total": dispatch_samples_total,
        "worker_idle_gap_samples_total": idle_gap_samples_total,
        "queue_wait_samples_total": queue_wait_samples_total,
        "best_fitness_final": float(final_pop.best_fitness),
        "avg_fitness_final": float(final_pop.avg_fitness),
        "worst_fitness_final": float(final_pop.worst_fitness),
        "best_fitness_peak": float(best_fitness_peak),
        "convergence_rate_per_gen": float(store.convergence_rate),
        "generations_planned": run_stats.generations_planned,
        "generations_completed": run_stats.generations_completed,
        "stopped_early": run_stats.stopped_early,
        "stop_reason": run_stats.stop_reason,
        "max_wall_clock_s": args.max_wall_clock_s,
        "expected_workers": args.expected_workers,
        "generation_kpis": [
            {
                "generation": kpi.generation,
                "jobs_count": kpi.jobs_count,
                "completed": kpi.completed,
                "failed": kpi.failed,
                "timed_out": kpi.timed_out,
                "active_workers": kpi.active_workers,
                "wall_clock_ms": kpi.wall_clock_ms,
                "train_total_ms": kpi.train_total_ms,
                "train_p50_ms": kpi.train_p50_ms,
                "train_p90_ms": kpi.train_p90_ms,
                "ideal_wall_ms": kpi.ideal_wall_ms,
                "queue_overhead_ms": kpi.queue_overhead_ms,
                "queue_overhead_pct": kpi.queue_overhead_pct,
                "gen_jobs_per_sec": kpi.gen_jobs_per_sec,
                "dispatch_latency_p50_ms": kpi.dispatch_latency_p50_ms,
                "dispatch_latency_p90_ms": kpi.dispatch_latency_p90_ms,
                "dispatch_latency_max_ms": kpi.dispatch_latency_max_ms,
                "worker_idle_gap_p50_ms": kpi.worker_idle_gap_p50_ms,
                "worker_idle_gap_p90_ms": kpi.worker_idle_gap_p90_ms,
                "worker_idle_gap_max_ms": kpi.worker_idle_gap_max_ms,
                "queue_wait_p50_ms": kpi.queue_wait_p50_ms,
                "queue_wait_p90_ms": kpi.queue_wait_p90_ms,
                "queue_wait_max_ms": kpi.queue_wait_max_ms,
                "dispatch_samples": kpi.dispatch_samples,
                "idle_gap_samples": kpi.idle_gap_samples,
                "queue_wait_samples": kpi.queue_wait_samples,
            }
            for kpi in generation_kpis
        ],
    }
    os.makedirs(os.path.dirname(summary_output_path) or ".", exist_ok=True)
    with open(summary_output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 38)
    print("  Results")
    print("=" * 38)
    print(f"  Jobs planned:   {jobs_planned}")
    print(f"  Jobs submitted: {jobs_submitted}")
    print(f"  Jobs returned:  {jobs_returned}")
    print(f"  Total time:     {total_s:.1f}s")
    print(f"  Throughput:     {jobs_per_sec:.2f} jobs/sec")
    print(f"  Avg time/job:   {avg_job_ms_raw:.0f}ms raw / {avg_job_ms_adjusted:.0f}ms adjusted")
    print(f"  Best fitness:   {final_pop.best_fitness:.4f}")
    print(f"  Peak fitness:   {best_fitness_peak:.4f}")
    print(f"  Convergence:    {store.convergence_rate:.6f}/gen")
    print(f"  Stop reason:    {run_stats.stop_reason}")
    print("=" * 38)
    print(f"  Results saved to {output_path}")
    print(f"  Summary saved to {summary_output_path}")


if __name__ == "__main__":
    main()
