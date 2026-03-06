"""Benchmark runner for local algorithm comparisons."""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from benchmarks.reporter import build_aggregate_summary, write_ranked_report
from benchmarks.strategies.base import SearchStrategy
from benchmarks.strategies.bayesian_search import BayesianSearch
from benchmarks.strategies.differential_evolution import DifferentialEvolution
from benchmarks.strategies.grid_search import GridSearch
from benchmarks.strategies.random_search import RandomSearch
from benchmarks.strategies.simulated_annealing import SimulatedAnnealing
from ga.search_space import SearchSpace
from metrics.collector import MetricsStore
from worker.trainer import Trainer

logger = logging.getLogger(__name__)

_DEFAULT_REFERENCE_SUMMARY = "results/modal_benchmark_t4_w10/ga_benchmark_g12_p120_summary.json"
_DEFAULT_WALL_CLOCK_S = 531.9059779640011


def _slugify(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _parse_seeds(raw: str) -> list[int]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one seed is required.")
    return [int(value) for value in values]


@dataclass
class BenchmarkConfig:
    """Configuration for local benchmark execution."""

    budget: int = 0
    max_wall_clock_s: float = _DEFAULT_WALL_CLOCK_S
    dataset: str = "mnist"
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])
    output_dir: str = "results/local_benchmark_g12_p120"
    space_mode: str = "stress"
    train_subset_size: int = 1536
    val_subset_size: int = 768
    timeout_seconds: float = 20.0
    gc_every_n_jobs: int = 3
    deterministic_eval: bool = True
    deterministic_seed_offset: int = 0
    run_eagerly: bool = False
    reference_summary_path: str = _DEFAULT_REFERENCE_SUMMARY

    def normalized_budget(self) -> int | None:
        if self.budget > 0:
            return int(self.budget)
        return None

    def normalized_max_wall_clock_s(self) -> float | None:
        if self.max_wall_clock_s > 0:
            return float(self.max_wall_clock_s)
        return None


class BenchmarkRunner:
    """Runs local search strategies and generates benchmark reports."""

    def __init__(
        self,
        config: BenchmarkConfig | None = None,
        strategies: list[SearchStrategy] | None = None,
    ) -> None:
        self.config = config or BenchmarkConfig()
        self.strategies = strategies

    def _default_strategies(self, seed: int) -> list[SearchStrategy]:
        return [
            RandomSearch(seed=seed, dataset=self.config.dataset),
            GridSearch(dataset=self.config.dataset),
            BayesianSearch(seed=seed, dataset=self.config.dataset),
            DifferentialEvolution(seed=seed, dataset=self.config.dataset),
            SimulatedAnnealing(seed=seed, dataset=self.config.dataset),
        ]

    def _resolve_space(self) -> SearchSpace:
        if self.config.space_mode == "stress":
            return SearchSpace.stress_test_cnn_space()
        if self.config.space_mode == "default":
            return SearchSpace.default_cnn_space()
        raise ValueError(f"Unsupported --space-mode: {self.config.space_mode}")

    def _build_trainer(self) -> Trainer:
        return Trainer(
            timeout_seconds=self.config.timeout_seconds,
            train_subset_size=self.config.train_subset_size,
            val_subset_size=self.config.val_subset_size,
            run_eagerly=self.config.run_eagerly,
            gc_every_n_jobs=self.config.gc_every_n_jobs,
            deterministic_eval=self.config.deterministic_eval,
            deterministic_seed_offset=self.config.deterministic_seed_offset,
        )

    def run(self) -> dict[str, dict[int, MetricsStore]]:
        """Execute all configured strategies for each seed."""
        output_dir = Path(self.config.output_dir)
        per_seed_dir = output_dir / "per_seed"
        per_seed_dir.mkdir(parents=True, exist_ok=True)

        space = self._resolve_space()
        results: dict[str, dict[int, MetricsStore]] = {}

        for seed in self.config.seeds:
            strategy_list = (
                self.strategies
                if self.strategies is not None
                else self._default_strategies(seed)
            )
            logger.info("Running seed=%d with %d strategies", seed, len(strategy_list))

            for strategy in strategy_list:
                logger.info(
                    "Running %s (seed=%d, budget=%s, max_wall_clock_s=%s)",
                    strategy.name(),
                    seed,
                    self.config.normalized_budget(),
                    self.config.normalized_max_wall_clock_s(),
                )
                trainer = self._build_trainer()
                started = time.monotonic()
                store = strategy.run(
                    space,
                    trainer,
                    budget=self.config.normalized_budget(),
                    max_wall_clock_s=self.config.normalized_max_wall_clock_s(),
                )
                elapsed = time.monotonic() - started

                strategy_name = strategy.name()
                results.setdefault(strategy_name, {})[seed] = store

                output_path = per_seed_dir / f"{_slugify(strategy_name)}_seed{seed}.json"
                store.to_json(output_path)

                peak = max((g.best_so_far for g in store.generations), default=0.0)
                logger.info(
                    "%s seed=%d complete: evals=%d peak=%.6f wall_s=%.2f",
                    strategy_name,
                    seed,
                    len(store.generations),
                    peak,
                    elapsed,
                )

        return results

    def run_and_report(self) -> dict:
        """Execute benchmark and write aggregate summary + markdown report."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = self.run()
        summary = build_aggregate_summary(
            results=results,
            config=asdict(self.config),
            reference_summary_path=self.config.reference_summary_path,
        )

        summary_path = output_dir / "benchmark_aggregate_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        report_path = output_dir / "ranked_report.md"
        write_ranked_report(summary=summary, output_path=report_path)

        logger.info("Aggregate summary saved to %s", summary_path)
        logger.info("Ranked markdown report saved to %s", report_path)
        return summary


def _print_topline(summary: dict) -> None:
    print("\nLocal algorithm ranking (reference-only rows excluded)")
    print("-" * 98)
    print(
        f"{'rank':>4}  {'algorithm':<24} {'median_peak':>12} {'mean_peak':>10} "
        f"{'median_evals':>12} {'mean_evals':>10} {'median_s':>10}"
    )
    print("-" * 98)
    for row in summary.get("ranked_algorithms", []):
        print(
            f"{row['rank']:>4}  {row['name']:<24.24} {row['median_best_fitness_peak']:>12.6f} "
            f"{row['mean_best_fitness_peak']:>10.6f} {row['median_evals_completed']:>12.1f} "
            f"{row['mean_evals_completed']:>10.1f} {row['median_wall_clock_s']:>10.2f}"
        )

    reference = summary.get("reference_row")
    if reference:
        print("\nReference (not ranked)")
        print("-" * 98)
        print(
            "  "
            f"{reference.get('name', 'reference')}: "
            f"best_peak={reference.get('best_fitness_peak', 0.0):.6f}, "
            f"wall_clock_total_s={reference.get('wall_clock_total_s', 0.0):.3f}, "
            f"jobs_submitted={reference.get('jobs_submitted', 0)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Epoch local benchmark runner")
    parser.add_argument(
        "--budget",
        type=int,
        default=0,
        help="Max evaluations per run (0 disables)",
    )
    parser.add_argument(
        "--max-wall-clock-s",
        type=float,
        default=_DEFAULT_WALL_CLOCK_S,
        help="Wall-clock budget per algorithm/seed run in seconds",
    )
    parser.add_argument("--dataset", default="mnist", help="Dataset (mnist or cifar10)")
    parser.add_argument(
        "--seeds",
        default="42,43,44",
        help="Comma-separated seed list (e.g. 42,43,44)",
    )
    parser.add_argument("--output-dir", default="results/local_benchmark_g12_p120")
    parser.add_argument(
        "--space-mode",
        choices=["stress", "default"],
        default="stress",
        help="Search space selection",
    )
    parser.add_argument("--train-subset-size", type=int, default=1536)
    parser.add_argument("--val-subset-size", type=int, default=768)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--gc-every-n-jobs", type=int, default=3)
    parser.add_argument(
        "--deterministic-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable deterministic per-genome evaluation seeding",
    )
    parser.add_argument("--deterministic-seed-offset", type=int, default=0)
    parser.add_argument("--run-eagerly", action="store_true", default=False)
    parser.add_argument(
        "--reference-summary-path",
        default=_DEFAULT_REFERENCE_SUMMARY,
        help="Modal GA summary used as reference-only row",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    config = BenchmarkConfig(
        budget=args.budget,
        max_wall_clock_s=args.max_wall_clock_s,
        dataset=args.dataset,
        seeds=_parse_seeds(args.seeds),
        output_dir=args.output_dir,
        space_mode=args.space_mode,
        train_subset_size=args.train_subset_size,
        val_subset_size=args.val_subset_size,
        timeout_seconds=args.timeout_seconds,
        gc_every_n_jobs=args.gc_every_n_jobs,
        deterministic_eval=args.deterministic_eval,
        deterministic_seed_offset=args.deterministic_seed_offset,
        run_eagerly=args.run_eagerly,
        reference_summary_path=args.reference_summary_path,
    )

    runner = BenchmarkRunner(config)
    summary = runner.run_and_report()
    _print_topline(summary)


if __name__ == "__main__":
    main()
