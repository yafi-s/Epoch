"""Benchmark runner: executes all strategies and generates comparison reports."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from benchmarks.strategies.base import SearchStrategy
from benchmarks.strategies.bayesian_search import BayesianSearch
from benchmarks.strategies.grid_search import GridSearch
from benchmarks.strategies.random_search import RandomSearch
from ga.search_space import SearchSpace
from metrics.collector import MetricsStore
from metrics.dashboard import plot_comparison
from worker.trainer import Trainer

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark runner.

    Attributes:
        budget: Total number of evaluations per strategy.
        dataset: Dataset to use for training.
        seed: RNG seed for reproducibility.
        output_dir: Directory for output files.
    """

    budget: int = 20
    dataset: str = "mnist"
    seed: int = 42
    output_dir: str = "results"


class BenchmarkRunner:
    """Runs multiple search strategies and compares their performance.

    Attributes:
        config: Benchmark configuration.
        strategies: List of strategies to evaluate.
    """

    def __init__(
        self,
        config: BenchmarkConfig | None = None,
        strategies: list[SearchStrategy] | None = None,
    ) -> None:
        self.config = config or BenchmarkConfig()
        self.strategies = strategies or self._default_strategies()

    def _default_strategies(self) -> list[SearchStrategy]:
        """Return the default set of strategies to benchmark."""
        return [
            RandomSearch(seed=self.config.seed, dataset=self.config.dataset),
            GridSearch(dataset=self.config.dataset),
            BayesianSearch(seed=self.config.seed, dataset=self.config.dataset),
        ]

    def run(self) -> dict[str, MetricsStore]:
        """Execute all strategies and return their results.

        Returns:
            Mapping of strategy name to its MetricsStore.
        """
        space = SearchSpace.default_cnn_space()
        trainer = Trainer()
        results: dict[str, MetricsStore] = {}

        for strategy in self.strategies:
            logger.info("═══ Running: %s (budget=%d) ═══", strategy.name(), self.config.budget)
            start = time.monotonic()

            store = strategy.run(space, self.config.budget, trainer)
            results[strategy.name()] = store

            elapsed = time.monotonic() - start
            logger.info(
                "%s complete: best=%.4f total_time=%.1fs",
                strategy.name(),
                store.final_best_fitness,
                elapsed,
            )

            # Save individual results
            store.to_json(f"{self.config.output_dir}/{strategy.name().lower().replace(' ', '_')}.json")

        return results

    def run_and_report(self) -> dict[str, MetricsStore]:
        """Execute all strategies, generate comparison plots, and print a summary table.

        Returns:
            Mapping of strategy name to its MetricsStore.
        """
        results = self.run()

        # Generate comparison plot
        plot_comparison(results, f"{self.config.output_dir}/comparison.png")
        logger.info("Comparison plot saved to %s/comparison.png", self.config.output_dir)

        # Print summary table
        self._print_summary(results)

        return results

    def _print_summary(self, results: dict[str, MetricsStore]) -> None:
        """Print a formatted comparison table."""
        print("\n" + "=" * 75)
        print("BENCHMARK RESULTS")
        print("=" * 75)
        print(f"{'Strategy':<25} {'Best Fitness':>14} {'Total Time (s)':>16} {'Conv. Rate':>12}")
        print("-" * 75)

        for name, store in results.items():
            summary = store.summary()
            print(
                f"{name:<25} {summary['final_best_fitness']:>14.4f} "
                f"{summary['total_wall_clock_ms'] / 1000.0:>16.1f} "
                f"{summary['convergence_rate']:>12.6f}"
            )

        print("=" * 75)


def main() -> None:
    """CLI entry point for running benchmarks."""
    import argparse

    parser = argparse.ArgumentParser(description="Epoch Benchmark Runner")
    parser.add_argument("--budget", type=int, default=20, help="Evaluations per strategy")
    parser.add_argument("--dataset", default="mnist", help="Dataset (mnist or cifar10)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    config = BenchmarkConfig(
        budget=args.budget,
        dataset=args.dataset,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    runner = BenchmarkRunner(config)
    runner.run_and_report()


if __name__ == "__main__":
    main()
