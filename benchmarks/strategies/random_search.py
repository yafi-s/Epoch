"""Random search strategy for benchmarking."""

from __future__ import annotations

import random
import time

from benchmarks.strategies.base import SearchStrategy
from ga.search_space import SearchSpace
from metrics.collector import MetricsStore
from worker.trainer import Trainer


class RandomSearch(SearchStrategy):
    """Uniformly random hyperparameter search.

    Samples configurations uniformly at random from the search space and
    evaluates them sequentially.

    Attributes:
        seed: RNG seed for reproducibility.
        dataset: Dataset to use for training.
    """

    def __init__(self, seed: int | None = None, dataset: str = "mnist") -> None:
        self.seed = seed
        self.dataset = dataset

    def name(self) -> str:
        return "Random Search"

    def run(
        self,
        space: SearchSpace,
        trainer: Trainer,
        *,
        budget: int | None = None,
        max_wall_clock_s: float | None = None,
    ) -> MetricsStore:
        """Evaluate random configurations within the provided limits."""
        rng = random.Random(self.seed)
        store = MetricsStore(run_name=self.name())
        best_fitness = 0.0
        eval_idx = 0
        budget, deadline = self._resolve_limits(
            budget=budget,
            max_wall_clock_s=max_wall_clock_s,
        )

        while not self._should_stop(
            evals_completed=eval_idx,
            budget=budget,
            deadline_monotonic=deadline,
        ):
            start = time.monotonic()
            genome = space.random_sample(rng)
            genome["dataset"] = self.dataset

            result = trainer.train(genome)
            fitness = result.validation_accuracy if result.status == "completed" else 0.0
            best_fitness = max(best_fitness, fitness)
            elapsed_ms = int((time.monotonic() - start) * 1000)

            store.record_generation(
                generation=eval_idx,
                best_fitness=best_fitness,
                best_so_far=best_fitness,
                avg_fitness=fitness,
                worst_fitness=fitness,
                wall_clock_ms=elapsed_ms,
            )
            eval_idx += 1

        return store
