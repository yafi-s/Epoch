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
        budget: int,
        trainer: Trainer,
    ) -> MetricsStore:
        """Evaluate `budget` random configurations."""
        rng = random.Random(self.seed)
        store = MetricsStore(run_name=self.name())
        best_fitness = 0.0

        for i in range(budget):
            start = time.monotonic()
            genome = space.random_sample(rng)
            genome["dataset"] = self.dataset

            result = trainer.train(genome)
            fitness = result.validation_accuracy if result.status == "completed" else 0.0
            best_fitness = max(best_fitness, fitness)
            elapsed_ms = int((time.monotonic() - start) * 1000)

            store.record_generation(
                generation=i,
                best_fitness=best_fitness,
                avg_fitness=fitness,
                worst_fitness=fitness,
                wall_clock_ms=elapsed_ms,
            )

        return store
