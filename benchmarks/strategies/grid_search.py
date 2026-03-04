"""Grid search strategy for benchmarking."""

from __future__ import annotations

import itertools
import time
from typing import Any

from benchmarks.strategies.base import SearchStrategy
from ga.search_space import ParamType, SearchSpace
from metrics.collector import MetricsStore
from worker.trainer import Trainer

# Number of grid points per parameter type
_GRID_POINTS = 3


class GridSearch(SearchStrategy):
    """Exhaustive grid search over discretized hyperparameter space.

    Discretizes each parameter into a small number of grid points and
    evaluates the Cartesian product (subsampled to budget if needed).

    Attributes:
        grid_points: Number of grid points per continuous/discrete parameter.
        dataset: Dataset to use for training.
    """

    def __init__(self, grid_points: int = _GRID_POINTS, dataset: str = "mnist") -> None:
        self.grid_points = grid_points
        self.dataset = dataset

    def name(self) -> str:
        return "Grid Search"

    def run(
        self,
        space: SearchSpace,
        budget: int,
        trainer: Trainer,
    ) -> MetricsStore:
        """Evaluate grid configurations up to `budget`."""
        store = MetricsStore(run_name=self.name())

        # Build grid values for each parameter
        grid_values: list[list[Any]] = []
        for param in space.params:
            if param.param_type == ParamType.CONTINUOUS:
                step = (param.high - param.low) / max(self.grid_points - 1, 1)
                values = [param.low + i * step for i in range(self.grid_points)]
                grid_values.append(values)
            elif param.param_type == ParamType.DISCRETE:
                step = max(1, int((param.high - param.low) / max(self.grid_points - 1, 1)))
                values = list(range(int(param.low), int(param.high) + 1, step))[: self.grid_points]
                grid_values.append(values)
            else:
                grid_values.append(list(param.choices))

        # Generate Cartesian product, limited to budget
        all_combos = list(itertools.islice(itertools.product(*grid_values), budget))

        best_fitness = 0.0
        for i, combo in enumerate(all_combos):
            start = time.monotonic()
            genome = {param.name: val for param, val in zip(space.params, combo)}
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
