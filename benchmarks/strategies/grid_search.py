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
        trainer: Trainer,
        *,
        budget: int | None = None,
        max_wall_clock_s: float | None = None,
    ) -> MetricsStore:
        """Evaluate grid configurations within the provided limits."""
        store = MetricsStore(run_name=self.name())
        budget, deadline = self._resolve_limits(
            budget=budget,
            max_wall_clock_s=max_wall_clock_s,
        )

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

        best_fitness = 0.0
        eval_idx = 0
        for combo in itertools.product(*grid_values):
            if self._should_stop(
                evals_completed=eval_idx,
                budget=budget,
                deadline_monotonic=deadline,
            ):
                break
            start = time.monotonic()
            genome = {param.name: val for param, val in zip(space.params, combo)}
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
