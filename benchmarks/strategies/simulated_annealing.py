"""Simulated annealing search strategy for benchmarking."""

from __future__ import annotations

import math
import random
import time
from typing import Any

from benchmarks.strategies.base import SearchStrategy
from ga.search_space import ParamType, SearchSpace
from metrics.collector import MetricsStore
from worker.trainer import Trainer


class SimulatedAnnealing(SearchStrategy):
    """Simulated annealing with mixed-type neighborhood proposals."""

    def __init__(
        self,
        seed: int | None = None,
        dataset: str = "mnist",
        initial_temp: float = 1.0,
        cooling_rate: float = 0.995,
        min_temp: float = 1e-3,
        neighbor_scale: float = 0.15,
    ) -> None:
        self.seed = seed
        self.dataset = dataset
        self.initial_temp = float(initial_temp)
        self.cooling_rate = float(cooling_rate)
        self.min_temp = float(min_temp)
        self.neighbor_scale = float(neighbor_scale)

    def name(self) -> str:
        return "Simulated Annealing"

    def _propose_neighbor(
        self,
        *,
        current: dict[str, Any],
        space: SearchSpace,
        temperature: float,
        rng: random.Random,
    ) -> dict[str, Any]:
        neighbor = dict(current)
        param = rng.choice(space.params)
        name = param.name

        if param.param_type == ParamType.CONTINUOUS:
            span = float(param.high - param.low)
            scale = max(temperature, 0.05)
            sigma = max(span * self.neighbor_scale * scale, 1e-12)
            neighbor[name] = param.clip(float(current[name]) + rng.gauss(0.0, sigma))
        elif param.param_type == ParamType.DISCRETE:
            span = int(param.high - param.low)
            scale = max(temperature, 0.05)
            max_step = max(1, int(round(max(span, 1) * self.neighbor_scale * scale)))
            step = rng.randint(-max_step, max_step)
            if step == 0:
                step = 1 if rng.random() < 0.5 else -1
            neighbor[name] = param.clip(int(current[name]) + step)
        else:
            choices = [choice for choice in param.choices if choice != current[name]]
            if choices:
                neighbor[name] = param.clip(rng.choice(choices))

        neighbor["dataset"] = self.dataset
        return neighbor

    def run(
        self,
        space: SearchSpace,
        trainer: Trainer,
        *,
        budget: int | None = None,
        max_wall_clock_s: float | None = None,
    ) -> MetricsStore:
        """Run simulated annealing within the provided limits."""
        store = MetricsStore(run_name=self.name())
        rng = random.Random(self.seed)
        budget, deadline = self._resolve_limits(
            budget=budget,
            max_wall_clock_s=max_wall_clock_s,
        )

        eval_idx = 0
        temperature = max(self.initial_temp, self.min_temp)

        if self._should_stop(
            evals_completed=eval_idx,
            budget=budget,
            deadline_monotonic=deadline,
        ):
            return store

        current = space.random_sample(rng)
        current["dataset"] = self.dataset

        start = time.monotonic()
        current_result = trainer.train(current)
        current_fitness = (
            current_result.validation_accuracy if current_result.status == "completed" else 0.0
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)
        best_fitness = current_fitness
        best_genome = dict(current)

        store.record_generation(
            generation=eval_idx,
            best_fitness=best_fitness,
            best_so_far=best_fitness,
            avg_fitness=current_fitness,
            worst_fitness=current_fitness,
            wall_clock_ms=elapsed_ms,
        )
        eval_idx += 1

        while not self._should_stop(
            evals_completed=eval_idx,
            budget=budget,
            deadline_monotonic=deadline,
        ):
            candidate = self._propose_neighbor(
                current=current,
                space=space,
                temperature=temperature,
                rng=rng,
            )
            start = time.monotonic()
            result = trainer.train(candidate)
            candidate_fitness = result.validation_accuracy if result.status == "completed" else 0.0
            elapsed_ms = int((time.monotonic() - start) * 1000)

            delta = candidate_fitness - current_fitness
            temp = max(temperature, 1e-9)
            accept_prob = math.exp(delta / temp) if delta < 0.0 else 1.0
            if delta >= 0.0 or rng.random() < accept_prob:
                current = candidate
                current_fitness = candidate_fitness

            if candidate_fitness > best_fitness:
                best_fitness = candidate_fitness
                best_genome = dict(candidate)

            store.record_generation(
                generation=eval_idx,
                best_fitness=best_fitness,
                best_so_far=best_fitness,
                avg_fitness=candidate_fitness,
                worst_fitness=candidate_fitness,
                wall_clock_ms=elapsed_ms,
            )
            eval_idx += 1
            temperature = max(self.min_temp, temperature * self.cooling_rate)

        _ = best_genome
        return store
