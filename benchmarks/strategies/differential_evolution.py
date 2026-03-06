"""Differential evolution search strategy for benchmarking."""

from __future__ import annotations

import random
import time
from typing import Any

from benchmarks.strategies.base import SearchStrategy
from ga.search_space import ParamType, SearchSpace
from metrics.collector import MetricsStore
from worker.trainer import Trainer


class DifferentialEvolution(SearchStrategy):
    """Differential Evolution adapted for mixed search spaces."""

    def __init__(
        self,
        seed: int | None = None,
        dataset: str = "mnist",
        population_size: int = 20,
        differential_weight: float = 0.8,
        crossover_rate: float = 0.9,
    ) -> None:
        self.seed = seed
        self.dataset = dataset
        self.population_size = max(4, int(population_size))
        self.differential_weight = float(differential_weight)
        self.crossover_rate = float(crossover_rate)

    def name(self) -> str:
        return "Differential Evolution"

    def _random_genome(self, space: SearchSpace, rng: random.Random) -> dict[str, Any]:
        genome = space.random_sample(rng)
        genome["dataset"] = self.dataset
        return genome

    def _mutant_value(
        self,
        *,
        param,
        a: dict[str, Any],
        b: dict[str, Any],
        c: dict[str, Any],
        target: dict[str, Any],
        rng: random.Random,
    ) -> Any:
        if param.param_type in (ParamType.CONTINUOUS, ParamType.DISCRETE):
            donor = (
                float(a[param.name])
                + self.differential_weight * (float(b[param.name]) - float(c[param.name]))
            )
            return param.clip(donor)
        candidates = [
            a[param.name],
            b[param.name],
            c[param.name],
            target[param.name],
        ]
        return param.clip(rng.choice(candidates))

    def _build_trial(
        self,
        *,
        space: SearchSpace,
        target: dict[str, Any],
        mutant: dict[str, Any],
        rng: random.Random,
    ) -> dict[str, Any]:
        trial: dict[str, Any] = {}
        forced_idx = rng.randrange(len(space.params))
        for idx, param in enumerate(space.params):
            use_mutant = (idx == forced_idx) or (rng.random() < self.crossover_rate)
            value = mutant[param.name] if use_mutant else target[param.name]
            trial[param.name] = param.clip(value)
        trial["dataset"] = self.dataset
        return trial

    def run(
        self,
        space: SearchSpace,
        trainer: Trainer,
        *,
        budget: int | None = None,
        max_wall_clock_s: float | None = None,
    ) -> MetricsStore:
        """Run differential evolution within the provided limits."""
        store = MetricsStore(run_name=self.name())
        rng = random.Random(self.seed)
        budget, deadline = self._resolve_limits(
            budget=budget,
            max_wall_clock_s=max_wall_clock_s,
        )

        population = [self._random_genome(space, rng) for _ in range(self.population_size)]
        fitnesses = [0.0 for _ in range(self.population_size)]

        best_fitness = 0.0
        eval_idx = 0

        # Evaluate initial population.
        for i in range(self.population_size):
            if self._should_stop(
                evals_completed=eval_idx,
                budget=budget,
                deadline_monotonic=deadline,
            ):
                break
            start = time.monotonic()
            result = trainer.train(population[i])
            fitness = result.validation_accuracy if result.status == "completed" else 0.0
            fitnesses[i] = fitness
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

        while not self._should_stop(
            evals_completed=eval_idx,
            budget=budget,
            deadline_monotonic=deadline,
        ):
            progressed = False
            for i in range(self.population_size):
                if self._should_stop(
                    evals_completed=eval_idx,
                    budget=budget,
                    deadline_monotonic=deadline,
                ):
                    break

                candidate_indices = list(range(self.population_size))
                candidate_indices.remove(i)
                idx_a, idx_b, idx_c = rng.sample(candidate_indices, 3)
                target = population[i]
                mutant: dict[str, Any] = {}

                for param in space.params:
                    mutant[param.name] = self._mutant_value(
                        param=param,
                        a=population[idx_a],
                        b=population[idx_b],
                        c=population[idx_c],
                        target=target,
                        rng=rng,
                    )

                trial = self._build_trial(space=space, target=target, mutant=mutant, rng=rng)

                start = time.monotonic()
                result = trainer.train(trial)
                fitness = result.validation_accuracy if result.status == "completed" else 0.0
                elapsed_ms = int((time.monotonic() - start) * 1000)

                if fitness >= fitnesses[i]:
                    population[i] = trial
                    fitnesses[i] = fitness

                best_fitness = max(best_fitness, fitness)
                store.record_generation(
                    generation=eval_idx,
                    best_fitness=best_fitness,
                    best_so_far=best_fitness,
                    avg_fitness=fitness,
                    worst_fitness=fitness,
                    wall_clock_ms=elapsed_ms,
                )
                eval_idx += 1
                progressed = True

            if not progressed:
                break

        return store
