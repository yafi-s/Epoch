"""Bayesian optimization strategy using Optuna's TPE sampler."""

from __future__ import annotations

import time
from typing import Any

import optuna

from benchmarks.strategies.base import SearchStrategy
from ga.search_space import ParamType, SearchSpace
from metrics.collector import MetricsStore
from worker.trainer import Trainer


class BayesianSearch(SearchStrategy):
    """Bayesian optimization using Optuna's Tree-structured Parzen Estimator (TPE).

    Wraps Optuna to sample hyperparameters, evaluate them via the Trainer,
    and record metrics in the same format as other strategies.

    Attributes:
        seed: RNG seed for reproducibility.
        dataset: Dataset to use for training.
    """

    def __init__(self, seed: int | None = None, dataset: str = "mnist") -> None:
        self.seed = seed
        self.dataset = dataset

    def name(self) -> str:
        return "Bayesian (Optuna TPE)"

    def run(
        self,
        space: SearchSpace,
        budget: int,
        trainer: Trainer,
    ) -> MetricsStore:
        """Run Bayesian optimization for `budget` trials."""
        store = MetricsStore(run_name=self.name())
        best_fitness = 0.0
        trial_index = [0]

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_fitness
            start = time.monotonic()

            genome: dict[str, Any] = {}
            for param in space.params:
                if param.param_type == ParamType.CONTINUOUS:
                    if param.log_scale:
                        genome[param.name] = trial.suggest_float(
                            param.name, param.low, param.high, log=True
                        )
                    else:
                        genome[param.name] = trial.suggest_float(
                            param.name, param.low, param.high
                        )
                elif param.param_type == ParamType.DISCRETE:
                    genome[param.name] = trial.suggest_int(
                        param.name, int(param.low), int(param.high)
                    )
                else:
                    genome[param.name] = trial.suggest_categorical(
                        param.name, list(param.choices)
                    )

            genome["dataset"] = self.dataset

            result = trainer.train(genome)
            fitness = result.validation_accuracy if result.status == "completed" else 0.0
            best_fitness = max(best_fitness, fitness)
            elapsed_ms = int((time.monotonic() - start) * 1000)

            store.record_generation(
                generation=trial_index[0],
                best_fitness=best_fitness,
                avg_fitness=fitness,
                worst_fitness=fitness,
                wall_clock_ms=elapsed_ms,
            )
            trial_index[0] += 1

            return fitness

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=budget)

        return store
