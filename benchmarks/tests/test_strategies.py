"""Tests for benchmark strategies.

Uses a mock trainer to avoid actual model training during tests.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from unittest.mock import MagicMock

from benchmarks.strategies.bayesian_search import BayesianSearch
from benchmarks.strategies.differential_evolution import DifferentialEvolution
from benchmarks.strategies.grid_search import GridSearch
from benchmarks.strategies.random_search import RandomSearch
from benchmarks.strategies.simulated_annealing import SimulatedAnnealing
from ga.search_space import ParamSpec, ParamType, SearchSpace
from worker.trainer import Trainer, TrainResult


def _mock_trainer(
    *,
    on_train: Callable[[dict], None] | None = None,
    deterministic_score: bool = False,
) -> Trainer:
    """Create a trainer that returns fixed results without actual training."""
    trainer = MagicMock(spec=Trainer)

    def _train(genome: dict) -> TrainResult:
        if on_train is not None:
            on_train(genome)
        score = 0.85
        if deterministic_score:
            score = 0.5 + (random.Random(str(sorted(genome.items()))).random() * 0.5)
        return TrainResult(
            validation_accuracy=score,
            training_loss=0.3,
            training_time_ms=100,
            status="completed",
        )

    trainer.train.side_effect = _train
    return trainer


def _small_space() -> SearchSpace:
    """A minimal search space for fast tests."""
    space = SearchSpace()
    space.add(ParamSpec("x", ParamType.CONTINUOUS, 0.0, 1.0))
    space.add(ParamSpec("y", ParamType.DISCRETE, 1, 5))
    space.add(ParamSpec("z", ParamType.CATEGORICAL, choices=("a", "b")))
    return space


class TestRandomSearch:
    def test_returns_metrics_store(self):
        strategy = RandomSearch(seed=42)
        store = strategy.run(_small_space(), _mock_trainer(), budget=5)
        assert store.run_name == "Random Search"
        assert len(store.generations) == 5

    def test_best_fitness_monotonically_increases(self):
        strategy = RandomSearch(seed=42)
        store = strategy.run(_small_space(), _mock_trainer(), budget=5)
        fitnesses = [g.best_fitness for g in store.generations]
        for i in range(1, len(fitnesses)):
            assert fitnesses[i] >= fitnesses[i - 1]

    def test_requires_budget_or_time_limit(self):
        strategy = RandomSearch(seed=42)
        try:
            strategy.run(_small_space(), _mock_trainer())
        except ValueError as exc:
            assert "At least one of budget or max_wall_clock_s" in str(exc)
        else:
            raise AssertionError("Expected ValueError when no limits are provided.")

    def test_name(self):
        assert RandomSearch().name() == "Random Search"


class TestGridSearch:
    def test_returns_metrics_store(self):
        strategy = GridSearch(grid_points=2)
        store = strategy.run(_small_space(), _mock_trainer(), budget=10)
        assert store.run_name == "Grid Search"
        assert len(store.generations) > 0

    def test_respects_budget(self):
        strategy = GridSearch(grid_points=2)
        store = strategy.run(_small_space(), _mock_trainer(), budget=3)
        assert len(store.generations) <= 3

    def test_name(self):
        assert GridSearch().name() == "Grid Search"


class TestBayesianSearch:
    def test_returns_metrics_store(self):
        strategy = BayesianSearch(seed=42)
        store = strategy.run(_small_space(), _mock_trainer(), budget=5)
        assert store.run_name == "Bayesian (Optuna TPE)"
        assert len(store.generations) == 5

    def test_name(self):
        assert BayesianSearch().name() == "Bayesian (Optuna TPE)"


class TestDifferentialEvolution:
    def test_returns_metrics_store(self):
        strategy = DifferentialEvolution(seed=42, population_size=6)
        store = strategy.run(_small_space(), _mock_trainer(), budget=7)
        assert store.run_name == "Differential Evolution"
        assert len(store.generations) == 7

    def test_emits_valid_values(self):
        emitted: list[dict] = []
        strategy = DifferentialEvolution(seed=42, population_size=6)
        store = strategy.run(
            _small_space(),
            _mock_trainer(on_train=lambda genome: emitted.append(dict(genome))),
            budget=10,
        )
        assert len(store.generations) == 10
        assert emitted
        for genome in emitted:
            assert 0.0 <= float(genome["x"]) <= 1.0
            assert 1 <= int(genome["y"]) <= 5
            assert genome["z"] in {"a", "b"}

    def test_name(self):
        assert DifferentialEvolution().name() == "Differential Evolution"


class TestSimulatedAnnealing:
    def test_returns_metrics_store(self):
        strategy = SimulatedAnnealing(seed=42)
        store = strategy.run(_small_space(), _mock_trainer(), budget=6)
        assert store.run_name == "Simulated Annealing"
        assert len(store.generations) == 6

    def test_emits_valid_values(self):
        emitted: list[dict] = []
        strategy = SimulatedAnnealing(seed=42)
        store = strategy.run(
            _small_space(),
            _mock_trainer(on_train=lambda genome: emitted.append(dict(genome))),
            budget=8,
        )
        assert len(store.generations) == 8
        assert emitted
        for genome in emitted:
            assert 0.0 <= float(genome["x"]) <= 1.0
            assert 1 <= int(genome["y"]) <= 5
            assert genome["z"] in {"a", "b"}

    def test_name(self):
        assert SimulatedAnnealing().name() == "Simulated Annealing"
