"""Tests for benchmark strategies.

Uses a mock trainer to avoid actual model training during tests.
"""

from unittest.mock import MagicMock

from benchmarks.strategies.bayesian_search import BayesianSearch
from benchmarks.strategies.grid_search import GridSearch
from benchmarks.strategies.random_search import RandomSearch
from ga.search_space import ParamSpec, ParamType, SearchSpace
from worker.trainer import TrainResult, Trainer


def _mock_trainer() -> Trainer:
    """Create a trainer that returns fixed results without actual training."""
    trainer = MagicMock(spec=Trainer)
    trainer.train.return_value = TrainResult(
        validation_accuracy=0.85,
        training_loss=0.3,
        training_time_ms=100,
        status="completed",
    )
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
        store = strategy.run(_small_space(), budget=5, trainer=_mock_trainer())
        assert store.run_name == "Random Search"
        assert len(store.generations) == 5

    def test_best_fitness_monotonically_increases(self):
        strategy = RandomSearch(seed=42)
        store = strategy.run(_small_space(), budget=5, trainer=_mock_trainer())
        fitnesses = [g.best_fitness for g in store.generations]
        for i in range(1, len(fitnesses)):
            assert fitnesses[i] >= fitnesses[i - 1]

    def test_name(self):
        assert RandomSearch().name() == "Random Search"


class TestGridSearch:
    def test_returns_metrics_store(self):
        strategy = GridSearch(grid_points=2)
        store = strategy.run(_small_space(), budget=10, trainer=_mock_trainer())
        assert store.run_name == "Grid Search"
        assert len(store.generations) > 0

    def test_respects_budget(self):
        strategy = GridSearch(grid_points=2)
        store = strategy.run(_small_space(), budget=3, trainer=_mock_trainer())
        assert len(store.generations) <= 3

    def test_name(self):
        assert GridSearch().name() == "Grid Search"


class TestBayesianSearch:
    def test_returns_metrics_store(self):
        strategy = BayesianSearch(seed=42)
        store = strategy.run(_small_space(), budget=5, trainer=_mock_trainer())
        assert store.run_name == "Bayesian (Optuna TPE)"
        assert len(store.generations) == 5

    def test_name(self):
        assert BayesianSearch().name() == "Bayesian (Optuna TPE)"
