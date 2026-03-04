"""Abstract base class for search strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ga.search_space import SearchSpace
from metrics.collector import MetricsStore
from worker.trainer import Trainer


class SearchStrategy(ABC):
    """Base class for hyperparameter search strategies.

    Each strategy searches over a SearchSpace with a fixed evaluation budget
    and records its progress in a MetricsStore.
    """

    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this strategy."""

    @abstractmethod
    def run(
        self,
        space: SearchSpace,
        budget: int,
        trainer: Trainer,
    ) -> MetricsStore:
        """Execute the search strategy.

        Args:
            space: Hyperparameter search space.
            budget: Total number of evaluations allowed.
            trainer: Trainer instance for evaluating configurations.

        Returns:
            MetricsStore with per-step metrics.
        """
