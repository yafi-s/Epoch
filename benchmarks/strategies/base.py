"""Abstract base class for search strategies."""

from __future__ import annotations

import time
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
        trainer: Trainer,
        *,
        budget: int | None = None,
        max_wall_clock_s: float | None = None,
    ) -> MetricsStore:
        """Execute the search strategy.

        Args:
            space: Hyperparameter search space.
            trainer: Trainer instance for evaluating configurations.
            budget: Optional total number of evaluations allowed.
            max_wall_clock_s: Optional max wall-clock budget in seconds.

        Returns:
            MetricsStore with per-step metrics.
        """

    @staticmethod
    def _resolve_limits(
        *,
        budget: int | None,
        max_wall_clock_s: float | None,
    ) -> tuple[int | None, float | None]:
        """Normalize run limits and return (budget, deadline_monotonic)."""
        normalized_budget = None
        if budget is not None and int(budget) > 0:
            normalized_budget = int(budget)

        normalized_time = None
        if max_wall_clock_s is not None and float(max_wall_clock_s) > 0:
            normalized_time = float(max_wall_clock_s)

        if normalized_budget is None and normalized_time is None:
            raise ValueError("At least one of budget or max_wall_clock_s must be > 0.")

        deadline = None
        if normalized_time is not None:
            deadline = time.monotonic() + normalized_time

        return normalized_budget, deadline

    @staticmethod
    def _should_stop(
        *,
        evals_completed: int,
        budget: int | None,
        deadline_monotonic: float | None,
    ) -> bool:
        """Return True when either budget or wall-clock limit has been reached."""
        if budget is not None and evals_completed >= budget:
            return True
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            return True
        return False
