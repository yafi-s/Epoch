"""Selection strategies for the genetic algorithm."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

from ga.individual import Individual


class SelectionStrategy(ABC):
    """Abstract base class for selection operators."""

    @abstractmethod
    def select(
        self, population: list[Individual], num_parents: int, rng: random.Random
    ) -> list[Individual]:
        """Select parents from the population.

        Args:
            population: Current generation individuals (with fitness assigned).
            num_parents: How many parents to select.
            rng: Random number generator for reproducibility.

        Returns:
            Selected parent individuals.
        """


class TournamentSelection(SelectionStrategy):
    """Tournament selection: pick k random individuals, keep the best.

    Attributes:
        tournament_size: Number of individuals per tournament.
    """

    def __init__(self, tournament_size: int = 3) -> None:
        self.tournament_size = tournament_size

    def select(
        self, population: list[Individual], num_parents: int, rng: random.Random
    ) -> list[Individual]:
        """Run tournaments to select parents."""
        parents: list[Individual] = []
        for _ in range(num_parents):
            contestants = rng.sample(population, min(self.tournament_size, len(population)))
            winner = max(contestants, key=lambda ind: ind.fitness)
            parents.append(winner)
        return parents


class ElitistSelection(SelectionStrategy):
    """Elitist selection: always include top-N, fill the rest with tournament.

    Attributes:
        elite_count: Number of top individuals guaranteed selection.
        tournament_size: Tournament size for remaining slots.
    """

    def __init__(self, elite_count: int = 2, tournament_size: int = 3) -> None:
        self.elite_count = elite_count
        self.tournament_size = tournament_size

    def select(
        self, population: list[Individual], num_parents: int, rng: random.Random
    ) -> list[Individual]:
        """Select elites first, then fill remaining slots via tournament."""
        sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
        elites = sorted_pop[: min(self.elite_count, num_parents)]

        remaining = num_parents - len(elites)
        if remaining <= 0:
            return elites[:num_parents]

        tournament = TournamentSelection(self.tournament_size)
        others = tournament.select(population, remaining, rng)
        return elites + others
