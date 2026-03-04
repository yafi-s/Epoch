"""Population container for the genetic algorithm."""

from __future__ import annotations

from dataclasses import dataclass, field

from ga.individual import Individual


@dataclass
class Population:
    """A collection of individuals representing one generation.

    Attributes:
        individuals: The candidate solutions in this generation.
        generation: The generation number (0-indexed).
    """

    individuals: list[Individual] = field(default_factory=list)
    generation: int = 0

    @property
    def size(self) -> int:
        """Number of individuals in the population."""
        return len(self.individuals)

    @property
    def best(self) -> Individual:
        """Individual with the highest fitness."""
        return max(self.individuals, key=lambda ind: ind.fitness)

    @property
    def worst(self) -> Individual:
        """Individual with the lowest fitness."""
        return min(self.individuals, key=lambda ind: ind.fitness)

    @property
    def avg_fitness(self) -> float:
        """Mean fitness across the population."""
        if not self.individuals:
            return 0.0
        return sum(ind.fitness for ind in self.individuals) / len(self.individuals)

    @property
    def best_fitness(self) -> float:
        """Highest fitness in the population."""
        return self.best.fitness if self.individuals else 0.0

    @property
    def worst_fitness(self) -> float:
        """Lowest fitness in the population."""
        return self.worst.fitness if self.individuals else 0.0

    def sorted_by_fitness(self, descending: bool = True) -> list[Individual]:
        """Return individuals sorted by fitness."""
        return sorted(self.individuals, key=lambda ind: ind.fitness, reverse=descending)
