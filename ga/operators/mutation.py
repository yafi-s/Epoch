"""Mutation strategies for the genetic algorithm."""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod

from ga.individual import Individual
from ga.search_space import ParamType, SearchSpace


class MutationStrategy(ABC):
    """Abstract base class for mutation operators."""

    @abstractmethod
    def mutate(
        self,
        individual: Individual,
        space: SearchSpace,
        rng: random.Random,
        generation: int = 0,
        max_generations: int = 1,
    ) -> Individual:
        """Mutate an individual in-place and return it.

        Args:
            individual: The individual to mutate (modified in-place).
            space: Search space for parameter metadata and bounds.
            rng: Random number generator.
            generation: Current generation number (for adaptive schemes).
            max_generations: Total number of generations (for adaptive schemes).

        Returns:
            The mutated individual.
        """


class GaussianMutation(MutationStrategy):
    """Gaussian mutation for continuous/discrete params, random reset for categorical.

    Attributes:
        mutation_rate: Per-gene probability of mutation.
        sigma: Standard deviation of Gaussian noise (as fraction of param range).
    """

    def __init__(self, mutation_rate: float = 0.1, sigma: float = 0.2) -> None:
        self.mutation_rate = mutation_rate
        self.sigma = sigma

    def mutate(
        self,
        individual: Individual,
        space: SearchSpace,
        rng: random.Random,
        generation: int = 0,
        max_generations: int = 1,
    ) -> Individual:
        """Apply Gaussian mutation to each gene with probability mutation_rate."""
        for param in space.params:
            if rng.random() >= self.mutation_rate:
                continue

            name = param.name
            if param.param_type == ParamType.CONTINUOUS:
                range_size = param.high - param.low
                noise = rng.gauss(0, self.sigma * range_size)
                individual.genome[name] = param.clip(individual.genome[name] + noise)
            elif param.param_type == ParamType.DISCRETE:
                range_size = param.high - param.low
                noise = rng.gauss(0, self.sigma * range_size)
                individual.genome[name] = param.clip(individual.genome[name] + noise)
            else:
                individual.genome[name] = rng.choice(param.choices)

        return individual


class AdaptiveMutation(MutationStrategy):
    """Adaptive mutation: rate decays linearly from initial to final over generations.

    Attributes:
        initial_rate: Mutation rate at generation 0.
        final_rate: Mutation rate at the last generation.
        sigma: Gaussian noise scale (fraction of param range).
    """

    def __init__(
        self,
        initial_rate: float = 0.3,
        final_rate: float = 0.05,
        sigma: float = 0.2,
    ) -> None:
        self.initial_rate = initial_rate
        self.final_rate = final_rate
        self.sigma = sigma

    def _current_rate(self, generation: int, max_generations: int) -> float:
        """Linearly interpolate mutation rate based on generation progress."""
        if max_generations <= 1:
            return self.initial_rate
        t = generation / (max_generations - 1)
        return self.initial_rate + t * (self.final_rate - self.initial_rate)

    def mutate(
        self,
        individual: Individual,
        space: SearchSpace,
        rng: random.Random,
        generation: int = 0,
        max_generations: int = 1,
    ) -> Individual:
        """Apply mutation with generation-adaptive rate."""
        rate = self._current_rate(generation, max_generations)

        for param in space.params:
            if rng.random() >= rate:
                continue

            name = param.name
            if param.param_type == ParamType.CONTINUOUS:
                range_size = param.high - param.low
                noise = rng.gauss(0, self.sigma * range_size)
                individual.genome[name] = param.clip(individual.genome[name] + noise)
            elif param.param_type == ParamType.DISCRETE:
                range_size = param.high - param.low
                noise = rng.gauss(0, self.sigma * range_size)
                individual.genome[name] = param.clip(individual.genome[name] + noise)
            else:
                individual.genome[name] = rng.choice(param.choices)

        return individual
