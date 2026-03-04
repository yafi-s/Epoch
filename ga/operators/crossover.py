"""Crossover strategies for the genetic algorithm."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

from ga.individual import Individual
from ga.search_space import ParamType, SearchSpace


class CrossoverStrategy(ABC):
    """Abstract base class for crossover operators."""

    @abstractmethod
    def crossover(
        self,
        parent_a: Individual,
        parent_b: Individual,
        space: SearchSpace,
        rng: random.Random,
    ) -> tuple[Individual, Individual]:
        """Produce two offspring from two parents.

        Args:
            parent_a: First parent.
            parent_b: Second parent.
            space: Search space for parameter metadata.
            rng: Random number generator.

        Returns:
            Two offspring individuals (fitness unset).
        """


class UniformCrossover(CrossoverStrategy):
    """Uniform crossover: each gene independently chosen from either parent.

    Attributes:
        swap_prob: Probability of taking the gene from parent_b.
    """

    def __init__(self, swap_prob: float = 0.5) -> None:
        self.swap_prob = swap_prob

    def crossover(
        self,
        parent_a: Individual,
        parent_b: Individual,
        space: SearchSpace,
        rng: random.Random,
    ) -> tuple[Individual, Individual]:
        """Perform uniform crossover."""
        child_a_genome: dict[str, Any] = {}
        child_b_genome: dict[str, Any] = {}

        for param in space.params:
            name = param.name
            if rng.random() < self.swap_prob:
                child_a_genome[name] = parent_b.genome[name]
                child_b_genome[name] = parent_a.genome[name]
            else:
                child_a_genome[name] = parent_a.genome[name]
                child_b_genome[name] = parent_b.genome[name]

        return Individual(genome=child_a_genome), Individual(genome=child_b_genome)


class BLXAlphaCrossover(CrossoverStrategy):
    """BLX-alpha crossover for continuous params, uniform for others.

    For continuous parameters, offspring values are sampled from the interval
    [min(p1,p2) - alpha*d, max(p1,p2) + alpha*d] where d = |p1-p2|.

    Attributes:
        alpha: Expansion factor beyond the parent range.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha

    def crossover(
        self,
        parent_a: Individual,
        parent_b: Individual,
        space: SearchSpace,
        rng: random.Random,
    ) -> tuple[Individual, Individual]:
        """Perform BLX-alpha crossover."""
        child_a_genome: dict[str, Any] = {}
        child_b_genome: dict[str, Any] = {}

        for param in space.params:
            name = param.name
            va = parent_a.genome[name]
            vb = parent_b.genome[name]

            if param.param_type == ParamType.CONTINUOUS:
                lo = min(va, vb)
                hi = max(va, vb)
                d = hi - lo
                expanded_lo = lo - self.alpha * d
                expanded_hi = hi + self.alpha * d
                child_a_genome[name] = param.clip(rng.uniform(expanded_lo, expanded_hi))
                child_b_genome[name] = param.clip(rng.uniform(expanded_lo, expanded_hi))
            elif param.param_type == ParamType.DISCRETE:
                lo = min(va, vb)
                hi = max(va, vb)
                child_a_genome[name] = param.clip(rng.randint(lo, hi))
                child_b_genome[name] = param.clip(rng.randint(lo, hi))
            else:
                # Categorical: randomly pick from either parent
                child_a_genome[name] = rng.choice([va, vb])
                child_b_genome[name] = rng.choice([va, vb])

        return Individual(genome=child_a_genome), Individual(genome=child_b_genome)
