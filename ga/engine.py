"""GA engine: manages the evolutionary loop."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ga.individual import Individual
from ga.operators.crossover import CrossoverStrategy, UniformCrossover
from ga.operators.mutation import AdaptiveMutation, MutationStrategy
from ga.operators.selection import ElitistSelection, SelectionStrategy
from ga.population import Population
from ga.search_space import SearchSpace


@dataclass
class GAConfig:
    """Configuration for the genetic algorithm.

    Attributes:
        population_size: Number of individuals per generation.
        num_generations: Total generations to evolve.
        elite_count: Number of top individuals preserved via elitism.
        dataset: Dataset identifier passed to workers.
        epochs: Default training epochs per individual.
        seed: RNG seed for reproducibility (None = random).
    """

    population_size: int = 20
    num_generations: int = 10
    elite_count: int = 2
    dataset: str = "mnist"
    epochs: int = 10
    seed: int | None = None


class GAEngine:
    """Core genetic algorithm engine.

    Manages population initialization and per-generation evolution using
    pluggable selection, crossover, and mutation operators.

    Attributes:
        config: GA configuration.
        space: Hyperparameter search space.
        selection: Selection strategy.
        crossover: Crossover strategy.
        mutation: Mutation strategy.
    """

    def __init__(
        self,
        config: GAConfig,
        space: SearchSpace | None = None,
        selection: SelectionStrategy | None = None,
        crossover: CrossoverStrategy | None = None,
        mutation: MutationStrategy | None = None,
    ) -> None:
        self.config = config
        self.space = space or SearchSpace.default_cnn_space()
        self.selection = selection or ElitistSelection(elite_count=config.elite_count)
        self.crossover = crossover or UniformCrossover()
        self.mutation = mutation or AdaptiveMutation()
        self._rng = random.Random(config.seed)

    def initialize(self) -> Population:
        """Create the initial population with random genomes.

        Returns:
            A Population at generation 0 with random individuals.
        """
        individuals = []
        for _ in range(self.config.population_size):
            genome = self.space.random_sample(self._rng)
            genome["dataset"] = self.config.dataset
            if "epochs" not in genome:
                genome["epochs"] = self.config.epochs
            individuals.append(Individual(genome=genome))
        return Population(individuals=individuals, generation=0)

    def evolve(self, population: Population) -> Population:
        """Produce the next generation from the current population.

        Steps:
        1. Select parents (elites pass through unchanged).
        2. Pair parents and apply crossover to produce offspring.
        3. Apply mutation to non-elite offspring.
        4. Return new population.

        Args:
            population: Current generation with fitness values assigned.

        Returns:
            Next generation population (fitness values unset on new offspring).
        """
        pop_size = self.config.population_size
        next_gen = population.generation + 1

        # Elitism: preserve top individuals unchanged
        sorted_pop = population.sorted_by_fitness(descending=True)
        elites = [ind.clone() for ind in sorted_pop[: self.config.elite_count]]

        # Select parents for breeding
        num_parents = pop_size - len(elites)
        parents = self.selection.select(population.individuals, num_parents, self._rng)

        # Crossover parents in pairs to produce offspring
        offspring: list[Individual] = []
        for i in range(0, len(parents) - 1, 2):
            child_a, child_b = self.crossover.crossover(
                parents[i], parents[i + 1], self.space, self._rng
            )
            offspring.extend([child_a, child_b])

        # Handle odd parent count
        if len(parents) % 2 == 1:
            offspring.append(parents[-1].clone())

        # Trim to exact size needed
        offspring = offspring[: num_parents]

        # Mutate offspring (not elites)
        for ind in offspring:
            self.mutation.mutate(
                ind,
                self.space,
                self._rng,
                generation=next_gen,
                max_generations=self.config.num_generations,
            )
            # Ensure dataset and epochs are set
            ind.genome["dataset"] = self.config.dataset
            if "epochs" not in ind.genome:
                ind.genome["epochs"] = self.config.epochs

        new_individuals = elites + offspring
        return Population(individuals=new_individuals, generation=next_gen)
