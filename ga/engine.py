"""GA engine: manages the evolutionary loop."""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

from ga.individual import Individual
from ga.operators.crossover import CrossoverStrategy, UniformCrossover
from ga.operators.mutation import AdaptiveMutation, MutationStrategy
from ga.operators.selection import ElitistSelection, SelectionStrategy
from ga.population import Population
from ga.search_space import ParamType, SearchSpace


@dataclass
class GAConfig:
    """Configuration for the genetic algorithm.

    Attributes:
        population_size: Number of individuals per generation.
        num_generations: Total generations to evolve.
        elite_count: Number of top individuals preserved via elitism.
        immigrant_rate: Fraction of non-elite slots replaced by random immigrants.
        plateau_patience_gens: Number of stale generations before triggering a temporary reheat.
        plateau_min_delta: Minimum best-fitness gain treated as meaningful progress.
        plateau_immigrant_rate: Temporary immigrant rate applied on plateau trigger.
        plateau_mutation_rate_floor: Temporary minimum mutation rate after plateau trigger.
        plateau_reheat_gens: Number of consecutive evolve steps to apply plateau reheating.
        plateau_reset_on_any_new_best: Reset stale counter on any new best, even below min_delta.
        genome_dedupe_max_retries: Attempts to resample a duplicate genome before fallback.
        adaptive_narrowing_enabled: Whether to narrow mutation/immigrant sampling around elites.
        adaptive_narrowing_start_gen: First generation index where narrowing can activate.
        adaptive_elite_fraction: Fraction of top individuals used to build adaptive bounds.
        adaptive_quantile: Quantile width used when computing elite-focused numeric bounds.
        adaptive_min_span_ratio: Minimum numeric span relative to original param span.
        adaptive_categorical_bias: Probability to prefer elite-supported categories.
        late_epoch_bias: Generation-progressive probability of biasing epochs upward.
        deterministic_eval: Metadata flag indicating deterministic worker evaluation is intended.
        deterministic_seed_offset: Metadata seed offset for deterministic worker evaluation.
        dataset: Dataset identifier passed to workers.
        epochs: Default training epochs per individual.
        seed: RNG seed for reproducibility (None = random).
    """

    population_size: int = 20
    num_generations: int = 10
    elite_count: int = 2
    immigrant_rate: float = 0.15
    plateau_patience_gens: int = 2
    plateau_min_delta: float = 1.0 / 1024.0
    plateau_immigrant_rate: float = 0.35
    plateau_mutation_rate_floor: float = 0.30
    plateau_reheat_gens: int = 2
    plateau_reset_on_any_new_best: bool = True
    genome_dedupe_max_retries: int = 8
    adaptive_narrowing_enabled: bool = True
    adaptive_narrowing_start_gen: int = 4
    adaptive_elite_fraction: float = 0.25
    adaptive_quantile: float = 0.20
    adaptive_min_span_ratio: float = 0.35
    adaptive_categorical_bias: float = 0.60
    late_epoch_bias: float = 0.0
    deterministic_eval: bool = True
    deterministic_seed_offset: int = 0
    dataset: str = "mnist"
    epochs: int = 10
    seed: int | None = None


@dataclass
class _ParamConstraint:
    low: float | None = None
    high: float | None = None
    preferred_choices: tuple[Any, ...] | None = None


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
        self._adaptive_constraints: dict[str, _ParamConstraint] = {}

    def initialize(self) -> Population:
        """Create the initial population with random genomes.

        Returns:
            A Population at generation 0 with random individuals.
        """
        self._adaptive_constraints.clear()
        individuals = []
        for _ in range(self.config.population_size):
            individuals.append(self._make_random_individual(use_constraints=False))
        return Population(individuals=individuals, generation=0)

    def evolve(
        self,
        population: Population,
        immigrant_rate_override: float | None = None,
        mutation_rate_floor_override: float | None = None,
        reexpand_constraints: bool = False,
    ) -> Population:
        """Produce the next generation from the current population.

        Steps:
        1. Select parents (elites pass through unchanged).
        2. Pair parents and apply crossover to produce offspring.
        3. Apply mutation to non-elite offspring.
        4. Return new population.

        Args:
            population: Current generation with fitness values assigned.
            immigrant_rate_override: Optional immigrant rate override for this evolve step.
            mutation_rate_floor_override: Optional mutation-rate floor for this step.
            reexpand_constraints: If True, clear adaptive bounds for this evolve step.

        Returns:
            Next generation population (fitness values unset on new offspring).
        """
        pop_size = self.config.population_size
        next_gen = population.generation + 1

        if reexpand_constraints:
            self._adaptive_constraints.clear()
        elif (
            self.config.adaptive_narrowing_enabled
            and population.generation >= self.config.adaptive_narrowing_start_gen
        ):
            self._update_adaptive_constraints(population)
        else:
            self._adaptive_constraints.clear()

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
                rate_floor=mutation_rate_floor_override,
            )
            self._apply_adaptive_constraints(ind)
            self._apply_late_epoch_bias(ind, generation=next_gen)
            # Ensure dataset and epochs are set
            ind.genome["dataset"] = self.config.dataset
            if "epochs" not in ind.genome:
                ind.genome["epochs"] = self.config.epochs

        immigrant_rate = (
            self.config.immigrant_rate
            if immigrant_rate_override is None
            else float(immigrant_rate_override)
        )
        self._inject_immigrants(offspring, immigrant_rate)

        new_individuals = elites + offspring
        self._dedupe_population(new_individuals, next_gen)
        return Population(individuals=new_individuals, generation=next_gen)

    def _make_random_individual(self, *, use_constraints: bool = True) -> Individual:
        genome = {}
        for param in self.space.params:
            constraint = self._adaptive_constraints.get(param.name) if use_constraints else None
            genome[param.name] = self._sample_param(param, constraint)
        genome["dataset"] = self.config.dataset
        if "epochs" not in genome:
            genome["epochs"] = self.config.epochs
        return Individual(genome=genome, fitness=0.0)

    @staticmethod
    def _genome_key(genome: dict[str, object]) -> str:
        return json.dumps(genome, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _quantile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        if len(sorted_values) == 1:
            return float(sorted_values[0])
        q = min(max(float(q), 0.0), 1.0)
        position = q * float(len(sorted_values) - 1)
        low_idx = int(math.floor(position))
        high_idx = int(math.ceil(position))
        if low_idx == high_idx:
            return float(sorted_values[low_idx])
        weight = position - float(low_idx)
        return float(
            sorted_values[low_idx] * (1.0 - weight) + sorted_values[high_idx] * weight
        )

    def _sample_param(self, param, constraint: _ParamConstraint | None) -> Any:
        if param.param_type == ParamType.CONTINUOUS:
            low = float(constraint.low) if constraint and constraint.low is not None else float(param.low)
            high = float(constraint.high) if constraint and constraint.high is not None else float(param.high)
            if low > high:
                low, high = high, low
            if param.log_scale:
                low = max(low, 1e-12)
                high = max(high, low + 1e-12)
                return math.exp(self._rng.uniform(math.log(low), math.log(high)))
            return self._rng.uniform(low, high)

        if param.param_type == ParamType.DISCRETE:
            low = int(round(constraint.low)) if constraint and constraint.low is not None else int(param.low)
            high = int(round(constraint.high)) if constraint and constraint.high is not None else int(param.high)
            if low > high:
                low, high = high, low
            return self._rng.randint(low, high)

        choices = list(param.choices)
        if not choices:
            return None
        preferred: list[Any] = []
        if constraint and constraint.preferred_choices:
            preferred = [c for c in constraint.preferred_choices if c in choices]
        bias = min(max(float(self.config.adaptive_categorical_bias), 0.0), 1.0)
        if preferred and len(preferred) < len(choices) and self._rng.random() < bias:
            return self._rng.choice(preferred)
        if preferred and len(preferred) == len(choices):
            return self._rng.choice(preferred)
        return self._rng.choice(choices)

    def _update_adaptive_constraints(self, population: Population) -> None:
        if not population.individuals:
            self._adaptive_constraints.clear()
            return

        elite_fraction = min(max(float(self.config.adaptive_elite_fraction), 0.05), 1.0)
        elite_count = max(2, int(round(len(population.individuals) * elite_fraction)))
        elite_count = min(len(population.individuals), elite_count)
        elites = population.sorted_by_fitness(descending=True)[:elite_count]

        q = min(max(float(self.config.adaptive_quantile), 0.0), 0.49)
        min_span_ratio = min(max(float(self.config.adaptive_min_span_ratio), 0.0), 1.0)
        constraints: dict[str, _ParamConstraint] = {}

        for param in self.space.params:
            values = [ind.genome.get(param.name) for ind in elites if param.name in ind.genome]
            if not values:
                continue

            if param.param_type in (ParamType.CONTINUOUS, ParamType.DISCRETE):
                numeric_values = [float(v) for v in values]
                lo = self._quantile(numeric_values, q)
                hi = self._quantile(numeric_values, 1.0 - q)
                if lo > hi:
                    lo, hi = hi, lo

                base_low = float(param.low)
                base_high = float(param.high)
                base_span = max(base_high - base_low, 1e-12)
                target_span = max(hi - lo, base_span * min_span_ratio)
                center = 0.5 * (lo + hi)
                narrowed_low = max(base_low, center - 0.5 * target_span)
                narrowed_high = min(base_high, center + 0.5 * target_span)

                if param.param_type == ParamType.DISCRETE:
                    narrowed_low = float(param.clip(int(round(narrowed_low))))
                    narrowed_high = float(param.clip(int(round(narrowed_high))))
                    if narrowed_low > narrowed_high:
                        narrowed_low, narrowed_high = narrowed_high, narrowed_low

                constraints[param.name] = _ParamConstraint(
                    low=float(narrowed_low),
                    high=float(narrowed_high),
                )
                continue

            counts = Counter(values)
            preferred = [choice for choice, _freq in counts.most_common() if choice in param.choices]
            if preferred:
                constraints[param.name] = _ParamConstraint(preferred_choices=tuple(preferred))

        self._adaptive_constraints = constraints

    def _apply_adaptive_constraints(self, individual: Individual) -> None:
        if not self._adaptive_constraints:
            return

        bias = min(max(float(self.config.adaptive_categorical_bias), 0.0), 1.0)
        for param in self.space.params:
            constraint = self._adaptive_constraints.get(param.name)
            if constraint is None or param.name not in individual.genome:
                continue

            if param.param_type in (ParamType.CONTINUOUS, ParamType.DISCRETE):
                low = float(constraint.low) if constraint.low is not None else float(param.low)
                high = float(constraint.high) if constraint.high is not None else float(param.high)
                if low > high:
                    low, high = high, low
                current = float(individual.genome[param.name])
                clipped = max(low, min(high, current))
                if param.param_type == ParamType.DISCRETE:
                    clipped = int(round(clipped))
                individual.genome[param.name] = param.clip(clipped)
                continue

            preferred = (
                [c for c in constraint.preferred_choices if c in param.choices]
                if constraint.preferred_choices
                else []
            )
            if preferred and individual.genome[param.name] not in preferred and self._rng.random() < bias:
                individual.genome[param.name] = self._rng.choice(preferred)
            else:
                individual.genome[param.name] = param.clip(individual.genome[param.name])

    def _apply_late_epoch_bias(self, individual: Individual, generation: int) -> None:
        if self.config.late_epoch_bias <= 0.0:
            return

        epoch_param = None
        for param in self.space.params:
            if param.name == "epochs":
                epoch_param = param
                break
        if epoch_param is None or "epochs" not in individual.genome:
            return

        max_generations = max(self.config.num_generations - 1, 1)
        progress = min(max(float(generation) / float(max_generations), 0.0), 1.0)
        chance = min(max(float(self.config.late_epoch_bias), 0.0), 1.0) * progress
        if self._rng.random() >= chance:
            return

        current_epoch = int(individual.genome.get("epochs", self.config.epochs))
        if epoch_param.param_type == ParamType.DISCRETE:
            low = max(int(epoch_param.low), current_epoch)
            high = int(epoch_param.high)
            if low <= high:
                individual.genome["epochs"] = epoch_param.clip(self._rng.randint(low, high))
            return

        if epoch_param.param_type == ParamType.CATEGORICAL and epoch_param.choices:
            numeric_choices = [int(v) for v in epoch_param.choices if isinstance(v, (int, float))]
            if not numeric_choices:
                return
            higher = [v for v in numeric_choices if v >= current_epoch]
            target_choices = higher if higher else numeric_choices
            individual.genome["epochs"] = epoch_param.clip(self._rng.choice(target_choices))

    def _inject_immigrants(self, offspring: list[Individual], immigrant_rate: float) -> None:
        if not offspring:
            return
        rate = max(0.0, min(1.0, float(immigrant_rate)))
        if rate <= 0.0:
            return

        immigrant_count = int(round(len(offspring) * rate))
        immigrant_count = min(len(offspring), immigrant_count)
        for i in range(immigrant_count):
            replace_idx = len(offspring) - 1 - i
            offspring[replace_idx] = self._make_random_individual(use_constraints=True)

    def _dedupe_population(self, individuals: list[Individual], generation: int) -> None:
        seen: set[str] = set()
        retries = max(0, int(self.config.genome_dedupe_max_retries))

        for ind in individuals:
            key = self._genome_key(ind.genome)
            if key not in seen:
                seen.add(key)
                continue

            replaced = False
            for _ in range(retries):
                candidate = self._make_random_individual(use_constraints=True)
                candidate_key = self._genome_key(candidate.genome)
                if candidate_key in seen:
                    continue
                ind.genome = candidate.genome
                ind.fitness = 0.0
                seen.add(candidate_key)
                replaced = True
                break

            if replaced:
                continue

            fallback = ind.clone()
            self.mutation.mutate(
                fallback,
                self.space,
                self._rng,
                generation=generation,
                max_generations=self.config.num_generations,
                rate_floor=None,
            )
            self._apply_adaptive_constraints(fallback)
            fallback.genome["dataset"] = self.config.dataset
            if "epochs" not in fallback.genome:
                fallback.genome["epochs"] = self.config.epochs
            fallback_key = self._genome_key(fallback.genome)
            if fallback_key in seen:
                continue
            ind.genome = fallback.genome
            ind.fitness = 0.0
            seen.add(fallback_key)
