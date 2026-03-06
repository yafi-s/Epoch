"""Tests for GAEngine."""

import random

from ga.engine import GAConfig, GAEngine
from ga.search_space import ParamSpec, ParamType, SearchSpace


def _simple_space() -> SearchSpace:
    space = SearchSpace()
    space.add(ParamSpec("x", ParamType.CONTINUOUS, 0.0, 1.0))
    space.add(ParamSpec("y", ParamType.DISCRETE, 1, 10))
    space.add(ParamSpec("z", ParamType.CATEGORICAL, choices=("a", "b", "c")))
    return space


class TestGAEngine:
    def test_initialize_creates_correct_population_size(self):
        config = GAConfig(population_size=15, seed=42)
        engine = GAEngine(config, space=_simple_space())
        pop = engine.initialize()
        assert pop.size == 15
        assert pop.generation == 0

    def test_initialize_genomes_are_valid(self):
        config = GAConfig(population_size=10, seed=42)
        space = _simple_space()
        engine = GAEngine(config, space=space)
        pop = engine.initialize()
        for ind in pop.individuals:
            assert 0.0 <= ind.genome["x"] <= 1.0
            assert 1 <= ind.genome["y"] <= 10
            assert ind.genome["z"] in ("a", "b", "c")

    def test_evolve_preserves_population_size(self):
        config = GAConfig(population_size=10, elite_count=2, seed=42)
        engine = GAEngine(config, space=_simple_space())
        pop = engine.initialize()
        # Assign mock fitness
        for i, ind in enumerate(pop.individuals):
            ind.fitness = i / 10.0
        next_pop = engine.evolve(pop)
        assert next_pop.size == 10
        assert next_pop.generation == 1

    def test_evolve_increments_generation(self):
        config = GAConfig(population_size=8, seed=42)
        engine = GAEngine(config, space=_simple_space())
        pop = engine.initialize()
        for ind in pop.individuals:
            ind.fitness = random.random()
        pop2 = engine.evolve(pop)
        assert pop2.generation == 1
        pop3 = engine.evolve(pop2)
        assert pop3.generation == 2

    def test_elites_preserved_in_next_generation(self):
        config = GAConfig(population_size=10, elite_count=2, seed=42)
        engine = GAEngine(config, space=_simple_space())
        pop = engine.initialize()
        for i, ind in enumerate(pop.individuals):
            ind.fitness = float(i)

        best_two = pop.sorted_by_fitness(descending=True)[:2]
        best_genomes = [dict(ind.genome) for ind in best_two]

        next_pop = engine.evolve(pop)
        next_genomes = [dict(ind.genome) for ind in next_pop.individuals]

        # The top 2 genomes should appear in the next generation
        for bg in best_genomes:
            assert bg in next_genomes

    def test_multiple_generations_with_mock_fitness(self):
        """Run several generations with a simple fitness function to ensure no crashes."""
        config = GAConfig(population_size=12, num_generations=5, elite_count=2, seed=42)
        space = _simple_space()
        engine = GAEngine(config, space=space)
        pop = engine.initialize()

        for gen in range(config.num_generations):
            # Mock fitness: closer x is to 0.7, the better
            for ind in pop.individuals:
                ind.fitness = 1.0 - abs(ind.genome["x"] - 0.7)
            pop = engine.evolve(pop)

        assert pop.generation == 5
        assert pop.size == 12

    def test_evolve_with_immigrants_keeps_population_size(self):
        config = GAConfig(population_size=10, elite_count=2, immigrant_rate=0.2, seed=42)
        engine = GAEngine(config, space=_simple_space())
        pop = engine.initialize()
        for i, ind in enumerate(pop.individuals):
            ind.fitness = i / 10.0

        next_pop = engine.evolve(pop)
        assert next_pop.size == 10
        assert next_pop.generation == 1

    def test_dedupe_replaces_duplicate_genomes_when_space_allows(self):
        config = GAConfig(
            population_size=8,
            elite_count=2,
            immigrant_rate=0.0,
            genome_dedupe_max_retries=50,
            seed=7,
        )
        space = _simple_space()
        engine = GAEngine(config, space=space)

        base = {"x": 0.3, "y": 5, "z": "a", "dataset": "mnist", "epochs": 1}
        pop = engine.initialize()
        for i, ind in enumerate(pop.individuals):
            ind.genome = dict(base)
            ind.fitness = float(i)

        next_pop = engine.evolve(pop)
        keys = {
            tuple(sorted(ind.genome.items()))
            for ind in next_pop.individuals
        }
        assert len(keys) == next_pop.size

    def test_dedupe_fallback_does_not_crash_when_retries_exhausted(self):
        tiny = SearchSpace()
        tiny.add(ParamSpec("x", ParamType.CATEGORICAL, choices=(0,)))
        config = GAConfig(
            population_size=6,
            elite_count=2,
            immigrant_rate=0.0,
            genome_dedupe_max_retries=0,
            seed=1,
        )
        engine = GAEngine(config, space=tiny)
        pop = engine.initialize()
        for i, ind in enumerate(pop.individuals):
            ind.fitness = float(i)

        next_pop = engine.evolve(pop)
        assert next_pop.size == 6

    def test_adaptive_narrowing_uses_elite_distribution(self):
        config = GAConfig(
            population_size=8,
            elite_count=2,
            adaptive_narrowing_enabled=True,
            adaptive_narrowing_start_gen=0,
            adaptive_elite_fraction=0.25,
            adaptive_quantile=0.0,
            adaptive_min_span_ratio=0.10,
            seed=11,
        )
        engine = GAEngine(config, space=_simple_space())
        pop = engine.initialize()

        for i, ind in enumerate(pop.individuals):
            if i < 2:
                ind.genome["x"] = 0.85 + (0.01 * i)
                ind.genome["y"] = 9 + i
                ind.genome["z"] = "c"
                ind.fitness = 1.0 - (i * 0.01)
            else:
                ind.genome["x"] = 0.1
                ind.genome["y"] = 1
                ind.genome["z"] = "a"
                ind.fitness = 0.1

        engine._update_adaptive_constraints(pop)
        constraints = engine._adaptive_constraints

        assert "x" in constraints
        assert constraints["x"].low is not None
        assert constraints["x"].high is not None
        assert 0.0 <= constraints["x"].low <= constraints["x"].high <= 1.0

        assert "y" in constraints
        assert constraints["y"].low is not None
        assert constraints["y"].high is not None
        assert 1 <= int(round(constraints["y"].low)) <= int(round(constraints["y"].high)) <= 10

        assert "z" in constraints
        assert constraints["z"].preferred_choices is not None
        assert constraints["z"].preferred_choices[0] == "c"

    def test_reexpand_constraints_clears_adaptive_bounds_for_next_evolve(self):
        config = GAConfig(
            population_size=8,
            elite_count=2,
            adaptive_narrowing_enabled=True,
            adaptive_narrowing_start_gen=0,
            seed=13,
        )
        engine = GAEngine(config, space=_simple_space())
        pop = engine.initialize()
        for i, ind in enumerate(pop.individuals):
            ind.fitness = float(i)

        _ = engine.evolve(pop)
        assert engine._adaptive_constraints

        _ = engine.evolve(pop, reexpand_constraints=True)
        assert engine._adaptive_constraints == {}
