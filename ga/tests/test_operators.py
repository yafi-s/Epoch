"""Tests for GA operators: selection, crossover, mutation."""

import random

from ga.individual import Individual
from ga.operators.crossover import BLXAlphaCrossover, UniformCrossover
from ga.operators.mutation import AdaptiveMutation, GaussianMutation
from ga.operators.selection import ElitistSelection, TournamentSelection
from ga.search_space import ParamSpec, ParamType, SearchSpace


def _make_space() -> SearchSpace:
    space = SearchSpace()
    space.add(ParamSpec("x", ParamType.CONTINUOUS, 0.0, 1.0))
    space.add(ParamSpec("y", ParamType.DISCRETE, 1, 10))
    space.add(ParamSpec("z", ParamType.CATEGORICAL, choices=("a", "b", "c")))
    return space


def _make_population(n: int = 10) -> list[Individual]:
    rng = random.Random(42)
    space = _make_space()
    pop = []
    for i in range(n):
        genome = space.random_sample(rng)
        pop.append(Individual(genome=genome, fitness=i / n))
    return pop


# ─── Selection ────────────────────────────────────────────────────────────────


class TestTournamentSelection:
    def test_returns_correct_count(self):
        pop = _make_population()
        sel = TournamentSelection(tournament_size=3)
        parents = sel.select(pop, 5, random.Random(0))
        assert len(parents) == 5

    def test_tends_to_select_fitter(self):
        pop = _make_population(20)
        sel = TournamentSelection(tournament_size=5)
        rng = random.Random(42)
        parents = sel.select(pop, 100, rng)
        avg_fitness = sum(p.fitness for p in parents) / len(parents)
        pop_avg = sum(p.fitness for p in pop) / len(pop)
        assert avg_fitness > pop_avg


class TestElitistSelection:
    def test_elites_always_included(self):
        pop = _make_population(10)
        sel = ElitistSelection(elite_count=2)
        parents = sel.select(pop, 5, random.Random(0))
        top_two_ids = {ind.id for ind in sorted(pop, key=lambda i: i.fitness, reverse=True)[:2]}
        parent_ids = {p.id for p in parents}
        assert top_two_ids.issubset(parent_ids)

    def test_returns_correct_count(self):
        pop = _make_population()
        sel = ElitistSelection(elite_count=2)
        parents = sel.select(pop, 6, random.Random(0))
        assert len(parents) == 6


# ─── Crossover ────────────────────────────────────────────────────────────────


class TestUniformCrossover:
    def test_produces_two_children(self):
        space = _make_space()
        rng = random.Random(42)
        a = Individual(genome=space.random_sample(rng))
        b = Individual(genome=space.random_sample(rng))
        cx = UniformCrossover()
        c1, c2 = cx.crossover(a, b, space, rng)
        assert set(c1.genome.keys()) == set(space.names())
        assert set(c2.genome.keys()) == set(space.names())

    def test_children_have_zero_fitness(self):
        space = _make_space()
        rng = random.Random(42)
        a = Individual(genome=space.random_sample(rng), fitness=0.9)
        b = Individual(genome=space.random_sample(rng), fitness=0.8)
        cx = UniformCrossover()
        c1, c2 = cx.crossover(a, b, space, rng)
        assert c1.fitness == 0.0
        assert c2.fitness == 0.0

    def test_genes_come_from_parents(self):
        space = _make_space()
        rng = random.Random(42)
        a = Individual(genome=space.random_sample(rng))
        b = Individual(genome=space.random_sample(rng))
        cx = UniformCrossover()
        c1, c2 = cx.crossover(a, b, space, random.Random(0))
        for name in space.names():
            assert c1.genome[name] in (a.genome[name], b.genome[name])
            assert c2.genome[name] in (a.genome[name], b.genome[name])


class TestBLXAlphaCrossover:
    def test_continuous_values_in_expanded_range(self):
        space = SearchSpace()
        space.add(ParamSpec("x", ParamType.CONTINUOUS, 0.0, 1.0))
        a = Individual(genome={"x": 0.3})
        b = Individual(genome={"x": 0.7})
        cx = BLXAlphaCrossover(alpha=0.5)
        rng = random.Random(42)
        c1, c2 = cx.crossover(a, b, space, rng)
        # Should be clipped to [0, 1]
        assert 0.0 <= c1.genome["x"] <= 1.0
        assert 0.0 <= c2.genome["x"] <= 1.0


# ─── Mutation ─────────────────────────────────────────────────────────────────


class TestGaussianMutation:
    def test_mutation_respects_bounds(self):
        space = _make_space()
        rng = random.Random(42)
        ind = Individual(genome=space.random_sample(rng))
        mut = GaussianMutation(mutation_rate=1.0, sigma=1.0)
        mut.mutate(ind, space, rng)
        assert 0.0 <= ind.genome["x"] <= 1.0
        assert 1 <= ind.genome["y"] <= 10
        assert ind.genome["z"] in ("a", "b", "c")

    def test_zero_rate_no_change(self):
        space = _make_space()
        rng = random.Random(42)
        ind = Individual(genome=space.random_sample(rng))
        original = dict(ind.genome)
        mut = GaussianMutation(mutation_rate=0.0)
        mut.mutate(ind, space, rng)
        assert ind.genome == original


class TestAdaptiveMutation:
    def test_rate_decays_over_generations(self):
        mut = AdaptiveMutation(initial_rate=0.5, final_rate=0.1)
        assert mut._current_rate(0, 10) == 0.5
        assert abs(mut._current_rate(9, 10) - 0.1) < 1e-9
        mid_rate = mut._current_rate(5, 10)
        assert 0.1 < mid_rate < 0.5

    def test_mutation_respects_bounds(self):
        space = _make_space()
        rng = random.Random(42)
        ind = Individual(genome=space.random_sample(rng))
        mut = AdaptiveMutation(initial_rate=1.0, sigma=1.0)
        mut.mutate(ind, space, rng, generation=0, max_generations=10)
        assert 0.0 <= ind.genome["x"] <= 1.0
        assert 1 <= ind.genome["y"] <= 10
        assert ind.genome["z"] in ("a", "b", "c")
