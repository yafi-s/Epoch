"""Tests for Individual."""

from ga.individual import Individual
from ga.search_space import SearchSpace


def test_individual_clone_is_independent():
    ind = Individual(genome={"lr": 0.01, "batch": 32}, fitness=0.9)
    clone = ind.clone()

    assert clone.genome == ind.genome
    assert clone.fitness == ind.fitness
    assert clone.id != ind.id

    clone.genome["lr"] = 0.1
    assert ind.genome["lr"] == 0.01


def test_individual_default_fitness_is_zero():
    ind = Individual(genome={"x": 1})
    assert ind.fitness == 0.0


def test_individual_genome_access():
    genome = {"learning_rate": 0.001, "batch_size": 64, "optimizer": "adam"}
    ind = Individual(genome=genome)
    assert ind.genome["learning_rate"] == 0.001
    assert ind.genome["batch_size"] == 64
