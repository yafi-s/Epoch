"""Metrics collection and persistence for GA optimization runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GenerationMetrics:
    """Metrics for a single generation.

    Attributes:
        generation: Generation number.
        best_fitness: Highest fitness in this generation.
        avg_fitness: Mean fitness across the population.
        worst_fitness: Lowest fitness in this generation.
        wall_clock_ms: Wall-clock time for the generation in milliseconds.
    """

    generation: int
    best_fitness: float
    avg_fitness: float
    worst_fitness: float
    wall_clock_ms: int


@dataclass
class MetricsStore:
    """Stores per-generation metrics for a single optimization run.

    Attributes:
        run_name: Human-readable name for this run.
        generations: List of per-generation metrics.
    """

    run_name: str = "default"
    generations: list[GenerationMetrics] = field(default_factory=list)

    def record_generation(
        self,
        generation: int,
        best_fitness: float,
        avg_fitness: float,
        worst_fitness: float,
        wall_clock_ms: int,
    ) -> None:
        """Record metrics for a completed generation."""
        self.generations.append(
            GenerationMetrics(
                generation=generation,
                best_fitness=best_fitness,
                avg_fitness=avg_fitness,
                worst_fitness=worst_fitness,
                wall_clock_ms=wall_clock_ms,
            )
        )

    @property
    def total_wall_clock_ms(self) -> int:
        """Total wall-clock time across all generations."""
        return sum(g.wall_clock_ms for g in self.generations)

    @property
    def final_best_fitness(self) -> float:
        """Best fitness from the last recorded generation."""
        return self.generations[-1].best_fitness if self.generations else 0.0

    @property
    def convergence_rate(self) -> float:
        """Average fitness improvement per generation.

        Computed as (final_best - initial_best) / num_generations.
        """
        if len(self.generations) < 2:
            return 0.0
        initial = self.generations[0].best_fitness
        final = self.generations[-1].best_fitness
        return (final - initial) / (len(self.generations) - 1)

    def summary(self) -> dict:
        """Return a summary dictionary of the run."""
        return {
            "run_name": self.run_name,
            "num_generations": len(self.generations),
            "final_best_fitness": self.final_best_fitness,
            "convergence_rate": self.convergence_rate,
            "total_wall_clock_ms": self.total_wall_clock_ms,
        }

    def to_json(self, path: str | Path) -> None:
        """Persist metrics to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "run_name": self.run_name,
            "generations": [
                {
                    "generation": g.generation,
                    "best_fitness": g.best_fitness,
                    "avg_fitness": g.avg_fitness,
                    "worst_fitness": g.worst_fitness,
                    "wall_clock_ms": g.wall_clock_ms,
                }
                for g in self.generations
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> MetricsStore:
        """Load metrics from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        store = cls(run_name=data["run_name"])
        for g in data["generations"]:
            store.generations.append(
                GenerationMetrics(
                    generation=g["generation"],
                    best_fitness=g["best_fitness"],
                    avg_fitness=g["avg_fitness"],
                    worst_fitness=g["worst_fitness"],
                    wall_clock_ms=g["wall_clock_ms"],
                )
            )
        return store
