"""Tests for MetricsStore."""

import json
import tempfile
from pathlib import Path

from metrics.collector import MetricsStore


class TestMetricsStore:
    def test_record_and_retrieve(self):
        store = MetricsStore(run_name="test")
        store.record_generation(0, best_fitness=0.8, avg_fitness=0.5, worst_fitness=0.2, wall_clock_ms=1000)
        store.record_generation(1, best_fitness=0.9, avg_fitness=0.7, worst_fitness=0.4, wall_clock_ms=900)

        assert len(store.generations) == 2
        assert store.generations[0].best_fitness == 0.8
        assert store.generations[1].best_fitness == 0.9

    def test_final_best_fitness(self):
        store = MetricsStore()
        store.record_generation(0, 0.7, 0.5, 0.3, 1000)
        store.record_generation(1, 0.9, 0.7, 0.5, 900)
        assert store.final_best_fitness == 0.9

    def test_final_best_fitness_empty(self):
        store = MetricsStore()
        assert store.final_best_fitness == 0.0

    def test_total_wall_clock(self):
        store = MetricsStore()
        store.record_generation(0, 0.7, 0.5, 0.3, 1000)
        store.record_generation(1, 0.9, 0.7, 0.5, 900)
        assert store.total_wall_clock_ms == 1900

    def test_convergence_rate(self):
        store = MetricsStore()
        store.record_generation(0, 0.5, 0.3, 0.1, 1000)
        store.record_generation(1, 0.7, 0.5, 0.3, 1000)
        store.record_generation(2, 0.9, 0.7, 0.5, 1000)
        # (0.9 - 0.5) / 2 = 0.2
        assert abs(store.convergence_rate - 0.2) < 1e-9

    def test_summary(self):
        store = MetricsStore(run_name="test_run")
        store.record_generation(0, 0.8, 0.5, 0.2, 1000)
        summary = store.summary()
        assert summary["run_name"] == "test_run"
        assert summary["num_generations"] == 1
        assert summary["final_best_fitness"] == 0.8

    def test_json_roundtrip(self):
        store = MetricsStore(run_name="roundtrip_test")
        store.record_generation(0, 0.8, 0.5, 0.2, 1000)
        store.record_generation(1, 0.9, 0.7, 0.4, 800)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.json"
            store.to_json(path)

            loaded = MetricsStore.from_json(path)
            assert loaded.run_name == "roundtrip_test"
            assert len(loaded.generations) == 2
            assert loaded.generations[0].best_fitness == 0.8
            assert loaded.generations[1].wall_clock_ms == 800
