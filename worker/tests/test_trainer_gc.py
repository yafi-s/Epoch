"""Fast tests for trainer cleanup cadence and deterministic seed derivation."""

from worker.trainer import Trainer, _stable_genome_seed


def test_gc_cadence_every_job() -> None:
    trainer = Trainer(gc_every_n_jobs=1)
    assert trainer._should_collect_gc() is True
    assert trainer._should_collect_gc() is True
    assert trainer._should_collect_gc() is True


def test_gc_cadence_every_third_job() -> None:
    trainer = Trainer(gc_every_n_jobs=3)
    assert trainer._should_collect_gc() is False
    assert trainer._should_collect_gc() is False
    assert trainer._should_collect_gc() is True
    assert trainer._should_collect_gc() is False
    assert trainer._should_collect_gc() is False
    assert trainer._should_collect_gc() is True


def test_gc_cadence_clamps_invalid_values() -> None:
    trainer = Trainer(gc_every_n_jobs=0)
    assert trainer._should_collect_gc() is True


def test_stable_genome_seed_is_key_order_invariant() -> None:
    genome_a = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "nested": {"alpha": 1, "beta": [1, 2, 3]},
    }
    genome_b = {
        "nested": {"beta": [1, 2, 3], "alpha": 1},
        "batch_size": 32,
        "learning_rate": 0.001,
    }
    assert _stable_genome_seed(genome_a) == _stable_genome_seed(genome_b)


def test_stable_genome_seed_changes_with_genome_contents() -> None:
    genome_a = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "optimizer": "adam",
    }
    genome_b = {
        "learning_rate": 0.002,
        "batch_size": 32,
        "optimizer": "adam",
    }
    assert _stable_genome_seed(genome_a) != _stable_genome_seed(genome_b)


def test_deterministic_eval_toggle_controls_job_seed_resolution() -> None:
    genome = {"learning_rate": 0.001, "batch_size": 32}
    trainer = Trainer(deterministic_eval=True, deterministic_seed_offset=5)
    seed = trainer._resolve_job_seed(genome)
    assert seed is not None

    nondeterministic = Trainer(deterministic_eval=False)
    assert nondeterministic._resolve_job_seed(genome) is None
