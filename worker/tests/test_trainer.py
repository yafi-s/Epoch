"""Tests for Trainer.

Note: These tests require TensorFlow and will download MNIST on first run.
They are tagged as slow and can be skipped with: pytest -m "not slow"
"""

import pytest

from worker.trainer import Trainer, TrainResult


@pytest.mark.slow
class TestTrainer:
    def test_train_mnist_minimal(self):
        """Train for 1 epoch on MNIST to verify the pipeline works end-to-end."""
        genome = {
            "dataset": "mnist",
            "epochs": 1,
            "batch_size": 128,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "conv_filters_1": 16,
            "conv_filters_2": 32,
            "conv_filters_3": 64,
            "kernel_size": 3,
            "dense_units_1": 64,
            "dropout_rate": 0.25,
            "activation": "relu",
        }
        trainer = Trainer(timeout_seconds=120.0)
        result = trainer.train(genome)

        assert result.status == "completed"
        assert result.validation_accuracy > 0.0
        assert result.training_time_ms > 0

    def test_train_returns_failed_on_bad_dataset(self):
        genome = {"dataset": "nonexistent", "epochs": 1}
        trainer = Trainer()
        result = trainer.train(genome)
        assert result.status == "failed"
        assert result.error_message != ""
