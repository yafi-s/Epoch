"""Tests for ModelBuilder."""

import pytest

from worker.model_builder import ModelBuilder


class TestModelBuilder:
    def test_builds_model_with_default_genome(self):
        genome = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "optimizer": "adam",
            "conv_filters_1": 32,
            "conv_filters_2": 64,
            "conv_filters_3": 128,
            "kernel_size": 3,
            "dense_units_1": 128,
            "dropout_rate": 0.25,
            "activation": "relu",
        }
        builder = ModelBuilder(num_classes=10, input_shape=(28, 28, 1))
        model = builder.build(genome)

        assert model is not None
        # Output layer should have 10 units (num_classes)
        assert model.output_shape[-1] == 10

    def test_builds_model_for_cifar10(self):
        genome = {
            "learning_rate": 0.01,
            "batch_size": 64,
            "optimizer": "sgd",
            "conv_filters_1": 16,
            "conv_filters_2": 32,
            "conv_filters_3": 64,
            "kernel_size": 5,
            "dense_units_1": 256,
            "dropout_rate": 0.5,
            "activation": "elu",
        }
        builder = ModelBuilder(num_classes=10, input_shape=(32, 32, 3))
        model = builder.build(genome)

        assert model is not None
        assert model.output_shape[-1] == 10

    def test_different_optimizers(self):
        base_genome = {
            "learning_rate": 0.001,
            "conv_filters_1": 32,
            "conv_filters_2": 64,
            "conv_filters_3": 128,
            "kernel_size": 3,
            "dense_units_1": 128,
            "dropout_rate": 0.25,
            "activation": "relu",
        }
        builder = ModelBuilder()
        for opt in ["sgd", "adam", "rmsprop", "adamw"]:
            genome = {**base_genome, "optimizer": opt}
            model = builder.build(genome)
            assert model is not None

    def test_different_activations(self):
        base_genome = {
            "learning_rate": 0.001,
            "optimizer": "adam",
            "conv_filters_1": 32,
            "conv_filters_2": 64,
            "conv_filters_3": 128,
            "kernel_size": 3,
            "dense_units_1": 128,
            "dropout_rate": 0.25,
        }
        builder = ModelBuilder()
        for act in ["relu", "elu", "selu", "swish"]:
            genome = {**base_genome, "activation": act}
            model = builder.build(genome)
            assert model is not None
