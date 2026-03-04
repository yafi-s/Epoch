"""Builds Keras CNN models from hyperparameter configurations."""

from __future__ import annotations

import logging
from typing import Any

import tensorflow as tf

logger = logging.getLogger(__name__)

# Maps proto enum string names to Keras objects
_OPTIMIZER_MAP = {
    "sgd": lambda lr: tf.keras.optimizers.SGD(learning_rate=lr),
    "adam": lambda lr: tf.keras.optimizers.Adam(learning_rate=lr),
    "rmsprop": lambda lr: tf.keras.optimizers.RMSprop(learning_rate=lr),
    "adamw": lambda lr: tf.keras.optimizers.AdamW(learning_rate=lr),
}

_ACTIVATION_MAP = {
    "relu": "relu",
    "elu": "elu",
    "selu": "selu",
    "tanh": "tanh",
    "swish": "swish",
}


class ModelBuilder:
    """Constructs a Keras Sequential CNN from a hyperparameter genome.

    The architecture follows a standard pattern:
    - N Conv2D + MaxPool blocks (filter counts from conv_filters)
    - Flatten
    - Dense layers (from dense_units) with dropout
    - Softmax output head
    """

    def __init__(
        self,
        num_classes: int = 10,
        input_shape: tuple[int, ...] = (28, 28, 1),
        run_eagerly: bool = False,
    ) -> None:
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.run_eagerly = run_eagerly

    def build(self, genome: dict[str, Any]) -> tf.keras.Model:
        """Build a compiled Keras model from a hyperparameter genome.

        Args:
            genome: Dictionary with keys like learning_rate, batch_size,
                    conv_filters_1, conv_filters_2, conv_filters_3,
                    kernel_size, dense_units_1, dropout_rate, activation, optimizer.

        Returns:
            A compiled tf.keras.Model.
        """
        activation = _ACTIVATION_MAP.get(genome.get("activation", "relu"), "relu")
        kernel_size = int(genome.get("kernel_size", 3))
        dropout_rate = float(genome.get("dropout_rate", 0.25))
        learning_rate = float(genome.get("learning_rate", 0.001))
        optimizer_name = genome.get("optimizer", "adam")

        conv_filters = [
            int(genome.get("conv_filters_1", 32)),
            int(genome.get("conv_filters_2", 64)),
            int(genome.get("conv_filters_3", 128)),
        ]
        dense_units = [int(genome.get("dense_units_1", 128))]

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=self.input_shape))

        # Conv blocks
        for i, filters in enumerate(conv_filters):
            model.add(
                tf.keras.layers.Conv2D(
                    filters,
                    kernel_size=(kernel_size, kernel_size),
                    activation=activation,
                    padding="same",
                )
            )
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Flatten())

        # Dense layers
        for units in dense_units:
            model.add(tf.keras.layers.Dense(units, activation=activation))
            model.add(tf.keras.layers.Dropout(dropout_rate))

        # Output
        model.add(tf.keras.layers.Dense(self.num_classes, activation="softmax"))

        # Compile
        optimizer_fn = _OPTIMIZER_MAP.get(optimizer_name, _OPTIMIZER_MAP["adam"])
        model.compile(
            optimizer=optimizer_fn(learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            jit_compile=False,
            run_eagerly=self.run_eagerly,
        )

        logger.info(
            "Built model: conv=%s dense=%s kernel=%d dropout=%.2f opt=%s lr=%.6f",
            conv_filters,
            dense_units,
            kernel_size,
            dropout_rate,
            optimizer_name,
            learning_rate,
        )
        return model
