"""Individual representation for the genetic algorithm."""

from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass, field
from typing import Any

# Proto imports are deferred to avoid hard dependency during unit testing.
# The generated module lives at generated/python/epoch_pb2.py.
_OPTIMIZER_MAP = {"sgd": 1, "adam": 2, "rmsprop": 3, "adamw": 4}
_OPTIMIZER_REV = {v: k for k, v in _OPTIMIZER_MAP.items()}

_ACTIVATION_MAP = {"relu": 1, "elu": 2, "selu": 3, "tanh": 4, "swish": 5}
_ACTIVATION_REV = {v: k for k, v in _ACTIVATION_MAP.items()}


@dataclass
class Individual:
    """A single candidate solution in the GA population.

    Attributes:
        genome: Mapping of parameter names to their values.
        fitness: Fitness score (validation accuracy). 0.0 until evaluated.
        id: Unique identifier for tracking.
    """

    genome: dict[str, Any]
    fitness: float = 0.0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def clone(self) -> Individual:
        """Create a deep copy with a new id."""
        return Individual(genome=dict(self.genome), fitness=self.fitness)

    def to_protobuf(self) -> Any:
        """Convert to a HyperparamConfig protobuf message.

        Returns:
            An epoch_pb2.HyperparamConfig instance.
        """
        sys.path.insert(0, "generated/python")
        import epoch_pb2

        config = epoch_pb2.HyperparamConfig()
        config.learning_rate = float(self.genome.get("learning_rate", 0.001))
        config.batch_size = int(self.genome.get("batch_size", 32))
        config.optimizer = _OPTIMIZER_MAP.get(self.genome.get("optimizer", "adam"), 2)
        config.conv_filters.extend([
            int(self.genome.get("conv_filters_1", 32)),
            int(self.genome.get("conv_filters_2", 64)),
            int(self.genome.get("conv_filters_3", 128)),
        ])
        config.kernel_size = int(self.genome.get("kernel_size", 3))
        config.dense_units.extend([int(self.genome.get("dense_units_1", 128))])
        config.dropout_rate = float(self.genome.get("dropout_rate", 0.25))
        config.activation = _ACTIVATION_MAP.get(self.genome.get("activation", "relu"), 1)
        config.epochs = int(self.genome.get("epochs", 10))
        config.dataset = self.genome.get("dataset", "mnist")
        return config

    @classmethod
    def from_protobuf(cls, config: Any, fitness: float = 0.0) -> Individual:
        """Create an Individual from a HyperparamConfig protobuf message.

        Args:
            config: An epoch_pb2.HyperparamConfig instance.
            fitness: Fitness to assign.

        Returns:
            A new Individual.
        """
        conv_filters = list(config.conv_filters) if config.conv_filters else [32, 64, 128]
        genome = {
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "optimizer": _OPTIMIZER_REV.get(config.optimizer, "adam"),
            "conv_filters_1": conv_filters[0] if len(conv_filters) > 0 else 32,
            "conv_filters_2": conv_filters[1] if len(conv_filters) > 1 else 64,
            "conv_filters_3": conv_filters[2] if len(conv_filters) > 2 else 128,
            "kernel_size": config.kernel_size or 3,
            "dense_units_1": list(config.dense_units)[0] if config.dense_units else 128,
            "dropout_rate": config.dropout_rate,
            "activation": _ACTIVATION_REV.get(config.activation, "relu"),
            "epochs": config.epochs or 10,
            "dataset": config.dataset or "mnist",
        }
        return cls(genome=genome, fitness=fitness)
