"""Search space definition for hyperparameter optimization."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence


class ParamType(Enum):
    """Type of a hyperparameter."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"


@dataclass(frozen=True)
class ParamSpec:
    """Specification of a single hyperparameter.

    Attributes:
        name: Unique identifier for this parameter.
        param_type: Whether the param is continuous, discrete, or categorical.
        low: Lower bound (continuous/discrete).
        high: Upper bound (continuous/discrete).
        choices: Valid values (categorical).
        log_scale: If True, sample in log space (continuous only).
    """

    name: str
    param_type: ParamType
    low: float | None = None
    high: float | None = None
    choices: tuple[Any, ...] | None = None
    log_scale: bool = False

    def __post_init__(self) -> None:
        if self.param_type in (ParamType.CONTINUOUS, ParamType.DISCRETE):
            if self.low is None or self.high is None:
                raise ValueError(f"{self.name}: continuous/discrete params need low and high")
        if self.param_type == ParamType.CATEGORICAL:
            if not self.choices:
                raise ValueError(f"{self.name}: categorical params need choices")

    def sample(self, rng: random.Random) -> Any:
        """Draw a random value from this parameter's domain."""
        if self.param_type == ParamType.CONTINUOUS:
            if self.log_scale:
                import math

                return math.exp(rng.uniform(math.log(self.low), math.log(self.high)))
            return rng.uniform(self.low, self.high)
        elif self.param_type == ParamType.DISCRETE:
            return rng.randint(int(self.low), int(self.high))
        else:
            return rng.choice(self.choices)

    def clip(self, value: Any) -> Any:
        """Clamp a value to this parameter's valid domain."""
        if self.param_type == ParamType.CONTINUOUS:
            return max(self.low, min(self.high, float(value)))
        elif self.param_type == ParamType.DISCRETE:
            return max(int(self.low), min(int(self.high), int(round(value))))
        else:
            return value if value in self.choices else self.choices[0]


@dataclass
class SearchSpace:
    """Collection of hyperparameter specifications defining the search domain.

    Attributes:
        params: Ordered list of parameter specs.
    """

    params: list[ParamSpec] = field(default_factory=list)

    def add(self, spec: ParamSpec) -> None:
        """Add a parameter to the search space."""
        self.params.append(spec)

    def names(self) -> list[str]:
        """Return all parameter names."""
        return [p.name for p in self.params]

    def get(self, name: str) -> ParamSpec:
        """Look up a parameter spec by name."""
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(f"Unknown parameter: {name}")

    def random_sample(self, rng: random.Random | None = None) -> dict[str, Any]:
        """Sample a random point from the full search space."""
        rng = rng or random.Random()
        return {p.name: p.sample(rng) for p in self.params}

    @staticmethod
    def default_cnn_space() -> SearchSpace:
        """Return the default search space for CNN hyperparameter optimization."""
        space = SearchSpace()
        space.add(ParamSpec("learning_rate", ParamType.CONTINUOUS, 1e-5, 1e-1, log_scale=True))
        space.add(ParamSpec("batch_size", ParamType.CATEGORICAL, choices=(16, 32, 64, 128, 256)))
        space.add(ParamSpec("optimizer", ParamType.CATEGORICAL, choices=("sgd", "adam", "rmsprop", "adamw")))
        space.add(ParamSpec("conv_filters_1", ParamType.DISCRETE, 16, 128))
        space.add(ParamSpec("conv_filters_2", ParamType.DISCRETE, 32, 256))
        space.add(ParamSpec("conv_filters_3", ParamType.DISCRETE, 64, 512))
        space.add(ParamSpec("kernel_size", ParamType.CATEGORICAL, choices=(3, 5)))
        space.add(ParamSpec("dense_units_1", ParamType.DISCRETE, 64, 512))
        space.add(ParamSpec("dropout_rate", ParamType.CONTINUOUS, 0.0, 0.5))
        space.add(ParamSpec("activation", ParamType.CATEGORICAL, choices=("relu", "elu", "selu", "swish")))
        space.add(ParamSpec("epochs", ParamType.DISCRETE, 5, 30))
        return space

    @staticmethod
    def stress_test_cnn_space() -> SearchSpace:
        """Return a constrained search space for scheduler stress testing.

        Uses small model architectures and minimal epochs to achieve
        100-200ms training times while preserving meaningful accuracy
        differentiation between hyperparameter configurations.
        """
        space = SearchSpace()
        space.add(ParamSpec("learning_rate", ParamType.CONTINUOUS, 1e-4, 1e-1, log_scale=True))
        space.add(ParamSpec("batch_size", ParamType.CATEGORICAL, choices=(64, 128, 256)))
        space.add(
            ParamSpec(
                "optimizer", ParamType.CATEGORICAL, choices=("sgd", "adam", "rmsprop", "adamw")
            )
        )
        space.add(ParamSpec("conv_filters_1", ParamType.DISCRETE, 8, 32))
        space.add(ParamSpec("conv_filters_2", ParamType.DISCRETE, 16, 64))
        space.add(ParamSpec("conv_filters_3", ParamType.DISCRETE, 16, 64))
        space.add(ParamSpec("kernel_size", ParamType.CATEGORICAL, choices=(3,)))
        space.add(ParamSpec("dense_units_1", ParamType.DISCRETE, 32, 128))
        space.add(ParamSpec("dropout_rate", ParamType.CONTINUOUS, 0.0, 0.5))
        space.add(
            ParamSpec(
                "activation", ParamType.CATEGORICAL, choices=("relu", "elu", "selu", "swish")
            )
        )
        space.add(ParamSpec("epochs", ParamType.CATEGORICAL, choices=(1, 2)))
        return space
