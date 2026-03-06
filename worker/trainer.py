"""Trains Keras models and reports results."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf

from worker.model_builder import ModelBuilder

logger = logging.getLogger(__name__)

# Dataset cache: loaded once, reused across training runs
_dataset_cache: dict[str, tuple[Any, Any, Any, Any]] = {}

_INPUT_SHAPES = {
    "mnist": (28, 28, 1),
    "cifar10": (32, 32, 3),
}

_NUM_CLASSES = {
    "mnist": 10,
    "cifar10": 10,
}
_MAX_SEED = 2_147_483_647


@dataclass
class TrainResult:
    """Result of a single training run.

    Attributes:
        validation_accuracy: Accuracy on the test set.
        training_loss: Final training loss.
        training_time_ms: Wall-clock training time in milliseconds.
        status: One of "completed", "failed", "timeout".
        error_message: Error details if status != "completed".
    """

    validation_accuracy: float = 0.0
    training_loss: float = 0.0
    training_time_ms: int = 0
    status: str = "completed"
    error_message: str = ""


def _load_dataset(name: str) -> tuple[Any, Any, Any, Any]:
    """Load and preprocess a dataset, caching for reuse."""
    if name in _dataset_cache:
        return _dataset_cache[name]

    if name == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
    elif name == "cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        y_train = y_train.flatten()
        y_test = y_test.flatten()
    else:
        raise ValueError(f"Unknown dataset: {name}")

    _dataset_cache[name] = (x_train, y_train, x_test, y_test)
    logger.info("Loaded dataset '%s': train=%d test=%d", name, len(x_train), len(x_test))
    return x_train, y_train, x_test, y_test


def _normalize_for_hash(value: Any) -> Any:
    """Normalize nested values to a deterministic JSON-serializable structure."""
    if isinstance(value, dict):
        return {str(k): _normalize_for_hash(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if np.isnan(value):
            return "NaN"
        if np.isposinf(value):
            return "Infinity"
        if np.isneginf(value):
            return "-Infinity"
    return value


def _stable_genome_seed(genome: dict[str, Any], seed_offset: int = 0) -> int:
    """Derive a stable positive seed from canonical genome content."""
    normalized = _normalize_for_hash(genome)
    canonical = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    base_seed = int.from_bytes(digest[:8], byteorder="big", signed=False) % _MAX_SEED
    seed = (base_seed + int(seed_offset)) % _MAX_SEED
    return seed if seed > 0 else 1


class Trainer:
    """Trains a Keras model according to a hyperparameter genome.

    Attributes:
        timeout_seconds: Maximum wall-clock time for a single training run.
    """

    def __init__(
        self,
        timeout_seconds: float = 300.0,
        train_subset_size: int = 0,
        val_subset_size: int = 0,
        run_eagerly: bool = False,
        gc_every_n_jobs: int = 1,
        deterministic_eval: bool = True,
        deterministic_seed_offset: int = 0,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.train_subset_size = train_subset_size
        self.val_subset_size = val_subset_size
        self.run_eagerly = run_eagerly
        self.gc_every_n_jobs = max(1, int(gc_every_n_jobs))
        self.deterministic_eval = bool(deterministic_eval)
        self.deterministic_seed_offset = int(deterministic_seed_offset)
        self._jobs_completed = 0

    def _should_collect_gc(self) -> bool:
        """Increment per-job counter and decide whether to trigger GC."""
        self._jobs_completed += 1
        return (self._jobs_completed % self.gc_every_n_jobs) == 0

    def _resolve_job_seed(self, genome: dict[str, Any]) -> int | None:
        if not self.deterministic_eval:
            return None
        return _stable_genome_seed(
            genome,
            seed_offset=self.deterministic_seed_offset,
        )

    def train(self, genome: dict[str, Any]) -> TrainResult:
        """Build and train a model, returning the result.

        Args:
            genome: Hyperparameter dictionary (same format as Individual.genome).

        Returns:
            TrainResult with accuracy, loss, timing, and status.
        """
        dataset_name = genome.get("dataset", "mnist")
        epochs = int(genome.get("epochs", 10))
        batch_size = int(genome.get("batch_size", 32))

        model = None
        build_ms = 0
        fit_ms = 0
        try:
            job_seed = self._resolve_job_seed(genome)
            if job_seed is not None:
                tf.keras.utils.set_random_seed(job_seed)

            x_train, y_train, x_test, y_test = _load_dataset(dataset_name)

            if self.train_subset_size > 0:
                x_train = x_train[: self.train_subset_size]
                y_train = y_train[: self.train_subset_size]
            if self.val_subset_size > 0:
                x_test = x_test[: self.val_subset_size]
                y_test = y_test[: self.val_subset_size]

            logger.debug(
                "Training with %d train / %d val samples, %d epochs, batch_size=%d",
                len(x_train),
                len(x_test),
                epochs,
                batch_size,
            )

            input_shape = _INPUT_SHAPES[dataset_name]
            num_classes = _NUM_CLASSES[dataset_name]

            builder = ModelBuilder(
                num_classes=num_classes,
                input_shape=input_shape,
                run_eagerly=self.run_eagerly,
            )
            build_started = time.monotonic()
            model = builder.build(genome)
            build_ms = int((time.monotonic() - build_started) * 1000)

            # Timeout callback
            class TimeoutCallback(tf.keras.callbacks.Callback):
                def __init__(self, timeout: float):
                    super().__init__()
                    self.timeout = timeout
                    self.start_time = 0.0

                def on_train_begin(self, logs=None):
                    self.start_time = time.monotonic()

                def on_train_batch_end(self, batch, logs=None):
                    if time.monotonic() - self.start_time > self.timeout:
                        self.model.stop_training = True
                        logger.warning("Training timeout after %.1fs", self.timeout)

                def on_epoch_end(self, epoch, logs=None):
                    if time.monotonic() - self.start_time > self.timeout:
                        self.model.stop_training = True
                        logger.warning("Training timeout after %.1fs", self.timeout)

            start = time.monotonic()
            history = model.fit(
                x_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test),
                verbose=0,
                callbacks=[TimeoutCallback(self.timeout_seconds)],
            )
            elapsed_ms = int((time.monotonic() - start) * 1000)
            fit_ms = elapsed_ms

            # Check if we timed out
            if time.monotonic() - start > self.timeout_seconds:
                return TrainResult(
                    validation_accuracy=0.0,
                    training_loss=0.0,
                    training_time_ms=elapsed_ms,
                    status="timeout",
                    error_message=f"Training exceeded {self.timeout_seconds}s limit",
                )

            # Keras 3 (TF 2.16+) may store metric under different keys
            hist = history.history
            if "val_accuracy" in hist:
                val_acc = float(hist["val_accuracy"][-1])
            elif "val_sparse_categorical_accuracy" in hist:
                val_acc = float(hist["val_sparse_categorical_accuracy"][-1])
            else:
                available = list(hist.keys())
                raise KeyError(
                    f"No accuracy metric in history. Available keys: {available}"
                )
            train_loss = float(hist["loss"][-1])

            logger.info(
                "Training complete: val_acc=%.4f loss=%.4f fit_ms=%d build_ms=%d",
                val_acc,
                train_loss,
                fit_ms,
                build_ms,
            )

            return TrainResult(
                validation_accuracy=val_acc,
                training_loss=train_loss,
                training_time_ms=elapsed_ms,
                status="completed",
            )

        except Exception as e:
            logger.error("Training failed: %s", e, exc_info=True)
            return TrainResult(
                status="failed",
                error_message=str(e),
            )

        finally:
            cleanup_started = time.monotonic()
            # Delete model weights without destroying the TF runtime.
            # clear_session() is too expensive (~2s) - it tears down the entire
            # GPU context, forcing full reinit on the next job.
            try:
                del model
            except Exception:
                pass
            import gc

            ran_gc = self._should_collect_gc()
            if ran_gc:
                gc.collect()
            cleanup_ms = int((time.monotonic() - cleanup_started) * 1000)
            logger.debug(
                "Job phases: build_ms=%d fit_ms=%d cleanup_ms=%d gc_ran=%s",
                build_ms,
                fit_ms,
                cleanup_ms,
                ran_gc,
            )
