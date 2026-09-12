"""Dataset loading and deterministic train/validation splitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class DatasetBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    input_shape: tuple[int, ...]
    class_names: tuple[str, ...]


CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
MNIST_CLASSES = tuple(str(i) for i in range(10))


def _normalise(images: np.ndarray) -> np.ndarray:
    return images.astype("float32") / 255.0


def _flatten_labels(labels: np.ndarray) -> np.ndarray:
    return np.asarray(labels).reshape(-1).astype("int64")


def _split_train_validation(
    images: np.ndarray,
    labels: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a deterministic class-stratified train/validation split."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []

    for class_index in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_index)
        rng.shuffle(class_indices)
        validation_size = max(1, int(round(len(class_indices) * validation_fraction)))
        validation_size = min(validation_size, len(class_indices) - 1)
        validation_parts.append(class_indices[:validation_size])
        train_parts.append(class_indices[validation_size:])

    train_indices = np.concatenate(train_parts)
    validation_indices = np.concatenate(validation_parts)
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)

    return (
        images[train_indices],
        labels[train_indices],
        images[validation_indices],
        labels[validation_indices],
    )


def _limit_training_data(
    images: np.ndarray,
    labels: np.ndarray,
    train_limit: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if train_limit is None:
        return images, labels
    if train_limit <= 0:
        raise ValueError("train_limit must be a positive integer.")
    if train_limit >= len(images):
        return images, labels

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(images), size=train_limit, replace=False)
    return images[indices], labels[indices]


def load_dataset(
    name: str,
    validation_fraction: float = 0.2,
    seed: int = 42,
    train_limit: int | None = None,
) -> DatasetBundle:
    """Load CIFAR-10 or MNIST using Keras' built-in dataset loaders."""
    name = name.lower()
    if name == "cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        class_names = CIFAR10_CLASSES
    elif name == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        class_names = MNIST_CLASSES
    else:
        raise ValueError(f"Unsupported dataset: {name!r}. Choose 'cifar10' or 'mnist'.")

    x_train = _normalise(x_train)
    x_test = _normalise(x_test)
    y_train = _flatten_labels(y_train)
    y_test = _flatten_labels(y_test)

    x_train, y_train, x_val, y_val = _split_train_validation(
        x_train,
        y_train,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    x_train, y_train = _limit_training_data(x_train, y_train, train_limit, seed)

    return DatasetBundle(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        input_shape=tuple(x_train.shape[1:]),
        class_names=class_names,
    )
