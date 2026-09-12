"""CNN model presets based on the original course-design experiments."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class ModelPreset:
    filters: tuple[int, ...]
    kernel_sizes: tuple[int, ...]
    pool_after: tuple[int, ...]
    pool_sizes: tuple[int, ...]
    description: str


PRESETS: dict[str, dict[str, ModelPreset]] = {
    "cifar10": {
        "baseline": ModelPreset(
            filters=(32, 64, 32),
            kernel_sizes=(3, 3, 3),
            pool_after=(0, 1, 2),
            pool_sizes=(2, 2, 2),
            description="Original three-convolution CIFAR-10 baseline.",
        ),
        "small_filters": ModelPreset(
            filters=(16, 32, 16),
            kernel_sizes=(3, 3, 3),
            pool_after=(0, 1, 2),
            pool_sizes=(2, 2, 2),
            description="Three convolution layers with smaller filter counts.",
        ),
        "wide_filters": ModelPreset(
            filters=(32, 64, 64),
            kernel_sizes=(3, 3, 3),
            pool_after=(0, 1, 2),
            pool_sizes=(2, 2, 2),
            description="Three convolution layers with a wider final layer.",
        ),
        "two_layer": ModelPreset(
            filters=(16, 32),
            kernel_sizes=(2, 2),
            pool_after=(0, 1),
            pool_sizes=(2, 2),
            description="Two convolution layers using 2x2 kernels.",
        ),
        "four_layer": ModelPreset(
            filters=(32, 64, 32, 64),
            kernel_sizes=(4, 4, 4, 4),
            pool_after=(0, 1, 2, 3),
            pool_sizes=(2, 2, 2, 2),
            description="Four convolution layers using 4x4 kernels.",
        ),
    },
    "mnist": {
        "baseline": ModelPreset(
            filters=(32, 64, 32),
            kernel_sizes=(3, 3, 3),
            pool_after=(0, 1, 2),
            pool_sizes=(2, 2, 1),
            description="Original three-convolution MNIST baseline.",
        ),
        "four_conv_two_pool": ModelPreset(
            filters=(32, 64, 32, 16),
            kernel_sizes=(3, 3, 3, 3),
            pool_after=(0, 1),
            pool_sizes=(2, 2),
            description="Four convolution layers with only two pooling layers.",
        ),
        "three_conv_two_pool": ModelPreset(
            filters=(32, 64, 32),
            kernel_sizes=(3, 3, 3),
            pool_after=(0, 1),
            pool_sizes=(2, 2),
            description="Three convolution layers with two pooling layers.",
        ),
    },
}


def available_presets(dataset: str) -> tuple[str, ...]:
    return tuple(PRESETS[dataset].keys())


def build_model(
    dataset: str,
    preset_name: str,
    input_shape: tuple[int, ...],
    num_classes: int = 10,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Build and compile a CNN from one of the named experiment presets."""
    try:
        preset = PRESETS[dataset][preset_name]
    except KeyError as exc:
        available = ", ".join(PRESETS.get(dataset, {}).keys())
        raise ValueError(
            f"Unknown preset {preset_name!r} for {dataset!r}. Available: {available}"
        ) from exc

    layers: list[tf.keras.layers.Layer] = [tf.keras.layers.Input(shape=input_shape)]
    pool_size_index = 0

    for index, (filters, kernel_size) in enumerate(zip(preset.filters, preset.kernel_sizes)):
        layers.append(
            tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                padding="same",
                activation="relu",
            )
        )
        if index in preset.pool_after:
            pool_size = preset.pool_sizes[pool_size_index]
            layers.append(tf.keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size)))
            pool_size_index += 1

    layers.extend(
        [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model = tf.keras.Sequential(layers, name=f"{dataset}_{preset_name}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model
