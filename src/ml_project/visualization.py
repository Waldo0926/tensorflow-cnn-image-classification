"""Plotting helpers for experiment artefacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .evaluation import confusion_matrix_counts


def save_training_curves(history: dict[str, list[float]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(history["loss"], label="train")
    axes[0].plot(history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["accuracy"], label="train")
    axes[1].plot(history["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple[str, ...],
    output_path: Path,
) -> None:
    matrix = confusion_matrix_counts(y_true, y_pred, len(class_names))

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _select_examples(
    indices: np.ndarray,
    requested: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if requested <= 0 or len(indices) == 0:
        return np.asarray([], dtype=int)
    count = min(requested, len(indices))
    return np.sort(rng.choice(indices, size=count, replace=False))


def save_prediction_examples(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    class_names: tuple[str, ...],
    output_path: Path,
    correct_count: int = 8,
    incorrect_count: int = 8,
    seed: int = 42,
) -> None:
    """Save an explicit mix of correct and incorrect predictions with confidence."""
    correct_indices = np.flatnonzero(y_true == y_pred)
    incorrect_indices = np.flatnonzero(y_true != y_pred)
    rng = np.random.default_rng(seed)

    selected_correct = _select_examples(correct_indices, correct_count, rng)
    selected_incorrect = _select_examples(incorrect_indices, incorrect_count, rng)
    groups = (
        ("Correct predictions", selected_correct),
        ("Incorrect predictions", selected_incorrect),
    )

    columns = 4
    rows_per_group = max(1, int(np.ceil(max(correct_count, incorrect_count) / columns)))
    total_rows = rows_per_group * len(groups)
    fig, axes = plt.subplots(total_rows, columns, figsize=(11, 2.8 * total_rows))
    axes = np.asarray(axes).reshape(total_rows, columns)

    for group_index, (group_title, selected) in enumerate(groups):
        row_start = group_index * rows_per_group
        axes[row_start, 0].text(
            -0.05,
            1.23,
            group_title,
            transform=axes[row_start, 0].transAxes,
            fontsize=12,
            fontweight="bold",
        )

        slots = rows_per_group * columns
        for local_index in range(slots):
            row = row_start + local_index // columns
            column = local_index % columns
            ax = axes[row, column]

            if local_index >= len(selected):
                ax.axis("off")
                continue

            image_index = int(selected[local_index])
            image = images[image_index]
            if image.shape[-1] == 1:
                ax.imshow(image.squeeze(-1), cmap="gray")
            else:
                ax.imshow(image)

            true_name = class_names[int(y_true[image_index])]
            predicted_index = int(y_pred[image_index])
            predicted_name = class_names[predicted_index]
            confidence = float(probabilities[image_index, predicted_index])
            ax.set_title(
                f"true: {true_name}\npred: {predicted_name} ({confidence:.1%})",
                fontsize=9,
            )
            ax.axis("off")

    if len(selected_incorrect) == 0:
        axes[rows_per_group, 0].text(
            0.0,
            0.5,
            "No incorrect predictions in this evaluation set.",
            transform=axes[rows_per_group, 0].transAxes,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
