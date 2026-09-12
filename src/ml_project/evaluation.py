"""Evaluation helpers that avoid heavyweight third-party metric dependencies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassMetrics:
    class_index: int
    class_name: str
    precision: float
    recall: float
    f1: float
    support: int


def confusion_matrix_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_count: int,
) -> np.ndarray:
    """Return an integer confusion matrix with rows=true and columns=predicted."""
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(matrix, (y_true.astype(int), y_pred.astype(int)), 1)
    return matrix


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple[str, ...],
) -> list[ClassMetrics]:
    """Compute precision, recall and F1 for each class from a confusion matrix."""
    matrix = confusion_matrix_counts(y_true, y_pred, len(class_names))
    rows: list[ClassMetrics] = []

    for index, class_name in enumerate(class_names):
        true_positive = int(matrix[index, index])
        predicted_positive = int(matrix[:, index].sum())
        actual_positive = int(matrix[index, :].sum())

        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        rows.append(
            ClassMetrics(
                class_index=index,
                class_name=class_name,
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
                support=actual_positive,
            )
        )

    return rows
