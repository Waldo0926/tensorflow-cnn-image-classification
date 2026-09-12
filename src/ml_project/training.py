"""Training, evaluation, and result-export helpers."""

from __future__ import annotations

import csv
import io
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import tensorflow as tf

from .data import DatasetBundle
from .evaluation import per_class_metrics
from .visualization import (
    save_confusion_matrix,
    save_prediction_examples,
    save_training_curves,
)


def set_global_seed(seed: int) -> None:
    """Seed NumPy/TensorFlow and request deterministic TensorFlow operations."""
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        # Deterministic kernels are not available on every platform/backend.
        pass


def _write_history_csv(history: dict[str, list[float]], output_path: Path) -> None:
    keys = list(history.keys())
    rows = zip(*(history[key] for key in keys))
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", *keys])
        for epoch, values in enumerate(rows, start=1):
            writer.writerow([epoch, *values])


def _write_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple[str, ...],
    output_path: Path,
) -> None:
    rows = per_class_metrics(y_true, y_pred, class_names)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_index", "class_name", "precision", "recall", "f1", "support"])
        for row in rows:
            writer.writerow(
                [
                    row.class_index,
                    row.class_name,
                    f"{row.precision:.8f}",
                    f"{row.recall:.8f}",
                    f"{row.f1:.8f}",
                    row.support,
                ]
            )


def _model_summary_text(model: tf.keras.Model) -> str:
    buffer = io.StringIO()
    model.summary(print_fn=lambda line: buffer.write(line + "\n"))
    return buffer.getvalue()


def train_and_export(
    model: tf.keras.Model,
    data: DatasetBundle,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    patience: int,
    metadata: dict[str, object],
    save_model: bool = True,
) -> dict[str, float | int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks: list[tf.keras.callbacks.Callback] = []
    if patience > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            )
        )

    start_time = time.perf_counter()
    history_object = model.fit(
        data.x_train,
        data.y_train,
        validation_data=(data.x_val, data.y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    training_seconds = time.perf_counter() - start_time
    history = {
        key: [float(value) for value in values]
        for key, values in history_object.history.items()
    }

    test_loss, test_accuracy = model.evaluate(data.x_test, data.y_test, verbose=0)
    probabilities = model.predict(data.x_test, verbose=0)
    predictions = probabilities.argmax(axis=1)

    class_rows = per_class_metrics(data.y_test, predictions, data.class_names)
    macro_precision = float(np.mean([row.precision for row in class_rows]))
    macro_recall = float(np.mean([row.recall for row in class_rows]))
    macro_f1 = float(np.mean([row.f1 for row in class_rows]))

    metrics: dict[str, float | int] = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "best_val_accuracy": float(max(history.get("val_accuracy", [0.0]))),
        "best_val_loss": float(min(history.get("val_loss", [float("nan")]))),
        "epochs_ran": int(len(history["loss"])),
        "parameter_count": int(model.count_params()),
        "training_seconds": float(training_seconds),
        "train_examples": int(len(data.x_train)),
        "validation_examples": int(len(data.x_val)),
        "test_examples": int(len(data.x_test)),
    }

    if save_model:
        model.save(output_dir / "model.keras")
    (output_dir / "model_summary.txt").write_text(
        _model_summary_text(model), encoding="utf-8"
    )
    _write_history_csv(history, output_dir / "history.csv")
    _write_per_class_metrics(
        data.y_test,
        predictions,
        data.class_names,
        output_dir / "per_class_metrics.csv",
    )
    save_training_curves(history, output_dir / "training_curves.png")
    save_confusion_matrix(
        data.y_test,
        predictions,
        data.class_names,
        output_dir / "confusion_matrix.png",
    )
    save_prediction_examples(
        data.x_test,
        data.y_test,
        predictions,
        probabilities,
        data.class_names,
        output_dir / "prediction_examples.png",
        seed=int(metadata.get("seed", 42)),
    )

    environment = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
    }
    payload = {**metadata, **metrics, "environment": environment}
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    return metrics
