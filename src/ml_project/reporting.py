"""Collect per-run metrics and export experiment comparison tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


SUMMARY_FIELDS = (
    "rank",
    "dataset",
    "preset",
    "test_accuracy",
    "test_loss",
    "best_val_accuracy",
    "parameter_count",
    "epochs_ran",
    "training_seconds",
    "seed",
    "batch_size",
    "train_limit",
)


def collect_metrics(results_root: Path) -> list[dict[str, Any]]:
    """Load all metrics.json files below a results directory."""
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(results_root.glob("**/metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if "dataset" not in payload or "preset" not in payload:
            continue
        payload["metrics_path"] = str(metrics_path.relative_to(results_root))
        rows.append(payload)
    return rows


def rank_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank experiments by test accuracy within each dataset."""
    ranked: list[dict[str, Any]] = []
    datasets = sorted({str(row["dataset"]) for row in rows})
    for dataset in datasets:
        dataset_rows = [row.copy() for row in rows if str(row["dataset"]) == dataset]
        dataset_rows.sort(
            key=lambda row: (
                -float(row.get("test_accuracy", 0.0)),
                float(row.get("test_loss", float("inf"))),
                str(row.get("preset", "")),
            )
        )
        for rank, row in enumerate(dataset_rows, start=1):
            row["rank"] = rank
            ranked.append(row)
    return ranked


def _format_markdown_value(field: str, value: Any) -> str:
    if value is None:
        return "-"
    if field in {"test_accuracy", "best_val_accuracy"}:
        return f"{float(value):.2%}"
    if field == "test_loss":
        return f"{float(value):.4f}"
    if field == "training_seconds":
        return f"{float(value):.1f}s"
    if field == "parameter_count":
        return f"{int(value):,}"
    return str(value)


def write_summary(
    rows: list[dict[str, Any]],
    csv_path: Path,
    markdown_path: Path,
) -> list[dict[str, Any]]:
    """Write machine-readable CSV and recruiter-friendly Markdown comparisons."""
    ranked = rank_metrics(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in ranked:
            writer.writerow({field: row.get(field) for field in SUMMARY_FIELDS})

    headers = (
        "Rank",
        "Dataset",
        "Preset",
        "Test accuracy",
        "Test loss",
        "Best val accuracy",
        "Parameters",
        "Epochs",
        "Training time",
    )
    markdown_fields = (
        "rank",
        "dataset",
        "preset",
        "test_accuracy",
        "test_loss",
        "best_val_accuracy",
        "parameter_count",
        "epochs_ran",
        "training_seconds",
    )
    lines = [
        "# Experiment Comparison",
        "",
        "Generated automatically from per-run `metrics.json` files.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in ranked:
        values = [_format_markdown_value(field, row.get(field)) for field in markdown_fields]
        lines.append("| " + " | ".join(values) + " |")

    if not ranked:
        lines.append("| - | - | No completed experiments found | - | - | - | - | - | - |")

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ranked


def summarize_results(results_root: Path) -> list[dict[str, Any]]:
    """Collect, rank and write the standard experiment summary files."""
    rows = collect_metrics(results_root)
    return write_summary(
        rows,
        csv_path=results_root / "experiment_summary.csv",
        markdown_path=results_root / "experiment_summary.md",
    )
