"""Tests for dependency-light experiment reporting helpers."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ml_project.reporting import collect_metrics, rank_metrics, write_summary


class ReportingTests(unittest.TestCase):
    def test_collect_rank_and_write_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "cifar10" / "baseline"
            deeper = root / "cifar10" / "four_layer"
            baseline.mkdir(parents=True)
            deeper.mkdir(parents=True)

            (baseline / "metrics.json").write_text(
                json.dumps(
                    {
                        "dataset": "cifar10",
                        "preset": "baseline",
                        "test_accuracy": 0.60,
                        "test_loss": 1.2,
                        "best_val_accuracy": 0.59,
                        "parameter_count": 1000,
                        "epochs_ran": 10,
                        "training_seconds": 5.0,
                        "seed": 42,
                        "batch_size": 128,
                        "train_limit": None,
                    }
                ),
                encoding="utf-8",
            )
            (deeper / "metrics.json").write_text(
                json.dumps(
                    {
                        "dataset": "cifar10",
                        "preset": "four_layer",
                        "test_accuracy": 0.65,
                        "test_loss": 1.0,
                        "best_val_accuracy": 0.64,
                        "parameter_count": 2000,
                        "epochs_ran": 10,
                        "training_seconds": 7.0,
                        "seed": 42,
                        "batch_size": 128,
                        "train_limit": None,
                    }
                ),
                encoding="utf-8",
            )

            rows = collect_metrics(root)
            self.assertEqual(len(rows), 2)
            ranked = rank_metrics(rows)
            self.assertEqual(ranked[0]["preset"], "four_layer")
            self.assertEqual(ranked[0]["rank"], 1)

            csv_path = root / "experiment_summary.csv"
            markdown_path = root / "experiment_summary.md"
            write_summary(rows, csv_path, markdown_path)
            self.assertTrue(csv_path.exists())
            self.assertIn("65.00%", markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
