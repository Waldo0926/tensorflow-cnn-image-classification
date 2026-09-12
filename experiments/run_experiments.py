#!/usr/bin/env python3
"""Run the documented CNN preset matrix and build a comparison table."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ml_project.reporting import summarize_results

TRAIN_SCRIPT = ROOT / "src" / "train.py"
RESULTS_ROOT = ROOT / "results"

DEFAULT_MATRIX = {
    "cifar10": ["baseline", "small_filters", "wide_filters", "two_layer", "four_layer"],
    "mnist": ["baseline", "four_conv_two_pool", "three_conv_two_pool"],
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the documented CNN experiment presets and summarize their metrics."
    )
    parser.add_argument("--dataset", choices=("cifar10", "mnist", "all"), default="all")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Also save model.keras for every batch experiment (off by default to save disk space).",
    )
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("--epochs and --batch-size must be positive integers.")
    if args.learning_rate <= 0:
        raise SystemExit("--learning-rate must be positive.")
    if not 0.0 < args.validation_fraction < 1.0:
        raise SystemExit("--validation-fraction must be between 0 and 1.")
    if args.train_limit is not None and args.train_limit <= 0:
        raise SystemExit("--train-limit must be positive.")
    if args.patience < 0:
        raise SystemExit("--patience cannot be negative.")

    datasets = tuple(DEFAULT_MATRIX) if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        for preset in DEFAULT_MATRIX[dataset]:
            command = [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--dataset",
                dataset,
                "--preset",
                preset,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--learning-rate",
                str(args.learning_rate),
                "--validation-fraction",
                str(args.validation_fraction),
                "--patience",
                str(args.patience),
                "--seed",
                str(args.seed),
            ]
            if args.train_limit is not None:
                command.extend(["--train-limit", str(args.train_limit)])
            if not args.save_models:
                command.append("--no-save-model")

            print("\n$", " ".join(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)

    ranked = summarize_results(RESULTS_ROOT)
    print("\nExperiment comparison written to:")
    print(f"  {RESULTS_ROOT / 'experiment_summary.csv'}")
    print(f"  {RESULTS_ROOT / 'experiment_summary.md'}")

    for dataset in datasets:
        matching = [row for row in ranked if row.get("dataset") == dataset]
        if matching:
            best = matching[0]
            print(
                f"Best {dataset} run: {best['preset']} "
                f"({float(best['test_accuracy']):.2%} test accuracy)"
            )


if __name__ == "__main__":
    main()
