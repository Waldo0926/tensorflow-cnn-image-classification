#!/usr/bin/env python3
"""Train one CNN experiment and export reproducible artefacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ml_project.data import load_dataset
from ml_project.models import PRESETS, build_model
from ml_project.training import set_global_seed, train_and_export


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or a positive integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train CNN presets on CIFAR-10 or MNIST using TensorFlow/Keras."
    )
    parser.add_argument("--dataset", choices=tuple(PRESETS.keys()), default="cifar10")
    parser.add_argument("--preset", default="baseline", help="Model preset for the selected dataset.")
    parser.add_argument("--epochs", type=positive_int, default=20)
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--learning-rate", type=positive_float, default=1e-3)
    parser.add_argument("--validation-fraction", type=fraction, default=0.2)
    parser.add_argument(
        "--train-limit",
        type=positive_int,
        default=None,
        help="Optional training-subset size for quick CPU tests. Validation/test sets stay intact.",
    )
    parser.add_argument(
        "--patience",
        type=non_negative_int,
        default=0,
        help="Early-stopping patience; 0 disables early stopping.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="Export metrics/figures without writing model.keras.",
    )
    parser.add_argument("--list-presets", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.list_presets:
        for dataset, presets in PRESETS.items():
            print(f"{dataset}:")
            for name, preset in presets.items():
                print(f"  {name:<20} {preset.description}")
        return

    if args.preset not in PRESETS[args.dataset]:
        available = ", ".join(PRESETS[args.dataset].keys())
        raise SystemExit(f"Unknown preset {args.preset!r}. Available for {args.dataset}: {available}")

    set_global_seed(args.seed)
    data = load_dataset(
        args.dataset,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        train_limit=args.train_limit,
    )
    model = build_model(
        dataset=args.dataset,
        preset_name=args.preset,
        input_shape=data.input_shape,
        num_classes=len(data.class_names),
        learning_rate=args.learning_rate,
    )
    model.summary()

    print(
        "\nDataset split: "
        f"train={len(data.x_train):,}, validation={len(data.x_val):,}, test={len(data.x_test):,}"
    )

    output_dir = args.output_dir or PROJECT_ROOT / "results" / args.dataset / args.preset
    metadata = {
        "dataset": args.dataset,
        "preset": args.preset,
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "validation_fraction": args.validation_fraction,
        "train_limit": args.train_limit,
        "patience": args.patience,
        "seed": args.seed,
        "model_saved": not args.no_save_model,
    }
    metrics = train_and_export(
        model=model,
        data=data,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        metadata=metadata,
        save_model=not args.no_save_model,
    )

    print(f"\nResults saved to: {output_dir}")
    print(f"Test loss: {float(metrics['test_loss']):.4f}")
    print(f"Test accuracy: {float(metrics['test_accuracy']):.2%}")
    print(f"Macro F1: {float(metrics['macro_f1']):.4f}")
    print(f"Parameters: {int(metrics['parameter_count']):,}")
    print(f"Training time: {float(metrics['training_seconds']):.1f}s")


if __name__ == "__main__":
    main()
