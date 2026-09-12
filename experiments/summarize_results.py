#!/usr/bin/env python3
"""Rebuild experiment_summary.csv/.md from existing metrics.json files."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ml_project.reporting import summarize_results


def main() -> None:
    results_root = ROOT / "results"
    ranked = summarize_results(results_root)
    print(f"Found {len(ranked)} completed experiment(s).")
    print(f"Wrote {results_root / 'experiment_summary.csv'}")
    print(f"Wrote {results_root / 'experiment_summary.md'}")


if __name__ == "__main__":
    main()
