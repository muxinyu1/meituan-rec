#!/usr/bin/env python3
"""Report missing value ratios for every column in a CSV file."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def compute_missing_rates(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    return df.isna().mean().sort_values(ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print per-column missing-value ratios.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/recsys_task_data/train_merged-20221014.csv"),
        help="Path to the merged CSV file.",
    )

    args = parser.parse_args()
    rates = compute_missing_rates(args.input)

    print(f"Missing-value ratios for {args.input}:")
    for column, ratio in rates.items():
        percentage = ratio * 100
        print(f" - {column}: {percentage}%")


if __name__ == "__main__":
    main()
