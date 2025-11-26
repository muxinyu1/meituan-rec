#!/usr/bin/env python3
"""Compute mean and median for selected continuous features."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

DEFAULT_COLUMNS: List[str] = [
    "distance",
    "user_home_dis",
    "user_work_dis",
    "item_ave_price",
    "price",
    "user_displayed_item_num",
    "online_days",
    "temp",
    "discount_rate",
    "distance_home_ratio",
    "sale",
    "price_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute mean and median statistics for continuous features."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/recsys_task_data/train_merged_transformed_train-20221014.csv"),
        help="CSV file containing the features.",
    )
    parser.add_argument(
        "--columns",
        type=str,
        nargs="*",
        default=DEFAULT_COLUMNS,
        help="Columns to summarize (defaults to predefined list).",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    return df


def summarize_columns(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for column in columns:
        if column not in df.columns:
            print(f"[!] Column '{column}' not found, skipping.")
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        mean = float(series.mean()) if series.notna().any() else float("nan")
        median = float(series.median()) if series.notna().any() else float("nan")
        summary[column] = {"mean": mean, "median": median}
    return summary


def main() -> None:
    args = parse_args()
    df = load_dataset(args.input)
    summary = summarize_columns(df, args.columns)

    print("Feature statistics (mean & median):")
    for column, stats in summary.items():
        print(f" - {column}: mean={stats['mean']:.6f}, median={stats['median']:.6f}")


if __name__ == "__main__":
    main()
