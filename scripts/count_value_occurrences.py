#!/usr/bin/env python3
"""Count distinct values for selected columns."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

DEFAULT_COLUMNS: List[str] = [
    "cityid",
    "loc_cityid",
    "online_days",
    "dtype",
    "cate_1",
    "cate_2",
    "cate_3",
    "age",
    "level",
    "mobile_type",
    "mobile_os",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report unique counts and frequency tables for selected columns."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/recsys_task_data/train_merged_transformed-20221014.csv"),
        help="CSV file to analyze.",
    )
    parser.add_argument(
        "--columns",
        type=str,
        nargs="*",
        default=DEFAULT_COLUMNS,
        help="Columns to analyze (defaults to predefined list).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Display the top-K most frequent values for each column.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file to dump full frequency tables.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    return df


def main() -> None:
    args = parse_args()
    df = load_dataset(args.input)

    results: Dict[str, Dict[str, int]] = {}

    for column in args.columns:
        if column not in df.columns:
            print(f"[!] Column '{column}' not found, skipping.")
            continue

        counts = df[column].astype("string").value_counts(dropna=False)
        unique = int(counts.shape[0])
        print("-" * 80)
        print(f"Column: {column}")
        print(f"  Unique values: {unique:,}")
        print("  Top values:")
        for value, freq in counts.head(args.top_k).items():
            label = "<NA>" if value == "<NA>" else value
            print(f"    {label}: {freq:,}")
        results[column] = counts.to_dict()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved full value counts to {args.output}")


if __name__ == "__main__":
    main()
