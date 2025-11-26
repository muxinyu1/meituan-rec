#!/usr/bin/env python3
"""Merge train samples with item and user info, then report missing columns."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    return df


def merge_datasets(
    train_path: Path,
    item_path: Path,
    user_path: Path,
    output_path: Path,
) -> List[str]:
    train_df = load_csv(train_path)
    item_df = load_csv(item_path)
    user_df = load_csv(user_path)

    merged = train_df.merge(item_df, on="itemid", how="left", suffixes=("", "_item"))
    merged = merged.merge(user_df, on="userid", how="left", suffixes=("", "_user"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    missing_columns = merged.columns[merged.isna().any()].tolist()
    return missing_columns


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge train, item, and user datasets.")
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/recsys_task_data/train_samples-20221014.csv"),
        help="Path to train samples CSV.",
    )
    parser.add_argument(
        "--items",
        type=Path,
        default=Path("data/recsys_task_data/item_infos-20221014.csv"),
        help="Path to item infos CSV.",
    )
    parser.add_argument(
        "--users",
        type=Path,
        default=Path("data/recsys_task_data/user_infos-20221014.csv"),
        help="Path to user infos CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/recsys_task_data/train_merged-20221014.csv"),
        help="Path to save merged CSV.",
    )

    args = parser.parse_args()

    missing_columns = merge_datasets(args.train, args.items, args.users, args.output)

    if missing_columns:
        print("Columns containing missing values:")
        for col in missing_columns:
            print(f" - {col}")
    else:
        print("No missing values detected in merged dataset.")
    print(f"Merged dataset saved to {args.output}")


if __name__ == "__main__":
    main()
