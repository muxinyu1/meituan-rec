#!/usr/bin/env python3
"""Bucketize continuous features in the transformed dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

QUANTILE_COLUMNS_20: List[str] = [
    "distance",
    "user_home_dis",
    "user_work_dis",
    "item_ave_price",
    "price",
    "user_displayed_item_num",
    "online_days",
    "discount_rate",
    "distance_home_ratio",
    "cityid_freq",
    "dtype_freq",
]

TEMP_COLUMN = "temp"


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    return df


def quantile_bin(series: pd.Series, q: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    try:
        binned = pd.qcut(numeric, q=q, labels=False, duplicates="drop")
    except ValueError:
        # Fallback if series has insufficient unique values
        binned = pd.Series(pd.NA, index=series.index, dtype="Int64")
    return binned.astype("Int64")


def add_bins(df: pd.DataFrame) -> Dict[str, int]:
    bins_added: Dict[str, int] = {}

    for column in QUANTILE_COLUMNS_20:
        if column not in df.columns:
            continue
        bins = quantile_bin(df[column], 20)
        df[f"{column}_bin"] = bins
        bins_added[column] = 20

    if TEMP_COLUMN in df.columns:
        bins = quantile_bin(df[TEMP_COLUMN], 5)
        df[f"{TEMP_COLUMN}_bin"] = bins
        bins_added[TEMP_COLUMN] = 5

    return bins_added


def add_dtype_distance(df: pd.DataFrame) -> None:
    if {"dtype", "distance_bin"}.issubset(df.columns):
        dtype_str = df["dtype"].astype(str).replace("nan", "<NA>")
        distance_str = df["distance_bin"].astype("Int64").astype(str).replace("<NA>", "NA")
        df["dtype_distance"] = dtype_str + "_" + distance_str


def drop_original_columns(df: pd.DataFrame) -> None:
    columns_to_drop = QUANTILE_COLUMNS_20 + [TEMP_COLUMN]
    existing = [col for col in columns_to_drop if col in df.columns]
    if existing:
        df.drop(columns=existing, inplace=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bucketize continuous features.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/recsys_task_data/train_merged_transformed-20221014.csv"),
        help="Path to the transformed dataset CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/recsys_task_data/train_merged_binned-20221014.csv"),
        help="Path to save the dataset with bucketed features.",
    )

    args = parser.parse_args()

    df = load_dataset(args.input)
    bins_added = add_bins(df)
    add_dtype_distance(df)
    drop_original_columns(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Saved bucketed dataset to {args.output}")
    for column, bins in bins_added.items():
        print(f" - {column}: {bins} buckets (column: {column}_bin)")


if __name__ == "__main__":
    main()
