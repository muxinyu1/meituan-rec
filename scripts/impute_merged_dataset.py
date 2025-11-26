#!/usr/bin/env python3
"""Fill missing values in the merged dataset using predefined strategies."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    return df


def fill_with_median(df: pd.DataFrame, columns: List[str]) -> Dict[str, float]:
    medians: Dict[str, float] = {}
    for column in columns:
        if column not in df.columns:
            continue
        numeric_series = pd.to_numeric(df[column], errors="coerce")
        median = float(numeric_series.median())
        df[column] = numeric_series.fillna(median)
        medians[column] = median
    return medians


def fill_with_mode(df: pd.DataFrame, columns: List[str]) -> Dict[str, object]:
    modes: Dict[str, object] = {}
    for column in columns:
        if column not in df.columns:
            continue
        mode_series = df[column].mode(dropna=True)
        if mode_series.empty:
            continue
        mode_value = mode_series.iloc[0]
        df[column] = df[column].fillna(mode_value)
        modes[column] = mode_value
    return modes


def add_missing_indicator(df: pd.DataFrame, column: str) -> None:
    if column in df.columns:
        indicator = f"{column}_is_missing"
        df[indicator] = df[column].isna().astype(int)


def main() -> None:
    parser = argparse.ArgumentParser(description="Impute missing values with specified rules.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/recsys_task_data/train_merged-20221014.csv"),
        help="Path to the merged CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/recsys_task_data/train_merged_imputed-20221014.csv"),
        help="Path to save the imputed CSV file.",
    )

    args = parser.parse_args()
    df = load_dataset(args.input)

    # Add indicator before filling
    add_missing_indicator(df, "level")

    median_columns = [
        "online_days",
        "price",
        "user_displayed_item_num",
        "level",
        "item_ave_price",
        "distance",
        "temp_high",
        "temp",
        "temp_low",
        "user_home_dis",
        "user_work_dis",
    ]

    mode_columns = [
        "cate_1",
        "cate_2",
        "cate_3",
        "married",
        "has_car",
        "loc_cityid",
        "cityid",
    ]

    median_stats = fill_with_median(df, median_columns)
    mode_stats = fill_with_mode(df, mode_columns)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Saved imputed dataset to {args.output}")
    print("Median fills:")
    for column, value in median_stats.items():
        print(f" - {column}: {value}")
    print("Mode fills:")
    for column, value in mode_stats.items():
        print(f" - {column}: {value}")

    remaining_missing = df.columns[df.isna().any()].tolist()
    if remaining_missing:
        print("Columns still containing missing values:")
        for col in remaining_missing:
            print(f" - {col}")
    else:
        print("All missing values handled as per instructions.")


if __name__ == "__main__":
    main()
