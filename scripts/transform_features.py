#!/usr/bin/env python3
"""Apply feature transformations on the merged dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

DROP_COLUMNS: List[str] = [
    "temp_high",
    "temp_low",
    "global_id",
    "geohash",
    "work_geohash",
    "sample_index",
]

CITY_RARE_THRESHOLD = 50
DTYPE_RARE_THRESHOLD = 50


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    return df


def add_timeslot(df: pd.DataFrame) -> None:
    hour_series = None
    if "hour" in df.columns:
        hour_series = pd.to_numeric(df["hour"], errors="coerce")
    elif "timestamp" in df.columns:
        hour_series = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce").dt.hour

    if hour_series is None:
        return

    timeslot = pd.Series(0, index=df.index, dtype="Int64")
    timeslot = timeslot.mask(hour_series.between(5, 9, inclusive="both"), 1)
    timeslot = timeslot.mask(hour_series.between(10, 10, inclusive="both"), 7)
    timeslot = timeslot.mask(hour_series.between(11, 13, inclusive="both"), 2)
    timeslot = timeslot.mask(hour_series.between(14, 15, inclusive="both"), 3)
    timeslot = timeslot.mask(hour_series.between(16, 19, inclusive="both"), 4)
    timeslot = timeslot.mask(hour_series.between(20, 23, inclusive="both"), 5)

    df["timeslot"] = timeslot.fillna(0).astype("Int64")


def add_is_same_city(df: pd.DataFrame) -> None:
    if {"loc_cityid", "cityid"}.issubset(df.columns):
        df["is_same_city"] = (
            (df["loc_cityid"].notna())
            & (df["cityid"].notna())
            & (df["loc_cityid"] == df["cityid"])
        ).astype(int)


def add_is_weekend(df: pd.DataFrame) -> None:
    if "weekday" in df.columns:
        df["is_weekend"] = df["weekday"].isin([6, 7]).astype(int)


def add_prebin_cross_features(df: pd.DataFrame) -> None:
    if {"item_ave_price", "price"}.issubset(df.columns):
        item_price = pd.to_numeric(df["item_ave_price"], errors="coerce").fillna(0)
        price = pd.to_numeric(df["price"], errors="coerce").fillna(0)
        df["discount_rate"] = (item_price - price) / (item_price + 1)

    if {"distance", "user_home_dis"}.issubset(df.columns):
        distance = pd.to_numeric(df["distance"], errors="coerce").fillna(0)
        user_home = pd.to_numeric(df["user_home_dis"], errors="coerce").fillna(0)
        df["distance_home_ratio"] = distance / (user_home + 1)


def add_frequency_features(df: pd.DataFrame) -> None:
    def _apply(column: str, threshold: int, freq_col: str) -> None:
        if column not in df.columns:
            return
        counts = df[column].value_counts(dropna=False)
        freq = df[column].map(counts).fillna(0)
        df[freq_col] = np.log1p(freq)
        rare_mask = freq < threshold
        df[column] = df[column].where(~rare_mask, "__RARE__")

    _apply("cityid", CITY_RARE_THRESHOLD, "cityid_freq")
    _apply("dtype", DTYPE_RARE_THRESHOLD, "dtype_freq")


def transform_dataset(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = load_dataset(input_path)

    add_timeslot(df)
    add_is_same_city(df)
    add_is_weekend(df)
    add_prebin_cross_features(df)
    add_frequency_features(df)

    if "timestamp" in df.columns:
        df.drop(columns=["timestamp"], inplace=True)

    df.drop(columns=[col for col in DROP_COLUMNS if col in df.columns], inplace=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Transform merged dataset features.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/recsys_task_data/train_merged_imputed-20221014.csv"),
        help="Path to the input merged CSV (after imputation).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/recsys_task_data/train_merged_transformed-20221014.csv"),
        help="Path to save the transformed CSV.",
    )

    args = parser.parse_args()
    df = transform_dataset(args.input, args.output)

    print(f"Transformed dataset saved to {args.output}")
    print("Columns after processing:")
    print(", ".join(df.columns))


if __name__ == "__main__":
    main()
