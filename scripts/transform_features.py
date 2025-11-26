#!/usr/bin/env python3
"""Apply feature transformations on the merged dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd
import numpy as np

DROP_COLUMNS: List[str] = [
    "temp_high",
    "temp_low",
    "global_id",
    "geohash",
    "work_geohash",
 #   "sample_index",
]

RARE_TOKEN = "<UNK>"

CITY_RARE_THRESHOLD = 300
LOC_CITY_RARE_THRESHOLD = 250
DTYPE_RARE_THRESHOLD = 100
CATE_1_RARE_THRESHOLD = 100
CATE_2_RARE_THRESHOLD = 100
CATE_3_RARE_THRESHOLD = 100
ITEM_RARE_THRESHOLD = 10
MOBILE_TYPE_RARE_THRESHOLD = 10_000

ITEM_UNKNOWN_TOKEN = RARE_TOKEN


def cap_distance_outliers(df: pd.DataFrame) -> None:
    target_columns = ["user_home_dis", "user_work_dis"]
    for col in target_columns:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        threshold = numeric.quantile(0.95)
        if pd.isna(threshold):
            df[f"{col}_is_far"] = 0
            continue
        non_outliers = numeric[numeric <= threshold]
        replacement = non_outliers.median()
        if pd.isna(replacement):
            replacement = numeric.median()
        is_far = (numeric >= threshold) & numeric.notna()
        df[f"{col}_is_far"] = is_far.astype(int)
        capped = numeric.where(~is_far, replacement)
        df[col] = capped


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


def add_cross_features(df: pd.DataFrame) -> None:
    if {"item_ave_price", "price"}.issubset(df.columns):
        item_price = pd.to_numeric(df["item_ave_price"], errors="coerce").fillna(0)
        price = pd.to_numeric(df["price"], errors="coerce").fillna(0)
        df["discount_rate"] = (item_price - price) / (item_price + 1)

    if {"distance", "user_home_dis"}.issubset(df.columns):
        distance = pd.to_numeric(df["distance"], errors="coerce").fillna(0)
        user_home = pd.to_numeric(df["user_home_dis"], errors="coerce").fillna(0)
        df["distance_home_ratio"] = distance / (user_home + 1)


def add_price_deviation_features(df: pd.DataFrame) -> None:
    required = {"price", "cityid", "dtype"}
    if not required.issubset(df.columns):
        return

    price = pd.to_numeric(df["price"], errors="coerce")
    city = df["cityid"].fillna("__MISSING__").astype(str)
    dtype = df["dtype"].fillna("__MISSING__").astype(str)

    combined_median = price.groupby([city, dtype]).transform("median")
    dtype_median = price.groupby(dtype).transform("median")
    global_median = price.median()

    reference_price = combined_median.fillna(dtype_median).fillna(global_median)

    df["sale"] = price - reference_price

    reference_safe = reference_price.where(reference_price.abs() > 1e-6, np.nan)
    ratio = price / reference_safe
    df["price_ratio"] = ratio.replace([np.inf, -np.inf], np.nan)


def add_frequency_features(df: pd.DataFrame) -> None:
    def _apply(column: str, freq_col: str) -> None:
        if column not in df.columns:
            return
        counts = df[column].value_counts(dropna=False)
        freq = df[column].map(counts).fillna(0)
        df[freq_col] = np.log1p(freq)
    _apply("cityid", "cityid_freq")
    _apply("dtype", "dtype_freq")


def bucket_itemid(df: pd.DataFrame) -> None:
    if "itemid" not in df.columns:
        return
    item_series = df["itemid"].astype("string")
    counts = item_series.value_counts(dropna=False)
    keep_ids = counts[counts >= ITEM_RARE_THRESHOLD].index
    df["itemid"] = item_series.where(item_series.isin(keep_ids), ITEM_UNKNOWN_TOKEN)


def bucket_additional_categories(df: pd.DataFrame) -> None:
    bucket_rules: Sequence[Tuple[str, int, str]] = [
        ("cityid", CITY_RARE_THRESHOLD, RARE_TOKEN),
        ("loc_cityid", LOC_CITY_RARE_THRESHOLD, RARE_TOKEN),
        ("dtype", DTYPE_RARE_THRESHOLD, RARE_TOKEN),
        ("cate_1", CATE_1_RARE_THRESHOLD, RARE_TOKEN),
        ("cate_2", CATE_2_RARE_THRESHOLD, RARE_TOKEN),
        ("cate_3", CATE_3_RARE_THRESHOLD, RARE_TOKEN),
        ("mobile_type", MOBILE_TYPE_RARE_THRESHOLD, "0"),
    ]

    for column, threshold, token in bucket_rules:
        if column not in df.columns or threshold <= 0:
            continue
        series = df[column].astype("string")
        counts = series.value_counts(dropna=False)
        keep = counts[counts >= threshold].index
        df[column] = series.where(series.isin(keep), token)


def transform_dataset(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = load_dataset(input_path)

    cap_distance_outliers(df)
    add_timeslot(df)
    add_is_same_city(df)
    add_is_weekend(df)
    add_cross_features(df)
    add_price_deviation_features(df)
    add_frequency_features(df)
    bucket_additional_categories(df)
    bucket_itemid(df)

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
