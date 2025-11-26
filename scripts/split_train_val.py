#!/usr/bin/env python3
"""Split the binned training dataset into train and validation subsets."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["weekday", "timeslot"]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    return df


def stratified_split(
    df: pd.DataFrame,
    val_ratio: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input data is missing required columns: {missing_cols}")

    rng = np.random.default_rng(random_state)
    val_indices = []

    for _, group in df.groupby(REQUIRED_COLUMNS):
        group_size = len(group)
        if group_size == 0:
            continue
        val_count = max(1, int(round(group_size * val_ratio)))
        val_count = min(group_size, val_count)
        seed = int(rng.integers(0, 1_000_000_000))
        sampled_idx = group.sample(n=val_count, random_state=seed).index
        val_indices.extend(sampled_idx.tolist())

    val_mask = df.index.isin(val_indices)
    val_df = df.loc[val_mask].copy()
    train_df = df.loc[~val_mask].copy()

    _verify_coverage(df, val_df)

    return train_df, val_df


def _verify_coverage(full_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    full_groups = set(tuple(idx) for idx in full_df[REQUIRED_COLUMNS].drop_duplicates().itertuples(index=False, name=None))
    val_groups = set(tuple(idx) for idx in val_df[REQUIRED_COLUMNS].drop_duplicates().itertuples(index=False, name=None))
    missing = full_groups - val_groups
    if missing:
        raise RuntimeError(
            "Validation set is missing some (weekday, timeslot) combinations: "
            f"{sorted(missing)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Split binned train data into train/validation subsets.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/recsys_task_data/train_merged_binned-20221014.csv"),
        help="Path to the binned training dataset.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("data/recsys_task_data/train_merged_binned_train-20221014.csv"),
        help="Output path for the training subset.",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=Path("data/recsys_task_data/train_merged_binned_val-20221014.csv"),
        help="Output path for the validation subset.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Approximate fraction of samples to allocate to validation (0-1).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    df = load_dataset(args.input)
    train_df, val_df = stratified_split(df, args.val_ratio, args.random_state)

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.val_output.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(args.train_output, index=False)
    val_df.to_csv(args.val_output, index=False)

    print(f"Train samples: {len(train_df)} -> {args.train_output}")
    print(f"Validation samples: {len(val_df)} -> {args.val_output}")
    print(
        "Validation coverage confirmed for all (weekday, timeslot) combinations."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error: {exc}", file=sys.stderr)
        raise
