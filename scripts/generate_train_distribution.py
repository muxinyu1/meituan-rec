#!/usr/bin/env python3
"""Generate distribution statistics for the Meituan recommendation training dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def _is_numeric(series: pd.Series, valid_ratio_threshold: float = 0.9) -> bool:
    """Determine whether a column should be treated as numeric."""
    if series.empty:
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    numeric = pd.to_numeric(non_null, errors="coerce")
    valid_ratio = numeric.notna().mean()
    return valid_ratio >= valid_ratio_threshold


def _summarize_numeric(series: pd.Series) -> Dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {
            "type": "numeric",
            "note": "No valid numeric values detected.",
        }

    desc = numeric.describe(percentiles=[0.25, 0.5, 0.75])
    return {
        "type": "numeric",
        "count": int(desc["count"]),
        "mean": float(desc["mean"]),
        "std": float(desc["std"]) if pd.notna(desc["std"]) else None,
        "min": float(desc["min"]),
        "25%": float(desc["25%"]),
        "50%": float(desc["50%"]),
        "75%": float(desc["75%"]),
        "max": float(desc["max"]),
    }


def _summarize_categorical(series: pd.Series, top_k: int = 5) -> Dict[str, Any]:
    non_null_series = series.dropna()
    value_counts = non_null_series.value_counts()
    top_values: List[Dict[str, Any]] = []
    total = len(non_null_series)
    for value, count in value_counts.head(top_k).items():
        top_values.append(
            {
                "value": value,
                "count": int(count),
                "ratio": float(count / total) if total else None,
            }
        )

    return {
        "type": "categorical",
        "unique": int(value_counts.shape[0]),
        "top_values": top_values,
    }


def _log_cap_scale(series: pd.Series, cap_value: float = 70000.0) -> pd.Series:
    """Apply log1p scaling to limit values to the 0-100 range."""
    numeric = pd.to_numeric(series, errors="coerce")
    clipped = numeric.clip(lower=0, upper=cap_value)
    scaled = np.log1p(clipped) / np.log1p(cap_value) * 100
    return scaled


def preprocess_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply requested preprocessing steps before summarization."""

    processed = df.copy()
    preprocessing_info: Dict[str, Any] = {
        "dropped_columns": [],
        "missing_indicators": [],
        "transformed_columns": {},
    }

    if "global_id" in processed.columns:
        processed = processed.drop(columns=["global_id"])
        preprocessing_info["dropped_columns"].append("global_id")

    missing_ratios = processed.isna().mean()
    for column, ratio in missing_ratios.items():
        if ratio > 0.2:
            indicator_name = f"{column}_is_missing"
            if indicator_name in processed.columns:
                continue
            processed[indicator_name] = processed[column].isna().astype(int)
            preprocessing_info["missing_indicators"].append(
                {
                    "column": column,
                    "indicator": indicator_name,
                    "missing_ratio": float(ratio),
                }
            )

    capped_columns = ["user_home_dis", "user_work_dis"]
    for column in capped_columns:
        if column in processed.columns:
            processed[column] = _log_cap_scale(processed[column])
            preprocessing_info["transformed_columns"][column] = {
                "transformation": "log_cap_scale",
                "cap_value": 70000.0,
                "target_range": [0, 100],
            }

    return processed, preprocessing_info


def summarize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total_rows": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "columns": {},
    }

    if "label" in df.columns:
        label_counts = df["label"].value_counts(dropna=False).to_dict()
        summary["label_distribution"] = {
            str(k): int(v) for k, v in label_counts.items()
        }

    for column in df.columns:
        col_series = df[column]
        total = int(col_series.shape[0])
        missing_mask = col_series.isna()
        non_null = total - int(missing_mask.sum())
        col_info: Dict[str, Any] = {
            "non_null": non_null,
            "null": total - non_null,
        }

        if non_null == 0:
            summary["columns"][column] = col_info
            continue

        if _is_numeric(col_series):
            col_info.update(_summarize_numeric(col_series))
        else:
            col_info.update(_summarize_categorical(col_series))

        summary["columns"][column] = col_info

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate column-level distribution statistics for the training dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/recsys_task_data/train_samples-20221014.csv"),
        help="Path to the training samples CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/recsys_task_data/train_distribution_stats.json"),
        help="Where to store the generated JSON summary.",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input, low_memory=False)
    # Treat empty strings or whitespace-only values as missing
    df = df.replace(r"^\s*$", pd.NA, regex=True)

    df, preprocessing_info = preprocess_dataframe(df)
    summary = summarize_dataframe(df)
    summary["preprocessing"] = preprocessing_info

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote distribution summary to {args.output}")


if __name__ == "__main__":
    main()
