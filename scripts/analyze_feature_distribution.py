#!/usr/bin/env python3
"""Analyze per-column distributions to detect long-tail categorical features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize feature distributions and long tails.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/recsys_task_data/train_merged_transformed-20221014.csv"),
        help="CSV file to analyze.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of most frequent values to display for categorical columns.",
    )
    parser.add_argument(
        "--tail-threshold",
        type=int,
        default=10,
        help="Frequency threshold to flag long-tail categories (<= threshold).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file to store the full statistics.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    return df


def detect_numeric(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    numeric = pd.to_numeric(series, errors="coerce")
    # treat as numeric if at least half the non-null values are numeric
    valid_ratio = numeric.notna().sum() / max(len(series) - series.isna().sum(), 1)
    return valid_ratio >= 0.5


def analyze_numeric(series: pd.Series) -> Dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    stats: Dict[str, Any] = {
        "min": float(numeric.min()) if numeric.notna().any() else None,
        "max": float(numeric.max()) if numeric.notna().any() else None,
        "mean": float(numeric.mean()) if numeric.notna().any() else None,
        "std": float(numeric.std()) if numeric.notna().any() else None,
        "p25": float(numeric.quantile(0.25)) if numeric.notna().any() else None,
        "p50": float(numeric.quantile(0.50)) if numeric.notna().any() else None,
        "p75": float(numeric.quantile(0.75)) if numeric.notna().any() else None,
    }
    return stats


def analyze_categorical(series: pd.Series, top_k: int, tail_threshold: int) -> Dict[str, Any]:
    value_counts = series.astype("string").value_counts(dropna=False)
    tail_mask = value_counts <= tail_threshold
    tail_counts = value_counts[tail_mask]
    categorical_stats: Dict[str, Any] = {
        "top_values": value_counts.head(top_k).to_dict(),
        "tail_unique_count": int(tail_mask.sum()),
        "tail_sample_ratio": float(tail_counts.sum() / len(series)) if len(series) else 0.0,
    }
    return categorical_stats


def summarize_column(
    series: pd.Series,
    top_k: int,
    tail_threshold: int,
) -> Dict[str, Any]:
    total = len(series)
    non_null = int(series.notna().sum())
    unique = int(series.nunique(dropna=True))
    summary: Dict[str, Any] = {
        "count": total,
        "non_null": non_null,
        "missing": total - non_null,
        "missing_ratio": float((total - non_null) / total) if total else 0.0,
        "unique": unique,
        "unique_ratio": float(unique / non_null) if non_null else 0.0,
    }

    if detect_numeric(series):
        summary["type"] = "numeric"
        summary["numeric_stats"] = analyze_numeric(series)
    else:
        summary["type"] = "categorical"
        summary["categorical_stats"] = analyze_categorical(series, top_k, tail_threshold)

    return summary


def main() -> None:
    args = parse_args()
    df = load_dataset(args.input)

    results: Dict[str, Dict[str, Any]] = {}
    for column in df.columns:
        summary = summarize_column(df[column], args.top_k, args.tail_threshold)
        results[column] = summary

        print("-" * 80)
        print(f"Column: {column}")
        print(
            f"  count={summary['count']:,}, missing={summary['missing']:,}"
            f" ({summary['missing_ratio']:.2%}), unique={summary['unique']:,}"
        )
        if summary["type"] == "numeric":
            stats = summary["numeric_stats"]
            values = {k: (stats.get(k) if stats.get(k) is not None else 0.0) for k in stats}
            print(
                "  numeric stats: min={min:.3f}, p25={p25:.3f}, median={p50:.3f}, "
                "p75={p75:.3f}, max={max:.3f}".format(**values)
            )
        else:
            cat = summary["categorical_stats"]
            print(
                f"  top {args.top_k}: "
                + ", ".join(f"{k} ({v})" for k, v in cat['top_values'].items())
            )
            print(
                f"  tail (<= {args.tail_threshold} freq): {cat['tail_unique_count']} categories"
                f" covering {cat['tail_sample_ratio']:.2%} of samples"
            )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nFull statistics saved to {args.output}")
if __name__ == "__main__":
    main()
