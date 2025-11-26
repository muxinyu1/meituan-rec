#!/usr/bin/env python3
"""Run inference using a trained CatBoost model."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
from catboost import CatBoostClassifier, Pool

# --- CONFIG (Must match training code exactly) ---
CATEGORICAL_COLUMNS: List[str] = [
    # "userid",  # Commented out in training, so we exclude it here too
    "itemid",
    #"cityid",
    # "loc_cityid",
    "weekday",
    "hour",
    "weather",
    "dtype",
    "cate_1",
    "cate_2",
    "cate_3",
    "age",
    "level",
    "gender",
    "married",
    "job",
    "has_car",
    "mobile_type",
    "mobile_os",
    "level_is_missing",
    "timeslot",
    "is_same_city",
    "is_weekend",
]

NUMERICAL_COLUMNS: List[str] = [
    "distance",
    "user_home_dis",
    "user_work_dis",
    "user_home_dis_is_far",
    "user_work_dis_is_far",
    "item_ave_price",
    "price",
    "user_displayed_item_num",
    "online_days",
    "temp",
    "discount_rate",
    "distance_home_ratio",
    "sale",
    "price_ratio",
    "cityid_freq",
    "dtype_freq",
]

# The strict list of features the model expects
FEATURE_COLUMNS: List[str] = [*CATEGORICAL_COLUMNS, *NUMERICAL_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict using trained CatBoost model.")
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("data/recsys_task_data/test_merged_transformed-20221019.csv"),
        help="Path to the test CSV file.",
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=Path("catboost_model.cbm"),
        help="Path to the trained .cbm model file.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("submission.csv"),
        help="Path to save the prediction CSV.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    print(f"Loading data from {path}...")
    return pd.read_csv(path, low_memory=False)


def cast_categorical(df: pd.DataFrame, columns: List[str], na_token: str = "<NA>") -> None:
    for col in columns:
        if col not in df.columns:
            # Create missing categorical columns if they don't exist in test set
            df[col] = na_token
        df[col] = df[col].astype("string").fillna(na_token)


def cast_numeric(df: pd.DataFrame, columns: List[str]) -> None:
    for col in columns:
        if col not in df.columns:
            # Create missing numeric columns with 0 or NaN
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")


def main() -> None:
    args = parse_args()

    # 1. Load Data
    test_df = load_dataset(args.test_file)

    # 2. Check for sample_index (Required for submission)
    if "sample_index" not in test_df.columns:
        raise ValueError("Test dataset must contain 'sample_index' column.")
    
    sample_indices = test_df["sample_index"]

    # 3. Preprocess Features (Must apply same transformations as training)
    # Filter only relevant columns to avoid noise, but operate on a copy
    features_df = test_df.copy()
    
    # Ensure all feature columns exist and are cast correctly
    cast_categorical(features_df, CATEGORICAL_COLUMNS)
    cast_numeric(features_df, NUMERICAL_COLUMNS)
    
    # Strictly order columns to match model signature
    X = features_df[FEATURE_COLUMNS]

    print(f"Feature shape for inference: {X.shape}")

    # 4. Load Model
    if not args.model_file.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_file}")
    
    print(f"Loading model from {args.model_file}...")
    model = CatBoostClassifier()
    model.load_model(args.model_file)

    # 5. Predict
    # predict_proba returns [prob_class_0, prob_class_1], we want index 1
    print("Predicting...")
    pred_proba = model.predict_proba(X)[:, 1]

    # 6. Save Submission
    submission = pd.DataFrame({
        "sample_index": sample_indices,
        "label": pred_proba
    })

    print(f"Saving submission to {args.output_file}...")
    submission.to_csv(args.output_file, index=False)
    
    # Preview
    print("\nOutput preview:")
    print(submission.head().to_string(index=False))


if __name__ == "__main__":
    main()