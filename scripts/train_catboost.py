#!/usr/bin/env python3
"""Train and evaluate a CatBoost model on the transformed Meituan dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
from catboost import CatBoostClassifier, Pool
from catboost.utils import get_gpu_device_count
from sklearn.metrics import classification_report, roc_auc_score

LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS: List[str] = [
    # "userid",
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

FEATURE_COLUMNS: List[str] = [*CATEGORICAL_COLUMNS, *NUMERICAL_COLUMNS]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CatBoost on transformed Meituan dataset.")
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/recsys_task_data/train_merged_transformed_train-20221014.csv"),
        help="Training CSV path (used only if --only-train is not set).",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=Path("data/recsys_task_data/train_merged_transformed_val-20221014.csv"),
        help="Validation CSV path (ignored if --only-train is True).",
    )
    parser.add_argument(
        "--only-train",
        action="store_true",
        help="If set, train on full dataset without validation (uses --full-train).",
    )
    parser.add_argument(
        "--full-train",
        type=Path,
        default=Path("data/recsys_task_data/train_merged_transformed-20221014.csv"),
        help="Path to full training dataset (used when --only-train is True).",
    )

    # 其他参数保持不变
    parser.add_argument(
        "--iterations",
        type=int,
        default=2000,
        help="Max boosting iterations.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate for CatBoost.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=8,
        help="Tree depth.",
    )
    parser.add_argument(
        "--l2-leaf-reg",
        type=float,
        default=3.0,
        help="L2 regularization coefficient.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=2025,
        help="Random seed.",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=100,
        help="Early stopping patience (ignored if --only-train is True).",
    )
    parser.add_argument(
        "--eval-metric",
        type=str,
        default="AUC",
        help="Primary evaluation metric (e.g., AUC, Logloss).",
    )
    parser.add_argument(
        "--cat-features",
        type=str,
        nargs="*",
        default=None,
        help="Override list of categorical feature names.",
    )
    parser.add_argument(
        "--auto-class-weights",
        action="store_true",
        help="Enable CatBoost's automatic class weight balancing.",
    )
    parser.add_argument(
        "--task-type",
        choices=["CPU", "GPU"],
        default=None,
        help="Device for CatBoost (GPU auto-detected if available).",
    )
    parser.add_argument(
        "--badcase-topk",
        type=int,
        default=10,
        help="Number of top false positives/negatives to print (ignored if --only-train).",
    )
    parser.add_argument(
        "--slice-cols",
        type=str,
        nargs="*",
        default=["weekday", "timeslot", "cityid", "dtype"],
        help="Feature columns for slice bias analysis (ignored if --only-train).",
    )
    parser.add_argument(
        "--slice-topk",
        type=int,
        default=5,
        help="Top slices to display per feature (ignored if --only-train).",
    )
    return parser.parse_args()

def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path, low_memory=False)


def ensure_columns(df: pd.DataFrame, columns: List[str], df_name: str) -> None:
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{df_name} dataset missing columns: {missing_cols}")


def cast_categorical(df: pd.DataFrame, columns: List[str], na_token: str = "<NA>") -> None:
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = df[col].astype("string").fillna(na_token)


def cast_numeric(df: pd.DataFrame, columns: List[str]) -> None:
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")


def analyze_feature_bias(
    df: pd.DataFrame,
    slice_cols: List[str],
    topk: int,
) -> None:
    required_cols = {LABEL_COLUMN, "pred_proba"}
    if not required_cols.issubset(df.columns):
        print("Skipping slice analysis due to missing columns.")
        return

    for col in slice_cols:
        if col not in df.columns:
            print(f"Column '{col}' not found for slice analysis, skipping.")
            continue
        agg = (
            df.groupby(col)
            .agg(
                samples=(LABEL_COLUMN, "size"),
                positive_rate=(LABEL_COLUMN, "mean"),
                pred_mean=("pred_proba", "mean"),
            )
            .reset_index()
        )
        agg["bias"] = agg["pred_mean"] - agg["positive_rate"]
        agg["abs_bias"] = agg["bias"].abs()
        shown = agg.sort_values("abs_bias", ascending=False).head(topk)
        if shown.empty:
            print(f"No data available for column '{col}'.")
            continue
        print(f"\nSlice bias for '{col}' (top {len(shown)} by |bias|):")
        print(
            shown[[col, "samples", "positive_rate", "pred_mean", "bias"]]
            .rename(
                columns={
                    "positive_rate": "label_rate",
                    "pred_mean": "pred_rate",
                }
            )
            .to_string(index=False)
        )

def main() -> None:
    args = parse_args()

    if args.only_train:
        print("=> Mode: ONLY TRAIN (no validation, no evaluation)")
        train_df = load_dataset(args.full_train)
        ensure_columns(train_df, [LABEL_COLUMN, *FEATURE_COLUMNS], "full-train")

        cat_features = args.cat_features if args.cat_features else CATEGORICAL_COLUMNS.copy()
        invalid_cats = [col for col in cat_features if col not in FEATURE_COLUMNS]
        if invalid_cats:
            raise ValueError(f"Categorical columns not in feature set: {invalid_cats}")

        numeric_features = [col for col in FEATURE_COLUMNS if col not in cat_features]
        train_features = train_df[FEATURE_COLUMNS].copy()

        cast_categorical(train_features, cat_features)
        cast_numeric(train_features, numeric_features)

        train_pool = Pool(
            data=train_features,
            label=train_df[LABEL_COLUMN],
            cat_features=cat_features,
        )

        task_type = args.task_type or ("GPU" if get_gpu_device_count() > 0 else "CPU")
        print(f"Using CatBoost task_type={task_type}")

        model = CatBoostClassifier(
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            loss_function="Logloss",
            l2_leaf_reg=args.l2_leaf_reg,
            random_seed=args.random_state,
            task_type=task_type,
            auto_class_weights="Balanced" if args.auto_class_weights else None,
            verbose=200,
            # ⚠️ No early stopping without validation
        )

        model.fit(train_pool)  # No eval_set

        model.save_model("catboost_model.cbm")
        print("Saved CatBoost model to catboost_model.cbm")

        feature_imp = model.get_feature_importance(prettified=True)
        print("Top 10 feature importances:")
        print(feature_imp.head(10))

    else:
        # Original train + validation logic
        print("=> Mode: TRAIN WITH VALIDATION")
        train_df = load_dataset(args.train)
        val_df = load_dataset(args.val)

        ensure_columns(train_df, [LABEL_COLUMN, *FEATURE_COLUMNS], "train")
        ensure_columns(val_df, [LABEL_COLUMN, *FEATURE_COLUMNS], "val")

        cat_features = args.cat_features if args.cat_features else CATEGORICAL_COLUMNS.copy()
        invalid_cats = [col for col in cat_features if col not in FEATURE_COLUMNS]
        if invalid_cats:
            raise ValueError(f"Categorical columns not found in feature set: {invalid_cats}")

        numeric_features = [col for col in FEATURE_COLUMNS if col not in cat_features]

        train_features = train_df[FEATURE_COLUMNS].copy()
        val_features = val_df[FEATURE_COLUMNS].copy()

        cast_categorical(train_features, cat_features)
        cast_categorical(val_features, cat_features)
        cast_numeric(train_features, numeric_features)
        cast_numeric(val_features, numeric_features)

        train_pool = Pool(
            data=train_features,
            label=train_df[LABEL_COLUMN],
            cat_features=cat_features,
        )
        val_pool = Pool(
            data=val_features,
            label=val_df[LABEL_COLUMN],
            cat_features=cat_features,
        )

        task_type = args.task_type or ("GPU" if get_gpu_device_count() > 0 else "CPU")
        print(f"Using CatBoost task_type={task_type}")

        model = CatBoostClassifier(
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            loss_function="Logloss",
            eval_metric=args.eval_metric,
            l2_leaf_reg=args.l2_leaf_reg,
            random_seed=args.random_state,
            early_stopping_rounds=args.early_stopping_rounds,
            task_type=task_type,
            auto_class_weights="Balanced" if args.auto_class_weights else None,
            verbose=200,
        )

        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        val_pred_proba = model.predict_proba(val_pool)[:, 1]
        val_pred_label = (val_pred_proba >= 0.5).astype(int)

        auc = roc_auc_score(val_df[LABEL_COLUMN], val_pred_proba)
        print(f"Validation AUC: {auc:.6f}")
        print("Classification report:\n", classification_report(val_df[LABEL_COLUMN], val_pred_label))

        model.save_model("catboost_model.cbm")
        print("Saved CatBoost model to catboost_model.cbm")

        feature_imp = model.get_feature_importance(prettified=True)
        print("Top 10 feature importances:")
        print(feature_imp.head(10))

        if args.badcase_topk > 0:
            val_results = val_df.copy()
            val_results["pred_proba"] = val_pred_proba
            val_results["pred_label"] = val_pred_label
            misclassified = val_results[val_results["pred_label"] != val_results[LABEL_COLUMN]]

            false_pos = misclassified[misclassified[LABEL_COLUMN] == 0].sort_values(
                "pred_proba", ascending=False
            ).head(args.badcase_topk)
            false_neg = misclassified[misclassified[LABEL_COLUMN] == 1].sort_values(
                "pred_proba", ascending=True
            ).head(args.badcase_topk)

            def _print_bad_cases(df: pd.DataFrame, title: str) -> None:
                if df.empty:
                    print(f"No {title.lower()} found.")
                    return
                print(f"\nTop {len(df)} {title} (showing label, pred_proba, key features):")
                cols_to_show = [LABEL_COLUMN, "pred_proba", "pred_label", "weekday", "timeslot", "itemid"]
                existing = [c for c in cols_to_show if c in df.columns]
                print(df[existing].to_string(index=False))

            _print_bad_cases(false_pos, "False Positives")
            _print_bad_cases(false_neg, "False Negatives")

            analyze_feature_bias(val_results, args.slice_cols, args.slice_topk)
            
if __name__ == "__main__":
    main()
