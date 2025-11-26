#!/usr/bin/env python3
"""Train and evaluate a DCN v2 model on the Meituan recommendation dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from deepctr_torch.inputs import SparseFeat, get_feature_names
from deepctr_torch.models import DCN
from sklearn.preprocessing import LabelEncoder

DEFAULT_FEATURE_ORDER: List[str] = [
    # "userid",
    # "itemid",
    "cityid",
    "loc_cityid",
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
    "distance_bin",
    "user_home_dis_bin",
    "user_work_dis_bin",
    "item_ave_price_bin",
    "price_bin",
    "user_displayed_item_num_bin",
    "online_days_bin",
    "temp_bin",
    "discount_rate_bin",
    "distance_home_ratio_bin",
        "cityid_freq_bin",
        "dtype_freq_bin",
    "dtype_distance",
]

LABEL_COLUMN = "label"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DCN v2 on binned datasets using DeepCTR-Torch.")
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/recsys_task_data/train_merged_binned_train-20221014.csv"),
        help="Path to the training split CSV.",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=Path("data/recsys_task_data/train_merged_binned_val-20221014.csv"),
        help="Path to the validation split CSV.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=16,
        help="Embedding dimension for sparse features.",
    )
    parser.add_argument(
        "--hidden-units",
        type=int,
        nargs="+",
        default=[256, 128, 64],
        help="DNN hidden layer sizes for DCN.",
    )
    parser.add_argument(
        "--cross-num",
        type=int,
        default=4,
        help="Number of cross layers in DCN v2.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="Optional additional split from the provided train CSV (if validation CSV is missing).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=2025,
        help="Random seed for label encoding and optional split.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path, low_memory=False)


def ensure_columns(df: pd.DataFrame, columns: List[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")


def prepare_dataframes(args: argparse.Namespace) -> Dict[str, pd.DataFrame]:
    train_df = load_dataset(args.train)
    ensure_columns(train_df, [LABEL_COLUMN, *DEFAULT_FEATURE_ORDER])

    if args.val.exists():
        val_df = load_dataset(args.val)
    else:
        if args.val_ratio is None:
            raise ValueError("Validation CSV missing; provide --val-ratio to create a split on the fly.")
        val_df = train_df.sample(frac=args.val_ratio, random_state=args.random_state)
        train_df = train_df.drop(index=val_df.index)

    ensure_columns(val_df, [LABEL_COLUMN, *DEFAULT_FEATURE_ORDER])
    return {"train": train_df.reset_index(drop=True), "val": val_df.reset_index(drop=True)}


def label_encode_columns(train_df: pd.DataFrame, val_df: pd.DataFrame, columns: List[str]) -> Dict[str, LabelEncoder]:
    encoders: Dict[str, LabelEncoder] = {}
    for col in columns:
        encoder = LabelEncoder()
        combined = pd.concat([train_df[col], val_df[col]], axis=0).astype(str)
        encoder.fit(combined.fillna("<UNK>"))
        train_df[col] = encoder.transform(train_df[col].astype(str).fillna("<UNK>"))
        val_df[col] = encoder.transform(val_df[col].astype(str).fillna("<UNK>"))
        encoders[col] = encoder
    return encoders


def build_feature_columns(encoders: Dict[str, LabelEncoder], embedding_dim: int) -> List[SparseFeat]:
    feature_columns: List[SparseFeat] = []
    for col, encoder in encoders.items():
        vocab_size = len(encoder.classes_)
        feature_columns.append(
            SparseFeat(col, vocabulary_size=vocab_size, embedding_dim=embedding_dim)
        )
    return feature_columns


def dataframe_to_model_input(df: pd.DataFrame, feature_names: List[str]) -> Dict[str, np.ndarray]:
    return {name: df[name].values for name in feature_names}


def main() -> None:
    args = parse_args()
    data = prepare_dataframes(args)

    train_df = data["train"].copy()
    val_df = data["val"].copy()

    encoders = label_encode_columns(train_df, val_df, DEFAULT_FEATURE_ORDER)
    feature_columns = build_feature_columns(encoders, args.embedding_dim)
    feature_names = get_feature_names(feature_columns)

    train_model_input = dataframe_to_model_input(train_df, feature_names)
    val_model_input = dataframe_to_model_input(val_df, feature_names)

    train_labels = train_df[LABEL_COLUMN].values
    val_labels = val_df[LABEL_COLUMN].values

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DCN(
        linear_feature_columns=feature_columns,
        dnn_feature_columns=feature_columns,
        task="binary",
        dnn_hidden_units=tuple(args.hidden_units),
        cross_num=args.cross_num,
        cross_parameterization="matrix",
        device=device,
    )

    model.compile("adam", "binary_crossentropy", metrics=["auc"])

    model.fit(
        train_model_input,
        train_labels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1,
        validation_data=(val_model_input, val_labels),
    )

    val_pred = model.predict(val_model_input, batch_size=args.batch_size)
    print("Validation prediction sample:", val_pred[:5].flatten())


if __name__ == "__main__":
    main()
