#!/usr/bin/env python3
"""Train CatBoost using K-Fold CV to utilize full dataset while monitoring AUC."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from catboost.utils import get_gpu_device_count
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# --- 配置部分保持不变 ---
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS: List[str] = [
    "itemid",
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
    parser = argparse.ArgumentParser(description="K-Fold CatBoost Training")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/recsys_task_data/train_merged_transformed-20221014.csv"),
        help="Path to the FULL transformed training dataset.",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("data/recsys_task_data/test_merged_transformed-20221019.csv"),
        help="Path to the test dataset for inference.",
    )
    parser.add_argument(
        "--n-splits", type=int, default=5, help="Number of K-Fold splits."
    )
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=2025)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument(
        "--output-file", type=Path, default=Path("submission_kfold.csv")
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path, low_memory=False)


def preprocess(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    # 简单的类型转换，防止报错
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("<NA>")
    for col in NUMERICAL_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def main() -> None:
    args = parse_args()

    # 1. 加载全量训练数据和测试数据
    print("Loading datasets...")
    full_train_df = load_dataset(args.train_file)
    test_df = load_dataset(args.test_file)

    full_train_df = preprocess(full_train_df, is_train=True)
    test_df = preprocess(test_df, is_train=False)

    X_full = full_train_df[FEATURE_COLUMNS]
    y_full = full_train_df[LABEL_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]

    # 准备存储测试集的预测结果（累加）
    test_preds_accum = np.zeros(len(test_df))
    # 存储每一折的验证集 AUC
    fold_aucs = []

    # 2. 初始化 K-Fold
    skf = StratifiedKFold(
        n_splits=args.n_splits, shuffle=True, random_state=args.random_state
    )

    task_type = args.task_type
    if task_type is None:
        task_type = "GPU" if get_gpu_device_count() > 0 else "CPU"
    print(f"Starting {args.n_splits}-Fold CV using {task_type}...")

    # 3. 循环训练
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"\n{'='*20} Fold {fold + 1}/{args.n_splits} {'='*20}")

        # 切分数据
        X_train, y_train = X_full.iloc[train_idx], y_full.iloc[train_idx]
        X_val, y_val = X_full.iloc[val_idx], y_full.iloc[val_idx]

        train_pool = Pool(data=X_train, label=y_train, cat_features=CATEGORICAL_COLUMNS)
        val_pool = Pool(data=X_val, label=y_val, cat_features=CATEGORICAL_COLUMNS)

        # 定义模型
        model = CatBoostClassifier(
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=args.random_state + fold,  # 让每折有点随机差异
            task_type=task_type,
            early_stopping_rounds=100,
            verbose=200,
            allow_writing_files=False,  # 防止生成大量临时文件
        )

        # 训练
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        # 验证集评估
        val_pred_proba = model.predict_proba(val_pool)[:, 1]
        auc = roc_auc_score(y_val, val_pred_proba)
        fold_aucs.append(auc)
        print(f"Fold {fold + 1} AUC: {auc:.6f}")

        # 对测试集进行预测，并累加
        # 注意：这里我们用当前折的模型对整个测试集预测
        test_preds_accum += model.predict_proba(X_test)[:, 1]

    # 4. 汇总结果
    mean_auc = np.mean(fold_aucs)
    print(f"\n{'='*40}")
    print(f"Mean Validation AUC: {mean_auc:.6f}")
    print(f"Fold AUCs: {[round(x, 4) for x in fold_aucs]}")

    # 5. 生成最终提交文件 (取平均)
    avg_test_preds = test_preds_accum / args.n_splits

    submission = pd.DataFrame(
        {"sample_index": test_df["sample_index"], "label": avg_test_preds}
    )

    submission.to_csv(args.output_file, index=False)
    print(f"Saved ensemble predictions to {args.output_file}")


if __name__ == "__main__":
    main()
