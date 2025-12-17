#!/usr/bin/env python3
"""
LightGBM 训练脚本 V3
- 参考 train_catboost_v3.py 的特征工程与验证集划分方式
- 适配 LightGBM (可选 GPU)
"""

import os
import gc
import time
import json
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


class CTRFeatureEncoder:
    """基于K折的Target Encoding编码器，避免数据泄露 (与 CatBoost V3 一致)"""

    def __init__(self, cols: List[str], n_folds: int = 5, smoothing: float = 20, random_state: int = 42):
        self.cols = cols
        self.n_folds = n_folds
        self.smoothing = smoothing
        self.random_state = random_state
        self.global_mean = None
        self.encodings = {}

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        result = X.copy()
        self.global_mean = y.mean()

        from sklearn.model_selection import StratifiedKFold

        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for col in self.cols:
            if col not in X.columns:
                continue

            print(f"  处理Target Encoding: {col}")
            col_name = f"{col}_ctr"
            # 预分配 float32 数组，避免在 DataFrame block 中分批写入引发 dtype 问题
            encoded = np.full(len(X), np.nan, dtype=np.float32)

            for train_idx, val_idx in kf.split(X, y):
                train_data = pd.DataFrame({"col": X.iloc[train_idx][col], "label": y.iloc[train_idx]})
                stats = train_data.groupby("col")["label"].agg(["sum", "count"])
                smooth_ctr = (stats["sum"] + self.smoothing * self.global_mean) / (stats["count"] + self.smoothing)

                encoded[val_idx] = (
                    X.iloc[val_idx][col].map(smooth_ctr).astype(np.float32).values
                )

            encoded = np.where(np.isnan(encoded), self.global_mean, encoded).astype(np.float32)
            result[col_name] = encoded

            full_data = pd.DataFrame({"col": X[col], "label": y})
            full_stats = full_data.groupby("col")["label"].agg(["sum", "count"])
            self.encodings[col] = (full_stats["sum"] + self.smoothing * self.global_mean) / (
                full_stats["count"] + self.smoothing
            )

        return result

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = X.copy()
        for col in self.cols:
            if col not in X.columns or col not in self.encodings:
                continue
            col_name = f"{col}_ctr"
            mapped = X[col].map(self.encodings[col]).astype(np.float32)
            mapped = mapped.fillna(self.global_mean).astype(np.float32)
            result[col_name] = mapped.values
        return result


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and col_type.name != "category":
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"内存: {start_mem:.2f}MB -> {end_mem:.2f}MB (减少{100 * (start_mem - end_mem) / start_mem:.1f}%)")
    return df


def load_data(base_dir: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """加载数据；默认使用脚本所在目录下的 train.csv / test.csv"""
    base_dir = (
        base_dir
        or os.getenv("DATA_BASE_DIR")
        or os.path.dirname(os.path.abspath(__file__))
    )
    train_path = os.path.join(base_dir, "train.csv")
    test_path = os.path.join(base_dir, "test.csv")

    print("=" * 60)
    print("加载数据...")
    print("=" * 60)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"训练集: {train_df.shape}")
    print(f"测试集: {test_df.shape}")
    if "label" in train_df.columns:
        print(f"正样本比例: {train_df['label'].mean():.4f}")

    return train_df, test_df


def create_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """复用 CatBoost V3 的特征工程"""
    print("=" * 60)
    print("特征工程 V3 (LightGBM) ...")
    print("=" * 60)

    test_sample_index = test_df["sample_index"].copy() if "sample_index" in test_df.columns else None
    y_train = train_df["label"].copy()

    train_len = len(train_df)
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # 1. 缺失值标记特征
    print("\n1. 创建缺失值标记特征...")
    high_missing_cols = [
        "price",
        "online_days",
        "user_displayed_item_num",
        "distance",
        "item_ave_price",
        "level",
        "cate_1",
        "cate_2",
        "cate_3",
    ]
    for col in high_missing_cols:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isnull().astype(np.int8)

    # 2. 时间特征
    print("\n2. 创建时间特征...")
    if "timestamp" in df.columns:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        df["hour_sin"] = np.sin(2 * np.pi * df["timestamp_dt"].dt.hour / 24).astype(np.float32)
        df["hour_cos"] = np.cos(2 * np.pi * df["timestamp_dt"].dt.hour / 24).astype(np.float32)
        df["weekday_sin"] = np.sin(2 * np.pi * df["timestamp_dt"].dt.weekday / 7).astype(np.float32)
        df["weekday_cos"] = np.cos(2 * np.pi * df["timestamp_dt"].dt.weekday / 7).astype(np.float32)
        df["is_weekend"] = df["weekday"].isin([6, 7]).astype(np.int8)

        def get_time_period(hour):
            if 6 <= hour < 11:
                return 1
            elif 11 <= hour < 14:
                return 2
            elif 14 <= hour < 18:
                return 3
            elif 18 <= hour < 22:
                return 4
            else:
                return 0

        df["time_period"] = df["hour"].apply(get_time_period).astype(np.int8)
        df.drop(["timestamp", "timestamp_dt"], axis=1, inplace=True)

    # 3. Count Encoding
    print("\n3. 创建Count Encoding特征...")
    count_cols = [
        "userid",
        "itemid",
        "geohash",
        "cityid",
        "loc_cityid",
        "cate_1",
        "cate_2",
        "cate_3",
        "dtype",
    ]
    train_part = df.iloc[:train_len]
    for col in count_cols:
        if col in df.columns:
            counts = train_part[col].value_counts()
            df[f"{col}_count"] = df[col].map(counts).fillna(0).astype(np.float32)
            df[f"{col}_count_log"] = np.log1p(df[f"{col}_count"]).astype(np.float32)

    # 4. 交叉特征
    print("\n4. 创建交叉特征 (V3增强)...")
    if "cityid" in df.columns and "loc_cityid" in df.columns:
        df["is_same_city"] = (df["cityid"] == df["loc_cityid"]).astype(np.int8)

    cross_pairs = [
        ("userid", "cate_1"),
        ("userid", "cate_2"),
        ("cityid", "cate_1"),
        ("cityid", "cate_2"),
        ("time_period", "cate_1"),
        ("time_period", "dtype"),
        ("gender", "cate_1"),
        ("age", "cate_1"),
        ("level", "cate_1"),
        ("gender", "dtype"),
        ("age", "dtype"),
    ]
    for c1, c2 in cross_pairs:
        if c1 in df.columns and c2 in df.columns:
            new_col = f"{c1}_{c2}"
            df[new_col] = df[c1].astype(str) + "_" + df[c2].astype(str)

    # 5. 数值特征
    print("\n5. 处理数值特征 (V3增强)...")
    numerical_cols = [
        "distance",
        "item_ave_price",
        "price",
        "user_home_dis",
        "user_work_dis",
        "temp",
        "temp_low",
        "temp_high",
        "user_displayed_item_num",
        "online_days",
    ]
    for col in numerical_cols:
        if col in df.columns:
            median_val = train_part[col].median()
            df[col] = df[col].fillna(median_val).astype(np.float32)

    log_cols = ["distance", "price", "item_ave_price", "user_home_dis", "user_work_dis"]
    for col in log_cols:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(np.maximum(df[col], 0)).astype(np.float32)

    if "price" in df.columns and "cate_1" in df.columns:
        cate1_price_mean = df.groupby("cate_1")["price"].transform("mean")
        df["price_relative_to_cate1"] = (df["price"] / (cate1_price_mean + 1)).astype(np.float32)
        df["price_diff_cate1"] = (df["price"] - cate1_price_mean).astype(np.float32)

    if "price" in df.columns and "cityid" in df.columns:
        city_price_mean = df.groupby("cityid")["price"].transform("mean")
        df["price_relative_to_city"] = (df["price"] / (city_price_mean + 1)).astype(np.float32)

    if "user_displayed_item_num" in df.columns and "level" in df.columns:
        level_disp_mean = df.groupby("level")["user_displayed_item_num"].transform("mean")
        df["disp_relative_to_level"] = (df["user_displayed_item_num"] / (level_disp_mean + 1)).astype(np.float32)

    if "temp" in df.columns:
        df["temp_comfort"] = ((df["temp"] >= 17) & (df["temp"] <= 26)).astype(np.int8)

    if "temp_high" in df.columns and "temp_low" in df.columns:
        df["temp_range"] = (df["temp_high"] - df["temp_low"]).astype(np.float32)

    if "distance" in df.columns:
        df["distance_bin"] = pd.cut(df["distance"], bins=[-1, 20, 40, 60, 80, 200], labels=[0, 1, 2, 3, 4]).astype(float)
        df["distance_bin"] = df["distance_bin"].fillna(2).astype(np.int8)

    if "price" in df.columns:
        df["price_bin"] = pd.cut(df["price"], bins=[-1, 20, 40, 60, 80, 200], labels=[0, 1, 2, 3, 4]).astype(float)
        df["price_bin"] = df["price_bin"].fillna(2).astype(np.int8)

    # 6. 类别特征
    print("\n6. 处理类别特征...")
    categorical_features = [
        "userid",
        "itemid",
        "geohash",
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
        "work_geohash",
        "mobile_type",
        "mobile_os",
        "is_weekend",
        "time_period",
        "is_same_city",
        "temp_comfort",
        "distance_bin",
        "price_bin",
    ]
    for c1, c2 in cross_pairs:
        col_name = f"{c1}_{c2}"
        if col_name in df.columns:
            categorical_features.append(col_name)

    categorical_features = [col for col in categorical_features if col in df.columns]
    for col in categorical_features:
        df[col] = df[col].fillna("-1").astype("category")

    # 拆分
    train_processed = df.iloc[:train_len].copy()
    test_processed = df.iloc[train_len:].copy()

    del df, train_part
    gc.collect()

    # 7. Target Encoding
    print("\n7. 创建Target Encoding特征...")
    ctr_cols = [
        "dtype",
        "cate_1",
        "cate_2",
        "cate_3",
        "cityid",
        "weather",
        "time_period",
        "age",
        "level",
        "gender",
        "price_bin",
        "distance_bin",
    ]
    ctr_cols = [col for col in ctr_cols if col in train_processed.columns]

    encoder = CTRFeatureEncoder(cols=ctr_cols, n_folds=5, smoothing=20)
    exclude_cols = ["sample_index", "label"]
    feature_cols = [col for col in train_processed.columns if col not in exclude_cols]

    X_train = train_processed[feature_cols].copy()
    X_test = test_processed[[col for col in feature_cols if col in test_processed.columns]].copy()

    X_train = encoder.fit_transform(X_train, y_train)
    X_test = encoder.transform(X_test)

    if test_sample_index is not None:
        X_test["sample_index"] = test_sample_index.values

    categorical_features = [col for col in categorical_features if col in X_train.columns]

    X_train = reduce_mem_usage(X_train)
    X_test = reduce_mem_usage(X_test)

    for col in categorical_features:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")
        if col in X_test.columns:
            X_test[col] = X_test[col].astype("category")

    print(f"\n最终特征数量: {len([c for c in X_train.columns if c not in exclude_cols])}")

    del train_processed, test_processed
    gc.collect()

    return X_train, y_train, X_test, categorical_features


def build_weekday_folds(X: pd.DataFrame, n_splits: int = 10, seed: int = 42):
    """与 CatBoost V3 相同的 weekday 分桶切分方式"""
    rng = np.random.default_rng(seed)
    weekdays = X["weekday"].astype(str)
    folds = [[] for _ in range(n_splits)]
    for weekday in weekdays.unique():
        weekday_indices = X[weekdays == weekday].index.to_list()
        rng.shuffle(weekday_indices)
        fold_size = len(weekday_indices) // n_splits
        for i in range(n_splits):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < n_splits - 1 else len(weekday_indices)
            folds[i].extend(weekday_indices[start_idx:end_idx])
    for fold_indices in folds:
        rng.shuffle(fold_indices)
    return folds


def train_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_features: List[str],
    n_splits: int = 10,
    use_gpu: bool = False,
):
    print("=" * 60)
    print(f"开始 {n_splits} 折 LightGBM 训练 ...")
    print("=" * 60)

    folds = build_weekday_folds(X, n_splits=n_splits, seed=42)
    cv_scores = []
    models = []

    exclude_cols = ["sample_index", "label"]
    feature_cols = [col for col in X.columns if col not in exclude_cols]

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.025,
        "num_leaves": 31,
        "max_depth": 7,
        "feature_fraction": 0.65,
        "feature_fraction_bynode": 0.65,
        "bagging_fraction": 0.6,
        "bagging_freq": 5,
        "lambda_l1": 10.0,
        "lambda_l2": 10.0,
        "min_data_in_leaf": 400,
        "min_sum_hessian_in_leaf": 1e-2,
        "min_gain_to_split": 0.05,
        "max_bin": 127,
        "cat_l2": 15.0,
        "cat_smooth": 30.0,
        "scale_pos_weight": 1.0,
        "extra_trees": True,
        "verbose": -1,
        "seed": 42,
        "boosting": "gbdt",
    }
    if use_gpu:
        params.update({"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0})

    for fold in range(n_splits):
        print(f"\n{'=' * 60}")
        print(f"第 {fold + 1}/{n_splits} 折")
        print(f"{'=' * 60}")

        val_indices = folds[fold]
        train_indices = []
        for i in range(n_splits):
            if i != fold:
                train_indices.extend(folds[i])

        X_train_fold = X.iloc[train_indices][feature_cols].copy()
        y_train_fold = y.iloc[train_indices].copy()
        X_val_fold = X.iloc[val_indices][feature_cols].copy()
        y_val_fold = y.iloc[val_indices].copy()

        for col in categorical_features:
            if col in X_train_fold.columns:
                X_train_fold[col] = X_train_fold[col].astype("category")
                X_val_fold[col] = X_val_fold[col].astype("category")

        lgb_train = lgb.Dataset(X_train_fold, label=y_train_fold, categorical_feature=categorical_features, free_raw_data=False)
        lgb_val = lgb.Dataset(X_val_fold, label=y_val_fold, categorical_feature=categorical_features, free_raw_data=False)

        del X_train_fold, y_train_fold
        gc.collect()

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=8000,
            valid_sets=[lgb_train, lgb_val],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(200, verbose=False),
                lgb.log_evaluation(100),
            ],
        )

        best_iter = model.best_iteration
        print(f"最佳迭代: {best_iter}")

        y_pred_val = model.predict(lgb_val.data, num_iteration=best_iter)
        auc = roc_auc_score(y_val_fold, y_pred_val)
        cv_scores.append(auc)
        models.append(model)

        print(f"第 {fold + 1} 折 AUC: {auc:.4f}")

        del lgb_train, lgb_val, X_val_fold, y_val_fold
        gc.collect()

    print(f"\n{'=' * 60}")
    print("交叉验证结果汇总")
    print("=" * 60)
    print(f"各折AUC: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"平均AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"最大AUC: {max(cv_scores):.4f}")
    print(f"最小AUC: {min(cv_scores):.4f}")
    print("=" * 60)

    return models, cv_scores, feature_cols


def ensemble_predict(models: List[lgb.Booster], X_test: pd.DataFrame, categorical_features: List[str], feature_cols: List[str]):
    print("集成预测...")
    X_test_features = X_test[feature_cols].copy()
    for col in categorical_features:
        if col in X_test_features.columns:
            X_test_features[col] = X_test_features[col].astype("category")

    preds = np.zeros((X_test_features.shape[0], len(models)))
    for i, model in enumerate(models):
        preds[:, i] = model.predict(X_test_features, num_iteration=model.best_iteration)
    return preds.mean(axis=1)


def save_results(models, cv_scores, predictions: np.ndarray, X_test: pd.DataFrame):
    print("保存结果...")
    submission = pd.DataFrame(
        {
            "sample_index": X_test["sample_index"].astype(int) if "sample_index" in X_test.columns else range(len(predictions)),
            "label": predictions,
        }
    )
    submission.to_csv("submission_lgbm_v3.csv", index=False)
    print("预测结果已保存到 submission_lgbm_v3.csv")

    models[0].save_model("lgbm_v3.txt")
    print("模型已保存到 lgbm_v3.txt")

    results = {
        "metrics": {
            "auc_mean": float(np.mean(cv_scores)),
            "auc_std": float(np.std(cv_scores)),
            "auc_max": float(max(cv_scores)),
            "auc_min": float(min(cv_scores)),
            "fold_scores": [float(s) for s in cv_scores],
        },
        "model_params": models[0].params,
    }

    with open("model_results_lgbm_v3.json", "w") as f:
        json.dump(results, f, indent=4)
    print("评估结果已保存到 model_results_lgbm_v3.json")

    importance = models[0].feature_importance(importance_type="gain")
    feature_cols = [col for col in X_test.columns if col not in ["sample_index", "label"]]
    importance_df = pd.DataFrame({"feature": feature_cols, "importance": importance}).sort_values(
        "importance", ascending=False
    )
    print("\n前20个重要特征:")
    print(importance_df.head(20).to_string(index=False))
    importance_df.to_csv("feature_importance_lgbm_v3.csv", index=False)


def main():
    np.random.seed(42)
    start_time = time.time()

    train_df, test_df = load_data()
    X_train, y_train, X_test, categorical_features = create_features(train_df, test_df)

    del train_df, test_df
    gc.collect()

    models, cv_scores, feature_cols = train_lightgbm(
        X_train, y_train, categorical_features, n_splits=10, use_gpu=False
    )

    predictions = ensemble_predict(models, X_test, categorical_features, feature_cols)
    save_results(models, cv_scores, predictions, X_test)

    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed/60:.1f} 分钟")
    print("V3版 LightGBM 训练完成！")


if __name__ == "__main__":
    main()
