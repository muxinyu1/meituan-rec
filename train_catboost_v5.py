#!/usr/bin/env python3
"""\
CatBoost训练脚本 V5 (Leakage-safe版)

你遇到的现象：V4 线下CV AUC显著高于 V3，但线上提交反而更差。
核心原因：V3/V4 都在“外层CV之前”对全量训练集做了 K-fold Target Encoding，
而外层CV的划分方式（按weekday再均分）与 TE 的 KFold（StratifiedKFold shuffle）不一致。
这会导致：外层验证集样本的 TE 特征中混入了同一外层验证折中其它样本的 label 统计信息 -> 线下 AUC 虚高。
V4 引入更多交叉TE特征后，这个问题被放大，因此线下看起来更强，但线上泛化更差。

V5 修复策略：
- 先构造“无label依赖”的基础特征（count/排名/统计聚合等）。
- 在外层CV的每一折内：
  - 仅用该折的训练部分 (outer-train) 计算 TE 映射；
  - outer-train 内部再做一层 KFold 生成训练集 OOF-TE（避免训练时自身label泄露）；
  - 用 outer-train 的全量映射去变换 outer-val 与 test；
  - 训练该折模型，并同时对 test 预测；
- 汇总每折 test 预测做(加权/平均)集成。

注意：
- 该脚本结构正确但计算更重（每外层fold都会重算TE）。
- 输出 submission 会使用 test.csv 原始 sample_index，而不是 range(len(pred))。
"""

import gc
import json
import os
import shutil
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# 与 v4 保持一致：大文件写到 /media 下
BASE_DIR = "/media/wenhao/7d8f46ef-9674-40b1-a999-148b79a69954/zhaoshanhui/meituan-rec"
CHECKPOINT_SUBDIR = "train_catboost_v5"


def get_output_dir() -> str:
    out_dir = os.path.join(BASE_DIR, CHECKPOINT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def safe_link_or_copy(src: str, dst: str) -> str:
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if src == dst:
        return dst
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
        return dst
    except OSError:
        pass
    try:
        os.symlink(src, dst)
        return dst
    except OSError:
        pass
    shutil.copy2(src, dst)
    return dst


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
                else:
                    df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"内存: {start_mem:.2f}MB -> {end_mem:.2f}MB (减少{100*(start_mem-end_mem)/start_mem:.1f}%)")
    return df


def load_data():
    train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))
    print(f"训练集: {train_df.shape}, 正样本: {train_df['label'].mean():.4f}")
    print(f"测试集: {test_df.shape}")
    return train_df, test_df


def _combine_key(df: pd.DataFrame, cols) -> pd.Series:
    if isinstance(cols, str):
        cols = [cols]
    s = df[cols[0]].astype(str)
    for c in cols[1:]:
        s = s + "_" + df[c].astype(str)
    return s


@dataclass
class TESpec:
    cols: list  # list[str] or list for cross
    name: str   # output column name


def _smooth_ctr(sum_cnt: pd.DataFrame, global_mean: float, smoothing: float) -> pd.Series:
    # sum_cnt has columns ['sum','count']
    return (sum_cnt["sum"] + smoothing * global_mean) / (sum_cnt["count"] + smoothing)


def add_te_features_per_fold(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    te_specs: list,
    smoothing: float = 20.0,
    inner_folds: int = 5,
    seed: int = 42,
):
    """\
    对外层fold内的数据，生成：
    - 训练集 OOF-TE（inner KFold，避免训练时自身label泄露）
    - 验证集/测试集 TE（用 outer-train 全量映射）

    返回 (X_tr2, X_val2, X_test2)
    """
    X_tr2 = X_tr.copy()
    X_val2 = X_val.copy()
    X_test2 = X_test.copy()

    global_mean = float(y_tr.mean())
    inner_kf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)

    for spec in te_specs:
        key_tr = _combine_key(X_tr2, spec.cols)
        key_val = _combine_key(X_val2, spec.cols)
        key_test = _combine_key(X_test2, spec.cols)

        # 1) 训练集 OOF TE
        oof = np.full(len(X_tr2), np.nan, dtype=np.float32)
        for inner_tr_idx, inner_va_idx in inner_kf.split(X_tr2, y_tr):
            tmp = pd.DataFrame({"k": key_tr.iloc[inner_tr_idx], "y": y_tr.iloc[inner_tr_idx]})
            stats = tmp.groupby("k")["y"].agg(["sum", "count"])
            mapping = _smooth_ctr(stats, global_mean, smoothing)
            oof[inner_va_idx] = key_tr.iloc[inner_va_idx].map(mapping).astype(np.float32)
        X_tr2[spec.name] = np.where(np.isnan(oof), global_mean, oof).astype(np.float32)

        # 2) val/test 用 outer-train 全量映射
        tmp_full = pd.DataFrame({"k": key_tr, "y": y_tr})
        stats_full = tmp_full.groupby("k")["y"].agg(["sum", "count"])
        mapping_full = _smooth_ctr(stats_full, global_mean, smoothing)

        X_val2[spec.name] = key_val.map(mapping_full).fillna(global_mean).astype(np.float32)
        X_test2[spec.name] = key_test.map(mapping_full).fillna(global_mean).astype(np.float32)

    return X_tr2, X_val2, X_test2


def create_base_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """\
    只做无label依赖的基础特征（借鉴 v4 的 count/曝光/排名/统计等）。
    关键点：不在这里做 Target Encoding。
    """
    test_sample_index = test_df["sample_index"].copy() if "sample_index" in test_df.columns else None

    y = train_df["label"].copy()
    train_len = len(train_df)

    df = pd.concat([train_df.drop(columns=["label"]), test_df], axis=0, ignore_index=True)

    # 1) missing flags
    high_missing_cols = [
        "price", "online_days", "user_displayed_item_num", "distance",
        "item_ave_price", "level", "cate_1", "cate_2", "cate_3",
    ]
    for col in high_missing_cols:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isnull().astype(np.int8)

    # 2) time features (keep v4 style)
    if "timestamp" in df.columns:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        df["is_weekend"] = (df["weekday"].isin([6, 7])).astype(np.int8)

        def get_time_period(hour):
            if 6 <= hour < 9:
                return 1
            elif 9 <= hour < 11:
                return 2
            elif 11 <= hour < 14:
                return 3
            elif 14 <= hour < 17:
                return 4
            elif 17 <= hour < 20:
                return 5
            elif 20 <= hour < 23:
                return 6
            else:
                return 0

        df["time_period"] = df["hour"].apply(get_time_period).astype(np.int8)
        df["is_rush_hour"] = ((df["hour"].isin([7, 8, 9, 17, 18, 19])) & (~df["is_weekend"].astype(bool))).astype(np.int8)
        df["is_meal_time"] = df["hour"].isin([11, 12, 13, 17, 18, 19, 20]).astype(np.int8)
        df.drop(["timestamp", "timestamp_dt"], axis=1, inplace=True)

    # 3) count encoding (counts from train only)
    train_part = df.iloc[:train_len]
    count_cols = ["userid", "itemid", "geohash", "cityid", "loc_cityid", "cate_1", "cate_2", "cate_3", "dtype"]
    for col in count_cols:
        if col in df.columns:
            vc = train_part[col].value_counts()
            df[f"{col}_count"] = df[col].map(vc).fillna(0).astype(np.float32)
            df[f"{col}_count_log"] = np.log1p(df[f"{col}_count"]).astype(np.float32)

    # 4) impressions (train only)
    if "userid" in train_df.columns:
        user_imp = train_df.groupby("userid").size().rename("user_impressions")
        df["user_impressions"] = df["userid"].map(user_imp).fillna(0).astype(np.float32)
        df["user_impressions_log"] = np.log1p(df["user_impressions"]).astype(np.float32)
        df["is_new_user"] = (df["user_impressions"] <= 1).astype(np.int8)
        df["user_activity_bin"] = pd.cut(df["user_impressions"], bins=[-1, 1, 2, 5, 10, 100], labels=[0, 1, 2, 3, 4]).astype(float).fillna(0).astype(np.int8)

    if "itemid" in train_df.columns:
        item_imp = train_df.groupby("itemid").size().rename("item_impressions")
        df["item_impressions"] = df["itemid"].map(item_imp).fillna(0).astype(np.float32)
        df["item_impressions_log"] = np.log1p(df["item_impressions"]).astype(np.float32)
        df["is_new_item"] = (df["item_impressions"] <= 1).astype(np.int8)
        df["item_popularity_bin"] = pd.cut(df["item_impressions"], bins=[-1, 1, 2, 5, 10, 100], labels=[0, 1, 2, 3, 4]).astype(float).fillna(0).astype(np.int8)

    # 5) rankings (train only)
    if "cate_1" in train_df.columns and "itemid" in train_df.columns:
        cate_item_cnt = train_df.groupby(["cate_1", "itemid"]).size().reset_index(name="cate_item_count")
        cate_item_cnt["cate_item_rank"] = cate_item_cnt.groupby("cate_1")["cate_item_count"].rank(method="dense", ascending=False)
        cate_item_cnt["cate_item_rank_pct"] = cate_item_cnt.groupby("cate_1")["cate_item_count"].rank(method="dense", pct=True)
        df = df.merge(cate_item_cnt[["cate_1", "itemid", "cate_item_rank", "cate_item_rank_pct"]], on=["cate_1", "itemid"], how="left")
        df["cate_item_rank"] = df["cate_item_rank"].fillna(9999).astype(np.float32)
        df["cate_item_rank_pct"] = df["cate_item_rank_pct"].fillna(0.5).astype(np.float32)

    if "cityid" in train_df.columns and "itemid" in train_df.columns:
        city_item_cnt = train_df.groupby(["cityid", "itemid"]).size().reset_index(name="city_item_count")
        city_item_cnt["city_item_rank_pct"] = city_item_cnt.groupby("cityid")["city_item_count"].rank(method="dense", pct=True)
        df = df.merge(city_item_cnt[["cityid", "itemid", "city_item_rank_pct"]], on=["cityid", "itemid"], how="left")
        df["city_item_rank_pct"] = df["city_item_rank_pct"].fillna(0.5).astype(np.float32)

    # 6) crosses (string)
    if "cityid" in df.columns and "loc_cityid" in df.columns:
        df["is_same_city"] = (df["cityid"] == df["loc_cityid"]).astype(np.int8)

    cross_pairs = [
        ("gender", "dtype"), ("age", "dtype"), ("level", "dtype"),
        ("gender", "cate_1"), ("age", "cate_1"), ("level", "cate_1"),
        ("time_period", "dtype"), ("time_period", "cate_1"),
        ("cityid", "cate_1"), ("cityid", "dtype"),
        ("weekday", "time_period"),
        ("is_weekend", "time_period"),
    ]
    for c1, c2 in cross_pairs:
        if c1 in df.columns and c2 in df.columns:
            df[f"{c1}_{c2}"] = df[c1].astype(str) + "_" + df[c2].astype(str)

    # 7) numeric processing
    numerical_cols = [
        "distance", "item_ave_price", "price", "user_home_dis", "user_work_dis",
        "temp", "temp_low", "temp_high", "user_displayed_item_num", "online_days",
    ]
    train_part_for_stats = df.iloc[:train_len]
    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(train_part_for_stats[col].median()).astype(np.float32)

    for col in ["distance", "price", "item_ave_price", "user_home_dis", "user_work_dis"]:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(np.maximum(df[col], 0)).astype(np.float32)

    if "price" in df.columns and "cate_1" in df.columns:
        cate1_price_mean = df.groupby("cate_1")["price"].transform("mean")
        cate1_price_std = df.groupby("cate_1")["price"].transform("std").fillna(1)
        df["price_relative_to_cate1"] = (df["price"] / (cate1_price_mean + 1)).astype(np.float32)
        df["price_zscore_cate1"] = ((df["price"] - cate1_price_mean) / (cate1_price_std + 0.01)).astype(np.float32)

    if "price" in df.columns and "cityid" in df.columns:
        city_price_mean = df.groupby("cityid")["price"].transform("mean")
        df["price_relative_to_city"] = (df["price"] / (city_price_mean + 1)).astype(np.float32)

    if "user_displayed_item_num" in df.columns and "level" in df.columns:
        level_disp_mean = df.groupby("level")["user_displayed_item_num"].transform("mean")
        df["disp_relative_to_level"] = (df["user_displayed_item_num"] / (level_disp_mean + 1)).astype(np.float32)

    if "distance" in df.columns and "cate_1" in df.columns:
        cate1_dist_mean = df.groupby("cate_1")["distance"].transform("mean")
        df["dist_relative_to_cate1"] = (df["distance"] / (cate1_dist_mean + 1)).astype(np.float32)

    if "temp" in df.columns:
        df["temp_comfort"] = ((df["temp"] >= 17) & (df["temp"] <= 26)).astype(np.int8)

    if "temp_high" in df.columns and "temp_low" in df.columns:
        df["temp_range"] = (df["temp_high"] - df["temp_low"]).astype(np.float32)

    if "distance" in df.columns:
        df["distance_bin"] = pd.cut(df["distance"], bins=[-1, 10, 20, 40, 60, 80, 200], labels=[0, 1, 2, 3, 4, 5]).astype(float).fillna(3).astype(np.int8)

    if "price" in df.columns:
        df["price_bin"] = pd.cut(df["price"], bins=[-1, 15, 30, 50, 75, 200], labels=[0, 1, 2, 3, 4]).astype(float).fillna(2).astype(np.int8)

    # 8) aggregations (train only)
    if "price" in df.columns and "cate_1" in df.columns:
        stats = train_part_for_stats.groupby("cate_1")["price"].agg(["mean", "std", "min", "max"])
        stats.columns = ["cate1_price_mean", "cate1_price_std", "cate1_price_min", "cate1_price_max"]
        df = df.merge(stats, on="cate_1", how="left")
        for c in stats.columns:
            df[c] = df[c].fillna(train_part_for_stats["price"].median()).astype(np.float32)

    if "distance" in df.columns and "cityid" in df.columns:
        stats = train_part_for_stats.groupby("cityid")["distance"].agg(["mean", "std"])
        stats.columns = ["city_dist_mean", "city_dist_std"]
        df = df.merge(stats, on="cityid", how="left")
        df["city_dist_mean"] = df["city_dist_mean"].fillna(train_part_for_stats["distance"].median()).astype(np.float32)
        df["city_dist_std"] = df["city_dist_std"].fillna(1).astype(np.float32)

    # categorical list (same spirit as v4)
    categorical_features = [
        "userid", "itemid", "geohash", "cityid", "loc_cityid",
        "weekday", "hour", "weather", "dtype", "cate_1", "cate_2", "cate_3",
        "age", "level", "gender", "married", "job", "has_car",
        "work_geohash", "mobile_type", "mobile_os",
        "is_weekend", "time_period", "is_same_city", "temp_comfort",
        "distance_bin", "price_bin",
        "is_rush_hour", "is_meal_time", "is_new_user", "is_new_item",
        "user_activity_bin", "item_popularity_bin",
    ]
    for c1, c2 in cross_pairs:
        col = f"{c1}_{c2}"
        if col in df.columns:
            categorical_features.append(col)

    categorical_features = [c for c in categorical_features if c in df.columns]
    for c in categorical_features:
        df[c] = df[c].fillna(-1).astype(str)

    # split
    X_train = df.iloc[:train_len].copy()
    X_test = df.iloc[train_len:].copy()

    if test_sample_index is not None:
        X_test["sample_index"] = test_sample_index.values

    del df, train_part, train_part_for_stats
    gc.collect()

    X_train = reduce_mem_usage(X_train)
    X_test = reduce_mem_usage(X_test)

    return X_train, y, X_test, categorical_features


def build_weekday_folds(X: pd.DataFrame, n_splits: int, seed: int = 42):
    rng = np.random.RandomState(seed)
    weekdays = X["weekday"].astype(str)
    folds = [[] for _ in range(n_splits)]
    for wd in weekdays.unique():
        idx = X[weekdays == wd].index.to_list()
        rng.shuffle(idx)
        fold_size = len(idx) // n_splits
        for i in range(n_splits):
            start = i * fold_size
            end = start + fold_size if i < n_splits - 1 else len(idx)
            folds[i].extend(idx[start:end])
    for f in folds:
        rng.shuffle(f)
    return folds


def train_v5(X_base: pd.DataFrame, y: pd.Series, X_test_base: pd.DataFrame, categorical_features: list, n_splits: int = 10):
    out_dir = get_output_dir()
    folds = build_weekday_folds(X_base, n_splits=n_splits, seed=42)

    # v5 的 TE 选择：建议先从少量稳定列开始（太多交叉TE容易过拟合）
    te_specs = [
        TESpec(["dtype"], "dtype_ctr"),
        TESpec(["cate_1"], "cate_1_ctr"),
        TESpec(["cate_2"], "cate_2_ctr"),
        TESpec(["cate_3"], "cate_3_ctr"),
        TESpec(["cityid"], "cityid_ctr"),
        TESpec(["weather"], "weather_ctr"),
        TESpec(["time_period"], "time_period_ctr"),
        TESpec(["age"], "age_ctr"),
        TESpec(["level"], "level_ctr"),
        TESpec(["gender"], "gender_ctr"),
        TESpec(["price_bin"], "price_bin_ctr"),
        TESpec(["distance_bin"], "distance_bin_ctr"),
        # 交叉TE：只保留少量最直观的
        TESpec(["cityid", "cate_1"], "cityid_cate_1_ctr"),
        TESpec(["cityid", "dtype"], "cityid_dtype_ctr"),
        TESpec(["gender", "cate_1"], "gender_cate_1_ctr"),
        TESpec(["time_period", "dtype"], "time_period_dtype_ctr"),
        TESpec(["weekday", "hour"], "weekday_hour_ctr"),
    ]

    # 模型参数：先偏保守，避免 v4 那种“线下好看线上掉”的过拟合
    params = {
        "task_type": "GPU",
        "iterations": 12000,
        "learning_rate": 0.03,
        "depth": 8,
        "l2_leaf_reg": 10.0,
        "random_strength": 1e-5,
        "border_count": 128,
        "thread_count": -1,
        "random_seed": 42,
        "verbose": 200,
        "eval_metric": "AUC",
        "loss_function": "Logloss",
        "auto_class_weights": "Balanced",
        "boosting_type": "Plain",
        "one_hot_max_size": 10,
        "nan_mode": "Min",
        "od_type": "Iter",
        "od_wait": 500,
        "bagging_temperature": 0.6,
        "gpu_ram_part": 0.95,
        "allow_writing_files": False,
        "use_best_model": True,
        "min_data_in_leaf": 100,
        "grow_policy": "SymmetricTree",
    }

    exclude_cols = ["sample_index"]
    feature_cols_base = [c for c in X_base.columns if c not in exclude_cols]

    cv_scores = []
    model_paths = []
    test_preds = []

    for fold in range(n_splits):
        print("=" * 60)
        print(f"第{fold+1}/{n_splits}折")
        print("=" * 60)

        val_idx = folds[fold]
        tr_idx = []
        for i in range(n_splits):
            if i != fold:
                tr_idx.extend(folds[i])

        X_tr = X_base.loc[tr_idx, feature_cols_base].copy()
        y_tr = y.loc[tr_idx].copy()
        X_val = X_base.loc[val_idx, feature_cols_base].copy()
        y_val = y.loc[val_idx].copy()

        X_test_fold = X_test_base[feature_cols_base].copy()

        # 类别列保证是 str
        for c in categorical_features:
            if c in X_tr.columns:
                X_tr[c] = X_tr[c].astype(str)
            if c in X_val.columns:
                X_val[c] = X_val[c].astype(str)
            if c in X_test_fold.columns:
                X_test_fold[c] = X_test_fold[c].astype(str)

        # 关键：每折内重新做 TE（Leakage-safe）
        X_tr, X_val, X_test_fold = add_te_features_per_fold(
            X_tr, y_tr, X_val, X_test_fold,
            te_specs=te_specs,
            smoothing=20.0,
            inner_folds=5,
            seed=42,
        )

        # 训练与评估
        train_pool = Pool(X_tr, y_tr, cat_features=[c for c in categorical_features if c in X_tr.columns])
        val_pool = Pool(X_val, y_val, cat_features=[c for c in categorical_features if c in X_val.columns])

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True, plot=False)

        y_val_pred = model.predict_proba(val_pool)[:, 1]
        auc = roc_auc_score(y_val, y_val_pred)
        cv_scores.append(float(auc))
        print(f"第{fold+1}折 AUC: {auc:.4f}, best_iter={model.get_best_iteration()}")

        # test 预测（注意：必须用该折的 X_test_fold，包含该折的TE）
        test_pool = Pool(X_test_fold, cat_features=[c for c in categorical_features if c in X_test_fold.columns])
        pred = model.predict_proba(test_pool)[:, 1]
        test_preds.append(pred.astype(np.float32))

        # 保存模型
        model_path = os.path.join(out_dir, f"fold_{fold+1}.cbm")
        meta_path = os.path.join(out_dir, f"fold_{fold+1}.json")
        model.save_model(model_path)
        with open(meta_path, "w") as f:
            json.dump({
                "fold": fold + 1,
                "auc": float(auc),
                "best_iteration": int(model.get_best_iteration()),
                "params": model.get_all_params(),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)
        model_paths.append(model_path)

        # cleanup
        del model, train_pool, val_pool, test_pool, X_tr, y_tr, X_val, y_val, X_test_fold
        gc.collect()

    print("=" * 60)
    print("交叉验证结果汇总")
    print("=" * 60)
    print(f"各折AUC: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"平均AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # 集成 test 预测
    scores = np.array(cv_scores)
    weights = np.exp((scores - scores.min()) * 50)
    weights = weights / weights.sum()

    test_preds = np.vstack(test_preds)  # [n_folds, n_test]
    pred_weighted = np.average(test_preds, axis=0, weights=weights)
    pred_simple = np.mean(test_preds, axis=0)

    return model_paths, cv_scores, pred_weighted, pred_simple


def save_submission(pred: np.ndarray, pred_simple: np.ndarray, X_test: pd.DataFrame, model_paths, cv_scores):
    out_dir = get_output_dir()

    if "sample_index" in X_test.columns:
        sample_index = X_test["sample_index"].values
    else:
        sample_index = np.arange(len(pred))

    sub = pd.DataFrame({"sample_index": sample_index, "label": pred})
    path = os.path.join(out_dir, "submission_v5.csv")
    sub.to_csv(path, index=False)
    print(f"预测结果已保存到 {path}")

    sub2 = pd.DataFrame({"sample_index": sample_index, "label": pred_simple})
    path2 = os.path.join(out_dir, "submission_v5_simple.csv")
    sub2.to_csv(path2, index=False)
    print(f"简单平均预测结果已保存到 {path2}")

    best_idx = int(np.argmax(cv_scores))
    main_model_path = os.path.join(out_dir, "catboost_v5.cbm")
    safe_link_or_copy(model_paths[best_idx], main_model_path)

    with open(os.path.join(out_dir, "model_results_v5.json"), "w") as f:
        json.dump({
            "version": "V5",
            "auc_mean": float(np.mean(cv_scores)),
            "auc_std": float(np.std(cv_scores)),
            "fold_scores": [float(s) for s in cv_scores],
            "best_fold": best_idx + 1,
            "notes": [
                "Leakage-safe TE (outer-fold aware + inner OOF)",
                "Per-fold test TE then predict, then ensemble",
                "Use original test sample_index in submission",
            ],
        }, f, indent=2)


def main():
    np.random.seed(42)
    start = time.time()

    print("=" * 60)
    print("CatBoost V5 训练 (Leakage-safe)")
    print("=" * 60)

    train_df, test_df = load_data()
    X_base, y, X_test_base, categorical_features = create_base_features(train_df, test_df)

    del train_df, test_df
    gc.collect()

    model_paths, cv_scores, pred_weighted, pred_simple = train_v5(
        X_base, y, X_test_base, categorical_features, n_splits=10
    )

    save_submission(pred_weighted, pred_simple, X_test_base, model_paths, cv_scores)

    print(f"总耗时: {(time.time()-start)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
