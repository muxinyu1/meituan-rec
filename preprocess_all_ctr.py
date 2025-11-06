import pandas as pd
import numpy as np
import argparse
import json
import itertools
from typing import Any
import holidays  # <-- 新增：用于计算节假日


def create_time_slot(hour):
    """根据小时划分时间段"""
    if 6 <= hour <= 9:
        return 1
    elif 10 <= hour <= 15:
        return 2
    elif 16 <= hour <= 20:
        return 3
    elif 21 <= hour <= 23:
        return 4
    else:
        return 0


def create_temp_comfort(temp):
    """根据温度划分舒适度"""
    if pd.isna(temp):
        return 0
    if temp < 10:
        return 0
    elif 10 <= temp <= 25:
        return 1
    else:
        return 2


# ==========================================================
# === 新增：时间戳处理的辅助函数 ===
# ==========================================================
def create_season(month):
    """根据月份划分季节 (北半球)"""
    if month in [3, 4, 5]:
        return 0  # 春季
    elif month in [6, 7, 8]:
        return 1  # 夏季
    elif month in [9, 10, 11]:
        return 2  # 秋季
    elif month in [12, 1, 2]:
        return 3  # 冬季
    else:
        return 4  # 未知/NaN


def create_part_of_month(day):
    """根据日期划分上中下旬"""
    if day <= 10:
        return 0  # 上旬
    elif day <= 20:
        return 1  # 中旬
    else:
        return 2  # 下旬


# ==================================================================
# === 最终特征名定义 (已更新) ===
# ==================================================================
# (这部分与你原来的一致，保持不变)
# 1. 最终的离散特征名
USER_DISCRETE_COLS = [
    "age",
    "level",
    "gender",
    "married",
    "job",
    "has_car",
    "mobile_type",
    "mobile_os",
    "cross_age_gender_encoded",
    "cross_age_married_encoded",
    "cross_gender_married_encoded",
    "cross_age_gender_married_encoded",
    "cross_married_has_car_encoded",
    "cross_level_age_encoded",
    "cross_level_gender_encoded",
    "cross_level_job_encoded",
    "cross_mobile_os_age_encoded",
]
ITEM_DISCRETE_COLS = ["dtype", "cate_1", "cate_2", "cate_3"]
CONTEXT_DISCRETE_COLS = [
    "cityid",
    "loc_cityid",
    "weekday",
    "hour",
    "weather",
    "is_weekend",
    "is_night",
    "time_slot",
    "is_same_city",
    "temp_comfort",
    "cityid_bucket",
    "price_is_missing",
    "online_days_is_missing",
    "user_displayed_item_num_is_missing",
    # --- 新增的时间特征 ---
    "month",
    "day_of_month",
    "season",
    "part_of_month",
    "is_holiday",
]

# 2. 最终的连续特征名
USER_CONTINUOUS_COLS = ["online_days", "user_displayed_item_num"]
ITEM_CONTINUOUS_COLS = []
CONTEXT_CONTINUOUS_COLS = [
    "distance",
    "user_home_dis",
    "user_work_dis",
    "temp",
    "temp_low",
    "temp_high",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "temp_range",
    "item_ave_price",
    "price",
    "itemid_ctr",
    "cate_1_ctr",
    "cate_2_ctr",
    "cate_3_ctr",
    "age_ctr",
    "level_ctr",
    "gender_ctr",
    "age_cate_1_ctr",
    "age_cate_2_ctr",
    "level_cate_1_ctr",
    "level_cate_2_ctr",
    "gender_cate_1_ctr",
    "job_cate_1_ctr",
    "cityid_cate_1_ctr",
    "age_gender_cate_1_ctr",
    # --- 新增的时间CTR特征 ---
    "month_ctr",
    "season_ctr",
    "is_holiday_ctr",
    "season_cate_1_ctr",
    "month_cate_1_ctr",
    "is_holiday_cate_1_ctr",
    "gender_cate_2_ctr",
    "gender_cate_3_ctr",
    "age_gender_cate_2_ctr",
    "season_cate_2_ctr",
    "season_cate_3_ctr",
    "month_cate_2_ctr",
    "month_cate_3_ctr",
]

# 3. 计算CTR需要的基础特征（单特征 + 交叉特征）
BASE_CTRS_TO_CREATE = [
    "age",
    "level",
    "gender",
    "itemid",
    "cate_1",
    "cate_2",
    "cate_3",
    # --- 新增 ---
    "month",
    "season",
    "is_holiday",
]
CROSS_CTRS_TO_CREATE = [
    ["age", "cate_1"],
    ["age", "cate_2"],
    ["level", "cate_1"],
    ["level", "cate_2"],
    ["gender", "cate_1"],
    ["gender", "cate_2"],
    ["gender", "cate_3"],
    ["job", "cate_1"],
    ["cityid", "cate_1"],
    ["age", "gender", "cate_1"],  # 对应 'age_gender_cate_1_ctr'
    ["age", "gender", "cate_2"],
    # --- 新增 ---
    ["season", "cate_1"],
    ["season", "cate_2"],
    ["season", "cate_3"],
    ["month", "cate_1"],
    ["month", "cate_2"],
    ["month", "cate_3"],
    ["is_holiday", "cate_1"],
]


def process_data(
    train_df_raw,  # <-- MODIFIED: 传入DataFrame
    test_df_raw,  # <-- MODIFIED: 传入DataFrame
    user_df,  # <-- MODIFIED: 传入DataFrame
    item_df,  # <-- MODIFIED: 传入DataFrame
    output_train_csv_path,  # <-- MODIFIED: 训练集输出路径
    output_test_csv_path,  # <-- MODIFIED: 测试集输出路径
    output_json_path,
    cardinality_cap=512,
):
    """
    加载、合并、处理数据，并根据 *新规范* 生成全局CTR特征。
    """
    # ==========================================================
    # === 1. 合并训练集和测试集 ===
    # ==========================================================
    print("1. Combining train and test sets for consistent processing...")
    # 为测试集添加一个 假的/空的 label 列，以便合并
    test_df_raw["label"] = np.nan
    # 添加一个标志，以便我们稍后可以拆分它们
    train_df_raw["is_train"] = 1
    test_df_raw["is_train"] = 0

    # 合并
    merged_df = pd.concat([train_df_raw, test_df_raw], ignore_index=True)
    print(f"  - Combined shape (pre-merge): {merged_df.shape}")

    print("2. Merging with user and item data...")
    merged_df = pd.merge(merged_df, user_df, on="userid", how="left")
    merged_df = pd.merge(merged_df, item_df, on="itemid", how="left")
    print(f"  - Merged shape: {merged_df.shape}")

    print("3. Handling missing values (from merge)...")
    # (这部分逻辑不变)
    cols_to_fill_median = [
        "age",
        "level",
        "gender",
        "married",
        "job",
        "has_car",
        "dtype",
        "cate_1",
        "cate_2",
        "cate_3",
        "cross_age_gender_encoded",
        "cross_age_married_encoded",
        "cross_gender_married_encoded",
        "cross_age_gender_married_encoded",
        "cross_married_has_car_encoded",
        "cross_level_age_encoded",
        "cross_level_gender_encoded",
        "cross_level_job_encoded",
        "cross_mobile_os_age_encoded",
    ]
    for col in cols_to_fill_median:
        if col in merged_df.columns and merged_df[col].isnull().any():
            # <-- 关键: 在 *整个* 数据集上计算中位数，以保持一致
            median_val = merged_df[col].median()
            merged_df[col].fillna(median_val, inplace=True)

    print("3.5. Comprehensive NaN handling for continuous features...")
    # (这部分逻辑不变, 在合并的数据集上操作是正确的)
    continuous_cols_to_check = [
        "price",
        "online_days",
        "user_displayed_item_num",
        "distance",
        "item_ave_price",
        "user_home_dis",
        "user_work_dis",
        "temp",
        "temp_low",
        "temp_high",
        "weather",
    ]
    continuous_cols_to_check = [
        col for col in continuous_cols_to_check if col in merged_df.columns
    ]

    total_rows = len(merged_df)
    new_discrete_features_created = []

    for col in continuous_cols_to_check:
        if merged_df[col].isnull().any():
            missing_count = merged_df[col].isnull().sum()
            missing_rate = missing_count / total_rows

            if missing_rate < 0.10:
                median_val = merged_df[col].median()
                merged_df[col].fillna(median_val, inplace=True)
            else:
                missing_col_name = f"{col}_is_missing"
                if missing_col_name in CONTEXT_DISCRETE_COLS:  # 只创建我们需要的
                    merged_df[missing_col_name] = merged_df[col].isnull().astype(int)
                    new_discrete_features_created.append(missing_col_name)
                merged_df[col].fillna(-1, inplace=True)

    for col in [
        "price_is_missing",
        "online_days_is_missing",
        "user_displayed_item_num_is_missing",
    ]:
        if col in merged_df.columns and merged_df[col].isnull().any():
            merged_df[col].fillna(0, inplace=True)

    print("4. Engineering new base features...")

    # ==========================================================
    # === 4.1 新增：从 timestamp 提取丰富的日期特征 ===
    # ==========================================================
    print("  - 4.1 Extracting rich features from 'timestamp'...")
    # (这部分逻辑不变, 在合并的数据集上操作是正确的)
    if "timestamp" in merged_df.columns:
        merged_df["datetime"] = pd.to_datetime(merged_df["timestamp"], unit="ms")
        merged_df["month"] = merged_df["datetime"].dt.month
        merged_df["day_of_month"] = merged_df["datetime"].dt.day
        merged_df["season"] = merged_df["month"].apply(create_season)
        merged_df["part_of_month"] = merged_df["day_of_month"].apply(
            create_part_of_month
        )
        print("    - Calculating holiday features (CN)...")
        years_in_data = merged_df["datetime"].dt.year.unique()
        cn_holidays = holidays.country_holidays(country="CN", years=years_in_data)
        merged_df["is_holiday"] = (
            merged_df["datetime"].dt.date.isin(cn_holidays)
        ).astype(int)
        print(f"    - Found {merged_df['is_holiday'].sum()} holiday-day samples.")
        merged_df.drop(columns=["datetime"], inplace=True)
        print(
            "  - Timestamp features created: month, day_of_month, season, part_of_month, is_holiday"
        )
    else:
        print("  - [Warning] 'timestamp' column not found, skipping time extraction.")
    # ==========================================================
    # === 4.2 原始特征工程 ===
    # ==========================================================
    print("  - 4.2 Engineering original base features...")
    # (这部分逻辑不变, 在合并的数据集上操作是正确的)
    merged_df["is_weekend"] = merged_df["weekday"].apply(
        lambda x: 1 if x in [5, 6] else 0
    )
    merged_df["is_night"] = merged_df["hour"].apply(
        lambda x: 1 if (x >= 21 or x <= 5) else 0
    )
    merged_df["time_slot"] = merged_df["hour"].apply(create_time_slot)
    merged_df["hour_sin"] = np.sin(2 * np.pi * merged_df["hour"] / 24)
    merged_df["hour_cos"] = np.cos(2 * np.pi * merged_df["hour"] / 24)
    merged_df["weekday_sin"] = np.sin(2 * np.pi * merged_df["weekday"] / 7)
    merged_df["weekday_cos"] = np.cos(2 * np.pi * merged_df["weekday"] / 7)
    merged_df["is_same_city"] = (merged_df["cityid"] == merged_df["loc_cityid"]).astype(
        int
    )
    merged_df["temp_range"] = merged_df["temp_high"] - merged_df["temp_low"]
    merged_df["temp_comfort"] = merged_df["temp"].apply(create_temp_comfort)
    print("  - Base features created.")

    print(f"6. Discretizing features (Applying Cap={cardinality_cap})...")
    # (这部分逻辑不变, 在合并的数据集上操作是正确的)
    all_discrete_features_to_encode = list(
        set(
            USER_DISCRETE_COLS
            + ITEM_DISCRETE_COLS
            + CONTEXT_DISCRETE_COLS
            + list(itertools.chain(*CROSS_CTRS_TO_CREATE))  # 添加所有交叉特征的组件
        )
    )
    all_discrete_features_to_encode = [
        col
        for col in all_discrete_features_to_encode
        if col in merged_df.columns and col != "cityid_bucket"  # cityid_bucket 单独处理
    ]
    all_discrete_features_to_encode.extend(new_discrete_features_created)
    all_discrete_features_to_encode = sorted(list(set(all_discrete_features_to_encode)))

    feature_cardinality_map = {}  # 存储所有特征的 *最终* 基数

    for col in all_discrete_features_to_encode:
        if col not in merged_df.columns:
            print(f"  - [Warning] Column '{col}' not found for encoding. Skipping.")
            continue

        merged_df[col] = merged_df[col].astype(str)
        # <-- 关键: 在 *整个* 数据集上计算value_counts，以保持一致
        value_counts = merged_df[col].value_counts()
        current_cardinality = len(value_counts)

        if current_cardinality > cardinality_cap:
            print(
                f"  - Capping '{col}' (Raw: {current_cardinality} -> Cap: {cardinality_cap})"
            )
            top_values = value_counts.index[: cardinality_cap - 1]
            mapping = {val: i for i, val in enumerate(top_values)}
            other_index = cardinality_cap - 1
            merged_df[col] = merged_df[col].map(mapping).fillna(other_index).astype(int)
            final_cardinality = cardinality_cap
        else:
            unique_vals = value_counts.index
            mapping = {val: i for i, val in enumerate(unique_vals)}
            merged_df[col] = merged_df[col].map(mapping)
            final_cardinality = current_cardinality
            print(f"  - Encoded '{col}' (cardinality: {final_cardinality})")

        feature_cardinality_map[col] = final_cardinality

    print("6.5. Creating 'cityid_bucket' (Special Cap=101)...")
    # (这部分逻辑不变)
    cap = 101
    value_counts = merged_df["cityid"].value_counts()
    top_values = value_counts.index[: cap - 1]
    mapping = {val: i for i, val in enumerate(top_values)}
    other_index = cap - 1
    merged_df["cityid_bucket"] = (
        merged_df["cityid"].map(mapping).fillna(other_index).astype(int)
    )
    final_cardinality = merged_df["cityid_bucket"].nunique()
    feature_cardinality_map["cityid_bucket"] = int(final_cardinality)
    print(f"  - Created 'cityid_bucket' with actual cardinality {final_cardinality}")

    # ==========================================================
    # === 7. MODIFIED: CTR 计算 ===
    # ==========================================================
    print("7. Calculating GLOBAL CTR features (from TRAIN data only)...")

    # --- 关键修改 ---
    # 创建一个 *只包含训练数据* 的视图，用于计算所有CTR值
    train_only_df = merged_df[merged_df["is_train"] == 1]
    
    # 全局CTR *只* 从训练集计算
    global_mean = train_only_df["label"].mean()
    print(f"  - Global mean CTR (from train): {global_mean:.4f}")

    # --- 7.1 单特征CTR ---
    print("  - Processing single-feature CTRs...")
    for col in BASE_CTRS_TO_CREATE:
        if col not in merged_df.columns:
            print(f"    - [Warning] Base CTR col '{col}' not found. Skipping.")
            continue
        print(f"    - Calculating global CTR for: '{col}'")
        
        # <-- 关键: *只* 用 train_only_df 计算 map
        global_ctr_map = train_only_df.groupby(col)["label"].mean()
        
        # <-- 关键: 将 map 应用于 *整个* merged_df
        merged_df[f"{col}_ctr"] = merged_df[col].map(global_ctr_map)
        # 用训练集的全局均值填充 (包括测试集中的新值)
        merged_df[f"{col}_ctr"].fillna(global_mean, inplace=True)

    # --- 7.2 交叉特征CTR ---
    print("  - Processing cross-feature CTRs...")
    for col_list in CROSS_CTRS_TO_CREATE:
        if not all(c in merged_df.columns for c in col_list):
            print(f"    - Skipping {col_list}, component missing.")
            continue
        cross_col_name = "_".join(col_list)
        ctr_col_name = f"{cross_col_name}_ctr"

        if (
            ctr_col_name not in CONTEXT_CONTINUOUS_COLS
            and ctr_col_name not in ITEM_CONTINUOUS_COLS
            and ctr_col_name not in USER_CONTINUOUS_COLS
        ):
            print(
                f"    - [Config Skip] '{ctr_col_name}' not in final feature lists. Skipping calculation."
            )
            continue

        print(f"    - Calculating global CTR for: '{cross_col_name}'")
        
        # <-- 关键: *只* 用 train_only_df 计算 map
        global_ctr_map = train_only_df.groupby(col_list)["label"].mean()
        global_ctr_df = global_ctr_map.reset_index()
        global_ctr_df = global_ctr_df.rename(columns={"label": ctr_col_name})
        
        # <-- 关键: 将 map 应用于 *整个* merged_df
        merged_df = pd.merge(merged_df, global_ctr_df, on=col_list, how="left")
        merged_df[ctr_col_name].fillna(global_mean, inplace=True)

    print("7.5. Dropping geo-hash features...")
    # (这部分逻辑不变)
    geo_cols_to_drop = ["geohash", "work_geohash", "geohash_prefix_6"]
    existing_geo_cols = [col for col in geo_cols_to_drop if col in merged_df.columns]
    if existing_geo_cols:
        merged_df.drop(columns=existing_geo_cols, inplace=True)
        print(f"  - Dropped columns: {existing_geo_cols}")

    # ==========================================================
    # === 8. MODIFIED: 拆分并保存 ===
    # ==========================================================
    print("8. Finalizing feature selection and splitting...")
    ALL_COLS_TO_KEEP = (
        ["label", "global_id", "timestamp"]  # 元数据
        + USER_DISCRETE_COLS
        + ITEM_DISCRETE_COLS
        + CONTEXT_DISCRETE_COLS
        + USER_CONTINUOUS_COLS
        + ITEM_CONTINUOUS_COLS
        + CONTEXT_CONTINUOUS_COLS
    )

    ALL_COLS_TO_KEEP = sorted(list(set(ALL_COLS_TO_KEEP)))

    missing_cols = [col for col in ALL_COLS_TO_KEEP if col not in merged_df.columns]
    if missing_cols:
        print(
            f"\n[ERROR] The script failed to create the following required columns: {missing_cols}"
        )

    final_cols_in_df = [col for col in ALL_COLS_TO_KEEP if col in merged_df.columns]
    
    # 包含 'is_train', 'label' 和 'sample_index' 以便拆分和提交
    utility_cols_to_keep = ['is_train', 'label', 'sample_index']
    
    # 过滤掉数据中可能不存在的工具列 (虽然它们应该都在)
    utility_cols_to_keep = [col for col in utility_cols_to_keep if col in merged_df.columns]
    
    # 使用 dict.fromkeys 来去重并保持顺序
    final_cols_to_select = list(dict.fromkeys(final_cols_in_df + utility_cols_to_keep))
    
    final_output_df = merged_df[final_cols_to_select]

    print(f"  - Final selected feature count: {len(final_cols_in_df)}")

    # --- 拆分数据 ---
    print("  - Splitting back into train and test...")
    # (更安全的方式) 丢弃 'is_train' 和 'sample_index'
    cols_to_drop_train = [col for col in ["is_train", "sample_index"] if col in final_output_df.columns]
    output_train_df = final_output_df[
        final_output_df["is_train"] == 1
    ].drop(columns=cols_to_drop_train)
    
    output_test_df = final_output_df[
        final_output_df["is_train"] == 0
    ].drop(columns=["is_train", "label"]) # 测试集不需要 'label' 列

    print(f"  - Final Train shape: {output_train_df.shape}")
    print(f"  - Final Test shape: {output_test_df.shape}")

    # --- 保存文件 ---
    output_train_df.to_csv(output_train_csv_path, index=False)
    print(f"Successfully saved processed TRAIN data to '{output_train_csv_path}'")
    
    output_test_df.to_csv(output_test_csv_path, index=False)
    print(f"Successfully saved processed TEST data to '{output_test_csv_path}'")


    # -----------------------------------------------------------------
    # --- 9. 动态生成JSON元数据 ---
    # -----------------------------------------------------------------
    print(f"9. Dynamically generating feature metadata JSON...")
    # (这部分逻辑不变, 它使用从合并数据中获得的正确的 cardinality_map)
    output_json_data: Any = {
        "user": {"discrete": {}, "continuous": []},
        "item": {"discrete": {}, "continuous": []},
        "context": {"discrete": {}, "continuous": []},
    }

    final_columns = output_train_df.columns.tolist() # 使用训练列作为参考

    for col in final_columns:
        if col in ["label", "global_id", "timestamp"]:
            continue

        if col in USER_DISCRETE_COLS:
            output_json_data["user"]["discrete"][col] = int(
                feature_cardinality_map[col]
            )
        elif col in ITEM_DISCRETE_COLS:
            output_json_data["item"]["discrete"][col] = int(
                feature_cardinality_map[col]
            )
        elif col in CONTEXT_DISCRETE_COLS:
            output_json_data["context"]["discrete"][col] = int(
                feature_cardinality_map[col]
            )
        elif col in USER_CONTINUOUS_COLS:
            output_json_data["user"]["continuous"].append(col)
        elif col in ITEM_CONTINUOUS_COLS:
            output_json_data["item"]["continuous"].append(col)
        elif col in CONTEXT_CONTINUOUS_COLS:
            output_json_data["context"]["continuous"].append(col)
        else:
            print(
                f"  - [Metadata Warning] Uncategorized col: '{col}'. Not included in JSON."
            )

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_json_data, f, indent=4)

    print(f"Successfully saved dynamic feature metadata to '{output_json_path}'")


# ==========================================================
# === MODIFIED: Main 执行入口 ===
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load raw data, merge, process, and generate GLOBAL CTR features based on spec.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # --- 原始参数 ---
    parser.add_argument(
        "--train_csv",
        type=str,
        default="data/recsys_task_data/train_samples-20221014.csv",
        help="Path to the RAW training samples CSV file (with timestamp).",
    )
    parser.add_argument(
        "--user_csv",
        type=str,
        default="data/recsys_task_data/user_infos_crossed_features.csv",
        help="Path to the CROSSED user information CSV file.",
    )
    parser.add_argument(
        "--item_csv",
        type=str,
        default="data/recsys_task_data/item_infos-20221014.csv",
        help="Path to the item information CSV file.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="data/processed/train.json",
        help="Path for the output companion JSON file (user-specified format).",
    )
    parser.add_argument(
        "--cardinality_cap",
        type=int,
        default=4096,
        help="Maximum cardinality for discrete features. Rarer values will be grouped. (Default: 4096)",
    )
    
    # --- 新增/修改的参数 ---
    parser.add_argument(
        "--test_csv",
        type=str,
        default="data/recsys_task_data/test_samples-20221019.csv", 
        help="Path to the RAW test samples CSV file (without label).",
    )
    parser.add_argument(
        "--output_train_csv",
        type=str,
        default="data/processed/train.csv", # <-- MODIFIED
        help="Path for the final output PROCESSED TRAIN CSV file.",
    )
    parser.add_argument(
        "--output_test_csv",
        type=str,
        default="data/processed/test.csv", # <-- NEW
        help="Path for the final output PROCESSED TEST CSV file.",
    )

    args = parser.parse_args()

    # --- 新的执行逻辑 ---
    print("1. Loading ALL raw data files...")
    try:
        train_df_raw = pd.read_csv(args.train_csv)
        print(f"  - Loaded train data: {train_df_raw.shape}")
        
        test_df_raw = pd.read_csv(args.test_csv)
        print(f"  - Loaded test data: {test_df_raw.shape}")

        user_df = pd.read_csv(args.user_csv)
        print(f"  - Loaded user data: {user_df.shape}")
        
        item_df = pd.read_csv(args.item_csv)
        print(f"  - Loaded item data: {item_df.shape}")
        
    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        exit(1) # 退出

    print("\n2. Starting unified data processing...")
    process_data(
        train_df_raw,
        test_df_raw,
        user_df,
        item_df,
        args.output_train_csv,
        args.output_test_csv,
        args.output_json,
        cardinality_cap=args.cardinality_cap,
    )
    print("\nProcessing complete for both train and test sets.")