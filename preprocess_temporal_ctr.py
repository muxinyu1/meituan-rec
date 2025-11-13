import pandas as pd
import numpy as np
import argparse
import json
import itertools
from typing import Any
import holidays
from tqdm import tqdm


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
# === 特征名定义 (已剔除0重要性特征) ===
# ==================================================================
USER_DISCRETE_COLS = [
    "job",
    "mobile_type",
    "mobile_os",
    "cross_age_gender_encoded",
    "cross_age_married_encoded",
    # "cross_gender_married_encoded",  # Removed (Importance 0)
    "cross_age_gender_married_encoded",
    # "cross_married_has_car_encoded", # Removed (Importance 0)
    "cross_level_age_encoded",
    "cross_level_gender_encoded",
    "cross_level_job_encoded",
    "cross_mobile_os_age_encoded",
    # "age",                           # Removed (Importance 0)
    # "level",                         # Removed (Importance 0)
    # "gender",                        # Removed (Importance 0)
    # "married",                       # Removed (Importance 0)
    # "has_car",                       # Removed (Importance 0)
]
ITEM_DISCRETE_COLS = ["dtype", "cate_1", "cate_2", "cate_3"]  # 无变化

CONTEXT_DISCRETE_COLS = [
    "cityid",
    "loc_cityid",
    "weekday",
    "hour",
    "weather",
    # "is_weekend",                    # Removed (Importance 0)
    # "is_night",                      # Removed (Importance 0)
    "time_slot",
    "is_same_city",
    # "temp_comfort",                  # Removed (Importance 0)
    "cityid_bucket",
    # "price_is_missing",              # Removed (Importance 0)
    # "online_days_is_missing",        # Removed (Importance 0)
    "user_displayed_item_num_is_missing",
    # "month",                         # Removed (Importance 0)
    "day_of_month",
    # "season",                        # Removed (Importance 0)
    # "part_of_month",                 # Removed (Importance 0)
    # "is_holiday",                    # Removed (Importance 0)
]

USER_CONTINUOUS_COLS = ["online_days", "user_displayed_item_num"] # 无变化

ITEM_CONTINUOUS_COLS = [] # 无变化

CONTEXT_CONTINUOUS_COLS = [
    "distance",
    "user_home_dis",
    "user_work_dis",
    # "temp",                          # Removed (Importance 0)
    # "temp_low",                      # Removed (Importance 0)
    # "temp_high",                     # Removed (Importance 0)
    "hour_sin",
    "hour_cos",
    # "weekday_sin",                   # Removed (Importance 0)
    # "weekday_cos",                   # Removed (Importance 0)
    # "temp_range",                    # Removed (Importance 0)
    "item_ave_price",
    "price",
    "itemid_ctr",
    "itemid_click_count_60d",
    # "cate_1_ctr",                    # Removed (Importance 0)
    "cate_2_ctr",
    "cate_3_ctr",
    # "age_ctr",                       # Removed (Importance 0)
    "level_ctr",
    # "gender_ctr",                    # Removed (Importance 0)
    # "age_cate_1_ctr",                # Removed (Importance 0)
    "age_cate_2_ctr",
    "age_cate_3_ctr",
    "level_cate_1_ctr",
    "level_cate_2_ctr",
    # "level_cate_3_ctr",              # Removed (Importance 0)
    # "gender_cate_1_ctr",             # Removed (Importance 0)
    "gender_cate_2_ctr",
    "gender_cate_3_ctr",
    # "job_cate_1_ctr",                # Removed (Importance 0)
    "job_cate_2_ctr",
    # "job_cate_3_ctr",                # Removed (Importance 0)
    # "cityid_cate_1_ctr",             # Removed (Importance 0)
    "age_gender_cate_1_ctr",
    "age_gender_cate_2_ctr",
    "age_gender_cate_3_ctr",
    # "month_ctr",                     # Removed (Importance 0)
    # "season_ctr",                    # Removed (Importance 0)
    # "is_holiday_ctr",                # Removed (Importance 0)
    # "season_cate_1_ctr",             # Removed (Importance 0)
    # "month_cate_1_ctr",              # Removed (Importance 0)
    # "is_holiday_cate_1_ctr",         # Removed (Importance 0)
    # "season_cate_2_ctr",             # Removed (Importance 0)
    # "season_cate_3_ctr",             # Removed (Importance 0)
    "month_cate_2_ctr",
    # "month_cate_3_ctr",              # Removed (Importance 0)
    
    # --- 上一轮补全的特征 (均有重要性) ---
    "cityid_cate_2_ctr",
    "cityid_cate_3_ctr",
    "age_gender_married_cate_1_ctr",
    "age_gender_married_cate_2_ctr",
    "age_gender_married_cate_3_ctr",
    "mobile_type_age_cate_1_ctr",
    "mobile_type_age_cate_2_ctr",
    "mobile_type_age_cate_3_ctr",
    "cityid_age_cate_1_ctr",
    "cityid_age_cate_2_ctr",
    "cityid_age_cate_3_ctr",
    "cityid_level_cate_2_ctr",
    "mobile_type_age_gender_cate_1_ctr",
    "mobile_type_age_gender_cate_2_ctr",
    "mobile_type_age_gender_cate_3_ctr",

    "cate_2_mobile_type_ctr",
    "cityid_mobile_type_ctr",
    "cate_2_mobile_os_ctr",
    "level_mobile_type_ctr"
]

# ==================================================================
# === CTR特征生成定义 (已剔除0重要性特征) ===
# ==================================================================

# 只对 itemid 使用时序CTR（避免信息泄漏）
TEMPORAL_CTRS = ["itemid"] # 无变化

# 其他特征使用全局CTR（基于训练集统计，具有群体普适性）
GLOBAL_BASE_CTRS = [
    # "age",         # Removed (Importance 0)
    # "level",       # Removed (Importance 0)
    # "gender",      # Removed (Importance 0)
    "cate_1",
    "cate_2",
    "cate_3",
    # "month",       # Removed (Importance 0)
    # "season",      # Removed (Importance 0)
    # "is_holiday",  # Removed (Importance 0)
]

GLOBAL_CROSS_CTRS = [
    # ["age", "cate_1"],     # Removed (age_cate_1_ctr)
    ["age", "cate_2"],
    ["age", "cate_3"],
    ["level", "cate_1"],
    ["level", "cate_2"],
    # ["level", "cate_3"],   # Removed (level_cate_3_ctr)
    # ["gender", "cate_1"],  # Removed (gender_cate_1_ctr)
    ["gender", "cate_2"],
    ["gender", "cate_3"],
    # ["job", "cate_1"],     # Removed (job_cate_1_ctr)
    ["job", "cate_2"],
    # ["job", "cate_3"],     # Removed (job_cate_3_ctr)
    # ["cityid", "cate_1"],  # Removed (cityid_cate_1_ctr)
    ["cityid", "cate_2"],
    ["cityid", "cate_3"],
    ["age", "gender", "cate_1"],
    ["age", "gender", "cate_2"],
    ["age", "gender", "cate_3"],
    ["age", "gender", "married", "cate_1"],
    ["age", "gender", "married", "cate_2"],
    ["age", "gender", "married", "cate_3"],
    # ["season", "cate_1"],  # Removed (season_cate_1_ctr)
    # ["season", "cate_2"],  # Removed (season_cate_2_ctr)
    # ["season", "cate_3"],  # Removed (season_cate_3_ctr)
    # ["month", "cate_1"],   # Removed (month_cate_1_ctr)
    ["month", "cate_2"],
    # ["month", "cate_3"],   # Removed (month_cate_3_ctr)
    # ["is_holiday", "cate_1"], # Removed (is_holiday_cate_1_ctr)
    
    ["mobile_type", "age", "cate_1"],
    ["mobile_type", "age", "cate_2"],
    ["mobile_type", "age", "cate_3"],

    ["cityid", "age", "cate_1"],
    ["cityid", "age", "cate_2"],
    ["cityid", "age", "cate_3"],
    ["cityid", "level", "cate_2"],
    ["mobile_type", "age", "gender", "cate_1"],
    ["mobile_type", "age", "gender", "cate_2"],
    ["mobile_type", "age", "gender", "cate_3"],

    ["mobile_type", "cate_2"],
    ["cityid", "mobile_type"],
    ["cate_2", "mobile_os"],
    ["level", "mobile_type"]
]


def calculate_item_click_count_temporal(
    df: pd.DataFrame,
    window_days: int = 60,
) -> pd.DataFrame:
    """
    按时间顺序计算每个物品在最近N天内的点击次数
    
    这个特征反映物品的热度趋势，需要时序计算以避免信息泄漏
    
    参数:
    - df: 已按timestamp排序的DataFrame，必须包含'itemid', 'timestamp', 'label', 'is_train'列
    - window_days: 时间窗口（天数），默认60天
    
    返回:
    - 添加了'itemid_click_count_Nd'特征的DataFrame
    """
    print(f"\n=== 开始计算物品最近{window_days}天点击次数（时序方法）===")
    print(f"  - 数据总量: {len(df)}")
    print(f"  - 时间窗口: {window_days} 天")
    
    # 确保数据已排序
    if not df['timestamp'].is_monotonic_increasing:
        print("  - 警告: 数据未按timestamp排序，正在排序...")
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 转换为numpy数组
    itemids = df['itemid'].values
    timestamps = df['timestamp'].values
    is_train = df['is_train'].values
    labels = df['label'].values
    n_samples = len(df)
    
    # 时间窗口（毫秒）
    window_ms = window_days * 24 * 60 * 60 * 1000
    
    # 存储每个物品的点击历史 (timestamp, is_click)
    from collections import defaultdict, deque
    item_click_history = defaultdict(deque)
    
    click_counts = np.zeros(n_samples, dtype=np.int32)
    
    print(f"  - 逐行计算（使用滑动窗口）...")
    for i in tqdm(range(n_samples), desc="    Progress", ncols=80):
        current_itemid = itemids[i]
        current_ts = timestamps[i]
        
        # 清理窗口外的历史记录
        history = item_click_history[current_itemid]
        while history and (current_ts - history[0][0]) > window_ms:
            history.popleft()
        
        # 统计窗口内的点击次数
        click_counts[i] = sum(1 for _, is_click in history if is_click == 1)
        
        # 只有训练集样本才更新历史
        if is_train[i] == 1 and not np.isnan(labels[i]):
            history.append((current_ts, int(labels[i])))
    
    col_name = f"itemid_click_count_{window_days}d"
    df[col_name] = click_counts
    
    print(f"  ✓ 完成！特征名: '{col_name}'")
    print(f"    - 统计信息: mean={click_counts.mean():.2f}, max={click_counts.max()}, std={click_counts.std():.2f}")
    
    return df


def calculate_global_ctr_features(
    df: pd.DataFrame,
    base_features: list,
    cross_features: list,
    smoothing_factor: float = 100.0,
    global_prior: float = 0.1,
) -> pd.DataFrame:
    """
    基于训练集计算全局CTR特征（快速向量化）
    
    这些特征反映的是群体统计规律，不存在信息泄漏问题：
    - gender_ctr: 性别对整体的点击偏好
    - age_cate_1_ctr: 年龄段对某类目的偏好
    等等...
    
    参数:
    - df: DataFrame，必须包含'label'和'is_train'列
    - base_features: 单特征CTR列表
    - cross_features: 交叉特征CTR列表
    - smoothing_factor: 平滑因子
    - global_prior: 全局先验CTR
    
    返回:
    - 添加了全局CTR特征的DataFrame
    """
    print("\n=== 开始计算全局CTR特征（基于训练集统计）===")
    print(f"  - 策略: 群体统计特征不存在时序泄漏问题")
    print(f"  - 平滑因子: {smoothing_factor}")
    print(f"  - 全局先验CTR: {global_prior:.4f}")
    
    # 只用训练集计算统计
    train_df = df[df['is_train'] == 1].copy()
    print(f"  - 使用 {len(train_df)} 条训练样本计算全局统计")
    
    # === 计算单特征全局CTR ===
    print("\n  计算单特征全局CTR:")
    for feature in base_features:
        if feature not in df.columns:
            print(f"    - 跳过 '{feature}' (列不存在)")
            continue
        
        ctr_col_name = f"{feature}_ctr"
        print(f"    - 计算 '{ctr_col_name}'", end=" ", flush=True)
        
        # 在训练集上计算每个特征值的点击率
        feature_stats = train_df.groupby(feature)['label'].agg(['sum', 'count']).reset_index()
        feature_stats.columns = [feature, 'clicks', 'shows']
        
        # 贝叶斯平滑
        feature_stats[ctr_col_name] = (
            (feature_stats['clicks'] + global_prior * smoothing_factor) / 
            (feature_stats['shows'] + smoothing_factor)
        )
        
        # 创建映射字典
        ctr_map = dict(zip(feature_stats[feature], feature_stats[ctr_col_name]))
        
        # 应用到全部数据（训练集+测试集）
        df[ctr_col_name] = df[feature].map(ctr_map).fillna(global_prior)
        
        print(f"✓ (映射了 {len(ctr_map)} 个唯一值)")
    
    # === 计算交叉特征全局CTR ===
    print("\n  计算交叉特征全局CTR:")
    for feature_list in cross_features:
        if not all(f in df.columns for f in feature_list):
            print(f"    - 跳过 {feature_list} (组件缺失)")
            continue
        
        feature_key = "_".join(feature_list)
        ctr_col_name = f"{feature_key}_ctr"
        
        # 检查是否在最终特征列表中
        if (ctr_col_name not in CONTEXT_CONTINUOUS_COLS 
            and ctr_col_name not in ITEM_CONTINUOUS_COLS 
            and ctr_col_name not in USER_CONTINUOUS_COLS):
            print(f"    - 跳过 '{ctr_col_name}' (不在最终特征列表)")
            continue
        
        print(f"    - 计算 '{ctr_col_name}'", end=" ", flush=True)
        
        # 在训练集上计算交叉特征的点击率
        feature_stats = train_df.groupby(feature_list)['label'].agg(['sum', 'count']).reset_index()
        feature_stats.columns = feature_list + ['clicks', 'shows']
        
        # 贝叶斯平滑
        feature_stats[ctr_col_name] = (
            (feature_stats['clicks'] + global_prior * smoothing_factor) / 
            (feature_stats['shows'] + smoothing_factor)
        )
        
        # 创建映射（用于多列组合）
        train_df_temp = train_df[feature_list].copy()
        train_df_temp[ctr_col_name] = 0.0
        train_df_temp = train_df_temp.merge(
            feature_stats[feature_list + [ctr_col_name]], 
            on=feature_list, 
            how='left',
            suffixes=('', '_map')
        )
        
        # 应用到全部数据
        df_temp = df[feature_list].copy()
        df_temp = df_temp.merge(
            feature_stats[feature_list + [ctr_col_name]], 
            on=feature_list, 
            how='left'
        )
        df[ctr_col_name] = df_temp[ctr_col_name].fillna(global_prior)
        
        print(f"✓ (映射了 {len(feature_stats)} 个组合)")
    
    print("\n=== 全局CTR特征计算完成 ===")
    return df


def calculate_temporal_ctr_features(
    df: pd.DataFrame,
    base_features: list,
    cross_features: list,
    smoothing_factor: float = 100.0,
    global_prior: float = 0.1,
) -> pd.DataFrame:
    """
    按时间顺序计算CTR特征，避免信息泄漏
    使用向量化操作加速计算
    
    参数:
    - df: 已按timestamp排序的DataFrame，必须包含'label'和'is_train'列
    - base_features: 单特征CTR列表，如['age', 'gender', 'itemid']
    - cross_features: 交叉特征CTR列表，如[['age', 'cate_1'], ['gender', 'cate_2']]
    - smoothing_factor: 平滑因子，用于贝叶斯平滑
    - global_prior: 全局先验CTR，用于冷启动
    
    返回:
    - 添加了CTR特征的DataFrame
    """
    print("\n=== 开始计算时序CTR特征（仅用于itemid）===")
    print(f"  - 数据总量: {len(df)}")
    print(f"  - 训练集样本: {(df['is_train'] == 1).sum()}")
    print(f"  - 测试集样本: {(df['is_train'] == 0).sum()}")
    print(f"  - 平滑因子: {smoothing_factor}")
    print(f"  - 全局先验CTR: {global_prior}")
    
    # 确保数据已排序
    if not df['timestamp'].is_monotonic_increasing:
        print("  - 警告: 数据未按timestamp排序，正在排序...")
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 转换为numpy数组以加速
    is_train = df['is_train'].values
    labels = df['label'].values
    n_samples = len(df)
    
    # === 处理单特征CTR ===
    print("\n  处理单特征CTR (时序方法):")
    for feature in base_features:
        if feature not in df.columns:
            print(f"    - 跳过 '{feature}' (列不存在)")
            continue
        
        ctr_col_name = f"{feature}_ctr"
        print(f"    - 计算 '{ctr_col_name}'", flush=True)
        
        feature_values = df[feature].values
        ctr_values = np.zeros(n_samples, dtype=np.float64)
        
        # 使用字典存储累积统计
        click_count = {}
        show_count = {}
        
        # 向量化计算
        for i in tqdm(range(n_samples), desc=f"      {feature}", ncols=80, leave=False):
            fval = feature_values[i]
            
            # 获取历史统计（使用get避免KeyError）
            clicks = click_count.get(fval, 0)
            shows = show_count.get(fval, 0)
            
            # 贝叶斯平滑CTR
            ctr_values[i] = (clicks + global_prior * smoothing_factor) / (shows + smoothing_factor)
            
            # 只有训练集样本才更新统计
            if is_train[i] == 1 and not np.isnan(labels[i]):
                show_count[fval] = shows + 1
                click_count[fval] = clicks + int(labels[i])
        
        df[ctr_col_name] = ctr_values
    
    # === 处理交叉特征CTR ===
    print("\n  处理交叉特征CTR (时序方法):")
    for feature_list in cross_features:
        if not all(f in df.columns for f in feature_list):
            print(f"    - 跳过 {feature_list} (组件缺失)")
            continue
        
        feature_key = "_".join(feature_list)
        ctr_col_name = f"{feature_key}_ctr"
        
        # 检查是否在最终特征列表中
        if (ctr_col_name not in CONTEXT_CONTINUOUS_COLS 
            and ctr_col_name not in ITEM_CONTINUOUS_COLS 
            and ctr_col_name not in USER_CONTINUOUS_COLS):
            print(f"    - 跳过 '{ctr_col_name}' (不在最终特征列表)")
            continue
        
        print(f"    - 计算 '{ctr_col_name}'", flush=True)
        
        # 提取所有特征列的值
        feature_arrays = [df[f].values for f in feature_list]
        ctr_values = np.zeros(n_samples, dtype=np.float64)
        
        # 使用字典存储累积统计
        click_count = {}
        show_count = {}
        
        # 向量化计算
        for i in tqdm(range(n_samples), desc=f"      {feature_key}", ncols=80, leave=False):
            # 构建组合键
            fval = tuple(arr[i] for arr in feature_arrays)
            
            # 获取历史统计
            clicks = click_count.get(fval, 0)
            shows = show_count.get(fval, 0)
            
            # 贝叶斯平滑CTR
            ctr_values[i] = (clicks + global_prior * smoothing_factor) / (shows + smoothing_factor)
            
            # 只有训练集样本才更新统计
            if is_train[i] == 1 and not np.isnan(labels[i]):
                show_count[fval] = shows + 1
                click_count[fval] = clicks + int(labels[i])
        
        df[ctr_col_name] = ctr_values
    
    print("\n=== 时序CTR特征计算完成 ===")
    return df


def process_data(
    train_df_raw,
    test_df_raw,
    user_df,
    item_df,
    output_train_csv_path,
    output_test_csv_path,
    output_json_path,
    cardinality_cap=512,
    smoothing_factor=100.0,
    global_prior=0.1,
):
    """
    加载、合并、处理数据，并按时间顺序生成CTR特征（无信息泄漏）
    """
    # ==========================================================
    # === 1. 合并训练集和测试集 ===
    # ==========================================================
    print("1. Combining train and test sets for consistent processing...")
    test_df_raw["label"] = np.nan
    train_df_raw["is_train"] = 1
    test_df_raw["is_train"] = 0

    merged_df = pd.concat([train_df_raw, test_df_raw], ignore_index=True)
    print(f"  - Combined shape (pre-merge): {merged_df.shape}")

    print("2. Merging with user and item data...")
    merged_df = pd.merge(merged_df, user_df, on="userid", how="left")
    merged_df = pd.merge(merged_df, item_df, on="itemid", how="left")
    print(f"  - Merged shape: {merged_df.shape}")

    print("3. Handling missing values (from merge)...")
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
            median_val = merged_df[col].median()
            merged_df[col].fillna(median_val, inplace=True)

    print("3.5. Comprehensive NaN handling for continuous features...")
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
                if missing_col_name in CONTEXT_DISCRETE_COLS:
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
    # === 4.1 从 timestamp 提取时间特征 ===
    # ==========================================================
    print("  - 4.1 Extracting rich features from 'timestamp'...")
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
    all_discrete_features_to_encode = list(
        set(
            USER_DISCRETE_COLS
            + ITEM_DISCRETE_COLS
            + CONTEXT_DISCRETE_COLS
            + list(itertools.chain(*GLOBAL_CROSS_CTRS))
        )
    )
    all_discrete_features_to_encode = [
        col
        for col in all_discrete_features_to_encode
        if col in merged_df.columns and col != "cityid_bucket"
    ]
    all_discrete_features_to_encode.extend(new_discrete_features_created)
    all_discrete_features_to_encode = sorted(list(set(all_discrete_features_to_encode)))

    feature_cardinality_map = {}

    for col in all_discrete_features_to_encode:
        if col not in merged_df.columns:
            print(f"  - [Warning] Column '{col}' not found for encoding. Skipping.")
            continue

        merged_df[col] = merged_df[col].astype(str)
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
    # === 7. 计算CTR特征（混合策略）===
    # ==========================================================
    print("\n7. Calculating CTR features (Hybrid Strategy)...")
    print("  - Strategy:")
    print("    * TEMPORAL CTR (time-sequential): itemid only")
    print("    * GLOBAL CTR (train-set statistics): all other features")
    print("    * TEMPORAL COUNT: itemid click count in recent 60 days")
    print("  - Rationale: User/category CTRs are population-level patterns, not subject to leakage")
    
    # 计算全局先验CTR（仅用于冷启动）
    train_only_df = merged_df[merged_df["is_train"] == 1]
    computed_global_prior = train_only_df["label"].mean()
    print(f"\n  - Computed global prior CTR from train set: {computed_global_prior:.4f}")
    
    # 先计算全局CTR特征（快速）
    print("\n  [Step 1/3] Calculating GLOBAL CTR features...")
    merged_df = calculate_global_ctr_features(
        merged_df,
        base_features=GLOBAL_BASE_CTRS,
        cross_features=GLOBAL_CROSS_CTRS,
        smoothing_factor=smoothing_factor,
        global_prior=computed_global_prior,
    )
    
    # 再计算时序CTR特征（仅itemid，需要排序）
    print("\n  [Step 2/3] Calculating TEMPORAL CTR features (itemid only)...")
    print("    - Sorting by timestamp to prevent information leakage...")
    merged_df = merged_df.sort_values("timestamp").reset_index(drop=True)
    print(f"    - Data sorted. Shape: {merged_df.shape}")
    
    merged_df = calculate_temporal_ctr_features(
        merged_df,
        base_features=TEMPORAL_CTRS,
        cross_features=[],  # itemid没有交叉特征
        smoothing_factor=smoothing_factor,
        global_prior=computed_global_prior,
    )
    
    # 计算物品最近60天点击次数（时序特征）
    print("\n  [Step 3/3] Calculating item click count in recent 60 days...")
    merged_df = calculate_item_click_count_temporal(
        merged_df,
        window_days=60,
    )

    print("7.5. Dropping geo-hash features...")
    geo_cols_to_drop = ["geohash", "work_geohash", "geohash_prefix_6"]
    existing_geo_cols = [col for col in geo_cols_to_drop if col in merged_df.columns]
    if existing_geo_cols:
        merged_df.drop(columns=existing_geo_cols, inplace=True)
        print(f"  - Dropped columns: {existing_geo_cols}")

    # ==========================================================
    # === 8. 拆分并保存 ===
    # ==========================================================
    print("8. Finalizing feature selection and splitting...")
    ALL_COLS_TO_KEEP = (
        ["label", "global_id", "timestamp"]
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
    
    utility_cols_to_keep = ['is_train', 'label', 'sample_index']
    utility_cols_to_keep = [col for col in utility_cols_to_keep if col in merged_df.columns]
    
    final_cols_to_select = list(dict.fromkeys(final_cols_in_df + utility_cols_to_keep))
    
    final_output_df = merged_df[final_cols_to_select]

    print(f"  - Final selected feature count: {len(final_cols_in_df)}")

    print("  - Splitting back into train and test...")
    cols_to_drop_train = [col for col in ["is_train", "sample_index"] if col in final_output_df.columns]
    output_train_df = final_output_df[
        final_output_df["is_train"] == 1
    ].drop(columns=cols_to_drop_train)
    
    output_test_df = final_output_df[
        final_output_df["is_train"] == 0
    ].drop(columns=["is_train", "label"])

    print(f"  - Final Train shape: {output_train_df.shape}")
    print(f"  - Final Test shape: {output_test_df.shape}")

    output_train_df.to_csv(output_train_csv_path, index=False)
    print(f"Successfully saved processed TRAIN data to '{output_train_csv_path}'")
    
    output_test_df.to_csv(output_test_csv_path, index=False)
    print(f"Successfully saved processed TEST data to '{output_test_csv_path}'")

    # -----------------------------------------------------------------
    # --- 9. 动态生成JSON元数据 ---
    # -----------------------------------------------------------------
    print(f"9. Dynamically generating feature metadata JSON...")
    output_json_data: Any = {
        "user": {"discrete": {}, "continuous": []},
        "item": {"discrete": {}, "continuous": []},
        "context": {"discrete": {}, "continuous": []},
    }

    final_columns = output_train_df.columns.tolist()

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
# === Main 执行入口 ===
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process data with HYBRID CTR features (global + temporal).",
        formatter_class=argparse.RawTextHelpFormatter,
    )

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
        "--test_csv",
        type=str,
        default="data/recsys_task_data/test_samples-20221019.csv",
        help="Path to the RAW test samples CSV file (without label).",
    )
    parser.add_argument(
        "--output_train_csv",
        type=str,
        default="data/processed/train_temporal.csv",
        help="Path for the final output PROCESSED TRAIN CSV file.",
    )
    parser.add_argument(
        "--output_test_csv",
        type=str,
        default="data/processed/test_temporal.csv",
        help="Path for the final output PROCESSED TEST CSV file.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="data/processed/train_temporal.json",
        help="Path for the output companion JSON file.",
    )
    parser.add_argument(
        "--cardinality_cap",
        type=int,
        default=4096,
        help="Maximum cardinality for discrete features. (Default: 4096)",
    )
    parser.add_argument(
        "--smoothing_factor",
        type=float,
        default=100.0,
        help="Smoothing factor for Bayesian CTR calculation. (Default: 100.0)",
    )
    parser.add_argument(
        "--global_prior",
        type=float,
        default=0.1,
        help="Global prior CTR for cold-start. Will be overridden by computed value. (Default: 0.1)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("HYBRID CTR PREPROCESSING - GLOBAL + TEMPORAL STRATEGY")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  - Train CSV: {args.train_csv}")
    print(f"  - Test CSV: {args.test_csv}")
    print(f"  - User CSV: {args.user_csv}")
    print(f"  - Item CSV: {args.item_csv}")
    print(f"  - Output Train CSV: {args.output_train_csv}")
    print(f"  - Output Test CSV: {args.output_test_csv}")
    print(f"  - Output JSON: {args.output_json}")
    print(f"  - Cardinality Cap: {args.cardinality_cap}")
    print(f"  - Smoothing Factor: {args.smoothing_factor}")
    print(f"  - Global Prior (initial): {args.global_prior}")
    print("\nCTR Strategy:")
    print("  - GLOBAL CTR (fast): population-level features (age, gender, categories, etc.)")
    print("  - TEMPORAL CTR (careful): itemid only (prevents leakage)")
    print("  - TEMPORAL COUNT: itemid click count in recent 60 days")
    print("\n" + "=" * 80)

    print("\n1. Loading ALL raw data files...")
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
        exit(1)

    print("\n2. Starting unified data processing with HYBRID CTR calculation...")
    process_data(
        train_df_raw,
        test_df_raw,
        user_df,
        item_df,
        args.output_train_csv,
        args.output_test_csv,
        args.output_json,
        cardinality_cap=args.cardinality_cap,
        smoothing_factor=args.smoothing_factor,
        global_prior=args.global_prior,
    )
    
    print("\n" + "=" * 80)
    print("Processing complete for both train and test sets.")
    print("✓ Hybrid CTR Strategy:")
    print("  - GLOBAL CTR: age, gender, level, categories, time features, and crosses")
    print("  - TEMPORAL CTR: itemid only (prevents information leakage)")
    print("  - TEMPORAL COUNT: itemid click count in recent 60 days (hotness indicator)")
    print("=" * 80)
