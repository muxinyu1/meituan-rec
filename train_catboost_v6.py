#!/usr/bin/env python3
"""
CatBoost training script V6 (feature-engineering enhanced)

Built on V3 with additions from the reference feature guide, including:
- User behavior/preference stats and binning
- Item-side statistics and temporal behaviors
- Richer time cycles, period flags, and activity binning
- Climate features (temperature gap/normalization, weather types)
- Multi-granularity distance bins, ratios/differences, geo prefixes, city aggregates
- High-value features (high-CTR periods/categories/jobs/levels combos, distance levels)

Training pipeline stays the same as V3: K-fold target encoding + CatBoost GPU CV/ensemble.
"""

import gc
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')


def safe_divide(num, den):
    """Safe division to avoid division-by-zero"""
    return np.where(den == 0, 0, num / den)


class CTRFeatureEncoder:
    """K-fold target encoding encoder to mitigate leakage"""
    
    def __init__(self, cols, n_folds=5, smoothing=20, random_state=42):
        self.cols = cols
        self.n_folds = n_folds
        self.smoothing = smoothing
        self.random_state = random_state
        self.global_mean = None
        self.encodings = {}  # 存储每列的全局编码映射
    
    def fit_transform(self, X, y):
        """在训练集上进行K折Target Encoding"""
        result = X.copy()
        self.global_mean = y.mean()
        
        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        for col in self.cols:
            if col not in X.columns:
                continue
                
            print(f"  处理Target Encoding: {col}")
            col_name = f'{col}_ctr'
            result[col_name] = np.nan
            
            # K折编码
            for train_idx, val_idx in kf.split(X, y):
                # 在训练折上计算统计
                train_data = pd.DataFrame({'col': X.iloc[train_idx][col], 'label': y.iloc[train_idx]})
                stats = train_data.groupby('col')['label'].agg(['sum', 'count'])
                
                # 平滑处理
                smooth_ctr = (stats['sum'] + self.smoothing * self.global_mean) / (stats['count'] + self.smoothing)
                
                # 应用到验证折
                result.iloc[val_idx, result.columns.get_loc(col_name)] = X.iloc[val_idx][col].map(smooth_ctr)
            
            # 填充缺失值（新类别）
            result[col_name] = result[col_name].fillna(self.global_mean)
            
            # 保存全局编码用于测试集
            full_data = pd.DataFrame({'col': X[col], 'label': y})
            full_stats = full_data.groupby('col')['label'].agg(['sum', 'count'])
            self.encodings[col] = (full_stats['sum'] + self.smoothing * self.global_mean) / (full_stats['count'] + self.smoothing)
        
        return result
    
    def transform(self, X):
        """对测试集应用Target Encoding"""
        result = X.copy()
        
        for col in self.cols:
            if col not in X.columns or col not in self.encodings:
                continue
            
            col_name = f'{col}_ctr'
            result[col_name] = X[col].map(self.encodings[col]).fillna(self.global_mean)
        
        return result


def reduce_mem_usage(df):
    """Downcast numeric dtypes to reduce DataFrame memory"""
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and col_type.name != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
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
    print(f'内存: {start_mem:.2f}MB -> {end_mem:.2f}MB (减少{100*(start_mem-end_mem)/start_mem:.1f}%)')
    return df


def load_data():
    """Load train/test from the script directory"""
    print("="*60)
    print("Loading data...")
    print("="*60)
    # 使用脚本所在目录作为数据目录，避免硬编码路径失效
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_df = pd.read_csv(os.path.join(base_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(base_dir, 'test.csv'))
    
    print(f"训练集: {train_df.shape}")
    print(f"测试集: {test_df.shape}")
    print(f"正样本比例: {train_df['label'].mean():.4f}")
    
    return train_df, test_df


def add_user_behavior_features(df, train_len):
    """User behavior features (per reference guide)"""
    if 'userid' not in df.columns:
        return df

    train_part = df.iloc[:train_len]
    filled_train = train_part.copy()
    for col in ['price', 'distance', 'user_home_dis', 'user_work_dis']:
        if col in filled_train.columns:
            filled_train[col] = filled_train[col].fillna(filled_train[col].median())

    user_group = filled_train.groupby('userid')
    user_aggs = {
        'user_item_count': ('itemid', 'count'),
        'user_unique_cates1': ('cate_1', 'nunique'),
        'user_unique_cates2': ('cate_2', 'nunique'),
        'user_unique_cities': ('cityid', 'nunique'),
        'user_avg_distance': ('distance', 'mean'),
        'user_distance_std': ('distance', 'std'),
        'user_price_mean': ('price', 'mean'),
        'user_price_std': ('price', 'std')
    }

    for new_feat, (col, op) in user_aggs.items():
        if col in filled_train.columns:
            agg = user_group[col].agg(op)
            df[new_feat] = df['userid'].map(agg)
            df[new_feat] = df[new_feat].fillna(0).astype(np.float32)

    ratio_features = [
        ('price', 'distance'),
        ('price', 'user_home_dis'),
        ('price', 'user_work_dis'),
        ('distance', 'user_home_dis'),
        ('distance', 'user_work_dis'),
        ('user_home_dis', 'user_work_dis')
    ]

    for num, den in ratio_features:
        if num in df.columns and den in df.columns:
            df[f'{num}_{den}_ratio'] = safe_divide(df[num], df[den] + 1).astype(np.float32)

    return df


def add_item_features(df, train_len):
    """Item-side features (per reference guide)"""
    if 'itemid' not in df.columns:
        return df

    train_part = df.iloc[:train_len].copy()
    for col in ['price', 'distance']:
        if col in train_part.columns:
            train_part[col] = train_part[col].fillna(train_part[col].median())

    item_group = train_part.groupby('itemid')
    item_aggs = {
        'item_user_count': ('userid', 'count'),
        'item_city_count': ('cityid', 'nunique'),
        'item_avg_distance': ('distance', 'mean'),
        'item_price_std': ('price', 'std')
    }

    for new_feat, (col, op) in item_aggs.items():
        if col in train_part.columns:
            agg = item_group[col].agg(op)
            df[new_feat] = df['itemid'].map(agg)
            df[new_feat] = df[new_feat].fillna(0).astype(np.float32)

    if 'hour' in df.columns:
        hour_stats = train_part.groupby('itemid')['hour'].agg(['mean', 'std'])
        hour_stats.columns = ['item_hour_mean', 'item_hour_std']
        df = df.merge(hour_stats, on='itemid', how='left')
        df['item_hour_mean'] = df['item_hour_mean'].fillna(df['hour'].mean()).astype(np.float32)
        df['item_hour_std'] = df['item_hour_std'].fillna(0).astype(np.float32)
        df['item_hour_diff'] = (np.abs(df['hour'] - df['item_hour_mean'])).astype(np.float32)

    return df


def add_time_features(df, train_len):
    """Expanded time features"""
    if 'hour' in df.columns:
        hour_float = df['hour'].astype(float)
        df['is_morning'] = ((hour_float >= 5) & (hour_float < 12)).astype(np.int8)
        df['is_afternoon'] = ((hour_float >= 12) & (hour_float < 18)).astype(np.int8)
        df['is_evening'] = ((hour_float >= 18) & (hour_float < 22)).astype(np.int8)
        df['is_night'] = ((hour_float >= 22) | (hour_float < 5)).astype(np.int8)
        df['is_work_hour'] = ((hour_float >= 9) & (hour_float <= 18)).astype(np.int8)
        df['is_rush_hour_flag'] = (((hour_float >= 8) & (hour_float <= 10)) | ((hour_float >= 17) & (hour_float <= 19))).astype(np.int8)

    if 'weekday' in df.columns:
        weekday_float = df['weekday'].astype(float)
        df['is_weekend_flag'] = (weekday_float >= 5).astype(np.int8)
        df['is_monday'] = (weekday_float == 0).astype(np.int8)
        df['is_friday'] = (weekday_float == 4).astype(np.int8)

    for period, max_val in [('hour', 24), ('weekday', 7)]:
        if period in df.columns:
            period_float = df[period].astype(float)
            for n in [1, 2]:
                df[f'{period}_sin_{n}'] = np.sin(2 * np.pi * n * period_float / max_val).astype(np.float32)
                df[f'{period}_cos_{n}'] = np.cos(2 * np.pi * n * period_float / max_val).astype(np.float32)

    # User activity / time-preference binning (train-only stats)
    if 'userid' in df.columns and 'hour' in df.columns:
        train_part = df.iloc[:train_len]
        user_activity = train_part.groupby('userid').size()
        if len(user_activity) > 0:
            activity_bin = pd.qcut(user_activity, q=10, duplicates='drop', labels=False)
            df['user_activity_bin'] = df['userid'].map(activity_bin).fillna(activity_bin.max()).astype(np.float32)

        hour_dist = pd.crosstab(train_part['userid'], train_part['hour'], normalize='index')
        hour_dist = hour_dist.reindex(columns=range(24), fill_value=0)
        period_slices = {
            'morning': range(5, 12),
            'afternoon': range(12, 18),
            'evening': range(18, 23)
        }
        for period_name, cols in period_slices.items():
            user_period_freq = hour_dist[list(cols)].sum(axis=1)
            bins = pd.qcut(user_period_freq, q=5, duplicates='drop', labels=False)
            df[f'user_{period_name}_freq_bin'] = df['userid'].map(bins).fillna(bins.max()).astype(np.float32)

        weekday_dist = pd.crosstab(train_part['userid'], train_part['weekday'], normalize='index')
        weekday_dist = weekday_dist.reindex(columns=range(7), fill_value=0)
        user_weekend_freq = weekday_dist[[5, 6]].sum(axis=1)
        bins = pd.qcut(user_weekend_freq, q=5, duplicates='drop', labels=False)
        df['user_weekend_freq_bin'] = df['userid'].map(bins).fillna(bins.max()).astype(np.float32)

        if 'timestamp_dt' in df.columns:
            last_active = df.groupby('userid')['timestamp_dt'].transform('max')
            time_diff = (df['timestamp_dt'].max() - last_active).dt.total_seconds() / 3600
            df['user_recency_bin'] = pd.qcut(time_diff, q=10, duplicates='drop', labels=False).astype(np.float32)

        user_stats_cols = ['price', 'distance', 'user_home_dis', 'user_work_dis']
        for col in user_stats_cols:
            if col in train_part.columns:
                stat_val = train_part.groupby('userid')[col].transform('mean')
                bins = pd.qcut(stat_val, q=10, duplicates='drop', labels=False)
                df[f'user_{col}_mean_bin'] = df['userid'].map(bins).fillna(bins.max()).astype(np.float32)

    # Distance preference by time period (current df)
    for dist_col in ['distance', 'user_home_dis', 'user_work_dis']:
        if dist_col not in df.columns or 'hour' not in df.columns:
            continue
        for time_period, mask in {
            'morning': df.get('is_morning', pd.Series(False, index=df.index)) == 1,
            'afternoon': df.get('is_afternoon', pd.Series(False, index=df.index)) == 1,
            'evening': df.get('is_evening', pd.Series(False, index=df.index)) == 1,
        }.items():
            period_dist = df.loc[mask, dist_col]
            if not period_dist.empty:
                df.loc[mask, f'{dist_col}_{time_period}_bin'] = pd.qcut(period_dist, q=5, duplicates='drop', labels=False).astype(np.float32)

    return df


def add_climate_features(df):
    """Climate-related features"""
    if {'temp_high', 'temp_low'}.issubset(df.columns):
        df['temp_gap'] = (df['temp_high'].astype(float) - df['temp_low'].astype(float)).astype(np.float32)
        df['temp_norm'] = safe_divide(df['temp'].astype(float) - df['temp_low'].astype(float), df['temp_gap'].astype(float) + 1).astype(np.float32)

    if 'weather' in df.columns:
        weather_str = df['weather'].astype(str)
        df['is_bad_weather'] = weather_str.isin(['2', '3', '4']).astype(np.int8)
        df['is_good_weather'] = weather_str.isin(['0', '1']).astype(np.int8)
        df['is_extreme_weather'] = weather_str.isin(['4', '5']).astype(np.int8)
    return df


def add_distance_features(df, train_len):
    """Multi-granularity distance and geo features"""
    # Geohash prefixes and work-geohash prefixes
    for i in range(1, 5):
        if 'geohash' in df.columns:
            df[f'geohash_prefix_{i}'] = df['geohash'].astype(str).str[:i]
        if 'work_geohash' in df.columns:
            df[f'work_geohash_prefix_{i}'] = df['work_geohash'].astype(str).str[:i]
        if f'geohash_prefix_{i}' in df.columns and f'work_geohash_prefix_{i}' in df.columns:
            df[f'area_match_{i}'] = (df[f'geohash_prefix_{i}'] == df[f'work_geohash_prefix_{i}']).astype(np.int8)

    # City-level aggregates
    if 'cityid' in df.columns:
        train_part = df.iloc[:train_len]
        city_group = train_part.groupby('cityid')
        city_aggs = {
            'city_unique_items': ('itemid', 'nunique'),
            'city_unique_cates1': ('cate_1', 'nunique'),
            'city_unique_cates2': ('cate_2', 'nunique'),
            'city_avg_price': ('item_ave_price', 'mean'),
            'city_avg_distance': ('distance', 'mean'),
            'city_item_density': ('itemid', 'count')
        }
        for new_feat, (col, op) in city_aggs.items():
            if col in train_part.columns:
                agg = city_group[col].agg(op)
                df[new_feat] = df['cityid'].map(agg).fillna(0).astype(np.float32)

    distance_cols = ['distance', 'user_home_dis', 'user_work_dis']
    for col in distance_cols:
        if col not in df.columns:
            continue
        df[f'{col}_bin_5'] = pd.qcut(df[col], q=5, duplicates='drop', labels=False).astype(np.float32)
        df[f'{col}_bin_10'] = pd.qcut(df[col], q=10, duplicates='drop', labels=False).astype(np.float32)
        df[f'{col}_bin_20'] = pd.qcut(df[col], q=20, duplicates='drop', labels=False).astype(np.float32)
        df[f'{col}_rank'] = df[col].rank(pct=True).astype(np.float32)
        mean_val = df[col].mean()
        df[f'{col}_diff_mean'] = (df[col] - mean_val).astype(np.float32)
        df[f'{col}_user_mean'] = df.groupby('userid')[col].transform('mean') if 'userid' in df.columns else 0
        df[f'{col}_user_diff'] = (df[col] - df[f'{col}_user_mean']).astype(np.float32)
        df[f'{col}_zscore'] = safe_divide(df[col] - df[col].mean(), df[col].std() + 1e-6).astype(np.float32)

    # Distance ratios and differences
    if {'distance', 'user_home_dis', 'user_work_dis'}.issubset(df.columns):
        df['home_work_ratio'] = safe_divide(df['user_home_dis'], df['user_work_dis'] + 1).astype(np.float32)
        df['distance_home_ratio'] = safe_divide(df['distance'], df['user_home_dis'] + 1).astype(np.float32)
        df['distance_work_ratio'] = safe_divide(df['distance'], df['user_work_dis'] + 1).astype(np.float32)
        df['home_work_diff'] = (df['user_home_dis'] - df['user_work_dis']).astype(np.float32)
        df['distance_home_diff'] = (df['distance'] - df['user_home_dis']).astype(np.float32)
        df['distance_work_diff'] = (df['distance'] - df['user_work_dis']).astype(np.float32)
        for col in ['distance', 'user_home_dis', 'user_work_dis']:
            city_mean = df.groupby('cityid')[col].transform('mean') if 'cityid' in df.columns else 0
            df[f'{col}_city_ratio'] = safe_divide(df[col], city_mean + 1).astype(np.float32)

    # Distance with category/price-level interactions
    if 'cate_1' in df.columns:
        for col in distance_cols:
            if col in df.columns:
                df[f'{col}_cate1_mean'] = df.groupby('cate_1')[col].transform('mean')
                df[f'{col}_cate1_diff'] = (df[col] - df[f'{col}_cate1_mean']).astype(np.float32)

    if 'price' in df.columns:
        price_level = pd.qcut(df['price'], q=5, duplicates='drop', labels=False)
        for col in distance_cols:
            if col in df.columns:
                df[f'{col}_price_level_mean'] = df.groupby(price_level)[col].transform('mean')
                df[f'{col}_price_level_diff'] = (df[col] - df[f'{col}_price_level_mean']).astype(np.float32)

    return df


def add_high_value_features(df):
    """High-value features (per reference guide)"""
    if 'hour' in df.columns:
        hour = pd.to_numeric(df['hour'], errors='coerce').fillna(-1)
        df['is_high_ctr_hour'] = ((hour >= 1) & (hour <= 2)).astype(np.int8)
        df['morning_display'] = ((hour >= 8) & (hour <= 11)).astype(np.int8)
        df['afternoon_display'] = ((hour >= 12) & (hour <= 17)).astype(np.int8)
        df['evening_display'] = ((hour >= 18) & (hour <= 22)).astype(np.int8)
        df['night_display'] = ((hour >= 23) | (hour <= 4)).astype(np.int8)

    if 'distance' in df.columns:
        dist = pd.to_numeric(df['distance'], errors='coerce').fillna(-1)
        quantiles = dist.quantile([0.2, 0.4, 0.6, 0.8]).values
        bins = np.unique(np.concatenate(([-np.inf], quantiles, [np.inf])))
        # 当分位点重复导致箱数不足时，labels 动态匹配
        labels = list(range(len(bins) - 1))
        df['distance_level'] = pd.cut(dist, bins=bins, labels=labels, duplicates='drop').astype(float).fillna(-1).astype(np.int8)

    if {'level', 'distance_level'}.issubset(df.columns):
        df['high_level_close_distance'] = ((df['level'] >= 4.0) & (df['distance_level'] == 0)).astype(np.int8)

    # High-value user/job/category flags
    if 'level' in df.columns:
        df['is_high_level'] = (pd.to_numeric(df['level'], errors='coerce').fillna(-1) >= 4.0).astype(np.int8)
    if 'job' in df.columns:
        df['is_high_ctr_job'] = (pd.to_numeric(df['job'], errors='coerce').fillna(-1) == 1).astype(np.int8)
    if {'level', 'job'}.issubset(df.columns):
        df['high_level_job1'] = ((pd.to_numeric(df['level'], errors='coerce').fillna(-1) >= 4.0) & (pd.to_numeric(df['job'], errors='coerce').fillna(-1) == 1)).astype(np.int8)
    if 'cate_1' in df.columns:
        cate1_num = pd.to_numeric(df['cate_1'], errors='coerce').fillna(-1)
        df['is_high_ctr_cate'] = (cate1_num == 209.0).astype(np.int8)
        if 'level' in df.columns:
            df['high_level_preferred_cate'] = ((pd.to_numeric(df['level'], errors='coerce').fillna(-1) >= 4.0) & (cate1_num == 209.0)).astype(np.int8)

    # Combination features
    if 'is_high_ctr_hour' in df.columns and 'is_high_ctr_cate' in df.columns:
        df['prime_time_prime_cate'] = (df['is_high_ctr_hour'].astype(bool) & df['is_high_ctr_cate'].astype(bool)).astype(np.int8)
    if 'is_high_ctr_hour' in df.columns and 'level' in df.columns:
        df['high_level_prime_time'] = ((pd.to_numeric(df['level'], errors='coerce').fillna(-1) >= 4.0) & df['is_high_ctr_hour'].astype(bool)).astype(np.int8)
    if 'job' in df.columns and 'is_high_ctr_price' in df.columns:
        df['job1_preferred_price'] = ((pd.to_numeric(df['job'], errors='coerce').fillna(-1) == 1) & df['is_high_ctr_price'].astype(bool)).astype(np.int8)
    if {'level', 'job', 'is_high_ctr_hour'}.issubset(df.columns):
        df['high_quality_combination'] = ((pd.to_numeric(df['level'], errors='coerce').fillna(-1) >= 4.0) & (pd.to_numeric(df['job'], errors='coerce').fillna(-1) == 1) & df['is_high_ctr_hour'].astype(bool)).astype(np.int8)

    # High-value distance stats (user/city/time based)
    distance_cols = ['distance', 'user_home_dis', 'user_work_dis']
    for col in distance_cols:
        if col not in df.columns:
            continue
        if 'userid' in df.columns:
            user_stats = df.groupby('userid')[col].agg(['mean', 'std']).add_prefix(f'user_{col}_')
            df = df.merge(user_stats, on='userid', how='left')
            # 若统计列因缺失未生成，补零以避免 KeyError
            if f'user_{col}_mean' not in df.columns:
                df[f'user_{col}_mean'] = 0
            if f'user_{col}_std' not in df.columns:
                df[f'user_{col}_std'] = 0
            df[f'{col}_user_diff'] = (df[col] - df[f'user_{col}_mean']).astype(np.float32)
            df[f'{col}_user_zscore'] = safe_divide(df[f'{col}_user_diff'], df[f'user_{col}_std'] + 1e-6).astype(np.float32)
        if 'cityid' in df.columns:
            city_stats = df.groupby('cityid')[col].agg(['mean', 'std']).add_prefix(f'city_{col}_')
            df = df.merge(city_stats, on='cityid', how='left')
            df[f'{col}_city_diff'] = (df[col] - df[f'city_{col}_mean']).astype(np.float32)
        if 'hour' in df.columns:
            df[f'{col}_per_hour'] = safe_divide(df[col], df['hour'] + 1).astype(np.float32)

    return df


def create_features(train_df, test_df):
    """Feature engineering for V6"""
    print("="*60)
    print("Feature engineering V6...")
    print("="*60)
    
    # Preserve key columns
    test_sample_index = test_df['sample_index'].copy() if 'sample_index' in test_df.columns else None
    y_train = train_df['label'].copy()
    
    train_len = len(train_df)
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    # ==========================================
    # 1) Missing-value flags (from V3)
    # ==========================================
    print("\n1) Building missing-value flags...")
    high_missing_cols = ['price', 'online_days', 'user_displayed_item_num', 'distance', 
                         'item_ave_price', 'level', 'cate_1', 'cate_2', 'cate_3']
    for col in high_missing_cols:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isnull().astype(np.int8)
    
    # ==========================================
    # 2) Time features (extended)
    # ==========================================
    print("\n2) Building time features...")
    if 'timestamp' in df.columns:
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        
        # Cyclic encodings (1x, 2x periods)
        df['hour_sin'] = np.sin(2 * np.pi * df['timestamp_dt'].dt.hour / 24).astype(np.float32)
        df['hour_cos'] = np.cos(2 * np.pi * df['timestamp_dt'].dt.hour / 24).astype(np.float32)
        df['weekday_sin'] = np.sin(2 * np.pi * df['timestamp_dt'].dt.weekday / 7).astype(np.float32)
        df['weekday_cos'] = np.cos(2 * np.pi * df['timestamp_dt'].dt.weekday / 7).astype(np.float32)
        
        # Weekend flag
        if 'weekday' in df.columns:
            df['is_weekend'] = df['weekday'].isin([6, 7]).astype(np.int8)
        else:
            df['is_weekend'] = df['timestamp_dt'].dt.weekday.isin([5, 6]).astype(np.int8)
        
        # Time-of-day buckets
        def get_time_period(hour):
            if 6 <= hour < 11:
                return 1  # 上午
            elif 11 <= hour < 14:
                return 2  # 午餐
            elif 14 <= hour < 18:
                return 3  # 下午
            elif 18 <= hour < 22:
                return 4  # 晚餐/晚间
            else:
                return 0  # 深夜/凌晨
        
        df['time_period'] = df['hour'].apply(get_time_period).astype(np.int8)
    else:
        # fallback: 保留 is_weekend
        if 'weekday' in df.columns:
            df['is_weekend'] = df['weekday'].isin([6, 7]).astype(np.int8)

    # ==========================================
    # 3) Numeric imputation + log transforms
    # ==========================================
    print("\n3) Numeric imputation and basic transforms...")
    numerical_cols = ['distance', 'item_ave_price', 'price', 'user_home_dis', 'user_work_dis',
                      'temp', 'temp_low', 'temp_high', 'user_displayed_item_num', 'online_days']
    train_part = df.iloc[:train_len]
    for col in numerical_cols:
        if col in df.columns:
            median_val = train_part[col].median()
            df[col] = df[col].fillna(median_val).astype(np.float32)
    
    log_cols = ['distance', 'price', 'item_ave_price', 'user_home_dis', 'user_work_dis']
    for col in log_cols:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(np.maximum(df[col], 0)).astype(np.float32)

    # Binning (multi-granularity per reference)
    if 'distance' in df.columns:
        df['distance_bin'] = pd.cut(df['distance'], bins=[-1, 20, 40, 60, 80, 200], 
                                     labels=[0, 1, 2, 3, 4]).astype(float).fillna(2).astype(np.int8)
    if 'price' in df.columns:
        df['price_bin'] = pd.cut(df['price'], bins=[-1, 20, 40, 60, 80, 200], 
                                  labels=[0, 1, 2, 3, 4]).astype(float).fillna(2).astype(np.int8)

    # Temperature comfort and range
    if 'temp' in df.columns:
        df['temp_comfort'] = ((df['temp'] >= 17) & (df['temp'] <= 26)).astype(np.int8)
    if 'temp_high' in df.columns and 'temp_low' in df.columns:
        df['temp_range'] = (df['temp_high'] - df['temp_low']).astype(np.float32)

    # High-CTR price flag (for combos)
    if 'price' in df.columns:
        df['is_high_ctr_price'] = (df['price'] <= df['price'].quantile(0.2)).astype(np.int8)

    # ==========================================
    # 4) Count encoding (from V3)
    # ==========================================
    print("\n4) Building count-encoding features...")
    count_cols = ['userid', 'itemid', 'geohash', 'cityid', 'loc_cityid', 
                  'cate_1', 'cate_2', 'cate_3', 'dtype']
    
    train_part = df.iloc[:train_len]
    for col in count_cols:
        if col in df.columns:
            counts = train_part[col].value_counts()
            df[f'{col}_count'] = df[col].map(counts).fillna(0).astype(np.float32)
            df[f'{col}_count_log'] = np.log1p(df[f'{col}_count']).astype(np.float32)
    
    # ==========================================
    # 5) Cross features (extended over V3)
    # ==========================================
    print("\n5) Building cross features...")
    if 'cityid' in df.columns and 'loc_cityid' in df.columns:
        df['is_same_city'] = (df['cityid'] == df['loc_cityid']).astype(np.int8)
    
    cross_pairs = [
        ('userid', 'cate_1'), ('userid', 'cate_2'), 
        ('cityid', 'cate_1'), ('cityid', 'cate_2'),
        ('time_period', 'cate_1'), ('time_period', 'dtype'),
        ('gender', 'cate_1'), ('age', 'cate_1'), ('level', 'cate_1'),
        ('gender', 'dtype'), ('age', 'dtype'),
        ('weekday', 'time_period'), ('is_weekend', 'time_period')
    ]
    
    for c1, c2 in cross_pairs:
        if c1 in df.columns and c2 in df.columns:
            new_col = f'{c1}_{c2}'
            df[new_col] = df[c1].astype(str) + '_' + df[c2].astype(str)

    # ==========================================
    # 6) User / item / time / climate / distance features (reference additions)
    # ==========================================
    print("\n6) User behavior features...")
    df = add_user_behavior_features(df, train_len)
    
    print("\n7) Item features...")
    df = add_item_features(df, train_len)

    print("\n8) Advanced time features...")
    df = add_time_features(df, train_len)

    print("\n9) Climate features...")
    df = add_climate_features(df)

    print("\n10) Distance & geo features...")
    df = add_distance_features(df, train_len)

    print("\n11) High-value features...")
    df = add_high_value_features(df)

    # Drop raw timestamp columns
    for col in ['timestamp', 'timestamp_dt']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # ==========================================
    # 7) Categorical handling
    # ==========================================
    print("\n12) Handling categorical features...")
    categorical_features = [
        'userid', 'itemid', 'geohash', 'cityid', 'loc_cityid', 
        'weekday', 'hour', 'weather', 'dtype', 'cate_1', 'cate_2', 'cate_3',
        'age', 'level', 'gender', 'married', 'job', 'has_car', 
        'work_geohash', 'mobile_type', 'mobile_os',
        'is_weekend', 'time_period', 'is_same_city', 'temp_comfort',
        'distance_bin', 'price_bin', 'distance_level',
        'is_morning', 'is_afternoon', 'is_evening', 'is_night',
        'is_work_hour', 'is_rush_hour_flag', 'is_weekend_flag', 'is_monday', 'is_friday',
        'is_bad_weather', 'is_good_weather', 'is_extreme_weather',
        'distance_bin_5', 'distance_bin_10', 'distance_bin_20',
        'user_activity_bin', 'user_morning_freq_bin', 'user_afternoon_freq_bin', 'user_evening_freq_bin', 'user_weekend_freq_bin',
        'is_high_ctr_hour', 'is_high_ctr_cate', 'is_high_ctr_job', 'is_high_level', 'high_level_close_distance',
        'prime_time_prime_cate', 'high_level_prime_time', 'job1_preferred_price', 'high_quality_combination',
    ]
    
    # Geohash prefixes and area matches
    for i in range(1, 5):
        for col in [f'geohash_prefix_{i}', f'work_geohash_prefix_{i}', f'area_match_{i}']:
            if col in df.columns:
                categorical_features.append(col)
    
    # Price/distance bin-derived features
    for col in ['distance', 'user_home_dis', 'user_work_dis']:
        for suf in ['bin_5', 'bin_10', 'bin_20']:
            name = f'{col}_{suf}'
            if name in df.columns:
                categorical_features.append(name)

    # Append cross features into categorical list
    for c1, c2 in cross_pairs:
        col_name = f'{c1}_{c2}'
        if col_name in df.columns:
            categorical_features.append(col_name)

    categorical_features = [col for col in categorical_features if col in df.columns]
    
    for col in categorical_features:
        df[col] = df[col].fillna(-1).astype(str)
    
    # 拆分数据
    train_processed = df.iloc[:train_len].copy()
    test_processed = df.iloc[train_len:].copy()
    
    del df, train_part
    gc.collect()
    
    # ==========================================
    # 8) Target encoding (from V2)
    # ==========================================
    print("\n13) Building target-encoding features...")
    
    ctr_cols = ['dtype', 'cate_1', 'cate_2', 'cate_3', 'cityid', 
                'weather', 'time_period', 'age', 'level', 'gender',
                'price_bin', 'distance_bin', 'distance_level', 'is_morning', 'is_afternoon', 'is_evening']
    ctr_cols = [col for col in ctr_cols if col in train_processed.columns]
    
    encoder = CTRFeatureEncoder(cols=ctr_cols, n_folds=5, smoothing=20)
    
    # 准备特征和标签
    exclude_cols = ['sample_index', 'label']
    feature_cols = [col for col in train_processed.columns if col not in exclude_cols]
    
    X_train = train_processed[feature_cols].copy()
    X_test = test_processed[[col for col in feature_cols if col in test_processed.columns]].copy()
    
    # 应用Target Encoding
    X_train = encoder.fit_transform(X_train, y_train)
    X_test = encoder.transform(X_test)
    
    # 恢复sample_index
    if test_sample_index is not None:
        X_test['sample_index'] = test_sample_index.values
    
    # 更新categorical_features列表（排除CTR特征）
    categorical_features = [col for col in categorical_features if col in X_train.columns]
    
    # 内存优化
    X_train = reduce_mem_usage(X_train)
    X_test = reduce_mem_usage(X_test)
    
    print(f"\nFinal feature count: {len([c for c in X_train.columns if c not in exclude_cols])}")
    
    del train_processed, test_processed
    gc.collect()
    
    return X_train, y_train, X_test, categorical_features


def cross_validation(X, y, categorical_features, n_splits=10):
    """K折交叉验证"""
    print("="*60)
    print(f"Starting {n_splits}-fold cross-validation...")
    print("="*60)
    
    weekdays = X['weekday'].astype(str)
    
    folds = [[] for _ in range(n_splits)]
    for weekday in weekdays.unique():
        weekday_indices = X[weekdays == weekday].index.tolist()
        np.random.shuffle(weekday_indices)
        
        fold_size = len(weekday_indices) // n_splits
        for i in range(n_splits):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < n_splits - 1 else len(weekday_indices)
            folds[i].extend(weekday_indices[start_idx:end_idx])
    
    for fold_indices in folds:
        np.random.shuffle(fold_indices)
    
    cv_scores = []
    models = []
    
    # 优化后的模型参数 V6（在 V3 基础上略调优）
    params = {
        'task_type': 'GPU',
        'iterations': 12000,
        'learning_rate': 0.04,
        'depth': 8,
        'l2_leaf_reg': 8.0,
        'random_strength': 1e-5,
        'border_count': 128,
        'thread_count': -1,
        'random_seed': 42,
        'verbose': 100,
        'eval_metric': 'AUC',
        'loss_function': 'Logloss',
        'auto_class_weights': 'Balanced',
        'boosting_type': 'Plain',
        'one_hot_max_size': 12,
        'nan_mode': 'Min',
        'od_type': 'Iter',
        'od_wait': 400,
        'bagging_temperature': 0.5,
        'gpu_ram_part': 0.95,
        'allow_writing_files': False,
        'use_best_model': True,
        'min_data_in_leaf': 60,
    }
    
    exclude_cols = ['sample_index', 'label']
    feature_cols = [col for col in X.columns if col not in exclude_cols]
    
    for fold in range(n_splits):
        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/{n_splits}")
        print(f"{'='*60}")
        
        val_indices = folds[fold]
        train_indices = []
        for i in range(n_splits):
            if i != fold:
                train_indices.extend(folds[i])
        
        X_train_fold = X.iloc[train_indices][feature_cols].copy()
        y_train_fold = y.iloc[train_indices].copy()
        X_val_fold = X.iloc[val_indices][feature_cols].copy()
        y_val_fold = y.iloc[val_indices].copy()
        
        print(f"Train: {X_train_fold.shape}, pos rate: {y_train_fold.mean():.4f}")
        print(f"Valid: {X_val_fold.shape}, pos rate: {y_val_fold.mean():.4f}")
        
        # 确保类别特征为字符串
        for col in categorical_features:
            if col in X_train_fold.columns:
                X_train_fold[col] = X_train_fold[col].astype(str)
                X_val_fold[col] = X_val_fold[col].astype(str)
        
        train_pool = Pool(X_train_fold, y_train_fold, cat_features=categorical_features)
        val_pool = Pool(X_val_fold, y_val_fold, cat_features=categorical_features)
        
        del X_train_fold, y_train_fold
        gc.collect()
        
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True, plot=False)
        
        print(f"Best iteration: {model.get_best_iteration()}")
        
        y_pred_proba = model.predict_proba(val_pool)[:, 1]
        auc = roc_auc_score(y_val_fold, y_pred_proba)
        cv_scores.append(auc)
        models.append(model)
        
        print(f"Fold {fold+1} AUC: {auc:.4f}")
        
        del train_pool, val_pool, X_val_fold, y_val_fold
        gc.collect()
    
    print(f"\n{'='*60}")
    print("Cross-validation summary")
    print(f"{'='*60}")
    print(f"Fold AUCs: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"Mean AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"Max AUC: {max(cv_scores):.4f}")
    print(f"Min AUC: {min(cv_scores):.4f}")
    print(f"{'='*60}\n")
    
    return models, cv_scores


def ensemble_predict(models, X_test, categorical_features):
    """集成预测"""
    print("Ensembling predictions...")
    
    exclude_cols = ['sample_index', 'label']
    feature_cols = [col for col in X_test.columns if col not in exclude_cols]
    
    X_test_features = X_test[feature_cols].copy()
    for col in categorical_features:
        if col in X_test_features.columns:
            X_test_features[col] = X_test_features[col].astype(str)
    
    test_pool = Pool(X_test_features, cat_features=categorical_features)
    
    predictions = np.zeros((X_test.shape[0], len(models)))
    for i, model in enumerate(models):
        predictions[:, i] = model.predict_proba(test_pool)[:, 1]
    
    ensemble_pred = np.mean(predictions, axis=1)
    
    del test_pool
    gc.collect()
    
    return ensemble_pred


def save_results(models, cv_scores, predictions, X_test):
    """保存结果"""
    print("Saving results...")
    
    submission = pd.DataFrame({
        'sample_index': range(len(predictions)) if 'sample_index' not in X_test.columns else X_test['sample_index'],
        'label': predictions
    })
    submission.to_csv('submission_v6.csv', index=False)
    print("Submission saved to submission_v6.csv")
    
    models[0].save_model('catboost_v6.cbm')
    print("Model saved to catboost_v6.cbm")
    
    results = {
        'metrics': {
            'auc_mean': float(np.mean(cv_scores)),
            'auc_std': float(np.std(cv_scores)),
            'auc_max': float(max(cv_scores)),
            'auc_min': float(min(cv_scores)),
            'fold_scores': [float(s) for s in cv_scores]
        },
        'model_params': models[0].get_all_params()
    }
    
    def convert(o):
        if isinstance(o, np.float32): return float(o)
        if isinstance(o, np.int64): return int(o)
        raise TypeError
    
    with open('model_results_v6.json', 'w') as f:
        json.dump(results, f, indent=4, default=convert)
    
    print("Evaluation saved to model_results_v6.json")
    
    importance = models[0].get_feature_importance()
    feature_cols = [col for col in X_test.columns if col not in ['sample_index', 'label']]
    
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop-20 important features:")
    print(importance_df.head(20).to_string(index=False))
    
    importance_df.to_csv('feature_importance_v6.csv', index=False)


def main():
    """主函数"""
    np.random.seed(42)
    start_time = time.time()
    
    train_df, test_df = load_data()
    
    X_train, y_train, X_test, categorical_features = create_features(train_df, test_df)
    
    del train_df, test_df
    gc.collect()
    
    models, cv_scores = cross_validation(X_train, y_train, categorical_features, n_splits=10)
    
    predictions = ensemble_predict(models, X_test, categorical_features)
    
    save_results(models, cv_scores, predictions, X_test)
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print("CatBoost V6 training done!")


if __name__ == "__main__":
    main()