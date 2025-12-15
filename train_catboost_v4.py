#!/usr/bin/env python3
"""
CatBoost训练脚本 V4 (深度优化版) - 修复版
V4优化点（基于V3）：
1. 用户/物品曝光次数统计（无泄露）：曝光数、活跃度分箱
2. 基于排名的特征：物品在类别内的热度排名、价格排名
3. 交叉组合Target Encoding：城市×类别、性别×类别等组合的CTR
4. 移除无效特征：sin/cos周期编码（重要性为0）
5. 加权集成：根据各折AUC进行加权平均
6. 更多统计聚合特征：均值、标准差等
7. 优化模型参数：调整学习率和正则化

⚠️ 重要修复：
- 移除了直接使用label计算的user_ctr/item_ctr特征（会导致严重的数据泄露）
- 改用曝光次数（count）等无监督统计，CTR类特征通过K-fold Target Encoding计算
"""

import pandas as pd
import numpy as np
import gc
import shutil
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import time
import os
import json
import warnings
warnings.filterwarnings('ignore')

# 尝试导入torch用于GPU内存管理
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# 全局基础路径与检查点目录
BASE_DIR = "/media/wenhao/7d8f46ef-9674-40b1-a999-148b79a69954/zhaoshanhui/meituan-rec"
CHECKPOINT_SUBDIR = "train_catboost_v4"


def get_checkpoint_dir():
    """返回并创建检查点目录"""
    ckpt_dir = os.path.join(BASE_DIR, CHECKPOINT_SUBDIR)
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir


def get_output_dir():
    """
    统一的输出目录（大文件必须落到 /media/... 下）。
    目前与 checkpoint 目录保持一致，便于管理训练产物。
    """
    return get_checkpoint_dir()


def safe_link_or_copy(src: str, dst: str) -> str:
    """
    尽量避免复制大文件占用额外空间：
    - 同一文件系统：优先硬链接（几乎不占空间）
    - 否则：退化为软链接
    - 最后兜底：copy2
    """
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if src == dst:
        return dst

    if os.path.lexists(dst):
        os.remove(dst)

    # 1) hard link (same filesystem)
    try:
        os.link(src, dst)
        return dst
    except OSError:
        pass

    # 2) symlink (cross filesystem ok)
    try:
        os.symlink(src, dst)
        return dst
    except OSError:
        pass

    # 3) fallback copy
    shutil.copy2(src, dst)
    return dst


def save_fold_metadata(fold_idx, auc, model, path):
    """保存单折的元信息"""
    meta = {
        "fold": fold_idx + 1,
        "auc": float(auc),
        "best_iteration": int(model.get_best_iteration()),
        "best_score": model.get_best_score().get("validation", {}),
        "params": model.get_all_params(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=4)


class AdvancedCTREncoder:
    """
    高级Target Encoding编码器
    支持：
    1. 单列编码
    2. 多列交叉编码（用户×类别 等）
    3. 可配置的平滑参数
    """
    
    def __init__(self, n_folds=5, smoothing=20, random_state=42):
        self.n_folds = n_folds
        self.smoothing = smoothing
        self.random_state = random_state
        self.global_mean = None
        self.encodings = {}
    
    def _get_smooth_ctr(self, stats, global_mean, smoothing):
        """计算平滑CTR"""
        return (stats['sum'] + smoothing * global_mean) / (stats['count'] + smoothing)
    
    def fit_transform_single(self, X, y, cols, prefix=''):
        """对单列或多列组合进行K折Target Encoding"""
        result = X.copy()
        self.global_mean = y.mean() if self.global_mean is None else self.global_mean
        
        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        # 创建组合键
        if isinstance(cols, str):
            cols = [cols]
        
        if len(cols) == 1:
            key_col = cols[0]
            col_name = f'{prefix}{key_col}_ctr' if prefix else f'{key_col}_ctr'
            combined_col = X[key_col].astype(str)
        else:
            key_col = '_'.join(cols)
            col_name = f'{prefix}{key_col}_ctr' if prefix else f'{key_col}_ctr'
            combined_col = X[cols[0]].astype(str)
            for c in cols[1:]:
                combined_col = combined_col + '_' + X[c].astype(str)
        
        if key_col in self.encodings:
            return result  # 已处理过
            
        print(f"  Target Encoding: {col_name}")
        result[col_name] = np.nan
        
        # K折编码
        for train_idx, val_idx in kf.split(X, y):
            train_data = pd.DataFrame({
                'col': combined_col.iloc[train_idx], 
                'label': y.iloc[train_idx]
            })
            stats = train_data.groupby('col')['label'].agg(['sum', 'count'])
            smooth_ctr = self._get_smooth_ctr(stats, self.global_mean, self.smoothing)
            result.iloc[val_idx, result.columns.get_loc(col_name)] = \
                combined_col.iloc[val_idx].map(smooth_ctr)
        
        result[col_name] = result[col_name].fillna(self.global_mean)
        
        # 保存全局编码
        full_data = pd.DataFrame({'col': combined_col, 'label': y})
        full_stats = full_data.groupby('col')['label'].agg(['sum', 'count'])
        self.encodings[key_col] = {
            'mapping': self._get_smooth_ctr(full_stats, self.global_mean, self.smoothing),
            'cols': cols,
            'col_name': col_name
        }
        
        return result
    
    def transform_single(self, X, key):
        """对测试集应用编码"""
        if key not in self.encodings:
            return X
            
        result = X.copy()
        enc = self.encodings[key]
        cols = enc['cols']
        col_name = enc['col_name']
        
        if len(cols) == 1:
            combined_col = X[cols[0]].astype(str)
        else:
            combined_col = X[cols[0]].astype(str)
            for c in cols[1:]:
                combined_col = combined_col + '_' + X[c].astype(str)
        
        result[col_name] = combined_col.map(enc['mapping']).fillna(self.global_mean)
        return result


def reduce_mem_usage(df):
    """优化DataFrame内存使用"""
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
    """加载数据"""
    print("="*60)
    print("加载数据...")
    print("="*60)
    
    train_df = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))
    test_df = pd.read_csv(os.path.join(BASE_DIR, 'test.csv'))
    
    print(f"训练集: {train_df.shape}")
    print(f"测试集: {test_df.shape}")
    print(f"正样本比例: {train_df['label'].mean():.4f}")
    
    return train_df, test_df


def create_features(train_df, test_df):
    """V4增强版特征工程"""
    print("="*60)
    print("特征工程 V4...")
    print("="*60)
    
    # 保存关键列
    test_sample_index = test_df['sample_index'].copy() if 'sample_index' in test_df.columns else None
    y_train = train_df['label'].copy()
    
    train_len = len(train_df)
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    # ==========================================
    # 1. 缺失值标记特征
    # ==========================================
    print("\n1. 创建缺失值标记特征...")
    high_missing_cols = ['price', 'online_days', 'user_displayed_item_num', 'distance', 
                         'item_ave_price', 'level', 'cate_1', 'cate_2', 'cate_3']
    for col in high_missing_cols:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isnull().astype(np.int8)
    
    # ==========================================
    # 2. 时间特征 (V4优化: 移除无效的sin/cos编码)
    # ==========================================
    print("\n2. 创建时间特征 (V4: 移除无效sin/cos)...")
    if 'timestamp' in df.columns:
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        
        # V4: 不再使用sin/cos (在V3中重要性为0)
        # 改用更直接的时间段特征
        
        # 是否周末
        df['is_weekend'] = (df['weekday'].isin([6, 7])).astype(np.int8)
        
        # 时段分类 (更细粒度)
        def get_time_period(hour):
            if 6 <= hour < 9:
                return 1  # 早高峰
            elif 9 <= hour < 11:
                return 2  # 上午
            elif 11 <= hour < 14:
                return 3  # 午餐
            elif 14 <= hour < 17:
                return 4  # 下午
            elif 17 <= hour < 20:
                return 5  # 晚高峰/晚餐
            elif 20 <= hour < 23:
                return 6  # 晚间
            else:
                return 0  # 深夜/凌晨
        
        df['time_period'] = df['hour'].apply(get_time_period).astype(np.int8)
        
        # 是否工作日高峰时段
        df['is_rush_hour'] = ((df['hour'].isin([7, 8, 9, 17, 18, 19])) & 
                               (~df['is_weekend'].astype(bool))).astype(np.int8)
        
        # 是否用餐时间
        df['is_meal_time'] = df['hour'].isin([11, 12, 13, 17, 18, 19, 20]).astype(np.int8)
        
        df.drop(['timestamp', 'timestamp_dt'], axis=1, inplace=True)
    
    # ==========================================
    # 3. Count Encoding
    # ==========================================
    print("\n3. 创建Count Encoding特征...")
    count_cols = ['userid', 'itemid', 'geohash', 'cityid', 'loc_cityid', 
                  'cate_1', 'cate_2', 'cate_3', 'dtype']
    
    train_part = df.iloc[:train_len]
    for col in count_cols:
        if col in df.columns:
            counts = train_part[col].value_counts()
            df[f'{col}_count'] = df[col].map(counts).fillna(0).astype(np.float32)
            df[f'{col}_count_log'] = np.log1p(df[f'{col}_count']).astype(np.float32)
    
    # ==========================================
    # 4. V4新增: 用户/物品历史行为统计特征
    # ==========================================
    print("\n4. 创建用户/物品历史行为统计特征 (V4新增 - 修复泄露)...")
    
    # ⚠️ 修复：只使用曝光次数（count），不使用label相关的统计（会导致泄露）
    # user_ctr 和 item_ctr 需要通过 K-fold Target Encoding 来处理，而不是直接计算
    
    # 用户曝光次数（不使用label，无泄露）
    user_impressions = train_df.groupby('userid').size().reset_index(name='user_impressions')
    df = df.merge(user_impressions, on='userid', how='left')
    df['user_impressions'] = df['user_impressions'].fillna(0).astype(np.float32)
    df['user_impressions_log'] = np.log1p(df['user_impressions']).astype(np.float32)
    
    # 物品曝光次数（不使用label，无泄露）
    item_impressions = train_df.groupby('itemid').size().reset_index(name='item_impressions')
    df = df.merge(item_impressions, on='itemid', how='left')
    df['item_impressions'] = df['item_impressions'].fillna(0).astype(np.float32)
    df['item_impressions_log'] = np.log1p(df['item_impressions']).astype(np.float32)
    
    # 用户是否是新用户（只有一次曝光）
    df['is_new_user'] = (df['user_impressions'] <= 1).astype(np.int8)
    
    # 物品是否是新物品
    df['is_new_item'] = (df['item_impressions'] <= 1).astype(np.int8)
    
    # 用户/物品的活跃度分箱（无泄露）
    df['user_activity_bin'] = pd.cut(df['user_impressions'], 
                                      bins=[-1, 1, 2, 5, 10, 100],
                                      labels=[0, 1, 2, 3, 4]).astype(float).fillna(0).astype(np.int8)
    df['item_popularity_bin'] = pd.cut(df['item_impressions'],
                                        bins=[-1, 1, 2, 5, 10, 100],
                                        labels=[0, 1, 2, 3, 4]).astype(float).fillna(0).astype(np.int8)
    
    # ==========================================
    # 5. V4新增: 排名特征
    # ==========================================
    print("\n5. 创建排名特征 (V4新增)...")
    
    # 物品在类别内的曝光排名
    if 'cate_1' in df.columns:
        # 类别内物品热度（曝光数）排名
        cate_item_rank = train_df.groupby(['cate_1', 'itemid']).size().reset_index(name='cate_item_count')
        cate_item_rank['cate_item_rank'] = cate_item_rank.groupby('cate_1')['cate_item_count'].rank(
            method='dense', ascending=False
        )
        cate_item_rank['cate_item_rank_pct'] = cate_item_rank.groupby('cate_1')['cate_item_count'].rank(
            method='dense', pct=True
        )
        df = df.merge(cate_item_rank[['cate_1', 'itemid', 'cate_item_rank', 'cate_item_rank_pct']], 
                      on=['cate_1', 'itemid'], how='left')
        df['cate_item_rank'] = df['cate_item_rank'].fillna(9999).astype(np.float32)
        df['cate_item_rank_pct'] = df['cate_item_rank_pct'].fillna(0.5).astype(np.float32)
    
    # 城市内物品热度排名
    if 'cityid' in df.columns:
        city_item_rank = train_df.groupby(['cityid', 'itemid']).size().reset_index(name='city_item_count')
        city_item_rank['city_item_rank_pct'] = city_item_rank.groupby('cityid')['city_item_count'].rank(
            method='dense', pct=True
        )
        df = df.merge(city_item_rank[['cityid', 'itemid', 'city_item_rank_pct']], 
                      on=['cityid', 'itemid'], how='left')
        df['city_item_rank_pct'] = df['city_item_rank_pct'].fillna(0.5).astype(np.float32)
    
    # ==========================================
    # 6. 交叉特征 (V4增强)
    # ==========================================
    print("\n6. 创建交叉特征 (V4增强)...")
    
    # 城市匹配
    if 'cityid' in df.columns and 'loc_cityid' in df.columns:
        df['is_same_city'] = (df['cityid'] == df['loc_cityid']).astype(np.int8)
    
    # 字符串组合交叉
    cross_pairs = [
        ('gender', 'dtype'), ('age', 'dtype'), ('level', 'dtype'),
        ('gender', 'cate_1'), ('age', 'cate_1'), ('level', 'cate_1'),
        ('time_period', 'dtype'), ('time_period', 'cate_1'),
        ('cityid', 'cate_1'), ('cityid', 'dtype'),
        ('weekday', 'time_period'),  # V4新增
        ('is_weekend', 'time_period'),  # V4新增
    ]
    
    for c1, c2 in cross_pairs:
        if c1 in df.columns and c2 in df.columns:
            new_col = f'{c1}_{c2}'
            df[new_col] = df[c1].astype(str) + '_' + df[c2].astype(str)

    # ==========================================
    # 7. 数值特征处理 (V4增强)
    # ==========================================
    print("\n7. 处理数值特征 (V4增强)...")
    
    numerical_cols = ['distance', 'item_ave_price', 'price', 'user_home_dis', 'user_work_dis',
                      'temp', 'temp_low', 'temp_high', 'user_displayed_item_num', 'online_days']
    
    # 计算训练集的统计量用于填充
    train_part_for_stats = df.iloc[:train_len]
    
    # 填充缺失值
    for col in numerical_cols:
        if col in df.columns:
            median_val = train_part_for_stats[col].median()
            df[col] = df[col].fillna(median_val).astype(np.float32)
    
    # 对数变换
    log_cols = ['distance', 'price', 'item_ave_price', 'user_home_dis', 'user_work_dis']
    for col in log_cols:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(np.maximum(df[col], 0)).astype(np.float32)

    # 相对特征
    if 'price' in df.columns and 'cate_1' in df.columns:
        cate1_price_mean = df.groupby('cate_1')['price'].transform('mean')
        cate1_price_std = df.groupby('cate_1')['price'].transform('std').fillna(1)
        df['price_relative_to_cate1'] = (df['price'] / (cate1_price_mean + 1)).astype(np.float32)
        df['price_zscore_cate1'] = ((df['price'] - cate1_price_mean) / (cate1_price_std + 0.01)).astype(np.float32)

    if 'price' in df.columns and 'cityid' in df.columns:
        city_price_mean = df.groupby('cityid')['price'].transform('mean')
        df['price_relative_to_city'] = (df['price'] / (city_price_mean + 1)).astype(np.float32)
        
    if 'user_displayed_item_num' in df.columns and 'level' in df.columns:
        level_disp_mean = df.groupby('level')['user_displayed_item_num'].transform('mean')
        df['disp_relative_to_level'] = (df['user_displayed_item_num'] / (level_disp_mean + 1)).astype(np.float32)

    # V4新增: 距离相对特征
    if 'distance' in df.columns and 'cate_1' in df.columns:
        cate1_dist_mean = df.groupby('cate_1')['distance'].transform('mean')
        df['dist_relative_to_cate1'] = (df['distance'] / (cate1_dist_mean + 1)).astype(np.float32)

    # 温度相关
    if 'temp' in df.columns:
        df['temp_comfort'] = ((df['temp'] >= 17) & (df['temp'] <= 26)).astype(np.int8)
    
    if 'temp_high' in df.columns and 'temp_low' in df.columns:
        df['temp_range'] = (df['temp_high'] - df['temp_low']).astype(np.float32)
    
    # 分箱
    if 'distance' in df.columns:
        df['distance_bin'] = pd.cut(df['distance'], bins=[-1, 10, 20, 40, 60, 80, 200], 
                                     labels=[0, 1, 2, 3, 4, 5]).astype(float).fillna(3).astype(np.int8)
    
    if 'price' in df.columns:
        df['price_bin'] = pd.cut(df['price'], bins=[-1, 15, 30, 50, 75, 200], 
                                  labels=[0, 1, 2, 3, 4]).astype(float).fillna(2).astype(np.int8)
    
    # V4: 用户行为强度分箱已在前面创建 (user_activity_bin)
    
    # ==========================================
    # 8. V4新增: 统计聚合特征
    # ==========================================
    print("\n8. 创建统计聚合特征 (V4新增)...")
    
    # 类别内价格统计
    if 'price' in df.columns and 'cate_1' in df.columns:
        cate1_price_stats = train_part_for_stats.groupby('cate_1')['price'].agg(['mean', 'std', 'min', 'max'])
        cate1_price_stats.columns = ['cate1_price_mean', 'cate1_price_std', 'cate1_price_min', 'cate1_price_max']
        df = df.merge(cate1_price_stats, on='cate_1', how='left')
        for col in cate1_price_stats.columns:
            df[col] = df[col].fillna(train_part_for_stats['price'].median()).astype(np.float32)
    
    # 城市内距离统计
    if 'distance' in df.columns and 'cityid' in df.columns:
        city_dist_stats = train_part_for_stats.groupby('cityid')['distance'].agg(['mean', 'std'])
        city_dist_stats.columns = ['city_dist_mean', 'city_dist_std']
        df = df.merge(city_dist_stats, on='cityid', how='left')
        df['city_dist_mean'] = df['city_dist_mean'].fillna(train_part_for_stats['distance'].median()).astype(np.float32)
        df['city_dist_std'] = df['city_dist_std'].fillna(1).astype(np.float32)
    
    # ==========================================
    # 9. 类别特征处理
    # ==========================================
    print("\n9. 处理类别特征...")
    
    categorical_features = [
        'userid', 'itemid', 'geohash', 'cityid', 'loc_cityid', 
        'weekday', 'hour', 'weather', 'dtype', 'cate_1', 'cate_2', 'cate_3',
        'age', 'level', 'gender', 'married', 'job', 'has_car', 
        'work_geohash', 'mobile_type', 'mobile_os',
        'is_weekend', 'time_period', 'is_same_city', 'temp_comfort',
        'distance_bin', 'price_bin',
        'is_rush_hour', 'is_meal_time', 'is_new_user', 'is_new_item',
        'user_activity_bin', 'item_popularity_bin'  # 修正：使用无泄露的分箱特征
    ]
    
    # 添加交叉特征
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
    
    del df, train_part, train_part_for_stats
    gc.collect()
    
    # ==========================================
    # 10. Target Encoding (V4增强: 更多交叉组合)
    # ==========================================
    print("\n10. 创建Target Encoding特征 (V4增强)...")
    
    # 单列CTR（使用K-fold Target Encoding避免泄露）
    # 注意：userid/itemid 因为太稀疏，CTR 可能效果有限，但不会导致泄露
    single_ctr_cols = ['dtype', 'cate_1', 'cate_2', 'cate_3', 'cityid', 
                       'weather', 'time_period', 'age', 'level', 'gender',
                       'price_bin', 'distance_bin', 'hour', 'weekday',
                       'is_rush_hour', 'is_meal_time',
                       'user_activity_bin', 'item_popularity_bin']  # 新增活跃度分箱的CTR
    single_ctr_cols = [col for col in single_ctr_cols if col in train_processed.columns]
    
    encoder = AdvancedCTREncoder(n_folds=5, smoothing=20)
    
    # 准备特征和标签
    exclude_cols = ['sample_index', 'label']
    feature_cols = [col for col in train_processed.columns if col not in exclude_cols]
    
    X_train = train_processed[feature_cols].copy()
    X_test = test_processed[[col for col in feature_cols if col in test_processed.columns]].copy()
    
    # 单列Target Encoding
    for col in single_ctr_cols:
        X_train = encoder.fit_transform_single(X_train, y_train, col)
        X_test = encoder.transform_single(X_test, col)
    
    # V4新增: 交叉组合Target Encoding
    cross_ctr_cols = [
        ['cityid', 'cate_1'],
        ['cityid', 'dtype'],
        ['gender', 'cate_1'],
        ['age', 'cate_1'],
        ['level', 'cate_1'],
        ['time_period', 'dtype'],
        ['weekday', 'hour'],
    ]
    
    for cols in cross_ctr_cols:
        if all(c in X_train.columns for c in cols):
            X_train = encoder.fit_transform_single(X_train, y_train, cols)
            X_test = encoder.transform_single(X_test, '_'.join(cols))
    
    # 恢复sample_index
    if test_sample_index is not None:
        X_test['sample_index'] = test_sample_index.values
    
    # 更新categorical_features列表
    categorical_features = [col for col in categorical_features if col in X_train.columns]
    
    # 内存优化
    X_train = reduce_mem_usage(X_train)
    X_test = reduce_mem_usage(X_test)
    
    print(f"\n最终特征数量: {len([c for c in X_train.columns if c not in exclude_cols])}")
    
    del train_processed, test_processed
    gc.collect()
    
    return X_train, y_train, X_test, categorical_features


def cross_validation(X, y, categorical_features, n_splits=10, resume=True):
    """K折交叉验证，支持检查点续训（内存优化版）"""
    print("="*60)
    print(f"开始{n_splits}折交叉验证...")
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
    # 内存优化：不在内存中保留所有模型，只保存模型路径
    model_paths = []
    checkpoint_dir = get_checkpoint_dir()
    
    # V4优化后的模型参数
    params = {
        'task_type': 'GPU',
        'iterations': 15000,  # 增加迭代上限
        'learning_rate': 0.03,  # 进一步降低学习率
        'depth': 8,
        'l2_leaf_reg': 8.0,  # 增加正则化
        'random_strength': 1e-5,
        'border_count': 128,
        'thread_count': -1,
        'random_seed': 42,
        'verbose': 100,
        'eval_metric': 'AUC',
        'loss_function': 'Logloss',
        'auto_class_weights': 'Balanced',
        'boosting_type': 'Plain',
        'one_hot_max_size': 10,
        'nan_mode': 'Min',
        'od_type': 'Iter',
        'od_wait': 500,  # 增加early stopping等待
        'bagging_temperature': 0.5,
        'gpu_ram_part': 0.95,
        'allow_writing_files': False,
        'use_best_model': True,
        'min_data_in_leaf': 80,  # 增加叶子节点最小样本
        'grow_policy': 'SymmetricTree',
    }
    
    exclude_cols = ['sample_index', 'label']
    feature_cols = [col for col in X.columns if col not in exclude_cols]
    
    for fold in range(n_splits):
        print(f"\n{'='*60}")
        print(f"第{fold+1}/{n_splits}折")
        print(f"{'='*60}")
        
        model_path = os.path.join(checkpoint_dir, f"fold_{fold+1}.cbm")
        meta_path = os.path.join(checkpoint_dir, f"fold_{fold+1}.json")
        model_paths.append(model_path)
        
        # 检查是否已完成该折（需要同时存在模型和元数据）
        if resume and os.path.exists(model_path) and os.path.exists(meta_path):
            print(f"检测到第{fold+1}折已完成，跳过...")
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            cv_scores.append(meta['auc'])
            print(f"第{fold+1}折 AUC: {meta['auc']:.4f} (从检查点读取)")
            continue
        
        val_indices = folds[fold]
        train_indices = []
        for i in range(n_splits):
            if i != fold:
                train_indices.extend(folds[i])
        
        X_val_fold = X.iloc[val_indices][feature_cols].copy()
        y_val_fold = y.iloc[val_indices].copy()
        
        # 确保类别特征为字符串
        for col in categorical_features:
            if col in X_val_fold.columns:
                X_val_fold[col] = X_val_fold[col].astype(str)
        
        # 检查是否只需要补评估（模型存在但元数据不存在）
        if resume and os.path.exists(model_path):
            print(f"检测到第{fold+1}折模型，补充评估...")
            model = CatBoostClassifier()
            model.load_model(model_path)
        else:
            X_train_fold = X.iloc[train_indices][feature_cols].copy()
            y_train_fold = y.iloc[train_indices].copy()
            
            print(f"训练集: {X_train_fold.shape}, 正样本: {y_train_fold.mean():.4f}")
            print(f"验证集: {X_val_fold.shape}, 正样本: {y_val_fold.mean():.4f}")
            
            for col in categorical_features:
                if col in X_train_fold.columns:
                    X_train_fold[col] = X_train_fold[col].astype(str)
            
            train_pool = Pool(X_train_fold, y_train_fold, cat_features=categorical_features)
            val_pool = Pool(X_val_fold, y_val_fold, cat_features=categorical_features)
            
            del X_train_fold, y_train_fold
            gc.collect()
            
            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=val_pool, use_best_model=True, plot=False)
            
            model.save_model(model_path)
            print(f"模型已保存: {model_path}")
            
            del train_pool, val_pool
            gc.collect()
        
        # 评估
        val_pool = Pool(X_val_fold, y_val_fold, cat_features=categorical_features)
        y_pred_proba = model.predict_proba(val_pool)[:, 1]
        auc = roc_auc_score(y_val_fold, y_pred_proba)
        print(f"最佳迭代: {model.get_best_iteration()}")
        save_fold_metadata(fold, auc, model, meta_path)
        cv_scores.append(auc)
        
        print(f"第{fold+1}折 AUC: {auc:.4f}")
        
        # 内存优化：释放模型和数据
        del model, val_pool, X_val_fold, y_val_fold
        gc.collect()
        
        # 显式清理GPU内存
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print("交叉验证结果汇总")
    print(f"{'='*60}")
    print(f"各折AUC: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"平均AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"最大AUC: {max(cv_scores):.4f}")
    print(f"最小AUC: {min(cv_scores):.4f}")
    print(f"{'='*60}\n")
    
    # 返回模型路径列表而非模型对象
    return model_paths, cv_scores


def weighted_ensemble_predict(model_paths, cv_scores, X_test, categorical_features):
    """V4: 加权集成预测（根据各折AUC加权）- 内存优化版"""
    print("加权集成预测 (V4) - 逐个加载模型...")
    
    exclude_cols = ['sample_index', 'label']
    feature_cols = [col for col in X_test.columns if col not in exclude_cols]
    
    X_test_features = X_test[feature_cols].copy()
    for col in categorical_features:
        if col in X_test_features.columns:
            X_test_features[col] = X_test_features[col].astype(str)
    
    test_pool = Pool(X_test_features, cat_features=categorical_features)
    
    # 计算权重（基于AUC，性能越好权重越大）
    scores = np.array(cv_scores)
    # 使用softmax计算权重
    weights = np.exp((scores - scores.min()) * 100)  # 放大差异
    weights = weights / weights.sum()
    print(f"集成权重: {[f'{w:.4f}' for w in weights]}")
    
    # 内存优化：逐个加载模型进行预测，不同时保留所有模型
    ensemble_pred = np.zeros(X_test.shape[0])
    simple_sum_pred = np.zeros(X_test.shape[0])
    
    for i, model_path in enumerate(model_paths):
        print(f"  加载模型 {i+1}/{len(model_paths)}: {os.path.basename(model_path)}")
        model = CatBoostClassifier()
        model.load_model(model_path)
        
        pred = model.predict_proba(test_pool)[:, 1]
        ensemble_pred += pred * weights[i]
        simple_sum_pred += pred
        
        # 立即释放模型
        del model
        gc.collect()
    
    # 简单平均
    simple_avg_pred = simple_sum_pred / len(model_paths)
    
    del test_pool, X_test_features
    gc.collect()
    
    return ensemble_pred, simple_avg_pred


def save_results(model_paths, cv_scores, predictions, simple_predictions, X_test):
    """保存结果 - 内存优化版"""
    print("保存结果...")

    output_dir = get_output_dir()
    
    # 保存加权集成预测
    submission = pd.DataFrame({
        'sample_index': range(len(predictions)),
        'label': predictions
    })
    submission_path = os.path.join(output_dir, 'submission_v4.csv')
    submission.to_csv(submission_path, index=False)
    print(f"预测结果已保存到 {submission_path}")
    
    # 保存简单平均预测（备用）
    submission_simple = pd.DataFrame({
        'sample_index': range(len(simple_predictions)),
        'label': simple_predictions
    })
    submission_simple_path = os.path.join(output_dir, 'submission_v4_simple.csv')
    submission_simple.to_csv(submission_simple_path, index=False)
    print(f"简单平均预测结果已保存到 {submission_simple_path}")
    
    # 保存“主模型”指针：选择AUC最高的一折作为代表模型（不再往 /home 写大文件）
    best_idx = int(np.argmax(cv_scores))
    best_model_path = model_paths[best_idx]
    main_model_path = os.path.join(output_dir, 'catboost_v4.cbm')
    saved_main_model_path = safe_link_or_copy(best_model_path, main_model_path)
    print(f"主模型已保存到 {saved_main_model_path} (best fold={best_idx+1}, auc={cv_scores[best_idx]:.4f})")
    
    # 临时加载模型获取参数和特征重要性
    model = CatBoostClassifier()
    model.load_model(best_model_path)
    
    results = {
        'metrics': {
            'auc_mean': float(np.mean(cv_scores)),
            'auc_std': float(np.std(cv_scores)),
            'auc_max': float(max(cv_scores)),
            'auc_min': float(min(cv_scores)),
            'fold_scores': [float(s) for s in cv_scores]
        },
        'version': 'V4',
        'optimizations': [
            '用户/物品历史行为统计特征',
            '排名特征（类别内、城市内热度排名）',
            '交叉组合Target Encoding',
            '移除无效sin/cos周期编码',
            '加权集成预测',
            '更多统计聚合特征',
            '优化模型参数',
            '内存优化（逐折释放）'
        ],
        'model_params': model.get_all_params()
    }
    
    def convert(o):
        if isinstance(o, np.float32): return float(o)
        if isinstance(o, np.int64): return int(o)
        raise TypeError
    
    results_path = os.path.join(output_dir, 'model_results_v4.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4, default=convert)
    
    print(f"评估结果已保存到 {results_path}")
    
    # 特征重要性
    importance = model.get_feature_importance()
    feature_cols = [col for col in X_test.columns if col not in ['sample_index', 'label']]
    
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\n前30个重要特征:")
    print(importance_df.head(30).to_string(index=False))
    
    importance_path = os.path.join(output_dir, 'feature_importance_v4.csv')
    importance_df.to_csv(importance_path, index=False)
    print(f"特征重要性已保存到 {importance_path}")
    
    # 释放模型
    del model
    gc.collect()


def main():
    """主函数"""
    np.random.seed(42)
    start_time = time.time()
    
    print("="*60)
    print("CatBoost V4 训练 - 深度优化版")
    print("="*60)
    
    train_df, test_df = load_data()
    
    X_train, y_train, X_test, categorical_features = create_features(train_df, test_df)
    
    del train_df, test_df
    gc.collect()
    
    models, cv_scores = cross_validation(X_train, y_train, categorical_features, n_splits=10)
    
    predictions, simple_predictions = weighted_ensemble_predict(
        models, cv_scores, X_test, categorical_features
    )
    
    save_results(models, cv_scores, predictions, simple_predictions, X_test)
    
    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed/60:.1f} 分钟")
    print("V4版CatBoost训练完成！")


if __name__ == "__main__":
    main()

