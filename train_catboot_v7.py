#!/usr/bin/env python3
"""
CatBoost训练脚本 V7

需求：不再划分验证集，使用全量训练集训练；每个 epoch（迭代）都对测试集生成预测 submission。
说明：
- 特征工程沿用 V3，含 Target Encoding。
- 训练一次，训练结束后按迭代数逐步生成提交文件（文件较多，请注意磁盘空间）。
"""

import gc
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')


class CTRFeatureEncoder:
    """基于K折的Target Encoding编码器，避免数据泄露"""
    
    def __init__(self, cols, n_folds=5, smoothing=20, random_state=42):
        self.cols = cols
        self.n_folds = n_folds
        self.smoothing = smoothing
        self.random_state = random_state
        self.global_mean = None
        self.encodings = {}
    
    def fit_transform(self, X, y):
        result = X.copy()
        self.global_mean = y.mean()
        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        for col in self.cols:
            if col not in X.columns:
                continue
            col_name = f'{col}_ctr'
            result[col_name] = np.nan
            for train_idx, val_idx in kf.split(X, y):
                train_data = pd.DataFrame({'col': X.iloc[train_idx][col], 'label': y.iloc[train_idx]})
                stats = train_data.groupby('col')['label'].agg(['sum', 'count'])
                smooth_ctr = (stats['sum'] + self.smoothing * self.global_mean) / (stats['count'] + self.smoothing)
                result.iloc[val_idx, result.columns.get_loc(col_name)] = X.iloc[val_idx][col].map(smooth_ctr)
            result[col_name] = result[col_name].fillna(self.global_mean)
            full_data = pd.DataFrame({'col': X[col], 'label': y})
            full_stats = full_data.groupby('col')['label'].agg(['sum', 'count'])
            self.encodings[col] = (full_stats['sum'] + self.smoothing * self.global_mean) / (full_stats['count'] + self.smoothing)
        return result
    
    def transform(self, X):
        result = X.copy()
        for col in self.cols:
            if col not in X.columns or col not in self.encodings:
                continue
            col_name = f'{col}_ctr'
            result[col_name] = X[col].map(self.encodings[col]).fillna(self.global_mean)
        return result


def reduce_mem_usage(df):
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
    print("="*60)
    print("加载数据...")
    print("="*60)
    # 使用脚本所在目录作为默认数据目录，避免硬编码失效
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_df = pd.read_csv(os.path.join(base_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(base_dir, 'test.csv'))
    print(f"训练集: {train_df.shape}")
    print(f"测试集: {test_df.shape}")
    print(f"正样本比例: {train_df['label'].mean():.4f}")
    return train_df, test_df


def create_features(train_df, test_df):
    print("="*60)
    print("特征工程 V7 (同V3)...")
    print("="*60)
    test_sample_index = train_df.get('sample_index')  # placeholder to avoid lint
    test_sample_index = test_df['sample_index'].copy() if 'sample_index' in test_df.columns else None
    y_train = train_df['label'].copy()
    
    train_len = len(train_df)
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    print("\n1. 缺失值标记特征...")
    high_missing_cols = ['price', 'online_days', 'user_displayed_item_num', 'distance', 
                         'item_ave_price', 'level', 'cate_1', 'cate_2', 'cate_3']
    for col in high_missing_cols:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isnull().astype(np.int8)
    
    print("\n2. 时间特征...")
    if 'timestamp' in df.columns:
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        df['hour_sin'] = np.sin(2 * np.pi * df['timestamp_dt'].dt.hour / 24).astype(np.float32)
        df['hour_cos'] = np.cos(2 * np.pi * df['timestamp_dt'].dt.hour / 24).astype(np.float32)
        df['weekday_sin'] = np.sin(2 * np.pi * df['timestamp_dt'].dt.weekday / 7).astype(np.float32)
        df['weekday_cos'] = np.cos(2 * np.pi * df['timestamp_dt'].dt.weekday / 7).astype(np.float32)
        df['is_weekend'] = (df['weekday'].isin([6, 7])).astype(np.int8)
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
        df['time_period'] = df['hour'].apply(get_time_period).astype(np.int8)
        df.drop(['timestamp', 'timestamp_dt'], axis=1, inplace=True)
    
    print("\n3. Count Encoding...")
    count_cols = ['userid', 'itemid', 'geohash', 'cityid', 'loc_cityid', 'cate_1', 'cate_2', 'cate_3', 'dtype']
    train_part = df.iloc[:train_len]
    for col in count_cols:
        if col in df.columns:
            counts = train_part[col].value_counts()
            df[f'{col}_count'] = df[col].map(counts).fillna(0).astype(np.float32)
            df[f'{col}_count_log'] = np.log1p(df[f'{col}_count']).astype(np.float32)
    
    print("\n4. 交叉特征...")
    cross_pairs = [
        ('userid', 'cate_1'), ('userid', 'cate_2'), 
        ('cityid', 'cate_1'), ('cityid', 'cate_2'),
        ('time_period', 'cate_1'), ('time_period', 'dtype'),
        ('gender', 'cate_1'), ('age', 'cate_1'), ('level', 'cate_1'),
        ('gender', 'dtype'), ('age', 'dtype')
    ]
    if 'cityid' in df.columns and 'loc_cityid' in df.columns:
        df['is_same_city'] = (df['cityid'] == df['loc_cityid']).astype(np.int8)
    for c1, c2 in cross_pairs:
        if c1 in df.columns and c2 in df.columns:
            df[f'{c1}_{c2}'] = df[c1].astype(str) + '_' + df[c2].astype(str)

    print("\n5. 数值特征处理...")
    numerical_cols = ['distance', 'item_ave_price', 'price', 'user_home_dis', 'user_work_dis',
                      'temp', 'temp_low', 'temp_high', 'user_displayed_item_num', 'online_days']
    for col in numerical_cols:
        if col in df.columns:
            median_val = train_part[col].median()
            df[col] = df[col].fillna(median_val).astype(np.float32)
    log_cols = ['distance', 'price', 'item_ave_price', 'user_home_dis', 'user_work_dis']
    for col in log_cols:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(np.maximum(df[col], 0)).astype(np.float32)
    if 'price' in df.columns and 'cate_1' in df.columns:
        cate1_price_mean = df.groupby('cate_1')['price'].transform('mean')
        df['price_relative_to_cate1'] = (df['price'] / (cate1_price_mean + 1)).astype(np.float32)
        df['price_diff_cate1'] = (df['price'] - cate1_price_mean).astype(np.float32)
    if 'price' in df.columns and 'cityid' in df.columns:
        city_price_mean = df.groupby('cityid')['price'].transform('mean')
        df['price_relative_to_city'] = (df['price'] / (city_price_mean + 1)).astype(np.float32)
    if 'user_displayed_item_num' in df.columns and 'level' in df.columns:
        level_disp_mean = df.groupby('level')['user_displayed_item_num'].transform('mean')
        df['disp_relative_to_level'] = (df['user_displayed_item_num'] / (level_disp_mean + 1)).astype(np.float32)
    if 'temp' in df.columns:
        df['temp_comfort'] = ((df['temp'] >= 17) & (df['temp'] <= 26)).astype(np.int8)
    if 'temp_high' in df.columns and 'temp_low' in df.columns:
        df['temp_range'] = (df['temp_high'] - df['temp_low']).astype(np.float32)
    if 'distance' in df.columns:
        df['distance_bin'] = pd.cut(df['distance'], bins=[-1, 20, 40, 60, 80, 200], 
                                     labels=[0, 1, 2, 3, 4]).astype(float).fillna(2).astype(np.int8)
    if 'price' in df.columns:
        df['price_bin'] = pd.cut(df['price'], bins=[-1, 20, 40, 60, 80, 200], 
                                  labels=[0, 1, 2, 3, 4]).astype(float).fillna(2).astype(np.int8)
    
    print("\n6. 类别特征处理...")
    categorical_features = [
        'userid', 'itemid', 'geohash', 'cityid', 'loc_cityid', 
        'weekday', 'hour', 'weather', 'dtype', 'cate_1', 'cate_2', 'cate_3',
        'age', 'level', 'gender', 'married', 'job', 'has_car', 
        'work_geohash', 'mobile_type', 'mobile_os',
        'is_weekend', 'time_period', 'is_same_city', 'temp_comfort',
        'distance_bin', 'price_bin'
    ]
    for c1, c2 in cross_pairs:
        col_name = f'{c1}_{c2}'
        if col_name in df.columns:
            categorical_features.append(col_name)
    categorical_features = [col for col in categorical_features if col in df.columns]
    for col in categorical_features:
        df[col] = df[col].fillna(-1).astype(str)

    train_processed = df.iloc[:train_len].copy()
    test_processed = df.iloc[train_len:].copy()
    del df, train_part
    gc.collect()

    print("\n7. Target Encoding...")
    ctr_cols = ['dtype', 'cate_1', 'cate_2', 'cate_3', 'cityid', 
                'weather', 'time_period', 'age', 'level', 'gender',
                'price_bin', 'distance_bin']
    ctr_cols = [col for col in ctr_cols if col in train_processed.columns]
    encoder = CTRFeatureEncoder(cols=ctr_cols, n_folds=5, smoothing=20)
    exclude_cols = ['sample_index', 'label']
    feature_cols = [col for col in train_processed.columns if col not in exclude_cols]
    X_train = train_processed[feature_cols].copy()
    X_test = test_processed[[col for col in feature_cols if col in test_processed.columns]].copy()
    X_train = encoder.fit_transform(X_train, y_train)
    X_test = encoder.transform(X_test)
    if test_sample_index is not None:
        X_test['sample_index'] = test_sample_index.values
    categorical_features = [col for col in categorical_features if col in X_train.columns]
    X_train = reduce_mem_usage(X_train)
    X_test = reduce_mem_usage(X_test)
    print(f"\n最终特征数量: {len([c for c in X_train.columns if c not in exclude_cols])}")
    del train_processed, test_processed
    gc.collect()
    return X_train, y_train, X_test, categorical_features


def train_full_data(X, y, categorical_features, iterations=1200):
    print("="*60)
    print("开始全量训练 (无验证集)...")
    print("="*60)
    exclude_cols = ['sample_index', 'label']
    feature_cols = [col for col in X.columns if col not in exclude_cols]
    X_train = X[feature_cols].copy()
    for col in categorical_features:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(str)
    train_pool = Pool(X_train, y, cat_features=categorical_features)
    params = {
        'task_type': 'GPU',
        'iterations': iterations,
        'learning_rate': 0.04,
        'depth': 8,
        'l2_leaf_reg': 7.0,
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
        'od_wait': 400,
        'bagging_temperature': 0.4,
        'gpu_ram_part': 0.95,
        'allow_writing_files': False,
        'use_best_model': False,
        'min_data_in_leaf': 60,
    }
    model = CatBoostClassifier(**params)
    model.fit(train_pool)
    del train_pool, X_train
    gc.collect()
    return model


def save_submission_per_epoch(model, X_test, categorical_features, output_dir='submissions_v7'):
    print("逐迭代生成提交文件...（每5轮保存一次，末轮必保存）")
    os.makedirs(output_dir, exist_ok=True)
    exclude_cols = ['sample_index', 'label']
    feature_cols = [col for col in X_test.columns if col not in exclude_cols]
    X_test_features = X_test[feature_cols].copy()
    for col in categorical_features:
        if col in X_test_features.columns:
            X_test_features[col] = X_test_features[col].astype(str)
    test_pool = Pool(X_test_features, cat_features=categorical_features)
    sample_index = X_test['sample_index'] if 'sample_index' in X_test.columns else pd.Series(range(len(X_test_features)))
    tree_count = model.tree_count_
    for i in range(1, tree_count + 1):
        if (i % 5 != 0) and (i != tree_count):
            continue
        preds = model.predict_proba(test_pool, ntree_end=i)[:, 1]
        submission = pd.DataFrame({'sample_index': sample_index, 'label': preds})
        path = os.path.join(output_dir, f'submission_v7_iter{i}.csv')
        submission.to_csv(path, index=False)
        if i % 50 == 0 or i == tree_count:
            print(f"已生成 {i}/{tree_count} 个提交文件")
    del test_pool, X_test_features
    gc.collect()


def save_final_artifacts(model, X_test, cv_scores=None):
    print("保存最终模型与特征重要度...")
    model.save_model('catboost_v7.cbm')
    importance = model.get_feature_importance()
    feature_cols = [col for col in X_test.columns if col not in ['sample_index', 'label']]
    importance_df = pd.DataFrame({'feature': feature_cols, 'importance': importance}).sort_values('importance', ascending=False)
    importance_df.to_csv('feature_importance_v7.csv', index=False)
    print("前20重要特征:")
    print(importance_df.head(20).to_string(index=False))
    results = {
        'metrics': {
            'note': 'No validation in V7; full-data training only'
        },
        'model_params': model.get_all_params()
    }
    with open('model_results_v7.json', 'w') as f:
        json.dump(results, f, indent=4)


def main():
    np.random.seed(42)
    start_time = time.time()
    train_df, test_df = load_data()
    X_train, y_train, X_test, categorical_features = create_features(train_df, test_df)
    del train_df, test_df
    gc.collect()
    model = train_full_data(X_train, y_train, categorical_features, iterations=1200)
    save_submission_per_epoch(model, X_test, categorical_features, output_dir='submissions_v7')
    save_final_artifacts(model, X_test)
    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed/60:.1f} 分钟")
    print("V7版CatBoost训练完成！")


if __name__ == "__main__":
    main()