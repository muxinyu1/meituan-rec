#!/usr/bin/env python3
"""
CatBoost训练脚本 V3 (深度优化版)
优化点：
1. 继承 V2 的所有优点 (Target Encoding, CTR特征, 基础交叉特征)
2. 新增相对统计特征 (Relative Features): 价格相对于类别/城市的偏差
3. 新增对数数值特征: 处理长尾分布
4. 增强交叉特征: 用户属性与物品属性的二阶交叉
5. 进一步微调模型参数
"""

import pandas as pd
import numpy as np
import gc
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import time
import os
import json
import warnings
warnings.filterwarnings('ignore')


class CTRFeatureEncoder:
    """基于K折的Target Encoding编码器，避免数据泄露"""
    
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
    
    base_dir = "/media/wenhao/7d8f46ef-9674-40b1-a999-148b79a69954/zhaoshanhui/meituan-rec"
    train_df = pd.read_csv(os.path.join(base_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(base_dir, 'test.csv'))
    
    print(f"训练集: {train_df.shape}")
    print(f"测试集: {test_df.shape}")
    print(f"正样本比例: {train_df['label'].mean():.4f}")
    
    return train_df, test_df


def create_features(train_df, test_df):
    """V3增强版特征工程"""
    print("="*60)
    print("特征工程 V3...")
    print("="*60)
    
    # 保存关键列
    test_sample_index = test_df['sample_index'].copy() if 'sample_index' in test_df.columns else None
    y_train = train_df['label'].copy()
    
    train_len = len(train_df)
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    # ==========================================
    # 1. 缺失值标记特征 (V2继承)
    # ==========================================
    print("\n1. 创建缺失值标记特征...")
    high_missing_cols = ['price', 'online_days', 'user_displayed_item_num', 'distance', 
                         'item_ave_price', 'level', 'cate_1', 'cate_2', 'cate_3']
    for col in high_missing_cols:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isnull().astype(np.int8)
    
    # ==========================================
    # 2. 时间特征 (V2继承)
    # ==========================================
    print("\n2. 创建时间特征...")
    if 'timestamp' in df.columns:
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        
        # 周期性编码
        df['hour_sin'] = np.sin(2 * np.pi * df['timestamp_dt'].dt.hour / 24).astype(np.float32)
        df['hour_cos'] = np.cos(2 * np.pi * df['timestamp_dt'].dt.hour / 24).astype(np.float32)
        df['weekday_sin'] = np.sin(2 * np.pi * df['timestamp_dt'].dt.weekday / 7).astype(np.float32)
        df['weekday_cos'] = np.cos(2 * np.pi * df['timestamp_dt'].dt.weekday / 7).astype(np.float32)
        
        # 是否周末
        df['is_weekend'] = (df['weekday'].isin([6, 7])).astype(np.int8)
        
        # 时段分类
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
        
        df.drop(['timestamp', 'timestamp_dt'], axis=1, inplace=True)
    
    # ==========================================
    # 3. Count Encoding (V2继承)
    # ==========================================
    print("\n3. 创建Count Encoding特征...")
    count_cols = ['userid', 'itemid', 'geohash', 'cityid', 'loc_cityid', 
                  'cate_1', 'cate_2', 'cate_3', 'dtype']
    
    train_part = df.iloc[:train_len]
    for col in count_cols:
        if col in df.columns:
            counts = train_part[col].value_counts()
            df[f'{col}_count'] = df[col].map(counts).fillna(0).astype(np.float32)
            
            # 对数变换
            df[f'{col}_count_log'] = np.log1p(df[f'{col}_count']).astype(np.float32)
    
    # ==========================================
    # 4. 交叉特征 (V3增强)
    # ==========================================
    print("\n4. 创建交叉特征 (V3增强)...")
    
    # 城市匹配
    if 'cityid' in df.columns and 'loc_cityid' in df.columns:
        df['is_same_city'] = (df['cityid'] == df['loc_cityid']).astype(np.int8)
    
    # 字符串组合交叉 (CatBoost会自动处理)
    # 用户属性 x 物品属性
    cross_pairs = [
        ('userid', 'cate_1'), ('userid', 'cate_2'), 
        ('cityid', 'cate_1'), ('cityid', 'cate_2'),
        ('time_period', 'cate_1'), ('time_period', 'dtype'),
        ('gender', 'cate_1'), ('age', 'cate_1'), ('level', 'cate_1'),
        ('gender', 'dtype'), ('age', 'dtype')
    ]
    
    for c1, c2 in cross_pairs:
        if c1 in df.columns and c2 in df.columns:
            new_col = f'{c1}_{c2}'
            df[new_col] = df[c1].astype(str) + '_' + df[c2].astype(str)

    # ==========================================
    # 5. 数值特征处理 (V3增强: 相对特征 & 对数)
    # ==========================================
    print("\n5. 处理数值特征 (V3增强)...")
    
    numerical_cols = ['distance', 'item_ave_price', 'price', 'user_home_dis', 'user_work_dis',
                      'temp', 'temp_low', 'temp_high', 'user_displayed_item_num', 'online_days']
    
    # 填充缺失值
    for col in numerical_cols:
        if col in df.columns:
            median_val = train_part[col].median()
            df[col] = df[col].fillna(median_val).astype(np.float32)
    
    # 对数变换 (V3新增: 处理长尾分布)
    log_cols = ['distance', 'price', 'item_ave_price', 'user_home_dis', 'user_work_dis']
    for col in log_cols:
        if col in df.columns:
            # log1p(x) = log(x + 1)
            # 确保非负
            df[f'{col}_log'] = np.log1p(np.maximum(df[col], 0)).astype(np.float32)

    # 相对特征 (V3新增: 价格相对于类别均值/城市均值的偏差)
    # 注意: 为了避免数据泄露，均值统计应该只在训练集上计算，或者使用全局统计（如果假设测试集分布一致）
    # 这里我们使用全局统计（基于整个df），但更严谨的做法是只用训练集统计。
    # 为了方便，这里使用全局groupby，但这是无监督的统计特征，通常可以接受。
    
    if 'price' in df.columns and 'cate_1' in df.columns:
        # 计算每个类别的平均价格
        cate1_price_mean = df.groupby('cate_1')['price'].transform('mean')
        df['price_relative_to_cate1'] = (df['price'] / (cate1_price_mean + 1)).astype(np.float32)
        df['price_diff_cate1'] = (df['price'] - cate1_price_mean).astype(np.float32)

    if 'price' in df.columns and 'cityid' in df.columns:
        # 计算每个城市的平均价格
        city_price_mean = df.groupby('cityid')['price'].transform('mean')
        df['price_relative_to_city'] = (df['price'] / (city_price_mean + 1)).astype(np.float32)
        
    if 'user_displayed_item_num' in df.columns and 'level' in df.columns:
        # 不同用户等级的平均浏览量差异
        level_disp_mean = df.groupby('level')['user_displayed_item_num'].transform('mean')
        df['disp_relative_to_level'] = (df['user_displayed_item_num'] / (level_disp_mean + 1)).astype(np.float32)

    # 温度相关
    if 'temp' in df.columns:
        df['temp_comfort'] = ((df['temp'] >= 17) & (df['temp'] <= 26)).astype(np.int8)
    
    if 'temp_high' in df.columns and 'temp_low' in df.columns:
        df['temp_range'] = (df['temp_high'] - df['temp_low']).astype(np.float32)
    
    # 分箱
    if 'distance' in df.columns:
        df['distance_bin'] = pd.cut(df['distance'], bins=[-1, 20, 40, 60, 80, 200], 
                                     labels=[0, 1, 2, 3, 4]).astype(float).fillna(2).astype(np.int8)
    
    if 'price' in df.columns:
        df['price_bin'] = pd.cut(df['price'], bins=[-1, 20, 40, 60, 80, 200], 
                                  labels=[0, 1, 2, 3, 4]).astype(float).fillna(2).astype(np.int8)
    
    # ==========================================
    # 6. 类别特征处理
    # ==========================================
    print("\n6. 处理类别特征...")
    
    categorical_features = [
        'userid', 'itemid', 'geohash', 'cityid', 'loc_cityid', 
        'weekday', 'hour', 'weather', 'dtype', 'cate_1', 'cate_2', 'cate_3',
        'age', 'level', 'gender', 'married', 'job', 'has_car', 
        'work_geohash', 'mobile_type', 'mobile_os',
        'is_weekend', 'time_period', 'is_same_city', 'temp_comfort',
        'distance_bin', 'price_bin'
    ]
    
    # 添加交叉特征到类别列表
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
    # 7. Target Encoding (V2继承)
    # ==========================================
    print("\n7. 创建Target Encoding特征...")
    
    ctr_cols = ['dtype', 'cate_1', 'cate_2', 'cate_3', 'cityid', 
                'weather', 'time_period', 'age', 'level', 'gender',
                'price_bin', 'distance_bin'] # 新增一些CTR
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
    
    print(f"\n最终特征数量: {len([c for c in X_train.columns if c not in exclude_cols])}")
    
    del train_processed, test_processed
    gc.collect()
    
    return X_train, y_train, X_test, categorical_features


def cross_validation(X, y, categorical_features, n_splits=10):
    """K折交叉验证"""
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
    models = []
    
    # 优化后的模型参数 V3
    params = {
        'task_type': 'GPU',
        'iterations': 12000,  # 稍微增加迭代上限
        'learning_rate': 0.04,  # 稍微降低学习率以适应更多特征
        'depth': 8,
        'l2_leaf_reg': 7.0,  # 增加正则化以防止由于新特征导致的过拟合
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
        'bagging_temperature': 0.4, # 增加一点随机性
        'gpu_ram_part': 0.95,
        'allow_writing_files': False,
        'use_best_model': True,
        'min_data_in_leaf': 60,
    }
    
    exclude_cols = ['sample_index', 'label']
    feature_cols = [col for col in X.columns if col not in exclude_cols]
    
    for fold in range(n_splits):
        print(f"\n{'='*60}")
        print(f"第{fold+1}/{n_splits}折")
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
        
        print(f"训练集: {X_train_fold.shape}, 正样本: {y_train_fold.mean():.4f}")
        print(f"验证集: {X_val_fold.shape}, 正样本: {y_val_fold.mean():.4f}")
        
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
        
        print(f"最佳迭代: {model.get_best_iteration()}")
        
        y_pred_proba = model.predict_proba(val_pool)[:, 1]
        auc = roc_auc_score(y_val_fold, y_pred_proba)
        cv_scores.append(auc)
        models.append(model)
        
        print(f"第{fold+1}折 AUC: {auc:.4f}")
        
        del train_pool, val_pool, X_val_fold, y_val_fold
        gc.collect()
    
    print(f"\n{'='*60}")
    print("交叉验证结果汇总")
    print(f"{'='*60}")
    print(f"各折AUC: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"平均AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"最大AUC: {max(cv_scores):.4f}")
    print(f"最小AUC: {min(cv_scores):.4f}")
    print(f"{'='*60}\n")
    
    return models, cv_scores


def ensemble_predict(models, X_test, categorical_features):
    """集成预测"""
    print("集成预测...")
    
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
    print("保存结果...")
    
    submission = pd.DataFrame({
        'sample_index': range(len(predictions)),
        'label': predictions
    })
    submission.to_csv('submission_v3.csv', index=False)
    print("预测结果已保存到 submission_v3.csv")
    
    models[0].save_model('catboost_v3.cbm')
    print("模型已保存到 catboost_v3.cbm")
    
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
    
    with open('model_results_v3.json', 'w') as f:
        json.dump(results, f, indent=4, default=convert)
    
    print("评估结果已保存到 model_results_v3.json")
    
    importance = models[0].get_feature_importance()
    feature_cols = [col for col in X_test.columns if col not in ['sample_index', 'label']]
    
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\n前20个重要特征:")
    print(importance_df.head(20).to_string(index=False))
    
    importance_df.to_csv('feature_importance_v3.csv', index=False)


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
    print(f"\n总耗时: {elapsed/60:.1f} 分钟")
    print("V3版CatBoost训练完成！")


if __name__ == "__main__":
    main()
