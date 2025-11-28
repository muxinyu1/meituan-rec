#!/usr/bin/env python3
"""
LightGBM 训练脚本 (用于模型融合)
特点：Label Encoding、内存优化、K折交叉验证
"""

import pandas as pd
import numpy as np
import gc
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import time
import warnings

# 忽略 LightGBM 的一些冗余警告
warnings.filterwarnings('ignore')

def load_data():
    """加载数据"""
    print("加载数据...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    return train_df, test_df

def reduce_mem_usage(df):
    """内存优化"""
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
    print(f'内存优化: {start_mem:.2f}MB -> {end_mem:.2f}MB')
    return df

def preprocess_data(train_df, test_df):
    print("数据预处理...")
    
    # 记录索引以便恢复
    test_sample_index = test_df['sample_index'].copy() if 'sample_index' in test_df.columns else None
    
    # --- 1. 基础特征工程 (保持与CatBoost一致) ---
    count_features = ['userid', 'itemid', 'geohash', 'cityid', 'loc_cityid']
    for col in count_features:
        if col in train_df.columns:
            counts = train_df[col].value_counts()
            train_df[f'{col}_count'] = train_df[col].map(counts).fillna(0).astype('float32')
            test_df[f'{col}_count'] = test_df[col].map(counts).fillna(0).astype('float32')

    # 时间特征
    for df in [train_df, test_df]:
        if 'timestamp' in df.columns:
            df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
            df['hour'] = df['timestamp_dt'].dt.hour
            df['weekday'] = df['timestamp_dt'].dt.weekday
            # 保留 sin/cos 特征
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24).astype(np.float32)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24).astype(np.float32)
            df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7).astype(np.float32)
            df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7).astype(np.float32)
            df.drop(['timestamp', 'timestamp_dt'], axis=1, inplace=True)

    # --- 2. 针对 LightGBM 的类别编码 (Label Encoding) ---
    categorical_features = [
        'userid', 'itemid', 'geohash', 'cityid', 'loc_cityid', 
        'weekday', 'hour', 'weather', 'dtype', 'cate_1', 'cate_2', 'cate_3',
        'age', 'level', 'gender', 'married', 'job', 'has_car', 
        'work_geohash', 'mobile_type', 'mobile_os'
    ]
    categorical_features = [col for col in categorical_features if col in train_df.columns]

    print("执行 Label Encoding...")
    for col in categorical_features:
        # 将 train 和 test 拼接在一起进行编码，防止编码不一致
        # 填充 NaN 为 'nan' 字符串
        train_df[col] = train_df[col].astype(str).fillna('unknown')
        test_df[col] = test_df[col].astype(str).fillna('unknown')
        
        le = LabelEncoder()
        # 拟合全部数据
        all_values = list(train_df[col].unique()) + list(test_df[col].unique())
        le.fit(all_values)
        
        # 转换
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
        
        # 转为 category 类型，LGBM 会自动处理
        train_df[col] = train_df[col].astype('category')
        test_df[col] = test_df[col].astype('category')

    # 内存优化
    train_df = reduce_mem_usage(train_df)
    test_df = reduce_mem_usage(test_df)
    
    # 准备输出
    exclude_cols = ['sample_index', 'label']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df['label']
    X_test = test_df[feature_cols]
    
    return X_train, y_train, X_test, feature_cols, categorical_features

def train_lgb_cv(X, y, X_test, categorical_features, n_splits=5):
    """LightGBM K折交叉验证"""
    
    # 获取weekday用于分层
    weekdays = X['weekday'] if 'weekday' in X.columns else np.zeros(len(X))
    unique_weekdays = sorted(weekdays.unique())
    print(f"Stratified by Weekday: {unique_weekdays}")

    # 自定义分层索引逻辑 (和CatBoost脚本保持一致)
    folds = [[] for _ in range(n_splits)]
    for wd in unique_weekdays:
        idx = X[X['weekday'] == wd].index.tolist()
        np.random.shuffle(idx)
        fold_size = len(idx) // n_splits
        for i in range(n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < n_splits - 1 else len(idx)
            folds[i].extend(idx[start:end])
    
    # LGBM 参数
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,        # 较低的学习率
        'num_leaves': 64,             # 叶子节点数，控制复杂度
        'max_depth': -1,              # 限制深度，防止过拟合
        'min_child_samples': 50,      # 叶子节点最少样本数
        'feature_fraction': 0.8,      # 特征采样
        'bagging_fraction': 0.8,      # 数据采样
        'bagging_freq': 5,            # 每5轮做一次bagging
        'lambda_l1': 0.1,             # L1正则
        'lambda_l2': 0.2,             # L2正则
        'n_jobs': -1,
        'seed': 42,
        'verbose': -1,
        # 'device': 'gpu',            # 如果有GPU，取消注释
        # 'gpu_platform_id': 0,
        # 'gpu_device_id': 0
    }
    
    oof_preds = np.zeros(X.shape[0])
    test_preds = np.zeros(X_test.shape[0])
    cv_scores = []
    
    print(f"开始 {n_splits} 折训练...")
    
    for fold in range(n_splits):
        val_idx = folds[fold]
        train_idx = []
        for f in range(n_splits):
            if f != fold: train_idx.extend(folds[f])
            
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # 创建 Dataset
        dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=categorical_features)
        dval = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_features, reference=dtrain)
        
        # 回调函数
        callbacks = [
            lgb.log_evaluation(period=100),
            lgb.early_stopping(stopping_rounds=100)
        ]
        
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=10000,
            valid_sets=[dtrain, dval],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )
        
        # 预测验证集
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred
        
        # 预测测试集
        test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits
        
        score = roc_auc_score(y_val, val_pred)
        cv_scores.append(score)
        print(f"Fold {fold+1} AUC: {score:.4f}")
        
        # 内存清理
        del X_tr, y_tr, X_val, y_val, dtrain, dval, model
        gc.collect()
        
    print(f"\nCV Mean AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    return test_preds, oof_preds

def main():
    train_df, test_df = load_data()
    
    # 预处理
    X_train, y_train, X_test, feature_cols, categorical_features = preprocess_data(train_df, test_df)
    
    # 训练
    lgb_preds, lgb_oof = train_lgb_cv(X_train, y_train, X_test, categorical_features, n_splits=10)
    
    # 保存结果
    submission = pd.DataFrame({
        'sample_index': range(len(lgb_preds)),
        'label': lgb_preds
    })
    submission.to_csv('submission_lgb.csv', index=False)
    print("LightGBM 预测结果已保存至 submission_lgb.csv")
    
    # 保存 OOF 结果用于 Stacking (进阶融合)
    # oof_df = pd.DataFrame({'oof_pred': lgb_oof})
    # oof_df.to_csv('oof_lgb.csv', index=False)

if __name__ == "__main__":
    main()