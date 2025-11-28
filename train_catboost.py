#!/usr/bin/env python3
"""
CatBoost训练脚本 (优化版)
优化点：内存管理、特征工程(Count Encoding)、数据类型压缩
使用K折交叉验证进行训练
"""

import pandas as pd
import numpy as np
import gc
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
import time
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns


def cross_validation(X, y, categorical_features, n_splits=5):
    """K折交叉验证，按照weekday进行分层抽样"""
    print(f"进行{n_splits}折交叉验证...")
    
    # 获取weekday
    weekdays = X['weekday'].unique()
    print(f"训练数据中的weekday值: {sorted(weekdays)}")

    folds = [[] for _ in range(n_splits)]
    
    for weekday in sorted(weekdays):
        weekday_indices = X[X['weekday'] == weekday].index.tolist()
        np.random.shuffle(weekday_indices)
        
        # 将当前weekday的数据均匀分配到各折
        fold_size = len(weekday_indices) // n_splits
        for i in range(n_splits):
            start_idx = i * fold_size
            if i < n_splits - 1:
                end_idx = start_idx + fold_size
            else:
                end_idx = len(weekday_indices)  # 最后一折包含剩余所有样本
            folds[i].extend(weekday_indices[start_idx:end_idx])
    
    # 打乱每折内的顺序
    for fold_indices in folds:
        np.random.shuffle(fold_indices)
    
    cv_scores = []
    models = []
    
    for fold in range(n_splits):
        print(f"\n{'='*60}")
        print(f"第{fold+1}/{n_splits}折交叉验证")
        print(f"{'='*60}")
        
        # 验证集索引
        val_indices = folds[fold]
        
        # 训练集索引（其他所有折）
        train_indices = []
        for i in range(n_splits):
            if i != fold:
                train_indices.extend(folds[i])
        
        # 验证weekday分布
        X_train_fold = X.iloc[train_indices].copy()
        y_train_fold = y.iloc[train_indices].copy()
        X_val_fold = X.iloc[val_indices].copy()
        y_val_fold = y.iloc[val_indices].copy()
        
        print(f"训练集大小: {X_train_fold.shape}, 正样本比例: {y_train_fold.mean():.4f}")
        print(f"验证集大小: {X_val_fold.shape}, 正样本比例: {y_val_fold.mean():.4f}")
        
        # 打印weekday分布
        print(f"训练集weekday分布:\n{X_train_fold['weekday'].value_counts().sort_index()}")
        print(f"验证集weekday分布:\n{X_val_fold['weekday'].value_counts().sort_index()}")
        
        # 确保类别特征为字符串
        for col in categorical_features:
            if col in X_train_fold.columns:
                X_train_fold[col] = X_train_fold[col].astype(str)
            if col in X_val_fold.columns:
                X_val_fold[col] = X_val_fold[col].astype(str)
        
        # 训练模型（传入验证集）
        model, _ = train_model(
            X_train_fold, y_train_fold, 
            categorical_features,
            X_val=X_val_fold,
            y_val=y_val_fold
        )
        
        # 预测验证集
        val_pool = Pool(X_val_fold, y_val_fold, cat_features=categorical_features)
        y_pred_proba = model.predict_proba(val_pool)[:, 1]
        
        # 计算AUC
        auc = roc_auc_score(y_val_fold, y_pred_proba)
        cv_scores.append(auc)
        models.append(model)
        
        print(f"\n第{fold+1}折最终验证集AUC: {auc:.4f}")
        
        del val_pool, X_train_fold, y_train_fold, X_val_fold, y_val_fold
        gc.collect()
    
    mean_auc = np.mean(cv_scores)
    std_auc = np.std(cv_scores)
    
    print(f"\n{'='*60}")
    print(f"交叉验证结果汇总")
    print(f"{'='*60}")
    print(f"各折AUC: {[f'{score:.4f}' for score in cv_scores]}")
    print(f"平均AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"最大AUC: {max(cv_scores):.4f}")
    print(f"最小AUC: {min(cv_scores):.4f}")
    print(f"{'='*60}\n")
    
    return models, cv_scores

def train_model(X_train, y_train, categorical_features, params=None, X_val=None, y_val=None):
    """训练CatBoost模型，支持验证集监控"""
    print("准备数据池 (Pool)...")
    
    train_pool = Pool(X_train, y_train, cat_features=categorical_features)
    
    # 如果提供了验证集，创建验证集Pool
    eval_set = None
    if X_val is not None and y_val is not None:
        eval_set = Pool(X_val, y_val, cat_features=categorical_features)
        print(f"验证集Pool已创建，大小: {X_val.shape}")
    
    # 优化：Pool创建后，原始DataFrame不再需要，立即释放内存
    print("Pool 已创建，释放原始 DataFrame 内存...")
    del X_train, y_train
    if X_val is not None:
        del X_val, y_val
    gc.collect()
    
    print("开始训练 CatBoost 模型...")
    
    if params is None:
        params = {
            'task_type': 'GPU',
            'iterations': 15000,
            'learning_rate': 0.06260215315409043,
            'depth': 10,
            'l2_leaf_reg': 3.6916543587180826,
            'random_strength': 5.030516129056867e-07,
            'border_count': 128,
            'thread_count': -1,
            'random_seed': 42,
            'verbose': 100,  # 每100轮打印一次
            'eval_metric': 'AUC',
            'loss_function': 'Logloss',
            'auto_class_weights': 'Balanced',
            'boosting_type': 'Ordered',
            'one_hot_max_size': 2,
            'nan_mode': 'Min',
            'od_type': 'Iter',
            'od_wait': 500,
            'bagging_temperature': 0.23983928972206614,
            
            # 显存优化参数
            'gpu_ram_part': 0.95,
            'allow_writing_files': False,
            
            # **关键修改：如果有验证集，使用early stopping**
            'use_best_model': True if eval_set is not None else False,
            'metric_period': 50  # 每50轮打印一次指标（而不是默认的5）
        }
    
    model = CatBoostClassifier(**params)
    
    # 传入验证集
    if eval_set is not None:
        model.fit(
            train_pool, 
            eval_set=eval_set,
            use_best_model=True,
            plot=False
        )
    else:
        model.fit(train_pool, use_best_model=False)
    
    print(f"最佳迭代轮数: {model.get_best_iteration()}")
    
    # 清理验证集Pool
    if eval_set is not None:
        del eval_set
        gc.collect()
    
    return model, time.time()

def ensemble_predict(models, X_test, categorical_features):
    """集成预测"""
    print("集成预测...")
    
    predictions = np.zeros((X_test.shape[0], len(models)))
    test_pool = Pool(X_test, cat_features=categorical_features)
    
    for i, model in enumerate(models):
        predictions[:, i] = model.predict_proba(test_pool)[:, 1]
    
    # 简单平均
    ensemble_pred = np.mean(predictions, axis=1)
    
    del test_pool
    gc.collect()
    
    return ensemble_pred

def load_data():
    """加载训练和测试数据"""
    print("加载数据...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    print(f"训练集大小: {train_df.shape}")
    print(f"测试集大小: {test_df.shape}")
    
    return train_df, test_df

def reduce_mem_usage(df):
    """
    遍历 DataFrame 的所有列，修改数据类型以减少内存使用。
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'内存优化前: {start_mem:.2f} MB')
    
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
                # CatBoost GPU 建议使用 float32 这里的兼容性最好
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'内存优化后: {end_mem:.2f} MB (减少了 {100 * (start_mem - end_mem) / start_mem:.1f}%)')
    return df

def preprocess_data(train_df, test_df):
    print("数据预处理...")
    
    # 保存test的sample_index
    test_sample_index = test_df['sample_index'].copy() if 'sample_index' in test_df.columns else None
    
    label_col = 'label'
    exclude_cols = ['sample_index', label_col]

    count_features = ['userid', 'itemid', 'geohash', 'cityid', 'loc_cityid']
    for col in count_features:
        if col in train_df.columns:
            # 只在训练集上统计
            counts = train_df[col].value_counts()
            
            # 分别应用到训练集和测试集
            train_df[f'{col}_count'] = train_df[col].map(counts).fillna(0).astype('float32')
            test_df[f'{col}_count'] = test_df[col].map(counts).fillna(0).astype('float32')
    
    # 处理 timestamp - 这个可以在各自数据集上做
    for df in [train_df, test_df]:
        if 'timestamp' in df.columns:
            df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
            df['hour_sin'] = np.sin(2 * np.pi * df['timestamp_dt'].dt.hour / 24).astype(np.float32)
            df['hour_cos'] = np.cos(2 * np.pi * df['timestamp_dt'].dt.hour / 24).astype(np.float32)
            df['weekday_sin'] = np.sin(2 * np.pi * df['timestamp_dt'].dt.weekday / 7).astype(np.float32)
            df['weekday_cos'] = np.cos(2 * np.pi * df['timestamp_dt'].dt.weekday / 7).astype(np.float32)
            df.drop(['timestamp', 'timestamp_dt'], axis=1, inplace=True)
    
    # 内存优化
    train_df = reduce_mem_usage(train_df)
    test_df = reduce_mem_usage(test_df)
    
    # 类别特征处理
    categorical_features = [
        'userid', 'itemid', 'geohash', 'cityid', 'loc_cityid', 
        'weekday', 'hour', 'weather', 'dtype', 'cate_1', 'cate_2', 'cate_3',
        'age', 'level', 'gender', 'married', 'job', 'has_car', 
        'work_geohash', 'mobile_type', 'mobile_os'
    ]
    categorical_features = [col for col in categorical_features if col in train_df.columns]
    
    print("清洗类别特征...")
    for col in categorical_features:
        for df in [train_df, test_df]:
            if col in df.columns:
                # CatBoost可以处理NaN，不需要填充-1
                # 直接转为str即可
                df[col] = df[col].astype(str)
    
    # 提取特征和标签
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df[label_col].copy()
    X_test = test_df[[col for col in feature_cols if col in test_df.columns]].copy()
    
    # 恢复test的sample_index
    if test_sample_index is not None:
        X_test['sample_index'] = test_sample_index.values
    
    del train_df, test_df
    gc.collect()
    
    return X_train, y_train, X_test, feature_cols, categorical_features

def evaluate_model(model, X_val, y_val):
    """评估模型性能"""
    print("评估模型性能...")
    
    # 预测概率
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # 预测类别
    y_pred = model.predict(X_val)
    
    # 计算评估指标
    auc = roc_auc_score(y_val, y_pred_proba)
    logloss = log_loss(y_val, y_pred_proba)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    print(f"AUC: {auc:.4f}")
    print(f"LogLoss: {logloss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    metrics = {
        'auc': auc,
        'logloss': logloss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics, y_pred_proba

def feature_importance(model, feature_cols, top_n=20):
    """分析特征重要性"""
    print("分析特征重要性...")
    
    # 获取特征重要性
    importance = model.get_feature_importance()
    
    # 创建特征重要性DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # 打印前N个重要特征
    print(f"前{top_n}个重要特征:")
    print(feature_importance_df.head(top_n))
    
    # 保存特征重要性
    feature_importance_df.to_csv('feature_importance.csv', index=False)
    
    # 绘制特征重要性图
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(top_n))
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    return feature_importance_df

def predict_test(model, X_test):
    """对测试集进行预测"""
    print("对测试集进行预测...")
    
    # 预测概率
    test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 创建提交文件
    submission = pd.DataFrame({
        'sample_index': range(len(test_pred_proba)),
        'label': test_pred_proba
    })
    
    # 保存预测结果
    submission.to_csv('submission.csv', index=False)
    print(f"预测结果已保存到 submission.csv")
    
    return test_pred_proba

def save_model(model, metrics, training_time):
    """保存模型和评估结果"""
    print("保存模型和评估结果...")
    
    # 保存模型
    model.save_model('catboost_model.cbm')
    print("模型已保存到 catboost_model.cbm")
    
    # 保存评估结果
    results = {
        'metrics': metrics,
        'training_time': training_time,
        'model_params': model.get_all_params()
    }
    
    # JSON 不支持 float32，需要转换
    def convert(o):
        if isinstance(o, np.float32): return float(o)
        raise TypeError
        
    with open('model_results.json', 'w') as f:
        json.dump(results, f, indent=4, default=convert)
    
    print("评估结果已保存到 model_results.json")

def main():
    """主函数"""
    # 设置随机种子
    np.random.seed(42)
    
    # 加载数据
    train_df, test_df = load_data()
    
    # 数据预处理
    X_train, y_train, X_test, feature_cols, categorical_features = preprocess_data(
        train_df, test_df
    )
    
    # 打印优化参数信息
    print("="*60)
    print("应用 Optuna 优化参数 (已锁定)")
    print("="*60)
    
    # K折交叉验证
    models, cv_scores = cross_validation(X_train, y_train, categorical_features, n_splits=10)
    
    # 集成预测
    test_pred_proba = ensemble_predict(models, X_test, categorical_features)
    
    # 保存预测结果
    submission = pd.DataFrame({
        'sample_index': range(len(test_pred_proba)),
        'label': test_pred_proba
    })
    submission.to_csv('submission.csv', index=False)
    print(f"预测结果已保存到 submission.csv")
    
    # 保存模型（保存第一个模型作为示例）
    if models:
        model = models[0]
        training_time = time.time()
        save_model(model, {'auc_mean': np.mean(cv_scores), 'auc_std': np.std(cv_scores)}, training_time)
    
    print("CatBoost模型K折交叉验证训练完成！")

if __name__ == "__main__":
    main()
