#!/usr/bin/env python3
"""
CatBoost训练脚本 + Optuna 自动调参
用于训练推荐系统模型
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, precision_score, recall_score, f1_score
import time
import os
import json
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import optuna  # ✅ 新增：引入 Optuna
from optuna.integration import CatBoostPruningCallback # ✅ 新增：用于剪枝（提前停止不好的尝试）

def load_data():
    """加载训练、验证和测试数据"""
    print("加载数据...")
    # 这里假设文件存在，实际运行时请确保文件路径正确
    train_df = pd.read_csv('train.csv')
    val_df = pd.read_csv('val.csv')
    test_df = pd.read_csv('test.csv')
    
    print(f"训练集大小: {train_df.shape}")
    print(f"验证集大小: {val_df.shape}")
    print(f"测试集大小: {test_df.shape}")
    
    return train_df, val_df, test_df

def preprocess_data(train_df, val_df, test_df):
    print("数据预处理...")
    
    label_col = 'label'
    exclude_cols = ['sample_index', label_col] 
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df[label_col].copy()
    X_val = val_df[feature_cols].copy()
    y_val = val_df[label_col].copy()
    X_test = test_df[feature_cols].copy()
    
    categorical_features = [
        'userid', 'itemid', 'geohash', 'cityid', 'loc_cityid', 
        'weekday', 'hour', 'weather', 'dtype', 'cate_1', 'cate_2', 'cate_3',
        'age', 'level', 'gender', 'married', 'job', 'has_car', 
        'work_geohash', 'mobile_type', 'mobile_os'
    ]
    categorical_features = [col for col in categorical_features if col in feature_cols]
    
    for col in categorical_features:
        X_train[col] = X_train[col].fillna('MISSING').astype(str)
        X_val[col] = X_val[col].fillna('MISSING').astype(str)
        X_test[col] = X_test[col].fillna('MISSING').astype(str)
    
    for df in [X_train, X_val, X_test]:
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        df['hour_sin'] = np.sin(2 * np.pi * df['timestamp_dt'].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['timestamp_dt'].dt.hour / 24)
        df['weekday_sin'] = np.sin(2 * np.pi * df['timestamp_dt'].dt.weekday / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['timestamp_dt'].dt.weekday / 7)
        df.drop(['timestamp', 'timestamp_dt'], axis=1, inplace=True)
    
    feature_cols = list(X_train.columns)
    categorical_features = [col for col in categorical_features if col in feature_cols]
    
    print(f"特征数量: {len(feature_cols)}")
    print(f"类别特征数量: {len(categorical_features)}")
    
    return X_train, y_train, X_val, y_val, X_test, feature_cols, categorical_features

# ✅ 修改：增加 best_params 参数，允许接收 Optuna 调优后的参数
def train_model(X_train, y_train, X_val, y_val, categorical_features, best_params=None):
    print("训练CatBoost模型...")
    
    train_pool = Pool(X_train, y_train, cat_features=categorical_features)
    val_pool = Pool(X_val, y_val, cat_features=categorical_features)
    
    # 默认基础参数
    params = {
        'task_type': 'GPU',
        'iterations': 15000,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 5.0,
        'random_strength': 2.0,
        'border_count': 128,
        'thread_count': -1,
        'random_seed': 42,
        'verbose': 100,
        'eval_metric': 'AUC',
        'loss_function': 'Logloss',
        'auto_class_weights': 'Balanced',
        'boosting_type': 'Ordered',
        'one_hot_max_size': 2,
        'nan_mode': 'Min',
        
        # ✅ 修正：统一只使用 od_wait
        'od_type': 'Iter',
        'od_wait': 300, 
    }
    
    # ✅ 如果有优化的参数，进行覆盖
    if best_params:
        print(f"应用 Optuna 优化参数: {best_params}")
        # 如果 best_params 里包含 early_stopping_rounds，必须删掉它，避免冲突
        if 'early_stopping_rounds' in best_params:
            params['od_wait'] = best_params.pop('early_stopping_rounds')
        
        params.update(best_params)
    
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    
    print(f"最佳迭代轮数: {model.get_best_iteration()}")
    return model, time.time()
def optimize_hyperparameters(X_train, y_train, X_val, y_val, categorical_features, n_trials=20):
    print(f"开始 Optuna 超参数优化 (Trials={n_trials}) [GPU模式 - 无剪枝]...")
    
    train_pool = Pool(X_train, y_train, cat_features=categorical_features)
    val_pool = Pool(X_val, y_val, cat_features=categorical_features)

    def objective(trial):
        param = {
            'task_type': 'GPU',
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'auto_class_weights': 'Balanced',
            'verbose': 0, 
            
            # 调参时限制最大轮数，防止耗时过长
            'iterations': 2000, 
            
            # ✅ 关键：既然不能用 Optuna 剪枝，就必须依赖 CatBoost 内部早停
            # 如果 50 轮内 AUC 没有提升，就自己停下来
            'od_type': 'Iter',
            'od_wait': 50,
            
            # 参数搜索空间
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 1e-9, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'border_count': trial.suggest_categorical('border_count', [128, 254]),
            'boosting_type': 'Ordered'
        }

        model = CatBoostClassifier(**param)
        
        # ❌ 删除：pruning_callback = CatBoostPruningCallback(trial, "AUC")
        
        # 直接训练，不带 callbacks
        model.fit(
            train_pool,
            eval_set=val_pool,
            verbose=0
        )
        
        # ❌ 删除：pruning_callback.check_pruned()
        
        return model.best_score_['validation']['AUC']

    # 创建 Study
    study = optuna.create_study(
        direction='maximize',
        # 由于无法使用回调剪枝，这里其实不需要特定的 pruner，但留着也不影响
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    study.optimize(objective, n_trials=n_trials)

    print("\nOptuna 优化完成!")
    # 检查是否依然是 0.0
    if study.best_value == 0.0:
        print("⚠️ 警告：最佳 AUC 为 0.0，调参可能完全失败。")
        
    return study.best_params

def evaluate_model(model, X_val, y_val):
    """评估模型性能"""
    print("评估模型性能...")
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)
    
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
        'auc': auc, 'logloss': logloss, 'accuracy': accuracy,
        'precision': precision, 'recall': recall, 'f1': f1
    }
    return metrics, y_pred_proba

def feature_importance(model, feature_cols, top_n=20):
    """分析特征重要性"""
    print("分析特征重要性...")
    importance = model.get_feature_importance()
    feature_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(f"前{top_n}个重要特征:")
    print(feature_importance_df.head(top_n))
    
    feature_importance_df.to_csv('feature_importance.csv', index=False)
    return feature_importance_df

def predict_test(model, X_test):
    """对测试集进行预测"""
    print("对测试集进行预测...")
    test_pred_proba = model.predict_proba(X_test)[:, 1]
    submission = pd.DataFrame({
        'sample_index': range(len(test_pred_proba)),
        'label': test_pred_proba
    })
    submission.to_csv('submission.csv', index=False)
    print(f"预测结果已保存到 submission.csv")
    return test_pred_proba

def save_model(model, metrics, training_time):
    """保存模型和评估结果"""
    print("保存模型和评估结果...")
    model.save_model('catboost_model.cbm')
    results = {
        'metrics': metrics,
        'training_time': training_time,
        'model_params': model.get_all_params()
    }
    # 转换 float32 为 float 以便 json 序列化
    def convert(o):
        if isinstance(o, np.float32): return float(o)
        raise TypeError
    
    with open('model_results.json', 'w') as f:
        json.dump(results, f, indent=4, default=convert)
    print("评估结果已保存到 model_results.json")

def main():
    np.random.seed(42)
    train_df, val_df, test_df = load_data()
    X_train, y_train, X_val, y_val, X_test, feature_cols, categorical_features = preprocess_data(
        train_df, val_df, test_df
    )
    
    RUN_OPTUNA = True
    best_params = None
    
    if RUN_OPTUNA:
        best_params = optimize_hyperparameters(
            X_train, y_train, X_val, y_val, 
            categorical_features, 
            n_trials=20
        )
    
    if best_params:
        # 确保最终训练轮数足够
        best_params['iterations'] = 15000  # 增加到 15000
        # ✅ 修正：不要用 early_stopping_rounds，统一用 od_wait
        best_params['od_wait'] = 500       
    
    model, training_time = train_model(
        X_train, y_train, X_val, y_val, 
        categorical_features, 
        best_params=best_params
    )
    
    metrics, y_pred_proba = evaluate_model(model, X_val, y_val)
    feature_importance_df = feature_importance(model, feature_cols)
    test_pred_proba = predict_test(model, X_test)
    save_model(model, metrics, training_time)
    
    print("CatBoost模型训练和评估完成！")

if __name__ == "__main__":
    main()