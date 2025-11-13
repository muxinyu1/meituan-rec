import numpy as np
import pandas as pd
import lightgbm as lgb
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss
import joblib

# 数据和特征定义路径
PROCESSED_DATA_PATH = "data/processed/train_temporal.csv"
FEATURE_DEFS_PATH = "data/processed/train_temporal.json"
MODEL_DIR = "models"

# 加载特征定义
print(f"Loading feature definitions from: {FEATURE_DEFS_PATH}")
with open(FEATURE_DEFS_PATH, "r") as f:
    feature_definitions = json.load(f)

USER_DISCRETE_VOCAB_SIZES = feature_definitions["user"]["discrete"]
USER_CONTINUOUS_FEATURES = feature_definitions["user"]["continuous"]
ITEM_DISCRETE_VOCAB_SIZES = feature_definitions["item"]["discrete"]
ITEM_CONTINUOUS_FEATURES = feature_definitions["item"]["continuous"]
CONTEXT_DISCRETE_VOCAB_SIZES = feature_definitions["context"]["discrete"]
CONTEXT_CONTINUOUS_FEATURES = feature_definitions["context"]["continuous"]

# 整合所有特征
ALL_DISCRETE_FEATURES = (
    list(USER_DISCRETE_VOCAB_SIZES.keys())
    + list(ITEM_DISCRETE_VOCAB_SIZES.keys())
    + list(CONTEXT_DISCRETE_VOCAB_SIZES.keys())
)
ALL_CONTINUOUS_FEATURES = (
    USER_CONTINUOUS_FEATURES + ITEM_CONTINUOUS_FEATURES + CONTEXT_CONTINUOUS_FEATURES
)
ALL_FEATURES = ALL_DISCRETE_FEATURES + ALL_CONTINUOUS_FEATURES

print("Feature definitions loaded successfully.")
print(f"Total features: {len(ALL_FEATURES)} ({len(ALL_DISCRETE_FEATURES)} discrete + {len(ALL_CONTINUOUS_FEATURES)} continuous)")


def prepare_data():
    """加载和准备数据"""
    print(f"\nLoading preprocessed data from: {PROCESSED_DATA_PATH}")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"Data loaded. Shape: {df.shape}")
    
    # 显示标签分布
    label_counts = df["label"].value_counts()
    print(f"Label distribution: {label_counts.to_dict()}")
    
    return df


def split_and_scale_data(df, validation_split=0.1):
    """划分训练集和验证集，并归一化连续特征"""
    print(f"\nSplitting data into train and validation sets ({1-validation_split:.0%}:{validation_split:.0%})")
    
    # 分离特征和标签
    X = df[ALL_FEATURES]
    y = df["label"]
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=42, stratify=y
    )
    
    print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # 归一化连续特征
    scaler = None
    if ALL_CONTINUOUS_FEATURES:
        print("Normalizing continuous features...")
        scaler = StandardScaler()
        # 在训练集上fit并transform
        X_train.loc[:, ALL_CONTINUOUS_FEATURES] = scaler.fit_transform(
            X_train[ALL_CONTINUOUS_FEATURES]
        )
        # 在验证集上只transform
        X_val.loc[:, ALL_CONTINUOUS_FEATURES] = scaler.transform(
            X_val[ALL_CONTINUOUS_FEATURES]
        )
        print("Normalization complete.")
    else:
        print("No continuous features to normalize.")
    
    return X_train, X_val, y_train, y_val, scaler


def train_lightgbm(X_train, y_train, X_val, y_val, params=None):
    """训练LightGBM模型"""
    print("\nTraining LightGBM model...")
    
    # 默认参数
    if params is None:
        # 计算正负样本比例
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'scale_pos_weight': scale_pos_weight,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42
        }
    
    print(f"Parameters: {params}")
    
    # 创建LightGBM数据集
    # 标记离散特征（类别特征）
    categorical_features = ALL_DISCRETE_FEATURES
    
    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=categorical_features,
        free_raw_data=False
    )
    val_data = lgb.Dataset(
        X_val,
        label=y_val,
        categorical_feature=categorical_features,
        reference=train_data,
        free_raw_data=False
    )
    
    # 训练模型
    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=10)
    ]
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1024,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=callbacks
    )
    
    return model


def evaluate_model(model, X, y, dataset_name="Validation"):
    """评估模型性能"""
    print(f"\nEvaluating on {dataset_name} set...")
    
    # 预测
    y_pred_proba = model.predict(X, num_iteration=model.best_iteration)
    
    # 计算指标
    auc = roc_auc_score(y, y_pred_proba)
    logloss = log_loss(y, y_pred_proba)
    
    print(f"{dataset_name} AUC: {auc:.4f}")
    print(f"{dataset_name} Log Loss: {logloss:.4f}")
    
    return auc, logloss


def save_model(model, scaler, feature_definitions):
    """保存模型和相关配置"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 保存LightGBM模型
    model_path = os.path.join(MODEL_DIR, "lightgbm.txt")
    model.save_model(model_path)
    print(f"\nLightGBM model saved to: {model_path}")
    
    # 保存scaler
    if scaler is not None:
        scaler_path = os.path.join(MODEL_DIR, "standard_scaler.joblib")
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")
    
    # 保存特征定义
    feature_defs_path = os.path.join(MODEL_DIR, "feature_definitions.json")
    with open(feature_defs_path, "w") as f:
        json.dump(feature_definitions, f, indent=2)
    print(f"Feature definitions saved to: {feature_defs_path}")


def plot_feature_importance(model, top_n=20):
    """绘制特征重要性"""
    try:
        import matplotlib.pyplot as plt
        
        # 获取特征重要性
        importance = model.feature_importance(importance_type='gain')
        feature_names = model.feature_name()
        
        # 创建DataFrame并排序
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # 绘制前N个最重要的特征
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance (Gain)')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        importance_plot_path = os.path.join(MODEL_DIR, "feature_importance.png")
        plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
        print(f"\nFeature importance plot saved to: {importance_plot_path}")
        plt.close()
        
        # 保存特征重要性到CSV
        importance_csv_path = os.path.join(MODEL_DIR, "feature_importance.csv")
        importance_df.to_csv(importance_csv_path, index=False)
        print(f"Feature importance data saved to: {importance_csv_path}")
        
    except ImportError:
        print("\nMatplotlib not available. Skipping feature importance plot.")
        # 仍然保存特征重要性数据
        importance = model.feature_importance(importance_type='gain')
        feature_names = model.feature_name()
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        importance_csv_path = os.path.join(MODEL_DIR, "feature_importance.csv")
        importance_df.to_csv(importance_csv_path, index=False)
        print(f"Feature importance data saved to: {importance_csv_path}")


def main():
    """主函数"""
    print("=" * 80)
    print("LightGBM CTR Prediction")
    print("=" * 80)
    
    # 1. 加载数据
    df = prepare_data()
    
    # 2. 划分和归一化数据
    X_train, X_val, y_train, y_val, scaler = split_and_scale_data(
        df, validation_split=0.1
    )
    
    # 3. 训练模型
    model = train_lightgbm(X_train, y_train, X_val, y_val)
    
    # 4. 评估模型
    train_auc, train_logloss = evaluate_model(model, X_train, y_train, "Training")
    val_auc, val_logloss = evaluate_model(model, X_val, y_val, "Validation")
    
    # 5. 保存模型
    save_model(model, scaler, feature_definitions)
    
    # 6. 绘制特征重要性
    plot_feature_importance(model, top_n=20)
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print(f"Best Validation AUC: {val_auc:.4f}")
    print(f"Best Validation Log Loss: {val_logloss:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
