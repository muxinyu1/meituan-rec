import numpy as np
import pandas as pd
import lightgbm as lgb
import json
import os
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import time

# ==================================================================
# === 配置项 ===
# ==================================================================

# --- 路径 ---
PROCESSED_DATA_PATH = "data/processed/train_temporal.csv"
FEATURE_DEFS_PATH = "data/processed/train_temporal.json"
# 输出目录：用于存放生成的叶节点特征和最终模型
OUTPUT_DIR = "data/leaf_features"

# --- K-Fold 配置 ---
N_SPLITS = 5  # 5 折交叉验证

# --- LGBM 核心参数 ---
# 这是 GBDT+DL 策略的关键超参数
# 它定义了你新特征的维度
N_ESTIMATORS = 200 
NUM_LEAVES = 31

# ==================================================================
# === 特征和数据加载 (与你之前的脚本相同) ===
# ==================================================================

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


def prepare_data():
    """加载和准备数据"""
    print(f"\nLoading preprocessed data from: {PROCESSED_DATA_PATH}")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"Data loaded. Shape: {df.shape}")
    
    label_counts = df["label"].value_counts()
    print(f"Label distribution: {label_counts.to_dict()}")
    
    X = df[ALL_FEATURES]
    y = df["label"]
    
    return X, y, df

# ==================================================================
# === 主程序：K-Fold OOF 叶节点特征提取 ===
# ==================================================================

def main():
    start_time = time.time()
    
    # 1. 加载数据
    X, y, df_full = prepare_data()
    
    # 2. 定义LGBM参数
    # 在K-Fold中，我们使用固定的 num_boost_round (N_ESTIMATORS)
    # 并且不使用 early_stopping，以确保所有折的输出维度一致
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    
    lgbm_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': NUM_LEAVES,
        'num_boost_round': N_ESTIMATORS, # 关键：固定树的数量
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1, # 在循环中保持安静
        'scale_pos_weight': scale_pos_weight,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # 3. 初始化OOF数组
    # oof_leaf_features: 存储每个样本的叶节点索引
    # oof_predictions: (可选) 存储OOF预测概率，用于评估LGBM的OOF AUC
    oof_leaf_features = np.zeros((len(X), N_ESTIMATORS), dtype=np.int32)
    oof_predictions = np.zeros(len(X))
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    print("\n" + "=" * 80)
    print(f"Starting K-Fold (N_SPLITS={N_SPLITS}) Leaf Feature Extraction...")
    print(f"LGBM params: N_ESTIMATORS={N_ESTIMATORS}, NUM_LEAVES={NUM_LEAVES}")
    print("=" * 80)
    
    # 4. K-Fold 循环
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        fold_start_time = time.time()
        print(f"\n--- FOLD {fold + 1}/{N_SPLITS} ---")
        
        # --- a. 拆分数据 ---
        # .copy() 避免 SettingWithCopyWarning
        X_train_fold, X_val_fold = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # --- b. 归一化 (在Fold内部) ---
        scaler = StandardScaler()
        if ALL_CONTINUOUS_FEATURES:
            X_train_fold.loc[:, ALL_CONTINUOUS_FEATURES] = scaler.fit_transform(
                X_train_fold[ALL_CONTINUOUS_FEATURES]
            )
            X_val_fold.loc[:, ALL_CONTINUOUS_FEATURES] = scaler.transform(
                X_val_fold[ALL_CONTINUOUS_FEATURES]
            )
        
        # --- c. 训练LGBM ---
        print("Training LGBM for fold...")
        train_data = lgb.Dataset(
            X_train_fold,
            label=y_train_fold,
            categorical_feature=ALL_DISCRETE_FEATURES
        )
        
        model_fold = lgb.train(
            lgbm_params,
            train_data,
            # 注意：没有验证集和 early_stopping
        )
        
        # --- d. 预测叶节点 (关键) ---
        print("Predicting leaf indices for validation set...")
        val_leaves = model_fold.predict(
            X_val_fold, 
            pred_leaf=True
        )
        # 形状: (n_val_samples, N_ESTIMATORS)
        
        # --- e. 存储OOF结果 ---
        oof_leaf_features[val_idx] = val_leaves
        
        # --- f. (可选) 评估OOF AUC ---
        val_proba = model_fold.predict(X_val_fold)
        oof_predictions[val_idx] = val_proba
        fold_auc = roc_auc_score(y_val_fold, val_proba)
        print(f"Fold {fold + 1} AUC: {fold_auc:.4f}")
        print(f"Fold {fold + 1} completed in {time.time() - fold_start_time:.2f}s")
        
    # 5. OOF 循环结束
    overall_oof_auc = roc_auc_score(y, oof_predictions)
    print("\n" + "=" * 80)
    print(f"K-Fold OOF extraction complete.")
    print(f"Overall OOF AUC: {overall_oof_auc:.4f}")
    
    # 6. 保存OOF叶节点特征
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    leaf_feature_path = os.path.join(OUTPUT_DIR, "oof_leaf_features.npy")
    np.save(leaf_feature_path, oof_leaf_features)
    print(f"OOF leaf features saved to: {leaf_feature_path}")
    print(f"Shape: {oof_leaf_features.shape}")

    # 7. 训练并保存最终模型 (用于未来Test集)
    # 这个模型将使用100%的数据训练，并保存，以便你之后为 test.csv 生成特征
    print("\nTraining final model on 100% of data...")
    
    # --- a. 归一化全部数据 ---
    scaler_final = StandardScaler()
    X_scaled = X.copy()
    if ALL_CONTINUOUS_FEATURES:
        X_scaled.loc[:, ALL_CONTINUOUS_FEATURES] = scaler_final.fit_transform(
            X[ALL_CONTINUOUS_FEATURES]
        )
        
    # --- b. 训练 ---
    full_train_data = lgb.Dataset(
        X_scaled,
        label=y,
        categorical_feature=ALL_DISCRETE_FEATURES
    )
    final_model = lgb.train(lgbm_params, full_train_data)
    
    # --- c. 保存 ---
    model_path = os.path.join(OUTPUT_DIR, "final_lgbm_model.txt")
    scaler_path = os.path.join(OUTPUT_DIR, "final_scaler.joblib")
    
    final_model.save_model(model_path)
    joblib.dump(scaler_final, scaler_path)
    print(f"Final model saved to: {model_path}")
    print(f"Final scaler saved to: {scaler_path}")
    
    print(f"\nTotal script time: {time.time() - start_time:.2f}s")
    print("=" * 80)

if __name__ == "__main__":
    main()