import numpy as np
import pandas as pd
import lightgbm as lgb
import json
import os
import joblib
from tqdm import tqdm

# 数据和模型路径
TEST_DATA_PATH = "data/processed/test_temporal.csv"
MODEL_PATH = "models/lightgbm.txt"
SCALER_PATH = "models/standard_scaler.joblib"
FEATURE_DEFS_PATH = "models/feature_definitions.json"
OUTPUT_PATH = "submission.csv"


def load_model_and_config():
    """加载模型和相关配置"""
    print("Loading model and configurations...")
    
    # 加载LightGBM模型
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    model = lgb.Booster(model_file=MODEL_PATH)
    print(f"LightGBM model loaded from: {MODEL_PATH}")
    
    # 加载scaler
    scaler = None
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print(f"Scaler loaded from: {SCALER_PATH}")
    else:
        print("No scaler found. Continuous features will not be normalized.")
    
    # 加载特征定义
    if not os.path.exists(FEATURE_DEFS_PATH):
        raise FileNotFoundError(f"Feature definitions not found at: {FEATURE_DEFS_PATH}")
    
    with open(FEATURE_DEFS_PATH, "r") as f:
        feature_definitions = json.load(f)
    print(f"Feature definitions loaded from: {FEATURE_DEFS_PATH}")
    
    return model, scaler, feature_definitions


def prepare_features(feature_definitions):
    """准备特征列表"""
    USER_DISCRETE_VOCAB_SIZES = feature_definitions["user"]["discrete"]
    USER_CONTINUOUS_FEATURES = feature_definitions["user"]["continuous"]
    ITEM_DISCRETE_VOCAB_SIZES = feature_definitions["item"]["discrete"]
    ITEM_CONTINUOUS_FEATURES = feature_definitions["item"]["continuous"]
    CONTEXT_DISCRETE_VOCAB_SIZES = feature_definitions["context"]["discrete"]
    CONTEXT_CONTINUOUS_FEATURES = feature_definitions["context"]["continuous"]
    
    ALL_DISCRETE_FEATURES = (
        list(USER_DISCRETE_VOCAB_SIZES.keys())
        + list(ITEM_DISCRETE_VOCAB_SIZES.keys())
        + list(CONTEXT_DISCRETE_VOCAB_SIZES.keys())
    )
    ALL_CONTINUOUS_FEATURES = (
        USER_CONTINUOUS_FEATURES + ITEM_CONTINUOUS_FEATURES + CONTEXT_CONTINUOUS_FEATURES
    )
    ALL_FEATURES = ALL_DISCRETE_FEATURES + ALL_CONTINUOUS_FEATURES
    
    return ALL_FEATURES, ALL_CONTINUOUS_FEATURES


def load_test_data(test_path):
    """加载测试数据"""
    print(f"\nLoading test data from: {test_path}")
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at: {test_path}")
    
    df = pd.read_csv(test_path)
    print(f"Test data loaded. Shape: {df.shape}")
    
    return df


def predict(model, X_test, batch_size=10000):
    """批量预测"""
    print("\nGenerating predictions...")
    
    n_samples = len(X_test)
    predictions = np.zeros(n_samples)
    
    # 分批预测以避免内存问题
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for i in tqdm(range(n_batches), desc="Predicting"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        
        batch_data = X_test.iloc[start_idx:end_idx]
        predictions[start_idx:end_idx] = model.predict(
            batch_data, num_iteration=model.best_iteration
        )
    
    return predictions


def save_predictions(predictions, sample_ids=None, output_path=OUTPUT_PATH):
    """保存预测结果"""
    print(f"\nSaving predictions to: {output_path}")
    
    # 创建提交文件
    if sample_ids is not None:
        submission_df = pd.DataFrame({
            'id': sample_ids,
            'label': predictions
        })
    else:
        submission_df = pd.DataFrame({
            'label': predictions
        })
    
    submission_df.to_csv(output_path, index=False)
    print(f"Predictions saved successfully!")
    print(f"Total predictions: {len(predictions)}")
    print(f"Prediction stats - Mean: {predictions.mean():.4f}, Std: {predictions.std():.4f}, Min: {predictions.min():.4f}, Max: {predictions.max():.4f}")


def main():
    """主函数"""
    print("=" * 80)
    print("LightGBM CTR Prediction - Inference")
    print("=" * 80)
    
    # 1. 加载模型和配置
    model, scaler, feature_definitions = load_model_and_config()
    
    # 2. 准备特征列表
    ALL_FEATURES, ALL_CONTINUOUS_FEATURES = prepare_features(feature_definitions)
    print(f"\nTotal features: {len(ALL_FEATURES)}")
    
    # 3. 加载测试数据
    test_df = load_test_data(TEST_DATA_PATH)
    
    # 保存样本ID（如果存在）
    sample_ids = None
    if 'id' in test_df.columns:
        sample_ids = test_df['id']
    elif 'sample_id' in test_df.columns:
        sample_ids = test_df['sample_id']
    
    # 4. 提取特征
    X_test = test_df[ALL_FEATURES]
    
    # 5. 归一化连续特征
    if scaler is not None and ALL_CONTINUOUS_FEATURES:
        print("Normalizing continuous features...")
        X_test.loc[:, ALL_CONTINUOUS_FEATURES] = scaler.transform(
            X_test[ALL_CONTINUOUS_FEATURES]
        )
        print("Normalization complete.")
    
    # 6. 预测
    predictions = predict(model, X_test, batch_size=10000)
    
    # 7. 保存结果
    save_predictions(predictions, sample_ids, OUTPUT_PATH)
    
    print("\n" + "=" * 80)
    print("Prediction Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
