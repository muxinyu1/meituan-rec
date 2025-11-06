import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import os
import json
import argparse
from sklearn.preprocessing import StandardScaler

# 假设 model.py 在同一个目录下
# 如果 model.py 不存在，这个脚本会报错
try:
    from model import Model
except ImportError:
    print("="*50)
    print("错误: 找不到 'model.py' 文件。")
    print("请确保 'model.py' 与 'predict.py' 在同一个目录中。")
    print("="*50)
    exit(1)


# ==============================================================================
# 1. 全局变量 (由 main 函数中的 checkpoint 动态填充)
# ==============================================================================
# 我们先将它们声明为全局变量，main()函数将从加载的checkpoint中填充它们

USER_DISCRETE_VOCAB_SIZES = {}
USER_CONTINUOUS_FEATURES = []
ITEM_DISCRETE_VOCAB_SIZES = {}
ITEM_CONTINUOUS_FEATURES = []
CONTEXT_DISCRETE_VOCAB_SIZES = {}
CONTEXT_CONTINUOUS_FEATURES = []
ALL_CONTINUOUS_FEATURES = []
USER_FEATURE_DEFS = ({}, [])
ITEM_FEATURE_DEFS = ({}, [])
CONTEXT_FEATURE_DEFS = ({}, [])


# ==============================================================================
# 2. 专用于测试集的 Dataset 和 Collate Function
# (这些函数依赖于上面动态填充的全局变量)
# ==============================================================================

class RecSysTestDataset(Dataset):
    """
    一个为测试集优化的Dataset。
    它不加载 'label'，而是加载 'sample_index'。
    """

    def __init__(self, df: pd.DataFrame):
        # 预先提取列名列表 (依赖全局变量)
        self.user_discrete_cols = list(USER_DISCRETE_VOCAB_SIZES.keys())
        self.user_continuous_cols = USER_CONTINUOUS_FEATURES
        self.item_discrete_cols = list(ITEM_DISCRETE_VOCAB_SIZES.keys())
        self.item_continuous_cols = ITEM_CONTINUOUS_FEATURES
        self.context_discrete_cols = list(CONTEXT_DISCRETE_VOCAB_SIZES.keys())
        self.context_continuous_cols = CONTEXT_CONTINUOUS_FEATURES
        
        self.sample_index_col = "sample_index"

        # 1. 整合所有 *特征* 列
        self.all_feature_cols = (
            self.user_discrete_cols
            + self.user_continuous_cols
            + self.item_discrete_cols
            + self.item_continuous_cols
            + self.context_discrete_cols
            + self.context_continuous_cols
        )
        
        # 检查测试集中是否缺少必要的特征列
        missing_cols = [col for col in self.all_feature_cols if col not in df.columns]
        if missing_cols:
            print(f"错误: 测试数据缺少必要的特征列: {missing_cols}")
            exit(1)
            
        if self.sample_index_col not in df.columns:
            print(f"错误: 测试数据缺少 '{self.sample_index_col}' 列。")
            exit(1)

        # 2. 将特征和sample_index转换为NumPy数组
        self.features_numpy = df[self.all_feature_cols].to_numpy()
        self.sample_index_numpy = df[self.sample_index_col].to_numpy()

        # 3. 预先计算好切片索引 (与 train.py 相同)
        start = 0
        end = len(self.user_discrete_cols)
        self.user_discrete_slice = slice(start, end)
        start = end
        end += len(self.user_continuous_cols)
        self.user_continuous_slice = slice(start, end)
        start = end
        end += len(self.item_discrete_cols)
        self.item_discrete_slice = slice(start, end)
        start = end
        end += len(self.item_continuous_cols)
        self.item_continuous_slice = slice(start, end)
        start = end
        end += len(self.context_discrete_cols)
        self.context_discrete_slice = slice(start, end)
        start = end
        end += len(self.context_continuous_cols)
        self.context_continuous_slice = slice(start, end)

    def __len__(self):
        return len(self.features_numpy)

    def __getitem__(self, idx):
        feature_row = self.features_numpy[idx]

        user_discrete_data = feature_row[self.user_discrete_slice]
        user_continuous_data = feature_row[self.user_continuous_slice]
        item_discrete_data = feature_row[self.item_discrete_slice]
        item_continuous_data = feature_row[self.item_continuous_slice]
        context_discrete_data = feature_row[self.context_discrete_slice]
        context_continuous_data = feature_row[self.context_continuous_slice]

        sample_index = self.sample_index_numpy[idx]

        return (
            user_discrete_data,
            user_continuous_data,
            item_discrete_data,
            item_continuous_data,
            context_discrete_data,
            context_continuous_data,
            sample_index,
        )


def collate_test_batch(batch):
    """
    用于测试集的collate_fn，处理 sample_index 而不是 label。
    """
    unzipped = zip(*batch)

    (
        user_discrete_data,
        user_continuous_data,
        item_discrete_data,
        item_continuous_data,
        context_discrete_data,
        context_continuous_data,
        sample_indexes,
    ) = unzipped

    user_discrete_batch = torch.tensor(np.array(user_discrete_data), dtype=torch.long)
    item_discrete_batch = torch.tensor(np.array(item_discrete_data), dtype=torch.long)
    context_discrete_batch = torch.tensor(
        np.array(context_discrete_data), dtype=torch.long
    )

    user_continuous_batch = torch.tensor(
        np.array(user_continuous_data, dtype=np.float32), dtype=torch.float
    )
    item_continuous_batch = torch.tensor(
        np.array(item_continuous_data, dtype=np.float32), dtype=torch.float
    )
    context_continuous_batch = torch.tensor(
        np.array(context_continuous_data, dtype=np.float32), dtype=torch.float
    )

    batched_user = {}
    if user_discrete_data[0].size > 0:
        for i, name in enumerate(USER_DISCRETE_VOCAB_SIZES.keys()):
            batched_user[name] = user_discrete_batch[:, i]
    if user_continuous_data[0].size > 0:
        for i, name in enumerate(USER_CONTINUOUS_FEATURES):
            batched_user[name] = user_continuous_batch[:, i]

    batched_item = {}
    if item_discrete_data[0].size > 0:
        for i, name in enumerate(ITEM_DISCRETE_VOCAB_SIZES.keys()):
            batched_item[name] = item_discrete_batch[:, i]
    if item_continuous_data[0].size > 0:
        for i, name in enumerate(ITEM_CONTINUOUS_FEATURES):
            batched_item[name] = item_continuous_batch[:, i]

    batched_context = {}
    if context_discrete_data[0].size > 0:
        for i, name in enumerate(CONTEXT_DISCRETE_VOCAB_SIZES.keys()):
            batched_context[name] = context_discrete_batch[:, i]
    if context_continuous_data[0].size > 0:
        for i, name in enumerate(CONTEXT_CONTINUOUS_FEATURES):
            batched_context[name] = context_continuous_batch[:, i]

    return batched_user, batched_item, batched_context, np.array(sample_indexes)


def predict_loop(model, data_loader, device):
    """
    执行预测并收集 sample_index 和概率。
    """
    model.eval()
    all_preds, all_sample_indexes = [], []
    with torch.no_grad():
        for user_batch, item_batch, context_batch, sample_indexes in tqdm(
            data_loader, desc="Predicting"
        ):
            user_batch = {k: v.to(device) for k, v in user_batch.items()}
            item_batch = {k: v.to(device) for k, v in item_batch.items()}
            context_batch = {k: v.to(device) for k, v in context_batch.items()}

            logits = model(user_batch, item_batch, context_batch)
            probs = torch.sigmoid(logits) # 转换为 0-1 之间的概率

            all_preds.append(probs.cpu())
            all_sample_indexes.append(sample_indexes)

    all_preds_tensor = torch.cat(all_preds).numpy()
    all_sample_indexes_tensor = np.concatenate(all_sample_indexes)
    
    return all_sample_indexes_tensor, all_preds_tensor


# ==============================================================================
# 3. 预测主程序
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Run prediction on the test set.")
    parser.add_argument(
        "--test_csv", 
        type=str, 
        required=True, 
        help="Path to the PROCESSED test CSV file (e.g., 'processed_test.csv')"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="models/attn_best_model.pth", 
        help="Path to the saved bundled model artifact (.pth file)"
    )
    parser.add_argument(
        "--output_csv", 
        type=str, 
        default="submission.csv", 
        help="Path to save the final submission CSV file"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096, # 预测时可以使用更大的batch_size
        help="Batch size for prediction"
    )
    args = parser.parse_args()

    # --- 1. 设备设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. 加载打包的训练产物 ---
    print(f"Loading bundled artifacts from: {args.model_path}")
    try:
        # 加载所有保存的对象
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"错误: 模型文件未找到: {args.model_path}")
        print("请确保你已经训练了模型，并且路径是正确的。")
        exit(1)
    except Exception as e:
        print(f"加载模型文件时出错: {e}")
        print("这可能是一个旧格式的模型文件。请使用新版 train.py 重新训练。")
        exit(1)

    # --- 3. 提取所有组件并设置全局变量 ---
    print("Extracting artifacts from checkpoint...")
    try:
        scaler = checkpoint['scaler']
        model_config = checkpoint['model_config']
        feature_definitions = checkpoint['feature_definitions']
        model_state_dict = checkpoint['model_state_dict']
    except KeyError as e:
        print(f"错误: 模型文件不完整，缺少键: {e}")
        print("请确保你的 train.py 保存了 'scaler', 'model_config', 'feature_definitions', 和 'model_state_dict'")
        exit(1)

    # 动态填充本脚本顶部的全局变量
    global USER_DISCRETE_VOCAB_SIZES, USER_CONTINUOUS_FEATURES, \
           ITEM_DISCRETE_VOCAB_SIZES, ITEM_CONTINUOUS_FEATURES, \
           CONTEXT_DISCRETE_VOCAB_SIZES, CONTEXT_CONTINUOUS_FEATURES, \
           ALL_CONTINUOUS_FEATURES, USER_FEATURE_DEFS, \
           ITEM_FEATURE_DEFS, CONTEXT_FEATURE_DEFS

    USER_DISCRETE_VOCAB_SIZES = feature_definitions["user"]["discrete"]
    USER_CONTINUOUS_FEATURES = feature_definitions["user"]["continuous"]
    ITEM_DISCRETE_VOCAB_SIZES = feature_definitions["item"]["discrete"]
    ITEM_CONTINUOUS_FEATURES = feature_definitions["item"]["continuous"]
    CONTEXT_DISCRETE_VOCAB_SIZES = feature_definitions["context"]["discrete"]
    CONTEXT_CONTINUOUS_FEATURES = feature_definitions["context"]["continuous"]
    
    ALL_CONTINUOUS_FEATURES = (
        USER_CONTINUOUS_FEATURES + ITEM_CONTINUOUS_FEATURES + CONTEXT_CONTINUOUS_FEATURES
    )
    
    USER_FEATURE_DEFS = (USER_DISCRETE_VOCAB_SIZES, USER_CONTINUOUS_FEATURES)
    ITEM_FEATURE_DEFS = (ITEM_DISCRETE_VOCAB_SIZES, ITEM_CONTINUOUS_FEATURES)
    CONTEXT_FEATURE_DEFS = (CONTEXT_DISCRETE_VOCAB_SIZES, CONTEXT_CONTINUOUS_FEATURES)
    
    print("Feature definitions, config, and scaler loaded from checkpoint.")

    # --- 4. 加载并处理测试数据 ---
    print(f"Loading test data from: {args.test_csv}")
    try:
        test_df = pd.read_csv(args.test_csv)
    except FileNotFoundError:
        print(f"错误: 测试集文件未找到: {args.test_csv}")
        exit(1)
        
    print(f"Test data loaded. Shape: {test_df.shape}")

    # --- 5. 应用加载的归一化器 ---
    if scaler and ALL_CONTINUOUS_FEATURES:
        print("Normalizing test data using the *loaded* scaler...")
        # 确保列存在
        cols_to_transform = [col for col in ALL_CONTINUOUS_FEATURES if col in test_df.columns]
        if cols_to_transform:
            test_df.loc[:, cols_to_transform] = scaler.transform(
                test_df[cols_to_transform]
            )
            print("Normalization complete.")
        else:
            print("警告: 连续特征在测试集中未找到，跳过归一化。")
    else:
        print("No continuous features to normalize or scaler was not saved.")

    # --- 6. 创建 Test Dataset 和 DataLoader ---
    num_worker_cores = min(os.cpu_count(), 32) # type: ignore

    test_dataset = RecSysTestDataset(test_df)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False, # 预测时绝不能打乱
        collate_fn=collate_test_batch,
        num_workers=num_worker_cores,
        pin_memory=True,
    )

    # --- 7. 初始化并加载模型 ---
    print(f"Loading model architecture...")
    try:
        model = Model(
            user_feature_defs=USER_FEATURE_DEFS,
            item_feature_defs=ITEM_FEATURE_DEFS,
            context_feature_defs=CONTEXT_FEATURE_DEFS,
            embedding_dim_per_feature=model_config["EMBEDDING_DIM_PER_FEATURE"],
            hidden_dim=model_config["HIDDEN_DIM"],
            emb_dim=model_config["FINAL_EMB_DIM"],
            head_nums=model_config["HEAD_NUMS"],
        )
    except KeyError as e:
        print(f"错误: 模型配置 'model_config' 中缺少键: {e}")
        print("请确保你的 train.py 正确保存了所有模型超参数。")
        exit(1)
    
    print(f"Loading model weights...")
    try:
        model.load_state_dict(model_state_dict)
    except RuntimeError as e:
        print(f"错误: 加载模型权重失败。模型架构是否已更改？")
        print(f"确保 predict.py 中的 'model.py' 与 train.py 训练时使用的完全一致。")
        print(f"详细信息: {e}")
        exit(1)
        
    model.to(device)
    model.eval() # 切换到评估模式

    # --- 8. 执行预测 ---
    print("Starting prediction loop...")
    sample_indexes, predictions = predict_loop(model, test_loader, device)
    print("Prediction complete.")

    # --- 9. 格式化并保存结果 ---
    print(f"Formatting results for submission...")
    results_df = pd.DataFrame({
        "sample_index": sample_indexes.flatten(),
        "label": predictions.flatten()
    })
    
    results_df["sample_index"] = results_df["sample_index"].astype(int)
    results_df = results_df.sort_values(by="sample_index") # 按索引排序

    results_df.to_csv(args.output_csv, index=False)
    print(f"Submission file saved successfully to: {args.output_csv}")
    print("\n--- 提交文件预览 (前5行) ---")
    print(results_df.head())
    print("---------------------------------")


if __name__ == "__main__":
    main()