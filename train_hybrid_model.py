import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import random

# === MODIFIED ===
# 假设你的 HybridModel 类现在位于 'model.py' 中，并且类名已更新
# 如果你保留了 'Model' 这个类名，这个导入是正确的
from hybrid_model import HybridModel as Model

# ==============================================================================
# 1. 路径和特征定义 (新增 GBDT 特征路径)
# ==============================================================================

PROCESSED_DATA_PATH = "data/processed/train_temporal.csv"
FEATURE_DEFS_PATH = "data/processed/train_temporal.json"

# === NEW ===
# OOF GBDT 叶节点特征的路径 (由 extract_leaf_features.py 生成)
LEAF_FEATURES_PATH = "data/leaf_features/oof_leaf_features.npy"

print(f"Loading feature definitions from: {FEATURE_DEFS_PATH}")
with open(FEATURE_DEFS_PATH, "r") as f:
    feature_definitions = json.load(f)

# ... (你原有的所有特征定义加载代码保持不变) ...
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

print("Feature definitions loaded successfully.")


# ==============================================================================
# 2. Dataset (修改以接受和返回叶节点特征)
# ==============================================================================

class RecSysDataset(Dataset):
    """
    一个经过优化的Dataset类。
    它在初始化时将特征转换为NumPy数组，以实现快速的数据访问。
    """

    # === MODIFIED ===
    def __init__(self, df: pd.DataFrame, leaf_features: np.ndarray): # 新增 leaf_features 参数
        # ... (你原有的列名提取代码保持不变) ...
        self.user_discrete_cols = list(USER_DISCRETE_VOCAB_SIZES.keys())
        self.user_continuous_cols = USER_CONTINUOUS_FEATURES
        self.item_discrete_cols = list(ITEM_DISCRETE_VOCAB_SIZES.keys())
        self.item_continuous_cols = ITEM_CONTINUOUS_FEATURES
        self.context_discrete_cols = list(CONTEXT_DISCRETE_VOCAB_SIZES.keys())
        self.context_continuous_cols = CONTEXT_CONTINUOUS_FEATURES
        self.label_col = "label"
        
        # ... (你原有的特征列整合代码保持不变) ...
        self.all_feature_cols = (
            self.user_discrete_cols
            + self.user_continuous_cols
            + self.item_discrete_cols
            + self.item_continuous_cols
            + self.context_discrete_cols
            + self.context_continuous_cols
        )
        
        # ... (你原有的 NumPy 转换代码保持不变) ...
        self.features_numpy = df[self.all_feature_cols].to_numpy()
        self.labels_numpy = df[self.label_col].to_numpy()

        # === NEW ===
        # 存储 GBDT 叶节点特征
        self.leaf_features_numpy = leaf_features
        
        # 验证数据是否对齐
        assert len(self.features_numpy) == len(self.labels_numpy), "Features and labels length mismatch"
        assert len(self.features_numpy) == len(self.leaf_features_numpy), "Features and leaf_features length mismatch"
        # === END NEW ===

        # ... (你原有的切片索引计算代码保持不变) ...
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
        return len(self.labels_numpy)

    def __getitem__(self, idx):
        # ... (你原有的特征提取代码保持不变) ...
        feature_row = self.features_numpy[idx]
        
        user_discrete_data = feature_row[self.user_discrete_slice]
        user_continuous_data = feature_row[self.user_continuous_slice]
        item_discrete_data = feature_row[self.item_discrete_slice]
        item_continuous_data = feature_row[self.item_continuous_slice]
        context_discrete_data = feature_row[self.context_discrete_slice]
        context_continuous_data = feature_row[self.context_continuous_slice]

        label = self.labels_numpy[idx]

        # === NEW ===
        # 获取 GBDT 叶节点索引
        leaf_indices = self.leaf_features_numpy[idx]
        # === END NEW ===

        # === MODIFIED ===
        # 返回元组 (现在有8个元素)
        return (
            user_discrete_data,
            user_continuous_data,
            item_discrete_data,
            item_continuous_data,
            context_discrete_data,
            context_continuous_data,
            leaf_indices,  # 新增
            label,
        )


# ==============================================================================
# 3. Collate Function (修改以处理叶节点特征)
# ==============================================================================

def collate_batch(batch):
    # === MODIFIED ===
    # 解包8个元素
    unzipped = zip(*batch)
    (
        user_discrete_data,
        user_continuous_data,
        item_discrete_data,
        item_continuous_data,
        context_discrete_data,
        context_continuous_data,
        leaf_indices,  # 新增
        labels,
    ) = unzipped

    # ... (你原有的离散和连续特征批处理代码保持不变) ...
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

    # === NEW ===
    # 为 GBDT 叶节点特征创建 Tensor
    leaf_indices_batch = torch.tensor(np.array(leaf_indices), dtype=torch.long)
    # === END NEW ===

    labels_tensor = torch.tensor(labels, dtype=torch.float).unsqueeze(1)

    # ... (你原有的字典创建代码保持不变) ...
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

    # === MODIFIED ===
    # 返回5个元素
    return batched_user, batched_item, batched_context, leaf_indices_batch, labels_tensor


# ==============================================================================
# 4. Evaluate Function (修改以处理叶节点特征)
# ==============================================================================

def evaluate(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        # === MODIFIED ===
        # 解包5个元素
        for user_batch, item_batch, context_batch, leaf_indices_batch, labels in tqdm(
            data_loader, desc="Validating"
        ):
            # ... (你原有的 to(device) 代码保持不变) ...
            user_batch = {k: v.to(device) for k, v in user_batch.items()}
            item_batch = {k: v.to(device) for k, v in item_batch.items()}
            context_batch = {k: v.to(device) for k, v in context_batch.items()}
            
            # === NEW ===
            # 移动 GBDT 特征到 device
            leaf_indices_batch = leaf_indices_batch.to(device)
            # === END NEW ===

            # === MODIFIED ===
            # 使用4个参数调用模型
            logits = model(user_batch, item_batch, context_batch, leaf_indices_batch)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_preds_tensor = torch.cat(all_preds).numpy()
    all_labels_tensor = torch.cat(all_labels).numpy()
    auc = roc_auc_score(all_labels_tensor, all_preds_tensor)
    return auc

def set_seed(seed_value=42):
    """设置所有随机种子以确保可复现性"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

# ==============================================================================
# 5. 训练主程序 (更新以使用新数据流)
# ==============================================================================
def main():
    set_seed(42)
    # --- 超参数 ---
    BATCH_SIZE = 4096
    LEARNING_RATE = 5e-4
    EPOCHS = 50
    EMBEDDING_DIM_PER_FEATURE = 64
    HIDDEN_DIM = 512
    FINAL_EMB_DIM = 256
    HEAD_NUMS = 16
    VALIDATION_SPLIT = 0.1
    CHECKPOINT_DIR = "models"
    
    # === NEW ===
    # GBDT+DL 独有的超参数
    N_TREES = 200        # 必须匹配 extract_leaf_features.py 中的 N_ESTIMATORS
    N_LEAVES = 31        # 必须匹配 extract_leaf_features.py 中的 NUM_LEAVES
    LEAF_EMBED_DIM = 8   # 可调超参数 (例如 8, 16)
    # === END NEW ===

    # --- 设备设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 加载数据 ---
    print(f"Loading preprocessed data from: {PROCESSED_DATA_PATH}")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print("Data loaded.")
    
    # === NEW ===
    # 加载 GBDT 叶节点特征
    print(f"Loading leaf features from: {LEAF_FEATURES_PATH}")
    leaf_features = np.load(LEAF_FEATURES_PATH)
    print(f"Leaf features loaded. Shape: {leaf_features.shape}")
    assert len(df) == len(leaf_features), "DataFrame and leaf features length mismatch!"
    # === END NEW ===

    # ... (你原有的 pos_weight 计算代码保持不变) ...
    label_counts = df["label"].value_counts()
    neg_samples, pos_samples = label_counts[0], label_counts[1]
    pos_weight = torch.tensor([neg_samples / pos_samples], device=device)
    print(f"Label distribution: {label_counts.to_dict()}")
    print(f"Calculated pos_weight for BCE loss: {pos_weight.item():.2f}")


    # --- 划分训练集和验证集 ---
    print(
        f"Splitting data into train and validation sets ({1-VALIDATION_SPLIT:.0%}:{VALIDATION_SPLIT:.0%})"
    )
    
    # === MODIFIED ===
    # 同时划分 DataFrame 和 GBDT 特征数组
    train_df, val_df, train_leaf_features, val_leaf_features = train_test_split(
        df,
        leaf_features, # 新增：同时划分 leaf_features
        test_size=VALIDATION_SPLIT,
        random_state=42,
        stratify=df["label"],
    )
    # === END MODIFIED ===
    
    # 重置索引
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")

    # ... (你原有的归一化代码保持不变) ...
    scaler = None
    if ALL_CONTINUOUS_FEATURES:
        print("Normalizing continuous features...")
        scaler = StandardScaler()
        train_df.loc[:, ALL_CONTINUOUS_FEATURES] = scaler.fit_transform(
            train_df[ALL_CONTINUOUS_FEATURES]
        )
        val_df.loc[:, ALL_CONTINUOUS_FEATURES] = scaler.transform(
            val_df[ALL_CONTINUOUS_FEATURES]
        )
        print("Normalization complete.")
    else:
        print("No continuous features to normalize.")

    # --- 创建Dataset和DataLoader ---
    num_worker_cores = min(os.cpu_count(), 32) # type: ignore

    # === MODIFIED ===
    # 将 GBDT 特征传递给 Dataset
    train_dataset = RecSysDataset(train_df, train_leaf_features)
    val_dataset = RecSysDataset(val_df, val_leaf_features)
    # === END MODIFIED ===

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=num_worker_cores,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=num_worker_cores,
        pin_memory=True,
    )

    # === MODIFIED ===
    # 保存所有超参数，包括 GBDT 的
    model_config = {
        "EMBEDDING_DIM_PER_FEATURE": EMBEDDING_DIM_PER_FEATURE,
        "HIDDEN_DIM": HIDDEN_DIM,
        "FINAL_EMB_DIM": FINAL_EMB_DIM,
        "HEAD_NUMS": HEAD_NUMS,
        "N_TREES": N_TREES,           # 新增
        "N_LEAVES": N_LEAVES,         # 新增
        "LEAF_EMBED_DIM": LEAF_EMBED_DIM, # 新增
    }

    # 假设你的 'model.py' 中的 'Model' 类现在是 HybridModel
    # 它在 __init__ 中接受 GBDT 的新参数
    model = Model(
        user_feature_defs=USER_FEATURE_DEFS,
        item_feature_defs=ITEM_FEATURE_DEFS,
        context_feature_defs=CONTEXT_FEATURE_DEFS,
        embedding_dim_per_feature=EMBEDDING_DIM_PER_FEATURE,
        hidden_dim=HIDDEN_DIM,
        emb_dim=FINAL_EMB_DIM,
        head_nums=HEAD_NUMS,
        
        # === NEW ===
        # 传入 GBDT 超参数
        n_trees=N_TREES,
        num_leaves=N_LEAVES,
        leaf_embed_dim=LEAF_EMBED_DIM
        # === END NEW ===
    ).to(device)
    # === END MODIFIED ===

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.2, patience=2,
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_auc = 0.0

    # --- 训练循环 ---
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

        # === MODIFIED ===
        # 解包5个元素
        for user_batch, item_batch, context_batch, leaf_indices_batch, labels in progress_bar:
            # ... (你原有的 to(device) 代码保持不变) ...
            user_batch = {k: v.to(device) for k, v in user_batch.items()}
            item_batch = {k: v.to(device) for k, v in item_batch.items()}
            context_batch = {k: v.to(device) for k, v in context_batch.items()}
            labels = labels.to(device)

            # === NEW ===
            # 移动 GBDT 特征到 device
            leaf_indices_batch = leaf_indices_batch.to(device)
            # === END NEW ===

            optimizer.zero_grad()
            
            # === MODIFIED ===
            # 使用4个参数调用模型
            logits = model(user_batch, item_batch, context_batch, leaf_indices_batch)
            loss = criterion(logits, labels)

            if torch.isnan(loss):
                print("\n!!! NaN loss detected. Stopping training. !!!")
                return

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        val_auc = evaluate(model, val_loader, device)
        scheduler.step(val_auc)

        print(
            f"\nEpoch {epoch+1}/{EPOCHS} Summary | Avg Train Loss: {avg_loss:.4f} | Validation AUC: {val_auc:.4f} | Best AUC: {best_val_auc:.4f}"
        )
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            # === MODIFIED ===
            # 更改保存路径以区分模型
            save_path = os.path.join(CHECKPOINT_DIR, "hybrid_attn_best_model.pth")
            print(
                f"🎉 New best model found! AUC improved to {best_val_auc:.4f}. Saving model to {save_path}\n"
            )
            save_bundle = {
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'model_config': model_config, # model_config 现在包含 GBDT 超参
                'feature_definitions': feature_definitions
            }
            torch.save(save_bundle, save_path)
        else:
            print(f"Validation AUC did not improve from {best_val_auc:.4f}.\n")

    print("Training complete.")
    print(f"Best validation AUC achieved: {best_val_auc:.4f}")
    print(
        f"The best model is saved at: {os.path.join(CHECKPOINT_DIR, 'hybrid_attn_best_model.pth')}"
    )


if __name__ == "__main__":
    main()