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

from model import Model

PROCESSED_DATA_PATH = "data/processed/train.csv"
FEATURE_DEFS_PATH = "data/processed/train.json"

print(f"Loading feature definitions from: {FEATURE_DEFS_PATH}")
with open(FEATURE_DEFS_PATH, "r") as f:
    feature_definitions = json.load(f)

USER_DISCRETE_VOCAB_SIZES = feature_definitions["user"]["discrete"]
USER_CONTINUOUS_FEATURES = feature_definitions["user"]["continuous"]
ITEM_DISCRETE_VOCAB_SIZES = feature_definitions["item"]["discrete"]
ITEM_CONTINUOUS_FEATURES = feature_definitions["item"]["continuous"]
CONTEXT_DISCRETE_VOCAB_SIZES = feature_definitions["context"]["discrete"]
CONTEXT_CONTINUOUS_FEATURES = feature_definitions["context"]["continuous"]

# 整合所有连续特征，方便归一化
ALL_CONTINUOUS_FEATURES = (
    USER_CONTINUOUS_FEATURES + ITEM_CONTINUOUS_FEATURES + CONTEXT_CONTINUOUS_FEATURES
)

# 将特征定义打包，方便传递给模型
USER_FEATURE_DEFS = (USER_DISCRETE_VOCAB_SIZES, USER_CONTINUOUS_FEATURES)
ITEM_FEATURE_DEFS = (ITEM_DISCRETE_VOCAB_SIZES, ITEM_CONTINUOUS_FEATURES)
CONTEXT_FEATURE_DEFS = (CONTEXT_DISCRETE_VOCAB_SIZES, CONTEXT_CONTINUOUS_FEATURES)

print("Feature definitions loaded successfully.")


class RecSysDataset(Dataset):
    """
    一个经过优化的Dataset类。
    它在初始化时将特征转换为NumPy数组，以实现快速的数据访问。
    """

    def __init__(self, df: pd.DataFrame):
        # 预先提取列名列表以提高效率
        self.user_discrete_cols = list(USER_DISCRETE_VOCAB_SIZES.keys())
        self.user_continuous_cols = USER_CONTINUOUS_FEATURES
        self.item_discrete_cols = list(ITEM_DISCRETE_VOCAB_SIZES.keys())
        self.item_continuous_cols = ITEM_CONTINUOUS_FEATURES
        self.context_discrete_cols = list(CONTEXT_DISCRETE_VOCAB_SIZES.keys())
        self.context_continuous_cols = CONTEXT_CONTINUOUS_FEATURES

        self.label_col = "label"

        # ---- 修改开始 ----

        # 1. 整合所有特征列，并保持一个固定的顺序
        self.all_feature_cols = (
            self.user_discrete_cols
            + self.user_continuous_cols
            + self.item_discrete_cols
            + self.item_continuous_cols
            + self.context_discrete_cols
            + self.context_continuous_cols
        )

        # 2. 将特征和标签从Pandas DataFrame一次性转换为NumPy数组
        #    NumPy数组的索引比Pandas iloc/loc快得多
        self.features_numpy = df[self.all_feature_cols].to_numpy()
        self.labels_numpy = df[self.label_col].to_numpy()

        # 3. 预先计算好每个特征组在NumPy数组中的切片索引
        #    这样在__getitem__中就不用再做任何计算了
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
        # ---- 修改结束 ----

    def __len__(self):
        return len(self.labels_numpy)  # 使用NumPy数组的长度

    def __getitem__(self, idx):
        # ---- 修改开始 ----
        # 直接从NumPy数组中获取一行特征，非常快
        feature_row = self.features_numpy[idx]

        # 使用预先计算好的切片来提取各个部分的特征，避免了字典创建
        user_discrete_data = feature_row[self.user_discrete_slice]
        user_continuous_data = feature_row[self.user_continuous_slice]
        item_discrete_data = feature_row[self.item_discrete_slice]
        item_continuous_data = feature_row[self.item_continuous_slice]
        context_discrete_data = feature_row[self.context_discrete_slice]
        context_continuous_data = feature_row[self.context_continuous_slice]

        label = self.labels_numpy[idx]

        # 返回简单的NumPy数组元组，而不是字典
        return (
            user_discrete_data,
            user_continuous_data,
            item_discrete_data,
            item_continuous_data,
            context_discrete_data,
            context_continuous_data,
            label,
        )


def collate_batch(batch):

    unzipped = zip(*batch)

    (
        user_discrete_data,
        user_continuous_data,
        item_discrete_data,
        item_continuous_data,
        context_discrete_data,
        context_continuous_data,
        labels,
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

    labels_tensor = torch.tensor(labels, dtype=torch.float).unsqueeze(1)

    # 在整理完批次数据后，仅创建一次字典，以匹配模型的输入格式
    batched_user = {}
    if user_discrete_data[0].size > 0:  # 检查是否有离散特征
        for i, name in enumerate(USER_DISCRETE_VOCAB_SIZES.keys()):
            batched_user[name] = user_discrete_batch[:, i]
    if user_continuous_data[0].size > 0:  # 检查是否有连续特征
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

    return batched_user, batched_item, batched_context, labels_tensor


def evaluate(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for user_batch, item_batch, context_batch, labels in tqdm(
            data_loader, desc="Validating"
        ):
            user_batch = {k: v.to(device) for k, v in user_batch.items()}
            item_batch = {k: v.to(device) for k, v in item_batch.items()}
            context_batch = {k: v.to(device) for k, v in context_batch.items()}

            logits = model(user_batch, item_batch, context_batch)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_preds_tensor = torch.cat(all_preds).numpy()
    all_labels_tensor = torch.cat(all_labels).numpy()
    auc = roc_auc_score(all_labels_tensor, all_preds_tensor)
    return auc


# ==============================================================================
# 4. 训练主程序 (更新以使用新数据流)
# ==============================================================================
def main():
    # --- 超参数 ---
    BATCH_SIZE = 2048  # 可以适当调大Batch Size，因为数据加载更快了
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    EMBEDDING_DIM_PER_FEATURE = 64
    HIDDEN_DIM = 512
    FINAL_EMB_DIM = 256
    HEAD_NUMS = 16
    VALIDATION_SPLIT = 0.1
    CHECKPOINT_DIR = "models"

    # --- 设备设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 加载数据 (极大简化！)---
    print(f"Loading preprocessed data from: {PROCESSED_DATA_PATH}")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print("Data loaded.")

    label_counts = df["label"].value_counts()
    neg_samples, pos_samples = label_counts[0], label_counts[1]
    pos_weight = torch.tensor([neg_samples / pos_samples], device=device)
    print(f"Label distribution: {label_counts.to_dict()}")
    print(f"Calculated pos_weight for BCE loss: {pos_weight.item():.2f}")

    # --- 归一化连续特征 ---
    # StandardScaler 应当在划分训练集后，仅在训练集上 fit，然后再 transform 训练集和验证集
    # 这样可以防止验证集的信息泄露到训练过程中

    # --- 划分训练集和验证集 ---
    print(
        f"Splitting data into train and validation sets ({1-VALIDATION_SPLIT:.0%}:{VALIDATION_SPLIT:.0%})"
    )
    train_df, val_df = train_test_split(
        df, test_size=VALIDATION_SPLIT, random_state=42, stratify=df["label"]
    )
    # 重置索引，保证iloc的连续性
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")

    scaler = None
    # --- 在训练集上fit归一化器，并应用到训练集和验证集 ---
    if ALL_CONTINUOUS_FEATURES:
        print("Normalizing continuous features...")
        scaler = StandardScaler()
        # Fit a-n-d transform on training data
        train_df.loc[:, ALL_CONTINUOUS_FEATURES] = scaler.fit_transform(
            train_df[ALL_CONTINUOUS_FEATURES]
        )
        # Only transform on validation data
        val_df.loc[:, ALL_CONTINUOUS_FEATURES] = scaler.transform(
            val_df[ALL_CONTINUOUS_FEATURES]
        )
        print("Normalization complete.")
    else:
        print("No continuous features to normalize.")

    # --- 创建Dataset和DataLoader (极大简化！) ---
    num_worker_cores = min(os.cpu_count(), 32)  # 安全地获取CPU核心数 # type: ignore

    # Dataset的初始化变得非常简单
    train_dataset = RecSysDataset(train_df)
    val_dataset = RecSysDataset(val_df)

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

    model_config = {
        "EMBEDDING_DIM_PER_FEATURE": EMBEDDING_DIM_PER_FEATURE,
        "HIDDEN_DIM": HIDDEN_DIM,
        "FINAL_EMB_DIM": FINAL_EMB_DIM,
        "HEAD_NUMS": HEAD_NUMS,
    }

    model = Model(
        user_feature_defs=USER_FEATURE_DEFS,
        item_feature_defs=ITEM_FEATURE_DEFS,
        context_feature_defs=CONTEXT_FEATURE_DEFS,
        embedding_dim_per_feature=EMBEDDING_DIM_PER_FEATURE,
        hidden_dim=HIDDEN_DIM,
        emb_dim=FINAL_EMB_DIM,
        head_nums=HEAD_NUMS,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",  # 我们希望AUC最大化，所以是'max'
        factor=0.5,  # 学习率衰减系数
        patience=1,  # 容忍1个epoch没有提升
    )
    # torch.autograd.set_detect_anomaly(True) # 调试时开启，正常训练时可以注释掉以提高速度

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # 确保目录存在
    best_val_auc = 0.0  # 初始化最佳AUC

    # --- 训练循环 ---
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

        for user_batch, item_batch, context_batch, labels in progress_bar:
            user_batch = {k: v.to(device) for k, v in user_batch.items()}
            item_batch = {k: v.to(device) for k, v in item_batch.items()}
            context_batch = {k: v.to(device) for k, v in context_batch.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(user_batch, item_batch, context_batch)
            loss = criterion(logits, labels)

            # 检查NaN loss
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
            save_path = os.path.join(CHECKPOINT_DIR, "attn_best_model.pth")
            print(
                f"🎉 New best model found! AUC improved to {best_val_auc:.4f}. Saving model to {save_path}\n"
            )
            save_bundle = {
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'model_config': model_config,
                'feature_definitions': feature_definitions
            }
            torch.save(save_bundle, save_path)
        else:
            print(f"Validation AUC did not improve from {best_val_auc:.4f}.\n")

    print("Training complete.")
    print(f"Best validation AUC achieved: {best_val_auc:.4f}")
    print(
        f"The best model is saved at: {os.path.join(CHECKPOINT_DIR, 'best_model.pth')}"
    )


if __name__ == "__main__":
    main()
