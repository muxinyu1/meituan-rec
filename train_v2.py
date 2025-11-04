import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import os
import json  # 我们将从 json 文件加载特征定义

# 新增导入
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# 假设你的模型定义在 model.py 文件中
from model_v2 import InteractionModel

# ==============================================================================
# 1. 从文件加载特征定义 (新方式！)
# ==============================================================================

# 定义预处理后的数据和特征定义文件路径
PROCESSED_DATA_PATH = "data/processed/processed_samples.csv"
FEATURE_DEFS_PATH = "data/processed/feature_definitions.json"

print(f"Loading feature definitions from: {FEATURE_DEFS_PATH}")
with open(FEATURE_DEFS_PATH, "r") as f:
    feature_definitions = json.load(f)

# 直接从加载的json文件中获取特征定义
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

# ==============================================================================
# 2. 自定义 Dataset (极大简化！)
# ==============================================================================


class RecSysDataset(Dataset):
    """
    一个极大简化的Dataset类。
    它只接收一个已经合并和预处理好的DataFrame。
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

        # 预先提取列名列表以提高效率
        self.user_discrete_cols = list(USER_DISCRETE_VOCAB_SIZES.keys())
        self.user_continuous_cols = USER_CONTINUOUS_FEATURES
        self.item_discrete_cols = list(ITEM_DISCRETE_VOCAB_SIZES.keys())
        self.item_continuous_cols = ITEM_CONTINUOUS_FEATURES
        self.context_discrete_cols = list(CONTEXT_DISCRETE_VOCAB_SIZES.keys())
        self.context_continuous_cols = CONTEXT_CONTINUOUS_FEATURES

        self.label_col = "label"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 直接从一行中获取所有数据，不再需要多表查找
        sample_row = self.df.iloc[idx]

        # 构建用户特征字典
        user_features = {name: sample_row[name] for name in self.user_discrete_cols}
        for name in self.user_continuous_cols:
            user_features[name] = sample_row[name]

        # 构建物品特征字典
        item_features = {name: sample_row[name] for name in self.item_discrete_cols}
        for name in self.item_continuous_cols:
            item_features[name] = sample_row[name]

        # 构建上下文特征字典
        context_features = {
            name: sample_row[name] for name in self.context_discrete_cols
        }
        for name in self.context_continuous_cols:
            context_features[name] = sample_row[name]

        label = sample_row[self.label_col]

        return user_features, item_features, context_features, label


# Collate Function 和 evaluate 函数保持不变，因为我们输出的数据结构是一致的
# ==============================================================================
# 3. Collate Function 和 评估函数 (保持不变)
# ==============================================================================


def collate_batch(batch):
    batched_user, batched_item, batched_context = {}, {}, {}
    labels = []

    # 初始化批处理字典
    for name in USER_DISCRETE_VOCAB_SIZES:
        batched_user[name] = []
    for name in USER_CONTINUOUS_FEATURES:
        batched_user[name] = []
    for name in ITEM_DISCRETE_VOCAB_SIZES:
        batched_item[name] = []
    for name in ITEM_CONTINUOUS_FEATURES:
        batched_item[name] = []
    for name in CONTEXT_DISCRETE_VOCAB_SIZES:
        batched_context[name] = []
    for name in CONTEXT_CONTINUOUS_FEATURES:
        batched_context[name] = []

    for user_feats, item_feats, context_feats, label in batch:
        for name, val in user_feats.items():
            batched_user[name].append(val)
        for name, val in item_feats.items():
            batched_item[name].append(val)
        for name, val in context_feats.items():
            batched_context[name].append(val)
        labels.append(label)

    # 转换为Tensor
    for name, vals in batched_user.items():
        dtype = torch.float if name in USER_CONTINUOUS_FEATURES else torch.long
        batched_user[name] = torch.tensor(vals, dtype=dtype)
    for name, vals in batched_item.items():
        dtype = torch.float if name in ITEM_CONTINUOUS_FEATURES else torch.long
        batched_item[name] = torch.tensor(vals, dtype=dtype)
    for name, vals in batched_context.items():
        dtype = torch.float if name in CONTEXT_CONTINUOUS_FEATURES else torch.long
        batched_context[name] = torch.tensor(vals, dtype=dtype)

    labels_tensor = torch.tensor(labels, dtype=torch.float).unsqueeze(1)
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
    BATCH_SIZE = 1024  # 可以适当调大Batch Size，因为数据加载更快了
    LEARNING_RATE = 5e-4
    EPOCHS = 50
    EMBEDDING_DIM_PER_FEATURE = 32
    HIDDEN_DIM = 256
    FINAL_EMB_DIM = 128
    HEAD_NUMS = 4
    VALIDATION_SPLIT = 0.2

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
    num_worker_cores = min(os.cpu_count(), 16)  # type: ignore

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

    # --- 初始化模型、损失函数、优化器 ---
    model = InteractionModel(  # 或者直接用 InteractionModel(...)
        user_feature_defs=USER_FEATURE_DEFS,
        item_feature_defs=ITEM_FEATURE_DEFS,
        context_feature_defs=CONTEXT_FEATURE_DEFS,
        embedding_dim_per_feature=EMBEDDING_DIM_PER_FEATURE,
        hidden_dim=HIDDEN_DIM,
        # emb_dim 和 head_nums 参数已经不存在了
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",  # 我们希望AUC最大化，所以是'max'
        factor=0.5,  # 学习率衰减系数
        patience=1,  # 容忍1个epoch没有提升
    )
    # torch.autograd.set_detect_anomaly(True) # 调试时开启，正常训练时可以注释掉以提高速度

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
            f"\nEpoch {epoch+1}/{EPOCHS} Summary | Avg Train Loss: {avg_loss:.4f} | Validation AUC: {val_auc:.4f}\n"
        )

    print("Training complete.")


if __name__ == "__main__":
    main()
