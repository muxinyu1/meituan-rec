import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import os
import json
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# ==============================================================================
# 0. 导入你的两个模型
# ==============================================================================
from model import Model as ModelV1
from model_v2 import InteractionModel as ModelV2


# ==============================================================================
# 1. 基础设置 (和之前一样)
# ==============================================================================

PROCESSED_DATA_PATH = "data/processed/processed_samples.csv"
FEATURE_DEFS_PATH = "data/processed/feature_definitions.json"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("--- Loading Feature Definitions ---")
with open(FEATURE_DEFS_PATH, "r") as f:
    feature_definitions = json.load(f)

USER_DISCRETE_VOCAB_SIZES = feature_definitions["user"]["discrete"]
USER_CONTINUOUS_FEATURES = feature_definitions["user"]["continuous"]
ITEM_DISCRETE_VOCAB_SIZES = feature_definitions["item"]["discrete"]
ITEM_CONTINUOUS_FEATURES = feature_definitions["item"]["continuous"]
CONTEXT_DISCRETE_VOCAB_SIZES = feature_definitions["context"]["discrete"]
CONTEXT_CONTINUOUS_FEATURES = feature_definitions["context"]["continuous"]

ALL_CONTINUOUS_FEATURES = (
    USER_CONTINUOUS_FEATURES + ITEM_CONTINUOUS_FEATURES + CONTEXT_CONTINUOUS_FEATURES
)
ALL_DISCRETE_FEATURES = (
    list(USER_DISCRETE_VOCAB_SIZES.keys())
    + list(ITEM_DISCRETE_VOCAB_SIZES.keys())
    + list(CONTEXT_DISCRETE_VOCAB_SIZES.keys())
)

USER_FEATURE_DEFS = (USER_DISCRETE_VOCAB_SIZES, USER_CONTINUOUS_FEATURES)
ITEM_FEATURE_DEFS = (ITEM_DISCRETE_VOCAB_SIZES, ITEM_CONTINUOUS_FEATURES)
CONTEXT_FEATURE_DEFS = (CONTEXT_DISCRETE_VOCAB_SIZES, CONTEXT_CONTINUOUS_FEATURES)
print("Feature definitions loaded.\n")


# ==============================================================================
# 2. 深度学习模型的共享组件 (和之前一样)
# ==============================================================================


class RecSysDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df
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
        sample_row = self.df.iloc[idx]
        user_features = {
            name: sample_row[name]
            for name in self.user_discrete_cols + self.user_continuous_cols
        }
        item_features = {
            name: sample_row[name]
            for name in self.item_discrete_cols + self.item_continuous_cols
        }
        context_features = {
            name: sample_row[name]
            for name in self.context_discrete_cols + self.context_continuous_cols
        }
        label = sample_row[self.label_col]
        return user_features, item_features, context_features, label


def collate_batch(batch):
    batched_user, batched_item, batched_context = {}, {}, {}
    labels = []
    for feat_type, feat_dict in [
        ("user", batched_user),
        ("item", batched_item),
        ("context", batched_context),
    ]:
        for name in feature_definitions[feat_type]["discrete"]:
            feat_dict[name] = []
        for name in feature_definitions[feat_type]["continuous"]:
            feat_dict[name] = []

    for user_feats, item_feats, context_feats, label in batch:
        for name, val in user_feats.items():
            batched_user[name].append(val)
        for name, val in item_feats.items():
            batched_item[name].append(val)
        for name, val in context_feats.items():
            batched_context[name].append(val)
        labels.append(label)

    for feat_dict, feat_type in [
        (batched_user, "user"),
        (batched_item, "item"),
        (batched_context, "context"),
    ]:
        for name, vals in feat_dict.items():
            is_continuous = name in feature_definitions[feat_type]["continuous"]
            dtype = torch.float if is_continuous else torch.long
            feat_dict[name] = torch.tensor(vals, dtype=dtype)

    labels_tensor = torch.tensor(labels, dtype=torch.float)
    return batched_user, batched_item, batched_context, labels_tensor


def get_dl_predictions(model, data_loader, device):
    """辅助函数：获取深度学习模型的预测概率"""
    model.to(device)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for user_batch, item_batch, context_batch, _ in tqdm(
            data_loader, desc="Generating DL predictions"
        ):
            user_batch = {k: v.to(device) for k, v in user_batch.items()}
            item_batch = {k: v.to(device) for k, v in item_batch.items()}
            context_batch = {k: v.to(device) for k, v in context_batch.items()}

            logits = model(user_batch, item_batch, context_batch)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu())

    return torch.cat(all_preds).numpy().flatten()  # 返回1维numpy数组


def evaluate_dl_model(model, data_loader, device):
    """评估函数现在只返回AUC"""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for user_batch, item_batch, context_batch, labels in data_loader:
            user_batch = {k: v.to(device) for k, v in user_batch.items()}
            item_batch = {k: v.to(device) for k, v in item_batch.items()}
            context_batch = {k: v.to(device) for k, v in context_batch.items()}

            logits = model(user_batch, item_batch, context_batch)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu())
            all_labels.append(labels.unsqueeze(1).cpu())

    all_preds_tensor = torch.cat(all_preds).numpy()
    all_labels_tensor = torch.cat(all_labels).numpy()
    return roc_auc_score(all_labels_tensor, all_preds_tensor)


def train_deep_model(
    model, model_name, train_loader, val_loader, device, pos_weight, params
):
    """通用深度学习模型训练函数 (与之前相同)"""
    print(f"\n--- Training Deep Learning Model: {model_name} ---")
    save_path = os.path.join(MODEL_DIR, f"{model_name}.pth")

    model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )

    best_val_auc = 0.0
    for epoch in range(params["epochs"]):
        model.train()
        total_loss = 0
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{params['epochs']} [Train]"
        )
        for user_batch, item_batch, context_batch, labels in progress_bar:
            user_batch = {k: v.to(device) for k, v in user_batch.items()}
            item_batch = {k: v.to(device) for k, v in item_batch.items()}
            context_batch = {k: v.to(device) for k, v in context_batch.items()}
            labels = labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(user_batch, item_batch, context_batch)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        val_auc = evaluate_dl_model(model, val_loader, device)
        scheduler.step(val_auc)
        print(
            f"Epoch {epoch+1}/{params['epochs']} | Avg Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), save_path)
            print(
                f"Validation AUC improved to {best_val_auc:.4f}. Model saved to {save_path}"
            )
    print(
        f"--- Finished training {model_name}. Best Validation AUC: {best_val_auc:.4f} ---\n"
    )
    return best_val_auc


# ==============================================================================
# 3. LightGBM 模型训练函数 (与之前相同)
# ==============================================================================
def train_lightgbm_model(train_df, val_df, scale_pos_weight):
    model_name = "lightgbm"
    print(f"\n--- Training Tree-based Model: {model_name} ---")
    save_path = os.path.join(MODEL_DIR, f"{model_name}.txt")
    y_train = train_df["label"]
    X_train = train_df.drop("label", axis=1)
    y_val = val_df["label"]
    X_val = val_df.drop("label", axis=1)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "scale_pos_weight": scale_pos_weight,
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "seed": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        categorical_feature=ALL_DISCRETE_FEATURES,
        callbacks=[lgb.early_stopping(30, verbose=True)],
    )
    preds_val = model.predict_proba(X_val)[:, 1]
    best_val_auc = roc_auc_score(y_val, preds_val)
    model.booster_.save_model(save_path)
    print(f"Model saved to {save_path}")
    print(
        f"--- Finished training {model_name}. Best Validation AUC: {best_val_auc:.4f} ---\n"
    )
    return best_val_auc


VALIDATION_SPLIT = 0.2
BATCH_SIZE = 1024
EMB_DIM_PER_FEATURE = 32
EMB_DIM = 256
HIDDEN_DIM = 512
HEAD_NUMS = 16
LR = 5e-4
EPOCHS = 5


# ==============================================================================
# 4. 主执行函数
# ==============================================================================
def main():

    num_worker_cores = min(int(os.cpu_count()), 16)  # type: ignore

    dl_params = {
        "lr": LR,
        "epochs": EPOCHS,
        "embedding_dim_per_feature": EMB_DIM_PER_FEATURE,
        "hidden_dim": HIDDEN_DIM,
    }
    dl_params_v1 = {**dl_params, "emb_dim": EMB_DIM, "head_nums": HEAD_NUMS}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    print("--- Loading & Preprocessing Data ---")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    label_counts = df["label"].value_counts()
    neg_samples, pos_samples = label_counts[0], label_counts[1]
    pos_weight = torch.tensor([neg_samples / pos_samples], device=device)
    scale_pos_weight = neg_samples / pos_samples
    print(f"Label distribution: Negative={neg_samples}, Positive={pos_samples}")
    print(f"Calculated pos_weight: {pos_weight.item():.2f}\n")

    train_df, val_df = train_test_split(
        df, test_size=VALIDATION_SPLIT, random_state=42, stratify=df["label"]
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    if ALL_CONTINUOUS_FEATURES:
        scaler = StandardScaler()
        train_df.loc[:, ALL_CONTINUOUS_FEATURES] = scaler.fit_transform(
            train_df[ALL_CONTINUOUS_FEATURES]
        )
        val_df.loc[:, ALL_CONTINUOUS_FEATURES] = scaler.transform(
            val_df[ALL_CONTINUOUS_FEATURES]
        )
        scaler_path = os.path.join(MODEL_DIR, "standard_scaler.joblib")
        joblib.dump(scaler, scaler_path)
        print(f"StandardScaler saved to {scaler_path}.\n")

    results = {}

    # --- DL模型的数据加载器 ---
    train_dataset = RecSysDataset(train_df)
    val_dataset = RecSysDataset(val_df)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=num_worker_cores,
        pin_memory=True if device == "cuda" else False,
    )
    val_loader = DataLoader(  # Dataloader for validation
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=num_worker_cores,
        pin_memory=True if device == "cuda" else False,
    )

    # --- 训练模型 ---
    model_v1 = ModelV1(
        embedding_dim_per_feature=EMB_DIM_PER_FEATURE,
        hidden_dim=HIDDEN_DIM,
        head_nums=HEAD_NUMS,
        emb_dim=EMB_DIM,
        user_feature_defs=USER_FEATURE_DEFS,
        item_feature_defs=ITEM_FEATURE_DEFS,
        context_feature_defs=CONTEXT_FEATURE_DEFS,
    )
    auc_v1 = train_deep_model(
        model_v1, "model_v1", train_loader, val_loader, device, pos_weight, dl_params_v1
    )
    results["Model V1 (Attention)"] = auc_v1

    model_v2 = ModelV2(
        embedding_dim_per_feature=EMB_DIM_PER_FEATURE,
        hidden_dim=HIDDEN_DIM,
        user_feature_defs=USER_FEATURE_DEFS,
        item_feature_defs=ITEM_FEATURE_DEFS,
        context_feature_defs=CONTEXT_FEATURE_DEFS,
    )
    auc_v2 = train_deep_model(
        model_v2, "model_v2", train_loader, val_loader, device, pos_weight, dl_params
    )
    results["Model V2 (Interaction MLP)"] = auc_v2

    auc_lgbm = train_lightgbm_model(train_df, val_df, scale_pos_weight)
    results["Model V3 (LightGBM)"] = auc_lgbm

    # +++++++++++++++++++++++ 【新增】模型融合评估环节 +++++++++++++++++++++++
    print("\n\n" + "=" * 50)
    print(" " * 10 + "Ensemble Model Evaluation on Validation Set")
    print("=" * 50)

    # --- 1. 加载所有最佳模型 ---
    print("Loading best saved models for ensemble evaluation...")
    # 加载模型V1
    best_model_v1 = ModelV1(
        hidden_dim=HIDDEN_DIM,
        embedding_dim_per_feature=EMB_DIM_PER_FEATURE,
        emb_dim=EMB_DIM,
        head_nums=HEAD_NUMS,
        user_feature_defs=USER_FEATURE_DEFS,
        item_feature_defs=ITEM_FEATURE_DEFS,
        context_feature_defs=CONTEXT_FEATURE_DEFS,
    )
    best_model_v1.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model_v1.pth")))

    # 加载模型V2
    best_model_v2 = ModelV2(
        embedding_dim_per_feature=EMB_DIM_PER_FEATURE,
        hidden_dim=HIDDEN_DIM,
        user_feature_defs=USER_FEATURE_DEFS,
        item_feature_defs=ITEM_FEATURE_DEFS,
        context_feature_defs=CONTEXT_FEATURE_DEFS,
    )
    best_model_v2.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model_v2.pth")))

    # 加载LightGBM
    best_lgbm = lgb.Booster(model_file=os.path.join(MODEL_DIR, "lightgbm.txt"))

    # --- 2. 在验证集上生成预测概率 ---
    print("Generating predictions from each model on the validation set...")
    # DL模型预测
    preds_v1 = get_dl_predictions(best_model_v1, val_loader, device)
    preds_v2 = get_dl_predictions(best_model_v2, val_loader, device)

    # LightGBM预测
    X_val = val_df.drop("label", axis=1)
    preds_lgbm = best_lgbm.predict(X_val)

    # 获取真实标签
    y_val = val_df["label"].values

    # --- 3. 融合预测并计算AUC ---
    # 确保所有预测都是相同长度
    assert (
        len(preds_v1) == len(y_val)
        and len(preds_v2) == len(y_val)
        and len(preds_lgbm) == len(y_val)
    )

    print("Ensembling predictions with equal weights (1/3 each)...")
    ensemble_preds = (preds_v1 + preds_v2 + preds_lgbm) / 3.0
    ensemble_auc = roc_auc_score(y_val, ensemble_preds)
    results["Ensemble (Equal Weights)"] = ensemble_auc
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # --- 最终总结 ---
    print("\n\n" + "=" * 50)
    print(" " * 15 + "Final Training Summary")
    print("=" * 50)
    # 使用 '*' 标记性能最佳的项
    best_auc = 0
    best_model_name = ""
    for model_name, auc in results.items():
        if auc > best_auc:
            best_auc = auc
            best_model_name = model_name

    for model_name, auc in results.items():
        is_best_str = "  <--- BEST" if model_name == best_model_name else ""
        print(f"{model_name:<30}: Best Validation AUC = {auc:.4f}{is_best_str}")

    print("=" * 50)
    print("All models and the scaler have been saved in the 'models/' directory.")


if __name__ == "__main__":
    main()
