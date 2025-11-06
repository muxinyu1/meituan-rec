import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import os
import json
import joblib
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# --- 【新增】deepctr-torch 相关导入 ---
# --- 【新增】deepctr-torch 相关导入 ---
from deepctr_torch.models import DeepFM, DCN, AutoInt, xDeepFM
from deepctr_torch.inputs import SparseFeat, DenseFeat


# deepctr-torch 使用 keras 的回调函数来实现早停
from tensorflow.python.keras.callbacks import EarlyStopping

# ==============================================================================
# 0. 导入你的两个模型
# ==============================================================================
from model import Model as ModelV1
from model_v2 import InteractionModel as ModelV2


# ==============================================================================
# 1. 基础设置 (和之前一样)
# ==============================================================================
PROCESSED_DATA_PATH = "data/processed/features_added_processed_samples.csv"
FEATURE_DEFS_PATH = "data/processed/feature_definitions_new.json"
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
# 【新增】一个包含所有特征名称的列表，方便 deepctr 使用
ALL_FEATURES = ALL_DISCRETE_FEATURES + ALL_CONTINUOUS_FEATURES

USER_FEATURE_DEFS = (USER_DISCRETE_VOCAB_SIZES, USER_CONTINUOUS_FEATURES)
ITEM_FEATURE_DEFS = (ITEM_DISCRETE_VOCAB_SIZES, ITEM_CONTINUOUS_FEATURES)
CONTEXT_FEATURE_DEFS = (CONTEXT_DISCRETE_VOCAB_SIZES, CONTEXT_CONTINUOUS_FEATURES)
print("Feature definitions loaded.\n")


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

    labels_tensor = torch.tensor(labels, dtype=torch.float)

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


def get_dl_predictions(model, data_loader, device):
    """辅助函数：获取深度学习模型的预测概率"""
    model.to(device)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for user_batch, item_batch, context_batch, _ in tqdm(
            data_loader, desc="Generating Custom DL predictions"
        ):
            user_batch = {k: v.to(device) for k, v in user_batch.items()}
            item_batch = {k: v.to(device) for k, v in item_batch.items()}
            context_batch = {k: v.to(device) for k, v in context_batch.items()}
            logits = model(user_batch, item_batch, context_batch)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu())
    return torch.cat(all_preds).numpy().flatten()


def evaluate_dl_model(model, data_loader, device):
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
    print(f"\n--- Training Custom PyTorch Model: {model_name} ---")
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
            print(f"Validation AUC improved. Model saved to {save_path}")
    print(
        f"--- Finished training {model_name}. Best Validation AUC: {best_val_auc:.4f} ---\n"
    )
    return best_val_auc


# ==============================================================================
# 3. LightGBM 和 deepctr 训练函数
# ==============================================================================


def train_lightgbm_model(train_df, val_df, scale_pos_weight):
    # 此函数与您提供的代码完全相同
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
    model = lgb.LGBMClassifier(**params)  # type: ignore
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        categorical_feature=ALL_DISCRETE_FEATURES,
        callbacks=[lgb.early_stopping(30, verbose=True)],
    )
    preds_val = model.predict_proba(X_val)[:, 1]  # type: ignore
    best_val_auc = roc_auc_score(y_val, preds_val)
    model.booster_.save_model(save_path)
    print(f"Model saved to {save_path}")
    print(
        f"--- Finished training {model_name}. Best Validation AUC: {best_val_auc:.4f} ---\n"
    )
    return best_val_auc


def train_deepctr_model(model_class, model_name, train_df, val_df, device, dl_params):
    """
    【修正后】通用的 deepctr 模型训练函数
    """
    print(f"\n--- Training deepctr Model: {model_name} ---")
    save_path = os.path.join(MODEL_DIR, f"{model_name}.pth")

    all_vocab_sizes = {
        **USER_DISCRETE_VOCAB_SIZES,
        **ITEM_DISCRETE_VOCAB_SIZES,
        **CONTEXT_DISCRETE_VOCAB_SIZES,
    }

    sparse_features = [
        SparseFeat(
            feat,
            vocabulary_size=all_vocab_sizes[feat],
            embedding_dim=dl_params["embedding_dim_per_feature"],
        )
        for feat in ALL_DISCRETE_FEATURES
    ]
    dense_features = [DenseFeat(feat, 1) for feat in ALL_CONTINUOUS_FEATURES]

    dnn_feature_columns = sparse_features + dense_features
    # 始终创建 linear_feature_columns，因为大部分模型都需要它作为第一个参数
    linear_feature_columns = sparse_features + dense_features

    # 关键修正：总是按顺序传入 linear 和 dnn 特征列。
    # 模型自身会决定如何使用它们。
    model = model_class(
        linear_feature_columns, dnn_feature_columns, task="binary", device=device
    )

    model.compile(
        torch.optim.Adam(model.parameters(), lr=dl_params["lr"]),
        "binary_crossentropy",
        metrics=["auc"],
    )

    train_model_input = {name: train_df[name].values for name in ALL_FEATURES}
    val_model_input = {name: val_df[name].values for name in ALL_FEATURES}

    y_train = train_df["label"].values
    y_val = val_df["label"].values

    es = EarlyStopping(monitor="val_auc", patience=3, verbose=1, mode="max")

    history = model.fit(
        train_model_input,
        y_train,
        batch_size=dl_params["batch_size"],
        epochs=dl_params["epochs"],
        verbose=1,
        validation_data=(val_model_input, y_val),
        callbacks=[es],
    )

    best_val_auc = max(history.history["val_auc"])
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    print(
        f"--- Finished training {model_name}. Best Validation AUC: {best_val_auc:.5f} ---\n"
    )
    return best_val_auc


# ==============================================================================
# 4. 主执行函数
# ==============================================================================

# 使用您定义的超参数
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 1024
EMB_DIM_PER_FEATURE = 64
EMB_DIM = 256
HIDDEN_DIM = 512
HEAD_NUMS = 16
LR = 5e-4
EPOCHS = 30


def main():

    num_worker_cores = min(int(os.cpu_count()), 16)  # type: ignore

    dl_params = {
        "lr": LR,
        "epochs": EPOCHS,
        "embedding_dim_per_feature": EMB_DIM_PER_FEATURE,
        "hidden_dim": HIDDEN_DIM,
        "batch_size": BATCH_SIZE,
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
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=num_worker_cores,
        pin_memory=True if device == "cuda" else False,
    )

    # --- 训练模型 1-3 ---
    model_v1 = ModelV1(
        embedding_dim_per_feature=EMB_DIM_PER_FEATURE,
        emb_dim=EMB_DIM,
        head_nums=HEAD_NUMS,
        hidden_dim=HIDDEN_DIM,
        user_feature_defs=USER_FEATURE_DEFS,
        item_feature_defs=ITEM_FEATURE_DEFS,
        context_feature_defs=CONTEXT_FEATURE_DEFS,
    )
    results["Model V1 (Attention)"] = train_deep_model(
        model_v1, "model_v1", train_loader, val_loader, device, pos_weight, dl_params_v1
    )

    model_v2 = ModelV2(
        embedding_dim_per_feature=EMB_DIM_PER_FEATURE,
        hidden_dim=HIDDEN_DIM,
        user_feature_defs=USER_FEATURE_DEFS,
        item_feature_defs=ITEM_FEATURE_DEFS,
        context_feature_defs=CONTEXT_FEATURE_DEFS,
    )
    results["Model V2 (Interaction MLP)"] = train_deep_model(
        model_v2, "model_v2", train_loader, val_loader, device, pos_weight, dl_params
    )

    results["Model V3 (LightGBM)"] = train_lightgbm_model(
        train_df, val_df, scale_pos_weight
    )

    # --- 【新增】训练模型 4-5 (deepctr) ---
    results["Model V4 (DeepFM)"] = train_deepctr_model(
        DeepFM, "deepfm", train_df, val_df, device, dl_params
    )
    results["Model V5 (DCN)"] = train_deepctr_model(
        DCN, "dcn", train_df, val_df, device, dl_params
    )

    results["Model V6 (xDeepFM)"] = train_deepctr_model(
        xDeepFM, "xdeepfm", train_df, val_df, device, dl_params
    )
    # 注意：为 AutoInt 传入 use_linear=False
    results["Model V7 (AutoInt)"] = train_deepctr_model(
        AutoInt, "autoint", train_df, val_df, device, dl_params
    )

    # +++++++++++++++++++++++ 扩展后的模型融合评估 +++++++++++++++++++++++
    print("\n\n" + "=" * 50)
    print(" " * 10 + "Ensemble Model Evaluation on Validation Set")
    print("=" * 50)

    print("Loading best saved models for ensemble evaluation...")
    # 1. 加载模型
    best_model_v1 = ModelV1(
        embedding_dim_per_feature=EMB_DIM_PER_FEATURE,
        hidden_dim=HIDDEN_DIM,
        emb_dim=EMB_DIM,
        head_nums=HEAD_NUMS,
        user_feature_defs=USER_FEATURE_DEFS,
        item_feature_defs=ITEM_FEATURE_DEFS,
        context_feature_defs=CONTEXT_FEATURE_DEFS,
    )
    best_model_v1.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model_v1.pth")))

    best_model_v2 = ModelV2(
        embedding_dim_per_feature=EMB_DIM_PER_FEATURE,
        hidden_dim=HIDDEN_DIM,
        user_feature_defs=USER_FEATURE_DEFS,
        item_feature_defs=ITEM_FEATURE_DEFS,
        context_feature_defs=CONTEXT_FEATURE_DEFS,
    )
    best_model_v2.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model_v2.pth")))

    best_lgbm = lgb.Booster(model_file=os.path.join(MODEL_DIR, "lightgbm.txt"))

    all_vocab_sizes = {
        **USER_DISCRETE_VOCAB_SIZES,
        **ITEM_DISCRETE_VOCAB_SIZES,
        **CONTEXT_DISCRETE_VOCAB_SIZES,
    }
    sparse_features = [
        SparseFeat(
            f,
            vocabulary_size=all_vocab_sizes[f],
            embedding_dim=dl_params["embedding_dim_per_feature"],  # type: ignore
        )
        for f in ALL_DISCRETE_FEATURES
    ]
    dense_features = [DenseFeat(f, 1) for f in ALL_CONTINUOUS_FEATURES]
    dnn_feature_columns = sparse_features + dense_features
    linear_feature_columns = sparse_features + dense_features

    best_deepfm = DeepFM(
        linear_feature_columns, dnn_feature_columns, task="binary", device=str(device)
    )
    best_deepfm.load_state_dict(torch.load(os.path.join(MODEL_DIR, "deepfm.pth")))

    best_dcn = DCN(
        linear_feature_columns, dnn_feature_columns, task="binary", device=str(device)
    )
    best_dcn.load_state_dict(torch.load(os.path.join(MODEL_DIR, "dcn.pth")))
    best_xdeepfm = xDeepFM(
        linear_feature_columns, dnn_feature_columns, task="binary", device=str(device)
    )
    best_xdeepfm.load_state_dict(torch.load(os.path.join(MODEL_DIR, "xdeepfm.pth")))

    best_autoint = AutoInt(
        linear_feature_columns, dnn_feature_columns, task="binary", device=str(device)
    )
    best_autoint.load_state_dict(torch.load(os.path.join(MODEL_DIR, "autoint.pth")))
    # 2. 生成预测
    print("Generating predictions from each model on the validation set...")
    preds_v1 = get_dl_predictions(best_model_v1, val_loader, device)
    preds_v2 = get_dl_predictions(best_model_v2, val_loader, device)

    X_val = val_df.drop("label", axis=1)
    preds_lgbm = best_lgbm.predict(X_val)

    val_model_input = {name: val_df[name].values for name in ALL_FEATURES}
    preds_deepfm = best_deepfm.predict(val_model_input, batch_size=BATCH_SIZE * 2)
    preds_dcn = best_dcn.predict(val_model_input, batch_size=BATCH_SIZE * 2)
    preds_xdeepfm = best_xdeepfm.predict(val_model_input, batch_size=BATCH_SIZE * 2)
    preds_autoint = best_autoint.predict(val_model_input, batch_size=BATCH_SIZE * 2)
    # 3. 融合预测
    y_val = val_df["label"].values

    print("Ensembling predictions with equal weights (1/5 each)...")
    ensemble_preds = (
        preds_v1.flatten()
        + preds_v2.flatten()
        + preds_lgbm.flatten()  # type: ignore
        + preds_deepfm.flatten()
        + preds_dcn.flatten()
        + preds_xdeepfm.flatten()
        + preds_autoint.flatten()
    ) / 7.0
    results["Ensemble (5 Models)"] = roc_auc_score(y_val, ensemble_preds)

    # --- 最终总结 ---
    print("\n\n" + "=" * 50)
    print(" " * 15 + "Final Training Summary")
    print("=" * 50)
    best_auc = 0
    best_model_name = ""
    for model_name, auc in results.items():
        if auc > best_auc:
            best_auc = auc
            best_model_name = model_name

    for model_name, auc in results.items():
        is_best_str = "  <--- BEST" if model_name == best_model_name else ""
        print(
            f"{model_name:<30}: Best Validation AUC = {auc:.5f}{is_best_str}"
        )  # 显示5位小数

    print("=" * 50)
    print("All models and the scaler have been saved in the 'models/' directory.")


if __name__ == "__main__":
    main()
