import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.metrics import roc_auc_score
from rkmixer import RankMixer
import os

# Configuration
TRAIN_FILE = "train.csv"
BATCH_SIZE = 512
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Feature definitions
DROP_COLS = ["timestamp", "geohash", "work_geohash", "userid"]
NUM_COLS = [
    "distance",
    "item_ave_price",
    "price",
    "user_home_dis",
    "user_work_dis",
    "temp",
    "temp_low",
    "temp_high",
    "user_displayed_item_num",
    "online_days",
]


def preprocess_data(df):
    print("Preprocessing data...")

    # 1. Drop columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Identify categorical columns (all remaining except label and numericals)
    cat_cols = [c for c in df.columns if c not in NUM_COLS and c != "label"]

    # 2. Missing Values
    print("Handling missing values...")
    for col in df.columns:
        if col == "label":
            continue
        missing_rate = df[col].isnull().mean()
        if missing_rate > 0:
            if missing_rate < 0.10:
                # Fill with median for numerical, mode for categorical
                if col in NUM_COLS:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
            else:
                # Create is_missing feature
                df[f"{col}_is_missing"] = df[col].isnull().astype(int)
                # Fill original with median/mode to avoid NaNs
                if col in NUM_COLS:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
                # Add to appropriate list
                if col in NUM_COLS:
                    # The is_missing feature is categorical (0/1)
                    cat_cols.append(f"{col}_is_missing")
                else:
                    cat_cols.append(f"{col}_is_missing")

    # 3. Item ID Handling
    if "itemid" in df.columns:
        print("Processing itemid...")
        item_counts = df["itemid"].value_counts()
        # Threshold: e.g., keep items appearing at least 5 times
        # Or keep top N. Let's use a threshold of 5 for now.
        threshold = 5
        valid_items = item_counts[item_counts >= threshold].index
        df.loc[~df["itemid"].isin(valid_items), "itemid"] = "<UNK>"

    # 4. Numerical Binning
    print("Binning numerical features...")
    # Use KBinsDiscretizer or qcut. qcut is simpler for quantiles.
    # We'll use 10 bins for simplicity.
    for col in NUM_COLS:
        if col in df.columns:
            # Fill any remaining NaNs just in case
            df[col] = df[col].fillna(df[col].median())
            try:
                df[col] = pd.qcut(df[col], q=10, labels=False, duplicates="drop")
            except ValueError:
                # Fallback if unique values are too few
                df[col] = pd.cut(df[col], bins=10, labels=False)

            # Treat binned numericals as categorical now
            if col not in cat_cols:
                cat_cols.append(col)

    # 5. Label Encoding (Compression)
    print("Label encoding...")
    feature_dims = []
    features_list = []  # List of (vocab_size, emb_dim)

    # We need to encode all categorical columns (which now includes binned numericals)
    # We'll process them in a fixed order to ensure consistency
    # Sort cat_cols to be deterministic
    cat_cols = sorted(list(set(cat_cols)))

    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        # Convert to string to handle mixed types if any
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        vocab_size = len(le.classes_)
        # Simple rule for embedding dim: min(50, (vocab_size + 1) // 2)
        emb_dim = min(16, (vocab_size + 1) // 2)
        # Ensure at least 2
        emb_dim = max(2, emb_dim)

        features_list.append((vocab_size, emb_dim))

    return df, cat_cols, features_list


class RecDataset(Dataset):
    def __init__(self, df, cat_cols):
        self.labels = torch.tensor(df["label"].values, dtype=torch.float32)
        # Extract features in order
        self.features = torch.tensor(df[cat_cols].values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def main():
    # Load Data
    print(f"Loading {TRAIN_FILE}...")
    df = pd.read_csv(TRAIN_FILE)

    # Preprocess
    df, cat_cols, features_list = preprocess_data(df)

    print(f"Total features: {len(cat_cols)}")
    print(f"Feature dims: {features_list}")

    # Split Data
    # "保证训练集和验证集的weekday和hour分布大体一致"
    # We'll create a stratify column
    print("Splitting data...")
    if "weekday" in df.columns and "hour" in df.columns:
        # Note: weekday and hour are already encoded/binned in cat_cols,
        # but we can use the columns from df directly as they are now integers.
        stratify_col = df["weekday"].astype(str) + "_" + df["hour"].astype(str)
    else:
        stratify_col = df["label"]  # Fallback

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    # Use a subset for splitting if stratify_col has too many rare classes
    # But StratifiedShuffleSplit handles this reasonably well usually.
    # However, if some combinations are singletons, it will error.
    # Let's just use label for stratification if weekday/hour is too complex,
    # but prompt specifically asked for weekday/hour.
    # To be safe, we can filter out rare combinations or just try/except.

    try:
        train_idx, val_idx = next(split.split(df, stratify_col))
    except ValueError:
        print(
            "Warning: Stratification by weekday/hour failed (likely singletons). Falling back to label stratification."
        )
        train_idx, val_idx = next(split.split(df, df["label"]))

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_dataset = RecDataset(train_df, cat_cols)
    val_dataset = RecDataset(val_df, cat_cols)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model Init
    # RankMixer(num_tokens, num_heads, num_layers, d_model, hidden_dim, dropout, features)
    # We need to decide num_tokens.
    # RankMixer projects input to num_tokens * d_model.
    # num_tokens is effectively the sequence length for the transformer blocks.
    # We can choose an arbitrary number, e.g., 16 or 32.

    num_tokens = 16
    num_heads = 8
    num_layers = 8
    d_model = 256  # Must be divisible by num_heads (32/4 = 8)
    hidden_dim = 4 * d_model
    dropout = 0.1

    model = RankMixer(
        num_tokens=num_tokens,
        num_heads=num_heads,
        num_layers=num_layers,
        d_model=d_model,
        hidden_dim=hidden_dim,
        dropout=dropout,
        features=features_list,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs.squeeze(), y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                outputs = model(X)
                loss = criterion(outputs.squeeze(), y)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs.squeeze())
                all_targets.extend(y.cpu().numpy())
                all_preds.extend(probs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        auc = roc_auc_score(all_targets, all_preds)

        print(
            f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val AUC: {auc:.4f}"
        )


if __name__ == "__main__":
    main()
