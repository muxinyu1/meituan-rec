import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import json

# --- 1. 配置区域 ---

# 定义输入文件路径
DATA_DIR = "data/recsys_task_data"
TRAIN_SAMPLES_PATH = os.path.join(DATA_DIR, "train_samples-20221014-crossed.csv")
ITEM_INFOS_PATH = os.path.join(DATA_DIR, "item_infos-20221014.csv")
USER_INFOS_PATH = os.path.join(DATA_DIR, "user_infos_crossed_features.csv")

# 定义输出文件路径
OUTPUT_DIR = "data/processed"
PROCESSED_DATA_PATH = os.path.join(OUTPUT_DIR, "processed_samples.csv")
FEATURE_DEFS_PATH = os.path.join(OUTPUT_DIR, "feature_definitions.json")


# 定义特征类型，这是整个脚本的核心
# 我们在这里明确区分用户、物品、上下文的离散和连续特征

# 物品特征
ITEM_DISCRETE_FEATURES = ["dtype", "cate_1", "cate_2", "cate_3"]
ITEM_CONTINUOUS_FEATURES = []

# 用户特征
USER_DISCRETE_FEATURES = [
    "age",
    "level",
    "gender",
    "married",
    "job",
    "has_car",
    "mobile_type",
    "mobile_os",
    "cross_age_gender_encoded",
    "cross_age_married_encoded",
    "cross_gender_married_encoded",
    "cross_age_gender_married_encoded",
    "cross_married_has_car_encoded",
    "cross_level_age_encoded",
    "cross_level_gender_encoded",
    "cross_level_job_encoded",
    "cross_mobile_os_age_encoded",
]
# 'work_geohash' 基数太大，暂时不作为简单离散特征处理
USER_CONTINUOUS_FEATURES = ["online_days", "user_displayed_item_num"]


# 上下文特征
CONTEXT_DISCRETE_FEATURES = [
    "cityid",
    "loc_cityid",
    "weekday",
    "hour",
    "weather",
    "is_weekend",
    "is_night",
    "time_slot",
    "is_same_city",
    "temp_comfort",
    "cityid_bucket",
]
CONTEXT_CONTINUOUS_FEATURES = [
    "distance",
    "user_home_dis",
    "user_work_dis",
    "temp",
    "temp_low",
    "temp_high",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "temp_range",
    "item_ave_price",
      "price"
]

# ID 特征，需要进行 Label Encoding，但不归入模型输入特征
ID_FEATURES = ["userid", "itemid"]


def preprocess():
    """
    执行完整的数据预处理流程。
    """
    print("--- Step 1: Loading data ---")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df_train = pd.read_csv(TRAIN_SAMPLES_PATH)
    df_item = pd.read_csv(ITEM_INFOS_PATH)
    df_user = pd.read_csv(USER_INFOS_PATH)

    print(f"Train samples: {len(df_train)}")
    print(f"Item infos: {len(df_item)}")
    print(f"User infos: {len(df_user)}")

    print("\n--- Step 2: Merging dataframes ---")
    df = pd.merge(df_train, df_user, on="userid", how="left")
    df = pd.merge(df, df_item, on="itemid", how="left")
    print(f"Merged dataframe shape: {df.shape}")

    print("\n--- Step 3: Handling data types and missing values ---")
    # 结合所有离散特征列表
    all_discrete_features = (
        ITEM_DISCRETE_FEATURES + USER_DISCRETE_FEATURES + CONTEXT_DISCRETE_FEATURES
    )
    # 结合所有连续特征列表
    all_continuous_features = (
        ITEM_CONTINUOUS_FEATURES
        + USER_CONTINUOUS_FEATURES
        + CONTEXT_CONTINUOUS_FEATURES
    )

    # 填充缺失值并转换类型
    for col in all_discrete_features:
        # 使用 -1 或一个特殊值填充离散特征的缺失值
        df[col] = df[col].fillna(-1).astype(int)

    for col in all_continuous_features:
        # 使用均值填充连续特征的缺失值
        mean_val = df[col].mean()
        df[col] = df[col].fillna(mean_val).astype(float)

    # 特殊处理 'price_is_missing' 等标志列
    flag_cols = [
        "price_is_missing",
        "online_days_is_missing",
        "user_displayed_item_num_is_missing",
    ]
    for col in flag_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
            # 将这些标志列也加入到离散特征中，因为它们是有用信息的
            if col not in all_discrete_features:
                CONTEXT_DISCRETE_FEATURES.append(col)
                all_discrete_features.append(col)

    print("Data types and missing values handled.")

    print("\n--- Step 4: Encoding discrete features and IDs ---")
    feature_definitions = {
        "user": {"discrete": {}, "continuous": USER_CONTINUOUS_FEATURES},
        "item": {"discrete": {}, "continuous": ITEM_CONTINUOUS_FEATURES},
        "context": {"discrete": {}, "continuous": CONTEXT_CONTINUOUS_FEATURES},
    }

    # 编码所有ID和离散特征
    # 这是关键步骤，确保了所有类别都从0开始映射
    for col in ID_FEATURES + all_discrete_features:
        print(f"  Encoding '{col}'...")
        le = LabelEncoder()
        # 使用 fit_transform 来创建从0开始的编码
        df[col] = le.fit_transform(df[col])

        # 计算词表大小（编码后的最大值 + 1）
        vocab_size = int(df[col].max() + 1)

        # 存储词表大小到 feature_definitions
        if col in USER_DISCRETE_FEATURES:
            feature_definitions["user"]["discrete"][col] = vocab_size
        elif col in ITEM_DISCRETE_FEATURES:
            feature_definitions["item"]["discrete"][col] = vocab_size
        elif col in CONTEXT_DISCRETE_FEATURES:
            feature_definitions["context"]["discrete"][col] = vocab_size
        # ID特征也被编码，但我们不把它们放入模型的特征定义中
        elif col in ID_FEATURES:
            print(
                f"    '{col}' is an ID feature. Encoded with vocab size: {vocab_size}"
            )

    print("Encoding complete.")

    # --- 最终检查 ---
    # 检查是否存在标准差为0的连续特征
    print("\n--- Step 5: Sanity check for continuous features ---")
    for col in all_continuous_features:
        if df[col].std() < 1e-9:  # 使用一个很小的值来比较浮点数
            print(
                f"!!! WARNING: Continuous feature '{col}' has zero standard deviation. !!!"
            )
            print(f"This feature might cause 'NaN' issues during normalization.")
            print(f"Consider removing it from features list if warning persists.")
    print("Sanity check complete.")

    print(f"\n--- Step 6: Saving processed data to '{PROCESSED_DATA_PATH}' ---")
    # 选择要保存的列，这里我们保存所有处理过的列
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print("Data saved.")

    print(f"\n--- Step 7: Saving feature definitions to '{FEATURE_DEFS_PATH}' ---")
    with open(FEATURE_DEFS_PATH, "w") as f:
        json.dump(feature_definitions, f, indent=4)
    print("Feature definitions saved.")

    print("\n--- Preprocessing Finished ---")
    print("You can now use the generated files in your training script.")
    print("\nCopy the following Python code snippet into your 'train.py':")

    # 打印可以直接复制到 train.py 的代码
    print("\n" + "=" * 50)
    print("# Generated Feature Definitions (copy to train.py)\n")
    print(
        f"USER_DISCRETE_VOCAB_SIZES = {json.dumps(feature_definitions['user']['discrete'], indent=4)}"
    )
    print(f"USER_CONTINUOUS_FEATURES = {feature_definitions['user']['continuous']}\n")
    print(
        f"ITEM_DISCRETE_VOCAB_SIZES = {json.dumps(feature_definitions['item']['discrete'], indent=4)}"
    )
    print(f"ITEM_CONTINUOUS_FEATURES = {feature_definitions['item']['continuous']}\n")
    print(
        f"CONTEXT_DISCRETE_VOCAB_SIZES = {json.dumps(feature_definitions['context']['discrete'], indent=4)}"
    )
    print(
        f"CONTEXT_CONTINUOUS_FEATURES = {feature_definitions['context']['continuous']}\n"
    )
    print("=" * 50 + "\n")


if __name__ == "__main__":
    preprocess()
