#!/usr/bin/env python3
"""
Analyze the coverage and distribution of Test Set Users/Items within the Training Set.
"""
import pandas as pd
from pathlib import Path
import numpy as np

# --- 配置路径 ---
TRAIN_PATH = Path("data/recsys_task_data/train_merged_transformed_train-20221014.csv")
TEST_PATH = Path("data/recsys_task_data/test_merged_transformed-20221019.csv")

def analyze_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame, col: str, col_name: str):
    print(f"\n{'='*20} 分析维度: {col_name} ({col}) {'='*20}")
    
    # 1. 基础集合分析 (Set Analysis)
    train_ids = set(train_df[col].unique())
    test_ids = set(test_df[col].unique())
    
    n_train_unique = len(train_ids)
    n_test_unique = len(test_ids)
    
    # 交集：测试集中有多少 ID 在训练集中出现过
    seen_ids = test_ids.intersection(train_ids)
    n_seen = len(seen_ids)
    n_unseen = len(test_ids) - n_seen
    
    print(f"[{col_name} - ID 维度]")
    print(f"  - 训练集唯一 ID 数: {n_train_unique}")
    print(f"  - 测试集唯一 ID 数: {n_test_unique}")
    print(f"  - 测试集中【已知】ID 数 (Warm Start): {n_seen} ({n_seen/n_test_unique:.2%})")
    print(f"  - 测试集中【未知】ID 数 (Cold Start): {n_unseen} ({n_unseen/n_test_unique:.2%})")

    # 2. 样本覆盖率分析 (Sample/Row Analysis)
    # 测试集的样本中，有多少条数据是属于“已知 ID”的？
    # 这一点往往比 ID 覆盖率更重要，因为活跃用户可能贡献了测试集的大部分流量
    test_in_train_mask = test_df[col].isin(seen_ids)
    n_test_rows = len(test_df)
    n_known_rows = test_in_train_mask.sum()
    n_unknown_rows = n_test_rows - n_known_rows
    
    print(f"[{col_name} - 样本流量维度]")
    print(f"  - 测试集总样本数: {n_test_rows}")
    print(f"  - 涉及【已知】ID 的样本数: {n_known_rows} ({n_known_rows/n_test_rows:.2%})")
    print(f"  - 涉及【未知】ID 的样本数: {n_unknown_rows} ({n_unknown_rows/n_test_rows:.2%})")

    # 3. 历史活跃度/热度分析 (History Frequency)
    # 对于那些“已知”的 User/Item，它们在训练集中到底有多“热”？
    if n_seen > 0:
        # 计算训练集中的频率
        train_counts = train_df[col].value_counts()
        
        # 提取测试集中出现的那些 ID 在训练集中的频次
        # 注意：这里我们只看 unique ID 的历史频次分布
        seen_id_counts = train_counts[list(seen_ids)]
        
        print(f"[{col_name} - 历史热度分布 (仅针对已知 ID)]")
        print(f"  这些 ID 在训练集中出现的次数统计:")
        print(f"  - Mean (平均次数): {seen_id_counts.mean():.2f}")
        print(f"  - Median (中位数): {seen_id_counts.median():.2f}")
        print(f"  - Min  (最少次数): {seen_id_counts.min()}")
        print(f"  - Max  (最多次数): {seen_id_counts.max()}")
        
        # 分位数概览
        quantiles = seen_id_counts.quantile([0.25, 0.5, 0.75, 0.9, 0.99])
        print(f"  - 25% 的 ID 历史记录少于: {quantiles[0.25]:.0f} 次")
        print(f"  - 75% 的 ID 历史记录少于: {quantiles[0.75]:.0f} 次")
        print(f"  - 99% 的 ID 历史记录少于: {quantiles[0.99]:.0f} 次")
    else:
        print("  无已知 ID，跳过热度分析。")

def main():
    # 仅读取需要的列以节省内存
    use_cols = ["userid", "itemid"]
    
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        print("错误：文件未找到，请检查路径。")
        return

    print("正在加载数据 (仅 userid, itemid)...")
    # 强制转为 string 类型，防止 int 和 str 混淆造成匹配错误
    train_df = pd.read_csv(TRAIN_PATH, usecols=use_cols, dtype=str)
    test_df = pd.read_csv(TEST_PATH, usecols=use_cols, dtype=str)
    
    print(f"训练集大小: {len(train_df)}")
    print(f"测试集大小: {len(test_df)}")

    # 分析 UserID
    analyze_overlap(train_df, test_df, "userid", "用户 (User)")
    
    # 分析 ItemID
    analyze_overlap(train_df, test_df, "itemid", "商户/商品 (Item)")

if __name__ == "__main__":
    main()