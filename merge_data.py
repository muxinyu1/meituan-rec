#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os

def main():
    # 设置文件路径
    data_dir = "data/recsys_task_data"
    
    # 读取各个CSV文件
    print("读取数据文件...")
    item_infos = pd.read_csv(os.path.join(data_dir, "item_infos-20221014.csv"))
    train_samples = pd.read_csv(os.path.join(data_dir, "train_samples-20221014.csv"))
    user_infos = pd.read_csv(os.path.join(data_dir, "user_infos-20221014.csv"))
    test_samples = pd.read_csv(os.path.join(data_dir, "test_samples-20221019.csv"))
    
    print(f"物品信息: {item_infos.shape}")
    print(f"训练样本: {train_samples.shape}")
    print(f"用户信息: {user_infos.shape}")
    print(f"测试样本: {test_samples.shape}")
    
    # 合并训练数据
    print("\n合并训练数据...")
    # 将物品信息合并到训练样本中
    train_merged = train_samples.merge(
        item_infos, 
        on='itemid', 
        how='left',
        suffixes=('', '_item')
    )
    
    # 将用户信息合并到训练样本中
    train_merged = train_merged.merge(
        user_infos, 
        on='userid', 
        how='left',
        suffixes=('', '_user')
    )
    
    print(f"合并后的训练数据: {train_merged.shape}")
    
    # 合并测试数据
    print("\n合并测试数据...")
    # 将物品信息合并到测试样本中
    test_merged = test_samples.merge(
        item_infos, 
        on='itemid', 
        how='left',
        suffixes=('', '_item')
    )
    
    # 将用户信息合并到测试样本中
    test_merged = test_merged.merge(
        user_infos, 
        on='userid', 
        how='left',
        suffixes=('', '_user')
    )
    
    print(f"合并后的测试数据: {test_merged.shape}")
    
    # 删除不需要的列，保留global_id作为唯一标识
    train_data = train_merged.drop(['global_id'], axis=1)
    test_data = test_merged.drop(['global_id'], axis=1)

    print(f"\n最终训练集大小: {train_data.shape}")
    print(f"测试集大小: {test_data.shape}")

    # 保存文件
    print("\n保存文件...")
    train_data.to_csv("train.csv", index=False)
    test_data.to_csv("test.csv", index=False)
    print("文件保存完成!")

if __name__ == "__main__":
    main()