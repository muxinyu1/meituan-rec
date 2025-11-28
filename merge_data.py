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
    
    # 从训练数据中划分验证集，确保各个weekday的数据都包含
    print("\n划分训练集和验证集...")
    
    # 按weekday分组，然后从每个组中抽取一定比例的数据作为验证集
    val_ratio = 0.2  # 验证集比例
    val_data = pd.DataFrame()
    train_data_final = pd.DataFrame()
    
    # 获取所有weekday值
    weekdays = train_data['weekday'].unique()
    print(f"训练数据中的weekday值: {sorted(weekdays)}")
    
    # 对每个weekday进行分层抽样
    for weekday in sorted(weekdays):
        weekday_data = train_data[train_data['weekday'] == weekday]
        # 随机打乱数据
        weekday_data = weekday_data.sample(frac=1, random_state=42)
        # 计算验证集大小
        val_size = int(len(weekday_data) * val_ratio)
        # 划分训练集和验证集
        weekday_val = weekday_data.iloc[:val_size]
        weekday_train = weekday_data.iloc[val_size:]
        
        val_data = pd.concat([val_data, weekday_val])
        train_data_final = pd.concat([train_data_final, weekday_train])
        
        print(f"Weekday {weekday}: 训练样本 {len(weekday_train)}, 验证样本 {len(weekday_val)}")
    
    print(f"\n最终训练集大小: {train_data_final.shape}")
    print(f"验证集大小: {val_data.shape}")
    print(f"测试集大小: {test_data.shape}")
    
    # 保存文件
    print("\n保存文件...")
    train_data_final.to_csv("train.csv", index=False)
    val_data.to_csv("val.csv", index=False)
    test_data.to_csv("test.csv", index=False)
    
    print("文件保存完成!")
    
    # 打印各数据集的标签分布
    print("\n训练集标签分布:")
    print(train_data_final['label'].value_counts(normalize=True))
    
    print("\n验证集标签分布:")
    print(val_data['label'].value_counts(normalize=True))
    
    print("\n验证集中各weekday的样本数量:")
    print(val_data['weekday'].value_counts().sort_index())

if __name__ == "__main__":
    main()