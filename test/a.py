import pandas as pd

def main():
    # ========== 配置 ==========
    train_path = 'data/recsys_task_data/train_samples-20221014.csv'      # 训练集路径
    valid_path = 'data/recsys_task_data/test_samples-20221019.csv'      # 验证集路径
    output_report_path = 'oov_report.txt'  # 报告输出路径（可选）
    
    # 要检查的列
    check_columns = [ 'geohash', 'cityid', 'loc_cityid']

    # ========== 读取数据 ==========
    print("正在读取训练集和验证集...")
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)

    # ========== 检查每列的 OOV（Out-of-Vocabulary）情况 ==========
    report_lines = []
    oov_info = {}

    for col in check_columns:
        if col not in train_df.columns or col not in valid_df.columns:
            print(f"警告：列 '{col}' 在训练集或验证集中不存在，跳过。")
            continue

        # 获取训练集中该列的唯一值（转为 set 加速查询）
        train_vals = set(train_df[col].dropna().unique())
        
        # 验证集中不在训练集中的值
        valid_not_in_train = valid_df[~valid_df[col].isin(train_vals) | valid_df[col].isna()]
        oov_count = len(valid_not_in_train)
        total_valid = len(valid_df)

        # 记录信息
        oov_info[col] = {
            'oov_count': oov_count,
            'total': total_valid,
            'oov_ratio': oov_count / total_valid if total_valid > 0 else 0,
            'examples': valid_not_in_train[col].dropna().head(5).tolist()  # 前5个示例
        }

        # 生成报告行
        line = (
            f"列 '{col}':\n"
            f"  - 验证集中总样本数: {total_valid}\n"
            f"  - 未在训练集中出现的样本数: {oov_count} ({oov_info[col]['oov_ratio']:.2%})\n"
            f"  - 示例（最多5个）: {oov_info[col]['examples']}\n"
        )
        report_lines.append(line)
        print(line)

    # ========== 保存报告 ==========
    with open(output_report_path, 'w', encoding='utf-8') as f:
        f.write("验证集 OOV（训练集未见值）分析报告\n")
        f.write("=" * 50 + "\n\n")
        f.write("\n".join(report_lines))
    print(f"报告已保存至: {output_report_path}")

    # ========== 可选：保存带标记的验证集（每列新增 _in_train 列）==========
    valid_with_flag = valid_df.copy()
    for col in check_columns:
        if col in valid_df.columns:
            train_set = set(train_df[col].dropna().unique())
            valid_with_flag[f"{col}_in_train"] = valid_df[col].isin(train_set) & valid_df[col].notna()
    
    valid_with_flag.to_csv('valid_with_oov_flags.csv', index=False)
    print("已保存带 OOV 标记的验证集: valid_with_oov_flags.csv")

if __name__ == '__main__':
    main()