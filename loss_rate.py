import pandas as pd

# 读取 CSV 文件（请将 'your_file.csv' 替换为你的实际文件路径）
df = pd.read_csv('data/recsys_task_data/train_samples-20221014.csv')

# 获取每列的缺失值数量
missing_count = df.isnull().sum()

# 获取总行数
total_rows = len(df)

# 计算每列的缺失率（百分比）
missing_rate = (missing_count / total_rows) * 100

# 创建一个 DataFrame 显示结果
missing_report = pd.DataFrame({
    'Column': missing_rate.index,
    'Missing Rate (%)': missing_rate.values
})

# 按缺失率降序排序（可选）
missing_report = missing_report.sort_values(by='Missing Rate (%)', ascending=False)

# 打印结果
print(missing_report.to_string(index=False))