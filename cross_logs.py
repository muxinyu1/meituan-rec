import pandas as pd
import numpy as np
import time

def process_data_v3(input_csv_path: str, output_csv_path: str):
    """
    根据高级洞察对用户点击商品数据集进行特征工程处理 (版本 3)。
    此版本重点增强了时间特征工程。

    Args:
        input_csv_path (str): 输入的原始 CSV 文件路径。
        output_csv_path (str): 处理后要保存的 CSV 文件路径。
    """
    print(f"--- 开始处理文件 (V3): {input_csv_path} ---")

    # 1. 加载数据
    start_time = time.time()
    try:
        # 在加载时直接指定部分列的类型，可以节省内存
        # 这里为了通用性先不加，但对于超大数据集是好习惯
        df = pd.read_csv(input_csv_path)
        print(f"数据加载成功. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"错误: 文件 '{input_csv_path}' 未找到。请检查文件路径。")
        return

    # --- 预处理和特征工程 ---

    # 2. 增强的时间特征工程
    print("\n步骤 2: 增强的时间特征工程...")
    # 2.1 转换时间戳
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

    # 2.2 提取基础时间特征
    df['hour'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.weekday  # 星期一=0, 星期日=6
    
    # 2.3 创建派生布尔特征
    df['is_weekend'] = (df['weekday'] >= 5).astype('int8')  # 周六或周日
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype('int8')  # 22:00 - 05:59

    # 2.4 创建周期性编码特征
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24).astype('float32')
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24).astype('float32')
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7).astype('float32')
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7).astype('float32')
    
    # 2.5 创建时段分箱特征
    # bins 定义了区间的边缘，labels 定义了每个区间的名称
    df['time_slot'] = pd.cut(
        df['hour'],
        bins=[-1, 5, 11, 14, 18, 23],  # (夜:0-5, 早:6-11, 午:12-14, 下午:15-18, 晚:19-23)
        labels=[0, 1, 2, 3, 4],
        ordered=False # 作为一个无序类别特征
    ).astype('category')
    
    # 原始的`weekday`和`hour`列（来自原始数据而非timestamp的）可能会被覆盖，这里确保删除无用的中间列
    # 并强制覆盖原始数据可能带有的 `weekday` 列，以 `timestamp` 提取的为准
    df.drop(columns=['timestamp', 'datetime'], inplace=True)
    if 'weekday' in pd.read_csv(input_csv_path, nrows=0).columns:
        print("已使用基于 `timestamp` 的 `weekday` 覆盖原始列。")
        
    print("时间特征创建完成，新增/更新了 'hour', 'weekday', 'is_weekend', 'is_night', \
'hour_sin/cos', 'weekday_sin/cos', 'time_slot'.")


    # 3. 缺失值处理 & 衍生特征 (与V2保持一致)
    print("\n步骤 3: 缺失值处理与衍生特征...")
    # (这部分代码与 V2 版本相同)
    high_missing_cols = ['price', 'online_days', 'user_displayed_item_num']
    for col in high_missing_cols:
        indicator_col_name = f'{col}_is_missing'
        df[indicator_col_name] = df[col].isnull().astype(int)
        df[col].fillna(-1, inplace=True)
    print(f"高缺失率列 {high_missing_cols} 已创建指示器并用 -1 填充。")
    
    df['cityid'].fillna(-1, inplace=True)
    df['loc_cityid'].fillna(-1, inplace=True)
    df['is_same_city'] = (df['cityid'] == df['loc_cityid']).astype(int)
    print("地理特征创建完成: 'is_same_city'.")
    
    city_median_distance = df.groupby('cityid')['distance'].transform('median')
    df['distance'].fillna(city_median_distance, inplace=True)
    df['distance'].fillna(df['distance'].median(), inplace=True)
    print("列 'distance' 已通过城市分组中位数填充。")
    
    for col in ['item_ave_price', 'temp_high', 'temp', 'temp_low']:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    print("中低缺失率数值列已用中位数填充。")
    # 假设 df 是你加载了数据的 DataFrame


    df['temp_range'] = (df['temp_high'] - df['temp_low']).astype('float32')

    df['temp_comfort'] = ((df['temp'] >= 16) & (df['temp'] <= 26)).astype('int8')

    print("Binning high-cardinality features...")
    # 保留Top 100城市（覆盖95%+数据）
    top_cities = df['cityid'].value_counts().index[:100].tolist()
    df['cityid_bucket'] = df['cityid'].apply(
        lambda x: x if x in top_cities else 999.0  # 999表示"其他城市"
    ).astype('float32')

    # 4. 高级特征交叉与变换 (与V2保持一致)
    # print("\n步骤 4: 高级特征交叉与变换...")
    # # (这部分代码与 V2 版本相同)
    # df['price'] = df['price'].apply(lambda x: np.log1p(x) if x >= 0 else x)
    # df['item_ave_price'] = df['item_ave_price'].apply(lambda x: np.log1p(x) if x >= 0 else x)
    # print("对 'price' 和 'item_ave_price' 进行了 log1p 变换。")
    
    # epsilon = 1e-6
    # df['price_to_ave_ratio'] = df['price'] / (df['item_ave_price'] + epsilon)
    # print("创建了特征 'price_to_ave_ratio'.")
    
    # df['city_hour_interaction'] = df['cityid'].astype(int).astype(str) + '_' + df['hour'].astype(str)
    # print("创建了特征 'city_hour_interaction'.")
    df.drop(columns=['geohash', 'global_id'], inplace=True)

    # 5. 保存结果
    print(f"\n步骤 5: 保存处理后的数据到 {output_csv_path}...")
    try:
        df.to_csv(output_csv_path, index=False)
        print("文件保存成功。")
    except Exception as e:
        print(f"文件保存失败: {e}")

    end_time = time.time()
    print(f"\n处理完成. 总耗时: {end_time - start_time:.2f} 秒。")
    print("\n最终数据预览 (部分列):")
    preview_cols = ['userid', 'itemid', 'label', 'hour', 'is_weekend', 'is_night', 'time_slot', 'hour_sin', 'weekday_cos', 'price', 'price_is_missing', 'distance', 'is_same_city']
    print(df[preview_cols].head())
    print(f"\n最终数据 Shape: {df.shape}")


if __name__ == '__main__':
    # --- 配置区 ---
    # 运行新的处理函数
    process_data_v3(input_csv_path='data/recsys_task_data/train_samples-20221014.csv', 
                    output_csv_path='data/recsys_task_data/train_samples-20221014-crossed.csv')