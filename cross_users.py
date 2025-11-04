import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# --- 配置 ---

# 原始CSV文件路径
INPUT_FILE = 'data/recsys_task_data/user_infos-20221014.csv'

# 处理后要保存的新CSV文件路径
OUTPUT_FILE = 'data/recsys_task_data/user_infos_crossed_features.csv'

# 定义要执行的特征交叉列表
# 每一项都是一个包含要交叉的列名的列表
CROSS_FEATURE_LIST = [
    # 1. 基础人口属性交叉
    ['age', 'gender'],
    ['age', 'married'],
    ['gender', 'married'],
    ['age', 'gender', 'married'],
    
    # 2. 社会经济状态交叉
    ['married', 'has_car'],
    
    # 3. 用户行为与属性交叉
    ['level', 'age'],
    ['level', 'gender'],
    ['level', 'job'],
    
    # 4. 设备与上下文交叉
    ['mobile_os', 'age']
]

# 在交叉前，用于填充NaN值的特殊字符串
# 这样 "level=NaN & age=1" 会变成一个唯一的类别 "NA_1"
FILL_NA_VALUE = 'NA'

# --- 脚本开始 ---

def main():
    print(f"--- 开始处理特征交叉脚本 ---")
    
    # 1. 加载数据
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"成功加载数据: {INPUT_FILE}，共 {len(df)} 行。")
    except FileNotFoundError:
        print(f"错误: 文件未找到 {INPUT_FILE}")
        print("请检查文件路径是否正确。")
        return
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return

    # 用于存储新创建的列名
    new_encoded_columns = []
    
    print("\n--- 正在创建、报告和编码交叉特征 ---")
    print("=" * 40)

    for col_list in CROSS_FEATURE_LIST:
        # 1. 创建新特征的名称
        # 例如: ['age', 'gender'] -> 'cross_age_gender'
        new_col_name = 'cross_' + '_'.join(col_list)
        new_col_name_encoded = new_col_name + '_encoded'
        new_encoded_columns.append(new_col_name_encoded)

        print(f"处理中: {new_col_name}")

        # 2. 创建字符串交叉特征
        # 我们使用 .fillna(FILL_NA_VALUE).astype(str) 来稳健地处理所有类型和缺失值
        
        # 获取要交叉的系列（Series）列表
        series_to_cross = [df[col].fillna(FILL_NA_VALUE).astype(str) for col in col_list]
        
        # 将它们用 '_' 连接起来
        # 例如: '1' + '_' + '1' -> '1_1'
        #       'NA' + '_' + '2' -> 'NA_2'
        df[new_col_name] = series_to_cross[0]
        for i in range(1, len(series_to_cross)):
            df[new_col_name] = df[new_col_name] + '_' + series_to_cross[i]
            
        # 3. 报告基数（唯一值的数量）
        # 这是您要求观察的关键信息
        cardinality = df[new_col_name].nunique()
        print(f"  > [报告] 特征 '{new_col_name}' 的唯一组合数量 (基数): {cardinality}")

        # 4. 对新特征进行Label Encoding
        encoder = LabelEncoder()
        df[new_col_name_encoded] = encoder.fit_transform(df[new_col_name])
        
        # （可选）删除临时的字符串列，只保留编码后的列
        df = df.drop(columns=[new_col_name])

        print(f"  > [完成] 已创建并编码为: {new_col_name_encoded}")
        print("-" * 20)

    print("=" * 40)
    
    # 5. 保存新的CSV文件
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(OUTPUT_FILE)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n--- 脚本执行完毕！ ---")
        print(f"已将包含原始特征和 {len(new_encoded_columns)} 个新交叉特征的CSV保存至:")
        print(f"{os.path.abspath(OUTPUT_FILE)}")
        
        print("\n新增加的编码列为:")
        for col in new_encoded_columns:
            print(f"  - {col}")
            
    except Exception as e:
        print(f"\n--- 错误 ---")
        print(f"保存文件 {OUTPUT_FILE} 时出错: {e}")

if __name__ == "__main__":
    main()