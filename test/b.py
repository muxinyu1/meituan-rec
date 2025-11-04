import pandas as pd
import json

def analyze_csv_to_json(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path, low_memory=False)
    
    result = {}
    
    for col in df.columns:
        col_info = {}
        
        # 尝试将列转换为数值类型
        numeric_series = pd.to_numeric(df[col], errors='coerce')
        
        # 判断是否全部无法转换为数字
        if numeric_series.isna().all():
            col_info['data_type'] = 'string'
        else:
            # 检查非NaN值是否都是整数
            non_na_values = numeric_series.dropna()
            if not non_na_values.empty and (non_na_values % 1 == 0).all():
                col_info['data_type'] = 'int'
                # 计算有效整数值的唯一值数量
                int_values = non_na_values.astype(int)
                col_info['unique_count'] = int(int_values.nunique())
            else:
                col_info['data_type'] = 'float'
        
        result[col] = col_info
    
    return result

def main():
    file_path = input("请输入CSV文件路径: ").strip()
    try:
        json_result = analyze_csv_to_json(file_path)
        # 格式化输出JSON
        print(json.dumps(json_result, indent=2, ensure_ascii=False))
    except Exception as e:
        error_result = {"error": str(e)}
        print(json.dumps(error_result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()