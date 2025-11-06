import pandas as pd
import argparse
import sys

def check_missing_values(filepath):
    """
    加载一个CSV文件并检查其中是否包含缺失值 (NaN)。
    """
    print(f"🔍 [Check] Loading data from: {filepath}")
    try:
        # 使用 low_memory=False 加速大文件读取
        df = pd.read_csv(filepath, low_memory=False)
    except FileNotFoundError:
        print(f"❌ [Error] File not found at '{filepath}'. Please check the path.")
        sys.exit(1) # 退出脚本，返回错误码
    except Exception as e:
        print(f"❌ [Error] Could not read file: {e}")
        sys.exit(1)

    print("🕵️  [Check] Scanning for missing values (NaNs)...")
    
    # 1. 获取每列的NaN总数
    missing_counts = df.isnull().sum()
    
    # 2. 获取整个DataFrame的NaN总数
    total_missing = missing_counts.sum()

    print("---" * 15)
    if total_missing == 0:
        print("🎉  Success!  🎉")
        print(f"   No missing values (NaNs) found in the file.")
        print(f"   Shape of the data: {df.shape}")
    else:
        print("🔥  Warning!  🔥")
        print(f"   Found a total of {total_missing} missing values in the file.")
        print("\n   --- Columns with missing values ---")
        
        # 3. 筛选出那些真正有NaN的列并打印
        cols_with_missing = missing_counts[missing_counts > 0]
        print(cols_with_missing)
        print("\n   Please re-check the preprocessing script (process_data.py).")
    
    print("---" * 15)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A small script to check a CSV file for missing values (NaNs).",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--csv_path',
        type=str,
        default='merged_features_final.csv', # 默认指向我们刚生成的文件
        help='Path to the CSV file you want to check.'
    )
    
    args = parser.parse_args()
    
    check_missing_values(args.csv_path)