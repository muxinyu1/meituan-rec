import pandas as pd
import numpy as np

def blend_models():
    # 1. 读取文件
    print("正在读取预测文件...")
    # 你的 CatBoost 结果 (主模型，分数较高)
    file_cat = 'submission.csv' 
    # 你的 LightGBM 结果 (辅助模型，分数较低但差异性好)
    file_lgb = 'submission_nn.csv' 
    
    df_cat = pd.read_csv(file_cat)
    df_lgb = pd.read_csv(file_lgb)
    
    # 2. 检查对齐
    if len(df_cat) != len(df_lgb):
        raise ValueError("错误：两个文件的行数不一致，无法融合！")
        
    # 3. 设置权重
    # 策略：CatBoost 单模强很多 (0.6685 vs 0.6628)，所以给高权重
    # 0.75 / 0.25 是一个非常稳健的起点
    w_cat = 0.5
    w_lgb = 0.5
    
    print(f"融合权重设置 -> CatBoost: {w_cat}, LightGBM: {w_lgb}")
    
    # 4. 执行线性加权融合 (Linear Blending)
    # 这种方式保留了概率的分布信息
    df_blend = df_cat.copy()
    df_blend['label'] = (df_cat['label'] * w_cat) + (df_lgb['label'] * w_lgb)
    
    # --- 备选方案：Rank Averaging (排名融合) ---
    # 如果上面的线性融合效果不好，可以尝试解开下面这段代码
    # 当两个模型预测值的分布差异极大（比如一个在0-0.1，另一个在0-1）时，Rank融合更稳
    """
    print("使用 Rank Averaging 进行融合...")
    df_blend['label'] = (df_cat['label'].rank(pct=True) * w_cat) + \
                        (df_lgb['label'].rank(pct=True) * w_lgb)
    """
    
    # 5. 保存结果
    output_file = 'submission_ensemble.csv'
    df_blend.to_csv(output_file, index=False)
    
    print(f"\n融合完成！结果已保存至: {output_file}")
    print("建议：提交该文件，预期 AUC 应该能突破 0.6695")

    # 简单打印前几行看看
    print("\n前5行预览:")
    print(df_blend.head())

if __name__ == "__main__":
    blend_models()