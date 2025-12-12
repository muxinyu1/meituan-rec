import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import gc
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

def preprocess_and_feature_engineering():
    print("Loading data...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    train_len = len(train)
    
    # 合并数据以便统一进行LabelEncoding和频次统计
    df = pd.concat([train, test], axis=0, ignore_index=True)
    
    # ==========================================
    # 1. 缺失值处理
    # ==========================================
    print("Preprocessing: Handling Missing Values...")
    
    # 定义原始列的类型（用于判断填充策略）
    # 注意：此时还未进行删除操作
    categorical_cols_raw = [
        'userid', 'itemid', 'geohash', 'cityid', 'loc_cityid', 
        'weekday', 'hour', 'weather', 'dtype', 'cate_1', 'cate_2', 'cate_3', 
        'age', 'level', 'gender', 'married', 'job', 'has_car', 
        'work_geohash', 'mobile_type', 'mobile_os'
    ]
    
    numerical_cols_raw = [
        'timestamp', 'distance', 'item_ave_price', 'price', 
        'user_home_dis', 'user_work_dis', 'temp', 'temp_low', 'temp_high', 
        'user_displayed_item_num', 'online_days'
    ]
    
    # 遍历所有列进行缺失值填充
    all_cols = [c for c in df.columns if c not in ['label', 'sample_index']]
    
    for col in all_cols:
        # 计算train部分的缺失率
        train_missing_rate = train[col].isnull().mean()
        
        # 缺失率高于10%添加标记列
        if train_missing_rate > 0.1:
            df[f'{col}_is_missing'] = df[col].isnull().astype('category')
        
        # 填充值
        if col in categorical_cols_raw:
            mode_val = train[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
        elif col in numerical_cols_raw:
            median_val = train[col].median()
            df[col] = df[col].fillna(median_val)

    # ==========================================
    # 2. 特征工程：构建类别特征
    # ==========================================
    print("Feature Engineering: Processing Categorical Logic...")

    # --- 2.1 特殊逻辑处理 ---
    
    # A. is_same_city (cityid == loc_cityid)
    # 需在删除 loc_cityid 之前进行
    # 转换为 int 或 category 都可以，LGBM对二分类处理很好
    df['is_same_city'] = (df['cityid'] == df['loc_cityid']).astype(int).astype('category')
    
    # B. itemid: 出现次数小于10 -> "UNK"
    # 使用 transform 计算全局频次
    item_counts = df.groupby('itemid')['itemid'].transform('count')
    # 为了保持一致性，先转为字符串再赋值，最后统一转category
    df['itemid'] = df['itemid'].astype(str)
    df.loc[item_counts < 10, 'itemid'] = "UNK"
    
    # C. cityid: 出现次数小于200 -> "UNK"
    city_counts = df.groupby('cityid')['cityid'].transform('count')
    df['cityid'] = df['cityid'].astype(str)
    df.loc[city_counts < 200, 'cityid'] = "UNK"
    
    # D. temp: 17~26 -> 1 (Comfortable), else 0
    # 这是一个将数值特征离散化的过程
    def process_temp(t):
        if 17 <= t <= 26:
            return 1 # 舒适
        else:
            return 0 # 其他
    df['temp_cat'] = df['temp'].apply(process_temp).astype('category')
    
    # --- 2.2 删除指定列 ---
    cols_to_drop = [
        'userid', 
        'timestamp', 
        'geohash', 
        'loc_cityid', # 已用于生成 is_same_city
        'distance', 
        'item_ave_price', 
        'price', 
        'user_home_dis', 
        'user_work_dis', 
        'temp',      # 已转换为 temp_cat
        'temp_low', 
        'temp_high', 
        'user_displayed_item_num', 
        'online_days'
    ]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    # --- 2.3 最终类型转换 ---
    # 确保所有特征列（除了label和index）都是 'category' 类型
    # LightGBM 原生支持 category 类型，会自动进行 Label Encoding 且处理 split
    feature_cols = [c for c in df.columns if c not in ['label', 'sample_index']]
    
    for col in feature_cols:
        df[col] = df[col].astype('category')
        
    print(f"Remaining Features: {feature_cols}")
    
    # 重新拆分
    train_processed = df.iloc[:train_len].copy()
    test_processed = df.iloc[train_len:].copy()
    
    # test不需要label
    if 'label' in test_processed.columns:
        test_processed.drop(columns=['label'], inplace=True)
        
    # 清理内存
    del df, train, test, item_counts, city_counts
    gc.collect()
    
    return train_processed, test_processed

def train_lgbm(train_df, test_df):
    print("\nStarting Training...")
    
    features = [c for c in train_df.columns if c != 'label']
    target = 'label'
    
    # 找出所有的 category 列名，显式传递给 LGBM (虽然 astype('category') 通常足够，但显式更安全)
    cat_feats = [c for c in features if train_df[c].dtype.name == 'category']
    print(f"Categorical features count: {len(cat_feats)}")

    # 参数配置
    params = {
        "bagging_fraction": 0.9879639408647978,
        "bagging_freq": 6,
        "feature_fraction": 0.6003115063364057,
        "lambda_l1": 1.984423118582435,
        "lambda_l2": 1.234963019255433,
        "learning_rate": 0.06504878444394528,
        "max_bin": 380,
        "max_depth": 7,
        "min_child_samples": 128,
        "min_child_weight": 0.030122914019804194,
        "min_gain_to_split": 0.030592644736118974,
        "num_leaves": 61,
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "n_jobs": -1,
        "seed": 2023
    }

    K = 8
    
    # 构造分层 Stratification 列
    # 这里的 weekday 和 hour 已经被转为了 category 类型保留在特征中，可以直接使用
    # 注意：需要转回 str 拼接
    stratify_col = train_df['weekday'].astype(str) + "_" + train_df['hour'].astype(str)
    
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(train_df.shape[0])
    test_preds = np.zeros(test_df.shape[0])
    fold_aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, stratify_col)):
        X_train, y_train = train_df.iloc[train_idx][features], train_df.iloc[train_idx][target]
        X_val, y_val = train_df.iloc[val_idx][features], train_df.iloc[val_idx][target]
        
        # 使用 categorical_feature 参数
        dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_feats)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, categorical_feature=cat_feats)
        
        clf = lgb.train(
            params,
            dtrain,
            num_boost_round=10000,
            valid_sets=[dtrain, dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0) # 简洁输出
            ]
        )
        
        val_pred = clf.predict(X_val, num_iteration=clf.best_iteration)
        oof_preds[val_idx] = val_pred
        
        score = roc_auc_score(y_val, val_pred)
        fold_aucs.append(score)
        print(f"Fold {fold + 1} AUC: {score:.5f}")
        
        test_preds += clf.predict(test_df[features], num_iteration=clf.best_iteration) / K
        
        del X_train, y_train, X_val, y_val, dtrain, dval, clf
        gc.collect()
        
    # 计算加权AUC
    weighted_auc = np.average(fold_aucs, weights=fold_aucs)
    mean_auc = np.mean(fold_aucs)
    
    print(f"\nTraining Finished.")
    print(f"Mean AUC: {mean_auc:.5f}")
    print(f"Weighted Average AUC: {weighted_auc:.5f}")
    
    # 生成提交文件
    submission = pd.DataFrame({
        'sample_index': test_df['sample_index'],
        'label': test_preds
    })
    
    submission.to_csv('submission_lgb.csv', index=False)
    print("submission_lgb.csv saved successfully.")

if __name__ == "__main__":
    train_df, test_df = preprocess_and_feature_engineering()
    train_lgbm(train_df, test_df)