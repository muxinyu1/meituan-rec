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
    
    # ==========================================
    # 0. 关键修正：按时间排序防止泄漏
    # ==========================================
    print("Sorting train data by timestamp to prevent leakage...")
    train = train.sort_values(by='timestamp').reset_index(drop=True)
    
    # 记录数据量，用于后续分离
    train_len = len(train)
    
    # 合并数据 (注意：Label列保留在df中以便计算，Test的label为NaN)
    df = pd.concat([train, test], axis=0, ignore_index=True)
    
    # ==========================================
    # 1. 数据预处理 (填充 NaN)
    # ==========================================
    print("Preprocessing: Handling Missing Values...")
    
    categorical_cols = [
        'userid', 'itemid', 'geohash', 'cityid', 'loc_cityid', 
        'weekday', 'hour', 'weather', 'dtype', 'cate_1', 'cate_2', 'cate_3', 
        'age', 'level', 'gender', 'married', 'job', 'has_car', 
        'work_geohash', 'mobile_type', 'mobile_os'
    ]
    
    numerical_cols = [
        'timestamp', 'distance', 'item_ave_price', 'price', 
        'user_home_dis', 'user_work_dis', 'temp', 'temp_low', 'temp_high', 
        'user_displayed_item_num', 'online_days'
    ]
    
    # 处理逻辑
    feature_candidates = [c for c in df.columns if c not in ['label', 'sample_index']]
    
    for col in feature_candidates:
        # 按照整个train.csv缺失率
        train_missing_rate = train[col].isnull().mean()
        
        # 缺失率高于10%添加一列
        if train_missing_rate > 0.1:
            df[f'{col}_is_missing'] = df[col].isnull().astype(int)
        
        # 填充值
        if col in categorical_cols:
            mode_val = train[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
        elif col in numerical_cols:
            median_val = train[col].median()
            df[col] = df[col].fillna(median_val)

    # ==========================================
    # 2. 特征工程：类别 -> 连续值
    # ==========================================
    print("Feature Engineering: Converting Categorical to Continuous...")
    
    # 辅助函数：Count 编码 (统计出现次数，通常不涉及严重泄漏，可用全量)
    def add_count(df, col):
        cnt = df[col].value_counts()
        df[f'{col}_count'] = df[col].map(cnt)
        return df

    # ----------------------------------------------------
    # 修正后的 CTR 计算函数：时间序列历史点击率
    # ----------------------------------------------------
    def add_temporal_ctr(df, col, train_len):
        # 分离出 Train 部分
        train_part = df.iloc[:train_len].copy()
        
        # --- 1. Train部分：Rolling/Expanding CTR ---
        # 已经在第0步按 timestamp 排序了
        # 逻辑：(当前累计点击数 - 当前点击) / (当前累计计数 - 1)
        # 这样确保了只使用了"当前样本之前"的信息
        
        # 计算该列每个类别的累计 Sum (点击次数) 和 Count (展现次数)
        # 注意：这里利用 groupby 的 cumsum 是向量化的，速度比 apply 快
        train_part['tmp_cumsum'] = train_part.groupby(col)['label'].cumsum()
        train_part['tmp_cumcount'] = train_part.groupby(col).cumcount() + 1
        
        # 历史 CTR = (Cumulative Sum - Current Label) / (Cumulative Count - 1)
        # 解释：减去 Current Label 是因为 cumsum 包含了当前行，我们需要的是"之前"的
        train_part[f'{col}_ctr'] = (train_part['tmp_cumsum'] - train_part['label']) / (train_part['tmp_cumcount'] - 1)
        
        # 处理第一次出现的样本 (Count-1=0的情况)，填充为全局平均 CTR 或 0
        global_mean = train_part['label'].mean()
        train_part[f'{col}_ctr'] = train_part[f'{col}_ctr'].fillna(global_mean)
        
        # --- 2. Test部分：使用 Train 的最终全局统计 ---
        # Test 无法进行 Expanding，因为它没有 Label。
        # Test 使用的是 Train 阶段该类别沉淀下来的最终 CTR。
        train_final_stats = train_part.groupby(col)['label'].agg(['sum', 'count'])
        train_final_stats['final_ctr'] = train_final_stats['sum'] / train_final_stats['count']
        mapper = train_final_stats['final_ctr'].to_dict()
        
        # 映射到全量 df (先覆盖 Train，再映射 Test)
        # 这里为了效率，我们直接把 Train 计算好的赋回去，Test 部分用 map 填充
        df.loc[:train_len-1, f'{col}_ctr'] = train_part[f'{col}_ctr']
        
        # Test 部分：如果 Test 中的类别在 Train 出现过，用 Train 的 Final CTR
        # 如果没出现过，用 Train 的全局均值
        test_vals = df.iloc[train_len:][col].map(mapper)
        df.loc[train_len:, f'{col}_ctr'] = test_vals.fillna(global_mean)
        
        return df

    # --- 执行映射 ---
    
    # 1. Count 编码
    count_cols = [
        'userid', 'itemid', 'cityid', 'weather', 'dtype', 
        'cate_1', 'cate_2', 'cate_3', 'age', 'level'
    ]
    for col in count_cols:
        df = add_count(df, col)
        
    # 2. Temporal CTR 编码 (修正点)
    ctr_cols = [
        'dtype', 'cate_1', 'cate_2', 'cate_3', 'age', 
        'level', 'gender', 'married', 'job'
    ]
    for col in ctr_cols:
        print(f"Processing Temporal CTR for: {col}")
        df = add_temporal_ctr(df, col, train_len)
        
    # 3. 周期性特征
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7.0)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7.0)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    
    # 4. 删除列表
    drop_cols = [
        'timestamp', 'geohash', 'loc_cityid', 'has_car', 
        'work_geohash', 'mobile_type', 'mobile_os',
        'userid', 'itemid', 'cityid', 'weekday', 'hour', 'weather',
        'dtype', 'cate_1', 'cate_2', 'cate_3', 'age', 'level',
        'gender', 'married', 'job'
    ]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    
    # 重新拆分
    train_processed = df.iloc[:train_len].copy()
    test_processed = df.iloc[train_len:].copy()
    
    if 'label' in test_processed.columns:
        test_processed.drop(columns=['label'], inplace=True)
        
    # 简单清理内存
    del df, train, test
    gc.collect()
    
    print(f"Final Feature Count: {train_processed.shape[1] - 1}")
    return train_processed, test_processed

def train_lgbm(train_df, test_df):
    print("\nStarting Training...")
    
    features = [c for c in train_df.columns if c != 'label']
    target = 'label'
    
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
    
    # 重新读取原始列用于分层 (因前面已删除或转换)
    # 注意：这里需要重新读取并按照 timestamp 排序，以匹配 train_df 的顺序
    raw_train = pd.read_csv('train.csv', usecols=['weekday', 'hour', 'timestamp'])
    raw_train = raw_train.sort_values(by='timestamp').reset_index(drop=True) # 必须同步排序！
    
    raw_train['weekday'] = raw_train['weekday'].fillna(raw_train['weekday'].mode()[0])
    raw_train['hour'] = raw_train['hour'].fillna(raw_train['hour'].mode()[0])
    
    stratify_col = raw_train['weekday'].astype(str) + "_" + raw_train['hour'].astype(str)
    
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(train_df.shape[0])
    test_preds = np.zeros(test_df.shape[0])
    fold_aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, stratify_col)):
        X_train, y_train = train_df.iloc[train_idx][features], train_df.iloc[train_idx][target]
        X_val, y_val = train_df.iloc[val_idx][features], train_df.iloc[val_idx][target]
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        clf = lgb.train(
            params,
            dtrain,
            num_boost_round=10000,
            valid_sets=[dtrain, dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0) # 减少日志输出
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
        
    mean_auc = np.mean(fold_aucs)
    weighted_auc = np.average(fold_aucs, weights=fold_aucs)
    print(f"\nMean AUC: {mean_auc:.5f}")
    print(f"Weighted AUC: {weighted_auc:.5f}")
    
    submission = pd.DataFrame({
        'sample_index': test_df['sample_index'].astype(int),
        'label': test_preds
    })
    submission.to_csv('submission_lgb.csv', index=False)
    print("submission_lgb.csv saved.")

if __name__ == "__main__":
    train_df, test_df = preprocess_and_feature_engineering()
    train_lgbm(train_df, test_df)