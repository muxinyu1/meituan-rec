import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class CTRPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        
    def load_data(self, train_path, user_path, item_path):
        """
        加载训练数据和辅助数据
        """
        print("Loading data...")
        # 加载训练样本
        train_data = pd.read_csv(train_path)
        
        # 加载用户信息
        user_data = pd.read_csv(user_path)
        
        # 加载物品信息
        item_data = pd.read_csv(item_path)
        
        # 合并数据
        train_data = train_data.merge(user_data, on='userid', how='left')
        train_data = train_data.merge(item_data, on='itemid', how='left')
        
        print(f"Data loaded. Shape: {train_data.shape}")
        return train_data
    
    def preprocess_data(self, df):
        """
        数据预处理和特征工程
        """
        print("Preprocessing data...")
        # 处理缺失值
        df = df.fillna(-1)
        
        # 选择特征列（排除id类和标签列）
        exclude_columns = ['global_id', 'userid', 'itemid', 'label']
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        self.feature_columns = feature_columns
        
        # 分离特征和标签
        X = df[feature_columns]
        y = df['label']
        
        print(f"Features: {len(feature_columns)}")
        return X, y, feature_columns
    
    def train_model(self, X_train, y_train, X_val, y_val, num_boost_round=100, early_stopping_rounds=10):
        """
        训练LightGBM模型
        :param X_train: 训练特征
        :param y_train: 训练标签
        :param X_val: 验证特征
        :param y_val: 验证标签
        :param num_boost_round: 提升轮数，默认100
        :param early_stopping_rounds: 早停轮数，默认10
        """
        print(f"Training LightGBM model with {num_boost_round} boosting rounds...")
        # 创建数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # 设置参数
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'num_threads': 32
        }
        
        # 训练模型
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            valid_names=['eval'],
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            callbacks=[lgb.log_evaluation(10)]
        )
        
        print("Model training completed.")
        
    def evaluate_model(self, X_test, y_test):
        """
        评估模型性能
        """
        print("Evaluating model...")
        # 预测
        y_pred = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        
        # 计算评估指标
        # 检查测试集中是否包含两种标签
        unique_labels = np.unique(y_test)
        if len(unique_labels) < 2:
            print(f"Warning: Test set contains only one label ({unique_labels[0]}). AUC cannot be calculated.")
            auc = 0.5  # 当只有一种标签时，AUC无意义，设为0.5
        else:
            auc = roc_auc_score(y_test, y_pred)
        
        # 对于logloss，我们需要确保预测值在(0,1)范围内
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        logloss = log_loss(y_test, y_pred, labels=[0, 1])
        
        print(f"AUC: {auc:.4f}")
        print(f"LogLoss: {logloss:.4f}")
        
        return auc, logloss, y_pred
    
    def save_model(self, model_path):
        """
        保存模型
        """
        self.model.save_model(model_path)
        print(f"Model saved to {model_path}")

def main():
    # 初始化预测器
    predictor = CTRPredictor()
    
    # 数据路径
    train_path = './data/recsys_task_data/train_samples-20221014.csv'
    user_path = './data/recsys_task_data/user_infos-20221014.csv'
    item_path = './data/recsys_task_data/item_infos-20221014.csv'
    
    # 加载数据
    data = predictor.load_data(train_path, user_path, item_path)
    
    # 数据预处理
    X, y, feature_columns = predictor.preprocess_data(data)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    
    print(f"Train set size: {X_train.shape}")
    print(f"Validation set size: {X_val.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # 训练模型，设置训练轮数和早停
    num_boost_round = 500  # 可以根据需要调整
    early_stopping_rounds = 10  # 早停轮数
    predictor.train_model(X_train, y_train, X_val, y_val, 
                         num_boost_round=num_boost_round, 
                         early_stopping_rounds=early_stopping_rounds)
    
    # 评估模型
    auc, logloss, y_pred = predictor.evaluate_model(X_test, y_test)
    
    # 保存模型
    predictor.save_model('ctr_model.txt')
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()