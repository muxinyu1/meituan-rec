import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import gc
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# 配置
CONFIG = {
    'seed': 2024,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'n_splits': 10,
    'batch_size': 1024,
    'epochs': 100,  # 增大 epoch 数，防止欠拟合，确保为 int 类型
    'lr': 0.001,
    'embed_dim': 8,       # Embedding维度
    'hidden_units': [256, 128, 64], # Deep部分隐层
    'dropout': 0.2
}

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

seed_everything(CONFIG['seed'])

# ==========================================
# 1. 数据预处理
# ==========================================

def preprocess_data(train_path, test_path):
    print("Loading data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # 合并处理以保证编码一致
    data = pd.concat([train, test], axis=0, ignore_index=True)
    
    # --------------------------------------
    # 1.1 填充缺失值
    # --------------------------------------
    print("Handling missing values...")
    missing_cols = []
    
    # 计算整个train的缺失率
    train_len = len(train)
    for col in data.columns:
        if col == 'label' or col == 'sample_index': continue
        
        # 只根据train集计算缺失率
        missing_rate = train[col].isnull().mean()
        
        # 缺失率 > 10% 添加指示列
        if missing_rate > 0.10:
            data[f'{col}_is_missing'] = data[col].isnull().astype(int)
            missing_cols.append(f'{col}_is_missing')
        
        # 填充
        if data[col].dtype == 'object' or col in ['userid', 'itemid', 'geohash', 'cityid', 'loc_cityid']:
            # 类别列：用众数
            mode_val = train[col].mode()[0]
            data[col] = data[col].fillna(mode_val)
        else:
            # 数值列：用中位数
            median_val = train[col].median()
            data[col] = data[col].fillna(median_val)

    # --------------------------------------
    # 1.2 特征工程 (变换与生成)
    # --------------------------------------
    print("Feature Engineering...")
    
    # is_same_city
    data['is_same_city'] = (data['cityid'] == data['loc_cityid']).astype(int)
    
    # is_weekend
    data['is_weekend'] = data['weekday'].isin([6, 7]).astype(int)
    
    # timeslot (对hour划分)
    # 假设划分: 0-6(凌晨), 7-12(上午), 13-18(下午), 19-23(晚上)
    def get_timeslot(h):
        if 0 <= h <= 6: return 0
        elif 7 <= h <= 12: return 1
        elif 13 <= h <= 18: return 2
        else: return 3
    data['timeslot'] = data['hour'].apply(get_timeslot)
    
    # temp comfort (17~26)
    data['temp_comfort'] = ((data['temp'] >= 17) & (data['temp'] <= 26)).astype(int)
    
    # --------------------------------------
    # 1.3 高基数特征处理 (UNK划分)
    # --------------------------------------
    print("Handling high cardinality features...")
    
    # itemid: 出现次数 < 10 -> UNK
    item_counts = train['itemid'].value_counts()
    valid_items = set(item_counts[item_counts >= 10].index)
    data.loc[~data['itemid'].isin(valid_items), 'itemid'] = 'UNK'
    
    # cityid: 出现次数 < 200 -> UNK
    city_counts = train['cityid'].value_counts()
    valid_cities = set(city_counts[city_counts >= 200].index)
    data.loc[~data['cityid'].isin(valid_cities), 'cityid'] = 'UNK'

    # --------------------------------------
    # 1.4 删除不需要的列
    # --------------------------------------
    drop_cols = [
        'userid', 'timestamp', 'geohash', 'loc_cityid', 'distance',
        'item_ave_price', 'price', 'user_home_dis', 'user_work_dis',
        'temp_low', 'temp_high', 'user_displayed_item_num', 'online_days',
        'temp' # 已转换为temp_comfort
    ]
    # 注意：不要删除用于StratifiedKFold的weekday和hour，稍后在Dataset中处理
    
    data = data.drop(columns=[c for c in drop_cols if c in data.columns])

    # --------------------------------------
    # 1.5 Label Encoding
    # --------------------------------------
    print("Label Encoding...")
    
    # 确定最终用于训练的类别特征列表
    # 排除 label, sample_index, 以及可能存在的数值列(虽然上面大多删了)
    exclude_cols = ['label', 'sample_index']
    
    # 识别所有剩下的特征作为类别特征 (根据题目要求，只使用类别特征训练NN)
    feature_cols = [c for c in data.columns if c not in exclude_cols]
    
    cat_dims = []
    
    for col in feature_cols:
        le = LabelEncoder()
        # 将所有列转为字符串处理，防止数字/字符串混用报错
        data[col] = le.fit_transform(data[col].astype(str))
        cat_dims.append(data[col].nunique())
        
    print(f"Final features ({len(feature_cols)}): {feature_cols}")
    
    return data, feature_cols, cat_dims

# ==========================================
# 2. PyTorch Dataset & Model
# ==========================================

class CTRDataset(Dataset):
    def __init__(self, df, feature_cols, label_col=None):
        self.X = torch.tensor(df[feature_cols].values, dtype=torch.long)
        if label_col:
            self.y = torch.tensor(df[label_col].values, dtype=torch.float32)
        else:
            self.y = None
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers=2):
        super(CrossNetwork, self).__init__()
        self.num_layers = num_layers
        self.W = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim)) for _ in range(num_layers)
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)
        ])

    def forward(self, x):
        # x: [batch_size, input_dim]
        x0 = x
        xi = x
        for i in range(self.num_layers):
            # DCN V2 Standard Cross Layer: x_{l+1} = x0 * (x_l . w_l + b_l) + x_l
            # 这里实现的是向量积形式，计算效率较高
            # x_l . w_l -> [batch_size, 1]
            xw = torch.sum(xi * self.W[i], dim=1, keepdim=True)
            xi = x0 * (xw + self.b[i]) + xi
        return xi

class DCNV2(nn.Module):
    def __init__(self, cat_dims, embed_dim, hidden_units, dropout, cross_layers=3):
        super(DCNV2, self).__init__()
        
        # Embedding Layers
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=dim, embedding_dim=embed_dim) 
            for dim in cat_dims
        ])
        
        # Input Dimension after flattening embeddings
        total_input_dim = len(cat_dims) * embed_dim
        
        # Deep Network
        layers = []
        in_dim = total_input_dim
        for hidden in hidden_units:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden
        self.deep_net = nn.Sequential(*layers)
        
        # Cross Network
        self.cross_net = CrossNetwork(total_input_dim, num_layers=cross_layers)
        
        # Final Output Layer (Deep Output + Cross Output)
        # Deep输出维度是 hidden_units[-1]
        # Cross输出维度是 total_input_dim
        # 结构：Parallel DCN V2
        final_dim = hidden_units[-1] + total_input_dim
        self.out_layer = nn.Linear(final_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch_size, num_features]
        embeds = []
        for i, emb_layer in enumerate(self.embeddings):
            embeds.append(emb_layer(x[:, i]))
        
        # Concat embeddings: [batch_size, total_input_dim]
        embed_concat = torch.cat(embeds, dim=1)
        
        # Deep Side
        deep_out = self.deep_net(embed_concat)
        
        # Cross Side
        cross_out = self.cross_net(embed_concat)
        
        # Stack Deep and Cross outputs
        stacked = torch.cat([deep_out, cross_out], dim=1)
        
        # Final prediction
        logits = self.out_layer(stacked)
        return self.sigmoid(logits).squeeze()

# ==========================================
# 3. 训练与验证流程
# ==========================================

def train_model():
    # 1. 准备数据
    data, feature_cols, cat_dims = preprocess_data('train.csv', 'test.csv')
    
    train_df = data[data['label'].notnull()].reset_index(drop=True)
    test_df = data[data['label'].isnull()].reset_index(drop=True)
    
    # 2. 准备 K-Fold Stratified Splitter
    # 要求：按照 weekday 和 hour 抽样
    # 创建一个 stratify key
    # 注意：weekday和hour在preprocessing阶段已经被label encode了，直接用即可
    train_df['stratify_key'] = train_df['weekday'].astype(str) + "_" + train_df['hour'].astype(str)
    
    skf = StratifiedKFold(n_splits=CONFIG['n_splits'], shuffle=True, random_state=CONFIG['seed'])
    
    # 存储预测结果
    test_preds = np.zeros(len(test_df))
    fold_aucs = []
    
    # 测试集Dataset (固定不变)
    test_dataset = CTRDataset(test_df, feature_cols, label_col=None)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # 开始 K 折训练
    # skf.split 的 y 参数传入 stratify_key 以实现针对 weekday 和 hour 的分层
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['stratify_key'])):
        print(f"\n===== Fold {fold + 1}/{CONFIG['n_splits']} =====")

        X_train, X_val = train_df.iloc[train_idx], train_df.iloc[val_idx]

        train_dataset = CTRDataset(X_train, feature_cols, 'label')
        val_dataset = CTRDataset(X_val, feature_cols, 'label')

        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

        # 初始化模型
        model = DCNV2(
            cat_dims=cat_dims,
            embed_dim=CONFIG['embed_dim'],
            hidden_units=CONFIG['hidden_units'],
            dropout=CONFIG['dropout']
        ).to(CONFIG['device'])

        optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
        criterion = nn.BCELoss()

        # 训练 Epochs + 早停
        best_auc = 0
        best_epoch = 0
        patience = 5  # 早停容忍轮数
        patience_counter = 0
        best_model_state = None
        for epoch in range(int(CONFIG['epochs'])):
            model.train()
            total_loss = 0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(CONFIG['device']), y_batch.to(CONFIG['device'])

                optimizer.zero_grad()
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # 验证
            model.eval()
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(CONFIG['device'])
                    preds = model(X_batch)
                    val_preds.extend(preds.cpu().numpy())
                    val_targets.extend(y_batch.numpy())

            val_auc = roc_auc_score(val_targets, val_preds)
            print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch
                patience_counter = 0
                # 保存最佳模型参数
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}, best epoch was {best_epoch+1} with AUC {best_auc:.4f}")
                    break

        fold_aucs.append(best_auc)

        # 对测试集进行预测 (使用该Fold最优的模型)
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        model.eval()
        fold_preds = []
        with torch.no_grad():
            for X_batch in test_loader:
                X_batch = X_batch.to(CONFIG['device'])
                preds = model(X_batch)
                fold_preds.extend(preds.cpu().numpy())

        test_preds += np.array(fold_preds) / CONFIG['n_splits']

        # 清理显存
        del model, optimizer, criterion, train_dataset, val_dataset
        torch.cuda.empty_cache()
        gc.collect()

    # 4. 评估与输出
    # 计算加权AUC (按照Fold的AUC加权平均，或者简单的AUC平均)
    # 题目要求：每个fold的按照该fold的auc加权平均 -> 可能是指最终结果的评估方式，
    # 或者是指用auc作为权重来ensemble prediction。通常KFold直接平均prediction即可。
    # 这里我们输出验证集平均AUC作为参考。
    mean_auc = np.mean(fold_aucs)
    print(f"\nOverall Mean Validation AUC: {mean_auc:.5f}")
    
    # 5. 生成提交文件
    submission = pd.DataFrame({
        'sample_index': test_df['sample_index'],
        'label': test_preds
    })
    submission.to_csv('submission_nn.csv', index=False)
    print("submission_nn.csv generated successfully.")

if __name__ == '__main__':
    train_model()