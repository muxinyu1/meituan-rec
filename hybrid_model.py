import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Tuple

# ==================================================================
# === 1. LGBM 叶节点编码器 (与 Hybrid 版本相同) ===
# ==================================================================

class LgbmLeafEncoder(nn.Module):
    """
    此模块为LGBM的N棵树中的每一棵树创建一个独立的Embedding层。
    
    输入形状: (batch_size, n_trees) e.g. (1024, 200)
    输出形状: (batch_size, n_trees * embedding_dim) e.g. (1024, 200 * 8)
    """
    def __init__(self, n_trees: int, num_leaves: int, embedding_dim: int):
        super().__init__()
        self.n_trees = n_trees
        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(num_leaves, embedding_dim) for _ in range(n_trees)]
        )
        self.output_dim = n_trees * embedding_dim

    def forward(self, x: Tensor) -> Tensor:
        x = x.long()
        embeddings = []
        for i in range(self.n_trees):
            emb_i = self.embedding_layers[i](x[:, i])
            embeddings.append(emb_i)
        return torch.cat(embeddings, dim=1)

# --- (备选方案：参数共享的编码器) ---
# class SharedLgbmLeafEncoder(nn.Module):
#     def __init__(self, n_trees: int, num_leaves: int, embedding_dim: int):
#         super().__init__()
#         self.n_trees = n_trees
#         self.embedding = nn.Embedding(num_leaves, embedding_dim)
#         self.output_dim = n_trees * embedding_dim
#     def forward(self, x: Tensor) -> Tensor:
#         x = x.long()
#         emb = self.embedding(x)
#         return emb.view(x.size(0), -1)

# ==================================================================
# === 2. 纯 GBDT-DL 模型 (替换你原有的 Model) ===
# ==================================================================

class HybridModel(nn.Module):
    """
    这是 "纯 GBDT-DL" 模型 (选项一)。
    它实现了和你 train.py 兼容的 "伪装" 接口，
    但内部只使用 GBDT 叶节点特征。
    """
    def __init__(self,
                 # --- "伪装" 参数 (将被忽略) ---
                 user_feature_defs: Tuple[Dict[str, int], List[str]],
                 item_feature_defs: Tuple[Dict[str, int], List[str]],
                 context_feature_defs: Tuple[Dict[str, int], List[str]],
                 embedding_dim_per_feature: int,
                 emb_dim: int,
                 head_nums: int,
                 # --- 实际使用参数 ---
                 hidden_dim: int,
                 n_trees: int,       
                 num_leaves: int,    
                 leaf_embed_dim: int,
                 dropout_rate: float = 0.4 # 简单的模型可以承受更高的 Dropout
                ) -> None:
        
        super().__init__()
        
        # 1. 初始化 LGBM 叶节点编码器
        # (你可以切换使用 LgbmLeafEncoder 或 SharedLgbmLeafEncoder)
        self.lgbm_leaf_encoder = LgbmLeafEncoder(
            n_trees=n_trees,
            num_leaves=num_leaves,
            embedding_dim=leaf_embed_dim
        )
        
        lgbm_output_dim = self.lgbm_leaf_encoder.output_dim # e.g. 200 * 8 = 1600

        # 2. 初始化一个简单的 MLP 打分器
        # 这个 MLP 是这个模型唯一的 "大脑"
        self.scorer_mlp = nn.Sequential(
            nn.Linear(lgbm_output_dim, hidden_dim), # (1600 -> 512)
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim // 2), # (512 -> 256)
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim // 2, 1) # (256 -> 1)
        )

    def forward(self,
                # --- "伪装" 输入 (将被忽略) ---
                user_features: Dict[str, Tensor],
                item_features: Dict[str, Tensor],
                context_features: Dict[str, Tensor],
                # --- 实际使用输入 ---
                leaf_indices: Tensor
               ) -> Tensor:
        
        # 1. 编码 GBDT 叶节点特征
        # (batch_size, 200) -> (batch_size, 1600)
        leaf_emb = self.lgbm_leaf_encoder(leaf_indices)
        
        # 2. 通过 MLP 打分
        # (batch_size, 1600) -> (batch_size, 1)
        logits = self.scorer_mlp(leaf_emb)

        return logits