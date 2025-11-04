# new model.py

from typing import Dict, List, Tuple
from torch import Tensor
import torch
import torch.nn as nn

class InteractionModel(nn.Module):
    """
    一个采用 "Combine-Then-Interact" 思想的模型。
    它将所有特征嵌入后拼接，然后送入一个深度网络进行交互。
    """
    def __init__(self,
                 user_feature_defs: Tuple[Dict[str, int], List[str]],
                 item_feature_defs: Tuple[Dict[str, int], List[str]],
                 context_feature_defs: Tuple[Dict[str, int], List[str]],
                 embedding_dim_per_feature: int,
                 hidden_dim: int,
                 # emb_dim 和 head_nums 不再需要
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # 1. --- 统一的特征嵌入层 ---
        # 将所有离散特征的定义合并到一个字典里
        all_discrete_vocab_sizes = {
            **user_feature_defs[0],
            **item_feature_defs[0],
            **context_feature_defs[0]
        }
        
        # 将所有连续特征的名字合并到一个列表里
        self.user_continuous_names = user_feature_defs[1]
        self.item_continuous_names = item_feature_defs[1]
        self.context_continuous_names = context_feature_defs[1]
        all_continuous_names = self.user_continuous_names + self.item_continuous_names + self.context_continuous_names

        self.embedding_layers = nn.ModuleDict({
            feat_name: nn.Embedding(vocab_size, embedding_dim_per_feature)
            for feat_name, vocab_size in all_discrete_vocab_sizes.items()
        })
        self.discrete_feature_names = list(all_discrete_vocab_sizes.keys())


        # 2. --- 计算深度交互网络的输入维度 ---
        # 输入维度 = (所有离散特征数 * 单个embedding维度) + 所有连续特征数
        total_input_dim = (len(self.discrete_feature_names) * embedding_dim_per_feature) + \
                          len(all_continuous_names)


        # 3. --- 深度交互网络 (Interaction MLP) ---
        # 这个MLP是模型的核心，负责学习所有特征之间的高阶交互
        self.interaction_mlp = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 4. --- Scorer ---
        # 最终的输出层
        self.scorer = nn.Linear(hidden_dim // 2, 1)


    def forward(self,
                user_features: Dict[str, Tensor],
                item_features: Dict[str, Tensor],
                context_features: Dict[str, Tensor]) -> Tensor:
        
        # --- 步骤1 & 2: 统一嵌入与拼接 (Combine) ---
        
        # 将输入的三个字典合并，方便处理
        all_features = {**user_features, **item_features, **context_features}
        
        embedded_discrete_parts = []
        for name in self.discrete_feature_names:
            # 从ModuleDict中找到对应的Embedding层并进行嵌入
            embedded_discrete_parts.append(self.embedding_layers[name](all_features[name]))

        continuous_parts = []
        # 注意：这里需要检查特征是否存在，因为DataLoader可能会传空字典
        if self.user_continuous_names:
            continuous_parts.extend([all_features[name] for name in self.user_continuous_names])
        if self.item_continuous_names:
            continuous_parts.extend([all_features[name] for name in self.item_continuous_names])
        if self.context_continuous_names:
            continuous_parts.extend([all_features[name] for name in self.context_continuous_names])

        # 将所有部分拼接成一个超宽向量
        all_parts = embedded_discrete_parts
        if continuous_parts:
            # 将连续特征堆叠成(batch_size, num_continuous_feats)的张量
            continuous_vector = torch.stack(continuous_parts, dim=1).float()
            all_parts.append(continuous_vector)
            
        mega_vector = torch.cat(all_parts, dim=1)

        # --- 步骤3: 深度交互 (Interact) ---
        fused_output = self.interaction_mlp(mega_vector)

        # --- 步骤4: 打分 ---
        logits = self.scorer(fused_output)

        return logits