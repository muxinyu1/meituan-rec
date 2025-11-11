from typing import Dict, List, Tuple
from torch import Tensor
import torch
import torch.nn as nn

class MixedFeatureEncoder(nn.Module):
    def __init__(self,
                 discrete_vocab_sizes: Dict[str, int],
                 continuous_feature_names: List[str],
                 embedding_dim_per_feature: int,
                 output_emb_dim: int,
                 hidden_dim: int):
        super().__init__()
        self.discrete_feature_names = list(discrete_vocab_sizes.keys())
        self.continuous_feature_names = continuous_feature_names
        # self.num_continuous_features = len(self.continuous_feature_names)

        self.embedding_layers = nn.ModuleDict({
            feat_name: nn.Embedding(vocab_size, embedding_dim_per_feature)
            for feat_name, vocab_size in discrete_vocab_sizes.items()
        })

        total_input_dim = (len(self.discrete_feature_names) * embedding_dim_per_feature) + \
                          len(self.continuous_feature_names)
        # self.continuous_bn = nn.BatchNorm1d(self.num_continuous_features)
        self.projection_mlp = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_emb_dim),
            nn.LayerNorm(output_emb_dim)
        )

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        embedded_discrete = []
        for name in self.discrete_feature_names:
            embedded_discrete.append(self.embedding_layers[name](x[name]))

        if self.continuous_feature_names:
            continuous_vectors = torch.stack([x[name] for name in self.continuous_feature_names], dim=1)
            # continuous_vectors = self.continuous_bn(continuous_vectors)
            all_vectors = embedded_discrete + [continuous_vectors.float()]
        else:
            all_vectors = embedded_discrete
            
        if len(all_vectors) > 1:
            combined_vector = torch.cat(all_vectors, dim=1)
        else:
            combined_vector = all_vectors[0]

        return self.projection_mlp(combined_vector)


class Model(nn.Module):
    def __init__(self,
                 user_feature_defs: Tuple[Dict[str, int], List[str]],
                 item_feature_defs: Tuple[Dict[str, int], List[str]],
                 context_feature_defs: Tuple[Dict[str, int], List[str]],
                 embedding_dim_per_feature: int,
                 hidden_dim: int,
                 emb_dim: int,
                 head_nums: int,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.emb_dim = emb_dim
        self.head_nums = head_nums
        assert self.emb_dim % self.head_nums == 0
        
        self.user_encoder = MixedFeatureEncoder(
            discrete_vocab_sizes=user_feature_defs[0],
            continuous_feature_names=user_feature_defs[1],
            embedding_dim_per_feature=embedding_dim_per_feature,
            output_emb_dim=emb_dim,
            hidden_dim=hidden_dim
        )
        self.item_encoder = MixedFeatureEncoder(
            discrete_vocab_sizes=item_feature_defs[0],
            continuous_feature_names=item_feature_defs[1],
            embedding_dim_per_feature=embedding_dim_per_feature,
            output_emb_dim=emb_dim,
            hidden_dim=hidden_dim
        )
        self.context_encoder = MixedFeatureEncoder(
            discrete_vocab_sizes=context_feature_defs[0],
            continuous_feature_names=context_feature_defs[1],
            embedding_dim_per_feature=embedding_dim_per_feature,
            output_emb_dim=emb_dim,
            hidden_dim=hidden_dim
        )

        self.user_query = nn.Linear(emb_dim, emb_dim)
        self.context_key = nn.Linear(emb_dim, emb_dim)
        self.item_value = nn.Linear(emb_dim, emb_dim)

        self.cat_norm = nn.LayerNorm(emb_dim)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self,
                user_features: Dict[str, Tensor],
                item_features: Dict[str, Tensor],
                context_features: Dict[str, Tensor]) -> Tensor:

        user_emb = self.user_encoder(user_features)
        item_emb = self.item_encoder(item_features)
        context_emb = self.context_encoder(context_features)

        q: Tensor = self.user_query(user_emb)
        k: Tensor = self.context_key(context_emb)
        v: Tensor = self.item_value(item_emb)

        head_dim = self.emb_dim // self.head_nums
        bsz = q.size(0)

        q_heads = q.view(bsz, self.head_nums, head_dim)
        k_heads = k.view(bsz, self.head_nums, head_dim)
        v_heads = v.view(bsz, self.head_nums, head_dim)

        scores = torch.sum(q_heads * k_heads, dim=-1)
        scores = scores / (head_dim ** 0.5)
        gates = torch.sigmoid(scores)
        gates = gates.unsqueeze(-1) 
        gated_v_heads = gates * v_heads
        context_aware_item_emb = gated_v_heads.view(bsz, self.emb_dim)
        context_aware_item_emb = self.cat_norm(context_aware_item_emb)

        combined_features = torch.cat([user_emb, context_aware_item_emb], dim=1)

        fused_output = self.fusion_mlp(combined_features)
        logits = self.scorer(fused_output)

        return logits