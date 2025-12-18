from typing import List, Tuple
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import ModuleList

from rmblk import RankMixerBlock


class RankMixer(nn.Module):

    def __init__(
        self,
        num_tokens: int,
        num_heads: int,
        num_layers: int,
        d_model: int,
        hidden_dim: int,
        dropout: float,
        features: List[Tuple[int, int]], # [[该特征类别数, 该特征emb_dim], ...]
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.embeddings = ModuleList([nn.Embedding(i, emb_dim) for i, emb_dim in features])
        self.total_raw_emb_dim = sum([emb_dim for _, emb_dim in features])
        self.proj = nn.Linear(self.total_raw_emb_dim, num_tokens * d_model)
        self.blocks = ModuleList(
            [
                RankMixerBlock(num_tokens, num_heads, d_model, hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.scorer = nn.Linear(d_model, 1)


    def forward(self, x: Tensor):
        # [bsz, num_features]
        # -> [bsz, len(self.embeddings) * emb_dim] 每个feature 的emb concat
        bsz = x.size(0)
        emb_lst = []
        for i, emb_layer in enumerate(self.embeddings):
            emb_lst.append(emb_layer(x[:, i]))
        concat_emb = torch.concat(emb_lst, -1)
        projected: Tensor = self.proj(concat_emb)
        x = projected.contiguous().view(bsz, self.num_tokens, self.d_model)

        for blk in self.blocks:
            x = blk(x)

        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.squeeze(-1)

        logits = self.scorer(x)

        return logits

