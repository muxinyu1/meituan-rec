import torch.nn as nn

from mhtm import MuiltiHeadTokenMixing
from pffn import PerTokenFFN
from torch import Tensor


class RankMixerBlock(nn.Module):

    def __init__(
        self,
        num_tokens: int,
        num_heads: int,
        d_model: int,
        hidden_dim: int,
        dropout: float,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.token_mixing = MuiltiHeadTokenMixing(num_tokens, num_heads, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.pffn = PerTokenFFN(num_tokens, d_model, hidden_dim, dropout)

    def forward(self, x: Tensor):
        # Pre-Norm
        res = x
        x = self.norm1(x)
        x = self.token_mixing(x)
        x = x + res

        res = x
        x = self.norm2(x)
        x = self.pffn(x)
        x = x + res

        return x
