from torch import Tensor
import torch.nn as nn

class MuiltiHeadTokenMixing(nn.Module):
    
    def __init__(self, num_tokens: int, num_heads: int, d_model: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0

        self.head_dim = d_model // num_heads

    def forward(self, x: Tensor):
        bsz, t, _ = x.size()

        x = x.contiguous().view(bsz, t, self.num_heads, -1)
        x = x.transpose(1, 2)
        x = x.contiguous().view(bsz, t, -1)

        return x
    
    