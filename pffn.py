import torch
import torch.nn as nn
from torch import Tensor

class PerTokenFFN(nn.Module):
    def __init__(self, num_tokens: int, d_model: int, hidden_dim: int, dropout: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim

        self.w1 = nn.Parameter(torch.Tensor(num_tokens, d_model, hidden_dim))
        self.b1 = nn.Parameter(torch.Tensor(num_tokens, hidden_dim))

        self.w2 = nn.Parameter(torch.Tensor(num_tokens, hidden_dim, d_model))
        self.b2 = nn.Parameter(torch.Tensor(num_tokens, d_model))

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_normal_(self.w2)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def forward(self, x: Tensor):

        x = torch.einsum('btd,tde->bte', x, self.w1) + self.b1
        x = self.activation(x)
        x = self.dropout(x)

        x = torch.einsum('btd,tde->bte', x, self.w2) + self.b2
        x = self.dropout(x)

        return x