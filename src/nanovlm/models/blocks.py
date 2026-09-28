"""Pre-normalized residual transformer shared by vision and language."""
import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout, causal=False):
        super().__init__()
        self.heads, self.dropout, self.causal = heads, dropout, causal
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=self.causal)
        return self.residual_dropout(self.out(a.transpose(1, 2).reshape(b, n, d)))


class Block(nn.Module):
    def __init__(self, dim, heads, ratio, dropout, causal=False):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = Attention(dim, heads, dropout, causal)
        self.mlp = nn.Sequential(nn.Linear(dim, ratio * dim), nn.GELU(), nn.Linear(ratio * dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))
