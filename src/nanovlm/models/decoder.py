import torch
from torch import nn

from .blocks import Block


class Decoder(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.max_tokens = config.max_text_tokens
        self.embedding = nn.Embedding(vocab_size, config.text_dim, padding_idx=0)
        self.position = nn.Embedding(config.max_text_tokens + 1, config.text_dim)
        self.blocks = nn.Sequential(*[Block(config.text_dim, config.text_heads, config.mlp_ratio, config.dropout, causal=True) for _ in range(config.text_layers)])
        self.norm = nn.LayerNorm(config.text_dim)
        self.head = nn.Linear(config.text_dim, vocab_size, bias=False)

    def forward(self, visual, input_ids):
        if input_ids.shape[1] > self.max_tokens:
            raise ValueError("Text exceeds configured context length")
        x = torch.cat((visual.unsqueeze(1), self.embedding(input_ids)), dim=1)
        x = x + self.position(torch.arange(x.shape[1], device=x.device))
        # Discard the image position: BOS predicts word 1, each word predicts next.
        return self.head(self.norm(self.blocks(x)))[:, 1:]
