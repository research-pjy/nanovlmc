import torch
from torch import nn

from nanovlm.models.blocks import Block
from .tokenizers import ConvTokenizer


class VisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.vision_dim
        self.tokenizer = ConvTokenizer(config)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.position = nn.Parameter(torch.empty(1, (config.image_size // config.patch_size) ** 2 + 1, d))
        nn.init.normal_(self.position, std=0.02)
        self.input_norm = nn.LayerNorm(d)
        self.blocks = nn.Sequential(*[Block(d, config.vision_heads, config.mlp_ratio, config.dropout) for _ in range(config.vision_layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, images):
        patches = self.tokenizer(images)
        x = torch.cat((self.cls.expand(images.shape[0], -1, -1), patches), dim=1)
        x = self.blocks(self.input_norm(x + self.position))
        return self.norm(x)[:, 0]
