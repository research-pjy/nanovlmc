import torch
from torch import nn

from .encoders import VisionEncoder
from .projector import Projector
from .decoder import Decoder


class NanoVLM(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.encoder = VisionEncoder(config)
        self.projector = Projector(config.vision_dim, config.text_dim)
        self.decoder = Decoder(config, vocab_size)

    def visual_prefix(self, images, zero_image=False):
        visual = self.projector(self.encoder(images))
        return torch.zeros_like(visual) if zero_image else visual

    def forward(self, images, input_ids, zero_image=False):
        return self.decoder(self.visual_prefix(images, zero_image), input_ids)

    def parameter_counts(self):
        counts = {name: sum(p.numel() for p in module.parameters()) for name, module in self.named_children()}
        return {**counts, "total": sum(counts.values())}

    @torch.no_grad()
    def generate(self, images, prefix_ids, eos_id=2, max_new_tokens=96, zero_image=False):
        """Greedy single-example decoding; no cache, no sampling ambiguity."""
        if self.training:
            raise ValueError("Call eval() before generating")
        if images.shape[0] != 1 or prefix_ids.shape[0] != 1:
            raise ValueError("Generation accepts one example at a time")
        visual = self.visual_prefix(images, zero_image)
        ids = prefix_ids
        for _ in range(max_new_tokens):
            if ids.shape[1] > self.decoder.max_tokens:
                break
            logits = self.decoder(visual, ids)[:, -1].clone()
            logits[:, 0] = logits[:, 1] = -torch.inf  # Never generate PAD or BOS.
            next_id = logits.argmax(-1, keepdim=True)
            ids = torch.cat((ids, next_id), dim=1)
            if next_id.item() == eos_id:
                break
        return ids
