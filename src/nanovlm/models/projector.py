from torch import nn


class Projector(nn.Sequential):
    def __init__(self, vision_dim, text_dim):
        super().__init__(nn.Linear(vision_dim, text_dim), nn.GELU())
