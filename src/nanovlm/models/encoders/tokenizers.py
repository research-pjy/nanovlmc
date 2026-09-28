"""Equal-parameter image tokenizers; only convolution boundaries differ."""
from torch import nn
from torch.nn import functional as F


class ChannelLayerNorm(nn.Module):
    """Normalize channels at each spatial location, never across patches."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvTokenizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c1, c2 = config.conv_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, 3, stride=2, padding=1), ChannelLayerNorm(c1), nn.ReLU(),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1), ChannelLayerNorm(c2), nn.ReLU(),
        )
        self.projection = nn.Linear(c2 * (config.patch_size // 4) ** 2, config.vision_dim)

    def forward(self, images):
        c = self.config
        b, channels, h, w = images.shape
        if (channels, h, w) != (3, c.image_size, c.image_size):
            raise ValueError(f"Expected RGB {c.image_size}x{c.image_size} images")
        if c.encoder == "image_conv":
            # Convolution receptive fields cross future patch boundaries.
            features = self.stem(images)
            side = c.patch_size // 4
            tokens = F.unfold(features, kernel_size=side, stride=side).transpose(1, 2)
        else:
            # Shared stem on isolated patches: padding occurs at EVERY patch edge.
            patches = F.unfold(images, kernel_size=c.patch_size, stride=c.patch_size).transpose(1, 2)
            n = patches.shape[1]
            patches = patches.reshape(b * n, 3, c.patch_size, c.patch_size)
            tokens = self.stem(patches).reshape(b, n, -1)
        return self.projection(tokens)
