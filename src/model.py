"""Faithful three-pool U-Net translation; returns logits rather than probabilities."""

import torch
import torch.nn.functional as F
from torch import nn

from ops import BACKENDS, ReplaceableConv2d


def block(in_channels, out_channels, backend):
    return nn.Sequential(
        ReplaceableConv2d(in_channels, out_channels, 3, padding=1, backend=backend),
        nn.ReLU(),
        ReplaceableConv2d(out_channels, out_channels, 3, padding=1, backend=backend),
        nn.ReLU(),
    )


class UNet(nn.Module):
    def __init__(self, backend="torch"):
        super().__init__()
        if backend not in BACKENDS:
            raise ValueError(f"Unknown backend: {backend}")
        self.enc1 = block(1, 16, backend)
        self.enc2 = block(16, 32, backend)
        self.enc3 = block(32, 64, backend)
        self.bridge = block(64, 128, backend)
        self.dec3 = block(192, 64, backend)
        self.dec2 = block(96, 32, backend)
        self.dec1 = block(48, 16, backend)
        self.head = ReplaceableConv2d(16, 1, 1, backend=backend)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 1 or any(s % 8 or s < 8 for s in x.shape[-2:]):
            raise ValueError("Expected NCHW with one channel and H/W divisible by 8")
        c1 = self.enc1(x)
        c2 = self.enc2(F.max_pool2d(c1, 2))
        c3 = self.enc3(F.max_pool2d(c2, 2))
        c4 = self.bridge(F.max_pool2d(c3, 2))
        c5 = self.dec3(torch.cat((F.interpolate(c4, scale_factor=2, mode="nearest"), c3), 1))
        c6 = self.dec2(torch.cat((F.interpolate(c5, scale_factor=2, mode="nearest"), c2), 1))
        c7 = self.dec1(torch.cat((F.interpolate(c6, scale_factor=2, mode="nearest"), c1), 1))
        return self.head(c7)
