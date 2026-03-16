"""
Template for custom models. Copy this file and adapt for your own architecture.

Usage:
    uv run profile.py --model models/custom.py --class-name MyModel --input-shape 8,3,224,224 --dtype float16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    """
    Example custom model -- a simple CNN + MLP.
    Replace this with your own architecture.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(256, num_classes)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"MyModel: {n_params / 1e6:.1f}M parameters")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
