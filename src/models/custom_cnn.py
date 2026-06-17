#!/usr/bin/env python3
"""Custom CNN baseline for binary seed-germination crop classification.

Input:
  Tensor with shape [batch_size, 3, 224, 224]

Output:
  Raw logits with shape [batch_size, 2].
  Use torch.nn.CrossEntropyLoss during training; do not apply softmax inside
  the model.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Convolution + batch norm + ReLU + max pool block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    """Small CNN baseline for two-class crop classification."""

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(input_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_custom_cnn(
    input_channels: int = 3,
    num_classes: int = 2,
    dropout: float = 0.3,
) -> CustomCNN:
    """Factory used by training scripts."""

    return CustomCNN(
        input_channels=input_channels,
        num_classes=num_classes,
        dropout=dropout,
    )


def count_trainable_parameters(model: nn.Module) -> int:
    """Return number of trainable parameters for reporting."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    model = build_custom_cnn()
    sample = torch.randn(4, 3, 224, 224)
    logits = model(sample)

    print(model)
    print(f"Output shape: {tuple(logits.shape)}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
