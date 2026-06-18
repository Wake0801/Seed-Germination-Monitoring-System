#!/usr/bin/env python3
"""Faster R-CNN builders used by object-detection inference.

The project requirement is to avoid pretrained models. The builder below
therefore sets both ``weights`` and ``weights_backbone`` to ``None``.
"""

from __future__ import annotations

from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn


def build_faster_rcnn_resnet50_fpn_scratch(
    num_classes: int = 3,
    min_size: int = 640,
    max_size: int = 640,
) -> nn.Module:
    """Build Faster R-CNN ResNet50-FPN without pretrained weights."""

    return fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
    )


def build_detection_model(
    architecture: str,
    num_classes: int = 3,
    min_size: int = 640,
    max_size: int = 640,
) -> nn.Module:
    architecture = architecture.lower()
    if architecture == "fasterrcnn_resnet50_fpn_scratch":
        return build_faster_rcnn_resnet50_fpn_scratch(
            num_classes=num_classes,
            min_size=min_size,
            max_size=max_size,
        )

    raise ValueError(f"Unsupported detection architecture: {architecture}")
