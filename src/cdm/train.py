"""EfficientNet-B0 classifier for HAM10000."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from cdm.data import CLASSES


def build_model(pretrained: bool = True) -> nn.Module:
    """ImageNet EfficientNet-B0 with a new 7-class head."""
    model = cast(
        nn.Module,
        efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None),
    )
    classifier = cast(nn.Sequential, model.classifier)
    head = classifier[1]
    assert isinstance(head, nn.Linear)
    classifier[1] = nn.Linear(head.in_features, len(CLASSES))
    return model


def features_and_logits(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The 1280-d pooled features the classifier head sees, and the head's logits."""
    backbone, pool, head = (
        cast(nn.Module, getattr(model, n)) for n in ("features", "avgpool", "classifier")
    )
    feats = torch.flatten(pool(backbone(x)), 1)
    return feats, head(feats)
