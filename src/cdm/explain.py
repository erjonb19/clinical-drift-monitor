"""Grad-CAM for the shipped EfficientNet, with the weight-randomization sanity check (research
Phase D; Selvaraju et al. 2017, Adebayo et al. 2018).

Grad-CAM weights a convolutional layer's feature maps by the mean gradient of the target
class's logit, keeps the positive part, and upsamples to the input size. On EfficientNet-B0 the
last block is 7×7 at 224 px, so maps are coarse; that is stated wherever they appear.

The sanity check: a map that barely changes when the model's weights are randomized shows the
input, not what the model learned, and is not interpreted. :func:`randomization_check` reports
the rank correlation between the trained model's maps and a randomized copy's.

CPU-only prep: this module runs no training and reads no data. The worksheet that uses it runs
on Databricks with the shipped checkpoints, after a committed spec (docs/EVAL_PLAN.md).
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch import nn


def last_conv_block(model: nn.Module) -> nn.Module:
    """EfficientNet's last feature block (``model.features[-1]``), the usual Grad-CAM layer."""
    features = getattr(model, "features", None)
    if not isinstance(features, nn.Sequential):
        raise ValueError("expected a torchvision EfficientNet with a .features Sequential")
    return features[-1]


def grad_cam(
    model: nn.Module, layer: nn.Module, images: torch.Tensor, target: torch.Tensor | None = None
) -> torch.Tensor:
    """Heatmaps (N, H, W) in [0, 1] at the input's size, for ``target`` classes (default: the
    predicted class). The model is left unchanged and in eval mode."""
    saved: dict[str, torch.Tensor] = {}

    def keep(_: nn.Module, __: object, output: torch.Tensor) -> None:
        output.retain_grad()
        saved["maps"] = output

    model.eval()
    handle = layer.register_forward_hook(keep)
    try:
        images = images.detach().clone().requires_grad_(True)
        logits = model(images)
        classes = logits.argmax(dim=1) if target is None else target
        model.zero_grad(set_to_none=True)
        logits.gather(1, classes.view(-1, 1)).sum().backward()
    finally:
        handle.remove()
    maps = saved["maps"]
    assert maps.grad is not None
    weights = maps.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * maps).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze(1).detach()
    low = cam.flatten(1).min(dim=1).values.view(-1, 1, 1)
    high = cam.flatten(1).max(dim=1).values.view(-1, 1, 1)
    return (cam - low) / (high - low).clamp_min(1e-12)


def randomized_copy(model: nn.Module, seed: int = 0) -> nn.Module:
    """A copy with every parameter re-drawn (normal, scaled like the original), the "full
    randomization" of Adebayo et al."""
    clone = copy.deepcopy(model)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in clone.parameters():
            scale = float(param.std()) if param.numel() > 1 else 1.0
            noise = torch.randn(param.shape, generator=generator, dtype=param.dtype)
            param.copy_(noise * (scale or 1.0))
    return clone


def randomization_check(
    model: nn.Module,
    images: torch.Tensor,
    layer: Callable[[nn.Module], nn.Module] = last_conv_block,
    seed: int = 0,
) -> float:
    """Mean Spearman correlation, per image, between the trained model's maps and a randomized
    copy's. Near 1 means the maps do not depend on the weights. ``layer`` picks the target
    layer from a model (default: EfficientNet's last block)."""
    random_model = randomized_copy(model, seed)
    a = grad_cam(model, layer(model), images).flatten(1).numpy()
    b = grad_cam(random_model, layer(random_model), images).flatten(1).numpy()
    correlations = []
    for x, y in zip(a, b, strict=True):
        rho = spearmanr(x, y).statistic
        correlations.append(0.0 if np.isnan(rho) else float(rho))
    return float(np.mean(correlations))
