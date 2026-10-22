"""Grad-CAM (Phase D prep): maps have the input's shape and range, follow what the model
uses, and change when its weights are randomized. No training, no data."""

from __future__ import annotations

import torch
from torch import nn

from cdm.explain import grad_cam, last_conv_block, randomization_check, randomized_copy
from cdm.train import build_model


def brightness_model() -> nn.Sequential:
    """Class 1's logit grows with brightness anywhere: a model whose evidence is known."""
    conv, head = nn.Conv2d(3, 4, 3, padding=1), nn.Linear(4, 2)
    with torch.no_grad():
        conv.weight.fill_(1 / 27)
        nn.init.zeros_(conv.bias)  # type: ignore[arg-type]
        head.weight.copy_(torch.tensor([[-1.0] * 4, [1.0] * 4]))
        head.bias.zero_()
    return nn.Sequential(conv, nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), head)


def first_conv(model: nn.Module) -> nn.Conv2d:
    return next(m for m in model.modules() if isinstance(m, nn.Conv2d))


def bright_corner(n: int = 4) -> torch.Tensor:
    images = torch.zeros(n, 3, 32, 32)
    images[:, :, :10, :10] = 1.0
    return images


def conv_layer(model: nn.Module) -> nn.Module:
    return [m for m in model.modules() if isinstance(m, nn.Conv2d)][-1]


def test_maps_have_the_input_size_and_point_at_the_evidence() -> None:
    model = brightness_model()
    cam = grad_cam(model, conv_layer(model), bright_corner(), torch.ones(4, dtype=torch.long))
    assert cam.shape == (4, 32, 32)
    assert float(cam.min()) >= 0 and float(cam.max()) <= 1
    assert cam[:, :10, :10].mean() > cam[:, 16:, 16:].mean() + 0.5


def test_randomized_weights_change_the_maps() -> None:
    model = brightness_model()
    copy_ = randomized_copy(model, seed=1)
    trained, randomized = first_conv(model).weight, first_conv(copy_).weight
    assert not torch.equal(randomized, trained)
    assert torch.equal(trained, torch.full_like(trained, 1 / 27))  # the original is untouched
    images = bright_corner() + 0.1 * torch.rand(
        4, 3, 32, 32, generator=torch.Generator().manual_seed(0)
    )
    assert randomization_check(model, images, conv_layer, seed=1) < 0.9


def test_efficientnet_target_layer_and_map_shape() -> None:
    model = build_model(pretrained=False)
    cam = grad_cam(model, last_conv_block(model), torch.rand(2, 3, 64, 64))
    assert cam.shape == (2, 64, 64)
