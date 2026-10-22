"""Grad-CAM (Phase D prep): maps have the input's shape and range, follow what the model
uses, and change when its weights are randomized. No training, no data."""

from __future__ import annotations

import copy

import pytest
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


def varied_images() -> torch.Tensor:
    noise = torch.rand(4, 3, 32, 32, generator=torch.Generator().manual_seed(0))
    return bright_corner() + noise


def test_positive_control_an_identical_model_correlates_fully() -> None:
    model = brightness_model()
    a = grad_cam(model, first_conv(model), varied_images()).flatten(1)
    twin = copy.deepcopy(model)
    b = grad_cam(twin, first_conv(twin), varied_images()).flatten(1)
    assert torch.allclose(a, b)


def test_randomized_batchnorm_statistics_are_reset_and_maps_vary_by_input() -> None:
    model = build_model(pretrained=False)
    bn = next(m for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    assert bn.running_var is not None
    with torch.no_grad():
        bn.running_var.fill_(5.0)  # pretend these were learned
    copy_ = randomized_copy(model, seed=0)
    bn_copy = next(m for m in copy_.modules() if isinstance(m, nn.BatchNorm2d))
    assert bn_copy.running_var is not None
    assert torch.equal(bn_copy.running_var, torch.ones_like(bn_copy.running_var))
    assert float(bn.running_var[0]) == 5.0  # the original is untouched
    maps = grad_cam(copy_, last_conv_block(copy_), torch.rand(3, 3, 64, 64))
    assert not torch.allclose(maps[0], maps[1])


def test_a_constant_map_raises_instead_of_passing() -> None:
    model = brightness_model()
    flat = torch.zeros(2, 3, 32, 32)  # no evidence anywhere: a flat map
    with pytest.raises(ValueError):
        randomization_check(model, flat, first_conv, seed=1)


def test_hooks_are_removed_and_mode_restored_after_an_error() -> None:
    model = brightness_model()
    model.train()
    layer = first_conv(model)
    with pytest.raises(RuntimeError):
        grad_cam(model, layer, torch.rand(1, 5, 32, 32))  # wrong channel count
    assert not layer._forward_hooks
    assert model.training
