"""cdm.serve_onnx: export a small EfficientNet-shaped model, the noise rule, latency."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from cdm.serve_onnx import (
    ENSEMBLE_KEYS,
    export,
    gaps,
    latency_ms,
    onnx_outputs,
    passes_against_noise,
    session,
    top2_margin,
    torch_outputs,
)


class Tiny(nn.Module):
    """The three attributes features_and_logits reads, at toy size."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.features = nn.Sequential(nn.Conv2d(3, 8, 3, stride=2), nn.ReLU())
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(8, 7)


def test_export_matches_pytorch_for_any_batch_size(tmp_path: Path) -> None:
    model = Tiny().eval()
    sess = session(export(model, tmp_path / "tiny.onnx", size=32), threads=1)
    x = torch.randn(6, 3, 32, 32)
    for batch in (1, 4):
        got = gaps(torch_outputs(model, x, batch), onnx_outputs(sess, x, batch))
        assert got["logits_max_abs_diff"] < 1e-4 and got["top1_disagreements"] == []


def test_noise_between_batch_sizes_is_measured_the_same_way() -> None:
    model = Tiny().eval()
    x = torch.randn(6, 3, 32, 32)
    noise = gaps(torch_outputs(model, x, 1), torch_outputs(model, x, 4))
    assert noise["logits_max_abs_diff"] < 1e-4 and noise["top1_disagreements"] == []


NOISE = {
    "logits_max_abs_diff": 0.04,
    "probabilities_max_abs_diff": 0.01,
    "features_max_abs_diff_relative": 0.005,
    "top1_disagreements": [3],
}


def gap(**changes: object) -> dict[str, object]:
    base = {"logits_max_abs_diff": 0.07, "probabilities_max_abs_diff": 0.019,
            "features_max_abs_diff_relative": 0.009, "top1_disagreements": []}  # fmt: skip
    return base | changes


MARGIN = np.array([0.5, 0.005, 0.5, 0.5, 0.5])


def test_the_noise_rule_checks_every_gap_at_twice_the_noise() -> None:
    assert passes_against_noise(gap(), NOISE, MARGIN) == []
    for key, value in (("logits_max_abs_diff", 0.09), ("probabilities_max_abs_diff", 0.021),
                       ("features_max_abs_diff_relative", 0.011)):  # fmt: skip
        failed = passes_against_noise(gap(**{key: value}), NOISE, MARGIN)
        assert len(failed) == 1 and failed[0].startswith(key)


def test_flips_are_allowed_only_where_pytorch_flips_or_the_top_two_are_close() -> None:
    assert passes_against_noise(gap(top1_disagreements=[1, 3]), NOISE, MARGIN) == []
    assert "[0]" in passes_against_noise(gap(top1_disagreements=[0]), NOISE, MARGIN)[0]
    at_limit = np.array([0.01, 0.5, 0.5, 0.5, 0.5])  # a margin of exactly 0.01 is not "close"
    assert passes_against_noise(gap(top1_disagreements=[0]), NOISE, at_limit) != []


def test_a_missing_gap_is_an_error_not_a_pass() -> None:
    with pytest.raises(KeyError, match="logits"):
        passes_against_noise(gap(), {"top1_disagreements": []}, MARGIN)
    ensemble_noise = {"probabilities_max_abs_diff": 0.01, "top1_disagreements": []}
    assert passes_against_noise(gap(), ensemble_noise, MARGIN, keys=ENSEMBLE_KEYS) == []


def test_top2_margin() -> None:
    logits = np.log(np.array([[0.6, 0.3, 0.1], [0.34, 0.33, 0.33]]))
    assert np.allclose(top2_margin(logits), [0.3, 0.01])


def test_latency_percentiles_are_ordered() -> None:
    out = latency_ms(lambda: np.zeros(1000).sum(), warmup=2, repeats=20)
    assert 0 <= out["p50"] <= out["p95"] and out["repeats"] == 20
