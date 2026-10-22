"""cdm.serve_onnx: export a small EfficientNet-shaped model, check parity, time it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

from cdm.serve_onnx import export, latency_ms, parity, session, within  # noqa: E402


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
    path = export(model, tmp_path / "tiny.onnx", size=32)
    sess = session(path, threads=1)
    for n in (1, 5):
        check = parity(model, sess, torch.randn(n, 3, 32, 32))
        assert check["images"] == n
        assert within(check, {"logits_max_abs_diff": 1e-4, "top1_agreement": 1.0}) == []


def test_within_reports_each_broken_tolerance() -> None:
    check = {"logits_max_abs_diff": 0.01, "top1_agreement": 0.99}
    failed = within(check, {"logits_max_abs_diff": 1e-3, "top1_agreement": 1.0})
    assert len(failed) == 2 and failed[0].startswith("logits_max_abs_diff")


def test_latency_percentiles_are_ordered() -> None:
    out = latency_ms(lambda: np.zeros(1000).sum(), warmup=2, repeats=20)
    assert 0 <= out["p50"] <= out["p95"] and out["repeats"] == 20


def test_the_noise_rule_allows_flips_only_where_pytorch_flips_or_the_top_two_are_close() -> None:
    from cdm.serve_onnx import passes_against_noise

    noise = {"logits_max_abs_diff": 0.04, "top1_disagreements": [3]}
    margin = np.array([0.5, 0.005, 0.5, 0.5, 0.5])
    ok = {"logits_max_abs_diff": 0.07, "top1_disagreements": [1, 3]}
    assert passes_against_noise(ok, noise, margin) == []
    too_far = {"logits_max_abs_diff": 0.09, "top1_disagreements": []}
    assert "noise" in passes_against_noise(too_far, noise, margin)[0]
    bad_flip = {"logits_max_abs_diff": 0.01, "top1_disagreements": [0]}
    assert "[0]" in passes_against_noise(bad_flip, noise, margin)[0]


def test_noise_between_batch_sizes_is_measured_the_same_way(tmp_path: Path) -> None:
    from cdm.serve_onnx import gaps, onnx_outputs, torch_outputs

    model = Tiny().eval()
    x = torch.randn(6, 3, 32, 32)
    noise = gaps(torch_outputs(model, x, 1), torch_outputs(model, x, 4))
    assert noise["logits_max_abs_diff"] < 1e-4 and noise["top1_disagreements"] == []
    sess = session(export(model, tmp_path / "t.onnx", size=32), threads=1)
    assert gaps(torch_outputs(model, x, 4), onnx_outputs(sess, x, 4))["logits_max_abs_diff"] < 1e-4
