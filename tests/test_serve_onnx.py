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
