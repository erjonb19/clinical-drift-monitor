"""Phase 3: the shipped EfficientNet as ONNX, with a parity check against PyTorch and a CPU
latency benchmark.

Each seed's network is exported with two outputs, the 1,280-d penultimate features (for the
drift score) and the logits, and a dynamic batch size, so the API (one image) and the monthly
job (batches) use the same file. The ensemble's probabilities are the mean of the three seeds'
softmax, computed by the caller, as in Phase 2.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from cdm.train import features_and_logits

Array = NDArray[np.float64]
OUTPUTS = ("features", "logits")


class FeaturesAndLogits(nn.Module):
    """The classifier with its penultimate features exposed as a second output."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return features_and_logits(self.model, image)


def export(model: nn.Module, path: Path, size: int = 224) -> Path:
    """Write ``model`` (in eval mode) as ONNX with a dynamic batch dimension."""
    wrapped = FeaturesAndLogits(model.eval()).eval()
    example = torch.zeros(2, 3, size, size)
    batch = torch.export.Dim("batch", min=1, max=1024)
    program = torch.onnx.export(
        wrapped,
        (example,),
        dynamo=True,
        input_names=["image"],
        output_names=list(OUTPUTS),
        dynamic_shapes={"image": {0: batch}},
        verbose=False,
    )
    if program is None:
        raise RuntimeError("torch.onnx.export returned no program")
    program.save(str(path))
    return path


def session(path: Path, threads: int | None = None) -> Any:
    """An ONNX Runtime CPU session (imported here: onnxruntime is the serve extra only)."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    if threads is not None:
        options.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def softmax(logits: Array) -> Array:
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    return np.asarray(shifted / shifted.sum(axis=1, keepdims=True), dtype=np.float64)


@torch.no_grad()
def parity(
    model: nn.Module, onnx_session: Any, images: torch.Tensor, batch: int = 32
) -> dict[str, float]:
    """How far ONNX Runtime's outputs are from PyTorch's on the same inputs, both run in
    batches of ``batch`` (one large batch ran a serverless machine out of memory)."""
    model.eval()
    parts_t, parts_o = [], []
    for start in range(0, len(images), batch):
        x = images[start : start + batch]
        parts_t.append([v.double().numpy() for v in features_and_logits(model, x)])
        parts_o.append(onnx_session.run(list(OUTPUTS), {"image": x.numpy()}))
    feats_t, logits_t = (np.concatenate([p[i] for p in parts_t]) for i in range(2))
    feats_o, logits_o = (
        np.concatenate([np.asarray(p[i], dtype=np.float64) for p in parts_o]) for i in range(2)
    )
    scale = np.abs(feats_t).max()
    return {
        "images": int(len(images)),
        "logits_max_abs_diff": float(np.abs(logits_t - logits_o).max()),
        "probabilities_max_abs_diff": float(np.abs(softmax(logits_t) - softmax(logits_o)).max()),
        "features_max_abs_diff_relative": float(np.abs(feats_t - feats_o).max() / scale),
        "top1_agreement": float(np.mean(logits_t.argmax(1) == logits_o.argmax(1))),
    }


def within(check: dict[str, float], tolerance: dict[str, float]) -> list[str]:
    """The tolerance keys a parity check fails (``top1_agreement`` is a floor, others ceilings)."""
    failed = []
    for key, limit in tolerance.items():
        value = check[key]
        bad = value < limit if key == "top1_agreement" else value > limit
        if bad:
            failed.append(f"{key} {value:.3g} (limit {limit:g})")
    return failed


def latency_ms(run: Callable[[], object], warmup: int = 50, repeats: int = 300) -> dict[str, float]:
    """Wall-clock p50 and p95 of ``run`` in milliseconds, after ``warmup`` untimed calls."""
    for _ in range(warmup):
        run()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        times.append(1000 * (time.perf_counter() - start))
    arr = np.asarray(times)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "repeats": repeats,
        "warmup": warmup,
    }
