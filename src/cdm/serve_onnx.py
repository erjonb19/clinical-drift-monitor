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


@torch.no_grad()
def torch_outputs(model: nn.Module, images: torch.Tensor, batch: int) -> tuple[Array, Array]:
    """PyTorch's features and logits, run in batches of ``batch``."""
    model.eval()
    parts = [
        [v.double().numpy() for v in features_and_logits(model, images[i : i + batch])]
        for i in range(0, len(images), batch)
    ]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def onnx_outputs(onnx_session: Any, images: torch.Tensor, batch: int) -> tuple[Array, Array]:
    """ONNX Runtime's features and logits, run in batches of ``batch``."""
    parts = [
        onnx_session.run(list(OUTPUTS), {"image": images[i : i + batch].numpy()})
        for i in range(0, len(images), batch)
    ]
    feats = np.concatenate([np.asarray(p[0], dtype=np.float64) for p in parts])
    logits = np.concatenate([np.asarray(p[1], dtype=np.float64) for p in parts])
    return feats, logits


def gaps(a: tuple[Array, Array], b: tuple[Array, Array]) -> dict[str, Any]:
    """How far two (features, logits) results are apart, and the images whose top class
    differs between them."""
    (fa, la), (fb, lb) = a, b
    return {
        "logits_max_abs_diff": float(np.abs(la - lb).max()),
        "probabilities_max_abs_diff": float(np.abs(softmax(la) - softmax(lb)).max()),
        "features_max_abs_diff_relative": float(np.abs(fa - fb).max() / np.abs(fa).max()),
        "top1_disagreements": [int(i) for i in np.flatnonzero(la.argmax(1) != lb.argmax(1))],
    }


def top2_margin(logits: Array) -> Array:
    """Gap between the two highest class probabilities, per image."""
    p = np.sort(softmax(logits), axis=1)
    return np.asarray(p[:, -1] - p[:, -2], dtype=np.float64)


SEED_KEYS = (
    "logits_max_abs_diff",
    "probabilities_max_abs_diff",
    "features_max_abs_diff_relative",
)
ENSEMBLE_KEYS = ("probabilities_max_abs_diff",)


def passes_against_noise(
    onnx_gap: dict[str, Any],
    noise: dict[str, Any],
    margin: Array,
    factor: float = 2.0,
    close: float = 0.01,
    keys: tuple[str, ...] = SEED_KEYS,
) -> list[str]:
    """The amended rule (owner's decision, 2026-10-22): every ONNX gap at most ``factor`` times
    PyTorch's own batch-size noise on the same images, and a top-class disagreement only where
    PyTorch also disagrees with itself or the top two classes are within ``close``."""
    failed = []
    for key in keys:
        if key not in noise or key not in onnx_gap:
            raise KeyError(f"{key} missing: every listed gap must be measured")
        if onnx_gap[key] > factor * noise[key]:
            failed.append(f"{key} {onnx_gap[key]:.3g} > {factor:g} x noise {noise[key]:.3g}")
    allowed = set(noise["top1_disagreements"]) | {int(i) for i in np.flatnonzero(margin < close)}
    extra = sorted(set(onnx_gap["top1_disagreements"]) - allowed)
    if extra:
        failed.append(f"top-class disagreements outside the allowed images: {extra}")
    return failed
