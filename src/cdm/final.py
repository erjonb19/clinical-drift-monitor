"""Final Phase 2 reporting: the 3-seed ensemble, bootstrap intervals by lesion, and
calibration before and after temperature scaling.

Temperature is fitted on validation and applied to test. It divides the logits by one
number, so it never changes a prediction: only confidence, and so calibration, moves.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cdm.eval import classification_metrics, softmax

Array = NDArray[np.float64]
BOOTSTRAP_METRICS = ("balanced_accuracy", "macro_auroc", "melanoma_sensitivity")
TEMPERATURES = np.exp(np.linspace(np.log(0.05), np.log(20.0), 2001))


def ensemble_logits(members: Sequence[Array]) -> Array:
    """Log of the mean class probabilities over ensemble members."""
    probs = np.mean(np.stack([softmax(m) for m in members]), axis=0)
    return np.asarray(np.log(np.clip(probs, 1e-12, None)), dtype=np.float64)


def nll(logits: Array, labels: NDArray[np.int64]) -> float:
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-np.mean(log_probs[np.arange(len(labels)), labels]))


def ece(logits: Array, labels: NDArray[np.int64], bins: int = 15) -> float:
    """Expected calibration error of the top-label confidence, equal-width bins."""
    probs = softmax(logits)
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (confidence > lo) & (confidence <= hi)
        if in_bin.any():
            total += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(total)


def fit_temperature(logits: Array, labels: NDArray[np.int64]) -> float:
    """The temperature (searched on a log grid from 0.05 to 20) minimising NLL."""
    losses = [nll(logits / t, labels) for t in TEMPERATURES]
    return float(TEMPERATURES[int(np.argmin(losses))])


def calibration(
    val_logits: Array,
    val_labels: NDArray[np.int64],
    test_logits: Array,
    test_labels: NDArray[np.int64],
) -> dict[str, float]:
    t = fit_temperature(val_logits, val_labels)
    return {
        "temperature_fitted_on_val": t,
        "ece_before": ece(test_logits, test_labels),
        "ece_after": ece(test_logits / t, test_labels),
        "nll_before": nll(test_logits, test_labels),
        "nll_after": nll(test_logits / t, test_labels),
    }


def bootstrap_by_lesion(
    logits: Array,
    labels: NDArray[np.int64],
    lesion_ids: Sequence[str],
    n: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """95% percentile intervals from resampling lesions with replacement (every image of a
    drawn lesion comes along), so images of one lesion are never treated as independent."""
    ids = np.asarray(lesion_ids, dtype=object)
    rows_by_lesion = [np.flatnonzero(ids == lesion) for lesion in np.unique(ids)]
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {m: [] for m in BOOTSTRAP_METRICS}
    for _ in range(n):
        picked = rng.integers(0, len(rows_by_lesion), len(rows_by_lesion))
        rows = np.concatenate([rows_by_lesion[i] for i in picked])
        metrics = classification_metrics(labels[rows], logits[rows])
        for m in BOOTSTRAP_METRICS:
            value = metrics[m]
            if value is not None:
                draws[m].append(float(value))  # type: ignore[arg-type]
    return {
        "resamples": n,
        "seed": seed,
        **{
            m: {"low": float(np.percentile(v, 2.5)), "high": float(np.percentile(v, 97.5))}
            for m, v in draws.items()
            if v
        },
    }
