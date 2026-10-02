"""Post-training tuning rungs: changes to how a trained model's logits become predictions.

Rung 4 (test-time flip averaging) lives in ``cdm.ood.logits_with_flips``. This module holds:

- Rung 5, per-class thresholds: an additive offset per class on the log-probabilities,
  chosen to maximise balanced accuracy. Offsets are cross-fitted on validation: validation
  lesions are split into two halves, offsets fitted on one half score the other, so no
  image is scored with offsets that saw it. For test, offsets are fitted on all of
  validation once.
- Rung 6, lesion-level averaging: every image of a lesion gets the lesion's mean class
  probabilities. Images are still counted one by one, so the metrics stay comparable.

Every function returns logits (log-probabilities), so ``cdm.eval`` scores them unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from cdm.eval import softmax

Array = NDArray[np.float64]
GRID = np.round(np.arange(-3.0, 3.0001, 0.1), 2)


def log_probs(logits: Array) -> Array:
    return np.log(np.clip(softmax(logits), 1e-12, None))


def lesion_average(logits: Array, lesion_ids: Sequence[str]) -> Array:
    """Each image's log of its lesion's mean class probabilities (rung 6)."""
    probs = softmax(logits)
    ids = np.asarray(lesion_ids, dtype=object)
    out = np.empty_like(probs)
    for lesion in np.unique(ids):
        rows = ids == lesion
        out[rows] = probs[rows].mean(axis=0)
    return np.log(np.clip(out, 1e-12, None))


def _balanced_accuracy(labels: NDArray[np.int64], preds: NDArray[np.int64]) -> float:
    present = np.unique(labels)
    return float(np.mean([np.mean(preds[labels == c] == c) for c in present]))


def fit_offsets(
    logits: Array, labels: NDArray[np.int64], passes: int = 3, grid: Array = GRID
) -> Array:
    """Per-class offsets on log-probabilities that maximise balanced accuracy, by coordinate
    ascent over ``grid``. Ties keep the offset closest to zero; a class absent from
    ``labels`` keeps offset 0."""
    base = log_probs(logits)
    offsets = np.zeros(base.shape[1])
    order = np.argsort(np.abs(grid), kind="stable")  # try small offsets first
    present = set(np.unique(labels).tolist())
    for _ in range(passes):
        changed = False
        for c in range(base.shape[1]):
            if c not in present:
                continue
            best, best_value = -1.0, offsets[c]
            for value in grid[order]:
                trial = offsets.copy()
                trial[c] = value
                score = _balanced_accuracy(labels, (base + trial).argmax(axis=1))
                if score > best + 1e-12:
                    best, best_value = score, float(value)
            changed = changed or best_value != offsets[c]
            offsets[c] = best_value
        if not changed:
            break
    return offsets


def apply_offsets(logits: Array, offsets: Array) -> Array:
    return log_probs(logits) + offsets


def lesion_halves(lesion_ids: Sequence[str], seed: int = 0) -> NDArray[np.int64]:
    """0 or 1 for every image; all images of a lesion share a half."""
    ids = np.asarray(lesion_ids, dtype=object)
    lesions = np.unique(ids)
    rng = np.random.default_rng(seed)
    half = dict(zip(lesions, rng.permutation(len(lesions)) % 2, strict=True))
    return np.array([half[i] for i in ids], dtype=np.int64)


def cross_fit_offsets(
    logits: Array, labels: NDArray[np.int64], lesion_ids: Sequence[str], seed: int = 0
) -> tuple[Array, list[list[float]]]:
    """Rung 5 on validation: each half scored with offsets fitted on the other half.
    Returns the adjusted logits and the two offset vectors (fitted on half 0, then half 1)."""
    half = lesion_halves(lesion_ids, seed)
    out = np.empty_like(logits)
    fitted = []
    for h in (0, 1):
        offsets = fit_offsets(logits[half == h], labels[half == h])
        fitted.append([float(v) for v in offsets])
        out[half != h] = apply_offsets(logits[half != h], offsets)
    return out, fitted
