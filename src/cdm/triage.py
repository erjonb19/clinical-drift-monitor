"""Triage operating point (triage framing, not a clinical claim).

An image is referred when its melanoma probability is at or above a threshold. The threshold
is the highest one that reaches the target melanoma sensitivity on validation, so it refers as
few images as the target allows. It is fixed before any test image is scored.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def threshold_for_sensitivity(
    p_mel: Array, is_mel: NDArray[np.bool_], target: float = 0.95
) -> float:
    """The highest threshold t with sensitivity(p_mel >= t) >= target on these images."""
    positives = np.sort(p_mel[is_mel])[::-1]
    if len(positives) == 0:
        raise ValueError("no melanoma images to set a threshold on")
    k = math.ceil(target * len(positives))  # positives that must be referred
    return float(positives[k - 1])


def triage_metrics(p_mel: Array, is_mel: NDArray[np.bool_], threshold: float) -> dict[str, Any]:
    refer = p_mel >= threshold
    n_mel, n_other = int(is_mel.sum()), int((~is_mel).sum())
    return {
        "images": len(p_mel),
        "melanoma_images": n_mel,
        "sensitivity": float(refer[is_mel].mean()) if n_mel else None,
        "specificity": float((~refer[~is_mel]).mean()) if n_other else None,
        "referral_rate": float(refer.mean()) if len(p_mel) else None,
    }


def bootstrap_triage(
    p_mel: Array,
    is_mel: NDArray[np.bool_],
    lesions: Sequence[str],
    threshold: float,
    n: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """95% percentile intervals, resampling whole lesions with replacement."""
    ids = np.asarray(lesions, dtype=object)
    rows_by_lesion = [np.flatnonzero(ids == lesion) for lesion in np.unique(ids)]
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {"sensitivity": [], "specificity": [], "referral_rate": []}
    for _ in range(n):
        picked = rng.integers(0, len(rows_by_lesion), len(rows_by_lesion))
        rows = np.concatenate([rows_by_lesion[i] for i in picked])
        m = triage_metrics(p_mel[rows], is_mel[rows], threshold)
        for k in draws:
            if m[k] is not None:
                draws[k].append(m[k])
    return {"resamples": n, "seed": seed,
            **{k: {"low": float(np.percentile(v, 2.5)), "high": float(np.percentile(v, 97.5))}
               for k, v in draws.items() if v}}  # fmt: skip
