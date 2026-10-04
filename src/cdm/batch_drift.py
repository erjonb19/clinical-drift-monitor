"""Batch-level drift detection: how often a two-sample KS test on detector scores flags a
batch that mixes in images from a new site (spec: results/phase2/batch_drift/spec.json)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import ks_2samp

Array = NDArray[np.float64]


def flag_rate(
    reference: Array,
    in_dist: Array,
    site: Array,
    size: int,
    share: float,
    batches: int,
    alpha: float,
    rng: np.random.Generator,
) -> float:
    """Share of random batches (``round(share * size)`` site images, the rest in-distribution,
    each without replacement within a batch) whose KS p-value against ``reference`` is below
    ``alpha``."""
    n_site = round(share * size)
    flagged = 0
    for _ in range(batches):
        batch = np.concatenate([
            rng.choice(site, n_site, replace=False),
            rng.choice(in_dist, size - n_site, replace=False),
        ])  # fmt: skip
        flagged += ks_2samp(batch, reference).pvalue < alpha
    return flagged / batches


def simulate(
    reference: Array,
    in_dist: Array,
    sites: dict[str, Array],
    sizes: Sequence[int],
    shares: Sequence[float],
    batches: int,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    """False-alarm rate (0% batches) per size, and detection rate per site, size and share."""
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {"false_alarm_rate": {}, "detection_rate": {}}
    for size in sizes:
        out["false_alarm_rate"][str(size)] = flag_rate(
            reference, in_dist, in_dist[:0], size, 0.0, batches, alpha, rng
        )
    for name, site in sites.items():
        out["detection_rate"][name] = {
            str(size): {str(share): flag_rate(reference, in_dist, site, size, share, batches,
                                              alpha, rng)
                        for share in shares if share > 0}
            for size in sizes
        }  # fmt: skip
    return out
