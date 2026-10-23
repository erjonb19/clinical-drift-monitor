"""Phase 3 monthly drift monitoring: the replay schedule, the drift tests and the alert rule,
per-skin-type alerts and the delayed accuracy check (``results/phase3/monthly/spec.json``).

The replay uses real images only: each month is drawn, without replacement across the whole
schedule, from real pools (the clients' test split and the two new hospitals) in a fixed order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import ks_2samp

from cdm.eval import MIN_GROUP
from cdm.serving import balanced_accuracy
from cdm.splits import CLASSES

Array = NDArray[np.float64]
ALERT_P, ALERT_PSI, WARN_PSI = 0.01, 0.2, 0.1


def draw_schedule(
    pools: Mapping[str, Sequence[str]],
    schedule: Mapping[str, Mapping[str, int]],
    seed: int = 0,
) -> dict[int, list[tuple[str, str]]]:
    """Month -> (pool, image ID) pairs. Each pool is shuffled once and handed out in order, so no
    image is used twice in the schedule."""
    rng = np.random.default_rng(seed)
    order = {
        name: [ids[i] for i in rng.permutation(len(ids))] for name, ids in sorted(pools.items())
    }
    used = {name: 0 for name in order}
    months: dict[int, list[tuple[str, str]]] = {}
    for month in sorted(schedule, key=int):
        batch = []
        for name, n in sorted(schedule[month].items()):
            if used[name] + n > len(order[name]):
                raise ValueError(f"pool {name} has too few images for month {month}")
            batch += [(name, i) for i in order[name][used[name] : used[name] + n]]
            used[name] += n
        months[int(month)] = batch
    return months


def psi(baseline: Array, current: Array, bins: int = 10, eps: float = 1e-6) -> float:
    """Population stability index over bins at the baseline's quantiles."""
    edges = np.quantile(baseline, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(baseline, edges)[0] / len(baseline) + eps
    actual = np.histogram(current, edges)[0] / len(current) + eps
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def drift_test(baseline: Array, current: Array) -> dict[str, Any]:
    """KS and PSI against the baseline, and the pre-registered alert and warning flags."""
    ks = ks_2samp(current, baseline)
    value = psi(baseline, current)
    alert = bool(ks.pvalue < ALERT_P and value >= ALERT_PSI)
    return {
        "images": int(len(current)),
        "ks_statistic": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "psi": value,
        "alert": alert,
        "warning": bool(not alert and value >= WARN_PSI),
    }


def by_skin_type(
    baseline: Array, scores: Array, types: Sequence[str | None], min_n: int = MIN_GROUP
) -> dict[str, Any]:
    """The drift test per Fitzpatrick type with at least ``min_n`` images this month; smaller
    types and unrecorded skin type get counts only and never alert."""
    keys = np.asarray([t if t else "not recorded" for t in types], dtype=object)
    out: dict[str, Any] = {}
    for t in sorted(set(keys.tolist())):
        mask = keys == t
        n = int(mask.sum())
        if t != "not recorded" and n >= min_n:
            out[t] = drift_test(baseline, scores[mask])
        else:
            out[t] = {
                "images": n,
                "alert": False,
                "note": f"counts only (under {min_n} images or not recorded)",
            }
    return out


def delayed_accuracy(
    labels: NDArray[np.int64], probs: Array, min_mel: int = MIN_GROUP
) -> dict[str, Any]:
    """A month's accuracy once its labels have arrived (labelled images only)."""
    keep = labels >= 0
    y, p = labels[keep], probs[keep].argmax(axis=1)
    mel = CLASSES.index("mel")
    n_mel = int((y == mel).sum())
    out: dict[str, Any] = {"labelled": int(keep.sum()), "melanoma_images": n_mel,
                           "balanced_accuracy": balanced_accuracy(y, p)}  # fmt: skip
    if n_mel >= min_mel:
        out["melanoma_sensitivity"] = float(np.mean(p[y == mel] == mel))
    return out
