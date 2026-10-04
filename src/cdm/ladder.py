"""The Phase 2 tuning ladder's decision rules, fixed before any tuning result.

A change is kept if, on pooled validation, balanced accuracy rises by at least 1.0 point,
macro AUROC falls by at most 0.5 points and melanoma sensitivity falls by at most 2.0 points
against the reference. A batch tests several single changes against one reference, then
trains one model with every kept change, kept only if it passes the same rule against the
best single change. Colour constancy must also not shrink held-out-site drift: its drift
AUROC may fall by at most 2.0 points for any held-out site and drift score.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

MIN_GAIN = {"balanced_accuracy": 1.0, "macro_auroc": -0.5, "melanoma_sensitivity": -2.0}
MAX_DRIFT_DROP = 2.0
# Thresholds are inclusive; this absorbs floating-point error at the exact edges
# (100 * (0.9197 - 0.9247) is -0.50000000001, which is "at most 0.5 points").
TOLERANCE = 1e-9


def deltas(candidate: Mapping[str, float], reference: Mapping[str, float]) -> dict[str, float]:
    """Points (percentage) gained by ``candidate`` over ``reference`` for each kept metric."""
    return {m: 100.0 * (candidate[m] - reference[m]) for m in MIN_GAIN}


def passes(candidate: Mapping[str, float], reference: Mapping[str, float]) -> bool:
    d = deltas(candidate, reference)
    return all(d[m] >= limit - TOLERANCE for m, limit in MIN_GAIN.items())


def drift_preserved(
    candidate: Mapping[str, float], reference: Mapping[str, float]
) -> dict[str, Any]:
    """Drift AUROCs keyed "<site>/<score>"; fails if any falls by more than MAX_DRIFT_DROP."""
    drops = {k: 100.0 * (reference[k] - candidate[k]) for k in reference}
    return {
        "drops": drops,
        "preserved": all(v <= MAX_DRIFT_DROP + TOLERANCE for v in drops.values()),
    }


def merge(base: Mapping[str, Any], changes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply single changes on top of the base configuration; conflicting changes are refused."""
    merged = dict(base)
    seen: dict[str, Any] = {}
    for change in changes:
        for key, value in change.items():
            if key in seen and seen[key] != value:
                raise ValueError(f"changes disagree on {key}: {seen[key]!r} and {value!r}")
            seen[key] = value
            merged[key] = value
    return merged


def combination_needed(kept: Sequence[str]) -> bool:
    """A combined model is trained only when two or more single changes were kept."""
    return len(kept) >= 2


# The v1 vs v2 data decision (2026-10-04), fixed before any v2 result: on v1's exact
# validation images, mean over the centralized seeds, v2 is kept only if pooled balanced
# accuracy rises by at least 1.0 point and no client's falls by more than 3.0 points.
VERSION_MIN_GAIN = 1.0
VERSION_MAX_CLIENT_DROP = 3.0


def keep_data_version(
    v1: Sequence[Mapping[str, Any]], v2: Sequence[Mapping[str, Any]], clients: Sequence[str]
) -> dict[str, Any]:
    """``v1`` and ``v2`` are per-seed validation reports (``score_rows`` groups: ``pooled``
    and one per client), both computed on v1's validation images. Returns the means, the
    point changes and the decision."""

    def mean(runs: Sequence[Mapping[str, Any]], group: str) -> float:
        return float(np.mean([r[group]["balanced_accuracy"] for r in runs]))

    groups = ["pooled", *clients]
    before = {g: mean(v1, g) for g in groups}
    after = {g: mean(v2, g) for g in groups}
    change = {g: 100.0 * (after[g] - before[g]) for g in groups}
    gain_ok = change["pooled"] >= VERSION_MIN_GAIN - TOLERANCE
    worst = min(change[c] for c in clients)
    clients_ok = worst >= -VERSION_MAX_CLIENT_DROP - TOLERANCE
    return {
        "seeds": [len(v1), len(v2)],
        "v1_mean": before,
        "v2_mean": after,
        "change_points": change,
        "pooled_gain_ok": gain_ok,
        "worst_client_change_points": worst,
        "clients_ok": clients_ok,
        "keep_v2": gain_ok and clients_ok,
    }
