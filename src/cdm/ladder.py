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
