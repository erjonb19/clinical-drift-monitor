"""Research Phase D: which images go on the review worksheet, and the error-analysis tables.

Pure functions over the committed predictions and saved drift scores; the Grad-CAM maps
themselves come from :mod:`cdm.explain` in the Databricks job.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cdm.eval import confusion
from cdm.splits import CLASSES

ENSEMBLE = "v2-ensemble-final-seeds0-1-2"
RANDOMIZATION_FLAG = 0.5


def probabilities(pred: pd.DataFrame, model: str = ENSEMBLE) -> NDArray[np.float64]:
    return pred[[f"{model}__{c}" for c in CLASSES]].to_numpy(np.float64)


def select(
    test: pd.DataFrame, ood_scores: NDArray[np.float64], per_class: int = 20, n_ood: int = 20
) -> pd.DataFrame:
    """The worksheet's images from the client test rows (``test``, in the drift run's order,
    with ``ood_scores`` row for row): the worst misses of each true class (lowest ensemble
    probability of the true class first) and the most out-of-distribution images. One row per
    image, with every reason it was chosen."""
    if len(test) != len(ood_scores):
        raise ValueError(f"{len(test)} rows, {len(ood_scores)} scores")
    probs = probabilities(test)
    labels = test["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    predicted = probs.argmax(1)
    p_true = probs[np.arange(len(test)), labels]
    reasons: dict[int, list[str]] = {}
    for c, name in enumerate(CLASSES):
        missed = np.flatnonzero((labels == c) & (predicted != c))
        for i in missed[np.argsort(p_true[missed], kind="stable")][:per_class]:
            reasons.setdefault(int(i), []).append(f"worst_miss:{name}")
    for i in np.argsort(-ood_scores, kind="stable")[:n_ood]:
        reasons.setdefault(int(i), []).append("most_ood")
    rows = sorted(reasons)
    out = test.iloc[rows][["isic_id", "source", "client", "label"]].reset_index(drop=True)
    out["reason"] = [";".join(reasons[i]) for i in rows]
    out["predicted"] = [CLASSES[k] for k in predicted[rows]]
    out["p_true"] = p_true[rows]
    out["p_predicted"] = probs[rows, predicted[rows]]
    out["ood_score"] = ood_scores[rows]
    return out


def confusion_by(pred: pd.DataFrame, column: str, min_n: int = 0) -> dict[str, Any]:
    """The ensemble's confusion matrix (rows true, columns predicted, in CLASSES order) per value
    of ``column``, over labelled rows; values under ``min_n`` images get their count only."""
    index = {c: i for i, c in enumerate(CLASSES)}
    out: dict[str, Any] = {"classes": list(CLASSES)}
    for value, group in pred[pred["label"].notna()].groupby(
        pred[column].fillna("not recorded"), sort=True
    ):
        labels: NDArray[np.int64] = np.asarray(group["label"].map(index), dtype=np.int64)
        out[str(value)] = (
            {"images": len(group), "matrix": confusion(labels, np.log(probabilities(group)))}
            if len(group) >= min_n
            else {"images": len(group), "warning": f"fewer than {min_n} images: count only"}
        )
    return out


def flagged(randomization: float) -> bool:
    """A map that barely changes when the weights are randomized is not interpreted."""
    return randomization >= RANDOMIZATION_FLAG
