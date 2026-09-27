"""Detection and classification metrics, and aggregation over seeds.

Scores follow the convention in ``cdm.ood``: higher means more likely OOD, and OOD is the
positive class for AUROC.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score

from cdm.data import CLASSES

Array = NDArray[np.float64]


def auroc(id_scores: Array, ood_scores: Array) -> float:
    y = np.r_[np.zeros(len(id_scores)), np.ones(len(ood_scores))]
    return float(roc_auc_score(y, np.r_[id_scores, ood_scores]))


def fpr_at_95_tpr(id_scores: Array, ood_scores: Array) -> float:
    """Share of OOD inputs accepted as in-distribution when 95% of in-distribution is kept.

    This is the convention of Liang et al. (2018) and Liu et al. (2020): in-distribution is
    the positive class for the 95% TPR, and lower is better.
    """
    threshold = np.quantile(id_scores, 0.95)
    return float(np.mean(ood_scores <= threshold))


def detection_metrics(id_scores: Array, ood_scores: Array) -> dict[str, float]:
    return {"auroc": auroc(id_scores, ood_scores), "fpr95": fpr_at_95_tpr(id_scores, ood_scores)}


def classification_metrics(labels: NDArray[np.int64], logits: Array) -> dict[str, object]:
    preds = logits.argmax(axis=1)
    recalls = recall_score(labels, preds, labels=range(len(CLASSES)), average=None, zero_division=0)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "accuracy": float(np.mean(preds == labels)),
        "recall": {c: float(r) for c, r in zip(CLASSES, recalls, strict=True)},
        "support": {c: int(np.sum(labels == i)) for i, c in enumerate(CLASSES)},
    }


def summarize(values: Sequence[float]) -> dict[str, float]:
    """Mean and range over seeds."""
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max())}
