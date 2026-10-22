"""Detection and classification metrics, and aggregation over seeds.

Scores follow the convention in ``cdm.ood``: higher means more likely OOD, and OOD is the
positive class for AUROC.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score, roc_auc_score

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
    """Balanced accuracy, macro AUROC, melanoma sensitivity, accuracy, per-class recall and
    support. Melanoma sensitivity is None when no melanoma is present."""
    preds = logits.argmax(axis=1)
    recalls = recall_score(labels, preds, labels=range(len(CLASSES)), average=None, zero_division=0)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_auroc": macro_auroc(labels, logits),
        "melanoma_sensitivity": float(recalls[CLASSES.index("mel")])
        if np.any(labels == CLASSES.index("mel"))
        else None,
        "accuracy": float(np.mean(preds == labels)),
        "recall": {c: float(r) for c, r in zip(CLASSES, recalls, strict=True)},
        "support": {c: int(np.sum(labels == i)) for i, c in enumerate(CLASSES)},
    }


def softmax(logits: Array) -> Array:
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    return np.asarray(shifted / shifted.sum(axis=1, keepdims=True), dtype=np.float64)


def macro_auroc(labels: NDArray[np.int64], logits: Array) -> float | None:
    """Mean one-vs-rest AUROC over the classes present (each needs positives and negatives)."""
    probs = softmax(logits)
    scores = [
        float(roc_auc_score(labels == c, probs[:, c]))
        for c in range(len(CLASSES))
        if 0 < np.sum(labels == c) < len(labels)
    ]
    return float(np.mean(scores)) if scores else None


def confusion(labels: NDArray[np.int64], logits: Array) -> list[list[int]]:
    """Confusion matrix in CLASSES order: row = true class, column = predicted class."""
    matrix = confusion_matrix(labels, logits.argmax(axis=1), labels=list(range(len(CLASSES))))
    return [[int(v) for v in row] for row in matrix]


def summarize(values: Sequence[float]) -> dict[str, float]:
    """Mean and range over seeds."""
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max())}


MIN_GROUP = 100  # the repo's minimum for a reported metric (research Phase C)


def subgroup_table(
    labels: NDArray[np.int64],
    logits: Array,
    groups: Sequence[str],
    min_n: int = MIN_GROUP,
) -> list[dict[str, object]]:
    """Balanced accuracy and melanoma sensitivity per group (age band, sex, body site, skin
    type). A group with fewer than ``min_n`` images gets its count and a warning, no metric;
    melanoma sensitivity also needs ``min_n`` melanoma images. Intervals come from the caller,
    resampled by patient or lesion as in :func:`cdm.final.bootstrap_by_lesion`."""
    keys = np.asarray(groups, dtype=object)
    mel = CLASSES.index("mel")
    rows: list[dict[str, object]] = []
    for group in sorted(set(keys.tolist()), key=str):
        mask = keys == group
        n, n_mel = int(mask.sum()), int((labels[mask] == mel).sum())
        row: dict[str, object] = {"group": group, "images": n, "melanoma_images": n_mel}
        if n < min_n:
            row["warning"] = f"fewer than {min_n} images: count only"
            rows.append(row)
            continue
        metrics = classification_metrics(labels[mask], logits[mask])
        row["balanced_accuracy"] = metrics["balanced_accuracy"]
        if n_mel >= min_n:
            row["melanoma_sensitivity"] = metrics["melanoma_sensitivity"]
        else:
            row["warning"] = f"fewer than {min_n} melanoma images: no melanoma sensitivity"
        rows.append(row)
    return rows
