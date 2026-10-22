"""cdm.review: worksheet selection and the confusion tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cdm.review import ENSEMBLE, confusion_by, flagged, select
from cdm.splits import CLASSES


def frame(labels: list[str], predicted: list[str], confidence: list[float]) -> pd.DataFrame:
    rows = []
    for i, (label, pred, conf) in enumerate(zip(labels, predicted, confidence, strict=True)):
        probs = {c: (1 - conf) / (len(CLASSES) - 1) for c in CLASSES}
        probs[pred] = conf
        rows.append({"isic_id": f"I{i}", "source": "s", "client": "x", "label": label}
                    | {f"{ENSEMBLE}__{c}": p for c, p in probs.items()})  # fmt: skip
    return pd.DataFrame(rows)


def test_worst_misses_come_first_and_correct_images_are_never_misses() -> None:
    test = frame(["mel", "mel", "mel", "nv"], ["nv", "nv", "mel", "nv"], [0.6, 0.9, 0.9, 0.9])
    out = select(test, np.zeros(4), per_class=1, n_ood=0)
    assert out["isic_id"].tolist() == ["I1"]  # mel missed with the lowest mel probability
    assert out["reason"].tolist() == ["worst_miss:mel"]
    assert out["predicted"].tolist() == ["nv"]


def test_an_image_chosen_twice_appears_once_with_both_reasons() -> None:
    test = frame(["mel", "nv"], ["nv", "nv"], [0.9, 0.9])
    out = select(test, np.array([5.0, 1.0]), per_class=20, n_ood=1)
    assert out["reason"].tolist() == ["worst_miss:mel;most_ood"]
    with pytest.raises(ValueError, match="scores"):
        select(test, np.zeros(3))


def test_confusion_counts_rows_by_true_class_and_hides_small_groups() -> None:
    pred = frame(["mel", "nv", "nv"], ["nv", "nv", "nv"], [0.9, 0.9, 0.9])
    pred["skin_type"] = ["II", "II", None]
    out = confusion_by(pred, "skin_type", min_n=2)
    m = np.asarray(out["II"]["matrix"])
    assert m[CLASSES.index("mel"), CLASSES.index("nv")] == 1
    assert m[CLASSES.index("nv"), CLASSES.index("nv")] == 1
    assert out["not recorded"] == {"images": 1, "warning": "fewer than 2 images: count only"}


def test_randomization_flag() -> None:
    assert flagged(0.5) and not flagged(0.49)
