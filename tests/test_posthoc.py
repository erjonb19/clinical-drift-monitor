"""Post-training rungs 5 and 6: per-class thresholds (cross-fitted by lesion) and
lesion-level averaging."""

from __future__ import annotations

import numpy as np

from cdm.eval import classification_metrics, softmax
from cdm.posthoc import (
    apply_offsets,
    cross_fit_offsets,
    fit_offsets,
    lesion_average,
    lesion_halves,
    log_probs,
)


def bacc(labels: np.ndarray, logits: np.ndarray) -> float:
    return float(classification_metrics(labels, logits)["balanced_accuracy"])  # type: ignore[arg-type]


def test_lesion_average_shares_probabilities_within_a_lesion_only() -> None:
    logits = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 0.0]])
    out = softmax(lesion_average(logits, ["a", "a", "b"]))
    np.testing.assert_allclose(out[0], out[1])
    np.testing.assert_allclose(out[0], [0.5, 0.5])
    np.testing.assert_allclose(out[2], softmax(logits)[2])  # a single-image lesion is unchanged


def test_offsets_fix_a_class_the_model_under_predicts() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, 600)
    logits = rng.normal(0, 1, (600, 3))
    logits[np.arange(600), labels] += 2.0
    logits[:, 2] -= 3.0  # class 2 is almost never predicted
    before = bacc(labels, logits)
    offsets = fit_offsets(logits, labels)
    after = bacc(labels, apply_offsets(logits, offsets))
    assert offsets[2] - offsets[[0, 1]].max() > 1.5
    assert after > before + 0.2


def test_offsets_stay_zero_when_nothing_helps_and_for_absent_classes() -> None:
    labels = np.array([0, 1, 0, 1])
    logits = np.array([[3.0, 0, 0], [0, 3.0, 0], [3.0, 0, 0], [0, 3.0, 0]])
    np.testing.assert_array_equal(fit_offsets(logits, labels), np.zeros(3))


def test_halves_never_split_a_lesion() -> None:
    ids = [f"L{i // 3}" for i in range(300)]
    half = lesion_halves(ids, seed=0)
    for lesion in set(ids):
        assert len({half[i] for i, x in enumerate(ids) if x == lesion}) == 1
    assert 0.3 < half.mean() < 0.7


def test_cross_fitting_scores_each_half_with_the_other_halfs_offsets() -> None:
    rng = np.random.default_rng(1)
    ids = [f"L{i // 2}" for i in range(400)]
    labels = rng.integers(0, 3, 400)
    logits = rng.normal(0, 1, (400, 3))
    logits[np.arange(400), labels] += 1.5
    out, fitted = cross_fit_offsets(logits, labels, ids, seed=0)
    half = lesion_halves(ids, seed=0)
    np.testing.assert_allclose(out[half == 1], log_probs(logits[half == 1]) + np.array(fitted[0]))
    np.testing.assert_allclose(out[half == 0], log_probs(logits[half == 0]) + np.array(fitted[1]))
