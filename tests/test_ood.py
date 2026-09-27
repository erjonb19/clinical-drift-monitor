"""OOD scores must point the right way and be fit on training data only."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cdm.eval import auroc, fpr_at_95_tpr, summarize
from cdm.ood import MahalanobisDetector, energy_score, msp_score
from cdm.train import build_model, features_and_logits

N_CLASSES = 4
DIM = 32


def gaussian_classes(
    rng: np.random.Generator, n_per_class: int, shift: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Well-separated class clusters; ``shift`` moves every sample off the training manifold."""
    centers = np.eye(N_CLASSES, DIM) * 6.0
    labels = np.repeat(np.arange(N_CLASSES), n_per_class)
    feats = centers[labels] + rng.normal(size=(len(labels), DIM))
    feats[:, N_CLASSES:] += shift
    return feats, labels


def confident_logits(rng: np.random.Generator, n: int, margin: float) -> np.ndarray:
    logits = rng.normal(size=(n, N_CLASSES))
    logits[np.arange(n), rng.integers(0, N_CLASSES, n)] += margin
    return logits


def test_mahalanobis_scores_ood_higher() -> None:
    rng = np.random.default_rng(0)
    train_x, train_y = gaussian_classes(rng, 200)
    id_x, _ = gaussian_classes(rng, 100)
    ood_x, _ = gaussian_classes(rng, 100, shift=3.0)
    det = MahalanobisDetector(n_components=16).fit(train_x, train_y)
    assert auroc(det.score(id_x), det.score(ood_x)) > 0.95
    assert fpr_at_95_tpr(det.score(id_x), det.score(ood_x)) < 0.10


def test_mahalanobis_uses_nearest_class_not_true_class() -> None:
    """The notebook scored ID images against their true class and OOD against the nearest,
    which makes OOD look closer than ID. ``score`` takes no labels, and equals the minimum
    over classes computed by brute force."""
    rng = np.random.default_rng(1)
    train_x, train_y = gaussian_classes(rng, 200)
    det = MahalanobisDetector(n_components=16).fit(train_x, train_y)
    x, _ = gaussian_classes(rng, 20)
    assert det.pca is not None and det.means is not None and det.precision is not None
    z = det.pca.transform(x)
    brute = np.array(
        [min(float((zi - mu) @ det.precision @ (zi - mu)) for mu in det.means) for zi in z]
    )
    np.testing.assert_allclose(det.score(x), brute, rtol=1e-9)


def test_mahalanobis_statistics_do_not_change_when_scoring() -> None:
    """The notebook refit PCA on PathMNIST, putting ID and OOD in different spaces."""
    rng = np.random.default_rng(2)
    train_x, train_y = gaussian_classes(rng, 200)
    det = MahalanobisDetector(n_components=16).fit(train_x, train_y)
    assert det.pca is not None and det.means is not None
    components, means = det.pca.components_.copy(), det.means.copy()
    det.score(gaussian_classes(rng, 50, shift=5.0)[0])
    np.testing.assert_array_equal(det.pca.components_, components)
    np.testing.assert_array_equal(det.means, means)


def test_mahalanobis_refuses_to_score_before_fit() -> None:
    with pytest.raises(RuntimeError, match="fit the detector"):
        MahalanobisDetector().score(np.zeros((2, DIM)))


@pytest.mark.parametrize("score", [msp_score, energy_score])
def test_logit_scores_rank_uncertain_inputs_as_ood(score: object) -> None:
    rng = np.random.default_rng(3)
    id_logits = confident_logits(rng, 200, margin=8.0)
    ood_logits = confident_logits(rng, 200, margin=0.0)
    assert callable(score)
    assert auroc(score(id_logits), score(ood_logits)) > 0.95


def test_auroc_direction() -> None:
    low, high = np.array([0.1, 0.2, 0.3]), np.array([0.7, 0.8, 0.9])
    assert auroc(low, high) == 1.0
    assert auroc(high, low) == 0.0


def test_fpr95_counts_ood_below_the_id_threshold() -> None:
    id_scores = np.arange(100, dtype=np.float64)  # 95th percentile is 94.05
    ood_scores = np.array([50.0, 94.0, 95.0, 200.0])
    assert fpr_at_95_tpr(id_scores, ood_scores) == 0.5


def test_summarize_reports_mean_and_range() -> None:
    assert summarize([0.8, 0.9, 1.0]) == pytest.approx({"mean": 0.9, "min": 0.8, "max": 1.0})


def test_features_feed_the_classifier_head() -> None:
    """The features scored for OOD are exactly what the head classifies."""
    model = build_model(pretrained=False).eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        feats, logits = features_and_logits(model, x)
        assert feats.shape == (2, 1280)
        torch.testing.assert_close(logits, model(x))
