"""Gram-matrix detector and the Mahalanobis PCA-size methods: scores point the right way,
no labels at scoring, training data only for fitting."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cdm.ood import (
    GramDetector,
    gram_features,
    pca_size_by_selection,
    pca_size_for_variance,
    stage_maps,
)
from cdm.train import build_model, features_and_logits


def test_gram_features_match_a_hand_computation() -> None:
    f = torch.tensor([[[1.0, 2.0], [3.0, -1.0]]]).reshape(1, 2, 1, 2)  # 1 image, 2 channels
    (out,) = gram_features([f], powers=(1, 2))
    flat = f.double().flatten(2)[0]
    g1 = (flat @ flat.T).sum(1)
    g2 = ((flat**2) @ (flat**2).T).sum(1)
    expected = np.r_[g1.numpy(), np.sign(g2.numpy()) * np.abs(g2.numpy()) ** 0.5]
    np.testing.assert_allclose(out[0], expected)
    assert out.shape == (1, 4)


def test_stage_maps_reproduce_the_model_output() -> None:
    torch.manual_seed(0)
    model = build_model(pretrained=False).eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        feats, logits, maps = stage_maps(model, x)
        ref_feats, ref_logits = features_and_logits(model, x)
    torch.testing.assert_close(logits, ref_logits)
    torch.testing.assert_close(feats, ref_feats)
    assert len(maps) == 9


def _layers(rng: np.random.Generator, n: int, shift: float = 0.0) -> list[np.ndarray]:
    return [rng.normal(shift, 1.0, (n, 8)), rng.normal(shift, 1.0, (n, 4))]


def test_gram_scores_point_the_right_way_and_need_no_labels() -> None:
    rng = np.random.default_rng(0)
    det = GramDetector(n_classes=2)
    for _ in range(5):  # streamed in batches, like the training pass
        det.update(_layers(rng, 200), rng.integers(0, 2, 200))
    val = det.deviations(_layers(rng, 300), rng.integers(0, 2, 300))
    det.fit_normalizer(val)
    ind = det.score(det.deviations(_layers(rng, 300), rng.integers(0, 2, 300)))
    ood = det.score(det.deviations(_layers(rng, 300, shift=4.0), rng.integers(0, 2, 300)))
    assert ood.mean() > 5 * ind.mean()
    inside = [np.zeros((1, 8)), np.zeros((1, 4))]
    assert det.deviations(inside, np.array([0]))[0].sum() == 0  # within range: no deviation


def test_a_class_never_predicted_in_training_falls_back_to_all_classes() -> None:
    rng = np.random.default_rng(1)
    det = GramDetector(n_classes=3)
    det.update(_layers(rng, 100), rng.integers(0, 2, 100))  # class 2 never predicted
    devs = det.deviations(_layers(rng, 50), np.full(50, 2))
    assert np.isfinite(devs).all()


def test_scoring_before_fitting_fails_loudly() -> None:
    det = GramDetector(n_classes=2)
    with pytest.raises(RuntimeError):
        det.deviations([np.zeros((1, 2))], np.array([0]))


def test_pca_size_for_variance_uses_the_95_percent_rule() -> None:
    rng = np.random.default_rng(0)
    # 3 strong directions, then weak noise: 3 components carry well over 95%.
    base = rng.normal(0, 1, (500, 3)) @ rng.normal(0, 1, (3, 40)) * 10
    feats = base + rng.normal(0, 0.1, (500, 40))
    assert pca_size_for_variance(feats) == 3


def test_pca_selection_picks_a_candidate_and_reports_every_one() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, 600)
    train = rng.normal(0, 1, (600, 32)) + labels[:, None]
    val = rng.normal(0, 1, (200, 32)) + rng.integers(0, 3, 200)[:, None]
    far = rng.normal(5, 1, (200, 32))
    k, table = pca_size_by_selection(train, labels, val, {"far": far}, candidates=(4, 8, 16))
    assert k in (4, 8, 16) and set(table) == {4, 8, 16}
    assert table[k]["far"] > 0.9
