"""Final reporting: ensemble, calibration with temperature scaling, bootstrap by lesion."""

from __future__ import annotations

import numpy as np

from cdm.eval import classification_metrics, softmax
from cdm.final import bootstrap_by_lesion, calibration, ece, ensemble_logits, fit_temperature


def _data(n: int = 2000, seed: int = 0, margin: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 7, n)
    logits = rng.normal(0, 1, (n, 7))
    logits[np.arange(n), labels] += margin
    return logits, labels


def test_ensemble_averages_probabilities() -> None:
    a = np.array([[2.0, 0.0]])
    b = np.array([[0.0, 2.0]])
    np.testing.assert_allclose(softmax(ensemble_logits([a, b])), [[0.5, 0.5]])
    np.testing.assert_allclose(softmax(ensemble_logits([a])), softmax(a))


def _calibrated(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Logits whose softmax is the true label distribution: calibrated at temperature 1."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 2, (n, 7))
    probs = softmax(logits)
    labels = np.array([rng.choice(7, p=p) for p in probs])
    return logits, labels


def test_temperature_recovers_overconfidence_and_lowers_ece() -> None:
    logits, labels = _calibrated(4000, seed=0)
    sharp = logits * 3.0  # the same model, three times too confident
    assert 2.5 < fit_temperature(sharp, labels) < 3.5
    val_logits, val_labels = _calibrated(4000, seed=1)
    out = calibration(val_logits * 3.0, val_labels, sharp, labels)
    assert out["ece_after"] < out["ece_before"] / 2
    assert out["nll_after"] < out["nll_before"]


def test_ece_is_zero_for_a_perfectly_confident_perfect_model() -> None:
    labels = np.array([0, 1, 2])
    assert ece(np.eye(3) * 50.0, labels) < 1e-9


def test_bootstrap_interval_brackets_the_estimate_and_resamples_whole_lesions() -> None:
    logits, labels = _data(n=600, margin=1.0)
    ids = [f"L{i // 3}" for i in range(600)]
    labels = np.repeat(labels[::3], 3)  # every image of a lesion shares its label
    out = bootstrap_by_lesion(logits, labels, ids, n=200)
    point = classification_metrics(labels, logits)
    for m in ("balanced_accuracy", "macro_auroc"):
        assert out[m]["low"] <= point[m] <= out[m]["high"]
        assert out[m]["high"] - out[m]["low"] < 0.2


def test_subgroups_below_the_minimum_get_counts_and_a_warning_only() -> None:
    from cdm.eval import subgroup_table

    rng = np.random.default_rng(0)
    labels = np.array([4] * 120 + [5] * 130 + [5] * 30, dtype=np.int64)  # mel, nv, nv
    logits = rng.normal(size=(280, 7))
    logits[np.arange(280), labels] += 3  # mostly right
    groups = ["60-69"] * 250 + ["90+"] * 30
    rows = {r["group"]: r for r in subgroup_table(labels, logits, groups)}
    assert rows["90+"] == {"group": "90+", "images": 30, "melanoma_images": 0,
                           "warning": "fewer than 100 images: count only"}  # fmt: skip
    assert rows["60-69"]["images"] == 250 and rows["60-69"]["melanoma_images"] == 120
    assert 0 <= float(rows["60-69"]["balanced_accuracy"]) <= 1  # type: ignore[arg-type]
    assert "melanoma_sensitivity" in rows["60-69"]
    # Groups partition the data.
    assert sum(int(r["images"]) for r in rows.values()) == 280  # type: ignore[call-overload]


def test_missing_group_values_become_unknown_and_nothing_is_dropped() -> None:
    from cdm.eval import subgroup_table

    labels = np.array([5] * 4, dtype=np.int64)
    groups: list[object] = ["a", None, float("nan"), "a"]
    rows = subgroup_table(labels, np.zeros((4, 7)), groups, min_n=1)  # type: ignore[arg-type]
    assert {r["group"]: r["images"] for r in rows} == {"a": 2, "unknown": 2}
