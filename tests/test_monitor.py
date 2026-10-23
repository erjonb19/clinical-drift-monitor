"""cdm.monitor: the replay never reuses an image, the drift tests point the right way, small
skin-type groups never alert, and accuracy waits for labels."""

from __future__ import annotations

import numpy as np
import pytest

from cdm.monitor import by_skin_type, delayed_accuracy, draw_schedule, drift_test, psi
from cdm.splits import CLASSES


def test_schedule_draws_without_replacement_in_month_order() -> None:
    pools = {"a": [f"a{i}" for i in range(10)], "b": [f"b{i}" for i in range(5)]}
    months = draw_schedule(pools, {"1": {"a": 4}, "2": {"a": 4, "b": 2}, "10": {"b": 3}})
    assert sorted(months) == [1, 2, 10]
    every = [i for batch in months.values() for _, i in batch]
    assert len(every) == len(set(every)) == 13
    assert draw_schedule(pools, {"1": {"a": 4}}) == {1: months[1]}  # fixed by the seed
    with pytest.raises(ValueError, match="too few"):
        draw_schedule(pools, {"1": {"b": 6}})


def test_psi_is_zero_for_the_same_distribution_and_large_for_a_shift() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=5000)
    assert psi(base, rng.normal(size=5000)) < 0.02
    assert psi(base, rng.normal(2, 1, size=5000)) > 1.0


def test_alert_needs_both_ks_and_psi() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(size=3000)
    same = drift_test(base, rng.normal(size=300))
    shifted = drift_test(base, rng.normal(1.5, 1, size=300))
    assert not same["alert"] and not same["warning"]
    assert shifted["alert"] and shifted["ks_p"] < 0.01 and shifted["psi"] >= 0.2


def test_small_or_unrecorded_skin_types_never_alert() -> None:
    rng = np.random.default_rng(2)
    base = rng.normal(size=3000)
    scores = np.r_[rng.normal(3, 1, 150), rng.normal(3, 1, 40), rng.normal(3, 1, 200)]
    types = ["II"] * 150 + ["V"] * 40 + [None] * 200
    out = by_skin_type(base, scores, types)
    assert out["II"]["alert"] is True
    assert out["V"]["alert"] is False and out["V"]["images"] == 40
    assert out["not recorded"]["alert"] is False


def test_delayed_accuracy_uses_labelled_images_and_the_melanoma_minimum() -> None:
    labels = np.array([0, 1, -1, CLASSES.index("mel")])
    probs = np.eye(len(CLASSES))[[0, 1, 2, CLASSES.index("mel")]].astype(float)
    out = delayed_accuracy(labels, probs)
    assert out["labelled"] == 3 and out["balanced_accuracy"] == 1.0
    assert "melanoma_sensitivity" not in out and out["melanoma_images"] == 1
