"""Triage threshold: the highest threshold meeting the target sensitivity on validation."""

from __future__ import annotations

import numpy as np
import pytest

from cdm.triage import bootstrap_triage, threshold_for_sensitivity, triage_metrics


def test_threshold_is_the_highest_meeting_the_target() -> None:
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.95, 0.01])
    is_mel = np.array([True] * 10 + [False, False])
    t = threshold_for_sensitivity(p, is_mel, target=0.9)  # 9 of 10 melanomas must be referred
    assert t == pytest.approx(0.1)
    assert triage_metrics(p, is_mel, t)["sensitivity"] >= 0.9
    assert triage_metrics(p, is_mel, t + 1e-9)["sensitivity"] < 0.9  # any higher misses it


def test_metrics_count_referrals_and_specificity() -> None:
    p = np.array([0.9, 0.2, 0.6, 0.1])
    is_mel = np.array([True, True, False, False])
    m = triage_metrics(p, is_mel, 0.5)
    assert (m["sensitivity"], m["specificity"], m["referral_rate"]) == (0.5, 0.5, 0.5)


def test_no_melanoma_cannot_set_a_threshold() -> None:
    with pytest.raises(ValueError):
        threshold_for_sensitivity(np.array([0.5]), np.array([False]))


def test_bootstrap_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(0)
    is_mel = rng.random(600) < 0.2
    p = np.clip(rng.normal(0.3, 0.15, 600) + 0.4 * is_mel, 0, 1)
    lesions = [f"L{i // 2}" for i in range(600)]
    t = threshold_for_sensitivity(p, is_mel)
    point = triage_metrics(p, is_mel, t)
    ci = bootstrap_triage(p, is_mel, lesions, t, n=200)
    for k in ("sensitivity", "specificity", "referral_rate"):
        assert ci[k]["low"] <= point[k] <= ci[k]["high"]
