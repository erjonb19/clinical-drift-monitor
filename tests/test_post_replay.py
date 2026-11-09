"""Backlog item 1 after replay month 12: label-free signals on synthetic data (no real images)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from cdm import post_replay as pr
from cdm.monitor import ALERT_P as MONITOR_P
from cdm.splits import CLASSES

REPO = Path(__file__).resolve().parents[1]
K = len(CLASSES)


def batch(
    rng: np.random.Generator, n: int, shift: float = 0.0, mix: bool = False
) -> dict[str, Any]:
    features = rng.standard_normal((n, 16)) + shift
    probs = rng.dirichlet(np.ones(K) * (0.6 if not shift else 2.0), size=n)
    predicted = probs.argmax(1) if not mix else np.full(n, 3)
    return {
        "ood": rng.standard_normal(n) + shift,
        "features": features,
        "predicted": predicted.tolist(),
        "top_prob": probs.max(1),
    }


def test_new_ids_are_unseen_sorted_and_unique() -> None:
    assert pr.new_image_ids(["ISIC_3", "ISIC_1", "ISIC_3", "ISIC_2"], {"ISIC_2"}) == [
        "ISIC_1",
        "ISIC_3",
    ]
    assert pr.new_image_ids([], {"ISIC_1"}) == []


def test_no_shift_raises_no_alert() -> None:
    rng = np.random.default_rng(0)
    base, month = batch(rng, 3000), batch(rng, 300)
    signals = pr.label_free_signals(base, month, K)
    assert not pr.month_alert(signals), signals


def test_a_shifted_month_alerts_and_scores_point_the_right_way() -> None:
    rng = np.random.default_rng(1)
    base, month = batch(rng, 3000), batch(rng, 300, shift=1.5)
    signals = pr.label_free_signals(base, month, K)
    assert signals["ood"]["alert"] and signals["embedding"]["alert"]
    assert np.mean(month["ood"]) > np.mean(base["ood"])  # higher OOD score = further from training
    assert pr.month_alert(signals)


def test_a_prediction_mix_shift_alone_alerts() -> None:
    rng = np.random.default_rng(2)
    base, month = batch(rng, 3000), batch(rng, 300, mix=True)
    mix = pr.prediction_mix(base["predicted"], month["predicted"], K)
    assert mix["alert"] and mix["tvd"] >= pr.MIX_TVD
    assert abs(sum(mix["observed_share"]) - 1) < 1e-9


def test_a_confidence_shift_alerts_only_when_large_enough() -> None:
    rng = np.random.default_rng(3)
    base = rng.beta(5, 2, 3000)
    assert pr.confidence_drift(base, rng.beta(2, 5, 300))["alert"]
    assert not pr.confidence_drift(base, rng.beta(5, 2, 300))["alert"]


def test_too_few_images_are_counts_only() -> None:
    rng = np.random.default_rng(4)
    base, month = batch(rng, 3000), batch(rng, pr.MIN_IMAGES - 1, shift=3.0)
    signals = pr.label_free_signals(base, month, K)
    assert "counts only" in signals["note"] and not pr.month_alert(signals)


def test_status_is_explicit() -> None:
    assert pr.month_status(0) == {"status": "no_new_data", "new_images": 0}
    assert pr.month_status(12)["status"] == "scored"
    assert pr.month_status(12, "download failed")["status"] == "failed"


def test_embedding_statistics_come_from_the_baseline_only() -> None:
    """The month's own spread must not change the projection: a copy of the baseline as the
    month scores as no drift, whatever is appended to the month afterwards."""
    rng = np.random.default_rng(5)
    base = rng.standard_normal((2000, 8))
    same = pr.embedding_drift(base, base[:500])
    assert not same["alert"] and same["alpha"] == pr.ALERT_P / pr.EMBEDDING_COMPONENTS


def test_the_spec_matches_the_code() -> None:
    spec = json.loads(
        (REPO / "results" / "phase3" / "monthly" / "post_replay_spec.json").read_text()
    )
    t = spec["signals"]
    assert spec["committed_before_running"] is True
    assert t["ood"]["alert"] == "KS p < 0.01 and PSI >= 0.2" and pr.ALERT_P == MONITOR_P
    assert t["embedding"]["components"] == pr.EMBEDDING_COMPONENTS
    assert t["embedding"]["psi"] == pr.EMBEDDING_PSI
    assert t["prediction_mix"]["tvd"] == pr.MIX_TVD
    assert t["confidence"]["median_change"] == pr.CONFIDENCE_SHIFT
    assert spec["min_images"] == pr.MIN_IMAGES
    assert spec["statuses"] == [pr.SCORED, pr.NO_NEW_DATA, pr.FAILED]
