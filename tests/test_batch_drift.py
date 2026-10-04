"""Batch-level drift: false alarms near alpha when nothing shifted, detection when it did."""

from __future__ import annotations

import numpy as np

from cdm.batch_drift import simulate


def test_false_alarms_near_alpha_and_detection_grows_with_share_and_size() -> None:
    rng = np.random.default_rng(0)
    reference, in_dist = rng.normal(0, 1, 3000), rng.normal(0, 1, 3000)
    site = rng.normal(1.0, 1, 1500)
    out = simulate(reference, in_dist, {"new": site}, [25, 200], [0.0, 0.25, 1.0], 400, 0.01, 0)
    assert all(rate < 0.04 for rate in out["false_alarm_rate"].values())
    d = out["detection_rate"]["new"]
    assert d["200"]["1.0"] > 0.99 and d["200"]["1.0"] >= d["25"]["1.0"]
    assert d["200"]["1.0"] >= d["200"]["0.25"]
    assert "0.0" not in d["25"]  # 0% batches are reported only as false alarms


def test_same_seed_same_result() -> None:
    rng = np.random.default_rng(1)
    a, b, c = rng.normal(0, 1, 500), rng.normal(0, 1, 500), rng.normal(0.5, 1, 300)
    args = (a, b, {"s": c}, [25], [0.0, 0.5], 50, 0.01, 7)
    assert simulate(*args) == simulate(*args)
