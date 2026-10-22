"""cdm.subgroups: group keys, intervals per group, drift per group, and the reproduction gate."""

from __future__ import annotations

import numpy as np
import pytest

from cdm.splits import CLASSES
from cdm.subgroups import age_band, check_reproduced, concept_group, drift, performance


@pytest.mark.parametrize(
    ("age", "band"),
    [(0, "0-29"), (29.0, "0-29"), (30, "30-44"), ("45", "45-59"), (74, "60-74"), (75, "75+"),
     (85.0, "75+"), (None, "unknown"), (float("nan"), "unknown"), ("", "unknown")],
)  # fmt: skip
def test_age_bands(age: object, band: str) -> None:
    assert age_band(age) == band


def test_concept_groups_keep_ids_and_treat_zero_as_unknown() -> None:
    assert concept_group(8507) == "concept:8507"
    assert concept_group(8532.0) == "concept:8532"
    assert concept_group(0) == "unknown"
    assert concept_group(None) == "unknown"


def test_performance_gives_intervals_only_to_groups_with_a_metric() -> None:
    rng = np.random.default_rng(0)
    labels = np.r_[np.repeat(np.arange(len(CLASSES)), 30), np.zeros(10, np.int64)]
    good = np.eye(len(CLASSES))[labels] * 5
    noisy = good + rng.normal(0, 3, good.shape)
    groups = ["big"] * 210 + ["small"] * 10
    units = [str(i) for i in range(len(labels))]
    out = performance(
        labels, {"a": good, "s0": noisy, "s1": noisy}, units, groups, ["s0", "s1"], n_boot=20
    )
    big, small = out["models"]["a"]
    assert big["group"] == "big" and big["balanced_accuracy"] == 1.0
    assert big["bootstrap_95"]["balanced_accuracy"] == {"low": 1.0, "high": 1.0}
    assert "bootstrap_95" not in small and "warning" in small
    assert set(out["probe_over_seeds"]) == {"big"}


def test_drift_by_group_points_the_right_way_and_hides_small_groups() -> None:
    reference = np.zeros(200)
    scores = np.r_[np.ones(150), np.zeros(50)]
    out = drift(reference, scores, ["far"] * 150 + ["near"] * 50)
    assert out["far"]["auroc"] == 1.0
    assert out["near"] == {"images": 50, "warning": "fewer than 100 images: count only"}
    with pytest.raises(ValueError, match="group values"):
        drift(reference, scores, ["far"])


def test_the_reproduction_gate_refuses_a_different_number() -> None:
    check_reproduced("x", 0.5, 0.5)
    with pytest.raises(RuntimeError, match="regenerated"):
        check_reproduced("x", 0.5, 0.5001)
