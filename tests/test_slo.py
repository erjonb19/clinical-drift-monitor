"""Backlog items 4 and 5: model targets, the pooled melanoma window, and the alert path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cdm import slo
from cdm.monitor import delayed_accuracy
from cdm.splits import CLASSES

REPO = Path(__file__).resolve().parents[1]
CONFIG = slo.load_config(REPO / "config" / "slo.json")


def month(m: int, mel: int, referred: int | None) -> dict[str, float]:
    r: dict[str, float] = {"month": m, "melanoma_images": mel, "balanced_accuracy": 0.6}
    if referred is not None:
        r["melanoma_referred"] = referred
    return r


def test_wilson_matches_a_known_interval() -> None:
    low, high = slo.wilson(81, 100)  # 81/100: 95% Wilson interval about 0.722 to 0.874
    assert low == pytest.approx(0.7222, abs=1e-3) and high == pytest.approx(0.8744, abs=1e-3)
    assert slo.wilson(0, 10)[0] == 0.0 and slo.wilson(10, 10)[1] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        slo.wilson(5, 0)


def test_the_window_grows_back_from_the_latest_month_until_100_melanomas() -> None:
    months = [month(1, 60, 57), month(2, 40, 38), month(3, 45, 40), month(4, 30, 25)]
    w = slo.melanoma_window(months, 100)
    assert w["months"] == [2, 3, 4] and w["melanoma_images"] == 115
    assert w["melanoma_referred"] == 103 and w["sensitivity"] == pytest.approx(103 / 115)


def test_too_few_melanomas_is_insufficient_not_met_or_broken() -> None:
    w = slo.melanoma_window([month(1, 40, 40), month(2, 30, 30)], 100)
    assert w["status"] == slo.INSUFFICIENT
    assert slo.judge_sensitivity(w, 0.9) == slo.INSUFFICIENT


def test_months_without_the_referral_count_cannot_join() -> None:
    w = slo.melanoma_window([month(1, 200, None)], 100)
    assert w["status"] == slo.INSUFFICIENT and "no melanoma_referred" in w["reason"]


def test_judged_on_the_point_estimate() -> None:
    def window(hits: int, n: int) -> dict[str, Any]:
        return slo.melanoma_window([month(1, n, hits)], 100)

    assert slo.judge_sensitivity(window(90, 100), 0.9) == slo.MET
    assert slo.judge_sensitivity(window(89, 100), 0.9) == slo.BROKEN
    assert window(89, 100)["wilson_95"][1] > 0.9  # reported beside it, not used to judge


def test_delayed_accuracy_counts_referred_melanomas() -> None:
    mel = CLASSES.index("mel")
    labels = np.array([mel, mel, mel, 0, -1], dtype=np.int64)
    probs = np.full((5, len(CLASSES)), 0.01)
    probs[0, mel], probs[1, mel], probs[2, 0], probs[3, 0], probs[4, mel] = 0.9, 0.07, 0.9, 0.9, 0.9
    out = delayed_accuracy(labels, probs, refer_threshold=0.0638)
    assert out["melanoma_images"] == 3 and out["melanoma_referred"] == 2
    assert out["melanoma_correct"] == 2  # argmax melanoma for images 0 and 1
    assert "melanoma_referred" not in delayed_accuracy(labels, probs)


def test_alerts_name_each_broken_target_and_each_drift_alert() -> None:
    record = {
        "month": 9,
        "abstain_share": 0.3,
        "drift": {"alert": True},
        "by_skin_type": {"II": {"alert": True}, "I": {"alert": False}},
        "delayed_accuracy_of_month": {"month": 8, "balanced_accuracy": 0.7},
    }
    reasons = slo.alerts(record, slo.evaluate(record, [], CONFIG))
    assert reasons == [
        "target broken: abstain_share",
        "drift alert (overall)",
        "drift alert (skin type II)",
    ]
    with pytest.raises(
        slo.AlertRaised, match="ALERT \\(not a job error\\), month 9: target broken"
    ):
        slo.raise_if_alerts({8: [], 9: reasons})
    slo.raise_if_alerts({8: [], 9: []})  # no alert, no exception


def test_a_backfill_alerts_once_at_the_end_naming_every_month() -> None:
    with pytest.raises(slo.AlertRaised) as err:
        slo.raise_if_alerts({9: ["drift alert (overall)"], 10: ["target broken: abstain_share"]})
    assert "month 9: drift alert (overall) | month 10: target broken" in str(err.value)
    text = (REPO / "databricks" / "notebooks" / "monthly_drift.py").read_text("utf-8")
    assert text.rindex("raise_if_alerts(raised)") > text.rindex("for m in range(START")


def test_few_labels_are_insufficient_for_accuracy() -> None:
    record = {
        "month": 5,
        "abstain_share": 0.05,
        "delayed_accuracy_of_month": {"month": 4, "labelled": 40, "balanced_accuracy": 0.2},
    }
    out = slo.evaluate(record, [], CONFIG)["delayed_balanced_accuracy"]
    assert out["status"] == slo.INSUFFICIENT


def test_the_committed_replay_report_is_current() -> None:
    """scripts/slo_report.py's output matches the committed months and targets."""
    report = json.loads((REPO / "results" / "phase3" / "monthly" / "slo_replay.json").read_text())
    monthly = REPO / "results" / "phase3" / "monthly"
    labelled: list[dict[str, object]] = []
    for entry in report["months"]:
        record = json.loads((monthly / f"month_{entry['month']:02d}.json").read_text())
        if record.get("delayed_accuracy_of_month"):
            labelled.append(record["delayed_accuracy_of_month"])
        assert entry["slo"] == json.loads(json.dumps(slo.evaluate(record, labelled, CONFIG)))
    in_distribution = [e for e in report["months"] if e["month"] <= 4]
    assert all(not e["alerts"] for e in in_distribution)  # no false alarm on months 1 to 4


def test_config_targets_match_their_stated_sources() -> None:
    triage = json.loads((REPO / "results" / "phase2" / "triage" / "threshold.json").read_text())
    assert "0.0638" in CONFIG["targets"]["melanoma_referral_sensitivity"]["measure"]
    assert triage["thresholds"]["v2-ensemble-final-seeds0-1-2"] == pytest.approx(0.0638, abs=1e-4)
    assert CONFIG["targets"]["melanoma_referral_sensitivity"]["min_melanoma_images"] == 100
