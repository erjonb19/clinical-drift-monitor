"""Backlog item 4: the months 1 to 12 melanoma-referral backfill (cdm.backfill)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cdm import slo
from cdm.backfill import (
    BackfillMismatch,
    check_committed,
    committed_records,
    merge_history,
    month_record,
)
from cdm.splits import CLASSES

REPO = Path(__file__).resolve().parents[1]
MEL = CLASSES.index("mel")
REFER = 0.0638


def _month(seed: int, n: int = 300) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.ones(len(CLASSES)), size=n)
    labels = rng.integers(0, len(CLASSES), size=n)
    labels[:10] = -1  # a few unlabelled images, as in the replay
    return labels.astype(np.int64), probs


def test_the_record_counts_referred_melanomas_from_the_stored_flag() -> None:
    labels, probs = _month(1)
    flags = probs[:, MEL] >= REFER
    record = month_record(4, labels, probs, flags, REFER)
    is_mel = labels == MEL
    assert record["month"] == 4
    assert record["melanoma_referred"] == int((is_mel & flags).sum())
    assert record["melanoma_images"] == int(is_mel.sum())
    assert record["labelled"] == 290


def test_a_stored_flag_that_disagrees_with_the_threshold_fails() -> None:
    labels, probs = _month(2)
    flags = probs[:, MEL] >= REFER
    flags[5] = not flags[5]
    with pytest.raises(BackfillMismatch, match="1 rows"):
        month_record(4, labels, probs, flags, REFER)
    with pytest.raises(BackfillMismatch):  # a wrong threshold is caught the same way
        month_record(4, labels, probs, probs[:, MEL] >= REFER, 0.5)


def test_rebuilt_records_must_match_the_committed_ones() -> None:
    labels, probs = _month(3)
    record = month_record(4, labels, probs, probs[:, MEL] >= REFER, REFER)
    committed = {4: {k: record[k] for k in ("labelled", "melanoma_images", "balanced_accuracy")}}
    assert check_committed([record], committed) == {4: "matches the committed record"}
    unchecked = {**record, "month": 12}
    assert check_committed([unchecked], committed)[12].startswith("unchecked")
    for key, value in [("labelled", 289), ("balanced_accuracy", 0.1)]:
        with pytest.raises(BackfillMismatch, match=key):
            check_committed([record], {4: {**committed[4], key: value}})


def test_the_committed_replay_records_are_keyed_by_the_month_they_describe() -> None:
    files = [
        json.loads((REPO / "results/phase3/monthly" / f"month_{m:02d}.json").read_text("utf-8"))
        for m in range(1, 13)
    ]
    committed = committed_records(files)
    assert sorted(committed) == list(range(1, 12))  # month 12's labels arrive after the replay
    assert all("melanoma_referred" not in r for r in committed.values())


def test_merge_fills_missing_counts_and_keeps_recorded_ones() -> None:
    history = [
        {"month": 1, "melanoma_images": 40},
        {"month": 2, "melanoma_images": 50, "melanoma_referred": 49},
    ]
    backfill = [
        {"month": 1, "melanoma_images": 40, "melanoma_referred": 38},
        {"month": 2, "melanoma_images": 50, "melanoma_referred": 1},
        {"month": 3, "melanoma_images": 60, "melanoma_referred": 50},
    ]
    merged = merge_history(history, backfill)
    assert [r["melanoma_referred"] for r in merged] == [38, 49, 50]
    assert [r["month"] for r in merge_history(history, backfill, before_month=3)] == [1, 2]


def test_with_the_backfill_the_melanoma_target_can_be_judged() -> None:
    """Without the counts the window is insufficient; with them it pools to 100 melanomas."""
    config = slo.load_config(REPO / "config" / "slo.json")
    history = [{"month": m, "melanoma_images": 40, "labelled": 300} for m in (1, 2, 3)]
    backfill = [{**r, "melanoma_referred": 36} for r in history]
    record = {"month": 4}
    before = slo.evaluate(record, history, config)["melanoma_referral_sensitivity"]
    after = slo.evaluate(record, merge_history(history, backfill, 4), config)[
        "melanoma_referral_sensitivity"
    ]
    assert before["status"] == slo.INSUFFICIENT
    assert after["status"] == slo.MET and after["months"] == [1, 2, 3]


def test_the_replay_report_judges_the_melanoma_target_once_backfilled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """scripts/slo_report.py with a (synthetic) backfill file: month m sees months before m."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("slo_report", REPO / "scripts/slo_report.py")
    assert spec and spec.loader
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    fake = [{"month": m, "melanoma_images": 40, "melanoma_referred": 37} for m in range(1, 13)]
    (tmp_path / "melanoma_backfill.json").write_text(json.dumps({"months": fake}))
    monkeypatch.setattr(report, "BACKFILL", tmp_path / "melanoma_backfill.json")
    monkeypatch.setattr(report, "OUT", tmp_path / "slo_replay.json")
    report.main()
    out = json.loads((tmp_path / "slo_replay.json").read_text())
    mel = {m["month"]: m["slo"]["melanoma_referral_sensitivity"] for m in out["months"]}
    assert mel[1]["status"] == slo.INSUFFICIENT  # no labelled month before month 1
    assert mel[4]["status"] == slo.MET and mel[4]["months"] == [1, 2, 3]
    assert mel[12]["months"] == [9, 10, 11]
