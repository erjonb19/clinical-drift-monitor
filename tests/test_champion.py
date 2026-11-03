"""The retraining champion check (silent-failures #12)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cdm.champion import (
    CLASSES_N,
    COMMITTED,
    SMALLEST_CLASS_IMAGES,
    SUPPORT_SOURCE,
    TOLERANCE_POINTS,
    ChampionMismatch,
    champion_record,
    check_champion_val,
    committed_val_balanced_accuracy,
)

REPO = Path(__file__).resolve().parents[1]


def test_committed_value_is_the_serving_validation_result() -> None:
    value = committed_val_balanced_accuracy(REPO / COMMITTED)
    assert value == pytest.approx(0.6826, abs=5e-5)  # 68.26%, as silent-failures #12 states


def test_the_run_behind_entry_12_is_within_tolerance() -> None:
    """68.45% against 68.26%: the gap #12 reports (0.19 points) is tolerated, not refused."""
    gap = check_champion_val(0.68445666477536, committed_val_balanced_accuracy(REPO / COMMITTED))
    assert 0.18 < gap < 0.20 < TOLERANCE_POINTS


def test_tolerance_is_two_images_of_the_smallest_validation_class() -> None:
    record = json.loads((REPO / SUPPORT_SOURCE).read_text())
    support = record["scores"]["val"]["pooled"]["support"]
    assert sum(support.values()) == 3210 and len(support) == CLASSES_N
    assert min(support.values()) == SMALLEST_CLASS_IMAGES == support["df"]
    assert pytest.approx(2 * 100 / (7 * 38)) == TOLERANCE_POINTS  # 0.75 points


@pytest.mark.parametrize("observed", [0.6740, 0.6910, 0.60, 0.75])
def test_a_gap_past_the_tolerance_refuses(observed: float) -> None:
    with pytest.raises(ChampionMismatch, match="refusing to retrain"):
        check_champion_val(observed, 0.6826)


def test_the_tolerance_edge() -> None:
    assert check_champion_val(0.6826 + 0.0074, 0.6826) == pytest.approx(0.74)
    with pytest.raises(ChampionMismatch):
        check_champion_val(0.6826 - 0.0076, 0.6826)


def test_points_not_fractions() -> None:
    """A score passed as a percentage is rejected, not compared on the wrong scale."""
    with pytest.raises(ValueError):
        check_champion_val(68.45, 0.6826)


def test_record_names_the_models_file_and_every_weight_hash(tmp_path: Path) -> None:
    models = tmp_path / "final_models.json"
    models.write_text(json.dumps([{"seed": 0}, {"seed": 1}]))
    record = champion_record(models, {1: "b" * 64, 0: "a" * 64}, 0.6826, 0.6845)
    assert record["models_file_sha256"] == hashlib.sha256(models.read_bytes()).hexdigest()
    assert record["weights_sha256"] == {"seed0": "a" * 64, "seed1": "b" * 64}
    assert record["gap_points"] == pytest.approx(0.19)
    assert record["tolerance_points"] == TOLERANCE_POINTS


def test_record_refuses_when_nothing_was_checked(tmp_path: Path) -> None:
    models = tmp_path / "m.json"
    models.write_text("[]")
    with pytest.raises(ValueError):
        champion_record(models, {}, 0.6826, 0.6826)


def test_the_notebook_checks_before_training() -> None:
    """The check sits before the training loop, so a mismatch costs no challenger training."""
    text = (REPO / "databricks" / "notebooks" / "retrain.py").read_text()
    assert text.index("check_champion_val(champ_val, COMMITTED_VAL)") < text.index(
        "train(copy.deepcopy"
    )
    assert '"champion_check": CHAMPION' in text
    assert "champion_refused_" in text  # a refusal leaves a record, not only a log line


def test_exactly_two_smallest_class_images_pass() -> None:
    """Two df images flipped the same way land on the boundary and are tolerated."""
    two = 2 / (CLASSES_N * SMALLEST_CLASS_IMAGES)
    assert check_champion_val(0.6826 + two, 0.6826) == pytest.approx(TOLERANCE_POINTS)
    assert check_champion_val(0.6826 - two, 0.6826) == pytest.approx(-TOLERANCE_POINTS)
