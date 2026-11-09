"""Backlog item 15: the staged-shift simulation (cdm.detector_quality, scripts/detector_quality.py).

Synthetic scores only: these tests check the simulation's logic, not the detector. Every number
they produce is made up and never reported.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from cdm import detector_quality as dq
from cdm.splits import CLASSES

REPO = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(0)
BASELINE = RNG.normal(10, 2, 3210)
POOL = RNG.normal(10, 2, 3262)


def test_shifts_are_applied_at_the_model_input_size() -> None:
    image = dq.to_input(Image.new("RGB", (600, 450), (180, 120, 100)))
    assert image.size == (224, 224)
    pattern = Image.effect_noise((224, 224), 80).convert("RGB")
    blurred = np.asarray(dq.apply_shift(pattern, "blur", 4.0), dtype=float)
    assert blurred.std() < np.asarray(pattern, dtype=float).std()
    darker = np.asarray(dq.apply_shift(image, "brightness", 0.7), dtype=float)
    assert darker.mean() < np.asarray(image, dtype=float).mean()
    assert dq.apply_shift(pattern, "jpeg", 10).size == (224, 224)
    with pytest.raises(ValueError):
        dq.apply_shift(image, "rotate", 1)


def test_a_month_is_drawn_without_replacement_and_the_age_mix_holds() -> None:
    rng = np.random.default_rng(1)
    idx, replaced = dq.draw_month(rng, 3262)
    assert len(set(idx.tolist())) == 300 and not replaced
    old = np.zeros(3262, dtype=bool)
    old[:500] = True
    idx, replaced = dq.draw_month(rng, 3262, old, 0.75)
    assert old[idx].sum() == 225 and not replaced
    few = np.zeros(3262, dtype=bool)
    few[:100] = True  # too few old patients for 90%: drawn with replacement, and said so
    idx, replaced = dq.draw_month(rng, 3262, few, 0.9)
    assert few[idx].sum() == 270 and replaced


def test_no_shift_rarely_alarms_and_a_large_shift_is_found_in_the_first_month() -> None:
    control = [dq.run_sequence(dq.sequence_seed(0, r), BASELINE, POOL) for r in range(20)]
    summary = dq.summarise(control, control=True)
    assert summary["unshifted_month_alert_rate"]["months"] == 240
    assert summary["unshifted_month_alert_rate"]["rate"] < 0.05
    shifted = POOL + 6.0  # far outside the baseline: every shifted month should alert
    seeds = [dq.sequence_seed(1, r) for r in range(20)]
    runs = [dq.run_sequence(s, BASELINE, POOL, shifted, 1.0) for s in seeds]
    paired = [dq.run_sequence(s, BASELINE, POOL) for s in seeds]
    out = dq.summarise(runs, paired=paired)
    assert out["paired_control_months_4_to_12_alert_rate"]["rate"] < 0.05
    assert out["share_detected"] == 1.0 and out["months_to_detect"]["median"] == 1.0
    assert out["unshifted_month_alert_rate"]["months"] == 60  # months 1 to 3 only


def test_time_to_detect_counts_from_the_first_shifted_month() -> None:
    never = [False] * 12
    assert dq.time_to_detect(never) is None
    early_false_alarm = [True] + [False] * 11  # an alarm before the shift is not a detection
    assert dq.time_to_detect(early_false_alarm) is None
    assert dq.time_to_detect([False] * 5 + [True] * 7) == 3


def test_sequences_are_seeded_and_paired_with_their_control() -> None:
    """The same seed gives the same months with or without the shift, so the paired control
    is the shifted sequence with no image altered (review of PR #75)."""
    seed = dq.sequence_seed(7, 3)
    a = dq.run_sequence(seed, BASELINE, POOL, POOL + 1, 0.5)
    assert a == dq.run_sequence(seed, BASELINE, POOL, POOL + 1, 0.5)
    tiny = POOL + 1e-9  # a shift too small to change any test: the paired run must match it
    assert dq.run_sequence(seed, BASELINE, POOL, tiny, 0.5) == dq.run_sequence(seed, BASELINE, POOL)


def test_a_shift_that_lowers_scores_is_listed_for_a_look() -> None:
    assert dq.check_direction(BASELINE, {"blur:4": 12.0, "brightness:1.3": 9.0}) == [
        "brightness:1.3"
    ]


def _script() -> Any:
    spec = importlib.util.spec_from_file_location("dqs", REPO / "scripts/detector_quality.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_scores(work: Path) -> None:
    """A scores.npz and score_meta.json like the score step writes, with made-up values."""
    spec = json.loads((REPO / "results/phase3/detector_quality/spec.json").read_text())
    rng = np.random.default_rng(3)
    n_val, n_test, d = 600, 640, 12
    out: dict[str, Any] = {}
    for split, n in (("val", n_val), ("test", n_test)):
        out[f"{split}/none:0/ood"] = rng.normal(10, 2, n)
        out[f"{split}/none:0/probs"] = rng.dirichlet(np.ones(len(CLASSES)), n)
        out[f"{split}/none:0/features"] = rng.normal(0, 1, (n, d)).astype(np.float32)
        out[f"{split}/ids"] = np.array([f"ISIC_{split}{i}" for i in range(n)])
    for kind in ("blur", "brightness", "jpeg"):
        for size in spec["shifts"][kind]["sizes"]:
            out[f"test/{kind}:{float(size):g}/ood"] = rng.normal(
                10 + 2 * (kind == "blur"), 2, n_test
            )
    out["test/age"] = rng.choice([30.0, 50.0, 80.0, np.nan], n_test)
    np.savez_compressed(work / "scores.npz", **out)
    (work / "score_meta.json").write_text(json.dumps({"bundle_version": "fake", "seconds": 0}))


def test_simulate_reports_every_cell_labelled_staged(tmp_path: Path) -> None:
    _fake_scores(tmp_path)
    result = _script().simulate(tmp_path, repeats=3)
    spec = json.loads((REPO / "results/phase3/detector_quality/spec.json").read_text())
    image_cells = 3 * 3 * len(spec["share_shifted"])
    assert len(result["cells"]) == 1 + image_cells + 3  # control, image shifts, age mix
    assert result["cells"][0]["shift"] == "none (control)"
    assert "staged" in result["staged"] and result["spec_sha256"]
    blur = [c for c in result["cells"] if c["shift"] == "blur" and c["share"] == 1.0]
    assert all(c["share_detected"] == 1.0 for c in blur)  # the fake blur adds 2 to every score
    assert result["post_replay_rule_false_alarms"]["months"] == 3 * 12
    assert "none:0" not in result["mean_ood"]
    shifted_cells = [c for c in result["cells"] if c["shift"] != "none (control)"]
    assert all("paired_control_months_4_to_12_alert_rate" in c for c in shifted_cells)
    assert result["pool_age_unknown"] > 0  # the fake ages include unknowns, left out of age mixes


def test_the_pool_check_wants_exactly_silvers_images(tmp_path: Path) -> None:
    script = _script()
    with (tmp_path / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, script.FIELDS)
        writer.writeheader()
        for i in range(3262 + 3210 - 1):  # one image short
            writer.writerow({"isic_id": f"ISIC_{i}", "split": "test" if i < 3262 else "val"})
    with pytest.raises(RuntimeError, match="not the spec's pool"):
        script.check_pool(tmp_path / "manifest.csv")


def test_without_the_id_list_the_fetch_says_how_to_export_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script()
    monkeypatch.setattr(script, "IDS", script.REPO / "config" / "no_such_file.csv")
    with pytest.raises(SystemExit, match="silver_images"):
        script.wanted_ids()


def test_the_kept_input_gives_exactly_what_eval_transform_gives() -> None:
    """The fetch keeps to_input's 224 px crop as PNG; normalised, it must equal the serving
    transform on the original image, or the scores would not be the shipped detector's."""
    from torchvision import transforms

    from cdm.data import IMAGENET_MEAN, IMAGENET_STD, eval_transform

    original = Image.effect_noise((613, 457), 60).convert("RGB")
    tensor = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
    assert float((eval_transform(224)(original) - tensor(dq.to_input(original))).abs().max()) == 0
