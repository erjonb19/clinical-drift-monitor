"""End to end on tiny synthetic images: the plumbing ``cdm.reproduce`` runs, on CPU, in CI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import TensorDataset

from cdm.data import CLASSES, HamDataset, OODImages, eval_transform, train_transform
from cdm.reproduce import (
    DETECTORS,
    Result,
    ResumeMismatchError,
    _limit,
    aggregate,
    check_plausible,
    load_finished,
    results_markdown,
    run_seed,
    write_json,
)
from cdm.train import TrainConfig


def fake_ham(tmp_path: Path, per_class: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for label, dx in enumerate(CLASSES):
        for k in range(per_class):
            path = tmp_path / f"{dx}_{k}.jpg"
            pixels = rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(path)
            rows.append({"dx": dx, "label": label, "image_path": str(path)})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def result_and_sets(tmp_path_factory: pytest.TempPathFactory) -> tuple[Result, int]:
    frame = fake_ham(tmp_path_factory.mktemp("ham"), per_class=3)
    sets = {
        "train": HamDataset(frame, train_transform()),
        "train_eval": HamDataset(frame, eval_transform()),
        "val": HamDataset(frame, eval_transform()),
        "test": HamDataset(frame, eval_transform()),
    }
    noise = TensorDataset(torch.randn(10, 3, 224, 224), torch.zeros(10))
    ood = {"noise": OODImages(noise, range(10))}  # type: ignore[arg-type]
    cfg = TrainConfig(epochs=1, batch_size=8, num_workers=0, pretrained=False)
    return run_seed(0, sets, ood, cfg, torch.device("cpu")), len(frame)


def test_run_seed_reports_every_metric(result_and_sets: tuple[Result, int]) -> None:
    result, n_images = result_and_sets
    assert set(result["test"]["recall"]) == set(CLASSES)
    assert sum(result["test"]["support"].values()) == n_images
    for detector in DETECTORS:
        metrics = result["ood"]["noise"][detector]
        assert 0.0 <= metrics["auroc"] <= 1.0
        assert 0.0 <= metrics["fpr95"] <= 1.0
    assert result["ood"]["noise"]["n"] == {"id": n_images, "ood": 10}


def test_aggregate_and_markdown(result_and_sets: tuple[Result, int]) -> None:
    result, _ = result_and_sets
    summary = aggregate([result, {**result, "seed": 1}])
    assert summary["seeds"] == [0, 1]
    provenance = {"code_version": "abc", "device": "cpu", "split_fingerprint": "f00"}
    table = results_markdown(summary, provenance)
    assert "Balanced accuracy" in table
    assert all(f"| {d} |" in table for d in DETECTORS)


def test_implausible_auroc_is_flagged(result_and_sets: tuple[Result, int]) -> None:
    result, _ = result_and_sets
    broken = {
        **result,
        "ood": {"noise": {d: {"auroc": 0.041, "fpr95": 1.0} for d in DETECTORS}},
    }
    assert len(check_plausible(broken)) == len(DETECTORS)


def test_limit_keeps_every_class() -> None:
    frame = pd.DataFrame({"dx": [c for c in CLASSES for _ in range(20)]})
    limited = _limit(frame, 20, seed=0)
    assert len(limited) == 20
    assert set(limited["dx"]) == set(CLASSES)


def test_resume_reuses_only_a_matching_seed(tmp_path: Path) -> None:
    key = {"code_version": "abc1234", "config": {"epochs": 12}}
    path = tmp_path / "seed0.json"
    assert load_finished(path, key) is None
    write_json(path, {"run_key": key, "seed": 0})
    assert load_finished(path, key) == {"run_key": key, "seed": 0}
    changed = {**key, "config": {"epochs": 1}}
    with pytest.raises(ResumeMismatchError):
        load_finished(path, changed)
