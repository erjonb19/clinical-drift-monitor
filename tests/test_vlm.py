"""BiomedCLIP baseline helpers (research Phase B), on a stand-in model: no download in CI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from cdm.probe import ProbeConfig, class_weights, select_config, train_probe
from cdm.splits import CLASSES
from cdm.vlm import (
    CLASS_TEXT,
    PINS,
    TEMPLATES,
    check_cache,
    class_text,
    clusters,
    drift_metrics,
    image_embeddings,
    select_templates,
    text_embeddings,
    zero_shot_logits,
)

K, DIM = len(CLASSES), 16


class FakeClip(nn.Module):
    """Images and texts map to fixed directions: a text names one class; template 1 is the
    informative one (the others add noise that swamps the class)."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, DIM, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(100.0)))
        self.directions = torch.eye(DIM)[:K]

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.proj(x.mean(dim=(2, 3)))
        return out

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        cls, template = tokens[:, 0], tokens[:, 1]
        noise = torch.randn(len(tokens), DIM, generator=torch.Generator().manual_seed(1))
        weight = torch.where(template == 1, 0.0, 5.0).unsqueeze(1)
        return self.directions[cls] + weight * noise


def fake_tokenizer(prompts: list[str], context_length: int) -> torch.Tensor:
    assert context_length == 256
    first = CLASS_TEXT[CLASSES[0]]
    template = next(i for i, t in enumerate(TEMPLATES) if prompts[0] == t.format(first))
    return torch.tensor([[i, template] for i in range(len(prompts))])


def test_check_cache_accepts_only_the_pinned_snapshots(tmp_path: Path) -> None:
    for repo, rev in PINS.items():
        (tmp_path / "hub" / repo / "snapshots" / rev).mkdir(parents=True)
        (tmp_path / "hub" / repo / "refs").mkdir()
        (tmp_path / "hub" / repo / "refs" / "main").write_text(rev)
    check_cache(tmp_path)
    repo = next(iter(PINS))
    (tmp_path / "hub" / repo / "snapshots" / ("0" * 40)).mkdir()
    with pytest.raises(ValueError, match="snapshots"):
        check_cache(tmp_path)


def test_check_cache_refuses_a_missing_cache(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected only"):
        check_cache(tmp_path)


def test_embeddings_are_normalized_with_the_right_shapes() -> None:
    model = FakeClip()
    batches = [torch.rand(4, 3, 8, 8), torch.rand(3, 3, 8, 8)]
    images = image_embeddings(model, batches, torch.device("cpu"))
    assert images.shape == (7, DIM) and np.allclose(np.linalg.norm(images, axis=1), 1, atol=1e-5)
    text = text_embeddings(model, fake_tokenizer, torch.device("cpu"))
    assert text.shape == (len(TEMPLATES), K, DIM)
    assert np.allclose(np.linalg.norm(text, axis=2), 1, atol=1e-5)


def test_zero_shot_picks_the_class_whose_text_matches() -> None:
    text = class_text(np.eye(DIM, dtype=np.float32)[None, :K], [0])
    images = np.eye(DIM, dtype=np.float32)[:K][[3, 0, 6]]
    assert zero_shot_logits(images, text, 100.0).argmax(1).tolist() == [3, 0, 6]


def test_template_selection_finds_the_informative_template() -> None:
    model = FakeClip()
    text = text_embeddings(model, fake_tokenizer, torch.device("cpu"))
    labels = np.repeat(np.arange(K), 5)
    images = np.eye(DIM, dtype=np.float32)[labels]
    best, scores = select_templates(images, labels, text, 100.0)
    assert best == "template_1" and scores["template_1"] == 1.0
    assert set(scores) == {*(f"template_{i}" for i in range(len(TEMPLATES))), "all_templates"}


def blobs(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, n)
    centers = np.random.default_rng(0).normal(0, 3, (K, DIM))
    return (centers[y] + rng.normal(0, 1, (n, DIM))).astype(np.float32), y.astype(np.int64)


def test_probe_learns_separable_classes_and_seeds_differ() -> None:
    x, y = blobs(700, 1)
    xv, yv = blobs(300, 2)
    a = train_probe(x, y, K, seed=0, cfg=ProbeConfig(lr=1e-2, epochs=10))
    b = train_probe(x, y, K, seed=1, cfg=ProbeConfig(lr=1e-2, epochs=10))
    assert (a.logits(xv).argmax(1) == yv).mean() > 0.9
    assert not torch.equal(a.head.weight, b.head.weight)
    again = train_probe(x, y, K, seed=0, cfg=ProbeConfig(lr=1e-2, epochs=10))
    assert torch.equal(a.head.weight, again.head.weight)


def test_probe_selection_uses_validation_only_and_reports_every_grid_point() -> None:
    x, y = blobs(400, 1)
    xv, yv = blobs(200, 2)
    grid = {"lr": (1e-6, 1e-2), "weight_decay": (0.0,)}
    cfg, table = select_config(x, y, xv, yv, K, seeds=(0, 1), grid=grid)
    assert len(table) == 2 and cfg.lr == 1e-2
    assert all(len(row["val_balanced_accuracy"]) == 2 for row in table)


def test_class_weights_refuse_a_missing_class() -> None:
    with pytest.raises(ValueError, match="no training examples"):
        class_weights(np.array([0, 1, 1]), 3)


def test_clusters_prefer_patient_then_lesion_then_image() -> None:
    got = clusters(["IP_1", None, ""], ["IL_1", "IL_2", None], ["a", "b", "c"])
    assert got == ["patient:IP_1", "lesion:IL_2", "image:c"]


def test_drift_scores_point_the_right_way() -> None:
    """Far-away features and flat logits must score as more out-of-distribution than the
    in-distribution test set, for every detector (AUROC well above 0.5)."""
    x, y = blobs(600, 1)
    test, _ = blobs(200, 3)
    far = (np.random.default_rng(4).normal(0, 1, (200, DIM)) + 25).astype(np.float32)
    confident = np.eye(K)[np.arange(200) % K] * 10.0
    flat = np.random.default_rng(5).normal(0, 0.1, (200, K))
    out = drift_metrics(
        x.astype(np.float64),
        y,
        {"test": test.astype(np.float64), "far": far.astype(np.float64)},
        {"test": confident, "far": flat},
        ["far"],
    )
    for detector, by_shift in out["metrics"].items():
        assert by_shift["far"]["auroc"] > 0.95, detector
    assert 1 <= out["pca_components_method_d"] <= DIM
