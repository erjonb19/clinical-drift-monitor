"""The tuning ladder's rules and the batch options (balanced sampling, EMA, colour constancy)."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from cdm.data import ShadesOfGray, eval_transform
from cdm.ladder import (
    combination_needed,
    deltas,
    drift_preserved,
    keep_data_version,
    merge,
    passes,
)
from cdm.silver import EncodedImages
from cdm.splits import CLASSES
from cdm.train import TrainConfig, balanced_weights, build_model, seed_everything, train

REF = {"balanced_accuracy": 0.6767, "macro_auroc": 0.9247, "melanoma_sensitivity": 0.6452}


@pytest.mark.parametrize(
    ("change", "kept"),
    [
        ({"balanced_accuracy": 0.6867}, True),  # exactly +1.00
        ({"balanced_accuracy": 0.6866}, False),  # +0.99
        ({"balanced_accuracy": 0.70, "macro_auroc": 0.9197}, True),  # AUROC -0.50 allowed
        ({"balanced_accuracy": 0.70, "macro_auroc": 0.9196}, False),  # -0.51
        ({"balanced_accuracy": 0.70, "melanoma_sensitivity": 0.6252}, True),  # mel -2.00 allowed
        ({"balanced_accuracy": 0.70, "melanoma_sensitivity": 0.6251}, False),
    ],
)
def test_keep_rule_edges(change: dict[str, float], kept: bool) -> None:
    candidate = {**REF, **change}
    assert passes(candidate, REF) is kept
    assert set(deltas(candidate, REF)) == set(REF)


def test_merge_combines_changes_and_refuses_conflicts() -> None:
    base = {"epochs": 12, "rotate": False}
    assert merge(base, [{"rotate": True}, {"epochs": 25, "warmup_epochs": 1}]) == {
        "epochs": 25, "rotate": True, "warmup_epochs": 1,
    }  # fmt: skip
    with pytest.raises(ValueError, match="disagree"):
        merge(base, [{"epochs": 25}, {"epochs": 30}])
    assert not combination_needed(["1b"]) and combination_needed(["1b", "1d"])


def test_drift_rule_allows_two_points_per_site_and_score() -> None:
    ref = {"buenos_aires/energy": 0.80, "pad_ufes/mahalanobis": 0.95}
    assert drift_preserved({"buenos_aires/energy": 0.78, "pad_ufes/mahalanobis": 0.96}, ref)[
        "preserved"
    ]
    assert not drift_preserved({"buenos_aires/energy": 0.779, "pad_ufes/mahalanobis": 0.96}, ref)[
        "preserved"
    ]


def test_balanced_weights_equalise_cells_up_to_the_cap() -> None:
    labels = [0] * 90 + [1] * 10 + [0] * 2
    groups = ["a"] * 100 + ["b"] * 2
    w = balanced_weights(labels, groups, cap=10.0)
    assert sum(w[:90]) == pytest.approx(sum(w[90:100]))  # each (group, class) cell totals 1
    assert max(w) <= 10.0 * float(np.median(w)) + 1e-12  # the 2-image cell is capped


def test_shades_of_gray_removes_a_colour_cast() -> None:
    cast = Image.new("RGB", (32, 32), (200, 120, 80))
    out = np.asarray(ShadesOfGray()(cast), dtype=np.float64)
    assert np.ptp(out.reshape(-1, 3).mean(axis=0)) < 2.0  # channels now equal
    assert eval_transform(64, color_constancy=True)(cast).shape == (3, 64, 64)


def _images(n: int) -> tuple[list[bytes], list[int], list[str]]:
    rng = np.random.default_rng(0)
    images = []
    for _ in range(n):
        out = io.BytesIO()
        Image.fromarray(rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)).save(out, format="PNG")
        images.append(out.getvalue())
    labels = [i % len(CLASSES) for i in range(n)]
    return images, labels, ["a" if i % 3 else "b" for i in range(n)]


def test_balanced_sampling_with_ema_trains_and_returns_the_averaged_model() -> None:
    images, labels, groups = _images(28)
    tr = EncodedImages(images, labels, eval_transform(32), groups)
    cfg = TrainConfig(epochs=1, batch_size=8, num_workers=0, pretrained=False, num_threads=2,
                      balanced_sampling=True, ema_decay=0.9)  # fmt: skip
    gen = seed_everything(0, 2)
    model, history = train(build_model(pretrained=False), tr, tr, cfg, 0, torch.device("cpu"), gen)
    assert len(history) == 1 and not any(k.startswith("module.") for k in model.state_dict())


def test_balanced_sampling_without_groups_is_refused() -> None:
    images, labels, _ = _images(8)
    tr = EncodedImages(images, labels, eval_transform(32))
    cfg = TrainConfig(
        epochs=1, batch_size=4, num_workers=0, pretrained=False, balanced_sampling=True
    )
    with pytest.raises(ValueError, match="group"):
        train(
            build_model(pretrained=False),
            tr,
            tr,
            cfg,
            0,
            torch.device("cpu"),
            seed_everything(0, 2),
        )


def _runs(pooled: float, clients: dict[str, float], n: int = 3) -> list[dict[str, Any]]:
    return [
        {"pooled": {"balanced_accuracy": pooled}, **{c: {"balanced_accuracy": v}
                                                      for c, v in clients.items()}}
        for _ in range(n)
    ]  # fmt: skip


CLIENTS = ("ham_vienna", "barcelona")


@pytest.mark.parametrize(
    ("v2_pooled", "v2_clients", "keep"),
    [
        (0.6900, {"ham_vienna": 0.70, "barcelona": 0.60}, True),  # +1.00, no client drop
        (0.6899, {"ham_vienna": 0.70, "barcelona": 0.60}, False),  # +0.99
        (0.7000, {"ham_vienna": 0.67, "barcelona": 0.65}, True),  # Vienna -3.00 allowed
        (0.7000, {"ham_vienna": 0.6699, "barcelona": 0.65}, False),  # Vienna -3.01
    ],
)
def test_data_version_rule_edges(
    v2_pooled: float, v2_clients: dict[str, float], keep: bool
) -> None:
    v1 = _runs(0.68, {"ham_vienna": 0.70, "barcelona": 0.60})
    out = keep_data_version(v1, _runs(v2_pooled, v2_clients), CLIENTS)
    assert out["keep_v2"] is keep


def test_data_version_rule_uses_the_mean_over_seeds() -> None:
    v1 = _runs(0.68, {"ham_vienna": 0.70, "barcelona": 0.60})
    v2 = [_runs(0.66, {"ham_vienna": 0.70, "barcelona": 0.60}, 1)[0],
          _runs(0.72, {"ham_vienna": 0.70, "barcelona": 0.60}, 1)[0]]  # fmt: skip
    out = keep_data_version(v1, v2, CLIENTS)
    assert out["change_points"]["pooled"] == pytest.approx(1.0) and out["keep_v2"]
