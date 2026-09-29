"""Gold task logic: scores point the right way, the baseline never uses train, weights are
checked before use."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torchvision.models import efficientnet_b0

from cdm.gold import baseline, embed, feature_stats, load_imagenet, score


def embeddings(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[tuple[str, str, str, str | None, np.ndarray]] = []
    centers = {"nv": 0.0, "mel": 4.0, "bcc": 8.0}
    for split, n in (("train", 60), ("val", 15), ("test", 15)):
        for label, c in centers.items():
            for _ in range(n):
                rows.append(("ham10000", split, label, None, rng.normal(c, 1.0, 16)))
    for _ in range(30):
        rows.append(("barcelona", "score", "nv", "III", rng.normal(0.0, 1.0, 16) + 6.0))
    frame = pd.DataFrame(
        rows, columns=["source", "split", "label", "fitzpatrick_skin_type", "features"]
    )
    frame["isic_id"] = [f"ISIC_{i}" for i in range(len(frame))]
    return frame


def test_shifted_site_scores_higher_than_held_out_ham10000() -> None:
    scores = score(embeddings(), n_components=8)
    test = scores[(scores.source == "ham10000") & (scores.split == "test")]["mahalanobis"]
    site = scores[scores.source == "barcelona"]["mahalanobis"]
    assert site.median() > test.quantile(0.95)


def test_baseline_uses_validation_and_test_only() -> None:
    table = baseline(score(embeddings(), n_components=8))
    assert set(table["split"]) == {"val", "test"}
    assert (table["n"] == 45).all()
    assert (table["q05"] <= table["q50"]).all() and (table["q50"] <= table["q99"]).all()


def test_feature_stats_cover_each_source_and_skin_type() -> None:
    emb = embeddings()
    stats = feature_stats(score(emb, n_components=8), emb)
    groups = set(zip(stats.source, stats.skin_type, strict=True))
    expected = {("ham10000", "all"), ("barcelona", "all"), ("barcelona", "III")}
    assert expected | {("ham10000", "not recorded")} <= groups
    assert stats.loc[(stats.source == "barcelona") & (stats.skin_type == "all"), "n"].item() == 30


def test_weights_are_refused_unless_their_hash_matches_the_name(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    torch.save(efficientnet_b0(weights=None).state_dict(), buffer)
    data = buffer.getvalue()
    good = tmp_path / f"effb0-{hashlib.sha256(data).hexdigest()[:8]}.pth"
    good.write_bytes(data)
    assert not load_imagenet(good).training
    bad = tmp_path / "effb0-00000000.pth"
    bad.write_bytes(data)
    with pytest.raises(ValueError, match="SHA-256"):
        load_imagenet(bad)


def test_embed_yields_one_1280_d_row_per_image_in_batches() -> None:
    out = io.BytesIO()
    Image.new("RGB", (224, 224), (10, 200, 30)).save(out, format="PNG")
    batches = list(embed([out.getvalue()] * 5, efficientnet_b0(weights=None).eval(), batch_size=2))
    assert [b.shape for b in batches] == [(2, 1280), (2, 1280), (1, 1280)]
