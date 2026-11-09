"""Backlog item 6: the champion bundle loads on CPU and scores a fixed set of public images.

Runs inference with the real serving code (cdm.serving.Service), unlike the eval gate, which
reads committed metrics. It needs two things the repo cannot hold: the champion bundle (a
release asset the owner uploads) and the image manifest (written by
scripts/make_smoke_manifest.py where the ISIC API is reachable). Until both exist it skips and
says which is missing; it never passes without scoring.
"""

from __future__ import annotations

import csv
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
CONFIG = json.loads((REPO / "config" / "champion_smoke.json").read_text(encoding="utf-8"))
CACHE = Path.home() / ".cache" / "cdm-smoke"


def missing() -> list[str]:
    out = []
    asset = CONFIG["bundle_asset"]
    if not asset.get("url") or not asset.get("sha256"):
        out.append("the champion bundle release asset (config/champion_smoke.json bundle_asset)")
    if not (REPO / CONFIG["images_manifest"]).exists():
        out.append(f"the image manifest ({CONFIG['images_manifest']})")
    return out


def fetch(url: str, sha256: str, dest: Path) -> bytes:
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == sha256:
        return dest.read_bytes()
    with urllib.request.urlopen(url, timeout=120) as response:
        data: bytes = response.read()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise AssertionError(f"{url}: SHA-256 does not match")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return data


def test_the_smoke_test_is_wired_and_says_what_it_waits_for() -> None:
    """Runs always: the config is complete and the reason for any skip is specific."""
    assert set(CONFIG["bundle_asset"]) >= {"url", "sha256"}
    assert 0 < CONFIG["max_abstain_share"] < 1
    for reason in missing():
        assert "release asset" in reason or "manifest" in reason


@pytest.mark.skipif(
    bool(missing()), reason="champion smoke test waits for: " + "; ".join(missing())
)
def test_the_champion_loads_and_scores_fixed_images() -> None:
    from cdm.serving import Service
    from cdm.splits import CLASSES

    asset = CONFIG["bundle_asset"]
    archive = CACHE / "bundle.tar.gz"
    fetch(asset["url"], asset["sha256"], archive)
    folder = CACHE / "bundle"
    if not folder.exists():
        with tarfile.open(archive) as tar:
            tar.extractall(folder, filter="data")
    roots = [p.parent for p in folder.rglob("manifest.json")]
    assert len(roots) == 1, roots
    service = Service.load(roots[0])  # checks every file's SHA-256 and each seed's weights hash

    with (REPO / CONFIG["images_manifest"]).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert 30 <= len(rows) <= 60
    answers: list[dict[str, Any]] = []
    for row in rows:
        image = fetch(row["url"], row["sha256"], CACHE / "images" / f"{row['isic_id']}.jpg")
        answers.append(service.predict(image))
    for a in answers:
        assert a["label"] in CLASSES
        assert abs(sum(a["probabilities"].values()) - 1) < 1e-3
        assert np.isfinite(a["ood_score"])
    abstain = np.mean([a["abstain"] for a in answers])
    assert abstain <= CONFIG["max_abstain_share"], f"{abstain:.0%} abstained on test-split images"
