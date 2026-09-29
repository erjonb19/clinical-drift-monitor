"""The ingest task end to end on a fake ISIC, then the gates on what it landed.

This runs everything the Phase 1 pipeline does except Spark: land, record, rebuild the gate
frame from the landed files, and check the gates on real and on broken input.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from PIL import Image

from cdm.gates import run_all, silver_frame
from cdm.images import DecodeError, decode_resize
from cdm.ingest import CORRUPT_ID, ingest, make_broken
from cdm.splits import lesion_split, split_fingerprint

ISIC_DIAGNOSIS = {"nv": "Nevus", "mel": "Melanoma, NOS", "bcc": "Basal cell carcinoma"}


def rec(isic_id: str, lesion: str, dx: str | None, license: str = "CC-BY-NC") -> dict[str, Any]:
    return {
        "isic_id": isic_id,
        "copyright_license": license,
        "attribution": "Somewhere",
        "metadata": {
            "clinical": {"lesion_id": lesion, "diagnosis_3": ISIC_DIAGNOSIS.get(dx or "")},
            "acquisition": {"image_type": "dermoscopic"},
        },
        "files": {"full": {"url": f"https://isic.test/{isic_id}.jpg"}},
    }


def fake_jpeg(url: str) -> bytes:
    """A valid JPEG whose pixels depend on the URL, so every file is distinct."""
    shade = hashlib.sha256(url.encode()).digest()
    out = io.BytesIO()
    Image.new("RGB", (48, 32), (shade[0], shade[1], shade[2])).save(out, format="JPEG")
    return out.getvalue()


@pytest.fixture
def world(tmp_path: Path) -> tuple[Path, dict[str, Any], Any]:
    rows = [
        {"image_id": f"ISIC_H{dx}{k}{j}", "lesion_id": f"HAM_{dx}{k}", "dx": dx}
        for dx, n in {"nv": 30, "mel": 15, "bcc": 15}.items()
        for k in range(n)
        for j in range(2)
    ]
    ham = pd.DataFrame(rows)
    raw = tmp_path / "raw"
    (raw / "uploads").mkdir(parents=True)
    csv = raw / "uploads" / "HAM10000_metadata.csv"
    ham.to_csv(csv, index=False)
    config = {
        "license_allowed": ["CC-0", "CC-BY", "CC-BY-NC"],
        "ham10000": {
            "collection": 212,
            "metadata_file": csv.name,
            "metadata_md5": hashlib.md5(csv.read_bytes()).hexdigest(),
            "split_seed": 0,
            "split_fingerprint": split_fingerprint(ham, lesion_split(ham, seed=0)),
            "class_counts": ham["dx"].value_counts().to_dict(),
        },
        "sites": {"barcelona": {"collections": [249]}},
    }
    collections = {
        212: [rec(r["image_id"], f"IL_{r['lesion_id']}", r["dx"]) for r in rows]
        + [rec("ISIC_NOT_HAM", "IL_X", "nv")],
        249: [rec(f"ISIC_S{i}", f"IL_S{i // 2}", "nv") for i in range(6)]
        + [rec("ISIC_S_UNLABELLED", "IL_S9", None)],
    }

    def fetch(url: str) -> dict[str, Any]:
        collection = int(url.split("collections=")[1].split("&")[0])
        return {"results": collections[collection], "next": None}

    return raw, config, fetch


def gate_frame(
    raw: Path, root: Path, config: dict[str, Any], image_dirs: list[Path]
) -> pd.DataFrame:
    """What the pipeline computes: every metadata and manifest row, plus the SHA-256 of each
    file actually present (bronze)."""

    def read(folder: Path) -> pd.DataFrame:
        lines = [
            json.loads(line)
            for f in sorted(folder.rglob("*.jsonl"))
            for line in f.read_text().splitlines()
        ]
        return pd.DataFrame(lines)

    landed = pd.DataFrame(
        {
            "source": p.parent.name,
            "isic_id": p.stem,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for d in image_dirs
        for p in d.glob("*/*.jpg")
    )
    seed = config["ham10000"]["split_seed"]
    return silver_frame(read(root / "metadata"), read(root / "manifest"), landed, seed)


def failing(frame: pd.DataFrame, config: dict[str, Any]) -> set[str]:
    return {g.name for g in run_all(frame, config) if g.violations}


def test_ingest_lands_everything_once_and_passes_every_gate(
    world: tuple[Path, dict[str, Any], Any],
) -> None:
    raw, config, fetch = world
    counts = ingest(config, raw, raw.parent / "tmp", 4, "abc1234", fetch, fake_jpeg)
    assert counts == {"ham10000": 120, "barcelona": 7}
    assert not (raw / "images" / "ham10000" / "ISIC_NOT_HAM.jpg").exists()

    def no_network(url: str) -> bytes:
        raise AssertionError("second run must not download")

    ingest(config, raw, raw.parent / "tmp", 4, "abc1234", fetch, no_network)
    frame = gate_frame(raw, raw, config, [raw / "images"])
    assert len(frame) == 127
    assert failing(frame, config) == set()
    assert set(frame.loc[frame.source == "barcelona", "split"]) == {"score"}


@pytest.mark.parametrize("faults", [False, True])
def test_broken_demo_uses_a_small_subset_and_fails_only_through_its_faults(
    world: tuple[Path, dict[str, Any], Any], faults: bool
) -> None:
    raw, config, fetch = world
    ingest(config, raw, raw.parent / "tmp", 4, "abc1234", fetch, fake_jpeg)
    expected = make_broken(raw, raw.parent / "tmp", config, lesions_per_class=10, site_images=3,
                           faults=faults)  # fmt: skip
    broken_config = json.loads((raw / "broken" / "sites.json").read_text(encoding="utf-8"))
    copied = list((raw / "broken" / "images").rglob("*.jpg"))
    assert len(copied) < 80  # 30 lesions x 2 images, 3 site images, plus the faults' files
    frame = gate_frame(raw, raw / "broken", broken_config, [raw / "broken" / "images"])
    assert failing(frame, broken_config) == set(expected)
    assert bool(expected) == faults
    if faults:
        corrupt = raw / "broken" / "images" / "barcelona" / f"{CORRUPT_ID}.jpg"
        with pytest.raises(DecodeError):
            decode_resize(corrupt.read_bytes())
