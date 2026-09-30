"""ISIC sources: pagination, diagnosis mapping, whole-lesion selection, landing. No network."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from cdm.sources import (
    Cap,
    Site,
    collection_records,
    fingerprint,
    flatten,
    ham_rows,
    land,
    load_sites,
    map_diagnosis,
    select_whole_lesions,
    site_rows,
)
from cdm.splits import DataError

CONFIG = Path(__file__).resolve().parents[1] / "config" / "sites.json"


def record(isic_id: str, lesion: str | None = None, d2: str | None = None, d3: str | None = None,
           license: str = "CC-BY") -> dict[str, Any]:  # fmt: skip
    clinical: dict[str, Any] = {"diagnosis_2": d2, "diagnosis_3": d3}
    if lesion:
        clinical["lesion_id"] = lesion
    return {
        "isic_id": isic_id,
        "copyright_license": license,
        "attribution": "Somewhere",
        "metadata": {"clinical": clinical, "acquisition": {"image_type": "dermoscopic"}},
        "files": {"full": {"url": f"https://example.test/{isic_id}.jpg"}},
    }


def fake_fetch(pages: dict[int, list[dict[str, Any]]]) -> Any:
    """Serve each collection as two pages to exercise pagination."""

    def fetch(url: str) -> dict[str, Any]:
        collection = int(url.split("collections=")[1].split("&")[0])
        records = pages[collection]
        if "page=2" in url:
            return {"results": records[1:], "next": None}
        return {"results": records[:1], "next": url + "&page=2"}

    return fetch


def test_collection_records_follows_pagination() -> None:
    pages = {7: [record("ISIC_1"), record("ISIC_2"), record("ISIC_3")]}
    assert [r["isic_id"] for r in collection_records(7, fake_fetch(pages))] == [
        "ISIC_1",
        "ISIC_2",
        "ISIC_3",
    ]


@pytest.mark.parametrize(
    ("d2", "d3", "label"),
    [
        ("Benign melanocytic proliferations", "Nevus", "nv"),
        ("Malignant melanocytic proliferations (Melanoma)", "Melanoma in situ", "mel"),
        ("Malignant epidermal proliferations", "Squamous cell carcinoma, NOS", "akiec"),
        ("Benign epidermal proliferations", "Solar lentigo", "bkl"),
        ("Benign soft tissue proliferations - Vascular", None, "vasc"),
        ("Hemorrhagic lesions", "Hemorrhage", "vasc"),
        ("Malignant melanocytic proliferations (Melanoma)", "Melanoma metastasis", None),
        ("Benign soft tissue proliferations - Fibro-histiocytic", "Scar", None),
        (None, None, None),
    ],
)
def test_map_diagnosis(d2: str | None, d3: str | None, label: str | None) -> None:
    assert map_diagnosis(d2, d3) == label


def test_flatten_tolerates_missing_metadata() -> None:
    bare = {"isic_id": "ISIC_9", "files": {"full": {"url": "u"}}}
    row = flatten(bare, "site", 1)
    assert row["isic_id"] == "ISIC_9" and row["isic_label"] is None and row["license"] is None


def test_whole_lesion_selection_never_splits_a_lesion_and_is_seeded() -> None:
    rows = [
        {"isic_id": f"ISIC_{lesion}_{k}", "isic_lesion_id": f"IL_{lesion}"}
        for lesion in range(40)
        for k in range(1 + lesion % 4)
    ]
    a = select_whole_lesions(rows, max_images=30, seed=0)
    assert len(a) <= 30
    chosen = {r["isic_lesion_id"] for r in a}
    for lesion in chosen:  # every image of a chosen lesion is included
        assert sum(r["isic_lesion_id"] == lesion for r in a) == sum(
            r["isic_lesion_id"] == lesion for r in rows
        )
    assert fingerprint(r["isic_id"] for r in a) == fingerprint(
        r["isic_id"] for r in select_whole_lesions(rows, 30, seed=0)
    )
    assert fingerprint(r["isic_id"] for r in a) != fingerprint(
        r["isic_id"] for r in select_whole_lesions(rows, 30, seed=1)
    )


def test_site_rows_deduplicates_and_enforces_the_recorded_selection() -> None:
    pages = {
        1: [record("ISIC_1", "IL_1"), record("ISIC_2", "IL_2")],
        2: [record("ISIC_2", "IL_2"), record("ISIC_3", "IL_3")],
    }
    rows = site_rows(Site("s", (1, 2)), fake_fetch(pages))
    assert [r["isic_id"] for r in rows] == ["ISIC_1", "ISIC_2", "ISIC_3"]

    capped = site_rows(Site("s", (1, 2), Cap(max_images=2, seed=0)), fake_fetch(pages))
    good = Cap(max_images=2, seed=0, fingerprint=fingerprint(r["isic_id"] for r in capped))
    assert site_rows(Site("s", (1, 2), good), fake_fetch(pages)) == capped
    with pytest.raises(DataError, match="selection fingerprint"):
        site_rows(Site("s", (1, 2), Cap(2, 0, fingerprint="0000000000000000")), fake_fetch(pages))


def test_ham_rows_take_labels_and_lesions_from_ham_metadata() -> None:
    meta = pd.DataFrame(
        {
            "image_id": ["ISIC_1", "ISIC_2"],
            "lesion_id": ["HAM_1", "HAM_1"],
            "dx": ["nv", "nv"],
            "dataset": ["rosendahl", "rosendahl"],
        }  # fmt: skip
    )
    pages = {212: [record("ISIC_1", "IL_9"), record("ISIC_2", "IL_9"), record("ISIC_99")]}
    rows = ham_rows(meta, 212, fake_fetch(pages))
    assert [(r["isic_id"], r["lesion_id"], r["label"]) for r in rows] == [
        ("ISIC_1", "HAM_1", "nv"),
        ("ISIC_2", "HAM_1", "nv"),
    ]
    with pytest.raises(DataError, match="missing from ISIC"):
        ham_rows(meta, 212, fake_fetch({212: [record("ISIC_1")]}))


def test_land_downloads_once_and_records_checksums(tmp_path: Path) -> None:
    row = {"isic_id": "ISIC_1", "source": "s", "url": "https://example.test/ISIC_1.jpg"}
    body = b"jpeg bytes"
    (tmp_path / "tmp").mkdir()
    first = land(row, tmp_path / "vol", tmp_path / "tmp", opener=lambda url: body)
    assert first["downloaded"] and first["sha256"] == hashlib.sha256(body).hexdigest()
    assert (tmp_path / "vol" / "s" / "ISIC_1.jpg").read_bytes() == body
    assert not any((tmp_path / "tmp").iterdir())

    def no_network(url: str) -> bytes:
        raise AssertionError("a landed file must not be downloaded again")

    again = land(row, tmp_path / "vol", tmp_path / "tmp", opener=no_network)
    assert not again["downloaded"] and again["sha256"] == first["sha256"]


def test_committed_config_parses_and_pins_the_barcelona_selection() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    sites = {s.name: s for s in load_sites(config)}
    assert set(sites) == {"barcelona", "buenos_aires", "msk", "pad_ufes"}
    cap = sites["barcelona"].cap
    assert cap is not None and cap.max_images == 5000 and cap.fingerprint is not None
    assert config["ham10000"]["split_fingerprint"] == "cc2b196cd5bf58ad"


def test_land_reuses_a_known_manifest_row_without_reading_the_file(tmp_path: Path) -> None:
    """Reruns must not re-read every landed file from the volume (about six per second);
    the pipeline's own hashing and files_match_manifest gate catch a changed file."""
    row = {"isic_id": "ISIC_1", "source": "s", "url": "https://example.test/ISIC_1.jpg"}
    (tmp_path / "tmp").mkdir()
    first = land(row, tmp_path / "vol", tmp_path / "tmp", opener=lambda url: b"original")
    (tmp_path / "vol" / "s" / "ISIC_1.jpg").write_bytes(b"changed after landing")
    reused = land(row, tmp_path / "vol", tmp_path / "tmp", opener=lambda url: b"", known=first)
    assert reused["sha256"] == first["sha256"] and not reused["downloaded"]
    rehashed = land(row, tmp_path / "vol", tmp_path / "tmp", opener=lambda url: b"")
    assert rehashed["sha256"] == hashlib.sha256(b"changed after landing").hexdigest()
