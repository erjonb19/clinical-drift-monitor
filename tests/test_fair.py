"""FAIR metadata stays in step with the code and config: every OMOP table and every source."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from cdm.omop import TABLES

REPO = Path(__file__).resolve().parents[1]


def test_dataset_metadata_covers_every_table_and_source() -> None:
    meta = json.loads((REPO / "metadata" / "dataset.jsonld").read_text(encoding="utf-8"))
    assert meta["@type"] == "Dataset"
    for key in ("name", "description", "url", "creator", "license", "conditionsOfAccess"):
        assert meta[key], key
    assert {v["name"] for v in meta["variableMeasured"]} == set(TABLES)
    config = json.loads((REPO / "config" / "sites.json").read_text(encoding="utf-8"))
    collections = {config["ham10000"]["collection"]}
    collections |= {c for site in config["sites"].values() for c in site["collections"]}
    cited = " ".join(str(b.get("url", "")) for b in meta["isBasedOn"])
    assert all(f"/collections/{c}/" in cited for c in collections)
    assert any("10.7910/DVN/DBW86T" in b.get("identifier", "") for b in meta["isBasedOn"])


def test_citation_file_is_valid_cff() -> None:
    cff = yaml.safe_load((REPO / "CITATION.cff").read_text(encoding="utf-8"))
    assert cff["cff-version"] == "1.2.0"
    assert cff["title"] and cff["authors"] and cff["message"]
