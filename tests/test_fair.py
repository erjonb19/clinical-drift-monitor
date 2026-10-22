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


def test_citation_and_zenodo_say_mit_for_code_and_nothing_about_data() -> None:
    cff = yaml.safe_load((REPO / "CITATION.cff").read_text(encoding="utf-8"))
    assert cff["type"] == "software" and cff["license"] == "MIT"
    for author in cff["authors"]:
        assert author.get("family-names") or author.get("name")
    zenodo = json.loads((REPO / ".zenodo.json").read_text(encoding="utf-8"))
    assert zenodo["license"] == "MIT" and zenodo["upload_type"] == "software"
    assert "No data is included" in zenodo["description"]


def test_every_source_collection_carries_its_license() -> None:
    meta = json.loads((REPO / "metadata" / "dataset.jsonld").read_text(encoding="utf-8"))
    assert all(b["license"].startswith("https://creativecommons.org/") for b in meta["isBasedOn"])
    assert "MIT" in meta["conditionsOfAccess"]


def test_public_concepts_carry_ids_and_links_but_no_vocabulary_text() -> None:
    from cdm.omop import load_concept_map, public_concepts

    cmap = load_concept_map(REPO / "config" / "omop_concepts.csv")
    table = public_concepts(cmap)
    assert set(table["concept_id"].astype(int)) == {int(c) for c in cmap["concept_id"] if c != 0}
    assert table["concept_name"].str.startswith("https://athena.ohdsi.org/").all()
    assert (table["concept_code"] == "not published").all()


def test_silver_from_isic_falls_back_to_the_image_as_lesion() -> None:
    from cdm.omop import silver_from_isic

    rows = [{"isic_id": "ISIC_1", "isic_lesion_id": "IL_1", "isic_label": "nv"},
            {"isic_id": "ISIC_2", "isic_lesion_id": None, "isic_label": None}]  # fmt: skip
    frame = silver_from_isic(rows)
    assert frame["lesion_id"].tolist() == ["IL_1", "ISIC_2"]
    assert frame["label"].tolist()[0] == "nv"
