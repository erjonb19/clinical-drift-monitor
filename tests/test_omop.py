"""OMOP concept map: its shape, the vocabulary check, and value lookup."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from cdm.omop import (
    TABLES,
    ConceptError,
    build_tables,
    check_against_vocabulary,
    check_tables,
    conform,
    data_dictionary,
    load_concept_map,
    lookup,
    map_values,
    profile,
    unmapped,
)

REPO = Path(__file__).resolve().parents[1]
HEADER = "source_field,scope,source_value,concept_id,expected_domain,match,note\n"


def write_map(tmp_path: Path, rows: str) -> Path:
    path = tmp_path / "map.csv"
    path.write_text(HEADER + rows, encoding="utf-8")
    return path


def test_committed_map_loads_and_every_diagnosis_label_is_mapped() -> None:
    cmap = load_concept_map(REPO / "config" / "omop_concepts.csv")
    labels = set(cmap.loc[cmap["source_field"] == "label", "source_value"])
    assert labels == {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}
    assert (cmap.loc[cmap["source_field"] == "label", "concept_id"] > 0).all()


def test_map_rejects_concept_zero_without_none_and_unexplained_approximations(
    tmp_path: Path,
) -> None:
    with pytest.raises(ConceptError, match="concept 0"):
        load_concept_map(write_map(tmp_path, "label,all,nv,0,Condition,exact,\n"))
    with pytest.raises(ConceptError, match="without a note"):
        load_concept_map(write_map(tmp_path, "label,all,nv,4271013,Condition,broader,\n"))
    with pytest.raises(ConceptError, match="duplicate"):
        load_concept_map(
            write_map(
                tmp_path, "label,all,nv,1,Condition,exact,\nlabel,all,nv,2,Condition,exact,\n"
            )
        )


def vocab(**overrides: str) -> pd.DataFrame:
    row = {"concept_id": "4271013", "domain_id": "Condition", "vocabulary_id": "SNOMED",
           "standard_concept": "S", "invalid_reason": ""} | overrides  # fmt: skip
    return pd.DataFrame([row])


def test_vocabulary_check_catches_missing_nonstandard_invalid_and_wrong_domain(
    tmp_path: Path,
) -> None:
    cmap = load_concept_map(write_map(tmp_path, "label,all,nv,4271013,Condition,exact,\n"))
    assert check_against_vocabulary(cmap, vocab()) == []
    assert check_against_vocabulary(cmap, vocab(concept_id="1"))[0].problem.startswith("not in")
    assert check_against_vocabulary(cmap, vocab(standard_concept=""))[0].problem.startswith(
        "not a standard"
    )
    assert check_against_vocabulary(cmap, vocab(invalid_reason="D"))[0].problem.startswith(
        "invalid"
    )
    assert "expected Condition" in check_against_vocabulary(cmap, vocab(domain_id="Obs"))[0].problem


def test_source_specific_rows_win_and_unknown_values_are_not_silently_zero(tmp_path: Path) -> None:
    rows = (
        "diagnosis_3,sites,SCC NOS,4300118,Condition,broader,may be in situ\n"
        "diagnosis_3,ham10000,SCC NOS,4112745,Condition,source_specific,HAM akiec\n"
        "diagnosis_3,all,Nevus,4271013,Condition,exact,\n"
    )
    table = lookup(load_concept_map(write_map(tmp_path, rows)), "diagnosis_3")
    got = map_values(
        ["ham10000", "msk", "msk", "msk", "msk"],
        ["SCC NOS", "SCC NOS", "Nevus", None, "New ISIC term"],
        table,
    )
    assert got == [4112745, 4300118, 4271013, None, None]


# ---- tables and gates on a synthetic silver frame -------------------------------------------

CMAP = load_concept_map(REPO / "config" / "omop_concepts.csv")


def concept_frame(cmap: pd.DataFrame) -> pd.DataFrame:
    """Stand-in for the private resolved concepts: one row per mapped concept, no SNOMED text."""
    ids = sorted({int(c) for c in cmap["concept_id"] if c != 0})
    return pd.DataFrame(
        {"concept_id": [str(i) for i in ids], "concept_name": "x", "domain_id": "x",
         "vocabulary_id": "x", "concept_class_id": "x", "standard_concept": "S",
         "concept_code": "x", "valid_start_date": "19700101", "valid_end_date": "20991231",
         "invalid_reason": ""}
    )  # fmt: skip


def image(isic_id: str, source: str, lesion: str, **fields: object) -> dict[str, object]:
    base: dict[str, object] = {
        "isic_id": isic_id, "source": source, "lesion_id": lesion, "patient_id": None,
        "sex": "female", "age_approx": 50.0, "image_type": "dermoscopic", "label": "nv",
        "diagnosis_3": "Nevus", "license": "CC-BY", "attribution": "x",
        "local_path": f"/Volumes/x/{isic_id}.jpg",
    }  # fmt: skip
    return base | fields


def silver() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # HAM10000: one lesion, two images, histology; diagnosis from diagnosis_3.
            image(
                "H1",
                "ham10000",
                "HAM_1",
                ham_localization="back",
                ham_dx_type="histo",
                label="mel",
                diagnosis_3="Melanoma, NOS",
            ),
            image(
                "H2",
                "ham10000",
                "HAM_1",
                ham_localization="back",
                ham_dx_type="histo",
                label="mel",
                diagnosis_3="Melanoma Invasive",
                age_approx=55.0,
            ),
            # MSK: one patient with two lesions; one image has no age, no sex, no site.
            image(
                "M1",
                "msk",
                "IL_1",
                patient_id="IP_1",
                anatom_site_1="Trunk",
                anatom_site_2="Posterior trunk",
                diagnosis_confirm_type="single image expert consensus",
            ),
            image(
                "M2",
                "msk",
                "IL_2",
                patient_id="IP_1",
                anatom_site_1="Head and neck",
                diagnosis_confirm_type="histopathology",
                label="bcc",
                diagnosis_3="Basal cell carcinoma",
                sex=None,
                age_approx=None,
            ),
            # A site image with no patient ID and no diagnosis at all.
            image(
                "B1",
                "barcelona",
                "IL_3",
                label=None,
                diagnosis_3=None,
                image_type="clinical: close-up",
            ),
        ]  # fmt: skip
    )


def built(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return build_tables(frame, CMAP, concept_frame(CMAP), "abc1234", "v5 test", date(2026, 1, 1))


def test_tables_follow_the_person_lesion_and_image_rules() -> None:
    t = built(silver())
    person = t["person"].set_index("person_source_value")
    # A patient ID makes one person of two lesions; otherwise one person per lesion.
    assert sorted(person.index) == ["barcelona:lesion:IL_3", "ham10000:lesion:HAM_1",
                                    "msk:patient:IP_1"]  # fmt: skip
    assert person.loc["ham10000:lesion:HAM_1", "year_of_birth"] == 2000 - 50
    assert (t["image_occurrence"]["person_id"].isin(set(t["person"]["person_id"]))).all()
    # One condition per diagnosed lesion; the undiagnosed lesion has none.
    cond = t["condition_occurrence"].set_index("lesion_source_value")
    assert sorted(cond.index) == ["ham10000:HAM_1", "msk:IL_1", "msk:IL_2"]
    # The specific diagnosis wins a tie with NOS; the HAM10000 label is kept.
    assert cond.loc["ham10000:HAM_1", "condition_source_value"] == "Melanoma Invasive"
    assert cond.loc["ham10000:HAM_1", "condition_label"] == "mel"
    assert cond.loc["ham10000:HAM_1", "condition_status_source_value"] == "histo"
    # Histopathology is a procedure; expert consensus is not.
    proc = t["procedure_occurrence"]
    assert sorted(proc["procedure_source_value"].dropna()) == [
        "clinical: close-up", "dermoscopic", "dermoscopic", "dermoscopic", "dermoscopic",
        "histo", "histopathology"]  # fmt: skip
    # The finest body site: anatom_site_2 when recorded, else anatom_site_1.
    sites = t["image_occurrence"].set_index("image_source_value")["anatomic_site_source_value"]
    assert sites.to_dict() == {"H1": "back", "H2": "back", "M1": "Posterior trunk",
                               "M2": "Head and neck", "B1": None}  # fmt: skip
    assert (
        t["image_occurrence"].set_index("image_source_value").loc["B1", "anatomic_site_concept_id"]
        == 0
    )
    assert all(g.violations == 0 for g in check_tables(silver(), t, CMAP)), check_tables(
        silver(), t, CMAP
    )


def test_ids_are_stable_exact_integers_and_uids_are_dicom_form() -> None:
    first, again = built(silver()), built(silver().sample(frac=1, random_state=1))
    for name in ("person", "condition_occurrence", "image_occurrence"):
        key = first[name].columns[0]
        assert set(first[name][key]) == set(again[name][key])
        assert all(type(v) is int and 0 < v < 2**63 for v in first[name][key])
    uid = first["image_occurrence"]["image_study_uid"].iloc[0]
    assert uid.startswith("2.25.") and uid[5:].isdigit() and len(uid) <= 64
    with pytest.raises(TypeError, match="not an integer"):
        conform(pd.DataFrame({"person_id": [1.0]}), "person")


def test_gates_catch_unmapped_values_label_conflicts_and_orphans() -> None:
    frame = silver()
    frame.loc[2, "diagnosis_3"] = "A new ISIC term"
    frame.loc[0, "label"] = "nv"  # HAM_1's two images now disagree on the 7-class label
    t = built(frame)
    t["image_occurrence"].loc[0, "procedure_occurrence_id"] = 1  # an orphan
    failed = {g.name for g in check_tables(frame, t, CMAP) if g.violations}
    # The orphan also leaves one image without its imaging procedure (row counts).
    assert failed == {"omop_every_value_mapped", "omop_one_diagnosis_per_lesion",
                      "omop_foreign_keys_resolve", "omop_row_counts"}  # fmt: skip
    gaps = unmapped(frame, CMAP)
    assert gaps.to_dict("records") == [
        {"field": "diagnosis_3", "source": "msk", "value": "A new ISIC term", "n": 1}
    ]


def test_profile_covers_every_column_of_every_table() -> None:
    t = built(silver())
    prof = profile(t)
    assert len(prof) == sum(len(cols) for cols in TABLES.values())
    assert set(prof["table_name"]) == set(TABLES)


def test_committed_data_dictionary_is_current() -> None:
    """docs/data_dictionary.md is generated: it must match the column definitions, the concept
    map and the committed build (rerun scripts/omop_build_report.py --render-only)."""
    build_file = REPO / "results" / "research" / "a_omop" / "build.json"
    build = json.loads(build_file.read_text(encoding="utf-8")) if build_file.exists() else None
    committed = (REPO / "docs" / "data_dictionary.md").read_text(encoding="utf-8")
    assert committed == data_dictionary(CMAP, build)
    for table, cols in TABLES.items():
        assert f"## omop_{table}" in committed
        assert all(f"| `{c.name}` |" in committed for c in cols)
