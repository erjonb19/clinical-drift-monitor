"""OMOP concept map: its shape, the vocabulary check, and value lookup."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cdm.omop import (
    ConceptError,
    check_against_vocabulary,
    load_concept_map,
    lookup,
    map_values,
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
