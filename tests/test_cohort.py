"""The governed cohort service: allowlist, approval by hash, suppression, audit, and counts."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from cdm.cohort import (
    COMPLEMENT,
    SUPPRESSED,
    Approvals,
    AuditLog,
    CohortService,
    PhenotypeError,
    Refused,
    compile_sql,
    duckdb_engine,
    load,
    parse,
    suppress,
    suppress_attrition,
    verify_chain,
)
from cdm.omop import build_tables, load_concept_map

REPO = Path(__file__).resolve().parents[1]
CMAP = load_concept_map(REPO / "config" / "omop_concepts.csv")
MEL, MEL_INSITU, NEVUS, HISTO = 4162276, 40391314, 4271013, 43531286

BASE = {"name": "mel", "version": 1, "title": "t",
        "index": {"condition_concept_ids": [MEL, MEL_INSITU]}}  # fmt: skip


# ---- the allowlisted language ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"sql": "SELECT * FROM person"}, "unknown keys"),
        ({"index": {"condition_concept_ids": ["4162276; DROP TABLE person"]}}, "positive integers"),
        ({"index": {"condition_concept_ids": [MEL], "age_min": "18 OR 1=1"}}, "integer age"),
        ({"include": [{"person_source_value": [1]}]}, "not an allowed criterion"),
        (
            {"include": [{"procedure_concept_ids": [HISTO], "gender_concept_ids": [1]}]},
            "exactly one",
        ),
        ({"breakdown": ["person_source_value"]}, "breakdown"),
        ({"version": 0}, "positive integer"),
    ],
)
def test_anything_outside_the_allowlist_is_refused(change: dict[str, object], message: str) -> None:
    with pytest.raises(PhenotypeError, match=message):
        parse(BASE | change)


def test_hash_follows_meaning_not_formatting(tmp_path: Path) -> None:
    folder = tmp_path / "mel"
    folder.mkdir()
    (folder / "v1.yaml").write_text(
        "name: mel\nversion: 1\ntitle: t\nindex: {condition_concept_ids: [40391314, 4162276]}\n"
    )
    a = load(folder / "v1.yaml")
    (folder / "v1.yaml").write_text(
        "# a comment\nname: mel\nversion: 1\ntitle: t\nindex:\n"
        "  condition_concept_ids: [4162276, 40391314]\n"
    )
    assert load(folder / "v1.yaml").sha256 == a.sha256
    assert parse(BASE | {"include": [{"procedure_concept_ids": [HISTO]}]}).sha256 != a.sha256
    (folder / "v2.yaml").write_text((folder / "v1.yaml").read_text())  # says version 1
    with pytest.raises(PhenotypeError, match="expected mel/v1.yaml"):
        load(folder / "v2.yaml")


def test_committed_phenotypes_are_valid_and_use_mapped_concepts() -> None:
    mapped = {int(c) for c in CMAP["concept_id"] if c != 0}
    files = sorted((REPO / "phenotypes").glob("*/v*.yaml"))
    assert len(files) >= 3
    for path in files:
        d = load(path).definition
        ids = list(d["index"]["condition_concept_ids"])
        for criterion in d["include"] + d["exclude"]:
            ids += next(iter(criterion.values()))
        assert set(ids) <= mapped, path


# ---- small cells -----------------------------------------------------------------------------


def test_small_cells_and_the_cell_that_would_reveal_one_are_hidden() -> None:
    assert suppress(pd.Series([5, 50, 30]), total=85) == [SUPPRESSED, "50", COMPLEMENT]
    assert suppress(pd.Series([5, 7, 30]), total=42) == [SUPPRESSED, SUPPRESSED, "30"]
    assert suppress(pd.Series([0, 50, 12]), total=62) == ["0", "50", "12"]
    assert suppress(pd.Series([5, 50])) == [SUPPRESSED, "50"]  # no total published
    # 100 -> 95 drops 5 people: 95 is hidden too; 95 -> 40 drops 55, so 40 is shown.
    assert suppress_attrition(pd.Series([100, 95, 40, 3])) == ["100", COMPLEMENT, "40", SUPPRESSED]


# ---- approval, audit and counts on real OMOP tables (built from synthetic silver) ----------


def silver(n_people: int = 40) -> pd.DataFrame:
    """Lesion-level people: every 4th has melanoma (in situ every 8th), histology on evens."""
    rows = []
    for i in range(n_people):
        mel = i % 4 == 0
        rows.append({
            "isic_id": f"I{i}", "source": "ham10000", "lesion_id": f"L{i}", "patient_id": None,
            "sex": "male", "age_approx": 15.0 + i, "image_type": "dermoscopic",
            "label": "mel" if mel else "nv",
            "diagnosis_3": ("Melanoma in situ" if i % 8 == 0 else "Melanoma, NOS") if mel
            else "Nevus",
            "ham_dx_type": "histo" if i % 2 == 0 else "follow_up", "ham_localization": "back",
            "license": "CC-BY-NC", "attribution": "x", "local_path": None,
        })  # fmt: skip
    return pd.DataFrame(rows)


def omop(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    concepts = pd.DataFrame(
        {"concept_id": sorted({str(c) for c in CMAP["concept_id"] if c != 0})}
    ).assign(concept_name="x", domain_id="x", vocabulary_id="x", concept_class_id="x",
             standard_concept="S", concept_code="x", valid_start_date="19700101",
             valid_end_date="20991231", invalid_reason="")  # fmt: skip
    tables = build_tables(frame, CMAP, concepts, "t", "v", date(2026, 1, 1))
    # DuckDB reads typed columns; the pipeline's object columns become typed in Spark.
    return {k: v.infer_objects() for k, v in tables.items()}


def make_service(tmp_path: Path, n_people: int = 100) -> CohortService:
    engine, tables = duckdb_engine(omop(silver(n_people)))
    return CohortService(engine, tables, Approvals(tmp_path / "approvals.jsonl"),
                         AuditLog(tmp_path / "audit.jsonl"))  # fmt: skip


def write(tmp_path: Path, definition: dict[str, object]) -> Path:
    import yaml

    folder = tmp_path / str(definition["name"])
    folder.mkdir(exist_ok=True)
    path = folder / f"v{definition['version']}.yaml"
    path.write_text(yaml.safe_dump(definition))
    return path


def test_only_an_approved_exact_definition_runs_and_everything_is_audited(tmp_path: Path) -> None:
    svc = make_service(tmp_path)
    path = write(tmp_path, BASE)
    with pytest.raises(Refused, match="pending"):
        svc.run(path, "fellow")
    svc.decide(path, "escalate", "data engineer", "needs the PI")
    with pytest.raises(Refused, match="pending"):  # escalation is not a verdict
        svc.run(path, "fellow")
    svc.decide(path, "approve", "PI", "public data, counts only")
    assert svc.run(path, "fellow").attrition[0]["persons"] == "25"  # i = 0, 4, ..., 96
    # Changing the definition after approval needs a new approval.
    write(tmp_path, BASE | {"index": {"condition_concept_ids": [MEL]}})
    with pytest.raises(Refused, match="pending"):
        svc.run(path, "fellow")
    write(tmp_path, BASE)
    svc.decide(path, "reject", "PI", "withdrawn")
    with pytest.raises(Refused, match="reject"):
        svc.run(path, "fellow")
    events = [e["event"] for e in svc.audit.entries()]
    assert events == ["refused", "decision", "refused", "decision", "ran", "refused", "decision",
                      "refused"]  # fmt: skip
    assert verify_chain(svc.audit.entries()) == []


def test_counts_match_the_data_and_small_cells_are_suppressed(tmp_path: Path) -> None:
    svc = make_service(tmp_path, n_people=40)
    definition = BASE | {
        "index": {"condition_concept_ids": [MEL, MEL_INSITU], "age_min": 30},
        "include": [{"procedure_concept_ids": [HISTO]}],
        "breakdown": ["age_band"],
    }
    path = write(tmp_path, definition)
    svc.decide(path, "approve", "PI", "ok")
    result = svc.run(path, "fellow")
    # Melanoma people are i = 0, 4, ..., 36 (age 15 + i); aged 30+ means i >= 16: 16..36,
    # six people, all even so all with histology. Six is a small cell.
    assert [r["persons"] for r in result.attrition] == [SUPPRESSED, SUPPRESSED]
    # Ages 31, 35, 39, 43, 47, 51: bands 30 (3), 40 (2), 50 (1), all small.
    assert {r["age_band"]: r["persons"] for r in result.breakdown} == {
        30: SUPPRESSED, 40: SUPPRESSED, 50: SUPPRESSED}  # fmt: skip
    # The SQL itself returns the exact counts the suppression hides.
    raw = svc.engine(compile_sql(load(path), svc.tables)["attrition"])
    assert raw["persons"].tolist() == [6, 6]


def test_exclusion_and_a_larger_cohort_show_unsuppressed_counts(tmp_path: Path) -> None:
    svc = make_service(tmp_path)
    definition = {"name": "nv", "version": 1, "title": "t",
                  "index": {"condition_concept_ids": [NEVUS]},
                  "exclude": [{"procedure_concept_ids": [HISTO]}]}  # fmt: skip
    path = write(tmp_path, definition)
    svc.decide(path, "approve", "PI", "ok")
    result = svc.run(path, "fellow")
    # 75 nevus people (i % 4 in 1, 2, 3); the 25 with i % 4 == 2 had histology.
    assert [r["persons"] for r in result.attrition] == ["75", "50"]


def test_audit_chain_detects_an_edited_or_deleted_line(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")
    for i in range(3):
        log.append("ran", phenotype=f"p{i}")
    lines = log.path.read_text().splitlines()
    edited = json.loads(lines[1]) | {"phenotype": "other"}
    entries = [json.loads(lines[0]), edited, json.loads(lines[2])]
    assert any("changed" in p for p in verify_chain(entries))
    assert any("does not follow" in p for p in verify_chain([json.loads(lines[0]),
                                                             json.loads(lines[2])]))  # fmt: skip
