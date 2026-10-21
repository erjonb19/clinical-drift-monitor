"""OMOP CDM v5.4-style tables from silver, with the OHDSI medical imaging extension proposal's
``image_occurrence``. No Spark or torch: the pipeline calls these through ``applyInPandas``,
and CI tests them on synthetic frames.

Concepts come from ``config/omop_concepts.csv``: one row per source value, with the OMOP
``concept_id`` chosen from an Athena vocabulary download and how well it matches. Only OHDSI
concept IDs are in the repository. SNOMED codes and names are SNOMED International content
under the Affiliate License, so they are resolved from the licensed download at build time,
never committed (docs/research_extension_plan.md, licenses). ``concept_id`` 0 is OMOP's
"no matching concept"; every such row says why.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cdm.gates import Gate

MATCHES = ("exact", "synonym", "broader", "approximate", "class_lca", "source_specific", "none")
SCOPES = ("all", "ham10000", "sites")
FIELDS = (
    "diagnosis_3",
    "label",
    "ham_localization",
    "anatom_site_1",
    "anatom_site_2",
    "ham_dx_type",
    "diagnosis_confirm_type",
    "image_type",
    "condition_type",
    "sex",
    "modality",
    "condition_status",
    "cdm_version",
)
# Vocabularies a mapped concept may come from.
VOCABULARIES = {"SNOMED", "Type Concept", "Gender", "DICOM", "Condition Status", "CDM"}


class ConceptError(RuntimeError):
    """The concept map or the vocabulary is not what the build expects."""


def load_concept_map(path: Path) -> pd.DataFrame:
    """The committed concept map, checked for shape: known fields, scopes and match types; one
    row per (field, scope, value); concept 0 exactly when the match is ``none``; a reason on
    every row that is not an exact match."""
    cmap = pd.read_csv(path, dtype=str, keep_default_na=False)
    problems: list[str] = []
    for col, allowed in (("source_field", FIELDS), ("scope", SCOPES), ("match", MATCHES)):
        bad = sorted(set(cmap[col]) - set(allowed))
        if bad:
            problems.append(f"unknown {col}: {bad}")
    dupes = cmap[cmap.duplicated(["source_field", "scope", "source_value"], keep=False)]
    if len(dupes):
        problems.append(f"duplicate rows: {dupes['source_value'].tolist()}")
    zero = cmap["concept_id"] == "0"
    if (zero != (cmap["match"] == "none")).any():
        problems.append("concept 0 must go with match 'none', and only with it")
    unexplained = cmap[(cmap["match"] != "exact") & (cmap["note"].str.strip() == "")]
    if len(unexplained):
        problems.append(f"non-exact rows without a note: {unexplained['source_value'].tolist()}")
    if not cmap["concept_id"].str.fullmatch(r"\d+").all():
        problems.append("concept_id must be a non-negative integer")
    if problems:
        raise ConceptError(f"{path}: " + "; ".join(problems))
    return cmap.assign(concept_id=cmap["concept_id"].astype(int))


@dataclass(frozen=True)
class ConceptCheck:
    concept_id: int
    problem: str


def check_against_vocabulary(cmap: pd.DataFrame, concepts: pd.DataFrame) -> list[ConceptCheck]:
    """Every mapped concept must exist in the vocabulary download, be a valid standard concept
    from an expected vocabulary, and sit in the domain its row expects. ``concepts`` has the
    Athena CONCEPT.csv columns (all strings)."""
    vocab = concepts.set_index("concept_id")
    out: list[ConceptCheck] = []
    mapped = cmap[cmap["concept_id"] != 0]
    ids, domains = mapped["concept_id"].astype(int), mapped["expected_domain"]
    for cid, domain in zip(ids, domains, strict=True):
        key = str(cid)
        if key not in vocab.index:
            out.append(ConceptCheck(cid, "not in the vocabulary download"))
            continue
        c = vocab.loc[key]
        if c["standard_concept"] != "S":
            out.append(ConceptCheck(cid, "not a standard concept"))
        if c["invalid_reason"]:
            out.append(ConceptCheck(cid, f"invalid ({c['invalid_reason']})"))
        if c["vocabulary_id"] not in VOCABULARIES:
            out.append(ConceptCheck(cid, f"vocabulary {c['vocabulary_id']}"))
        if c["domain_id"] != domain:
            wrong = f"domain {c['domain_id']}, expected {domain}"
            out.append(ConceptCheck(cid, wrong))
    return out


def lookup(cmap: pd.DataFrame, field: str) -> dict[tuple[str, str], int]:
    """(scope, source value) -> concept_id for one source field."""
    rows = cmap[cmap["source_field"] == field]
    return {
        (str(s), str(v)): int(c)
        for s, v, c in zip(rows["scope"], rows["source_value"], rows["concept_id"], strict=True)
    }


def map_values(
    sources: Iterable[str], values: Iterable[object], table: Mapping[tuple[str, str], int]
) -> list[int | None]:
    """Concept per value: a source-specific row (``ham10000`` or ``sites``) wins over ``all``.
    A missing value gives None (not recorded); a value with no row also gives None, which the
    ``every_value_mapped`` check reports, so a new ISIC term cannot slip through as 0."""
    out: list[int | None] = []
    for source, value in zip(sources, values, strict=True):
        if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
            out.append(None)
            continue
        scope = "ham10000" if source == "ham10000" else "sites"
        hit = table.get((scope, str(value)), table.get(("all", str(value))))
        out.append(hit)
    return out


# ---- tables ---------------------------------------------------------------------------------

# HAM10000 and ISIC give no acquisition dates. Every OMOP date is this placeholder, and
# year_of_birth is its year minus the approximate age: neither is a real date.
ANCHOR_DATE = date(2000, 1, 1)
HAM = "ham10000"
# Fixed namespace for the surrogate DICOM UIDs (the UUID-derived "2.25." form).
_UID_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL, "https://github.com/erjonb19/clinical-drift-monitor"
)


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str  # Spark SQL type
    required: bool  # OMOP NOT NULL, enforced by the omop_required_not_null gate
    source: str  # where the value comes from: silver column, constant or rule
    note: str = ""


_PLACEHOLDER = "placeholder date 2000-01-01; not a real date"
_UNUSED = "not recorded in the source"
_EXT = "extension column, not in OMOP CDM v5.4"
_BY_PERSON = "the person's person_id"
_REGISTRY = "concept map condition_type (Registry)"


def _unused(name: str, dtype: str) -> Column:
    return Column(name, dtype, False, "always null", _UNUSED)


def _zero(name: str) -> Column:
    return Column(name, "INT", False, "constant 0", _UNUSED)


# Column order and types follow OMOP CDM v5.4; image_occurrence follows the OHDSI medical
# imaging extension proposal (Park et al., JAMIA 2025). IDs are BIGINT because they are
# 60-bit hashes of source keys, stable across rebuilds.
TABLES: dict[str, tuple[Column, ...]] = {
    "person": (
        Column("person_id", "BIGINT", True, "hash of person_source_value"),
        Column("gender_concept_id", "INT", True, "sex via concept map field sex; 0 if missing"),
        Column(
            "year_of_birth",
            "INT",
            False,
            "2000 minus the person's lowest age_approx",
            "OMOP requires it; null where no age is recorded; artificial (placeholder year)",
        ),
        _unused("month_of_birth", "INT"),
        _unused("day_of_birth", "INT"),
        _unused("birth_datetime", "TIMESTAMP"),
        Column("race_concept_id", "INT", True, "constant 0", _UNUSED),
        Column("ethnicity_concept_id", "INT", True, "constant 0", _UNUSED),
        _unused("location_id", "BIGINT"),
        _unused("provider_id", "BIGINT"),
        _unused("care_site_id", "BIGINT"),
        Column(
            "person_source_value",
            "STRING",
            True,
            "source:patient:<patient_id> where ISIC records a patient, else "
            "source:lesion:<lesion_id>",
            "one person per lesion where there is no patient ID: an assumption that "
            "over-counts people",
        ),
        Column("gender_source_value", "STRING", False, "most common sex over the person's images"),
        _zero("gender_source_concept_id"),
        _unused("race_source_value", "STRING"),
        _zero("race_source_concept_id"),
        _unused("ethnicity_source_value", "STRING"),
        _zero("ethnicity_source_concept_id"),
    ),
    "observation_period": (
        Column("observation_period_id", "BIGINT", True, "hash of person_source_value"),
        Column("person_id", "BIGINT", True, _BY_PERSON),
        Column("observation_period_start_date", "DATE", True, "constant", _PLACEHOLDER),
        Column("observation_period_end_date", "DATE", True, "constant", _PLACEHOLDER),
        Column("period_type_concept_id", "INT", True, _REGISTRY),
    ),
    "condition_occurrence": (
        Column("condition_occurrence_id", "BIGINT", True, "hash of source:lesion_id"),
        Column("person_id", "BIGINT", True, _BY_PERSON),
        Column(
            "condition_concept_id",
            "INT",
            True,
            "diagnosis_3 via concept map; the 7-class label where diagnosis_3 is missing",
        ),
        Column("condition_start_date", "DATE", True, "constant", _PLACEHOLDER),
        _unused("condition_start_datetime", "TIMESTAMP"),
        _unused("condition_end_date", "DATE"),
        _unused("condition_end_datetime", "TIMESTAMP"),
        Column("condition_type_concept_id", "INT", True, _REGISTRY),
        Column(
            "condition_status_concept_id",
            "INT",
            False,
            "confirmation method via concept map field condition_status; 0 if not recorded",
        ),
        _unused("stop_reason", "STRING"),
        _unused("provider_id", "BIGINT"),
        _unused("visit_occurrence_id", "BIGINT"),
        _unused("visit_detail_id", "BIGINT"),
        Column("condition_source_value", "STRING", False, "diagnosis_3, else the 7-class label"),
        _zero("condition_source_concept_id"),
        Column(
            "condition_status_source_value",
            "STRING",
            False,
            "HAM10000 dx_type; ISIC diagnosis_confirm_type at the sites",
        ),
        Column("condition_label", "STRING", False, "the project's 7-class label", _EXT),
        Column("lesion_source_value", "STRING", True, "source:lesion_id", _EXT),
    ),
    "procedure_occurrence": (
        Column(
            "procedure_occurrence_id",
            "BIGINT",
            True,
            "hash of source:isic_id (imaging) or of source:lesion_id (histopathology)",
        ),
        Column("person_id", "BIGINT", True, _BY_PERSON),
        Column(
            "procedure_concept_id",
            "INT",
            True,
            "imaging: image_type via concept map; histopathology: dx_type or "
            "diagnosis_confirm_type via concept map",
        ),
        Column("procedure_date", "DATE", True, "constant", _PLACEHOLDER),
        _unused("procedure_datetime", "TIMESTAMP"),
        _unused("procedure_end_date", "DATE"),
        _unused("procedure_end_datetime", "TIMESTAMP"),
        Column("procedure_type_concept_id", "INT", True, _REGISTRY),
        _zero("modifier_concept_id"),
        _unused("quantity", "INT"),
        _unused("provider_id", "BIGINT"),
        _unused("visit_occurrence_id", "BIGINT"),
        _unused("visit_detail_id", "BIGINT"),
        Column(
            "procedure_source_value",
            "STRING",
            False,
            "image_type, or the confirmation method for a histopathology row",
        ),
        _zero("procedure_source_concept_id"),
        _unused("modifier_source_value", "STRING"),
    ),
    "image_occurrence": (
        Column("image_occurrence_id", "BIGINT", True, "hash of source:isic_id"),
        Column("person_id", "BIGINT", True, _BY_PERSON),
        Column("procedure_occurrence_id", "BIGINT", True, "the image's imaging procedure"),
        _unused("visit_occurrence_id", "BIGINT"),
        Column(
            "anatomic_site_concept_id",
            "INT",
            False,
            "finest body site via concept map: HAM10000 localization; ISIC anatom_site_2, "
            "else anatom_site_1; 0 if not recorded",
        ),
        Column("wadors_uri", "STRING", False, "always null", "no DICOMweb server"),
        Column("local_path", "STRING", False, "the image file in the Unity Catalog volume"),
        Column("image_occurrence_date", "DATE", True, "constant", _PLACEHOLDER),
        Column(
            "image_study_uid",
            "STRING",
            True,
            "2.25.<UUID5 of isic_id>",
            "no DICOM exists: a deterministic surrogate, not a real study UID",
        ),
        Column(
            "image_series_uid",
            "STRING",
            True,
            "2.25.<UUID5 of isic_id and 'series'>",
            "no DICOM exists: a deterministic surrogate, not a real series UID",
        ),
        Column("modality_concept_id", "INT", True, "image_type via concept map field modality"),
        Column("image_source_value", "STRING", True, "isic_id", _EXT),
        Column("anatomic_site_source_value", "STRING", False, "the body site value used", _EXT),
        Column("condition_occurrence_id", "BIGINT", False, "the lesion's condition", _EXT),
        Column("license", "STRING", False, "ISIC copyright_license of the image", _EXT),
        Column("attribution", "STRING", False, "ISIC attribution of the image", _EXT),
    ),
    "cdm_source": (
        Column("cdm_source_name", "STRING", True, "constant"),
        Column("cdm_source_abbreviation", "STRING", True, "constant"),
        Column("cdm_holder", "STRING", True, "constant"),
        Column("source_description", "STRING", False, "constant"),
        Column("source_documentation_reference", "STRING", False, "constant: this dictionary"),
        Column("cdm_etl_reference", "STRING", False, "git commit of the deployed code"),
        Column("source_release_date", "DATE", True, "build date"),
        Column("cdm_release_date", "DATE", True, "build date"),
        Column("cdm_version", "STRING", False, "constant v5.4"),
        Column("cdm_version_concept_id", "INT", True, "concept map field cdm_version"),
        Column("vocabulary_version", "STRING", True, "Athena release, from config/sites.json"),
    ),
    "concept": (
        Column("concept_id", "INT", True, "Athena CONCEPT.csv: the concepts in the concept map"),
        Column("concept_name", "STRING", True, "Athena", "SNOMED content: workspace only"),
        Column("domain_id", "STRING", True, "Athena"),
        Column("vocabulary_id", "STRING", True, "Athena"),
        Column("concept_class_id", "STRING", True, "Athena"),
        Column("standard_concept", "STRING", False, "Athena"),
        Column("concept_code", "STRING", True, "Athena", "SNOMED content: workspace only"),
        Column("valid_start_date", "DATE", True, "Athena"),
        Column("valid_end_date", "DATE", True, "Athena"),
        Column("invalid_reason", "STRING", False, "Athena"),
    ),
}
PRIMARY_KEYS = {name: cols[0].name for name, cols in TABLES.items() if name != "cdm_source"}
# (column, parent table, parent column); a null value is not checked (optional link).
FOREIGN_KEYS = {
    "observation_period": [("person_id", "person", "person_id")],
    "condition_occurrence": [("person_id", "person", "person_id")],
    "procedure_occurrence": [("person_id", "person", "person_id")],
    "image_occurrence": [
        ("person_id", "person", "person_id"),
        ("procedure_occurrence_id", "procedure_occurrence", "procedure_occurrence_id"),
        ("condition_occurrence_id", "condition_occurrence", "condition_occurrence_id"),
    ],
}
# Columns holding a concept ID; every non-zero one must be in the concept table.
CONCEPT_COLUMNS = {
    name: [c.name for c in cols if c.name.endswith("concept_id") and c.name != "concept_id"]
    for name, cols in TABLES.items()
}


def spark_schema(table: str) -> str:
    return ", ".join(f"{c.name} {c.dtype}" for c in TABLES[table])


def stable_id(*parts: str) -> int:
    """A 60-bit ID from source keys, the same on every rebuild (fits a signed BIGINT)."""
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:15], 16)


def dicom_uid(*parts: str) -> str:
    return f"2.25.{uuid.uuid5(_UID_NAMESPACE, '|'.join(parts)).int}"


def _blank(value: object) -> bool:
    return value is None or value == "" or (isinstance(value, float) and pd.isna(value))


def _text(value: object) -> str | None:
    return None if _blank(value) else str(value)


def _col(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame:
        return frame[name].map(_text)
    return pd.Series(None, index=frame.index, dtype=object)


def _integer(value: object, column: str) -> int:
    """An exact integer. A float is refused: a 60-bit ID that went through a float column
    (pandas does this to an int column with a gap) has already lost its low bits."""
    if isinstance(value, float | np.floating) or not isinstance(value, int | np.integer):
        raise TypeError(f"{column}: {value!r} is not an integer")
    return int(value)


def conform(frame: pd.DataFrame, table: str) -> pd.DataFrame:
    """Columns in OMOP order, as Python values Arrow converts to the Spark types: int or None
    for integers (never a float NaN), date or None for dates, str or None for strings."""
    out: dict[str, list[object]] = {}
    for col in TABLES[table]:
        default = 0 if col.source == "constant 0" else None
        values = frame[col.name].tolist() if col.name in frame else [default] * len(frame)
        if col.dtype in ("BIGINT", "INT"):
            out[col.name] = [None if _blank(v) else _integer(v, col.name) for v in values]
        elif col.dtype == "STRING":
            out[col.name] = [_text(v) for v in values]
        else:  # DATE, TIMESTAMP
            out[col.name] = [None if _blank(v) else v for v in values]
    return pd.DataFrame(out, index=range(len(frame)), dtype=object)


def concepts_for(
    cmap: pd.DataFrame,
    fields: Iterable[str | None],
    sources: Iterable[str],
    values: Iterable[object],
) -> list[int]:
    """Concept per row, the field varying by row. 0 for a missing value or field, and for an
    unmapped value, which :func:`unmapped` reports and the omop_every_value_mapped gate fails."""
    tables: dict[str, dict[tuple[str, str], int]] = {}
    out = []
    for field, source, value in zip(fields, sources, values, strict=True):
        if field is None:
            out.append(0)
            continue
        if field not in tables:
            tables[field] = lookup(cmap, field)
        table = tables[field]
        hit = map_values([source], [value], table)[0]
        out.append(0 if hit is None else hit)
    return out


def constant(cmap: pd.DataFrame, field: str, value: str) -> int:
    return lookup(cmap, field)[("all", value)]


def prepare(silver: pd.DataFrame) -> pd.DataFrame:
    """One row per image with the keys and source values the OMOP tables need. ``silver`` is
    silver_metadata, plus ``local_path`` where the file decoded."""
    f = silver.reset_index(drop=True)
    source = f["source"].astype(str)
    is_ham = source == HAM
    lesion = source + ":" + f["lesion_id"].astype(str)
    patient = _col(f, "patient_id")
    person = (
        source
        + ":"
        + ("patient:" + patient.fillna("")).where(
            patient.notna(), "lesion:" + f["lesion_id"].astype(str)
        )
    )
    site2, site1, ham_site = (
        _col(f, c) for c in ("anatom_site_2", "anatom_site_1", "ham_localization")
    )
    site_field = pd.Series(
        np.select(
            [is_ham & ham_site.notna(), ~is_ham & site2.notna(), ~is_ham & site1.notna()],
            ["ham_localization", "anatom_site_2", "anatom_site_1"],
            "",
        ),
        index=f.index,
    )
    site = pd.Series(
        np.select([site_field == "ham_localization", site_field == "anatom_site_2",
                   site_field == "anatom_site_1"], [ham_site, site2, site1], ""),
        index=f.index,
    )  # fmt: skip
    confirm = _col(f, "ham_dx_type").where(is_ham, _col(f, "diagnosis_confirm_type"))
    return pd.DataFrame(
        {
            "source": source,
            "isic_id": f["isic_id"].astype(str),
            "lesion_key": lesion,
            "person_source_value": person,
            "person_id": person.map(stable_id),
            "sex": _col(f, "sex"),
            "age_approx": pd.to_numeric(_col(f, "age_approx"), errors="coerce"),
            "diagnosis_3": _col(f, "diagnosis_3"),
            "label": _col(f, "label"),
            "confirm": confirm,
            "confirm_field": np.where(is_ham, "ham_dx_type", "diagnosis_confirm_type"),
            "site": site.map(_text),
            "site_field": site_field.map(_text),
            "image_type": _col(f, "image_type"),
            "local_path": _col(f, "local_path"),
            "license": _col(f, "license"),
            "attribution": _col(f, "attribution"),
        }
    )


def most_common(frame: pd.DataFrame, key: str, column: str) -> pd.Series:
    """Per ``key``, the most common non-null ``column`` value; ties go to a specific value over
    a "NOS" one, then alphabetical. This decides a lesion's diagnosis when its images disagree
    (they may only refine one 7-class label: the omop_one_diagnosis_per_lesion gate fails on
    anything else) and a person's sex. Keys with no value are absent from the result."""
    counts = frame.dropna(subset=[column]).groupby([key, column]).size().rename("n").reset_index()
    counts["nos"] = counts[column].astype(str).str.endswith("NOS")
    counts["neg"] = -counts["n"]
    ranked = counts.sort_values([key, "neg", "nos", column])
    return ranked.drop_duplicates(key).set_index(key)[column]


def lesions(images: pd.DataFrame) -> pd.DataFrame:
    """One row per lesion: its person, and its diagnosis, label and confirmation method
    (:func:`most_common` of its images)."""
    out = images.groupby("lesion_key", sort=True).agg(
        person_id=("person_id", "first"),
        source=("source", "first"),
        confirm_field=("confirm_field", "first"),
    )
    for column in ("diagnosis_3", "label", "confirm"):
        picked = most_common(images, "lesion_key", column)
        out[column] = pd.Series([picked.get(k) for k in out.index], index=out.index, dtype=object)
    out = out.reset_index()
    has_d3 = out["diagnosis_3"].notna()
    out["condition_source_value"] = out["diagnosis_3"].where(has_d3, out["label"])
    out["condition_field"] = np.where(has_d3, "diagnosis_3", "label")
    return out


def build_tables(
    silver: pd.DataFrame,
    cmap: pd.DataFrame,
    concepts: pd.DataFrame,
    etl_reference: str,
    vocabulary_version: str,
    build_date: date,
) -> dict[str, pd.DataFrame]:
    """Every OMOP table, conformed to ``TABLES``. ``concepts`` has the Athena CONCEPT.csv
    columns for the concepts in the map (the private resolved file)."""
    img = prepare(silver)
    registry = constant(cmap, "condition_type", "isic_archive")

    people = (
        img.groupby("person_id", sort=True)
        .agg(
            person_source_value=("person_source_value", "first"),
            source=("source", "first"),
            age=("age_approx", "min"),
        )
        .reset_index()
    )
    sex = most_common(img, "person_id", "sex")
    people["sex"] = pd.Series([sex.get(p) for p in people["person_id"]], dtype=object)
    sex_field = ["sex"] * len(people)
    person = people.assign(
        gender_concept_id=concepts_for(cmap, sex_field, people["source"], people["sex"]),
        year_of_birth=pd.Series(
            [None if pd.isna(a) else ANCHOR_DATE.year - int(a) for a in people["age"]], dtype=object
        ),
        race_concept_id=0,
        ethnicity_concept_id=0,
        gender_source_value=people["sex"],
    )
    observation_period = people.assign(
        observation_period_id=[stable_id("op", p) for p in people["person_source_value"]],
        observation_period_start_date=ANCHOR_DATE,
        observation_period_end_date=ANCHOR_DATE,
        period_type_concept_id=registry,
    )

    les = lesions(img)
    dx = les[les["condition_source_value"].notna()].reset_index(drop=True)
    status_field = [None if c is None else "condition_status" for c in dx["confirm"]]
    condition = dx.assign(
        condition_occurrence_id=dx["lesion_key"].map(stable_id),
        condition_concept_id=concepts_for(
            cmap, dx["condition_field"], dx["source"], dx["condition_source_value"]
        ),
        condition_start_date=ANCHOR_DATE,
        condition_type_concept_id=registry,
        condition_status_concept_id=concepts_for(cmap, status_field, dx["source"], dx["confirm"]),
        condition_status_source_value=dx["confirm"],
        condition_label=dx["label"],
        lesion_source_value=dx["lesion_key"],
    )

    image_ids = list(zip(img["source"], img["isic_id"], strict=True))
    imaging = img.assign(
        procedure_occurrence_id=[stable_id("img", s, i) for s, i in image_ids],
        procedure_concept_id=concepts_for(
            cmap, ["image_type"] * len(img), img["source"], img["image_type"]
        ),
        procedure_source_value=img["image_type"],
    )
    conf = les[les["confirm"].notna()].reset_index(drop=True)
    conf = conf.assign(
        procedure_occurrence_id=[stable_id("confirm", k) for k in conf["lesion_key"]],
        procedure_concept_id=concepts_for(
            cmap, conf["confirm_field"], conf["source"], conf["confirm"]
        ),
        procedure_source_value=conf["confirm"],
    )
    # Only a confirmation method that is a procedure (histopathology) becomes a procedure row.
    conf = conf[conf["procedure_concept_id"] != 0]
    procedure = pd.concat([imaging, conf], ignore_index=True).assign(
        procedure_date=ANCHOR_DATE, procedure_type_concept_id=registry
    )

    condition_of = dict(
        zip(condition["lesion_key"], condition["condition_occurrence_id"], strict=True)
    )
    image = imaging.assign(
        image_occurrence_id=[stable_id("image", s, i) for s, i in image_ids],
        anatomic_site_concept_id=concepts_for(cmap, img["site_field"], img["source"], img["site"]),
        image_occurrence_date=ANCHOR_DATE,
        image_study_uid=[dicom_uid(i) for i in img["isic_id"]],
        image_series_uid=[dicom_uid(i, "series") for i in img["isic_id"]],
        modality_concept_id=concepts_for(
            cmap, ["modality"] * len(img), img["source"], img["image_type"]
        ),
        image_source_value=img["isic_id"],
        anatomic_site_source_value=img["site"],
        # A list, not Series.map: a gap would turn the IDs into floats and corrupt them.
        condition_occurrence_id=pd.Series(
            [condition_of.get(k) for k in img["lesion_key"]], dtype=object
        ),
    )

    cdm_source = pd.DataFrame([{
        "cdm_source_name": "Clinical Drift Monitor OMOP tables (public dermoscopy archives)",
        "cdm_source_abbreviation": "CDM-ISIC",
        "cdm_holder": "Erjon Brucaj (portfolio project; not a clinical data source)",
        "source_description": "HAM10000 and ISIC Archive collections as landed by the cdm "
        "pipeline: public research images, no real dates, one person per lesion where ISIC "
        "records no patient ID.",
        "source_documentation_reference": "https://github.com/erjonb19/clinical-drift-monitor/"
        "blob/main/docs/data_dictionary.md",
        "cdm_etl_reference": etl_reference,
        "source_release_date": build_date,
        "cdm_release_date": build_date,
        "cdm_version": "v5.4",
        "cdm_version_concept_id": constant(cmap, "cdm_version", "5.4"),
        "vocabulary_version": vocabulary_version,
    }])  # fmt: skip

    concept = concepts.assign(
        concept_id=concepts["concept_id"].astype(int),
        valid_start_date=pd.to_datetime(concepts["valid_start_date"], format="%Y%m%d").dt.date,
        valid_end_date=pd.to_datetime(concepts["valid_end_date"], format="%Y%m%d").dt.date,
    )
    built = {
        "person": person,
        "observation_period": observation_period,
        "condition_occurrence": condition,
        "procedure_occurrence": procedure,
        "image_occurrence": image,
        "cdm_source": cdm_source,
        "concept": concept,
    }
    return {name: conform(frame, name) for name, frame in built.items()}


# ---- checks ---------------------------------------------------------------------------------


def unmapped(silver: pd.DataFrame, cmap: pd.DataFrame) -> pd.DataFrame:
    """Every (field, source, value) present in the data with no row in the concept map, and
    how many images (or lesions, for lesion-level fields) have it. A value mapped to concept 0
    on purpose has a row, so it is not listed."""
    img = prepare(silver)
    les = lesions(img)

    def same(field: str, frame: pd.DataFrame) -> pd.Series:
        return pd.Series(field, index=frame.index, dtype=object)

    checks = [
        (same("sex", img), img["source"], img["sex"]),
        (same("image_type", img), img["source"], img["image_type"]),
        (same("modality", img), img["source"], img["image_type"]),
        (img["site_field"], img["source"], img["site"]),
        (les["condition_field"], les["source"], les["condition_source_value"]),
        (les["confirm_field"], les["source"], les["confirm"]),
        (same("condition_status", les), les["source"], les["confirm"]),
    ]
    tables: dict[str, dict[tuple[str, str], int]] = {}
    rows: list[tuple[str, str, str]] = []
    for fields, sources, values in checks:
        for field, source, value in zip(fields, sources, values, strict=True):
            if field is None or _blank(value):
                continue
            if field not in tables:
                tables[field] = lookup(cmap, field)
            table = tables[field]
            if map_values([source], [value], table)[0] is None:
                rows.append((field, source, str(value)))
    found = pd.DataFrame(rows, columns=["field", "source", "value"])
    counts = found.groupby(["field", "source", "value"]).size().rename("n").reset_index()
    return counts.sort_values(["field", "source", "value"]).reset_index(drop=True)


def profile(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Rows and null count per table column: the null rates in the data dictionary."""
    rows = [
        {"table_name": name, "column_name": col.name, "rows": len(frame),
         "nulls": int(frame[col.name].isna().sum())}
        for name, frame in tables.items()
        for col in TABLES[name]
    ]  # fmt: skip
    return pd.DataFrame(rows)


def _gate(name: str, problems: list[str]) -> Gate:
    return Gate(name, len(problems), "; ".join(problems[:5]) if problems else "ok")


def check_tables(
    silver: pd.DataFrame, tables: Mapping[str, pd.DataFrame], cmap: pd.DataFrame
) -> list[Gate]:
    """The OMOP quality gates. Violations are counted per problem found."""
    img = prepare(silver)
    les = lesions(img)
    gates = []

    dupes = [
        f"{name}.{key}: {int(tables[name][key].duplicated().sum())} duplicates"
        for name, key in PRIMARY_KEYS.items()
        if tables[name][key].duplicated().any()
    ]
    gates.append(_gate("omop_primary_keys_unique", dupes))

    orphans = []
    for name, links in FOREIGN_KEYS.items():
        for col, parent, parent_col in links:
            values = tables[name][col].dropna()
            missing = int((~values.isin(set(tables[parent][parent_col]))).sum())
            if missing:
                orphans.append(f"{name}.{col}: {missing} not in {parent}")
    gates.append(_gate("omop_foreign_keys_resolve", orphans))

    gaps = unmapped(silver, cmap)
    gates.append(
        _gate(
            "omop_every_value_mapped",
            [f"{r.field} {r.source} {r.value!r} x{r.n}" for r in gaps.itertuples()],
        )
    )

    known = set(tables["concept"]["concept_id"])
    stray = sorted(
        {
            f"{name}.{col}={cid}"
            for name, cols in CONCEPT_COLUMNS.items()
            for col in cols
            for cid in tables[name][col].dropna()
            if cid != 0 and cid not in known
        }
    )
    gates.append(_gate("omop_concepts_resolve", stray))

    counts = []
    expected = {
        "image_occurrence": len(img),
        "person": img["person_id"].nunique(),
        "observation_period": img["person_id"].nunique(),
        "condition_occurrence": int(les["condition_source_value"].notna().sum()),
    }
    for name, n in expected.items():
        if len(tables[name]) != n:
            counts.append(f"{name}: {len(tables[name])} rows, expected {n}")
    imaging = tables["procedure_occurrence"]["procedure_occurrence_id"].isin(
        set(tables["image_occurrence"]["procedure_occurrence_id"])
    )
    if int(imaging.sum()) != len(img):
        counts.append(f"imaging procedures: {int(imaging.sum())}, expected {len(img)}")
    gates.append(_gate("omop_row_counts", counts))

    # A lesion's images may refine one diagnosis (Melanoma, NOS and Melanoma Invasive): the
    # rule in most_common picks one. Images disagreeing on the 7-class label fail.
    conflicts = [
        f"{k}: {sorted(v)}"
        for k, v in img.dropna(subset=["label"]).groupby("lesion_key")["label"].unique().items()
        if len(v) > 1
    ]
    gates.append(_gate("omop_one_diagnosis_per_lesion", conflicts))

    nulls = [
        f"{name}.{col.name}: {int(tables[name][col.name].isna().sum())} null"
        for name, cols in TABLES.items()
        for col in cols
        if col.required and tables[name][col.name].isna().any()
    ]
    gates.append(_gate("omop_required_not_null", nulls))
    return gates


# ---- data dictionary ------------------------------------------------------------------------

GATES = {
    "omop_primary_keys_unique": "every table's first column is unique",
    "omop_foreign_keys_resolve": "person, procedure and condition links all resolve",
    "omop_every_value_mapped": "every source value in the data has a concept map row",
    "omop_concepts_resolve": "every non-zero concept ID is in the concept table",
    "omop_row_counts": "one image row per silver image, one person per person key, one "
    "condition per diagnosed lesion, one imaging procedure per image",
    "omop_one_diagnosis_per_lesion": "a lesion's images never disagree on the 7-class label",
    "omop_required_not_null": "no null in a column marked required",
}


def _md(text: object) -> str:
    return str(text).replace("|", "\\|")


def data_dictionary(cmap: pd.DataFrame, build: Mapping[str, Any] | None) -> str:
    """docs/data_dictionary.md: every OMOP column from ``TABLES``, null rates from a committed
    build (results/research/a_omop/build.json), and every unmapped source value."""
    nulls: dict[tuple[str, str], tuple[int, int]] = {}
    if build:
        for row in build["profile"]:
            nulls[(row["table_name"], row["column_name"])] = (int(row["nulls"]), int(row["rows"]))
    where = (
        f"build `{build['etl_reference']}` (pipeline update `{build['update_id']}`, "
        f"{build['built_utc']})"
        if build
        else "no build yet: null rates fill in from the first pipeline update"
    )
    lines = [
        "# OMOP data dictionary",
        "",
        "Generated by `scripts/omop_build_report.py` from `src/cdm/omop.py` (the column "
        "definitions) and `results/research/a_omop/build.json` (the null rates). Do not edit by "
        "hand; a test checks this file matches. Null rates are from " + where + ".",
        "",
        "These tables are a **portfolio build on public research images** (HAM10000 and ISIC "
        "Archive collections), not a clinical data source. They live in the Databricks "
        "workspace as `workspace.cdm.omop_<table>`, built by the Lakeflow pipeline from "
        "`silver_metadata`, so Unity Catalog lineage runs bronze → silver → OMOP.",
        "",
        "## Conventions and assumptions",
        "",
        "- **Standard:** OMOP CDM v5.4 column names, order and types. `image_occurrence` follows "
        "the OHDSI medical imaging extension proposal (Park et al., JAMIA 32(10):1533, 2025, "
        "doi:10.1093/jamia/ocaf091). Extension columns are marked as such.",
        "- **Dates are placeholders.** The sources record no dates. Every date is 2000-01-01, "
        "and `year_of_birth` is 2000 minus the approximate age (5-year bins). Use them for age "
        "only, never for time.",
        "- **Person** is a real patient only where ISIC records a patient ID (MSK collection "
        "289, Buenos Aires, PAD-UFES-20). Elsewhere one person is one lesion, which "
        "over-counts people. `person_source_value` says which rule applied.",
        "- **Condition** is one per lesion: ISIC `diagnosis_3`, else the 7-class label. When a "
        "lesion's images disagree, the most common value wins, then a specific diagnosis over "
        "a NOS one. `condition_status_source_value` keeps how the diagnosis was made.",
        "- **IDs** are 60-bit hashes of source keys, stable across rebuilds. The DICOM UIDs are "
        "surrogates (`2.25.` + a UUID5 of the image ID), because there is no DICOM.",
        "- **Concepts:** `config/omop_concepts.csv` maps each source value to an OMOP concept "
        "looked up in an Athena download (`results/research/a_omop/concept_check.json`). "
        "Concept 0 means no verified concept. SNOMED codes and names are not in the "
        "repository (SNOMED license); `omop_concept` holds them inside the workspace only.",
        "",
        "## FAIR, honestly",
        "",
        "- **Findable:** table and column comments in Unity Catalog, and this dictionary.",
        "- **Accessible:** inside the Databricks workspace only. There is no public endpoint.",
        "- **Interoperable:** OMOP CDM v5.4 tables and OMOP standard concepts.",
        "- **Reusable:** license and attribution per image in `omop_image_occurrence`. CC-BY-NC "
        "images are non-commercial only.",
        "",
        "## Quality gates",
        "",
        "Each gate is a row in `omop_gate_results`. Any violation fails the pipeline update, and "
        "each gate also runs in CI on synthetic data (`tests/test_omop.py`).",
        "",
        "| Gate | Checks | Violations at this build |",
        "| --- | --- | --- |",
    ]
    results = {g["name"]: g for g in build["gates"]} if build else {}
    for name, what in GATES.items():
        got = results.get(name)
        lines.append(f"| `{name}` | {what} | {got['violations'] if got else 'pending'} |")
    for table, cols in TABLES.items():
        lines += ["", f"## omop_{table}", ""]
        if build:
            lines += [f"{build['row_counts'].get(table, 'pending')} rows.", ""]
        lines += [
            "| Column | Type | Required | From | Note | Null |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for c in cols:
            n = nulls.get((table, c.name))
            rate = "pending" if n is None else f"{100 * n[0] / n[1]:.1f}%" if n[1] else "0 rows"
            lines.append(
                f"| `{c.name}` | {c.dtype} | {'yes' if c.required else ''} | {_md(c.source)} "
                f"| {_md(c.note)} | {rate} |"
            )
    lines += [
        "",
        "## Source values with no verified concept",
        "",
        "These map to concept 0 on purpose. Each gives its reason. Modality is among them: "
        "the Athena downloads offered no DICOM vocabulary.",
        "",
        "| Field | Scope | Source value | Reason |",
        "| --- | --- | --- | --- |",
    ]
    for r in cmap[cmap["concept_id"] == 0].itertuples():
        lines.append(f"| {r.source_field} | {r.scope} | {_md(r.source_value)} | {_md(r.note)} |")
    return "\n".join(lines) + "\n"


# ---- public extracts ------------------------------------------------------------------------


def public_concepts(cmap: pd.DataFrame) -> pd.DataFrame:
    """A concept table without SNOMED text, for anything outside the workspace (quickstart,
    demo): each mapped concept ID with its expected domain and an Athena link for the name."""
    used = cmap[cmap["concept_id"] != 0].drop_duplicates("concept_id")
    return pd.DataFrame(
        {
            "concept_id": used["concept_id"].astype(str).tolist(),
            "concept_name": [
                f"https://athena.ohdsi.org/search-terms/terms/{c}" for c in used["concept_id"]
            ],
            "domain_id": used["expected_domain"].tolist(),
            "vocabulary_id": "see Athena",
            "concept_class_id": "see Athena",
            "standard_concept": "S",
            "concept_code": "not published",
            "valid_start_date": "19700101",
            "valid_end_date": "20991231",
            "invalid_reason": "",
        }
    )


def silver_from_isic(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Rows from :func:`cdm.sources.flatten` (public ISIC metadata), shaped like silver: the
    lesion is ISIC's lesion ID, else the image, and the label is ISIC's mapped label. The
    pipeline's site label rule is not applied here."""
    frame = pd.DataFrame(list(rows))
    return frame.assign(
        lesion_id=frame["isic_lesion_id"].fillna(frame["isic_id"]), label=frame["isic_label"]
    )
