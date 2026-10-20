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

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

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
)
# Vocabularies a mapped concept may come from.
VOCABULARIES = {"SNOMED", "Type Concept", "Gender"}


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
