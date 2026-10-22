"""The public cohort demo's logic (research Phase G6), kept apart from Streamlit so CI can
test it. ``app/cohort_demo.py`` is the page.

The demo builds the OMOP tables from public ISIC metadata (PAD-UFES-20, collection 406,
CC-BY), fetched when the app starts. Nothing is committed and no image is shown. The concept
table carries concept IDs and Athena links only, so no SNOMED text appears. Each visitor gets
a scratch approvals file and audit log in a temporary folder: they play the reviewer, and
their decisions never touch ``phenotypes/approvals.jsonl``.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from cdm.cohort import (
    Approvals,
    AuditLog,
    CohortResult,
    CohortService,
    duckdb_engine,
    load,
    parse,
)
from cdm.extract import choices
from cdm.omop import build_tables, load_concept_map, public_concepts, silver_from_isic
from cdm.sources import collection_records, flatten

REPO = Path(__file__).resolve().parents[2]
COLLECTION, SOURCE = 406, "pad_ufes"
ATTRIBUTION = (
    "Data: ISIC Archive collection 406, PAD-UFES-20 (CC-BY), Federal University of "
    "Espírito Santo. Counts only; no images are shown."
)
HISTOPATHOLOGY = 43531286
REVIEWER = "demo visitor (scratch approvals, this session only)"


def concept_map() -> pd.DataFrame:
    return load_concept_map(REPO / "config" / "omop_concepts.csv")


def tables_from(records: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    """OMOP tables from ISIC records, with the public concept table (IDs only)."""
    cmap = concept_map()
    silver = silver_from_isic(flatten(r, SOURCE, COLLECTION) for r in records)
    tables = build_tables(silver, cmap, public_concepts(cmap), "demo", "public", date.today())
    return {name: frame.infer_objects() for name, frame in tables.items()}


def fetch_tables(
    fetch: Callable[[int], list[dict[str, Any]]] = collection_records,
) -> dict[str, pd.DataFrame]:
    return tables_from(fetch(COLLECTION))


def diagnosis_options() -> dict[str, int]:
    """ISIC diagnosis labels (source values) with their OMOP concept IDs."""
    return choices(concept_map()).concept["diagnosis"]


def custom_definition(
    labels: list[str],
    age_min: int | None,
    age_max: int | None,
    histopathology: bool,
    breakdown: list[str],
) -> dict[str, Any]:
    """A phenotype from the page's controls. It still goes through the allowlist parser and
    needs an approval like any other definition."""
    options = diagnosis_options()
    index: dict[str, Any] = {"condition_concept_ids": sorted({options[x] for x in labels})}
    if age_min is not None:
        index["age_min"] = age_min
    if age_max is not None:
        index["age_max"] = age_max
    return {
        "name": "visitor_cohort",
        "version": 1,
        "title": "A cohort built on the demo page",
        "purpose": "demo",
        "author": "demo visitor",
        "index": index,
        "include": [{"procedure_concept_ids": [HISTOPATHOLOGY]}] if histopathology else [],
        "breakdown": breakdown,
    }


@dataclass
class DemoSession:
    """One visitor's service: their own scratch approvals and audit log."""

    tables: dict[str, pd.DataFrame]
    folder: Path = field(default_factory=lambda: Path(tempfile.mkdtemp(prefix="cdm-demo-")))

    def __post_init__(self) -> None:
        engine, names = duckdb_engine(self.tables)
        self.service = CohortService(
            engine,
            names,
            Approvals(self.folder / "approvals.jsonl"),
            AuditLog(self.folder / "audit.jsonl"),
            source="demo: PAD-UFES-20 public metadata",
        )

    def write(self, definition: dict[str, Any]) -> Path:
        """A visitor's definition as a versioned file in the scratch folder. A changed
        definition is a new version, as in the library."""
        folder = self.folder / str(definition["name"])
        folder.mkdir(exist_ok=True)
        versions = sorted(folder.glob("v*.yaml"))
        for path in versions:
            existing = load(path)
            if existing.definition == parse({**definition, "version": existing.version}).definition:
                return path
        definition = {**definition, "version": len(versions) + 1}
        path = folder / f"v{definition['version']}.yaml"
        path.write_text(yaml.safe_dump(definition, sort_keys=False), encoding="utf-8")
        return path

    def status(self, path: Path) -> str:
        return self.service.approvals.status(load(path))

    def approve(self, path: Path) -> None:
        self.service.decide(path, "approve", REVIEWER, "approved on the demo page")

    def run(self, path: Path) -> CohortResult:
        return self.service.run(path, "demo visitor")

    def audit(self) -> pd.DataFrame:
        entries = self.service.audit.entries()
        columns = ["seq", "ts", "event", "phenotype", "version", "reason", "final", "hash"]
        return pd.DataFrame(entries).reindex(columns=columns)


def library() -> list[Path]:
    return sorted((REPO / "phenotypes").glob("*/v*.yaml"))
