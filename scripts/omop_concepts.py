"""Check config/omop_concepts.csv against an Athena vocabulary download.

Every mapped concept must exist in the download, be a valid standard concept from an expected
vocabulary, and sit in the expected domain. Writes:

- results/research/a_omop/concept_check.json (committed): the vocabulary release, counts by
  match type, and the check result per concept ID. No SNOMED codes or names.
- --resolved FILE (not committed; default in the vocabulary folder): concept ID, name, code,
  vocabulary, domain and class for every mapped concept, for the private ``concept`` table.

Usage: python scripts/omop_concepts.py --vocab C:/Users/Erjon/athena_vocab
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from cdm.omop import check_against_vocabulary, load_concept_map

REPO = Path(__file__).resolve().parents[1]
CMAP = REPO / "config" / "omop_concepts.csv"
OUT = REPO / "results" / "research" / "a_omop" / "concept_check.json"


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--resolved", type=Path)
    args = parser.parse_args()
    cmap = load_concept_map(CMAP)
    concepts = read_tsv(args.vocab / "CONCEPT.csv")
    vocabularies = read_tsv(args.vocab / "VOCABULARY.csv").set_index("vocabulary_id")
    problems = check_against_vocabulary(cmap, concepts)
    used = cmap[cmap["concept_id"] != 0]
    resolved = concepts[concepts["concept_id"].isin(used["concept_id"].astype(str))]
    release = {
        v: vocabularies.loc[v, "vocabulary_version"]
        for v in ("None", "SNOMED", "Type Concept", "Gender")
        if v in vocabularies.index
    }
    result: dict[str, Any] = {
        "checked_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_version": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "vocabulary_release": release,
        "citation": "https://athena.ohdsi.org/search-terms/terms/<concept_id>",
        "map_rows": len(cmap),
        "rows_by_match": cmap["match"].value_counts().sort_index().to_dict(),
        "rows_by_field": cmap["source_field"].value_counts().sort_index().to_dict(),
        "distinct_concepts": int(used["concept_id"].nunique()),
        "problems": [{"concept_id": p.concept_id, "problem": p.problem} for p in problems],
        "passed": not problems,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    target = args.resolved or (args.vocab / "cdm_resolved_concepts.csv")
    resolved.to_csv(target, index=False)
    print(json.dumps({k: v for k, v in result.items() if k != "problems"}, indent=1))
    for p in problems:
        print(f"PROBLEM {p.concept_id}: {p.problem}")
    print(f"resolved concepts (private, not committed): {target}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
