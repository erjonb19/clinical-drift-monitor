"""Record an OMOP build and regenerate docs/data_dictionary.md.

After a pipeline update builds the omop_* tables, this reads them through the SQL warehouse
and writes results/research/a_omop/build.json (committed): row counts, null counts per
column, gate results, and condition and site concept counts (concept IDs only, no SNOMED
text). Then it renders the dictionary. Lineage is recorded separately, because
system.access.table_lineage lags the pipeline by hours.

    python scripts/omop_build_report.py --update-id <pipeline update ID>
    python scripts/omop_build_report.py --render-only   (dictionary from the committed build)
    python scripts/omop_build_report.py --lineage       (results/research/a_omop/lineage.json)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cdm.omop import TABLES, data_dictionary, load_concept_map

REPO = Path(__file__).resolve().parents[1]
BUILD = REPO / "results" / "research" / "a_omop" / "build.json"
DICTIONARY = REPO / "docs" / "data_dictionary.md"
LINEAGE = REPO / "results" / "research" / "a_omop" / "lineage.json"
SCHEMA = "workspace.cdm"


def sql_client() -> Any:
    spec = importlib.util.spec_from_file_location(
        "evidence_export", REPO / "scripts" / "phase1_export_evidence.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Sql()


def records(frame: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = json.loads(frame.to_json(orient="records"))
    return out


def collect(update_id: str) -> dict[str, Any]:
    sql = sql_client()
    counts = {t: int(sql(f"SELECT COUNT(*) AS n FROM {SCHEMA}.omop_{t}")["n"][0]) for t in TABLES}
    etl = sql(f"SELECT cdm_etl_reference FROM {SCHEMA}.omop_cdm_source")["cdm_etl_reference"][0]
    return {
        "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "update_id": update_id,
        "etl_reference": etl,
        "report_code_version": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "row_counts": counts,
        "gates": records(sql(f"SELECT * FROM {SCHEMA}.omop_gate_results ORDER BY name")),
        "profile": records(sql(f"SELECT * FROM {SCHEMA}.omop_profile")),
        "persons_by_rule": records(
            sql(
                "SELECT split_part(person_source_value, ':', 1) AS source, "
                "split_part(person_source_value, ':', 2) AS rule, COUNT(*) AS persons "
                f"FROM {SCHEMA}.omop_person GROUP BY 1, 2 ORDER BY 1, 2"
            )
        ),
        "conditions_by_concept": records(
            sql(
                "SELECT condition_concept_id, condition_label, COUNT(*) AS lesions "
                f"FROM {SCHEMA}.omop_condition_occurrence GROUP BY 1, 2 ORDER BY 1, 2"
            )
        ),
        "images_by_site_concept": records(
            sql(
                "SELECT anatomic_site_concept_id, COUNT(*) AS images "
                f"FROM {SCHEMA}.omop_image_occurrence GROUP BY 1 ORDER BY 1"
            )
        ),
    }


def lineage() -> dict[str, Any]:
    """Unity Catalog's lineage edges from bronze through silver into the OMOP tables. Bronze
    tables read files from the volume, so their edges have no source table."""
    edges = sql_client()(
        "SELECT source_table_full_name AS source, target_table_full_name AS target, "
        "MAX(event_time) AS last_seen FROM system.access.table_lineage "
        f"WHERE target_table_full_name LIKE '{SCHEMA}.omop_%' "
        f"OR target_table_full_name IN ('{SCHEMA}.silver_metadata', '{SCHEMA}.silver_decoded') "
        f"OR target_table_full_name LIKE '{SCHEMA}.bronze_%' GROUP BY 1, 2 ORDER BY 2, 1"
    )
    return {"queried_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "table": "system.access.table_lineage", "edges": records(edges)}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--update-id")
    group.add_argument("--render-only", action="store_true")
    group.add_argument("--lineage", action="store_true")
    args = parser.parse_args()
    if args.lineage:
        LINEAGE.write_bytes((json.dumps(lineage(), indent=2) + "\n").encode())
        print(f"wrote {LINEAGE}")
        return
    if args.update_id:
        BUILD.parent.mkdir(parents=True, exist_ok=True)
        BUILD.write_bytes((json.dumps(collect(args.update_id), indent=2) + "\n").encode())
    build = json.loads(BUILD.read_text(encoding="utf-8")) if BUILD.exists() else None
    cmap = load_concept_map(REPO / "config" / "omop_concepts.csv")
    DICTIONARY.write_bytes(data_dictionary(cmap, build).encode())
    print(f"wrote {DICTIONARY}" + (f" from build {build['etl_reference']}" if build else ""))


if __name__ == "__main__":
    main()
