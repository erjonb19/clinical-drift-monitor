"""Read the Unity Catalog lineage edges into the audit and drift tables (Phase 3).

Reads ``system.access.table_lineage`` on the workspace's SQL warehouse (read only) for every
edge whose target is ``gold_prediction_audit``, ``gold_drift_baseline`` or
``gold_drift_monthly``, and writes them to ``results/phase3/lineage/spark_writes_edges.json``.
Edges with no source table (writes from pandas) are counted, not listed. The system table lags,
so an edge from a recent run may not be there yet: the script fails rather than write a file
that is missing an expected edge.

Usage: python scripts/lineage_edges.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_export_evidence import Sql, records  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "phase3" / "lineage" / "spark_writes_edges.json"
TARGETS = tuple(
    f"workspace.cdm.{t}"
    for t in ("gold_prediction_audit", "gold_drift_baseline", "gold_drift_monthly")
)
# What the spec expects once the rewrite has run (results/phase3/lineage/spark_writes_spec.json).
EXPECTED = {
    ("workspace.cdm.silver_images", "workspace.cdm.gold_prediction_audit"),
    ("workspace.cdm.silver_images", "workspace.cdm.gold_drift_baseline"),
    ("workspace.cdm.gold_prediction_audit", "workspace.cdm.gold_drift_monthly"),
    ("workspace.cdm.gold_drift_baseline", "workspace.cdm.gold_drift_monthly"),
}


def main() -> None:
    sql = Sql()
    targets = ", ".join(f"'{t}'" for t in TARGETS)
    edges = sql(
        f"""
        SELECT source_table_full_name AS source, target_table_full_name AS target,
               entity_type, entity_run_id, MIN(event_time) AS first_seen, COUNT(*) AS events
        FROM system.access.table_lineage
        WHERE target_table_full_name IN ({targets}) AND source_table_full_name IS NOT NULL
          AND source_table_full_name <> target_table_full_name
        GROUP BY ALL ORDER BY target, source, first_seen
        """
    )
    sourceless = sql(
        f"""
        SELECT target_table_full_name AS target, COUNT(*) AS events
        FROM system.access.table_lineage
        WHERE target_table_full_name IN ({targets}) AND source_table_full_name IS NULL
        GROUP BY ALL ORDER BY target
        """
    )
    found = {(str(s), str(t)) for s, t in zip(edges["source"], edges["target"], strict=True)}
    missing = sorted(EXPECTED - found)
    if missing:
        sys.exit(f"not in system.access.table_lineage yet: {missing}")
    out = {
        "spec": "results/phase3/lineage/spark_writes_spec.json",
        "read_from": "system.access.table_lineage",
        "edges": records(edges),
        "writes_with_no_source_table": records(sourceless),
        "expected_edges_present": True,
        "read_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"{len(edges)} edges written to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
