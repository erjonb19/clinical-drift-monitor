"""Go/no-go checks after the v2 ingest and pipeline update, saved to results/phase2/v2/.

Passes only if: the Phase 1 job run succeeded; every gate in gate_results has zero
violations (frozen_splits_kept and split_fingerprint_matches_phase0 named explicitly);
bronze_images grew by exactly the new Barcelona images (Auto Loader picked up only new
files, Phase 1's deferred check), read both from the table and from the update's own
flow metrics; and v1's scoring-only Barcelona lesions are all still scoring-only.

Usage: python scripts/v2_data_checks.py --run <phase 1 job run id>
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_export_evidence import Sql, _events, job_run, pipeline_id  # noqa: E402

BEFORE = {"bronze_images": 22657, "barcelona": 5000}
NEW_IMAGES = 4999  # 9,999 selected by the 10,000 cap minus v1's 5,000
OUT = Path("results/phase2/v2/data_checks.json")


def flow_rows_added(pipeline: str, flow: str, since: str) -> int | None:
    """Rows the bronze flow wrote in updates that started after ``since`` (flow metrics)."""
    total, seen = 0, False
    for e in _events(pipeline, "METRICS") + _events(pipeline, "INFO"):
        origin = e.get("origin") or {}
        if origin.get("flow_name", "").split(".")[-1] != flow or e["timestamp"] < since:
            continue
        progress = (e.get("details") or {}).get("flow_progress") or {}
        rows = (progress.get("metrics") or {}).get("num_output_rows")
        if progress.get("status") == "COMPLETED" and rows is not None:
            total, seen = total + int(rows), True
    return total if seen else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    sql = Sql()
    run = job_run(args.run)
    gates = sql("SELECT * FROM workspace.cdm.gate_results")
    gates["violations"] = gates["violations"].astype(int)
    by_source = sql("SELECT source, count(*) n FROM workspace.cdm.bronze_images GROUP BY source")
    counts = {r["source"]: int(r["n"]) for r in by_source.to_dict("records")}
    total = sum(counts.values())
    start = min(t["start"] for t in run["tasks"] if t.get("start"))
    added = flow_rows_added(pipeline_id("cdm-phase1"), "bronze_images", start[:19])
    stayed = sql(
        """SELECT count(DISTINCT v.lesion_id) lesions,
             sum(CASE WHEN s.split = 'score' THEN 1 ELSE 0 END) still_score,
             sum(CASE WHEN s.split IS NULL THEN 1 ELSE 0 END) missing, count(*) images
           FROM workspace.cdm.silver_images_v1_snapshot v
           LEFT JOIN workspace.cdm.silver_images s USING (source, isic_id)
           WHERE v.source = 'barcelona' AND v.label IS NULL"""
    ).to_dict("records")[0]
    splits = sql(
        """SELECT client, split, count(*) n FROM workspace.cdm.silver_images
           WHERE role = 'client' GROUP BY client, split ORDER BY client, split"""
    ).to_dict("records")
    named = {g: int(gates.loc[gates["name"] == g, "violations"].sum())
             for g in ("frozen_splits_kept", "split_fingerprint_matches_phase0")}  # fmt: skip
    checks = {
        "job_succeeded": run.get("result") == "SUCCESS",
        "all_gates_pass": bool((gates["violations"] == 0).all()) and len(gates) > 0,
        "frozen_splits_kept_present_and_passing": "frozen_splits_kept" in set(gates["name"])
        and named["frozen_splits_kept"] == 0,
        "ham_fingerprint_gate_passing": named["split_fingerprint_matches_phase0"] == 0,
        "bronze_grew_by_new_images": total - BEFORE["bronze_images"] == NEW_IMAGES
        and counts.get("barcelona", 0) - BEFORE["barcelona"] == NEW_IMAGES,
        "update_flow_added_only_new_files": added == NEW_IMAGES,
        "v1_scoring_lesions_still_scoring": int(stayed["images"]) == int(stayed["still_score"]),
    }
    evidence: dict[str, Any] = {
        "exported_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "job_run": run,
        "checks": checks,
        "passed": all(checks.values()),
        "gates": gates.to_dict("records"),
        "bronze_images_before": BEFORE,
        "bronze_images_after": {"total": total, **counts},
        "bronze_rows_added_by_update_flow_metrics": added,
        "v1_scoring_only_barcelona": stayed,
        "silver_client_images_by_split": splits,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(evidence, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"passed": evidence["passed"], **checks}, indent=2))
    print("bronze after", evidence["bronze_images_after"], "flow rows added", added)


if __name__ == "__main__":
    main()
