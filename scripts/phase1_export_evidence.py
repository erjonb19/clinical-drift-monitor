"""Export the Phase 1 Databricks run evidence into results/phase1/run_evidence.json.

Reads, with the logged-in Databricks CLI:
- the job runs (task results, times) for the successful run and the broken demo runs;
- every gate_results row, and row counts per source at every layer, from workspace.cdm;
- the HAM10000 split, JPEG compression and resolution per source, gold baseline and stats;
- for the broken demo: the expectation failure Databricks reported, the number of pipeline
  updates the demo run made, the quarantine rows, and every gate recomputed with
  cdm.gates.run_all on workspace.cdm_broken.silver_metadata with the demo's own config.

Usage: python scripts/phase1_export_evidence.py --run <job run id> --demo-runs <id> [<id> ...]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.request
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from cdm.gates import run_all

OUT = Path("results/phase1/run_evidence.json")

C = "workspace.cdm"
QUERIES = {
    "gate_results": f"SELECT name, violations, detail FROM {C}.gate_results ORDER BY name",
    "rows_per_source": f"""
WITH b AS (SELECT source, count(*) n FROM {C}.bronze_images GROUP BY source),
sm AS (SELECT source, count(*) n, count_if(label IS NOT NULL) labelled,
       count(DISTINCT manifest_sha256) manifest_unique_sha256
       FROM {C}.silver_metadata GROUP BY source),
si AS (SELECT source, count(*) n FROM {C}.silver_images GROUP BY source),
q AS (SELECT source, count(*) n FROM {C}.silver_quarantine GROUP BY source),
g AS (SELECT source, count(*) n FROM {C}.gold_embeddings GROUP BY source)
SELECT source, b.n AS bronze_files, sm.n AS silver_metadata, sm.labelled,
       sm.manifest_unique_sha256, si.n AS silver_images, coalesce(q.n, 0) AS quarantined,
       g.n AS gold_embeddings
FROM b FULL JOIN sm USING (source) FULL JOIN si USING (source)
FULL JOIN q USING (source) FULL JOIN g USING (source) ORDER BY source""",
    "ham10000_split": f"""
SELECT split, count(*) AS images, count(DISTINCT lesion_id) AS lesions
FROM {C}.silver_metadata WHERE source = 'ham10000' GROUP BY split ORDER BY split""",
    "images_per_source": f"""
SELECT source, min(jpeg_quant_mean) AS jpeg_quant_min, max(jpeg_quant_mean) AS jpeg_quant_max,
       min(width) AS min_width, max(width) AS max_width
FROM {C}.silver_images GROUP BY source ORDER BY source""",
    "gold_baseline": f"SELECT * FROM {C}.gold_baseline ORDER BY split",
    "gold_feature_stats": f"SELECT * FROM {C}.gold_feature_stats ORDER BY source, skin_type",
}


def cli(*args: str) -> Any:
    done = subprocess.run(
        ["databricks", *args, "--output", "json"], capture_output=True, text=True, check=True
    )
    return json.loads(done.stdout)


def cli_text(*args: str) -> str:
    return subprocess.run(["databricks", *args], capture_output=True, text=True, check=True).stdout


class Sql:
    def __init__(self) -> None:
        self.host = cli("auth", "describe")["details"]["host"].rstrip("/")
        self.token = cli("auth", "token")["access_token"]
        self.warehouse = cli("warehouses", "list")[0]["id"]

    def __call__(self, statement: str) -> pd.DataFrame:
        body = json.dumps(
            {"warehouse_id": self.warehouse, "wait_timeout": "50s", "statement": statement}
        ).encode()
        request = urllib.request.Request(
            f"{self.host}/api/2.0/sql/statements/",
            data=body,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
        if result["status"]["state"] != "SUCCEEDED":
            raise RuntimeError(f"{statement}: {result['status']}")
        cols = [c["name"] for c in result["manifest"]["schema"]["columns"]]
        return pd.DataFrame(result["result"].get("data_array", []), columns=cols)


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = json.loads(frame.to_json(orient="records"))
    return out


def job_run(run_id: str) -> dict[str, Any]:
    r = cli("jobs", "get-run", run_id)

    def ts(ms: int | None) -> str | None:
        return datetime.fromtimestamp(ms / 1000, UTC).isoformat(timespec="seconds") if ms else None

    return {
        "run_id": r["run_id"],
        "job_id": r["job_id"],
        "result": r["state"].get("result_state"),
        "start": ts(r.get("start_time")),
        "end": ts(r.get("end_time")),
        "tasks": [
            {
                "task": t["task_key"],
                "result": t["state"].get("result_state"),
                "start": ts(t.get("start_time")),
                "end": ts(t.get("end_time")),
            }
            for t in r["tasks"]
        ],  # fmt: skip
    }


def pipeline_id(name: str) -> str:
    return str(
        next(p["pipeline_id"] for p in cli("pipelines", "list-pipelines") if p["name"] == name)
    )


def _events(pipeline: str, level: str) -> list[dict[str, Any]]:
    """Events of one level only: the full log exceeds one 250-event page."""
    got = cli("pipelines", "list-pipeline-events", pipeline, "--filter", f"level='{level}'",
              "--max-results", "250")  # fmt: skip
    return list(got if isinstance(got, list) else got.get("events", []))


def demo_updates(pipeline: str, run: dict[str, Any]) -> dict[str, Any]:
    """Pipeline updates that failed during the demo run's pipeline task, and the expectation
    messages Databricks reported (it reports only the first violating row)."""
    task = next(t for t in run["tasks"] if t["task"] == "pipeline")
    start, end = task["start"], task["end"] or datetime.now(UTC).isoformat()

    def within(e: dict[str, Any]) -> bool:
        return bool(start <= e["timestamp"][:19] + "+00:00" <= end)

    errors = [e for e in _events(pipeline, "ERROR") if within(e)]
    failed = sorted(
        {e["origin"]["update_id"] for e in errors
         if e.get("event_type") == "update_progress" and "is FAILED" in e.get("message", "")}
    )  # fmt: skip
    violations = sorted(
        {
            x.get("message", "")
            for e in errors + [w for w in _events(pipeline, "WARN") if within(w)]
            if (e.get("origin") or {}).get("flow_name", "").endswith("gate_results")
            for x in ((e.get("error") or {}).get("exceptions") or [])[:1]
            if "EXPECTATION_VIOLATION" in x.get("message", "")
        }
    )
    return {
        "failed_updates": len(failed),
        "failed_update_ids": failed,
        "reported_violations": violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="the successful cdm-phase1 job run")
    parser.add_argument("--demo-runs", nargs="+", required=True, help="broken-demo job runs")
    args = parser.parse_args()
    sql = Sql()

    real = job_run(args.run)
    evidence: dict[str, Any] = {
        "exported_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "successful_run": real,
        **{name: records(sql(query)) for name, query in QUERIES.items()},
    }

    broken_pipeline = pipeline_id("cdm-phase1-broken")
    demos = []
    for run_id in args.demo_runs:
        run = job_run(run_id)
        demos.append({**run, **demo_updates(broken_pipeline, run)})
    broken_config = json.loads(
        cli_text("fs", "cat", "dbfs:/Volumes/workspace/cdm/raw/broken/sites.json")
    )
    frame = sql("SELECT * FROM workspace.cdm_broken.silver_metadata")
    evidence["broken_demo"] = {
        "runs": demos,
        "subset": broken_config.get("broken_demo"),
        "silver_metadata_rows_per_source": frame["source"].value_counts().sort_index().to_dict(),
        "gates_recomputed_with_cdm_gates_run_all": [
            asdict(g) for g in run_all(frame, broken_config)
        ],
        "quarantine": records(
            sql("SELECT source, isic_id, error FROM workspace.cdm_broken.silver_quarantine")
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
