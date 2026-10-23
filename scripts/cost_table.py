"""Per-run cost tracking (Phase 3): DBUs and run time for every job run and pipeline update,
from Databricks system tables, written to ``workspace.cdm.gold_job_costs``; a summary in
``results/phase3/costs/job_costs.json``; and a trace from each committed results file to the
cost of the job run that produced it, ``results/phase3/costs/result_costs.json``.

The SQL runs on the workspace's SQL warehouse, so nothing heavy runs locally. Databricks Free
Edition charges nothing; ``list_price_usd`` is the list price of the recorded SKU's DBUs
(discounts excluded). Billing records arrive with a delay, so the newest runs are missing until
a later refresh. Lakeflow pipeline compute is billed to the pipeline update, not to the job run
that started it, so it appears as its own row.

Checks after the build fail loudly: the price join must neither drop nor duplicate usage rows,
every row must have a USD list price, and every SKU must be on the GPU or CPU list below.

Usage: python scripts/cost_table.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_export_evidence import Sql, records  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "phase3" / "costs" / "job_costs.json"
TRACE = REPO / "results" / "phase3" / "costs" / "result_costs.json"
INDEX = REPO / "results" / "mlflow_runs.json"
TABLE = "workspace.cdm.gold_job_costs"
# Every SKU a job or pipeline here may bill, by compute type; anything else stops the build.
GPU_SKUS = ("PREMIUM_MODEL_TRAINING_US_EAST_OHIO",)
CPU_SKUS = (
    "PREMIUM_JOBS_SERVERLESS_COMPUTE_US_EAST_OHIO",
    "PREMIUM_ALL_PURPOSE_SERVERLESS_COMPUTE_US_EAST_OHIO",
)

FILTER = """u.usage_unit = 'DBU'
  AND (u.usage_metadata.job_run_id IS NOT NULL OR u.usage_metadata.dlt_update_id IS NOT NULL)"""

BUILD = f"""
CREATE OR REPLACE TABLE {TABLE}
COMMENT 'DBUs, list-price equivalent and wall time per job run and pipeline update, from
system.billing and system.lakeflow. Free Edition charges nothing.'
AS
WITH usage AS (
  SELECT CASE WHEN u.usage_metadata.job_run_id IS NOT NULL THEN 'job_run'
              ELSE 'pipeline_update' END AS kind,
         COALESCE(u.usage_metadata.job_run_id, u.usage_metadata.dlt_update_id) AS run_key,
         u.usage_metadata.job_id AS job_id, u.usage_metadata.dlt_pipeline_id AS pipeline_id,
         u.sku_name, u.usage_quantity, u.usage_start_time,
         p.pricing.default AS unit_price
  FROM system.billing.usage u
  LEFT JOIN system.billing.list_prices p
    ON p.sku_name = u.sku_name AND p.usage_unit = u.usage_unit AND p.currency_code = 'USD'
   AND u.usage_start_time >= p.price_start_time
   AND (p.price_end_time IS NULL OR u.usage_start_time < p.price_end_time)
  WHERE {FILTER}
),
names AS (
  SELECT job_id, name, ROW_NUMBER() OVER (PARTITION BY job_id ORDER BY change_time DESC) AS rn
  FROM system.lakeflow.jobs
),
runs AS (
  SELECT run_id, MIN(period_start_time) AS run_start, MAX(period_end_time) AS run_end,
         MAX_BY(result_state, period_end_time) AS result_state, MAX(run_name) AS run_name
  FROM system.lakeflow.job_run_timeline GROUP BY run_id
)
SELECT u.kind, u.run_key, u.job_id, u.pipeline_id,
       COALESCE(n.name, r.run_name, CONCAT('job ', u.job_id), CONCAT('pipeline ', u.pipeline_id))
         AS name,
       u.sku_name, ROUND(SUM(u.usage_quantity), 6) AS dbus,
       ROUND(SUM(u.usage_quantity * CAST(u.unit_price AS DOUBLE)), 6) AS list_price_usd,
       COUNT(*) AS usage_rows, SUM(CASE WHEN u.unit_price IS NULL THEN 1 ELSE 0 END) AS unpriced,
       MIN(u.usage_start_time) AS first_usage, r.run_start, r.run_end,
       ROUND((UNIX_TIMESTAMP(r.run_end) - UNIX_TIMESTAMP(r.run_start)) / 60.0, 2) AS wall_minutes,
       r.result_state
FROM usage u
LEFT JOIN names n ON n.job_id = u.job_id AND n.rn = 1
LEFT JOIN runs r ON u.kind = 'job_run' AND r.run_id = u.run_key
GROUP BY ALL
"""

COLUMNS = (
    "kind, run_key, name, sku_name, dbus, list_price_usd, wall_minutes, result_state, first_usage"
)
NUMERIC = ("runs", "dbus", "list_price_usd", "wall_minutes", "usage_rows", "unpriced")


def typed(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """The SQL API returns every value as text: numbers back to numbers (counts as integers)."""
    out = []
    for row in rows:
        fixed: dict[str, object] = {}
        for k, v in row.items():
            if k in NUMERIC and v is not None:
                number = float(str(v))
                fixed[k] = int(number) if k in ("runs", "usage_rows", "unpriced") else number
            else:
                fixed[k] = v
        out.append(fixed)
    return out


def compute_type(sku: str) -> str:
    """GPU or CPU from the explicit lists; an unknown SKU is an error, never a guess."""
    if sku in GPU_SKUS:
        return "gpu"
    if sku in CPU_SKUS:
        return "cpu"
    raise ValueError(f"unknown SKU {sku}: add it to GPU_SKUS or CPU_SKUS in scripts/cost_table.py")


def check(sql: Sql) -> None:
    """The price join kept every usage row once, and every row has a price."""
    count = sql(f"SELECT COUNT(*) FROM system.billing.usage u WHERE {FILTER}")
    usage_rows = int(str(count.iloc[0, 0]))
    table = sql(f"SELECT SUM(usage_rows), SUM(unpriced) FROM {TABLE}").iloc[0]
    joined, unpriced = int(str(table.iloc[0])), int(str(table.iloc[1]))
    if joined != usage_rows:
        raise RuntimeError(f"price join changed the row count: {usage_rows} usage, {joined} joined")
    if unpriced:
        raise RuntimeError(f"{unpriced} usage rows have no USD list price")


def summarise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per name and compute type: distinct runs, DBUs, list price, and wall minutes counted once
    per run (a run billed under two SKUs is not counted twice)."""
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        g = groups.setdefault((r["name"], r["compute"]), {"runs": {}, "dbus": 0.0, "usd": 0.0})
        g["runs"][r["run_key"]] = r["wall_minutes"] or 0.0
        g["dbus"] += r["dbus"]
        g["usd"] += r["list_price_usd"]
    out = [
        {"name": name, "compute": compute, "runs": len(g["runs"]), "dbus": round(g["dbus"], 4),
         "list_price_usd": round(g["usd"], 4), "wall_minutes": round(sum(g["runs"].values()), 1)}
        for (name, compute), g in groups.items()
    ]  # fmt: skip
    return sorted(out, key=lambda x: -x["dbus"])


def trace(rows: list[dict[str, Any]], billed_until: str) -> dict[str, Any]:
    """Each committed results file indexed with a Databricks job run, and that run's cost; a file
    whose run has no billing row says why."""
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    by_run: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_run.setdefault(str(r["run_key"]), []).append(r)
    files: dict[str, Any] = {}
    for path, entry in sorted(index.items()):
        run = entry.get("databricks_job_run")
        if not run:
            continue
        billed = by_run.get(str(run), [])
        if billed:
            files[path] = {"job_run": run, "dbus": round(sum(b["dbus"] for b in billed), 6),
                           "list_price_usd": round(sum(b["list_price_usd"] for b in billed), 6),
                           "compute": sorted({b["compute"] for b in billed})}  # fmt: skip
        else:
            files[path] = {"job_run": run, "dbus": None, "reason": "no billing row under this job "
                           f"run: billed after {billed_until} (not yet in system.billing), or its "
                           "compute was billed to a pipeline update"}  # fmt: skip
    covered = sum(1 for f in files.values() if f["dbus"] is not None)
    return {"files": files, "covered": covered, "with_job_run": len(files)}


def main() -> None:
    sql = Sql()
    sql(BUILD)
    check(sql)
    rows = typed(records(sql(f"SELECT {COLUMNS} FROM {TABLE} ORDER BY first_usage")))
    for r in rows:
        r["compute"] = compute_type(str(r["sku_name"]))
    billed_until = str(sql("SELECT MAX(usage_end_time) FROM system.billing.usage").iloc[0, 0])
    note = ("Databricks Free Edition charges nothing; list_price_usd is the list price of the "
            "recorded SKU's DBUs, discounts excluded. Billing arrives with a delay. Wall minutes "
            "are from the job timeline, queue and start-up included, counted once per run. "
            "Pipeline compute is billed to the pipeline update, not the job run.")  # fmt: skip
    built = datetime.now(UTC).isoformat(timespec="seconds")
    common = {"table": TABLE, "built_utc": built, "billing_records_up_to": billed_until}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({**common, "note": note, "by_name": summarise(rows), "runs": rows},
                              indent=2) + "\n", encoding="utf-8", newline="\n")  # fmt: skip
    traced = trace(rows, billed_until)
    TRACE.write_text(json.dumps({**common, "note": note, **traced}, indent=2) + "\n",
                     encoding="utf-8", newline="\n")  # fmt: skip
    print(f"{len(rows)} cost rows; {traced['covered']} of {traced['with_job_run']} results traced")


if __name__ == "__main__":
    main()
